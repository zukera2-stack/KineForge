"""Optional neutral-relative joint models. SI spring/damper and sampled load gates.
Not forward dynamics; contacts do not yet resolve internal actuator loads.
"""
import math
import numpy as np
import torch
from dynamics import rotation_vector, evaluate


def prepare(body):
    specs=body.get('jointModels',{})
    if not isinstance(specs,dict): raise ValueError('jointModels must be object')
    names=body['rig']['bones']; models=[]
    def num(spec,key,default,lo=0.,hi=float('inf')):
        x=spec.get(key,default)
        if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or not lo<=x<=hi: raise ValueError('invalid '+key)
        return float(x)
    for name,s in specs.items():
        if name not in names[1:] or not isinstance(s,dict): raise ValueError('invalid joint name/model: '+name)
        kind=s.get('type'); i=names.index(name)
        if kind not in ('hinge','ball','fixed'): raise ValueError('unsupported joint type')
        allowed={'type','toleranceDeg','stiffnessNmPerRad','dampingNmsPerRad','actuatorLimitNm'}|({'axis','minDeg','maxDeg','restDeg'} if kind=='hinge' else {'maxAngleDeg'} if kind=='ball' else set())
        if set(s)-allowed: raise ValueError('unknown joint model field')
        m={'name':name,'i':i,'type':kind,'tol':math.radians(num(s,'toleranceDeg',.5,0,10)), 'k':num(s,'stiffnessNmPerRad',0),'c':num(s,'dampingNmsPerRad',0),'cap':num(s,'actuatorLimitNm',0) if 'actuatorLimitNm' in s else None}
        if kind=='fixed' and (any(m[x] for x in ('k','c')) or m['cap'] is not None): raise ValueError('fixed joint cannot have motor or spring')
        if kind=='hinge':
            a=np.asarray(s.get('axis'),float)
            if a.shape!=(3,) or not np.isfinite(a).all() or abs(np.linalg.norm(a)-1)>1e-6: raise ValueError('hinge axis must be unit vector')
            lo=num(s,'minDeg',-179,-179,179);hi=num(s,'maxDeg',179,-179,179)
            if lo>hi: raise ValueError('reversed hinge limits')
            rest=num(s,'restDeg',0,lo,hi);m.update(axis=a,lo=math.radians(lo),hi=math.radians(hi),rest=math.radians(rest))
        elif kind=='ball':m['max']=math.radians(num(s,'maxAngleDeg',179,0,179))
        models.append(m)
    return sorted(models,key=lambda x:x['i'])


def samples(L,neutral,models,fps):
    """All vectors in neutral local coordinates; parent-relative spatial omega."""
    for m in models:
        q=neutral[m['i']].T@L[:,m['i']]
        rv=rotation_vector(q)
        increments=rotation_vector(q[1:]@q[:-1].transpose(-1,-2))
        omega=increments*fps
        undersampled=torch.linalg.vector_norm(increments,dim=-1)>math.radians(170)
        # central average of interval angular velocities, no extrapolation at endpoints
        w=(omega[1:]+omega[:-1])*.5
        angle=torch.linalg.vector_norm(rv,dim=-1)
        if m['type']=='hinge':
            a=torch.as_tensor(m['axis'],dtype=L.dtype,device=L.device)
            qa=q@a;dot=(qa*a).sum(-1).clamp(-1,1)
            swing=torch.atan2(torch.linalg.vector_norm(torch.linalg.cross(a.expand_as(qa),qa),dim=-1),dot)
            skew=torch.stack([q[:,2,1]-q[:,1,2],q[:,0,2]-q[:,2,0],q[:,1,0]-q[:,0,1]],-1)
            sn=(skew*a).sum(-1);cs=q.diagonal(dim1=-2,dim2=-1).sum(-1)-dot
            theta=torch.atan2(sn,cs+1e-12)
            # singular 180-degree swing (no unique twist); flagged via swing
            violation=torch.maximum(swing,torch.maximum(torch.relu(m['lo']-theta),torch.relu(theta-m['hi'])))
            displacement=(theta-m['rest'])[:,None]*a
            w=(w*a).sum(-1,keepdim=True)*a
            ambiguous=(swing>math.radians(179))|(theta.abs()>math.radians(179.5))
            undersampled=undersampled|(torch.diff(theta).abs()>math.pi)
        elif m['type']=='ball':
            violation=torch.relu(angle-m['max']);displacement=rv;ambiguous=angle>math.radians(179.5)
        else:
            violation=angle;displacement=rv;ambiguous=angle>math.radians(179.5)
        ambiguous=ambiguous.clone();ambiguous[1:] |= undersampled;ambiguous[:-1] |= undersampled
        passive=-m['k']*displacement[1:-1]-m['c']*w
        energy=.5*m['k']*displacement.square().sum(-1)
        yield m,violation,passive,energy,ambiguous


def assess(L,G,P,neutral,models,physical,p,fps,contacts):
    # Known gravity-only subtree inverse dynamics; unknown support reactions invalidate loads.
    loads=evaluate(G,P,*physical[:3],p,fps,physical[3]) if physical is not None and len(P)>=3 and (contacts is None or not contacts['items']) and any(m['cap'] is not None for m in models) else None
    for m,violation,passive,energy,ambiguous in samples(L,neutral,models,fps):
        demand=None
        if m['cap'] is not None and loads is not None:
            i=m['i'];basis=G[1:-1,p[i]]@neutral[i]
            req=(basis.transpose(-1,-2)@loads['joint_torque'][:,i,:,None]).squeeze(-1)
            if m['type']=='hinge':
                a=torch.as_tensor(m['axis'],dtype=L.dtype,device=L.device);req=(req*a).sum(-1,keepdim=True)*a
            demand=torch.linalg.vector_norm(req-passive,dim=-1)
        yield m,violation,passive,energy,ambiguous,demand


def cost(L,G,P,neutral,models,physical,p,fps,contacts):
    result=L.sum()*0
    for m,v,_,_,_,demand in assess(L,G,P,neutral,models,physical,p,fps,contacts):
        result=result+10*v.square().mean()
        if demand is not None:
            result=result+torch.relu(demand-m['cap']).square().mean()/max(1.,m['cap'])**2
    return result/max(1,len(models))


def report(L,G,P,neutral,models,physical,p,fps,contacts,start=0):
    T=lambda x:torch.as_tensor(x,dtype=torch.float64)
    rows=[]; flags=set()
    with torch.no_grad():
        for m,v,passive,energy,ambiguous,demand in assess(T(L),T(G),T(P),T(neutral),models,physical,p,fps,contacts):
            peak=lambda x:float(x.max()) if x.numel() else None
            bad=(v>m['tol']+1e-9).nonzero().flatten().tolist()
            if bad:flags.add('JOINT_RANGE_OR_AXIS_LIMIT')
            if ambiguous.any():flags.add('JOINT_ROTATION_AMBIGUOUS')
            if m['cap'] is not None and demand is None:flags.add('ACTUATOR_LOAD_UNVERIFIED')
            over=[] if demand is None else (demand>m['cap']+1e-6).nonzero().flatten().tolist()
            if over:flags.add('ACTUATOR_LIMIT')
            rows.append({'joint':m['name'],'type':m['type'],'max_violation_deg':math.degrees(peak(v)), 'violation_frames':[start+i for i in bad], 'ambiguous_frames':[start+i for i in ambiguous.nonzero().flatten().tolist()], 'passive_peak_Nm':peak(torch.linalg.vector_norm(passive,dim=-1)), 'spring_energy_peak_J':peak(energy),'actuator_limit_Nm':m['cap'],'actuator_peak_Nm':None if demand is None else peak(demand),'actuator_over_frames':[start+i+1 for i in over]})
    return {'status':'REVIEW_REQUIRED' if flags else 'SAMPLED_JOINT_CONSTRAINTS_PASS','flags':sorted(flags),'joints':rows,'load_assumption':'gravity-only inverse dynamics, interior samples; supported internal loads not resolved','limitations':'hinge principal twist chart, isotropic ball angular cone, no impacts, forward dynamics or biological calibration; unspecified joints unconstrained'}
