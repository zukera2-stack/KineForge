#!/usr/bin/env python3
"""Unified motion optimization, independent of HTML/RTM. Python 3 + NumPy/SciPy/Torch."""
import argparse, copy, csv, hashlib, json, math
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R, Slerp
import torch
VERSION='0.7.0-beta'
import dynamics
import contact as contact_model
import joints as joint_model
import collision as collision_model
C=np.array([[1,0,0],[0,0,-1],[0,1,0]],float) # legacy coordinate conversion

def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,x):
    with Path(p).open('x',encoding='utf8') as f: json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(c,m):
    if not c: raise ValueError(m)
def rig_check(r):
    n=len(r['bones']); p=r['parents']; need(2<=n<=256 and len(set(r['bones']))==n and len(p)==n and p[0]==-1 and all(type(v)==int and 0<=v<i for i,v in enumerate(p[1:],1)), 'invalid ordered single-root rig')
    return np.array(p)
def valid(d,b):
    need(d['schema']=='sam.keyposes.v1','expected keyposes');p=rig_check(d['rig']);need(d['rig']==b['rig'],'rig mismatch');t=d['timeline']; f=t['end']-t['start']+1; fps=t['fps'];need(type(t['start'])==int and type(t['end'])==int and 2<=f<=10000 and np.isfinite(fps) and fps>0,'invalid timeline'); k=d['keys'];ids=np.array([x['frame']-t['start'] for x in k]);y=np.array([x['positions'] for x in k],float);need(ids.dtype.kind in 'iu' and len(ids)>=2 and ids[0]==0 and ids[-1]==f-1 and (np.diff(ids)>0).all() and y.shape==(len(ids),len(p),3) and np.isfinite(y).all(),'invalid keys');return p,f,float(fps),ids,y

def fk(L,root,o,p):
    G=[L[:,0]]; P=[root]
    for i,j in enumerate(p[1:],1): G.append(G[j]@L[:,i]);P.append(P[j]+(G[j]@o[i,:,None]).squeeze(-1))
    return np.stack(G,1),np.stack(P,1)
def reference(path,b,f):
    d=read(path);need(d['rig']==b['rig'],'reference rig mismatch');q=np.asarray(d['localQuaternionXYZW']);need(q.ndim==3 and q.shape[1:]==(len(b['rig']['bones']),4) and np.isfinite(q).all(),'bad reference');t=np.asarray(d['phase']);need(t.shape==(len(q),) and np.all(np.diff(t)>0) and abs(t[0])<1e-6 and abs(t[-1]-1)<1e-6,'reference phases');return np.stack([Slerp(t,R.from_quat(q[:,i]))(np.linspace(0,1,f)).as_matrix() for i in range(q.shape[1])],1)
def exp(v):
    x,y,z=v.unbind(-1);o=torch.zeros_like(x);S=torch.stack((o,-z,y,z,o,-x,-y,x,o),-1).reshape(*v.shape[:-1],3,3);a=torch.linalg.vector_norm(v,dim=-1);return torch.eye(3,dtype=v.dtype)+torch.sinc(a/torch.pi)[...,None,None]*S+.5*torch.sinc(a/(2*torch.pi)).square()[...,None,None]*(S@S)
def rms(x): return float(np.sqrt(np.mean(np.sum(x*x,axis=-1)))) if x.size else 0.
def metrics(P,L,fps,ids,y,names,o,p):
    v=np.diff(P,axis=0)*fps;a=np.diff(v,axis=0)*fps;j=np.diff(a,axis=0)*fps;w=R.from_matrix((L[1:]@L[:-1].swapaxes(-1,-2)).reshape(-1,3,3)).as_rotvec().reshape(len(P)-1,len(p),3)*180/np.pi;step=np.linalg.norm(w,axis=-1);i,b=np.unravel_index(step.argmax(),step.shape);e=np.linalg.norm(P[ids]-y,axis=-1);bl=np.linalg.norm(P[:,1:]-P[:,p[1:]],axis=-1); peak=lambda a:float(np.linalg.norm(a,axis=-1).max()) if a.size else 0.
    return dict(key_rms_mm=float(np.sqrt(np.mean(e**2))*1000),key_max_mm=float(e.max()*1000),boundary_max_mm=float(e[[0,-1]].max()*1000),length_max_error_m=float(np.abs(bl-np.linalg.norm(o[1:],axis=-1)).max()),max_speed_mps=peak(v),max_accel_mps2=peak(a),rms_jerk_mps3=rms(j),max_step_deg=float(step.max()),rms_angular_jerk_deg_s3=rms(np.diff(w,n=2,axis=0)*fps**3),worst_joint=names[b],worst_transition_indices=[int(i),int(i+1)])
def align_pose(y,L0,o,p):
    """Recover global bone frames from child offsets, closest-prior swing for rank-one joints.
    Rank-one twist and leaf rotations are not observable from positions alone."""
    G=[];L=[]
    for i,parent in enumerate(p):
        prior=L0[i] if parent<0 else G[parent]@L0[i]
        children=np.flatnonzero(p==i)
        x=o[children];z=y[children]-y[i];g=prior
        if len(children) and np.linalg.norm(z)>1e-12:
            if np.linalg.matrix_rank(x,tol=1e-8)>=2 and np.linalg.matrix_rank(z,tol=1e-8)>=2:
                u,_,vt=np.linalg.svd(x.T@z);fix=np.eye(3);fix[-1,-1]=np.linalg.det(vt.T@u.T);g=vt.T@fix@u.T
            else:
                j=int(np.argmax(np.linalg.norm(x,axis=-1)));a=prior@x[j];b=z[j]
                if np.linalg.norm(b)>1e-12:
                    a=a/np.linalg.norm(a);b=b/np.linalg.norm(b);cross=np.cross(a,b);sn=np.linalg.norm(cross);cs=np.clip(a@b,-1,1)
                    if sn>1e-12:g=R.from_rotvec(cross/sn*np.arctan2(sn,cs)).as_matrix()@prior
                    elif cs<0:
                        axis=np.cross(a,np.eye(3)[np.argmin(np.abs(a))]);g=R.from_rotvec(axis/np.linalg.norm(axis)*np.pi).as_matrix()@prior
        G.append(g);L.append(g if parent<0 else G[parent].T@g)
    return np.asarray(L)
def fit_pose(y,L0,o,p):
    aligned=align_pose(y,L0,o,p)
    err=lambda L:np.linalg.norm(fk(L[None],y[None,0],o,p)[1][0]-y,axis=-1)
    if err(aligned).max()<1e-5:return aligned
    seed=min([L0,aligned],key=lambda L:np.mean(err(L)**2))
    T=lambda a:torch.tensor(a,dtype=torch.float64);base=T(seed)[None];off=T(o);target=T(y)[None];delta=torch.zeros((1,len(p),3),dtype=torch.float64,requires_grad=True);opt=torch.optim.LBFGS([delta],max_iter=100,line_search_fn='strong_wolfe',tolerance_grad=1e-11)
    def closure():
        opt.zero_grad();L=exp(delta)@base;P=fkt(L,target[:,0],off,p)[1];loss=(P-target).square().mean()+1e-7*delta.square().mean();loss.backward();return loss
    opt.step(closure);return min([seed,(exp(delta)@base).detach().numpy()[0]],key=lambda L:np.mean(err(L)**2))
def fkt(L,root,o,p):
    G=[L[:,0]];P=[root]
    for i,j in enumerate(p[1:],1):G.append(G[j]@L[:,i]);P.append(P[j]+(G[j]@o[i,:,None]).squeeze(-1))
    return torch.stack(G,1),torch.stack(P,1)
def solve(d,b,ref,iterations=120,strength=.6,max_key_mm=100,initial=None,max_step_deg=None):
    need(type(iterations)==int and iterations>0 and np.isfinite(strength) and strength>=0 and np.isfinite(max_key_mm) and max_key_mm>0 and (max_step_deg is None or np.isfinite(max_step_deg) and max_step_deg>0), 'invalid optimizer settings');p,F,fps,ids,Y=valid(d,b);n=len(p);names=d['rig']['bones'];o=np.array(b['offsets_m']);need(o.shape==(n,3) and np.isfinite(o).all() and (np.linalg.norm(o[1:],axis=-1)>1e-9).all(),'bad offsets');need(np.isfinite(ref).all() and ref.shape==(F,n,3,3),'bad reference rotations');t=np.arange(F)/fps;root=CubicSpline(t[ids],Y[:,0],bc_type='natural')(t)
    physical=dynamics.tensors(b['physical'],n) if 'physical' in b else None
    contact_schedule=contact_model.prepare(d,physical)
    joint_models=joint_model.prepare(b)
    collision_spec=collision_model.prepare(b)
    # A wholly unsupported clip has a ballistic translation seed. Interpolate
    # the gravity-compensated coordinates, then restore the known acceleration.
    if contact_schedule is not None and not contact_schedule['items']:
        drift=.5*t[:,None]**2*physical[3].numpy();root=CubicSpline(t[ids],Y[:,0]-drift[ids],bc_type='natural')(t)+drift
    # For interior keys seed from reference or previously computed rotations, never editor rotations.
    seed=ref if initial is None else initial
    if initial is not None:need(initial.shape==ref.shape and np.isfinite(initial).all(),'bad initial')
    K=np.array([seed[i] if np.linalg.norm(fk(seed[i:i+1],y[None,0],o,p)[1][0]-y,axis=-1).max()<1e-5 else fit_pose(y,seed[i],o,p) for i,y in zip(ids,Y)])
    L0=np.stack([Slerp(t[ids],R.from_matrix(K[:,j]))(t).as_matrix() for j in range(n)],1);_,P0=fk(L0,root,o,p)
    initial_metrics=metrics(P0,L0,fps,ids,Y,names,o,p)
    rate=np.asarray(b.get('observedAngularRateDegS',[3600.]*n),float);need(rate.shape==(n,) and np.isfinite(rate).all() and (rate>0).all(),'invalid rate profile')
    hard_peak=min(179.,initial_metrics['max_step_deg']*1.10+.01, max_step_deg if max_step_deg is not None else 179.);need(hard_peak>0,'invalid step limit')
    cap=np.minimum(np.clip(rate*1.25/fps,.5,179),hard_peak);capcos=torch.tensor(np.cos(np.radians(cap)),dtype=torch.float64);budget=max_key_mm/1000
    T=lambda a:torch.tensor(a,dtype=torch.float64);off=T(o);base=T(L0);target=T(Y);r0=T(root);prior=T(ref);mask=torch.ones(F,1,1,dtype=torch.float64);mask[[0,-1]]=0
    w=T(b.get('keyWeights',[1]*n));need(w.shape==(n,) and torch.all(w>0),'bad key weights');neutral=T(R.from_quat(b['neutralLocalQuaternionXYZW']).as_matrix());lim=T(np.cos(np.radians(b['observedAngleEnvelopeDeg'])));axis=T(b['primaryAxes']);twlim=T(np.radians(b['observedTwistEnvelopeDeg']));has_axis=T(np.linalg.norm(np.array(b['primaryAxes']),axis=-1)>0)
    mass=T(b.get('massFractions',[1/n]*n));need(mass.shape==(n,) and torch.all(mass>0),'bad mass proxies');mass=mass/mass.sum(); inertia=mass* T(b.get('inertiaLengthM',[.1]*n)).square()/3
    physical_weight=float(b.get('physicalObjectiveWeight',.15));need(np.isfinite(physical_weight) and physical_weight>=0,'invalid physical weight')
    # Explicit user/profile ground contacts, not inferred as hard foot plants in every clip.
    contacts=d.get('solverHints',{}).get('contacts',[]);contact_terms=[]
    for c in contacts:
        j=names.index(c['bone']);lo=c['start']-d['timeline']['start'];hi=c['end']-d['timeline']['start'];need(0<=lo<=hi<F,'bad contact range');q=T(c['position']);need(q.shape==(3,) and torch.isfinite(q).all(),'bad contact point');contact_terms.append((j,lo,hi,q))
    pairs=b.get('collisionProxies',[]); # optional explicit sphere pairs, conservative sampled proxy only
    for a,z,r in pairs:need(0<=a<n and 0<=z<n and a!=z and r>0,'bad collision proxy')
    v=torch.zeros((F,n,3),dtype=torch.float64,requires_grad=True);dr=torch.zeros((F,3),dtype=torch.float64,requires_grad=True);opt=torch.optim.LBFGS([v,dr],lr=.85,max_iter=iterations,history_size=15,line_search_fn='strong_wolfe',tolerance_grad=1e-9,tolerance_change=1e-12)
    def state():
        L=exp(v*mask)@base;rt=r0+dr*mask[:,0];G,P=fkt(L,rt,off,p);return L,G,P
    def mean(x):return x.square().mean() if x.numel() else x.sum()*0
    def terms(L,G,P):
        diff=lambda x,k:torch.diff(x,n=k,dim=0); rel=neutral.transpose(-1,-2)@L;cos=(rel.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2
        # twist of relative rotation about a declared axis; undefined near 180° swing, diagnostic only there.
        sk=torch.stack([rel[...,2,1]-rel[...,1,2],rel[...,0,2]-rel[...,2,0],rel[...,1,0]-rel[...,0,1]],-1);s=(sk*axis).sum(-1);c=rel.diagonal(dim1=-2,dim2=-1).sum(-1)-(axis[...,None,:]@rel@axis[...,:,None]).squeeze(-1).squeeze(-1);tw=torch.atan2(s,c+1e-12)
        env=mean(torch.relu(lim-cos))+mean(torch.relu(tw.abs()-twlim)*has_axis)
        # Geometry (changes of unit tangent) independent of timing, downweight near-stationary segments.
        dp=diff(P,1);speed=torch.linalg.vector_norm(dp,dim=-1,keepdim=True);u=dp/(speed+1e-5);bend=mean(diff(u,1)*torch.minimum(speed[1:],speed[:-1]))
        acc=diff(P,2);ang=diff(G,2); # dimensionless finite-step proxies, not forces or torque solution
        inertia_proxy=(acc.square().sum(-1)*mass).sum(-1).mean() if acc.numel() else acc.sum()*0
        angular_proxy=(ang.square().sum((-1,-2))*inertia).sum(-1).mean() if ang.numel() else ang.sum()*0
        if physical is not None:
            # Gravity-only subtree loads are not contact-aware actuator loads.
            # With prescribed contacts the external wrench residual is the physical term.
            inertia_proxy=dynamics.cost(G,P,physical,p,fps) if contact_schedule is None else P.sum()*0;angular_proxy=0
        contact=sum(mean(P[lo:hi+1,j]-q) for j,lo,hi,q in contact_terms) if contact_terms else P.sum()*0
        collision=sum(mean(torch.relu(r-torch.linalg.vector_norm(P[:,a]-P[:,z],dim=-1))) for a,z,r in pairs) if pairs else P.sum()*0
        err=(P[ids]-target).square().sum(-1);step_rel=L[1:]@L[:-1].transpose(-1,-2);step_cos=(step_rel.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2
        peak_key=torch.relu(torch.sqrt(err+1e-18)-budget*.85).square().sum()/len(ids)
        peak_rot=torch.relu(torch.sqrt(torch.clamp(2*(1-step_cos),min=1e-18))-torch.sqrt(2*(1-capcos))).square().sum()/max(1,F-1)
        return {'keys':(err*w).mean(),'key_budget':peak_key,'rotation_budget':peak_rot,'linear':2*mean(acc)+5*mean(diff(P,3))+.4*bend,'angular':1.2*mean(diff(L,2))+3*mean(diff(L,3))+.035*mean(diff(L,1)),'prior':.0003*mean(L-prior),'observed_envelope':.02*env,'inertia_proxy':(physical_weight if physical is not None else .15)*(inertia_proxy+angular_proxy),'contacts':2*contact,'contact_wrench':contact_model.evaluate(G,P,physical,p,fps,contact_schedule) if contact_schedule is not None else P.sum()*0,'collision':2*collision+(collision_model.cost(L,G,P,off,p,collision_spec,fkt,exp) if collision_spec is not None else P.sum()*0),'joints':joint_model.cost(L,G,P,neutral,joint_models,physical,p,fps,contact_schedule) if joint_models else P.sum()*0,'root':.025*mean(dr)}
    def total(q):return 4*q['keys']+40*q['key_budget']+8*q['rotation_budget']+strength*(q['linear']+q['angular'])+sum(q[k] for k in ['prior','observed_envelope','inertia_proxy','contacts','contact_wrench','collision','joints','root'])
    def closure():opt.zero_grad();q=terms(*state());loss=total(q);loss.backward();return loss
    with torch.no_grad():before=float(total(terms(*state())))
    opt.step(closure)
    with torch.no_grad():L,G,P=state();q=terms(L,G,P);after=float(total(q));L=L.numpy();P=P.numpy();q={k:float(x) for k,x in q.items()}
    need(np.isfinite(P).all() and np.isfinite(L).all() and after<=before+1e-10,'optimization failed');m=metrics(P,L,fps,ids,Y,names,o,p);initial_metrics=metrics(P0,L0,fps,ids,Y,names,o,p);optimized_metrics=copy.deepcopy(m);rejected=False
    def rates(L):
        return np.linalg.norm(R.from_matrix((L[1:]@L[:-1].swapaxes(-1,-2)).reshape(-1,3,3)).as_rotvec().reshape(F-1,n,3),axis=-1)*180/np.pi
    def violations(m,L,P=None):
        flags=[]
        if m['key_max_mm']>max_key_mm:flags.append('KEY_DEVIATION_LIMIT')
        if m['boundary_max_mm']>.1:flags.append('BOUNDARY_NOT_EXACT')
        if m['max_step_deg']>initial_metrics['max_step_deg']*1.1+.01:flags.append('PEAK_ROTATION_REGRESSION')
        if max_step_deg is not None and m['max_step_deg']>max_step_deg:flags.append('PEAK_ROTATION_LIMIT')
        for key in ['rms_jerk_mps3','rms_angular_jerk_deg_s3']:
            if m[key]>initial_metrics[key]*1.05+1e-6:flags.append('REGRESSION_'+key)
        if physical is not None and P is not None and F>=3:
            phys=dynamics.report(fk(L,P[:,0],o,p)[0],P,b['physical'],p,fps)
            if phys['near_pi_warning']:flags.append('WORLD_ROTATION_UNDERSAMPLED')
            if contact_schedule is None and phys['root_force_rms_N']>physical_baseline['root_force_rms_N']*1.05+1e-6:flags.append('ROOT_FORCE_REGRESSION')
            if contact_schedule is None and not physical_baseline['near_pi_warning'] and phys['segment_com_torque_rms_Nm']>physical_baseline['segment_com_torque_rms_Nm']*1.05+1e-6:flags.append('SEGMENT_TORQUE_REGRESSION')
        if contact_schedule is not None and P is not None:
            flags.extend(contact_model.report(fk(L,P[:,0],o,p)[0],P,physical,p,fps,contact_schedule)['flags'])
        if joint_models and P is not None:
            flags.extend(joint_model.report(L,fk(L,P[:,0],o,p)[0],P,neutral,joint_models,physical,p,fps,contact_schedule,d['timeline']['start'])['flags'])
        if collision_spec is not None and P is not None:
            flags.extend(collision_model.report(L,P[:,0],o,p,collision_spec,fk,d['timeline']['start'])['flags'])
        return flags
    physical_baseline=dynamics.report(fk(L0,root,o,p)[0],P0,b['physical'],p,fps) if physical is not None else None
    attempt_flags=violations(m,L,P);initial_flags=violations(initial_metrics,L0,P0)
    # Line search in valid rotation space, not independent point interpolation. A useful partial
    # correction can pass the explicit gates even if the unconstrained optimizer overshoots.
    candidates=[(0.,L0,P0,initial_metrics,initial_flags),(1.,L,P,m,attempt_flags)]
    if attempt_flags:
        rv=R.from_matrix((L@L0.swapaxes(-1,-2)).reshape(-1,3,3)).as_rotvec()
        for fraction in [.125,.25,.375,.5,.625,.75,.875]:
            Li=R.from_rotvec(rv*fraction).as_matrix().reshape(F,n,3,3)@L0
            Pi=fk(Li,root+fraction*(P[:,0]-root),o,p)[1];mi=metrics(Pi,Li,fps,ids,Y,names,o,p);candidates.append((fraction,Li,Pi,mi,violations(mi,Li,Pi)))
    def rank(c):
        _,_,_,m,flags=c
        hard=sum(f in ('KEY_DEVIATION_LIMIT','BOUNDARY_NOT_EXACT','CONTACT_WRENCH_RESIDUAL','CONTACT_ANCHOR_DRIFT','JOINT_RANGE_OR_AXIS_LIMIT','JOINT_ROTATION_AMBIGUOUS','ACTUATOR_LIMIT','ACTUATOR_LOAD_UNVERIFIED','COLLISION_PENETRATION','COLLISION_INTERVAL_UNCERTIFIED','COLLISION_ROTATION_AMBIGUOUS') for f in flags) if contact_schedule is not None or joint_models or collision_spec is not None else 0
        return (hard,len(flags), m['rms_jerk_mps3']/max(initial_metrics['rms_jerk_mps3'],1e-6)+m['rms_angular_jerk_deg_s3']/max(initial_metrics['rms_angular_jerk_deg_s3'],1e-6)+.15*m['key_max_mm']/max_key_mm)
    fraction,L,P,m,selected_flags=min(candidates,key=rank);rejected=fraction==0.
    need(m['length_max_error_m']<1e-8,'FK length failure')
    # Explicit quality gate; never claim arbitrary author pose / physical validity.
    ql=R.from_matrix(L.reshape(-1,3,3)).as_quat().reshape(F,n,4)
    for i in range(1,F):ql[i]*=np.where((ql[i]*ql[i-1]).sum(-1)<0,-1,1)[:,None]
    flags=violations(m,L,P)
    if rejected and (flags or attempt_flags):flags.append('OPTIMIZATION_REJECTED_KEEP_INITIAL')
    report={'status':'REVIEW_REQUIRED' if flags else 'KINEMATIC_CANDIDATE','flags':flags,'version':VERSION,'initial':initial_metrics,'optimized_attempt':optimized_metrics,'attempt_flags':attempt_flags,'selected_fraction':fraction,'candidate_metrics':[{'fraction':c[0],'flags':c[4],'metrics':c[3]} for c in candidates],'initial_flags':initial_flags,'rotation_budget_deg':cap.tolist(),'observed_rate_excess_deg_max':float(np.maximum(rates(L)-cap,0).max()),'observed_rate_warning':bool(np.any(rates(L)>cap[None,:]+.05)),'optimization_rejected':rejected,'final':m,'objective_before':before,'objective_after':after,'objective_terms_attempt':q,'iterations_requested':iterations,'optimizer_iterations':opt.state[v].get('n_iter',0),'strength':strength,'limits':{'max_key_mm':max_key_mm,'boundary_mm':.1,'max_step_deg':max_step_deg},'key_errors':[{'frame':int(i+d['timeline']['start']),'max_mm':float(np.linalg.norm(P[i]-y,axis=-1).max()*1000)} for i,y in zip(ids,Y)],'limitations':['Observed rotation envelope is statistical, not calibrated anatomical limits.','Mass/inertia proxy is not coupled rigid-body dynamics.','Collision and contact terms are optional explicit sampled proxies.','No mesh deformation or runtime validation.']}
    if physical is not None:
        report['physical_objective_weight']=physical_weight
        report['physical_initial']=physical_baseline
        report['physical_final']=dynamics.report(fk(L,P[:,0],o,p)[0],P,b['physical'],p,fps)
        report['limitations'].append('User-supplied mass model; sampled inverse dynamics only. No actuator solution.')
        if contact_schedule is not None:
            report['contact_initial']=contact_model.report(fk(L0,root,o,p)[0],P0,physical,p,fps,contact_schedule)
            report['contact_final']=contact_model.report(fk(L,P[:,0],o,p)[0],P,physical,p,fps,contact_schedule)
    if joint_models:
        report['joint_initial']=joint_model.report(L0,fk(L0,root,o,p)[0],P0,neutral,joint_models,physical,p,fps,contact_schedule,d['timeline']['start'])
        report['joint_final']=joint_model.report(L,fk(L,P[:,0],o,p)[0],P,neutral,joint_models,physical,p,fps,contact_schedule,d['timeline']['start'])
    if collision_spec is not None:
        report['collision_initial']=collision_model.report(L0,root,o,p,collision_spec,fk,d['timeline']['start'])
        report['collision_final']=collision_model.report(L,P[:,0],o,p,collision_spec,fk,d['timeline']['start'])
    out=copy.deepcopy(d);out['schema']='sam.skeletal.solved.v1';out['originalAuthorKeys']=out.pop('keys');out['keyframes']=[{'frame':i+d['timeline']['start'],'positions':a.tolist()} for i,a in enumerate(P)];out['sourceKeys']=[{**k,'positions':out['keyframes'][i]['positions']} for k,i in zip(d['keys'],ids)];out['lockedFrames']=[k['frame'] for k in d['keys']];out['localQuaternionXYZW']=ql.tolist();out['solver']={'name':'KineForge '+VERSION,'diagnostics':report};out['notes']='Original author keys retained separately. Editor sourceKeys represent solved positions. REVIEW_REQUIRED is not acceptance.';return out,report,P,L,ids,Y

def diagnostics(path,P,L,fps,names,start,ids,b):
    if 'physical' in b and len(P)>=3:
        p=rig_check(b['rig']);G,_=fk(L,P[:,0],np.array(b['offsets_m']),p);dynamics.write_csv(Path(path).with_name('physical_loads.csv'),G,P,b['physical'],p,fps,names,start)
    q=R.from_matrix(L.reshape(-1,3,3)).as_quat().reshape(len(P),len(names),4)
    for i in range(1,len(P)):q[i]*=np.where((q[i]*q[i-1]).sum(-1)<0,-1,1)[:,None]
    v=np.diff(P,axis=0)*fps;a=np.diff(v,axis=0)*fps;j=np.diff(a,axis=0)*fps;omega=R.from_matrix((L[1:]@L[:-1].swapaxes(-1,-2)).reshape(-1,3,3)).as_rotvec().reshape(len(P)-1,len(names),3)*fps*180/np.pi;alpha=np.diff(omega,axis=0)*fps;aj=np.diff(alpha,axis=0)*fps
    neutral=R.from_quat(b['neutralLocalQuaternionXYZW']).as_matrix();rel=neutral.transpose(0,2,1)@L;qr=R.from_matrix(rel.reshape(-1,3,3)).as_quat().reshape(q.shape);axes=np.array(b['primaryAxes']);tw=2*np.arctan2((qr[...,:3]*axes).sum(-1),qr[...,3]);tw=np.degrees((tw+np.pi)%(2*np.pi)-np.pi);tw[:,np.linalg.norm(axes,axis=-1)<1e-8]=np.nan
    turns=[]
    for ax in [(0,2),(1,2)]:
        z=np.diff(P[:,:,ax],axis=0);a0=z[:-1];a1=z[1:];den=np.linalg.norm(a0,axis=-1)*np.linalg.norm(a1,axis=-1);cos=(a0*a1).sum(-1)/np.maximum(den,1e-20);angle=np.degrees(np.arccos(np.clip(cos,-1,1)));angle[den<1e-8]=np.nan;turns.append(angle)
    scalar=lambda a:float(a) if np.isfinite(a) else ''
    with Path(path).open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['frame' ,'seconds','key','bone','x_m','y_m','z_m','qx','qy','qz','qw','twist_vs_neutral_deg','turn_XZ_deg','turn_YZ_deg']+[f'{k}_{a}' for k in ['v_mps','a_mps2','jerk_mps3','omega_deg_s','alpha_deg_s2','angular_jerk_deg_s3'] for a in 'xyz'])
        for i in range(len(P)):
            for b,n in enumerate(names):w.writerow([i+start,i/fps,int(i in ids),n,*P[i,b],*q[i,b],scalar(tw[i,b]),*[scalar(z[i-1,b]) if 0<i<len(P)-1 else '' for z in turns],*[c for arr,k in [(v,1),(a,2),(j,3),(omega,1),(alpha,2),(aj,3)] for c in (arr[i-k,b] if i>=k else ['']*3)]])
def main():
    a=argparse.ArgumentParser();sub=a.add_subparsers(dest='mode',required=True);sim=sub.add_parser('simulate');sim.add_argument('body');sim.add_argument('scenario');sim.add_argument('output');tr=sub.add_parser('track');tr.add_argument('body');tr.add_argument('target');tr.add_argument('options');tr.add_argument('output');s=sub.add_parser('solve');s.add_argument('keys');s.add_argument('body');s.add_argument('reference');s.add_argument('output');s.add_argument('--initial');s.add_argument('--iterations',type=int,default=120);s.add_argument('--strength',type=float,default=.6);s.add_argument('--max-key-mm',type=float,default=100);s.add_argument('--max-step-deg',type=float);z=sub.add_parser('analyze');z.add_argument('solved');z.add_argument('body');z.add_argument('output');c=sub.add_parser('compare');c.add_argument('first');c.add_argument('second');c.add_argument('body');c.add_argument('output');args=a.parse_args();torch.set_num_threads(1)
    b=read(args.body)
    if args.mode=='track':
        from tracking import track
        write(args.output,track(b,read(args.target),read(args.options)));return
    if args.mode=='simulate':
        from coupled import simulate
        write(args.output,simulate(b,read(args.scenario)));return
    def analyze(path):
        d=read(path);need(d['rig']==b['rig'],'rig mismatch');p=rig_check(d['rig']);P=np.array([k['positions'] for k in d['keyframes']]);L=R.from_quat(np.array(d['localQuaternionXYZW']).reshape(-1,4)).as_matrix().reshape(len(P),len(p),3,3);keys=d['originalAuthorKeys'];ids=np.array([k['frame']-d['timeline']['start'] for k in keys]);Y=np.array([k['positions'] for k in keys]);fps=d['timeline']['fps'];o=np.array(b['offsets_m']);r=metrics(P,L,fps,ids,Y,d['rig']['bones'],o,p)
        if 'physical' in b:
            G,expected=fk(L,P[:,0],o,p);need(np.max(np.abs(expected-P))<1e-6,'inconsistent solved coordinates and rotations');r['physical']=dynamics.report(G,P,b['physical'],p,fps)
            params=dynamics.tensors(b['physical'],len(p));schedule=contact_model.prepare(d,params)
            if schedule is not None:r['contact']=contact_model.report(G,P,params,p,fps,schedule)
        jm=joint_model.prepare(b)
        if jm:
            G,expected=fk(L,P[:,0],o,p);need(np.max(np.abs(expected-P))<1e-6,'inconsistent solved coordinates and rotations');params=dynamics.tensors(b['physical'],len(p)) if 'physical' in b else None; schedule=contact_model.prepare(d,params);r['joints']=joint_model.report(L,G,P,R.from_quat(b['neutralLocalQuaternionXYZW']).as_matrix(),jm,params,p,fps,schedule,d['timeline']['start'])
        cm=collision_model.prepare(b)
        if cm is not None:
            G,expected=fk(L,P[:,0],o,p);need(np.max(np.abs(expected-P))<1e-6,'inconsistent solved coordinates and rotations');r['collision']=collision_model.report(L,P[:,0],o,p,cm,fk,d['timeline']['start'])
        return r
    if args.mode=='analyze':write(args.output,analyze(args.solved));return
    if args.mode=='compare':write(args.output,{'first':analyze(args.first),'second':analyze(args.second),'note':'Each key error is relative to its own author keys.'});return
    need(args.iterations>0 and args.strength>=0 and args.max_key_mm>0,'invalid options');d=read(args.keys);_,f,_,_,_=valid(d,b);ref=reference(args.reference,b,f);initial=None
    if args.initial: initial=reference(args.initial,b,f)
    out,r,P,L,ids,Y=solve(d,b,ref,args.iterations,args.strength,args.max_key_mm,initial,args.max_step_deg);r.update(source_sha256=sha(args.keys),body_sha256=sha(args.body),reference_sha256=sha(args.reference),initial_sha256=sha(args.initial) if args.initial else None);out['solver']['sourceSha256']=r['source_sha256'];p=Path(args.output);p.mkdir(parents=True,exist_ok=False);write(p/'solved.json',out);write(p/'report.json',r);diagnostics(p/'trajectories.csv',P,L,d['timeline']['fps'],d['rig']['bones'],d['timeline']['start'],ids,b);print(json.dumps(r,ensure_ascii=False))
if __name__=='__main__':main()
