"""Bounded whole-horizon direct shooting; explicit contact schedule candidates.
Not a general optimal-control certificate or automatic contact discovery.
"""
import copy
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R
from coupled import Tree, vec
from tracking import Target
from support import Schedule
from multi_contact import impulse


def optimize(body, clip, options):
    if options.get('schema') != 'kineforge.horizon.v1': raise ValueError('invalid horizon schema')
    fixed=options.get('fixedBase',False); sub=options.get('substeps',4); blocks=options.get('controlBlocks',2); budget=options.get('maxEvaluations',30)
    if type(fixed)!=bool or type(sub)!=int or not 1<=sub<=16 or type(blocks)!=int or not 1<=blocks<=8 or type(budget)!=int or not 1<=budget<=60: raise ValueError('invalid horizon bounds')
    if clip['rig']!=body['rig']: raise ValueError('rig mismatch')
    if clip.get('contactSchedule') or any(clip.get('solverHints',{}).get(k) for k in ('contactSchedule','physicalContacts')): raise ValueError('convert legacy contacts to explicit scheduleCandidates')
    if 'supportSchedule' in options or 'stickingContacts' in options: raise ValueError('use scheduleCandidates, not legacy support fields')
    timeline=clip['timeline']; span=timeline['end']-timeline['start']
    if type(span)!=int or not 1<=span<=24 or not 1<=len(body['rig']['bones'])<=16: raise ValueError('horizon size exceeds experimental limit')
    tr=Tree(body,fixed); ref=Target(tr,clip); first=0 if fixed else 6; na=tr.d-first; intervals=ref.F-1
    if intervals>24 or blocks>intervals or na>8 or blocks*na>32 or tr.n>16: raise ValueError('horizon size exceeds experimental limit')
    tol=float(options.get('maxKeyErrorMm',10)); angle_tol=float(options.get('maxTargetAngleDeg',5)); replay_tol=float(options.get('maxReplayDifferenceMm',1)); replay_angle=float(options.get('maxReplayAngleDeg',.5)); torque=float(options.get('maxTorqueNm',50)); reg=float(options.get('effortWeight',1e-5)); smooth=float(options.get('smoothWeight',1e-5)); angle_weight=float(options.get('orientationWeightMPerRad',.1)); target_weight=float(options.get('targetWeight',.1));
    if not np.isfinite([tol,angle_tol,replay_tol,replay_angle,torque,reg,smooth,angle_weight,target_weight]).all() or min(tol,angle_tol,replay_tol,replay_angle,torque)<=0 or torque>1e5 or min(reg,smooth,angle_weight,target_weight)<0: raise ValueError('invalid horizon tolerance/weights')
    rollout_limit=options.get('maxRolloutsPerSchedule',200)
    if type(rollout_limit)!=int or not 2<=rollout_limit<=1000: raise ValueError('invalid rollout limit')
    raw=options.get('scheduleCandidates',[[]]); activation=options.get('supportActivationImpulse',False)
    if not isinstance(raw,list) or not 1<=len(raw)<=4 or type(activation)!=bool: raise ValueError('invalid schedule candidates')
    schedules=[Schedule(tr,ref,{'supportSchedule':x}) for x in raw]
    if any(len(s.items)>16 for s in schedules): raise ValueError('too many contact points')
    initial,_=ref.at(0)
    if 'initialVelocity' in options: initial=tr.state(initial.root,initial.local,vec(options['initialVelocity'],(tr.d,)))
    loads=options.get('externalLoads',[])
    # Conservative box bounds for vector actuators (never exceeds declared norm).
    caps=np.full(na,torque)
    for i,m in tr.models.items():
        sl=tr.sl[i]; w=sl.stop-sl.start
        if w and m['cap'] is not None: caps[sl.start-first:sl.stop-first]=np.minimum(caps[sl.start-first:sl.stop-first],m['cap']/np.sqrt(w))
    if np.any(caps<=0): raise ValueError('zero actuator cap not supported in horizon; use fixed joint')
    block_for=lambda k:min(blocks-1,k*blocks//intervals)
    def tau(x,k): return np.r_[np.zeros(first),x[block_for(min(k,intervals-1))]*caps]
    def replay(x,schedule,substeps):
        s=tr.state(initial.root,initial.local,initial.v); frames=[]; events=[]; old=[]; dt=1/ref.fps/substeps; res=0.; failure=None
        for k in range(intervals*substeps+1):
            active=schedule.active(k,substeps); ids=[c['id'] for c in active]
            try:
                if any(i not in old for i in ids) and activation:
                    kin=tr.kinematics(s)
                    for c in active:
                        pos,_,_=tr.point(kin,c['bone'],c['local'])
                        if np.linalg.norm(pos-c['anchor'])>1e-5:raise ValueError('impact activation anchor mismatch')
                    s,rep=impulse(tr,s,active);events.append(dict(frame=ref.start+k/substeps,**rep))
                old=ids
                contacts=[dict(bone=c['bone'],localPointM=c['local'],anchorM=c['anchor'],normal=c['normal'],mu=c['mu']) for c in active]
                act=tau(x,k//substeps); a,rep=tr.forward(s,act,loads,contacts);res=max(res,rep['residual'])
                if res>1e-6:raise ValueError('coupled equation residual exceeds 1e-6')
                if k%substeps==0:
                    kin=tr.kinematics(s);frames.append(dict(frame=ref.start+k//substeps,positions=kin['P'].tolist(),localQuaternionXYZW=R.from_matrix(s.local).as_quat().tolist(),velocity=s.v.tolist(),actuation=act.tolist(),contactForcesN=rep['contactForcesN'],activeContacts=ids,**tr.invariants(s)))
                if k==intervals*substeps:break
                # Midpoint with constant torque for this interval, strict sticking checks at all evaluations.
                vm=s.v+.5*dt*a;mid=tr.advance(s,s.v,.5*dt,vm);am,mrep=tr.forward(mid,act,loads,contacts);res=max(res,mrep['residual']);s=tr.advance(s,vm,dt,s.v+dt*am);_,end=tr.forward(s,act,loads,contacts);res=max(res,end['residual'])
                if res>1e-6:raise ValueError('coupled equation residual exceeds 1e-6')
            except (ValueError,np.linalg.LinAlgError,FloatingPointError) as ex:
                failure={'frame':ref.start+k/substeps,'reason':str(ex)};break
        return dict(frames=frames,activationImpacts=events,failure=failure,equationResidualMax=res)
    def arrays(run):
        p=np.array([f['positions'] for f in run['frames']]);l=R.from_quat(np.array([f['localQuaternionXYZW'] for f in run['frames']]).reshape(-1,4)).as_matrix().reshape(ref.F,tr.n,3,3);return p,l
    nr=len(ref.ids)*tr.n*3+2*ref.F*tr.n*3+blocks*na+max(0,blocks-1)*na
    def residual(run,x):
        if run['failure'] is not None or len(run['frames'])!=ref.F:return np.full(nr,100.+(ref.F-len(run['frames'])))
        p,l=arrays(run);ang=R.from_matrix((l@ref.L.swapaxes(-1,-2)).reshape(-1,3,3)).as_rotvec()
        return np.r_[(p[ref.ids]-ref.Y).ravel()/np.sqrt(len(ref.ids)*tr.n),np.sqrt(target_weight)*(p-ref.P).ravel()/np.sqrt(ref.F*tr.n),angle_weight*ang.ravel()/np.sqrt(ref.F*tr.n),np.sqrt(reg)*x.ravel()/np.sqrt(max(1,x.size)),np.sqrt(smooth)*np.diff(x,axis=0).ravel()/np.sqrt(max(1,(blocks-1)*na))]
    # Target inverse dynamics provides a seed, but is not an accepted replay.
    seed=np.zeros((blocks,na))
    for j in range(blocks):
        ks=[k for k in range(intervals) if block_for(k)==j];t=(ks[0]+ks[-1]+1)/2/ref.fps;s,a=ref.at(t);seed[j]=np.clip(tr.inverse(s,a,loads)[first:]/caps,-1,1)
    results=[]; candidates=[]
    class RolloutBudget(Exception): pass
    for si,schedule in enumerate(schedules):
        best=None; evaluated=0; converged=False
        def evaluate(flat):
            nonlocal best,evaluated
            if evaluated>=rollout_limit:raise RolloutBudget
            x=np.asarray(flat).reshape(blocks,na);run=replay(x,schedule,sub);r=residual(run,x);cost=float(r@r);evaluated+=1
            if run['failure'] is None and len(run['frames'])==ref.F and (best is None or cost<best[0]): best=(cost,x.copy(),run)
            return r
        z=np.zeros_like(seed);r0=evaluate(z.ravel());baseline=float(r0@r0);rs=evaluate(seed.ravel());seed_cost=float(rs@rs)
        if na:
            start=seed if seed_cost<baseline else z
            try:
                fit=least_squares(evaluate,start.ravel(),bounds=(-1,1),max_nfev=budget,ftol=1e-8,xtol=1e-8,gtol=1e-8,diff_step=1e-4)
                converged=bool(fit.success)
            except RolloutBudget:converged=False
        else:converged=True
        row=dict(scheduleIndex=si,evaluations=evaluated,zeroControlObjective=baseline,inverseSeedObjective=seed_cost,searchConverged=converged)
        if best is None:
            row.update(status='REVIEW_REQUIRED',flags=['NO_COMPLETE_REPLAY'],failure=replay(z,schedule,sub)['failure']);results.append(row);continue
        cost,x,coarse=best;fine=replay(x,schedule,2*sub);flags=[];row.update(objective=cost,controlTorquesNm=(x*caps).tolist(),fineFailure=fine['failure'],equationResidualMax=max(coarse['equationResidualMax'],fine['equationResidualMax']))
        if not converged:flags.append('SEARCH_NOT_CONVERGED')
        if fine['failure'] is not None or len(fine['frames'])!=ref.F:flags.append('REFINED_REPLAY_FAILED')
        else:
            p,l=arrays(fine);pc,lc=arrays(coarse);key=np.linalg.norm(p[ref.ids]-ref.Y,axis=-1)*1000; angle=np.degrees(R.from_matrix((l@ref.L.swapaxes(-1,-2)).reshape(-1,3,3)).magnitude()); drift=np.linalg.norm(p-pc,axis=-1)*1000;da=np.degrees(R.from_matrix((l@lc.swapaxes(-1,-2)).reshape(-1,3,3)).magnitude())
            row.update(keyMaxMm=float(key.max()),keyRmsMm=float(np.sqrt(np.mean(key**2))),targetAngleMaxDeg=float(angle.max()),replayDifferenceMaxMm=float(drift.max()),replayAngleMaxDeg=float(da.max()))
            if key.max()>tol:flags.append('KEY_DEVIATION_LIMIT')
            if angle.max()>angle_tol:flags.append('TARGET_ORIENTATION_DEVIATION')
            if drift.max()>replay_tol or da.max()>replay_angle:flags.append('STEP_REFINEMENT_LIMIT')
            import collision
            spec=collision.prepare(body)
            if spec is not None:
                from solver import fk
                row['exportCollision']=collision.report(l,p[:,0],tr.o,np.array(tr.p),spec,fk,ref.start);flags.extend(row['exportCollision']['flags'])
            else:row['exportCollision']='NOT_CONFIGURED'
        row.update(status='HORIZON_CANDIDATE' if not flags else 'REVIEW_REQUIRED',flags=flags);results.append(row);candidates.append((len(flags),cost,si,fine if fine['failure'] is None else coarse))
    chosen=min(candidates,key=lambda a:a[:3]) if candidates else None;solved=None;frames=[];events=[]
    if chosen is not None:
        _,_,si,run=chosen;frames=run['frames'];events=run['activationImpacts'];row=results[si]
        solved=copy.deepcopy(clip);solved['keyframes']=[dict(frame=f['frame'],positions=f['positions']) for f in frames];q=np.array([f['localQuaternionXYZW'] for f in frames]);
        for j in range(1,len(q)):q[j]*=np.where(np.sum(q[j]*q[j-1],axis=-1)<0,-1,1)[:,None]
        solved['localQuaternionXYZW']=q.tolist();solved['sourceKeys']=[{**k,'positions':frames[i]['positions']} for k,i in zip(ref.keys,ref.ids)];solved['solver']={'name':'KineForge 0.7 zeta horizon','status':row['status']};solved['notes']='Bounded physical shooting candidate; original keys and timing unchanged.'
    report=dict(status=results[chosen[2]]['status'] if chosen else 'REVIEW_REQUIRED',selectedScheduleIndex=chosen[2] if chosen else None,candidates=results,rootActuationAllowed=False,actuatorComponentCapsNm=caps.tolist(),maxRolloutsPerSchedule=rollout_limit,initialVelocitySource='explicit' if 'initialVelocity' in options else 'target derivative',controlBlockByInterval=[block_for(k) for k in range(intervals)],substeps=sub,verificationSubsteps=2*sub,limitations=['Small bounded direct shooting, local numerical search only.','Only supplied schedule alternatives; no automatic contact discovery or sliding tracking.','Strict minimum-norm sticking reactions can reject feasible redundant support.','Refinement and export proxy checks are sampled numerical checks, not continuous physics or game acceptance.'])
    return dict(schema='kineforge.horizon.result.v1',report=report,dof=tr.dof,frames=frames,activationImpacts=events,solved=solved)
