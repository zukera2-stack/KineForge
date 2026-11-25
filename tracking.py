"""Physical replay of an authored kinematic target; no hidden floating-base actuation.
Feedforward inverse dynamics + configuration/velocity feedback. Not optimal control.
"""
import copy
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R, RotationSpline
from coupled import Tree, vec

class Target:
    def __init__(self, tree, clip):
        if clip.get('schema')!='sam.skeletal.solved.v1': raise ValueError('expected solved kinematic target')
        if clip['rig']['bones']!=tree.names or list(clip['rig']['parents'])!=list(tree.p): raise ValueError('rig mismatch')
        t=clip['timeline']; self.fps=float(t['fps']); self.start=t['start']; self.F=t['end']-self.start+1
        if type(self.F)!=int or not 2<=self.F<=10000 or not np.isfinite(self.fps) or self.fps<=0: raise ValueError('invalid timeline')
        if [k['frame'] for k in clip['keyframes']]!=list(range(self.start,t['end']+1)):raise ValueError('nonconsecutive target frames')
        self.P=vec([k['positions'] for k in clip['keyframes']],(self.F,tree.n,3)); q=vec(clip['localQuaternionXYZW'],(self.F,tree.n,4))
        if np.max(abs(np.linalg.norm(q,axis=-1)-1))>1e-6:raise ValueError('invalid target quaternion')
        L=R.from_quat(q.reshape(-1,4)).as_matrix().reshape(self.F,tree.n,3,3)
        self.L=L.copy();times=np.arange(self.F)/self.fps; self.end=times[-1];self.root=CubicSpline(times,self.P[:,0]);self.spl=[];self.tree=tree
        if tree.fixed and (np.max(abs(self.P[:,0]-self.P[0,0]))>1e-7 or np.max(abs(L[:,0]-L[0,0]))>1e-7):raise ValueError('fixed base target moves')
        for i in range(self.F):
            s=tree.state(self.P[i,0],L[i]);k=tree.kinematics(s)
            if np.max(abs(k['P']-self.P[i]))>1e-6:raise ValueError('target positions inconsistent with FK')
            if tree.passive(s)[2]:raise ValueError('target joint range exceeded')
        for i in range(tree.n):
            if i and tree.models[i]['type']=='hinge':
                ang=R.from_matrix(tree.N[i].T@L[:,i]).as_rotvec()@tree.models[i]['axis'];
                if np.max(abs(np.diff(ang)))>=np.pi:raise ValueError('ambiguous hinge target')
                self.spl.append(CubicSpline(times,ang))
            else:
                if np.max(R.from_matrix(L[1:,i]@L[:-1,i].transpose(0,2,1)).magnitude())>=np.pi-1e-6:raise ValueError('ambiguous target rotation')
                self.spl.append(RotationSpline(times,R.from_matrix(L[:,i])))
        self.keys=copy.deepcopy(clip['originalAuthorKeys']);ids=[x['frame'] for x in self.keys]
        if not ids or ids[0]!=self.start or ids[-1]!=t['end'] or any(type(x)!=int for x in ids) or any(a>=b for a,b in zip(ids,ids[1:])):raise ValueError('invalid original author keys')
        self.Y=vec([x['positions'] for x in self.keys],(len(ids),tree.n,3));self.ids=np.array(ids)-self.start
    def at(self,t):
        tr=self.tree;t=np.clip(t,0,self.end);L=tr.N.copy();v=np.zeros(tr.d);a=v.copy()
        for i,sp in enumerate(self.spl):
            if i and tr.models[i]['type']=='hinge':
                axis=tr.models[i]['axis'];L[i]=tr.N[i]@R.from_rotvec(axis*sp(t)).as_matrix();v[tr.sl[i]]=sp(t,1);a[tr.sl[i]]=sp(t,2)
            else:
                L[i]=sp(t).as_matrix()
                if i==0 and not tr.fixed: v[3:6]=L[i]@sp(t,1);a[3:6]=L[i]@sp(t,2)
                elif i and tr.models[i]['type']=='ball':v[tr.sl[i]]=tr.N[i].T@L[i]@sp(t,1);a[tr.sl[i]]=tr.N[i].T@L[i]@sp(t,2)
        if not tr.fixed:v[:3]=self.root(t,1);a[:3]=self.root(t,2)
        return tr.state(self.root(t),L,v),a

def desired_acceleration(tr,s,target,acc,kp,kd,root=False):
    err=np.zeros(tr.d)
    for i in range(1,tr.n):
        delta=R.from_matrix(tr.N[i].T@target.local[i]@s.local[i].T@tr.N[i]).as_rotvec();err[tr.sl[i]]=tr.B[i].T@delta
    if root and not tr.fixed:
        err[:3]=target.root-s.root;err[3:6]=R.from_matrix(target.local[0]@s.local[0].T).as_rotvec()
    return acc+kp*err+kd*(target.v-s.v)

def controller(tr,s,target,acc,kp,kd,loads):
    desired=desired_acceleration(tr,s,target,acc,kp,kd);k=tr.kinematics(s);M,h=tr.terms(s,k);passive,_,_=tr.passive(s);bias=h-passive-tr.external(k,loads)
    # Solve root acceleration from zero applied generalized root wrench. No root servo.
    if not tr.fixed:desired[:6]=np.linalg.solve(M[:6,:6],-bias[:6]-M[:6,6:]@desired[6:])
    tau=M@desired+bias
    if not tr.fixed:tau[:6]=0
    saturated=[]
    for i,m in tr.models.items():
        cap=m['cap'];sl=tr.sl[i];norm=np.linalg.norm(tau[sl])
        if cap is not None and norm>cap:tau[sl]*=cap/norm;saturated.append(tr.names[i])
    return tau,saturated

def track(body,clip,options):
    if options.get('schema')!='kineforge.tracking.v1':raise ValueError('invalid tracking schema')
    if options.get('stickingContacts') or clip.get('contactSchedule') or (clip.get('solverHints',{}).get('contactSchedule') or clip.get('solverHints',{}).get('physicalContacts')):raise ValueError('legacy contact tracking not implemented; explicitly convert to supportSchedule')
    fixed=options.get('fixedBase',False);sub=options.get('substeps',4);kp=float(options.get('kp',100));kd=float(options.get('kd',20));tol=float(options.get('maxKeyErrorMm',10));angtol=float(options.get('maxTargetAngleDeg',5))
    if type(fixed)!=bool or type(sub)!=int or not 1<=sub<=64 or not np.isfinite([kp,kd,tol,angtol]).all() or min(kp,kd)<0 or min(tol,angtol)<=0:raise ValueError('invalid tracking options')
    if clip['rig']!=body['rig']:raise ValueError('rig mismatch')
    tr=Tree(body,fixed);ref=Target(tr,clip);loads=options.get('externalLoads',[]);dt=1/ref.fps/sub
    if kp*dt*dt>0.5 or kd*dt>0.5:raise ValueError('tracking gains too high for explicit integration step')
    from support import Schedule, allocate
    schedule=Schedule(tr,ref,options) if 'supportSchedule' in options else None
    impact_enabled=options.get('supportActivationImpulse',False)
    if type(impact_enabled)!=bool or (impact_enabled and schedule is None):raise ValueError('supportActivationImpulse requires supportSchedule and boolean flag')
    impacts=[]
    s,_=ref.at(0); frames=[];sat=set();peak=0.;failure=None;gapmax=speedmax=resmax=accresmax=0.;transitions=[];previous=None;methods=set()
    def evaluate(state,time,active):
        target,acc=ref.at(time)
        if active:
            desired=desired_acceleration(tr,state,target,acc,kp,kd,root=True)
            return allocate(tr,state,desired,loads,active,schedule,dt)
        tau,ss=controller(tr,state,target,acc,kp,kd,loads);a,rep=tr.forward(state,tau,loads);return a,tau,ss,rep
    for k in range((ref.F-1)*sub+1):
        time=k*dt
        try:
            active=schedule.active(k,sub) if schedule is not None else [];ids=[x['id'] for x in active]
            if ids!=previous:
                if impact_enabled and active and any(x not in (previous or []) for x in ids):
                    from multi_contact import impulse
                    kin=tr.kinematics(s)
                    for item in active:
                        pos,J,_=tr.point(kin,item['bone'],item['local'])
                        if np.linalg.norm(pos-item['anchor'])>schedule.gap:raise ValueError('impact activation anchor mismatch: '+item['id'])
                    s,ir=impulse(tr,s,active)
                    impacts.append(dict(seconds=time,**ir))
                    kin=tr.kinematics(s)
                    for item in active:
                        _,J,_=tr.point(kin,item['bone'],item['local'])
                        if np.linalg.norm(J@s.v)>schedule.speed:raise ValueError('activation impact did not establish sticking: '+item['id'])
                transitions.append({'seconds':time,'active':ids});previous=ids
            a,tau,saturated,rep=evaluate(s,time,active);sat.update(saturated);peak=max(peak,float(np.max(abs(tau))))
            gapmax=max(gapmax,rep.get('supportGapMaxM',0.));speedmax=max(speedmax,rep.get('supportSpeedMaxMps',0.));resmax=max(resmax,rep.get('residual',0.));accresmax=max(accresmax,rep.get('contactAccelerationResidualMps2',0.));methods.update([rep['allocationMethod']] if 'allocationMethod' in rep else [])
            if k%sub==0:
                kin=tr.kinematics(s);frames.append(dict(frame=ref.start+k//sub,positions=kin['P'].tolist(),localQuaternionXYZW=R.from_matrix(s.local).as_quat().tolist(),velocity=s.v.tolist(),actuation=tau.tolist(),**tr.invariants(s),**rep))
            if k==(ref.F-1)*sub:break
            vm=s.v+.5*dt*a;mid=tr.advance(s,s.v,.5*dt,vm);am,tm,ss,mrep=evaluate(mid,time+.5*dt,active);sat.update(ss);peak=max(peak,float(np.max(abs(tm))));gapmax=max(gapmax,mrep.get('supportGapMaxM',0.));speedmax=max(speedmax,mrep.get('supportSpeedMaxMps',0.));resmax=max(resmax,mrep.get('residual',0.));accresmax=max(accresmax,mrep.get('contactAccelerationResidualMps2',0.));methods.update([mrep['allocationMethod']] if 'allocationMethod' in mrep else []);s=tr.advance(s,vm,dt,s.v+dt*am)
            # Check outgoing constraints at release; do not mask drift by switching off support.
            if active:
                kin=tr.kinematics(s)
                for item in active:
                    pos,J,_=tr.point(kin,item['bone'],item['local']);gap=float(np.linalg.norm(pos-item['anchor']));speed=float(np.linalg.norm(J@s.v));gapmax=max(gapmax,gap);speedmax=max(speedmax,speed)
                    if gap>schedule.gap or speed>schedule.speed:raise ValueError('outgoing support drift: '+item['id'])
        except (ValueError,np.linalg.LinAlgError) as ex:
            failure={'seconds':time,'reason':str(ex)};break
    flags=[];report={'status':'REVIEW_REQUIRED','framesIntegrated':len(frames),'framesRequested':ref.F,'failure':failure,'saturatedJoints':sorted(sat),'peakActuationComponent':peak,'fixedBase':fixed,'initialVelocitySource':'kinematic target derivative','rootActuationAllowed':False,'substeps':sub,'maxKeyErrorMm':tol}
    if schedule is not None:report['support']={'transitions':transitions,'gapMaxM':gapmax,'speedMaxMps':speedmax,'equationResidualMax':resmax,'contactAccelerationResidualMaxMps2':accresmax,'allocationMethods':sorted(methods),'schedule':copy.deepcopy(options['supportSchedule']),'gapToleranceM':schedule.gap,'speedToleranceMps':schedule.speed}
    if impact_enabled:report['activationImpacts']=impacts
    if failure:flags.append('INTEGRATION_FAILED')
    if sat:flags.append('ACTUATOR_SATURATION')
    out=None
    if len(frames)==ref.F and failure is None:
        P=np.array([f['positions'] for f in frames]);e=np.linalg.norm(P[ref.ids]-ref.Y,axis=-1)*1000;te=np.linalg.norm(P-ref.P,axis=-1)*1000;report.update(keyMaxMm=float(e.max()),keyRmsMm=float(np.sqrt(np.mean(e**2))),targetRmsMm=float(np.sqrt(np.mean(te**2))),targetMaxMm=float(te.max()),keyErrors=[{'frame':int(i+ref.start),'maxMm':float(x.max())} for i,x in zip(ref.ids,e)])
        L=R.from_quat(np.array([f['localQuaternionXYZW'] for f in frames]).reshape(-1,4)).as_matrix().reshape(ref.F,tr.n,3,3)
        angle=np.degrees(R.from_matrix((L@ref.L.swapaxes(-1,-2)).reshape(-1,3,3)).magnitude()).reshape(ref.F,tr.n)
        report['targetAngleMaxDeg']=float(angle.max());report['maxTargetAngleDeg']=angtol
        if e.max()>tol:flags.append('KEY_DEVIATION_LIMIT')
        if angle.max()>angtol:flags.append('TARGET_ORIENTATION_DEVIATION')
        if not flags:report['status']='TRACKING_CANDIDATE'
        import collision
        spec=collision.prepare(body)
        if spec is not None:
            from solver import fk
            L=R.from_quat(np.array([f['localQuaternionXYZW'] for f in frames]).reshape(-1,4)).as_matrix().reshape(ref.F,tr.n,3,3)
            report['exportCollision']=collision.report(L,P[:,0],tr.o,np.array(tr.p),spec,fk,ref.start)
            flags.extend(report['exportCollision']['flags'])
            if flags:report['status']='REVIEW_REQUIRED'
        else:report['exportCollision']='NOT_CONFIGURED'
        out=copy.deepcopy(clip);out['keyframes']=[{'frame':f['frame'],'positions':f['positions']} for f in frames];out['localQuaternionXYZW']=[f['localQuaternionXYZW'] for f in frames];out['sourceKeys']=[{**x,'positions':out['keyframes'][i]['positions']} for x,i in zip(ref.keys,ref.ids)];out['solver']={'name':'KineForge 0.7 epsilon tracking','diagnostics':report};out['notes']='Physical replay candidate, not game acceptance. Original author keys and timing preserved.'
    if out is not None:
        q=np.array(out['localQuaternionXYZW']);
        for i in range(1,len(q)):q[i]*=np.where(np.sum(q[i]*q[i-1],axis=-1)<0,-1,1)[:,None]
        out['localQuaternionXYZW']=q.tolist()
    report['flags']=flags;report['limitations']=['Prescribed initial velocity; fixed base is an external support.','Feedback tracking, not optimal control. Explicit scheduled sticking with optional inelastic activation impulses; sliding tracking and automatic contact discovery are not implemented.','Tracking acceptance is limited to key error and sampled dynamics; no full motion acceptance.']
    return dict(schema='kineforge.tracking.result.v1',report=report,dof=tr.dof,frames=frames,solved=out)
