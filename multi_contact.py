"""Simultaneous perfectly inelastic point impacts and Coulomb velocity constraints.
The complementarity residual is solved numerically, then independently checked.
No position projection, no hidden base wrench, no uniqueness guarantee.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from coupled import Tree, vec


def disk(x,r):
    length=np.linalg.norm(x)
    return x*min(1.,max(0.,r)/max(length,1e-30))


def impulse(tree,state,items):
    if not 1<=len(items)<=16:raise ValueError('multi-contact count must be 1..16')
    k=tree.kinematics(state);M,_=tree.terms(state,k)
    eig=np.linalg.eigvalsh(M)
    if eig[0]<=0 or eig[-1]/eig[0]>1e12:raise ValueError('ill-conditioned mass matrix')
    rows=[];bases=[];mus=[];ids=[]
    for c in sorted(items,key=lambda c:c['id']):
        ident=c['id'];n=vec(c['normal'],(3,));mu=float(c['mu'])
        if not isinstance(ident,str) or not ident or ident in ids or abs(np.linalg.norm(n)-1)>1e-8 or not np.isfinite(mu) or not 0<=mu<=5:raise ValueError('invalid contact id/normal/friction')
        if float(c.get('restitution',0))!=0:raise ValueError('multi-contact restitution must be zero')
        a=np.eye(3)[np.argmin(abs(n))];a-=n*(a@n);a/=np.linalg.norm(a);B=np.stack([n,a,np.cross(n,a)])
        _,J,_=tree.point(k,c['bone'],vec(c['local'],(3,)));rows.append(B@J);bases.append(B);mus.append(mu);ids.append(ident)
    J=np.vstack(rows);Z=np.linalg.solve(M,J.T);W=J@Z;u=(J@state.v).reshape(-1,3);N=len(ids)
    # One velocity-to-impulse scale per point; normalize unknowns to velocity units.
    scale=np.maximum(np.linalg.norm(W.reshape(N,3,3*N),axis=(1,2)),1e-12)
    def unpack(x):return x.reshape(N,3)/scale[:,None]
    def residual(x):
        y=x.reshape(N,3);post=u+(W@unpack(x).ravel()).reshape(N,3);r=np.empty_like(y)
        for i in range(N):
            r[i,0]=y[i,0]-max(0.,y[i,0]-post[i,0]);r[i,1:]=y[i,1:]-disk(y[i,1:]-post[i,1:],mus[i]*y[i,0])
        return r.ravel()
    if np.min(u[:,0])>=-1e-10:
        x=np.zeros(3*N);iterations=0
    else:
        seed=np.zeros((N,3));seed[:,0]=np.maximum(-u[:,0],0);fit=least_squares(residual,seed.ravel(),ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=500);x=fit.x;iterations=fit.nfev
    p=unpack(x);v=state.v+Z@p.ravel();post=(J@v).reshape(N,3);scalev=max(1.,np.linalg.norm(u));vtol=1e-7*scalev;ptol=1e-7*max(1.,np.linalg.norm(p));forces=[]
    if not np.isfinite(v).all() or np.linalg.norm(residual(x))>vtol:raise ValueError('multi-contact complementarity unresolved')
    for i,ident in enumerate(ids):
        pn=p[i,0];pt=p[i,1:];vn=post[i,0];vt=post[i,1:];ts=np.linalg.norm(vt)
        if pn < -ptol or vn < -vtol or abs(pn*vn)>ptol*scalev or np.linalg.norm(pt)>mus[i]*max(pn,0.)+ptol:raise ValueError('multi-contact unilateral/cone verification failed')
        if ts>vtol and np.linalg.norm(pt+mus[i]*max(pn,0.)*vt/ts)>ptol:raise ValueError('multi-contact slip verification failed')
        mode='separating' if pn<=ptol else ('frictionless' if mus[i]==0 else ('slide' if ts>vtol else 'stick'))
        forces.append(dict(id=ident,mode=mode,impulseNs=(bases[i].T@p[i]).tolist(),normalBeforeMps=float(u[i,0]),normalAfterMps=float(vn),tangentAfterMps=float(ts)))
    gen=J.T@p.ravel();momentum=float(np.linalg.norm(M@(v-state.v)-gen));loss=float(.5*(state.v@M@state.v-v@M@v))
    if momentum>1e-8*max(1.,np.linalg.norm(gen)) or loss< -1e-8*max(1.,state.v@M@state.v):raise ValueError('multi-contact momentum/energy verification failed')
    return tree.state(state.root,state.local,v),dict(contacts=forces,kineticLossJ=max(0.,loss),momentumResidual=momentum,complementarityResidual=float(np.linalg.norm(residual(x))),iterations=iterations)


def simulate_multi(body,scenario):
    """Sampled plane events, first-order persistent contact, inelastic only."""
    if scenario.get('schema')!='kineforge.multi_contact_simulation.v1':raise ValueError('invalid multi-contact schema')
    fixed=scenario.get('fixedBase',False)
    if type(fixed)!=bool:raise ValueError('invalid fixedBase')
    if scenario.get('stickingContacts') or scenario.get('supportSchedule'):raise ValueError('cannot combine contact modes')
    tr=Tree(body,fixed);init=scenario.get('initial',{});q=init.get('localQuaternionXYZW');L=None
    if q is not None:
        q=vec(q,(tr.n,4))
        if np.max(abs(np.linalg.norm(q,axis=1)-1))>1e-6:raise ValueError('invalid initial quaternion')
        L=Rotation.from_quat(q).as_matrix()
    s=tr.state(init.get('rootM'),L,init.get('velocity'));dt=float(scenario['dtSeconds']);steps=scenario['steps'];tol=float(scenario.get('gapToleranceM',1e-6))
    if type(steps)!=int or not 1<=steps<=10000 or not 0<dt<=.02 or not 1e-9<=tol<=1e-3:raise ValueError('invalid step/gap parameters')
    items=[]
    for c in scenario['planeContacts']:
        n=vec(c['normal'],(3,));mu=float(c['mu']);e=float(c.get('restitution',0))
        if abs(np.linalg.norm(n)-1)>1e-8 or not 0<=mu<=5 or e!=0:raise ValueError('invalid contact parameters (inelastic only)')
        items.append(dict(id=c['id'],bone=c['bone'],local=vec(c['localPointM'],(3,)),normal=n,mu=mu,origin=vec(c['planePointM'],(3,))))
    if not 1<=len(items)<=16 or any(not isinstance(c['id'],str) or not c['id'] for c in items) or len({c['id'] for c in items})!=len(items):raise ValueError('invalid contact count or ids')
    items.sort(key=lambda c:c['id']);act=vec(scenario.get('actuation',np.zeros(tr.d)),(tr.d,));loads=scenario.get('externalLoads',[])
    if not fixed and np.any(act[:6]!=0):raise ValueError('free root actuation forbidden')
    def gaps(s):
        k=tr.kinematics(s);return np.array([c['normal']@(tr.point(k,c['bone'],c['local'])[0]-c['origin']) for c in items])
    if gaps(s).min()<-tol:raise ValueError('initial point penetration')
    events=[];frames=[];failure=None
    def resolve(s,active,time,kind):
        out,r=impulse(tr,s,active)
        if any(np.linalg.norm(c['impulseNs'])>1e-10 for c in r['contacts']):events.append(dict(seconds=time,kind=kind,**r))
        return out
    def step(s,t):
        rem=dt;elapsed=0.
        for _ in range(16):
            if rem<1e-11:return s
            g=gaps(s)
            if g.min()<-tol:raise ValueError('point penetration exceeds tolerance')
            active=[c for c,x in zip(items,g) if x<=tol]
            if active:
                s=resolve(s,active,t+elapsed,'impact')
                a,_=tr.forward(s,act,loads);pred=tr.state(s.root,s.local,s.v+rem*a);corrected=resolve(pred,active,t+elapsed,'support_step')
                out=tr.advance(s,corrected.v,rem,corrected.v)
                if gaps(out).min()<-tol or gaps(tr.advance(s,corrected.v,rem/2,corrected.v)).min()<-tol:raise ValueError('contact drift or new impact during support step; decrease timestep')
                tr.forward(out,act,loads);return out
            end=tr.step(s,act,rem,loads);half=tr.step(s,act,rem/2,loads)
            if gaps(half).min()<=0:hi=rem/2
            elif gaps(end).min()<=0:hi=rem
            else:return end
            lo=0.
            for _ in range(48):
                mid=(lo+hi)/2
                if gaps(tr.step(s,act,mid,loads)).min()>0:lo=mid
                else:hi=mid
                if hi-lo<1e-11:break
            s=tr.step(s,act,hi,loads);g=gaps(s)
            if g.min()<-tol:raise ValueError('unresolved crossing')
            active=[c for c,x in zip(items,g) if x<=tol];s=resolve(s,active,t+elapsed+hi,'impact');elapsed+=hi;rem-=hi
        raise ValueError('impact event budget exceeded')
    for i in range(steps+1):
        k=tr.kinematics(s);frames.append(dict(seconds=i*dt,positionsM=k['P'].tolist(),rootM=s.root.tolist(),velocity=s.v.tolist(),gapsM=gaps(s).tolist(),localQuaternionXYZW=Rotation.from_matrix(s.local).as_quat().tolist(),**tr.invariants(s)))
        if i==steps:break
        try:s=step(s,i*dt)
        except (ValueError,np.linalg.LinAlgError) as ex:failure=dict(seconds=i*dt,reason=str(ex));break
    return dict(schema='kineforge.multi_contact_simulation.result.v1',status='REVIEW_REQUIRED' if failure else 'CONTACT_SCENARIO_CANDIDATE',failure=failure,frames=frames,events=events,contactIds=[c['id'] for c in items],limitations=['Inelastic declared point/plane contacts only; finite nonlinear complementarity search, no uniqueness certificate.','Sampled crossing and first-order support; new contact during support can require smaller step; not full CCD.','No automatic contact discovery or character acceptance.'])
