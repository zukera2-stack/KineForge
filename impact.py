"""Single declared point/static plane contact; SI units, no position projection."""
import numpy as np
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation
from coupled import Tree, vec


def point_impulse(tree, state, bone, local, normal, mu, restitution):
    """Newton restitution with Coulomb sticking or post-slip friction direction."""
    n=vec(normal,(3,)); mu=float(mu); e=float(restitution)
    if abs(np.linalg.norm(n)-1)>1e-8 or not np.isfinite([mu,e]).all() or not 0<=mu<=5 or not 0<=e<=1:raise ValueError('invalid plane normal/friction/restitution')
    k=tree.kinematics(state); _,J,_=tree.point(k,bone,local); M,_=tree.terms(state,k)
    eig=np.linalg.eigvalsh(M)
    if eig[0]<=0 or eig[-1]/eig[0]>1e12:raise ValueError('ill-conditioned mass matrix')
    Z=np.linalg.solve(M,J.T); W=J@Z; u=J@state.v; un=float(n@u); T=np.eye(3)-np.outer(n,n)
    if un>=-1e-10:return state,dict(mode='separating',impulseNs=[0.,0.,0.],normalBeforeMps=un,normalAfterMps=un,kineticLossJ=0.,momentumResidual=0.)
    target=-e*un; delta=target-un
    if n@W@n<=1e-12:raise ValueError('immovable contact')
    if mu==0:j=n*(delta/(n@W@n));mode='frictionless'
    else:
        js=np.linalg.lstsq(W,target*n-u,rcond=1e-12)[0]
        if np.linalg.norm(W@js-(target*n-u))<=1e-8 and n@js>=0 and np.linalg.norm(T@js)<=mu*(n@js)+1e-9:j=js;mode='stick'
        else:
            a=np.eye(3)[np.argmin(abs(n))];a=T@a;a/=np.linalg.norm(a);b=np.cross(n,a)
            def candidate(theta):
                t=np.cos(theta)*a+np.sin(theta)*b; direction=n-mu*t;den=n@W@direction
                if den<=1e-12:return None
                impulse=direction*delta/den; post=T@(u+W@impulse)
                return impulse,post,t,float(np.cross(t,post)@n)
            def fun(theta):
                c=candidate(theta);return c[3] if c is not None else np.nan
            grid=np.linspace(0,2*np.pi,129); roots=[];vals=[fun(x) for x in grid]
            for lo,hi,fl,fh in zip(grid[:-1],grid[1:],vals[:-1],vals[1:]):
                if not np.isfinite([fl,fh]).all():continue
                if abs(fl)<1e-10:roots.append(lo)
                if fl*fh<0:
                    try:roots.append(brentq(fun,lo,hi,xtol=1e-13))
                    except ValueError:pass
            valid=[]
            for x in roots:
                c=candidate(x)
                if c is None:continue
                imp,post,t,r=c
                if abs(r)<1e-7 and post@t>1e-9 and not any(np.linalg.norm(imp-y)<1e-7 for y in valid):valid.append(imp)
            if len(valid)!=1:raise ValueError('sliding impulse search unresolved or ambiguous')
            j=valid[0];mode='slide'
    v=state.v+Z@j; post=J@v; loss=float(.5*(state.v@M@state.v-v@M@v)); residual=float(np.linalg.norm(M@(v-state.v)-J.T@j))
    scale=max(1.,np.linalg.norm(j),abs(un));tol=1e-7*scale
    if n@j< -tol or np.linalg.norm(T@j)>mu*max(0.,n@j)+tol or abs(n@post-target)>tol or residual>tol or loss< -1e-8*max(1.,state.v@M@state.v):raise ValueError('impulse verification failed (cone/restitution/momentum/energy)')
    if mode=='slide' and (np.linalg.norm(np.cross(T@j,T@post))>1e-7*max(1.,np.linalg.norm(j)*np.linalg.norm(post)) or (T@j)@(T@post)>tol):raise ValueError('slip direction verification failed')
    out=tree.state(state.root,state.local,v)
    return out,dict(mode=mode,impulseNs=j.tolist(),normalBeforeMps=un,normalAfterMps=float(n@post),tangentBeforeMps=float(np.linalg.norm(T@u)),tangentAfterMps=float(np.linalg.norm(T@post)),kineticLossJ=max(0.,loss),momentumResidual=residual)


def simulate_contact(body,scenario):
    """Bracketed impacts and first-order resting contact; not full CCD."""
    if scenario.get('schema')!='kineforge.contact_simulation.v1':raise ValueError('invalid contact simulation schema')
    fixed=scenario.get('fixedBase',False)
    if type(fixed)!=bool:raise ValueError('invalid fixedBase')
    if scenario.get('stickingContacts') or scenario.get('supportSchedule'):raise ValueError('cannot combine contact modes')
    tr=Tree(body,fixed); init=scenario.get('initial',{}); q=init.get('localQuaternionXYZW'); L=None
    if q is not None:
        q=vec(q,(tr.n,4))
        if np.max(abs(np.linalg.norm(q,axis=1)-1))>1e-6:raise ValueError('invalid initial quaternion')
        L=Rotation.from_quat(q).as_matrix()
    s=tr.state(init.get('rootM'),L,init.get('velocity'));dt=float(scenario['dtSeconds']);steps=scenario['steps']
    if type(steps)!=int or not 1<=steps<=10000 or not 0<dt<=.02:raise ValueError('invalid steps/dt (max 0.02 s)')
    c=scenario['planeContact'];bone=c['bone'];local=vec(c['localPointM'],(3,));n=vec(c['normal'],(3,));origin=vec(c['planePointM'],(3,));mu=float(c['mu']);e=float(c['restitution']);gap_tol=float(scenario.get('gapToleranceM',1e-6))
    if abs(np.linalg.norm(n)-1)>1e-8 or not np.isfinite([mu,e,gap_tol]).all() or not 0<=mu<=5 or not 0<=e<=1 or not 1e-9<=gap_tol<=1e-3:raise ValueError('invalid contact parameters')
    act=vec(scenario.get('actuation',np.zeros(tr.d)),(tr.d,));loads=scenario.get('externalLoads',[])
    if not fixed and np.any(act[:6]!=0):raise ValueError('free root actuation forbidden')
    def point(s):
        pos,J,_=tr.point(tr.kinematics(s),bone,local);return float(n@(pos-origin)),float(n@(J@s.v))
    if point(s)[0]<-gap_tol:raise ValueError('initial point penetration')
    events=[];frames=[];failure=None;min_gap=point(s)[0]
    def resolve(s,coef,kind,time):
        out,r=point_impulse(tr,s,bone,local,n,mu,coef)
        if r['mode']!='separating':events.append(dict(seconds=time,kind=kind,**r))
        return out
    def step(s,t):
        rem=dt; elapsed=0.
        for _ in range(12):
            if rem<=1e-12:return s
            gap,vn=point(s)
            if gap< -gap_tol:raise ValueError('point penetration exceeds tolerance')
            if gap<=gap_tol and vn< -1e-8:
                s=resolve(s,e,'impact',t+elapsed);gap,vn=point(s)
            if gap<=gap_tol and abs(vn)<=1e-8:
                a,_=tr.forward(s,act,loads);pred=tr.state(s.root,s.local,s.v+rem*a)
                if point(pred)[1]< -1e-10:
                    corrected=resolve(pred,0.,'support_step',t+elapsed)
                    out=tr.advance(s,corrected.v,rem,corrected.v);tr.forward(out,act,loads)
                    if point(out)[0]<-gap_tol:raise ValueError('resting contact drift; decrease timestep')
                    return out
            end=tr.step(s,act,rem,loads);half=tr.step(s,act,rem/2,loads)
            if point(half)[0]<0:hi=rem/2
            elif point(end)[0]<0 or (point(end)[0]==0 and point(end)[1]<-1e-10):hi=rem
            else:return end
            lo=0.
            for _ in range(48):
                mid=(lo+hi)/2
                if point(tr.step(s,act,mid,loads))[0]>0:lo=mid
                else:hi=mid
                if hi-lo<1e-11:break
            hit=tr.step(s,act,hi,loads)
            if point(hit)[0]<-gap_tol or point(hit)[1]>=-1e-10:raise ValueError('grazing/unresolved contact')
            s=resolve(hit,e,'impact',t+elapsed+hi);elapsed+=hi;rem-=hi
        raise ValueError('impact event budget exceeded')
    for i in range(steps+1):
        gap,vn=point(s);min_gap=min(min_gap,gap)
        if gap< -gap_tol:failure={'seconds':i*dt,'reason':'point penetration exceeds tolerance'};break
        k=tr.kinematics(s);frames.append(dict(seconds=i*dt,rootM=s.root.tolist(),positionsM=k['P'].tolist(),localQuaternionXYZW=Rotation.from_matrix(s.local).as_quat().tolist(),velocity=s.v.tolist(),gapM=gap,normalVelocityMps=vn,**tr.invariants(s)))
        if i==steps:break
        try:s=step(s,i*dt)
        except (ValueError,np.linalg.LinAlgError) as ex:failure={'seconds':i*dt,'reason':str(ex)};break
    return dict(schema='kineforge.contact_simulation.result.v1',status='REVIEW_REQUIRED' if failure else 'CONTACT_SCENARIO_CANDIDATE',failure=failure,dof=tr.dof,frames=frames,events=events,minSampledGapM=min_gap,limitations=['One explicit point and static plane only; no multi-contact, mesh, automatic discovery or full CCD.','Newton restitution and post-slip Coulomb direction; finite search does not prove uniqueness.','First-order support step, no position projection; drift may cause rejection.','No tracking/optimal control or character acceptance.'])
