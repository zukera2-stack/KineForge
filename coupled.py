"""Coupled rigid-tree dynamics in spatial velocity coordinates (SI).
Dense reference backend. Not a collision/impact or automatic contact solver.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation
from physical_body import validate
from joints import prepare


def vec(x, shape):
    x=np.asarray(x,float)
    if x.shape!=shape or not np.isfinite(x).all(): raise ValueError('invalid vector/array shape or nonfinite value')
    return x

@dataclass
class State:
    root: np.ndarray
    local: np.ndarray
    v: np.ndarray

class Tree:
    """Root translation/world rotation, then parent-relative neutral-basis joint rates.
    Hinge 1 DOF, ball 3, fixed 0; unspecified joints are ball. Fixed base has no root DOFs.
    Local angular velocities are spatial (left increments) in the neutral parent basis.
    """
    def __init__(self, body, fixed_base=False):
        self.names=body['rig']['bones']; self.n=len(self.names); self.p=body['rig']['parents']
        if not 1<=self.n<=256 or len(set(self.names))!=self.n or len(self.p)!=self.n or self.p[0]!=-1 or any(type(j)!=int or not 0<=j<i for i,j in enumerate(self.p[1:],1)): raise ValueError('invalid tree')
        self.o=vec(body['offsets_m'],(self.n,3)); q=vec(body['neutralLocalQuaternionXYZW'],(self.n,4))
        if np.max(abs(np.linalg.norm(q,axis=1)-1))>1e-6: raise ValueError('invalid neutral quaternion')
        self.N=Rotation.from_quat(q).as_matrix(); self.m,self.c,self.I=validate(body['physical'],self.n)
        self.g=vec(body['physical'].get('gravityMps2',[0,0,-9.81]),(3,)); self.fixed=bool(fixed_base)
        models={m['i']:m for m in prepare(body)}; self.models={}; self.B={}; self.sl={}; self.dof=[]
        if not self.fixed: self.dof=['root.translation.'+a for a in 'xyz']+['root.rotation.'+a for a in 'xyz']
        for i in range(1,self.n):
            m=models.get(i,{'type':'ball','k':0.,'c':0.,'cap':None,'max':np.pi}); self.models[i]=m
            B=m['axis'][:,None] if m['type']=='hinge' else np.eye(3) if m['type']=='ball' else np.zeros((3,0)); self.B[i]=B
            k=len(self.dof); self.sl[i]=slice(k,k+B.shape[1]); self.dof.extend(self.names[i]+'.'+a for a in (['hinge'] if B.shape[1]==1 else list('xyz') if B.shape[1]==3 else []))
        self.d=len(self.dof)
        if self.d==0: raise ValueError('no dynamic degrees of freedom')
    def state(self,root=None,local=None,v=None):
        root=vec(np.zeros(3) if root is None else root,(3,)); L=vec(self.N if local is None else local,(self.n,3,3)); v=vec(np.zeros(self.d) if v is None else v,(self.d,))
        if np.max(abs(L@L.transpose(0,2,1)-np.eye(3)))>1e-7 or np.max(abs(np.linalg.det(L)-1))>1e-7: raise ValueError('nonrotation')
        for i,m in self.models.items():
            rv=Rotation.from_matrix(self.N[i].T@L[i]).as_rotvec()
            if m['type']=='fixed' and np.linalg.norm(rv)>1e-7: raise ValueError('fixed joint displaced')
            if m['type']=='hinge' and np.linalg.norm(rv-m['axis']*(rv@m['axis']))>1e-7: raise ValueError('hinge off axis')
        return State(root.copy(),L.copy(),v.copy())
    def kinematics(self,s):
        """Jacobians plus bias accelerations at generalized acceleration zero."""
        n,d=self.n,self.d; G=np.zeros((n,3,3)); P=np.zeros((n,3)); Jp=np.zeros((n,3,d)); Jw=Jp.copy(); a=np.zeros((n,3)); al=a.copy()
        G[0]=s.local[0];P[0]=s.root
        if not self.fixed:Jp[0,:,:3]=np.eye(3);Jw[0,:,3:6]=np.eye(3)
        for i,j in enumerate(self.p[1:],1):
            r=G[j]@self.o[i]; P[i]=P[j]+r; G[i]=G[j]@s.local[i]; basis=G[j]@self.N[i]@self.B[i]
            Jp[i]=Jp[j]+np.cross(Jw[j].T,r).T; Jw[i]=Jw[j]; Jw[i,:,self.sl[i]]=basis
            w=Jw[j]@s.v; wr=basis@s.v[self.sl[i]]
            a[i]=a[j]+np.cross(al[j],r)+np.cross(w,np.cross(w,r)); al[i]=al[j]+np.cross(w,wr)
        w=Jw@s.v; r=(G@self.c[...,None]).squeeze(-1); X=P+r
        Jc=Jp+np.cross(Jw.transpose(0,2,1),r[:,None,:]).transpose(0,2,1)
        ac=a+np.cross(al,r)+np.cross(w,np.cross(w,r)); I=G@self.I@G.transpose(0,2,1)
        return dict(G=G,P=P,Jp=Jp,Jw=Jw,a=a,al=al,w=w,X=X,Jc=Jc,ac=ac,I=I)
    def terms(self,s,k=None):
        k=self.kinematics(s) if k is None else k; J,W,I=k['Jc'],k['Jw'],k['I']
        M=np.einsum('n,nai,naj->ij',self.m,J,J)+np.einsum('nai,nab,nbj->ij',W,I,W)
        F=self.m[:,None]*(k['ac']-self.g); T=(I@k['al'][...,None]).squeeze(-1)+np.cross(k['w'],(I@k['w'][...,None]).squeeze(-1))
        h=np.einsum('nai,na->i',J,F)+np.einsum('nai,na->i',W,T)
        return (M+M.T)/2,h
    def passive(self,s):
        f=np.zeros(self.d); E=0.; violations=[]
        for i,m in self.models.items():
            if m['type']=='fixed':continue
            rv=Rotation.from_matrix(self.N[i].T@s.local[i]).as_rotvec(); q=self.B[i].T@rv; v=s.v[self.sl[i]]
            if m['type']=='hinge':
                if q[0]<m['lo']-1e-9 or q[0]>m['hi']+1e-9:violations.append(self.names[i])
                q=q-m['rest']
            elif np.linalg.norm(q)>m['max']+1e-9:violations.append(self.names[i])
            f[self.sl[i]]=-m['k']*q-m['c']*v; E+=.5*m['k']*float(q@q)
        return f,E,violations
    def point(self,k,i,local):
        if type(i)!=int or not 0<=i<self.n: raise ValueError('invalid body index')
        r=k['G'][i]@vec(local,(3,)); J=k['Jp'][i]+np.cross(k['Jw'][i].T,r).T; w=k['w'][i]
        return k['P'][i]+r,J,k['a'][i]+np.cross(k['al'][i],r)+np.cross(w,np.cross(w,r))
    def external(self,k,loads):
        f=np.zeros(self.d)
        for item in loads:
            i=item['bone']; _,J,_=self.point(k,i,item.get('localPointM',[0,0,0])); f+=J.T@vec(item.get('forceN',[0,0,0]),(3,))+k['Jw'][i].T@vec(item.get('torqueNm',[0,0,0]),(3,))
        return f
    def inverse(self,s,acc,loads=()):
        k=self.kinematics(s); M,h=self.terms(s,k); passive,_,_=self.passive(s)
        return M@vec(acc,(self.d,))+h-passive-self.external(k,loads)
    def forward(self,s,act,loads=(),contacts=()):
        act=vec(act,(self.d,)); k=self.kinematics(s); M,h=self.terms(s,k); passive,_,bad=self.passive(s)
        if bad: raise ValueError('joint range exceeded: '+','.join(bad))
        for i,m in self.models.items():
            if m['cap'] is not None and np.linalg.norm(act[self.sl[i]])>m['cap']+1e-9:raise ValueError('actuator limit exceeded: '+self.names[i])
        eig=np.linalg.eigvalsh(M)
        if eig[0]<=0 or eig[-1]/eig[0]>1e12:raise ValueError('singular or ill-conditioned mass matrix; verify inertia/DOFs')
        rhs=act+passive+self.external(k,loads)-h; acc=np.linalg.solve(M,rhs); forces=[]
        if contacts:
            rows=[]; bias=[]; normals=[]
            for item in contacts:
                pos,J,a=self.point(k,item['bone'],item['localPointM']); n=vec(item['normal'],(3,)); mu=float(item['mu'])
                if abs(np.linalg.norm(n)-1)>1e-8 or not np.isfinite(mu) or mu<0:raise ValueError('invalid contact normal/friction')
                # Constant sticking anchors. Never create an impulse or teleport to an anchor.
                if np.linalg.norm(pos-vec(item['anchorM'],(3,)))>1e-5 or np.linalg.norm(J@s.v)>1e-4:raise ValueError('sticking contact drift/velocity; impact or projection not implemented')
                rows.append(J);bias.append(a);normals.append((n,mu))
            J=np.concatenate(rows); b=np.concatenate(bias); Z=np.linalg.solve(M,J.T); S=J@Z
            f=np.linalg.lstsq(S,-b-J@acc,rcond=1e-12)[0]; acc+=Z@f
            if np.linalg.norm(J@acc+b)>1e-7*max(1.,np.linalg.norm(b)):raise ValueError('inconsistent sticking constraints')
            for x,(n,mu) in zip(f.reshape(-1,3),normals):
                normal=x@n
                if normal< -1e-7 or np.linalg.norm(x-normal*n)>mu*max(0.,normal)+1e-7:raise ValueError('minimum-norm sticking reaction outside friction cone; no feasible reaction certified')
                forces.append(x.tolist())
            rhs+=J.T@f
        return acc,dict(residual=float(np.linalg.norm(M@acc-rhs)),contactForcesN=forces,massCondition=float(eig[-1]/eig[0]))
    def advance(self,s,v,dt,next_v):
        root=s.root.copy(); L=s.local.copy()
        if not self.fixed:root+=dt*v[:3];L[0]=Rotation.from_rotvec(dt*v[3:6]).as_matrix()@L[0]
        for i,B in self.B.items():
            if B.shape[1]:L[i]=self.N[i]@Rotation.from_rotvec(dt*(B@v[self.sl[i]])).as_matrix()@self.N[i].T@L[i]
        return self.state(root,L,next_v)
    def step(self,s,act,dt,loads=(),contacts=()):
        if not np.isfinite(dt) or dt<=0:raise ValueError('positive dt required')
        a,_=self.forward(s,act,loads,contacts); vm=s.v+.5*dt*a; mid=self.advance(s,s.v,.5*dt,vm); am,_=self.forward(mid,act,loads,contacts)
        out=self.advance(s,vm,dt,s.v+dt*am); self.forward(out,act,loads,contacts) # reject limit/drift before returning
        return out
    def invariants(self,s):
        k=self.kinematics(s); v=k['Jc']@s.v; M,_=self.terms(s,k); _,spring,_=self.passive(s); com=np.average(k['X'],axis=0,weights=self.m)
        momentum=self.m[:,None]*v; H=np.cross(k['X']-com,momentum)+(k['I']@k['w'][...,None]).squeeze(-1)
        return dict(energyJ=float(.5*s.v@M@s.v-np.sum(self.m*(k['X']@self.g))+spring),comM=com.tolist(),linearMomentumKgMps=momentum.sum(0).tolist(),angularMomentumCOMKgM2ps=H.sum(0).tolist())


def simulate(body, scenario):
    """Run a bounded, constant-input experiment; never mutate authored keys."""
    if scenario.get('schema')!='kineforge.simulation.v1':raise ValueError('invalid simulation schema')
    fixed=scenario.get('fixedBase',False)
    if type(fixed)!=bool:raise ValueError('fixedBase must be boolean')
    tree=Tree(body,fixed); initial=scenario.get('initial',{}); q=initial.get('localQuaternionXYZW'); L=None
    if q is not None:
        q=vec(q,(tree.n,4))
        if np.max(abs(np.linalg.norm(q,axis=1)-1))>1e-6:raise ValueError('invalid initial quaternion')
        L=Rotation.from_quat(q).as_matrix()
    s=tree.state(initial.get('rootM'),L,initial.get('velocity')); dt=float(scenario['dtSeconds']); steps=scenario['steps']
    if type(steps)!=int or not 1<=steps<=10000 or not np.isfinite(dt) or dt<=0:raise ValueError('invalid steps/dt')
    act=vec(scenario.get('actuation',np.zeros(tree.d)),(tree.d,)); loads=scenario.get('externalLoads',[]); contacts=scenario.get('stickingContacts',[])
    if not fixed and not scenario.get('allowRootActuation',False) and np.any(act[:6]!=0):raise ValueError('free root actuation requires explicit allowRootActuation; use externalLoads for external forces')
    frames=[]
    for i in range(steps+1):
        a,report=tree.forward(s,act,loads,contacts);k=tree.kinematics(s)
        frames.append(dict(seconds=i*dt,rootM=s.root.tolist(),positionsM=k['P'].tolist(),localQuaternionXYZW=Rotation.from_matrix(s.local).as_quat().tolist(),velocity=s.v.tolist(),acceleration=a.tolist(),**tree.invariants(s),**report))
        if i<steps:s=tree.step(s,act,dt,loads,contacts)
    return dict(schema='kineforge.simulation.result.v1',status='INTEGRATED_EXPLICIT_SCENARIO',dof=tree.dof,dtSeconds=dt,frames=frames,limitations=['Dense rigid-tree dynamics with constant applied generalized actuation and explicit world loads.','Fixed sticking contacts only; no impact, sliding, hard-stop response, or automatic collision/contact discovery.','No keyframe tracking, trajectory optimization or continuous physical acceptance in this mode.'])
