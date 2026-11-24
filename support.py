"""Explicit, piecewise sticking support and coupled acceleration/actuator allocation.
No impulses, state projection, automatic contact discovery or sliding.
"""
import numpy as np
from scipy.optimize import minimize, linprog, lsq_linear
from coupled import vec
from contact import rays

class Schedule:
    def __init__(self,tr,ref,options):
        raw=options['supportSchedule']; self.items=[]; seen=set()
        if not isinstance(raw,list) or len(raw)>32:raise ValueError('supportSchedule must be a list of at most 32 contacts')
        self.gap=float(options.get('supportGapToleranceM',1e-4)); self.speed=float(options.get('supportSpeedToleranceMps',1e-3))
        if not np.isfinite([self.gap,self.speed]).all() or not 0<self.gap<=.005 or not 0<self.speed<=.05:raise ValueError('invalid support tolerance')
        for x in raw:
            name=x['id'];lo=x['startFrame'];hi=x['endFrameExclusive']
            if not isinstance(name,str) or not name or name in seen:raise ValueError('unique support id required')
            if type(lo)!=int or type(hi)!=int or not ref.start<=lo<hi<=ref.start+ref.F:raise ValueError('invalid half-open support interval')
            seen.add(name);n=vec(x['normal'],(3,));mu=float(x['mu']);D=rays(n,mu)
            self.items.append(dict(id=name,lo=lo-ref.start,hi=hi-ref.start,bone=tr.names.index(x['bone']),local=vec(x['localPointM'],(3,)),anchor=vec(x['anchorM'],(3,)),normal=n,mu=mu,D=D))
    def active(self,k,sub):return [x for x in self.items if x['lo']*sub<=k<x['hi']*sub]


def allocate(tr,s,desired,loads,items,schedule,dt):
    """Minimize acceleration error with exact sticking acceleration constraints.
    Unknowns: joint torques and nonnegative inscribed friction ray coefficients.
    Return actual acceleration, applied torques, forces, independently checked residuals.
    """
    k=tr.kinematics(s); M,h=tr.terms(s,k); passive,_,bad=tr.passive(s)
    if bad:raise ValueError('joint range exceeded: '+','.join(bad))
    eig=np.linalg.eigvalsh(M)
    if eig[0]<=0 or eig[-1]/eig[0]>1e12:raise ValueError('singular or ill-conditioned mass matrix')
    bias=h-passive-tr.external(k,loads); first=0 if tr.fixed else 6; nact=tr.d-first
    weight=max(1.,float(tr.m.sum())*max(1.,np.linalg.norm(tr.g)))
    B=[np.eye(tr.d)[:,first:]*weight]; Js=[]; bs=[]; gaps=[]; speeds=[]; spans=[]; end=nact
    for c in items:
        pos,J,b=tr.point(k,c['bone'],c['local']); gap=pos-c['anchor']; v=J@s.v
        gn=float(np.linalg.norm(gap)); vn=float(np.linalg.norm(v));gaps.append(gn);speeds.append(vn)
        if gn>schedule.gap or vn>schedule.speed:raise ValueError('support drift or nonzero activation velocity: '+c['id'])
        # Bounded acceleration stabilization of small numerical drift; no position/velocity projection.
        w=min(20.,.1/dt); bs.append(b+2*w*v+w*w*gap); Js.append(J)
        B.append(J.T@c['D'].T*weight);spans.append(slice(end,end+len(c['D'])));end+=len(c['D'])
    B=np.concatenate(B,axis=1);Z=np.linalg.solve(M,B);a0=np.linalg.solve(M,-bias);J=np.concatenate(Js);b=np.concatenate(bs)
    A=J@Z; y=-b-J@a0
    # Remove redundant equality rows only if RHS lies in the same range.
    U,S,V=np.linalg.svd(A,full_matrices=False);rank=int(np.sum(S>max(A.shape)*max(S[0] if len(S) else 0.,1.)*1e-12));u=U[:,:rank]
    if np.linalg.norm(y-u@(u.T@y))>1e-7*max(1.,np.linalg.norm(y)):raise ValueError('inconsistent support constraints')
    C=V[:rank];d=(u.T@y)/S[:rank] if rank else np.zeros(0)
    bounds=[(None,None)]*nact+[(0,None)]*(end-nact); caps=[]
    for i,m in tr.models.items():
        sl=tr.sl[i];cap=m['cap'];q=slice(sl.start-first,sl.stop-first)
        if cap is None or q.start==q.stop:continue
        cap=cap/weight
        if q.stop-q.start==1:bounds[q.start]=(-cap,cap)
        else:caps.append((q,cap))
    # Equality-constrained convex quadratic objective, plus convex norm caps.
    scale=max(1.,np.linalg.norm(desired-a0));H=Z/scale;target=(desired-a0)/scale;reg=1e-9
    def obj(x):r=H@x-target;return .5*(r@r+reg*(x@x))
    def jac(x):return H.T@(H@x-target)+reg*x
    cons=[]
    if rank:cons.append({'type':'eq','fun':lambda x:C@x-d,'jac':lambda x:C})
    for sl,cap in caps:
        def fun(x,sl=sl,cap=cap):return cap-np.linalg.norm(x[sl])
        def grad(x,sl=sl):
            g=np.zeros(end);norm=np.linalg.norm(x[sl]);g[sl]=-x[sl]/max(norm,1e-30);return g
        cons.append({'type':'ineq','fun':fun,'jac':grad})
    lp=linprog(np.zeros(end),A_eq=C if rank else None,b_eq=d if rank else None,bounds=bounds,method='highs')
    if not lp.success:raise ValueError(('support allocation infeasible: ' if lp.status==2 else 'support feasibility not certified: ')+lp.message)
    opt=minimize(obj,lp.x,jac=jac,bounds=bounds,constraints=cons,method='SLSQP',options={'ftol':1e-12,'maxiter':200})
    method='SLSQP'
    if (not opt.success or not np.isfinite(opt.x).all()) and not caps:
        # Ill-conditioned light segments can defeat SLSQP. Bounded least squares
        # with progressively stronger equality penalty is a fallback, not an
        # equality certificate; the independent residual checks below remain mandatory.
        lo=np.array([-np.inf if l is None else l for l,u in bounds]);hi=np.array([np.inf if u is None else u for l,u in bounds])
        if np.all(lo<hi):
            for penalty in (1e3,1e5,1e7):
                row_scale=max(1.,np.linalg.norm(y));D=np.vstack((H,np.sqrt(reg)*np.eye(end),penalty*A/row_scale));rhs=np.r_[target,np.zeros(end),penalty*y/row_scale]
                col=np.maximum(np.linalg.norm(D,axis=0),1e-12);fit=lsq_linear(D/col,rhs,bounds=(lo*col,hi*col),method='bvls',tol=1e-11,max_iter=300);candidate=fit.x/col
                if fit.success and np.isfinite(candidate).all() and np.linalg.norm(A@candidate-y)<=1e-7*max(1.,np.linalg.norm(b)):
                    opt=fit;opt.x=candidate;method='bounded_ls_feasible_fallback';break
    if not opt.success or not np.isfinite(opt.x).all():raise ValueError('support optimization not certified: '+opt.message)
    x=opt.x;acc=a0+Z@x;act=np.zeros(tr.d);act[first:]=x[:nact]*weight;forces=[];general=np.zeros(tr.d);sat=[]
    for c,sl in zip(items,spans):
        f=x[sl]@c['D']*weight;n=float(f@c['normal']);t=float(np.linalg.norm(f-n*c['normal']))
        if np.min(x[sl])< -1e-8 or n< -1e-7 or t>c['mu']*max(n,0)+1e-7:raise ValueError('friction cone violation')
        _,jp,_=tr.point(k,c['bone'],c['local']);general+=jp.T@f;forces.append(dict(id=c['id'],forceN=f.tolist(),normalN=n,tangentN=t))
    for i,m in tr.models.items():
        cap=m['cap'];v=float(np.linalg.norm(act[tr.sl[i]]))
        if cap is not None:
            if v>cap+1e-7:raise ValueError('actuator cap violation')
            if v>=cap-1e-7:sat.append(tr.names[i])
    residual=float(np.linalg.norm(M@acc+bias-act-general));cres=float(np.linalg.norm(J@acc+b))
    if residual>1e-7*max(1.,np.linalg.norm(bias),np.linalg.norm(M@acc)) or cres>1e-7*max(1.,np.linalg.norm(b)):raise ValueError('support equation residual exceeded')
    return acc,act,sat,dict(residual=residual,contactAccelerationResidualMps2=cres,contactForcesN=forces,massCondition=float(eig[-1]/eig[0]),supportGapMaxM=max(gaps),supportSpeedMaxMps=max(speeds),desiredAccelerationError=float(np.linalg.norm(acc-desired)),allocationIterations=int(opt.nit),allocationMethod=method)
