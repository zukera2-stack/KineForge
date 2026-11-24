"""Explicit sphere-pair collision constraints with conservative interframe bounds."""
import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from dynamics import rotation_vector


def prepare(body):
    if 'collisionModel' not in body:return None
    s=body['collisionModel'];names=body['rig']['bones'];p=body['rig']['parents'];o=np.asarray(body['offsets_m'],float)
    if not isinstance(s,dict) or set(s)-{'spheres','pairs','toleranceM','maxSubsteps','objectiveSubsteps'}:raise ValueError('invalid collision model')
    spheres=s.get('spheres');pairs=s.get('pairs')
    if not isinstance(spheres,list) or not 2<=len(spheres)<=512 or not isinstance(pairs,list) or not 1<=len(pairs)<=4096:raise ValueError('explicit spheres and pairs required')
    def number(x,lo,hi):
        if isinstance(x,bool) or not isinstance(x,(int,float)) or not math.isfinite(x) or not lo<=x<=hi:raise ValueError('invalid collision number')
        return float(x)
    tol=number(s.get('toleranceM',.001),0,.05);steps=s.get('maxSubsteps',64);obj=s.get('objectiveSubsteps',4)
    if type(steps)!=int or not 2<=steps<=256 or steps&(steps-1) or type(obj)!=int or not 1<=obj<=min(16,steps):raise ValueError('invalid substeps')
    ids=[];bones=[];centers=[];radii=[];arms=[];anc=[]
    for x in spheres:
        if not isinstance(x,dict) or set(x)!={'id','bone','centerLocalM','radiusM'} or not isinstance(x['id'],str) or not x['id'] or x['id'] in ids or x['bone'] not in names:raise ValueError('invalid sphere')
        i=names.index(x['bone']);c=np.asarray(x['centerLocalM'],float)
        if c.shape!=(3,) or not np.isfinite(c).all():raise ValueError('invalid sphere center')
        ids.append(x['id']);bones.append(i);centers.append(c);radii.append(number(x['radiusM'],1e-6,1e6));a=np.zeros(len(p));length=np.linalg.norm(c);j=i;chain=set()
        while j>=0:
            chain.add(j);a[j]=length
            if p[j]>=0:length+=np.linalg.norm(o[j])
            j=p[j]
        arms.append(a);anc.append(chain)
    ix=[];bounds=[];seen=set()
    for pair in pairs:
        if not isinstance(pair,list) or len(pair)!=2 or any(not isinstance(x,str) or x not in ids for x in pair):raise ValueError('invalid pair')
        a,b=sorted(ids.index(x) for x in pair)
        if a==b or bones[a]==bones[b] or (a,b) in seen:raise ValueError('duplicate or same-body pair')
        seen.add((a,b));ix.append((a,b));v=arms[a]+arms[b];v[list(anc[a]&anc[b])]=0;bounds.append(v)
    ix=np.asarray(ix);return dict(ids=ids,bones=np.asarray(bones),centers=np.asarray(centers),pairs=ix,radii=np.asarray(radii)[ix].sum(1),bounds=np.asarray(bounds),tol=tol,steps=steps,obj=obj)


def clearance_t(G,P,m):
    c=torch.as_tensor(m['centers'],dtype=G.dtype,device=G.device);i=m['bones'];X=P[:,i]+(G[:,i]@c[...,None]).squeeze(-1);a,b=m['pairs'].T
    return torch.linalg.vector_norm(X[:,a]-X[:,b],dim=-1)-torch.as_tensor(m['radii'],dtype=G.dtype,device=G.device)


def cost(L,G,P,o,p,m,fkt,exp):
    values=[clearance_t(G,P,m)];rv=rotation_vector(L[1:]@L[:-1].transpose(-1,-2))
    for k in range(1,m['obj']):
        u=k/m['obj'];g,x=fkt(exp(rv*u)@L[:-1],P[:-1,0]*(1-u)+P[1:,0]*u,o,p);values.append(clearance_t(g,x,m))
    z=torch.relu(-torch.cat(values,0));return 20*(z.square().mean()+z.max().square())


def report(L,root,o,p,m,fk,start):
    # Pair distance is Lipschitz with constant sum(angle * max lever arm).
    # Translation and shared ancestor motion cancel in relative distance.
    F,n=L.shape[:2];rv=R.from_matrix((L[1:]@L[:-1].swapaxes(-1,-2)).reshape(-1,3,3)).as_rotvec().reshape(F-1,n,3);angles=np.linalg.norm(rv,axis=-1);bound=angles@m['bounds'].T
    def clearance(l,r):
        g,x=fk(l,r,o,p);i=m['bones'];X=x[:,i]+(g[:,i]@m['centers'][...,None]).squeeze(-1);a,b=m['pairs'].T;return np.linalg.norm(X[:,a]-X[:,b],axis=-1)-m['radii']
    knots=clearance(L,root);minimum=np.minimum(knots[:-1],knots[1:]);witness=np.where(knots[:-1]<=knots[1:],0.,1.);steps=1;density=np.ones(F-1,dtype=int)
    while steps<m['steps']:
        active=((minimum>=-m['tol'])&(minimum-bound/(2*density[:,None])<-m['tol'])).any(1)
        if not active.any():break
        inds=np.flatnonzero(active);steps*=2
        for k in range(1,steps,2):
            u=k/steps;l=R.from_rotvec((rv[inds]*u).reshape(-1,3)).as_matrix().reshape(len(inds),n,3,3)@L[inds];c=clearance(l,root[inds]*(1-u)+root[inds+1]*u);better=c<minimum[inds];witness[inds]=np.where(better,u,witness[inds]);minimum[inds]=np.minimum(minimum[inds],c)
        density[inds]=steps
    lower=minimum-bound/(2*density[:,None]);bad=minimum<-m['tol'];uncertain=(~bad)&(lower<-m['tol']);flags=[]
    if bad.any():flags.append('COLLISION_PENETRATION')
    if uncertain.any():flags.append('COLLISION_INTERVAL_UNCERTIFIED')
    if (angles[:,np.any(m['bounds']>0,axis=0)]>=math.radians(179.5)).any():flags.append('COLLISION_ROTATION_AMBIGUOUS')
    rows=[]
    for k,(a,b) in enumerate(m['pairs']):
        i=int(np.argmin(minimum[:,k]));rows.append({'pair':[m['ids'][a],m['ids'][b]],'min_sampled_clearance_m':float(minimum[i,k]),'certified_lower_bound_m':float(lower[:,k].min()),'worst_frame':float(start+i+witness[i,k]),'penetrating_intervals':[int(start+j) for j in np.flatnonzero(bad[:,k])],'uncertified_intervals':[int(start+j) for j in np.flatnonzero(uncertain[:,k])]})
    return {'status':'REVIEW_REQUIRED' if flags else 'DECLARED_PROXY_PATH_PASS','flags':flags,'pairs':rows,'max_substeps_used':int(density.max()),'tolerance_m':m['tol'],'coverage':'explicit sphere pairs; linear root and shortest-arc local SO(3) between frames; no mesh, impulse or environment collision'}
