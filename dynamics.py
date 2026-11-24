"""Sampled Newton-Euler inverse dynamics for a rigid articulated body.
Assumes no external forces except gravity on each segment; computed root wrench
is the additional required external wrench, not necessarily feasible contact.
No contact force distribution, muscles, joint-limit dynamics or forward integration.
"""
import numpy as np
import torch

def rotation_vector(Q):
    # Q is relative SO(3). Smooth near zero; near pi ill-conditioned, flagged by diagnostics.
    v=torch.stack([Q[...,2,1]-Q[...,1,2],Q[...,0,2]-Q[...,2,0],Q[...,1,0]-Q[...,0,1]],-1)*.5
    sn=torch.sqrt(v.square().sum(-1)+1e-24);cs=((Q.diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)
    ordinary=v*(torch.atan2(sn,cs)/sn)[...,None]
    # At pi, skew vanishes. Recover axis from the symmetric part instead;
    # sign is intrinsically ambiguous exactly at pi (undersampled orientation).
    A=(Q+Q.transpose(-1,-2))*.5+torch.eye(3,dtype=Q.dtype,device=Q.device)
    k=A.diagonal(dim1=-2,dim2=-1).argmax(-1)
    axis=torch.gather(A,-1,k[...,None,None].expand(*k.shape,3,1)).squeeze(-1)
    axis=axis/torch.linalg.vector_norm(axis,dim=-1,keepdim=True).clamp_min(1e-12)
    axis=axis*torch.where((axis*v).sum(-1)<0,-1.,1.)[...,None]
    return torch.where((cs<-.9999)[...,None],axis*torch.atan2(sn,cs)[...,None],ordinary)

def evaluate(G,P,m,c,I,p,fps,gravity):
    X=P+(G@c[...,None]).squeeze(-1);J=G@I@G.transpose(-1,-2)
    w=rotation_vector(G[1:]@G[:-1].transpose(-1,-2))*fps
    H=((J[1:]+J[:-1])*.5@w[...,None]).squeeze(-1)
    a=torch.diff(X,n=2,dim=0)*fps**2
    F=m[None,:,None]*(a-gravity);N=torch.diff(H,dim=0)*fps
    # moment about individual joint, then accumulate children's wrenches at parent joints.
    A=[F[:,i] for i in range(len(p))];B=[N[:,i]+torch.linalg.cross(X[1:-1,i]-P[1:-1,i],A[i]) for i in range(len(p))]
    for i in range(len(p)-1,0,-1):
        j=p[i];B[j]=B[j]+B[i]+torch.linalg.cross(P[1:-1,i]-P[1:-1,j],A[i]);A[j]=A[j]+A[i]
    return {'com':X,'total_com':(X*m[None,:,None]).sum(1)/m.sum(),'force':F,'com_torque':N,'joint_force':torch.stack(A,1),'joint_torque':torch.stack(B,1),'omega':w}

def tensors(ph,n):
    from physical_body import validate
    m,c,I=validate(ph,n);g=np.array(ph.get('gravityMps2',[0,0,-9.81]),float)
    if g.shape!=(3,) or not np.isfinite(g).all():raise ValueError('invalid gravity')
    return tuple(torch.tensor(x,dtype=torch.float64) for x in (m,c,I,g))

def cost(G,P,params,p,fps,flight=()):
    m,c,I,g=params
    if len(P)<3:return P.sum()*0
    q=evaluate(G,P,m,c,I,p,fps,g)
    # N/kg scaled by dt² -> meters; N*m/kg scaled by dt² / 1m -> meters.
    # Use per-segment wrench, not sum of subtree torques (would overcount ancestors).
    # All terms physical units before normalization, but weight in objective remains heuristic.
    scale=m.sum()*fps**2
    z=(q['force']/scale).square().sum(-1).mean()+(q['com_torque']/scale).square().sum(-1).mean()
    if flight:
        ii=torch.tensor(flight,dtype=torch.long);z=z+((q['joint_force'][ii,0]/scale).square().sum(-1)+(q['joint_torque'][ii,0]/scale).square().sum(-1)).mean()
    return z

def report(G,P,ph,p,fps):
    m,c,I,g=tensors(ph,len(p));G=torch.as_tensor(G,dtype=torch.float64);P=torch.as_tensor(P,dtype=torch.float64)
    if len(P)<3:return {'status':'INSUFFICIENT_FRAMES'}
    q=evaluate(G,P,m,c,I,p,fps,g);mag=lambda x:torch.linalg.vector_norm(x,dim=-1);peak=lambda x:float(mag(x).max());rms=lambda x:float(torch.sqrt(mag(x).square().mean()));return {'status':'INVERSE_DYNAMICS_ESTIMATE','total_mass_kg':float(m.sum()),'gravity_mps2':g.tolist(),'root_force_peak_N':peak(q['joint_force'][:,0]),'root_force_rms_N':rms(q['joint_force'][:,0]),'root_torque_peak_Nm':peak(q['joint_torque'][:,0]),'internal_joint_torque_peak_Nm_without_contacts':peak(q['joint_torque'][:,1:]),'segment_force_rms_N':rms(q['force']),'segment_com_torque_rms_Nm':rms(q['com_torque']),'world_step_max_deg':peak(q['omega'])*180/np.pi/fps,'near_pi_warning':peak(q['omega'])*180/np.pi/fps>170,'total_com_first_m':q['total_com'][0].tolist(),'total_com_last_m':q['total_com'][-1].tolist(),'assumption':'gravity only per segment; unknown external wrench at root. No contact feasibility or forward simulation.'}

def write_csv(path,G,P,ph,p,fps,names,start):
    import csv
    m,c,I,g=tensors(ph,len(p));q=evaluate(torch.as_tensor(G,dtype=torch.float64),torch.as_tensor(P,dtype=torch.float64),m,c,I,p,fps,g)
    with open(path,'x',newline='') as f:
        w=csv.writer(f);w.writerow(['frame','seconds','bone','mass_kg']+[f'{key}_{ax}' for key in ['com_m','force_N','com_torque_Nm','subtree_force_N','subtree_torque_Nm'] for ax in 'xyz'])
        for i in range(1,len(P)-1):
            for j,name in enumerate(names):w.writerow([start+i,i/fps,name,float(m[j]),*q['com'][i,j].tolist(),*[v for key in ['force','com_torque','joint_force','joint_torque'] for v in q[key][i-1,j].tolist()]])
