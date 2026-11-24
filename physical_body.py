"""Generic point-mass moments and rigid-body parameter validation."""
import numpy as np

def moments(points,weights):
 points=np.asarray(points,float);weights=np.asarray(weights,float)
 if points.ndim!=2 or points.shape[1]!=3 or weights.shape!=(len(points),) or not np.isfinite(points).all() or not np.isfinite(weights).all() or (weights<0).any() or weights.sum()<=0:raise ValueError('invalid mass cloud')
 m=weights.sum();c=weights@points/m;x=points-c;Q=(x.T*weights)@x;I=np.trace(Q)*np.eye(3)-Q
 return float(m),c,I

def validate(ph,n):
 m=np.array(ph['massKg'],float);c=np.array(ph['centerOfMassLocalM'],float);I=np.array(ph['inertiaCOMKgM2'],float)
 if m.shape!=(n,) or c.shape!=(n,3) or I.shape!=(n,3,3) or not all(np.isfinite(x).all() for x in [m,c,I]) or (m<=0).any():raise ValueError('invalid physical body dimensions/mass')
 if not np.allclose(I,I.swapaxes(-1,-2),atol=1e-10,rtol=0):raise ValueError('nonsymmetric inertia')
 e=np.linalg.eigvalsh(I)
 if (e< -1e-9).any() or (e[:,2]>e[:,0]+e[:,1]+1e-9).any():raise ValueError('nonphysical inertia')
 return m,c,I

