"""Build an experimental body profile from user-supplied rotation samples."""
import numpy as np
from scipy.spatial.transform import Rotation as R
import solver as s

def make_profile(rig,o,clips,provenance):
    p=s.rig_check(rig);n=len(p);neutral=clips[0][0];axes=np.zeros((n,3));length=np.full(n,.1)
    for i in range(n):
        ch=np.flatnonzero(p==i)
        if len(ch):
            j=ch[np.argmax(np.linalg.norm(o[ch],axis=-1))];length[i]=np.linalg.norm(o[j]);axes[i]=o[j]/length[i]
        elif i: length[i]=np.linalg.norm(o[i])*.5
    rel=neutral.transpose(0,2,1)@np.concatenate(clips);q=R.from_matrix(rel.reshape(-1,3,3)).as_quat().reshape(len(rel),n,4);ang=np.degrees(2*np.arctan2(np.linalg.norm(q[...,:3],axis=-1),np.abs(q[...,3])));tw=2*np.arctan2((q[...,:3]*axes).sum(-1),q[...,3]);tw=np.abs(np.degrees((tw+np.pi)%(2*np.pi)-np.pi));mass=np.maximum(length,.03)**2;mass/=mass.sum()
    return {'schema':'sam.body.experimental.v1','rig':rig,'offsets_m':o.tolist(),'neutralLocalQuaternionXYZW':R.from_matrix(neutral).as_quat().tolist(),'primaryAxes':axes.tolist(),'observedAngleEnvelopeDeg':np.minimum(180,np.max(ang,axis=0)+10).tolist(),'observedTwistEnvelopeDeg':np.minimum(180,np.max(tw,axis=0)+10).tolist(),'massFractions':mass.tolist(),'inertiaLengthM':length.tolist(),'massModel':'uncalibrated relative length^2 proxy, no absolute mass','keyWeights':[1.]*n,'collisionProxies':[],'provenance':provenance}
