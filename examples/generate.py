"""Create a completely synthetic four-joint motion and physical profile."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial.transform import Rotation as R
import solver as s, calibrate as c

def generate(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    rig={'bones':['root','branch_a','branch_b','tip'],'parents':[-1,0,0,1],'coordinateSystem':'XYZ meters; Z up'}
    offsets=np.array([[0,0,0],[1,0,0],[0,0,1],[.7,0,0]],float)
    t=np.linspace(0,1,21);L=np.broadcast_to(np.eye(3),(len(t),4,3,3)).copy()
    L[:,0]=R.from_rotvec(np.c_[t*0,t*0,.5*np.sin(np.pi*t)]).as_matrix()
    L[:,1]=R.from_rotvec(np.c_[t*0,.4*np.sin(2*np.pi*t),t*0]).as_matrix()
    P=s.fk(L,np.c_[t*.1,t*0,t*0+1],offsets,np.array(rig['parents']))[1]
    body=c.make_profile(rig,offsets,[L],['synthetic analytic example'])
    body['physical']={'massKg':[3,2,1,1],'centerOfMassLocalM':[[.1,0,.1],[.3,0,0],[0,0,.2],[.1,0,0]],'inertiaCOMKgM2':[np.diag([.1,.2,.2]).tolist()]*4,'gravityMps2':[0,0,-9.81]}
    keys={'schema':'sam.keyposes.v1','rig':rig,'timeline':{'start':1,'end':21,'fps':30},'keys':[{'frame':int(i+1),'positions':P[i].tolist()} for i in [0,5,10,15,20]]}
    # Neutral-only reference: no withheld intermediate pose information.
    ref={'rig':rig,'phase':[0,1],'localQuaternionXYZW':np.tile([0,0,0,1],(2,4,1)).tolist()}
    for name,data in [('body',body),('keys',keys),('reference',ref)]:s.write(output/(name+'.json'),data)
if __name__=='__main__':generate(sys.argv[1])
