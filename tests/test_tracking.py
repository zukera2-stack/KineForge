import copy, sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial.transform import Rotation as R
from tracking import track, Target, controller
from coupled import Tree
from test_coupled import body

def target(b, fixed=True, fps=20, frames=7, root_fn=None, angle_fn=None):
    tr=Tree(b,fixed);T=np.arange(frames)/fps;P=[];Q=[]
    for t in T:
        L=tr.N.copy()
        for i,m in tr.models.items():
            if m['type']=='hinge': L[i]=tr.N[i]@R.from_rotvec(m['axis']*(angle_fn(t,i) if angle_fn else .3*t+.2*t*t)).as_matrix()
        root=root_fn(t) if root_fn else [0,0,0];P.append(tr.kinematics(tr.state(root,L))['P'].tolist());Q.append(R.from_matrix(L).as_quat().tolist())
    k=[dict(frame=i+1,positions=p) for i,p in enumerate(P)]
    return dict(schema='sam.skeletal.solved.v1',rig=b['rig'],timeline=dict(start=1,end=frames,fps=fps),keyframes=k,originalAuthorKeys=[k[0],k[frames//2],k[-1]],localQuaternionXYZW=Q,sourceKeys=[k[0],k[frames//2],k[-1]],lockedFrames=[1,frames//2+1,frames])

def opts(**kw):return {**dict(schema='kineforge.tracking.v1',fixedBase=True,substeps=4,kp=100,kd=20,maxKeyErrorMm=1),**kw}

class Tracking(unittest.TestCase):
    def test_fixed_two_link_follow(self):
        b=body(3);d=target(b);o=track(b,d,opts());self.assertEqual(o['report']['status'],'TRACKING_CANDIDATE');self.assertLess(o['report']['keyMaxMm'],.01);self.assertEqual(o['solved']['originalAuthorKeys'],d['originalAuthorKeys']);self.assertEqual(o['solved']['timeline'],d['timeline']);self.assertEqual(o['solved']['sourceKeys'][1]['positions'],o['solved']['keyframes'][3]['positions'])
    def test_step_convergence(self):
        b=body(3);d=target(b);errors=[track(b,d,opts(substeps=i))['report']['keyMaxMm'] for i in (2,4,8)];self.assertLess(errors[1],errors[0]/3);self.assertLess(errors[2],errors[1]/3)
    def test_free_fall_no_root_servo(self):
        b=body(2);d=target(b,False,root_fn=lambda t:[t,0,3-.5*9.81*t*t],angle_fn=lambda t,i:0);o=track(b,d,opts(fixedBase=False));self.assertEqual(o['report']['status'],'TRACKING_CANDIDATE');self.assertLess(o['report']['keyMaxMm'],1e-6);self.assertTrue(all(f['actuation'][:6]==[0]*6 for f in o['frames']))
    def test_hover_rejected(self):
        b=body(2);d=target(b,False,root_fn=lambda t:[0,0,3],angle_fn=lambda t,i:0);o=track(b,d,opts(fixedBase=False));self.assertIn('KEY_DEVIATION_LIMIT',o['report']['flags']);self.assertAlmostEqual(o['report']['keyMaxMm'],441.45,places=5)
    def test_saturation(self):
        b=body(3);b['jointModels']['link1']['actuatorLimitNm']=.01;d=target(b);o=track(b,d,opts());self.assertIn('ACTUATOR_SATURATION',o['report']['flags']);self.assertEqual(o['report']['status'],'REVIEW_REQUIRED');self.assertTrue(all(abs(f['actuation'][0])<=.010000001 for f in o['frames']))
    def test_target_mutation_and_determinism(self):
        b=body();d=target(b);orig=copy.deepcopy(d);a=track(b,d,opts());self.assertEqual(d,orig);self.assertEqual(a,track(b,d,opts()))
    def test_reject_contacts(self):
        b=body();d=target(b);d['solverHints']={'physicalContacts':[{'id':'x'}]};self.assertRaisesRegex(ValueError,'contact tracking',track,b,d,opts())
    def test_reject_moving_fixed_base(self):
        b=body();d=target(b,False,root_fn=lambda t:[t,0,0]);self.assertRaisesRegex(ValueError,'fixed base',track,b,d,opts())
    def test_reject_inconsistent_coordinates(self):
        b=body();d=target(b);d['keyframes'][1]['positions'][2][0]+=.1;self.assertRaisesRegex(ValueError,'FK',track,b,d,opts())
    def test_reject_unstable_gains(self):
        b=body();d=target(b);self.assertRaisesRegex(ValueError,'gains',track,b,d,opts(kd=10000))
    def test_offneutral_hinge(self):
        b=body();b['neutralLocalQuaternionXYZW'][1]=R.from_rotvec([0,0,.3]).as_quat().tolist();d=target(b);o=track(b,d,opts());self.assertLess(o['report']['keyMaxMm'],.01)
    def test_free_internal_recoil(self):
        b=body(3,(0,0,0));d=target(b,False,angle_fn=lambda t,i:t*t);o=track(b,d,opts(fixedBase=False));self.assertGreater(np.linalg.norm(np.array(o['frames'][-1]['positions'][0])),1e-4);np.testing.assert_allclose(o['frames'][-1]['linearMomentumKgMps'],o['frames'][0]['linearMomentumKgMps'],atol=1e-4);self.assertTrue(all(f['actuation'][:6]==[0]*6 for f in o['frames']))
if __name__=='__main__':unittest.main()
