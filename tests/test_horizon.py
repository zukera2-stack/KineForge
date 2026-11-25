import copy, unittest
import numpy as np
from scipy.spatial.transform import Rotation as R
from test_coupled import body
from test_tracking import target
from horizon import optimize

def opts(**kw):
    return {**dict(schema='kineforge.horizon.v1',fixedBase=True,controlBlocks=2,substeps=2,maxEvaluations=20,maxKeyErrorMm=1,maxReplayDifferenceMm=.2,effortWeight=0,smoothWeight=0),**kw}


def land(b):
    # A falling body, hits plane at frame 3, remains there. Explicit initial velocity is zero.
    d=target(b,False,fps=10,frames=5,root_fn=lambda t:[0,0,max(0,.1962-.5*9.81*t*t)])
    def c(start):return [dict(id='floor',bone='root',localPointM=[0,0,0],anchorM=[0,0,0],normal=[0,0,1],mu=.5,startFrame=start,endFrameExclusive=6)]
    return d,c

class Horizon(unittest.TestCase):
    def test_global_control_improves(self):
        b=body(3,(0,0,0));d=target(b,fps=20,frames=5,angle_fn=lambda t,i:.1*t+.3*t*t);o=optimize(b,d,opts());r=o['report']['candidates'][0];self.assertEqual(o['report']['status'],'HORIZON_CANDIDATE');self.assertLess(r['objective'],r['zeroControlObjective']/100);self.assertLess(r['keyMaxMm'],.1)
    def test_preserve_and_repeat(self):
        b=body(2,(0,0,0));d=target(b,frames=4);original=copy.deepcopy(d);o=optimize(b,d,opts());self.assertEqual(d,original);self.assertEqual(o, optimize(b,d,opts()));self.assertEqual(o['solved']['originalAuthorKeys'],d['originalAuthorKeys']);self.assertEqual(o['solved']['timeline'],d['timeline'])
    def test_ballistic_no_actuators(self):
        b=body(1);d=target(b,False,frames=5,root_fn=lambda t:[t,0,2-.5*9.81*t*t]);o=optimize(b,d,opts(fixedBase=False));self.assertEqual(o['report']['status'],'HORIZON_CANDIDATE');self.assertTrue(all(f['actuation']==[0]*6 for f in o['frames']))
    def test_hover_rejected(self):
        b=body(1);d=target(b,False,frames=5,root_fn=lambda t:[0,0,2]);o=optimize(b,d,opts(fixedBase=False));self.assertIn('KEY_DEVIATION_LIMIT',o['report']['candidates'][0]['flags'])
    def test_schedule_choice(self):
        b=body(1);d,c=land(b);o=optimize(b,d,opts(fixedBase=False,scheduleCandidates=[c(2),c(3)],supportActivationImpulse=True,initialVelocity=[0]*6));self.assertEqual(o['report']['selectedScheduleIndex'],1);self.assertEqual(o['report']['status'],'HORIZON_CANDIDATE');self.assertEqual(len(o['activationImpacts']),1);self.assertEqual(o['report']['candidates'][0]['flags'],['NO_COMPLETE_REPLAY'])
    def test_cap(self):
        b=body(2);b['jointModels']['link1']['actuatorLimitNm']=.01;d=target(b,frames=4);o=optimize(b,d,opts());self.assertTrue(all(abs(f['actuation'][0])<=.01000001 for f in o['frames']));self.assertEqual(o['report']['status'],'REVIEW_REQUIRED')
    def test_invalid_size(self):
        b=body(2);d=target(b,frames=26);self.assertRaisesRegex(ValueError,'size',optimize,b,d,opts())
    def test_rollout_budget(self):
        b=body(3);d=target(b,frames=4);o=optimize(b,d,opts(maxRolloutsPerSchedule=2));r=o['report']['candidates'][0];self.assertEqual(r['evaluations'],2);self.assertIn('SEARCH_NOT_CONVERGED',r['flags'])
    def test_legacy_schedule_rejected(self):
        b=body(2);d=target(b,frames=4);self.assertRaisesRegex(ValueError,'scheduleCandidates',optimize,b,d,opts(supportSchedule=[]))
    def test_fixed_base_support_no_snap(self):
        b=body(1);d,c=land(b);o=optimize(b,d,opts(fixedBase=False,scheduleCandidates=[c(2)],supportActivationImpulse=True,initialVelocity=[0]*6));self.assertIsNone(o['solved']);self.assertEqual(o['report']['status'],'REVIEW_REQUIRED')
    def test_refinement_gate(self):
        b=body(3);d=target(b,frames=4);o=optimize(b,d,opts(maxReplayDifferenceMm=1e-9));self.assertIn('STEP_REFINEMENT_LIMIT',o['report']['candidates'][0]['flags'])
if __name__=='__main__':unittest.main()
