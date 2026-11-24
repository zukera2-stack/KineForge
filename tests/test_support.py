import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from test_coupled import body
from test_tracking import target,opts
from tracking import track, Target
from coupled import Tree
from support import Schedule,allocate

def c(id='floor',bone='root',local=(0,0,0),anchor=(0,0,0),lo=1,hi=8,mu=.5,normal=(0,0,1)):
    return dict(id=id,bone=bone,localPointM=list(local),anchorM=list(anchor),startFrame=lo,endFrameExclusive=hi,mu=mu,normal=list(normal))
def options(items,**kw):return opts(fixedBase=False,supportSchedule=items,**kw)
def static(b):return target(b,False,angle_fn=lambda t,i:0)
class Support(unittest.TestCase):
    def test_static_weight(self):
        b=body(1);o=track(b,static(b),options([c()]));r=o['report'];self.assertEqual(r['status'],'TRACKING_CANDIDATE',r);self.assertLess(r['keyMaxMm'],1e-5)
        for f in o['frames']:
            np.testing.assert_allclose(f['contactForcesN'][0]['forceN'],[0,0,19.62],atol=1e-5);self.assertEqual(f['actuation'],[0]*6)
    def test_two_supports_and_joint_torque(self):
        b=body(2);d=static(b);o=track(b,d,options([c(),c('tip','link1',local=(1,0,0),anchor=(1,0,0))]));self.assertEqual(o['report']['status'],'TRACKING_CANDIDATE',o['report']);f=o['frames'][0];np.testing.assert_allclose(np.sum([x['forceN'] for x in f['contactForcesN']],axis=0),[0,0,29.43],atol=1e-4);self.assertAlmostEqual(f['contactForcesN'][1]['normalN'],4.905,places=5);self.assertAlmostEqual(f['actuation'][-1],-0.0,places=4);self.assertEqual(f['actuation'][:6],[0]*6)
    def test_redundant_supports(self):
        b=body(1);o=track(b,static(b),options([c('a'),c('b')]));self.assertEqual(o['report']['status'],'TRACKING_CANDIDATE',o['report']);self.assertAlmostEqual(sum(f['normalN'] for f in o['frames'][0]['contactForcesN']),19.62,places=5)
    def test_one_support_coupled_torque(self):
        b=body(2);d=static(b);o=track(b,d,options([c('tip','link1',local=(.5,0,0),anchor=(.5,0,0)),c()]));self.assertEqual(o['report']['status'],'TRACKING_CANDIDATE',o['report']);self.assertAlmostEqual(o['frames'][0]['contactForcesN'][0]['normalN'],9.81,places=4)
    def test_release_to_flight(self):
        b=body(1);d=static(b);o=track(b,d,options([c(hi=4)]));r=o['report'];self.assertEqual(r['framesIntegrated'],7,r)
        self.assertEqual([t['active'] for t in r['support']['transitions']],[['floor'],[]]);self.assertAlmostEqual(r['support']['transitions'][1]['seconds'],.15);self.assertAlmostEqual(o['frames'][-1]['positions'][0][2],-.5*9.81*.15**2,places=6);self.assertIn('KEY_DEVIATION_LIMIT',r['flags']);self.assertTrue(all(f['actuation']==[0]*6 for f in o['frames']))
    def test_switch_anchor_at_rest(self):
        b=body(1);o=track(b,static(b),options([c('a',hi=4),c('b',lo=4)]));self.assertEqual(o['report']['status'],'TRACKING_CANDIDATE',o['report']);self.assertEqual([t['active'] for t in o['report']['support']['transitions']],[['a'],['b']]);self.assertEqual(o['frames'][3]['contactForcesN'][0]['id'],'b')
    def test_recontact_zero_velocity(self):
        b=body(1,(0,0,0));o=track(b,static(b),options([c('a',hi=3),c('b',lo=5)]));self.assertEqual(o['report']['status'],'TRACKING_CANDIDATE',o['report']);self.assertEqual([t['active'] for t in o['report']['support']['transitions']],[['a'],[],['b']])
    def test_impact_rejected(self):
        b=body(1);d=target(b,False,root_fn=lambda t:[0,0,.5*9.81*(.15**2-t*t)]);o=track(b,d,options([c(lo=4)]));self.assertIn('INTEGRATION_FAILED',o['report']['flags']);self.assertIn('activation velocity',o['report']['failure']['reason']);self.assertIsNone(o['solved'])
    def test_unilateral_rejected(self):
        b=body(1,(0,0,9.81));o=track(b,static(b),options([c()]));self.assertIn('infeasible',o['report']['failure']['reason']);self.assertIsNone(o['solved'])
    def test_friction_rejected(self):
        b=body(1);o=track(b,static(b),options([c(mu=.2)],externalLoads=[dict(bone=0,forceN=[10,0,0])]));self.assertIn('infeasible',o['report']['failure']['reason'])
    def test_friction_inside(self):
        b=body(1);o=track(b,static(b),options([c(mu=.5)],externalLoads=[dict(bone=0,forceN=[4,0,0])]));self.assertEqual(o['report']['status'],'TRACKING_CANDIDATE',o['report']);np.testing.assert_allclose(o['frames'][0]['contactForcesN'][0]['forceN'],[-4,0,19.62],atol=1e-6)
    def test_anchor_rejected(self):
        b=body(1);o=track(b,static(b),options([c(anchor=(0,0,1))]));self.assertIn('drift',o['report']['failure']['reason'])
    def test_schedule_validation(self):
        b=body(1);d=static(b);self.assertRaisesRegex(ValueError,'half-open',track,b,d,options([c(lo=4,hi=4)]));self.assertRaisesRegex(ValueError,'unique',track,b,d,options([c(),c()]));self.assertRaisesRegex(ValueError,'friction',track,b,d,options([c(mu=-1)]))
    def test_determinism_and_no_mutation(self):
        b=body(1);d=static(b);op=options([c()]);original=copy.deepcopy((b,d,op));a=track(b,d,op);self.assertEqual((b,d,op),original);self.assertEqual(a,track(b,d,op));self.assertEqual(a['solved']['originalAuthorKeys'],d['originalAuthorKeys'])
    def test_moving_articulated_support(self):
        b=body(3);d=target(b,False);cs=[c(str(i),local=p,anchor=p) for i,p in enumerate([(-1,-1,0),(2,-1,0),(2,1,0),(-1,1,0)])];o=track(b,d,options(cs,substeps=4));self.assertEqual(o['report']['status'],'TRACKING_CANDIDATE',o['report']);self.assertLess(o['report']['keyMaxMm'],.1);self.assertGreater(max(abs(f['actuation'][-1]) for f in o['frames']),.1);self.assertTrue(all(f['actuation'][:6]==[0]*6 for f in o['frames']));self.assertLess(o['report']['support']['equationResidualMax'],1e-7)
    def test_nonvertical_normal(self):
        b=body(1,(-9.81,0,0));o=track(b,static(b),options([c(normal=(1,0,0))]));self.assertEqual(o['report']['status'],'TRACKING_CANDIDATE',o['report']);np.testing.assert_allclose(o['frames'][0]['contactForcesN'][0]['forceN'],[19.62,0,0],atol=1e-6)
    def test_verified_fallback(self):
        from unittest.mock import patch
        from scipy.optimize import OptimizeResult
        b=body(1);tr=Tree(b);ref=Target(tr,static(b));s=tr.state();sch=Schedule(tr,ref,options([c()]));
        with patch('support.minimize',return_value=OptimizeResult(success=False,x=np.zeros(8),message='forced optimizer failure')):
            a,act,sat,r=allocate(tr,s,np.zeros(tr.d),[],sch.items,sch,.01)
        self.assertEqual(r['allocationMethod'],'bounded_ls_feasible_fallback');np.testing.assert_allclose(a,0,atol=1e-7);self.assertEqual(act.tolist(),[0]*6)
    def test_actuator_limit(self):
        # Fixed base, constrained tip on a 1m lever. External downward force at tip must be supported by unilateral floor or actuator.
        b=body(2);b['jointModels']['link1']['actuatorLimitNm']=.01;t=Tree(b,True);s=t.state();d=target(b);ref=Target(t,d);sch=Schedule(t,ref,{'supportSchedule':[c('tip','link1',local=(1,0,0),anchor=(1,0,0))]});a,act,sat,r=allocate(t,s,np.zeros(t.d),[],sch.items,sch,.01);self.assertLessEqual(abs(act[0]),.0100001);self.assertAlmostEqual(a[0],0,places=8);self.assertGreater(r['contactForcesN'][0]['normalN'],4.8)
if __name__=='__main__':unittest.main()
