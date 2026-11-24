import sys,unittest,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from coupled import Tree
from impact import point_impulse,simulate_contact
from test_coupled import body


def scenario(z=0.,vx=0.,vz=0.,mu=.5,e=0.,dt=.01,steps=20):
    return dict(schema='kineforge.contact_simulation.v1',dtSeconds=dt,steps=steps,initial={'rootM':[0,0,z],'velocity':[vx,0,vz,0,0,0]},planeContact=dict(bone=0,localPointM=[0,0,0],normal=[0,0,1],planePointM=[0,0,0],mu=mu,restitution=e))

class Impact(unittest.TestCase):
    def test_elastic(self):
        t=Tree(body(1));s=t.state(v=[1,0,-3,0,0,0]);o,r=point_impulse(t,s,0,[0,0,0],[0,0,1],0,1)
        np.testing.assert_allclose(o.v,[1,0,3,0,0,0]);np.testing.assert_allclose(r['impulseNs'],[0,0,12]);self.assertAlmostEqual(r['kineticLossJ'],0);np.testing.assert_array_equal(o.root,s.root);np.testing.assert_array_equal(o.local,s.local)
    def test_inelastic(self):
        t=Tree(body(1));o,r=point_impulse(t,t.state(v=[0,0,-3,0,0,0]),0,[0,0,0],[0,0,1],0,0);np.testing.assert_allclose(o.v,0);self.assertAlmostEqual(r['kineticLossJ'],9)
    def test_slide(self):
        t=Tree(body(1));o,r=point_impulse(t,t.state(v=[3,0,-2,0,0,0]),0,[0,0,0],[0,0,1],.25,0);self.assertEqual(r['mode'],'slide');np.testing.assert_allclose(o.v,[2.5,0,0,0,0,0],atol=1e-10);np.testing.assert_allclose(r['impulseNs'],[-1,0,4],atol=1e-10)
    def test_stick(self):
        t=Tree(body(1));o,r=point_impulse(t,t.state(v=[.2,0,-2,0,0,0]),0,[0,0,0],[0,0,1],.5,0);self.assertEqual(r['mode'],'stick');np.testing.assert_allclose(o.v,0,atol=1e-12)
    def test_separation(self):
        t=Tree(body(1));s=t.state(v=[3,0,2,0,0,0]);o,r=point_impulse(t,s,0,[0,0,0],[0,0,1],1,1);self.assertEqual(r['mode'],'separating');np.testing.assert_array_equal(o.v,s.v)
    def test_offcenter_impulse(self):
        t=Tree(body(1));s=t.state(v=[0,0,-3,0,0,0]);o,r=point_impulse(t,s,0,[1,0,0],[0,0,1],0,0);j=3/10.5
        np.testing.assert_allclose(o.v,[0,0,-3+j/2,0,-j/.1,0],atol=1e-12);self.assertAlmostEqual(r['impulseNs'][2],j);self.assertLess(r['momentumResidual'],1e-12)
    def test_coupled_impulse(self):
        t=Tree(body(2));s=t.state(v=[0,0,-2,0,0,0,0]);o,r=point_impulse(t,s,1,[1,0,0],[0,0,1],0,.2);self.assertGreater(abs(o.v[-1]),.01);self.assertGreater(abs(o.v[2]-s.v[2]),.01);p0=t.invariants(s);p1=t.invariants(o);np.testing.assert_allclose(np.array(p1['linearMomentumKgMps'])-p0['linearMomentumKgMps'],r['impulseNs'],atol=1e-12);self.assertGreaterEqual(r['kineticLossJ'],0)
    def test_invalid(self):
        t=Tree(body(1));s=t.state();
        for n,mu,e in [([0,0,2],0,0),([0,0,1],-1,0),([0,0,1],1,1.1)]:
            with self.assertRaises(ValueError):point_impulse(t,s,0,[0,0,0],n,mu,e)

class ContactScenario(unittest.TestCase):
    def test_drop_bounce(self):
        r=simulate_contact(body(1),scenario(z=.1,e=.5,steps=20));self.assertEqual(r['status'],'CONTACT_SCENARIO_CANDIDATE',r['failure']);i=r['events'][0];self.assertAlmostEqual(i['seconds'],np.sqrt(.2/9.81),8);self.assertAlmostEqual(i['normalAfterMps'],-.5*i['normalBeforeMps'],10);self.assertGreater(r['frames'][-1]['gapM'],0)
    def test_drop_stop(self):
        r=simulate_contact(body(1),scenario(z=.1,steps=30));self.assertEqual(r['status'],'CONTACT_SCENARIO_CANDIDATE',r['failure']);self.assertLess(abs(r['frames'][-1]['gapM']),1e-8);np.testing.assert_allclose(r['frames'][-1]['velocity'],0,atol=1e-9)
    def test_slide_to_stop(self):
        r=simulate_contact(body(1),scenario(vx=1,steps=30));self.assertEqual(r['status'],'CONTACT_SCENARIO_CANDIDATE',r['failure']);self.assertAlmostEqual(r['frames'][10]['velocity'][0],1-.5*9.81*.1,10);self.assertAlmostEqual(r['frames'][-1]['velocity'][0],0,10);self.assertEqual(r['events'][-1]['mode'],'stick');self.assertLess(r['frames'][-1]['energyJ'],1e-9)
    def test_slide_step_convergence(self):
        err=[]
        for dt in [.02,.01,.005]:
            r=simulate_contact(body(1),scenario(vx=2,dt=dt,steps=round(.1/dt)));self.assertEqual(r['status'],'CONTACT_SCENARIO_CANDIDATE',r['failure']);err.append(abs(r['frames'][-1]['rootM'][0]-(.2-.5*.5*9.81*.1**2)))
        self.assertLess(err[1],err[0]*.51);self.assertLess(err[2],err[1]*.51)
    def test_airborne(self):
        r=simulate_contact(body(1),scenario(z=2,steps=10));self.assertFalse(r['events']);self.assertAlmostEqual(r['frames'][-1]['gapM'],2-.5*9.81*.1**2,12)
    def test_liftoff(self):
        d=scenario();d['externalLoads']=[dict(bone=0,forceN=[0,0,25])];r=simulate_contact(body(1),d);self.assertFalse(r['events']);self.assertGreater(r['frames'][-1]['gapM'],0)
    def test_initial_penetration_and_root_guard(self):
        with self.assertRaisesRegex(ValueError,'penetration'):simulate_contact(body(1),scenario(z=-.001))
        d=scenario();d['actuation']=[0,0,1,0,0,0]
        with self.assertRaisesRegex(ValueError,'root actuation'):simulate_contact(body(1),d)
    def test_determinism(self):
        d=scenario(z=.01,vx=.3);self.assertEqual(simulate_contact(body(1),d),simulate_contact(body(1),d))

if __name__=='__main__':unittest.main()
