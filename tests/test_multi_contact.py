import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from coupled import Tree
from multi_contact import impulse,simulate_multi
from tracking import track
from test_coupled import body
from test_support import c,options,static
from test_tracking import target

def points(mu=0.):
    return [dict(id=i,bone=0,local=[x,0,0],normal=[0,0,1],mu=mu) for i,x in [('a',-1),('b',1)]]
def scenario(z=.1,vx=0,vz=0,mu=.5,steps=30,dt=.01):
    return dict(schema='kineforge.multi_contact_simulation.v1',initial=dict(rootM=[0,0,z],velocity=[vx,0,vz,0,0,0]),steps=steps,dtSeconds=dt,planeContacts=[dict(id=x['id'],bone=x['bone'],localPointM=x['local'],normal=x['normal'],mu=mu,planePointM=[0,0,0]) for x in points(mu)])
class Multi(unittest.TestCase):
    def test_symmetric_impact(self):
        t=Tree(body(1));s=t.state(v=[0,0,-3,0,0,0]);o,r=impulse(t,s,points());np.testing.assert_allclose(o.v,0,atol=1e-8);np.testing.assert_allclose([c['impulseNs'] for c in r['contacts']],[[0,0,3],[0,0,3]],atol=1e-8);self.assertAlmostEqual(r['kineticLossJ'],9,places=8);np.testing.assert_array_equal(o.root,s.root);np.testing.assert_array_equal(o.local,s.local)
    def test_asymmetric_analytic(self):
        b=body(1);b['physical']['centerOfMassLocalM'][0]=[.25,0,0];t=Tree(b);s=t.state(v=[0,0,-3,0,0,0]);o,r=impulse(t,s,points());np.testing.assert_allclose(o.v,0,atol=1e-7);np.testing.assert_allclose([c['impulseNs'][2] for c in r['contacts']],[2.25,3.75],atol=1e-7)
    def test_energy_and_impulse_balance(self):
        t=Tree(body(1));s=t.state(v=[3,0,-2,0,0,0]);o,r=impulse(t,s,points(.25));a=t.invariants(s);b=t.invariants(o);np.testing.assert_allclose(np.array(b['linearMomentumKgMps'])-a['linearMomentumKgMps'],np.sum([c['impulseNs'] for c in r['contacts']],axis=0),atol=1e-8);self.assertAlmostEqual(r['kineticLossJ'],6.75,places=7)
    def test_order(self):
        t=Tree(body(1));s=t.state(v=[.2,0,-2,0,0,0]);a,r=impulse(t,s,points(.5));b,q=impulse(t,s,points(.5)[::-1]);np.testing.assert_array_equal(a.v,b.v);self.assertEqual(r,q)
    def test_friction(self):
        t=Tree(body(1));s=t.state(v=[3,0,-2,0,0,0]);o,r=impulse(t,s,points(.25));np.testing.assert_allclose(o.v,[2.5,0,0,0,0,0],atol=1e-7);self.assertTrue(all(c['mode']=='slide' for c in r['contacts']))
    def test_stick(self):
        t=Tree(body(1));o,r=impulse(t,t.state(v=[.2,0,-2,0,0,0]),points(.5));np.testing.assert_allclose(o.v,0,atol=1e-7)
    def test_separating(self):
        t=Tree(body(1));s=t.state(v=[0,0,2,0,0,0]);o,r=impulse(t,s,points());np.testing.assert_array_equal(s.v,o.v);self.assertTrue(all(c['mode']=='separating' for c in r['contacts']))
    def test_one_separates(self):
        t=Tree(body(1));s=t.state(v=[0,0,0,0,2,0]);o,r=impulse(t,s,points());self.assertEqual(r['contacts'][0]['mode'],'separating');self.assertGreater(r['contacts'][0]['normalAfterMps'],0);self.assertAlmostEqual(r['contacts'][1]['normalAfterMps'],0,places=7)
    def test_redundant(self):
        t=Tree(body(1));cs=points();cs.append(dict(cs[0],id='duplicate_location'));o,r=impulse(t,t.state(v=[0,0,-3,0,0,0]),cs);np.testing.assert_allclose(o.v,0,atol=1e-7);self.assertAlmostEqual(sum(x['impulseNs'][2] for x in r['contacts']),6,places=7)
    def test_coupled(self):
        t=Tree(body(2));cs=points();cs[1]['bone']=1;o,r=impulse(t,t.state(v=[0,0,-2,0,0,0,0]),cs);self.assertLess(r['momentumResidual'],1e-9);self.assertGreaterEqual(r['kineticLossJ'],0);self.assertTrue(all(x['normalAfterMps']>=-1e-7 for x in r['contacts']))
    def test_invalid(self):
        t=Tree(body(1));s=t.state(v=[0,0,-2,0,0,0]);cs=points();cs[0]['restitution']=.5
        with self.assertRaisesRegex(ValueError,'restitution'):impulse(t,s,cs)
        with self.assertRaisesRegex(ValueError,'id'):impulse(t,s,[points()[0]]*2)
    def test_drop(self):
        r=simulate_multi(body(1),scenario());self.assertEqual(r['status'],'CONTACT_SCENARIO_CANDIDATE',r['failure']);self.assertAlmostEqual(r['events'][0]['seconds'],np.sqrt(.2/9.81),8);np.testing.assert_allclose(r['frames'][-1]['velocity'],0,atol=1e-7);self.assertEqual(len(r['events'][0]['contacts']),2)
    def test_sliding(self):
        r=simulate_multi(body(1),scenario(z=0,vx=1));self.assertEqual(r['status'],'CONTACT_SCENARIO_CANDIDATE',r['failure']);self.assertAlmostEqual(r['frames'][10]['velocity'][0],1-.5*9.81*.1,places=7);np.testing.assert_allclose(r['frames'][-1]['velocity'],0,atol=1e-7)
    def test_no_root_wrench(self):
        d=scenario();d['actuation']=[0,0,1,0,0,0]
        with self.assertRaisesRegex(ValueError,'root actuation'):simulate_multi(body(1),d)
    def test_activation(self):
        b=body(1);d=target(b,False,root_fn=lambda t:[0,0,.5*9.81*(.15**2-t*t)]);cs=[c(str(i),lo=4,local=p,anchor=p) for i,p in enumerate([(-1,0,0),(1,0,0)])];op=options(cs,supportActivationImpulse=True);o=track(b,d,op);self.assertIsNone(o['report']['failure'],o['report']);self.assertEqual(len(o['report']['activationImpacts']),1);self.assertEqual(len(o['report']['activationImpacts'][0]['contacts']),2);np.testing.assert_allclose(o['frames'][3]['velocity'],0,atol=1e-7);self.assertEqual(o['solved']['originalAuthorKeys'],d['originalAuthorKeys']);self.assertEqual(o['report']['status'],'REVIEW_REQUIRED')
    def test_activation_anchor_guard(self):
        b=body(1);o=track(b,static(b),options([c(anchor=(0,0,1))],supportActivationImpulse=True));self.assertIn('anchor mismatch',o['report']['failure']['reason'])
    def test_activation_slide_reject(self):
        b=body(1);d=target(b,False,root_fn=lambda t:[t,0,-t]);o=track(b,d,options([c(mu=0)],supportActivationImpulse=True));self.assertIn('did not establish sticking',o['report']['failure']['reason'])
if __name__=='__main__':unittest.main()
