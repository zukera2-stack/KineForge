import sys, unittest, copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import solver as s, calibrate as c, dynamics, contact
T=lambda x:torch.tensor(np.asarray(x),dtype=torch.float64)

def fixture(kind='static'):
    n=3; F=9; fps=20; t=np.arange(F)/fps
    rig={'bones':['root','left','right'],'parents':[-1,0,0],'coordinateSystem':'XYZ meters; Z up'}
    o=np.array([[0,0,0],[-1,0,-1],[1,0,-1]],float);L=np.tile(np.eye(3),(F,n,1,1));root=np.tile([0.,0.,1.],(F,1))
    if kind=='flight':root[:,2]=3+2*t-4.905*t*t
    if kind=='slip':root[:,0]=.01*t
    if kind=='accelerate':root[:,0]=5*t*t
    G,P=s.fk(L,root,o,rig['parents']);b=c.make_profile(rig,o,[L],['analytic synthetic'])
    b['physical']={'massKg':[2,1,1],'centerOfMassLocalM':[[.5,0,0],[0,0,0],[0,0,0]],'inertiaCOMKgM2':[np.diag([.1,.2,.2]).tolist()]*n,'gravityMps2':[0,0,-9.81]}
    contacts=[{'id':name,'bone':name,'start':5,'end':13,'localPointM':[0,0,0],'anchorM':P[0,j].tolist(),'normal':[0,0,1],'mu':.4} for j,name in [(1,'left'),(2,'right')]]
    if kind in ('flight','hover'):contacts=[]
    d={'schema':'sam.keyposes.v1','rig':rig,'timeline':{'start':5,'end':13,'fps':fps},'keys':[{'frame':i+5,'positions':P[i].tolist()} for i in range(F)],'solverHints':{'physicalContacts':contacts}}
    return d,b,G,P,L

def check(d,b,G,P):
    p=d['rig']['parents'];params=dynamics.tensors(b['physical'],len(p));return contact.report(G,P,params,p,d['timeline']['fps'],contact.prepare(d,params))

class ContactTests(unittest.TestCase):
    def test_static_load_distribution(self):
        d,b,G,P,L=fixture();r=check(d,b,G,P);self.assertEqual(r['status'],'SAMPLED_CONTACT_FEASIBLE');self.assertLess(r['force_residual_max_N'],1e-9);self.assertLess(r['moment_residual_max_Nm'],1e-9)
        for f in r['frames']:
            self.assertEqual(f['mode'],'CONTACT');np.testing.assert_allclose([a['normal_N'] for a in f['forces']],[14.715,24.525],atol=1e-9)
            for a in f['forces']:self.assertGreaterEqual(a['normal_N'],0);self.assertLessEqual(a['tangent_N'],a['mu']*a['normal_N']+1e-9)
    def test_friction_cone_and_unilateral(self):
        D=contact.rays([0,0,1],.4).T
        for target, feasible in [([0,0,10],True),([3,0,10],True),([5,0,10],False),([0,0,-10],False)]:
            f=D@contact.project(D,np.array(target,float));self.assertGreaterEqual(f[2],-1e-10);self.assertLessEqual(np.linalg.norm(f[:2]),.4*f[2]+1e-10);self.assertEqual(np.linalg.norm(f-target)<1e-8,feasible)
    def test_frictionless(self):
        D=contact.rays([0,0,1],0).T;np.testing.assert_allclose(D@contact.project(D,np.array([4,0,10.])),[0,0,10])
    def test_impossible_moment(self):
        d,b,G,P,L=fixture();d['solverHints']['physicalContacts']=d['solverHints']['physicalContacts'][:1];r=check(d,b,G,P);self.assertIn('CONTACT_WRENCH_RESIDUAL',r['flags']);self.assertGreater(r['moment_residual_max_Nm'],1)
    def test_flight(self):
        d,b,G,P,L=fixture('flight');r=check(d,b,G,P);self.assertEqual(r['status'],'SAMPLED_CONTACT_FEASIBLE');self.assertLess(r['force_residual_max_N'],1e-9);self.assertTrue(all(f['mode']=='FLIGHT' and f['forces']==[] for f in r['frames']))
    def test_hover_is_not_flight(self):
        d,b,G,P,L=fixture('hover');r=check(d,b,G,P);self.assertIn('CONTACT_WRENCH_RESIDUAL',r['flags']);self.assertAlmostEqual(r['force_residual_max_N'],39.24)
    def test_sliding_anchor(self):
        d,b,G,P,L=fixture('slip');d['solverHints']['contactOptions']={'anchorToleranceM':.001};r=check(d,b,G,P);self.assertIn('CONTACT_ANCHOR_DRIFT',r['flags']);self.assertAlmostEqual(r['anchor_gap_max_m'],.004)
    def test_contact_range_and_local_offset(self):
        d,b,G,P,L=fixture();a=d['solverHints']['physicalContacts'][0];a['localPointM']=[0,0,-.1];a['anchorM'][2]-=.1;a['end']=8;r=check(d,b,G,P);self.assertAlmostEqual(r['anchor_gap_max_m'],0);self.assertEqual([len(x['forces']) for x in r['frames']],[2,2,2,1,1,1,1])
    def test_invalid_contract(self):
        d,b,G,P,L=fixture();params=dynamics.tensors(b['physical'],3)
        with self.assertRaises(ValueError):contact.prepare(d,None)
        for k,v in [('mu',-1),('normal',[0,0,2]),('start',4),('anchorM',[0,0,float('nan')])]:
            bad=copy.deepcopy(d);bad['solverHints']['physicalContacts'][0][k]=v
            with self.assertRaises(ValueError):contact.prepare(bad,params)
        d['solverHints'].pop('physicalContacts');self.assertIsNone(contact.prepare(d,params))
    def test_objective_gradient(self):
        d,b,G,P,L=fixture('accelerate');params=dynamics.tensors(b['physical'],3);schedule=contact.prepare(d,params);p=d['rig']['parents'];v=T(P).requires_grad_();g=T(G)
        loss=contact.evaluate(g,v,params,p,20,schedule);loss.backward();self.assertTrue(torch.isfinite(v.grad).all());idx=(4,0,0);h=1e-6;vp=P.copy();vm=P.copy();vp[idx]+=h;vm[idx]-=h
        numeric=float((contact.evaluate(g,T(vp),params,p,20,schedule)-contact.evaluate(g,T(vm),params,p,20,schedule))/(2*h));self.assertAlmostEqual(float(v.grad[idx]),numeric,places=5)
    def test_unified_solver_gate(self):
        d,b,G,P,L=fixture('hover');out,r,*_=s.solve(d,b,L,20,max_key_mm=.01);self.assertEqual(r['status'],'REVIEW_REQUIRED');self.assertIn('CONTACT_WRENCH_RESIDUAL',r['flags']);self.assertIn('contact_wrench',r['objective_terms_attempt']);self.assertEqual(out['originalAuthorKeys'],d['keys']);self.assertEqual(out['timeline'],d['timeline'])
    def test_unified_static(self):
        d,b,G,P,L=fixture();out,r,*_=s.solve(d,b,L,20,max_key_mm=1);self.assertEqual(r['contact_final']['status'],'SAMPLED_CONTACT_FEASIBLE');self.assertLess(r['final']['key_max_mm'],1e-6);self.assertEqual(out['originalAuthorKeys'],d['keys'])
    def test_sparse_ballistic(self):
        d,b,G,P,L=fixture('flight');d['keys']=d['keys'][::4];out,r,*_=s.solve(d,b,L,30,max_key_mm=1);self.assertEqual(r['contact_final']['status'],'SAMPLED_CONTACT_FEASIBLE');self.assertLess(r['contact_final']['force_residual_max_N'],1e-6);self.assertEqual(out['originalAuthorKeys'],d['keys'])
    def test_unified_ballistic(self):
        d,b,G,P,L=fixture('flight');out,r,*_=s.solve(d,b,L,20,max_key_mm=.01);self.assertEqual(r['contact_final']['status'],'SAMPLED_CONTACT_FEASIBLE');self.assertEqual(out['timeline'],d['timeline'])
if __name__=='__main__':unittest.main(verbosity=2)
