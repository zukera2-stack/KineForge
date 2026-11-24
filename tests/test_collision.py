import copy, unittest
from test_joints import fixture, s, np, R, torch
import collision as c


def case(angles, obstacle=(1,1,1),radius=.2,steps=64):
    d,b,G,P,L=fixture([0]*len(angles));L[:,2]=R.from_rotvec(np.radians(angles)[:,None]*[0,0,1]).as_matrix();b.pop('jointModels');b['collisionModel']={'spheres':[{'id':'fixed','bone':'base','centerLocalM':list(obstacle),'radiusM':radius},{'id':'moving','bone':'tip','centerLocalM':[1,0,0],'radiusM':radius}],'pairs':[['fixed','moving']],'maxSubsteps':steps,'objectiveSubsteps':2,'toleranceM':.001};return d,b,P,L

def check(d,b,P,L):
    return c.report(L,P[:,0],np.array(b['offsets_m']),np.array(b['rig']['parents']),c.prepare(b),s.fk,d['timeline']['start'])

class CollisionTests(unittest.TestCase):
    def test_clear(self):
        d,b,P,L=case([0,30,0],obstacle=(5,5,5));r=check(d,b,P,L);self.assertEqual(r['flags'],[]);self.assertGreater(r['pairs'][0]['certified_lower_bound_m'],0)
    def test_static_penetration(self):
        d,b,P,L=case([90,90]);r=check(d,b,P,L);self.assertIn('COLLISION_PENETRATION',r['flags']);self.assertAlmostEqual(r['pairs'][0]['min_sampled_clearance_m'],-.4)
    def test_between_frames(self):
        d,b,P,L=case([10,170]);r=check(d,b,P,L);self.assertIn('COLLISION_PENETRATION',r['flags']);self.assertAlmostEqual(r['pairs'][0]['worst_frame'],10.5);self.assertAlmostEqual(r['pairs'][0]['min_sampled_clearance_m'],-.4)
    def test_uncertified(self):
        d,b,P,L=case([10,170],obstacle=(1,1.401,1),steps=2);r=check(d,b,P,L);self.assertIn('COLLISION_INTERVAL_UNCERTIFIED',r['flags']);self.assertNotIn('COLLISION_PENETRATION',r['flags'])
    def test_common_motion(self):
        d,b,P,L=case([0,30,0],obstacle=(5,5,5));r=check(d,b,P,L);L[:,0]=R.from_rotvec(np.radians([0,120,-30])[:,None]*[1,0,0]).as_matrix();P[:,0]=[[5,2,1],[9,3,2],[0,0,0]];z=check(d,b,P,L);self.assertAlmostEqual(r['pairs'][0]['certified_lower_bound_m'],z['pairs'][0]['certified_lower_bound_m'])
    def test_ambiguous_pi(self):
        d,b,P,L=case([0,180],obstacle=(5,5,5));self.assertIn('COLLISION_ROTATION_AMBIGUOUS',check(d,b,P,L)['flags'])
    def test_invalid(self):
        d,b,P,L=case([0,30]);
        for key,value in [('maxSubsteps',3),('toleranceM',float('nan')),('pairs',[['fixed','fixed']]),('pairs',[['fixed','moving'],['moving','fixed']]),('objectiveSubsteps',0)]:
            a=copy.deepcopy(b);a['collisionModel'][key]=value
            with self.assertRaises(ValueError):c.prepare(a)
    def test_bound_dense_random(self):
        rng=np.random.default_rng(17)
        for _ in range(12):
            d,b,P,L=case([0,0],obstacle=(1.5,1.2,1));L=R.from_rotvec(rng.normal(0,.9,(6,3))).as_matrix().reshape(2,3,3,3);m=c.prepare(b);r=check(d,b,P,L);u=np.linspace(0,1,1001);rv=R.from_matrix(L[1]@L[0].transpose(0,2,1)).as_rotvec();LL=R.from_rotvec((u[:,None,None]*rv).reshape(-1,3)).as_matrix().reshape(len(u),3,3,3)@L[0];G,X=s.fk(LL,np.zeros((len(u),3)),np.array(b['offsets_m']),np.array(b['rig']['parents']));actual=c.clearance_t(torch.tensor(G),torch.tensor(X),m).min().item();self.assertLessEqual(r['pairs'][0]['certified_lower_bound_m'],actual+1e-10)
    def test_gradient(self):
        d,b,P,L=case([0,75,0]);v=torch.zeros((3,3,3),dtype=torch.float64,requires_grad=True);l=s.exp(v)@torch.tensor(L);o=torch.tensor(b['offsets_m'],dtype=torch.float64);p=np.array(b['rig']['parents']);g,x=s.fkt(l,torch.tensor(P[:,0]),o,p);loss=c.cost(l,g,x,o,p,c.prepare(b),s.fkt,s.exp);loss.backward();self.assertTrue(torch.isfinite(v.grad).all());self.assertGreater(v.grad.norm().item(),0)
    def test_fixed_boundary_conflict(self):
        d,b,P,L=case([90,90,90]);out,r,*_=s.solve(d,b,L,20,max_key_mm=1);self.assertIn('COLLISION_PENETRATION',r['flags']);self.assertEqual(r['status'],'REVIEW_REQUIRED');self.assertEqual(out['originalAuthorKeys'],d['keys']);self.assertEqual(out['timeline'],d['timeline'])
    def test_repair(self):
        d,b,P,L=case([0,25,50,75,50,25,0]);out,r,*_=s.solve(d,b,L,100,max_key_mm=5);self.assertIn('COLLISION_PENETRATION',r['collision_initial']['flags']);self.assertEqual(r['collision_final']['flags'],[]);self.assertLess(r['final']['key_max_mm'],5);self.assertEqual(out['originalAuthorKeys'],d['keys'])
if __name__=='__main__':unittest.main()
