import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.integrate import solve_ivp
from coupled import Tree


def body(n=3,gravity=(0,0,-9.81)):
    names=['root']+['link'+str(i) for i in range(1,n)]
    return {'rig':{'bones':names,'parents':[-1]+list(range(n-1))},'offsets_m':[[0,0,0]]*min(n,2)+[[1,0,0]]*max(n-2,0),'neutralLocalQuaternionXYZW':[[0,0,0,1]]*n,'physical':{'massKg':[2]+[1]*(n-1),'centerOfMassLocalM':[[0,0,0]]+[[.5,0,0]]*(n-1),'inertiaCOMKgM2':[np.diag([.08,.1,.1]).tolist()]*n,'gravityMps2':list(gravity)},'jointModels':{name:{'type':'hinge','axis':[0,1,0],'minDeg':-179,'maxDeg':179} for name in names[1:]}}

def angles(tree,q,v):
    L=tree.N.copy()
    for i,x in enumerate(q,1): L[i]=R.from_rotvec([0,x,0]).as_matrix()
    return tree.state(local=L,v=v)

def analytic(y):
    q1,q2,v1,v2=y; c=np.cos(q2); sn=np.sin(q2)
    M=np.array([[1.7+c,.35+.5*c],[.35+.5*c,.35]]) # m1=m2=1, lc=.5, I=.1, l1=1
    # M11 = .35 + .35 + 1 + cos(q2)
    h=np.array([-.5*sn*(2*v1*v2+v2*v2)-9.81*(1.5*np.cos(q1)+.5*np.cos(q1+q2)),.5*sn*v1*v1-4.905*np.cos(q1+q2)])
    return np.r_[v1,v2,np.linalg.solve(M,-h)]

class Coupled(unittest.TestCase):
    def test_double_pendulum_analytic(self):
        t=Tree(body(),True);s=angles(t,[.2,-.4],[.3,-.2]);a,report=t.forward(s,[0,0]);np.testing.assert_allclose(a,analytic([.2,-.4,.3,-.2])[2:],atol=1e-12);self.assertLess(report['residual'],1e-12)
    def test_independent_holdout_convergence(self):
        t=Tree(body(),True);y=[.2,-.4,.3,-.2];truth=solve_ivp(lambda _,z:analytic(z),(0,.15),y,rtol=1e-11,atol=1e-13).y[:,-1];errors=[]
        for dt in [.005,.0025]:
            s=angles(t,y[:2],y[2:]);e0=t.invariants(s)['energyJ']
            for _ in range(round(.15/dt)):s=t.step(s,[0,0],dt)
            q=[R.from_matrix(s.local[i]).as_rotvec()[1] for i in [1,2]];errors.append(np.linalg.norm(np.r_[q,s.v]-truth));self.assertLess(abs(t.invariants(s)['energyJ']-e0),.002)
        self.assertLess(errors[1],errors[0]/3)
    def test_coupling_and_inverse(self):
        t=Tree(body(gravity=(0,0,0)),True);s=angles(t,[.2,.3],[0,0]);a,_=t.forward(s,[0,1]);self.assertGreater(abs(a[0]),.1);np.testing.assert_allclose(t.inverse(s,a),[0,1],atol=1e-12)
    def test_free_recoil(self):
        t=Tree(body(2,(0,0,0)));s=t.state();act=np.zeros(t.d);act[-1]=1;initial=t.invariants(s);a,_=t.forward(s,act);self.assertGreater(np.linalg.norm(a[:6]),.1)
        for _ in range(20):s=t.step(s,act,.002)
        end=t.invariants(s);np.testing.assert_allclose(end['linearMomentumKgMps'],initial['linearMomentumKgMps'],atol=2e-6);np.testing.assert_allclose(end['angularMomentumCOMKgM2ps'],initial['angularMomentumCOMKgM2ps'],atol=2e-6)
    def test_ballistic(self):
        t=Tree(body(2));s=t.state(root=[0,0,3],v=[1,0,2,0,0,0,0]);a,_=t.forward(s,np.zeros(t.d));np.testing.assert_allclose(a,[0,0,-9.81,0,0,0,0],atol=1e-12)
        for _ in range(10):s=t.step(s,np.zeros(t.d),.01)
        np.testing.assert_allclose(s.root,[.1,0,3.2-.04905],atol=1e-12)
    def test_sticking_and_friction(self):
        t=Tree(body(1));s=t.state();c=[dict(bone=0,localPointM=[0,0,0],anchorM=[0,0,0],normal=[0,0,1],mu=.5)];a,r=t.forward(s,np.zeros(6),contacts=c);np.testing.assert_allclose(a,0,atol=1e-12);np.testing.assert_allclose(r['contactForcesN'],[[0,0,19.62]],atol=1e-12)
        with self.assertRaisesRegex(ValueError,'friction cone'):t.forward(s,[20,0,0,0,0,0],contacts=c)
        with self.assertRaisesRegex(ValueError,'friction cone'):t.forward(s,[0,0,30,0,0,0],contacts=c)
        with self.assertRaisesRegex(ValueError,'drift'):t.forward(t.state(root=[0,0,.1]),np.zeros(6),contacts=c)
    def test_known_force(self):
        t=Tree(body(1,(0,0,0)));s=t.state();loads=[dict(bone=0,localPointM=[0,1,0],forceN=[2,0,0])];a,_=t.forward(s,np.zeros(6),loads);np.testing.assert_allclose(a,[1,0,0,0,0,-20],atol=1e-12)
    def test_damper_and_limits(self):
        b=body(2,(0,0,0));b['jointModels']['link1'].update(stiffnessNmPerRad=4,dampingNmsPerRad=2,actuatorLimitNm=1,minDeg=-30,maxDeg=30);t=Tree(b,True);s=angles(t,[.2],[.1]);f,E,bad=t.passive(s);np.testing.assert_allclose(f,[-1]);self.assertAlmostEqual(E,.08);self.assertFalse(bad)
        with self.assertRaisesRegex(ValueError,'actuator'):t.forward(s,[2])
        with self.assertRaisesRegex(ValueError,'range'):t.forward(angles(t,[.8],[0]),[0])
    def test_ball_and_fixed(self):
        b=body(3);b['jointModels']={'link1':{'type':'ball'},'link2':{'type':'fixed'}};t=Tree(b,True);self.assertEqual(t.d,3);s=t.state();a,_=t.forward(s,[1,2,3]);np.testing.assert_allclose(t.inverse(s,a),[1,2,3],atol=1e-12)
    def test_singular_rejected(self):
        b=body(2);b['physical']['centerOfMassLocalM'][1]=[0,0,0];b['physical']['inertiaCOMKgM2'][1]=np.zeros((3,3)).tolist();t=Tree(b,True)
        with self.assertRaisesRegex(ValueError,'singular'):t.forward(t.state(),[0])
    def test_nonidentity_neutral(self):
        b=body(2,(0,0,0));b['neutralLocalQuaternionXYZW'][1]=R.from_rotvec([0,0,np.pi/2]).as_quat().tolist();t=Tree(b,True);s=t.state(v=[1]);k=t.kinematics(s);np.testing.assert_allclose(k['w'][1],[-1,0,0],atol=1e-12)

class SimulationContract(unittest.TestCase):
    def test_deterministic_scenario(self):
        from coupled import simulate
        d={'schema':'kineforge.simulation.v1','dtSeconds':.01,'steps':4,'initial':{'rootM':[0,0,3]}}
        a=simulate(body(2),d);b=simulate(body(2),d);self.assertEqual(a,b);self.assertEqual(len(a['frames']),5);self.assertAlmostEqual(a['frames'][-1]['rootM'][2],3-.5*9.81*.04**2,12)
    def test_free_root_actuation_guard(self):
        from coupled import simulate
        d={'schema':'kineforge.simulation.v1','dtSeconds':.01,'steps':1,'actuation':[1,0,0,0,0,0]}
        with self.assertRaisesRegex(ValueError,'free root actuation'):simulate(body(1),d)

if __name__=='__main__':unittest.main()
