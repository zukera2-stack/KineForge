import sys,unittest,copy
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import dynamics as d, physical_body as pb, solver as s
T=lambda a:torch.tensor(np.asarray(a),dtype=torch.float64)
class Physics(unittest.TestCase):
 def test_mass_cloud(self):
  m,c,I=pb.moments([[-1,0,0],[1,0,0]],[2,2]);self.assertEqual(m,4);np.testing.assert_allclose(c,0);np.testing.assert_allclose(I,np.diag([0,4,4]))
 def test_static(self):
  G=T(np.tile(np.eye(3),(5,2,1,1)));P=T(np.tile([[0,0,0],[1,0,0]],(5,1,1)));q=d.evaluate(G,P,T([2,3]),T([[0,0,0],[0,0,0]]),T(np.tile(np.eye(3),(2,1,1))),[-1,0],20,T([0,0,-9.81]));np.testing.assert_allclose(q['joint_force'][:,0],[[0,0,49.05]]*3,atol=1e-10);np.testing.assert_allclose(q['joint_torque'][:,0],[[0,-29.43,0]]*3,atol=1e-10)
 def test_ballistic(self):
  t=np.arange(20)/30;P=np.zeros((20,1,3));P[:,0,2]=5+3*t-4.905*t*t;G=T(np.tile(np.eye(3),(20,1,1,1)));q=d.evaluate(G,T(P),T([2]),T([[0,0,0]]),T([np.eye(3)]),[-1],30,T([0,0,-9.81]));self.assertLess(float(q['joint_force'].abs().max()),1e-10)
 def test_constant_spin(self):
  t=np.arange(20)/100;G=T(R.from_rotvec(np.c_[t*0,t*0,t*2]).as_matrix()[:,None]);q=d.evaluate(G,T(np.zeros((20,1,3))),T([1]),T([[0,0,0]]),T([np.diag([1,2,3])]),[-1],100,T([0,0,0]));self.assertLess(float(q['joint_torque'].abs().max()),1e-9)
 def test_angular_acceleration(self):
  t=np.arange(20)/100;G=T(R.from_rotvec(np.c_[t*0,t*0,2*t*t]).as_matrix()[:,None]);q=d.evaluate(G,T(np.zeros((20,1,3))),T([1]),T([[0,0,0]]),T([np.diag([1,2,3])]),[-1],100,T([0,0,0]));np.testing.assert_allclose(q['joint_torque'][:,0,2],12,atol=1e-8)
 def test_pi_rotation(self):
  axis=np.array([1.,2.,3.]);axis/=np.linalg.norm(axis);v=d.rotation_vector(T(R.from_rotvec(axis*np.pi).as_matrix()));self.assertAlmostEqual(float(torch.linalg.vector_norm(v)),np.pi,10)
 def test_invalid_inertia(self):
  ph={'massKg':[1],'centerOfMassLocalM':[[0,0,0]],'inertiaCOMKgM2':[np.diag([1,1,3]).tolist()]};self.assertRaises(ValueError,pb.validate,ph,1)
 def test_grad(self):
  v=torch.zeros(6,2,3,dtype=torch.float64,requires_grad=True);G=s.exp(v);P=T(np.tile([[0,0,0],[1,0,0]],(6,1,1)));params=(T([2,3]),T([[.1,0,0],[0,0,0]]),T(np.tile(np.eye(3),(2,1,1))),T([0,0,-9.81]));loss=d.cost(G,P,params,[-1,0],30);loss.backward();self.assertTrue(torch.isfinite(v.grad).all())
 def test_body(self):
  ph={'massKg':[2,3],'centerOfMassLocalM':[[0,0,0],[0,.1,0]],'inertiaCOMKgM2':[np.diag([1,2,2]).tolist(),np.diag([2,3,4]).tolist()]};m,c,I=pb.validate(ph,2);self.assertAlmostEqual(m.sum(),5,9);self.assertGreater(np.linalg.eigvalsh(I).min(),0)
if __name__=='__main__':unittest.main(verbosity=2)
