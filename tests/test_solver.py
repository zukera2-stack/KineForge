import sys,copy,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.spatial.transform import Rotation as R
import solver as s, calibrate as c
import torch
torch.set_num_threads(1)
class Tests(unittest.TestCase):
 def setUp(self):
  self.rig={'bones':['root','branchA','branchB','tip'],'parents':[-1,0,0,1],'coordinateSystem':'XYZ meters'};self.o=np.array([[0,0,0],[1,0,0],[0,0,1],[1,0,0]],float);self.L=np.broadcast_to(np.eye(3),(9,4,3,3)).copy();self.L[:,0]=R.from_rotvec(np.array([[0,0,.3*np.sin(i/8*np.pi)] for i in range(9)])).as_matrix();self.P=s.fk(self.L,np.zeros((9,3)),self.o,np.array(self.rig['parents']))[1];self.b=c.make_profile(self.rig,self.o,[self.L],['synthetic']);self.d={'schema':'sam.keyposes.v1','rig':self.rig,'timeline':{'start':4,'end':12,'fps':30},'keys':[{'frame':i+4,'positions':self.P[i].tolist()} for i in [0,4,8]]}
 def test_generic(self):
  out,r,P,L,*_=s.solve(self.d,self.b,self.L,30);self.assertEqual(len(P),9);self.assertLess(r['final']['length_max_error_m'],1e-10);self.assertLess(r['final']['boundary_max_mm'],.1);self.assertEqual(out['originalAuthorKeys'],self.d['keys']);self.assertEqual(out['lockedFrames'],[4,8,12]);self.assertEqual(out['keyframes'][4]['positions'],out['sourceKeys'][1]['positions']);self.assertLessEqual(r['objective_after'],r['objective_before']+1e-12)
 def test_ignore_editor_angles(self):
  d=copy.deepcopy(self.d);d['keys'][1]['angleCorrectionsDegrees']=[[999,999,999]]*4;self.assertTrue(np.array_equal(s.solve(d,self.b,self.L,20)[2],s.solve(self.d,self.b,self.L,20)[2]))
 def test_invalid(self):
  for variant in ['duplicate','wrongrig','nan','noend']:
   d=copy.deepcopy(self.d)
   if variant=='duplicate':d['keys'][1]['frame']=4
   if variant=='wrongrig':d['rig']['bones'][1]='changed'
   if variant=='nan':d['keys'][1]['positions'][1][1]=float('nan')
   if variant=='noend':d['keys'].pop()
   with self.assertRaises(ValueError):s.valid(d,self.b)
 def test_impossible_key_flag(self):
  d=copy.deepcopy(self.d);d['keys'][1]['positions'][3][0]+=8;self.assertIn('KEY_DEVIATION_LIMIT',s.solve(d,self.b,self.L,20,max_key_mm=10)[1]['flags'])
 def test_contact_collision(self):
  d=copy.deepcopy(self.d);d['solverHints']={'contacts':[{'bone':'branchB','start':4,'end':12,'position':[0,0,1]}]};b=copy.deepcopy(self.b);b['collisionProxies']=[[2,3,3.0]];r=s.solve(d,b,self.L,20)[1];self.assertGreater(r['objective_terms_attempt']['collision'],0);self.assertIn('contacts',r['objective_terms_attempt'])
 def test_short(self):
  d=copy.deepcopy(self.d);d['timeline']['end']=5;d['keys']=[d['keys'][0],{**d['keys'][-1],'frame':5}];r=s.solve(d,self.b,self.L[[0,-1]],10)[1];self.assertEqual(r['final']['rms_jerk_mps3'],0)
 def test_exact_branch_alignment(self):
  p=np.array(self.rig['parents']);q=R.from_rotvec([[.6,.3,-.2],[.2,-.4,.3],[0,0,0],[0,0,0]]).as_matrix();P=s.fk(q[None],np.array([[.2,.3,.4]]),self.o,p)[1][0];L=s.align_pose(P,np.tile(np.eye(3),(4,1,1)),self.o,p);self.assertLess(np.abs(s.fk(L[None],P[None,0],self.o,p)[1][0]-P).max(),1e-10);self.assertTrue(np.allclose(np.linalg.det(L),1))
 def test_antiparallel_alignment(self):
  p=np.array([-1,0]);o=np.array([[0,0,0],[1,0,0]],float);y=np.array([[0,0,0],[-1,0,0]],float);L=s.align_pose(y,np.tile(np.eye(3),(2,1,1)),o,p);self.assertLess(np.abs(s.fk(L[None],y[None,0],o,p)[1][0]-y).max(),1e-10)
 def test_gate_reports_impossible_rotation_limit(self):
  d=copy.deepcopy(self.d);d['keys']=d['keys'][:2];d['timeline']['end']=8;d['keys'][1]['frame']=8;r=s.solve(d,self.b,self.L[:5],20,max_step_deg=.01)[1];self.assertIn('PEAK_ROTATION_LIMIT',r['flags']);self.assertEqual(r['status'],'REVIEW_REQUIRED')
if __name__=='__main__':unittest.main(verbosity=2)
