import copy,sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import solver as s, calibrate as c, joints as j, dynamics

torch.set_num_threads(1)

def fixture(angles=None,axis=(0,1,0)):
    angles=np.array([0.,10.,20.,30.,20.,10.,0.] if angles is None else angles)
    F=len(angles);rig={'bones':['base','hinge','tip'],'parents':[-1,0,1],'coordinateSystem':'XYZ meters'}
    o=np.array([[0,0,0],[0,0,1],[1,0,0]],float);p=np.array(rig['parents']);L=np.tile(np.eye(3),(F,3,1,1));L[:,1]=R.from_rotvec(np.radians(angles)[:,None]*axis).as_matrix();G,P=s.fk(L,np.zeros((F,3)),o,p)
    b=c.make_profile(rig,o,[L],['synthetic']);b['neutralLocalQuaternionXYZW']=[[0,0,0,1]]*3
    b['jointModels']={'hinge':{'type':'hinge','axis':[0,1,0],'minDeg':-45,'maxDeg':45,'stiffnessNmPerRad':12,'dampingNmsPerRad':2}}
    d={'schema':'sam.keyposes.v1','rig':rig,'timeline':{'start':10,'end':10+F-1,'fps':10},'keys':[{'frame':10+i,'positions':P[i].tolist()} for i in range(F)]}
    return d,b,G,P,L

def check(d,b,G,P,L,contacts=None):
    p=np.array(b['rig']['parents']);neutral=R.from_quat(b['neutralLocalQuaternionXYZW']).as_matrix();ph=dynamics.tensors(b['physical'],len(p)) if 'physical' in b else None
    return j.report(L,G,P,neutral,j.prepare(b),ph,p,d['timeline']['fps'],contacts,d['timeline']['start'])

def physical(b):
    b['physical']={'massKg':[1,2,0.1],'centerOfMassLocalM':[[0,0,0],[.5,0,0],[0,0,0]],'inertiaCOMKgM2':[(np.eye(3)*.01).tolist()]*3,'gravityMps2':[0,0,-9.81]}

class JointTests(unittest.TestCase):
    def test_hinge_legal(self):
        d,b,G,P,L=fixture();r=check(d,b,G,P,L);self.assertEqual(r['flags'],[]);self.assertAlmostEqual(r['joints'][0]['max_violation_deg'],0,8)
    def test_hinge_wrong_axis(self):
        d,b,G,P,L=fixture(axis=(0,0,1));r=check(d,b,G,P,L);self.assertIn('JOINT_RANGE_OR_AXIS_LIMIT',r['flags']);self.assertAlmostEqual(r['joints'][0]['max_violation_deg'],30,7);self.assertIn(13,r['joints'][0]['violation_frames'])
    def test_hinge_range(self):
        d,b,G,P,L=fixture([0,60,0]);self.assertAlmostEqual(check(d,b,G,P,L)['joints'][0]['max_violation_deg'],15,7)
    def test_ball_and_fixed(self):
        d,b,G,P,L=fixture([0,60,0]);b['jointModels']={'hinge':{'type':'ball','maxAngleDeg':40}};self.assertAlmostEqual(check(d,b,G,P,L)['joints'][0]['max_violation_deg'],20,7);b['jointModels']['hinge']={'type':'fixed'};self.assertAlmostEqual(check(d,b,G,P,L)['joints'][0]['max_violation_deg'],60,7)
    def test_spring_and_energy(self):
        d,b,G,P,L=fixture([30]*3);r=check(d,b,G,P,L)['joints'][0];self.assertAlmostEqual(r['passive_peak_Nm'],12*np.pi/6,7);self.assertAlmostEqual(r['spring_energy_peak_J'],6*(np.pi/6)**2,7)
    def test_damping(self):
        d,b,G,P,L=fixture([0,10,20]);b['jointModels']['hinge']['stiffnessNmPerRad']=0;row=check(d,b,G,P,L)['joints'][0];self.assertAlmostEqual(row['passive_peak_Nm'],2*np.pi/18*10,7)
    def test_nonidentity_neutral(self):
        d,b,G,P,L=fixture([0,20,0]);N=R.from_euler('x',67,degrees=True);L[:,1]=N.as_matrix()@L[:,1];b['neutralLocalQuaternionXYZW'][1]=N.as_quat().tolist();G,P=s.fk(L,P[:,0],np.array(b['offsets_m']),np.array(b['rig']['parents']));self.assertEqual(check(d,b,G,P,L)['flags'],[])
    def test_invalid_profiles(self):
        _,b,*_=fixture();
        for change in ({'axis':[0,2,0]},{'minDeg':60},{'dampingNmsPerRad':-1},{'actuatorLimitNm':float('nan')},{'type':'unknown'},{'typo':1}):
            z=copy.deepcopy(b);z['jointModels']['hinge'].update(change)
            with self.assertRaises(ValueError):j.prepare(z)
    def test_actuator_gravity(self):
        d,b,G,P,L=fixture([0]*3);physical(b);b['jointModels']['hinge']['actuatorLimitNm']=10
        r=check(d,b,G,P,L);self.assertAlmostEqual(r['joints'][0]['actuator_peak_Nm'],(2*.5+.1)*9.81,7);self.assertIn('ACTUATOR_LIMIT',r['flags']);b['jointModels']['hinge']['actuatorLimitNm']=12;self.assertEqual(check(d,b,G,P,L)['flags'],[])
    def test_actuator_contact_unknown(self):
        d,b,G,P,L=fixture([0]*3);physical(b);b['jointModels']['hinge']['actuatorLimitNm']=20;r=check(d,b,G,P,L,{'items':[{}]});self.assertIn('ACTUATOR_LOAD_UNVERIFIED',r['flags']);self.assertIsNone(r['joints'][0]['actuator_peak_Nm'])
    def test_actuator_no_mass(self):
        d,b,G,P,L=fixture([0]*3);b['jointModels']['hinge']['actuatorLimitNm']=20;self.assertIn('ACTUATOR_LOAD_UNVERIFIED',check(d,b,G,P,L)['flags'])
    def test_ambiguous_pi(self):
        d,b,G,P,L=fixture([0,180,0],axis=(0,0,1));self.assertIn('JOINT_ROTATION_AMBIGUOUS',check(d,b,G,P,L)['flags'])
    def test_endpoints_impossible_not_silent(self):
        d,b,G,P,L=fixture([60]*3);out,r,*_=s.solve(d,b,L,30,max_key_mm=5);self.assertIn('JOINT_RANGE_OR_AXIS_LIMIT',r['flags']);self.assertEqual(r['status'],'REVIEW_REQUIRED');self.assertEqual(out['originalAuthorKeys'],d['keys']);self.assertEqual(out['timeline'],d['timeline'])
    def test_solver_repairs_unobservable_twist(self):
        # Tip position is not affected by a twist about its own X axis. Constraint
        # should remove intermediate leaf rotations without altering authored XYZ.
        d,b,G,P,L=fixture([0]*7);L[1:-1,2]=R.from_rotvec(np.radians([0,30,50,30,0])[:,None]*[1,0,0]).as_matrix();b['jointModels']['tip']={'type':'fixed'};out,r,*_=s.solve(d,b,L,80,max_key_mm=1);self.assertEqual(r['joint_final']['flags'],[]);self.assertLess(r['final']['key_max_mm'],1);self.assertEqual(out['originalAuthorKeys'],d['keys'])
    def test_gradient_finite_neutral(self):
        d,b,G,P,L=fixture([0]*3);p=np.array(b['rig']['parents']);v=torch.zeros((3,3,3),dtype=torch.float64,requires_grad=True);loc=s.exp(v);g,pp=s.fkt(loc,torch.zeros((3,3),dtype=torch.float64),torch.tensor(b['offsets_m'],dtype=torch.float64),p);loss=j.cost(loc,g,pp,torch.eye(3,dtype=torch.float64).repeat(3,1,1),j.prepare(b),None,p,10,None);loss.backward();self.assertTrue(torch.isfinite(v.grad).all())
    def test_passive_reduces_gravity_actuator_demand(self):
        d,b,G,P,L=fixture([30]*3);physical(b);b['jointModels']['hinge']['actuatorLimitNm']=100
        r=check(d,b,G,P,L)['joints'][0];self.assertAlmostEqual(r['actuator_peak_Nm'],(2*.5+.1)*9.81*np.cos(np.pi/6)-12*np.pi/6,7)
    def test_damping_opposes_motion(self):
        d,b,G,P,L=fixture([0,10,20]);b['jointModels']['hinge']['stiffnessNmPerRad']=0
        m,v,passive,e,amb=next(j.samples(torch.tensor(L),torch.eye(3,dtype=torch.float64).repeat(3,1,1),j.prepare(b),10));self.assertLess(float(passive[0,1]),0)
    def test_twist_chart_wrap_reported(self):
        d,b,G,P,L=fixture([170,-170,170]);self.assertIn('JOINT_ROTATION_AMBIGUOUS',check(d,b,G,P,L)['flags'])
    def test_two_frame_actuator_unverified(self):
        d,b,G,P,L=fixture([0,0]);physical(b);b['jointModels']['hinge']['actuatorLimitNm']=20;self.assertIn('ACTUATOR_LOAD_UNVERIFIED',check(d,b,G,P,L)['flags'])
if __name__=='__main__':unittest.main()
