"""Reproduce analytic and unified optimization checks, synthetic fixtures only."""
import sys, json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from test_joints import fixture,check,physical,s,np,R

def main(path):
    cases=[]
    for name,angles,axis,expected in [('legal',[0,30,0],(0,1,0),True),('range',[0,60,0],(0,1,0),False),('wrong_axis',[0,30,0],(0,0,1),False)]:
        d,b,G,P,L=fixture(angles,axis);r=check(d,b,G,P,L);assert (not r['flags'])==expected;cases.append({'case':name,**r})
    d,b,G,P,L=fixture([0]*3);physical(b);b['jointModels']['hinge']['actuatorLimitNm']=10;r=check(d,b,G,P,L);assert 'ACTUATOR_LIMIT' in r['flags'];cases.append({'case':'gravity_actuator_overload',**r})
    d,b,G,P,L=fixture([0]*7);L[1:-1,2]=R.from_rotvec(np.radians([0,30,50,30,0])[:,None]*[1,0,0]).as_matrix();b['jointModels']['tip']={'type':'fixed'};out,r,*_=s.solve(d,b,L,80,max_key_mm=1);assert not r['joint_final']['flags'] and out['originalAuthorKeys']==d['keys'] and out['timeline']==d['timeline'];cases.append({'case':'unobservable_twist_repair','initial':r['joint_initial'],'final':r['joint_final'],'key_max_mm':r['final']['key_max_mm'],'solver_status':r['status']})
    s.write(path,{'version':s.VERSION,'passed':len(cases),'cases':cases})
if __name__=='__main__':main(sys.argv[1])
