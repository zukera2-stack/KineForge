"""Reproducible tracking benchmarks: analytic target, no hidden root force, cap rejection."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_tracking import target,opts
from test_coupled import body
from tracking import track

def run():
    b=body();d=target(b);cases={}
    for sub in [2,4,8]:cases['fixed_substeps_'+str(sub)]=track(b,d,opts(substeps=sub))['report']
    b=body(2);cases['ballistic']=track(b,target(b,False,root_fn=lambda t:[t,0,3-.5*9.81*t*t],angle_fn=lambda t,i:0),opts(fixedBase=False))['report']
    cases['hover']=track(b,target(b,False,root_fn=lambda t:[0,0,3],angle_fn=lambda t,i:0),opts(fixedBase=False))['report']
    b=body();b['jointModels']['link1']['actuatorLimitNm']=.01;cases['limited_actuator']=track(b,target(b),opts())['report']
    assert all(cases['fixed_substeps_'+str(i)]['status']=='TRACKING_CANDIDATE' for i in (2,4,8))
    assert cases['ballistic']['status']=='TRACKING_CANDIDATE' and 'KEY_DEVIATION_LIMIT' in cases['hover']['flags'] and 'ACTUATOR_SATURATION' in cases['limited_actuator']['flags']
    return cases
if __name__=='__main__':
    data=run();Path(sys.argv[1]).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
