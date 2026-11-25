"""Synthetic analytic and replay tests for the simultaneous impact candidate."""
import json,sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'tests'),str(Path(__file__).resolve().parents[1])]
import numpy as np
from test_multi_contact import points,scenario
from test_coupled import body
from test_support import c,options
from test_tracking import target
from coupled import Tree
from multi_contact import impulse,simulate_multi
from tracking import track

def run():
    out={}
    for offset in [0,.25]:
        b=body(1);b['physical']['centerOfMassLocalM'][0]=[offset,0,0];t=Tree(b);s=t.state(v=[0,0,-3,0,0,0]);z,r=impulse(t,s,points());expected=[3*(1-offset),3*(1+offset)];np.testing.assert_allclose([x['impulseNs'][2] for x in r['contacts']],expected,atol=1e-7);np.testing.assert_allclose(z.v,0,atol=1e-7);out['symmetric' if offset==0 else 'asymmetric']={'expectedNormalImpulseNs':expected,'report':r,'postVelocity':z.v.tolist()}
    r=simulate_multi(body(1),scenario());assert not r['failure'];assert abs(r['events'][0]['seconds']-np.sqrt(.2/9.81))<1e-8;out['drop']={'firstImpact':r['events'][0],'lastVelocity':r['frames'][-1]['velocity'],'status':r['status']}
    out['slidingConvergence']=[]
    for dt in [.02,.01,.005]:
        r=simulate_multi(body(1),scenario(z=0,vx=2,steps=round(.1/dt),dt=dt));assert not r['failure'];out['slidingConvergence'].append({'dtSeconds':dt,'positionErrorM':abs(r['frames'][-1]['rootM'][0]-(.2-.5*.5*9.81*.1**2))})
    assert out['slidingConvergence'][2]['positionErrorM']<.51*out['slidingConvergence'][1]['positionErrorM']
    b=body(1);d=target(b,False,root_fn=lambda t:[0,0,.5*9.81*(.15**2-t*t)]);cs=[c(str(i),lo=4,local=p,anchor=p) for i,p in enumerate([(-1,0,0),(1,0,0)])];o=track(b,d,options(cs,supportActivationImpulse=True));assert not o['report']['failure'];assert o['report']['status']=='REVIEW_REQUIRED';assert o['solved']['originalAuthorKeys']==d['originalAuthorKeys'];out['scheduledLanding']={'report':o['report'],'postImpactVelocity':o['frames'][3]['velocity']}
    return out
if __name__=='__main__':Path(sys.argv[1]).write_text(json.dumps(run(),indent=2,allow_nan=False)+'\n')
