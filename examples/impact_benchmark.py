"""Reproducible public point/plane impact and sliding cases; SI units."""
import json,sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'tests'),str(Path(__file__).resolve().parents[1])]
import numpy as np
from test_impact import scenario
from test_coupled import body
from impact import simulate_contact,point_impulse
from coupled import Tree

def run():
    cases={}
    for name,d in [('elastic_drop',scenario(z=.1,e=1,steps=20)),('inelastic_drop',scenario(z=.1,steps=30)),('slide_stop',scenario(vx=1,steps=30)),('airborne',scenario(z=2,steps=10))]:
        r=simulate_contact(body(1),d);assert r['status']=='CONTACT_SCENARIO_CANDIDATE',r['failure'];cases[name]={'status':r['status'],'minSampledGapM':r['minSampledGapM'],'firstFrame':r['frames'][0],'lastFrame':r['frames'][-1],'impactEvents':[e for e in r['events'] if e['kind']=='impact'],'supportSteps':sum(e['kind']=='support_step' for e in r['events'])}
    c=cases['elastic_drop']['impactEvents'][0];assert abs(c['seconds']-np.sqrt(.2/9.81))<1e-8 and c['kineticLossJ']<1e-9
    c=cases['inelastic_drop']['impactEvents'][0];assert abs(c['kineticLossJ']-1.962)<1e-8
    cases['slidingConvergence']=[]
    for dt in [.02,.01,.005]:
        r=simulate_contact(body(1),scenario(vx=2,dt=dt,steps=round(.1/dt)));assert not r['failure'];cases['slidingConvergence'].append({'dtSeconds':dt,'positionErrorM':abs(r['frames'][-1]['rootM'][0]-(.2-.5*.5*9.81*.1**2))})
    t=Tree(body(2));s=t.state(v=[0,0,-2,0,0,0,0]);o,r=point_impulse(t,s,1,[1,0,0],[0,0,1],0,.2);cases['articulatedImpact']={'report':r,'beforeVelocity':s.v.tolist(),'afterVelocity':o.v.tolist(),'before':t.invariants(s),'after':t.invariants(o)}
    return cases
if __name__=='__main__':Path(sys.argv[1]).write_text(json.dumps(run(),indent=2,allow_nan=False)+'\n')
