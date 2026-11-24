"""Analytic double-pendulum holdout, time-step convergence and free-body recoil."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from test_coupled import body, angles, analytic
from coupled import Tree,simulate
from scipy.integrate import solve_ivp
from scipy.spatial.transform import Rotation as R
import numpy as np

def main(path):
    t=Tree(body(),True); y=[.2,-.4,.3,-.2]; duration=.15
    reference=solve_ivp(lambda _,z:analytic(z),(0,duration),y,rtol=1e-12,atol=1e-14,dense_output=True)
    rows=[]
    for dt in [.01,.005,.0025]:
        s=angles(t,y[:2],y[2:]);E=t.invariants(s)['energyJ'];errors=[];pos_errors=[];drift=[];residual=[]
        for i in range(round(duration/dt)+1):
            q=np.array([R.from_matrix(s.local[j]).as_rotvec()[1] for j in [1,2]]); ref=reference.sol(i*dt);errors.append(q-ref[:2]);
            # Independent Cartesian COM and endpoint positions from analytical angles.
            actual=t.kinematics(s); r1,r2=ref[:2]; u=lambda a:np.array([np.cos(a),0,-np.sin(a)]); predicted=np.stack([.5*u(r1),u(r1)+.5*u(r1+r2)]);pos_errors.append(actual['X'][1:]-predicted);drift.append(t.invariants(s)['energyJ']-E);a,r=t.forward(s,[0,0]);residual.append(r['residual'])
            if i<round(duration/dt):s=t.step(s,[0,0],dt)
        rows.append(dict(dt_s=dt,angle_rms_deg=float(np.sqrt(np.mean(np.array(errors)**2))*180/np.pi),com_rms_mm=float(np.sqrt(np.mean(np.sum(np.array(pos_errors)**2,axis=-1)))*1000),energy_drift_peak_J=float(np.max(np.abs(drift))),equation_residual_peak=float(max(residual))))
    t=Tree(body(2,(0,0,0)));s=t.state();act=np.zeros(t.d);act[-1]=1;acc,r=t.forward(s,act);initial=t.invariants(s)
    for _ in range(20):s=t.step(s,act,.002)
    final=t.invariants(s)
    report=dict(analytic_holdout=rows,free_recoil=dict(initial_acceleration=acc.tolist(),linear_momentum_drift=float(np.linalg.norm(np.array(final['linearMomentumKgMps'])-initial['linearMomentumKgMps'])),angular_momentum_drift=float(np.linalg.norm(np.array(final['angularMomentumCOMKgM2ps'])-initial['angularMomentumCOMKgM2ps']))))
    assert rows[-1]['angle_rms_deg']<rows[0]['angle_rms_deg']/10
    assert rows[-1]['energy_drift_peak_J']<.002
    assert report['free_recoil']['angular_momentum_drift']<2e-6
    with open(path,'x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main(sys.argv[1])
