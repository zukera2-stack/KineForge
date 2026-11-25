"""Independent two-link holdout plus prescribed schedule selection and rejection."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from test_coupled import body,analytic
from test_tracking import target
from test_horizon import opts,land
from horizon import optimize
from coupled import Tree

def run():
    b=body();times=np.arange(9)/40;ids=[0,4,8];q0=[.2,-.4,.3,-.2];torque=np.array([1.2,-.6])
    def ode(t,y):
        c=np.cos(y[1]);M=np.array([[1.7+c,.35+.5*c],[.35+.5*c,.35]]);a=analytic(y);a[2:]+=np.linalg.solve(M,torque);return a
    truth=solve_ivp(ode,(0,times[-1]),q0,t_eval=times,rtol=1e-12,atol=1e-14).y.T
    sp=CubicSpline(times[ids],truth[ids,:2],axis=0);d=target(b,fps=40,frames=9,angle_fn=lambda t,i:float(sp(t)[i-1]));d['originalAuthorKeys']=[d['keyframes'][i] for i in ids];d['sourceKeys']=d['originalAuthorKeys'];d['lockedFrames']=[i+1 for i in ids]
    o=optimize(b,d,opts(controlBlocks=1,substeps=4,initialVelocity=q0[2:],maxKeyErrorMm=1,maxReplayDifferenceMm=.2,targetWeight=0,orientationWeightMPerRad=0));r=o['report']['candidates'][0]
    tr=Tree(b,True);p=[];q=[]
    for y in truth:
        L=tr.N.copy();L[1:]=R.from_rotvec(np.c_[np.zeros(2),y[:2],np.zeros(2)]).as_matrix();p.append(tr.kinematics(tr.state(local=L))['P']);q.append(L)
    pc=np.array([f['positions'] for f in o['frames']]);lc=R.from_quat(np.array([f['localQuaternionXYZW'] for f in o['frames']]).reshape(-1,4)).as_matrix().reshape(9,3,3,3);hidden=[i for i in range(9) if i not in ids];p=np.array(p);q=np.array(q)
    err=np.linalg.norm(pc[hidden]-p[hidden],axis=-1)*1000;initial=np.linalg.norm(np.array([f['positions'] for f in d['keyframes']])[hidden]-p[hidden],axis=-1)*1000
    angular=np.degrees(R.from_matrix((lc[hidden]@q[hidden].swapaxes(-1,-2)).reshape(-1,3,3)).magnitude())
    holdout=dict(report=o['report'],trueControlNm=torque.tolist(),hiddenKeyframeIndices=hidden,hiddenPositionRmsMm=float(np.sqrt(np.mean(err**2))),interpolatedTargetHiddenRmsMm=float(np.sqrt(np.mean(initial**2))),hiddenLocalAngleRmsDeg=float(np.sqrt(np.mean(angular**2))))
    b=body(1);d,c=land(b);landing=optimize(b,d,opts(fixedBase=False,scheduleCandidates=[c(2),c(3)],supportActivationImpulse=True,initialVelocity=[0]*6));hover=optimize(b,target(b,False,frames=5,root_fn=lambda t:[0,0,2]),opts(fixedBase=False));b=body(2);b['jointModels']['link1']['actuatorLimitNm']=.01;limited=optimize(b,target(b,frames=4),opts())
    assert o['report']['status']=='HORIZON_CANDIDATE' and holdout['hiddenPositionRmsMm']<holdout['interpolatedTargetHiddenRmsMm'] and landing['report']['selectedScheduleIndex']==1 and landing['report']['status']=='HORIZON_CANDIDATE' and hover['report']['status']=='REVIEW_REQUIRED' and limited['report']['status']=='REVIEW_REQUIRED'
    return dict(analyticHoldout=holdout,landingSchedule=landing['report'],hover=hover['report'],limitedActuator=limited['report'])
if __name__=='__main__':Path(sys.argv[1]).write_text(json.dumps(run(),indent=2,allow_nan=False)+'\n')
