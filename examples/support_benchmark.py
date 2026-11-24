"""Public analytic scheduled-contact fixtures; independent of any game."""
import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from test_support import c,options,static
from test_tracking import target
from test_coupled import body
from tracking import track

def run():
    b=body(1);d=static(b);cases={}
    def save(name,b,d,op):
        x=track(b,d,op);cases[name]={'report':x['report'],'firstFrame':x['frames'][0] if x['frames'] else None,'lastFrame':x['frames'][-1] if x['frames'] else None};return x
    save('static',b,d,options([c()]));save('release',b,d,options([c(hi=4)]));save('switch',b,d,options([c('a',hi=4),c('b',lo=4)]));save('insufficient_friction',b,d,options([c(mu=.2)],externalLoads=[dict(bone=0,forceN=[10,0,0])]))
    save('impact_rejected',b,target(b,False,root_fn=lambda t:[0,0,.5*9.81*(.15**2-t*t)]),options([c(lo=4)]))
    b=body(3);d=target(b,False);cs=[c(str(i),local=p,anchor=p) for i,p in enumerate([(-1,-1,0),(2,-1,0),(2,1,0),(-1,1,0)])];save('moving_articulated',b,d,options(cs))
    assert all(cases[k]['report']['status']=='TRACKING_CANDIDATE' for k in ('static','switch','moving_articulated'))
    assert all(cases[k]['report']['status']=='REVIEW_REQUIRED' for k in ('release','insufficient_friction','impact_rejected'))
    return cases
if __name__=='__main__':Path(sys.argv[1]).write_text(json.dumps(run(),indent=2,allow_nan=False)+'\n')
