"""Reproduce contact acceptance cases using only analytic synthetic fixtures."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from test_contact import fixture, check, s, torch

def main(path):
    torch.set_num_threads(1);cases=[]
    for name in ('static','flight','hover','accelerate','single_support'):
        d,b,G,P,L=fixture('static' if name=='single_support' else name)
        if name=='single_support':d['solverHints']['physicalContacts']=d['solverHints']['physicalContacts'][:1]
        # Withhold intermediate keys for static and flight; no hidden rotation motion.
        if name in ('static','flight'):d['keys']=d['keys'][::4]
        d['solverHints']['contactOptions']={'forceToleranceN':.1,'momentToleranceNm':.1,'anchorToleranceM':.001}
        exact=check(d,b,G,P);out,r,*_=s.solve(d,b,L,120,max_key_mm=5)
        final=r['contact_final']; expected=name in ('static','flight')
        assert (final['status']=='SAMPLED_CONTACT_FEASIBLE')==expected, (name,r['flags'])
        assert out['originalAuthorKeys']==d['keys'] and out['timeline']==d['timeline']
        cases.append({'case':name,'keys':len(d['keys']),'frames':len(P),'analytic_contact_status':exact['status'],'solver_status':r['status'],'flags':r['flags'],'contact_status':final['status'],'force_residual_max_N':final['force_residual_max_N'],'moment_residual_max_Nm':final['moment_residual_max_Nm'],'anchor_gap_max_m':final['anchor_gap_max_m'],'key_max_mm':r['final']['key_max_mm'],'first_frame_forces':final['frames'][0]['forces']})
    s.write(path,{'version':s.VERSION,'cases':cases,'passed':len(cases),'note':'Expected failures must be rejected. Sampled wrench feasibility is not full physical acceptance.'})
if __name__=='__main__':main(sys.argv[1])
