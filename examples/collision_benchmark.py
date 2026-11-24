"""Reproduce synthetic collision tests without private data."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
from test_collision import case,check,s

def main(path):
    cases=[]
    for name,angles,obstacle,steps,flag in [('clear',[0,30,0],(5,5,5),64,None),('static',[90,90],(1,1,1),64,'COLLISION_PENETRATION'),('tunneling',[10,170],(1,1,1),64,'COLLISION_PENETRATION'),('uncertified',[10,170],(1,1.401,1),2,'COLLISION_INTERVAL_UNCERTIFIED')]:
        d,b,P,L=case(angles,obstacle,steps=steps);r=check(d,b,P,L);assert (flag in r['flags']) if flag else not r['flags'];cases.append({'case':name,**r})
    d,b,P,L=case([0,25,50,75,50,25,0]);out,r,*_=s.solve(d,b,L,100,max_key_mm=5);assert not r['collision_final']['flags'];assert out['originalAuthorKeys']==d['keys'];assert out['timeline']==d['timeline'];cases.append({'case':'repair','initial':r['collision_initial'],'final':r['collision_final'],'key_max_mm':r['final']['key_max_mm'],'solver_status':r['status'],'solver_flags':r['flags']})
    s.write(path,{'version':s.VERSION,'passed':len(cases),'cases':cases})
if __name__=='__main__':main(sys.argv[1])
