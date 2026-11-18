import sys, json, numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0,'tools'); from livestereo import LiveStereo, unpack_raw
d=sys.argv[1]; ls=LiveStereo(scale=0.5,max_depth=2.5)
recs={json.loads(l)['i']:json.loads(l) for l in open(f'{d}/frames.jsonl')}
def cloud(i):
    L=unpack_raw(open(f'{d}/{i:05d}_left.y10p','rb').read()); R=unpack_raw(open(f'{d}/{i:05d}_right.y10p','rb').read())
    P,_,_=ls.points(L,R); P=P[np.random.default_rng(0).choice(len(P),min(len(P),60000),replace=False)]; return P
import cv2
def H(i): M=np.eye(4); M[:3,:]=np.array(recs[i]['left']['pose']); return M
for a,b in ((20,24),(60,64),(100,104),(20,40),(60,80)):
    Pa,Pb=cloud(a),cloud(b)
    for name,X in (("X0",ls.head_from_camL),):
        Wa=H(a)@X; Wb=H(b)@X
        wa=Pa@Wa[:3,:3].T+Wa[:3,3]; wb=Pb@Wb[:3,:3].T+Wb[:3,3]
        dd,_=cKDTree(wb).query(wa)
        rel=np.linalg.inv(H(b))@H(a)
        print(f"{a}-{b}: head moved {np.linalg.norm(rel[:3,3])*100:.0f} cm / {np.degrees(np.linalg.norm(cv2.Rodrigues(rel[:3,:3])[0])):.0f}°, world NN median {np.median(dd)*100:.1f} cm, p25 {np.percentile(dd,25)*100:.1f} cm")
