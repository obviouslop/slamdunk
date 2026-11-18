#!/usr/bin/env python3
"""Find cam0 convention S (24 rotations) and a small correction Rcorr so per-frame stereo clouds
coincide in world: X_w = W[f] (S Rcorr X_cv) . Score: median NN distance between frame clouds."""
import sys, json, itertools, numpy as np, cv2
from scipy.spatial import cKDTree
from scipy.optimize import least_squares
m=np.load(sys.argv[1]); wd=sys.argv[2]; ids=[int(x) for x in sys.argv[3].split(',')]
def read(f):
    raw=open(f'{wd}/f{f}.ply','rb').read(); h=raw.index(b'end_header\n')+11
    P=np.frombuffer(raw[h:],dtype=[('p','<f4',3),('c','u1',3)])['p'].astype(float); P=P[np.linalg.norm(P,axis=1)<4]
    return P[np.random.default_rng(0).choice(len(P),min(len(P),40000),replace=False)]
cl={f:read(f) for f in ids}; W={f:m['world_from_cam0'][f] for f in ids}
pairs=[(ids[i],ids[j]) for i in range(len(ids)) for j in range(i+1,len(ids))]
for i,j in pairs:
    dR=W[i][:3,:3].T@W[j][:3,:3]; print(f"pair {i}-{j}: move {np.linalg.norm(W[i][:3,3]-W[j][:3,3])*100:.1f} cm, rot {np.degrees(np.linalg.norm(cv2.Rodrigues(dR)[0])):.1f}°")
Ss=[]
for perm in itertools.permutations(range(3)):
    for sg in itertools.product((1,-1),repeat=3):
        S=np.zeros((3,3))
        for k,(p,q) in enumerate(zip(perm,sg)): S[k,p]=q
        if np.linalg.det(S)>0: Ss.append(S)
def score(M):
    d=[]
    for i,j in pairs:
        a=cl[i]@M.T@W[i][:3,:3].T+W[i][:3,3]; b=cl[j]@M.T@W[j][:3,:3].T+W[j][:3,3]
        dd,_=cKDTree(b).query(a[::4]); d.append(np.median(dd))
    return np.mean(d)
res=sorted((score(S),k) for k,S in enumerate(Ss))
for sc,k in res[:3]: print(f"S#{k} {Ss[k].astype(int).tolist()}: median NN {sc*100:.2f} cm")
S=Ss[res[0][1]]
def f(x): return score(S@cv2.Rodrigues(np.radians(x))[0])
best=(f([0,0,0]),np.zeros(3))
for it in range(3):   # coarse-to-fine grid search on Rcorr
    step=[4,1,0.25][it]; c=best[1]
    for dx in (-2,-1,0,1,2):
        for dy in (-2,-1,0,1,2):
            for dz in (-2,-1,0,1,2):
                x=c+step*np.array([dx,dy,dz]); s=f(x)
                if s<best[0]: best=(s,x)
    print(f"  refine step {step}°: Rcorr {np.round(best[1],2)}° -> {best[0]*100:.2f} cm")
print(json.dumps({"S":S.tolist(),"rcorr_deg":best[1].tolist(),"median_nn_cm":best[0]*100}))
