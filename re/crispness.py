#!/usr/bin/env python3
"""Local surface thickness of a point cloud: per sampled point, sqrt of smallest PCA eigenvalue of
neighbours within R. Lower = crisper surfaces (less fog)."""
import sys, numpy as np
from scipy.spatial import cKDTree
def load(fn):
    raw=open(fn,'rb').read(); h=raw.index(b'end_header\n')+11
    return np.frombuffer(raw[h:],dtype=[('p','<f4',3),('c','u1',3)])['p'].astype(float)
R=0.05
for fn in sys.argv[1:]:
    P=load(fn); t=cKDTree(P); rng=np.random.default_rng(0); s=P[rng.choice(len(P),min(len(P),15000),replace=False)]
    th=[]
    for p,nb in zip(s,t.query_ball_point(s,R)):
        if len(nb)<8: continue
        Q=P[nb]-P[nb].mean(0); ev=np.linalg.eigvalsh(Q.T@Q/len(nb)); th.append(np.sqrt(max(ev[0],0)))
    th=np.array(th)
    print(f"{fn.split('/')[-1]:28s} pts {len(P):7d}  thickness median {np.median(th)*1000:5.1f} mm  p75 {np.percentile(th,75)*1000:5.1f} mm")
