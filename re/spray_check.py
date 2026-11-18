#!/usr/bin/env python3
"""Spray metric: fraction of a frame's world points confirmed by another frame within TOL (cross-frame)."""
import sys, os, json, numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0,'tools'); from livestereo import LiveStereo, unpack_raw
d=sys.argv[1]; recs={json.loads(l)['i']:json.loads(l) for l in open(f'{d}/frames.jsonl')}
pairs=[(i,i+1) for i in range(5,55,7)]
CFG=[dict(),dict(block=7),dict(block=7,uniq=15),dict(block=7,uniq=15,pscale=2),dict(block=7,uniq=15,median=5),dict(block=9,uniq=15,pscale=2,median=5)]
for cfg in CFG:
    edge=cfg
    ls=LiveStereo(scale=0.35,max_depth=2.5,**cfg); conf=[]; npts=[]
    cache={}
    def W(i):
        if i not in cache:
            L=unpack_raw(open(f'{d}/{i:05d}_left.y10p','rb').read()); R=unpack_raw(open(f'{d}/{i:05d}_right.y10p','rb').read())
            P,_,_=ls.points(L,R); cache[i]=ls.to_world(P,recs[i]['left']['pose'])
        return cache[i]
    for a,b in pairs:
        A,B=W(a),W(b); dd,_=cKDTree(B).query(A[::3]); conf.append(np.mean(dd<0.03)); npts.append(len(A))
    print(f"{edge}: points/frame {int(np.mean(npts))}, confirmed by next frame (<3 cm) {np.mean(conf)*100:.1f}%  -> unconfirmed points/frame {int(np.mean(npts)*(1-np.mean(conf)))}")
