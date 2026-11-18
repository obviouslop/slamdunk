#!/usr/bin/env python3
"""Find the transform that makes roomview3d vertices static in world space."""
import sys, numpy as np, itertools
from scipy.spatial import cKDTree
NV=17280; rec=np.dtype([('seq','<u4'),('t','<f8'),('m','<f4',(3,4)),('hmd','<f4',(3,4)),('xyz','<f4',(NV,3))])
fr=np.fromfile(sys.argv[1],dtype=rec)
print(len(fr),"frames, t span %.2fs"%(fr['t'][-1]-fr['t'][0]))
def rt(m): return m[:,:3].astype(np.float64), m[:,3].astype(np.float64)
def rotdeg(a,b):
    R=a[:,:3].T@b[:,:3]; return np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))
# pick pairs with large head rotation
pairs=[]
for i in range(0,len(fr),10):
    for j in range(i+15,min(len(fr),i+60),5):
        pairs.append((rotdeg(fr['m'][i],fr['m'][j]),i,j))
pairs=sorted(pairs)[-6:]
print("test pairs (deg,i,j):",[(round(a,1),i,j) for a,i,j in pairs])
flips={f"S{s}":np.diag(s) for s in [(1,1,1),(1,-1,-1),(-1,1,-1),(-1,-1,1)]}
def cand(name,f,v):
    S=flips[name.split('|')[0]]; kind=name.split('|')[1]
    R,t=rt(f['m'] if 'trailer' in kind else f['hmd'])
    vv=v@S.T
    if kind.endswith('inv'): return (vv-t)@R   # R^T (v - t)
    return vv@R.T+t
names=[f"{s}|{k}" for s in flips for k in ('trailer','trailer_inv','hmd')]
res=[]
for nm in names:
    errs=[]
    for _,i,j in pairs:
        a=cand(nm,fr[i],fr['xyz'][i].astype(np.float64)); b=cand(nm,fr[j],fr['xyz'][j].astype(np.float64))
        d,_=cKDTree(b).query(a[::7]); errs.append(np.median(d))
    res.append((np.mean(errs),nm))
for e,nm in sorted(res): print(f"  {nm:22s} median NN {e*100:6.2f} cm")

print("-- identity (vertices already world-anchored?)")
for _,i,j in pairs:
    a=fr['xyz'][i].astype(np.float64); b=fr['xyz'][j].astype(np.float64)
    d,_=cKDTree(b).query(a[::7]); print(f"  pair {i},{j}: {np.median(d)*100:.2f} cm")
print("-- baseline consecutive frames, identity / S(1,1,1)|trailer")
for i in (5,100,200):
    a=fr['xyz'][i].astype(np.float64); b=fr['xyz'][i+1].astype(np.float64)
    d0,_=cKDTree(b).query(a[::7])
    a2=cand('S(1, 1, 1)|trailer',fr[i],a); b2=cand('S(1, 1, 1)|trailer',fr[i+1],b)
    d1,_=cKDTree(b2).query(a2[::7]); print(f"  {i}: id {np.median(d0)*100:.2f} cm, trailer {np.median(d1)*100:.2f} cm, rot {rotdeg(fr['m'][i],fr['m'][i+1]):.2f} deg")
