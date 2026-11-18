#!/usr/bin/env python3
"""Ground-truth check: per phase (floor/wall/ceiling by head pitch), RANSAC the dominant plane in
RAW mesh coordinates and report its distance and normal (in mesh axes and in head axes)."""
import sys, numpy as np
NV=17280
rec=np.dtype([('seq','<u4'),('t','<f8'),('m','<f4',(3,4)),('hmd','<f4',(13,3,4)),('xyz','<f4',(NV,3))])
fr=np.fromfile(sys.argv[1],dtype=rec)
H=fr['hmd'][:,5]                       # pose at -50 ms
fwd=-H[:,:,2]; pitch=np.degrees(np.arcsin(np.clip(fwd[:,1],-1,1)))
tt=fr['t']-fr['t'][0]
print("pitch timeline (every ~1s):", [f"{tt[i]:.0f}s:{pitch[i]:+.0f}" for i in range(0,len(fr),max(1,len(fr)//25))])
rng=np.random.default_rng(0)
def ransac(P,iters=400,th=0.03):
    best=(0,None)
    for _ in range(iters):
        s=P[rng.choice(len(P),3,replace=False)]; n=np.cross(s[1]-s[0],s[2]-s[0]); nn=np.linalg.norm(n)
        if nn<1e-9: continue
        n/=nn; d=n@s[0]; inl=np.abs(P@n-d)<th
        if inl.sum()>best[0]: best=(inl.sum(),(n,d,inl))
    n,d,inl=best[1]; Q=P[inl]; c=Q.mean(0); n=np.linalg.svd(Q-c)[2][2]; d=n@c
    if d<0: n,d=-n,-d
    return n,d,inl.mean()
for name,sel in (('FLOOR (look down)',pitch<-55),('WALL (level)',np.abs(pitch)<15),('CEILING (look up)',pitch>55)):
    idx=np.nonzero(sel)[0]
    if len(idx)==0: print(f"{name}: no frames"); continue
    print(f"== {name}: {len(idx)} frames, head standing-ish y n/a; mean pitch {pitch[idx].mean():+.1f}")
    for i in idx[[len(idx)//4, len(idx)//2, 3*len(idx)//4]]:
        P=fr['xyz'][i].astype(float); dist=np.linalg.norm(P,axis=1)
        n,d,frac=ransac(P)
        # gravity "up" in head axes: R^T * (0,1,0)
        up_head=H[i][:,:3].T@np.array([0,1,0.])
        print(f"   f{i} t={tt[i]:5.1f}s pitch {pitch[i]:+5.1f}: plane dist {d:.3f} m, inliers {frac*100:4.1f}%, normal(mesh) {np.round(n,2)}, |v| median {np.median(dist):.2f} min {dist.min():.2f} max {dist.max():.2f}; up(head) {np.round(up_head,2)}")
