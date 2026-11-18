#!/usr/bin/env python3
"""Calibrate pose time offset for live recordings: world consistency of stereo clouds vs. delta.
Poses in poses.jsonl were sampled at t_mid = ts - readout/2 - exp/2 (arcgrab). We interpolate
pose(t_mid_i + delta) and score cross-frame nearest-neighbour distance."""
import sys, json, numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot, Slerp
sys.path.insert(0,'tools'); from livestereo import LiveStereo, unpack_raw
d=sys.argv[1]; LT=4.36e-6
ls=LiveStereo(scale=0.5,max_depth=2.5)
pl=[json.loads(l) for l in open(f'{d}/poses.jsonl')]
tp=np.array([p['ts']-2464*LT/2-p['exp_lines']*LT/2 for p in pl]); o=np.argsort(tp); tp=tp[o]
Pm=np.array([p['pose'] for p in pl])[o]
keep=np.r_[True,np.diff(tp)>1e-4]; tp,Pm=tp[keep],Pm[keep]
sl=Slerp(tp,Rot.from_matrix(Pm[:,:,:3]))
def pose_at(t):
    t=np.clip(t,tp[0],tp[-1]); M=np.eye(4); M[:3,:3]=sl([t]).as_matrix()[0]
    M[:3,3]=[np.interp(t,tp,Pm[:,k,3]) for k in range(3)]; return M
recs=[json.loads(l) for l in open(f'{d}/frames.jsonl')]
rng=np.random.default_rng(0); clouds={}; tm={}
ids=list(range(0,len(recs),2))
for r in (recs[i] for i in ids):
    i=r['i']; L=unpack_raw(open(f'{d}/{i:05d}_left.y10p','rb').read()); R=unpack_raw(open(f'{d}/{i:05d}_right.y10p','rb').read())
    P,_,_=ls.points(L,R); clouds[i]=P[rng.choice(len(P),min(len(P),25000),replace=False)]
    tm[i]=r['left']['ts']-2464*LT/2-r['left']['exp_lines']*LT/2
print(len(clouds),"frames prepared")
# angular speed per frame (for reporting)
pairs=[(a,b) for k,a in enumerate(ids) for b in ids[k+1:k+5]]
X=ls.head_from_camL
def score(delta):
    W={i:pose_at(tm[i]+delta)@X for i in clouds}
    wc={i:clouds[i]@W[i][:3,:3].T+W[i][:3,3] for i in clouds}
    trees={}; s=[]
    for a,b in pairs:
        if b not in trees: trees[b]=cKDTree(wc[b])
        dd,_=trees[b].query(wc[a][::5]); s.append(np.percentile(dd,40))
    return np.mean(s)
res=[(dl,score(dl/1000)) for dl in range(-80,81,10)]
for dl,s in res: print(f"delta {dl:+4d} ms: score {s*100:.2f} cm")
best=min(res,key=lambda x:x[1])[0]
fine=[(dl,score(dl/1000)) for dl in np.arange(best-8,best+9,2)]
for dl,s in fine: print(f"  fine {dl:+5.1f} ms: {s*100:.3f} cm")
print("BEST delta (ms):",min(fine,key=lambda x:x[1])[0])
