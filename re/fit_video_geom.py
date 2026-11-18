#!/usr/bin/env python3
"""Joint fit for recorder video frames: principal points of both cams (4) + small rotation
correction to metadata stereo R (3); t from metadata. Minimises epipolar error on matches.
usage: fit_video_geom.py meta.npz frames_dir f1,f2,.. out.json"""
import sys, json, numpy as np, cv2
from scipy.optimize import least_squares
m=np.load(sys.argv[1]); kb0=json.loads(str(m['kb'])); d=sys.argv[2]; frames=[int(x) for x in sys.argv[3].split(',')]
S0=np.array([[0,1,0],[-1,0,0],[0,0,1.]]); S1=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
Rm=m['cam0_from_cam1'][frames[0]][:3,:3]; tm=m['cam0_from_cam1'][frames[0]][:3,3]
Rmeta=S1.T@Rm.T@S0; tmeta=-S1.T@Rm.T@tm
s=cv2.SIFT_create(10000); P0=[];P1=[]
for f in frames:
    G=[cv2.imread(f'{d}/f{f}_cam{i}.png',0) for i in (0,1)]
    k0,d0=s.detectAndCompute(G[0],None); k1,d1=s.detectAndCompute(G[1],None)
    g=[a for a,b in cv2.BFMatcher().knnMatch(d0,d1,k=2) if a.distance<0.75*b.distance]
    P0.append(np.float64([k0[x.queryIdx].pt for x in g])); P1.append(np.float64([k1[x.trainIdx].pt for x in g]))
P0=np.vstack(P0); P1=np.vstack(P1); print(len(P0),"matches")
def bear(p,k,dcx,dcy):
    K=np.array([[k['fx'],0,k['cx']+dcx],[0,k['fy'],k['cy']+dcy],[0,0,1.]])
    n=cv2.fisheye.undistortPoints(p[:,None].astype(np.float64),K,np.array(k['D'],float).reshape(4,1))[:,0]
    b=np.c_[n,np.ones(len(n))]; return b/np.linalg.norm(b,axis=1,keepdims=True)
def geom(x):
    R=cv2.Rodrigues(np.radians(x[4:7]))[0]@Rmeta; return R,tmeta/np.linalg.norm(tmeta)
FIX0='--fix-cam0' in sys.argv
def res(x):
    if FIX0: x=np.r_[0,0,x[2:]]
    b0=bear(P0,kb0[0],x[0],x[1]); b1=bear(P1,kb0[1],x[2],x[3]); R,t=geom(x)
    E=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])@R; n=(E@b0.T).T; n/=np.linalg.norm(n,axis=1,keepdims=True)
    return np.sum(n*b1,1)*1475
best=None
for start in ([0,0,284,0,0,0,0],[0,0,300,0,0,0,0],[-142,0,142,0,0,0,0],[0,0,0,0,0,0,0]):
    r=least_squares(res,np.array(start,float),loss='cauchy',f_scale=1.0,diff_step=1e-3,x_scale=[50,50,50,50,0.5,0.5,0.5])
    e=np.abs(r.fun); print(f"start {start[:4]} -> pp0 ({r.x[0]:+.1f},{r.x[1]:+.1f}) pp1 ({r.x[2]:+.1f},{r.x[3]:+.1f}) dR {np.round(r.x[4:],2)}°: epi median {np.median(e):.2f}px <1px {np.mean(e<1)*100:.0f}% <2px {np.mean(e<2)*100:.0f}%")
    if FIX0: r.x[:2]=0
    if best is None or np.median(e)<best[0]: best=(np.median(e),r.x)
x=best[1]; R,tu=geom(x); t=tu*np.linalg.norm(tmeta)
kb=[dict(k) for k in kb0]; kb[0]['cx']+=x[0]; kb[0]['cy']+=x[1]; kb[1]['cx']+=x[2]; kb[1]['cy']+=x[3]
json.dump({"cams":kb,"R":R.tolist(),"t":t.tolist(),"pp_offsets_px":x[:4].tolist(),"dR_deg":x[4:].tolist(),"frames":frames,
  "S_meta_from_cv":{"cam0":S0.tolist(),"cam1":S1.tolist()},"epi_median_px":best[0]},open(sys.argv[4],'w'),indent=1)
print("wrote",sys.argv[4], "toe angle %.2f°"%np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0])))
