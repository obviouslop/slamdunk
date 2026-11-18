#!/usr/bin/env python3
"""Fit principal-point offsets of the recorder video frames so the metadata stereo geometry
(avcam1InAvcam0, with per-camera ±90° axis convention) explains feature matches.
Uses several frames. Writes captures/video_intrinsics.json."""
import sys, json, numpy as np, cv2
from scipy.optimize import least_squares
m=np.load(sys.argv[1]); kb0=json.loads(str(m['kb'])); d=sys.argv[2]; frames=[int(x) for x in sys.argv[3].split(',')]
S0=np.array([[0,1,0],[-1,0,0],[0,0,1.]]); S1=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
s=cv2.SIFT_create(6000); P0=[];P1=[];FR=[]
for f in frames:
    G=[cv2.imread(f'{d}/f{f}_cam{i}.png',0) for i in (0,1)]
    k0,d0=s.detectAndCompute(G[0],None); k1,d1=s.detectAndCompute(G[1],None)
    g=[a for a,b in cv2.BFMatcher().knnMatch(d0,d1,k=2) if a.distance<0.7*b.distance]
    P0.append(np.float32([k0[x.queryIdx].pt for x in g])); P1.append(np.float32([k1[x.trainIdx].pt for x in g])); FR+= [f]*len(g)
P0=np.vstack(P0); P1=np.vstack(P1); FR=np.array(FR); print(len(P0),"matches from",frames)
Rm=m['cam0_from_cam1'][frames[0]][:3,:3]; tm=m['cam0_from_cam1'][frames[0]][:3,3]
Rc=S1.T@Rm.T@S0; tc=-S1.T@Rm.T@tm; t_=tc/np.linalg.norm(tc)
E=np.array([[0,-t_[2],t_[1]],[t_[2],0,-t_[0]],[-t_[1],t_[0],0]])@Rc
def intr(x):
    kb=[dict(k) for k in kb0]; kb[0]['cx']+=x[0]; kb[0]['cy']+=x[1]; kb[1]['cx']+=x[2]; kb[1]['cy']+=x[3]; return kb
def bear(p,k):
    K=np.array([[k['fx'],0,k['cx']],[0,k['fy'],k['cy']],[0,0,1.]]); n=cv2.fisheye.undistortPoints(p[:,None],K,np.array(k['D']).reshape(4,1))[:,0]
    b=np.c_[n,np.ones(len(n))]; return b/np.linalg.norm(b,axis=1,keepdims=True)
def res(x):
    kb=intr(x); b0,b1=bear(P0,kb[0]),bear(P1,kb[1]); nrm=(E@b0.T).T; nrm/=np.linalg.norm(nrm,axis=1,keepdims=True)
    return np.sum(nrm*b1,1)*1475
for x0 in ([0,0,284,0],[0,0,300,0],[-150,0,150,0]):
    r=least_squares(res,np.array(x0,float),loss='cauchy',f_scale=1.0)
    e=np.abs(res(r.x)); print(f"start {x0}: offsets cam0 ({r.x[0]:+.1f},{r.x[1]:+.1f}) cam1 ({r.x[2]:+.1f},{r.x[3]:+.1f}) px -> epipolar median {np.median(e):.2f}px, <1px {np.mean(e<1)*100:.0f}%, <2px {np.mean(e<2)*100:.0f}%")
kb=intr(r.x); json.dump({"note":"KB intrinsics adjusted to recorder video frames (principal point offsets fitted, see re/fit_pp.py)","offsets_px":r.x.tolist(),"cams":kb,
  "S_meta_from_cv":{"cam0":S0.tolist(),"cam1":S1.tolist()}},open(sys.argv[4],'w'),indent=1)
print("wrote",sys.argv[4])
