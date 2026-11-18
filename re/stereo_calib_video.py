#!/usr/bin/env python3
"""Stereo extrinsics for the recorder video frames: R from far points (pure rotation RANSAC),
t direction from all matches with R fixed, |t| = metadata baseline. Multi-frame.
usage: stereo_calib_video.py meta.npz frames_dir f1,f2,.. out.json"""
import sys, json, numpy as np, cv2
from scipy.optimize import least_squares
m=np.load(sys.argv[1]); kb=json.loads(str(m['kb'])); d=sys.argv[2]; frames=[int(x) for x in sys.argv[3].split(',')]
K=[np.array([[k['fx'],0,k['cx']],[0,k['fy'],k['cy']],[0,0,1.]]) for k in kb]; D=[np.array(k['D']).reshape(4,1) for k in kb]
def bear(p,i):
    n=cv2.fisheye.undistortPoints(p[:,None],K[i],D[i])[:,0]; b=np.c_[n,np.ones(len(n))]; return b/np.linalg.norm(b,axis=1,keepdims=True)
s=cv2.SIFT_create(10000); B0=[];B1=[]
for f in frames:
    G=[cv2.imread(f'{d}/f{f}_cam{i}.png',0) for i in (0,1)]
    k0,d0=s.detectAndCompute(G[0],None); k1,d1=s.detectAndCompute(G[1],None)
    g=[a for a,b in cv2.BFMatcher().knnMatch(d0,d1,k=2) if a.distance<0.75*b.distance]
    B0.append(bear(np.float32([k0[x.queryIdx].pt for x in g]),0)); B1.append(bear(np.float32([k1[x.trainIdx].pt for x in g]),1))
b0=np.vstack(B0); b1=np.vstack(B1); print(len(b0),"matches")
def kabsch(a,b):
    U,_,Vt=np.linalg.svd(a.T@b); S=np.diag([1,1,np.sign(np.linalg.det(Vt.T@U.T))]); return Vt.T@S@U.T
rng=np.random.default_rng(0); best=(0,None)
for _ in range(4000):
    i=rng.choice(len(b0),2,replace=False); R=kabsch(b0[i],b1[i]); inl=np.linalg.norm(b1-b0@R.T,axis=1)<1.2/1475
    if inl.sum()>best[0]: best=(inl.sum(),inl)
far=best[1]; R=kabsch(b0[far],b1[far])
print(f"R from {far.sum()} far points: {np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0])):.2f}°, residual {np.median(np.linalg.norm(b1[far]-b0[far]@R.T,axis=1))*1475:.2f}px")
def tvec(p): th,ph=p; return np.array([np.cos(ph)*np.cos(th),np.cos(ph)*np.sin(th),np.sin(ph)])
def epi(t,R):
    E=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])@R; n=(E@b0.T).T; n/=np.linalg.norm(n,axis=1,keepdims=True); return np.sum(n*b1,1)*1475
near=~far
fits=[]
for th0 in np.linspace(-np.pi,np.pi,12,endpoint=False):
    r=least_squares(lambda p: epi(tvec(p),R)[near],[th0,0.0],loss='cauchy',f_scale=1.0,diff_step=1e-4); fits.append((np.median(np.abs(r.fun)),r.x))
fits.sort(key=lambda x:x[0]); t=tvec(fits[0][1])
# disambiguate sign with cheirality: points must be in front of both cameras
def tri_ok(t):
    # midpoint depth sign check on near matches
    A=np.stack([b0[near],-(b1[near]@R)],2)  # solve l0*b0 - l1*R^T b1 = -R^T t  (cam0 frame)
    rhs=-(R.T@t); l=np.array([np.linalg.lstsq(a,rhs,rcond=None)[0] for a in A[:500]]); return np.mean((l[:,0]>0)&(l[:,1]>0))
if tri_ok(-t)>tri_ok(t): t=-t
base=float(np.linalg.norm(m['cam0_from_cam1'][frames[0]][:3,3])); t=t*base
e=np.abs(epi(t/np.linalg.norm(t),R))
print(f"t = {np.round(t*1000,2)} mm (|t| from metadata {base*1000:.2f}); epipolar err all matches median {np.median(e):.2f}px, <1px {np.mean(e<1)*100:.0f}%, <2px {np.mean(e<2)*100:.0f}%; cheirality {tri_ok(t/base*1.0):.2f}")
json.dump({"R":R.tolist(),"t":t.tolist(),"frames":frames,"baseline_m":base,"method":"far-point rotation + fixed-R translation direction"},open(sys.argv[4],'w'),indent=1)
print("wrote",sys.argv[4])
