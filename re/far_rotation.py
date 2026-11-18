#!/usr/bin/env python3
"""Fit the cam0->cam1 rotation from distant points only (pure-rotation model on bearings,
RANSAC). Decides between metadata (22.1°) and essential-matrix (31°) and checks intrinsics."""
import sys, json, numpy as np, cv2
m=np.load(sys.argv[1]); kb=json.loads(str(m['kb'])); f=int(sys.argv[2]); d=sys.argv[3]
K=[np.array([[k['fx'],0,k['cx']],[0,k['fy'],k['cy']],[0,0,1.]]) for k in kb]; D=[np.array(k['D']).reshape(4,1) for k in kb]
G=[cv2.imread(f'{d}/f{f}_cam{i}.png',0) for i in (0,1)]
s=cv2.SIFT_create(10000); k0,d0=s.detectAndCompute(G[0],None); k1,d1=s.detectAndCompute(G[1],None)
g=[a for a,b in cv2.BFMatcher().knnMatch(d0,d1,k=2) if a.distance<0.75*b.distance]
p0=np.float32([k0[x.queryIdx].pt for x in g]); p1=np.float32([k1[x.trainIdx].pt for x in g])
def bear(p,i):
    n=cv2.fisheye.undistortPoints(p[:,None],K[i],D[i])[:,0]; b=np.c_[n,np.ones(len(n))]; return b/np.linalg.norm(b,axis=1,keepdims=True)
b0,b1=bear(p0,0),bear(p1,1)
def kabsch(a,b):  # R s.t. b ≈ R a
    U,_,Vt=np.linalg.svd(a.T@b); S=np.diag([1,1,np.sign(np.linalg.det(Vt.T@U.T))]); return Vt.T@S@U.T
rng=np.random.default_rng(1); best=(0,None); thr=np.radians(1.5/1475*57.3)*57.3  # ~1.5 px in rad
thr=1.5/1475
for _ in range(3000):
    i=rng.choice(len(b0),2,replace=False); R=kabsch(b0[i],b1[i])
    err=np.linalg.norm(b1-b0@R.T,axis=1); inl=err<thr
    if inl.sum()>best[0]: best=(inl.sum(),inl)
inl=best[1]; R=kabsch(b0[inl],b1[inl]); err=np.linalg.norm(b1-b0@R.T,axis=1)
ang=np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0]))
print(f"{len(g)} matches; far-point (pure rotation) inliers {inl.sum()} ({inl.mean()*100:.0f}%), rotation {ang:.2f}°, residual median {np.median(err[inl])*1475:.2f} px")
ys=p0[inl,1]; print("  inlier rows in cam0 image p10/p50/p90:",np.percentile(ys,[10,50,90]).round())
# compare with metadata rotation under the convention found earlier
S0=np.array([[0,1,0],[-1,0,0],[0,0,1.]]); S1=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
Rm=m['cam0_from_cam1'][f][:3,:3]; Rmeta=S1.T@Rm.T@S0
print(f"  metadata rotation (conv.) {np.degrees(np.linalg.norm(cv2.Rodrigues(Rmeta)[0])):.2f}°, angle between far-fit and metadata: {np.degrees(np.linalg.norm(cv2.Rodrigues(R@Rmeta.T)[0])):.2f}°")
errm=np.linalg.norm(b1[inl]-b0[inl]@Rmeta.T,axis=1); print(f"  metadata rotation residual on far points: median {np.median(errm)*1475:.1f} px")
json.dump({"R_far":R.tolist(),"frame":f},open(sys.argv[4],'w'))
