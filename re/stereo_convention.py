#!/usr/bin/env python3
"""Estimate cam0<->cam1 relative pose from image features (fisheye-undistorted) and find the
axis convention that makes the recorder's avcam1InAvcam0 agree."""
import sys, json, itertools, numpy as np, cv2
m=np.load(sys.argv[1]); kb=json.loads(str(m['kb'])); f=int(sys.argv[2]); d=sys.argv[3]
I0=cv2.imread(f'{d}/f{f}_cam0.png',0); I1=cv2.imread(f'{d}/f{f}_cam1.png',0)
def K(k): return np.array([[k['fx'],0,k['cx']],[0,k['fy'],k['cy']],[0,0,1.]])
K0,K1=K(kb[0]),K(kb[1]); D0,D1=np.array(kb[0]['D']),np.array(kb[1]['D'])
sift=cv2.SIFT_create(8000)
k0,d0=sift.detectAndCompute(I0,None); k1,d1=sift.detectAndCompute(I1,None)
mt=cv2.BFMatcher().knnMatch(d0,d1,k=2); good=[a for a,b in mt if a.distance<0.75*b.distance]
p0=np.float32([k0[g.queryIdx].pt for g in good]); p1=np.float32([k1[g.trainIdx].pt for g in good])
n0=cv2.fisheye.undistortPoints(p0[:,None],K0,D0)[:,0]; n1=cv2.fisheye.undistortPoints(p1[:,None],K1,D1)[:,0]
E,mask=cv2.findEssentialMat(n0,n1,np.eye(3),cv2.RANSAC,0.999,1e-3)
_,R,t,mask2=cv2.recoverPose(E,n0,n1,np.eye(3),mask=mask)   # X1 = R X0 + t  (OpenCV cams)
print(f"{len(good)} matches, {int(mask2.sum()//255 if mask2.max()>1 else mask2.sum())} inliers")
# image says: cam1_from_cam0 (OpenCV) = R, t (unit). Metadata: cam0_from_cam1 = C
C=m['cam0_from_cam1'][f]; Rm=C[:3,:3]; tm=C[:3,3]
Rm_c1c0=Rm.T; tm_c1c0=-Rm.T@tm
def ang(A,B): return np.degrees(np.arccos(np.clip((np.trace(A.T@B)-1)/2,-1,1)))
print("image-estimated cam1_from_cam0: R euler(deg)",np.round(cv2.RQDecomp3x3(R)[0],2)," t dir",np.round(t.ravel(),3))
best=[]
Ss=[]
for perm in itertools.permutations(range(3)):
    for sg in itertools.product((1,-1),repeat=3):
        S=np.zeros((3,3))
        for k,(p,q) in enumerate(zip(perm,sg)): S[k,p]=q
        if np.linalg.det(S)>0: Ss.append(S)
tdir=t.ravel()/np.linalg.norm(t)
for S0 in Ss:
    for S1 in Ss:
        # meta frames: X_meta_i = S_i X_cv_i ; meta: X_m0 = Rm X_m1 + tm
        # => X_cv1 = S1^T Rm^T (S0 X_cv0 - tm)
        Rc=S1.T@Rm.T@S0; tc=-S1.T@Rm.T@tm
        e=ang(Rc,R); et=np.degrees(np.arccos(np.clip(tc@tdir/np.linalg.norm(tc),-1,1)))
        best.append((e+et,e,et,S0,S1,Rc,tc))
best.sort(key=lambda x:x[0])
for tot,e,et,S0,S1,Rc,tc in best[:4]:
    print(f"total {tot:6.2f}: rot err {e:5.2f}°, t-dir err {et:5.2f}°; S0=\n{S0.astype(int)}\nS1=\n{S1.astype(int)}\n t_cv {np.round(tc*1000,1)} mm, rot angle {np.degrees(np.arccos((np.trace(Rc)-1)/2)):.1f}°")

# --- epipolar residuals (angular, deg) for candidate geometries on inlier bearings
inl=mask2.ravel()>0
def bearing(n): b=np.c_[n,np.ones(len(n))]; return b/np.linalg.norm(b,axis=1,keepdims=True)
b0,b1=bearing(n0[inl]),bearing(n1[inl])
def epi(Rc,tc):
    t_=tc/np.linalg.norm(tc); E=np.array([[0,-t_[2],t_[1]],[t_[2],0,-t_[0]],[-t_[1],t_[0],0]])@Rc
    # angle of b1 from epipolar plane normal to (E b0)
    nrm=(E@b0.T).T; nrm/=np.linalg.norm(nrm,axis=1,keepdims=True)
    return np.degrees(np.abs(np.arcsin(np.sum(nrm*b1,1))))
_,e_,et_,S0b,S1b,Rcb,tcb=best[0]
for name,(Rc,tc) in {"image essential":(R,t.ravel()),"metadata (best convention)":(Rcb,tcb)}.items():
    r=epi(Rc,tc); print(f"{name:28s} epipolar err median {np.median(r)*60:6.2f} arcmin, p90 {np.percentile(r,90)*60:6.2f} arcmin (1 px ≈ {np.degrees(1/1475)*60:.2f} arcmin)")
# where do features lie in the image (radius)?
rr=np.linalg.norm(p0[inl]-np.array([kb[0]['cx'],kb[0]['cy']]),axis=1); print("feature radius px: p10/p50/p90",np.percentile(rr,[10,50,90]).round())

print("--- variants of metadata interpretation")
res=[]
for vname,(RmV,tmV) in {"child-in-parent":(Rm,tm),"R transposed":(Rm.T,tm),"inverse":(Rm.T,-Rm.T@tm),"R^T, t as-is neg":(Rm.T,-tm)}.items():
    for S0 in Ss:
        for S1 in Ss:
            Rc=S1.T@RmV.T@S0; tc=-S1.T@RmV.T@tmV
            r=epi(Rc,tc); res.append((np.median(r)*60,vname,S0,S1,Rc,tc))
res.sort(key=lambda x:x[0])
for med,vn,S0,S1,Rc,tc in res[:3]:
    print(f"{vn:18s} median epi {med:6.2f} arcmin  rot-vs-image {ang(Rc,R):5.2f}°  toe {np.degrees(np.arccos((np.trace(Rc)-1)/2)):.1f}°  t_cv {np.round(tc*1000,1)}  S0 {S0.astype(int).tolist()} S1 {S1.astype(int).tolist()}")
