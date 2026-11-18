#!/usr/bin/env python3
"""Pose/camera time-offset calibration from fast head rotations.
Camera inter-frame rotation (thumbnails, pure-rotation RANSAC on fisheye bearings) vs HMD rotation
over the same interval from the 500 Hz pose log, shifted by delta. Reports the best delta (relative
to arcgrab's mid-exposure estimate t_mid = ts_EOF - readout/2 - exposure/2)."""
import sys, os, json, numpy as np, cv2
from scipy.spatial.transform import Rotation as Rot, Slerp
sys.path.insert(0,'tools'); from livestereo import LiveStereo
d=sys.argv[1]; LT=4.36e-6
ls=LiveStereo(scale=0.25)
K=ls.KL.copy(); K[0,0]/=4; K[1,1]/=4; K[0,2]=(K[0,2]-1)/4; K[1,2]=K[1,2]/4; D=ls.DL
meta=[json.loads(l) for l in open(f'{d}/poses.jsonl')]
th=np.memmap(f'{d}/thumbs_left.u8',dtype=np.uint8,mode='r').reshape(-1,616,616)
n=min(len(meta),len(th)); meta=meta[:n]
tmid=np.array([m['ts']-2464*LT/2-m['exp_lines']*LT/2 for m in meta])
H=np.loadtxt(f'{d}/hmd_500hz.txt'); H=H[H[:,1]>0]
th_t=H[:,0]; q=H[:,[3,4,5,2]]   # scipy wants x,y,z,w
sl=Slerp(th_t,Rot.from_quat(q))
def kabsch(a,b):
    U,_,Vt=np.linalg.svd(a.T@b); S=np.diag([1,1,np.sign(np.linalg.det(Vt.T@U.T))]); return Vt.T@S@U.T
orb=cv2.ORB_create(1500,fastThreshold=10); bf=cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
def bear(p): u=cv2.fisheye.undistortPoints(p[:,None].astype(np.float32),K,D)[:,0]; b=np.c_[u,np.ones(len(u))]; return b/np.linalg.norm(b,axis=1,keepdims=True)
clahe=cv2.createCLAHE(3.0,(8,8))
prev=None; cam_rv=[]; idx=[]; rng=np.random.default_rng(0)
for i in range(n):
    img=clahe.apply(np.ascontiguousarray(th[i])); k,ds=orb.detectAndCompute(img,None)
    if prev is not None and ds is not None and prev[1] is not None and meta[i]['frame']==meta[i-1]['frame']+1:
        m=bf.match(prev[1],ds)
        if len(m)>=20:
            a=bear(np.float32([prev[0][x.queryIdx].pt for x in m])); b=bear(np.float32([k[x.trainIdx].pt for x in m]))
            best=(0,None)
            for _ in range(200):
                s=rng.choice(len(a),2,replace=False); R=kabsch(a[s],b[s]); inl=np.linalg.norm(b-a@R.T,axis=1)<3/368
                if inl.sum()>best[0]: best=(inl.sum(),inl)
            if best[0]>=15:
                R=kabsch(a[best[1]],b[best[1]])     # b = R a : camera(i) <- camera(i-1) for static points -> cam motion R^T
                cam_rv.append(Rot.from_matrix(R).as_rotvec()); idx.append(i)
    prev=(k,ds)
cam_rv=np.array(cam_rv); idx=np.array(idx)
print(f"{len(idx)} camera inter-frame rotations; |rot| p50 {np.degrees(np.median(np.linalg.norm(cam_rv,axis=1))):.2f}° p95 {np.degrees(np.percentile(np.linalg.norm(cam_rv,axis=1),95)):.2f}° per frame")
Xr=ls.head_from_camL[:3,:3]
def hmd_rv(delta):
    ta=np.clip(tmid[idx-1]+delta,th_t[0],th_t[-1]); tb=np.clip(tmid[idx]+delta,th_t[0],th_t[-1])
    Ra=sl(ta).as_matrix(); Rb=sl(tb).as_matrix()
    # head motion world: Hb = Ha * Rrel  -> camera: cam_b_from_cam_a = X^T Rrel^T X ; static points map b = that @ a
    out=[]
    for A,B in zip(Ra,Rb):
        Rrel=A.T@B; out.append(Rot.from_matrix(Xr.T@Rrel.T@Xr).as_rotvec())
    return np.array(out)
fast=np.linalg.norm(cam_rv,axis=1)>np.radians(0.5)
res=[]
for dl in np.arange(float(os.environ.get("DMIN","-0.5")),float(os.environ.get("DMAX","0.1"))+1e-4,0.0025):
    e=np.linalg.norm(hmd_rv(dl)-cam_rv,axis=1); res.append((dl,np.mean(e[fast]),np.median(e)))
res=np.array(res); b=res[np.argmin(res[:,1])]
for r in res[::4]: print(f"delta {r[0]*1000:+7.1f} ms: mean err (moving) {np.degrees(r[1]):.3f}°, median all {np.degrees(r[2]):.3f}°")
print(f"BEST delta {b[0]*1000:+.1f} ms: mean err {np.degrees(b[1]):.3f}° on {fast.sum()} moving frame pairs")
e0=np.linalg.norm(hmd_rv(b[0])-cam_rv,axis=1)
print("at best: mean err slow frames %.3f°, fast %.3f°"%(np.degrees(e0[~fast].mean()),np.degrees(e0[fast].mean())))
json.dump({"pose_delta_s":float(b[0]),"relative_to":"t_mid = ts_EOF - 2464*4.36us/2 - exp_lines*4.36us/2","recording":d},open(f'{d}/delay.json','w'),indent=1)

# ---- convention-free diagnostics
print("--- magnitude-only (convention free)")
cm=np.linalg.norm(cam_rv,axis=1); mres=[]
for dl in np.arange(float(os.environ.get("DMIN","-0.5")),float(os.environ.get("DMAX","0.1"))+1e-4,0.0025):
    hm=np.linalg.norm(hmd_rv(dl),axis=1); mres.append((dl,np.corrcoef(cm,hm)[0,1],np.mean(np.abs(cm-hm))))
mres=np.array(mres); bm=mres[np.argmax(mres[:,1])]
for r in mres[::8]: print(f"  delta {r[0]*1000:+7.1f} ms: corr {r[1]:.3f}, mean |d|w| {np.degrees(r[2]):.3f}°")
print(f"BEST (magnitude corr) delta {bm[0]*1000:+.1f} ms corr {bm[1]:.3f}")
hv=hmd_rv(bm[0]); hv_head=np.array([Xr@v for v in hv])   # back into head frame
def kab(a,b):
    U,_,Vt=np.linalg.svd(a.T@b); S=np.diag([1,1,np.sign(np.linalg.det(Vt.T@U.T))]); return Vt.T@S@U.T
Y=kab(hv_head[fast],cam_rv[fast])     # cam_rv ≈ Y @ head_rv
print("best-fit Y (cam <- head rotvec map):\n",np.round(Y,3),"\nassumed Xr^T:\n",np.round(Xr.T,3))
print("residual with Y: %.3f°, with Xr^T: %.3f°"%(np.degrees(np.mean(np.linalg.norm(cam_rv[fast]-hv_head[fast]@Y.T,axis=1))),np.degrees(np.mean(np.linalg.norm(cam_rv[fast]-hv_head[fast]@Xr,axis=1)))))
print("scale check |cam|/|head| median %.3f"%np.median(cm[fast]/np.maximum(np.linalg.norm(hv,axis=1)[fast],1e-6)))

print("--- around camera speed peaks: HMD angular speed (deg/s) vs time offset")
H_t=th_t; Rall=Rot.from_quat(q)
hw=np.r_[0,np.degrees(np.linalg.norm((Rall[:-1].inv()*Rall[1:]).as_rotvec(),axis=1))/np.diff(H_t)]
hw=np.convolve(hw,np.ones(15)/15,'same')      # ~30 ms smoothing
camw=np.degrees(cm)/np.maximum(np.diff(tmid)[idx-1],1e-3)
order=np.argsort(-camw)
picked=[]
for j in order:
    if all(abs(tmid[idx[j]]-tmid[idx[p]])>0.6 for p in picked): picked.append(j)
    if len(picked)==6: break
offs=np.arange(-0.4,0.41,0.05)
print("offset(ms): "+" ".join(f"{o*1000:+5.0f}" for o in offs))
for j in picked:
    t=0.5*(tmid[idx[j]]+tmid[idx[j]-1])
    print(f"cam {camw[j]:5.0f}°/s @t={t-tmid[0]:5.2f}s: "+" ".join(f"{np.interp(t+o,H_t,hw):5.0f}" for o in offs))
