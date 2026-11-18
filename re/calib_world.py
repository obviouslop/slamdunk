#!/usr/bin/env python3
"""Joint calibration of head_from_camL (6 DoF) and pose time offset delta from a live recording.
Residual = world-space distance between the same physical feature triangulated (stereo) in two
different frames: H(t_a+d) X Xa - H(t_b+d) X Xb. Pairs up to 2 s apart (big rotations constrain X)."""
import sys, json, numpy as np, cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot, Slerp
sys.path.insert(0,'tools'); from livestereo import LiveStereo, unpack_raw
d=sys.argv[1]; LT=4.36e-6
ls=LiveStereo(scale=0.5)
import os
if os.path.exists(f'{d}/hmd_500hz.txt'):   # true pose history (arcgrab --pose-log): t valid qw qx qy qz px py pz
    Hh=np.loadtxt(f'{d}/hmd_500hz.txt'); Hh=Hh[Hh[:,1]>0]; tp=Hh[:,0]
    Rm=Rot.from_quat(Hh[:,[3,4,5,2]]).as_matrix(); Pm=np.concatenate([Rm,Hh[:,6:9,None]],2); print("pose source: 500 Hz history")
else:
    pl=[json.loads(l) for l in open(f'{d}/poses.jsonl')]
    tp=np.array([p['ts']-2464*LT/2-p['exp_lines']*LT/2 for p in pl]); o=np.argsort(tp); tp=tp[o]; Pm=np.array([p['pose'] for p in pl])[o]
k=np.r_[True,np.diff(tp)>1e-4]; tp,Pm=tp[k],Pm[k]; sl=Slerp(tp,Rot.from_matrix(Pm[:,:,:3]))
def pose_at(t):
    t=np.clip(t,tp[0],tp[-1]); M=np.eye(4); M[:3,:3]=sl([t]).as_matrix()[0]; M[:3,3]=[np.interp(t,tp,Pm[:,j,3]) for j in range(3)]; return M
recs=[json.loads(l) for l in open(f'{d}/frames.jsonl')]
def gray(raw):
    x=np.clip(raw.astype(np.float32)-64,0,None); g=cv2.cvtColor(np.clip(x*64,0,65535).astype(np.uint16),cv2.COLOR_BayerBG2GRAY).astype(np.float32)
    return np.clip(g/np.percentile(g,99.5)*255,0,255).astype(np.uint8)
sift=cv2.SIFT_create(5000); bf=cv2.BFMatcher()
def nrm(p,K,D): return cv2.fisheye.undistortPoints(p[:,None].astype(np.float32),K,D)[:,0]
F={}
P1=np.hstack([np.eye(3),np.zeros((3,1))]); P2=np.hstack([ls.R,ls.t[:,None]])
for r in recs:
    i=r['i']; L=gray(unpack_raw(open(f'{d}/{i:05d}_left.y10p','rb').read())); R=gray(unpack_raw(open(f'{d}/{i:05d}_right.y10p','rb').read()))
    kL,dL=sift.detectAndCompute(L,None); kR,dR=sift.detectAndCompute(R,None)
    g=[a for a,b in bf.knnMatch(dL,dR,k=2) if a.distance<0.7*b.distance] if dL is not None and dR is not None and len(kL)>2 and len(kR)>2 else []
    if len(g)<10: continue
    nL=nrm(np.float32([kL[x.queryIdx].pt for x in g]),ls.KL,ls.DL); nR=nrm(np.float32([kR[x.trainIdx].pt for x in g]),ls.KR,ls.DR)
    X=cv2.triangulatePoints(P1,P2,nL.T,nR.T); X=(X[:3]/X[3]).T; xr=X@ls.R.T+ls.t
    ok=(X[:,2]>0.3)&(X[:,2]<3.0)&(xr[:,2]>0.3)&(np.linalg.norm(xr[:,:2]/xr[:,2:3]-nR,axis=1)<1.5/1475)&(np.linalg.norm(X[:,:2]/X[:,2:3]-nL,axis=1)<1.5/1475)
    F[i]=dict(X=X[ok],D=dL[[x.queryIdx for x in g]][ok],t=r['left']['ts']-2464*LT/2-r['left']['exp_lines']*LT/2)
print("features/frame median",int(np.median([len(f['X']) for f in F.values()])))
C=[]
ids=sorted(F)
for a in ids:
    for gap in (1,2,4,8):
        b=a+gap
        if b not in F or len(F[a]['X'])<20 or len(F[b]['X'])<20: continue
        m=[x for x,y in bf.knnMatch(F[a]['D'],F[b]['D'],k=2) if x.distance<0.7*y.distance]
        if len(m)<12: continue
        Xa=F[a]['X'][[x.queryIdx for x in m]]; Xb=F[b]['X'][[x.trainIdx for x in m]]
        C.append((a,b,Xa,Xb))
import os
SPLIT=int(os.environ.get('SPLIT','-1'))
CT=[c for c in C if SPLIT>=0 and c[0]>=SPLIT]   # held-out
if SPLIT>=0: C=[c for c in C if c[1]<SPLIT]
print(len(C),"frame pairs,",sum(len(c[2]) for c in C),"3D correspondences")
X0=ls.head_from_camL
def unpack(v):
    X=np.eye(4); X[:3,:3]=Rot.from_rotvec(v[:3]).as_matrix()@X0[:3,:3]; X[:3,3]=X0[:3,3]+v[3:6]; return X,v[6]
Hc={}
def H(i,dl):
    key=(i,round(dl,5))
    if key not in Hc: Hc[key]=pose_at(F[i]['t']+dl)
    return Hc[key]
def res(v):
    X,dl=unpack(v); out=[]
    for a,b,Xa,Xb in C:
        Wa=H(a,dl)@X; Wb=H(b,dl)@X
        e=(Xa@Wa[:3,:3].T+Wa[:3,3])-(Xb@Wb[:3,:3].T+Wb[:3,3])
        sig=0.005+0.006*(Xa[:,2:3]**2)        # ~stereo noise model (m)
        out.append((e/sig).ravel())
    return np.concatenate(out)
def report(v,name):
    X,dl=unpack(v); ds=[]
    for a,b,Xa,Xb in C:
        Wa=H(a,dl)@X; Wb=H(b,dl)@X; ds.append(np.linalg.norm((Xa@Wa[:3,:3].T+Wa[:3,3])-(Xb@Wb[:3,:3].T+Wb[:3,3]),axis=1))
    ds=np.concatenate(ds); print(f"{name}: 3D residual median {np.median(ds)*100:.2f} cm, p75 {np.percentile(ds,75)*100:.2f} cm  (delta {dl*1000:+.1f} ms)")
v0=np.zeros(7); report(v0,"prior X0, delta 0")
if os.environ.get('DELTA_SCAN'):
    for dms in [int(x) for x in os.environ.get("DELTA_LIST","-60,-45,-30,-15,0").split(",")]:
        v=v0.copy(); v[6]=dms/1000; report(v,f"X0 delta {dms:+d}")
    raise SystemExit
best=None
for d0 in (-0.03,0.0,0.03):
    v=v0.copy(); v[6]=d0
    s=least_squares(res,v,loss='cauchy',f_scale=1.0,diff_step=[1e-3]*6+[1e-2],x_scale=[0.02,0.02,0.02,0.01,0.01,0.01,0.01])
    if best is None or s.cost<best.cost: best=s
report(best.x,"FITTED")
if SPLIT>=0:
    Cfit=C; C=CT; report(v0,"HELD-OUT prior"); report(best.x,"HELD-OUT fitted")
    v1=best.x.copy(); v1[3:6]=0; report(v1,"HELD-OUT fitted rot+delta only"); C=Cfit
X,dl=unpack(best.x)
print("dR (deg):",np.round(np.degrees(best.x[:3]),2)," dt (mm):",np.round(best.x[3:6]*1000,1)," delta (ms): %.1f"%(dl*1000))
json.dump({"head_from_camL":X.tolist(),"pose_delta_s":dl,"prior":X0.tolist(),"recording":d},open(f'{d}/calib_world.json','w'),indent=1)
