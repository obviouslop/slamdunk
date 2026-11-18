#!/usr/bin/env python3
"""Hand-eye check/fit for live recordings: X = head_from_camL.
Camera relative motion from images (stereo 3D in frame i -> PnP in frame j, metric), HMD relative
motion from recorded OpenVR poses. Prior X0 = diag(-1,1,-1) @ cad_from_camL (EEPROM + device_config)."""
import sys, os, json, numpy as np, cv2
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot
d=sys.argv[1]
cal=json.load(open('captures/arcturus_calib.json')); cams={c['sourceCamera']:c for c in cal['cameras']}
def KD(c):
    i=c['intrinsics'][0]; return np.array([[i['fx'],0,i['cx']],[0,i['fy'],i['cy']],[0,0,1.]]), np.array([i['k1'],i['k2'],i['k3'],i['k4']]).reshape(4,1)
def T(c):
    e=c['extrinsics']; x=np.array(e['plus_x']); z=np.array(e['plus_z']); y=np.cross(z,x); M=np.eye(4); M[:3,:3]=np.stack([x,y,z],1); M[:3,3]=np.array(e['position'])/1000; return M
KL,DL=KD(cams['passthrough_left']); KR,DR=KD(cams['passthrough_right'])
TL,TR=T(cams['passthrough_left']),T(cams['passthrough_right'])
M=np.linalg.inv(TR)@TL; R_RL,t_RL=M[:3,:3],M[:3,3]
X0=np.diag([-1.,1,-1,1])@TL
recs=[json.loads(l) for l in open(f'{d}/frames.jsonl')]
def H(r,cam='left'): P=np.eye(4); P[:3,:]=np.array(r[cam]['pose']); return P
def dec(fn,W=2464,H_=2464,S=3088):
    b=np.fromfile(fn,dtype=np.uint8)[:S*H_].reshape(H_,S)[:,:W*5//4].reshape(H_,W//4,5).astype(np.uint16); lo=b[:,:,4]
    r=np.stack([(b[:,:,i]<<2)|((lo>>(2*i))&3) for i in range(4)],-1).reshape(H_,W).astype(np.float32)-64
    g=cv2.cvtColor(np.clip(r*64,0,65535).astype(np.uint16),cv2.COLOR_BayerBG2GRAY).astype(np.float32)
    return np.clip(g/np.percentile(g,99.5)*255,0,255).astype(np.uint8)
sift=cv2.SIFT_create(6000); bf=cv2.BFMatcher()
def norm(p,K,D): return cv2.fisheye.undistortPoints(p[:,None].astype(np.float32),K,D)[:,0]
cache={}
def feats(i):
    if i not in cache:
        L=dec(f'{d}/{i:05d}_left.y10p'); R=dec(f'{d}/{i:05d}_right.y10p')
        kL,dL=sift.detectAndCompute(L,None); kR,dR=sift.detectAndCompute(R,None)
        g=[a for a,b in bf.knnMatch(dL,dR,k=2) if a.distance<0.7*b.distance]
        pL=np.float32([kL[x.queryIdx].pt for x in g]); pR=np.float32([kR[x.trainIdx].pt for x in g])
        nL,nR=norm(pL,KL,DL),norm(pR,KR,DR)
        P1=np.hstack([np.eye(3),np.zeros((3,1))]); P2=np.hstack([R_RL,t_RL[:,None]])
        X=cv2.triangulatePoints(P1,P2,nL.T,nR.T); X=(X[:3]/X[3]).T
        xr=(X@R_RL.T+t_RL); ok=(X[:,2]>0.2)&(X[:,2]<6)&(xr[:,2]>0.2)
        # reprojection check in R
        rp=xr[:,:2]/xr[:,2:3]; ok&=np.linalg.norm(rp-nR,axis=1)<2/1475
        cache[i]=dict(kL=kL,dL=dL,idx=np.array([g[k].queryIdx for k in range(len(g))])[ok],X=X[ok])
    return cache[i]
good=[r['i'] for r in recs if abs(r['left']['ts']-r['right']['ts'])<0.005 and r['left']['pose_valid']]
print(len(good),"usable pairs of",len(recs))
B=[];A=[];info=[]
for a_,b_ in zip(good[:-1],good[1:]):
    if b_-a_>int(os.environ.get("MAXGAP","3")): continue
    fa,fb=feats(a_),feats(b_)
    if len(fa['X'])<30: continue
    m=[x for x,y in bf.knnMatch(fa['dL'][fa['idx']],fb['dL'],k=2) if x.distance<0.75*y.distance]
    if len(m)<25: continue
    X3=fa['X'][[x.queryIdx for x in m]]; p2=norm(np.float32([fb['kL'][x.trainIdx].pt for x in m]),KL,DL)
    ok,rv,tv,inl=cv2.solvePnPRansac(X3,p2,np.eye(3),None,reprojectionError=2/1475,iterationsCount=500,flags=cv2.SOLVEPNP_EPNP)
    if not ok or inl is None or len(inl)<20: continue
    rv,tv=cv2.solvePnPRefineLM(X3[inl[:,0]],p2[inl[:,0]],np.eye(3),None,rv,tv)
    Bm=np.eye(4); Bm[:3,:3]=cv2.Rodrigues(rv)[0]; Bm[:3,3]=tv.ravel()      # camB_from_camA
    Am=np.linalg.inv(H(recs[b_]))@H(recs[a_])                                  # headB_from_headA
    B.append(Bm); A.append(Am); info.append((a_,b_,len(inl)))
print(len(B),"relative motions; HMD rot per pair p50 %.1f°, trans p50 %.1f cm"%(np.median([np.degrees(np.linalg.norm(cv2.Rodrigues(x[:3,:3])[0])) for x in A]),np.median([np.linalg.norm(x[:3,3])*100 for x in A])))
def resid(X):
    Xi=np.linalg.inv(X); out=[]
    for Am,Bm in zip(A,B):
        P=Xi@Am@X; E=np.linalg.inv(Bm)@P
        out.append([np.degrees(np.linalg.norm(cv2.Rodrigues(E[:3,:3])[0])),np.linalg.norm(E[:3,3])*100])
    return np.array(out)
r0=resid(X0); print(f"PRIOR X0 (EEPROM+device_config): rot err median {np.median(r0[:,0]):.2f}°, trans err median {np.median(r0[:,1]):.2f} cm")
def pack(X): return np.r_[Rot.from_matrix(X[:3,:3]).as_rotvec(),X[:3,3]]
def unpack(v): X=np.eye(4); X[:3,:3]=Rot.from_rotvec(v[:3]).as_matrix(); X[:3,3]=v[3:]; return X
def f(v):
    X=unpack(v); Xi=np.linalg.inv(X); out=[]
    for Am,Bm in zip(A,B):
        E=np.linalg.inv(Bm)@Xi@Am@X; out+=list(Rot.from_matrix(E[:3,:3]).as_rotvec()*1.0)+list(E[:3,3]*5)   # 1 rad ~ 20 cm weight
    return np.array(out)
sol=least_squares(f,pack(X0),loss='soft_l1',f_scale=0.02); Xf=unpack(sol.x); rf=resid(Xf)
print(f"FITTED X: rot err median {np.median(rf[:,0]):.2f}°, trans err median {np.median(rf[:,1]):.2f} cm")
dX=np.linalg.inv(X0)@Xf
print(f"fitted vs prior: rot {np.degrees(np.linalg.norm(cv2.Rodrigues(dX[:3,:3])[0])):.2f}°, trans {np.round(dX[:3,3]*1000,1)} mm")
print("X0 t (mm):",np.round(X0[:3,3]*1000,1)," Xf t (mm):",np.round(Xf[:3,3]*1000,1))
json.dump({"head_from_camL":Xf.tolist(),"prior":X0.tolist(),"pairs":len(A)},open(f'{d}/handeye.json','w'),indent=1)

# ---- translation scale: image-derived camera motion vs HMD-derived camera motion (prior X0)
Xi=np.linalg.inv(X0); ratios=[]; rows=[]
for Am,Bm in zip(A,B):
    P=Xi@Am@X0                      # predicted camB_from_camA from HMD poses
    tb=np.linalg.norm(Bm[:3,3]); tp=np.linalg.norm(P[:3,3])
    if tp>0.05 and tb>0.05: ratios.append(tb/tp); rows.append((tp,tb))
r=np.array(ratios)
print(f"translation scale image/HMD over {len(r)} pairs (>5 cm): median {np.median(r):.3f}, p25 {np.percentile(r,25):.3f}, p75 {np.percentile(r,75):.3f}")
