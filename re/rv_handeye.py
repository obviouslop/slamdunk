#!/usr/bin/env python3
"""ICP between raw roomview clouds -> true relative motion; compare rotation angles with
trailer/HMD relative poses (angle is invariant to a fixed extrinsic), then solve the
extrinsic rotation X (hand-eye) for the source that matches."""
import sys, numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
NV=17280; rec=np.dtype([('seq','<u4'),('t','<f8'),('m','<f4',(3,4)),('hmd','<f4',(3,4)),('xyz','<f4',(NV,3))])
fr=np.fromfile(sys.argv[1],dtype=rec)
def T(m): M=np.eye(4); M[:3,:]=m; return M
def kabsch(a,b):
    ca,cb=a.mean(0),b.mean(0); U,_,Vt=np.linalg.svd((a-ca).T@(b-cb)); D=np.diag([1,1,np.sign(np.linalg.det(Vt.T@U.T))])
    R=Vt.T@D@U.T; return R, cb-R@ca
def icp(a,b,R=np.eye(3),t=np.zeros(3)):
    tree=cKDTree(b)
    for _ in range(40):
        a2=a@R.T+t; d,k=tree.query(a2); keep=d<max(np.percentile(d,70),0.005)
        R,t=kabsch(a[keep],b[k[keep]])
    d,_=tree.query(a@R.T+t); return R,t,np.median(d)
ang=lambda R: np.degrees(Rot.from_matrix(R).magnitude())
rows=[]
for i in range(0,len(fr)-6,2):
    j=i+3
    a=fr['xyz'][i][::3].astype(float); b=fr['xyz'][j].astype(float)
    R,t,res=icp(a,b)                       # maps frame-i local -> frame-j local
    Ptr=np.linalg.inv(T(fr['m'][j]))@T(fr['m'][i]); Phm=np.linalg.inv(T(fr['hmd'][j]))@T(fr['hmd'][i])
    if not (1.5<ang(Ptr[:3,:3])<8) or res>0.02: continue
    rows.append((i,ang(R),ang(Ptr[:3,:3]),ang(Phm[:3,:3]),res*100,R,Ptr,Phm))
print(" i    icp°  trailer°  hmd°   icp-res(cm)")
for r in rows[:14]: print(f"{r[0]:3d} {r[1]:7.2f} {r[2]:8.2f} {r[3]:6.2f}  {r[4]:6.2f}")
A=np.array([[r[1],r[2],r[3]] for r in rows])
print(f"angle mismatch vs ICP: trailer {np.abs(A[:,0]-A[:,1]).mean():.2f}°, hmd {np.abs(A[:,0]-A[:,2]).mean():.2f}°  (n={len(rows)})")
# hand-eye rotation: icp axes a_k = X^T * pose axes p_k  (ICP local motion = X^-1 Prel X)
for name,idx in (('trailer',6),('hmd',7)):
    P=np.array([Rot.from_matrix(r[idx][:3,:3]).as_rotvec() for r in rows]); Q=np.array([Rot.from_matrix(r[5]).as_rotvec() for r in rows])
    H=Q.T@P; U,_,Vt=np.linalg.svd(H); D=np.diag([1,1,np.sign(np.linalg.det(Vt.T@U.T))]); X=Vt.T@D@U.T  # P ≈ X Q
    err=np.degrees(np.linalg.norm(P-Q@X.T,axis=1)).mean()
    print(f"hand-eye {name}: X=\n{np.round(X,3)}\n  axis residual {err:.2f}°  euler(xyz) {np.round(Rot.from_matrix(X).as_euler('xyz',degrees=True),1)}")
