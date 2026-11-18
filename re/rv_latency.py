#!/usr/bin/env python3
"""For each look-back offset, solve full extrinsic X (6DOF, via joint ICP over many frame pairs)
world = H(t+off) * X * v and report static-consistency. Picks best latency + X."""
import sys, numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import least_squares
NV=17280
rec=np.dtype([('seq','<u4'),('t','<f8'),('m','<f4',(3,4)),('hmd','<f4',(13,3,4)),('xyz','<f4',(NV,3))])
fr=np.fromfile(sys.argv[1],dtype=rec)
def T(m): M=np.eye(4); M[:3,:]=m; return M
def rdeg(a,b): R=a[:3,:3].T@b[:3,:3]; return np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))
pairs=[(i,j) for i in range(0,len(fr)-25,5) for j in (i+8,i+15,i+22) if 5<rdeg(fr['hmd'][i][0],fr['hmd'][j][0])<25][:40]
print(len(fr),"frames,",len(pairs),"pairs")
X0=np.diag([-1.,1,-1])
def world(k,i,x,sub=1):
    R=Rot.from_rotvec(x[:3]).as_matrix()@X0; c=x[3:]
    H=T(fr['hmd'][i][k]); v=fr['xyz'][i][::sub].astype(float)
    return (v@R.T+c)@H[:3,:3].T+H[:3,3]
def score(k,x):
    d=[]
    for i,j in pairs:
        a=world(k,i,x,4); b=world(k,j,x); dd,_=cKDTree(b).query(a); d.append(np.median(dd))
    return np.mean(d)
def fit(k):
    x=np.zeros(6)
    for it in range(4):  # alternate: fix correspondences, LS-solve x
        corr=[]
        for i,j in pairs:
            a=world(k,i,x,6); b=world(k,j,x); dd,idx=cKDTree(b).query(a); keep=dd<np.percentile(dd,60); corr.append((i,j,np.nonzero(keep)[0]*6,idx[keep]))
        def res(x):
            out=[]
            for i,j,ia,jb in corr:
                R=Rot.from_rotvec(x[:3]).as_matrix()@X0; c=x[3:]
                Hi,Hj=T(fr['hmd'][i][k]),T(fr['hmd'][j][k])
                a=(fr['xyz'][i][ia].astype(float)@R.T+c)@Hi[:3,:3].T+Hi[:3,3]
                b=(fr['xyz'][j][jb].astype(float)@R.T+c)@Hj[:3,:3].T+Hj[:3,3]
                out.append((a-b).ravel())
            return np.concatenate(out)
        x=least_squares(res,x,loss='soft_l1',f_scale=0.02).x
    return x
for k in range(0,13,2):
    x=fit(k); s=score(k,x)
    print(f"offset -{k*10:3d} ms: static median {s*100:5.2f} cm  dR={np.round(np.degrees(x[:3]),2)} deg  c={np.round(x[3:]*100,1)} cm")
