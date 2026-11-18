#!/usr/bin/env python3
"""Solve a fixed local offset c so world = R_i (S v + c) + t_i is static. Prints residuals."""
import sys, numpy as np
from scipy.spatial import cKDTree
NV=17280; rec=np.dtype([('seq','<u4'),('t','<f8'),('m','<f4',(3,4)),('hmd','<f4',(3,4)),('xyz','<f4',(NV,3))])
fr=np.fromfile(sys.argv[1],dtype=rec)
def rotdeg(a,b):
    R=a[:,:3].T@b[:,:3]; return np.degrees(np.arccos(np.clip((np.trace(R)-1)/2,-1,1)))
pairs=[(i,j) for i in range(0,len(fr)-30,7) for j in range(i+3,min(len(fr),i+40),3) if 8<rotdeg(fr['m'][i],fr['m'][j])<20][:20]
def run(src,S,label):
    c=np.zeros(3)
    for it in range(8):
        A=[];B=[];res=[]
        for i,j in pairs:
            Ri,ti=fr[src][i][:,:3].astype(float),fr[src][i][:,3].astype(float)
            Rj,tj=fr[src][j][:,:3].astype(float),fr[src][j][:,3].astype(float)
            vi=fr['xyz'][i][::4].astype(float)@S.T; vj=fr['xyz'][j].astype(float)@S.T
            a=(vi+c)@Ri.T+ti; b=(vj+c)@Rj.T+tj
            d,k=cKDTree(b).query(a); keep=d<np.percentile(d,60)
            res.append(np.median(d))
            # residual r = (Ri-Rj)c + (Ri vi + ti - Rj vj_k - tj)
            M=Ri-Rj; r0=(vi[keep]@Ri.T+ti)-(vj[k[keep]]@Rj.T+tj)
            A.append(np.repeat(M[None],keep.sum(),0).reshape(-1,3)); B.append(-r0.reshape(-1))
        c=np.linalg.lstsq(np.vstack(A),np.concatenate(B),rcond=None)[0]
    print(f"{label:28s} c=({c[0]*100:6.2f},{c[1]*100:6.2f},{c[2]*100:6.2f}) cm  median NN {np.mean(res)*100:.2f} cm")
    return c
S0=np.eye(3); Sy=np.diag([-1.,1,-1])
run('m',S0,'trailer, no flip')
run('hmd',Sy,'hmd, yaw flip')
run('hmd',S0,'hmd, no flip')
