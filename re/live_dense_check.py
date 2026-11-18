import sys, json, os, numpy as np, cv2
sys.path.insert(0,'tools'); from livestereo import LiveStereo, unpack_raw
d=sys.argv[1]; i=int(sys.argv[2]); S=sys.argv[3]
import os
ls=LiveStereo(scale=float(os.environ.get("SCALE","0.5")), max_depth=4, edge_rel=float(os.environ.get("EDGE","0.04")))
L=unpack_raw(open(f'{d}/{i:05d}_left.y10p','rb').read()); R=unpack_raw(open(f'{d}/{i:05d}_right.y10p','rb').read())
P,C,disp=ls.points(L,R)
print("left_first",ls.left_first,"f",round(ls.f,1),"B",round(ls.baseline*1000,1),"mm; valid %.1f%%"%((disp>0.5).mean()*100),"; depth p5/50/95",np.percentile(P[:,2],[5,50,95]).round(2))
# sparse reference: SIFT L/R + triangulation (same as handeye)
def gray(raw):
    x=np.clip(raw.astype(np.float32)-64,0,None); g=cv2.cvtColor(np.clip(x*64,0,65535).astype(np.uint16),cv2.COLOR_BayerBG2GRAY).astype(np.float32)
    return np.clip(g/np.percentile(g,99.5)*255,0,255).astype(np.uint8)
gL,gR=gray(L),gray(R); s=cv2.SIFT_create(4000); kL,dL=s.detectAndCompute(gL,None); kR,dR=s.detectAndCompute(gR,None)
m=[a for a,b in cv2.BFMatcher().knnMatch(dL,dR,k=2) if a.distance<0.7*b.distance]
pL=np.float32([kL[x.queryIdx].pt for x in m]); pR=np.float32([kR[x.trainIdx].pt for x in m])
nL=cv2.fisheye.undistortPoints(pL[:,None],ls.KL,ls.DL)[:,0]; nR=cv2.fisheye.undistortPoints(pR[:,None],ls.KR,ls.DR)[:,0]
X=cv2.triangulatePoints(np.hstack([np.eye(3),np.zeros((3,1))]),np.hstack([ls.R,ls.t[:,None]]),nL.T,nR.T); X=(X[:3]/X[3]).T
ok=(X[:,2]>0.25)&(X[:,2]<4); X=X[ok]
from scipy.spatial import cKDTree
dd,_=cKDTree(P).query(X); print(f"sparse pts {len(X)} depth p50 {np.median(X[:,2]):.2f}; NN dist sparse->dense median {np.median(dd)*100:.1f} cm, p90 {np.percentile(dd,90)*100:.1f} cm")
# visuals
s_=255/max(np.percentile(cv2.cvtColor(ls._rgb(L),cv2.COLOR_BGR2GRAY),99.5),1)
rL=cv2.remap(gL,*ls.mapL,cv2.INTER_LINEAR); rR=cv2.remap(gR,*ls.mapR,cv2.INTER_LINEAR)
dv=np.clip(disp/ (ls.f*ls.baseline/0.3)*255,0,255).astype(np.uint8); dv=cv2.applyColorMap(dv,cv2.COLORMAP_TURBO); dv[disp<=0.5]=0
cv2.imwrite(f'{S}/dense_check.jpg',cv2.resize(np.hstack([cv2.cvtColor(rL,cv2.COLOR_GRAY2BGR),cv2.cvtColor(rR,cv2.COLOR_GRAY2BGR),dv]),(1500,500)))
