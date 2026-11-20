#!/usr/bin/env python3
"""Perspective render of a coloured PLY from an orbit viewpoint (z-buffered splats).
usage: plyrender.py in.ply out.png [--yaw 30] [--pitch 20] [--dist 6] [--maxr 12] [--size 900]
Camera looks at the cloud's median point; input frame is OpenCV (x right, y down, z fwd)."""
import argparse, numpy as np, cv2
ap=argparse.ArgumentParser(); ap.add_argument("ply"); ap.add_argument("out")
for k,v in (("yaw",30.),("pitch",20.),("dist",6.),("maxr",12.),("size",900),("fov",70.)): ap.add_argument("--"+k,type=type(v),default=v)
ap.add_argument("--target",type=float,nargs=3)
ap.add_argument("--yup",action="store_true",help="input is y-up (OpenVR world): flip to y-down")
a=ap.parse_args()
raw=open(a.ply,'rb').read(); h=raw.index(b'end_header\n')+11
hdr=raw[:h].decode(); has_col='red' in hdr
dt=[('p','<f4',3)]+([('c','u1',3)] if has_col else [('h','u1')])+([('l','u1')] if 'label' in hdr else [])+([('t','<f8')] if 'double time' in hdr else [])+([('d','<f4')] if 'float distance' in hdr else [])
pts=np.frombuffer(raw[h:],dtype=dt); P=pts['p'].astype(float)
C=pts['c'].astype(np.uint8) if has_col else np.full((len(P),3),200,np.uint8)
if a.yup: P=P*np.array([1,-1,-1.])
m=np.linalg.norm(P-np.median(P,0),axis=1)<a.maxr; P,C=P[m],C[m]
T=np.array(a.target) if a.target else np.median(P,0)
yaw,pitch=np.radians(a.yaw),np.radians(a.pitch)
# orbit camera around target (in OpenCV world: y down)
fwd=np.array([np.sin(yaw)*np.cos(pitch), np.sin(pitch), np.cos(yaw)*np.cos(pitch)])
eye=T-fwd*a.dist; z=fwd/np.linalg.norm(fwd); x=np.cross([0,1,0],z); x/=np.linalg.norm(x); y=np.cross(z,x)
Rc=np.stack([x,y,z]); Q=(P-eye)@Rc.T
ok=Q[:,2]>0.1; Q,C=Q[ok],C[ok]
f=a.size/2/np.tan(np.radians(a.fov)/2); u=(Q[:,0]/Q[:,2]*f+a.size/2).astype(int); v=(Q[:,1]/Q[:,2]*f+a.size/2).astype(int)
ok=(u>=1)&(u<a.size-1)&(v>=1)&(v<a.size-1); u,v,zz,C=u[ok],v[ok],Q[ok,2],C[ok]
img=np.full((a.size,a.size,3),20,np.uint8); zb=np.full((a.size,a.size),np.inf)
o=np.argsort(-zz)
for du,dv in ((0,0),(1,0),(0,1),(1,1)):
    img[v[o]+dv,u[o]+du]=C[o]
cv2.imwrite(a.out,img[:,:,::-1] if has_col else img); print(f"{len(u)} pts rendered, target {np.round(T,2)}")
