#!/usr/bin/env python3
"""Check factory (EEPROM) stereo calibration against live frames: epipolar error of SIFT matches.
Tries per-camera axis conventions (24x24) in case the cal camera frame isn't OpenCV's."""
import sys, json, itertools, numpy as np, cv2
sys.path.insert(0, 'tools'); from arcshm import ArcShm
d=sys.argv[1]; ids=[int(x) for x in sys.argv[2].split(',')]
cal=json.load(open('captures/arcturus_calib.json')); cams={c['sourceCamera']:c for c in cal['cameras']}
def KD(c):
    i=c['intrinsics'][0]; return np.array([[i['fx'],0,i['cx']],[0,i['fy'],i['cy']],[0,0,1.]]), np.array([i['k1'],i['k2'],i['k3'],i['k4']]).reshape(4,1)
def T(c):
    e=c['extrinsics']; x=np.array(e['plus_x']); z=np.array(e['plus_z']); y=np.cross(z,x); M=np.eye(4); M[:3,:3]=np.stack([x,y,z],1); M[:3,3]=np.array(e['position'])/1000; return M
KL,DL=KD(cams['passthrough_left']); KR,DR=KD(cams['passthrough_right'])
TL,TR=T(cams['passthrough_left']),T(cams['passthrough_right'])     # cad_from_cam
def dec(fn,W=2464,H=2464,S=3088):
    b=np.fromfile(fn,dtype=np.uint8)[:S*H].reshape(H,S)[:,:W*5//4].reshape(H,W//4,5).astype(np.uint16); lo=b[:,:,4]
    r=np.stack([(b[:,:,i]<<2)|((lo>>(2*i))&3) for i in range(4)],-1).reshape(H,W).astype(np.float32)-64
    g=cv2.cvtColor(np.clip(r*64,0,65535).astype(np.uint16),cv2.COLOR_BayerBG2GRAY)
    g=g.astype(np.float32); return np.clip(g/np.percentile(g,99.5)*255,0,255).astype(np.uint8)
sift=cv2.SIFT_create(8000); P0=[];P1=[]
for i in ids:
    L=dec(f'{d}/{i:05d}_left.y10p'); R=dec(f'{d}/{i:05d}_right.y10p')
    if i==ids[0]: cv2.imwrite('/tmp/claude-1000/-home-steamos-playground/680cd019-9aa1-49e9-963b-60009e6aa3d0/scratchpad/live_LR.jpg',cv2.resize(np.hstack([L,R]),(1232,616)))
    k0,d0=sift.detectAndCompute(L,None); k1,d1=sift.detectAndCompute(R,None)
    g=[a for a,b in cv2.BFMatcher().knnMatch(d0,d1,k=2) if a.distance<0.7*b.distance]
    P0.append(np.float32([k0[x.queryIdx].pt for x in g])); P1.append(np.float32([k1[x.trainIdx].pt for x in g]))
P0=np.vstack(P0); P1=np.vstack(P1); print(len(P0),"matches")
def bear(p,K,D):
    n=cv2.fisheye.undistortPoints(p[:,None],K,D)[:,0]; b=np.c_[n,np.ones(len(n))]; return b/np.linalg.norm(b,axis=1,keepdims=True)
bL,bR=bear(P0,KL,DL),bear(P1,KR,DR)
def epi(R,t):   # X_R = R X_L + t
    t=t/np.linalg.norm(t); E=np.array([[0,-t[2],t[1]],[t[2],0,-t[0]],[-t[1],t[0],0]])@R
    n=(E@bL.T).T; n/=np.linalg.norm(n,axis=1,keepdims=True); return np.abs(np.sum(n*bR,1))*1475
Ss=[]
for perm in itertools.permutations(range(3)):
    for sg in itertools.product((1,-1),repeat=3):
        S=np.zeros((3,3))
        for k,(p,q) in enumerate(zip(perm,sg)): S[k,p]=q
        if np.linalg.det(S)>0: Ss.append(S)
M=np.linalg.inv(TR)@TL          # camR_from_camL in cal convention
res=[]
for SL in Ss:
    for SR in Ss:
        R=SR.T@M[:3,:3]@SL; t=SR.T@M[:3,3]; e=epi(R,t); res.append((np.median(e),SL,SR,R,t))
res.sort(key=lambda x:x[0])
for e,SL,SR,R,t in res[:3]:
    print(f"median epi {e:.2f}px  SL {SL.astype(int).tolist()} SR {SR.astype(int).tolist()}  t {np.round(t*1000,1)} mm  rot {np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0])):.1f}°")
e=epi(*res[0][3:]); print(f"best: <1px {np.mean(e<1)*100:.0f}%  <2px {np.mean(e<2)*100:.0f}%")
