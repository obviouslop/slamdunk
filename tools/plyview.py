#!/usr/bin/env python3
"""Render top/front/side orthographic views of a binary PLY (x,y,z float [+uchar/rgb]) to PNG.
Colour = height (y). usage: plyview.py in.ply out.png [px_per_m=100]"""
import sys, zlib, struct, numpy as np
def read_ply(fn):
    raw=open(fn,'rb').read(); h_end=raw.index(b'end_header\n')+11; hdr=raw[:h_end].decode().split('\n')
    n=int([l for l in hdr if l.startswith('element vertex')][0].split()[-1])
    props=[l.split() for l in hdr if l.startswith('property')]
    dt=np.dtype([(p[2],{'float':'<f4','uchar':'u1','double':'<f8'}[p[1]]) for p in props])
    a=np.frombuffer(raw[h_end:h_end+n*dt.itemsize],dtype=dt); return a
def png(fn,img):
    h,w,_=img.shape; raw=b''.join(b'\0'+img[y].tobytes() for y in range(h))
    ch=lambda t,d: struct.pack('>I',len(d))+t+d+struct.pack('>I',zlib.crc32(t+d))
    open(fn,'wb').write(b'\x89PNG\r\n\x1a\n'+ch(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+ch(b'IDAT',zlib.compress(raw,6))+ch(b'IEND',b''))
def turbo(t):
    t=np.clip(t,0,1)[:,None]; c=np.array([[0.19,0.07,0.23],[0.12,0.57,0.98],[0.18,0.95,0.55],[0.98,0.82,0.15],[0.85,0.20,0.05]])
    x=t*(len(c)-1); i=np.minimum(x.astype(int),len(c)-2); f=x-i; return ((c[i[:,0]]*(1-f)+c[i[:,0]+1]*f)*255).astype(np.uint8)
a=read_ply(sys.argv[1]); P=np.stack([a['x'],a['y'],a['z']],1).astype(float); ppm=float(sys.argv[3]) if len(sys.argv)>3 else 100
lo,hi=np.percentile(P,0.5,0),np.percentile(P,99.5,0); m=np.all((P>=lo-0.1)&(P<=hi+0.1),1); P=P[m]
col=turbo((P[:,1]-lo[1])/(hi[1]-lo[1]+1e-9))
def view(u,v,depth,flipv=True):
    W=int((hi[u]-lo[u])*ppm)+20; H=int((hi[v]-lo[v])*ppm)+20; img=np.full((H,W,3),18,np.uint8)
    x=((P[:,u]-lo[u])*ppm+10).astype(int); y=((P[:,v]-lo[v])*ppm+10).astype(int)
    if flipv: y=H-1-y
    o=np.argsort(depth); ok=(x>=0)&(x<W)&(y>=0)&(y<H); o=o[ok[o]]
    img[y[o],x[o]]=col[o]; return img
top=view(0,2,P[:,1],flipv=False)      # looking down -y: x right, z down
front=view(0,1,-P[:,2])               # looking along -z
side=view(2,1,P[:,0])                 # looking along -x
H=max(top.shape[0],front.shape[0],side.shape[0]); pad=lambda im: np.pad(im,((0,H-im.shape[0]),(0,8),(0,0)))
png(sys.argv[2],np.concatenate([pad(top),pad(front),pad(side)],1))
print(f"{len(P)} pts; extent x {lo[0]:.2f}..{hi[0]:.2f} y {lo[1]:.2f}..{hi[1]:.2f} z {lo[2]:.2f}..{hi[2]:.2f} m; panels: top(x,z) | front(x,y) | side(z,y)")
