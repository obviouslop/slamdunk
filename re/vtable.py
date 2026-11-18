#!/usr/bin/env python3
"""Recover a C++ vtable from a stripped aarch64 .so via RTTI + R_AARCH64_RELATIVE relocs.
usage: vtable.py lib.so ClassName"""
import subprocess, sys, re, struct
lib, cls = sys.argv[1], sys.argv[2]
mangled = f"{len(cls)}{cls}".encode()
data = open(lib,'rb').read()
# section map: vaddr <-> offset
secs=[]
for l in subprocess.run(['readelf','-SW',lib],capture_output=True,text=True).stdout.splitlines():
    m=re.match(r'\s*\[\s*\d+\]\s+(\S+)\s+\S+\s+([0-9a-f]+)\s+([0-9a-f]+)\s+([0-9a-f]+)',l)
    if m: secs.append((m[1],int(m[2],16),int(m[3],16),int(m[4],16)))
def off2va(o):
    for n,va,of,sz in secs:
        if of<=o<of+sz and va: return va+o-of
rel={}
for l in subprocess.run(['readelf','-rW',lib],capture_output=True,text=True).stdout.splitlines():
    p=l.split()
    if len(p)>=4 and p[2]=='R_AARCH64_RELATIVE': rel[int(p[0],16)]=int(p[3],16)
inv={}
for k,v in rel.items(): inv.setdefault(v,[]).append(k)
pos=0; found=False
while True:
    pos=data.find(mangled+b'\0',pos)
    if pos<0: break
    if data[pos-1:pos].isdigit(): pos+=1; continue  # avoid suffix match
    nameva=off2va(pos); found=True
    for nameptr in inv.get(nameva,[]):
        ti=nameptr-8
        for tiref in inv.get(ti,[]):
            # vtable: [offset_to_top][typeinfo*][fn0][fn1]...
            vt=tiref+8; fns=[]
            while vt in rel and rel[vt]>0x1000:
                fns.append(rel[vt]); vt+=8
            if fns:
                print(f"vtable @0x{tiref+8:x} ({len(fns)} slots)")
                for i,f in enumerate(fns): print(f"  [{i}] 0x{f:x}")
    pos+=1
if not found: print("typeinfo name not found")
