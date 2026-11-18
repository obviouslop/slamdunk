#!/usr/bin/env python3
"""Publish a coloured point cloud for the Godot viewer (viewer/pointcloud.gd).

File (default /dev/shm/pointcloud.bin), written atomically (tmp + rename):
  header 32 B: magic b"PCLD0001", u32 count, u32 flags, u64 seq, f32 point_size_m, u32 reserved
  body: count x 16 f32 = Godot MultiMesh buffer layout (TRANSFORM_3D + colors):
        [sx,0,0,x, 0,sy,0,y, 0,0,sz,z, r,g,b,a]   (row-major 3x4 + RGBA in 0..1)
Coordinates: OpenVR Standing universe == OpenXR STAGE space in Godot (metres, y up, -z forward).

usage (CLI): pcpub.py cloud.ply [--max 400000] [--size 0.012]   -> publishes a static PLY
"""
import os, struct, sys
import numpy as np

PATH = "/dev/shm/pointcloud.bin"
_seq = 0


def publish(xyz, rgb, path=PATH, point_size=0.012, max_points=400000, seq=None):
    """xyz: (N,3) float metres (Standing); rgb: (N,3) uint8 or float 0..1."""
    global _seq
    xyz = np.asarray(xyz, np.float32); rgb = np.asarray(rgb)
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255
    if len(xyz) > max_points:
        idx = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[idx], rgb[idx]
    n = len(xyz)
    buf = np.zeros((n, 16), np.float32)
    buf[:, 0] = buf[:, 5] = buf[:, 10] = 1.0          # identity basis (size handled in shader)
    buf[:, 3], buf[:, 7], buf[:, 11] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    buf[:, 12:15] = rgb[:, :3]; buf[:, 15] = 1.0
    _seq = _seq + 1 if seq is None else seq
    hdr = struct.pack("<8sIIQfI", b"PCLD0001", n, 0, _seq, point_size, 0)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(hdr); f.write(buf.tobytes())
    os.replace(tmp, path)
    return n


def read_ply(fn):
    raw = open(fn, "rb").read(); h = raw.index(b"end_header\n") + 11
    a = np.frombuffer(raw[h:], dtype=[("p", "<f4", 3), ("c", "u1", 3)])
    return a["p"], a["c"]


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("ply"); ap.add_argument("--max", type=int, default=400000)
    ap.add_argument("--size", type=float, default=0.012); ap.add_argument("--out", default=PATH)
    a = ap.parse_args()
    p, c = read_ply(a.ply)
    n = publish(p, c, a.out, a.size, a.max)
    print(f"published {n} points -> {a.out} ({os.path.getsize(a.out)/1e6:.1f} MB)")
