#!/usr/bin/env python3
"""Publish a coloured point cloud for the Godot viewer (viewer/pc_main.gd), split into spatial tiles so
the viewer only re-uploads tiles whose content changed (no frame hitches on large clouds).

File (default /dev/shm/pointcloud.bin), written atomically (tmp + rename):
  header 32 B: magic b"PCLD0002", u32 ntiles, u32 total points, u64 seq, f32 tile size (m), u32 reserved
  tile table: ntiles x 32 B: i64 key, u64 version (crc32 of the tile data | count << 32), u64 byte offset, u32 count, u32 0
  tile data: count x 16 f32 = Godot MultiMesh buffer layout (TRANSFORM_3D + colors):
        [1,0,0,x, 0,1,0,y, 0,0,1,z, r,g,b,a]   (row-major 3x4 + RGBA in 0..1)
Coordinates: OpenVR Standing universe == OpenXR STAGE space in Godot (metres, y up, -z forward).

usage (CLI): pcpub.py cloud.ply [--max 400000]   -> publishes a static PLY
"""
import os, struct, zlib
import numpy as np

PATH = "/dev/shm/pointcloud.bin"
_seq = 0
TILE_ENTRY = struct.Struct("<qQQII")


def tile_keys(xyz, tile):
    q = np.floor(xyz / tile).astype(np.int64) + (1 << 20)
    return (q[:, 0] & 0x1FFFFF) | ((q[:, 1] & 0x1FFFFF) << 21) | ((q[:, 2] & 0x1FFFFF) << 42)


def publish(xyz, rgb, path=PATH, max_points=400000, seq=None, tile=1.0):
    """xyz: (N,3) float metres (Standing); rgb: (N,3) uint8 or float 0..1. Returns points published."""
    global _seq
    xyz = np.asarray(xyz, np.float32); rgb = np.asarray(rgb)
    if rgb.dtype == np.uint8:
        rgb = rgb.astype(np.float32) / 255
    if len(xyz) > max_points:   # thinning by a per-point position hash (unchanged tiles stay mostly identical)
        q = np.round(xyz * 1000).astype(np.int64)
        h = ((q[:, 0] * 73856093) ^ (q[:, 1] * 19349663) ^ (q[:, 2] * 83492791)) & 0xFFFFFF
        m = h < int(max_points / len(xyz) * 0x1000000)
        xyz, rgb = xyz[m], rgb[m]
    n = len(xyz)
    keys = tile_keys(xyz, tile)
    order = np.argsort(keys, kind="stable"); keys = keys[order]
    buf = np.zeros((n, 16), np.float32)
    buf[:, 0] = buf[:, 5] = buf[:, 10] = 1.0          # identity basis (size handled in shader)
    buf[:, 3], buf[:, 7], buf[:, 11] = xyz[order, 0], xyz[order, 1], xyz[order, 2]
    buf[:, 12:15] = rgb[order, :3]; buf[:, 15] = 1.0
    uk, start, cnt = np.unique(keys, return_index=True, return_counts=True)
    _seq = _seq + 1 if seq is None else seq
    data_off = 32 + len(uk) * TILE_ENTRY.size
    table = bytearray(); blobs = []
    for k, s0, c in zip(uk.tolist(), start.tolist(), cnt.tolist()):
        b = buf[s0:s0 + c].tobytes()
        table += TILE_ENTRY.pack(k - (1 << 63) if k >= (1 << 63) else k, zlib.crc32(b) | (c << 32), data_off, c, 0)
        blobs.append(b); data_off += len(b)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<8sIIQfI", b"PCLD0002", len(uk), n, _seq, tile, 0)); f.write(table)
        for b in blobs:
            f.write(b)
    os.replace(tmp, path)
    return n


def read_ply(fn):
    raw = open(fn, "rb").read(); h = raw.index(b"end_header\n") + 11
    lab = b"property uchar label" in raw[:h]
    a = np.frombuffer(raw[h:], dtype=[("p", "<f4", 3), ("c", "u1", 3)] + ([("l", "u1")] if lab else []))
    return a["p"], a["c"]


def write_ply(fn, P, C, L=None):
    """P (N,3) float, C (N,3) RGB float 0..1 or uint8, L (N,) uint8 label ids (tools/zones.py LABELS) or None."""
    os.makedirs(os.path.dirname(os.path.abspath(fn)), exist_ok=True)
    C = np.asarray(C); C = C if C.dtype == np.uint8 else np.clip(C * 255 + 0.5, 0, 255).astype(np.uint8)
    dt = [("p", "<f4", 3), ("c", "u1", 3)] + ([("l", "u1")] if L is not None else [])
    rec = np.empty(len(P), dtype=dt); rec["p"] = P; rec["c"] = C
    if L is not None:
        rec["l"] = L
    with open(fn, "wb") as fh:
        fh.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n".encode())
        if L is not None:
            fh.write(b"property uchar label\n")
        fh.write(b"end_header\n"); fh.write(rec.tobytes())


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("ply"); ap.add_argument("--max", type=int, default=400000)
    ap.add_argument("--out", default=PATH)
    a = ap.parse_args()
    p, c = read_ply(a.ply)
    n = publish(p, c, a.out, a.max)
    print(f"published {n} points -> {a.out} ({os.path.getsize(a.out)/1e6:.1f} MB)")
