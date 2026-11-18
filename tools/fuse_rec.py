#!/usr/bin/env python3
"""Offline fusion of a live recording with the same Fuser as the live server (tools/fusion.py).
usage: fuse_rec.py RECDIR out.ply [--scale 0.35] [--voxel 0.015] [--min-hits 2] [--no-carve] [--maxdepth 2.5]
       [--delta S]  (re-pose from hmd_500hz.txt at t_mid+delta; only for recordings before the ts fix)"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from livestereo import LiveStereo, unpack_raw
from fusion import Fuser
import pcpub

ap = argparse.ArgumentParser(); ap.add_argument("rec"); ap.add_argument("out")
ap.add_argument("--scale", type=float, default=0.35); ap.add_argument("--voxel", type=float, default=0.015)
ap.add_argument("--min-hits", type=int, default=2); ap.add_argument("--no-carve", action="store_true")
ap.add_argument("--maxdepth", type=float, default=2.5); ap.add_argument("--delta", type=float, default=None)
a = ap.parse_args()
ls = LiveStereo(scale=a.scale, max_depth=a.maxdepth)
fu = Fuser(voxel=a.voxel, min_hits=a.min_hits, carve=not a.no_carve)
recs = [json.loads(l) for l in open(os.path.join(a.rec, "frames.jsonl"))]
recs = [r for r in recs if abs(r["left"]["ts"] - r["right"]["ts"]) < 0.015 and (r["left"]["pose_valid"] or a.delta is not None)]
pose_of = lambda r: np.array(r["left"]["pose"])
if a.delta is not None:
    from scipy.spatial.transform import Rotation as Rot, Slerp
    LT = 4.36e-6; Hh = np.loadtxt(os.path.join(a.rec, "hmd_500hz.txt")); Hh = Hh[Hh[:, 1] > 0]; sl = Slerp(Hh[:, 0], Rot.from_quat(Hh[:, [3, 4, 5, 2]]))
    def pose_of(r):
        m = r["left"]; t = np.clip(m["ts"] - 2464 * LT / 2 - m["exp_lines"] * LT / 2 + a.delta, Hh[0, 0], Hh[-1, 0])
        M = np.zeros((3, 4)); M[:, :3] = sl([t]).as_matrix()[0]; M[:, 3] = [np.interp(t, Hh[:, 0], Hh[:, 6 + j]) for j in range(3)]; return M
t0 = time.time()
for r in recs:
    i = r["i"]
    L = unpack_raw(open(os.path.join(a.rec, f"{i:05d}_left.y10p"), "rb").read())
    R = unpack_raw(open(os.path.join(a.rec, f"{i:05d}_right.y10p"), "rb").read())
    P, C, disp = ls.points(L, R)
    H = np.eye(4); H[:3, :] = pose_of(r); W = H @ ls.head_from_camL
    Pw = P @ W[:3, :3].T + W[:3, 3]
    st = fu.add_frame(Pw, C[:, ::-1], ls, disp, W)
print(f"{len(recs)} pairs in {time.time()-t0:.0f}s; last stats {st}")
P, C = fu.confirmed()
rec = np.empty(len(P), dtype=[("p", "<f4", 3), ("c", "u1", 3)]); rec["p"] = P; rec["c"] = np.clip(C * 255, 0, 255)
with open(a.out, "wb") as fh:
    fh.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\n"
             "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n".encode()); fh.write(rec.tobytes())
print(f"wrote {len(P)} confirmed voxels -> {a.out}")
