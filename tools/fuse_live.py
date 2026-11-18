#!/usr/bin/env python3
"""Fuse a live recording (tools/liverec.py) into a coloured world point cloud (OpenVR Standing).
usage: fuse_live.py RECDIR out.ply [--step 2] [--voxel 0.02] [--min-frames 2] [--maxdepth 3]"""
import argparse, json, os, sys, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from livestereo import LiveStereo, unpack_raw
ap = argparse.ArgumentParser(); ap.add_argument("rec"); ap.add_argument("out")
ap.add_argument("--step", type=int, default=2); ap.add_argument("--voxel", type=float, default=0.02)
ap.add_argument("--min-frames", type=int, default=2); ap.add_argument("--maxdepth", type=float, default=3.0)
ap.add_argument("--delta", type=float, default=None, help="re-pose frames at t_mid+delta from poses.jsonl (s)")
a = ap.parse_args()
ls = LiveStereo(scale=0.5, max_depth=a.maxdepth)
recs = [json.loads(l) for l in open(os.path.join(a.rec, "frames.jsonl"))]
recs = [r for r in recs if abs(r["left"]["ts"] - r["right"]["ts"]) < 0.005 and (r["left"]["pose_valid"] or a.delta is not None)][::a.step]
acc = {}; t0 = time.time()
pose_of = lambda r: r["left"]["pose"]
if a.delta is not None:   # interpolate the full-rate pose log (entries stamped at their own t_mid)
    from scipy.spatial.transform import Rotation as Rot, Slerp
    LT = 4.36e-6; tmid = lambda m: m["ts"] - 2464 * LT / 2 - m["exp_lines"] * LT / 2
    hist = os.path.join(a.rec, "hmd_500hz.txt")
    if os.path.exists(hist):   # true 500 Hz pose history (arcgrab --pose-log): t valid qw qx qy qz px py pz
        Hh = np.loadtxt(hist); Hh = Hh[Hh[:, 1] > 0]; tp = Hh[:, 0]
        Pm = np.concatenate([Rot.from_quat(Hh[:, [3, 4, 5, 2]]).as_matrix(), Hh[:, 6:9, None]], 2)
        print("pose source: 500 Hz history")
    else:
        pl = sorted([json.loads(l) for l in open(os.path.join(a.rec, "poses.jsonl"))], key=tmid)
        tp = np.array([tmid(m) for m in pl]); Pm = np.array([m["pose"] for m in pl])
    keep = np.r_[True, np.diff(tp) > 1e-4]
    tp, Pm = tp[keep], Pm[keep]; sl = Slerp(tp, Rot.from_matrix(Pm[:, :, :3]))
    def pose_of(r):
        t = np.clip(tmid(r["left"]) + a.delta, tp[0], tp[-1]); M = np.zeros((3, 4))
        M[:, :3] = sl([t]).as_matrix()[0]; M[:, 3] = [np.interp(t, tp, Pm[:, j, 3]) for j in range(3)]; return M
for r in recs:
    i = r["i"]
    L = unpack_raw(open(os.path.join(a.rec, f"{i:05d}_left.y10p"), "rb").read())
    R = unpack_raw(open(os.path.join(a.rec, f"{i:05d}_right.y10p"), "rb").read())
    P, C, _ = ls.points(L, R)
    Pw = ls.to_world(P, pose_of(r))
    key = np.floor(Pw / a.voxel).astype(np.int64)
    k = (key[:, 0] & 0x1FFFFF) | ((key[:, 1] & 0x1FFFFF) << 21) | ((key[:, 2] & 0x1FFFFF) << 42)
    uk, inv = np.unique(k, return_inverse=True)
    sums = np.zeros((len(uk), 7)); np.add.at(sums, inv, np.c_[Pw, C[:, ::-1] * 255, np.ones(len(Pw))])
    for kk, sv in zip(uk.tolist(), sums):
        e = acc.get(kk)
        if e is None: acc[kk] = [sv, 1]
        else: e[0] = e[0] + sv; e[1] += 1
    print(f"pair {i}: {len(P)} pts, total voxels {len(acc)}", flush=True)
out = [(s[:3] / s[6], s[3:6] / s[6]) for s, n in acc.values() if n >= a.min_frames]
P = np.array([o[0] for o in out], np.float32); C = np.clip(np.array([o[1] for o in out]), 0, 255).astype(np.uint8)
with open(a.out, "wb") as fh:
    fh.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\n"
             "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n".encode())
    rec = np.empty(len(P), dtype=[("p", "<f4", 3), ("c", "u1", 3)]); rec["p"] = P; rec["c"] = C; fh.write(rec.tobytes())
print(f"{len(recs)} pairs in {time.time()-t0:.0f}s -> {len(P)} voxels (>= {a.min_frames} frames) -> {a.out}")
