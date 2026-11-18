#!/usr/bin/env python3
"""Fuse per-frame stereo point clouds from an Arcturus recording into one world cloud.

Per frame f:  X_world = W0[f] @ S0 @ Rcorr @ X_cv0
  X_cv0   points from tools/stereo_depth.py (cam0, OpenCV axes, video intrinsics)
  Rcorr   small rotation video-cam0 -> calibrated cam0 (principal point re-centering by
          the recorder; fitted by re/handeye_video.py), default identity
  S0      cam0 OpenCV -> metadata cam0 axes (sensor mounted rotated 90°; re/stereo_convention.py)
  W0[f]   avcam0InWorld from the recording's metadata track

usage: fuse_recording.py meta.npz frames_dir calib.json out.ply f1,f2,... [--voxel 0.02]
       [--rcorr rx ry rz (deg)] [--maxdepth 6] [--min-frames 2]
"""
import argparse, json, os, subprocess, sys, numpy as np, cv2

ap = argparse.ArgumentParser()
ap.add_argument("meta"); ap.add_argument("frames"); ap.add_argument("calib"); ap.add_argument("out"); ap.add_argument("ids")
ap.add_argument("--voxel", type=float, default=0.02)
ap.add_argument("--rcorr", type=float, nargs=3, default=[0, 0, 0])
ap.add_argument("--maxdepth", type=float, default=6.0)
ap.add_argument("--min-frames", type=int, default=2)
ap.add_argument("--workdir", default=None)
a = ap.parse_args()
here = os.path.dirname(os.path.abspath(__file__))
m = np.load(a.meta)
S0 = np.array(json.load(open(a.calib))["S_meta_from_cv"]["cam0"]) if "S_meta_from_cv" in json.load(open(a.calib)) else np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.]])
Rcorr = cv2.Rodrigues(np.radians(a.rcorr))[0]
wd = a.workdir or os.path.join(os.path.dirname(a.out), "perframe"); os.makedirs(wd, exist_ok=True)


def read_ply(fn):
    raw = open(fn, "rb").read(); h = raw.index(b"end_header\n") + 11
    return np.frombuffer(raw[h:], dtype=[("p", "<f4", 3), ("c", "u1", 3)])


ids = [int(x) for x in a.ids.split(",")]
acc = {}
for f in ids:
    ply = os.path.join(wd, f"f{f}.ply")
    if not os.path.exists(ply):
        subprocess.run([sys.executable, "-I", os.path.join(here, "stereo_depth.py"), a.meta, a.frames, str(f), ply[:-4],
                        "--scale", "0.5", "--calib", a.calib, "--max-depth", str(a.maxdepth)], check=True, stdout=subprocess.DEVNULL)
    pts = read_ply(ply)
    P = pts["p"].astype(np.float64); C = pts["c"].astype(np.float64)
    keep = np.linalg.norm(P, axis=1) < a.maxdepth
    P, C = P[keep], C[keep]
    W = m["world_from_cam0"][f]
    Pw = (P @ (S0 @ Rcorr).T) @ W[:3, :3].T + W[:3, 3]
    key = np.floor(Pw / a.voxel).astype(np.int64)
    k = (key[:, 0] & 0x1FFFFF) | ((key[:, 1] & 0x1FFFFF) << 21) | ((key[:, 2] & 0x1FFFFF) << 42)
    uk, inv = np.unique(k, return_inverse=True)
    sums = np.zeros((len(uk), 7)); np.add.at(sums, inv, np.c_[Pw, C, np.ones(len(Pw))])
    for kk, s in zip(uk.tolist(), sums):
        e = acc.get(kk)
        if e is None: acc[kk] = [s, 1]
        else: e[0] = e[0] + s; e[1] += 1
    print(f"frame {f}: {len(P)} pts -> {len(uk)} voxels; total {len(acc)}", flush=True)

out = [(s[:3] / s[6], s[3:6] / s[6]) for s, nf in acc.values() if nf >= a.min_frames]
P = np.array([o[0] for o in out], np.float32); C = np.clip(np.array([o[1] for o in out]), 0, 255).astype(np.uint8)
with open(a.out, "wb") as fh:
    fh.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\n"
             "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n".encode())
    rec = np.empty(len(P), dtype=[("p", "<f4", 3), ("c", "u1", 3)]); rec["p"] = P; rec["c"] = C; fh.write(rec.tobytes())
print(f"wrote {len(P)} voxels (seen in >= {a.min_frames} frames) -> {a.out}")
