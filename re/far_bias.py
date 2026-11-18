#!/usr/bin/env python3
"""Far-range depth error from multi-view consistency (no external ground truth needed).

Near depth (< NEAR m) is accurate (controller check, live13: ~1.5 cm). Build a reference cloud from the near pixels
of all frames, project it into every frame (z-buffer, front-most), and compare with the depth that frame measured at
the same pixel when the surface was FAR from it. Pairs (measured disparity, reference distance) give the depth model
directly:  disp = k*fB/Z_true - d0   (linear in fB/Z_true) -> refit k, d0 from far pairs, rebuild near map, iterate.

  far_bias.py captures/live16 [--every 1]
Cache: RECDIR/raw_disp_s0.35.npz (disparity at half the rectified resolution, float32, + poses).
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from livestereo import LiveStereo
from tsdf import bin_raw10

ap = argparse.ArgumentParser(); ap.add_argument("rec"); ap.add_argument("--every", type=int, default=1)
ap.add_argument("--near", type=float, default=1.2); ap.add_argument("--k", type=float); ap.add_argument("--d0", type=float); ap.add_argument("--iters", type=int, default=1)
a = ap.parse_args()
os.environ["PC_NO_DEPTH_CAL"] = "1"
ls = LiveStereo(scale=0.35, max_depth=6.0)
fB = ls.f * ls.baseline; SUB = 2
cache = os.path.join(a.rec, "raw_disp_s0.35.npz")
if not os.path.exists(cache):
    recs = [json.loads(l) for l in open(os.path.join(a.rec, "frames.jsonl"))]
    recs = [r for r in recs if r["left"]["pose_valid"] and abs(r["left"]["ts"] - r["right"]["ts"]) < 0.015][::a.every]
    Ds, Ts, I = [], [], []
    for n, r in enumerate(recs):
        i = r["i"]
        cL = bin_raw10(open(os.path.join(a.rec, f"{i:05d}_left.y10p"), "rb").read())
        cR = bin_raw10(open(os.path.join(a.rec, f"{i:05d}_right.y10p"), "rb").read())
        disp, *_ = ls.rectified(cL, cR, half=True)
        Ds.append(disp[::SUB, ::SUB].astype(np.float32)); Ts.append(ls.world_from_rect(r["left"]["pose"])); I.append(i)
        print(f"\r{n + 1}/{len(recs)}", end="", flush=True)
    np.savez(cache, disp=np.array(Ds), T=np.array(Ts), i=np.array(I)); print()
z = np.load(cache); Ds, Ts = z["disp"], z["T"]
N, h, w = Ds.shape
f2, cx2, cy2 = ls.f / SUB, (ls.P1[0, 2] - 0.5) / SUB, (ls.P1[1, 2] - 0.5) / SUB          # half-res intrinsics
vv, uu = np.mgrid[0:h, 0:w].astype(np.float32)
rays = np.stack([(uu - cx2) / f2, (vv - cy2) / f2, np.ones_like(uu)], -1)


def depth(d, k, d0):
    return np.where(d > 0.5, k * fB / np.maximum(d + d0, 1e-3), 0)


def near_map(k, d0, near):
    P = []
    for j in range(N):
        Z = depth(Ds[j], k, d0); m = (Z > 0.3) & (Z < near)
        X = rays[m] * Z[m, None]; T = Ts[j]
        P.append(X @ T[:3, :3].T + T[:3, 3])
    P = np.concatenate(P)
    q = np.floor(P / 0.01).astype(np.int64)                       # 1 cm voxel means (denoise, thin)
    key = (q[:, 0] + (1 << 20)) * (1 << 42) + (q[:, 1] + (1 << 20)) * (1 << 21) + (q[:, 2] + (1 << 20))
    u, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    S = np.zeros((len(u), 3)); np.add.at(S, inv, P)
    keep = cnt >= 2                                                # seen at least twice
    return (S[keep] / cnt[keep, None]).astype(np.float32)


def pairs(M, k, d0, far_lo=1.4):
    """-> measured disp, reference Z, frame id for far pixels that see a near-mapped surface"""
    out_d, out_z, out_j = [], [], []
    for j in range(N):
        T = Ts[j]; Pc = (M - T[:3, 3]) @ T[:3, :3]
        ok = Pc[:, 2] > 0.3; Pc = Pc[ok]
        u = np.round(f2 * Pc[:, 0] / Pc[:, 2] + cx2).astype(int); v = np.round(f2 * Pc[:, 1] / Pc[:, 2] + cy2).astype(int)
        inb = (u >= 1) & (u < w - 1) & (v >= 1) & (v < h - 1)
        u, v, zz = u[inb], v[inb], Pc[inb, 2]
        zb = np.full((h, w), np.inf, np.float32)
        for du in (-1, 0, 1):                                       # 3x3 splat, front-most
            for dv in (-1, 0, 1):
                np.minimum.at(zb, (v + dv, u + du), zz)
        Zm = depth(Ds[j], k, d0)
        sel = np.isfinite(zb) & (zb > far_lo) & (Zm > 0) & (np.abs(Zm - zb) < 0.15 * zb)
        out_d.append(Ds[j][sel]); out_z.append(zb[sel]); out_j.append(np.full(sel.sum(), j))
    return np.concatenate(out_d), np.concatenate(out_z), np.concatenate(out_j)


def fit(d, Zr):
    """disp = k fB/Z - d0, robust (Huber IRLS) -> k, d0"""
    x = fB / Zr; A = np.c_[x, -np.ones_like(x)]; wgt = np.ones_like(x)
    for _ in range(8):
        p = np.linalg.lstsq(A * wgt[:, None], d * wgt, rcond=None)[0]
        r = d - A @ p; s = 1.4826 * np.median(np.abs(r)) + 1e-6
        wgt = np.sqrt(np.minimum(1, 1.5 * s / np.maximum(np.abs(r), 1e-9)))
    return p[0], p[1]


def report(d, Zr, k, d0, label):
    Zm = k * fB / np.maximum(d + d0, 1e-3); e = Zm - Zr
    print(f"{label}:  k={k:.4f} d0={d0:+.3f}")
    for lo, hi in ((1.4, 1.8), (1.8, 2.2), (2.2, 2.6), (2.6, 3.0), (3.0, 3.5), (3.5, 4.5), (4.5, 6)):
        m = (Zr >= lo) & (Zr < hi)
        if m.sum() > 200:
            print(f"   ref {lo:.1f}-{hi:.1f} m: n={m.sum():7d}  measured - ref: median {np.median(e[m])*100:+6.1f} cm "
                  f"({np.median(e[m] / Zr[m])*100:+5.1f} %)")


cal = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "captures", "depth_calib.json")))
k = a.k if a.k is not None else cal["depth_scale"]; d0 = a.d0 if a.d0 is not None else cal["disp_offset_px"]
print(f"{N} frames, fB {fB:.3f}, near map < {a.near} m")
for it in range(a.iters):
    M = near_map(k, d0, a.near)
    d, Zr, J = pairs(M, k, d0)
    report(d, Zr, k, d0, f"iter {it}: current model ({len(M)} map points, {len(d)} far pairs from {len(np.unique(J))} frames)")
    k_new, d0_new = fit(d, Zr)
    report(d, Zr, k_new, d0_new, f"iter {it}: refit")
    if abs(k_new - k) < 0.002 and abs(d0_new - d0) < 0.02:
        k, d0 = k_new, d0_new; break
    k, d0 = k_new, d0_new
print(f"\nRESULT k={k:.4f} d0={d0:+.3f}")
