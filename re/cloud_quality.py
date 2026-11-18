#!/usr/bin/env python3
"""Accuracy proxies for a fused cloud (OpenVR Standing, y up):
   * floor: robust thickness of the dominant horizontal plane (MAD of y around the floor peak)
   * top-down slice image (walls / furniture sides should be thin lines)
usage: cloud_quality.py a.ply [b.ply ...] --out slices.jpg"""
import sys, numpy as np, cv2, argparse
ap = argparse.ArgumentParser(); ap.add_argument("ply", nargs="+"); ap.add_argument("--out"); ap.add_argument("--band", type=float, nargs=2, default=[0.9, 1.5])
a = ap.parse_args()

def load(p):
    raw = open(p, 'rb').read(); h = raw.index(b'end_header\n') + 11
    d = np.frombuffer(raw[h:], dtype=[('p', '<f4', 3), ('c', 'u1', 3)]); return d['p'].astype(float), d['c']

ims = []; ref = None
for p in a.ply:
    P, C = load(p)
    hist, e = np.histogram(P[:, 1], bins=np.arange(P[:, 1].min(), P[:, 1].max(), 0.01))
    y0 = e[np.argmax(hist[: len(hist) // 2])] + 0.005           # floor = densest low layer
    near = np.abs(P[:, 1] - y0) < 0.08; r = P[near, 1] - np.median(P[near, 1])
    mad = 1.4826 * np.median(np.abs(r))
    sl = (P[:, 1] > y0 + a.band[0]) & (P[:, 1] < y0 + a.band[1])
    print(f"{p}: {len(P)} pts, floor y={y0:.2f} ({near.sum()} pts, thickness sigma {mad*100:.1f} cm), slice {sl.sum()} pts")
    if ref is None:
        ref = np.percentile(P[sl][:, [0, 2]], [2, 98], axis=0)
    lo, hi = ref; res = 0.01; W, H = ((hi - lo) / res).astype(int) + 1
    im = np.zeros((H, W, 3), np.uint8)
    q = ((P[sl][:, [0, 2]] - lo) / res).astype(int); ok = (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H)
    np.add.at(im[:, :, 1], (q[ok, 1], q[ok, 0]), 60)
    im[:, :, 0] = im[:, :, 2] = im[:, :, 1]
    cv2.putText(im, p.split('/')[-1], (5, 20), 0, 0.6, (0, 200, 255), 1)
    ims.append(im)
if a.out:
    cv2.imwrite(a.out, np.hstack(ims))
