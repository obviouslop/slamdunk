#!/usr/bin/env python3
"""Fit depth correction Z = k*fB/(disp+d0) on cached rays (re/depth_bias.py).
Criterion (scale-free): the same patch of floor/ceiling (0.3 m xz cell) must have the same height whether it
was observed from near (Z<1.6 m) or far (Z>2.0 m). Reports median(far - near) per surface for a d0 grid;
the correct d0 is the zero crossing. Also prints the floor-ceiling distance for the user's 2.50 m check."""
import sys, numpy as np
z = np.load(sys.argv[1]); O, D, disp, fB = z["O"], z["D"], z["disp"], float(z["fB"])
Dn = D / np.linalg.norm(D, axis=1, keepdims=True)
CELL = 0.3

def mode(y, lo, hi, step=0.01):
    h, e = np.histogram(y, bins=np.arange(lo, hi, step)); c = np.convolve(h, np.ones(5), "same"); return e[np.argmax(c)] + step / 2

def near_far(P, Z, sel, ylo, yhi):
    """per cell: robust height of near and far observations of the surface (points within +-6 cm of cell mode)"""
    s = sel & (P[:, 1] > ylo) & (P[:, 1] < yhi)
    key = np.floor(P[s, 0] / CELL).astype(np.int64) * 100000 + np.floor(P[s, 2] / CELL).astype(np.int64)
    y, Zs = P[s, 1], Z[s]
    order = np.argsort(key); key, y, Zs = key[order], y[order], Zs[order]
    cuts = np.flatnonzero(np.diff(key)) + 1
    diffs, w = [], []
    for yy, zz in zip(np.split(y, cuts), np.split(Zs, cuts)):
        if len(yy) < 200: continue
        m = mode(yy, ylo, yhi); keep = np.abs(yy - m) < 0.08
        n, f = keep & (zz < 1.6), keep & (zz > 2.0)
        if n.sum() > 30 and f.sum() > 30:
            diffs.append(np.median(yy[f]) - np.median(yy[n])); w.append(min(n.sum(), f.sum()))
    return np.array(diffs), np.array(w)

down = Dn[:, 1] < -0.4; up = Dn[:, 1] > 0.4
print("  d0     k    floor far-near (cells)  ceil far-near (cells)   floor_y  ceil_y  ceil-floor")
for k, d0 in [(1.0, x) for x in (-0.5, 0, 0.25, 0.5, 0.75, 1.0, 1.5)] + [(x, 0.0) for x in (0.9, 0.95)]:
    Z = k * fB / np.maximum(disp + d0, 0.3); ok = (Z < 3.5) & (Z > 0.3)
    P = O + D * Z[:, None]
    df, wf = near_far(P, Z, down & ok, 0.6, 1.6); dc, wc = near_far(P, Z, up & ok, 2.9, 4.3)
    yf = mode(P[down & ok, 1], 0.6, 1.6); yc = mode(P[up & ok, 1], 2.9, 4.3)
    wm = lambda d, w: np.sum(d * w) / max(w.sum(), 1)
    print(f"{d0:5.2f}  {k:.2f}   {wm(df,wf):+.3f} ({len(df):3d})            {wm(dc,wc):+.3f} ({len(dc):3d})"
          f"            {yf:.3f}   {yc:.3f}   {yc-yf:.3f}")
