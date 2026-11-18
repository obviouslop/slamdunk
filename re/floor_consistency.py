#!/usr/bin/env python3
"""Depth-model check from the floor seen at many distances (single frames, no fusion, no drift between frames that matters
much: SteamVR's floor is y = 0 and tracking height drift is small over a recording).
Per frame: RANSAC floor plane in disparity space (a plane is linear in (u, v, disp)), then for a depth model
Z = k fB/(disp + d0) the plane height y (STAGE) and its median distance. A correct model gives the same y at every
distance (ideally 0). Fits (d0, k) minimising the spread of y over frames + |mean y|.
  floor_consistency.py captures/live12/rays_e1_s0.35.npz"""
import sys, numpy as np
z = np.load(sys.argv[1]); O, D, disp, fB, F, UV = z["O"], z["D"], z["disp"], float(z["fB"]), z["frame"], z["uv"].astype(float)
rng = np.random.default_rng(0)


def ransac(uv, d, iters=300, tol=0.25):
    A = np.c_[uv, np.ones(len(d))]; best = None
    for _ in range(iters):
        i = rng.choice(len(d), 3, replace=False)
        try: p = np.linalg.solve(A[i], d[i])
        except np.linalg.LinAlgError: continue
        inl = np.abs(A @ p - d) < tol
        if best is None or inl.sum() > best.sum(): best = inl
    return best


fl = {}
for f in range(F.max() + 1):
    s = np.flatnonzero(F == f)
    if len(s) < 1000: continue
    Dn = D[s] / np.linalg.norm(D[s], axis=1, keepdims=True); P = O[s] + D[s] * (fB / disp[s])[:, None]; eye = O[s][0, 1]
    cand = s[(Dn[:, 1] < -0.3) & (P[:, 1] < eye - 0.9) & (disp[s] > 2)]
    if len(cand) < 1500: continue
    idx = cand[ransac(UV[cand], disp[cand])]
    if len(idx) >= 1500: fl[f] = idx
print(f"floor frames: {len(fl)}")


def planes(d0, k):
    rows = []
    for f, idx in fl.items():
        P = O[idx] + D[idx] * (k * fB / np.maximum(disp[idx] + d0, 0.3))[:, None]
        c = P.mean(0); _, _, vt = np.linalg.svd(P - c, full_matrices=False); n = vt[2] * np.sign(vt[2][1])
        tilt = np.degrees(np.arccos(min(1, abs(n[1]))))
        Zm = np.median(k * fB / (disp[idx] + d0))
        rows.append((f, c[1], tilt, Zm, O[idx[0], 1]))
    return np.array(rows)


def score(r):
    ok = r[:, 2] < 8                        # near-horizontal planes only
    y = r[ok, 1]
    return np.std(y) + 0.3 * abs(np.median(y)), ok


for d0, k in [(0, 1), (0.785, 1.024), (0.999, 1.0313)]:
    r = planes(d0, k); sc, ok = score(r)
    print(f"\nd0={d0:+.3f} k={k:.4f}: floor y median {np.median(r[ok,1]):+.3f}  std {np.std(r[ok,1]):.3f}  (frames {ok.sum()})")
    for lo, hi in ((0.8, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 5.0)):
        m = ok & (r[:, 3] >= lo) & (r[:, 3] < hi)
        if m.sum(): print(f"   floor at Z {lo:.1f}-{hi:.1f} m: n={m.sum():3d}  y median {np.median(r[m,1]):+.3f}  (eye {np.median(r[m,4]):.2f})")

best = None
for d0 in np.arange(-0.5, 2.51, 0.1):
    for k in np.arange(0.94, 1.081, 0.01):
        sc, _ = score(planes(d0, k))
        if best is None or sc < best[0]: best = (sc, d0, k)
sc, d0, k = best
r = planes(d0, k); _, ok = score(r)
print(f"\nbest: d0={d0:+.2f} k={k:.3f}  floor y median {np.median(r[ok,1]):+.3f} std {np.std(r[ok,1]):.3f}")
for lo, hi in ((0.8, 1.5), (1.5, 2.0), (2.0, 2.5), (2.5, 3.0), (3.0, 5.0)):
    m = ok & (r[:, 3] >= lo) & (r[:, 3] < hi)
    if m.sum(): print(f"   floor at Z {lo:.1f}-{hi:.1f} m: n={m.sum():3d}  y median {np.median(r[m,1]):+.3f}")
# slope-only fit: which d0 makes y independent of distance (k scales y about the eye, not the slope much)
for d0t in (0, 0.5, 0.785, 1.0, 1.5, 2.0):
    r = planes(d0t, 1.0); _, ok = score(r)
    A = np.c_[r[ok, 3], np.ones(ok.sum())]; sl = np.linalg.lstsq(A, r[ok, 1], rcond=None)[0][0]
    print(f"d0={d0t:.2f} k=1: dy/dZ = {sl*100:+.1f} cm per m distance")
