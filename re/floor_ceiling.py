#!/usr/bin/env python3
"""Floor-to-ceiling distance from short time windows (drift-free), vs depth model Z = k fB/(disp+d0).
Per frame: RANSAC plane (disparity space) among rays looking down (floor, >1 m below eyes) or up
(ceiling, >0.3 m above eyes); keep near-horizontal planes. Pair each ceiling frame with floor frames
within +-W frames (~1 s apart in the recording), use the eye height to remove tracking drift:
  H = (ceil_y - eye_y) + (eye_y - floor_y) measured relative to the same head pose stream."""
import sys, json, numpy as np
z = np.load(sys.argv[1]); O, D, disp, fB, F, UV = z["O"], z["D"], z["disp"], float(z["fB"]), z["frame"], z["uv"].astype(float)
W = int(sys.argv[2]) if len(sys.argv) > 2 else 5
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
fl, ce = {}, {}
for f in range(F.max() + 1):
    s = np.flatnonzero(F == f)
    if len(s) < 1000: continue
    Dn = D[s] / np.linalg.norm(D[s], axis=1, keepdims=True); P = O[s] + D[s] * (fB / disp[s])[:, None]; eye = O[s][0, 1]
    for store, cand, mn in ((fl, s[(Dn[:, 1] < -0.35) & (P[:, 1] < eye - 1.0)], 2000), (ce, s[(Dn[:, 1] > 0.35) & (P[:, 1] > eye + 0.3)], 800)):
        if len(cand) < mn: continue
        idx = cand[ransac(UV[cand], disp[cand])]
        if len(idx) >= mn: store[f] = idx
def plane_y(idx, d0, k):
    P = O[idx] + D[idx] * (k * fB / np.maximum(disp[idx] + d0, 0.3))[:, None]
    c = P.mean(0); _, _, vt = np.linalg.svd(P - c, full_matrices=False); n = vt[2] * np.sign(vt[2][1])
    tilt = np.degrees(np.arccos(abs(n[1])))
    Z = np.median(k * fB / (disp[idx] + d0))
    return c[1] - O[idx[0], 1], tilt, Z      # height relative to the eye
print(f"floor frames {len(fl)}, ceiling frames {len(ce)}")
for d0, k in [(0, 1), (0.5, 1), (1.0, 1), (0, 0.96), (0, 0.95)]:
    Hs = []; rows = []
    for fc, ic in ce.items():
        yc, tc, Zc = plane_y(ic, d0, k)
        if tc > 10: continue
        near = [plane_y(fl[ff], d0, k) for ff in fl if abs(ff - fc) <= W]
        near = [x for x in near if x[1] < 10]
        if not near: continue
        yf = np.median([x[0] for x in near])
        Hs.append(yc - yf); rows.append((fc, yc, -yf, Zc, np.median([x[2] for x in near]), len(near)))
    Hs = np.array(Hs)
    print(f"d0={d0:.2f} k={k:.2f}: n={len(Hs)} floor-ceiling median {np.median(Hs):.3f} m  IQR {np.subtract(*np.percentile(Hs,[75,25])):.3f}")
    if d0 == 0 and k == 1:
        for r in rows: print("   ceil frame %3d: eye->ceil %.3f (Z %.2f)  eye->floor %.3f (Z %.2f, %d frames)" % (r[0], r[1], r[3], r[2], r[4], r[5]))
