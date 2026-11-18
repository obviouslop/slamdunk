#!/usr/bin/env python3
"""Absolute floor height (tracking STAGE y) per frame over a recording -> vertical tracking drift."""
import sys, numpy as np
z = np.load(sys.argv[1]); O, D, disp, fB, F, UV = z["O"], z["D"], z["disp"], float(z["fB"]), z["frame"], z["uv"].astype(float)
rng = np.random.default_rng(0)
def ransac(uv, d, iters=200, tol=0.25):
    A = np.c_[uv, np.ones(len(d))]; best = None
    for _ in range(iters):
        i = rng.choice(len(d), 3, replace=False)
        try: p = np.linalg.solve(A[i], d[i])
        except np.linalg.LinAlgError: continue
        inl = np.abs(A @ p - d) < tol
        if best is None or inl.sum() > best.sum(): best = inl
    return best
rows = []
for f in range(F.max() + 1):
    s = np.flatnonzero(F == f)
    if len(s) < 1000: continue
    Dn = D[s] / np.linalg.norm(D[s], axis=1, keepdims=True); P = O[s] + D[s] * (fB / disp[s])[:, None]; eye = O[s][0, 1]
    cand = s[(Dn[:, 1] < -0.35) & (P[:, 1] < eye - 1.3)]
    if len(cand) < 2000: continue
    idx = cand[ransac(UV[cand], disp[cand])]
    if len(idx) < 2000: continue
    Q = O[idx] + D[idx] * (fB / disp[idx])[:, None]; c = Q.mean(0)
    _, _, vt = np.linalg.svd(Q - c, full_matrices=False)
    if abs(vt[2][1]) < 0.985: continue
    rows.append((f, c[1], eye, c[0], c[2], O[s][0, 0], O[s][0, 2]))
r = np.array(rows)
for x in r: print("frame %3d  floor y %.3f  eye %.3f  eye-floor %.3f   floor at x %.2f z %.2f" % (x[0], x[1], x[2], x[2] - x[1], x[3], x[4]))
