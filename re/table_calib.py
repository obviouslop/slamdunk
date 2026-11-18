#!/usr/bin/env python3
"""Single-frame metric calibration against a known table (live11: top 1.40 x 0.90 m, 0.755 m above the floor).
Per frame: RANSAC planes (disparity space) -> floor plane + table-top plane (horizontal, 0.5-1.0 m apart);
height = y_top - y_floor; table-top outline (min-area rectangle of the plane inliers' largest blob).
Depth model Z = k * fB / (disp + d0): reports the fit of k and d0 to the known height / size."""
import sys, os, json, numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from livestereo import LiveStereo, unpack_raw
rec = sys.argv[1]; H_TRUE = 0.755; L_TRUE, W_TRUE = 1.40, 0.90
ls = LiveStereo(scale=0.35, max_depth=4.0); fB = ls.f * ls.baseline; f = ls.f; cx, cy = ls.P1[0, 2], ls.P1[1, 2]
recs = [json.loads(l) for l in open(os.path.join(rec, "frames.jsonl"))]
recs = [r for r in recs if r["left"]["pose_valid"] and abs(r["left"]["ts"] - r["right"]["ts"]) < 0.015]
cache = os.path.join(rec, "all_depth.npz")
if not os.path.exists(cache):
    Zs, Cs, Ts = [], [], []
    for r in recs:
        i = r["i"]
        L = unpack_raw(open(os.path.join(rec, f"{i:05d}_left.y10p"), "rb").read()); R = unpack_raw(open(os.path.join(rec, f"{i:05d}_right.y10p"), "rb").read())
        Z, c, _ = ls.depth(L, R, far=4.0); Zs.append(Z.astype(np.float16)); Cs.append(c); Ts.append(ls.world_from_rect(r["left"]["pose"]))
    np.savez(cache, Z=np.array(Zs), C=np.array(Cs), T=np.array(Ts))
z = np.load(cache); Zs, Ts = z["Z"], z["T"]
n = Zs.shape[1]; vv, uu = np.mgrid[0:n:2, 0:n:2]; rng = np.random.default_rng(0)

def ransac(uv, d, iters=250, tol=0.2):
    A = np.c_[uv, np.ones(len(d))]; best = None
    for _ in range(iters):
        i = rng.choice(len(d), 3, replace=False)
        try: p = np.linalg.solve(A[i], d[i])
        except np.linalg.LinAlgError: continue
        inl = np.abs(A @ p - d) < tol
        if best is None or inl.sum() > best.sum(): best = inl
    return best

def planes(j):
    """-> list of (uv, disp) inlier sets of horizontal planes in frame j (at k=1, d0=0), lowest first"""
    Z = Zs[j].astype(np.float32)[vv, uu]; ok = Z > 0
    uv = np.c_[uu[ok], vv[ok]].astype(float); d = fB / Z[ok]
    T = Ts[j]; rays = np.c_[(uv[:, 0] - cx) / f, (uv[:, 1] - cy) / f, np.ones(len(d))] @ T[:3, :3].T
    out = []; rem = np.ones(len(d), bool)
    for _ in range(4):
        idx = np.flatnonzero(rem)
        if len(idx) < 3000: break
        inl = idx[ransac(uv[idx], d[idx])]
        rem[inl] = False
        P = T[:3, 3] + rays[inl] * (fB / d[inl])[:, None]
        c = P.mean(0); _, _, vt = np.linalg.svd(P - c, full_matrices=False)
        if abs(vt[2][1]) > 0.985 and len(inl) > 1500:
            out.append((uv[inl], d[inl], rays[inl], c[1]))
    return sorted(out, key=lambda p: p[3])

def geom(pl, T, k, d0):
    uv, d, rays, _ = pl
    P = T[:3, 3] + rays * (k * fB / np.maximum(d + d0, 0.3))[:, None]
    return P

rows = []
for j in range(len(Zs)):
    pls = planes(j)
    if len(pls) < 2: continue
    fl = pls[0]
    for tp in pls[1:]:
        if 0.55 < tp[3] - fl[3] < 0.95:
            rows.append((j, fl, tp)); break
print(f"{len(rows)} frames with floor + table top")

def evaluate(k, d0, verbose=False):
    hs, zs, dims = [], [], []
    for j, fl, tp in rows:
        T = Ts[j]
        Pf = geom(fl, T, k, d0); Pt = geom(tp, T, k, d0)
        h = np.median(Pt[:, 1]) - np.median(Pf[:, 1]); hs.append(h)
        zs.append(np.median(k * fB / (tp[1] + d0)))
        # table outline: largest blob of the top-plane inliers on a 1 cm grid -> min-area rectangle
        g = np.floor(Pt[:, [0, 2]] / 0.01).astype(int); g -= g.min(0)
        if g.max() > 600: dims.append((np.nan, np.nan)); continue
        img = np.zeros(g.max(0)[::-1] + 1, np.uint8); img[g[:, 1], g[:, 0]] = 255
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        nl, lab, st, _ = cv2.connectedComponentsWithStats(img)
        b = 1 + np.argmax(st[1:, cv2.CC_STAT_AREA]); ys, xs = np.nonzero(lab == b)
        (_, _), (w, hh), _ = cv2.minAreaRect(np.c_[xs, ys].astype(np.float32))
        dims.append((max(w, hh) / 100, min(w, hh) / 100))
    return np.array(hs), np.array(zs), np.array(dims)

hs, zs, dims = evaluate(1.0, 0.0)
for (j, fl, tp), h, zz, dm in zip(rows, hs, zs, dims):
    print(f"  frame {j:3d}: table height {h:.3f} m (table at Z {zz:.2f} m)  top outline {dm[0]:.2f} x {dm[1]:.2f} m")
good = (dims[:, 0] > 1.2) & (dims[:, 0] < 1.6) & (dims[:, 1] > 0.75) & (dims[:, 1] < 1.05)
print(f"median height {np.median(hs):.3f} (true {H_TRUE})  ratio {np.median(hs)/H_TRUE:.4f}")
print(f"complete outlines ({good.sum()}): {np.median(dims[good,0]):.3f} x {np.median(dims[good,1]):.3f} (true {L_TRUE} x {W_TRUE}), "
      f"ratios {np.median(dims[good,0])/L_TRUE:.4f} {np.median(dims[good,1])/W_TRUE:.4f}")
# fit: height error vs distance -> separate k and d0
from scipy.optimize import least_squares
def res(x):
    h, _, _ = evaluate(x[0], x[1]); return h - H_TRUE
sol = least_squares(res, [1.0, 0.0], loss="soft_l1", f_scale=0.02)
h2, _, d2 = evaluate(*sol.x)
print(f"fit k={sol.x[0]:.4f} d0={sol.x[1]:+.3f} px -> height median {np.median(h2):.3f}, outline {np.median(d2[good,0]):.3f} x {np.median(d2[good,1]):.3f}")
for k in (1.02, 1.04):
    h3, _, d3 = evaluate(k, 0.0); print(f"k={k} d0=0: height {np.median(h3):.3f}, outline {np.median(d3[good,0]):.3f} x {np.median(d3[good,1]):.3f}")
for d0 in (-0.2, -0.4):
    h3, _, d3 = evaluate(1.0, d0); print(f"k=1 d0={d0}: height {np.median(h3):.3f}, outline {np.median(d3[good,0]):.3f} x {np.median(d3[good,1]):.3f}")
