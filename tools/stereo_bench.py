#!/usr/bin/env python3
"""Offline benchmark of stereo matchers on an export_colmap dataset (left_/right_ pinhole pairs + poses).

    .venv-bench/bin/python tools/stereo_bench.py ~/splat/in/live17 [--size 800] [--n 40] [--methods sgbm,sgbm_wls,census_sgm,bm_wls]

No ground truth exists, so each method is scored by
  * density      : fraction of pixels with a depth in [min_depth, max_depth]
  * mv_agree     : multi-view consistency - points of frame i projected into frames i+-1, i+-2 (left cams, same method):
                   share whose depth agrees with that frame's depth within 3 % (higher = less noise/spray)
  * mv_front     : share that lands > 5 % IN FRONT of the other frame's surface (= floaters / spray)
  * ply_med      : median |dZ|/Z vs the fused TSDF cloud (fused.ply) rendered into the frame (biased to SGBM, which built it)
  * ms           : wall time per pair on this machine (relative only)
"""
import argparse, os, time, json
import numpy as np, cv2
from numba import njit, prange

ap = argparse.ArgumentParser()
ap.add_argument("ds"); ap.add_argument("--size", type=int, default=800); ap.add_argument("--n", type=int, default=40)
ap.add_argument("--methods", default="sgbm,sgbm_wls,census_sgm,bm_wls")
ap.add_argument("--min-depth", type=float, default=0.35); ap.add_argument("--max-depth", type=float, default=4.0)
ap.add_argument("--stride", type=int, default=1, help="rectified set: every k-th frame of each window")
ap.add_argument("--nbr", default="1,2", help="neighbour offsets (in ids) for the multi-view consistency")
ap.add_argument("--dump", help="write per-frame results json")
ap.add_argument("--save", help="dir for disparity PNGs of the first pair")
a = ap.parse_args()


def qrot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


WFR = {}; WIN = {}; BLUR = {}
if os.path.exists(os.path.join(a.ds, "set.json")):      # rectified set from tools/video_evalset.py
    st = json.load(open(os.path.join(a.ds, "set.json")))
    f, B, flip, N = st["f"], st["B"], st["flip"], st["size"]
    P1 = np.array([[f, 0, st["cx"], 0], [0, f, st["cy"], 0], [0, 0, 1, 0.]]); R1 = np.eye(3)
    fr = st["frames"]
    if a.stride > 1: fr = [x for x in fr if (x["i"] - x["win"]) % a.stride == 0]
    fr = fr[: a.n]; ids = [x["i"] for x in fr]
    for x in fr: WFR[x["i"]] = np.array(x["world_from_rect"]); WIN[x["i"]] = x["win"]; BLUR[x["i"]] = x["blur_px"]

    def load(i):
        return tuple(cv2.imread(os.path.join(a.ds, "rect", f"{i:06d}_{c}.png"), cv2.IMREAD_GRAYSCALE) for c in "LR")
else:
    sp = os.path.join(a.ds, "sparse", "0")
    cam = {}
    for l in open(os.path.join(sp, "cameras.txt")):
        if l[0] == "#" or not l.strip(): continue
        p = l.split(); cam[int(p[0])] = (int(p[2]), [float(v) for v in p[4:8]])
    poses = {}
    for l in open(os.path.join(sp, "images.txt")):
        p = l.split()
        if len(p) != 10 or l[0] == "#": continue
        poses[p[9]] = (qrot([float(v) for v in p[1:5]]), np.array([float(v) for v in p[5:8]]))   # world-to-cam
    W0, (fx, fy, cx, cy) = cam[1]
    s = a.size / W0
    K = np.array([[fx * s, 0, (cx + 0.5) * s - 0.5], [0, fy * s, (cy + 0.5) * s - 0.5], [0, 0, 1]])
    ids = sorted(int(n[5:10]) for n in poses if n.startswith("left_") and f"right_{n[5:]}" in poses)[: a.n]
    N = a.size

    # rectification is fixed by the rig: compute it once from the first pair
    Rl, tl = poses[f"left_{ids[0]:05d}.jpg"]; Rr, tr = poses[f"right_{ids[0]:05d}.jpg"]
    R = Rr @ Rl.T; t = tr - R @ tl
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K, None, K, None, (N, N), R, t.reshape(3, 1), flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    mapL = cv2.initUndistortRectifyMap(K, None, R1, P1, (N, N), cv2.CV_16SC2)
    mapR = cv2.initUndistortRectifyMap(K, None, R2, P2, (N, N), cv2.CV_16SC2)
    f = P1[0, 0]; B = abs(P2[0, 3] / P2[0, 0]); flip = P2[0, 3] > 0     # camL must be the LEFT image of the pair


    def load(i):
        gl = cv2.imread(os.path.join(a.ds, "images", f"left_{i:05d}.jpg"), cv2.IMREAD_GRAYSCALE)
        gr = cv2.imread(os.path.join(a.ds, "images", f"right_{i:05d}.jpg"), cv2.IMREAD_GRAYSCALE)
        gl = cv2.resize(gl, (N, N), interpolation=cv2.INTER_AREA); gr = cv2.resize(gr, (N, N), interpolation=cv2.INTER_AREA)
        rl = cv2.remap(gl, *mapL, cv2.INTER_LINEAR); rr = cv2.remap(gr, *mapR, cv2.INTER_LINEAR)
        if flip: rl, rr = np.ascontiguousarray(rl[:, ::-1]), np.ascontiguousarray(rr[:, ::-1])
        return rl, rr
    for i in ids:
        Rw, tw = poses[f"left_{i:05d}.jpg"]; T = np.eye(4); T[:3, :3] = Rw; T[:3, 3] = tw
        Rr_ = np.eye(4); Rr_[:3, :3] = R1.T; WFR[i] = np.linalg.inv(T) @ Rr_; WIN[i] = 0
ND = 16 * int(np.ceil(f * B / a.min_depth / 16))
print(f"pairs {len(ids)}  rect f {f:.1f}px  B {B*1000:.1f}mm  numDisp {ND}  flip {flip}")


# ---------------------------------------------------------------- matchers (all return float32 disparity, <= 0 invalid)
blk = 5
sgbm = cv2.StereoSGBM_create(0, ND, blk, P1=8 * blk * blk, P2=32 * blk * blk, uniquenessRatio=10, speckleWindowSize=100,
                             speckleRange=2, disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)   # = livestereo.py
bm = cv2.StereoBM_create(ND, 11); bm.setUniquenessRatio(10); bm.setTextureThreshold(10)


def m_sgbm(l, r): return sgbm.compute(l, r).astype(np.float32) / 16


def wls(matcher, l, r, lam=8000.0, sig=1.2, cthr=0.0):
    rm = cv2.ximgproc.createRightMatcher(matcher)
    dl = matcher.compute(l, r); dr = rm.compute(r, l)
    w = cv2.ximgproc.createDisparityWLSFilter(matcher); w.setLambda(lam); w.setSigmaColor(sig)
    d = w.filter(dl, l, disparity_map_right=dr).astype(np.float32) / 16
    c = w.getConfidenceMap()
    d[(c <= cthr) | (dl < 0)] = -1   # keep only pixels with a raw match support (edge-preserving smoothing, no hole filling)
    return d


def m_sgbm_wls(l, r): return wls(sgbm, l, r)
def m_bm_wls(l, r): return wls(bm, l, r)


@njit(parallel=True, cache=True)
def census(img, hr, hc):
    H, W = img.shape; out = np.zeros((H, W), np.uint64)
    for y in prange(hr, H - hr):
        for x in range(hc, W - hc):
            c = img[y, x]; v = np.uint64(0)
            for dy in range(-hr, hr + 1):
                for dx in range(-hc, hc + 1):
                    if dy == 0 and dx == 0: continue
                    v = (v << np.uint64(1)) | np.uint64(img[y + dy, x + dx] < c)
            out[y, x] = v
    return out


@njit(cache=True)
def popc(v):
    c = 0
    while v:
        v &= v - np.uint64(1); c += 1
    return c


@njit(parallel=True, cache=True)
def cost_volume(cl, cr, D):
    H, W = cl.shape; C = np.full((H, W, D), 64, np.uint8)
    for y in prange(H):
        for x in range(W):
            for d in range(min(D, x + 1)):
                C[y, x, d] = popc(cl[y, x] ^ cr[y, x - d])
    return C


@njit(cache=True)
def _path(C, img, S, y0, x0, dy, dx, P1, P2):
    H, W, D = C.shape; prev = np.empty(D, np.int32); cur = np.empty(D, np.int32)
    y = np.int64(y0); x = np.int64(x0); ddy = np.int64(dy); ddx = np.int64(dx); first = True; pi = 0
    while 0 <= y < H and 0 <= x < W:
        if first:
            for d in range(D): prev[d] = C[y, x, d]
            first = False
        else:
            mn = prev.min()
            p2 = max(P1 + 1, P2 // (1 + abs(int(img[y, x]) - pi) // 8))       # intensity-adaptive P2 (edge-aware)
            for d in range(D):
                v = prev[d]
                if d > 0 and prev[d - 1] + P1 < v: v = prev[d - 1] + P1
                if d < D - 1 and prev[d + 1] + P1 < v: v = prev[d + 1] + P1
                if mn + p2 < v: v = mn + p2
                cur[d] = C[y, x, d] + v - mn
            prev[:] = cur
        for d in range(D): S[y, x, d] += prev[d]
        pi = int(img[y, x]); y += ddy; x += ddx


@njit(parallel=True, cache=True)
def sgm(C, img, P1, P2):
    H, W, D = C.shape; S = np.zeros((H, W, D), np.uint16)
    # 8 directions; each direction's scanlines are independent -> parallel (this is exactly the GPU decomposition)
    for kk in prange(H):
        k = np.int64(kk)
        _path(C, img, S, k, 0, 0, 1, P1, P2); _path(C, img, S, k, W - 1, 0, -1, P1, P2)
    for kk in prange(W):
        k = np.int64(kk)
        _path(C, img, S, 0, k, 1, 0, P1, P2); _path(C, img, S, H - 1, k, -1, 0, P1, P2)
    for kk in prange(H + W - 1):
        k = np.int64(kk)   # diagonals, start on the top / bottom border or left / right column
        if k < W:
            _path(C, img, S, 0, k, 1, 1, P1, P2); _path(C, img, S, H - 1, k, -1, -1, P1, P2)
            _path(C, img, S, 0, k, 1, -1, P1, P2); _path(C, img, S, H - 1, k, -1, 1, P1, P2)
        else:
            j = int(k) - W + 1
            _path(C, img, S, j, 0, 1, 1, P1, P2); _path(C, img, S, j, W - 1, -1, -1, P1, P2)
            _path(C, img, S, j, W - 1, 1, -1, P1, P2); _path(C, img, S, j, 0, -1, 1, P1, P2)
    return S


@njit(parallel=True, cache=True)
def wta(S, uniq):
    H, W, D = S.shape; dl = np.full((H, W), -1.0, np.float32); dr = np.full((H, W), -1, np.int32)
    for y in prange(H):
        best_r = np.full(W, 1 << 30, np.int64); arg_r = np.full(W, -1, np.int32)
        for x in range(W):
            b = 1 << 30; bd = -1
            for d in range(D):
                v = S[y, x, d]
                if v < b: b = v; bd = d
                xr = x - d
                if xr >= 0 and v < best_r[xr]: best_r[xr] = v; arg_r[xr] = d
            ok = True
            for d in range(D):
                if abs(d - bd) > 1 and S[y, x, d] * (100 - uniq) < b * 100: ok = False; break
            if ok and 0 < bd < D - 1:
                c0, c1, c2 = float(S[y, x, bd - 1]), float(S[y, x, bd]), float(S[y, x, bd + 1])
                den = c0 - 2 * c1 + c2
                dl[y, x] = bd + (0.5 * (c0 - c2) / den if den > 0 else 0.0)
        for x in range(W): dr[y, x] = arg_r[x]
        for x in range(W):   # left-right check
            d = dl[y, x]
            if d > 0:
                xr = int(x - d + 0.5)
                if xr < 0 or abs(dr[y, xr] - d) > 1.0: dl[y, x] = -1
    return dl


def m_census_sgm(l, r, P1=10, P2=120):
    cl = census(l, 3, 4); cr = census(r, 3, 4)       # 7x9 census (63 bits) - libSGM's window
    S = sgm(cost_volume(cl, cr, ND), l, P1, P2)
    d = wta(S, 5)
    dm = cv2.medianBlur(d, 3); d[np.abs(dm - d) > 1] = -1
    return d


PRIOR = {}   # set by the run loop: predicted disparity samples (x, y, d) in the current pair's (possibly mirrored) left image
TILE, PAD, MARGIN = 4, 24, 4.0
stat = dict(tiles=0, full=0, work=0.0)


def m_sgbm_tiles(l, r):
    """Per-tile SGBM with the disparity window predicted from the previous frame's depth (reprojected). Tiles without a usable
    prior are matched over the full range. The right crop is shifted by the window start, so minDisparity is 0."""
    n = l.shape[0]; ts = n // TILE; out = np.full((n, n), -1, np.float32)
    pr = PRIOR.get("d")
    for ty in range(TILE):
        for tx in range(TILE):
            x0, y0 = tx * ts, ty * ts; x1, y1 = x0 + ts, y0 + ts
            d0, nd = 0, ND
            if pr is not None:
                m = (pr[:, 0] >= x0) & (pr[:, 0] < x1) & (pr[:, 1] >= y0) & (pr[:, 1] < y1)
                if m.sum() > 0.05 * ts * ts / 64:        # prior is subsampled 8x8
                    lo, hi = np.percentile(pr[m, 2], [3, 97])
                    d0 = int(max(0, np.floor(lo - MARGIN))); nd = int(min(ND, 16 * np.ceil((hi + MARGIN - d0) / 16)))
            stat["tiles"] += 1; stat["full"] += nd == ND and d0 == 0
            xa = x0 - PAD - nd; ya, yb = max(0, y0 - PAD), min(n, y1 + PAD); xb = min(n, x1 + PAD)
            def crop(img, shift):
                a0, b0 = xa - shift, xb - shift
                c = img[ya:yb, max(0, a0):max(0, b0)]
                return np.ascontiguousarray(cv2.copyMakeBorder(c, 0, 0, max(0, -a0), (b0 - a0) - c.shape[1] - max(0, -a0), cv2.BORDER_REPLICATE)) \
                    if c.shape[1] != b0 - a0 else np.ascontiguousarray(c)
            cl, cr = crop(l, 0), crop(r, d0)
            sg = cv2.StereoSGBM_create(0, nd, blk, P1=8 * blk * blk, P2=32 * blk * blk, uniquenessRatio=10, speckleWindowSize=100,
                                       speckleRange=2, disp12MaxDiff=1, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
            d = sg.compute(cl, cr).astype(np.float32) / 16
            stat["work"] += cl.size * nd
            sub = d[y0 - ya: y0 - ya + ts, x0 - xa: x0 - xa + ts]
            edge = (sub < 1.5) & (d0 > 0) | (sub > nd - 2.5)      # pinned to the window border: true disparity is outside
            out[y0:y1, x0:x1] = np.where((sub >= 0) & ~edge, sub + d0, -1)
    xs = np.arange(n, dtype=np.float32)[None, :]
    out[xs - out - blk // 2 < 0] = -1        # match would lie outside the right image (border padding)
    return out


METHODS = dict(sgbm_tiles=m_sgbm_tiles, sgbm=m_sgbm, sgbm_wls=m_sgbm_wls, census_sgm=m_census_sgm, bm_wls=m_bm_wls)

# ---------------------------------------------------------------- geometry for scoring
u, v = np.meshgrid(np.arange(N, dtype=np.float32), np.arange(N, dtype=np.float32))
if flip: u = N - 1 - u
ccx, ccy = P1[0, 2], P1[1, 2]


def depth(d):
    Z = np.where(d > 0.5, f * B / np.maximum(d, 1e-3), 0).astype(np.float32)
    Z[(Z < a.min_depth) | (Z > a.max_depth)] = 0
    return Z   # pixel (row, col) of the possibly mirrored image; true column = u


def world_pts(i, Z):
    m = Z > 0
    X = np.stack([(u[m] - ccx) * Z[m] / f, (v[m] - ccy) * Z[m] / f, Z[m]], 1)      # rect frame
    T = WFR[i]; return X @ T[:3, :3].T + T[:3, 3]


def project(i, Pw):
    T = np.linalg.inv(WFR[i]); Xr = Pw @ T[:3, :3].T + T[:3, 3]
    z = Xr[:, 2]; ok = z > 0.05
    uu = f * Xr[:, 0] / np.where(ok, z, 1) + ccx; vv = f * Xr[:, 1] / np.where(ok, z, 1) + ccy
    if flip: uu = N - 1 - uu
    ui = np.round(uu).astype(int); vi = np.round(vv).astype(int)
    ok &= (ui >= 0) & (ui < N) & (vi >= 0) & (vi < N)
    return ui, vi, z, ok


def read_ply(fn):
    with open(fn, "rb") as fh:
        props = []; n = 0
        while True:
            l = fh.readline().decode().strip()
            if l.startswith("element vertex"): n = int(l.split()[-1])
            elif l.startswith("property"): props.append(l.split()[1:])
            elif l == "end_header": break
        tm = {"float": "<f4", "uchar": "u1", "double": "<f8"}
        d = np.frombuffer(fh.read(), np.dtype([(nm, tm[t]) for t, nm in props]), count=n)
    return np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64)


PLY = read_ply(os.path.join(a.ds, "fused.ply")) if not BLUR and os.path.exists(os.path.join(a.ds, "fused.ply")) else None


def ply_depth(i):
    ui, vi, z, ok = project(i, PLY)
    Zr = np.full((N, N), np.inf, np.float32)
    np.minimum.at(Zr, (vi[ok], ui[ok]), z[ok].astype(np.float32))
    Zr = -cv2.dilate(-np.where(np.isinf(Zr), 1e9, Zr), np.ones((3, 3), np.uint8))   # fill splat gaps (nearest surface)
    Zr[Zr > 1e8] = 0
    return Zr


# ---------------------------------------------------------------- run
NBR = [int(x) for x in a.nbr.split(",")]; NBR = sorted(set(NBR + [-x for x in NBR]))
pairs = {i: load(i) for i in ids}
plyZ = {i: ply_depth(i) for i in ids} if PLY is not None else {}
results = {}
for name in a.methods.split(","):
    fn = METHODS[name]; fn(*pairs[ids[0]])        # warm-up (numba JIT)
    Zs = {}; t0 = time.perf_counter()
    PRIOR.clear(); stat.update(tiles=0, full=0, work=0.0); prev = None
    for i in ids:
        if prev is not None and WIN[prev[0]] != WIN[i]: prev = None; PRIOR["d"] = None
        if os.environ.get("PRIOR") == "map" and plyZ:   # fused map rendered into this view (= TSDF raycast)
            Zr = plyZ[i]; yy, xx = np.mgrid[0:N:8, 0:N:8]; zz = Zr[yy, xx]; ok = zz > a.min_depth
            PRIOR["d"] = np.stack([xx[ok], yy[ok], f * B / zz[ok]], 1)
        elif prev is not None:      # previous frame's depth -> this frame's disparity (8x8 subsampled), like the live loop would
            Zp = prev[1].copy(); keep = np.zeros_like(Zp, bool); keep[::8, ::8] = True; Zp[~keep] = 0
            ui, vi, z, ok = project(i, world_pts(prev[0], Zp))
            ok &= z > a.min_depth
            PRIOR["d"] = np.stack([ui[ok], vi[ok], f * B / z[ok]], 1) if ok.sum() else None
        Zs[i] = depth(fn(*pairs[i])); prev = (i, Zs[i])
    ms = (time.perf_counter() - t0) / len(ids) * 1000
    if a.save:
        os.makedirs(a.save, exist_ok=True)
        Z = Zs[ids[0]]; vis = np.where(Z > 0, 255 * (1 - (Z - a.min_depth) / (a.max_depth - a.min_depth)), 0).astype(np.uint8)
        cv2.imwrite(os.path.join(a.save, f"{name}.png"), cv2.applyColorMap(vis, cv2.COLORMAP_TURBO) * (Z > 0)[..., None])
        cv2.imwrite(os.path.join(a.save, "left.png"), pairs[ids[0]][0])
    dens = np.mean([(Z > 0).mean() for Z in Zs.values()])
    agree = front = tot = 0; pm = []; per = {}
    for k, i in enumerate(ids):
        Pw = world_pts(i, Zs[i])
        for o in NBR:
            if not 0 <= k + o < len(ids) or WIN[ids[k + o]] != WIN[i]: continue
            j = ids[k + o]
            ui, vi, z, ok = project(j, Pw)
            Zj = Zs[j][vi[ok], ui[ok]]; zz = z[ok]; h = Zj > 0
            rel = (zz[h] - Zj[h]) / Zj[h]
            agree += np.sum(np.abs(rel) < 0.03); front += np.sum(rel < -0.05); tot += len(rel)
            pa = per.setdefault(i, [0, 0]); pa[0] += np.sum(np.abs(rel) < 0.03); pa[1] += len(rel)
        if plyZ:
            Zr = plyZ[i]; m = (Zs[i] > 0) & (Zr > 0)
            pm.append(np.abs(Zs[i][m] - Zr[m]) / Zr[m])
    pmed = float(np.median(np.concatenate(pm))) if pm else float("nan")
    results[name] = dict(ms=ms, density=dens, mv_agree=agree / max(tot, 1), mv_front=front / max(tot, 1), ply_med=pmed)
    r = results[name]
    if name == "sgbm_tiles":
        print(f"  tiles with prior {1 - stat['full']/stat['tiles']:.2f}, matching work vs full frame {stat['work'] / (len(ids) * N * N * ND):.2f}")
    print(f"{name:11s} {ms:7.0f} ms  density {r['density']:.3f}  mv_agree {r['mv_agree']:.3f}  mv_front {r['mv_front']:.3f}  "
          f"ply_med {100*pmed:.2f} %", flush=True)
    if a.dump: r["per"] = {int(i): [int(x) for x in per[i]] for i in per}
    if BLUR:   # per-frame consistency vs estimated blur
        bins = [0, 0.5, 1, 1.5, 2, 3, 99]; bb = np.array([BLUR[i] for i in per]); ag = np.array([per[i][0] / max(per[i][1], 1) for i in per])
        print("    blur px  " + "  ".join(f"[{lo:g},{hi:g}):{ag[(bb >= lo) & (bb < hi)].mean():.3f}(n{((bb >= lo) & (bb < hi)).sum()})"
                                     for lo, hi in zip(bins[:-1], bins[1:]) if ((bb >= lo) & (bb < hi)).any()))
print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "per"} for k, v in results.items()}))
if a.dump: json.dump(results, open(a.dump, "w"))
