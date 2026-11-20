#!/usr/bin/env python3
"""Recording -> COLMAP text dataset for Gaussian-splat training on a workstation (Brush / gsplat / OpenSplat / Nerfstudio).

usage: export_colmap.py RECDIR OUTDIR [--ply fused.ply] [--fov 110] [--size 1600] [--target 300] [--density 1]
                        [--blur-pct 15] [--cams left,right]

  * poses: SteamVR head pose of each camera's own exposure (STAGE frame, y up) x factory hand-eye -> COLMAP world-to-camera
    (COLMAP camera axes == OpenCV, which our calibration uses). Poses carry 10-20 mm drift: let the trainer refine them
    (Brush/gsplat pose optimisation) or run a short bundle adjustment first.
  * images: full-resolution RAW10 -> demosaic -> exposure-normalised (exp_lines x analog gain x digital gain, so the auto
    exposure does not make frames differ), one global white balance and scale for the whole recording -> sRGB -> fisheye
    (Kannala-Brandt) undistorted to a pinhole of --fov degrees. Floaters from AE changes are the classic splat artefact.
  * image selection: drop the blurriest --blur-pct %, then greedy spatial decimation in (position, viewing direction): sharpest
    first, keep an image when no kept view is nearer than a radius that shrinks where there are more features and where more time
    was spent (--density), with the radius tuned to give about --target images.
  * points3D: --ply fused cloud, else the stereo depth points of the exported frames themselves (4 cm voxels seen twice).
"""
import argparse, json, os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from livestereo import LiveStereo, unpack_raw
from stills import upright
import handmask
import ctrl_overlay as co

ap = argparse.ArgumentParser()
ap.add_argument("rec"); ap.add_argument("out")
ap.add_argument("--ply"); ap.add_argument("--fov", type=float, default=110.0); ap.add_argument("--size", type=int, default=1600)
ap.add_argument("--target", type=int, default=300, help="about this many images, chosen for even spatial coverage"); ap.add_argument("--density", type=float, default=1.0, help="0 = uniform coverage, 1 = denser where there are more features / more time spent")
ap.add_argument("--blur-pct", type=float, default=15.0)
ap.add_argument("--masks", action="store_true", help="masks/<image>.png (255 keep, 0 ignore): fisheye border, own body / near things, dynamic objects vs the static map, controllers")
ap.add_argument("--depth", action="store_true", help="depth/<image>.png 16-bit mm (800x800, 0 invalid) from the stereo depth of the same frame")
ap.add_argument("--near", type=float, default=0.55, help="masks: ignore anything nearer than this (m) in the lower image half, and nearer than 0.7x this anywhere")
ap.add_argument("--static-margin", type=float, default=0.15, help="masks: with --ply, ignore pixels nearer than the fused map by more than this (m)")
ap.add_argument("--cams", default="left,right"); ap.add_argument("--max-points", type=int, default=300000)
a = ap.parse_args()

ls = LiveStereo(scale=0.5)


def progress(frac, stage):
    """progress file next to the output dir (read by pcserver for the viewer's progress bar)"""
    try:
        with open(a.out.rstrip("/") + ".progress", "w") as f: json.dump(dict(frac=round(float(frac), 3), stage=stage), f)
    except OSError: pass


UP = {}          # per camera: (cv2 rotate code or None, R_old_from_new). The sensors are mounted +-90 deg: rotate the images upright.
KB = {"left": (ls.KL, ls.DL), "right": (ls.KR, ls.DR)}


def exposure_factor(m):
    """Relative light-gathering of a frame: integration lines x analog gain (code c -> 1024/(1024-c), 960 = 16x) x digital gain."""
    return m["exp_lines"] * (1024.0 / max(1024 - m["gain_code"], 1)) * (m.get("dgain", 256) or 256) / 256.0


def cam_pose(side, m):
    """STAGE-from-camera 4x4 (OpenCV axes) for the exposure of this camera."""
    H = np.eye(4); H[:3, :] = np.asarray(m["pose"])
    T = H @ ls.head_from_camL
    if side == "right":
        C = np.eye(4); C[:3, :3] = ls.R.T; C[:3, 3] = -ls.R.T @ ls.t      # camL_from_camR
        T = T @ C
    if side not in UP:
        code, Rn, _ = upright(ls, side); UP[side] = (code, Rn)
    U = np.eye(4); U[:3, :3] = UP[side][1]
    return T @ U                                                           # pose of the upright (rotated) image


def rot_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def demosaic(raw):
    m = np.clip(raw.astype(np.int32) - 64, 0, 1023).astype(np.uint16)
    return cv2.cvtColor(m, cv2.COLOR_BayerBG2BGR).astype(np.float32)       # OpenCV's BG code == RGGB here (tested in stills.py)


recs = [json.loads(l) for l in open(os.path.join(a.rec, "frames.jsonl"))]
sides = a.cams.split(",")
cand = []
for r in recs:
    for s in sides:
        m = r[s]
        if m.get("pose_valid"): cand.append((s, r["i"], m))
print(f"{len(cand)} candidate images")


def load(s, i):
    return unpack_raw(open(os.path.join(a.rec, f"{i:05d}_{s}.y10p"), "rb").read())


# 1) per image: sharpness (exposure-normalised Laplacian variance) and feature richness (FAST corners) on a quarter-res grey image
fast = cv2.FastFeatureDetector_create(threshold=12)
score, feat = {}, {}
for ci, (s, i, m) in enumerate(cand):
    if ci % 10 == 0: progress(0.2 * ci / max(len(cand), 1), "scoring")
    raw = load(s, i).astype(np.float32)
    g = np.clip(raw[::4, ::4] - 64, 0, None) / exposure_factor(m)
    g = np.sqrt(np.clip(g / max(np.percentile(g, 99), 1e-9), 0, 1)).astype(np.float32)
    score[(s, i)] = cv2.Laplacian(g, cv2.CV_32F).var()
    feat[(s, i)] = len(fast.detect((g * 255).astype(np.uint8), None))
thr = np.percentile(list(score.values()), a.blur_pct)
pool = [(s, i, m, cam_pose(s, m)) for s, i, m in cand if score[(s, i)] >= thr]            # drop the blurriest
print(f"{len(pool)} sharp enough (blur threshold {thr:.2e})")

# 2) spatial selection: view distance = |d position| + W * angle (W metres per radian); greedy in order of sharpness, an image
#    is kept when no kept image is closer than its own radius r_i = r0 / density_i. density_i grows with feature richness and
#    with dwell (how many candidate views were taken around there), so busy / lingered-at places get denser coverage and empty
#    walls / transit get sparse coverage. r0 is bisected so that about --target images are kept (--density 0 = uniform coverage).
W = 0.5
pos = np.array([T[:3, 3] for *_, T in pool]); zax = np.array([T[:3, 2] for *_, T in pool])
D = np.linalg.norm(pos[:, None] - pos[None], axis=2) + W * np.arccos(np.clip(zax @ zax.T, -1, 1))
fe = np.array([feat[(s, i)] for s, i, *_ in pool], float); dw = (D < 0.25).sum(1).astype(float)
dens = np.clip(np.sqrt(fe / max(np.median(fe), 1) * dw / max(np.median(dw), 1)), 0.5, 2.5) ** a.density
order = np.argsort([-score[(s, i)] for s, i, *_ in pool])


def select(r0):
    kept = []
    for k in order:
        if not kept or D[k, kept].min() > r0 / dens[k]: kept.append(k)
    return kept


lo, hi = 0.005, 3.0
for _ in range(25):
    mid = (lo * hi) ** 0.5
    if len(select(mid)) > a.target: lo = mid
    else: hi = mid
kept = sorted(select(hi), key=lambda k: (pool[k][1], pool[k][0]))
keep = [pool[k] for k in kept]
print(f"{len(keep)} images selected (r0 {hi:.3f} m-equivalent, density gain {a.density})")
if not keep: sys.exit("nothing selected: raise --target / lower --blur-pct")

# 3) global white balance / scale from a sample of the selected frames
wbs, hi = [], []
for s, i, m, T in keep[::max(1, len(keep) // 12)]:
    x = demosaic(load(s, i)) / exposure_factor(m)
    mm = x[::8, ::8].reshape(-1, 3).mean(0); wbs.append(mm.mean() / np.maximum(mm, 1e-9)); hi.append(np.percentile(x[::8, ::8], 99.5))
wb = np.median(wbs, 0).astype(np.float32); sc = float(np.median(hi) * 1.0 * wb.max())
print("global white balance (BGR gains)", wb.round(3), "scale", sc)

# 4) pinhole target and per-camera undistortion maps
N = a.size; f_new = (N / 2) / np.tan(np.radians(a.fov / 2))
Knew = np.array([[f_new, 0, N / 2 - 0.5], [0, f_new, N / 2 - 0.5], [0, 0, 1.]])
maps = {s: cv2.fisheye.initUndistortRectifyMap(KB[s][0], KB[s][1], np.eye(3), Knew, (N, N), cv2.CV_16SC2) for s in sides}

img_dir = os.path.join(a.out, "images"); sp = os.path.join(a.out, "sparse", "0")
import shutil
for sub in ("images", "masks", "depth"): shutil.rmtree(os.path.join(a.out, sub), ignore_errors=True)      # no leftovers of an earlier export
os.makedirs(img_dir, exist_ok=True); os.makedirs(sp, exist_ok=True)
with open(os.path.join(sp, "cameras.txt"), "w") as fc:
    fc.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
    for ci, s in enumerate(sides, 1):
        fc.write(f"{ci} PINHOLE {N} {N} {f_new:.4f} {f_new:.4f} {N/2-0.5:.2f} {N/2-0.5:.2f}\n")
cam_id = {s: k for k, s in enumerate(sides, 1)}


def qvec(R):
    """rotation matrix -> COLMAP quaternion (w, x, y, z), w >= 0"""
    t = np.trace(R)
    if t > 0:
        S = np.sqrt(t + 1) * 2; w = S / 4; x = (R[2, 1] - R[1, 2]) / S; y = (R[0, 2] - R[2, 0]) / S; z = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2; w = (R[2, 1] - R[1, 2]) / S; x = S / 4; y = (R[0, 1] + R[1, 0]) / S; z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2; w = (R[0, 2] - R[2, 0]) / S; x = (R[0, 1] + R[1, 0]) / S; y = S / 4; z = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2; w = (R[1, 0] - R[0, 1]) / S; x = (R[0, 2] + R[2, 0]) / S; y = (R[1, 2] + R[2, 1]) / S; z = S / 4
    q = np.array([w, x, y, z]); return q if w >= 0 else -q


def read_ply(fn):
    with open(fn, "rb") as fh:
        props = []; n = 0
        while True:
            l = fh.readline().decode().strip()
            if l.startswith("element vertex"): n = int(l.split()[-1])
            elif l.startswith("property"): props.append(l.split()[1:])
            elif l == "end_header": break
        tm = {"float": "<f4", "double": "<f8", "uchar": "u1", "int": "<i4", "uint": "<u4", "short": "<i2", "ushort": "<u2"}
        d = np.frombuffer(fh.read(), np.dtype([(nm, tm[t]) for t, nm in props]), count=n)
    return np.stack([d["x"], d["y"], d["z"]], 1).astype(np.float64), np.stack([d["red"], d["green"], d["blue"]], 1)



NM = N // 2; fm = f_new / 2; cm = (N / 2 - 0.5) / 2           # depth / mask working resolution (800)
by_i = {r["i"]: r for r in recs}
tracks = {}
cfn = os.path.join(a.rec, "hmd_500hz.txt.controllers")
if a.masks and os.path.exists(cfn): tracks = co.load_controllers(cfn)
valid_fish = {}
for s_ in sides:                                                    # fisheye image circle in the upright pinhole image
    v = cv2.remap(np.full((2464, 2464), 255, np.uint8), *maps[s_], cv2.INTER_NEAREST)
    if UP[s_][0] is not None: v = cv2.rotate(v, UP[s_][0])
    valid_fish[s_] = cv2.erode(cv2.resize(v, (NM, NM), interpolation=cv2.INTER_NEAREST), np.ones((9, 9), np.uint8)) > 0
if a.masks or a.depth:
    os.makedirs(os.path.join(a.out, "depth"), exist_ok=True); os.makedirs(os.path.join(a.out, "masks"), exist_ok=True)
map_pts = None
auto_pts = not a.ply; auto_P, auto_C = [], []; rng_pts = np.random.default_rng(0)
if a.masks and a.ply: map_pts = read_ply(a.ply)[0]
fcache = {}


def zbuf(Pw, Tcam):
    """world points -> depth image (NM x NM, 0 = none) in the upright pinhole of camera pose Tcam (min filter 3x3 closes the gaps)"""
    Tc = np.linalg.inv(Tcam); Pc = Pw @ Tc[:3, :3].T + Tc[:3, 3]
    ok = Pc[:, 2] > 0.1; Pc = Pc[ok]
    u = np.round(fm * Pc[:, 0] / Pc[:, 2] + cm).astype(int); v = np.round(fm * Pc[:, 1] / Pc[:, 2] + cm).astype(int)
    k = (u >= 0) & (u < NM) & (v >= 0) & (v < NM)
    D = np.full(NM * NM, np.inf, np.float32); np.minimum.at(D, v[k] * NM + u[k], Pc[k, 2].astype(np.float32))
    D = D.reshape(NM, NM); D[~np.isfinite(D)] = 1e6
    D = cv2.erode(D, np.ones((3, 3), np.uint8)); D[D > 1e5] = 0
    return D


def frame_points(i):
    """stereo depth of pair i as world points (and the hand-mask points), cached"""
    if i in fcache: return fcache[i]
    r = by_i[i]; hp = np.array(r["left"]["pose"] if r["left"].get("pose_valid") else r["right"]["pose"])      # head pose of the exposure
    Zr, rgbr, _ = ls.depth(load("left", i), load("right", i), far=4.0, half=False)
    W = ls.world_from_rect(hp)
    vv, uu = np.nonzero(Zr > 0); z = Zr[vv, uu]
    Pr = np.stack([(uu - ls.P1[0, 2]) * z / ls.f, (vv - ls.P1[1, 2]) * z / ls.f, z], 1)
    Pw = Pr @ W[:3, :3].T + W[:3, 3]
    if auto_pts:                                                         # initial points from the capture itself (no fused cloud given)
        k = rng_pts.choice(len(Pw), min(len(Pw), 20000), replace=False)
        auto_P.append(Pw[k].astype(np.float32)); auto_C.append(rgbr[vv[k], uu[k]])
    hw = np.zeros((0, 3))
    if tracks:
        hands = []
        for role in (1, 2):
            if role in tracks:
                M = co.pose_at(tracks[role], r["left"]["ts"])
                if M is not None: hands.append(M[:3, 3])
        hm = handmask.hand_mask(Zr, np.linalg.inv(W), ls.f, ls.P1[0, 2], ls.P1[1, 2], hands)
        hw = Pw[hm[vv, uu]]
    fcache.clear(); fcache[i] = (Pw, hw); return fcache[i]


def depth_and_mask(s, i, T):
    Pw, hw = frame_points(i); D = zbuf(Pw, T)
    if a.depth:
        cv2.imwrite(os.path.join(a.out, "depth", f"{s}_{i:05d}.jpg.png"), np.clip(D * 1000, 0, 65535).astype(np.uint16))
    if a.masks:
        ign = ~valid_fish[s]
        lower = np.zeros((NM, NM), bool); lower[NM // 2:] = True
        ign |= (D > 0) & (((D < a.near) & lower) | (D < 0.7 * a.near))                    # own body / legs / things at the lens
        if map_pts is not None:
            Dm = zbuf(map_pts, T); ign |= (D > 0) & (Dm > 0) & (D < Dm - a.static_margin)  # moved objects, people
        if len(hw):
            Dh = zbuf(hw, T); ign |= cv2.dilate((Dh > 0).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        ign = cv2.morphologyEx(ign.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))      # drop speckle
        ign = cv2.dilate(ign, np.ones((7, 7), np.uint8)) > 0
        mk = cv2.resize((~ign).astype(np.uint8) * 255, (N, N), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(os.path.join(a.out, "masks", f"{s}_{i:05d}.jpg.png"), mk)


with open(os.path.join(sp, "images.txt"), "w") as fi:
    fi.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n# (empty line: no 2D points)\n")
    for n, (s, i, m, T) in enumerate(keep, 1):
        lin = demosaic(load(s, i)); E = exposure_factor(m)
        x = lin / E * wb / sc
        x += np.maximum(x.max(-1, keepdims=True) - 1, 0)                 # clipped highlights -> white, not magenta
        x[lin.max(-1) > 0.97 * 959] = 1.0                                # sensor-saturated pixels
        x = np.clip(x, 0, 1)
        x = np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)
        im = cv2.remap((x * 255).astype(np.uint8), *maps[s], cv2.INTER_LINEAR)
        if UP[s][0] is not None: im = cv2.rotate(im, UP[s][0])
        name = f"{s}_{i:05d}.jpg"; cv2.imwrite(os.path.join(img_dir, name), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if a.masks or a.depth: depth_and_mask(s, i, T)
        elif auto_pts: frame_points(i)
        Tc = np.linalg.inv(T)                                   # world-to-camera
        q = qvec(Tc[:3, :3]); t = Tc[:3, 3]
        fi.write(f"{n} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {cam_id[s]} {name}\n\n")
        progress(0.2 + 0.75 * n / len(keep), "images")
        if n % 25 == 0: print(f"  {n}/{len(keep)}", flush=True)


progress(0.95, "points")
npts = 0
with open(os.path.join(sp, "points3D.txt"), "w") as fp:
    fp.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
    if a.ply:
        P, C = read_ply(a.ply)
        if len(P) > a.max_points:
            k = np.random.default_rng(0).choice(len(P), a.max_points, replace=False); P, C = P[k], C[k]
        for k in range(len(P)):
            fp.write(f"{k+1} {P[k,0]:.4f} {P[k,1]:.4f} {P[k,2]:.4f} {C[k,0]} {C[k,1]} {C[k,2]} 1.0\n")
        npts = len(P)
    elif auto_P:        # stereo points of all exported frames, 4 cm voxels seen at least twice (drops noise), capped
        P = np.vstack(auto_P); C = np.vstack(auto_C)
        g = np.floor(P / 0.04).astype(np.int64) + 50000; key = (g[:, 0] << 40) | (g[:, 1] << 20) | g[:, 2]
        _, idx, cnt = np.unique(key, return_index=True, return_counts=True); k = idx[cnt >= 2]
        P, C = P[k], C[k]
        if len(P) > a.max_points: k = rng_pts.choice(len(P), a.max_points, replace=False); P, C = P[k], C[k]
        for k in range(len(P)):
            fp.write(f"{k+1} {P[k,0]:.4f} {P[k,1]:.4f} {P[k,2]:.4f} {C[k,0]} {C[k,1]} {C[k,2]} 1.0\n")
        npts = len(P)
json.dump(dict(rec=a.rec, images=len(keep), fov=a.fov, size=N, points=npts, wb=wb.tolist(), scale=sc, frame="STAGE (SteamVR), y up, metres"),
          open(os.path.join(a.out, "export.json"), "w"), indent=1)
progress(1.0, "done")
print(f"wrote {a.out}: {len(keep)} images, {npts} points. Train e.g.:  brush {a.out}")
