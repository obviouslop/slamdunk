#!/usr/bin/env python3
"""Recording -> COLMAP text dataset for Gaussian-splat training on a workstation (Brush / gsplat / OpenSplat / Nerfstudio).

usage: export_colmap.py RECDIR OUTDIR [--ply fused.ply] [--fov 110] [--size 1600] [--min-move 0.05] [--min-rot 5]
                        [--blur-pct 25] [--max-images 400] [--cams left,right]

  * poses: SteamVR head pose of each camera's own exposure (STAGE frame, y up) x factory hand-eye -> COLMAP world-to-camera
    (COLMAP camera axes == OpenCV, which our calibration uses). Poses carry 10-20 mm drift: let the trainer refine them
    (Brush/gsplat pose optimisation) or run a short bundle adjustment first.
  * images: full-resolution RAW10 -> demosaic -> exposure-normalised (exp_lines x analog gain x digital gain, so the auto
    exposure does not make frames differ), one global white balance and scale for the whole recording -> sRGB -> fisheye
    (Kannala-Brandt) undistorted to a pinhole of --fov degrees. Floaters from AE changes are the classic splat artefact.
  * frame selection: keep a frame when it moved > --min-move m or rotated > --min-rot deg from the last kept frame of the
    same camera AND is not among the blurriest (--blur-pct percentile of Laplacian variance), capped at --max-images.
  * points3D: optional fused cloud (PLY from pcserver / postprocess) used as the initial Gaussians.
"""
import argparse, json, os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from livestereo import LiveStereo, unpack_raw
from stills import upright

ap = argparse.ArgumentParser()
ap.add_argument("rec"); ap.add_argument("out")
ap.add_argument("--ply"); ap.add_argument("--fov", type=float, default=110.0); ap.add_argument("--size", type=int, default=1600)
ap.add_argument("--min-move", type=float, default=0.05); ap.add_argument("--min-rot", type=float, default=5.0)
ap.add_argument("--blur-pct", type=float, default=25.0); ap.add_argument("--max-images", type=int, default=400)
ap.add_argument("--cams", default="left,right"); ap.add_argument("--max-points", type=int, default=300000)
a = ap.parse_args()

ls = LiveStereo(scale=0.5)
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


# 1) blur score on a cheap quarter-res grey image (exposure-normalised)
score = {}
for s, i, m in cand:
    raw = load(s, i).astype(np.float32)
    g = np.clip(raw[::4, ::4] - 64, 0, None) / exposure_factor(m)
    g = g / max(np.percentile(g, 99), 1e-9)
    score[(s, i)] = cv2.Laplacian(np.sqrt(np.clip(g, 0, 1)).astype(np.float32), cv2.CV_32F).var()
thr = np.percentile(list(score.values()), a.blur_pct)

# 2) selection by motion
keep = []; last = {}
for s, i, m in cand:
    if score[(s, i)] < thr: continue
    T = cam_pose(s, m)
    if s in last and np.linalg.norm(T[:3, 3] - last[s][:3, 3]) < a.min_move and rot_deg(T[:3, :3], last[s][:3, :3]) < a.min_rot: continue
    last[s] = T; keep.append((s, i, m, T))
if len(keep) > a.max_images:
    idx = np.linspace(0, len(keep) - 1, a.max_images).round().astype(int); keep = [keep[k] for k in idx]
print(f"{len(keep)} images selected (blur threshold {thr:.2e})")
if not keep: sys.exit("nothing selected: relax --min-move/--min-rot/--blur-pct")

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
        Tc = np.linalg.inv(T)                                   # world-to-camera
        q = qvec(Tc[:3, :3]); t = Tc[:3, 3]
        fi.write(f"{n} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {q[3]:.9f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {cam_id[s]} {name}\n\n")
        if n % 25 == 0: print(f"  {n}/{len(keep)}", flush=True)


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
json.dump(dict(rec=a.rec, images=len(keep), fov=a.fov, size=N, points=npts, wb=wb.tolist(), scale=sc, frame="STAGE (SteamVR), y up, metres"),
          open(os.path.join(a.out, "export.json"), "w"), indent=1)
print(f"wrote {a.out}: {len(keep)} images, {npts} points. Train e.g.:  brush {a.out}")
