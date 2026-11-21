#!/usr/bin/env python3
"""Arcturus recorder video (+ timed metadata) -> rectified stereo evaluation set for tools/stereo_bench.py.

    video_evalset.py VIDEO META.npz CALIB.json OUT [--scale 0.35] [--windows auto|i0:n,i1:n,...] [--nwin 6] [--len 120]

META.npz from tools/avmeta.py, CALIB.json from re/fit_video_geom.py (video principal points + stereo R, t).
Writes OUT/rect/{i:06d}_L.png, _R.png (grey, the reference camera cam0 is always the LEFT image, mirrored if needed) and
OUT/set.json: f, B, flip, cx, cy, size and per frame {i, ts, exposure_ms, gain, blur_px, world_from_rect (4x4)}.
blur_px = f * t_exp * (omega + v / 2 m) from the metadata poses of the neighbouring frames (the pcserver gate's estimate).
auto windows: --nwin windows of --len consecutive frames spread over the blur range (sharpest .. most blurred).
"""
import argparse, json, os, subprocess
import numpy as np, cv2

ap = argparse.ArgumentParser()
ap.add_argument("video"); ap.add_argument("meta"); ap.add_argument("calib"); ap.add_argument("out")
ap.add_argument("--scale", type=float, default=0.35); ap.add_argument("--windows", default="auto")
ap.add_argument("--nwin", type=int, default=6); ap.add_argument("--len", type=int, default=120)
a = ap.parse_args()

m = np.load(a.meta); cal = json.load(open(a.calib))
W0 = m["world_from_cam0"]; ts = m["pts"]; ex = m["exposure_ms"]; gain = m["gain"]; n_all = len(ts)
S0 = np.array(cal["S_meta_from_cv"]["cam0"]); R = np.array(cal["R"]); t = np.array(cal["t"])   # X_cv1 = R X_cv0 + t

FULL = 2464; HALF = FULL // 2; N = int(round(FULL * a.scale))


def KD(c):   # 2x2-area-downsampled (half-res) intrinsics: c_half = (c - 0.5) / 2
    K = np.array([[c["fx"] / 2, 0, (c["cx"] - 0.5) / 2], [0, c["fy"] / 2, (c["cy"] - 0.5) / 2], [0, 0, 1.]])
    return K, np.array(c["D"], float).reshape(4, 1)


K0, D0 = KD(cal["cams"][0]); K1, D1 = KD(cal["cams"][1])
R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(K0, D0, K1, D1, (HALF, HALF), R, t.reshape(3, 1), flags=cv2.CALIB_ZERO_DISPARITY,
                                              newImageSize=(N, N), balance=0.0, fov_scale=1.0)
map0 = cv2.fisheye.initUndistortRectifyMap(K0, D0, R1, P1, (N, N), cv2.CV_16SC2)
map1 = cv2.fisheye.initUndistortRectifyMap(K1, D1, R2, P2, (N, N), cv2.CV_16SC2)
f = float(P1[0, 0]); B = float(np.linalg.norm(t))
if abs(P2[0, 3]) < abs(P2[1, 3]): raise SystemExit("vertical rectification")
flip = bool(P2[0, 3] > 0)          # cam1 lies to the LEFT of cam0 -> mirror both so that cam0 is the left image
print(f"rectified {N}^2, f {f:.1f} px, B {B*1000:.1f} mm, flip {flip}")

# per-frame blur estimate from the metadata poses (central differences)
Rw = W0[:, :3, :3]; P = W0[:, :3, 3]
om = np.zeros(n_all); v = np.zeros(n_all)
for i in range(1, n_all - 1):
    dt = ts[i + 1] - ts[i - 1]
    om[i] = np.linalg.norm(cv2.Rodrigues(Rw[i - 1].T @ Rw[i + 1])[0]) / dt; v[i] = np.linalg.norm(P[i + 1] - P[i - 1]) / dt
blur = f * ex[:, 0] / 1000 * (om + v / 2.0)

if a.windows == "auto":   # windows ordered by their mean blur, picked at evenly spaced blur quantiles
    starts = np.arange(1, n_all - a.len - 1, a.len)
    mb = np.array([blur[s:s + a.len].mean() for s in starts]); order = starts[np.argsort(mb)]
    wins = [(int(order[int(q * (len(order) - 1))]), a.len) for q in np.linspace(0, 1, a.nwin)]
else:
    wins = [tuple(int(x) for x in w.split(":")) for w in a.windows.split(",")]

pts0 = None
os.makedirs(os.path.join(a.out, "rect"), exist_ok=True)
# video packet i <-> metadata record i; seek by the stream's own presentation times
vt = np.array(sorted(float(x) for x in subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
              "packet=pts_time", "-of", "csv=p=0", a.video], capture_output=True, text=True, check=True).stdout.split()))


def decode(stream, i0, n):
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{vt[i0]:.6f}", "-i", a.video, "-map", f"0:v:{stream}", "-frames:v", str(n),
           "-vf", f"scale={HALF}:{HALF}:flags=area,format=gray", "-f", "rawvideo", "-"]
    b = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(b, np.uint8).reshape(-1, HALF, HALF)


frames = []
Rrect = np.eye(4); Rrect[:3, :3] = R1.T            # cam0(cv) <- rect
Smat = np.eye(4); Smat[:3, :3] = S0                 # meta cam0 <- cv cam0
for (i0, n) in wins:
    g0 = decode(0, i0, n); g1 = decode(1, i0, n); n = min(len(g0), len(g1))
    print(f"window {i0}..{i0+n-1}: mean blur {blur[i0:i0+n].mean():.2f} px, exposure {ex[i0:i0+n,0].mean():.1f} ms", flush=True)
    for k in range(n):
        i = i0 + k
        l = cv2.remap(g0[k], *map0, cv2.INTER_LINEAR); r = cv2.remap(g1[k], *map1, cv2.INTER_LINEAR)
        if flip: l, r = l[:, ::-1], r[:, ::-1]
        cv2.imwrite(os.path.join(a.out, "rect", f"{i:06d}_L.png"), l); cv2.imwrite(os.path.join(a.out, "rect", f"{i:06d}_R.png"), r)
        T = W0[i] @ Smat @ Rrect
        frames.append(dict(i=i, win=i0, ts=float(ts[i]), exposure_ms=float(ex[i, 0]), gain=float(gain[i, 0]), blur_px=float(blur[i]),
                           world_from_rect=T.tolist()))
json.dump(dict(f=f, B=B, flip=flip, cx=float(P1[0, 2]), cy=float(P1[1, 2]), size=N, frames=frames), open(os.path.join(a.out, "set.json"), "w"))
print(f"wrote {len(frames)} pairs to {a.out}")
