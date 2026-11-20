#!/usr/bin/env python3
"""Synthetic recording for tools/fid_analyze.py: a cal board on the floor + the id-4 origin marker on a wall, rendered through the
real fisheye calibration into raw10 frames; the 'tracked' head pose carries a known error (slow drift + jitter).
  make_fid_rec.py OUTDIR [--n 24]    -> OUTDIR/frames.jsonl + NNNNN_{left,right}.y10p + truth.json"""
import argparse, json, os, sys, numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import fiducials as F
from livestereo import LiveStereo
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--n", type=int, default=24); a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); rng = np.random.default_rng(3)
ls = LiveStereo(scale=1.0); camL_from_camR = np.eye(4); camL_from_camR[:3, :3] = ls.R.T; camL_from_camR[:3, 3] = -ls.R.T @ ls.t
N = 2464; ys, xs = np.mgrid[0:N, 0:N].astype(np.float64)

def texture(ids, board):
    """big raster of the board plane (white paper + tags): returns img, metres-per-px, origin"""
    b = F.BOARDS[board]; mpp = 0.0005; half = 0.2
    S = int(2 * half / mpp); img = np.full((S, S), 255, np.uint8)
    for i in ids:
        cx, cy = np.array(b["centres_mm"][i]) / 1000; w = int(round(b["tag_mm"] / 1000 / mpp))
        t = cv2.resize(F.render_tag(i, 20), (w, w), interpolation=cv2.INTER_AREA)
        x0 = int((cx + half) / mpp - w / 2); y0 = int((half - cy) / mpp - w / 2); img[y0:y0 + w, x0:x0 + w] = t
    return img, mpp, half
TEX = {"cal": texture([0, 1, 2, 3], "cal"), "origin": texture([4], "origin")}

def render(W_cam, K, D, boards):
    """boards: [(name, stage_from_board)] -> raw RGGB mosaic uint16 (10-bit)"""
    pts = np.stack([xs.ravel(), ys.ravel()], 1).reshape(-1, 1, 2)
    rays = cv2.fisheye.undistortPoints(pts, K, D).reshape(N, N, 2)
    d = np.concatenate([rays, np.ones((N, N, 1))], -1) @ W_cam[:3, :3].T; o = W_cam[:3, 3]
    img = np.full((N, N), 90.0)
    for name, Tb in boards:
        tex, mpp, half = TEX[name]; Rb, tb = Tb[:3, :3], Tb[:3, 3]
        n = Rb[:, 2]; den = d @ n; t = ((tb - o) @ n) / np.where(np.abs(den) < 1e-9, 1e-9, den)
        P = o + d * t[..., None]; loc = (P - tb) @ Rb                      # board coordinates (x right, y up)
        u = (loc[..., 0] + half) / mpp; v = (half - loc[..., 1]) / mpp
        ok = (t > 0) & (u >= 0) & (u < tex.shape[1] - 1) & (v >= 0) & (v < tex.shape[0] - 1)
        s = cv2.remap(tex, u.astype(np.float32), v.astype(np.float32), cv2.INTER_AREA)
        img = np.where(ok, s, img)
    img = cv2.GaussianBlur(img.astype(np.float32), (0, 0), 0.9) + rng.normal(0, 2.0, img.shape)
    return np.clip(64 + img * 3.2, 0, 1023).astype(np.uint16)

def pack(m):
    H, W = m.shape; S = 3088; g = m.reshape(H, W // 4, 4); buf = np.zeros((H, S), np.uint8)
    out = np.zeros((H, W // 4, 5), np.uint8); out[:, :, :4] = (g >> 2).astype(np.uint8)
    out[:, :, 4] = sum(((g[:, :, i] & 3) << (2 * i)) for i in range(4)).astype(np.uint8)
    buf[:, :W * 5 // 4] = out.reshape(H, -1); return buf.tobytes()

def rotz(t):
    c, s = np.cos(t), np.sin(t); return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
# boards: cal on the floor, origin marker on a wall (normal +z of stage = facing the room)
Tfloor = np.eye(4); Tfloor[:3, :3] = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0.]]) ; Tfloor[:3, 3] = [0.0, 0.0, -1.5]       # x=(1,0,0) y=(0,0,-1) z=(0,1,0)
Tfloor[:3, :3] = np.stack([[1, 0, 0], [0, 0, -1], [0, 1, 0]], 1)
Twall = np.eye(4); Twall[:3, 3] = [0.4, 1.3, -3.0]; Twall[:3, :3] = np.diag([1., 1, 1])          # normal +z (faces the camera at z > -3)
truth = dict(Tfloor=Tfloor.tolist(), Twall=Twall.tolist(), drift_mm=[], jitter_mm=0.0)
recs = []
for k in range(a.n):
    t = 1000.0 + k * 1.0
    if k % 2 == 0: target, Tb, dist = Tfloor[:3, 3], Tfloor, rng.uniform(0.6, 1.1)       # look at the floor board
    else: target, Tb, dist = Twall[:3, 3], Twall, rng.uniform(0.5, 0.8)
    nrm = Tb[:3, 2]; ang = np.radians(rng.uniform(0, 35)); az = rng.uniform(0, 2 * np.pi)
    perp = np.cross(nrm, [1, 0, 0] if abs(nrm[0]) < .9 else [0, 1, 0]); perp /= np.linalg.norm(perp)
    pos = target + dist * (np.cos(ang) * nrm + np.sin(ang) * (np.cos(az) * perp + np.sin(az) * np.cross(nrm, perp)))
    z = (target - pos) / np.linalg.norm(target - pos); xr = np.cross(z, np.cross([0, 1, 0], z) if abs(z[1]) < .95 else [1, 0, 0]); xr /= np.linalg.norm(xr)
    yd = np.cross(z, xr); Wl = np.eye(4); Wl[:3, :3] = np.stack([xr, yd, z], 1) @ rotz(rng.uniform(-.5, .5)); Wl[:3, 3] = pos
    H_true = Wl @ np.linalg.inv(ls.head_from_camL)
    drift = np.array([0.0006 * k * 40, -0.0004 * k * 20, 0.0003 * k * 20]) * 1.0       # grows ~1.4 cm / 0.9 cm over the run
    E = np.eye(4); E[:3, :3] = cv2.Rodrigues(np.array([0, 0.0006 * k, 0]))[0]; E[:3, 3] = drift + rng.normal(0, 0.002, 3)
    H_trk = E @ H_true; truth["drift_mm"].append(float(np.linalg.norm(E[:3, 3]) * 1000))
    Wr = Wl @ camL_from_camR
    for side, W, K, D in (("left", Wl, ls.KL, ls.DL), ("right", Wr, ls.KR, ls.DR)):
        open(os.path.join(a.out, f"{k:05d}_{side}.y10p"), "wb").write(pack(render(W, K, D, [("cal", Tfloor), ("origin", Twall)])))
    pose = H_trk[:3, :].tolist(); m = dict(frame=k, ts=t, exp_lines=3000, gain_code=0, dgain=256, pose=pose, pose_valid=True, buf_flags=0)
    recs.append(dict(i=k, left=m, right=dict(m)))
    print(f"\rframe {k + 1}/{a.n}", end="", flush=True)
open(os.path.join(a.out, "frames.jsonl"), "w").write("\n".join(json.dumps(r) for r in recs) + "\n")
truth["H_true_note"] = "tracked = E @ true; E = drift (+jitter) in stage"; json.dump(truth, open(os.path.join(a.out, "truth.json"), "w"))
print("\nwrote", a.out)
