#!/usr/bin/env python3
"""Draw the tracked controllers (SteamVR render model at the logged pose) into the recorded Arcturus images.

Checks how well the camera calibration + HMD pose + controller tracking agree: if everything is consistent, the
projected model outline sits exactly on the controller in the image.

Needs a recording made with  tools/pcsession.sh --record DIR  (arcgrab --pose-log writes hmd_500hz.txt and, since
2026-11-18, hmd_500hz.txt.controllers: t role valid qw qx qy qz px py pz, role 1 = left, 2 = right, STAGE frame).

usage: ctrl_overlay.py RECDIR [--frames 10,20,30 | --every 5] [--out RECDIR/ctrl_overlay] [--model-dir DIR]
Also compares the stereo depth on the controller pixels with the tracked model depth (per frame + summary by
distance, depth_check.csv) - a near-range depth test with a known object.
Writes per frame NNNNN.jpg: rectified left | right image with the model points (left = cyan, right = orange), and
prints the controller distance and the pixel position of the controller origin.
"""
import argparse, json, os, sys
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from livestereo import LiveStereo
from tsdf import bin_raw10

MODELS = "/opt/steamvr/drivers/frame_controller/resources/rendermodels"


def load_obj(fn, step=1):
    V = [list(map(float, l.split()[1:4])) for l in open(fn) if l.startswith("v ")]
    return np.asarray(V[::step], np.float64)


def quat_mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def load_controllers(fn):
    """-> {role: (t (N,), valid (N,), q (N,4), p (N,3))}"""
    a = np.loadtxt(fn, ndmin=2)
    out = {}
    for role in (1, 2):
        b = a[a[:, 1] == role]
        if len(b):
            q = b[:, 3:7].copy()
            for i in range(1, len(q)):           # hemisphere continuity for interpolation
                if np.dot(q[i], q[i - 1]) < 0: q[i] = -q[i]
            out[role] = (b[:, 0], b[:, 2] > 0, q, b[:, 7:10])
    return out


def pose_at(track, t):
    """4x4 STAGE-from-device at time t (linear position, normalised-lerp rotation), None outside / invalid."""
    T_, V, Q, P = track
    i = np.searchsorted(T_, t)
    if i <= 0 or i >= len(T_) or T_[i] - T_[i - 1] > 0.05 or not (V[i] and V[i - 1]):
        return None
    u = (t - T_[i - 1]) / (T_[i] - T_[i - 1])
    M = np.eye(4); M[:3, :3] = quat_mat((1 - u) * Q[i - 1] + u * Q[i]); M[:3, 3] = (1 - u) * P[i - 1] + u * P[i]
    return M


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("rec"); ap.add_argument("--frames"); ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--out"); ap.add_argument("--model-dir", default=MODELS); ap.add_argument("--scale", type=float, default=0.35)
    a = ap.parse_args()
    out = a.out or os.path.join(a.rec, "ctrl_overlay"); os.makedirs(out, exist_ok=True)
    cfile = os.path.join(a.rec, "hmd_500hz.txt.controllers")
    if not os.path.exists(cfile):
        sys.exit(f"{cfile} missing: record with the new arcgrab (pcsession.sh --record)")
    tracks = load_controllers(cfile)
    models = {1: load_obj(os.path.join(a.model_dir, "frame_controller_left", "frame_controller_left.obj"), 3),
              2: load_obj(os.path.join(a.model_dir, "frame_controller_right", "frame_controller_right.obj"), 3)}
    ls = LiveStereo(scale=a.scale)
    recs = [json.loads(l) for l in open(os.path.join(a.rec, "frames.jsonl"))]
    idx = [int(x) for x in a.frames.split(",")] if a.frames else list(range(0, len(recs), a.every))
    # rectified right camera: rect_R = R2 @ camR, camR = R @ camL + t
    camL_from_camR = np.eye(4); camL_from_camR[:3, :3] = ls.R.T; camL_from_camR[:3, 3] = -ls.R.T @ ls.t
    stats = []
    for k in idx:
        r = recs[k]; i = r["i"]
        bL = open(os.path.join(a.rec, f"{i:05d}_left.y10p"), "rb").read(); bR = open(os.path.join(a.rec, f"{i:05d}_right.y10p"), "rb").read()
        cL, cR = bin_raw10(bL), bin_raw10(bR)
        Zs, _, _ = ls.depth(cL, cR, far=4.0, half=True)                # stereo depth (live pipeline model) in rect left
        imgs = []
        for side, c in (("left", cL), ("right", cR)):
            mp = ls.mapL if side == "left" else ls.mapR
            g = cv2.remap(c, *mp, cv2.INTER_LINEAR)
            g = np.clip(g / max(np.percentile(g, 99.5), 1), 0, 1) ** (1 / 2.2) * 255
            imgs.append(np.ascontiguousarray(g.astype(np.uint8)))
        W = ls.world_from_rect(r["left"]["pose"])                     # STAGE <- rectified left
        H = np.eye(4); H[:3, :] = np.asarray(r["left"]["pose"])
        Wr = H @ ls.head_from_camL @ camL_from_camR; Rr = np.eye(4); Rr[:3, :3] = ls.R2.T; Wr = Wr @ Rr   # STAGE <- rect right
        line = [f"frame {i} (cam {r['left']['frame']})"]
        for role, col in ((1, (255, 255, 0)), (2, (0, 160, 255))):
            if role not in tracks: continue
            M = pose_at(tracks[role], r["left"]["ts"])
            if M is None: line.append(f"{'LR'[role-1]}: no pose"); continue
            Xw = M[:3, :3] @ models[role].T + M[:3, 3:4]
            for side, img, Wx, P in (("L", imgs[0], W, ls.P1), ("R", imgs[1], Wr, ls.P2)):
                Xc = np.linalg.inv(Wx)[:3, :3] @ Xw + np.linalg.inv(Wx)[:3, 3:4]
                ok = Xc[2] > 0.05
                u = P[0, 0] * Xc[0, ok] / Xc[2, ok] + P[0, 2]; v = P[1, 1] * Xc[1, ok] / Xc[2, ok] + P[1, 2]
                m = np.zeros(img.shape[:2], np.uint8)
                inb = (u >= 0) & (u < img.shape[1]) & (v >= 0) & (v < img.shape[0])
                m[v[inb].astype(int), u[inb].astype(int)] = 255
                if side == "L":            # depth check: model z-buffer vs stereo depth on the controller pixels
                    zb = np.full(img.shape[:2], np.inf, np.float32)
                    np.minimum.at(zb, (v[inb].astype(int), u[inb].astype(int)), Xc[2, ok][inb].astype(np.float32))
                    zb = cv2.erode(zb, np.ones((3, 3), np.uint8))            # fill gaps between model points (front-most)
                    sel = np.isfinite(zb) & (Zs > 0)
                    dz = Zs[sel] - zb[sel]
                    near = np.abs(dz) < 0.25                                # stereo pixels that see the controller
                    if near.sum() > 30:
                        zm = float(np.median(zb[sel][near])); d = float(np.median(dz[near]))
                        stats.append((i, role, zm, d, int(near.sum())))
                        line.append(f"{'LR'[role-1]} depth: model {zm:.3f} m, stereo {d*100:+.1f} cm ({d/zm*100:+.1f} %, {near.sum()} px)")
                m = cv2.morphologyEx(cv2.dilate(m, np.ones((3, 3), np.uint8)), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
                img[m > 0] = (0.65 * img[m > 0] + 0.35 * np.array(col)).astype(np.uint8)       # translucent silhouette
                cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(img, cs, -1, col, 1)                                            # crisp outline
            o = np.linalg.inv(W) @ np.r_[M[:3, 3], 1]
            line.append(f"{'LR'[role-1]}: {np.linalg.norm(o[:3]):.2f} m at px ({ls.P1[0,0]*o[0]/o[2]+ls.P1[0,2]:.0f},{ls.P1[1,1]*o[1]/o[2]+ls.P1[1,2]:.0f})")
        cv2.imwrite(os.path.join(out, f"{i:05d}.jpg"), np.hstack(imgs))
        print("  ".join(line))
    if stats:
        S = np.array(stats, float)
        np.savetxt(os.path.join(out, "depth_check.csv"), S, fmt="%.4f", header="frame role model_z_m stereo_minus_model_m n_px")
        print("\nstereo - tracked depth of the controllers, by distance:")
        for lo, hi in ((0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.2), (1.2, 3.0)):
            m = (S[:, 2] >= lo) & (S[:, 2] < hi)
            if m.sum():
                print(f"  {lo:.1f}-{hi:.1f} m: {m.sum():3d} frames  median {np.median(S[m, 3])*100:+.1f} cm  ({np.median(S[m, 3] / S[m, 2])*100:+.1f} %)")


if __name__ == "__main__":
    main()
