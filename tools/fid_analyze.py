#!/usr/bin/env python3
"""Analyse fiducial boards (tools/fiducials.py) in a recording (pcsession.sh --record): how well do SteamVR's head
poses and the camera calibration agree with markers that do not move?

  fid_analyze.py captures/live17 [--every 2] [--cams LR] [--hmd500 [--delta MS]] [--refit] [--write-corrected]

Per frame the markers are detected on the FULL-resolution grey image (2464^2, demosaiced), the corners are undistorted
with the stock fisheye calibration, each physical board (several copies share ids: ceiling / floor / table, told apart by
their position in STAGE) is solved with PnP, and  stage_from_board = stage_from_cam @ cam_from_board  is collected.

Reports (and CSVs in RECDIR/fid/):
 1. pose error of the tracker: the spread of stage_from_board of an IMMOBILE board = SteamVR pose error + detection noise.
    By distance, head speed, and split into slow drift (> 2 s) and jitter.  [pose_error.csv]
 2. time offset between camera frames and tracker poses (--hmd500: uses the 500 Hz pose log; scans the delay that makes
    the board hold still)
 3. stereo consistency: board pose seen by camL vs camR, against the calibrated extrinsics (baseline / scale error)
 4. undistortion: reprojection residual of the board corners by image radius, per camera [corners.csv]; --refit fits
    new fisheye intrinsics from the boards (written to RECDIR/fid/refit_calib.json, not applied)
 5. --write-corrected: frames_fid.jsonl = frames.jsonl with the poses corrected so the immobile boards stay put
    (every frame in which an anchor board is seen; smoothed over +-1 s in between) - feed it to fuse_rec.py / replay.
"""
import argparse, json, os, sys
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fiducials as F
from livestereo import LiveStereo, unpack_raw

ap = argparse.ArgumentParser(); ap.add_argument("rec")
ap.add_argument("--every", type=int, default=1); ap.add_argument("--cams", default="LR")
ap.add_argument("--hmd500", action="store_true", help="use the 500 Hz pose log at the exposure centre instead of the latched pose")
ap.add_argument("--delta", type=float, default=0.0, help="time offset (ms) added when sampling the 500 Hz log")
ap.add_argument("--refit", action="store_true"); ap.add_argument("--write-corrected", action="store_true")
ap.add_argument("--anchors", default="all", help="boards used for the correction: all | origin | cal")
ap.add_argument("--cluster", type=float, default=0.35, help="m: detections of one physical board lie within this radius in STAGE")
ap.add_argument("--origin-mm", type=float, help="edge of the id 4 marker (default tools/fiducials.ORIGIN_MM)")
a = ap.parse_args()
if a.origin_mm: F.BOARDS["origin"]["tag_mm"] = a.origin_mm
out = os.path.join(a.rec, "fid"); os.makedirs(out, exist_ok=True)
ls = LiveStereo(scale=1.0)
KL, DL, KR, DR = ls.KL, ls.DL, ls.KR, ls.DR
camL_from_camR = np.eye(4); camL_from_camR[:3, :3] = ls.R.T; camL_from_camR[:3, 3] = -ls.R.T @ ls.t
LT = 4.36e-6                                              # sensor line time (s)
recs = [json.loads(l) for l in open(os.path.join(a.rec, "frames.jsonl"))]


def gray8(raw):
    m = np.clip(raw.astype(np.float32) - 64, 0, None)
    g = cv2.cvtColor(np.clip(m, 0, 65535).astype(np.uint16), cv2.COLOR_BayerBG2GRAY).astype(np.float32)
    return np.clip(g * (255 / max(np.percentile(g[::4, ::4], 99.7), 1)), 0, 255).astype(np.uint8)


def head_pose_at(r, side, t=None):
    H = np.eye(4); H[:3, :] = np.asarray(r[side]["pose"]); return H


hmd = None
if a.hmd500:
    from scipy.spatial.transform import Rotation as Rot, Slerp
    Hh = np.loadtxt(os.path.join(a.rec, "hmd_500hz.txt")); Hh = Hh[Hh[:, 1] > 0]
    slerp = Slerp(Hh[:, 0], Rot.from_quat(Hh[:, [3, 4, 5, 2]]))
    def pose500(m, delta_ms):
        t = np.clip(m["ts"] - 2464 * LT / 2 - m["exp_lines"] * LT / 2 + delta_ms * 1e-3, Hh[0, 0], Hh[-1, 0])
        H = np.eye(4); H[:3, :3] = slerp([t]).as_matrix()[0]; H[:3, 3] = [np.interp(t, Hh[:, 0], Hh[:, 6 + j]) for j in range(3)]; return H
    hmd = pose500


def stage_from_cam(H, cam):
    W = H @ ls.head_from_camL
    return W if cam == "L" else W @ camL_from_camR


def undist(cam):
    K, D = (KL, DL) if cam == "L" else (KR, DR)
    return lambda c: cv2.fisheye.undistortPoints(np.asarray(c, np.float64).reshape(-1, 1, 2), K, D).reshape(-1, 2)


# ------------------------------------------------------------------ 1. detection pass
rows = []                       # one per (frame, camera, board group)
corners_log = []                # per corner residuals
cache = os.path.join(out, "detections.npz")
t_start = __import__("time").time()
sel = [r for r in recs if r["left"]["pose_valid"]][::a.every]
for n, r in enumerate(sel):
    i = r["i"]
    for cam in a.cams:
        side = "left" if cam == "L" else "right"
        g = gray8(unpack_raw(open(os.path.join(a.rec, f"{i:05d}_{side}.y10p"), "rb").read()))
        dets = F.detect(g)
        if not dets: continue
        xn_of = undist(cam); H0 = head_pose_at(r, side); Wc = stage_from_cam(H0, cam)
        for grp in F.group_boards(dets, Wc, xn_of, radius=a.cluster):
            b = F.board_of(grp[0]["id"]); O = F.board_object(b, [d["id"] for d in grp])
            obj = np.concatenate([O[d["id"]] for d in grp]); xn = np.concatenate([xn_of(d["corners"]) for d in grp])
            cands = F.solve_pose_all(obj, xn)
            if not cands: continue
            rows.append(dict(i=i, cam=cam, ts=r[side]["ts"], exp=r[side]["exp_lines"], board=b, ids=[d["id"] for d in grp],
                             cands=cands, edge=float(np.mean([d["edge_px"] for d in grp])), ntag=len(grp),
                             amb=(cands[1][1] / max(cands[0][1], 1e-12)) if len(cands) > 1 else np.inf,
                             obj=obj, px=np.concatenate([d["corners"] for d in grp]), xn=xn, r=r))
    print(f"\r{n + 1}/{len(sel)} frames, {len(rows)} board detections", end="", flush=True)
print(f"\n{len(rows)} board detections in {__import__('time').time() - t_start:.0f} s")
if not rows: sys.exit("no markers found")


def pose_row(row, delta_ms=None):
    r = row["r"]; side = "left" if row["cam"] == "L" else "right"
    H = hmd(r[side], a.delta if delta_ms is None else delta_ms) if hmd else head_pose_at(r, side)
    return stage_from_cam(H, row["cam"])


# ------------------------------------------------------------------ 2. physical boards (clusters in STAGE)
def stage_pos(row, cand=0): return (pose_row(row) @ row["cands"][cand][0])[:3, 3]
inst = []                       # [(board, centre)]
for row in rows:
    p = stage_pos(row)
    for k, (b, c, cnt) in enumerate(inst):
        if b == row["board"] and np.linalg.norm(c - p) < a.cluster:
            row["inst"] = k; inst[k] = (b, (c * cnt + p) / (cnt + 1), cnt + 1); break
    else:
        row["inst"] = len(inst); inst.append((row["board"], p, 1))
names = {k: f"{b}#{k}@({c[0]:+.2f},{c[1]:+.2f},{c[2]:+.2f})" for k, (b, c, n) in enumerate(inst)}
print("physical boards:", ", ".join(f"{names[k]} n={inst[k][2]}" for k in names))


def reliable(row): return row["ntag"] >= 2 or (row["amb"] > 1.6 and row["cands"][0][1] * KL[0, 0] < 0.6)   # single tag: unambiguous and well-refined


# reference pose of each board = robust mean of its reliable detections; then resolve single-tag flips with it
ref = {}
for k in names:
    Ts = [pose_row(r) @ r["cands"][0][0] for r in rows if r["inst"] == k and reliable(r)]
    if len(Ts) < 3: Ts = [pose_row(r) @ r["cands"][0][0] for r in rows if r["inst"] == k]
    if not Ts: continue
    med = np.median([T[:3, 3] for T in Ts], 0)
    ok = [T for T in Ts if np.linalg.norm(T[:3, 3] - med) < 0.15] or Ts
    ref[k] = F.mean_pose(ok)
for row in rows:
    if row["inst"] in ref and row["ntag"] == 1 and len(row["cands"]) > 1:
        Wc = pose_row(row); prior = np.linalg.inv(Wc) @ ref[row["inst"]]
        row["cands"] = sorted(row["cands"], key=lambda c: F.pose_diff(prior, c[0])[1])
for row in rows:
    row["T_cam_board"], row["rms"] = row["cands"][0]
    row["Tsb"] = pose_row(row) @ row["T_cam_board"]
    row["dist"] = float(np.linalg.norm(row["T_cam_board"][:3, 3])); row["angle"] = F.view_angle_deg(row["T_cam_board"])


def dev(row, k=None):
    k = row["inst"] if k is None else k
    dp, da = F.pose_diff(ref[k], row["Tsb"]); return np.linalg.norm(ref[k][:3, :3] @ dp), da


# ------------------------------------------------------------------ report 1: tracker pose error
print("\n=== 1. stage pose of immobile boards (deviation from the board's own median pose) ===")
csv = ["i,cam,inst,ts,dist_m,angle_deg,ntag,rms_px,dev_mm,dev_deg,x,y,z"]
for row in rows:
    dp, da = dev(row)
    csv.append(f"{row['i']},{row['cam']},{row['inst']},{row['ts']:.4f},{row['dist']:.3f},{row['angle']:.1f},{row['ntag']},"
               f"{row['rms'] * KL[0, 0]:.3f},{dp * 1000:.2f},{da:.3f},{row['Tsb'][0, 3]:.4f},{row['Tsb'][1, 3]:.4f},{row['Tsb'][2, 3]:.4f}")
open(os.path.join(out, "pose_error.csv"), "w").write("\n".join(csv) + "\n")
for k in names:
    R = [r for r in rows if r["inst"] == k and (r["ntag"] >= 2 or reliable(r))]
    if len(R) < 5: print(f"{names[k]}: only {len(R)} reliable detections"); continue
    d = np.array([dev(r) for r in R]); dist = np.array([r["dist"] for r in R]); ts = np.array([r["ts"] for r in R])
    P = np.array([r["Tsb"][:3, 3] for r in R]); o = np.argsort(ts); ts, P, d, dist = ts[o], P[o], d[o], dist[o]
    sm = np.array([np.median(P[np.abs(ts - t) < 1.0], 0) for t in ts])                 # +-1 s running median = slow part
    jit = np.linalg.norm(P - sm, axis=1); drift = np.linalg.norm(sm - np.median(P, 0), axis=1)
    print(f"\n{names[k]}  ({len(R)} detections, {'4-tag board' if inst[k][0] == 'cal' else 'single marker'})")
    print(f"  position dev: median {np.median(d[:, 0]) * 1000:.1f} mm  p90 {np.percentile(d[:, 0], 90) * 1000:.1f} mm  max {d[:, 0].max() * 1000:.0f} mm;"
          f" rotation dev median {np.median(d[:, 1]):.2f} deg p90 {np.percentile(d[:, 1], 90):.2f}")
    print(f"  slow drift (|running median - overall|) p90 {np.percentile(drift, 90) * 1000:.1f} mm, max {drift.max() * 1000:.0f} mm; fast jitter rms {np.sqrt(np.mean(jit ** 2)) * 1000:.1f} mm")
    for lo, hi in ((0, .5), (.5, 1), (1, 1.5), (1.5, 2.5), (2.5, 9)):
        m = (dist >= lo) & (dist < hi)
        if m.sum() >= 3: print(f"    distance {lo:.1f}-{hi:.1f} m: n={m.sum():4d}  dev median {np.median(d[m, 0]) * 1000:5.1f} mm  {np.median(d[m, 1]):5.2f} deg")
    # head speed
    sp = []
    for r in R:
        H = np.eye(4); H[:3, :] = np.asarray(r["r"]["left"]["pose"]); sp.append(H[:3, 3])
    sp = np.array(sp)[o]; v = np.r_[0, np.linalg.norm(np.diff(sp, axis=0), axis=1) / np.maximum(np.diff(ts), 1e-3)]
    for lo, hi in ((0, .1), (.1, .3), (.3, .8), (.8, 9)):
        m = (v >= lo) & (v < hi)
        if m.sum() >= 3: print(f"    head speed {lo:.1f}-{hi:.1f} m/s: n={m.sum():4d}  dev median {np.median(d[m, 0]) * 1000:5.1f} mm")

# ------------------------------------------------------------------ report 2: time offset
if hmd:
    print("\n=== 2. camera <-> pose time offset (spread of the immobile boards' stage position vs delay) ===")
    best = None
    for dms in range(-80, 81, 10):
        tot = []
        for k in names:
            R = [r for r in rows if r["inst"] == k and reliable(r)]
            if len(R) < 5: continue
            P = np.array([(pose_row(r, dms) @ r["T_cam_board"])[:3, 3] for r in R]); tot.append(np.sqrt(np.mean(np.sum((P - np.median(P, 0)) ** 2, 1))))
        if tot:
            s = float(np.mean(tot)); best = (s, dms) if best is None or s < best[0] else best
            print(f"  delay {dms:+4d} ms: rms position spread {s * 1000:.1f} mm")
    if best: print(f"  -> best delay {best[1]:+d} ms (use --delta {best[1]})")

# ------------------------------------------------------------------ report 3: stereo consistency
print("\n=== 3. stereo: board seen by camL vs camR, against the calibrated extrinsics ===")
byf = {}
for r in rows: byf.setdefault((r["i"], r["inst"]), {})[r["cam"]] = r
E = []
for (i, k), d in byf.items():
    if "L" in d and "R" in d and reliable(d["L"]) and reliable(d["R"]):
        pred = np.linalg.inv(camL_from_camR) @ d["L"]["T_cam_board"]            # camR <- board, predicted from camL
        dt = d["R"]["T_cam_board"][:3, 3] - pred[:3, 3]
        E.append((d["L"]["dist"], dt[0], dt[1], dt[2], F.pose_diff(pred, d["R"]["T_cam_board"])[1]))
if E:
    E = np.array(E)
    for lo, hi in ((0, .6), (.6, 1.0), (1.0, 1.5), (1.5, 9)):
        m = (E[:, 0] >= lo) & (E[:, 0] < hi)
        if m.sum() >= 3:
            print(f"  distance {lo:.1f}-{hi:.1f} m: n={m.sum():3d}  R-minus-predicted position (camR axes) median "
                  f"x {np.median(E[m, 1]) * 1000:+.1f} y {np.median(E[m, 2]) * 1000:+.1f} z {np.median(E[m, 3]) * 1000:+.1f} mm "
                  f"({np.median(E[m, 3] / E[m, 0]) * 100:+.2f} % of range), rot {np.median(E[m, 4]):.2f} deg")
else:
    print("  (no frames where both cameras see a board)")

# ------------------------------------------------------------------ report 4: undistortion
print("\n=== 4. undistortion: board corner reprojection residual by image radius (px at full resolution) ===")
lines = ["cam,theta_deg,u,v,res_radial_px,res_tangential_px,dist_m,angle_deg"]
for cam, K, D in (("L", KL, DL), ("R", KR, DR)):
    res = []
    for r in rows:
        if r["cam"] != cam or r["ntag"] < 2 and r["amb"] < 3: continue
        T = r["T_cam_board"]; P = (T[:3, :3] @ r["obj"].T + T[:3, 3:4]).T
        proj, _ = cv2.fisheye.projectPoints(P.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, D)
        e = r["px"] - proj.reshape(-1, 2)                                        # detected - model-projected (distorted image px)
        c = np.array([K[0, 2], K[1, 2]]); rad = r["px"] - c; rn = np.linalg.norm(rad, axis=1); u = rad / np.maximum(rn, 1e-9)[:, None]
        th = np.degrees(np.arctan2(rn, K[0, 0]))
        for j in range(len(e)):
            er, et = float(e[j] @ u[j]), float(e[j, 0] * -u[j, 1] + e[j, 1] * u[j, 0])
            res.append((th[j], er, et)); lines.append(f"{cam},{th[j]:.1f},{r['px'][j, 0]:.1f},{r['px'][j, 1]:.1f},{er:.3f},{et:.3f},{r['dist']:.3f},{r['angle']:.1f}")
    if res:
        res = np.array(res)
        print(f"  cam{cam}: {len(res)} corners, rms {np.sqrt(np.mean(res[:, 1:] ** 2)):.2f} px")
        for lo, hi in ((0, 15), (15, 30), (30, 45), (45, 60), (60, 90)):
            m = (res[:, 0] >= lo) & (res[:, 0] < hi)
            if m.sum() >= 8:
                print(f"    {lo:2d}-{hi:2d} deg from axis: n={m.sum():5d}  radial {np.median(res[m, 1]):+.2f} px (rms {np.sqrt(np.mean(res[m, 1] ** 2)):.2f})  tangential rms {np.sqrt(np.mean(res[m, 2] ** 2)):.2f}")
open(os.path.join(out, "corners.csv"), "w").write("\n".join(lines) + "\n")

if a.refit:
    cal = {}
    for cam, K, D in (("L", KL, DL), ("R", KR, DR)):
        V = [r for r in rows if r["cam"] == cam and r["ntag"] >= 2]
        if len(V) < 15: print(f"refit cam{cam}: only {len(V)} multi-tag views (need >= 15, from varied angles)"); continue
        objp = [r["obj"].reshape(-1, 1, 3).astype(np.float64) for r in V]; imgp = [r["px"].reshape(-1, 1, 2).astype(np.float64) for r in V]
        K2, D2 = K.copy(), D.copy().reshape(4, 1)
        rms, K2, D2, _, _ = cv2.fisheye.calibrate(objp, imgp, (2464, 2464), K2, D2, flags=cv2.fisheye.CALIB_USE_INTRINSIC_GUESS | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
                                                  | cv2.fisheye.CALIB_FIX_SKEW, criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-9))
        cal[cam] = dict(rms_px=float(rms), K=K2.tolist(), D=D2.ravel().tolist(), K_old=K.tolist(), D_old=D.ravel().tolist(), views=len(V))
        print(f"refit cam{cam}: {len(V)} views, rms {rms:.3f} px; f {K[0, 0]:.1f}->{K2[0, 0]:.1f}, c ({K[0, 2]:.1f},{K[1, 2]:.1f})->({K2[0, 2]:.1f},{K2[1, 2]:.1f}), "
              f"D {np.round(D.ravel(), 4).tolist()} -> {np.round(D2.ravel(), 4).tolist()}")
    json.dump(cal, open(os.path.join(out, "refit_calib.json"), "w"), indent=1)

# ------------------------------------------------------------------ 5. corrected poses
if a.write_corrected:
    use = [k for k in names if (a.anchors == "all" or inst[k][0] == a.anchors) and k in ref]
    C = {}
    for row in rows:
        if row["inst"] not in use or not reliable(row): continue
        Wc = pose_row(row)
        Corr = ref[row["inst"]] @ np.linalg.inv(row["T_cam_board"]) @ np.linalg.inv(Wc)
        C.setdefault(row["i"], []).append((Corr, (row["edge"] ** 2) * (4 if row["ntag"] > 1 else 1)))
    idx = np.array(sorted(C)); Cf = {i: F.mean_pose([c for c, w in C[i]], [w for c, w in C[i]]) for i in idx}
    ts_of = {r["i"]: r["left"]["ts"] for r in recs}
    tt = np.array([ts_of[i] for i in idx])
    fn = os.path.join(a.rec, "frames_fid.jsonl"); nfix = 0
    with open(fn, "w") as f:
        for r in recs:
            t = r["left"]["ts"]; near = idx[np.abs(tt - t) < 1.0]
            if len(near):
                Cs = F.mean_pose([Cf[i] for i in near]); nfix += 1
                for side in ("left", "right"):
                    H = np.eye(4); H[:3, :] = np.asarray(r[side]["pose"]); r[side]["pose_tracked"] = r[side]["pose"]; r[side]["pose"] = (Cs @ H)[:3, :].tolist()
                r["fid_corrected"] = True
            f.write(json.dumps(r) + "\n")
    print(f"\n=== 5. corrected poses: {nfix}/{len(recs)} frames within 1 s of an anchor detection -> {fn}")
    cs = np.array([F.pose_diff(np.eye(4), Cf[i]) for i in idx], dtype=object)
    mag = np.array([np.linalg.norm(x[0]) for x in cs]); ang = np.array([x[1] for x in cs])
    print(f"    correction applied: translation median {np.median(mag) * 1000:.1f} mm, max {mag.max() * 1000:.0f} mm; rotation median {np.median(ang):.2f} deg, max {ang.max():.2f}")
print(f"\nCSV: {out}/pose_error.csv, corners.csv")
