#!/usr/bin/env python3
"""Synthetic test of tools/fiducials.py: render the 4-marker board (and the origin marker) into a pinhole camera image
at known poses (blur + noise), detect, solve, compare. Also checks the Anchor correction."""
import os, sys, numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import fiducials as F
rng = np.random.default_rng(1)
W = 2464; f = 1200.0; K = np.array([[f, 0, W / 2], [0, f, W / 2], [0, 0, 1.]])

def render(T_cam_board, board, ids, noise=3.0, margin_mm=8):
    img = np.full((W, W), 150, np.uint8)
    b = F.BOARDS[board]; s = b["tag_mm"] / 1000
    for i in ids:
        cx, cy = np.array(b["centres_mm"][i]) / 1000
        # 9x9 tag image (with a white margin) -> board plane: image px (u right, v down) -> board (x right, y up)
        m = 1.0 + margin_mm / b["tag_mm"] * 0   # (no margin: the real print has paper around the sheet)
        pad = 3; tag = cv2.copyMakeBorder(F.render_tag(i, 30), pad * 30, pad * 30, pad * 30, pad * 30, cv2.BORDER_CONSTANT, value=255)
        n = tag.shape[0]; half = (9 + 2 * pad) / 9 * s / 2
        src = np.float32([[0, 0], [n, 0], [n, n], [0, n]])
        obj = np.array([[cx - half, cy + half, 0], [cx + half, cy + half, 0], [cx + half, cy - half, 0], [cx - half, cy - half, 0]])
        P = (T_cam_board[:3, :3] @ obj.T + T_cam_board[:3, 3:4]).T; uv = (K @ P.T).T; uv = (uv[:, :2] / uv[:, 2:]).astype(np.float32)
        Hm = cv2.getPerspectiveTransform(src, uv)
        w = cv2.warpPerspective(tag, Hm, (W, W), flags=cv2.INTER_AREA); msk = cv2.warpPerspective(np.full_like(tag, 255), Hm, (W, W))
        img = np.where(msk > 0, w, img)
    img = cv2.GaussianBlur(img, (0, 0), 0.8).astype(np.float32) + rng.normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)

def pose(rx, ry, rz, t):
    """camera <- board; the board faces the camera when the rotation vector is 0 (board z points at the viewer = -z_cam,
    board y up = -y_cam)"""
    T = np.eye(4); T[:3, :3] = cv2.Rodrigues(np.array([rx, ry, rz], float))[0] @ np.diag([1., -1, -1]); T[:3, 3] = t; return T

def xn_of(c): return (np.asarray(c, float) - K[:2, 2]) / f

fails = 0
for name, T in [("head-on 0.6 m", pose(0, 0, 0, [0, 0, .6])), ("tilted 40 deg 0.6 m", pose(.7, .2, .1, [.05, -.03, .6])),
                ("0.5 m rotated 90 deg", pose(0, 0, 1.57, [.02, .02, .5])), ("1.0 m", pose(.3, -.4, 0, [-.1, .05, 1.0]))]:
    img = render(T, "cal", [0, 1, 2, 3])
    dets = F.detect(img)
    ids = sorted(d["id"] for d in dets)
    Tb, rms = F.solve_board(dets, xn_of) if dets else (None, 9)
    if Tb is None or ids != [0, 1, 2, 3]:
        print(f"FAIL {name}: ids {ids}"); fails += 1; continue
    dt, da = F.pose_diff(T, Tb)
    ok = np.linalg.norm(dt) < 0.004 and da < 0.8
    print(f"{'ok  ' if ok else 'FAIL'} {name}: ids {ids}, pos err {np.linalg.norm(dt)*1000:.2f} mm, rot err {da:.2f} deg, rms {rms*f:.2f} px")
    fails += not ok
# single origin marker (id 4), 0.7 m, 3 mm accuracy expected from one 50 mm tag
img = render(pose(.3, .2, 0, [0.02, 0, .7]), "origin", [4]); dets = F.detect(img)
print("origin marker detected:", [d["id"] for d in dets])
if [d["id"] for d in dets] != [4]: fails += 1
else:
    Tt = pose(.3, .2, 0, [0.02, 0, .7]); obj = F.board_object("origin", [4])[4]; xn = xn_of(dets[0]["corners"])
    cand = F.solve_pose_all(obj, xn); print("      single tag: candidates (rot err deg, rms px):", [(round(F.pose_diff(Tt, c[0])[1], 1), round(c[1] * f, 2)) for c in cand],
                                              f"ambiguity {F.ambiguity(obj, xn):.2f}")
    Tb, _ = F.solve_pose(obj, xn, prior=np.block([[cv2.Rodrigues(np.array([.05, 0, 0]))[0] @ Tt[:3, :3], Tt[:3, 3:]], [np.zeros((1, 3)), np.ones((1, 1))]]))          # with a rough prior the flip is resolved
    dt, da = F.pose_diff(Tt, Tb); print(f"      origin pose with prior: {np.linalg.norm(dt)*1000:.1f} mm, {da:.1f} deg"); fails += (da > 10)
# no false positives on a noise / checker image
bad = rng.integers(0, 255, (W, W), np.uint8); bad = cv2.GaussianBlur(bad, (0, 0), 2)
print("false detections on noise:", len(F.detect(bad))); fails += len(F.detect(bad)) > 0
# anchor: a drifting tracker (true camera moves, tracked pose has +3 cm / 1 deg error) is corrected
an = F.Anchor(n_ref=5); Tm = pose(0.1, 0.2, 0.3, [0.5, 1.0, -2.0])   # (any 4x4 will do here)           # marker in the stage
for k in range(8):
    Tsc = pose(0.1 * k, 0, 0, [0, 1.5, 0]); Tcm = np.linalg.inv(Tsc) @ Tm
    an.add(Tsc @ Tcm)
Tsc = pose(0.3, 0.1, 0, [0.2, 1.4, 0.1]); Tcm = np.linalg.inv(Tsc) @ Tm
bias = pose(0, 0, 0.017, [0.03, 0, 0]); Tsc_tracked = bias @ Tsc                 # what SteamVR reports
C = an.correction(Tsc_tracked, Tcm); dt, da = F.pose_diff(Tsc, C @ Tsc_tracked)
print(f"anchor correction residual {np.linalg.norm(dt)*1000:.3f} mm, {da:.3f} deg"); fails += np.linalg.norm(dt) > 1e-6
print("FAILS:", fails); sys.exit(1 if fails else 0)
