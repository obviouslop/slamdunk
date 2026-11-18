#!/usr/bin/env python3
"""Fisheye stereo depth for one Arcturus recorder frame pair.

Stereo extrinsics come from image features (essential matrix on fisheye-undistorted SIFT
matches), scaled to the metadata baseline (63.98 mm): the recorder's avcam1InAvcam0 rotation
(22°) disagrees with what the video frames show (~31°), see NOTES.md.

usage: stereo_depth.py meta.npz frames_dir FRAME out_prefix [--scale 0.5] [--calib stereo.json]
writes  <out>_depth.png (preview), <out>.ply (coloured points, cam0 OpenCV frame (video intrinsics), metres),
        <out>_stereo.json (R, t used)
"""
import argparse, json, numpy as np, cv2

ap = argparse.ArgumentParser()
ap.add_argument("meta"); ap.add_argument("frames"); ap.add_argument("frame", type=int); ap.add_argument("out")
ap.add_argument("--scale", type=float, default=0.5, help="rectified image scale vs 2464")
ap.add_argument("--calib", help="reuse R,t json instead of estimating")
ap.add_argument("--fov-scale", type=float, default=1.0)
ap.add_argument("--max-depth", type=float, default=8.0)
a = ap.parse_args()

m = np.load(a.meta); kb = json.loads(str(m["kb"]))
K = [np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1.]]) for k in kb]
D = [np.array(k["D"], float).reshape(4, 1) for k in kb]
C0 = cv2.imread(f"{a.frames}/f{a.frame}_cam0.png"); C1 = cv2.imread(f"{a.frames}/f{a.frame}_cam1.png")
G0, G1 = cv2.cvtColor(C0, cv2.COLOR_BGR2GRAY), cv2.cvtColor(C1, cv2.COLOR_BGR2GRAY)
size = G0.shape[::-1]
baseline = float(np.linalg.norm(m["cam0_from_cam1"][a.frame][:3, 3]))

if a.calib:
    c = json.load(open(a.calib)); R = np.array(c["R"]); t = np.array(c["t"])
    if "cams" in c:   # intrinsics fitted to the video frames (re/fit_video_geom.py)
        kb = c["cams"]
        K = [np.array([[k["fx"], 0, k["cx"]], [0, k["fy"], k["cy"]], [0, 0, 1.]]) for k in kb]
        D = [np.array(k["D"], float).reshape(4, 1) for k in kb]
else:
    sift = cv2.SIFT_create(8000)
    k0, d0 = sift.detectAndCompute(G0, None); k1, d1 = sift.detectAndCompute(G1, None)
    good = [p for p, q in cv2.BFMatcher().knnMatch(d0, d1, k=2) if p.distance < 0.75 * q.distance]
    p0 = np.float32([k0[g.queryIdx].pt for g in good]); p1 = np.float32([k1[g.trainIdx].pt for g in good])
    n0 = cv2.fisheye.undistortPoints(p0[:, None], K[0], D[0])[:, 0]; n1 = cv2.fisheye.undistortPoints(p1[:, None], K[1], D[1])[:, 0]
    E, mask = cv2.findEssentialMat(n0, n1, np.eye(3), cv2.RANSAC, 0.999, 1e-3)
    _, R, t, _ = cv2.recoverPose(E, n0, n1, np.eye(3), mask=mask)
    t = t.ravel() / np.linalg.norm(t) * baseline
    print(f"estimated from {int(mask.sum())} inliers: rot {np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0])):.2f}°, t {np.round(t*1000,1)} mm")
json.dump({"R": R.tolist(), "t": np.asarray(t).ravel().tolist(), "frame": a.frame, "baseline_m": baseline}, open(a.out + "_stereo.json", "w"), indent=1)

# rectification (X1 = R X0 + t)
new = (int(2464 * a.scale), int(2464 * a.scale))
R0, R1, P0, P1, Q = cv2.fisheye.stereoRectify(K[0], D[0], K[1], D[1], size, R, np.asarray(t).reshape(3, 1),
                                              flags=cv2.CALIB_ZERO_DISPARITY, newImageSize=new, balance=0.0, fov_scale=a.fov_scale)
maps = [cv2.fisheye.initUndistortRectifyMap(K[i], D[i], Ri, Pi, new, cv2.CV_32FC1) for i, Ri, Pi in ((0, R0, P0), (1, R1, P1))]
r0 = cv2.remap(G0, *maps[0], cv2.INTER_LINEAR); r1 = cv2.remap(G1, *maps[1], cv2.INTER_LINEAR)
rc0 = cv2.remap(C0, *maps[0], cv2.INTER_LINEAR)
horizontal = abs(P1[0, 3]) > abs(P1[1, 3])
if not horizontal:   # vertical baseline -> rotate so SGBM sees horizontal disparity
    raise SystemExit("vertical rectification not handled yet")
f_rect = P0[0, 0]
# cam0 must be the LEFT image for SGBM (disparity = xL - xR > 0). Tx<0 means cam1 is to the right of cam0.
left_is_cam0 = P1[0, 3] < 0
L, Rr = (r0, r1) if left_is_cam0 else (r1, r0)
nd = 16 * int(np.ceil(f_rect * baseline / 0.3 / 16))   # disparities down to ~0.3 m
sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=nd, blockSize=5, P1=8 * 25, P2=32 * 25,
                             uniquenessRatio=10, speckleWindowSize=100, speckleRange=2, disp12MaxDiff=1,
                             mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
disp = sgbm.compute(L, Rr).astype(np.float32) / 16
if not left_is_cam0:
    # disparity is defined on cam1's rectified grid; keep it simple: report, points in cam1-rect frame
    print("note: disparity on cam1 grid")
valid = disp > 0.5
Z = f_rect * baseline / np.maximum(disp, 1e-6)
valid &= Z < a.max_depth
print(f"rectified {new}, f_rect {f_rect:.1f}px, numDisp {nd}, valid {valid.mean()*100:.1f}%, "
      f"depth p5/p50/p95 {np.percentile(Z[valid],[5,50,95]).round(2) if valid.any() else '-'} m, left_is_cam0={left_is_cam0}")

# preview: rectified left | colour depth
dv = np.zeros_like(disp); dv[valid] = np.clip(1.0 / Z[valid], 0, 2) / 2
dep = cv2.applyColorMap((dv * 255).astype(np.uint8), cv2.COLORMAP_TURBO); dep[~valid] = 0
grid = L if left_is_cam0 else r1
cv2.imwrite(a.out + "_depth.png", np.hstack([cv2.cvtColor(grid, cv2.COLOR_GRAY2BGR), dep]))

# 3D points in rectified frame -> camera frame of the left image
h, w = disp.shape
u, v = np.meshgrid(np.arange(w), np.arange(h))
X = (u - P0[0, 2]) * Z / f_rect; Y = (v - P0[1, 2]) * Z / f_rect
pts_rect = np.stack([X, Y, Z], -1)[valid]
Rl = R0 if left_is_cam0 else R1
pts_cam = pts_rect @ Rl          # rect = Rl @ cam  ->  cam = Rl^T rect
if not left_is_cam0:             # points are in cam1's frame -> cam0: X0 = R^T (X1 - t)
    pts_cam = (pts_cam - np.asarray(t).ravel()) @ R
col = (rc0 if left_is_cam0 else cv2.remap(C1, *maps[1], cv2.INTER_LINEAR))[valid][:, ::-1]
with open(a.out + ".ply", "wb") as f:
    f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(pts_cam)}\nproperty float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n".encode())
    rec = np.empty(len(pts_cam), dtype=[("p", "<f4", 3), ("c", "u1", 3)]); rec["p"] = pts_cam; rec["c"] = col
    f.write(rec.tobytes())
print(f"wrote {len(pts_cam)} points -> {a.out}.ply")
