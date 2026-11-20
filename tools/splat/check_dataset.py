#!/usr/bin/env python3
"""Sanity-check a COLMAP text dataset made by tools/export_colmap.py before training.

usage: python3 -I check_dataset.py DATASET [--max-centre 20] [--points 60000] [--worst 5]

Checks
  * cameras.txt / images.txt / points3D.txt parse; every listed image exists and has the camera's size
  * quaternions are unit (|q|-1 < 1e-3), camera centres (-R^T t) are within --max-centre metres of the median centre
  * projects points3D into every image: fraction in front of the camera and inside the image
  * colour consistency: colour of the projected point vs the pixel it lands on (pixels outside the fisheye
    circle, exactly 0,0,0, are skipped). Reported as Pearson r of luminance and mean abs RGB difference, next to a
    control with the point colours shuffled (r ~ 0). No occlusion test, so r is modest even when poses are right;
    a wrong convention (e.g. camera-to-world, flipped axes) drops it to the control level. The same score is
    computed for a few alternative conventions so a failure points at the cause.
Exit code 1 on a hard failure (missing image, non-unit quaternion, far centres, r not clearly above control).
"""
import argparse, os, sys
import numpy as np, cv2


def read_lines(p):
    with open(p) as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]


def load(ds):
    cams = {}
    for l in read_lines(f"{ds}/sparse/0/cameras.txt"):
        t = l.split()
        cams[int(t[0])] = dict(model=t[1], w=int(t[2]), h=int(t[3]), p=[float(x) for x in t[4:]])
    ims = []
    # images.txt: one line per image, followed by a (possibly empty) 2D-point line; empty lines were stripped by read_lines
    # but the exporter writes an empty point line, so parse by field count instead.
    for l in read_lines(f"{ds}/sparse/0/images.txt"):
        t = l.split()
        if len(t) == 10 and t[9].lower().endswith((".jpg", ".jpeg", ".png")):
            ims.append(dict(id=int(t[0]), q=np.array(t[1:5], float), t=np.array(t[5:8], float), cam=int(t[8]), name=t[9]))
    xyz, rgb = [], []
    for l in read_lines(f"{ds}/sparse/0/points3D.txt"):
        t = l.split()
        xyz.append([float(x) for x in t[1:4]]); rgb.append([float(x) for x in t[4:7]])
    return cams, ims, np.array(xyz).reshape(-1, 3), np.array(rgb).reshape(-1, 3)


def qmat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                     [2*(x*y + z*w), 1 - 2*(x*x+z*z), 2*(y*z - x*w)],
                     [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x+y*y)]])


def score(R, t, cam, X, C, img, mode="w2c"):
    """Project X (N,3) with the given convention; returns (n_front, n_inside, r, mad, n_used)."""
    fx, fy, cx, cy = cam["p"][:4]
    if mode == "w2c":
        Xc = X @ R.T + t
    elif mode == "c2w":      # misread the pose as camera-to-world
        Xc = (X - t) @ R
    elif mode == "flip_yz":  # OpenGL-style camera axes
        Xc = (X @ R.T + t) * np.array([1, -1, -1])
    else:
        raise ValueError(mode)
    front = Xc[:, 2] > 0.05
    z = np.where(front, Xc[:, 2], 1.0)
    u = fx * Xc[:, 0] / z + cx
    v = fy * Xc[:, 1] / z + cy
    H, W = img.shape[:2]
    ins = front & (u >= 0) & (u < W - 1) & (v >= 0) & (v < H - 1)
    ui, vi = np.round(u[ins]).astype(int), np.round(v[ins]).astype(int)
    pix = img[vi, ui].astype(float)
    ok = pix.sum(1) > 0               # skip the black outside-fisheye region
    pix, col = pix[ok], C[ins][ok]
    if len(pix) < 200:
        return int(front.sum()), int(ins.sum()), float("nan"), float("nan"), len(pix)
    lp, lc = pix @ [.299, .587, .114], col @ [.299, .587, .114]
    r = np.corrcoef(lp, lc)[0, 1]
    mad = np.abs(pix - col).mean()
    return int(front.sum()), int(ins.sum()), r, mad, len(pix)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ds"); ap.add_argument("--max-centre", type=float, default=20.0)
    ap.add_argument("--points", type=int, default=60000); ap.add_argument("--worst", type=int, default=5)
    a = ap.parse_args()
    fail = []
    cams, ims, X, C = load(a.ds)
    print(f"{len(cams)} cameras, {len(ims)} images, {len(X)} points")
    if not ims:
        sys.exit("no images parsed")
    for k, c in cams.items():
        print(f"  camera {k}: {c['model']} {c['w']}x{c['h']} params {c['p']}")
        if c["model"] != "PINHOLE":
            fail.append(f"camera {k} is {c['model']}, expected PINHOLE")

    miss = [i["name"] for i in ims if not os.path.isfile(f"{a.ds}/images/{i['name']}")]
    if miss:
        fail.append(f"{len(miss)} listed images missing, e.g. {miss[:3]}")
    qn = np.array([abs(np.linalg.norm(i["q"]) - 1) for i in ims])
    print(f"quaternion norm error: max {qn.max():.2e}")
    if qn.max() > 1e-3:
        fail.append(f"non-unit quaternions (max err {qn.max():.2e})")
    cen = np.array([-qmat(i["q"]).T @ i["t"] for i in ims])
    med = np.median(cen, 0)
    dist = np.linalg.norm(cen - med, axis=1)
    print(f"camera centres: bbox min {cen.min(0).round(2)} max {cen.max(0).round(2)}, max dist from median {dist.max():.2f} m")
    if dist.max() > a.max_centre:
        fail.append(f"camera centre {dist.max():.1f} m from the median (> {a.max_centre})")
    if len(X) == 0:
        print("WARNING: points3D empty, skipping projection checks")
    elif not miss:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(X), min(a.points, len(X)), replace=False)
        Xs, Cs = X[sel], C[sel]
        modes = ["w2c", "c2w", "flip_yz"]
        res = {m: [] for m in modes}
        ctrl, fr, inn, rows = [], [], [], []
        for i in ims:
            img = cv2.imread(f"{a.ds}/images/{i['name']}")[:, :, ::-1]
            cam = cams[i["cam"]]
            if img.shape[:2] != (cam["h"], cam["w"]):
                fail.append(f"{i['name']}: image {img.shape[1]}x{img.shape[0]} != camera {cam['w']}x{cam['h']}")
                continue
            R = qmat(i["q"])
            nf, ni, r, mad, n = score(R, i["t"], cam, Xs, Cs, img)
            fr.append(nf / len(Xs)); inn.append(ni / len(Xs)); rows.append((i["name"], ni, r, mad))
            res["w2c"].append(r)
            for m in modes[1:]:
                res[m].append(score(R, i["t"], cam, Xs, Cs, img, m)[2])
            ctrl.append(score(R, i["t"], cam, Xs, Cs[rng.permutation(len(Cs))], img)[2])
        f = lambda v: np.nanmedian(v)
        print(f"points in front: median {np.median(fr)*100:.1f} %, inside image: median {np.median(inn)*100:.1f} % "
              f"(of {len(Xs)} sampled; the rest are behind/outside, normal for a room scan)")
        print("colour consistency (median over images of Pearson r, luminance):")
        print(f"  as exported (world-to-camera, OpenCV axes): r = {f(res['w2c']):.3f}")
        print(f"  control, point colours shuffled           : r = {f(ctrl):.3f}")
        print(f"  alt. c2w / flip_yz (should be worse)      : r = {f(res['c2w']):.3f} / {f(res['flip_yz']):.3f}")
        good = [x for x in res["w2c"] if x == x]
        print(f"  images with r > 0.2: {sum(x > 0.2 for x in good)}/{len(good)}; mean abs RGB diff (median) {np.nanmedian([r[3] for r in rows]):.1f}/255")
        rows.sort(key=lambda r: (r[2] if r[2] == r[2] else -1))
        print(f"  worst {a.worst}: " + ", ".join(f"{n} (r={r:.2f}, {ni} pts)" for n, ni, r, _ in rows[:a.worst]))
        if not f(res["w2c"]) > f(ctrl) + 0.15 or f(res["w2c"]) <= max(f(res["c2w"]), f(res["flip_yz"])):
            fail.append("projection colour score is not clearly above control / alternative conventions: pose convention suspect")
    print("RESULT:", "FAIL" if fail else "OK")
    for m in fail:
        print("  -", m)
    sys.exit(1 if fail else 0)


main()
