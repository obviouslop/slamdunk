#!/usr/bin/env python3
"""Clean up / extend the training masks of an exported COLMAP dataset (tools/export_colmap.py --masks --depth).

usage: python3 -I make_masks.py DATASET OUT [--body-radius 0.40] [--body-top 0.25] [--near 1.2] [--min-area 8000]
                                [--solidity 0.6] [--fill 0.8] [--body]

OUT = a dataset directory with images/, sparse/, depth/ symlinked from DATASET and a new masks/ (X.jpg.png, 255 keep,
0 ignore: Brush picks it up from the `masks` folder). Per image:
  1. fisheye region: pixels <= 3 in every channel, connected regions touching the border (own detection, dilated).
  2. the exporter's "interior" ignore pixels (stereo near-field, nearer-than-map, controllers) are re-checked as blobs:
     a blob is kept only if it is compact (solidity = area / convex hull >= --solidity) AND solid (>= --fill of the blob is masked) AND either large (area >=
     --min-area) or near (median valid depth < --near m). This removes the blocky stereo-noise patches on textureless ceilings / glossy surfaces, while a person,
     hand or controller close to the camera stays masked.
  3. wearer's body: the depth map covers only the stereo overlap (a skewed quadrilateral), so legs/feet in the lower
     image are often outside it. They are masked geometrically instead: a vertical cylinder of --body-radius metres
     around the camera's vertical axis, from the floor (3rd percentile of the point cloud height) + 5 cm up to
     --body-top metres below the camera, projected with the known pose. Anything inside that column (legs, shoes, a
     chair you stand against) is ignored. EXPERIMENTAL, off by default (--body): at radius 0.4 m it covers ~38 % of the image because the camera sits on the column axis.
Stats and a few overlay images are written to OUT/mask_report.txt.
"""
import argparse, os
from multiprocessing import Pool
import numpy as np, cv2

ap = argparse.ArgumentParser()
ap.add_argument("ds"); ap.add_argument("out")
ap.add_argument("--body-radius", type=float, default=0.40)
ap.add_argument("--body-top", type=float, default=0.25)
ap.add_argument("--near", type=float, default=1.2)
ap.add_argument("--min-area", type=int, default=8000)
ap.add_argument("--solidity", type=float, default=0.6)
ap.add_argument("--fill", type=float, default=0.8)
ap.add_argument("--body", action="store_true", help="mask a vertical column under the headset (experimental: too coarse, covers ~38 %% of the image at the default radius)")
a = ap.parse_args()
ds, out = os.path.abspath(a.ds), os.path.abspath(a.out)


def qmat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
                     [2*(x*y + z*w), 1 - 2*(x*x+z*z), 2*(y*z - x*w)],
                     [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x+y*y)]])


def lines(p):
    return [l.split() for l in open(p) if l.strip() and not l.startswith("#")]


cams = {int(t[0]): [float(x) for x in t[4:8]] for t in lines(f"{ds}/sparse/0/cameras.txt")}
ims = []
for t in lines(f"{ds}/sparse/0/images.txt"):
    if len(t) == 10 and t[9].lower().endswith(".jpg"):
        q = np.array(t[1:5], float); tr = np.array(t[5:8], float)
        R = qmat(q); ims.append((t[9], R, tr, -R.T @ tr, cams[int(t[8])]))
ys = []
with open(f"{ds}/sparse/0/points3D.txt") as f:
    for l in f:
        if l[0] != "#" and l.strip():
            ys.append(float(l.split()[2]))
floor_y = float(np.percentile(ys, 3)) if ys else 0.0

# body column template: horizontal disc points (offsets), later stacked in y
step = 0.04
gx = np.arange(-a.body_radius, a.body_radius + 1e-6, step)
disc = np.array([(x, z) for x in gx for z in gx if x * x + z * z <= a.body_radius ** 2])


def process(item):
    name, R, t, c, (fx, fy, cx, cy) = item
    im = cv2.imread(f"{ds}/images/{name}")
    H, W = im.shape[:2]
    # 1. fisheye region
    blk = (im.max(2) <= 3).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(blk, connectivity=4)
    fish = np.zeros((H, W), np.uint8)
    for i in range(1, n):
        x, y, w, h, ar = st[i]
        if ar > 3000 and (x == 0 or y == 0 or x + w >= W or y + h >= H):
            fish[lab == i] = 1
    fish = cv2.dilate(fish, np.ones((9, 9), np.uint8))
    # 2. exporter's interior ignore pixels, filtered by blob shape / distance
    base = cv2.imread(f"{ds}/masks/{name}.png", 0)
    interior = ((base == 0) & (cv2.dilate(fish, np.ones((25, 25), np.uint8)) == 0)).astype(np.uint8)
    kept = np.zeros((H, W), np.uint8)
    dep = cv2.imread(f"{ds}/depth/{name}.png", cv2.IMREAD_UNCHANGED)
    if dep is not None:
        dep = cv2.resize(dep, (W, H), interpolation=cv2.INTER_NEAREST)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.morphologyEx(interior, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)), connectivity=8)
    nkept = 0
    for i in range(1, n):
        comp = lab == i
        pix = comp & (interior > 0)
        area = int(comp.sum())
        cnts, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hull = cv2.contourArea(cv2.convexHull(np.vstack(cnts))) if cnts else 1
        sol = area / max(hull, 1)
        fill = pix.sum() / max(area, 1)   # blocky noise clusters are ~half holes, real objects are solid
        near = False
        if dep is not None:
            d = dep[pix]; d = d[d > 0]
            near = len(d) > 20 and np.median(d) < a.near * 1000
        if sol >= a.solidity and fill >= a.fill and (near or area >= a.min_area):
            kept[pix] = 1; nkept += 1
    # 3. body column
    body = np.zeros((H, W), np.uint8)
    if a.body:
        y_lo, y_hi = floor_y + 0.05, c[1] - a.body_top
        if y_hi > y_lo:
            yy = np.arange(y_lo, y_hi, step)
            P = np.concatenate([np.column_stack([c[0] + disc[:, 0], np.full(len(disc), y), c[2] + disc[:, 1]]) for y in yy])
            Xc = P @ R.T + t
            z = Xc[:, 2]; ok = z > 0.05
            u = fx * Xc[ok, 0] / z[ok] + cx; v = fy * Xc[ok, 1] / z[ok] + cy
            rad = np.maximum(2, (fx * 0.035 / z[ok])).astype(int)
            for uu, vv, rr in zip(u, v, rad):
                if -rr < uu < W + rr and -rr < vv < H + rr:
                    cv2.circle(body, (int(uu), int(vv)), int(rr), 1, -1)
            body &= 1
    ign = ((fish > 0) | (kept > 0) | (body > 0)).astype(np.uint8)
    ign = cv2.dilate(ign, np.ones((7, 7), np.uint8))
    m = np.where(ign > 0, 0, 255).astype(np.uint8)
    cv2.imwrite(f"{out}/masks/{name}.png", m)
    valid = im.max(2) > 3
    return (name, float(((base == 0) & valid).sum() / valid.sum()), float(((m == 0) & valid).sum() / valid.sum()),
            float(((body > 0) & valid).sum() / valid.sum()), int(interior.sum() > 0) * nkept)


if __name__ == "__main__":
    os.makedirs(f"{out}/masks", exist_ok=True)
    for d in ("images", "sparse", "depth"):
        if not os.path.lexists(f"{out}/{d}"):
            os.symlink(f"{ds}/{d}", f"{out}/{d}")
    for f in ("export.json", "fused.ply"):
        if os.path.exists(f"{ds}/{f}") and not os.path.lexists(f"{out}/{f}"):
            os.symlink(f"{ds}/{f}", f"{out}/{f}")
    with Pool(8) as p:
        rows = p.map(process, ims, chunksize=4)
    old = np.array([r[1] for r in rows]); new = np.array([r[2] for r in rows]); body = np.array([r[3] for r in rows])
    rep = (f"{len(rows)} images, floor y = {floor_y:.3f} m\n"
           f"interior ignored fraction: exporter median {np.median(old):.3f} (max {old.max():.3f})  ->  now (incl. body) median {np.median(new):.3f} (max {new.max():.3f})\n"
           f"body column covers: median {np.median(body):.3f}, p90 {np.percentile(body, 90):.3f}, max {body.max():.3f}\n")
    open(f"{out}/mask_report.txt", "w").write(rep); print(rep, end="")
