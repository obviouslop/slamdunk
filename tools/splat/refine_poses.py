#!/usr/bin/env python3
"""Refine the SteamVR camera poses of an exported COLMAP dataset with pycolmap (CPU SIFT + bundle adjustment).

usage: refine_poses.py IN_DATASET OUT_DATASET [--work DIR] [--match exhaustive|spatial] [--refine-focal]

IN_DATASET/sparse/0 holds the exported poses (world-to-camera, STAGE frame). Steps:
  1. SIFT on every image (the black fisheye region and a rim are masked out), pairwise matching + geometric verification
  2. triangulate points from the *given* poses (poses fixed) -> reprojection error of the headset poses
  3. bundle adjustment (poses + points; intrinsics fixed unless --refine-focal) -> reprojection error after
  4. gauge: the BA result is only defined up to a similarity; it is aligned back onto the input camera centres
     (Umeyama) so the dataset stays in the STAGE frame. The alignment (scale, rotation, translation) is printed;
     it should be ~identity, otherwise something is wrong.
OUT_DATASET = copy of IN_DATASET with sparse/0/images.txt replaced by the refined poses (images are symlinked,
points3D.txt is the original fused cloud). Run with the pycolmap venv (~/splat/venv/bin/python).
"""
import argparse, os, shutil, json
import numpy as np, cv2
import pycolmap

ap = argparse.ArgumentParser()
ap.add_argument("ds"); ap.add_argument("out")
ap.add_argument("--work", default=None)
ap.add_argument("--match", default="exhaustive", choices=["exhaustive", "spatial"])
ap.add_argument("--refine-focal", action="store_true")
ap.add_argument("--max-features", type=int, default=8192)
ap.add_argument("--rim", type=int, default=12, help="px eroded from the valid fisheye region before SIFT")
a = ap.parse_args()
ds, out = os.path.abspath(a.ds), os.path.abspath(a.out)
work = os.path.abspath(a.work or os.path.join(os.path.dirname(out), "work_" + os.path.basename(out)))
os.makedirs(work, exist_ok=True)
imgdir = f"{ds}/images"
names = sorted(os.listdir(imgdir))

# 1. masks + features + matches --------------------------------------------------------------
mdir = f"{work}/masks"
if not os.path.isdir(mdir):
    os.makedirs(mdir)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.rim + 1, 2 * a.rim + 1))
    for n in names:
        im = cv2.imread(f"{imgdir}/{n}")
        valid = (im.sum(2) > 0).astype(np.uint8)
        valid = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        cv2.imwrite(f"{mdir}/{n}.png", cv2.erode(valid, k) * 255)
db = f"{work}/database.db"
if not os.path.isfile(db):
    ro = pycolmap.ImageReaderOptions()
    ro.mask_path = mdir
    ro.camera_model = "PINHOLE"
    cams = [l.split() for l in open(f"{ds}/sparse/0/cameras.txt") if l.strip() and not l.startswith("#")]
    ro.camera_params = ",".join(cams[0][4:8])
    eo = pycolmap.FeatureExtractionOptions()
    eo.sift.max_num_features = a.max_features
    print("extracting features ...", flush=True)
    pycolmap.extract_features(db, imgdir, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=ro, extraction_options=eo)
    print("matching ...", flush=True)
    if a.match == "exhaustive":
        pycolmap.match_exhaustive(db)
    else:
        pycolmap.match_spatial(db)
else:
    print("reusing", db)

# 2. triangulate with fixed poses ---------------------------------------------------------------
tri_dir = f"{work}/tri"; os.makedirs(tri_dir, exist_ok=True)
# the database numbers images alphabetically and has ONE camera for all (both cams share intrinsics); write the input model with
# the database ids so triangulation can match it. Original ids/camera ids are restored when writing the output.
_db = pycolmap.Database.open(db)
db_img = {i.name: (i.image_id, i.camera_id) for i in _db.read_all_images()}
db_cam = _db.read_all_cameras()[0]
_db.close()
src = {}
for l in open(f"{ds}/sparse/0/images.txt"):
    t = l.split()
    if len(t) == 10 and t[9].lower().endswith(".jpg"):
        src[t[9]] = t
with open(f"{tri_dir}/cameras.txt", "w") as f:
    f.write(f"{db_cam.camera_id} PINHOLE {db_cam.width} {db_cam.height} " + " ".join(f"{x:.6f}" for x in db_cam.params) + "\n")
with open(f"{tri_dir}/images.txt", "w") as f:
    for n, (iid, cid) in sorted(db_img.items(), key=lambda kv: kv[1][0]):
        t = src[n]
        f.write(f"{iid} {' '.join(t[1:8])} {cid} {n}\n\n")
open(f"{tri_dir}/points3D.txt", "w").close()
rec = pycolmap.Reconstruction(tri_dir)
tri = pycolmap.triangulate_points(rec, db, imgdir, f"{work}/tri_out")
print("triangulated:", tri.summary())
orig = {i.name: (i.cam_from_world().rotation.matrix().copy(), i.cam_from_world().translation.copy()) for i in tri.images.values()}
err0 = tri.compute_mean_reprojection_error()
print(f"mean reprojection error with the headset poses: {err0:.3f} px, {tri.num_points3D()} points")

# 3. bundle adjustment ----------------------------------------------------------------------------
bo = pycolmap.BundleAdjustmentOptions()
bo.refine_focal_length = a.refine_focal
bo.refine_principal_point = False
bo.refine_extra_params = False
pycolmap.bundle_adjustment(tri, bo)
err1 = tri.compute_mean_reprojection_error()
print(f"mean reprojection error after BA: {err1:.3f} px, {tri.num_points3D()} points")

# 4. gauge: align back to the input centres --------------------------------------------------------
ids = sorted(tri.images)
def centre(R, t): return -R.T @ t
c_in = np.array([centre(*orig[tri.images[i].name]) for i in ids])
c_ba = np.array([tri.images[i].projection_center() for i in ids])
mu_a, mu_b = c_ba.mean(0), c_in.mean(0)
H = (c_ba - mu_a).T @ (c_in - mu_b)
U, S, Vt = np.linalg.svd(H)
D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
Ra = Vt.T @ D @ U.T
s = np.trace(np.diag(S) @ D) / ((c_ba - mu_a) ** 2).sum()
ta = mu_b - s * Ra @ mu_a
ang = np.degrees(np.arccos(np.clip((np.trace(Ra) - 1) / 2, -1, 1)))
print(f"gauge alignment BA->STAGE: scale {s:.5f}, rotation {ang:.3f} deg, translation {np.linalg.norm(ta)*100:.1f} cm")

lines = ["# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME", "# (empty line: no 2D points)"]
cam_id = {l.split()[-1]: int(l.split()[8]) for l in open(f"{ds}/sparse/0/images.txt")
          if l.strip() and not l.startswith("#") and l.split()[-1].lower().endswith(".jpg")}
img_id = {l.split()[-1]: int(l.split()[0]) for l in open(f"{ds}/sparse/0/images.txt")
          if l.strip() and not l.startswith("#") and l.split()[-1].lower().endswith(".jpg")}
dc, dr = [], []
for i in sorted(ids, key=lambda i: img_id[tri.images[i].name]):
    im = tri.images[i]
    R_ba, t_ba = im.cam_from_world().rotation.matrix(), im.cam_from_world().translation
    # similarity BA->STAGE: x_stage = s Ra x_ba + ta  =>  metric camera coords: R = R_ba Ra^T, t = s t_ba - R ta
    Rn = R_ba @ Ra.T
    tn = s * t_ba - Rn @ ta   # = s*(R_ba X_ba + t_ba) form, i.e. camera coordinates in metres
    q = pycolmap.Rotation3d(Rn).quat   # x,y,z,w
    c_new = -Rn.T @ tn
    R0, t0 = orig[im.name]
    dc.append(np.linalg.norm(c_new - centre(R0, t0)))
    dr.append(np.degrees(np.arccos(np.clip((np.trace(R0 @ Rn.T) - 1) / 2, -1, 1))))
    lines.append(f"{img_id[im.name]} {q[3]:.9f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} {tn[0]:.6f} {tn[1]:.6f} {tn[2]:.6f} {cam_id[im.name]} {im.name}")
    lines.append("")
dc, dr = np.array(dc) * 100, np.array(dr)
print(f"pose change vs headset: centre median {np.median(dc):.1f} cm, p90 {np.percentile(dc,90):.1f} cm, max {dc.max():.1f} cm; "
      f"rotation median {np.median(dr):.2f} deg, p90 {np.percentile(dr,90):.2f}, max {dr.max():.2f}")

if os.path.exists(out): shutil.rmtree(out)
os.makedirs(f"{out}/sparse/0")
os.symlink(imgdir, f"{out}/images")
for f in ("cameras.txt", "points3D.txt"):
    shutil.copy(f"{ds}/sparse/0/{f}", f"{out}/sparse/0/{f}")
for f in ("fused.ply", "export.json"):
    if os.path.exists(f"{ds}/{f}"): os.symlink(f"{ds}/{f}", f"{out}/{f}")
open(f"{out}/sparse/0/images.txt", "w").write("\n".join(lines) + "\n")
json.dump(dict(err_before_px=err0, err_after_px=err1, n_points=tri.num_points3D(), gauge_scale=s, gauge_rot_deg=ang,
               gauge_trans_m=float(np.linalg.norm(ta)), centre_shift_cm_median=float(np.median(dc)),
               centre_shift_cm_max=float(dc.max()), rot_shift_deg_median=float(np.median(dr))),
          open(f"{out}/refine.json", "w"), indent=1)
print("wrote", out)
