#!/usr/bin/env python3
"""Run the TSDF pose refiner over a recording and write captures/scratch/<rec>_refined/ (symlinked images, refined frames.jsonl,
patched detection cache) so that tools/fid_analyze.py on it measures the residual SteamVR error with / without refinement.
usage: test_refine.py RECDIR [--every 1] [--no-refine]   then   fid_analyze.py captures/scratch/<rec>_refined  (and the original)"""
import argparse, glob, json, os, pickle, shutil, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from livestereo import LiveStereo, unpack_raw
from tsdf import TSDF
from poserefine import PoseRefiner
ap = argparse.ArgumentParser(); ap.add_argument("rec"); ap.add_argument("--every", type=int, default=1); ap.add_argument("--no-refine", action="store_true")
ap.add_argument("--tag", default="refined")
a = ap.parse_args()
rec = a.rec.rstrip("/"); out = os.path.join("captures/scratch", os.path.basename(rec) + "_" + a.tag); os.makedirs(out, exist_ok=True)
for f in glob.glob(os.path.join(rec, "*.y10p")):
    d = os.path.join(out, os.path.basename(f))
    if not os.path.exists(d): os.symlink(os.path.abspath(f), d)
ls = LiveStereo(scale=0.5, max_depth=4.0); ts = TSDF(voxel=0.02, max_depth=2.5); rf = PoseRefiner()
recs = [json.loads(l) for l in open(os.path.join(rec, "frames.jsonl"))]
fuse = set(r["i"] for r in recs[::a.every] if abs(r["left"]["ts"] - r["right"]["ts"]) < 0.015 and r["left"]["pose_valid"])
t0 = time.time(); corr = []; tref = []
for r in recs:
    if r["i"] not in fuse: continue
    i = r["i"]
    L = unpack_raw(open(os.path.join(rec, f"{i:05d}_left.y10p"), "rb").read()); R = unpack_raw(open(os.path.join(rec, f"{i:05d}_right.y10p"), "rb").read())
    Z, rgb, _ = ls.depth(L, R, far=ts.carve_depth, half=False)
    T = ls.world_from_rect(np.array(r["left"]["pose"]))
    t1 = time.time(); T2 = T if a.no_refine else rf.refine(ts, ls, Z, T); tref.append(time.time() - t1)
    D = T2 @ np.linalg.inv(T)
    for side in ("left", "right"):
        H = np.eye(4); H[:3, :] = np.array(r[side]["pose"]); r[side]["pose"] = (D @ H)[:3, :].tolist()
    corr.append((np.linalg.norm(D[:3, 3]) * 1000, np.degrees(np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1)))))
    ts.integrate(ls, Z, rgb, T2, t=r["left"]["ts"])
with open(os.path.join(out, "frames.jsonl"), "w") as f:
    for r in recs: f.write(json.dumps(r) + "\n")
os.makedirs(os.path.join(out, "fid"), exist_ok=True)
for pk in glob.glob(os.path.join(rec, "fid", "detections_*.pkl")):
    rows = pickle.load(open(pk, "rb")); byi = {r["i"]: r for r in recs}
    for row in rows: row["r"] = byi[row["i"]]
    pickle.dump(rows, open(os.path.join(out, "fid", os.path.basename(pk)), "wb"))
for f in ("hmd_500hz.txt", "stream.json"):
    if os.path.exists(os.path.join(rec, f)) and not os.path.exists(os.path.join(out, f)): shutil.copy(os.path.join(rec, f), out)
c = np.array(corr)
print(f"{len(corr)} frames in {time.time()-t0:.0f}s, refine {np.mean(tref)*1000:.0f} ms/frame; used {rf.stats['used']} skipped {rf.stats['skipped']}; "
      f"correction median {np.median(c[:,0]):.1f} mm / {np.median(c[:,1]):.2f} deg, p90 {np.percentile(c[:,0],90):.1f} mm")
print("->", out)
