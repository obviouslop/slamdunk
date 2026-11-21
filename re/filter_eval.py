#!/usr/bin/env python3
"""Floater filters compared by multi-view depth consistency on live recordings (same metric as re/tone_eval.py).

    re/filter_eval.py [--n 40] captures/live16 captures/live17 ...

Variants (all with the default log tone): orientation control (old upside-down rectification), SGBM uniqueness, a per-pixel
confidence computed on the MATCHER INPUT (texture of the tone-mapped grey + local depth jump; LiveStereo.confidence uses the
dark colour image and is ~0 indoors), thresholded, and the flying-pixel edge filter. Prints agree / floaters / density and the
share of the base's agreeing points that each variant keeps.
"""
import argparse, json, os, sys
import numpy as np, cv2
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import livestereo
from tsdf import bin_raw10

ap = argparse.ArgumentParser()
ap.add_argument("recs", nargs="+"); ap.add_argument("--n", type=int, default=40); ap.add_argument("--scale", type=float, default=0.35)
a = ap.parse_args()


def conf_matcher(grey, Z, g0=6.0, jump=0.04):
    """texture of the matcher input (mean |grad| over 7x7, grey levels) x depth-jump term -> 0..1"""
    g = grey.astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) * 0.125; gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) * 0.125
    ct = np.clip(cv2.blur(np.hypot(gx, gy), (7, 7)) / g0, 0, 1)
    k = np.ones((3, 3), np.uint8)
    zmax = cv2.dilate(Z, k); zmin = cv2.erode(np.where(Z > 0, Z, 1e3).astype(np.float32), k)
    ce = np.clip(1 - (zmax - np.minimum(zmin, zmax)) / np.maximum(Z, 1e-3) / jump, 0, 1)
    return np.where(Z > 0, ct * ce, 0)


def edge_filter(Z, rel=0.05):
    k = np.ones((3, 3), np.uint8)
    zmax = cv2.dilate(Z, k); zmin = cv2.erode(np.where(Z > 0, Z, 1e3).astype(np.float32), k)
    return np.where((Z > 0) & ((zmax - np.minimum(zmin, zmax)) < rel * Z), Z, 0)


VARIANTS = {   # name: (env PC_RECT_NO_ROTATE, LiveStereo kwargs, post filter(Z, debug) -> Z)
    "base (upright)":        (False, {}, None),
    "old upside-down":       (True, {}, None),
    "uniq 15":               (False, dict(uniq=15), None),
    "uniq 25":               (False, dict(uniq=25), None),
    "conf >= 0.3":           (False, {}, lambda Z, D: np.where(conf_matcher(D["rect_grey_L"], Z) >= 0.3, Z, 0)),
    "conf >= 0.5":           (False, {}, lambda Z, D: np.where(conf_matcher(D["rect_grey_L"], Z) >= 0.5, Z, 0)),
    "edge 5 %":              (False, {}, lambda Z, D: edge_filter(Z, 0.05)),
    "uniq 15 + conf >= 0.3": (False, dict(uniq=15), lambda Z, D: np.where(conf_matcher(D["rect_grey_L"], Z) >= 0.3, Z, 0)),
}


def score(ls, Zs, Ts):
    n = ls.size[0]; u, v = np.meshgrid(np.arange(n, dtype=np.float32), np.arange(n, dtype=np.float32))
    cx, cy, f = ls.P1[0, 2], ls.P1[1, 2], ls.f; agree = front = tot = 0
    for i in range(len(Zs)):
        m = Zs[i] > 0; Z = Zs[i][m]
        X = np.stack([(u[m] - cx) * Z / f, (v[m] - cy) * Z / f, Z], 1)[::4]
        Pw = X @ Ts[i][:3, :3].T + Ts[i][:3, 3]
        for j in (i - 2, i - 1, i + 1, i + 2):
            if not 0 <= j < len(Zs): continue
            Ti = np.linalg.inv(Ts[j]); Xr = Pw @ Ti[:3, :3].T + Ti[:3, 3]; z = Xr[:, 2]; ok = z > 0.1
            uu = np.round(f * Xr[ok, 0] / z[ok] + cx).astype(int); vv = np.round(f * Xr[ok, 1] / z[ok] + cy).astype(int)
            ib = (uu >= 0) & (uu < n) & (vv >= 0) & (vv < n)
            Zj = Zs[j][vv[ib], uu[ib]]; zz = z[ok][ib]; h = Zj > 0; rel = (zz[h] - Zj[h]) / Zj[h]
            agree += np.sum(np.abs(rel) < 0.03); front += np.sum(rel < -0.05); tot += len(rel)
    return agree, front, tot


tot = {k: np.zeros(4) for k in VARIANTS}
for d in a.recs:
    st = json.load(open(os.path.join(d, "stream.json")))
    kw = dict(W=st.get("line_px", 2464), stride=st.get("line_bytes", 3088), skip=st.get("skip", 0), black=float(st.get("black", 64)), threads=3)
    fr = [json.loads(l) for l in open(os.path.join(d, "frames.jsonl"))]
    fr = [r for r in fr if r["left"].get("pose_valid") and os.path.exists(os.path.join(d, f"{r['i']:05d}_right.y10p"))][: a.n]
    pairs = []
    for r in fr:
        c = []
        for s in ("left", "right"):
            b = open(os.path.join(d, f"{r['i']:05d}_{s}.y10p"), "rb").read()
            c.append(bin_raw10(b, H=min(len(b) // kw["stride"] - kw["skip"], kw["W"]), **kw))
        pairs.append(c)
    for name, (norot, kwa, post) in VARIANTS.items():
        if norot: os.environ["PC_RECT_NO_ROTATE"] = "1"
        else: os.environ.pop("PC_RECT_NO_ROTATE", None)
        ls = livestereo.LiveStereo(scale=a.scale, **kwa); Zs, Ts = [], []
        for k, (cL, cR) in enumerate(pairs):
            ls.debug = {}; Z, _, _ = ls.depth(cL, cR, far=4.0, half=True)
            if post: Z = post(Z, ls.debug).astype(np.float32)
            ls.debug = None; Zs.append(Z); Ts.append(ls.world_from_rect(fr[k]["left"]["pose"]))
        ag, fo, n = score(ls, Zs, Ts); dens = float(np.mean([(Z > 0).mean() for Z in Zs]))
        tot[name] += [ag, fo, n, dens * len(Zs)]
        print(f"{os.path.basename(d):7s} {name:22s} density {dens:.3f}  agree {ag / max(n, 1):.3f}  floaters {fo / max(n, 1):.3f}", flush=True)
os.environ.pop("PC_RECT_NO_ROTATE", None)
base = tot["base (upright)"]
print("ALL (kept = agreeing points relative to base)")
for name, (ag, fo, n, _) in tot.items():
    print(f"  {name:22s} agree {ag / max(n, 1):.3f}  floaters {fo / max(n, 1):.3f}  kept {ag / max(base[0], 1):.2f}")
