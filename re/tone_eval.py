#!/usr/bin/env python3
"""Compare matcher tone curves (LiveStereo tone=) on live recordings by multi-view depth consistency.

    re/tone_eval.py [--tones linear,sqrt,log,auto] [--n 60] [--scale 0.35] captures/live17 [captures/live20 ...]

Per recording and tone: every frame's depth (same pipeline as pcserver: bin_raw10 -> LiveStereo.depth) is projected into the
frames i+-1, i+-2 (tracked poses) and compared with their depth: agree = |dZ|/Z < 3 %, front = > 5 % in front (floaters).
Also density and the median grey level of the matcher input. No ground truth needed; pose error caps agree for all tones alike.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from livestereo import LiveStereo
from tsdf import bin_raw10

ap = argparse.ArgumentParser()
ap.add_argument("recs", nargs="+"); ap.add_argument("--tones", default="linear,sqrt,log,auto")
ap.add_argument("--n", type=int, default=60); ap.add_argument("--scale", type=float, default=0.35)
ap.add_argument("--far", type=float, default=4.0)
a = ap.parse_args()


def load(d, frames, k):
    st = json.load(open(os.path.join(d, "stream.json")))
    kw = dict(W=st.get("line_px", 2464), stride=st.get("line_bytes", 3088), skip=st.get("skip", 0), black=float(st.get("black", 64)), threads=3)
    r = frames[k]; out = []
    for s in ("left", "right"):
        b = open(os.path.join(d, f"{r['i']:05d}_{s}.y10p"), "rb").read()
        H = min(len(b) // kw["stride"] - kw["skip"], kw["W"])
        out.append(bin_raw10(b, H=H, **kw))
    return out


tones = a.tones.split(",")
total = {t: np.zeros(4) for t in tones}
for d in a.recs:
    frames = [json.loads(l) for l in open(os.path.join(d, "frames.jsonl"))]
    frames = [r for r in frames if r["left"].get("pose_valid") and os.path.exists(os.path.join(d, f"{r['i']:05d}_right.y10p"))][: a.n]
    pairs = [load(d, frames, k) for k in range(len(frames))]
    for t in tones:
        ls = LiveStereo(scale=a.scale, tone=t); n = ls.size[0]
        Zs, Ts, med = [], [], []; t0 = time.monotonic()
        for k, (cL, cR) in enumerate(pairs):
            ls.debug = {}
            Z, _, _ = ls.depth(cL, cR, far=a.far, half=True)
            g = ls.debug["rect_grey_L"]; med.append(np.median(g[g > 0])); ls.debug = None
            Zs.append(Z); Ts.append(ls.world_from_rect(frames[k]["left"]["pose"]))
        ms = (time.monotonic() - t0) / len(pairs) * 1000
        u, v = np.meshgrid(np.arange(n, dtype=np.float32), np.arange(n, dtype=np.float32))
        cx, cy, f = ls.P1[0, 2], ls.P1[1, 2], ls.f
        agree = front = tot = 0
        for i in range(len(Zs)):
            m = Zs[i] > 0; Z = Zs[i][m]
            X = np.stack([(u[m] - cx) * Z / f, (v[m] - cy) * Z / f, Z], 1)[::4]
            Pw = X @ Ts[i][:3, :3].T + Ts[i][:3, 3]
            for j in (i - 2, i - 1, i + 1, i + 2):
                if not 0 <= j < len(Zs): continue
                Ti = np.linalg.inv(Ts[j]); Xr = Pw @ Ti[:3, :3].T + Ti[:3, 3]; z = Xr[:, 2]; ok = z > 0.1
                uu = np.round(f * Xr[ok, 0] / z[ok] + cx).astype(int); vv = np.round(f * Xr[ok, 1] / z[ok] + cy).astype(int)
                ib = (uu >= 0) & (uu < n) & (vv >= 0) & (vv < n)
                Zj = Zs[j][vv[ib], uu[ib]]; zz = z[ok][ib]; h = Zj > 0
                rel = (zz[h] - Zj[h]) / Zj[h]
                agree += np.sum(np.abs(rel) < 0.03); front += np.sum(rel < -0.05); tot += len(rel)
        dens = float(np.mean([(Z > 0).mean() for Z in Zs]))
        total[t] += [agree, front, tot, dens * len(Zs)]
        print(f"{os.path.basename(d):8s} {t:7s} grey median {np.median(med):5.1f}  density {dens:.3f}  agree {agree / max(tot, 1):.3f}  "
              f"floaters {front / max(tot, 1):.3f}  {ms:.0f} ms/pair", flush=True)
print("ALL")
for t in tones:
    a_, f_, n_, _ = total[t]
    print(f"  {t:7s} agree {a_ / max(n_, 1):.3f}  floaters {f_ / max(n_, 1):.3f}  matched points {int(a_)}")
