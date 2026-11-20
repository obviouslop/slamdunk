#!/usr/bin/env python3
"""Offline check of tools/farstereo.py on a recording: keyframes, pairs, far points, and the claimed depth sigma.
The sigma model is validated where the near stereo still has depth: wide-baseline depth of each pair is compared with the
reference frame's own stereo depth (z 1.2-3.8 m; --zmin lowers the far range for this).
usage: test_far.py RECDIR [--every 1] [--limit N] [--zmin 1.2]"""
import argparse, json, os, shutil, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from livestereo import LiveStereo, unpack_raw
from farstereo import FarStereo
ap = argparse.ArgumentParser(); ap.add_argument("rec"); ap.add_argument("--every", type=int, default=1); ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--zmin", type=float, default=1.2); ap.add_argument("--near-frac", type=float, default=0.2); ap.add_argument("--blur", type=float, default=1.0)
ap.add_argument("--out", default="captures/scratch/far_test"); ap.add_argument("--ply")
a = ap.parse_args()
ls = LiveStereo(scale=0.5, max_depth=4.0)
fs = FarStereo(ls, a.out, z_min=a.zmin, near_frac=a.near_frac, blur_px=a.blur, rel_sigma=0.3)
recs = [json.loads(l) for l in open(os.path.join(a.rec, "frames.jsonl"))]
recs = [r for r in recs if abs(r["left"]["ts"] - r["right"]["ts"]) < 0.015 and r["left"]["pose_valid"]][::a.every]
if a.limit: recs = recs[:a.limit]
Zs = {}; t0 = time.time()
for r in recs:
    i = r["i"]
    L = unpack_raw(open(os.path.join(a.rec, f"{i:05d}_left.y10p"), "rb").read()); R = unpack_raw(open(os.path.join(a.rec, f"{i:05d}_right.y10p"), "rb").read())
    Z, rgb, _ = ls.depth(L, R, far=4.0, half=False)
    T = ls.world_from_rect(np.array(r["left"]["pose"]))
    s = fs.offer(rgb, Z, T, r["left"]["ts"], r["left"]["exp_lines"])
    if s: Zs[s] = Z.astype(np.float16)
print(f"{len(recs)} frames {time.time()-t0:.0f}s: {len(fs.kfs)} keyframes {fs.bytes/1e6:.1f} MB; stats {fs.stats}")
res = []; batches = []
t0 = time.time()
while fs.new:
    s = fs.new.pop(0); k = fs.kfs[s]
    for j in fs.partners(k):
        fs.pairs_done.add((min(j["seq"], s), max(j["seq"], s))); dbg = {}
        b = fs.pair_points(k, j, debug=dbg); fs.stats["pairs"] += 1
        if b is None: fs.stats["rejected_pair"] += 1; print(f"  pair {s}-{j['seq']}: rejected {dbg}"); continue
        fs.stats["pairs_ok"] += 1; fs.merge(b, k["ts"]); batches.append(b)
        # validate against the stereo depth of the reference keyframe
        kr = fs.kfs[b["ref"]]; Zr = Zs.get(b["ref"])
        if Zr is None: continue
        Pc = (b["P"] - kr["T"][:3, 3]) @ kr["T"][:3, :3]
        u = np.round(ls.f * Pc[:, 0] / Pc[:, 2] + ls.P1[0, 2]).astype(int); v = np.round(ls.f * Pc[:, 1] / Pc[:, 2] + ls.P1[1, 2]).astype(int)
        ok = (Pc[:, 2] > 0.1) & (u >= 0) & (u < Zr.shape[1]) & (v >= 0) & (v < Zr.shape[0])
        zs = np.zeros(len(Pc)); zs[ok] = Zr[v[ok], u[ok]]
        m = ok & (zs > 1.2) & (zs < 3.8)
        if m.sum() > 50:
            sst = zs[m] ** 2 / (ls.f * ls.baseline) * 0.5          # stereo depth sigma (0.5 px disparity)
            e = Pc[m, 2] - zs[m]; sg = np.sqrt(b["sig"][m] ** 2 + sst ** 2)
            res.append((m.sum(), b["B"], np.median(e), 1.4826 * np.median(np.abs(e - np.median(e))), np.median(b["sig"][m]), np.median(sst), np.median(np.abs(e) / sg)))
            print(f"  pair {s}-{j['seq']} B={b['B']:.2f} m n_in={dbg['n_in']}/{dbg['n_matches']}: {len(b['P'])} pts, {m.sum()} vs stereo: "
                  f"bias {np.median(e)*100:+.1f} cm, spread {res[-1][3]*100:.1f} cm (claimed far {res[-1][4]*100:.1f} + stereo {res[-1][5]*100:.1f} cm), |e|/sigma median {res[-1][6]:.2f}")
        else: print(f"  pair {s}-{j['seq']} B={b['B']:.2f} m: {len(b['P'])} pts (z {b['z'].min():.1f}-{b['z'].max():.1f}), none in the stereo range")
print(f"pairs {fs.stats['pairs_ok']}/{fs.stats['pairs']} in {time.time()-t0:.0f}s")
P, C, rng, S, T = fs.points()
print(f"far store {len(fs.P)} pts, confirmed output {len(P)}; range median {np.median(rng) if len(rng) else 0:.1f} m, sigma median {np.median(S) if len(S) else 0:.2f} m")
if a.ply and len(P):
    import pcpub; pcpub.write_ply(a.ply, P, C, dist=rng)
