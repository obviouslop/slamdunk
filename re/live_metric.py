#!/usr/bin/env python3
"""User-facing accuracy: map as it exists at time k (fused from frames < k) vs. the depth measured at frame k.
Sequential replay of the even live7 frames (re/tsdf_sweep.py cache). Prints median signed bias (cm, - = map
behind the real surface), fraction within +-3 cm (of points within +-12 cm), and point count."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from livestereo import LiveStereo; from tsdf import TSDF
ls = LiveStereo(scale=0.35, max_depth=2.5)
z = np.load(sys.argv[1]); Zs, Cs, Ts = z["Z"], z["C"], z["T"]
f = ls.f; cx, cy = ls.P1[0, 2], ls.P1[1, 2]
def resid(P, Z, T):
    Pc = (P - T[:3, 3]) @ T[:3, :3]; zz = Pc[:, 2]; ok = zz > 0.3
    u = np.round(f * Pc[ok, 0] / zz[ok] + cx).astype(int); v = np.round(f * Pc[ok, 1] / zz[ok] + cy).astype(int); n = Z.shape[0]
    i = (u >= 0) & (u < n) & (v >= 0) & (v < n); Zm = Z[v[i], u[i]]; zq = zz[ok][i]; g = (Zm > 0) & (np.abs(Zm - zq) < 0.12)
    return Zm[g] - zq[g]
for v in sys.argv[2:]:
    kw = dict(voxel=0.02, max_depth=2.5)
    for item in v.split(",")[1:]:
        k, val = item.split("="); kw[k] = float(val)
    mf = int(kw.pop("min_frames", 4)); al = kw.pop("align", 0); pr = kw.pop("prior_rot", 0.01); pt = kw.pop("prior_trans", 0.02); at = kw.pop("abs_trans", 0.15); ar = kw.pop("abs_rot", 0.03)
    ts = TSDF(**kw)
    D = []; npts = []; Tc = np.eye(4); corr = []
    for j in range(len(Zs)):
        Zj = Zs[j].astype(np.float32)
        Tg = Tc @ Ts[j]                                   # map_from_stage (from earlier frames) applied to this pose
        if j >= 20 and j % 6 == 0:                        # honest test: correction from frames < j only
            P, C, _ = ts.extract(min_frames=mf); npts.append(len(P))
            D.append(resid(P, Zj, Tg))
        Ti = Tg
        if al and j >= 3:
            Ta, info = ts.align(ls, Zj, Tg, prior_rot=pr, prior_trans=pt, tracking=Ts[j], abs_rot=ar, abs_trans=at)
            if Ta is not None:
                Ti = Ta; Tc = Ta @ np.linalg.inv(Ts[j])
        ph = Ts[j][:3, 3]; up = Tc[:3, :3] @ [0, 1, 0]
        corr.append((np.linalg.norm(Tc[:3, :3] @ ph + Tc[:3, 3] - ph), np.degrees(np.arccos(np.clip(up[1], -1, 1))), np.degrees(np.arccos(np.clip((np.trace(Tc[:3, :3]) - 1) / 2, -1, 1)))))
        ts.integrate(ls, Zj, Cs[j], Ti)
    meds = [np.median(d) for d in D]; Dc = np.concatenate(D)
    print(f"{v:28s} bias median {np.median(Dc)*100:+.2f} cm  |d|<3cm {np.mean(np.abs(Dc)<0.03)*100:.1f}%  "
          f"points {int(np.mean(npts))}  corr at head: max {max(c[0] for c in corr)*100:.1f} cm, final {corr[-1][0]*100:.1f} cm; tilt max {max(c[1] for c in corr):.2f} deg; rot final {corr[-1][2]:.2f} deg; final offset at head {np.round((Tc[:3,:3] @ Ts[-1][:3,3] + Tc[:3,3] - Ts[-1][:3,3])*100,1)} cm", flush=True)
