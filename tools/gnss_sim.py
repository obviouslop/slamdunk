#!/usr/bin/env python3
"""Simulated GNSS log for testing tools/gnssd.py without hardware: antenna on a recorded head trajectory
(hmd_500hz.txt), a known georeference, RTK-like noise and variable serial latency.
usage: gnss_sim.py hmd_500hz.txt out.log [--rate 5] [--noise 0.015] [--yaw-deg 37] [--lever 0 0.12 0.05]
Prints the true georef (compare with: gnssd.py --offline out.log --pose-file hmd_500hz.txt)."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geo
from gnssd import Poses

ap = argparse.ArgumentParser(); ap.add_argument("poses"); ap.add_argument("out")
ap.add_argument("--rate", type=float, default=5); ap.add_argument("--noise", type=float, default=0.015)
ap.add_argument("--yaw-deg", type=float, default=37.0); ap.add_argument("--lever", type=float, nargs=3, default=[0, 0.12, 0.05])
ap.add_argument("--lat0", type=float, default=48.2082); ap.add_argument("--lon0", type=float, default=16.3738); ap.add_argument("--h0", type=float, default=230.0)
ap.add_argument("--q", type=int, default=4)
a = ap.parse_args()
rng = np.random.default_rng(1)
P = Poses(); P.load(a.poses)
truth = {"lat0": a.lat0, "lon0": a.lon0, "h0": a.h0, "yaw": np.radians(a.yaw_deg), "t": [3.0, -7.0, 1.2]}
UTC_OFF = 30000.0 - P.buf[0, 0]            # utc_of_day = t_mono + UTC_OFF
def cs(b):
    c = 0
    for ch in b.encode(): c ^= ch
    return f"${b}*{c:02X}"
def dm(x, pos, neg, w):
    h = pos if x >= 0 else neg; x = abs(x); d = int(x); return f"{d:0{w}d}{(x - d) * 60:08.5f}", h
out = open(a.out, "w"); n = 0
t = np.ceil(P.buf[0, 0] * a.rate) / a.rate
while t < P.buf[P.n - 1, 0]:
    pr = P.at(t)
    if pr is not None:
        R, p = pr
        ant = R @ np.array(a.lever) + p
        enu = geo.stage_to_enu(ant, truth) + rng.normal(0, a.noise, 3) * [1, 1, 2]
        lat, lon, h = geo.enu_to_lla(enu, a.lat0, a.lon0, a.h0)
        utc = t + UTC_OFF; hh = int(utc // 3600); mm = int(utc % 3600 // 60); ss = utc % 60
        ts = f"{hh:02d}{mm:02d}{ss:05.2f}"
        la, lh = dm(float(lat), "N", "S", 2); lo, oh = dm(float(lon), "E", "W", 3)
        sep = 45.0
        t_rx = t + 0.05 + rng.exponential(0.04)
        out.write(f"{t_rx:.6f} {cs(f'GNGST,{ts},0.01,0.01,0.01,0,{a.noise:.3f},{a.noise:.3f},{2*a.noise:.3f}')}\n")
        out.write(f"{t_rx:.6f} {cs(f'GNGGA,{ts},{la},{lh},{lo},{oh},{a.q},18,0.7,{float(h)-sep:.3f},M,{sep:.1f},M,1.0,0000')}\n")
        n += 1
    t += 1 / a.rate
print(f"{n} fixes -> {a.out}; truth:", json.dumps({**truth, "yaw_deg": a.yaw_deg}))
