#!/usr/bin/env python3
"""Replay a live recording (tools/liverec.py) into /dev/shm/arcgrab, in arcgrab's format, so the
live pipeline (pcserver.py) and viewer can be developed without touching the cameras.

Poses: the per-frame poses recorded by arcgrab (default), or with --delta from the recording's 500 Hz
history (hmd_500hz.txt) at the frame's mid-exposure time + delta (use --delta -0.340 for recordings
made before the V4L2 timestamp fix: liverec1-3, delay1/2).

usage: replay_shm.py RECDIR [--delta S] [--speed 1.0] [--loop]
"""
import argparse, json, mmap, os, struct, sys, time
import numpy as np
from scipy.spatial.transform import Rotation as Rot, Slerp

ap = argparse.ArgumentParser(); ap.add_argument("rec"); ap.add_argument("--delta", type=float, default=None)
ap.add_argument("--speed", type=float, default=1.0); ap.add_argument("--loop", action="store_true")
a = ap.parse_args()
LT = 4.36e-6
recs = [json.loads(l) for l in open(os.path.join(a.rec, "frames.jsonl"))]
Hh = np.loadtxt(os.path.join(a.rec, "hmd_500hz.txt")); Hh = Hh[Hh[:, 1] > 0]
tp = Hh[:, 0]; sl = Slerp(tp, Rot.from_quat(Hh[:, [3, 4, 5, 2]]))
def pose_at(t):
    t = float(np.clip(t, tp[0], tp[-1])); M = np.zeros((3, 4), np.float32)
    M[:, :3] = sl([t]).as_matrix()[0]; M[:, 3] = [np.interp(t, tp, Hh[:, 6 + j]) for j in range(3)]; return M
slot = os.path.getsize(os.path.join(a.rec, f"{recs[0]['i']:05d}_left.y10p"))
HDR = struct.Struct("<8s8I"); CAM = struct.Struct("<Q4IdQ12f4I"); CAM_OFF = 40
fd = os.open("/dev/shm/arcgrab", os.O_RDWR | os.O_CREAT | os.O_TRUNC, 0o644)
os.ftruncate(fd, 4096 + 4 * slot); m = mmap.mmap(fd, 4096 + 4 * slot); os.close(fd)
m[0:HDR.size] = HDR.pack(b"ARCGRAB1", 2, 2464, 3088, 64, 0, slot, 2, 0)
state = [dict(seq=0, slot=0) for _ in range(2)]
def publish(k, buf, meta, pose):
    st = state[k]; s = (st["slot"] + 1) & 1; off = CAM_OFF + k * CAM.size
    st["seq"] += 1; struct.pack_into("<Q", m, off, st["seq"])                    # odd: writing
    m[4096 + (k * 2 + s) * slot: 4096 + (k * 2 + s) * slot + len(buf)] = buf
    st["slot"] = s; st["seq"] += 1
    m[off:off + CAM.size] = CAM.pack(st["seq"], s, meta["exp_lines"], meta["gain_code"], len(buf), meta["ts"], meta["frame"],
                                     *pose.ravel().tolist(), 1, meta.get("buf_flags", 0), meta.get("dgain", 256), 0)
print(f"replaying {len(recs)} pairs from {a.rec} (slot {slot} B) into /dev/shm/arcgrab", flush=True)
while True:
    t_wall0 = time.monotonic(); t_rec0 = recs[0]["left"]["ts"]
    for r in recs:
        tw = t_wall0 + (r["left"]["ts"] - t_rec0) / a.speed
        time.sleep(max(0.0, tw - time.monotonic()))
        for k, name in ((0, "left"), (1, "right")):
            meta = r[name]; tmid = meta["ts"] - 2464 * LT / 2 - meta["exp_lines"] * LT / 2
            buf = open(os.path.join(a.rec, f"{r['i']:05d}_{name}.y10p"), "rb").read()
            if a.delta is None and meta.get("pose_valid") and "pose" in meta:
                pose = np.asarray(meta["pose"], np.float32)
            else:
                pose = pose_at(tmid + (a.delta if a.delta is not None else -0.340))
            publish(k, buf, meta, pose)
    if not a.loop:
        break
print("replay done")
