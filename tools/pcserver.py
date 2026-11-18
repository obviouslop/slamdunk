#!/usr/bin/env python3
"""Live point-cloud fusion service.

Reads stereo pairs + poses from /dev/shm/arcgrab (src/arcgrab live, or tools/replay_shm.py),
computes stereo depth (tools/livestereo.py), transforms to the OpenVR Standing universe and fuses
into a voxel map; publishes the confirmed voxels to /dev/shm/pointcloud.bin (tools/pcpub.py) for
the Godot viewer (viewer/).

usage: pcserver.py [--scale 0.5] [--voxel 0.015] [--min-hits 2] [--maxdepth 2.5] [--publish-every 1.0]
                   [--stride 1] [--reset-on-start]
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arcshm import ArcShm
from livestereo import LiveStereo
import pcpub
from fusion import Fuser

ap = argparse.ArgumentParser()
ap.add_argument("--scale", type=float, default=0.5)
ap.add_argument("--voxel", type=float, default=0.015)
ap.add_argument("--min-hits", type=int, default=3)
ap.add_argument("--maxdepth", type=float, default=2.5)
ap.add_argument("--publish-every", type=float, default=1.0)
ap.add_argument("--stride", type=int, default=1, help="use every n-th valid point per frame")
ap.add_argument("--max-pub", type=int, default=400000)
ap.add_argument("--seconds", type=float, default=0, help="stop after N s (0 = run forever)")
ap.add_argument("--pair-tol", type=float, default=0.015, help="max |ts_L - ts_R| (s)")
ap.add_argument("--no-carve", action="store_true", help="disable free-space carving")
ap.add_argument("--cone", type=float, default=12.0, help="paint-mode cone half angle (deg)")
a = ap.parse_args()


CTL_PATH = "/dev/shm/pcctl.json"
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "captures", "sessions")


def read_ctl():
    """Viewer control state (viewer/pc_main.gd): mode auto|paint, painting, aim pose (STAGE), commands."""
    try:
        if time.time() - os.path.getmtime(CTL_PATH) > 2.0:   # viewer writes at 10 Hz; stale -> ignore
            return None
        with open(CTL_PATH) as f:
            return json.load(f)
    except Exception:
        return None


def cone_mask(Pw, aim, half_angle_deg):
    A = np.asarray(aim, float).reshape(3, 4); o = A[:, 3]; d = -A[:, 2]       # OpenXR aim: -Z forward
    v = Pw - o; n = np.linalg.norm(v, axis=1)
    return (v @ d) > np.cos(np.radians(half_angle_deg)) * np.maximum(n, 1e-6)


def save_ply(P, C, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rec = np.empty(len(P), dtype=[("p", "<f4", 3), ("c", "u1", 3)]); rec["p"] = P; rec["c"] = np.clip(C * 255, 0, 255)
    with open(path, "wb") as fh:
        fh.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(P)}\nproperty float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n".encode()); fh.write(rec.tobytes())


while True:   # wait for the frame source (arcgrab or replay_shm.py)
    try: s = ArcShm(); break
    except (FileNotFoundError, RuntimeError, ValueError): print("waiting for /dev/shm/arcgrab ...", flush=True); time.sleep(1)
ls = LiveStereo(scale=a.scale, max_depth=a.maxdepth)
fu = Fuser(voxel=a.voxel, min_hits=a.min_hits, carve=not a.no_carve); last_cmd_seq = None; mode = "auto"
last_frame = -1; t_pub = 0; n_frames = 0; t0 = time.monotonic(); tstat = []; tmerge = []
print(f"pcserver: scale {a.scale} ({ls.size[0]} px), voxel {a.voxel*100:.1f} cm, min hits {a.min_hits}", flush=True)
while not a.seconds or time.monotonic() - t0 < a.seconds:
    try:
        mL, rawL = s.latest(0, after_frame=last_frame, timeout=5.0)
        mR, rawR = s.latest(1, after_frame=mL["frame"] - 2, timeout=1.0)
    except TimeoutError:
        print("waiting for frames ...", flush=True); continue
    last_frame = mL["frame"]
    if not mL["pose_valid"] or abs(mL["ts"] - mR["ts"]) > a.pair_tol:
        continue
    t1 = time.monotonic()
    ctl = read_ctl()
    if ctl is None:
        mode = "auto"
    if ctl:
        mode = ctl.get("mode", "auto")
        if ctl.get("cmd_seq") != last_cmd_seq:
            if last_cmd_seq is not None:
                if ctl.get("cmd") == "reset":
                    fu.reset(); print("map reset", flush=True)
                elif ctl.get("cmd") == "save":
                    Ps, Cs = fu.confirmed(); fn = os.path.join(SAVE_DIR, time.strftime("pc_%Y%m%d_%H%M%S.ply"))
                    save_ply(Ps, np.clip(Cs[:, ::-1], 0, 1), fn); print(f"saved {len(Ps)} points -> {fn}", flush=True)
            last_cmd_seq = ctl.get("cmd_seq")
    painting = bool(ctl and ctl.get("painting") and ctl.get("aim"))
    if mode == "paint" and not painting:
        continue                                      # paint mode: fuse only while the trigger is held
    P, C, disp = ls.points(rawL, rawR)
    if a.stride > 1:
        P, C = P[::a.stride], C[::a.stride]
    H = np.eye(4); H[:3, :] = mL["pose"]; Wc = H @ ls.head_from_camL
    Pw = P @ Wc[:3, :3].T + Wc[:3, 3]
    if mode == "paint":
        m = cone_mask(Pw, ctl["aim"], a.cone); Pw, C = Pw[m], C[m]
    t2 = time.monotonic(); fu.add_frame(Pw, C, ls, disp, Wc)
    n_frames += 1; tstat.append(time.monotonic() - t1); tmerge.append(time.monotonic() - t2)
    if time.monotonic() - t_pub > a.publish_every:
        t_pub = time.monotonic()
        Pc, Cc = fu.confirmed()
        n = pcpub.publish(Pc, np.clip(Cc[:, ::-1], 0, 1), max_points=a.max_pub)   # BGR -> RGB
        print(f"frame {last_frame}: {n_frames} fused, {np.mean(tstat[-10:])*1000:.0f} ms/frame, "
              f"fuse {np.mean(tmerge[-10:])*1000:.0f} ms, voxels {len(fu.keys)} ({n} confirmed published), mode {mode}", flush=True)
