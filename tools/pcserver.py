#!/usr/bin/env python3
"""Live point-cloud fusion service.

Reads stereo pairs + poses from /dev/shm/arcgrab (src/arcgrab live, or tools/replay_shm.py), computes
stereo depth (tools/livestereo.py, RAW10 binning in C++), fuses into a sparse TSDF (src/tsdf.cpp via
tools/tsdf.py) in the OpenVR Standing universe (== OpenXR STAGE) and publishes the extracted surface
points as tiles (tools/pcpub.py) for the Godot viewer (viewer/).

Viewer interface (all in /dev/shm, JSON, written atomically):
  pcctl.json    viewer -> server, 10-30 Hz: {t_ms, scan: auto|paint|off, painting, aim[12], cmd, cmd_seq, name}
  pchit.json    server -> viewer, 30 Hz: first TSDF surface along the right controller aim ray
                {seq, hit, dist, p[3], n[3]}   (picking for anchors / boxes / paths)
  pcalign.json  server -> viewer, per fused frame: {seq, map_from_stage[12] (3x4 row-major)}: drift correction.
                The map (cloud, raycast hits, saved PLY) lives in the MAP frame; the viewer shows it through
                stage_from_map = inverse(map_from_stage) so old scans follow tracking drift.
  pcstatus.json server -> viewer, 1 Hz: {fps, points, blocks, scan, saved, still, stills}
  commands (cmd + incremented cmd_seq): reset | save (name) | still (stereo still -> captures/stills/, tools/stills.py)

usage: pcserver.py [--scale 0.35] [--voxel 0.02] [--min-frames 4] [--maxdepth 2.5] [--publish-every 1.0]
"""
import argparse, json, os, sys, threading, time
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arcshm import ArcShm
from livestereo import LiveStereo
from tsdf import TSDF, bin_raw10, cone_pixels
import ptpub
import pcpub
import stills

ap = argparse.ArgumentParser()
ap.add_argument("--scale", type=float, default=0.35)
ap.add_argument("--voxel", type=float, default=0.02)
ap.add_argument("--min-frames", type=int, default=6, help="frames a surface must be seen in before it is shown")
ap.add_argument("--carve-w", type=float, default=0.3, help="weight of free-space votes (balanced against --back-w)")
ap.add_argument("--back-k", type=float, default=8.0, help="solid votes up to (1+k) bands behind each measured surface")
ap.add_argument("--back-w", type=float, default=3.0, help="weight of the solid votes (removes surfaces left behind by drift)")
ap.add_argument("--no-align", action="store_true", help="disable drift correction (frame-to-map registration)")
ap.add_argument("--abs-trans", type=float, default=0.08, help="1-sigma of the total drift correction (m)")
ap.add_argument("--maxdepth", type=float, default=2.5, help="surfaces are only taken up to this distance (m)")
ap.add_argument("--carve-depth", type=float, default=4.0, help="free-space carving up to this distance (m)")
ap.add_argument("--min-depth", type=float, default=0.35)
ap.add_argument("--publish-every", type=float, default=1.0)
ap.add_argument("--tile", type=float, default=1.0, help="tile size (m) of the published cloud")
ap.add_argument("--max-pub", type=int, default=300000)
ap.add_argument("--threads", type=int, default=3, help="OpenCV / TSDF worker threads (leave cores for XR)")
ap.add_argument("--passthrough", action="store_true", help="always publish depth-passthrough frames (else only when the viewer asks)")
ap.add_argument("--seconds", type=float, default=0, help="stop after N s (0 = run forever)")
ap.add_argument("--pair-tol", type=float, default=0.015, help="max |ts_L - ts_R| (s)")
ap.add_argument("--cone", type=float, default=12.0, help="paint-mode cone half angle (deg)")
a = ap.parse_args()

SHM = "/dev/shm"
CTL_PATH, HIT_PATH, STATUS_PATH = f"{SHM}/pcctl.json", f"{SHM}/pchit.json", f"{SHM}/pcstatus.json"
ALIGN_PATH = f"{SHM}/pcalign.json"
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "captures", "sessions")
cv2.setNumThreads(a.threads)


def write_json(path, d):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f)
    os.replace(tmp, path)


def read_ctl():
    """Viewer control state (viewer/pc_main.gd); None if the viewer is not running (stale file)."""
    try:
        if time.time() - os.path.getmtime(CTL_PATH) > 2.0:
            return None
        with open(CTL_PATH) as f:
            return json.load(f)
    except Exception:
        return None


lock = threading.Lock()
ts = TSDF(voxel=a.voxel, max_depth=a.maxdepth, carve_depth=a.carve_depth, threads=a.threads,
          carve_w=a.carve_w, back_k=a.back_k, back_w=a.back_w)
Tc = np.eye(4)            # map_from_stage drift correction
align_seq = 0


def write_align():
    global align_seq
    align_seq += 1
    try: write_json(ALIGN_PATH, {"seq": align_seq, "map_from_stage": Tc[:3, :4].ravel().round(6).tolist()})
    except OSError: pass


def to_map(A34):
    """3x4 STAGE pose/aim -> MAP frame"""
    M = np.eye(4); M[:3, :4] = np.asarray(A34, float).reshape(3, 4); return (Tc @ M)[:3, :4]
stop = False


def pick_loop():
    """30 Hz: raycast the right-controller aim ray into the TSDF for the viewer's placement reticle."""
    seq = 0; last_t = None
    while not stop:
        time.sleep(1 / 30)
        ctl = read_ctl()
        if not ctl or len(ctl.get("aim") or []) != 12 or ctl.get("t_ms") == last_t:
            continue
        last_t = ctl.get("t_ms")
        A = to_map(ctl["aim"])
        with lock:
            h = ts.raycast(A[:, 3], -A[:, 2], tmin=0.05, tmax=8.0, min_frames=max(2, a.min_frames - 1))
        seq += 1
        d = {"seq": seq, "aim_t_ms": last_t, "hit": h is not None}
        if h is not None:
            Ti = np.linalg.inv(Tc)                       # report in STAGE (dist is frame independent)
            d.update(p=(Ti[:3, :3] @ h[0] + Ti[:3, 3]).tolist(), n=(Ti[:3, :3] @ h[1]).tolist(), dist=h[2])
        try: write_json(HIT_PATH, d)
        except OSError: pass


while True:   # wait for the frame source (arcgrab or replay_shm.py)
    try: s = ArcShm(); break
    except (FileNotFoundError, RuntimeError, ValueError): print("waiting for /dev/shm/arcgrab ...", flush=True); time.sleep(1)
ls = LiveStereo(scale=a.scale, max_depth=a.maxdepth, min_depth=a.min_depth)
threading.Thread(target=pick_loop, daemon=True).start()
last_cmd_seq = None; scan = "auto"; saved = ""; still = ""; n_stills = 0
last_frame = -1; n_pt = 0; t_frame_status = 0.0; t_pub = 0; n_frames = 0; t0 = time.monotonic(); tst = []; tfu = []; t_fr = []
npub = 0
print(f"pcserver: scale {a.scale} ({ls.size[0]} px), voxel {a.voxel*100:.1f} cm, min frames {a.min_frames}, "
      f"depth {a.min_depth}-{a.maxdepth} m", flush=True)


def take_still():
    global still, n_stills
    try:
        still = stills.save_still(s, ls); n_stills += 1
        print(f"still -> {still}", flush=True)
    except Exception as e:
        still = f"failed: {e}"; print("still failed:", e, flush=True)
    write_status()


def write_status(fps=None):
    write_json(STATUS_PATH, dict(fps=round(fps if fps is not None else last_fps, 1), points=npub, blocks=ts.nblocks,
                                 scan=scan, saved=saved, still=still, stills=n_stills, frame=last_frame))


last_fps = 0.0


def handle_cmd(ctl):
    global last_cmd_seq, saved, Tc
    if ctl.get("cmd_seq") == last_cmd_seq:
        return
    if last_cmd_seq is not None:
        c = ctl.get("cmd")
        if c == "reset":
            with lock: ts.reset(); Tc = np.eye(4)
            write_align()
            print("map reset", flush=True)
        elif c == "save":
            name = os.path.basename(ctl.get("name") or time.strftime("scan_%Y%m%d_%H%M%S"))
            with lock: P, C, w = ts.extract(min_frames=a.min_frames)
            fn = os.path.join(SAVE_DIR, name + ".ply"); pcpub.write_ply(fn, P, C); saved = fn
            print(f"saved {len(P)} points -> {fn}", flush=True)
        elif c == "still":
            threading.Thread(target=take_still, daemon=True).start()
    last_cmd_seq = ctl.get("cmd_seq")


try:
    while not a.seconds or time.monotonic() - t0 < a.seconds:
        ctl = read_ctl()
        scan = ctl.get("scan", ctl.get("mode", "auto")) if ctl else "auto"
        if ctl:
            handle_cmd(ctl)
        painting = bool(ctl and ctl.get("painting") and len(ctl.get("aim") or []) == 12)
        want_pt = a.passthrough or bool(ctl and ctl.get("passthrough"))
        fuse = not (scan == "off" or (scan == "paint" and not painting))
        if not fuse and not want_pt:
            time.sleep(0.03)                  # idle, keep polling the viewer for commands
            if time.monotonic() - t_pub > a.publish_every:
                t_pub = time.monotonic()
                write_status(0.0)
            continue
        try:
            mL, bL = s.latest_packed(0, after_frame=last_frame, timeout=5.0)
            mR, bR = s.latest_packed(1, after_frame=mL["frame"] - 2, timeout=1.0)
        except TimeoutError:
            print("waiting for frames ...", flush=True); continue
        last_frame = mL["frame"]
        if time.monotonic() - t_frame_status > 0.2:      # HUD frame stamp (arcgrab frame no. = frames.jsonl 'frame')
            t_frame_status = time.monotonic(); write_status()
        if not mL["pose_valid"] or abs(mL["ts"] - mR["ts"]) > a.pair_tol:
            continue
        t1 = time.monotonic()
        kw = dict(W=s.line_px, stride=s.line_bytes, skip=s.skip, black=float(s.black), threads=a.threads)
        H = (min(len(bL), len(bR)) // s.line_bytes) - s.skip
        cL = bin_raw10(bL, H=min(H, s.line_px), **kw); cR = bin_raw10(bR, H=min(H, s.line_px), **kw)
        Z, rgb, _ = ls.depth(cL, cR, far=a.carve_depth, half=True)
        Tt = ls.world_from_rect(mL["pose"])            # tracking (STAGE)
        T = Tc @ Tt                                     # predicted MAP pose
        if want_pt:                                     # depth-correct passthrough (STAGE frame, raw tracking pose)
            n_pt += 1
            ptpub.publish(ptpub.pack(Z, rgb, ls.f, ls.P1[0, 2], ls.P1[1, 2], Tt, mL["ts"], mL["frame"], n_pt))
        if not fuse:
            continue
        t2 = time.monotonic()
        with lock:
            if not a.no_align and n_frames >= 3:
                Ta, info = ts.align(ls, Z, T, tracking=Tt, abs_trans=a.abs_trans)
                if Ta is not None:
                    T = Ta; Tc = Ta @ np.linalg.inv(Tt)
            mask = cone_pixels(ls, Z, T, to_map(ctl["aim"]).ravel().tolist(), a.cone) if scan == "paint" else None
            ts.integrate(ls, Z, rgb, T, mask=mask)
        write_align()
        n_frames += 1; tst.append(t2 - t1); tfu.append(time.monotonic() - t2); t_fr.append(time.monotonic())
        if time.monotonic() - t_pub > a.publish_every:
            t_pub = time.monotonic()
            with lock:
                P, C, w = ts.extract(min_frames=a.min_frames)
            npub = pcpub.publish(P, C, max_points=a.max_pub, tile=a.tile)
            fps = (len(t_fr[-20:]) - 1) / max(t_fr[-1] - t_fr[-20:][0], 1e-3) if len(t_fr) > 1 else 0.0
            last_fps = fps; write_status(fps)
            print(f"frame {last_frame}: {n_frames} fused ({fps:.1f}/s), stereo {np.mean(tst[-10:])*1000:.0f} ms, "
                  f"tsdf {np.mean(tfu[-10:])*1000:.0f} ms, publish {(time.monotonic()-t_pub)*1000:.0f} ms, "
                  f"{ts.nblocks} blocks, {npub} points, scan {scan}", flush=True)
finally:
    stop = True
