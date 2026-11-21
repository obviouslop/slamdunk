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
import zones as zonemod
import pcpub
import stills
import handmask
import floor as floormod
import fiducials
from origin import OriginTracker
from farstereo import FarStereo, LT, _rot_angle
import debugcap
from tags import TagWatcher
from planes import PlaneTracker
from splatcap import SplatCapture

ap = argparse.ArgumentParser()
ap.add_argument("--scale", type=float, default=0.35)
ap.add_argument("--voxel", type=float, default=0.02)
ap.add_argument("--min-frames", type=int, default=6, help="frames a surface must be seen in before it is shown")
ap.add_argument("--carve-w", type=float, default=0.3, help="weight of free-space votes (balanced against --back-w)")
ap.add_argument("--back-k", type=float, default=8.0, help="solid votes up to (1+k) bands behind each measured surface")
ap.add_argument("--back-w", type=float, default=3.0, help="weight of the solid votes (removes surfaces left behind by drift)")
ap.add_argument("--maxdepth", type=float, default=2.5, help="surfaces are only taken up to this distance (m)")
ap.add_argument("--carve-depth", type=float, default=4.0, help="free-space carving up to this distance (m)")
ap.add_argument("--min-depth", type=float, default=0.35)
ap.add_argument("--publish-every", type=float, default=1.0)
ap.add_argument("--tile", type=float, default=1.0, help="tile size (m) of the published cloud")
ap.add_argument("--max-pub", type=int, default=300000)
ap.add_argument("--threads", type=int, default=3, help="OpenCV / TSDF worker threads (leave cores for XR)")
ap.add_argument("--underfloor-cm", type=float, default=7.0, help="drop points more than this far below a FLAT floor (mirror images of windows in glossy floors); 0 = off")
ap.add_argument("--occ-hz", type=float, default=15.0, help="max rate of the low-res occlusion depth frames")
ap.add_argument("--conf", action="store_true", help="weight TSDF observations by per-pixel confidence (texture / depth-edge); A/B showed no gain on floor thickness, see re/test_conf.py")
ap.add_argument("--sat", type=float, default=1.4, help="point colour saturation boost (1 = raw white-balanced sensor colours)")
ap.add_argument("--color-gamma", type=float, default=1.0, help="point colour mid-tone lift: <1 brightens (e.g. 0.7)")
ap.add_argument("--no-planes", action="store_true", help="no live coarse plane fitting (tools/planes.py, viewer PLANES toggle)")
ap.add_argument("--plane-hz", type=float, default=0.5, help="plane fitting rate (it extracts the whole fused cloud each time)")
ap.add_argument("--no-splat", action="store_true", help="no capture-for-splatting mode (tools/splatcap.py, viewer SPLAT toggle)")
ap.add_argument("--splat-min-dist", type=float, default=0.06, help="splat capture: a pair is stored when the view is this far (m, +0.5 m/rad of viewing angle) from all stored ones")
ap.add_argument("--splat-gb", type=float, default=6.0, help="splat capture stops by itself at this size")
ap.add_argument("--no-far", action="store_true", help="no wide-baseline far-range depth (tools/farstereo.py)")
ap.add_argument("--far-mb", type=float, default=100.0, help="size of the on-disk keyframe ring buffer for the far-range depth (MB)")
ap.add_argument("--far-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "captures", "scratch", "far"))
ap.add_argument("--far-blur", type=float, default=1.0, help="keyframes only below this estimated motion blur (px)")
ap.add_argument("--blur", type=float, default=1.5, help="fuse only pairs whose estimated motion blur (px, rectified image; f*t_exp*(omega+v/z) "
                "from the tracked pose change between frames) is below this; 0 = off (evaluation: NOTES.md 2026-11-21)")
ap.add_argument("--blur-max-wait", type=float, default=1.0, help="after this many s without a sharp pair, fuse the next one anyway")
ap.add_argument("--far-zmin", type=float, default=2.5, help="far points start at this range (m)")
ap.add_argument("--no-handmask", action="store_true", help="do not keep hands / controllers out of the fused cloud")
ap.add_argument("--no-origin", action="store_true", help="do not look for the origin marker (AprilTag 41h12 id 4)")
ap.add_argument("--apply-origin-corr", action="store_true", help="correct the SteamVR pose with the origin marker (fades out over 3 s after it was last seen)")
ap.add_argument("--origin-mm", type=float, help="printed edge of the origin marker in mm (default fiducials.ORIGIN_MM)")
ap.add_argument("--origin-frame", action="store_true", help="save PLYs in the origin-marker frame (x right, y up, z out of the paper) "
                "when the marker is locked; default: STAGE coordinates, with the marker pose in the <name>.ply.json sidecar")
ap.add_argument("--passthrough", action="store_true", help="always publish depth-passthrough frames (else only when the viewer asks)")
ap.add_argument("--seconds", type=float, default=0, help="stop after N s (0 = run forever)")
ap.add_argument("--pair-tol", type=float, default=0.015, help="max |ts_L - ts_R| (s)")
ap.add_argument("--cone", type=float, default=12.0, help="paint-mode cone half angle (deg)")
a = ap.parse_args()

SHM = "/dev/shm"
CTL_PATH, HIT_PATH, STATUS_PATH = f"{SHM}/pcctl.json", f"{SHM}/pchit.json", f"{SHM}/pcstatus.json"
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
def to_map(A34):
    """3x4 STAGE pose/aim (the map frame is STAGE)"""
    M = np.eye(4); M[:3, :4] = np.asarray(A34, float).reshape(3, 4); return M[:3, :4]
stop = False
zf = zonemod.ZoneFile()       # inclusion / exclusion / label zones from the viewer (MAP frame)


def extract_kept():
    """Fused surface points with the zones applied -> P, C, labels, acquisition time (s), view distance (m)"""
    with lock:
        P, C, w, t, d = ts.extract(min_frames=a.min_frames, attrs=True)
    if a.underfloor_cm > 0 and len(P):
        uf, fy, fs = floormod.underfloor_mask(P, tol=a.underfloor_cm / 100)
        if uf.any():
            k = ~uf; P, C, w, t, d = P[k], C[k], w[k], t[k], d[k]
    if far is not None:                         # far-range points from wide-baseline keyframe pairs
        Pf, Cf, rf, sf, tf = far.points()
        if len(Pf):
            P = np.vstack([P, Pf]); C = np.vstack([C, Cf]); w = np.concatenate([w, np.ones(len(Pf), np.float32)])
            t = np.concatenate([t, tf]); d = np.concatenate([d, rf])
    if not zf.zones:
        return P, C, np.zeros(len(P), np.uint8), t, d
    keep, lab = zonemod.evaluate(P, zf.zones)
    return P[keep], C[keep], lab[keep], t[keep], d[keep]


def publish_cloud():
    P, C = extract_kept()[:2]
    return pcpub.publish(P, C, max_points=a.max_pub, tile=a.tile)


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
        t0r = 0.05
        for _ in range(6):            # skip surfaces removed by zones (continue the ray behind them)
            with lock:
                h = ts.raycast(A[:, 3], -A[:, 2], tmin=t0r, tmax=8.0, min_frames=max(2, a.min_frames - 1))
            if h is None or not zf.zones or zonemod.point_kept(h[0], zf.zones):
                break
            t0r = h[2] + 0.04; h = None
        seq += 1
        d = {"seq": seq, "aim_t_ms": last_t, "hit": h is not None}
        if h is not None:
            d.update(p=np.asarray(h[0]).tolist(), n=np.asarray(h[1]).tolist(), dist=h[2])
        try: write_json(HIT_PATH, d)
        except OSError: pass


while True:   # wait for the frame source (arcgrab or replay_shm.py)
    try: s = ArcShm(); break
    except (FileNotFoundError, RuntimeError, ValueError): print("waiting for /dev/shm/arcgrab ...", flush=True); time.sleep(1)
ls = LiveStereo(scale=a.scale, max_depth=a.maxdepth, min_depth=a.min_depth)
ls.sat, ls.gamma = a.sat, a.color_gamma
if a.origin_mm: fiducials.BOARDS["origin"]["tag_mm"] = a.origin_mm
occ_state = {"on": False}


def occ_loop():
    """Occlusion mask producer: its own frame reader and a cheap low-res stereo pass at up to --occ-hz, independent of the (slow)
    fusion loop, so the mask is fresh (speed over accuracy). Colour is not published (8x8 placeholder)."""
    so = ArcShm(); lso = LiveStereo(scale=0.2, max_depth=a.maxdepth, min_depth=a.min_depth); last = -1; seq = 0
    kw = dict(W=so.line_px, stride=so.line_bytes, skip=so.skip, black=float(so.black), threads=1)
    while not stop:
        if not occ_state["on"]:
            time.sleep(0.05); continue
        t0 = time.monotonic()
        try:
            mL, bL = so.latest_packed(0, after_frame=last, timeout=0.5)
            mR, bR = so.latest_packed(1, after_frame=mL["frame"] - 2, timeout=0.5)
        except TimeoutError:
            continue
        last = mL["frame"]
        if not mL["pose_valid"] or abs(mL["ts"] - mR["ts"]) > a.pair_tol: continue
        try:
            H = (min(len(bL), len(bR)) // so.line_bytes) - so.skip
            cL = bin_raw10(bL, H=min(H, so.line_px), **kw); cR = bin_raw10(bR, H=min(H, so.line_px), **kw)
            Zo, rgbo, _ = lso.depth(cL, cR, far=a.carve_depth, half=True)
            seq += 1
            ptpub.publish(ptpub.pack(Zo, rgbo, lso.f, lso.P1[0, 2], lso.P1[1, 2], lso.world_from_rect(mL["pose"]), mL["ts"], mL["frame"],
                                     (int(time.time() * 1000) + seq) & 0xFFFFFFF, colour_size=8, block=3))
        except Exception as e:
            print("occlusion thread:", e, flush=True); time.sleep(0.5)
        time.sleep(max(0.0, 1.0 / a.occ_hz - (time.monotonic() - t0)))


threading.Thread(target=occ_loop, daemon=True).start()
origin = None if a.no_origin else OriginTracker(ls, path=f"{SHM}/pcorigin.json")
far = None if a.no_far else FarStereo(ls, a.far_dir, max_bytes=a.far_mb * 1e6, z_min=a.far_zmin, blur_px=a.far_blur)
if far: far.start()
splatcap = None if a.no_splat else SplatCapture(root=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "captures", "splat"), min_dist=a.splat_min_dist, max_gb=a.splat_gb)
if splatcap: splatcap.start()
planes_state = {"on": False, "reset": False}
plane_tr = None if a.no_planes else PlaneTracker(path=f"{SHM}/pcplanes.json")


def plane_loop():
    """Fits planes to the fused cloud at --plane-hz while the viewer's PLANES toggle is on (ctl.planes), only when the map changed."""
    last = None
    while not stop:
        time.sleep(1.0 / max(a.plane_hz, 0.05))
        if plane_tr is None or not planes_state["on"]: last = None; continue
        if planes_state["reset"]: planes_state["reset"] = False; plane_tr.clear(); last = None
        key = (ts.frames, len(zf.zones))
        if key == last: continue
        try:
            plane_tr.update(extract_kept()[0]); last = key
        except Exception as e:
            print("planes:", e, flush=True)


threading.Thread(target=plane_loop, daemon=True).start()
tagwatch = TagWatcher(ls, origin, path=f"{SHM}/pctags.json")      # fiducials + QR codes -> pctags.json (viewer labels them), also feeds the origin
threading.Thread(target=pick_loop, daemon=True).start()
last_cmd_seq = None; scan = "auto"; saved = ""; still = ""; n_stills = 0
last_frame = -1; n_pt = 0; t_frame_status = 0.0; t_pub = 0; n_frames = 0; t0 = time.monotonic(); tst = []; tfu = []; t_fr = []
npub = 0
prev_pose = None; z_med = 1.5; t_last_fused = 0.0; n_blurry = 0
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


dbg_state = {"armed": False, "last": "", "n": 0}
DEBUG_TRIGGER = f"{SHM}/pcdebug"      # `touch /dev/shm/pcdebug` arms a debug capture without the viewer


def take_debug(pair, bufs, info):
    try:
        dbg_state["last"] = debugcap.save(s, ls, pair, bufs, info); dbg_state["n"] += 1
        print(f"debug capture -> {dbg_state['last']}", flush=True)
    except Exception as e:
        dbg_state["last"] = f"failed: {e}"; print("debug capture failed:", e, flush=True)
    write_status()


def write_status(fps=None):
    write_json(STATUS_PATH, dict(fps=round(fps if fps is not None else last_fps, 1), points=npub, blocks=ts.nblocks,
                                 scan=scan, saved=saved, still=still, stills=n_stills, frame=last_frame,
                                 debug=dbg_state["last"], debugs=dbg_state["n"],
                                 origin=origin.hud() if origin else "", far=far.hud() if far else "",
                                 splat=splatcap.hud() if splatcap else "", splat_on=bool(splatcap and splatcap.enabled), splat_dir=(splatcap.dir or "") if splatcap else "", splat_n=splatcap.n if splatcap else 0,
                                 splat_prog=(-1.0 if not splatcap or splatcap.export is None else (splatcap.progress() if splatcap.export.poll() is None else 1.0)) if splatcap else -1.0))


last_fps = 0.0


def handle_cmd(ctl):
    global last_cmd_seq, saved
    if ctl.get("cmd_seq") == last_cmd_seq:
        return
    if last_cmd_seq is not None:
        c = ctl.get("cmd")
        if c == "reset":
            with lock: ts.reset()
            planes_state["reset"] = True
            if far is not None: far.clear()
            print("map reset", flush=True)
        elif c == "save":
            name = os.path.basename(ctl.get("name") or time.strftime("scan_%Y%m%d_%H%M%S"))
            P, C, L, tt, dd = extract_kept()
            fn = os.path.join(SAVE_DIR, name + ".ply")
            Po = origin.to_origin_frame(P) if origin else None
            pcpub.write_ply(fn, P if Po is None or not a.origin_frame else Po, C, L if L.any() else None, t=tt, dist=dd); saved = fn
            if origin and origin.stage_from_origin is not None:            # ties the cloud to the real world (re-usable across sessions)
                json.dump(dict(frame="origin" if a.origin_frame else "stage", origin_id=fiducials.ORIGIN_ID,
                               origin_mm=fiducials.BOARDS["origin"]["tag_mm"], stage_from_origin=origin.stage_from_origin.tolist(),
                               marker_frame="x right, y up, z out of the paper, origin at the marker centre",
                               time_range=[float(tt.min()), float(tt.max())] if len(tt) else None), open(fn + ".json", "w"), indent=1)
            print(f"saved {len(P)} points -> {fn}", flush=True)
        elif c == "splat_export":
            if splatcap: splatcap.start_export()
        elif c == "still":
            threading.Thread(target=take_still, daemon=True).start()
        elif c == "debug":
            dbg_state["armed"] = True; print("debug capture armed (next processed pair)", flush=True)
    last_cmd_seq = ctl.get("cmd_seq")


try:
    while not a.seconds or time.monotonic() - t0 < a.seconds:
        ctl = read_ctl()
        scan = ctl.get("scan", ctl.get("mode", "auto")) if ctl else "auto"
        if ctl:
            handle_cmd(ctl)
        painting = bool(ctl and ctl.get("painting") and len(ctl.get("aim") or []) == 12)
        want_view = a.passthrough or bool(ctl and ctl.get("passthrough"))          # viewer shows the camera image (needs colour)
        occ_state["on"] = bool(ctl and ctl.get("occlude")) and not want_view     # occlusion-only: its own fast low-res thread
        fuse = not (scan == "off" or (scan == "paint" and not painting))
        tagwatch.enabled = bool(ctl and ctl.get("tags"))
        planes_state["on"] = bool(ctl and ctl.get("planes"))
        if splatcap: splatcap.enabled = bool(ctl and ctl.get("splat")) and splatcap.bytes < splatcap.max_bytes
        if os.path.exists(DEBUG_TRIGGER):
            try: os.remove(DEBUG_TRIGGER)
            except OSError: pass
            dbg_state["armed"] = True; print("debug capture armed (trigger file)", flush=True)
        dbg = dbg_state["armed"]                 # debug capture: process the next pair even if nothing else needs it
        if not fuse and not want_view and not tagwatch.enabled and not dbg:
            time.sleep(0.03)                  # idle, keep polling the viewer for commands
            if zf.poll():                     # zones edited while not scanning: show the result right away
                npub = publish_cloud(); print(f"zones: {len(zf.zones)} -> {npub} points", flush=True)
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
        # motion-blur gate (pose velocities only, before any image work): blurred pairs are not fused
        Tcur = np.eye(4); Tcur[:3, :] = np.asarray(mL["pose"]); blur = 0.0; om = v = 0.0
        if prev_pose is not None and 1e-3 < mL["ts"] - prev_pose[1] < 0.2:
            dtp = mL["ts"] - prev_pose[1]
            om = np.radians(_rot_angle(prev_pose[0][:3, :3].T @ Tcur[:3, :3])) / dtp
            v = np.linalg.norm(Tcur[:3, 3] - prev_pose[0][:3, 3]) / dtp
            blur = ls.f * mL.get("exp_lines", 3000) * LT * (om + v / max(z_med, 0.5))
        prev_pose = (Tcur, mL["ts"])
        sharp = a.blur <= 0 or blur <= a.blur or time.monotonic() - t_last_fused > a.blur_max_wait
        if not sharp:
            n_blurry += 1
            if not want_view and not tagwatch.enabled and not dbg:
                continue                      # nothing else needs this pair: skip the stereo entirely
        fuse = fuse and sharp
        if dbg:
            dbg_state["armed"] = False; ls.debug = {}; tdb = {}
        t1 = time.monotonic()
        kw = dict(W=s.line_px, stride=s.line_bytes, skip=s.skip, black=float(s.black), threads=a.threads)
        H = (min(len(bL), len(bR)) // s.line_bytes) - s.skip
        cL = bin_raw10(bL, H=min(H, s.line_px), **kw); cR = bin_raw10(bR, H=min(H, s.line_px), **kw)
        Tt = ls.world_from_rect(mL["pose"])            # tracking (STAGE)
        if dbg: tdb["bin_raw10"] = time.monotonic() - t1
        Z, rgb, _ = ls.depth(cL, cR, far=a.carve_depth, half=True)
        if dbg: tdb["depth"] = time.monotonic() - t1 - tdb["bin_raw10"]; dbuf = ls.debug; ls.debug = None
        zv = Z[::8, ::8]; zv = zv[zv > 0]
        if len(zv): z_med = float(np.median(zv))
        T = Tt
        if a.apply_origin_corr and origin:
            Cc = origin.correction_now()
            if Cc is not None: T = Cc @ Tt
        tagwatch.submit(cL, mL["pose"], Z)
        if want_view:                                   # depth-correct passthrough (STAGE frame, raw tracking pose)
            n_pt += 1
            ptpub.publish(ptpub.pack(Z, rgb, ls.f, ls.P1[0, 2], ls.P1[1, 2], Tt, mL["ts"], mL["frame"], n_pt))
        def debug_out(Zf, hm, mask):
            info = dict(timings_ms={k: round(x * 1000, 2) for k, x in tdb.items()}, far_m=float(a.carve_depth), fused=Zf is not None,
                        scan=scan, frame_left=mL["frame"], frame_right=mR["frame"], ts_left=mL["ts"], ts_right=mR["ts"],
                        blur_gate=dict(blur_px=float(blur), omega_rad_s=float(om), v_m_s=float(v), z_med_m=float(z_med), threshold_px=a.blur,
                                       sharp=bool(sharp), skipped_so_far=n_blurry),
                        pcserver_args=vars(a))
            bufs = dict(dbuf, rgb_R=ls.rectified_rgb_pair(cL, cR, half=True)[1],
                        hand_mask=hm if hm is not None else np.zeros(Z.shape, bool),
                        cone_mask=mask if mask is not None else np.ones(Z.shape, bool),
                        Z_fused=Zf if Zf is not None else np.zeros_like(Z), conf=ls.confidence(dbuf["Z"], dbuf["rgb"]),
                        stage_from_rect_tracked=Tt, stage_from_rect_used=T)
            threading.Thread(target=take_debug, args=((mL, bL, mR, bR), bufs, info), daemon=True).start()
        if not fuse:
            if dbg: debug_out(None, None, None)
            continue
        t2 = time.monotonic()
        hm = None
        if not a.no_handmask and ctl and ctl.get("hands"):        # controllers + the hands holding them stay out of the map
            hp = [(h[3], h[7], h[11]) for h in ctl["hands"] if len(h) == 12]
            hm = handmask.hand_mask(Z, np.linalg.inv(Tt), ls.f, ls.P1[0, 2], ls.P1[1, 2], hp)
            if hm.any(): Z = Z.copy(); Z[hm] = 0
        if far is not None:
            far.enabled = bool(ctl.get("far", True)) if ctl else True
            far.offer(rgb, Z, T, mL["ts"], mL.get("exp_lines", 3000))
        with lock:
            mask = cone_pixels(ls, Z, T, to_map(ctl["aim"]).ravel().tolist(), a.cone) if scan == "paint" else None
            ts.integrate(ls, Z, rgb, T, mask=mask, t=mL["ts"], conf=ls.confidence(Z, rgb) if a.conf else None)
        t_last_fused = time.monotonic()
        if dbg:
            tdb["handmask_far_integrate"] = time.monotonic() - t2; debug_out(Z, hm, mask)
        n_frames += 1; tst.append(t2 - t1); tfu.append(time.monotonic() - t2); t_fr.append(time.monotonic())
        if zf.poll() or time.monotonic() - t_pub > a.publish_every:
            t_pub = time.monotonic()
            npub = publish_cloud()
            fps = (len(t_fr[-20:]) - 1) / max(t_fr[-1] - t_fr[-20:][0], 1e-3) if len(t_fr) > 1 else 0.0
            last_fps = fps; write_status(fps)
            print(f"frame {last_frame}: {n_frames} fused ({fps:.1f}/s), stereo {np.mean(tst[-10:])*1000:.0f} ms, "
                  f"tsdf {np.mean(tfu[-10:])*1000:.0f} ms, publish {(time.monotonic()-t_pub)*1000:.0f} ms, "
                  f"{ts.nblocks} blocks, {npub} points, scan {scan}, blurry skipped {n_blurry}", flush=True)
finally:
    stop = True
