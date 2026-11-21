#!/usr/bin/env python3
"""Stereo still capture: the newest synchronised Arcturus pair + pose + timestamps (+ GNSS / georeference) in ONE multi-page TIFF.

Writes captures/stills/<YYYYmmdd_HHMMSS_mmm>/
  still.tiff   multi-page TIFF (tools/tiffmulti.py; libtiff-readable: tiffinfo / tiffcp / ImageMagick / GIMP):
                 page 0  left   full-resolution colour (2464^2, demosaiced, white-balanced, upright), JPEG compressed;
                                 ImageDescription = ALL metadata as JSON (capture, poses, calibration, depth model, GNSS, lla ...),
                                 plus Make / Model / DateTime tags (tifffile cannot write Exif / GPS sub-IFDs: GPS is in the JSON)
                 page 1  right  same for the right camera
                 page 2  rectified left colour (the geometry of the depth map), JPEG
                 page 3  rectified right colour (row-aligned with page 2: a ready stereo pair), JPEG
                 page 4  depth map: float32 metres in the rectified-left geometry (0 = invalid), deflate; description gives
                         intrinsics, stage_from_rect and the depth model
                 page 5  composited headset view (what you saw incl. cloud / HUD), JPEG - APPENDED a moment later: the file is written
                         right after the stereo capture, the next composited frame is added when it exists (record sessions:
                         the next c_*.jpg of composited/; otherwise one frame is grabbed from /dev/video99)
  left.y10p, right.y10p  raw RAW10 sensor lines (lossless; tools/arcshm.py unpack / tools/livestereo.unpack_raw)
  meta.json    the same metadata as page 0 (poses: stage_from_cam for the raw sensor and stage_from_cam_jpg for the upright JPEG)
Used by tools/pcserver.py (viewer command "still"); standalone:  stills.py [--out DIR]  (needs arcgrab running)
Legacy stills (left.jpg / right.jpg): re-render the JPGs from their raw files:  stills.py --regen DIR...
"""
import json, os, struct, sys, time, datetime
import numpy as np
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from livestereo import unpack_raw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "captures", "stills")
CAL = os.path.join(HERE, "..", "captures", "arcturus_calib.json")


# ------------------------------------------------------------------ EXIF (minimal TIFF/EXIF writer)

def _rat(x, den=1000000):
    return (int(round(abs(x) * den)), den)


def _dms(deg):
    deg = abs(deg); d = int(deg); m = int((deg - d) * 60); s = (deg - d - m / 60) * 3600
    return [(d, 1), (m, 1), _rat(s, 10000)]


def exif_bytes(dt_local, subsec, offset, make, model, comment, gps=None, desc=""):
    """-> APP1 payload (b'Exif\\0\\0' + TIFF). gps: {lat, lon, h, utc: datetime, dir_deg}"""
    ASCII, SHORT, LONG, RAT, UNDEF, SRAT = 2, 3, 4, 5, 7, 10

    def ifd(entries, base, next_off=0):
        """entries: list (tag, type, value) -> bytes; offsets relative to TIFF start, IFD placed at `base`."""
        entries = sorted(entries, key=lambda e: e[0])
        n = len(entries); head = struct.pack("<H", n); data = b""; data_off = base + 2 + 12 * n + 4
        body = b""
        for tag, typ, val in entries:
            if typ == ASCII:
                raw = val.encode("utf-8", "replace") + b"\0"; cnt = len(raw)
            elif typ == UNDEF:
                raw = val; cnt = len(raw)
            elif typ in (RAT, SRAT):
                vals = val if isinstance(val, list) else [val]
                raw = b"".join(struct.pack("<II" if typ == RAT else "<ii", *v) for v in vals); cnt = len(vals)
            elif typ == SHORT:
                vals = val if isinstance(val, list) else [val]; raw = b"".join(struct.pack("<H", v) for v in vals); cnt = len(vals)
            elif typ == LONG:
                vals = val if isinstance(val, list) else [val]; raw = b"".join(struct.pack("<I", v) for v in vals); cnt = len(vals)
            elif typ == 1:   # BYTE
                vals = val if isinstance(val, list) else [val]; raw = bytes(vals); cnt = len(vals)
            if len(raw) <= 4:
                body += struct.pack("<HHI", tag, typ, cnt) + raw.ljust(4, b"\0")
            else:
                body += struct.pack("<HHII", tag, typ, cnt, data_off + len(data)); data += raw
                if len(data) % 2: data += b"\0"
        return head + body + struct.pack("<I", next_off) + data

    # layout: header(8) | IFD0 | ExifIFD | GPSIFD ; compute sizes iteratively (pointers are fixed-size)
    def build(exif_off, gps_off):
        e0 = [(0x010F, ASCII, make), (0x0110, ASCII, model), (0x0132, ASCII, dt_local), (0x8769, LONG, exif_off)]
        if desc: e0.append((0x010E, ASCII, desc))
        if gps: e0.append((0x8825, LONG, gps_off))
        ex = [(0x9000, UNDEF, b"0232"), (0x9003, ASCII, dt_local), (0x9291, ASCII, subsec), (0x9011, ASCII, offset),
              (0x9286, UNDEF, b"ASCII\0\0\0" + comment.encode("ascii", "replace"))]
        g = []
        if gps:
            u = gps["utc"]
            g = [(0x0000, 1, [2, 3, 0, 0]), (0x0001, ASCII, "N" if gps["lat"] >= 0 else "S"), (0x0002, RAT, _dms(gps["lat"])),
                 (0x0003, ASCII, "E" if gps["lon"] >= 0 else "W"), (0x0004, RAT, _dms(gps["lon"])),
                 (0x0005, 1, [0 if gps["h"] >= 0 else 1]), (0x0006, RAT, _rat(gps["h"], 1000)),
                 (0x0007, RAT, [(u.hour, 1), (u.minute, 1), _rat(u.second + u.microsecond / 1e6, 1000)]),
                 (0x001D, ASCII, u.strftime("%Y:%m:%d")), (0x0012, ASCII, "WGS-84")]
            if gps.get("dir_deg") is not None:
                g += [(0x0010, ASCII, "T"), (0x0011, RAT, _rat(gps["dir_deg"] % 360, 100))]
            if gps.get("hacc") is not None:
                g.append((0x001F, RAT, _rat(gps["hacc"], 1000)))
        b0 = ifd(e0, 8); bex = ifd(ex, exif_off); bg = ifd(g, gps_off) if gps else b""
        return b0, bex, bg
    b0, bex, bg = build(0, 0)
    exif_off = 8 + len(b0); gps_off = exif_off + len(bex)
    b0, bex, bg = build(exif_off, gps_off)
    assert 8 + len(b0) == exif_off and exif_off + len(bex) == gps_off
    return b"Exif\0\0" + b"II*\0" + struct.pack("<I", 8) + b0 + bex + bg


def jpeg_with_exif(img_bgr, app1, quality=92):
    ok, jpg = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    jpg = jpg.tobytes()
    seg = b"\xff\xe1" + struct.pack(">H", len(app1) + 2) + app1
    return jpg[:2] + seg + jpg[2:]


# ------------------------------------------------------------------ capture

def _read_json(p):
    try:
        with open(p) as f: return json.load(f)
    except Exception: return {}


def colour_full(raw16, black=64, wb=None):
    """RGGB 10-bit mosaic (R at (0,0)) -> 8-bit BGR full resolution, grey-world white balance, gamma 2.2."""
    m = np.clip(raw16.astype(np.int32) - black, 0, 1023).astype(np.uint16)
    bgr = cv2.cvtColor(m, cv2.COLOR_BayerBG2BGR).astype(np.float32)   # OpenCV's BG code = R at (0,0) (tested)
    if wb is None:
        mm = bgr[::8, ::8].reshape(-1, 3).mean(0); wb = mm.mean() / np.maximum(mm, 1e-3)
    bgr *= wb
    s = 1.0 / max(np.percentile(bgr[::8, ::8], 99.5), 1.0)
    return (np.clip(bgr * s, 0, 1) ** (1 / 2.2) * 255).astype(np.uint8), wb


def camera_poses(ls, hmd_pose):
    """STAGE-from-camera (OpenCV camera axes) 4x4 for left and right."""
    H = np.eye(4); H[:3, :] = np.asarray(hmd_pose)
    TL = H @ ls.head_from_camL
    camL_from_camR = np.eye(4); camL_from_camR[:3, :3] = ls.R.T; camL_from_camR[:3, 3] = -ls.R.T @ ls.t
    return TL, TL @ camL_from_camR


def upright(ls, side):
    """Rotation that makes the sensor image upright (the modules are mounted +-90 deg) -> (cv2 rotate code or None,
    R_old_from_new for the camera axes, name). Decided from the camera x/y axes in the head frame (head y = up)."""
    hc = ls.head_from_camL[:3, :3]
    if side == "right":
        hc = hc @ ls.R.T
    x_up, y_up = hc[1, 0], hc[1, 1]
    if x_up > 0.7:      # image right points up -> rotate counter-clockwise: new x = old y, new y = -old x
        return cv2.ROTATE_90_COUNTERCLOCKWISE, np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.]]), "90ccw"
    if x_up < -0.7:     # image right points down -> rotate clockwise: new x = -old y, new y = old x
        return cv2.ROTATE_90_CLOCKWISE, np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1.]]), "90cw"
    if y_up > 0.7:
        return cv2.ROTATE_180, np.diag([-1., -1, 1]), "180"
    return None, np.eye(3), "none"


def _write_page(path, arr, role, desc, name=None, dt=None, make=None, model=None, append=False, jpeg=True, quality=92):
    """One TIFF page via tifffile: JPEG-compressed colour (arr RGB uint8) or lossless deflate (float depth). The ImageDescription of
    every page is JSON (role + what is needed to interpret it)."""
    import tifffile
    tags = [(285, "s", 0, name or role, True)]
    if make: tags.append((271, "s", 0, make, True))
    if model: tags.append((272, "s", 0, model, True))
    if dt: tags.append((306, "s", 0, dt, True))
    kw = dict(compression="jpeg", compressionargs={"level": quality}, photometric="rgb") if jpeg else dict(compression="zlib", photometric="minisblack")
    tifffile.imwrite(path, arr, append=append, description=json.dumps(desc, separators=(",", ":")), metadata=None, software="pcstill/2 (tools/stills.py)",
                     extratags=tags, **kw)


# ------------------------------------------------------------------ composited headset view (next frame after the trigger)

def _active_composited_dir():
    """The composited/ dir of a running `pcsession.sh --record` (a c_*.jpg written within the last 3 s), else None."""
    import glob
    best = None
    for d in glob.glob(os.path.join(HERE, "..", "captures", "live*", "composited")):
        fs = glob.glob(os.path.join(d, "c_*.jpg"))
        if fs:
            m = max(os.path.getmtime(f) for f in fs)
            if time.time() - m < 3.0 and (best is None or m > best[0]): best = (m, d)
    return best[1] if best else None


def composited_frame(t_after, timeout=4.0, tmp_dir=None, device="/dev/video99"):
    """The first composited headset frame captured after wall-clock time t_after -> (jpeg bytes, source, t_file) or (None, reason, None).
    Recording session: the next file of composited/. Otherwise a single frame is read from the SteamVR mirror device."""
    import glob, subprocess
    t_end = time.time() + timeout
    d = _active_composited_dir()
    if d:
        while time.time() < t_end:
            fs = sorted((os.path.getmtime(f), f) for f in glob.glob(os.path.join(d, "c_*.jpg")))
            fs = [x for x in fs if x[0] > t_after]
            if fs:
                m, f = fs[0]; time.sleep(0.03)
                b = open(f, "rb").read()
                if b[-2:] == b"\xff\xd9": return b, "record:" + os.path.basename(f), m
            time.sleep(0.03)
        return None, "no composited frame within %.0f s" % timeout, None
    tmp = os.path.join(tmp_dir or os.path.join(HERE, "..", "captures", "scratch"), "composited_grab_%d.jpg" % os.getpid())
    os.makedirs(os.path.dirname(tmp), exist_ok=True)
    try:
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "v4l2", "-i", device, "-frames:v", "1", "-q:v", "2", "-y", tmp],
                       timeout=timeout, check=True, capture_output=True)
        t = time.time(); b = open(tmp, "rb").read(); os.remove(tmp)
        return b, "ffmpeg:" + device, t
    except Exception as e:
        return None, f"ffmpeg grab failed: {e}", None


def _append_composited(d, tiff_path, t_trigger, ready, meta):
    try:
        b, src, t_file = composited_frame(t_trigger)
        ready.wait(10.0)
        if b is None:
            meta["composited"] = {"error": src}
        else:
            img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)[..., ::-1]
            desc = {"role": "composited", "what": "headset composited view (SteamVR mirror: passthrough + viewer overlay), re-compressed as JPEG q92",
                    "source": src, "captured_utc": datetime.datetime.fromtimestamp(t_file, datetime.timezone.utc).isoformat(),
                    "after_trigger_s": round(t_file - t_trigger, 3)}
            _write_page(tiff_path, np.ascontiguousarray(img), "composited", desc, append=True)
            meta["composited"] = {k: desc[k] for k in ("source", "captured_utc", "after_trigger_s")}
    except Exception as e:
        meta["composited"] = {"error": str(e)}
    try:
        with open(os.path.join(d, "meta.json"), "w") as f: json.dump(meta, f, indent=1)
    except OSError:
        pass


# ------------------------------------------------------------------ capture

def save_still(shm, ls, out_root=OUT, gnss_path="/dev/shm/gnss.json", georef_path="/dev/shm/georef.json",
               pair_tol=0.015, timeout=2.0, composited=True, pair=None, depth=None):
    """Grab the newest synchronised pair from ArcShm `shm` and write a still directory with still.tiff. Returns its path.
    The stereo frames are taken first (small capture delay); everything else, incl. the composited view, follows.
    pair = (mL, bL, mR, bR): use this already grabbed pair instead (debug capture: the pair pcserver just processed);
    depth = (Z, rgbL, rgbR): its already computed depth page instead of recomputing it."""
    import threading
    from tsdf import bin_raw10
    t_wall = time.time()
    t_end = time.monotonic() + timeout
    while pair is None:
        mL, bL = shm.latest_packed(0, timeout=timeout)
        mR, bR = shm.latest_packed(1, after_frame=mL["frame"] - 2, timeout=timeout)
        if abs(mL["ts"] - mR["ts"]) <= pair_tol or time.monotonic() > t_end:
            break
    if pair is not None:
        mL, bL, mR, bR = pair
    ready = threading.Event(); meta = {}
    # wall-clock time of the exposure: shift "now" by the monotonic age of the frame
    age = time.monotonic() - mL["ts"]
    t_utc = t_wall - max(age, 0.0)
    dt = datetime.datetime.fromtimestamp(t_utc).astimezone()
    name = dt.strftime("%Y%m%d_%H%M%S_") + f"{dt.microsecond // 1000:03d}"
    d = os.path.join(out_root, name); os.makedirs(d, exist_ok=True)
    gnss, georef = _read_json(gnss_path), _read_json(georef_path)
    if gnss and time.time() - float(gnss.get("t", 0)) > 5:
        gnss = dict(gnss, stale=True)
    TL, TR = camera_poses(ls, mL["pose"])
    lla = {}
    if georef.get("ok"):
        import geo
        for side, T in (("left", TL), ("right", TR)):
            lat, lon, h = geo.stage_to_lla(T[:3, 3], georef)
            fwd = geo.stage_rot_to_enu(georef) @ T[:3, 2]          # camera optical axis in ENU
            lla[side] = dict(lat=float(lat), lon=float(lon), h=float(h), heading_deg=float(np.degrees(np.arctan2(fwd[0], fwd[1])) % 360),
                             pitch_deg=float(np.degrees(np.arcsin(np.clip(fwd[2], -1, 1)))), source="georef")
    elif gnss.get("q"):
        for side in ("left", "right"):
            lla[side] = dict(lat=gnss["lat"], lon=gnss["lon"], h=gnss["h"], source="gnss_antenna (no georef yet)")
    try:
        calib = json.load(open(CAL)); cams = [c for c in calib["cameras"] if c.get("sourceCamera") in ("passthrough_left", "passthrough_right")]
        calibration = {"file": os.path.relpath(CAL, os.path.dirname(d)), "cameras": cams}
    except Exception:
        calibration = {"file": os.path.relpath(CAL, os.path.dirname(d))}
    meta.update({"format": "pcstill/2", "t_utc": t_utc, "t_utc_iso": datetime.datetime.fromtimestamp(t_utc, datetime.timezone.utc).isoformat(),
            "trigger_wall_time": t_wall, "capture_delay_s": round(age, 4),
            "clock": "ts = CLOCK_MONOTONIC (s) at mid-exposure as stamped by src/arcgrab; poses = OpenVR Standing == OpenXR STAGE",
            "left": {k: mL[k] for k in ("frame", "ts", "exp_lines", "gain_code", "dgain", "pose_valid")},
            "right": {k: mR[k] for k in ("frame", "ts", "exp_lines", "gain_code", "dgain", "pose_valid")},
            "hmd_pose": np.asarray(mL["pose"]).tolist(),
            "stage_from_cam": {"left": TL.tolist(), "right": TR.tolist(),
                               "axes": "OpenCV camera (x right, y down, z forward), metres; raw sensor orientation"},
            "stage_from_cam_jpg": {}, "jpg_rotation": {},
            "calibration": calibration,
            "raw": {"width": shm.line_px, "line_bytes": shm.line_bytes, "skip_lines": shm.skip, "black": shm.black,
                    "bayer": "RGGB", "format": "RAW10 packed (MIPI)"},
            "gnss": gnss, "georef": georef, "lla": lla,
            "tiff_pages": ["left (JPEG, upright)", "right (JPEG, upright)", "rectified left (JPEG)", "rectified right (JPEG)", "depth (float32 m, rectified left)",
                           "composited headset view (JPEG, appended later)"]})
    if composited:      # starts waiting for the next composited frame right away; appends once still.tiff exists
        threading.Thread(target=_append_composited, args=(d, os.path.join(d, "still.tiff"), t_wall, ready, meta), daemon=True).start()
    imgs, descs = {}, {}
    for side, m, b, T in (("left", mL, bL, TL), ("right", mR, bR, TR)):
        with open(os.path.join(d, side + ".y10p"), "wb") as f:
            f.write(b)
        H = min(len(b) // shm.line_bytes - shm.skip, shm.line_px)
        raw = unpack_raw(b[shm.skip * shm.line_bytes:], W=shm.line_px, H=H, S=shm.line_bytes)
        img, wb = colour_full(raw, black=shm.black)
        rot, R_on, rname = upright(ls, side)
        if rot is not None:
            img = cv2.rotate(img, rot)
        Tj = T.copy(); Tj[:3, :3] = T[:3, :3] @ R_on
        meta["stage_from_cam_jpg"][side] = Tj.tolist(); meta["jpg_rotation"][side] = rname
        com = json.dumps({"t_utc": round(t_utc, 4), "ts": m["ts"], "side": side, "stage_from_cam": np.round(Tj[:3], 5).tolist(),
                          "lla": lla.get(side)}, separators=(",", ":"))
        gps = None
        L = lla.get(side)
        if L:
            gps = dict(lat=L["lat"], lon=L["lon"], h=L["h"], utc=datetime.datetime.fromtimestamp(t_utc, datetime.timezone.utc),
                       dir_deg=L.get("heading_deg"), hacc=gnss.get("hacc"))
        imgs[side] = np.ascontiguousarray(img[..., ::-1])
        descs[side] = {"role": side, "what": f"{side} camera, full resolution colour, upright ({rname} from the sensor)", "stage_from_cam": Tj.tolist(),
                       "frame": m["frame"], "ts": m["ts"], "lla": lla.get(side)}
    # depth map in the rectified-left geometry (same pipeline as the live fusion)
    kw = dict(W=shm.line_px, stride=shm.line_bytes, skip=shm.skip, black=float(shm.black), threads=2)
    Hh = min(len(bL) // shm.line_bytes - shm.skip, shm.line_px)
    cLh, cRh = bin_raw10(bL, H=Hh, **kw), bin_raw10(bR, H=Hh, **kw)
    if depth is not None:
        Z, rgbL, rgbR = depth
    else:
        Z, _, _ = ls.depth(cLh, cRh, far=4.0, half=True)
        rgbL, rgbR = ls.rectified_rgb_pair(cLh, cRh, half=True)
    Wr = ls.world_from_rect(mL["pose"])
    geom = {"intrinsics_rect": {"f": float(ls.f), "cx": float(ls.P1[0, 2]), "cy": float(ls.P1[1, 2]), "size": list(ls.size)},
            "stage_from_rect": Wr.tolist(), "rect_axes": "OpenCV (x right, y down, z forward)", "baseline_m": float(ls.baseline),
            "depth_model": "Z = k*f*B/(disparity + d0), disparity at this resolution", "k": float(getattr(ls, "k", 1.0)), "d0_px": float(getattr(ls, "d0", 0.0)),
            "R1": ls.R1.tolist(), "R2": ls.R2.tolist(), "P1": ls.P1.tolist(), "P2": ls.P2.tolist(),
            "K_left": ls.KL.tolist(), "D_left": ls.DL.ravel().tolist(), "K_right": ls.KR.tolist(), "D_right": ls.DR.ravel().tolist(),
            "R_right_from_left": ls.R.tolist(), "t_right_from_left": ls.t.tolist(), "head_from_camL": ls.head_from_camL.tolist()}
    meta["geometry"] = geom
    dt_s = dt.strftime("%Y:%m:%d %H:%M:%S")
    tp = os.path.join(d, "still.tiff")
    for k, side in enumerate(("left", "right")):
        desc = descs[side] if side == "right" else dict(descs[side], metadata=meta)           # page 0 carries ALL metadata
        _write_page(tp, imgs[side], side, desc, dt=dt_s, make="Valve", model=f"Steam Frame passthrough {side}", append=k > 0)
    _write_page(tp, rgbL, "rect_left", dict(role="rect_left", what="rectified left colour (geometry of the depth page)", **geom), append=True)
    _write_page(tp, rgbR, "rect_right", dict(role="rect_right", what="rectified right colour: row-aligned with rect_left, disparity = x_left - x_right", **geom), append=True)
    _write_page(tp, Z.astype(np.float32), "depth", dict(role="depth", units="m", invalid=0, what="stereo depth, rectified left geometry", **geom), append=True, jpeg=False)
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    ready.set()                                  # (the composited thread appends its page and rewrites meta.json after this)
    return d


def regen_jpgs(d):
    """Re-render left/right.jpg of an existing still directory from its raw .y10p files (keeps the EXIF segment)."""
    meta = json.load(open(os.path.join(d, "meta.json")))
    r = meta["raw"]; rots = {"90ccw": cv2.ROTATE_90_COUNTERCLOCKWISE, "90cw": cv2.ROTATE_90_CLOCKWISE, "180": cv2.ROTATE_180}
    for side in ("left", "right"):
        b = open(os.path.join(d, side + ".y10p"), "rb").read()
        H = min(len(b) // r["line_bytes"] - r["skip_lines"], r["width"])
        raw = unpack_raw(b[r["skip_lines"] * r["line_bytes"]:], W=r["width"], H=H, S=r["line_bytes"])
        img, _ = colour_full(raw, black=r["black"])
        rot = rots.get(meta["jpg_rotation"].get(side, "none"))
        if rot is not None:
            img = cv2.rotate(img, rot)
        jp = os.path.join(d, side + ".jpg"); old = open(jp, "rb").read()
        assert old[2:4] == b"\xff\xe1"
        n = struct.unpack(">H", old[4:6])[0]
        with open(jp, "wb") as f:
            f.write(jpeg_with_exif(img, old[6:4 + n]))
    meta["raw"]["bayer"] = "RGGB"
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    import argparse
    if len(sys.argv) > 2 and sys.argv[1] == "--regen":       # stills.py --regen DIR...  (re-render JPGs from the raw files)
        for d in sys.argv[2:]:
            regen_jpgs(d); print("regenerated", d)
        sys.exit(0)
    from arcshm import ArcShm
    from livestereo import LiveStereo
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=OUT); a = ap.parse_args()
    print(save_still(ArcShm(), LiveStereo(scale=0.35), a.out))
