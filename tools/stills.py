#!/usr/bin/env python3
"""Stereo still capture: the newest synchronised Arcturus pair + pose + timestamps (+ GNSS / georeference).

Writes captures/stills/<YYYYmmdd_HHMMSS_mmm>/
  left.jpg, right.jpg   full-resolution colour (2464^2, demosaiced, white-balanced, rotated upright), EXIF: DateTimeOriginal +
                        SubSec + OffsetTime, Make/Model, GPS (lat/lon/alt/time/img direction) when available,
                        UserComment = compact JSON {t_utc, ts, pose (camera-to-STAGE 3x4), lla, side}
  left.y10p, right.y10p raw RAW10 sensor lines (lossless; tools/arcshm.py unpack / tools/livestereo.unpack_raw)
  meta.json             everything: per-camera metadata (frame, ts CLOCK_MONOTONIC, exposure, gain), HMD pose,
                        camera-to-STAGE poses for the raw sensor (stage_from_cam, matches the factory
                        calibration) and for the upright JPEG (stage_from_cam_jpg, jpg_rotation), calibration file, GNSS fix, georeference, camera lat/lon/h
Used by tools/pcserver.py (viewer command "still"); standalone:  stills.py [--out DIR]  (needs arcgrab running)
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
    """BGGR 10-bit mosaic -> 8-bit BGR full resolution, grey-world white balance, gamma 2.2."""
    m = np.clip(raw16.astype(np.int32) - black, 0, 1023).astype(np.uint16)
    bgr = cv2.cvtColor(m, cv2.COLOR_BayerRG2BGR).astype(np.float32)   # OpenCV names Bayer by the 2nd row: BGGR -> RG
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


def save_still(shm, ls, out_root=OUT, gnss_path="/dev/shm/gnss.json", georef_path="/dev/shm/georef.json",
               pair_tol=0.015, timeout=2.0):
    """Grab the newest synchronised pair from ArcShm `shm` and write a still directory. Returns its path."""
    t_wall = time.time()
    t_end = time.monotonic() + timeout
    while True:
        mL, bL = shm.latest_packed(0, timeout=timeout)
        mR, bR = shm.latest_packed(1, after_frame=mL["frame"] - 2, timeout=timeout)
        if abs(mL["ts"] - mR["ts"]) <= pair_tol or time.monotonic() > t_end:
            break
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
    meta = {"format": "pcstill/1", "t_utc": t_utc, "t_utc_iso": datetime.datetime.fromtimestamp(t_utc, datetime.timezone.utc).isoformat(),
            "clock": "ts = CLOCK_MONOTONIC (s) at mid-exposure as stamped by src/arcgrab; poses = OpenVR Standing == OpenXR STAGE",
            "left": {k: mL[k] for k in ("frame", "ts", "exp_lines", "gain_code", "dgain", "pose_valid")},
            "right": {k: mR[k] for k in ("frame", "ts", "exp_lines", "gain_code", "dgain", "pose_valid")},
            "hmd_pose": np.asarray(mL["pose"]).tolist(),
            "stage_from_cam": {"left": TL.tolist(), "right": TR.tolist(),
                               "axes": "OpenCV camera (x right, y down, z forward), metres"},
            "stage_from_cam_jpg": {}, "jpg_rotation": {},
            "calibration": os.path.relpath(CAL, os.path.dirname(d)),
            "raw": {"width": shm.line_px, "line_bytes": shm.line_bytes, "skip_lines": shm.skip, "black": shm.black,
                    "bayer": "BGGR", "format": "RAW10 packed (MIPI)"},
            "gnss": gnss, "georef": georef, "lla": lla}
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
        T = Tj
        com = json.dumps({"t_utc": round(t_utc, 4), "ts": m["ts"], "side": side, "stage_from_cam": np.round(T[:3], 5).tolist(),
                          "lla": lla.get(side)}, separators=(",", ":"))
        gps = None
        L = lla.get(side)
        if L:
            gps = dict(lat=L["lat"], lon=L["lon"], h=L["h"], utc=datetime.datetime.fromtimestamp(t_utc, datetime.timezone.utc),
                       dir_deg=L.get("heading_deg"), hacc=gnss.get("hacc"))
        app1 = exif_bytes(dt.strftime("%Y:%m:%d %H:%M:%S"), f"{dt.microsecond // 1000:03d}", dt.strftime("%z")[:3] + ":" + dt.strftime("%z")[3:],
                          "Valve", f"Steam Frame passthrough {side}", com, gps, desc=f"pcstill {name} {side}")
        with open(os.path.join(d, side + ".jpg"), "wb") as f:
            f.write(jpeg_with_exif(img, app1))
    with open(os.path.join(d, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return d


if __name__ == "__main__":
    import argparse
    from arcshm import ArcShm
    from livestereo import LiveStereo
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=OUT); a = ap.parse_args()
    print(save_still(ArcShm(), LiveStereo(scale=0.25), a.out))
