#!/usr/bin/env python3
"""Record pose-stamped stereo pairs from /dev/shm/arcgrab (run src/arcgrab concurrently).
usage: liverec.py OUTDIR [--seconds 30] [--fps 3]
Writes OUTDIR/NNNNN_{left,right}.y10p (packed RAW10, 2464x2464, stride 3088, BGGR) and
OUTDIR/frames.jsonl (one line per pair: ts/pose/exposure per camera) + OUTDIR/stream.json."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arcshm import ArcShm
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--seconds", type=float, default=30); ap.add_argument("--fps", type=float, default=3)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
s = ArcShm()
json.dump(dict(line_px=s.line_px, line_bytes=s.line_bytes, black=s.black, bayer="BGGR", cams=["left", "right"],
               devices=["/dev/video0 arcimx616 0-001a crop x=550", "/dev/video3 arcimx616 0-0010 crop x=250"]),
          open(os.path.join(a.out, "stream.json"), "w"), indent=1)
log = open(os.path.join(a.out, "frames.jsonl"), "w"); n = 0; t0 = time.monotonic(); last = -1; dts = []
# full-rate log of every left frame: metadata + 616x616 green-channel thumbnail (8-bit MSBs sliced
# straight from the packed RAW10: every 4th row (even -> B/G row), byte 1 of each 5-byte group = G)
import threading, numpy as np
plog = open(os.path.join(a.out, "poses.jsonl"), "w"); tbin = open(os.path.join(a.out, "thumbs_left.u8"), "wb"); stop = False
def pose_logger():
    s2 = ArcShm(); lf = -1
    while not stop:
        try: m, buf = s2.latest_packed(0, after_frame=lf, timeout=0.5)
        except TimeoutError: continue
        lf = m["frame"]
        th_ = np.frombuffer(buf, np.uint8, count=s2.line_bytes * 2464).reshape(2464, s2.line_bytes)[0::4, :3080].reshape(616, 616, 5)[:, :, 1]
        tbin.write(np.ascontiguousarray(th_).tobytes())
        plog.write(json.dumps(dict(frame=m["frame"], ts=m["ts"], exp_lines=m["exp_lines"], pose_valid=m["pose_valid"], pose=m["pose"].tolist())) + "\n")
th = threading.Thread(target=pose_logger, daemon=True); th.start()
while time.monotonic() - t0 < a.seconds:
    tn = time.monotonic()
    m0, b0 = s.latest_packed(0, after_frame=last)
    m1, b1 = s.latest_packed(1)
    if abs(m1["ts"] - m0["ts"]) > 0.010:          # wait for the matching right frame
        try: m1, b1 = s.latest_packed(1, after_frame=m1["frame"], timeout=0.1)
        except TimeoutError: pass
    last = m0["frame"]; dts.append(m1["ts"] - m0["ts"])
    for name, b in (("left", b0), ("right", b1)):
        open(os.path.join(a.out, f"{n:05d}_{name}.y10p"), "wb").write(b)
    rec = {"i": n}
    for name, m in (("left", m0), ("right", m1)):
        rec[name] = dict(frame=m["frame"], ts=m["ts"], exp_lines=m["exp_lines"], gain_code=m["gain_code"], dgain=m["dgain"],
                         pose=m["pose"].tolist(), pose_valid=m["pose_valid"], buf_flags=m["buf_flags"])
    log.write(json.dumps(rec) + "\n"); log.flush(); n += 1
    time.sleep(max(0, 1 / a.fps - (time.monotonic() - tn)))
stop = True; th.join(timeout=1); plog.close(); tbin.close()
import numpy as np
print(f"{n} pairs; left-right ts diff median {np.median(dts)*1e3:.2f} ms, max {np.max(np.abs(dts))*1e3:.2f} ms; buf_flags 0x{m0['buf_flags']:x}")
