#!/usr/bin/env python3
"""Live QR scanning from the Arcturus cameras (frames from src/arcgrab via /dev/shm/arcgrab).

usage: qrscan.py [--seconds 30] [--cams 0,1] [--save DIR]
Prints each newly decoded payload once (with camera, frame, timestamp and corner points).
The images are fisheye; QR detection works well near the centre, less so at the rim.
"""
import argparse, os, sys, time
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from arcshm import ArcShm

ap = argparse.ArgumentParser()
ap.add_argument("--seconds", type=float, default=30)
ap.add_argument("--cams", default="0,1")
ap.add_argument("--save", default=None)
a = ap.parse_args()

det = cv2.QRCodeDetector()
s = ArcShm()
cams = [int(c) for c in a.cams.split(",")]
last = {c: -1 for c in cams}
seen = {}
t0 = time.monotonic(); n = 0; tdet = 0.0
while time.monotonic() - t0 < a.seconds:
    for c in cams:
        try:
            meta, raw = s.latest(c, after_frame=last[c], timeout=1.0)
        except TimeoutError:
            continue
        last[c] = meta["frame"]
        g = s.gray(raw)
        g8 = np.clip(g / max(np.percentile(g, 99.5), 1) * 255, 0, 255).astype(np.uint8)
        t1 = time.monotonic()
        ok, texts, pts, _ = det.detectAndDecodeMulti(g8)
        tdet += time.monotonic() - t1; n += 1
        if ok:
            for txt, p in zip(texts, pts):
                if not txt:
                    continue
                if txt not in seen:
                    print(f"QR cam{c} frame {meta['frame']} t={meta['ts']:.3f}: {txt!r} corners(half-res px)={np.round(p).astype(int).tolist()}", flush=True)
                    if a.save:
                        os.makedirs(a.save, exist_ok=True)
                        im = cv2.cvtColor(g8, cv2.COLOR_GRAY2BGR); cv2.polylines(im, [p.astype(np.int32)], True, (0, 255, 0), 3)
                        cv2.imwrite(os.path.join(a.save, f"qr_cam{c}_{meta['frame']}.jpg"), im)
                seen[txt] = seen.get(txt, 0) + 1
print(f"done: {n} frames scanned, {tdet / max(n, 1) * 1000:.0f} ms/frame detect, payloads: {seen}", file=sys.stderr)
