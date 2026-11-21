#!/usr/bin/env python3
"""Validate the pose-based motion-blur estimate against measured image sharpness on live recordings.

    re/blur_vs_sharp.py captures/live17 [captures/live21 ...]

blur_px (at the splat export's pinhole: 1600 px, 110 deg -> f 560) = f * t_exp * (omega + v / 1.5 m), omega / v from the
500 Hz HMD pose log averaged over the exposure window. Sharpness = splatcap._sharp (exposure-normalised Laplacian variance of
the green channel, quarter res). Prints the rank correlation and sharpness per blur bin (relative to each recording's median).
"""
import json, os, sys
import numpy as np, cv2
from scipy.stats import spearmanr
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from farstereo import LT
from splatcap import SplatCapture

F_EXPORT = 800 / np.tan(np.radians(55))


def sharp_robust(buf, shm):
    """noise-robust sharpness: half-res green, Gaussian sigma 1.5 (kills pixel noise), mean gradient magnitude / mean level"""
    a = np.frombuffer(buf, np.uint8, count=shm.line_bytes * 2464).reshape(2464, shm.line_bytes)[0::2, :3080].reshape(1232, 616, 5)[:, :, :4]
    g = a.reshape(1232, 2464)[:, 1::2].astype(np.float32)                 # G of the G/R... rows: one green sample per 2x2 cell
    g = cv2.GaussianBlur(g, (0, 0), 1.5)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0); gy = cv2.Sobel(g, cv2.CV_32F, 0, 1)
    return float(np.percentile(np.hypot(gx, gy), 90) / max(np.percentile(g, 90), 1))


class Shm:     # minimal stand-in for ArcShm geometry
    def __init__(self, d):
        s = json.load(open(os.path.join(d, "stream.json"))); self.line_bytes = s.get("line_bytes", 3088); self.skip = s.get("skip", 0)


def quat_R(q):   # w x y z
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


rows = []
for d in sys.argv[1:]:
    H = np.loadtxt(os.path.join(d, "hmd_500hz.txt")); H = H[H[:, 1] > 0]
    shm = Shm(d); bins = []
    for l in open(os.path.join(d, "frames.jsonl")):
        r = json.loads(l); m = r["left"]; t = m["ts"]; te = m["exp_lines"] * LT
        k = np.searchsorted(H[:, 0], [t - te / 2 - 0.004, t + te / 2 + 0.004])
        if k[1] - k[0] < 2 or k[1] >= len(H): continue
        a, b = H[k[0]], H[k[1]]; dt = b[0] - a[0]
        Ra, Rb = quat_R(a[2:6]), quat_R(b[2:6])
        om = np.linalg.norm(cv2.Rodrigues(Ra.T @ Rb)[0]) / dt; v = np.linalg.norm(b[6:9] - a[6:9]) / dt
        blur = F_EXPORT * te * (om + v / 1.5)
        fn = os.path.join(d, f"{r['i']:05d}_left.y10p")
        if not os.path.exists(fn): continue
        raw = open(fn, "rb").read()[shm.skip * shm.line_bytes:]
        sh = SplatCapture._sharp(raw, m, shm)
        bins.append((blur, sh, np.degrees(om), te * 1000, sharp_robust(raw, shm)))
    if len(bins) < 5: continue
    B = np.array(bins); B[:, 1] /= np.median(B[:, 1]); B[:, 4] /= np.median(B[:, 4])
    print(f"{d}: {len(B)} frames, exposure {np.median(B[:, 3]):.1f} ms, rho(blur, laplacian) {spearmanr(B[:, 0], B[:, 1])[0]:+.2f}"
          f"  rho(blur, robust gradient) {spearmanr(B[:, 0], B[:, 4])[0]:+.2f}")
    rows.append(B)
B = np.vstack(rows)
print(f"all: rho laplacian {spearmanr(B[:, 0], B[:, 1])[0]:+.2f}  robust {spearmanr(B[:, 0], B[:, 4])[0]:+.2f}")
edges = [0, 0.5, 0.75, 1, 1.5, 2, 3, 5, 1e9]
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (B[:, 0] >= lo) & (B[:, 0] < hi)
    if m.any(): print(f"  blur {lo:g}-{hi:g} px: n {m.sum():4d}  laplacian/median {np.median(B[m, 1]):.2f}  robust/median {np.median(B[m, 4]):.2f} (p25 {np.percentile(B[m, 4], 25):.2f})  omega {np.median(B[m, 2]):.1f} deg/s")
