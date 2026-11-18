#!/usr/bin/env python3
"""GNSS receiver on the headset -> georeference of the STAGE frame (absolute positions for scans/objects).

Reads NMEA (GGA + optional GST/RMC) from a USB serial receiver, pairs every fix with the HMD pose at the
fix time (src/posestream, OpenVR Standing == OpenXR STAGE), moves the pose to the antenna with the lever
arm, and fits ENU = Rz(yaw) M stage + t (4 DoF; gravity is known on both sides; tools/geo.py) once the
antenna positions spread enough horizontally. Publishes for the viewer (and scene saves / export):
  /dev/shm/gnss.json    latest fix {fix, q, sats, hacc, vacc, lat, lon, h, age}
  /dev/shm/georef.json  {ok, lat0, lon0, h0, yaw, yaw_deg, yaw_sigma_deg, t, rms, n, spread, lever}
and logs raw input to captures/gnss/gnss_<date>.log ("<t_monotonic> <NMEA sentence>") for reprocessing.

usage:
  gnssd.py --port /dev/ttyACM0 [--baud 115200] [--lever 0 0.12 0.05] [--min-spread 3]
  gnssd.py --offline LOG --pose-file hmd_500hz.txt   (reprocess a log; also the simulator test)
Fix time: the GGA UTC time is mapped to CLOCK_MONOTONIC with the minimum observed (t_received - t_utc)
over all fixes (serial latency varies, its minimum is close to the true offset), + --latency-floor.
Lever arm: antenna phase centre in the HMD frame (x right, y up, z back), metres.
"""
import argparse, json, os, sys, threading, time, subprocess
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import geo

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
QUAL = {0: "none", 1: "GPS", 2: "DGPS", 3: "PPS", 4: "RTK fixed", 5: "RTK float", 6: "dead reckoning", 7: "manual", 8: "sim"}
UERE = {1: 2.5, 2: 0.7, 4: 0.02, 5: 0.3, 8: 0.02}     # rough 1-sigma per HDOP when no GST


# ------------------------------------------------------------------ NMEA
def nmea_ok(s):
    s = s.strip()
    if not s.startswith("$") or "*" not in s:
        return None
    body, cs = s[1:].rsplit("*", 1)
    c = 0
    for ch in body.encode(errors="ignore"):
        c ^= ch
    try:
        return body.split(",") if int(cs[:2], 16) == c else None
    except ValueError:
        return None


def _deg(v, hemi):
    if not v:
        return None
    d = int(float(v) / 100); m = float(v) - 100 * d
    x = d + m / 60
    return -x if hemi in ("S", "W") else x


def _utc_s(v):
    return int(v[0:2]) * 3600 + int(v[2:4]) * 60 + float(v[4:]) if v else None


class Nmea:
    def __init__(self):
        self.gst = None; self.date = None

    def feed(self, s):
        """-> fix dict for GGA sentences (with GST accuracy if seen for the same epoch), else None"""
        f = nmea_ok(s)
        if not f:
            return None
        typ = f[0][2:]
        try:
            if typ == "GST" and len(f) >= 9:
                lat_sd, lon_sd, alt_sd = (float(x) if x else None for x in f[6:9])
                self.gst = (_utc_s(f[1]), lat_sd, lon_sd, alt_sd)
            elif typ == "RMC" and len(f) >= 10:
                self.date = f[9]
            elif typ == "GGA" and len(f) >= 12:
                q = int(f[6] or 0)
                if q == 0:
                    return {"q": 0}
                utc = _utc_s(f[1]); hdop = float(f[8] or 99)
                fix = {"q": q, "utc": utc, "lat": _deg(f[2], f[3]), "lon": _deg(f[4], f[5]), "sats": int(f[7] or 0), "hdop": hdop,
                       "h": float(f[9] or 0) + float(f[11] or 0)}       # ellipsoidal = MSL + geoid separation
                if self.gst and self.gst[0] is not None and abs(self.gst[0] - utc) < 0.05 and self.gst[1] is not None:
                    fix["hacc"] = float(np.hypot(self.gst[1], self.gst[2])); fix["vacc"] = self.gst[3] or 2 * fix["hacc"]
                else:
                    fix["hacc"] = UERE.get(q, 5.0) * hdop; fix["vacc"] = 2 * fix["hacc"]
                return fix
        except (ValueError, IndexError):
            return None
        return None


# ------------------------------------------------------------------ poses
class Poses:
    """HMD pose history (t valid qw qx qy qz px py pz), from src/posestream or a file."""
    def __init__(self, n=800000):          # 100 Hz x 2 h
        self.buf = np.zeros((n, 9)); self.n = 0; self.lock = threading.Lock()

    def add(self, row):
        with self.lock:
            if self.n == len(self.buf):
                self.buf[:-len(self.buf) // 2] = self.buf[len(self.buf) // 2:]; self.n = len(self.buf) - len(self.buf) // 2
            self.buf[self.n] = row; self.n += 1

    def load(self, fn):
        H = np.loadtxt(fn); self.buf = H.copy(); self.n = len(H)

    def at(self, t):
        """-> (R 3x3, p) at time t, or None (no valid pose within 20 ms on both sides)"""
        with self.lock:
            B = self.buf[:self.n]
            if self.n < 2 or t < B[0, 0] or t > B[-1, 0]:
                return None
            i = int(np.searchsorted(B[:, 0], t))
            a, b = B[max(i - 1, 0)], B[min(i, self.n - 1)]
        if not (a[1] > 0 and b[1] > 0) or b[0] - a[0] > 0.04:
            return None
        w = 0 if b[0] == a[0] else (t - a[0]) / (b[0] - a[0])
        qa, qb = a[2:6], b[2:6] * (1 if np.dot(a[2:6], b[2:6]) >= 0 else -1)
        q = (1 - w) * qa + w * qb; q /= np.linalg.norm(q)
        ww, x, y, z = q
        R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * ww), 2 * (x * z + y * ww)],
                      [2 * (x * y + z * ww), 1 - 2 * (x * x + z * z), 2 * (y * z - x * ww)],
                      [2 * (x * z - y * ww), 2 * (y * z + x * ww), 1 - 2 * (x * x + y * y)]])
        return R, (1 - w) * a[6:9] + w * b[6:9]


# ------------------------------------------------------------------ georeference
class Georef:
    def __init__(self, lever, min_spread=3.0, max_hacc=0.5, latency_floor=0.0):
        self.lever = np.asarray(lever, float); self.min_spread = min_spread; self.max_hacc = max_hacc
        self.lat_floor = latency_floor
        self.raw = []          # (t_rx, fix)
        self.off = None        # min(t_rx - utc)
        self.origin = None; self.result = {"ok": False}

    def add(self, t_rx, fix, poses):
        if fix.get("q", 0) == 0 or fix.get("utc") is None:
            return
        o = t_rx - fix["utc"]
        if self.off is not None and abs(o - self.off) > 3600:      # UTC day wrap (or clock jump): restart offset
            self.off = None
        self.off = o if self.off is None else min(self.off, o)
        self.raw.append((t_rx, fix))

    def solve(self, poses):
        if self.off is None:
            return self.result
        S, E, W = [], [], []
        step = max(1, len(self.raw) // 3000)    # bound the cost on long sessions
        for t_rx, f in self.raw[::step]:
            if f["hacc"] > self.max_hacc:
                continue
            t = f["utc"] + self.off + self.lat_floor
            if t_rx - t > 1.0:                       # outdated offset (day wrap); skip
                continue
            pr = poses.at(t)
            if pr is None:
                continue
            R, p = pr
            if self.origin is None:
                self.origin = (f["lat"], f["lon"], f["h"])
            S.append(R @ self.lever + p)
            E.append(geo.lla_to_enu(f["lat"], f["lon"], f["h"], *self.origin)); W.append(1 / max(f["hacc"], 0.005) ** 2)
        n = len(S)
        res = {"ok": False, "n": n, "lever": self.lever.tolist()}
        if n >= 5:
            S, E, W = np.array(S), np.array(E), np.array(W)
            spread = float(np.sqrt(np.max(np.linalg.eigvalsh(np.cov((S @ geo.M_STAGE_TO_ENU0.T)[:, :2].T)))) * 2)
            keep = np.ones(n, bool)
            for _ in range(3):
                yaw, t, r = geo.fit_yaw_translation(S[keep], E[keep], W[keep])
                r_all = np.linalg.norm(S @ (geo.Rz(yaw) @ geo.M_STAGE_TO_ENU0).T + t - E, axis=1)
                sig = 1 / np.sqrt(W)
                new = r_all < np.maximum(4 * sig, 0.05)
                if new.sum() < 5 or (new == keep).all():
                    break
                keep = new
            rms = float(np.sqrt(np.mean(r_all[keep] ** 2)))
            yaw_sigma = float(np.degrees(np.median(1 / np.sqrt(W[keep])) / max(spread / 2, 1e-3) / np.sqrt(keep.sum())))
            res.update(ok=bool(spread >= self.min_spread and keep.sum() >= 5), lat0=self.origin[0], lon0=self.origin[1], h0=self.origin[2],
                       yaw=float(yaw), yaw_deg=float(np.degrees(yaw)), yaw_sigma_deg=yaw_sigma, t=[float(x) for x in t],
                       rms=rms, n=int(keep.sum()), spread=spread, time_offset=self.off)
        self.result = res
        return res


def write_json(path, d):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f)
    os.replace(tmp, path)


def open_serial(port, baud):
    import termios
    fd = os.open(port, os.O_RDONLY | os.O_NOCTTY)
    at = termios.tcgetattr(fd)
    speed = getattr(termios, f"B{baud}", termios.B115200)
    at[0] = 0; at[1] = 0; at[2] = termios.CS8 | termios.CREAD | termios.CLOCAL; at[3] = 0; at[4] = speed; at[5] = speed
    at[6][termios.VMIN] = 1; at[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, at)
    return os.fdopen(fd, "rb", buffering=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port"); ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--lever", type=float, nargs=3, default=[0.0, 0.12, 0.05], help="antenna in HMD frame (x right, y up, z back), m")
    ap.add_argument("--min-spread", type=float, default=3.0, help="horizontal spread (m) of fixes before the georef is trusted")
    ap.add_argument("--max-hacc", type=float, default=0.5, help="ignore fixes with worse horizontal accuracy (m)")
    ap.add_argument("--latency-floor", type=float, default=0.0)
    ap.add_argument("--offline"); ap.add_argument("--pose-file")
    ap.add_argument("--no-shm", action="store_true")
    a = ap.parse_args()
    poses = Poses(); nm = Nmea(); gr = Georef(a.lever, a.min_spread, a.max_hacc, a.latency_floor)

    if a.offline:
        poses.load(a.pose_file)
        for line in open(a.offline):
            t, s = line.split(" ", 1)
            f = nm.feed(s)
            if f:
                gr.add(float(t), f, poses)
        r = gr.solve(poses)
        print(json.dumps(r, indent=1))
        if not a.no_shm:
            write_json("/dev/shm/georef.json", r)
        return

    if not a.port:
        sys.exit("--port or --offline needed")
    if a.pose_file:
        poses.load(a.pose_file)
    else:
        ps = subprocess.Popen([os.path.join(ROOT, "src", "posestream"), "--hz", "100"], stdout=subprocess.PIPE, text=True)
        def pose_reader():
            for line in ps.stdout:
                try: poses.add(np.array(line.split(), float))
                except ValueError: pass
        threading.Thread(target=pose_reader, daemon=True).start()
    os.makedirs(os.path.join(ROOT, "captures", "gnss"), exist_ok=True)
    log = open(os.path.join(ROOT, "captures", "gnss", time.strftime("gnss_%Y%m%d_%H%M%S.log")), "a")
    ser = open_serial(a.port, a.baud)
    buf = b""; last_solve = 0; last = None
    print(f"gnssd: reading {a.port}, lever {a.lever}", flush=True)
    while True:
        chunk = ser.read(512)
        if not chunk:
            time.sleep(0.01); continue
        t_rx = time.monotonic(); buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            s = line.decode(errors="ignore").strip()
            if not s:
                continue
            log.write(f"{t_rx:.6f} {s}\n")
            f = nm.feed(s)
            if f is None:
                continue
            if f.get("q"):
                gr.add(t_rx, f, poses); last = (t_rx, f)
            g = {"fix": QUAL.get(f.get("q", 0), str(f.get("q"))), "q": f.get("q", 0), "t": time.time()}
            if f.get("q"):
                g.update(sats=f["sats"], hacc=round(f["hacc"], 3), vacc=round(f["vacc"], 3), lat=f["lat"], lon=f["lon"], h=f["h"])
            write_json("/dev/shm/gnss.json", g)
        if time.monotonic() - last_solve > 1.0:
            last_solve = time.monotonic(); log.flush()
            r = gr.solve(poses); write_json("/dev/shm/georef.json", r)


if __name__ == "__main__":
    main()
