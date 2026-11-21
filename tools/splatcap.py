#!/usr/bin/env python3
"""Capture-for-splatting: pose-stamped full-resolution stereo pairs, selected online for spatial coverage.

SplatCapture runs its own reader thread on /dev/shm/arcgrab (independent of the point-cloud scan / REC state) and, while
`enabled`, writes a recording in the liverec.py format (OUT/NNNNN_{left,right}.y10p + frames.jsonl + stream.json) to
captures/splat/splat_YYYYmmdd_HHMMSS/, ready for  tools/export_colmap.py DIR OUTDIR.
A pair is stored when
  * it is sharp (exposure-normalised Laplacian variance >= 0.5 x the running median of the recent frames),
  * the cameras are synchronous and the pose is valid, and
  * its view is new: no stored view is closer than `min_dist` in (head position + 0.5 m/rad * viewing angle).
So standing still stores nothing, slow walking stores a view every few cm, and revisits of covered space are skipped.
Stops by itself at `max_gb`.  (The export then picks the final, denser/sparser subset with feature / dwell weighting.)
"""
import json, os, subprocess, sys, threading, time
import numpy as np
import cv2
from arcshm import ArcShm

W_ANG = 0.5


class SplatCapture:
    def __init__(self, root="captures/splat", min_dist=0.06, max_hz=3.0, max_gb=6.0, pair_tol=0.012):
        self.root = root; self.min_dist = min_dist; self.max_hz = max_hz; self.max_bytes = max_gb * 1e9; self.pair_tol = pair_tol
        self.enabled = False; self.dir = None; self.n = 0; self.bytes = 0; self.msg = ""
        self._stop = False; self._thread = None; self.last_dir = None; self.export = None; self.export_msg = ""

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True); self._thread.start()

    def stop(self):
        self._stop = True

    def start_export(self, extra=()):
        """Background tools/export_colmap.py on the latest capture (masks + depth + points from the capture itself) -> <dir>_colmap."""
        if self.enabled: self.export_msg = "export: switch SPLAT off first"; return False
        if not self.last_dir or not os.path.exists(os.path.join(self.last_dir, "frames.jsonl")) or self.n < 3:
            self.export_msg = "export: no splat capture yet"; return False
        if self.export is not None and self.export.poll() is None: return False
        out = self.last_dir.rstrip("/") + "_colmap"; here = os.path.dirname(os.path.abspath(__file__))
        log = open(self.last_dir.rstrip("/") + "_export.log", "w")
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PC_EXPORT_THREADS="1")   # leave the cores to the live pipeline
        self.export = subprocess.Popen(["nice", "-n", "19", "ionice", "-c3", sys.executable, "-I", os.path.join(here, "export_colmap.py"), self.last_dir, out,
                                        "--masks", "--depth", *extra], env=env, stdout=log, stderr=subprocess.STDOUT)
        self.export_out = out; self.export_msg = ""; print(f"splat export -> {out}", flush=True); return True

    def progress(self):
        try: return float(json.load(open(self.export_out.rstrip("/") + ".progress"))["frac"])
        except Exception: return None

    def export_state(self):
        if self.export is None: return self.export_msg
        rc = self.export.poll()
        if rc is None:
            f = self.progress()
            return "export: starting ..." if f is None else f"export [{'#' * int(round(f * 10))}{'-' * (10 - int(round(f * 10)))}] {int(f * 100)}%"
        return f"export: done -> {os.path.basename(self.export_out)}" if rc == 0 else "export: FAILED (see *_export.log)"

    def hud(self):
        e = self.export_state()
        if not self.enabled and not self.n: return e
        return (e + "   " if e else "") + f"splat: {'REC' if self.enabled else 'idle'} {self.n} views {self.bytes / 1e9:.1f} GB{(' ' + self.msg) if self.msg else ''}"

    # ---- helpers -------------------------------------------------------------------------------------------------------
    @staticmethod
    def _sharp(buf, meta, shm):
        a = np.frombuffer(buf, np.uint8, count=shm.line_bytes * 2464).reshape(2464, shm.line_bytes)[0::4, :3080].reshape(616, 616, 5)[:, :, 1]
        g = a.astype(np.float32)
        g = np.clip(g * 4 - 64, 0, None) / max(meta["exp_lines"], 1)                 # 8-bit MSBs of the green channel, exposure-normalised
        g = np.sqrt(np.clip(g / max(np.percentile(g, 99), 1e-9), 0, 1)).astype(np.float32)
        return float(cv2.Laplacian(g, cv2.CV_32F).var())

    @staticmethod
    def _view(pose):
        P = np.asarray(pose); return P[:, 3].copy(), P[:, 2].copy()                 # head position, head z axis (STAGE)

    def _novel(self, pos, z):
        if not self.views: return True
        V = np.array([p for p, _ in self.views]); Z = np.array([q for _, q in self.views])
        d = np.linalg.norm(V - pos, axis=1) + W_ANG * np.arccos(np.clip(Z @ z, -1, 1))
        return d.min() > self.min_dist

    def _open(self):
        self.dir = os.path.join(self.root, time.strftime("splat_%Y%m%d_%H%M%S")); os.makedirs(self.dir, exist_ok=True); self.last_dir = self.dir
        s = self.shm
        json.dump(dict(line_px=s.line_px, line_bytes=s.line_bytes, black=s.black, bayer="RGGB", cams=["left", "right"], kind="splat-capture",
                       devices=["/dev/video0 arcimx616 0-001a crop x=550", "/dev/video3 arcimx616 0-0010 crop x=250"]),
                  open(os.path.join(self.dir, "stream.json"), "w"), indent=1)
        self.log = open(os.path.join(self.dir, "frames.jsonl"), "w")
        self.n = 0; self.bytes = 0; self.views = []; self.sharp = []; self.msg = ""

    def _close(self):
        try: self.log.close()
        except Exception: pass
        if self.dir: print(f"splat capture: {self.n} views, {self.bytes / 1e9:.2f} GB -> {self.dir}", flush=True)
        self.dir = None

    # ---- thread ----------------------------------------------------------------------------------------------------------
    def _run(self):
        self.shm = ArcShm(); last = -1; self.views = []; self.sharp = []; t_last = 0.0; was = False
        while not self._stop:
            if self.enabled != was:
                was = self.enabled
                if was: self._open()
                else: self._close()
            if not self.enabled:
                time.sleep(0.1); continue
            if self.bytes >= self.max_bytes:
                self.msg = "disk cap reached"; self.enabled = False; continue
            try:
                m0, b0 = self.shm.latest_packed(0, after_frame=last, timeout=0.5)
                m1, b1 = self.shm.latest_packed(1, after_frame=m0["frame"] - 2, timeout=0.5)
            except TimeoutError:
                continue
            last = m0["frame"]
            if time.monotonic() - t_last < 1.0 / self.max_hz: continue
            if not (m0["pose_valid"] and m1["pose_valid"]) or abs(m0["ts"] - m1["ts"]) > self.pair_tol: continue
            try:
                sh = self._sharp(b0, m0, self.shm)
                self.sharp = (self.sharp + [sh])[-60:]
                if len(self.sharp) >= 5 and sh < 0.5 * float(np.median(self.sharp)): continue          # motion blur
                pos, z = self._view(m0["pose"])
                if not self._novel(pos, z): continue
                for name, b in (("left", b0), ("right", b1)):
                    with open(os.path.join(self.dir, f"{self.n:05d}_{name}.y10p"), "wb") as f: f.write(b)
                rec = {"i": self.n}
                for name, m in (("left", m0), ("right", m1)):
                    rec[name] = dict(frame=m["frame"], ts=m["ts"], exp_lines=m["exp_lines"], gain_code=m["gain_code"], dgain=m["dgain"],
                                     pose=m["pose"].tolist(), pose_valid=m["pose_valid"], buf_flags=m["buf_flags"])
                self.log.write(json.dumps(rec) + "\n"); self.log.flush()
                self.views.append((pos, z)); self.n += 1; self.bytes += len(b0) + len(b1); t_last = time.monotonic()
            except Exception as e:
                print("splat capture:", e, flush=True); time.sleep(0.5)
        if self.enabled or self.dir: self._close()
