#!/usr/bin/env python3
"""Live origin marker (AprilTag 41h12 id 4): ties virtual and real coordinates together, repeatedly.

The origin marker is single-instance and guaranteed not to move during a capture session. OriginTracker watches for it in
the left camera, collects its pose in STAGE (SteamVR's world) until a reference is frozen (n_ref good views), and then
reports for every detection
  * stage_from_origin  - the frozen reference pose (marker frame: x right, y up, z out of the paper)
  * dev                - how far this frame's measured marker pose is from the reference = the current SteamVR pose error
  * correction         - 4x4 C with  true_stage_from_cam = C @ tracked_stage_from_cam  (smoothed)
Published to /dev/shm/pcorigin.json (state, n, stage_from_origin[12], dev_mm, dev_deg, correction[12], age).
More immobile markers can be added later: pass several ids / boards; each gets its own Anchor.
"""
import json, os, threading, time
import numpy as np
import cv2
import fiducials as F


class OriginTracker:
    def __init__(self, ls, ids=(F.ORIGIN_ID,), every=3, n_ref=12, min_edge=12.0, max_rms_px=0.12, path="/dev/shm/pcorigin.json"):
        self.ls = ls; self.ids = tuple(ids); self.every = every; self.min_edge = min_edge; self.max_rms_px = max_rms_px; self.path = path
        def half(K):
            Kh = K.copy(); Kh[0, 0] /= 2; Kh[1, 1] /= 2; Kh[0, 2] = (K[0, 2] - 0.5) / 2; Kh[1, 2] = (K[1, 2] - 0.5) / 2; return Kh
        self.K, self.D = half(ls.KL), ls.DL
        self.anchor = F.Anchor(n_ref=n_ref); self.n = 0; self.seq = 0; self.calls = 0
        self.state = "searching"; self.dev = None; self.corr = np.eye(4); self.t_seen = None; self.nlocked_seen = 0
        self.edge_ref = []; self._busy = False; self._job = None; self._thread = None; self._ev = threading.Event()

    def _xn(self, c):
        return cv2.fisheye.undistortPoints(np.asarray(c, np.float64).reshape(-1, 1, 2), self.K, self.D).reshape(-1, 2)

    def submit(self, cL, pose34):
        """Non-blocking: hand the newest frame to the worker thread (frames arriving while it is busy are dropped)."""
        if self._busy: return
        self._busy = True; self._job = (cL, pose34)
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True); self._thread.start()
        self._ev.set()

    def _run(self):
        while True:
            self._ev.wait(); self._ev.clear()
            try:
                cL, pose34 = self._job; self.every = 1; self.calls = 0
                self.update(cL, pose34)
            except Exception as e:                       # never take the server down
                print("origin tracker:", e, flush=True)
            self._busy = False

    def update(self, cL, pose34, ts=None):
        """cL: half-resolution BGR float image (tsdf.bin_raw10 output); pose34: tracked head pose 3x4 (STAGE)."""
        self.calls += 1
        if self.calls % self.every: return self.state
        g = cv2.cvtColor(cL, cv2.COLOR_BGR2GRAY)
        g8 = np.clip(g * (255 / max(float(np.percentile(g[::4, ::4], 99.7)), 1.0)), 0, 255).astype(np.uint8)
        return self.feed(F.detect(g8, min_edge=7, blocks=(31,)), pose34)

    def feed(self, dets, pose34):
        """dets: fiducials.detect() output on the half-resolution left image (any ids; the origin id is picked here)."""
        dets = [d for d in dets if d["id"] in self.ids and d["ham"] <= 1 and d["edge_px"] >= self.min_edge * 0.5]
        if not dets: self._write(); return self.state
        d = max(dets, key=lambda x: x["edge_px"])
        H = np.eye(4); H[:3, :] = np.asarray(pose34); W = H @ self.ls.head_from_camL
        obj = F.board_object("origin", [d["id"]])[d["id"]]; xn = self._xn(d["corners"])
        prior = None
        if self.anchor.ref is not None: prior = np.linalg.inv(W) @ self.anchor.ref
        cands = F.solve_pose_all(obj, xn)
        if not cands: self._write(); return self.state
        if prior is None:
            if len(cands) > 1 and cands[1][1] < 1.6 * cands[0][1]: self._write(); return self.state   # flip not resolvable yet
            T = cands[0][0]
        else:
            T, rms = min(cands, key=lambda c: F.pose_diff(prior, c[0])[1])
            cands = [(T, rms)]
        if cands[0][1] * self.K[0, 0] > self.max_rms_px: self._write(); return self.state          # poorly refined corners
        Tsm = W @ T; self.n += 1; self.t_seen = time.monotonic()
        self.anchor.add(Tsm, w=d["edge_px"] ** 2)
        if self.anchor.ref is None:
            self.state = "collecting"
        else:
            self.state = "locked"
            dp, da = F.pose_diff(self.anchor.ref, Tsm)
            self.dev = (float(np.linalg.norm(self.anchor.ref[:3, :3] @ dp) * 1000), da)
            C = self.anchor.correction(W, T)
            a = 0.2; self.corr = F.mean_pose([self.corr, C], [1 - a, a])        # smoothed
        self._write()
        return self.state

    @property
    def stage_from_origin(self):
        return self.anchor.ref

    def to_origin_frame(self, P):
        """STAGE points (N,3) -> origin-marker frame, or None when not locked."""
        R = self.anchor.ref
        return None if R is None else (np.asarray(P) - R[:3, 3]) @ R[:3, :3]

    def _write(self):
        self.seq += 1
        d = dict(seq=self.seq, state=self.state, n=self.n, ids=list(self.ids),
                 age=None if self.t_seen is None else round(time.monotonic() - self.t_seen, 2))
        if self.anchor.ref is not None:
            d["stage_from_origin"] = self.anchor.ref[:3, :].ravel().round(6).tolist()
            d["correction"] = self.corr[:3, :].ravel().round(6).tolist()
        if self.dev: d["dev_mm"], d["dev_deg"] = round(self.dev[0], 1), round(self.dev[1], 2)
        try:
            tmp = self.path + ".tmp"; json.dump(d, open(tmp, "w")); os.replace(tmp, self.path)
        except OSError:
            pass

    def hud(self):
        if self.state == "locked" and self.dev:
            return f"origin: locked, SteamVR dev {self.dev[0]:.0f} mm / {self.dev[1]:.1f}°"
        return f"origin: {self.state}"
