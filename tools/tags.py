#!/usr/bin/env python3
"""Tags in view: AprilTag fiducials (tools/fiducials.py) and QR codes (zxing-cpp), placed in 3-D for the viewer to label.

TagWatcher.submit(cL, pose34, Z, ls) is non-blocking (worker thread, newest frame wins). Per frame:
  * fiducials: all known ids (fiducials.BOARDS): boards seen with >= 2 tags are solved jointly (unambiguous), single tags use the
    two-solution PnP and prefer the pose whose normal is closest to a world axis (floor / ceiling / wall markers); the origin id also
    feeds OriginTracker (reference lock + live SteamVR deviation).
  * QR codes: decoded on the grey image; no physical size is known, so the 3-D placement comes from the stereo depth Z inside the code's
    polygon (plane fit -> centre, normal, size, corner positions).
Items persist in a dict (matched by kind + id/text + position) and are published to /dev/shm/pctags.json for `ttl` seconds after they
were last seen:  {"seq", "items": [{"key", "kind": "fid"|"qr", "id" | "text", "T": 3x4 STAGE-from-tag row-major,
"size": m (edge), "corners": 4x3 STAGE (QR), "age": s}]}   (fid frame: x right, y up, z out of the paper)
"""
import json, os, threading, time
import numpy as np
import cv2
import fiducials as F
try:
    import zxingcpp
except ImportError:                                      # QR decoding is optional
    zxingcpp = None


class TagWatcher:
    def __init__(self, ls, origin=None, path="/dev/shm/pctags.json", ttl=20.0, min_period=0.25):
        self.ls = ls; self.origin = origin; self.path = path; self.ttl = ttl; self.min_period = min_period
        def half(K):
            Kh = K.copy(); Kh[0, 0] /= 2; Kh[1, 1] /= 2; Kh[0, 2] = (K[0, 2] - 0.5) / 2; Kh[1, 2] = (K[1, 2] - 0.5) / 2; return Kh
        self.K, self.D = half(ls.KL), ls.DL
        self.enabled = False                              # publish fiducial / QR items (viewer toggle); the origin marker always runs
        self.items = {}; self.seq = 0; self._busy = False; self._job = None; self._thread = None; self._ev = threading.Event(); self._t_last = 0.0

    # ---------------------------------------------------------------- worker plumbing
    def submit(self, cL, pose34, Z):
        if self._busy or time.monotonic() - self._t_last < self.min_period: return
        self._busy = True; self._job = (cL, pose34, Z)
        if self._thread is None:
            self._thread = threading.Thread(target=self._run, daemon=True); self._thread.start()
        self._ev.set()

    def _run(self):
        while True:
            self._ev.wait(); self._ev.clear()
            try:
                self.process(*self._job)
            except Exception as e:
                print("tag watcher:", e, flush=True)
            self._t_last = time.monotonic(); self._busy = False

    # ---------------------------------------------------------------- geometry helpers
    def _xn(self, c):
        return cv2.fisheye.undistortPoints(np.asarray(c, np.float64).reshape(-1, 1, 2), self.K, self.D).reshape(-1, 2)

    def _rect_px(self, c):
        """raw half-res pixels (N,2) -> rectified-left pixels (N,2) at the LiveStereo scale"""
        xn = self._xn(c); r = (self.ls.R1 @ np.c_[xn, np.ones(len(xn))].T).T
        return self.ls.f * r[:, :2] / r[:, 2:3] + np.array([self.ls.P1[0, 2], self.ls.P1[1, 2]]), r

    # ---------------------------------------------------------------- per frame
    def process(self, cL, pose34, Z):
        g = cv2.cvtColor(cL, cv2.COLOR_BGR2GRAY)
        g8 = np.clip(g * (255 / max(float(np.percentile(g[::4, ::4], 99.7)), 1.0)), 0, 255).astype(np.uint8)
        dets = F.detect(g8, min_edge=7, blocks=(31,))
        if self.origin: self.origin.feed(dets, pose34)
        if not self.enabled: return
        H = np.eye(4); H[:3, :] = np.asarray(pose34); W = H @ self.ls.head_from_camL          # STAGE <- camL (raw sensor axes)
        now = time.monotonic()
        for it in self._fiducials(dets, W): self._merge(it, now)
        if zxingcpp is not None and Z is not None:
            for it in self._qr(g8, Z, pose34): self._merge(it, now)
        self._publish(now)

    def _fiducials(self, dets, W):
        dets = [d for d in dets if d["ham"] <= 1 and F.board_of(d["id"]) and d["edge_px"] >= 6]
        out = []
        xn_of = self._xn
        for grp in F.group_boards(dets, W, xn_of, radius=0.6):
            b = F.board_of(grp[0]["id"]); O = F.board_object(b, [d["id"] for d in grp])
            if len(grp) >= 2:
                T, rms = F.solve_board(grp, xn_of)
                if T is None or rms * self.K[0, 0] > 1.5: continue
                for d in grp:                                       # tag pose = board pose * (tag centre offset in the board frame)
                    cx, cy = F.BOARDS[b]["centres_mm"][d["id"]]; To = np.eye(4); To[:3, 3] = [cx / 1000, cy / 1000, 0]
                    out.append(self._fid_item(d["id"], W @ T @ To, b, len(grp)))
            else:
                d = grp[0]; cands = F.solve_pose_all(O[d["id"]], xn_of(d["corners"]))
                if not cands: continue
                # a single small tag has two near-equal solutions: take the one whose normal is closest to a world axis (markers sit on floor / ceiling / walls)
                def axisness(c):
                    n = (W @ c[0])[:3, 2]; return max(abs(n[0]), abs(n[1]), abs(n[2]))
                c = max(cands, key=axisness) if len(cands) > 1 and cands[1][1] < 1.6 * cands[0][1] else cands[0]
                if c[1] * self.K[0, 0] > 0.3: continue
                cx, cy = F.BOARDS[b]["centres_mm"][d["id"]]; To = np.eye(4); To[:3, 3] = [cx / 1000, cy / 1000, 0]
                out.append(self._fid_item(d["id"], W @ c[0] @ To, b, 1))
        return out

    def _fid_item(self, tag_id, T, board, ntag):
        # NB: board_object() placed the tag at its board position; T here is the board pose composed with the tag offset => tag origin
        return dict(kind="fid", id=int(tag_id), T=np.asarray(T)[:3, :].ravel().round(5).tolist(), size=F.BOARDS[board]["tag_mm"] / 1000.0, ntag=ntag)

    def _qr(self, g8, Z, pose34):
        out = []
        try:
            res = zxingcpp.read_barcodes(g8, formats=zxingcpp.BarcodeFormat.QRCode | zxingcpp.BarcodeFormat.MicroQRCode)
        except Exception:
            return out
        Wr = self.ls.world_from_rect(pose34)
        for b in res:
            if not b.valid or not b.text: continue
            p = b.position; c = np.array([[p.top_left.x, p.top_left.y], [p.top_right.x, p.top_right.y],
                                          [p.bottom_right.x, p.bottom_right.y], [p.bottom_left.x, p.bottom_left.y]], float)
            # sample the depth inside the code (bilinear grid over the quad), in rectified coordinates
            u, v = np.meshgrid(np.linspace(0.1, 0.9, 9), np.linspace(0.1, 0.9, 9))
            P = ((1 - v)[..., None] * ((1 - u)[..., None] * c[0] + u[..., None] * c[1]) + v[..., None] * ((1 - u)[..., None] * c[3] + u[..., None] * c[2])).reshape(-1, 2)
            rp, ray = self._rect_px(P)
            xi = np.round(rp[:, 0]).astype(int); yi = np.round(rp[:, 1]).astype(int)
            ok = (xi >= 0) & (xi < Z.shape[1]) & (yi >= 0) & (yi < Z.shape[0])
            z = np.zeros(len(P)); z[ok] = Z[yi[ok], xi[ok]]
            ok &= z > 0
            if ok.sum() < 12: continue
            zm = np.median(z[ok]); ok &= np.abs(z - zm) < 0.15
            if ok.sum() < 10: continue
            X = ray[ok] / ray[ok][:, 2:3] * z[ok][:, None]                                   # rectified-camera 3-D points of the code surface
            ctr = X.mean(0); _, _, vt = np.linalg.svd(X - ctr, full_matrices=False); n = vt[2]
            if n[2] > 0: n = -n                                                              # normal towards the camera
            # corners: intersect their rays with the plane
            _, rc = self._rect_px(c); cor = []
            for r in rc:
                den = r @ n
                cor.append(r / r[2] * ((ctr @ n) / (den / r[2])) if abs(den) > 1e-6 else ctr)
            cor = np.array(cor); cs = (Wr[:3, :3] @ cor.T).T + Wr[:3, 3]; n_s = Wr[:3, :3] @ n
            ex = cs[1] - cs[0]; up = cs[0] - cs[3]                                          # code's own right / up in STAGE
            ex = ex - n_s * (ex @ n_s); ex /= np.linalg.norm(ex) + 1e-9
            upv = np.cross(n_s, ex)                                                          # right-handed (x, y=up, z=n)
            if upv @ up < 0:                                                                 # mirrored reading: keep text upright anyway
                ex = -ex; upv = np.cross(n_s, ex)
            T = np.eye(4); T[:3, 0] = ex; T[:3, 1] = upv; T[:3, 2] = n_s; T[:3, 3] = cs.mean(0)
            size = float(0.5 * (np.linalg.norm(cs[1] - cs[0]) + np.linalg.norm(cs[3] - cs[0])))
            if not (0.01 < size < 1.5): continue
            out.append(dict(kind="qr", text=b.text, T=T[:3, :].ravel().round(5).tolist(), size=round(size, 4), corners=cs.round(4).tolist()))
        return out

    # ---------------------------------------------------------------- persistence / output
    def _merge(self, it, now):
        pos = np.array(it["T"]).reshape(3, 4)[:, 3]
        for k, old in self.items.items():
            if old["kind"] != it["kind"]: continue
            same = old.get("id") == it.get("id") if it["kind"] == "fid" else old.get("text") == it.get("text")
            if same and np.linalg.norm(np.array(old["T"]).reshape(3, 4)[:, 3] - pos) < 0.5:
                it["key"] = k; it["seen"] = now; self.items[k] = it; return
        k = f"{it['kind']}:{it.get('id', it.get('text'))}:{len(self.items)}:{int(now)}"
        it["key"] = k; it["seen"] = now; self.items[k] = it

    def _publish(self, now):
        for k in [k for k, v in self.items.items() if now - v["seen"] > self.ttl]: del self.items[k]
        self.seq += 1
        out = [dict({kk: vv for kk, vv in v.items() if kk != "seen"}, age=round(now - v["seen"], 2)) for v in self.items.values()]
        try:
            tmp = self.path + ".tmp"; json.dump({"seq": self.seq, "items": out}, open(tmp, "w")); os.replace(tmp, self.path)
        except OSError:
            pass
