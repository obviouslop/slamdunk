#!/usr/bin/env python3
"""Wide-baseline ("motion stereo") depth for far surfaces.

The head-mounted stereo pair has a 6.4 cm baseline: depth noise ~ z^2/(f*B) is 0.3 m per pixel of disparity error at 3 m and
useless beyond ~5 m. Walking / turning gives baselines of 0.3 .. 5 m, so images taken at different moments form much better
stereo pairs of the (static) far scene. This module

  1. KEYFRAMES  picks sharp frames (motion blur estimated from the tracked pose velocities: blur_px = f * t_exp * (omega + v/z),
                no image analysis) that add a new viewpoint, only while a good part of the image has no near-range depth;
                stores the rectified left image as JPEG in an on-disk circular buffer (default 100 MB, oldest deleted first).
  2. PAIRS      for each new keyframe, partners with a baseline 0.3-5 m, a similar view direction and a lateral (not along the
                optical axis) baseline. The relative pose is re-estimated from SIFT matches (essential matrix; its rotation is far
                better than SteamVR's ~1 deg), the baseline LENGTH is taken from the tracked poses; stereoRectify + SGBM.
  3. ACCURACY   per-point depth sigma  sigma_z = z * sqrt((z*sigma_d/(f*B))^2 + (sigma_B/B)^2)  (disparity noise + baseline scale
                error from the tracked positions). Points with a too large relative sigma, no left-right agreement, a flying-pixel
                depth jump or no confirmation by a second, independent pair (moving objects / mismatches) are dropped.
  4. STORE      the far points (STAGE coordinates, with sigma / confirmation count) are merged: a better (smaller sigma)
                observation replaces a worse one nearby.
"""
import collections, glob, json, os, threading, time
import cv2
import numpy as np
from scipy.spatial import cKDTree

LT = 4.36e-6                    # s per sensor line (exp_lines * LT = exposure time)


def _rot_angle(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


class FarStereo:
    def __init__(self, ls, directory, max_bytes=100e6, z_min=2.5, z_max=80.0, rel_sigma=0.12, sigma_d=1.0, sigma_b=0.015,
                 min_base=0.30, max_base=5.0, max_view_deg=35.0, blur_px=1.0, near_frac=0.2, min_gap_s=0.4, period=2.0,
                 max_points=400000, jpeg_q=88, cone_deg=18.0):
        self.ls = ls; self.dir = directory; self.max_bytes = int(max_bytes)
        self.z_min = z_min; self.z_max = z_max; self.rel_sigma = rel_sigma; self.sigma_d = sigma_d; self.sigma_b = sigma_b
        self.min_base = min_base; self.max_base = max_base; self.max_view = max_view_deg; self.blur_px = blur_px
        self.near_frac = near_frac; self.min_gap = min_gap_s; self.period = period; self.max_points = max_points; self.jpeg_q = jpeg_q; self.cone_deg = cone_deg
        self.K = np.array([[ls.f, 0, ls.P1[0, 2]], [0, ls.f, ls.P1[1, 2]], [0, 0, 1.0]])
        self.enabled = True
        self.kfs = collections.OrderedDict()       # seq -> dict(seq, path, size, T, ts)
        self.bytes = 0; self.seq = 0; self.done = set(); self.pairs_done = set(); self.new = []
        self._feat = collections.OrderedDict(); self.lock = threading.Lock()
        self.prev = None; self.last_ts = -1e9; self.blur = 0.0
        # far point store
        self.P = np.zeros((0, 3), np.float32); self.C = np.zeros((0, 3), np.uint8); self.S = np.zeros(0, np.float32)
        self.N = np.zeros(0, np.uint8); self.R = np.zeros(0, np.int32); self.Tm = np.zeros(0, np.float64)
        self.stats = dict(offered=0, kept=0, blurry=0, covered=0, redundant=0, pairs=0, pairs_ok=0, rejected_pair=0)
        os.makedirs(directory, exist_ok=True)
        for f in glob.glob(os.path.join(directory, "kf*")): os.remove(f)     # a new session starts with an empty buffer
        self._sift = None; self._thread = None; self._stop = False

    # ------------------------------------------------------------------ accuracy model
    def sigma_z(self, z, B):
        """expected 1-sigma depth error (m) at range z for a pair of baseline B (m)"""
        z = np.asarray(z, np.float64)
        return z * np.sqrt((z * self.sigma_d / (self.ls.f * B)) ** 2 + (self.sigma_b / B) ** 2)

    # ------------------------------------------------------------------ keyframe selection (pose velocities only)
    def motion_blur(self, T, ts, exp_lines, z_med):
        """blur (px, rectified half-res image) of this frame from the pose change since the previous offered frame"""
        if self.prev is None: self.prev = (T, ts); return 99.0
        Tp, tp = self.prev; dt = ts - tp; self.prev = (T, ts)
        if dt <= 1e-3 or dt > 0.5: return 99.0
        om = np.radians(_rot_angle(Tp[:3, :3].T @ T[:3, :3])) / dt
        v = np.linalg.norm(T[:3, 3] - Tp[:3, 3]) / dt
        return float(self.ls.f * exp_lines * LT * (om + v / max(z_med, 0.5)))

    def offer(self, rgb, Z, T, ts, exp_lines):
        """Called for every fused frame (cheap unless the frame is kept). rgb: rectified left image (RGB uint8), Z its stereo depth,
        T = world_from_rect (4x4). Returns the keyframe seq when stored, else None."""
        zv = Z[::8, ::8]; ok = zv > 0
        z_med = float(np.median(zv[ok])) if ok.any() else 4.0
        blur = self.motion_blur(T, ts, exp_lines, z_med); self.blur = blur
        if not self.enabled: return None
        self.stats["offered"] += 1
        if blur > self.blur_px: self.stats["blurry"] += 1; return None
        if 1.0 - ok.mean() < self.near_frac: self.stats["covered"] += 1; return None     # near stereo already covers the view
        if ts - self.last_ts < self.min_gap: return None
        zax = T[:3, 2]
        for k in self.kfs.values():                                  # novelty: baseline / view angle against what we hold
            ang = np.degrees(np.arccos(np.clip(zax @ k["T"][:3, 2], -1, 1)))
            if np.linalg.norm(T[:3, 3] - k["T"][:3, 3]) / 0.25 + ang / 12.0 < 1.0: self.stats["redundant"] += 1; return None
        return self.store(rgb, T, ts)

    # ------------------------------------------------------------------ on-disk circular buffer
    def store(self, rgb, T, ts):
        ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(rgb[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_q])
        if not ok: return None
        with self.lock:
            self.seq += 1; s = self.seq; path = os.path.join(self.dir, f"kf{s:06d}.jpg")
            with open(path, "wb") as f: f.write(buf.tobytes())
            json.dump(dict(seq=s, ts=ts, T=np.asarray(T).tolist()), open(path[:-4] + ".json", "w"))
            self.kfs[s] = dict(seq=s, path=path, size=len(buf) + 600, T=np.array(T, np.float64), ts=ts); self.bytes += len(buf) + 600
            self.new.append(s); self.last_ts = ts; self.stats["kept"] += 1
            while self.bytes > self.max_bytes and len(self.kfs) > 2:         # circular: drop the oldest keyframe
                _, k = self.kfs.popitem(last=False); self.bytes -= k["size"]; self._feat.pop(k["seq"], None)
                for p in (k["path"], k["path"][:-4] + ".json"):
                    try: os.remove(p)
                    except OSError: pass
        return s

    def _load(self, seq):
        k = self.kfs.get(seq)
        if k is None: return None
        im = cv2.imread(k["path"], cv2.IMREAD_COLOR)
        return None if im is None else im[:, :, ::-1]              # RGB

    def _features(self, seq, rgb):
        if seq in self._feat: self._feat.move_to_end(seq); return self._feat[seq]
        if self._sift is None: self._sift = cv2.SIFT_create(nfeatures=6000)
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        kp, d = self._sift.detectAndCompute(g, None)
        f = (np.array([k.pt for k in kp], np.float32).reshape(-1, 2), d)
        self._feat[seq] = f
        while len(self._feat) > 24: self._feat.popitem(last=False)
        return f

    # ------------------------------------------------------------------ pairs
    def partners(self, k, n=2):
        """best partner keyframes for k: baseline in range, similar view direction, lateral baseline"""
        out = []
        for j in list(self.kfs.values()):
            if j["seq"] == k["seq"] or (min(j["seq"], k["seq"]), max(j["seq"], k["seq"])) in self.pairs_done: continue
            d = j["T"][:3, 3] - k["T"][:3, 3]; B = np.linalg.norm(d)
            if not (self.min_base <= B <= self.max_base): continue
            ang = np.degrees(np.arccos(np.clip(k["T"][:3, 2] @ j["T"][:3, 2], -1, 1)))
            if ang > self.max_view: continue
            out.append((abs(np.log(B / 1.0)) + ang / 30.0, j))        # prefer ~1 m baselines, parallel views
        out.sort(key=lambda x: x[0]); return [j for _, j in out[:n]]

    @staticmethod
    def _rej(why, reason):
        why['why'] = reason; return None

    def _sampson(self, R, t, pa, pb):
        """Sampson epipolar error (px) of matches under camera-B-from-camera-A pose (R, t)"""
        tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]]); Ki = np.linalg.inv(self.K)
        F = Ki.T @ tx @ R @ Ki
        xa = np.c_[pa, np.ones(len(pa))]; xb = np.c_[pb, np.ones(len(pb))]
        Fx = xa @ F.T; Ftx = xb @ F
        num = np.sum(xb * Fx, 1)
        return num / np.sqrt(Fx[:, 0] ** 2 + Fx[:, 1] ** 2 + Ftx[:, 0] ** 2 + Ftx[:, 1] ** 2 + 1e-12)

    def _relative_pose(self, pa, pb, M, why):
        """camera-B-from-camera-A rotation and unit translation from the matches: 5-point RANSAC when it works, else a robust
        refinement of the tracked relative pose (rotation + translation direction) on the Sampson error."""
        from scipy.optimize import least_squares
        from scipy.spatial.transform import Rotation as Rot
        R0 = M[:3, :3]; t0 = M[:3, 3] / np.linalg.norm(M[:3, 3])
        try:
            E, mk = cv2.findEssentialMat(pa, pb, self.K, cv2.RANSAC, 0.999, 1.0)
            if E is not None and E.shape == (3, 3):
                n_in, R, t, _ = cv2.recoverPose(E, pa, pb, self.K, mask=mk.copy())
                if n_in >= 40: why['pose'] = 'E'; return R, t.ravel(), n_in
        except cv2.error:
            pass
        def res(x):
            R = Rot.from_rotvec(x[:3]).as_matrix() @ R0
            th, ph = x[3], x[4]; d = Rot.from_rotvec([th, ph, 0]).as_matrix() @ t0; return self._sampson(R, d, pa, pb)
        sol = least_squares(res, np.zeros(5), loss='soft_l1', f_scale=1.0, max_nfev=60)
        R = Rot.from_rotvec(sol.x[:3]).as_matrix() @ R0; t = Rot.from_rotvec([sol.x[3], sol.x[4], 0]).as_matrix() @ t0
        n_in = int(np.sum(np.abs(self._sampson(R, t, pa, pb)) < 1.5)); why['pose'] = 'refine'
        return R, t, n_in

    def pair_points(self, ka, kb, work=0.5, debug=None):
        """-> dict(P world (N,3), C, sig, z, ref, B) or None."""
        why = debug if debug is not None else {}
        A = self._load(ka["seq"]); Bm = self._load(kb["seq"])
        if A is None or Bm is None: return self._rej(why, 'load')
        pa_all, da = self._features(ka["seq"], A); pb_all, db = self._features(kb["seq"], Bm)
        if da is None or db is None or len(da) < 80 or len(db) < 80: return self._rej(why, 'few features')
        m = cv2.BFMatcher(cv2.NORM_L2).knnMatch(da, db, k=2)
        good = [x[0] for x in m if len(x) == 2 and x[0].distance < 0.8 * x[1].distance]
        why['n_matches'] = len(good)
        if len(good) < 60: return self._rej(why, 'few matches')
        pa = pa_all[[g.queryIdx for g in good]]; pb = pb_all[[g.trainIdx for g in good]]
        M = np.linalg.inv(kb["T"]) @ ka["T"]; Bl = float(np.linalg.norm(M[:3, 3]))
        R, tt, n_in = self._relative_pose(pa, pb, M, why)
        if R is None: return self._rej(why, 'relative pose failed')
        why['n_in'] = int(n_in)
        if n_in < 40 or n_in < 0.3 * len(good): return self._rej(why, 'few E inliers')
        why['rot_diff'] = _rot_angle(R.T @ M[:3, :3])
        if why['rot_diff'] > 6.0: return self._rej(why, 'rotation disagrees with tracker')
        why['dir_diff'] = float(np.degrees(np.arccos(np.clip(tt @ M[:3, 3] / Bl, -1, 1))))
        if why['dir_diff'] > 20.0: return self._rej(why, 'baseline direction disagrees')
        return self._polar_points(ka, kb, A, Bm, R, tt, Bl, pa, pb, work, why)

    # ------------------------------------------------------------------ epipolar (polar) rectification + SGBM
    def _polar_points(self, ka, kb, A, Bm, R, tt, Bl, pa, pb, work, why):
        """Rectify the pair in epipolar-plane coordinates around the baseline axis b (any baseline direction, also forward motion):
        row = angle alpha of the epipolar plane around b, column = angle phi between the ray and b.  Epipolar lines are exact
        rows, a point at phi in the reference (far end of the baseline, camera B) is at phi - delta in the other (A) where
        delta = parallax angle, so SGBM (reference = B, match at x - d) gives  range r_B = B sin(phi_B - delta)/sin(delta).
        Columns start at phi_min (the cone around the epipole, where parallax vanishes, is left out)."""
        X_B = lambda X: X @ R.T                                        # camera A frame -> camera B frame (rotation part)
        b = -R.T @ tt; b /= np.linalg.norm(b)                          # B's centre in A's frame, unit
        zr = np.array([0.0, 0.0, 1.0]) - b[2] * b
        if np.linalg.norm(zr) < 0.3: zr = np.array([1.0, 0.0, 0.0]) - b[0] * b
        zr /= np.linalg.norm(zr); yr = np.cross(zr, b); xr = b         # alpha = 0 -> optical-axis side
        fw = self.K[0, 0] * work; cxw = (self.K[0, 2] + 0.5) * work - 0.5; cyw = (self.K[1, 2] + 0.5) * work - 0.5
        h, w = int(round(A.shape[0] * work)), int(round(A.shape[1] * work))
        dphi = 1.0 / fw
        ph0, ph1 = np.radians(self.cone_deg), np.radians(130.0)
        phis = ph0 + np.arange(int((ph1 - ph0) / dphi)) * dphi
        nal = int(np.ceil(2 * np.pi / dphi))
        als = np.arange(nal) * dphi - np.pi
        cp, sp = np.cos(phis)[None, :, None], np.sin(phis)[None, :, None]
        ca, sa = np.cos(als)[:, None, None], np.sin(als)[:, None, None]
        dA = cp * xr + sp * (ca * zr + sa * yr)                        # (nal, nphi, 3) rays in camera A
        dB = X_B(dA)
        def pix(d):
            z = d[..., 2]; ok = z > 0.05; zz = np.where(ok, z, 1.0)
            u = fw * d[..., 0] / zz + cxw; v = fw * d[..., 1] / zz + cyw
            ok &= (u > 3) & (u < w - 4) & (v > 3) & (v < h - 4)
            return u.astype(np.float32), v.astype(np.float32), ok
        uA, vA, okA = pix(dA); uB, vB, okB = pix(dB)
        ok = okA & okB
        rows = ok.sum(1) >= 24
        if rows.sum() < 40: return self._rej(why, 'no common view (polar)')
        # cut the (cyclic) row range at the longest gap of unusable rows, crop the columns to the usable range
        if not rows.all():
            rr = np.r_[rows, rows]; best, bs, run = 0, 0, 0
            for i, x in enumerate(rr):
                run = 0 if x else run + 1
                if run > best: best, bs = run, i + 1
            start = bs % nal
        else:
            start = 0
        ridx = (start + np.arange(nal)) % nal
        if not rows.all(): ridx = ridx[:nal - int(best)] if best else ridx
        cols = np.nonzero(ok[ridx].sum(0) >= 24)[0]
        c0, c1 = int(cols[0]), int(cols[-1]) + 1
        sub = np.ix_(ridx, np.arange(c0, c1))
        uA, vA, uB, vB, ok = uA[sub], vA[sub], uB[sub], vB[sub], ok[sub]
        al_r = als[ridx]; ph_c = phis[c0:c1]
        why['rect'] = (int(len(ridx)), int(c1 - c0))
        if len(ridx) * (c1 - c0) > 1.2e6: return self._rej(why, 'rectified image too large')
        gA = cv2.cvtColor(cv2.resize(A, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        gB = cv2.cvtColor(cv2.resize(Bm, (w, h), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        rA = cv2.remap(gA, uA, vA, cv2.INTER_LINEAR); rB = cv2.remap(gB, uB, vB, cv2.INTER_LINEAR)
        rA[~ok] = 0; rB[~ok] = 0
        dmax = Bl / max(self.z_min, 1.0) / dphi * 1.05
        nd = int(min(256, np.ceil(dmax / 16 + 1) * 16))
        if c1 - c0 < nd + 16: nd = int((c1 - c0 - 16) // 16 * 16)
        if nd < 16: return self._rej(why, 'rectified image too narrow')
        bs_ = 5
        sg = cv2.StereoSGBM_create(0, nd, bs_, 8 * bs_ * bs_, 32 * bs_ * bs_, 1, 31, 12, 120, 2, cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        d = sg.compute(rB, rA).astype(np.float32) / 16.0               # reference = B; match in A at x - d
        # pair verification with the sparse matches (rows must coincide, SGBM must reproduce their parallax)
        def to_ar(p, Rm):
            dd = cv2.undistortPoints(p.reshape(-1, 1, 2).astype(np.float64), self.K, np.zeros(5)).reshape(-1, 2)
            dd = np.c_[dd, np.ones(len(dd))]; dd /= np.linalg.norm(dd, axis=1, keepdims=True)
            return (np.arctan2(dd @ (Rm @ yr), dd @ (Rm @ zr)), np.arccos(np.clip(dd @ (Rm @ xr), -1, 1)))
        aA, fA = to_ar(pa, np.eye(3)); aB, fB = to_ar(pb, R)
        dy = np.abs(np.angle(np.exp(1j * (aA - aB)))) / dphi; dm = (fB - fA) / dphi
        inl = dy < 3.0; why['dy_med'] = float(np.median(dy[inl])) if inl.any() else 99.0
        ar_pos = np.angle(np.exp(1j * (aB - al_r[0]))) % (2 * np.pi)
        ix = np.round((fB - ph_c[0]) / dphi).astype(int); iy = np.round(ar_pos / dphi).astype(int)
        inb = (ix >= 0) & (ix < d.shape[1]) & (iy >= 0) & (iy < d.shape[0])
        ds = np.full(len(pa), -1.0); ds[inb] = d[iy[inb], ix[inb]]
        sel = inl & inb & (dm > 1.5) & (dm < nd - 6)                      # matches nearer than z_min are outside the search range
        why['sgbm_agree'] = float(np.mean(np.abs(ds[sel] - dm[sel]) < 2.0)) if sel.sum() >= 10 else 0.0
        why['n_check'] = int(sel.sum()); why['dbg'] = (int(inl.sum()), int(inb.sum()), int((dm > 0).sum()), int((ds > 1.5).sum()), float(np.median(dm)) if len(dm) else 0.0)
        if why['dy_med'] > 0.8: return self._rej(why, 'rows misaligned')
        if why['sgbm_agree'] < 0.6: return self._rej(why, 'dense matching disagrees with sparse matches')
        valid = d >= 1.5
        med = cv2.medianBlur(np.where(valid, d, 0).astype(np.float32), 5)
        valid &= np.abs(d - med) < np.maximum(1.0, 0.04 * d)                   # flying pixels / depth jumps
        valid &= cv2.erode(ok.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
        sl = np.zeros_like(valid); sl[::3, ::3] = True; valid &= sl
        v, u = np.nonzero(valid)
        if not len(v): return self._rej(why, 'no disparity')
        delta = d[v, u] * dphi; phB = ph_c[u]; phA = phB - delta
        good = (phA > 0.02) & (delta > 1e-4)
        v, u, delta, phB, phA = v[good], u[good], delta[good], phB[good], phA[good]
        r = Bl * np.sin(phA) / np.sin(delta)
        sig = r * np.sqrt((self.sigma_d * dphi / np.tan(delta)) ** 2 + (self.sigma_b / Bl) ** 2)
        k = (r >= self.z_min) & (r <= self.z_max) & (sig / r <= self.rel_sigma)
        v, u, r, sig = v[k], u[k], r[k], sig[k]
        if not len(r): return self._rej(why, 'all points gated')
        dirs = dB[ridx[v], c0 + u]                                       # rays in camera B (unit)
        Xc = dirs * r[:, None]
        Pw = Xc @ kb["T"][:3, :3].T + kb["T"][:3, 3]
        col = (np.stack([uB[v, u], vB[v, u]], 1))
        col = A_col = Bm[np.clip(np.round((col[:, 1] + 0.5) / work - 0.5).astype(int), 0, Bm.shape[0] - 1),
                          np.clip(np.round((col[:, 0] + 0.5) / work - 0.5).astype(int), 0, Bm.shape[1] - 1)]
        why.update(Br=Bl, nd=nd, phi_min=float(np.degrees(ph_c[0])))
        return dict(P=Pw.astype(np.float32), C=col, sig=sig.astype(np.float32), z=r.astype(np.float32),
                    ref=kb["seq"], other=ka["seq"], B=Bl)

    # ------------------------------------------------------------------ store of far points
    def merge(self, b, ts):
        """merge a pair's points; better observations replace worse ones nearby, a second independent pair confirms."""
        P, C, S = b["P"], b["C"], b["sig"]
        pair_id = (min(b["ref"], b["other"]), max(b["ref"], b["other"]))
        with self.lock:
            if len(self.P):
                tree = cKDTree(self.P); r = np.minimum(np.maximum(2.0 * S, 0.15), 1.5)
                dd, ii = tree.query(P, distance_upper_bound=1.5)
                near = dd <= np.minimum(r + 0.0, 1.5) * 1.0
                # radius test against the larger of the two sigmas (stored sigma used as well)
                near &= np.isfinite(dd)
                if near.any():
                    j = ii[near]; k = np.flatnonzero(near)
                    far_enough = dd[near] <= 2.0 * np.maximum(S[k], self.S[j])
                    k, j = k[far_enough], j[far_enough]
                    diff = self.R[j] != b["ref"]                                  # confirmed by a different reference view
                    self.N[j[diff]] = np.minimum(self.N[j[diff]].astype(int) + 1, 250).astype(np.uint8)
                    better = S[k] < self.S[j]
                    jb, kb = j[better], k[better]
                    self.P[jb] = P[kb]; self.C[jb] = C[kb]; self.S[jb] = S[kb]; self.R[jb] = b["ref"]; self.Tm[jb] = ts
                    used = np.zeros(len(P), bool); used[k] = True
                    P, C, S = P[~used], C[~used], S[~used]
            if len(P):
                self.P = np.vstack([self.P, P]); self.C = np.vstack([self.C, C]); self.S = np.concatenate([self.S, S])
                self.N = np.concatenate([self.N, np.ones(len(P), np.uint8)]); self.R = np.concatenate([self.R, np.full(len(P), b["ref"], np.int32)])
                self.Tm = np.concatenate([self.Tm, np.full(len(P), ts)])
            if len(self.P) > self.max_points:                                   # keep the most accurate
                k = np.argsort(self.S)[:self.max_points]
                self.P, self.C, self.S, self.N, self.R, self.Tm = self.P[k], self.C[k], self.S[k], self.N[k], self.R[k], self.Tm[k]

    def points(self, min_confirm=2, sure_rel=0.02):
        """-> P (N,3), C uint8, range (m, from the keyframe), sigma, t. Confirmed by >= 2 views, or very accurate on their own."""
        with self.lock:
            if not self.enabled or not len(self.P): return (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8), np.zeros(0, np.float32),
                                                             np.zeros(0, np.float32), np.zeros(0))
            rng = np.zeros(len(self.P), np.float32)
            # range of each point from its reference keyframe
            for s in np.unique(self.R):
                k = self.kfs.get(int(s)); m = self.R == s
                rng[m] = np.linalg.norm(self.P[m] - (k["T"][:3, 3] if k else 0), axis=1)
            keep = (self.N >= min_confirm) | (self.S / np.maximum(rng, 1e-3) <= sure_rel)
            return self.P[keep].copy(), self.C[keep].copy(), rng[keep], self.S[keep].copy(), self.Tm[keep].copy()

    def clear(self):
        with self.lock:
            self.P = self.P[:0]; self.C = self.C[:0]; self.S = self.S[:0]; self.N = self.N[:0]; self.R = self.R[:0]; self.Tm = self.Tm[:0]
            self.pairs_done.clear(); self.new.clear()

    # ------------------------------------------------------------------ periodic extraction
    def step(self, budget=8.0):
        """process the keyframes added since the last call (call from a worker thread or synchronously)"""
        t0 = time.monotonic(); n = 0
        while self.new and time.monotonic() - t0 < budget:
            with self.lock: s = self.new.pop(0); k = self.kfs.get(s)
            if k is None: continue
            for j in self.partners(k):
                pid = (min(j["seq"], k["seq"]), max(j["seq"], k["seq"])); self.pairs_done.add(pid)
                self.stats["pairs"] += 1
                try: b = self.pair_points(k, j)
                except cv2.error as e: print("far pair:", e, flush=True); b = None
                if b is None: self.stats["rejected_pair"] += 1; continue
                self.stats["pairs_ok"] += 1; self.merge(b, k["ts"]); n += 1
        return n

    def start(self):
        def run():
            while not self._stop:
                time.sleep(self.period)
                try:
                    if self.enabled: self.step()
                except Exception as e:
                    print("far stereo:", e, flush=True)
        self._thread = threading.Thread(target=run, daemon=True); self._thread.start()

    def hud(self):
        if not self.enabled: return "far: off"
        P = self.points()[3]
        s = f"far: {len(self.kfs)} kf {self.bytes / 1e6:.0f} MB, pairs {self.stats['pairs_ok']}/{self.stats['pairs']}, {len(P)} pts"
        return s + (f", sigma {np.median(P):.2f} m" if len(P) else "")
