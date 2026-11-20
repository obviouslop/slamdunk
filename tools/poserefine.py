#!/usr/bin/env python3
"""Pose refinement of the tracked (SteamVR) camera pose against the TSDF map (LSD-SLAM style direct tracking, depth term).

Per frame: take ~N valid stereo points of the new frame, transform them with the tracked pose, sample the map TSDF
(tsdf.sample: value + gradient -> distance to the zero crossing d = tsdf/|g|, normal n = g/|g|) and solve a robust
point-to-surface Gauss-Newton for a small 6-DoF correction D (world frame): T' = D @ T.
  * Huber weights, |d| gate, points with unobserved neighbourhoods are skipped;
  * Tikhonov prior toward the previous correction (SteamVR error is slowly varying) + hard cap on |D| (cm / deg);
  * skipped when the map is too small or too few points match, so it never runs away from the tracker.
"""
import numpy as np


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def _exp(w, t):
    th = np.linalg.norm(w); K = _skew(w)
    R = np.eye(3) + K + 0.5 * K @ K if th < 1e-6 else np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th ** 2 * K @ K
    D = np.eye(4); D[:3, :3] = R; D[:3, 3] = t; return D


class PoseRefiner:
    def __init__(self, n_points=3000, iters=6, huber=0.02, gate=0.06, sigma_t=0.02, sigma_r=np.radians(1.0),
                 cap_t=0.06, cap_r=np.radians(3.0), min_pts=400, min_frames=8, smooth=0.5):
        self.n = n_points; self.iters = iters; self.huber = huber; self.gate = gate
        self.sig_t = sigma_t; self.sig_r = sigma_r; self.cap_t = cap_t; self.cap_r = cap_r
        self.min_pts = min_pts; self.min_frames = min_frames; self.smooth = smooth
        self.D = np.eye(4); self.stats = dict(used=0, skipped=0, last=None); self.rng = np.random.default_rng(1)

    def points(self, ls, Z):
        """camera-frame points (rectified camL) of a random subset of valid pixels"""
        v, u = np.nonzero(Z > 0)
        if len(v) > self.n: k = self.rng.choice(len(v), self.n, replace=False); v, u = v[k], u[k]
        z = Z[v, u]
        return np.stack([(u - ls.P1[0, 2]) * z / ls.f, (v - ls.P1[1, 2]) * z / ls.f, z], 1).astype(np.float64)

    def refine(self, ts, ls, Z, T):
        """T: tracked world_from_rect (4x4) -> refined 4x4 (D @ T). ts.frames gates the start."""
        if ts.frames < self.min_frames:
            return T
        Pc = self.points(ls, Z)
        if len(Pc) < self.min_pts: return T
        D = self.D.copy(); info = None
        for it in range(self.iters):
            Pw = (Pc @ T[:3, :3].T + T[:3, 3]) @ D[:3, :3].T + D[:3, 3]
            s = ts.sample(Pw); d = s[:, 0]; g = s[:, 1:]; gn = np.linalg.norm(g, axis=1)
            ok = np.isfinite(d) & (gn > 1e-3)
            if ok.sum() < self.min_pts: self.stats["skipped"] += 1; return T
            n = g[ok] / gn[ok, None]; r = d[ok] / gn[ok]; P = Pw[ok]
            keep = np.abs(r) < self.gate; n, r, P = n[keep], r[keep], P[keep]
            if len(r) < self.min_pts: self.stats["skipped"] += 1; return T
            w = np.where(np.abs(r) < self.huber, 1.0, self.huber / np.abs(r))
            J = np.hstack([np.cross(P, n), n])                         # [omega, tau]
            A = (J * w[:, None]).T @ J; b = -(J * w[:, None]).T @ r
            sc = len(r) / (self.sig_t ** 2) * 1e-4                      # prior strength relative to the data (weak, grows with N)
            Lam = np.diag([1 / self.sig_r ** 2] * 3 + [1 / self.sig_t ** 2] * 3) * sc * 0.01
            x = np.linalg.solve(A + Lam, b)
            D = _exp(x[:3], x[3:]) @ D
            info = (len(r), float(np.sqrt(np.mean(r ** 2))))
            if np.linalg.norm(x[3:]) < 5e-5 and np.linalg.norm(x[:3]) < 2e-4: break
        # cap and smooth
        dt = D[:3, 3]; ang = np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1))
        if np.linalg.norm(dt) > self.cap_t or ang > self.cap_r: self.stats["skipped"] += 1; return T
        self.D = D if self.smooth <= 0 else self._blend(self.D, D)
        self.stats["used"] += 1; self.stats["last"] = (info, float(np.linalg.norm(dt) * 1000), float(np.degrees(ang)))
        return self.D @ T

    @staticmethod
    def _blend(A, B, a=0.5):
        from scipy.spatial.transform import Rotation as Rot
        r = Rot.from_matrix(np.stack([A[:3, :3], B[:3, :3]]))
        m = (Rot.from_rotvec(a * Rot.from_matrix(A[:3, :3].T @ B[:3, :3]).as_rotvec()) ).as_matrix()
        C = np.eye(4); C[:3, :3] = A[:3, :3] @ m; C[:3, 3] = (1 - a) * A[:3, 3] + a * B[:3, 3]; return C
