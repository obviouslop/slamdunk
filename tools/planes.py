#!/usr/bin/env python3
"""Live coarse plane fitting on the fused cloud -> /dev/shm/pcplanes.json (viewer/planes.gd draws them 10 % translucent).

PlaneTracker.update(P) takes the fused STAGE-frame points (N,3, y up) and
  1. voxel-downsamples (5 cm) so a plane's weight is its area, not how often it was scanned,
  2. iterative RANSAC (<= 8 planes, tol 5 cm, min_area m^2), each plane refined by SVD on its inliers,
  3. snaps near-horizontal / near-vertical planes (within 10 deg) to exactly horizontal / vertical (rooms are rectilinear),
  4. labels them: floor (lowest big up-facing), ceiling (highest big, > 1.8 m above the floor), table (other horizontal),
     wall (vertical), plane (other),
  5. tracks them over updates (match by normal + offset, EMA smoothing, ids stay stable; new planes must be seen twice,
     vanished ones are kept for a few updates) so nothing flickers.
Plane: n . x + d = 0 with unit n (y >= 0 for horizontal ones), in-plane basis u, v with u x v = n, a centre in the plane and
robust half extents (1-99 percentile of the inliers' in-plane coordinates) -> the viewer draws one quad per plane.
"""
import json, os
import numpy as np
import cv2

SNAP_DEG = 10.0


def _fit(P):
    c = P.mean(0); u, s, vt = np.linalg.svd(P - c, full_matrices=False)
    n = vt[2]; return n, -float(n @ c)


def _basis(n):
    """in-plane axes u, v with u x v = n: horizontal -> u = x, v = z-ish; vertical -> u horizontal, v = up"""
    if abs(n[1]) > 0.7:
        u = np.cross([0.0, 0.0, 1.0], n); v = None            # x-ish
    else:
        u = np.cross([0.0, 1.0, 0.0], n)
    u /= np.linalg.norm(u); v = np.cross(n, u)
    return u, v


class PlaneTracker:
    def __init__(self, path="/dev/shm/pcplanes.json", max_planes=8, voxel=0.05, tol=0.05, min_area=0.8, iters=500, hold=4, seed=0, min_fill=0.12):
        self.path = path; self.max_planes = max_planes; self.voxel = voxel; self.tol = tol
        self.min_pts = int(min_area / voxel ** 2); self.iters = iters; self.hold = hold; self.min_fill = min_fill
        self.rng = np.random.default_rng(seed); self.tracks = {}; self.next_id = 1; self.seq = 0; self.items = []

    # ---- detection -------------------------------------------------------------------------------------------------
    def detect(self, P):
        if len(P) < self.min_pts: return []
        if len(P) > 300000: P = P[self.rng.choice(len(P), 300000, replace=False)]
        g = np.floor(P / self.voxel).astype(np.int64) + 50000
        key = (g[:, 0] << 40) | (g[:, 1] << 20) | g[:, 2]
        Q = P[np.unique(key, return_index=True)[1]]
        if len(Q) > 60000: Q = Q[self.rng.choice(len(Q), 60000, replace=False)]
        out = []; rest = Q; fails = 0
        for _ in range(self.max_planes * 3):
            if len(rest) < self.min_pts or len(out) >= self.max_planes or fails >= 3: break
            best = None
            # local sampling: the 2nd and 3rd point come from the 0.5 m cell of the 1st, so small planes (tables) are hit often
            cell = np.floor(rest / 0.5).astype(np.int64); _, inv = np.unique(cell, axis=0, return_inverse=True); inv = inv.ravel()
            order = np.argsort(inv, kind="stable"); cnt_c = np.bincount(inv); start = np.cumsum(cnt_c) - cnt_c
            i0 = self.rng.integers(0, len(rest), self.iters); c0 = inv[i0]
            i1 = order[start[c0] + (self.rng.random(self.iters) * cnt_c[c0]).astype(int)]
            i2 = order[start[c0] + (self.rng.random(self.iters) * cnt_c[c0]).astype(int)]
            A, B, C = rest[i0], rest[i1], rest[i2]
            N = np.cross(B - A, C - A); nn = np.linalg.norm(N, axis=1); ok = nn > 1e-6
            N = N[ok] / nn[ok, None]; A = A[ok]
            if not len(N): break
            sub = rest[self.rng.choice(len(rest), min(len(rest), 6000), replace=False)]       # score hypotheses on a subsample
            cnt = (np.abs(sub @ N.T - (A * N).sum(1)) < self.tol).sum(0)
            n = N[int(np.argmax(cnt))]; d = -float(n @ A[int(np.argmax(cnt))])
            for _r in range(2):                                      # refine on the full inlier set
                inl = np.abs(rest @ n + d) < self.tol
                if inl.sum() < 3: break
                n, d = _fit(rest[inl])
            inl = np.abs(rest @ n + d) < self.tol
            if inl.sum() < self.min_pts: fails += 1; continue                # unlucky / spurious hypothesis: draw again
            n, d = self._snap(n, rest[inl])
            inl = np.abs(rest @ n + d) < self.tol
            dup = any(abs(float(n @ o["n"])) > 0.98 and abs(d - o["d"] * (1 if n @ o["n"] > 0 else -1)) < 0.12 for o in out)
            if inl.sum() >= self.min_pts and not dup:
                pl = self._describe(n, d, rest[inl])
                fill = pl["npts"] * self.voxel ** 2 / max(4 * pl["hu"] * pl["hv"], 1e-6)      # scattered noise slabs are mostly empty
                if pl["npts"] >= self.min_pts and fill >= self.min_fill: out.append(pl)
            rest = rest[np.abs(rest @ n + d) >= 2 * self.tol]               # thick walls: also drop the near-by remnants
        return self._label(out)

    @staticmethod
    def _snap(n, X):
        n = n / np.linalg.norm(n)
        if abs(n[1]) > np.cos(np.radians(SNAP_DEG)): n = np.array([0.0, np.sign(n[1]), 0.0])
        elif abs(n[1]) < np.sin(np.radians(SNAP_DEG)):
            n = np.array([n[0], 0.0, n[2]]); n /= np.linalg.norm(n)
        if n[1] < 0 or (abs(n[1]) < 1e-6 and (n[0] < 0 or (n[0] == 0 and n[2] < 0))): n = -n     # canonical sign
        return n, -float(np.mean(X @ n))

    @staticmethod
    def _describe(n, d, X, cell=0.10):
        u, v = _basis(n)
        a, b = X @ u, X @ v
        ia = np.floor((a - a.min()) / cell).astype(int) + 1; ib = np.floor((b - b.min()) / cell).astype(int) + 1
        if ia.max() < 4000 and ib.max() < 4000:                    # keep the biggest connected patch: coplanar clutter elsewhere must not stretch the quad
            im = np.zeros((ib.max() + 2, ia.max() + 2), np.uint8); im[ib, ia] = 1
            nl, lab = cv2.connectedComponents(cv2.dilate(im, np.ones((3, 3), np.uint8)))
            if nl > 2:
                cnt = np.bincount(lab[ib, ia], minlength=nl); cnt[0] = 0
                k = lab[ib, ia] == int(np.argmax(cnt)); X, a, b = X[k], a[k], b[k]
        lo_a, hi_a = np.percentile(a, [1, 99]); lo_b, hi_b = np.percentile(b, [1, 99])
        ca, cb = 0.5 * (lo_a + hi_a), 0.5 * (lo_b + hi_b)
        c = u * ca + v * cb - n * d                                   # point of the plane (n.c + d = 0)
        return dict(n=n, d=d, u=u, v=v, c=c, hu=0.5 * (hi_a - lo_a), hv=0.5 * (hi_b - lo_b), npts=int(len(X)))

    def _label(self, pl):
        hor = [p for p in pl if abs(p["n"][1]) > 0.95]
        floor = None
        if hor:
            ys = [-p["d"] for p in hor]                                   # plane height (n = +y => y = -d)
            fl = hor[int(np.argmin(ys))]
            if fl["hu"] * fl["hv"] * 4 > 1.5: floor = fl
        for p in pl:
            if abs(p["n"][1]) > 0.95:
                y = -p["d"]
                if p is floor: p["label"] = "floor"
                elif floor is not None and y - (-floor["d"]) > 1.8 and p["hu"] * p["hv"] * 4 > 1.5: p["label"] = "ceiling"
                else: p["label"] = "table"
            elif abs(p["n"][1]) < 0.17: p["label"] = "wall"
            else: p["label"] = "plane"
        return pl

    # ---- tracking ------------------------------------------------------------------------------------------------------
    def update(self, P):
        det = self.detect(np.asarray(P, np.float32))
        used = set()
        for p in det:
            best, bs = None, 1e9
            for tid, t in self.tracks.items():
                if tid in used: continue
                dot = float(p["n"] @ t["n"]); dd = p["d"] - t["d"] if dot > 0 else p["d"] + t["d"]
                if abs(dot) > np.cos(np.radians(12)) and abs(dd) < 0.10 and abs(dd) < bs: best, bs = tid, abs(dd)
            if best is None:
                self.tracks[self.next_id] = dict(p, seen=1, miss=0, id=self.next_id); used.add(self.next_id); self.next_id += 1
            else:
                t = self.tracks[best]; used.add(best); s = 1 if float(p["n"] @ t["n"]) > 0 else -1
                a = 0.4
                n = t["n"] * (1 - a) + p["n"] * s * a; n /= np.linalg.norm(n)
                d = t["d"] * (1 - a) + p["d"] * s * a
                u, v = _basis(n)
                c = t["c"] * (1 - a) + p["c"] * a; c = c - n * float(n @ c + d)            # keep it in the plane
                keep = dict(label=p["label"], hu=t["hu"] * (1 - a) + p["hu"] * a, hv=t["hv"] * (1 - a) + p["hv"] * a, npts=p["npts"])
                # extents/centre are in the old basis if the basis rotated: recompute from p when the normal changed noticeably
                if p["n"] @ t["n"] * s < 0.9995: keep["hu"], keep["hv"] = p["hu"], p["hv"]
                t.update(n=n, d=d, u=u, v=v, c=c, seen=t["seen"] + 1, miss=0, **keep)
        for tid in list(self.tracks):
            if tid not in used:
                self.tracks[tid]["miss"] += 1
                if self.tracks[tid]["miss"] > self.hold or self.tracks[tid]["seen"] < 2: del self.tracks[tid]
        self.items = [t for t in self.tracks.values() if t["seen"] >= 2]
        self._write()
        return self.items

    def _write(self):
        self.seq += 1
        items = [dict(id=t["id"], label=t["label"], n=t["n"].round(5).tolist(), d=round(float(t["d"]), 4), centre=t["c"].round(4).tolist(),
                      u=t["u"].round(5).tolist(), v=t["v"].round(5).tolist(), hu=round(float(t["hu"]), 3), hv=round(float(t["hv"]), 3),
                      npts=t["npts"], stale=t["miss"] > 0) for t in self.items]
        try:
            tmp = self.path + ".tmp"; json.dump(dict(seq=self.seq, items=items), open(tmp, "w")); os.replace(tmp, self.path)
        except OSError:
            pass

    def clear(self):
        self.tracks = {}; self.items = []; self._write()
