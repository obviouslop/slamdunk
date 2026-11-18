#!/usr/bin/env python3
"""Voxel fusion with free-space carving, shared by tools/pcserver.py (live) and tools/fuse_rec.py (offline).

Per frame:
  * hits:  per-frame mean point per voxel is accumulated (sum xyz/rgb, hits, last seen)
  * misses (free-space carving): every existing voxel is projected into this frame's rectified depth
    map; if the frame measured a surface clearly BEHIND the voxel along that pixel ray (beyond the
    stereo noise margin for that depth), the frame saw through it -> miss.
  * voxels with misses > carve_ratio * hits are removed; unconfirmed voxels (hits < min_hits) that
    were not re-observed for keep_frames frames are pruned.
"""
import numpy as np


class Fuser:
    def __init__(self, voxel=0.015, min_hits=2, carve=True, carve_ratio=1.0, keep_frames=6, sigma_disp=0.35):
        self.v = voxel; self.min_hits = min_hits; self.carve = carve; self.carve_ratio = carve_ratio
        self.keep_frames = keep_frames; self.sigma_disp = sigma_disp
        self.keys = np.zeros(0, np.int64); self.sums = np.zeros((0, 6)); self.hits = np.zeros(0, np.int32)
        self.miss = np.zeros(0, np.int32); self.last = np.zeros(0, np.int32); self.frame = 0
        self.stats = {}

    def key(self, P):
        q = np.floor(P / self.v).astype(np.int64) + (1 << 20)
        return (q[:, 0] & 0x1FFFFF) | ((q[:, 1] & 0x1FFFFF) << 21) | ((q[:, 2] & 0x1FFFFF) << 42)

    def centers(self):
        return self.sums[:, :3] / np.maximum(self.hits, 1)[:, None]

    def _carve(self, ls, disp, world_from_camL):
        """Increment misses of voxels that this frame saw through. ls: LiveStereo (rectification)."""
        if not len(self.keys):
            return 0
        C = self.centers()
        camL_from_world = np.linalg.inv(world_from_camL)
        Pc = C @ camL_from_world[:3, :3].T + camL_from_world[:3, 3]      # camL frame
        Pr = Pc @ ls.R1.T                                                 # rectified frame (rect = R1 camL)
        z = Pr[:, 2]
        h, w = disp.shape
        front = z > ls.min_depth
        u = np.full(len(z), -1, np.int32); v = np.full(len(z), -1, np.int32)
        u[front] = np.round(ls.P1[0, 0] * Pr[front, 0] / z[front] + ls.P1[0, 2]).astype(np.int32)
        v[front] = np.round(ls.P1[1, 1] * Pr[front, 1] / z[front] + ls.P1[1, 2]).astype(np.int32)
        inview = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        idx = np.nonzero(inview)[0]
        d = disp[v[idx], u[idx]]
        valid = d > 0.5
        idx, d = idx[valid], d[valid]
        Zm = ls.f * ls.baseline / d                                       # measured depth along that ray
        sigma = Zm * Zm / (ls.f * ls.baseline) * self.sigma_disp           # stereo depth noise at Zm
        margin = np.maximum(3 * sigma, 2 * self.v)
        seen_through = Zm > z[idx] + margin
        self.miss[idx[seen_through]] += 1
        return int(seen_through.sum())

    def add_frame(self, Pw, Cw, ls=None, disp=None, world_from_camL=None):
        """Pw, Cw: world points/colours of one frame (all points). Optional carving inputs."""
        self.frame += 1
        k = self.key(Pw); uk, inv, cnt = np.unique(k, return_inverse=True, return_counts=True)
        fs = np.zeros((len(uk), 6)); np.add.at(fs, inv, np.c_[Pw, Cw]); fs /= cnt[:, None]
        carved = self._carve(ls, disp, world_from_camL) if (self.carve and ls is not None) else 0
        idx = np.searchsorted(self.keys, uk)
        hit = idx < len(self.keys); hit[hit] = self.keys[idx[hit]] == uk[hit]
        hi = idx[hit]
        self.sums[hi] += fs[hit]; self.hits[hi] += 1; self.last[hi] = self.frame
        new = ~hit
        if new.any():
            pos = idx[new]
            self.keys = np.insert(self.keys, pos, uk[new]); self.sums = np.insert(self.sums, pos, fs[new], axis=0)
            self.hits = np.insert(self.hits, pos, 1); self.miss = np.insert(self.miss, pos, 0)
            self.last = np.insert(self.last, pos, self.frame)
        removed = self._prune()
        self.stats = dict(frame=self.frame, voxels=len(self.keys), carved_hits=carved, removed=removed)
        return self.stats

    def _prune(self):
        keep = ((self.hits >= self.min_hits) | (self.last > self.frame - self.keep_frames))
        if self.carve:
            keep &= self.miss <= self.carve_ratio * self.hits
        n = int((~keep).sum())
        if n:
            self.keys, self.sums, self.hits, self.miss, self.last = (self.keys[keep], self.sums[keep], self.hits[keep],
                                                                     self.miss[keep], self.last[keep])
        return n

    def confirmed(self):
        sel = self.hits >= self.min_hits
        n = self.hits[sel][:, None]
        return self.sums[sel, :3] / n, self.sums[sel, 3:] / n

    def reset(self):
        self.__init__(self.v, self.min_hits, self.carve, self.carve_ratio, self.keep_frames, self.sigma_disp)
