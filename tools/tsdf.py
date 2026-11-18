#!/usr/bin/env python3
"""ctypes wrapper of src/tsdf.cpp (sparse block TSDF, projective integration with free-space carving).

    from tsdf import TSDF
    ts = TSDF(voxel=0.02)
    Z, rgb, _ = ls.depth(rawL, rawR, far=4.0)                 # tools/livestereo.py
    ts.integrate(ls, Z, rgb, ls.world_from_rect(hmd_pose))
    P, C, w = ts.extract(min_frames=3)                         # surface points (world), RGB 0..1, weight
"""
import ctypes, os, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
LIB = os.path.join(ROOT, "build", "libtsdf.so")
SRC = os.path.join(ROOT, "src", "tsdf.cpp")


def _load():
    if not os.path.exists(LIB) or os.path.getmtime(LIB) < os.path.getmtime(SRC):
        os.makedirs(os.path.dirname(LIB), exist_ok=True)
        subprocess.check_call(["g++", "-O3", "-std=c++17", "-fPIC", "-shared", "-fopenmp", "-o", LIB, SRC])
    L = ctypes.CDLL(LIB)
    vp, fp, ip, dp, up = ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p
    L.tsdf_new.restype = vp; L.tsdf_new.argtypes = [fp, fp, fp, fp, ip]
    L.tsdf_free.argtypes = [vp]; L.tsdf_reset.argtypes = [vp]; L.tsdf_nblocks.argtypes = [vp]
    L.tsdf_integrate.restype = ctypes.c_long
    L.tsdf_integrate.argtypes = [vp, up, up, up, ip, ip, fp, fp, fp, dp, fp, fp, fp, fp, ip]
    L.tsdf_extract.argtypes = [vp, ip, fp]
    L.tsdf_get.argtypes = [vp, up, up, up]
    L.tsdf_gc.argtypes = [vp, ip, ip]
    return L


class TSDF:
    def __init__(self, voxel=0.02, trunc_min=None, trunc_max=0.12, wmax=64.0, threads=3, sigma_disp=0.35,
                 max_depth=2.5, carve_depth=4.0, alloc_stride=2, gc_every=20):
        self.L = _load()
        self.voxel = voxel; self.sigma_disp = sigma_disp; self.max_depth = max_depth; self.carve_depth = carve_depth
        self.alloc_stride = alloc_stride
        self.h = self.L.tsdf_new(voxel, trunc_min or 2 * voxel, trunc_max, wmax, threads)
        self.frames = 0; self.gc_every = gc_every

    def __del__(self):
        try: self.L.tsdf_free(self.h)
        except Exception: pass

    def reset(self):
        self.L.tsdf_reset(self.h); self.frames = 0

    @property
    def nblocks(self):
        return self.L.tsdf_nblocks(self.h)

    def integrate(self, ls, Z, rgb, world_from_rect, mask=None):
        """Z: HxW float32 depth (rectified camL view, 0 = invalid), rgb HxWx3 uint8, mask HxW uint8/bool or None."""
        H, W = Z.shape
        Z = np.ascontiguousarray(Z, np.float32); rgb = np.ascontiguousarray(rgb, np.uint8)
        T = np.ascontiguousarray(world_from_rect, np.float64)
        mk = None if mask is None else np.ascontiguousarray(mask, np.uint8)
        n = self.L.tsdf_integrate(self.h, Z.ctypes.data, rgb.ctypes.data, None if mk is None else mk.ctypes.data, W, H,
                                  float(ls.f), float(ls.P1[0, 2]), float(ls.P1[1, 2]),
                                  T.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), float(ls.f * ls.baseline),
                                  self.sigma_disp, self.max_depth, self.carve_depth, self.alloc_stride)
        self.frames += 1
        if self.gc_every and self.frames % self.gc_every == 0:
            self.gc()
        return n

    def gc(self, min_age=30, min_nsurf=2):
        return self.L.tsdf_gc(self.h, min_age, min_nsurf)

    def extract(self, min_frames=3, min_weight=0.0):
        n = self.L.tsdf_extract(self.h, min_frames, min_weight)
        P = np.empty((n, 3), np.float32); C = np.empty((n, 3), np.uint8); w = np.empty(n, np.float32)
        if n:
            self.L.tsdf_get(self.h, P.ctypes.data, C.ctypes.data, w.ctypes.data)
        return P, C.astype(np.float32) / 255, w


def cone_pixels(ls, Z, world_from_rect, aim, half_angle_deg):
    """Mask of pixels whose measured 3-D point lies inside the cone around the controller aim ray (-Z fwd)."""
    A = np.asarray(aim, float).reshape(3, 4); d = -A[:, 2]; o = A[:, 3]
    n = Z.shape[0]
    if getattr(ls, "_uvn", None) is None or ls._uvn[0].shape != Z.shape:
        u, v = np.meshgrid(np.arange(n, dtype=np.float32), np.arange(n, dtype=np.float32))
        ls._uvn = ((u - ls.P1[0, 2]) / ls.f, (v - ls.P1[1, 2]) / ls.f)
    T = np.asarray(world_from_rect, float)
    oc = T[:3, :3].T @ (o - T[:3, 3]); dc = T[:3, :3].T @ d          # aim ray in the rectified frame
    vx = ls._uvn[0] * Z - oc[0]; vy = ls._uvn[1] * Z - oc[1]; vz = Z - oc[2]
    along = vx * dc[0] + vy * dc[1] + vz * dc[2]
    return (Z > 0) & (along > np.cos(np.radians(half_angle_deg)) * np.sqrt(vx * vx + vy * vy + vz * vz))
