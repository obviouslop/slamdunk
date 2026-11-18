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
LIB = os.path.join(ROOT, "build", "libpc.so")
SRCS = [os.path.join(ROOT, "src", f) for f in ("tsdf.cpp", "raw10.cpp")]


def _load():
    if not os.path.exists(LIB) or os.path.getmtime(LIB) < max(os.path.getmtime(s) for s in SRCS):
        os.makedirs(os.path.dirname(LIB), exist_ok=True)
        subprocess.check_call(["g++", "-O3", "-std=c++17", "-fPIC", "-shared", "-fopenmp", "-o", LIB] + SRCS)
    L = ctypes.CDLL(LIB)
    vp, fp, ip, dp, up = ctypes.c_void_p, ctypes.c_float, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_void_p
    L.tsdf_new.restype = vp; L.tsdf_new.argtypes = [fp, fp, fp, fp, ip]
    L.tsdf_set.argtypes = [vp, ip, fp]
    L.tsdf_align.argtypes = [vp, up, ip, ip, fp, fp, fp, dp, fp, fp, fp, ip, ip, ip, fp, fp, dp, fp, fp, up]
    L.tsdf_free.argtypes = [vp]; L.tsdf_reset.argtypes = [vp]; L.tsdf_nblocks.argtypes = [vp]
    L.tsdf_integrate.restype = ctypes.c_long
    L.tsdf_integrate.argtypes = [vp, up, up, up, ip, ip, fp, fp, fp, dp, fp, fp, fp, fp, ip]
    L.tsdf_extract.argtypes = [vp, ip, fp]
    L.tsdf_get.argtypes = [vp, up, up, up]
    L.tsdf_gc.argtypes = [vp, ip, ip]
    L.tsdf_raycast.argtypes = [vp, up, up, fp, fp, ip, up]
    L.pc_bin_raw10.argtypes = [up, ip, ip, ip, ip, fp, up, ip]
    return L


_L = None


def lib():
    global _L
    if _L is None:
        _L = _load()
    return _L


def bin_raw10(buf, W=2464, H=2464, stride=3088, skip=0, black=64.0, threads=2):
    """Packed RAW10 BGGR bytes -> (H/2, W/2, 3) float32 BGR, black removed (== LiveStereo._rgb(unpack(buf)))."""
    a = np.frombuffer(buf, np.uint8)
    assert a.size >= (H + skip) * stride
    out = np.empty((H // 2, W // 2, 3), np.float32)
    lib().pc_bin_raw10(a.ctypes.data, W, H, stride, skip, float(black), out.ctypes.data, threads)
    return out


class TSDF:
    def __init__(self, voxel=0.02, trunc_min=None, trunc_max=0.12, wmax=64.0, threads=3, sigma_disp=0.35,
                 max_depth=2.5, carve_depth=4.0, alloc_stride=2, gc_every=20,
                 back_k=0.0, back_w=0.5, carve_w=1.0):
        self.L = lib()
        self.voxel = voxel; self.sigma_disp = sigma_disp; self.max_depth = max_depth; self.carve_depth = carve_depth
        self.alloc_stride = alloc_stride
        self.h = self.L.tsdf_new(voxel, trunc_min or 2 * voxel, trunc_max, wmax, threads)
        self.frames = 0; self.gc_every = gc_every
        for k, v in enumerate((back_k, back_w, carve_w)):
            self.L.tsdf_set(self.h, k, float(v))

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

    def align(self, ls, Z, world_from_rect, stride=4, iters=6, min_nsurf=3, prior_rot=0.005, prior_trans=0.02,
              tracking=None, abs_rot=0.03, abs_trans=0.15):
        """Frame-to-model registration: refine world_from_rect (4x4) so the depth frame Z fits the map (4 DoF: yaw +
        translation; prior_* = 1-sigma per-frame step, abs_* = 1-sigma total correction w.r.t. `tracking` (raw pose)).
        -> (T_refined 4x4, info dict) or (None, info) if there was not enough overlap."""
        H, W = Z.shape
        Z = np.ascontiguousarray(Z, np.float32)
        T = np.ascontiguousarray(np.asarray(world_from_rect, np.float64)[:3, :4].copy())
        out = np.zeros(4, np.float32)
        Tp = None if tracking is None else np.ascontiguousarray(np.asarray(tracking, np.float64)[:3, :4].copy())
        ok = self.L.tsdf_align(self.h, Z.ctypes.data, W, H, float(ls.f), float(ls.P1[0, 2]), float(ls.P1[1, 2]),
                               T.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), float(ls.f * ls.baseline), self.sigma_disp,
                               self.max_depth, stride, iters, min_nsurf, prior_rot, prior_trans,
                               None if Tp is None else Tp.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), abs_rot, abs_trans,
                               out.ctypes.data)
        info = dict(n=int(out[0]), rms=float(out[1]), rot=float(out[2]), trans=float(out[3]))
        if not ok:
            return None, info
        R = np.eye(4); R[:3, :4] = T
        return R, info

    def gc(self, min_age=30, min_nsurf=2):
        return self.L.tsdf_gc(self.h, min_age, min_nsurf)

    def raycast(self, origin, direction, tmin=0.1, tmax=6.0, min_frames=3):
        """-> (point, outward normal, distance) of the first surface along the ray, or None"""
        o = np.ascontiguousarray(origin, np.float32); d = np.asarray(direction, np.float32)
        d = np.ascontiguousarray(d / np.linalg.norm(d)); out = np.zeros(7, np.float32)
        if self.L.tsdf_raycast(self.h, o.ctypes.data, d.ctypes.data, tmin, tmax, min_frames, out.ctypes.data):
            return out[:3].copy(), out[3:6].copy(), float(out[6])
        return None

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
