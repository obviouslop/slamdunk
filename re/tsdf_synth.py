#!/usr/bin/env python3
"""Synthetic TSDF bias test: floor plane y=0 seen from cameras at 1.6 m looking down at various angles,
exact or noisy depth (stereo-like sigma ~ Z^2). Prints the median height of extracted points."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
from tsdf import TSDF
class LS: pass
ls = LS(); n = 400; ls.f = 313.6; ls.P1 = np.array([[ls.f, 0, n / 2], [0, ls.f, n / 2], [0, 0, 1.]]); ls.baseline = 0.064
def run(noise, label, y0=0.0, jitter=0.0, **kw):
    rng = np.random.default_rng(1); ts = TSDF(voxel=0.02, max_depth=2.5, **kw)
    u, v = np.meshgrid(np.arange(n), np.arange(n))
    for k in range(60):
        pitch = np.radians(rng.uniform(25, 70)); yaw = rng.uniform(0, 2 * np.pi)
        pos = np.array([rng.uniform(-1, 1), 1.6, rng.uniform(-1, 1)])
        fwd = np.array([np.cos(pitch) * np.sin(yaw), -np.sin(pitch), np.cos(pitch) * np.cos(yaw)])
        right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right); down = np.cross(fwd, right)
        R = np.stack([right, down, fwd], 1)                # world_from_cam (x right, y down, z fwd)
        d = np.stack([(u - n / 2) / ls.f, (v - n / 2) / ls.f, np.ones_like(u, float)], -1) @ R.T
        t = (y0 - pos[1]) / np.minimum(d[..., 1], -1e-6)    # ray hits y=y0
        Z = t.astype(np.float32)                              # depth along optical axis (d has z=1 in cam)
        Z[(d[..., 1] >= -1e-3) | (Z > 4)] = 0
        if noise:
            sig = Z ** 2 / (ls.f * ls.baseline) * 0.35; Z = np.where(Z > 0, Z + rng.normal(0, 1, Z.shape) * sig, 0).astype(np.float32)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = pos
        if jitter:
            from scipy.spatial.transform import Rotation as Rot
            T[:3, :3] = Rot.from_rotvec(rng.normal(0, np.radians(jitter), 3)).as_matrix() @ R; T[:3, 3] = pos + rng.normal(0, jitter / 100, 3)
        ts.integrate(ls, Z, np.full((n, n, 3), 128, np.uint8), T)
    P, C, w = ts.extract(min_frames=4)
    P = P[(np.abs(P[:, 0]) < 1.5) & (np.abs(P[:, 2]) < 1.5)]
    print(f"{label:34s} n={len(P):6d}  floor y-y0 median {(np.median(P[:,1])-y0)*100:+.2f} cm  p10 {np.percentile(P[:,1],10)*100:+.2f}  p90 {np.percentile(P[:,1],90)*100:+.2f}")
for y0 in (0.0, 0.005, 0.01, 0.015):
    run(True, f"noisy y0={y0}", y0=y0)
run(True, "noisy, pose jitter 1 deg/1 cm", jitter=1.0)

# --- aligner bias on the synthetic floor (true offset 0)
def frames(noise, k=60, seed=1):
    rng = np.random.default_rng(seed); u, v = np.meshgrid(np.arange(n), np.arange(n)); out = []
    for _ in range(k):
        pitch = np.radians(rng.uniform(25, 70)); yaw = rng.uniform(0, 2 * np.pi)
        pos = np.array([rng.uniform(-1, 1), 1.6, rng.uniform(-1, 1)])
        fwd = np.array([np.cos(pitch) * np.sin(yaw), -np.sin(pitch), np.cos(pitch) * np.cos(yaw)])
        right = np.cross(fwd, [0, 1, 0]); right /= np.linalg.norm(right); down = np.cross(fwd, right)
        R = np.stack([right, down, fwd], 1)
        d = np.stack([(u - n / 2) / ls.f, (v - n / 2) / ls.f, np.ones_like(u, float)], -1) @ R.T
        Z = (-pos[1] / np.minimum(d[..., 1], -1e-6)).astype(np.float32); Z[(d[..., 1] >= -1e-3) | (Z > 4)] = 0
        if noise:
            sig = Z ** 2 / (ls.f * ls.baseline) * 0.35; Z = np.where(Z > 0, Z + rng.normal(0, 1, Z.shape) * sig, 0).astype(np.float32)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = pos; out.append((Z, T))
    return out
for noise in (False, True):
    ts = TSDF(voxel=0.02, max_depth=2.5); dy = []
    for j, (Z, T) in enumerate(frames(noise)):
        if j >= 5:
            Ta, info = ts.align(ls, Z, T, prior_rot=0.005, prior_trans=0.05)
            if Ta is not None: dy.append(Ta[1, 3] - T[1, 3])
        ts.integrate(ls, Z, np.full((n, n, 3), 128, np.uint8), T)
    print(f"aligner on synthetic floor, noise={noise}: dy median {np.median(dy)*100:+.2f} cm (truth 0)")
