#!/usr/bin/env python3
"""Geodesy helpers: WGS84 <-> ECEF <-> local ENU, and the STAGE <-> ENU georeference.

STAGE = OpenXR STAGE == OpenVR Standing (x right, y up, z back; metres).
Georeference (tools/gnssd.py):  ENU = Rz(yaw) @ M @ stage + t,   M: (x, y, z) -> (x, -z, y)
  i.e. stage x -> East and stage -z (forward) -> North when yaw = 0; yaw rotates counter-clockwise
  seen from above. georef dict: {lat0, lon0, h0, yaw, t:[E,N,U], rms, n, ok}
"""
import numpy as np

A = 6378137.0
F = 1 / 298.257223563
E2 = F * (2 - F)


def lla_to_ecef(lat, lon, h):
    lat, lon = np.radians(lat), np.radians(lon)
    n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    return np.stack([(n + h) * np.cos(lat) * np.cos(lon), (n + h) * np.cos(lat) * np.sin(lon), (n * (1 - E2) + h) * np.sin(lat)], -1)


def ecef_to_lla(x):
    x = np.asarray(x, float)
    X, Y, Z = x[..., 0], x[..., 1], x[..., 2]
    lon = np.arctan2(Y, X); p = np.hypot(X, Y)
    lat = np.arctan2(Z, p * (1 - E2))
    for _ in range(6):
        n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(Z, p * (1 - E2 * n / (n + h)))
    n = A / np.sqrt(1 - E2 * np.sin(lat) ** 2)
    h = p / np.cos(lat) - n
    return np.degrees(lat), np.degrees(lon), h


def _R_enu(lat0, lon0):
    la, lo = np.radians(lat0), np.radians(lon0)
    return np.array([[-np.sin(lo), np.cos(lo), 0],
                     [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                     [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])


def lla_to_enu(lat, lon, h, lat0, lon0, h0):
    d = lla_to_ecef(lat, lon, h) - lla_to_ecef(lat0, lon0, h0)
    return d @ _R_enu(lat0, lon0).T


def enu_to_lla(enu, lat0, lon0, h0):
    x = lla_to_ecef(lat0, lon0, h0) + np.asarray(enu, float) @ _R_enu(lat0, lon0)
    return ecef_to_lla(x)


M_STAGE_TO_ENU0 = np.array([[1., 0, 0], [0, 0, -1], [0, 1, 0]])


def Rz(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])


def stage_to_enu(P, g):
    """P (...,3) STAGE metres -> local ENU metres around (lat0, lon0, h0)."""
    R = Rz(g["yaw"]) @ M_STAGE_TO_ENU0
    return np.asarray(P, float) @ R.T + np.asarray(g["t"], float)


def stage_rot_to_enu(g):
    return Rz(g["yaw"]) @ M_STAGE_TO_ENU0


def stage_to_lla(P, g):
    return enu_to_lla(stage_to_enu(P, g), g["lat0"], g["lon0"], g["h0"])


def fit_yaw_translation(stage_pts, enu_pts, w=None):
    """Weighted least squares ENU = Rz(yaw) M stage + t (4 DoF: gravity is known on both sides).
    Returns yaw, t, residuals (m)."""
    S = np.asarray(stage_pts, float) @ M_STAGE_TO_ENU0.T; E = np.asarray(enu_pts, float)
    w = np.ones(len(S)) if w is None else np.asarray(w, float)
    ws = w / w.sum()
    cs, ce = ws @ S, ws @ E
    a, b = S[:, :2] - cs[:2], E[:, :2] - ce[:2]
    num = np.sum(ws * (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0])); den = np.sum(ws * (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]))
    yaw = float(np.arctan2(num, den))
    t = ce - Rz(yaw) @ cs
    res = np.linalg.norm(S @ Rz(yaw).T + t - E, axis=1)
    return yaw, t, res
