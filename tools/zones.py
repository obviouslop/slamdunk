#!/usr/bin/env python3
"""Inclusion / exclusion / label zones for the fused cloud (MAP frame), defined in the viewer (SELECT / ZONE tools).

The viewer writes /dev/shm/pczones.json = {"seq": n, "zones": [zone, ...]} whenever the scene changes; the zones are
also stored in the scene JSON (type "zone"). The TSDF itself is never modified: zones are applied when the cloud is
extracted (publish, save) and to the pick ray, so deleting is permanent across rescans and undo is free.

zone = {"mode": "include" | "exclude" | "label", "label": name (mode label), "shape": ...}
  shape "box":     center [3], axes [[3] x3] (unit columns x, y, z), size [3]
  shape "prism":   points [[3], ...] polygon traced on the floor (x/z used), ymin, ymax (vertical column)
  shape "spheres": centers [[3], ...], radii [r, ...]  (SELECT brush strokes)
  shape "cut":     y, side "below" | "above"  (horizontal half space; mode is always exclude)

A point is kept iff (there is no include zone or it is inside at least one) and it is in no exclude zone.
Its label is the one of the LAST label zone that contains it (0 = none); LABELS gives the ids.
"""
import json, os
import numpy as np

PATH = "/dev/shm/pczones.json"
LABELS = ["floor", "wall", "ceiling", "furniture", "object", "other"]       # id = index + 1


def label_id(name):
    return LABELS.index(name) + 1 if name in LABELS else len(LABELS)


def inside(P, z):
    """Boolean mask of the points P (N,3) inside zone z."""
    P = np.asarray(P, np.float32)
    s = z.get("shape")
    if s == "box":
        A = np.asarray(z["axes"], np.float32)                   # rows = axis vectors
        q = (P - np.asarray(z["center"], np.float32)) @ A.T
        return np.all(np.abs(q) <= np.asarray(z["size"], np.float32) / 2, axis=1)
    if s == "prism":
        poly = np.asarray(z["points"], np.float32)[:, [0, 2]]
        if len(poly) < 3:
            return np.zeros(len(P), bool)
        m = (P[:, 1] >= z.get("ymin", -1e9)) & (P[:, 1] <= z.get("ymax", 1e9))
        x, y = P[:, 0], P[:, 2]
        ins = np.zeros(len(P), bool)
        j = len(poly) - 1
        for i in range(len(poly)):                               # even-odd ray casting
            xi, yi = poly[i]; xj, yj = poly[j]
            c = ((yi > y) != (yj > y)) & (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
            ins ^= c
            j = i
        return ins & m
    if s == "spheres":
        C = np.asarray(z["centers"], np.float32).reshape(-1, 3)
        if not len(C):
            return np.zeros(len(P), bool)
        R = np.broadcast_to(np.asarray(z.get("radii", 0.05), np.float32), (len(C),))
        from scipy.spatial import cKDTree
        d, idx = cKDTree(C).query(P, k=min(8, len(C)), distance_upper_bound=float(R.max()))
        d = d.reshape(len(P), -1); idx = idx.reshape(len(P), -1)
        ok = idx < len(C)
        r = np.where(ok, R[np.minimum(idx, len(C) - 1)], 0)
        return np.any(ok & (d <= r), axis=1)
    if s == "cut":
        return P[:, 1] < z["y"] if z.get("side", "below") == "below" else P[:, 1] > z["y"]
    return np.zeros(len(P), bool)


def evaluate(P, zones):
    """-> keep mask (N,), label ids uint8 (N,)"""
    n = len(P)
    keep = np.ones(n, bool); lab = np.zeros(n, np.uint8)
    inc = [z for z in zones if z.get("mode") == "include" and z.get("shape") != "cut"]
    if inc:
        keep[:] = False
        for z in inc:
            keep |= inside(P, z)
    for z in zones:
        if z.get("mode") == "exclude" or z.get("shape") == "cut":
            keep &= ~inside(P, z)
        elif z.get("mode") == "label":
            lab[inside(P, z)] = label_id(z.get("label"))
    return keep, lab


def point_kept(p, zones):
    k, _ = evaluate(np.asarray(p, np.float32).reshape(1, 3), zones)
    return bool(k[0])


class ZoneFile:
    """Reloads PATH when its seq changes."""
    def __init__(self, path=PATH):
        self.path = path; self.seq = None; self.zones = []

    def poll(self):
        """-> True if the zones changed."""
        try:
            with open(self.path) as f:
                d = json.load(f)
        except (OSError, ValueError):
            return False
        if d.get("seq") == self.seq:
            return False
        self.seq = d.get("seq"); self.zones = list(d.get("zones", []))
        return True
