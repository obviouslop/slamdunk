#!/usr/bin/env python3
"""Floor level + under-floor (mirror image) removal.

A glossy floor reflecting a window / the sky shows up in stereo as a consistent virtual image BEHIND the floor plane (a mirror
image is at a fixed place for every viewpoint, so multi-view consistency cannot reject it). Nothing real is below the floor of a
room, so points clearly below a FLAT floor are removed. Guarded: only when the floor is really a sharp horizontal plane
(outdoors / slopes / stairs give a broad histogram -> no filtering).
"""
import numpy as np


def floor_level(P, band=0.02):
    """-> (y, sharpness) of the lowest strong horizontal-plane peak of the y histogram, or (None, 0).
    sharpness = peak bin / median bin in the 8-40 cm above it (furniture, clutter): ~1 on slopes and rough ground, > 2 on a floor."""
    if len(P) < 3000: return None, 0.0
    lo, hi = np.percentile(P[:, 1], [0.5, 60])
    h, e = np.histogram(P[:, 1], bins=np.arange(lo - 0.1, hi + 0.5, band))
    if not len(h) or h.max() < 1500: return None, 0.0
    k = int(np.flatnonzero(h > 0.4 * h.max())[0]); k += int(np.argmax(h[k:k + 4]))
    above = h[k + int(0.08 / band):k + int(0.40 / band)]
    if len(above) < 5: return None, 0.0
    return float(0.5 * (e[k] + e[k + 1])), float(h[k] / max(np.median(above), 1.0))


def underfloor_mask(P, tol=0.07, min_sharp=2.0):
    """bool mask of points more than `tol` below a flat floor (all False when the floor is not flat / not found)."""
    y, sharp = floor_level(P)
    if y is None or sharp < min_sharp: return np.zeros(len(P), bool), y, sharp
    return P[:, 1] < y - tol, y, sharp
