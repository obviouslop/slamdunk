#!/usr/bin/env python3
"""Depth-correct passthrough frame publisher (server side).

Writes the latest rectified left colour image + a coarse depth grid + the camera pose to /dev/shm/pcpass.bin, which
the viewer (viewer/passthrough.gd) renders as a world-locked mesh: every grid vertex is unprojected with its measured
depth, so the camera image sits on the real surfaces (not on SteamVR's fixed-depth proxy).  Same idea as the stereo
mode of Rectus' openxr-steamvr-passthrough layer (disparity -> displaced grid in the vertex shader, depth edges cut).

File layout (little endian), written to a temp file and renamed (readers never see a partial frame):
  0   4s   b"PCPT"
  4   u32  seq
  8   u32  Wc, Hc       colour size (RGB8)
  16  u32  Wd, Hd       depth grid size (float32 metres, 0 = invalid); grid vertex (i, j) sits at depth pixel centre
  24  f32  f, cx, cy, n intrinsics of the rectified image (pixels) and its size n (square)
  40  f32  stage_from_rect 3x4 row-major (OpenCV camera axes: x right, y down, z forward -> OpenXR STAGE)
  88  f64  ts           CLOCK_MONOTONIC exposure time
  96  u32  frame        arcgrab frame number
  100 f32  block        depth pixels per grid cell (grid cell (i, j) centre = rectified pixel ((j+.5)*block-.5, (i+.5)*block-.5))
  104 pad to 128
  128 colour Wc*Hc*3, then depth Wd*Hd*4
"""
import os, struct
import numpy as np
import cv2

PATH = "/dev/shm/pcpass.bin"
HDR = 128


def depth_grid(Z, block):
    """Block-average of the valid depths (blocks with < 50 % valid pixels -> 0)."""
    n = Z.shape[0] // block * block
    Zc = Z[:n, :n]
    v = (Zc > 0).astype(np.float32)
    s = cv2.resize(Zc, (n // block, n // block), interpolation=cv2.INTER_AREA)       # mean of Z (zeros included)
    c = cv2.resize(v, (n // block, n // block), interpolation=cv2.INTER_AREA)        # valid fraction
    return np.where(c >= 0.5, s / np.maximum(c, 1e-6), 0).astype(np.float32)


def pack(Z, rgb, f, cx, cy, stage_from_rect, ts=0.0, frame=0, seq=0, colour_size=432, block=8):
    n = Z.shape[0]
    D = depth_grid(Z, block)
    C = cv2.resize(rgb, (colour_size, colour_size), interpolation=cv2.INTER_AREA)
    T = np.asarray(stage_from_rect, np.float32)[:3, :4]
    h = struct.pack("<4sI4I4f12fdIf", b"PCPT", seq, C.shape[1], C.shape[0], D.shape[1], D.shape[0],
                    f, cx, cy, n, *T.ravel(), ts, frame, block)
    return h.ljust(HDR, b"\0") + np.ascontiguousarray(C, np.uint8).tobytes() + D.tobytes()


def publish(buf, path=PATH):
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        fh.write(buf)
    os.replace(tmp, path)


def unpack(buf):
    magic, seq, Wc, Hc, Wd, Hd, f, cx, cy, n, *rest = struct.unpack_from("<4sI4I4f12fdIf", buf)
    T = np.array(rest[:12], np.float32).reshape(3, 4); ts, frame, block = rest[12], rest[13], rest[14]
    C = np.frombuffer(buf, np.uint8, Wc * Hc * 3, HDR).reshape(Hc, Wc, 3)
    D = np.frombuffer(buf, np.float32, Wd * Hd, HDR + Wc * Hc * 3).reshape(Hd, Wd)
    return dict(seq=seq, f=f, cx=cx, cy=cy, n=n, T=T, ts=ts, frame=frame, block=block, colour=C, depth=D)
