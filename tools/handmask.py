#!/usr/bin/env python3
"""Keep hands / arms holding the controllers out of the point cloud.

hand_mask(Z, rect_from_stage, f, cx, cy, hands) -> bool mask (H, W), True = pixel belongs to a controller or to whatever
is connected to it in the depth map (the hand, wrist).  The server zeroes those depths before fusing; the depth-passthrough /
occlusion frames keep them (the hand SHOULD occlude the virtual scene).

How: each tracked controller position (STAGE) is projected into the rectified left image. If it is in view, the stereo depth
around it is searched for pixels at about the controller's distance (the controller itself, +- latency slack); the region
connected to that seed through depth-continuous steps (3 % per pixel step in log depth) is the controller + hand, limited to
a ball around the controller so it cannot leak into the floor or the table the hand touches; then dilated.
hands: iterable of 3-vectors (STAGE metres).
"""
import numpy as np
import cv2


def hand_mask(Z, rect_from_stage, f, cx, cy, hands, radius=0.25, seed_slack=0.15, step=0.03, grow_px=4):
    H, W = Z.shape
    out = np.zeros((H, W), bool)
    if not hands: return out
    valid = Z > 0
    L = np.where(valid, np.log(np.maximum(Z, 1e-3)), -10.0).astype(np.float32)
    R = rect_from_stage[:3, :3]; t = rect_from_stage[:3, 3]
    for p in hands:
        pc = R @ np.asarray(p, float) + t
        if pc[2] < 0.15: continue
        u = f * pc[0] / pc[2] + cx; v = f * pc[1] / pc[2] + cy
        rpx = max(4, int(f * (0.07 + seed_slack) / pc[2]))                     # controller radius + motion since the pose sample
        if u < -rpx or u > W + rpx or v < -rpx or v > H + rpx: continue         # not in view
        x0, x1 = int(max(u - rpx, 0)), int(min(u + rpx + 1, W)); y0, y1 = int(max(v - rpx, 0)), int(min(v + rpx + 1, H))
        if x1 <= x0 or y1 <= y0: continue
        win = Z[y0:y1, x0:x1]
        near = (win > 0) & (np.abs(win - pc[2]) < 0.12 + seed_slack)
        if near.sum() < 5: continue                                              # nothing at the controller's distance
        yy, xx = np.nonzero(near)
        d2 = (xx + x0 - u) ** 2 + (yy + y0 - v) ** 2; k = int(np.argmin(d2))
        seed = (int(xx[k] + x0), int(yy[k] + y0))
        # continuity-connected region (floating range, 8-connected) in log-depth
        m = np.zeros((H + 2, W + 2), np.uint8)
        img = L.copy()
        cv2.floodFill(img, m, seed, 0, loDiff=step, upDiff=step, flags=8 | cv2.FLOODFILL_MASK_ONLY | (255 << 8))
        reg = m[1:-1, 1:-1] > 0
        # ball around the controller (3-D) - Z in metres, pixel -> point
        yy, xx = np.nonzero(reg)
        z = Z[yy, xx]
        X = np.stack([(xx - cx) / f * z, (yy - cy) / f * z, z], 1)
        keep = np.linalg.norm(X - pc, axis=1) < radius
        reg2 = np.zeros((H, W), bool); reg2[yy[keep], xx[keep]] = True
        out |= reg2
    if out.any() and grow_px > 0:
        out = cv2.dilate(out.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow_px + 1,) * 2)) > 0
    return out
