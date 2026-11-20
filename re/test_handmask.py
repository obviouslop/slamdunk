#!/usr/bin/env python3
"""tools/handmask.py on a recording with controller poses (hmd_500hz.txt.controllers): writes overlays of the masked depth
  test_handmask.py captures/live16 [--frames 5,20,...]   -> captures/scratch/handmask/NNNNN.jpg + summary"""
import argparse, json, os, sys, numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from livestereo import LiveStereo
from tsdf import bin_raw10
import handmask, ctrl_overlay as co
ap = argparse.ArgumentParser(); ap.add_argument("rec"); ap.add_argument("--every", type=int, default=6); ap.add_argument("--max", type=int, default=24)
a = ap.parse_args()
ls = LiveStereo(scale=0.35); tracks = co.load_controllers(os.path.join(a.rec, "hmd_500hz.txt.controllers"))
recs = [json.loads(l) for l in open(os.path.join(a.rec, "frames.jsonl"))]
out = os.path.join(HERE, "..", "captures", "scratch", "handmask"); os.makedirs(out, exist_ok=True)
n_in = n_masked = 0; fr = []
for r in recs[::a.every][:a.max]:
    i = r["i"]; Z, rgb, _ = ls.depth(bin_raw10(open(os.path.join(a.rec, f"{i:05d}_left.y10p"), "rb").read()),
                                      bin_raw10(open(os.path.join(a.rec, f"{i:05d}_right.y10p"), "rb").read()), far=4.0, half=True)
    W = ls.world_from_rect(r["left"]["pose"]); hands = []
    for role in (1, 2):
        if role in tracks:
            M = co.pose_at(tracks[role], r["left"]["ts"])
            if M is not None: hands.append(M[:3, 3])
    m = handmask.hand_mask(Z, np.linalg.inv(W), ls.f, ls.P1[0, 2], ls.P1[1, 2], hands)
    img = np.clip(np.nan_to_num(Z) / 3.0, 0, 1); img = cv2.applyColorMap((255 - img * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[m] = (0.4 * img[m] + 0.6 * np.array([255, 255, 255])).astype(np.uint8)
    for p in hands:
        o = np.linalg.inv(W)[:3, :3] @ p + np.linalg.inv(W)[:3, 3]
        if o[2] > 0.1: cv2.circle(img, (int(ls.f * o[0] / o[2] + ls.P1[0, 2]), int(ls.f * o[1] / o[2] + ls.P1[1, 2])), 6, (0, 0, 0), 2)
    cv2.imwrite(os.path.join(out, f"{i:05d}.jpg"), img)
    fr.append((i, len(hands), int(m.sum()), float((Z > 0).sum()))); print(f"frame {i}: hands {len(hands)}, masked px {int(m.sum())} ({100 * m.sum() / max((Z > 0).sum(), 1):.1f} % of valid)")
