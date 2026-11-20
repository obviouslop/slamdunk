#!/usr/bin/env python3
"""AprilTag (tagStandard41h12) fiducials: detection, boards, pose, and the immobile-anchor logic.

Why our own detector: OpenCV's aruco has no 41h12 dictionary and the pupil/apriltag wheels are not installed. The code
table + bit layout come from the AprilRobotics sources (tools/data/tag41h12.json, BSD-2). Quad finding is plain OpenCV.

Tag geometry (tagStandard41h12, "reversed border"): a 9x9 cell image. Ring 0 (outer) = data bits, ring 1 = black,
ring 2 = white, inner 3x3 = data bits. The detected quad is the white 5x5 square (cells 2..6): its corners are the four
points whose pose we measure. Printed "marker size" S mm = the whole 9x9 image, so the quad edge is 5/9 S.

Tag frame (like OpenCV aruco): origin = tag centre, x right, y UP, z out of the paper (towards the viewer).
Corner order everywhere: TL, TR, BR, BL as seen when the tag is upright.

Boards (BOARDS): "cal" = the 4-marker calibration target ids 0-3 (150 x 200 mm centre to centre, 50 mm markers;
0 TL, 1 TR, 2 BR, 3 BL in the printed sheet), "origin" = the single immobile origin marker id 4 (ORIGIN_MM edge).
Several physical copies of the cal board can exist (ceiling, floor, table): group detections of one frame by
position (group_boards) before solving.
"""
import json, os
import numpy as np
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
_fam = json.load(open(os.path.join(HERE, "data", "tag41h12.json")))
CODES = np.array(_fam["codes"], np.uint64)
BIT_COL = np.array(_fam["bit_x"]) + 2           # grid column / row of each of the 41 bits (MSB first)
BIT_ROW = np.array(_fam["bit_y"]) + 2
NBITS, MAXHAM = 41, 12                          # min distance of the family is 12 -> up to 5 bit errors are correctable
CELL_Q = np.array([[2, 2], [7, 2], [7, 7], [2, 7]], np.float32)          # TL TR BR BL of the white 5x5 square, in cells

TAG_MM = 50.0                                   # printed size of the calibration markers (9x9 image)
ORIGIN_MM = 150.0                               # id 4 origin marker: 150 mm printed edge (full 9x9 image)
BOARDS = {
    "cal": {"tag_mm": TAG_MM, "centres_mm": {0: (-75, 100), 1: (75, 100), 2: (75, -100), 3: (-75, -100)}},   # board x right, y up
    "origin": {"tag_mm": ORIGIN_MM, "centres_mm": {4: (0, 0)}},
}
ORIGIN_ID = 4


def quad_object(tag_mm):
    """3-D corners (TL TR BR BL, metres) of the detected quad in the tag frame (x right, y up, z out)."""
    a = tag_mm / 1000.0 * 5 / 9 / 2
    return np.array([[-a, a, 0], [a, a, 0], [a, -a, 0], [-a, -a, 0]], np.float64)


def board_object(board, ids):
    """-> {id: (4,3) metres in the board frame (origin = board centre, x right, y up, z out)}"""
    b = BOARDS[board]; q = quad_object(b["tag_mm"])
    return {i: q + np.array([b["centres_mm"][i][0] / 1000.0, b["centres_mm"][i][1] / 1000.0, 0]) for i in ids if i in b["centres_mm"]}


def board_of(tag_id):
    for n, b in BOARDS.items():
        if tag_id in b["centres_mm"]: return n
    return None


def render_tag(tag_id, cell=20):
    """Upright 9x9 tag image, uint8 (white 255). Used by the tests."""
    g = np.zeros((9, 9), np.uint8); g[2:7, 2:7] = 1         # ring 2 white; inner data overwritten below
    g[3:6, 3:6] = 0
    c = int(CODES[tag_id])
    for k in range(NBITS):
        g[BIT_ROW[k], BIT_COL[k]] = (c >> (NBITS - 1 - k)) & 1
    return cv2.resize(g * 255, (9 * cell, 9 * cell), interpolation=cv2.INTER_NEAREST)


_POP = np.array([bin(i).count("1") for i in range(1 << 16)], np.uint8)
def _ham(a, b):
    x = np.bitwise_xor(a, b)
    return _POP[(x & np.uint64(0xFFFF)).astype(np.int64)] + _POP[((x >> np.uint64(16)) & np.uint64(0xFFFF)).astype(np.int64)] \
        + _POP[((x >> np.uint64(32)) & np.uint64(0xFFFF)).astype(np.int64)]


def _sample_word(gray, H):
    """Homography H: cells -> image. Returns (word, contrast) or None when the ring structure does not look like a tag."""
    off = np.array([-0.22, 0.0, 0.22])
    gx, gy = np.meshgrid((np.arange(9)[:, None] + 0.5 + off[None, :]).ravel(), (np.arange(9)[:, None] + 0.5 + off[None, :]).ravel())
    p = cv2.perspectiveTransform(np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float32).reshape(-1, 1, 2), H).reshape(27, 27, 2)
    h, w = gray.shape
    if p[..., 0].min() < 1 or p[..., 1].min() < 1 or p[..., 0].max() > w - 2 or p[..., 1].max() > h - 2:
        return None
    v = cv2.remap(gray, p[..., 0].astype(np.float32), p[..., 1].astype(np.float32), cv2.INTER_LINEAR).astype(np.float32)
    v = v.reshape(9, 3, 9, 3).mean((1, 3))                                     # 3x3 samples per cell, averaged
    black = np.r_[v[1, 1:8], v[7, 1:8], v[2:7, 1], v[2:7, 7]]                 # ring 1
    white = np.r_[v[2, 2:7], v[6, 2:7], v[3:6, 2], v[3:6, 6]]                 # ring 2
    mb, mw = float(np.median(black)), float(np.median(white))
    if mw - mb < 12 or np.percentile(black, 80) > mb + 0.6 * (mw - mb) or np.percentile(white, 20) < mw - 0.6 * (mw - mb):
        return None
    thr = 0.5 * (mb + mw)
    word = 0
    for k in range(NBITS):
        word = (word << 1) | int(v[BIT_ROW[k], BIT_COL[k]] > thr)
    return np.uint64(word), mw - mb


def _order_cw(c):
    c = np.asarray(c, np.float32).reshape(4, 2)
    area = 0.5 * np.sum(c[:, 0] * np.roll(c[:, 1], -1) - np.roll(c[:, 0], -1) * c[:, 1])   # y down: >0 = clockwise on screen
    return c if area > 0 else c[::-1].copy()


def detect(gray, min_edge=8, refine=True, max_ham=4, blocks=None):
    """gray: uint8 image (rectified or raw). -> list of dict(id, corners (4,2) float32 TL TR BR BL, ham, edge_px, contrast)."""
    if gray.dtype != np.uint8:
        gray = np.clip(gray, 0, 255).astype(np.uint8)
    out = []
    cs = []
    for bs in (blocks or sorted({15, max(15, (min(gray.shape) // 40) | 1), 61})):
        bin_ = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, bs, -3)   # white regions
        cs += cv2.findContours(bin_, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[0]
    quads = []
    for c in cs:                                   # cheap pass: 4-vertex convex polygons only
        if len(c) < 4: continue
        ap = cv2.approxPolyDP(c, 0.04 * cv2.arcLength(c, True), True)
        if len(ap) == 4: quads.append(ap.reshape(4, 2))
    if not quads: return []
    Q = np.array(quads, np.float32)
    area = 0.5 * (Q[:, :, 0] * np.roll(Q[:, :, 1], -1, 1) - np.roll(Q[:, :, 0], -1, 1) * Q[:, :, 1]).sum(1)      # signed (y down: >0 = clockwise)
    Q = np.where((area < 0)[:, None, None], Q[:, ::-1], Q)
    E = np.linalg.norm(Q - np.roll(Q, -1, 1), axis=2); em = E.mean(1)
    d1 = np.roll(Q, -1, 1) - Q; d2 = np.roll(d1, -1, 1); cr = d1[:, :, 0] * d2[:, :, 1] - d1[:, :, 1] * d2[:, :, 0]
    keep = (E.min(1) >= min_edge) & (E.max(1) < 3.0 * E.min(1)) & (np.abs(area) > 0.6 * em ** 2) & (cr.min(1) > 0)
    Q = Q[keep]; E = E[keep]
    # the same quad shows up for several thresholds: de-duplicate by rounded centre + size
    key = np.round(np.c_[Q.mean(1) / 2, E.mean(1) / 2]).astype(int); _, ui = np.unique(key, axis=0, return_index=True)
    for q, e in zip(Q[ui], E[ui]):
        def sub(q0, div):
            cc = q0.reshape(-1, 1, 2).astype(np.float32).copy()
            cv2.cornerSubPix(gray, cc, (int(np.clip(e.mean() / div, 2, 7)),) * 2, (-1, -1),
                             (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, 0.01))
            cc = cc.reshape(4, 2)
            return cc if np.abs(cc - q0).max() < 0.35 * e.min() else q0
        tries = [sub(q, 6), sub(q, 4), q] if refine else [q]
        for qt in tries:
            hit = None
            for rot in range(4):
                qr = np.roll(qt, -rot, 0)
                s_ = _sample_word(gray, cv2.getPerspectiveTransform(CELL_Q, qr.astype(np.float32)))
                if s_ is None: continue
                d = _ham(CODES, s_[0]); i = int(np.argmin(d))
                if d[i] <= max_ham: hit = (i, int(d[i]), qr, s_[1]); break
            if hit:
                qr = hit[2]
                if refine:                                # polish with the decoded geometry
                    qr = np.roll(sub(np.roll(qr, rot, 0), 10), -rot, 0)
                out.append(dict(id=hit[0], corners=qr.copy(), ham=hit[1], edge_px=float(e.mean()), contrast=float(hit[3])))
                break
    # drop duplicates (same id, same place)
    res = []
    for d in sorted(out, key=lambda d: d["ham"]):
        m = d["corners"].mean(0)
        if all(not (r["id"] == d["id"] and np.linalg.norm(r["corners"].mean(0) - m) < 0.5 * d["edge_px"]) for r in res):
            res.append(d)
    return res


def solve_pose_all(obj, xn):
    """obj (N,3) metres in the board frame, xn (N,2) NORMALISED image coordinates (undistorted, K = I).
    -> list of (T_cam_board 4x4, rms in normalised units), best first. A single planar tag has TWO near-equal pose solutions
    (rotation flip) when it is small or seen head-on: compare the two rms values (ambiguity) or pass a prior."""
    obj = np.asarray(obj, np.float64).reshape(-1, 3); xn = np.asarray(xn, np.float64).reshape(-1, 2)
    K = np.eye(3); cands = []
    if len(obj) == 4 and _is_square(obj):
        ok, rvs, tvs, _ = cv2.solvePnPGeneric(obj, xn, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
        cands = list(zip(rvs, tvs)) if ok else []
    else:
        ok, rv, tv = cv2.solvePnP(obj, xn, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
        cands = [(rv, tv)] if ok else []
    out = []
    for rv, tv in cands:
        if tv[2, 0] <= 0: continue
        rv, tv = cv2.solvePnPRefineLM(obj, xn, K, None, rv, tv)
        pr, _ = cv2.projectPoints(obj, rv, tv, K, None)
        rms = float(np.sqrt(np.mean(np.sum((pr.reshape(-1, 2) - xn) ** 2, 1))))
        T = np.eye(4); T[:3, :3] = cv2.Rodrigues(rv)[0]; T[:3, 3] = tv.ravel(); out.append((T, rms))
    return sorted(out, key=lambda c: c[1])


def solve_pose(obj, xn, prior=None):
    """Best pose -> (T_cam_board, rms) or (None, inf). prior = an expected T_cam_board: picks the candidate whose rotation
    is closest to it instead of the lowest-rms one (resolves the single-tag flip)."""
    c = solve_pose_all(obj, xn)
    if not c: return None, np.inf
    if prior is not None and len(c) > 1:
        c = sorted(c, key=lambda x: pose_diff(prior, x[0])[1])
    return c[0]


def ambiguity(obj, xn):
    """rms of the second pose candidate / rms of the best (1 = indistinguishable, >2 = reliable). inf for multi-tag solves."""
    c = solve_pose_all(obj, xn)
    return np.inf if len(c) < 2 else c[1][1] / max(c[0][1], 1e-12)


def _is_square(obj):
    return len(obj) == 4 and abs(np.linalg.norm(obj[0] - obj[1]) - np.linalg.norm(obj[1] - obj[2])) < 1e-9 and np.ptp(obj[:, 2]) < 1e-9 \
        and abs(np.linalg.norm(obj[0] - obj[2]) - np.sqrt(2) * np.linalg.norm(obj[0] - obj[1])) < 1e-9


def view_angle_deg(T_cam_board):
    """angle between the board normal and the line of sight (0 = head-on)"""
    n = T_cam_board[:3, 2]; r = T_cam_board[:3, 3] / np.linalg.norm(T_cam_board[:3, 3])
    return float(np.degrees(np.arccos(np.clip(abs(np.dot(n, r)), 0, 1))))


def group_boards(dets, stage_from_cam, xn_of, radius=0.6):
    """Several copies of the same board (same ids) can be in view. Cluster the detections of one frame by the stage position
    of their tag. dets: detect() output, xn_of(corners)->normalised coords, stage_from_cam 4x4.
    -> list of lists of dets (one list per physical board, at most one detection per id)."""
    pos = []
    for d in dets:
        b = board_of(d["id"]); T, _ = solve_pose(board_object(b, [d["id"]])[d["id"]], xn_of(d["corners"]))
        pos.append(None if T is None else (stage_from_cam @ T)[:3, 3])
    groups = []
    for d, p in zip(dets, pos):
        if p is None: continue
        for g in groups:
            if board_of(g[0][0]["id"]) == board_of(d["id"]) and np.linalg.norm(g[0][1] - p) < radius and all(x[0]["id"] != d["id"] for x in g):
                g.append((d, p)); break
        else:
            groups.append([(d, p)])
    return [[x[0] for x in g] for g in groups]


def solve_board(group, xn_of):
    """All detections of one physical board -> (T_cam_board, rms) using every corner jointly."""
    b = board_of(group[0]["id"]); O = board_object(b, [d["id"] for d in group])
    obj = np.concatenate([O[d["id"]] for d in group]); xn = np.concatenate([xn_of(d["corners"]) for d in group])
    return solve_pose(obj, xn)


# ------------------------------------------------------------------ pose maths helpers
def mean_pose(Ts, w=None):
    """Weighted mean of 4x4 poses (chordal rotation mean) -> 4x4."""
    Ts = np.asarray(Ts); w = np.ones(len(Ts)) if w is None else np.asarray(w, float)
    t = (Ts[:, :3, 3] * w[:, None]).sum(0) / w.sum()
    M = (Ts[:, :3, :3] * w[:, None, None]).sum(0); U, _, Vt = np.linalg.svd(M)
    R = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; return T


def pose_diff(Ta, Tb):
    """Tb relative to Ta: (translation difference in Ta-frame axes (m), rotation angle (deg))"""
    D = np.linalg.inv(Ta) @ Tb
    return D[:3, 3], float(np.degrees(np.arccos(np.clip((np.trace(D[:3, :3]) - 1) / 2, -1, 1))))


class Anchor:
    """An immobile marker (origin marker or any fixed board): its stage pose is constant in reality, so the spread of the
    measured stage_from_marker is the SteamVR pose error (plus detection noise), and  ref @ inv(measured)  corrects it.

    add(stage_from_marker, weight) collects detections; once `n_ref` good ones are in, the reference is frozen
    (robust median pose) and correction(...) becomes available."""
    def __init__(self, n_ref=15, max_dev=0.15):
        self.n_ref = n_ref; self.max_dev = max_dev; self.ref = None; self.samples = []

    def add(self, T_stage_marker, w=1.0):
        if self.ref is None:
            self.samples.append((np.asarray(T_stage_marker), w))
            if len(self.samples) >= self.n_ref:
                Ts = np.array([s[0] for s in self.samples]); med = np.median(Ts[:, :3, 3], 0)
                ok = np.linalg.norm(Ts[:, :3, 3] - med, axis=1) < self.max_dev          # drop gross outliers
                self.ref = mean_pose(Ts[ok], np.array([s[1] for s in self.samples])[ok])
        return self.ref is not None

    def correction(self, T_stage_cam, T_cam_marker):
        """4x4 C with  true_stage_from_cam = C @ T_stage_cam  (so the marker lands on its reference pose)."""
        if self.ref is None: return None
        return self.ref @ np.linalg.inv(T_cam_marker) @ np.linalg.inv(T_stage_cam)
