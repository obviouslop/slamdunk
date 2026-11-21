#!/usr/bin/env python3
"""Stereo depth + world placement for live Arcturus frames (factory EEPROM calibration).

    from livestereo import LiveStereo
    ls = LiveStereo(scale=0.5)
    pts_camL, rgb = ls.points(raw_left, raw_right)        # uint16 mosaics (RGGB, black 64)
    pts_world = ls.to_world(pts_camL, hmd_pose_3x4)       # OpenVR Standing universe, metres

Geometry (verified 2026-11-17, see NOTES.md):
  * intrinsics/extrinsics from captures/arcturus_calib.json (EEPROM), camera frames = OpenCV
    (epipolar median 1.1 px on live frames, no axis conversion needed)
  * head_from_camL = diag(-1,1,-1) @ cad_from_camL  (OpenVR head = CAD rotated 180° about y,
    /persist/device_config.json "head"); hand-eye on live data: 0.93° / 0.68 cm per pair.
"""
import json, os
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
CAL = os.path.join(HERE, "..", "captures", "arcturus_calib.json")
DEPTH_CAL = os.path.join(HERE, "..", "captures", "depth_calib.json")   # Z = k * f B / (disp + d0), re/table_calib.py


def _kd(c):
    i = c["intrinsics"][0]
    return (np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1.]]),
            np.array([i["k1"], i["k2"], i["k3"], i["k4"]], float).reshape(4, 1))


def _cad_from_cam(c):
    e = c["extrinsics"]; x = np.array(e["plus_x"], float); z = np.array(e["plus_z"], float); y = np.cross(z, x)
    M = np.eye(4); M[:3, :3] = np.stack([x, y, z], 1); M[:3, 3] = np.array(e["position"], float) / 1000; return M


def unpack_raw(buf, W=2464, H=2464, S=3088):
    b = np.frombuffer(buf, np.uint8, count=S * H).reshape(H, S)[:, :W * 5 // 4].reshape(H, W // 4, 5).astype(np.uint16)
    lo = b[:, :, 4]
    return np.stack([(b[:, :, i] << 2) | ((lo >> (2 * i)) & 3) for i in range(4)], -1).reshape(H, W)


class LiveStereo:
    def __init__(self, scale=0.5, max_depth=4.0, min_depth=0.25, cal=CAL, edge_rel=0.0, block=5, uniq=10, pscale=1.0, median=0, tone="log"):
        cams = {c["sourceCamera"]: c for c in json.load(open(cal))["cameras"]}
        self.KL, self.DL = _kd(cams["passthrough_left"]); self.KR, self.DR = _kd(cams["passthrough_right"])
        TL, TR = _cad_from_cam(cams["passthrough_left"]), _cad_from_cam(cams["passthrough_right"])
        M = np.linalg.inv(TR) @ TL; self.R, self.t = np.ascontiguousarray(M[:3, :3]), np.ascontiguousarray(M[:3, 3])   # X_R = R X_L + t
        self.head_from_camL = np.diag([-1., 1, -1, 1]) @ TL
        self.baseline = float(np.linalg.norm(self.t)); self.max_depth, self.min_depth = max_depth, min_depth
        self.edge_rel = edge_rel   # flying-pixel filter: drop if local (3x3) depth range > edge_rel * depth (0 = off)
        n = int(2464 * scale); self.size = (n, n)
        R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(self.KL, self.DL, self.KR, self.DR, (2464, 2464), self.R, self.t.reshape(3, 1),
                                                      flags=cv2.CALIB_ZERO_DISPARITY, newImageSize=self.size, balance=0.0, fov_scale=1.0)
        self.R1, self.R2, self.P1, self.P2 = R1, R2, P1, P2
        # Source images are 2x2-binned (1232^2): binned pixel i covers full pixels 2i, 2i+1 -> centre 2i+0.5,
        # so K_half = K/2 with c_half = (c - 0.5)/2. Distortion is in normalised coordinates (unchanged).
        def half(K):
            Kh = K.copy(); Kh[0, 0] /= 2; Kh[1, 1] /= 2; Kh[0, 2] = (K[0, 2] - 0.5) / 2; Kh[1, 2] = (K[1, 2] - 0.5) / 2; return Kh
        self.mapL = cv2.fisheye.initUndistortRectifyMap(half(self.KL), self.DL, R1, P1, self.size, cv2.CV_16SC2)
        self.mapR = cv2.fisheye.initUndistortRectifyMap(half(self.KR), self.DR, R2, P2, self.size, cv2.CV_16SC2)
        self.f = P1[0, 0]
        self.horizontal = abs(P2[0, 3]) > abs(P2[1, 3])
        if not self.horizontal:
            raise RuntimeError("vertical rectification")
        self.left_first = P2[0, 3] < 0       # camL is the left image of the rectified pair
        nd = 16 * int(np.ceil(self.f * self.baseline / min_depth / 16))
        self.median = median
        self.d0, self.k = 0.0, 1.0
        if os.path.exists(DEPTH_CAL) and not os.environ.get("PC_NO_DEPTH_CAL"):
            dc = json.load(open(DEPTH_CAL)); self.d0 = float(dc["disp_offset_px"]) * scale / dc.get("valid_for_scale", scale); self.k = float(dc["depth_scale"])
        self.sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=nd, blockSize=block,
                                          P1=int(8 * block * block * pscale), P2=int(32 * block * block * pscale),
                                          uniquenessRatio=uniq, speckleWindowSize=100, speckleRange=2, disp12MaxDiff=1,
                                          mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        u, v = np.meshgrid(np.arange(n), np.arange(n)); self.uv = (u - P1[0, 2], v - P1[1, 2])
        self.tone = tone    # matcher grey tone curve (see grey8); colours are not affected
        self.debug = None   # set to a dict: the next rectified()/depth() call stores its intermediate buffers there (debug capture)

    @staticmethod
    def _rgb(raw):
        """RGGB mosaic (R at (0,0)) -> half-resolution BGR float (2x2 binning demosaic, 1232^2)."""
        r = raw.astype(np.float32)
        rr = r[0::2, 0::2]; g = (r[0::2, 1::2] + r[1::2, 0::2]) * 0.5; b = r[1::2, 1::2]
        return np.clip(np.stack([b, g, rr], -1) - 64, 0, None)

    def rectified(self, rawL, rawR, wb=None, half=False):
        """-> disp (rectified camL view, px), colour (rectified, BGR float in raw units), wb, s (gain to 0..255)
        rawL/rawR: uint16 mosaics, or (half=True) already binned half-res BGR float (tsdf.bin_raw10)"""
        cL, cR = (rawL, rawR) if half else (self._rgb(rawL), self._rgb(rawR))
        if wb is None:
            m = cL.reshape(-1, 3).mean(0); wb = m.mean() / np.maximum(m, 1e-6)
        gL = cv2.cvtColor(cL, cv2.COLOR_BGR2GRAY); gR = cv2.cvtColor(cR, cv2.COLOR_BGR2GRAY)
        s = 255 / max(np.percentile(gL[::4, ::4], 99.5), 1)
        g8L, g8R = self.grey8(gL, gR, s)
        rL = cv2.remap(g8L, *self.mapL, cv2.INTER_LINEAR)
        rR = cv2.remap(g8R, *self.mapR, cv2.INTER_LINEAR)
        rc = cv2.remap(cL, *self.mapL, cv2.INTER_LINEAR)
        mL, mR = (rL, rR) if self.left_first else (np.ascontiguousarray(rL[:, ::-1]), np.ascontiguousarray(rR[:, ::-1]))
        draw = self.sgbm.compute(mL, mR)                     # int16, 1/16 px, matcher orientation (mirrored if not left_first)
        disp = draw.astype(np.float32) / 16
        if not self.left_first:   # camL is the right image of the rectified pair: both were mirrored so it became the left one
            disp = disp[:, ::-1]
        disp_sgbm = disp
        if self.median:
            dm = cv2.medianBlur(disp, self.median)          # suppress isolated noisy matches
            disp = np.where(np.abs(dm - disp) < 1.0, disp, -1)
        if self.debug is not None:
            self.debug.update(binned_L=cL, binned_R=cR, wb=np.asarray(wb, np.float32), grey_gain=np.float32(s),
                              grey_L=g8L, grey_R=g8R, tone=np.array(self.tone),
                              rect_grey_L=rL, rect_grey_R=rR, matcher_in_L=mL, matcher_in_R=mR, matcher_out_int16=draw,
                              disp_sgbm=np.ascontiguousarray(disp_sgbm), disp=np.ascontiguousarray(disp), rect_colour_raw=rc)
        return disp, rc, wb, s

    def grey8(self, gL, gR, s):
        """float grey (raw units, more than 8 bits after binning) -> uint8 matcher input with the SAME curve for both cameras.
        x = g * s / 255 (1.0 = 99.5th percentile of the left image).
          linear: 255 x (dark scenes with a lamp in view end up in a few grey levels)
          sqrt:   255 sqrt(x)
          log:    255 log1p(50 x) / log1p(50)   (DEFAULT: best multi-view consistency on live16/17/19/20, re/tone_eval.py)
          auto:   255 x^gamma, gamma per frame so the median maps to 96, clamped to [0.3, 1]"""
        if self.tone == "linear":            # exactly the old behaviour (no LUT quantisation)
            return np.clip(gL * s, 0, 255).astype(np.uint8), np.clip(gR * s, 0, 255).astype(np.uint8)
        x = (np.arange(4096, dtype=np.float32) + 0.5) / 4096          # LUT over x in [0, 1], 4096 steps (> 10 bit)
        if self.tone == "sqrt":
            lut = 255 * np.sqrt(x)
        elif self.tone == "log":
            lut = 255 * np.log1p(50 * x) / np.log1p(50)
        elif self.tone == "auto":
            med = float(np.median(gL[::4, ::4])) * s / 255
            gam = float(np.clip(np.log(96 / 255) / np.log(min(max(med, 1e-4), 0.99)), 0.3, 1.0))
            lut = 255 * np.power(x, gam)
        else:
            raise ValueError(f"tone {self.tone}")
        lut = (lut + 0.5).astype(np.uint8); q = s / 255 * 4096
        f = lambda g: lut[np.clip(g * q, 0, 4095).astype(np.int32)]
        return f(gL), f(gR)

    def rectified_rgb_pair(self, rawL, rawR, half=False):
        """-> (left, right) rectified RGB uint8 with the SAME white balance / tone as depth()'s colour output (for stereo viewing)."""
        cL, cR = (rawL, rawR) if half else (self._rgb(rawL), self._rgb(rawR))
        m = cL.reshape(-1, 3).mean(0); wb = m.mean() / np.maximum(m, 1e-6)
        gL = cv2.cvtColor(cL, cv2.COLOR_BGR2GRAY); s = 255 / max(np.percentile(gL[::4, ::4], 99.5), 1)
        out = []
        for c, mp in ((cL, self.mapL), (cR, self.mapR)):
            rc = cv2.remap(c, *mp, cv2.INTER_LINEAR); x = rc * (wb * s)
            x += np.maximum(x.max(-1, keepdims=True) - 255, 0); x[rc.max(-1) > 0.97 * (1023 - 64)] = 255
            out.append(np.ascontiguousarray(np.clip(x, 0, 255).astype(np.uint8)[:, :, ::-1]))
        return out[0], out[1]

    sat = 1.0       # colour finishing of the point colours (the passthrough image is tone mapped / saturated, raw linear data is not)
    gamma = 1.0     # <1 brightens mid-tones (x -> x**gamma on 0..1)

    def finish(self, x):
        """white-balanced BGR float (0..255 scale) -> uint8 BGR with optional saturation boost and mid-tone lift."""
        if self.sat != 1.0:
            lum = x @ np.array([0.114, 0.587, 0.299], np.float32)
            x = lum[..., None] + self.sat * (x - lum[..., None])
        x = np.clip(x, 0, 255)
        if self.gamma != 1.0:
            x = 255.0 * (x / 255.0) ** self.gamma
        return (x + 0.5).astype(np.uint8)

    def depth(self, rawL, rawR, far=None, half=False):
        """-> Z (rectified camL view, metres along the optical axis, 0 = invalid, valid up to `far`),
              RGB uint8 rectified colour, disp"""
        disp, rc, wb, s = self.rectified(rawL, rawR, half=half)
        far = far or self.max_depth
        Z = np.where(disp > 0.5, self.k * self.f * self.baseline / np.maximum(disp + self.d0, 1e-6), 0).astype(np.float32)
        Z[(Z > far) | (Z < self.min_depth)] = 0
        x = rc * (wb * s)
        # highlights: blend towards white where a channel clips (else white-balanced highlights turn magenta),
        # and force white where the sensor itself saturated
        x += np.maximum(x.max(-1, keepdims=True) - 255, 0)
        x[rc.max(-1) > 0.97 * (1023 - 64)] = 255
        rgb = self.finish(x)[:, :, ::-1]
        if self.debug is not None:
            self.debug.update(Z=np.ascontiguousarray(Z), rgb=np.ascontiguousarray(rgb), far=np.float32(far))
        return np.ascontiguousarray(Z), np.ascontiguousarray(rgb), disp

    @staticmethod
    def confidence(Z, rgb, g0=8.0, gmin=1.5, jump=0.04):
        """Per-pixel measurement confidence 0..1 (LSD-SLAM style depth-hypothesis variance, cheap version).
        Depth variance of a block-matching pixel is large where the image has little texture (the disparity is then
        interpolated) and where the local depth jumps (flying pixels at occlusion edges / foreground fattening).
          texture : mean |gradient| of the left image over 7x7 (grey levels) -> c_t = clip(g / g0, 0.15, 1), 0 below gmin
          edge    : (max - min of Z over 3x3) / Z -> c_e = clip(1 - rel / jump, 0.1, 1)
        -> float32 HxW (0 where Z invalid or the texture is below gmin: such pixels should not be integrated at all)."""
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) * 0.125; gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) * 0.125
        tex = cv2.blur(np.hypot(gx, gy), (7, 7))
        ct = np.clip(tex / g0, 0.15, 1.0); ct[tex < gmin] = 0
        k = np.ones((3, 3), np.uint8)
        zmax = cv2.dilate(Z, k); zmin = cv2.erode(np.where(Z > 0, Z, 1e3).astype(np.float32), k)
        rel = (zmax - np.minimum(zmin, zmax)) / np.maximum(Z, 1e-3)
        ce = np.clip(1.0 - rel / jump, 0.1, 1.0)
        return np.where(Z > 0, ct * ce, 0).astype(np.float32)

    def debug_params(self):
        """matcher / rectification parameters for a debug capture (JSON-able)"""
        g = self.sgbm
        return dict(scale_size=list(self.size), f=float(self.f), baseline_m=float(self.baseline), left_first=bool(self.left_first),
                    k=float(self.k), d0_px=float(self.d0), min_depth=float(self.min_depth), max_depth=float(self.max_depth),
                    median=int(self.median), edge_rel=float(self.edge_rel), sat=float(self.sat), gamma=float(self.gamma),
                    sgbm=dict(minDisparity=g.getMinDisparity(), numDisparities=g.getNumDisparities(), blockSize=g.getBlockSize(),
                              P1=g.getP1(), P2=g.getP2(), uniquenessRatio=g.getUniquenessRatio(), speckleWindowSize=g.getSpeckleWindowSize(),
                              speckleRange=g.getSpeckleRange(), disp12MaxDiff=g.getDisp12MaxDiff(), preFilterCap=g.getPreFilterCap(),
                              mode=int(g.getMode())),
                    R1=self.R1.tolist(), R2=self.R2.tolist(), P1=self.P1.tolist(), P2=self.P2.tolist())

    def world_from_rect(self, hmd_pose_3x4):
        H = np.eye(4); H[:3, :] = np.asarray(hmd_pose_3x4); W = H @ self.head_from_camL
        Rr = np.eye(4); Rr[:3, :3] = self.R1.T             # camL <- rect
        return W @ Rr

    def points(self, rawL, rawR, wb=None):
        disp, rc, wb, s = self.rectified(rawL, rawR, wb)
        Z = self.k * self.f * self.baseline / np.maximum(disp + self.d0, 1e-6)
        ok = (disp > 0.5) & (Z < self.max_depth) & (Z > self.min_depth)
        if self.edge_rel > 0:   # flying pixels at depth discontinuities = most of the "spray" along view rays
            Zv = np.where(ok, Z, np.nan).astype(np.float32)
            Zmax = cv2.dilate(np.nan_to_num(Zv, nan=0.0), np.ones((3, 3), np.uint8))
            Zmin = -cv2.dilate(-np.nan_to_num(Zv, nan=1e6), np.ones((3, 3), np.uint8))
            ok &= (Zmax - Zmin) < self.edge_rel * Z
        P = np.stack([self.uv[0][ok] * Z[ok] / self.f, self.uv[1][ok] * Z[ok] / self.f, Z[ok]], -1)
        P = P @ self.R1            # rectified -> camL (rect = R1 @ camL)
        col = np.clip(rc[ok] * wb * s / 255, 0, 1)
        return P, col, disp

    def to_world(self, P, hmd_pose_3x4):
        H = np.eye(4); H[:3, :] = np.asarray(hmd_pose_3x4); W = H @ self.head_from_camL
        return P @ W[:3, :3].T + W[:3, 3]
