#!/usr/bin/env python3
"""Reader for frames published by src/arcgrab in /dev/shm/arcgrab.

Layout (see ShmHdr in src/arcgrab.cpp): 4096-byte header, then 2 cameras x 2 slots of packed
RAW10 (MIPI, 4 px / 5 B) frames, line stride `line_bytes`, `line_px` pixels per line.
Camera 0 = passthrough_left (/dev/video0), camera 1 = passthrough_right (/dev/video3).
Bayer pattern BGGR. Seqlock per camera: seq is odd while the writer copies.

    from arcshm import ArcShm
    s = ArcShm()
    meta, raw = s.latest(0)          # uint16 HxW mosaic (black level 64 not subtracted)
    rgb = s.rgb(raw, half=True)      # float32 BGR 0..1, gray-world balanced
"""
import mmap, os, struct, time
import numpy as np

HDR = struct.Struct("<8s8I")                       # magic + 8 u32 (version..pad)
CAM = struct.Struct("<Q4IdQ12f4I")                 # seq, slot, exp, gain, bytes, ts, frame, pose[12], pose_valid, buf_flags, dgain, pad
CAM_OFF = HDR.size + (-HDR.size % 8)               # ShmCam is 8-byte aligned in the C struct


class ArcShm:
    def __init__(self, path="/dev/shm/arcgrab"):
        self.f = open(path, "rb")
        self.m = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        magic, ver, self.line_px, self.line_bytes, self.black, self.skip, self.slot_bytes, self.ncams, _ = HDR.unpack_from(self.m, 0)
        if magic != b"ARCGRAB1":
            raise RuntimeError("bad magic %r" % magic)

    def _hdr(self):
        _, _, self.line_px, self.line_bytes, self.black, self.skip, self.slot_bytes, self.ncams, _ = HDR.unpack_from(self.m, 0)

    def _cam(self, k):
        v = CAM.unpack_from(self.m, CAM_OFF + k * CAM.size)
        seq, slot, exp, gain, nbytes, ts, frame = v[:7]
        pose = np.array(v[7:19], np.float64).reshape(3, 4)
        return dict(seq=seq, slot=slot, exp_lines=exp, gain_code=gain, bytes=nbytes, ts=ts, frame=frame,
                    pose=pose, pose_valid=bool(v[19]), buf_flags=v[20], dgain=v[21])

    def latest_packed(self, k, after_frame=-1, timeout=2.0):
        """Like latest() but returns the packed RAW10 bytes (cheap; decode later with unpack())."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            self._hdr(); a = self._cam(k)
            if a["seq"] % 2 or a["frame"] <= after_frame or a["frame"] == 0:
                time.sleep(0.002); continue
            off = 4096 + (k * 2 + a["slot"]) * self.slot_bytes
            buf = bytes(self.m[off:off + self.slot_bytes])
            if self._cam(k)["seq"] == a["seq"]:
                return a, buf
        raise TimeoutError("no new frame from camera %d" % k)

    def latest(self, k, after_frame=-1, timeout=2.0):
        """Return (meta, uint16 mosaic) of the newest frame of camera k (newer than after_frame)."""
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            self._hdr()
            a = self._cam(k)
            if a["seq"] % 2 or a["frame"] <= after_frame or a["frame"] == 0:
                time.sleep(0.002); continue
            off = 4096 + (k * 2 + a["slot"]) * self.slot_bytes
            buf = bytes(self.m[off:off + self.slot_bytes])
            b = self._cam(k)
            if b["seq"] != a["seq"]:
                continue                              # writer touched it meanwhile; retry
            return a, self.unpack(buf)
        raise TimeoutError("no new frame from camera %d (is arcgrab running?)" % k)

    def unpack(self, buf):
        W, S = self.line_px, self.line_bytes
        H = min(len(buf) // S, 4096) - self.skip
        a = np.frombuffer(buf, np.uint8, count=(H + self.skip) * S).reshape(-1, S)[self.skip:, :W * 5 // 4]
        g = a.reshape(H, W // 4, 5).astype(np.uint16)
        lo = g[:, :, 4]
        return np.stack([(g[:, :, i] << 2) | ((lo >> (2 * i)) & 3) for i in range(4)], -1).reshape(H, W)

    def rgb(self, raw, half=True, wb=None):
        """BGGR demosaic. half=True: 2x2 binning (fast, half-res). Returns float32 BGR in 0..1."""
        x = np.clip(raw.astype(np.float32) - self.black, 0, None) / (1023 - self.black)
        if half:
            b = x[0::2, 0::2]; g = (x[0::2, 1::2] + x[1::2, 0::2]) / 2; r = x[1::2, 1::2]
            img = np.stack([b, g, r], -1)
        else:
            import cv2
            img = cv2.cvtColor((x * 65535).astype(np.uint16), cv2.COLOR_BayerBG2BGR).astype(np.float32) / 65535  # BGGR
        if wb is None:
            wb = img.reshape(-1, 3).mean(0); wb = wb.mean() / np.maximum(wb, 1e-6)
        return np.clip(img * wb, 0, 1)

    def gray(self, raw):
        """Half-res luminance (2x2 average) — good input for QR / feature detection."""
        x = np.clip(raw.astype(np.float32) - self.black, 0, None)
        return (x[0::2, 0::2] + x[0::2, 1::2] + x[1::2, 0::2] + x[1::2, 1::2]) / 4


if __name__ == "__main__":
    import sys, cv2
    s = ArcShm()
    for k, name in ((0, "left"), (1, "right")):
        meta, raw = s.latest(k)
        img = s.rgb(raw)
        out = (sys.argv[1] if len(sys.argv) > 1 else ".") + f"/shm_{name}.jpg"
        cv2.imwrite(out, (np.clip(img / np.percentile(img, 99.5), 0, 1) ** (1 / 2.2) * 255).astype(np.uint8))
        print(name, {k_: meta[k_] for k_ in ("frame", "ts", "exp_lines", "gain_code")}, raw.shape, "->", out)
