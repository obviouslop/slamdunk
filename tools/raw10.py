#!/usr/bin/env python3
"""Decode MIPI-packed RAW10 (Y10P) frames; helpers for previews. Import or run:
   raw10.py in.y10p out_prefix [W H STRIDE]"""
import sys, zlib, struct, numpy as np
def unpack(fn, W=2464, H=2464, stride=3088):
    b = np.fromfile(fn, dtype=np.uint8)[:stride*H].reshape(H, stride)[:, :W*5//4]
    g = b.reshape(H, W//4, 5).astype(np.uint16)
    lo = g[:, :, 4]
    px = np.stack([(g[:, :, i] << 2) | ((lo >> (2*i)) & 3) for i in range(4)], -1)
    return px.reshape(H, W)
def png(fn, img):
    img = np.ascontiguousarray(img.astype(np.uint8)); h, w = img.shape[:2]; c = 1 if img.ndim == 2 else 3
    raw = b''.join(b'\0'+img[y].tobytes() for y in range(h))
    ch = lambda t, d: struct.pack('>I', len(d))+t+d+struct.pack('>I', zlib.crc32(t+d))
    open(fn, 'wb').write(b'\x89PNG\r\n\x1a\n'+ch(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 0 if c == 1 else 2, 0, 0, 0))+ch(b'IDAT', zlib.compress(raw, 6))+ch(b'IEND', b''))
def bayer_half(raw, pattern="RGGB"):
    """2x2 binning demosaic to half-res RGB."""
    q = {pattern[0]: raw[0::2, 0::2], pattern[1]: raw[0::2, 1::2], pattern[3]: raw[1::2, 1::2]}
    g = (raw[0::2, 1::2].astype(np.float32) + raw[1::2, 0::2]) / 2
    return np.stack([q['R'].astype(np.float32), g, q['B'].astype(np.float32)], -1)
def tone(img, gamma=2.2):
    lo, hi = np.percentile(img, 1), np.percentile(img, 99.5)
    return (np.clip((img - lo) / (hi - lo + 1e-6), 0, 1) ** (1/gamma) * 255)
if __name__ == "__main__":
    r = unpack(sys.argv[1]); p = sys.argv[2]
    print("raw stats: min %d max %d mean %.1f; 2x2 phase means:" % (r.min(), r.max(), r.mean()),
          [round(float(r[i::2, j::2].mean()), 1) for i in (0, 1) for j in (0, 1)])
    png(p + "_mono.png", tone(r[::4, ::4].astype(np.float32)))
    crop = r[1100:1356, 1100:1356]; png(p + "_crop.png", tone(np.kron(crop.astype(np.float32), np.ones((2, 2)))))
    for pat in ("RGGB", "BGGR"):
        rgb = bayer_half(r, pat); rgb /= rgb.reshape(-1, 3).mean(0)  # gray-world WB
        png(f"{p}_{pat}.png", tone(rgb[::2, ::2]))
