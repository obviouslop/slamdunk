#!/usr/bin/env python3
"""Pipeline trace page for a debug capture: DIR/index.html + DIR/preview/*.jpg, built from the capture directory alone
(pipeline.npz, pipeline.json, meta.json, left.y10p). Written by tools/debugcap.py after every capture; regenerate with

    debugreport.py captures/stills/<capture> [...]

The page is self-contained for offline viewing (relative images, system fonts, light/dark).
"""
import html, json, os, sys
import numpy as np
import cv2

DARK = (24, 20, 20)


def _w(pv, name, im, size=560, q=82, png=False):
    if im.ndim == 2: im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    h, w = im.shape[:2]; f = size / max(h, w)
    im = cv2.resize(im, (max(1, int(w * f)), max(1, int(h * f))), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_NEAREST)
    ext = ".png" if png else ".jpg"                   # PNG where per-pixel colour matters (JPEG subsamples chroma 2x2)
    cv2.imwrite(os.path.join(pv, name + ext), im, [] if png else [cv2.IMWRITE_JPEG_QUALITY, q])
    return f"preview/{name}{ext}"


def _turbo(x, lo, hi, valid, inv=False):
    v = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1); v = 1 - v if inv else v
    im = cv2.applyColorMap((255 * v).astype(np.uint8), cv2.COLORMAP_TURBO); im[~valid] = DARK; return im


def _raw_left(d, meta):
    raw = meta.get("raw", {}); W = raw.get("width", 2464); S = raw.get("line_bytes", 3088); skip = raw.get("skip_lines", 0)
    b = np.frombuffer(open(os.path.join(d, "left.y10p"), "rb").read(), np.uint8)[skip * S:]
    H = min(len(b) // S, W)
    p = b[:H * S].reshape(H, S)[:, :W * 5 // 4].reshape(H, W // 4, 5).astype(np.uint16)
    lo = p[:, :, 4]
    return np.stack([(p[:, :, i] << 2) | ((lo >> (2 * i)) & 3) for i in range(4)], -1).reshape(H, W)   # full 10-bit samples


def _bayer_rgb(raw10):
    """RGGB mosaic -> BGR image where every sensor pixel shows only its own colour, brightness = 10-bit value / 4 (no gain)"""
    v = (raw10 >> 2).astype(np.uint8); out = np.zeros(raw10.shape + (3,), np.uint8)
    out[0::2, 0::2, 2] = v[0::2, 0::2]                                  # R
    out[0::2, 1::2, 1] = v[0::2, 1::2]; out[1::2, 0::2, 1] = v[1::2, 0::2]   # G
    out[1::2, 1::2, 0] = v[1::2, 1::2]                                  # B
    return out


def build(d):
    z = np.load(os.path.join(d, "pipeline.npz")); A = {k: z[k] for k in z.files}
    man = json.load(open(os.path.join(d, "pipeline.json")))
    meta = json.load(open(os.path.join(d, "meta.json"))) if os.path.exists(os.path.join(d, "meta.json")) else {}
    pv = os.path.join(d, "preview"); os.makedirs(pv, exist_ok=True)
    P = man["params"]; sg = P["sgbm"]; T = man.get("timings_ms", {}); G = man.get("blur_gate", {})
    far = float(man.get("far_m", 4.0)); mn = float(P["min_depth"])
    img = {}
    # 1 raw
    try:
        raw = _raw_left(d, meta); c = (raw.shape[0] // 2) & ~1                 # even offset: crop starts on an R pixel
        rf = raw.astype(np.float32)
        img["raw_full"] = _w(pv, "raw_full", np.clip(255 * rf / max(np.percentile(rf, 99.5), 1), 0, 255).astype(np.uint8))
        img["raw_crop"] = _w(pv, "raw_crop", _bayer_rgb(raw[c - 48:c + 48, c - 48:c + 48]), size=288, png=True)
    except Exception:
        pass
    # 2 binned
    wb = A["wb"].astype(np.float32)
    def bgr(x):
        x = x * wb[None, None, :]; return np.clip(255 * (x / max(np.percentile(x, 99.5), 1e-6)) ** (1 / 2.2), 0, 255).astype(np.uint8)
    img["binned_L"] = _w(pv, "binned_L", bgr(A["binned_L"])); img["binned_R"] = _w(pv, "binned_R", bgr(A["binned_R"]))
    # 3 grey
    img["grey_L"] = _w(pv, "grey_L", A["grey_L"]); img["grey_R"] = _w(pv, "grey_R", A["grey_R"])
    gl = A["grey_L"]; gmed = float(np.median(gl[gl > 0])) if (gl > 0).any() else 0.0
    tone = str(A["tone"]) if "tone" in A else "linear"
    # 4 rectify
    mp = A["remap_L"].astype(np.float32); src = max(A["binned_L"].shape[:2])
    ok = (mp[..., 0] >= 0) & (mp[..., 0] < src) & (mp[..., 1] >= 0) & (mp[..., 1] < src)
    grid = np.zeros(mp.shape[:2] + (3,), np.uint8)
    grid[..., 1] = np.where(((mp[..., 0] // 64).astype(int) + (mp[..., 1] // 64).astype(int)) % 2 == 0, 150, 60)
    grid[..., 2] = (np.clip(mp[..., 0] / src, 0, 1) * 255).astype(np.uint8); grid[..., 0] = (np.clip(mp[..., 1] / src, 0, 1) * 255).astype(np.uint8)
    grid[~ok] = DARK; img["remap_L"] = _w(pv, "remap_L", grid)
    for n in ("rect_grey_L", "rect_grey_R"):
        im = cv2.cvtColor(A[n], cv2.COLOR_GRAY2BGR)
        for y in range(40, im.shape[0], 70): cv2.line(im, (0, y), (im.shape[1], y), (60, 170, 255), 1)
        img[n] = _w(pv, n, im)
    rect_valid = float((A["rect_grey_L"] > 0).mean())
    # 5 matcher input
    img["matcher_in_L"] = _w(pv, "matcher_in_L", A["matcher_in_L"]); img["matcher_in_R"] = _w(pv, "matcher_in_R", A["matcher_in_R"])
    # 6 SGBM
    mo = A["matcher_out_int16"]; okm = mo > 0
    img["matcher_out"] = _w(pv, "matcher_out", _turbo(mo.astype(np.float32), 0, float(np.percentile(mo[okm], 99)) if okm.any() else 1, okm))
    dp = A["disp"]; okd = dp > 0; dhi = float(np.percentile(dp[okd], 99)) if okd.any() else 1.0
    img["disp"] = _w(pv, "disp", _turbo(dp, 0, dhi, okd))
    removed = float(((A["disp_sgbm"] > 0) & ~okd).mean())
    # 7 depth
    Z = A["Z"]; okz = Z > 0
    img["Z"] = _w(pv, "Z", _turbo(Z, mn, far, okz, inv=True)); img["rgb"] = _w(pv, "rgb", A["rgb"][..., ::-1])
    if "conf" in A:
        c = A["conf"]; im = cv2.applyColorMap((255 * np.clip(c, 0, 1)).astype(np.uint8), cv2.COLORMAP_VIRIDIS); im[~okz] = DARK
        img["conf"] = _w(pv, "conf", im); conf_med = float(np.median(c[okz])) if okz.any() else 0.0
    else:
        conf_med = None
    zq = np.percentile(Z[okz], [5, 50, 95]) if okz.any() else [0, 0, 0]
    # 8 masks / fused
    hm = A.get("hand_mask", np.zeros(Z.shape, bool)); cm = A.get("cone_mask", np.ones(Z.shape, bool))
    im = cv2.cvtColor(A["rect_grey_L"], cv2.COLOR_GRAY2BGR) // 2; im[hm] = (80, 80, 255); im[~cm] = (255, 120, 0)
    img["masks"] = _w(pv, "masks", im)
    Zf = A.get("Z_fused", Z); img["Z_fused"] = _w(pv, "Z_fused", _turbo(Zf, mn, far, Zf > 0, inv=True))

    # ---------------------------------------------------------------- findings (automatic checks)
    findings = []
    if gmed < 16:
        findings.append(("bad", "input", f"The matcher input is very dark: median grey level {gmed:.0f} of 255 (tone curve <code>{tone}</code>). "
                         "Something bright (a lamp, a window) sets the gain; detail in the dark end is lost."))
    if conf_med is not None and conf_med < 0.05:
        findings.append(("warn", "confidence", f"Confidence is near 0 for most valid pixels (median {conf_med:.2f}); it only affects fusion with --conf."))
    if G:
        b = G.get("blur_px", 0.0)
        findings.append(("ok" if G.get("sharp", True) else "bad", "blur gate",
                         f"Estimated motion blur {b:.2f} px (ω {np.degrees(G.get('omega_rad_s', 0)):.1f}°/s), gate {G.get('threshold_px')} px: "
                         f"{'sharp' if G.get('sharp', True) else 'BLURRED (not fused)'}. Skipped so far this session: {G.get('skipped_so_far', 0)}."))
    if not man.get("fused", True):
        findings.append(("warn", "fusion", "This pair was not fused (scanning paused or blurred); buffers show what the pipeline computed."))
    if float(okz.mean()) < 0.25:
        findings.append(("warn", "depth", f"Only {100 * okz.mean():.0f} % of pixels have depth."))

    # ---------------------------------------------------------------- page
    L = meta.get("left", {}); e = lambda s: html.escape(str(s))
    exp_ms = L.get("exp_lines", 0) * 4.36e-6 * 1000
    def fig(key, cap):
        return f'<figure><img src="{img[key]}" alt="{e(cap)}" loading="lazy"><figcaption>{cap}</figcaption></figure>' if key in img else ""
    def stage(n, title, t, text, bufs, figs, extra=""):
        return (f'<article class="stage"><div class="num">{n}</div><div class="card"><div class="head"><h2>{title}</h2><span class="time">{t}</span></div>'
                f'<p>{text}</p><ul class="bufs">{"".join(f"<li>{b}</li>" for b in bufs)}</ul><div class="imgs">{"".join(figs)}</div>{extra}</div></article>')
    tb, td, tf = T.get("bin_raw10", 0), T.get("depth", 0), T.get("handmask_far_integrate", 0); tt = max(tb + td + tf, 1e-6)
    name = os.path.basename(os.path.normpath(d))
    when = name[:8] + " " + name[9:11] + ":" + name[11:13] + ":" + name[13:15] if len(name) >= 15 else name
    S = []
    S.append(stage(1, "Sensor: packed RAW10", "arcgrab → /dev/shm",
                   f"{e(meta.get('raw', {}).get('width', 2464))}² RGGB mosaic per camera, 5 bytes per 4 pixels. The crop shows the Bayer pattern at the centre.",
                   ["<b>left.y10p</b>", "<b>right.y10p</b>"], [fig("raw_full", "left mosaic, sensor orientation"), fig("raw_crop", "96 × 96 px centre crop, each pixel in its filter colour, 10-bit / 4")]))
    S.append(stage(2, "bin_raw10: 2×2 binning demosaic", f"{tb:.1f} ms",
                   "Each RGGB cell becomes one BGR pixel, black level removed, float32. Shown white-balanced with gamma for viewing.",
                   [f"<b>binned_L/R</b> {A['binned_L'].shape[0]}² ×3 f32", "<b>wb</b> [" + ", ".join(f"{x:.2f}" for x in wb) + "]"],
                   [fig("binned_L", "binned_L"), fig("binned_R", "binned_R")]))
    S.append(stage(3, "Grey + tone curve to 8 bit", "part of depth",
                   f"BGR → grey, x = grey / p99.5, tone curve <code>{tone}</code> → uint8. Median grey level of the matcher input: <b>{gmed:.0f}</b>.",
                   [f"<b>grey_L/R</b> {A['grey_L'].shape[0]}² u8", f"<b>grey_gain</b> {float(np.ravel(A['grey_gain'])[0]):.3f}"],
                   [fig("grey_L", "grey_L (as fed on)"), fig("grey_R", "grey_R")]))
    S.append(stage(4, "Rectify (fisheye remap)", "part of depth",
                   f"Factory calibration, fisheye stereoRectify, {P['scale_size'][0]}² output. Orange lines are rows: a scene point sits on the same row in both images. {100 * rect_valid:.0f} % of the output has source pixels.",
                   ["<b>remap_L/R</b> i16 + frac u16", "<b>rect_grey_L/R</b> u8"],
                   [fig("remap_L", "remap_L: source x (red), y (blue), 64 px checker"), fig("rect_grey_L", "rect_grey_L"), fig("rect_grey_R", "rect_grey_R")]))
    S.append(stage(5, "Matcher input" + (" (mirrored)" if not P["left_first"] else ""), "part of depth",
                   "Rectification puts camL on the right of the pair, so both images are mirrored before SGBM and the result is mirrored back." if not P["left_first"]
                   else "camL is the left image of the rectified pair; no mirroring.",
                   ["<b>matcher_in_L/R</b> u8"], [fig("matcher_in_L", "matcher_in_L"), fig("matcher_in_R", "matcher_in_R")]))
    S.append(stage(6, "SGBM → disparity", "part of depth",
                   f"SGBM mode {sg['mode']}: {sg['numDisparities']} disparities, block {sg['blockSize']}, P1 {sg['P1']}, P2 {sg['P2']}, uniqueness {sg['uniquenessRatio']}, "
                   f"speckle {sg['speckleWindowSize']}/{sg['speckleRange']}, LR check {sg['disp12MaxDiff']} px. Valid for <b>{100 * okd.mean():.0f} %</b> of pixels"
                   + (f"; the median test removed {100 * removed:.1f} %." if P.get("median") else "; median test off."),
                   ["<b>matcher_out_int16</b> 1/16 px", "<b>disp_sgbm</b>, <b>disp</b> f32"],
                   [fig("matcher_out", "matcher_out_int16 (matcher orientation)"), fig("disp", f"disp, 0–{dhi:.0f} px")]))
    S.append(stage(7, "Depth + colour", f"{td:.1f} ms (stages 3–7)",
                   f"Z = {P['k']:.3f}·f·B / (d + {P['d0_px']:.3f}), f {P['f']:.1f} px, B {1000 * P['baseline_m']:.1f} mm, kept in {mn}–{far} m: <b>{100 * okz.mean():.0f} %</b> of pixels. "
                   f"Median {zq[1]:.2f} m, 5–95 % {zq[0]:.2f}–{zq[2]:.2f} m.",
                   ["<b>Z</b> f32 m", "<b>rgb</b>, <b>rgb_R</b> u8"],
                   [fig("Z", f"Z, {mn} m (red) → {far:g} m (blue)"), fig("rgb", "rgb (rectified camL)"), fig("conf", "conf (viridis)")]))
    S.append(stage(8, "Masks → TSDF integrate", f"{tf:.1f} ms",
                   f"Hand/controller mask {100 * hm.mean():.1f} %, outside the paint cone {100 * (~cm).mean():.1f} %. "
                   + ("Z_fused went into TSDF.integrate." if man.get("fused", True) else "Not fused."),
                   ["<b>hand_mask</b>, <b>cone_mask</b> bool", "<b>Z_fused</b> f32", "<b>stage_from_rect_*</b> 4×4"],
                   [fig("masks", "hand (red) / outside cone (orange)"), fig("Z_fused", "Z_fused → TSDF.integrate")]))
    F = "".join(f'<div class="finding"><span class="tag {k}">{e(t)}</span><p>{x}</p></div>' for k, t, x in findings)
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Depth pipeline trace {e(when)}</title><style>{CSS}</style></head><body><div class="wrap">
<h1>Depth pipeline trace, {e(when)}</h1>
<p class="sub">Every buffer pcserver produced for one stereo pair, in processing order. Rendered from <code>pipeline.npz</code> in this directory.</p>
<ul class="facts"><li>frame <b>{e(man.get('frame_left'))}</b></li><li>exposure <b>{e(L.get('exp_lines'))} lines = {exp_ms:.1f} ms</b></li>
<li>analog gain code <b>{e(L.get('gain_code'))}</b></li><li>tone <b>{e(tone)}</b></li><li>scan <b>{e(man.get('scan'))}</b></li></ul>
<section class="findings">{F}</section>
<h2 style="margin:26px 0 10px">Time per stage (ms, on the Frame)</h2>
<div class="bar"><span style="width:{100 * tb / tt:.1f}%;background:var(--muted)"></span><span style="width:{100 * td / tt:.1f}%;background:var(--accent)"></span><span style="width:{100 * tf / tt:.1f}%;background:var(--warn)"></span></div>
<div class="legend"><span><i class="sw" style="background:var(--muted)"></i>bin_raw10 {tb:.1f}</span><span><i class="sw" style="background:var(--accent)"></i>grey + rectify + SGBM + depth {td:.1f}</span>
<span><i class="sw" style="background:var(--warn)"></i>masks + far + TSDF {tf:.1f}</span><span>write: still {T.get('write_still', 0):.0f}, pipeline {T.get('write_pipeline', 0):.0f}</span></div>
<div class="flow">{''.join(S)}</div>
<footer>Buffers and their meaning: <code>pipeline.json</code>. Regenerate: <code>tools/debugreport.py {e(d)}</code></footer></div></body></html>"""
    with open(os.path.join(d, "index.html"), "w") as f:
        f.write(page)
    return os.path.join(d, "index.html")


CSS = """
:root{--bg:#f3f4f1;--panel:#fff;--ink:#1b1f1d;--muted:#5d6662;--line:#d4d9d5;--accent:#0a7d6e;--warn:#b8560f;--bad:#b3261e;--img:#141716;
--display:"Archivo","Helvetica Neue",Arial,sans-serif;--body:"IBM Plex Sans","Segoe UI",system-ui,sans-serif;--mono:"IBM Plex Mono",ui-monospace,Menlo,monospace}
@media (prefers-color-scheme:dark){:root{--bg:#121614;--panel:#1a1f1d;--ink:#e4e8e5;--muted:#98a39e;--line:#2c3431;--accent:#3cc2ac;--warn:#f0954c;--bad:#ff7a70;--img:#0b0d0c;color-scheme:dark}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 var(--body)}
.wrap{max-width:1080px;margin:0 auto;padding:28px max(16px,env(safe-area-inset-right,0px)) 64px max(16px,env(safe-area-inset-left,0px))}
h1{font:800 clamp(24px,4vw,38px)/1.1 var(--display);margin:0;text-wrap:balance}h2{font:700 19px/1.25 var(--display);margin:0}
code{font-family:var(--mono);font-size:12.5px}.sub{color:var(--muted);margin:8px 0 0;max-width:70ch}
.facts{display:flex;flex-wrap:wrap;gap:6px 18px;margin:18px 0 0;padding:0;list-style:none;font:12.5px/1.4 var(--mono);color:var(--muted)}.facts b{color:var(--ink);font-weight:500}
.findings{margin:26px 0 8px;display:grid;gap:10px}.finding{display:grid;grid-template-columns:auto 1fr;gap:4px 12px;padding:12px 14px;background:var(--panel);border:1px solid var(--line);border-radius:8px}
.finding p{margin:0;min-width:0}.tag{font:500 11px/1 var(--mono);letter-spacing:.06em;text-transform:uppercase;padding:5px 7px;border-radius:4px;align-self:start;border:1px solid currentColor}
.tag.bad{color:var(--bad)}.tag.warn{color:var(--warn)}.tag.ok{color:var(--accent)}
.bar{display:flex;height:14px;border-radius:4px;overflow:hidden;border:1px solid var(--line)}.bar span{display:block;height:100%}
.legend{display:flex;flex-wrap:wrap;gap:6px 16px;margin-top:8px;font:12px/1.4 var(--mono);color:var(--muted)}.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px}
.flow{position:relative;margin-top:30px;display:grid;gap:34px}.flow::before{content:"";position:absolute;left:19px;top:8px;bottom:8px;width:2px;background:var(--line)}
.stage{position:relative;display:grid;grid-template-columns:40px 1fr;gap:0 14px}
.stage+.stage::after{content:"";position:absolute;left:14px;top:-22px;border:6px solid transparent;border-top:9px solid var(--accent)}
.num{width:40px;height:40px;border-radius:50%;display:grid;place-items:center;background:var(--panel);border:2px solid var(--accent);font:700 14px/1 var(--display);color:var(--accent);position:relative;z-index:1}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px;min-width:0;display:grid;gap:12px}
.head{display:flex;flex-wrap:wrap;align-items:baseline;gap:6px 14px;justify-content:space-between}.time{font:500 12.5px/1 var(--mono);color:var(--accent);white-space:nowrap}
.card p{margin:0;max-width:72ch}.bufs{margin:0;padding:0;list-style:none;display:flex;flex-wrap:wrap;gap:4px 14px;font:12px/1.5 var(--mono);color:var(--muted)}.bufs b{color:var(--ink);font-weight:500}
.imgs{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px}figure{margin:0;display:grid;gap:6px;min-width:0}
figure img{width:100%;aspect-ratio:1;object-fit:contain;background:var(--img);border-radius:6px;display:block}figcaption{font:12px/1.4 var(--mono);color:var(--muted)}
footer{margin-top:40px;color:var(--muted);font-size:13px}
@media (max-width:560px){.stage{grid-template-columns:30px 1fr;gap:0 10px}.num{width:30px;height:30px;font-size:12px}.flow::before{left:14px}.stage+.stage::after{left:9px}}
"""

if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(build(d))
