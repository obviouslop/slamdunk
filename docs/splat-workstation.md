# Gaussian splatting: protocol and workstation-agent prompt

## Protocol (Frame -> workstation -> Frame)

```
Frame (SteamOS, ARM, no CUDA)                     Workstation (24 GB RDNA3, no CUDA)
  pcsession.sh --record  -> captures/liveNN/        
  tools/export_colmap.py liveNN splat_liveNN \
        [--ply fused.ply]            -- COLMAP dataset (JPEG, ~0.3-0.6 MB/img)
  rsync -a splat_liveNN/  ws:splat/in/liveNN/  ---->  train (Brush) -> splat/out/liveNN/
  rsync -a ws:splat/out/liveNN/ captures/splats/liveNN/  <----  splat.ply + splat_meta.json + preview.mp4
```

1. **Capture**: a normal recording (`pcsession.sh record`, ~10 fps, both cameras, RAW10 + per-frame head pose + exposure,
   gain, GNSS if present). The auto exposure keeps running; its settings are stored per frame and normalised at export.
2. **Export (on the Frame)**: `tools/export_colmap.py RECDIR OUTDIR [--ply ...]`.
   Raw is 7.6 MB/image, JPEG ~0.5 MB, so only the export crosses the network.
3. **Dataset contract** (`OUTDIR`):
   - `images/{left,right}_NNNNN.jpg`: 1600x1600 pinhole, `--fov` 110 deg, sRGB, exposure-normalised, one global white balance.
     Pixels outside the fisheye circle are exactly (0,0,0): mask them (and the hand/controller, see below).
   - `sparse/0/cameras.txt` (PINHOLE; camera 1 = left cam, 2 = right cam, different poses/extrinsics but the same intrinsics model),
     `images.txt` (world-to-camera quaternion+translation, COLMAP/OpenCV axes: x right, y down, z forward),
     `points3D.txt` (initial points from the TSDF fusion; optional, subsampled, colours 0..255, track empty).
   - `export.json`: provenance (rec, fov, size, wb, scale, frame).
   - **World frame** = SteamVR STAGE: metres, y up, origin at the play-space origin. Keep it: the output must be in the
     same frame so it lines up with the live cloud, tags and GNSS georef (`tools/geo.py`).
4. **Return contract** (`splat/out/NAME/`): `splat.ply` (standard 3DGS PLY, unchanged coordinates, metres),
   `splat_meta.json` (trainer + version, iterations, #gaussians, held-out PSNR/SSIM/LPIPS, wall time, command line),
   `preview.mp4` (orbit or the training camera path), and `points.ply` (see task 4 below).
5. **Viewing on the Frame** comes later (a Godot splat renderer, ARM Mali/Adreno budget ~300k gaussians). The first use
   is `points.ply`: gaussian centres with SH-0 colours, opacity-filtered, in the format `tools/pcpub.py write_ply` makes
   (x,y,z float; red,green,blue uchar), which the existing viewer already displays.

Known data issues (the agent must plan for them)
- Poses are SteamVR poses with 10-20 mm drift and ~tens of ms timing error: enable pose refinement in the trainer, or do a
  short bundle adjustment first; expect soft results otherwise. The depth-fused `points3D` is accurate to ~1-3 cm at 1-2 m.
- Hands/controllers appear in many frames and move: mask them (e.g. skin/controller segmentation, or drop pixels whose
  depth is nearer than the static map) or the splat gets floaters in front of the cameras.
- Rolling shutter and motion blur exist (blurriest 25 % are already dropped at export).
- Auto exposure was normalised, but vignetting is not corrected; per-image appearance embeddings help if the trainer has them.
- Left and right images are from two cameras with their own pose each; treat them as two cameras (as exported).

## Prompt for the workstation agent

> You are implementing the workstation side of a Gaussian-splatting pipeline for a Steam Frame headset scanner
> (repo: this one, `pointcloud/`; read `docs/splat-workstation.md` and the "Gaussian splatting" part of `NOTES.md`).
> Environment: Linux workstation, 24 GB AMD RDNA3 GPU, **no CUDA**. Use **Brush** (Rust/WebGPU 3DGS trainer; Vulkan on AMD,
> no CUDA) as the trainer. Do not touch the headset side.
>
> Goal: given a COLMAP text dataset exported by `tools/export_colmap.py` (contract above), train a splat and return
> `splat.ply`, `splat_meta.json`, `preview.mp4`, `points.ply`.
>
> Work in this order and report after each step:
> 1. **Get Brush running**: build/install it (`cargo`; confirm the Vulkan device is the RDNA3 card, not llvmpipe), run its
>    bundled sample or a public COLMAP scene to prove training + the viewer work. Record versions and the exact commands.
>    If Brush cannot use the GPU, say so and propose OpenSplat (HIP/ROCm) or gsplat on ROCm; do not silently fall back to CPU.
> 2. **Dataset**: I will give you a dataset directory (from `export_colmap.py`; sample at `captures/scratch/colmap_t` on the
>    Frame, ask the user to rsync it to `~/splat/in/live20`). Check that Brush loads it: cameras/images/points3D parse, images
>    are found. Write `tools/splat/check_dataset.py`: verify every image listed exists, quaternions are unit, camera
>    centres are within ~20 m, project `points3D` into each image and report the fraction landing inside the image and, using
>    colour consistency of the projected points vs the pixel, a sanity score (a wrong pose convention shows up here first).
> 3. **Masks**: write `tools/splat/make_masks.py`: mask out the black fisheye region (pixels exactly 0,0,0 in a large connected
>    region touching the border) and, as a first version, hand/controller pixels (try a cheap approach first: skin-colour +
>    dark blob near the lower image area; justify any model you add). Output in the form Brush accepts (alpha-channel PNG or
>    its masks convention; check its docs/source).
> 4. **Training run**: `tools/splat/train.sh IN OUT` wrapping Brush with fixed, documented flags, holding out every 8th image
>    for evaluation, pose refinement ON if available, log the loss/PSNR. Then `tools/splat/eval.py` for held-out PSNR/SSIM
>    (LPIPS if easy). Produce `splat_meta.json`.
> 5. **Outputs**: `tools/splat/to_points.py splat.ply points.ply --min-opacity 0.5 --max-points 500000`: Gaussian means with
>    SH-0 colours (`0.5 + 0.2820948*f_dc`, clamp), opacity-filtered, written like `tools/pcpub.py:write_ply`. `preview.mp4`:
>    render a path through the training cameras (Brush export or its CLI; otherwise ffmpeg over rendered frames).
> 6. **Transport**: `tools/splat/sync.sh push|pull NAME` using rsync over ssh, host from `$FRAME_HOST` (user `steamos`),
>    directories `~/splat/in/NAME` and `captures/splats/NAME` on the Frame. No credentials in the repo.
>
> Constraints: Python 3 with `-I` for anything reading dataset files; keep large outputs out of /tmp (RAM) and out of git;
> all new code under `tools/splat/` and documented in a short `tools/splat/README.md`; commit small, working steps; if any
> contract detail seems wrong or missing (poses look inconsistent, axes, scale), report it with evidence instead of working
> around it silently. Hand-off report: numbers (held-out PSNR, training time, #gaussians, VRAM), commands, open problems.
