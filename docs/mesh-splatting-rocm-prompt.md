# Prompt: port mesh-splatting to ROCm (AMD RDNA3, no CUDA)

> You are porting `~/projects/mesh-splatting` (MeshSplatting, CVPR 2026: differentiable rendering with opaque meshes, PyTorch +
> custom CUDA) to run on an AMD Radeon RX 7900 XT (gfx1100, 24 GB) with ROCm (system ROCm in `/opt/rocm`, `hipcc` present).
> There is no NVIDIA GPU and no CUDA toolkit. Do not modify the upstream algorithm; the goal is the same results on HIP.
> Work in a branch of that repo (`git checkout -b rocm`), commit small working steps, keep upstream files readable
> (prefer `#ifdef __HIP_PLATFORM_AMD__` / build-script switches over rewriting). Do not touch `~/projects/slamdunk`.
>
> Known facts (already checked, verify rather than trust):
> - Submodules (already pulled) under `submodules/`:
>   - `diff-triangle-mesh-rasterization`: ~2.8k lines of CUDA (`cuda_rasterizer/{forward,backward,rasterizer_impl,adam,utils}.cu`,
>     `rasterize_points.cu`, `ext.cpp`). 3DGS-style: `cub::DeviceScan` / `cub::DeviceRadixSort`, `cooperative_groups`
>     (`this_grid().thread_rank()`, `this_thread_block()`), 16x16 blocks. No `__shfl*`/`__ballot*`/`warpSize` found.
>   - `simple-knn`: only `distCUDA2` (mean squared distance to the 3 nearest neighbours of each point), uses cub/thrust.
>     Acceptable to replace by a torch/scipy `cKDTree` implementation with the same signature and numbers.
>   - `effrdel`: pure C++/pybind11 (`rdel.lbvh_bind`) plus torch; no CUDA, should build as is.
> - Python uses `device="cuda"` / `torch.cuda.Event`; ROCm PyTorch maps these to HIP, so it should need no edits.
> - `compile.sh` builds the CUDA extensions; README targets Python 3.11, torch 2.7.1, CUDA 12.6.
> - The system default Python is 3.15, which many wheels lack: use `uv venv --python 3.11` (or 3.12) inside the repo dir or `~/splat/`-style
>   work dirs outside the repo, not the system Python.
>
> Steps, report after each:
> 1. **Torch on ROCm**: create the venv, install a ROCm build of PyTorch that matches the system ROCm (check `/opt/rocm/.info/version`
>    and https://pytorch.org for the matching `--index-url .../rocmX.Y`; if the pinned 2.7.1 has no matching wheel, pick the nearest
>    and say so). Prove: `torch.version.hip` is set, `torch.cuda.is_available()` is True, device name is the RX 7900 XT, a matmul runs
>    on it. If the system ROCm and the wheel disagree, tell me before working around it.
> 2. **Build `simple-knn`** (hipify or replace) and check it against a CPU reference (`scipy.spatial.cKDTree`) on random and on a real
>    point cloud: max relative error < 1e-5.
> 3. **Build `effrdel`** and run its own tests/examples if any.
> 4. **Port the rasterizer**: `hipify-perl` (or `hipify-clang`) on the `.cu/.cuh/.h/.cpp`, `cub` -> `hipcub`, set
>    `--offload-arch=gfx1100`, build through `setup.py` using `torch.utils.cpp_extension` (`CUDAExtension` works for HIP). Fix what the
>    compiler reports. Then **verify**: forward image vs a slow pure-PyTorch reference rasterizer on a tiny scene (a few triangles) and
>    gradient check (`torch.autograd.gradcheck` style or finite differences on the backward) for every differentiable input. Watch for
>    float `atomicAdd` ordering and 32-bit vs 64-bit wave assumptions; RDNA3 runs wave32 by default.
> 5. **Train**: run `train.py` on a small public scene (e.g. a Mip-NeRF 360 scene or the repo's example, at reduced resolution and
>    iterations first), then a full run if it works. Record: wall time, peak VRAM (`rocm-smi`), final PSNR/SSIM/LPIPS from `eval.py`,
>    number of triangles. If a CUDA reference number exists in the paper/README, compare and report the gap.
> 6. **Hand-off**: `ROCM.md` in the repo: exact install commands, versions (ROCm, torch, python, hipify), env vars, patches, known
>    issues, and a `rocm_setup.sh` that reproduces the environment. Commit.
>
> Constraints: keep large datasets/checkpoints/outputs out of `/tmp` (RAM) and out of git; the disk is ~90 % full, so check `df -h ~`
> before downloading and say if a dataset would not fit. Never fall back to CPU silently: if a kernel cannot be ported, say which,
> why, and propose the smallest replacement. Report evidence (commands, numbers, errors), not summaries.
