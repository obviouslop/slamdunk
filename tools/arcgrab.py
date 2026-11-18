#!/usr/bin/env python3
"""Grab raw stereo frames from the Arcturus Vision IMX616 pair via CAMSS RDI.

Cooperative with CameraSwitch: take /tmp/arcturus_camera_recorder.lock (CameraSwitch then
releases /tmp/passthrough_camera.lock), take the passthrough lock, capture, release both.
Media routes used (already present on this unit; verified 2026-11-17):
  arcimx616 0-001a -> csiphy3 -> csid0 -> vfe0_rdi0 -> /dev/video0
  arcimx616 0-0010 -> csiphy5 -> csid1 -> vfe1_rdi0 -> /dev/video3
Output: <out>/cam0.y10p, cam1.y10p (MIPI packed RAW10, 2464x2464) + meta.json

usage: arcgrab.py OUTDIR [--skip N] [--count N]
"""
import argparse, fcntl, json, os, subprocess, sys, time

DEVS = {"cam0": "/dev/video0", "cam1": "/dev/video3"}
SENSORS = {"cam0": "arcimx616 0-001a", "cam1": "arcimx616 0-0010"}   # csiphy3 / csiphy5


def subdev_for(name):
    import glob
    for p in glob.glob("/sys/class/video4linux/v4l-subdev*/name"):
        if open(p).read().strip() == name:
            return "/dev/" + p.split("/")[-2]
    raise FileNotFoundError(name)
W = H = 2464


def poke(subdev, reg, val):
    """arcimx616 'brightness' ctrl is a raw register write: (val << 16) | reg (see NOTES.md)."""
    subprocess.run(["v4l2-ctl", "-d", subdev, f"--set-ctrl=brightness={((val & 0xff) << 16) | (reg & 0xffff)}"], check=True)


def set_exposure(subdev, lines=None, gain=None):
    """Standard CCS/SMIA registers, wrapped in grouped parameter hold."""
    poke(subdev, 0x0104, 1)
    if lines is not None:
        lines = max(1, min(int(lines), 7348))
        poke(subdev, 0x0202, lines >> 8); poke(subdev, 0x0203, lines)
    if gain is not None:
        poke(subdev, 0x0204, gain >> 8); poke(subdev, 0x0205, gain)
    poke(subdev, 0x0104, 0)


def lock(path, timeout=10.0):
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    t0 = time.time()
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fd
        except BlockingIOError:
            if time.time() - t0 > timeout:
                os.close(fd)
                raise TimeoutError(f"could not lock {path}")
            time.sleep(0.1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--skip", type=int, default=20, help="frames to discard (AE settle)")
    ap.add_argument("--count", type=int, default=1)
    ap.add_argument("--exp-lines", type=int, help="coarse integration time in lines (frame = 7396 lines)")
    ap.add_argument("--gain", type=int, default=None, help="analog gain code (driver default 112)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rec = lock("/tmp/arcturus_camera_recorder.lock")
    pt = None
    try:
        pt = lock("/tmp/passthrough_camera.lock")
        if a.exp_lines is not None or a.gain is not None:
            for cam, sn in SENSORS.items():
                set_exposure(subdev_for(sn), a.exp_lines, a.gain)
        procs = {}
        for name, dev in DEVS.items():
            cmd = ["v4l2-ctl", "-d", dev,
                   f"--set-fmt-video=width={W},height={H},pixelformat=Y10P",
                   "--stream-mmap=4", f"--stream-skip={a.skip}", f"--stream-count={a.count}",
                   f"--stream-to={os.path.join(a.out, name + '.y10p')}"]
            procs[name] = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        t0 = time.monotonic()
        results = {}
        for name, p in procs.items():
            try:
                out, _ = p.communicate(timeout=30)
            except subprocess.TimeoutExpired:
                p.kill(); out, _ = p.communicate()
            results[name] = {"rc": p.returncode, "log": out[-400:]}
        fmt = subprocess.run(["v4l2-ctl", "-d", DEVS["cam0"], "--get-fmt-video"], capture_output=True, text=True).stdout
        meta = {"width": W, "height": H, "format": "Y10P", "skip": a.skip, "exp_lines": a.exp_lines, "gain": a.gain, "count": a.count,
                "elapsed_s": round(time.monotonic() - t0, 3), "results": results, "fmt": fmt,
                "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
        for name, r in results.items():
            size = os.path.getsize(os.path.join(a.out, name + ".y10p")) if os.path.exists(os.path.join(a.out, name + ".y10p")) else 0
            print(f"{name}: rc={r['rc']} bytes={size}")
            if r["rc"]:
                print(r["log"])
    finally:
        if pt is not None:
            fcntl.flock(pt, fcntl.LOCK_UN); os.close(pt)
        fcntl.flock(rec, fcntl.LOCK_UN); os.close(rec)


if __name__ == "__main__":
    sys.exit(main())
