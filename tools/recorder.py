#!/usr/bin/env python3
"""Session recording toggled from the viewer (toolbox ● REC), replacing pcsession.sh --record.

Start: next free captures/liveN; runs tools/liverec.py (stereo pairs 1 fps + full-rate poses/thumbnails) and an ffmpeg grab of the
composited headset view (/dev/video99, 2 fps -> composited/c_NNNN.jpg).
Stop: liverec / ffmpeg end cleanly (SIGTERM), then the recording's time range of the session's 500 Hz pose log (arcgrab
--pose-log, always on in pcsession.sh) is copied to hmd_500hz.txt (+ .controllers). Same layout as the old --record.
"""
import os, signal, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "captures")


def next_live_dir():
    n = 0
    for d in os.listdir(ROOT):
        if d.startswith("live") and d[4:].isdigit(): n = max(n, int(d[4:]) + 1)
    return os.path.join(ROOT, f"live{n}")


class Recorder:
    def __init__(self, pose_log=None, log_dir=None, fps=1.0):
        self.pose_log = pose_log; self.log_dir = log_dir or os.path.join(ROOT, "logs", "pcsession"); self.fps = fps
        self.dir = None; self.procs = []; self.t0 = None; self.last = ""; self.msg = ""

    @property
    def on(self):
        return self.dir is not None

    def set(self, want):
        if want and not self.on: self.start()
        elif not want and self.on: self.stop()

    def start(self):
        self.dir = next_live_dir(); os.makedirs(os.path.join(self.dir, "composited"), exist_ok=True)
        self.t0 = time.monotonic(); self.msg = ""
        py = sys.executable
        lr = open(os.path.join(self.log_dir, "liverec.log"), "a")
        self.procs = [subprocess.Popen(["nice", "-n", "8", py, "-I", os.path.join(HERE, "liverec.py"), self.dir, "--seconds", "1e9",
                                        "--fps", str(self.fps)], stdout=lr, stderr=subprocess.STDOUT)]
        try:
            self.procs.append(subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "v4l2", "-i", "/dev/video99", "-vf", "fps=2",
                                                "-q:v", "3", os.path.join(self.dir, "composited", "c_%04d.jpg")],
                                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=lr))
        except OSError as e:
            self.msg = f"no composited view ({e})"
        print(f"recording -> {self.dir}", flush=True)

    def stop(self):
        if not self.on: return
        t1 = time.monotonic()
        for p in self.procs:
            try: p.send_signal(signal.SIGTERM if "liverec.py" in " ".join(p.args) else signal.SIGINT)   # ffmpeg: SIGINT = finish file
            except OSError: pass
        for p in self.procs:
            try: p.wait(timeout=5)
            except subprocess.TimeoutExpired: p.kill()
        n = self._slice_poses(self.t0 - 1.0, t1 + 1.0)
        print(f"recording stopped: {self.dir} ({t1 - self.t0:.0f} s, {n} pose lines)", flush=True)
        self.last = self.dir; self.dir = None; self.procs = []

    def _slice_poses(self, ta, tb):
        """copy the [ta, tb] part (CLOCK_MONOTONIC s, first column) of the session pose logs into the recording"""
        n = 0
        if not self.pose_log: return 0
        for src, dst in ((self.pose_log, "hmd_500hz.txt"), (self.pose_log + ".controllers", "hmd_500hz.txt.controllers")):
            if not os.path.exists(src): continue
            with open(src) as f, open(os.path.join(self.dir, dst), "w") as g:
                for line in f:
                    try: t = float(line.split(None, 1)[0])
                    except (ValueError, IndexError): continue
                    if ta <= t <= tb: g.write(line); n += 1
        return n

    def hud(self):
        if self.on:
            s = int(time.monotonic() - self.t0)
            return f"● REC {s // 60}:{s % 60:02d} {os.path.basename(self.dir)}" + (f" ({self.msg})" if self.msg else "")
        return f"recorded {os.path.basename(self.last)}" if self.last else ""
