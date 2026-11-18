#!/usr/bin/env python3
"""Parse the Arcturus Vision recorder's timed-metadata track (recordingMode raw-dual-track-v1).

Each video frame has one JSON record with per-camera exposure/gain/intrinsics/midExpTs and
extrinsics (metres):
  avcam0InWorld    pose of av camera 0 (track 0 = "right") in the tracking world
  avcam1InAvcam0   pose of av camera 1 (track 1 = "left") in camera 0's frame
  avcam0InFrontcam0, headliveInFrontcam0, headofflineInHeadlive
Poses are given as plus_x / plus_z axis vectors + position of the child frame expressed in
the parent frame, i.e. X_parent = R @ X_child + t with R = [plus_x, plus_y, plus_z] columns.

usage: avmeta.py video.mp4 [out.npz]   (extracts stream with ffmpeg, writes arrays)
"""
import json, subprocess, sys, tempfile, os
import numpy as np


def pose(d):
    x = np.asarray(d["plus_x"], float); z = np.asarray(d["plus_z"], float)
    y = np.cross(z, x)
    R = np.stack([x, y, z], 1)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = d["position"]
    return T


def records_from_bytes(b):
    dec = json.JSONDecoder(); s = b.decode("latin-1"); out = []; i = 0
    while True:
        i = s.find('{"cameras"', i)
        if i < 0:
            return out
        obj, end = dec.raw_decode(s, i)
        out.append(obj); i = end


def extract(video):
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "meta.bin")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", video, "-map", "0:d:0", "-c", "copy", "-f", "data", p], check=True)
        return records_from_bytes(open(p, "rb").read())


def load(video):
    recs = extract(video)
    n = len(recs)
    W0 = np.stack([pose(r["extrinsics"]["avcam0InWorld"]) for r in recs])
    C10 = np.stack([pose(r["extrinsics"]["avcam1InAvcam0"]) for r in recs])
    pts = np.array([r["pts"] for r in recs])
    exp = np.array([[c["exposureMs"] for c in r["cameras"]] for r in recs])
    gain = np.array([[c["gain"] for c in r["cameras"]] for r in recs])
    cams = recs[0]["cameras"]
    kb = []
    for c in cams:
        k = [i for i in c["intrinsics"] if i["cameraModel"] == "kb"][0]
        kb.append(dict(name=c["sourceCamera"], track=c["track"], fx=k["fx"], fy=k["fy"], cx=k["cx"], cy=k["cy"],
                       D=[k["k1"], k["k2"], k["k3"], k["k4"]], w=k["width"], h=k["height"]))
    return dict(n=n, pts=pts, world_from_cam0=W0, cam0_from_cam1=C10, exposure_ms=exp, gain=gain, kb=kb, raw=recs)


if __name__ == "__main__":
    m = load(sys.argv[1])
    print(f"{m['n']} records, pts {m['pts'][0]:.3f}..{m['pts'][-1]:.3f}s")
    for k in m["kb"]:
        print(f"  track {k['track']} ({k['name']}): f=({k['fx']:.1f},{k['fy']:.1f}) c=({k['cx']:.1f},{k['cy']:.1f}) D={np.round(k['D'],4).tolist()}")
    C = m["cam0_from_cam1"][0]
    print("  cam0_from_cam1 t (mm):", np.round(C[:3, 3] * 1000, 2), " |t| %.2f mm" % (np.linalg.norm(C[:3, 3]) * 1000))
    print("  R:\n", np.round(C[:3, :3], 3))
    drift = np.linalg.norm(m["cam0_from_cam1"][:, :3, 3] - C[:3, 3], axis=1).max() * 1000
    print(f"  stereo extrinsic variation over clip: {drift:.3f} mm")
    P = m["world_from_cam0"][:, :3, 3]
    steps = np.linalg.norm(np.diff(P, axis=0), axis=1)
    print(f"  camera path: length {steps.sum():.2f} m, extent {np.round(P.max(0) - P.min(0), 2)} m, max step {steps.max()*1000:.1f} mm")
    print(f"  exposure {m['exposure_ms'].min():.1f}..{m['exposure_ms'].max():.1f} ms, gain {m['gain'].min():.2f}..{m['gain'].max():.2f}")
    if len(sys.argv) > 2:
        np.savez(sys.argv[2], pts=m["pts"], world_from_cam0=m["world_from_cam0"], cam0_from_cam1=m["cam0_from_cam1"],
                 exposure_ms=m["exposure_ms"], gain=m["gain"], kb=json.dumps(m["kb"]))
        print("  wrote", sys.argv[2])
