#!/usr/bin/env bash
# One-command live point-cloud session on the Steam Frame:
#   src/arcgrab (Arcturus cameras + 500 Hz poses) -> tools/pcserver.py (stereo + fusion) -> viewer (Godot/OpenXR)
#
# usage: tools/pcsession.sh [--seconds N] [--replay RECDIR] [--scale S] [--no-viewer] [--gnss PORT] [--no-depth]
#   --replay RECDIR  feed a recording through /dev/shm/arcgrab instead of the cameras (no camera handover)
#   Recording (stereo pairs 1 fps, 500 Hz poses + controllers, composited view 2 fps -> captures/liveN) is the toolbox
#   toggle ● REC (tools/recorder.py); arcgrab always writes the session pose log $LOG/hmd_500hz.txt for it.
#   Point-cloud scanning starts OFF (toolbox ● POINTS). No start delay / sounds: if SteamVR shows the app as paused,
#   close the dashboard (or press "Resume").
#   --no-depth       do not submit the viewer's depth buffer (XR_KHR_composition_layer_depth). Default is to submit
#                    it: SteamVR then re-warps late frames with per-pixel depth, which keeps the cloud anchored when
#                    the head moves (without it the cloud drifts / scales vs. the real world; live14 A/B)
#   --gnss PORT      run tools/gnssd.py on a USB serial GNSS receiver (e.g. /dev/ttyACM0) for georeferencing
# Saved sessions (toolbox SAVE) land in captures/sessions/scan_*.{json,ply};
# export with tools/scene_export.py captures/sessions/scan_*.json  (-> .glb for Blender)
#
# Notes: every camera session hands the colour cameras over from XRService (short tracking blip at
# start/end). Controller tracking survived the last sessions, but if the laser/controllers stop
# tracking afterwards, restart SteamVR. See NOTES.md.
set -u
cd "$(dirname "$0")/.."
SECONDS_RUN=600; REPLAY=""; SCALE=0.35; VIEWER=1; GNSS=""; DEPTH=1
while [ $# -gt 0 ]; do
  case "$1" in
    --seconds) SECONDS_RUN=$2; shift 2;;
    --replay) REPLAY=$2; shift 2;;
    --record|--delay) echo "pcsession: $1 was removed (recording = toolbox ● REC; no start delay)" >&2; exit 1;;
    --scale) SCALE=$2; shift 2;;
    --no-viewer) VIEWER=0; shift;;
    --gnss) GNSS=$2; shift 2;;
    --depth) DEPTH=1; shift;;
    --no-depth) DEPTH=0; shift;;
    *) echo "unknown option $1"; exit 1;;
  esac
done
PY=/home/steamos/playground/.venv/bin/python
GODOT=/home/steamos/code/Godot_v4.7.2-stable_linux.arm64
LOG=captures/logs/pcsession; mkdir -p "$LOG"      # on disk (/tmp is RAM)
if [ $DEPTH = 1 ]; then printf '[xr]\nopenxr/submit_depth_buffer=true\n' > viewer/override.cfg; else rm -f viewer/override.cfg; fi
PIDS=()
cleanup() {
  trap - INT TERM EXIT
  for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done
  # give arcgrab time to hand the cameras back (restores sensor crop/exposure, releases locks)
  for i in $(seq 1 30); do alive=0; for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done; [ $alive = 0 ] && break; sleep 0.2; done
  echo "session ended; logs in $LOG"
}
trap cleanup INT TERM EXIT
rm -f /dev/shm/pointcloud.bin /dev/shm/pcctl.json /dev/shm/pchit.json /dev/shm/pcstatus.json /dev/shm/gnss.json /dev/shm/georef.json /dev/shm/pcpass.bin

if [ -n "$REPLAY" ]; then
  $PY -I tools/replay_shm.py "$REPLAY" --loop > "$LOG/replay.log" 2>&1 & PIDS+=($!)
else
  rm -f "$LOG/hmd_500hz.txt" "$LOG/hmd_500hz.txt.controllers"
  src/arcgrab --seconds "$((SECONDS_RUN + 10))" --max-exp 3000 --pose-log "$LOG/hmd_500hz.txt" 2> "$LOG/arcgrab.log" & PIDS+=($!)
fi
# no wait: pcserver retries until /dev/shm/arcgrab exists
nice -n 8 $PY -I tools/pcserver.py --scale "$SCALE" --pose-log "$LOG/hmd_500hz.txt" ${PCSERVER_ARGS:-} > "$LOG/pcserver.log" 2>&1 & PIDS+=($!)
if [ -n "$GNSS" ]; then
  $PY -I tools/gnssd.py --port "$GNSS" > "$LOG/gnssd.log" 2>&1 & PIDS+=($!)
fi
if [ $VIEWER = 1 ]; then
  timeout "$SECONDS_RUN" "$GODOT" --path viewer --xr-mode on > "$LOG/viewer.log" 2>&1 & VPID=$!; PIDS+=($VPID)
fi
echo "session running (${SECONDS_RUN}s max, Ctrl-C to stop); logs in $LOG"
if [ $VIEWER = 1 ]; then wait "$VPID"; else sleep "$SECONDS_RUN"; fi
