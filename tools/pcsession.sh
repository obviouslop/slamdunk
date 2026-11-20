#!/usr/bin/env bash
# One-command live point-cloud session on the Steam Frame:
#   src/arcgrab (Arcturus cameras + 500 Hz poses) -> tools/pcserver.py (stereo + fusion) -> viewer (Godot/OpenXR)
#
# usage: tools/pcsession.sh [--seconds N] [--replay RECDIR] [--record [OUTDIR]] [--scale S] [--no-viewer] [--gnss PORT] [--delay S] [--no-depth]
#   --replay RECDIR  feed a recording through /dev/shm/arcgrab instead of the cameras (no camera handover)
#   --record [OUTDIR]  also save stereo pairs (1 fps), 500 Hz poses (+ controllers) and the composited headset view
#                    (2 fps); without OUTDIR the next free captures/liveN is used
#   --delay S        wait S seconds (beeping) before the viewer starts: close the SteamVR dashboard meanwhile,
#                    otherwise SteamVR shows the app as paused until you press "Resume"
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
SECONDS_RUN=600; REPLAY=""; RECORD=""; SCALE=0.35; VIEWER=1; GNSS=""; DELAY=0; DEPTH=1
while [ $# -gt 0 ]; do
  case "$1" in
    --seconds) SECONDS_RUN=$2; shift 2;;
    --replay) REPLAY=$2; shift 2;;
    --record)                          # --record [OUTDIR]: without a directory -> next free captures/liveN
      if [ $# -ge 2 ] && [ "${2#--}" = "$2" ] && [ "$2" != auto ]; then RECORD=$2; shift 2
      else
        n=0; for d in captures/live[0-9]*; do [ -d "$d" ] || continue; k=${d#captures/live}; [ "$k" -eq "$k" ] 2>/dev/null && [ "$k" -ge "$n" ] && n=$((k + 1)); done
        RECORD=captures/live$n; shift; [ "${1:-}" = auto ] && shift
      fi
      echo "recording to $RECORD";;
    --scale) SCALE=$2; shift 2;;
    --no-viewer) VIEWER=0; shift;;
    --gnss) GNSS=$2; shift 2;;
    --delay) DELAY=$2; shift 2;;
    --depth) DEPTH=1; shift;;
    --no-depth) DEPTH=0; shift;;
    *) echo "unknown option $1"; exit 1;;
  esac
done
PY=/home/steamos/playground/.venv/bin/python
GODOT=/home/steamos/code/Godot_v4.7.2-stable_linux.arm64
if [ -n "$RECORD" ] && { [ -e "$RECORD/frames.jsonl" ] || [ -e "$RECORD/hmd_500hz.txt" ]; }; then
  echo "pcsession: $RECORD already contains a recording - choose a new directory (nothing was started)" >&2; exit 1
fi
LOG=captures/logs/pcsession; mkdir -p "$LOG"      # on disk (/tmp is RAM)
if [ $DEPTH = 1 ]; then printf '[xr]\nopenxr/submit_depth_buffer=true\n' > viewer/override.cfg; else rm -f viewer/override.cfg; fi
PIDS=()
cleanup() {
  trap - INT TERM EXIT
  for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done
  # give arcgrab time to hand the cameras back (restores sensor crop/exposure, releases locks)
  for i in $(seq 1 30); do alive=0; for p in "${PIDS[@]}"; do kill -0 "$p" 2>/dev/null && alive=1; done; [ $alive = 0 ] && break; sleep 0.2; done
  pw-play tools/sounds/done.wav 2>/dev/null
  echo "session ended; logs in $LOG"
}
trap cleanup INT TERM EXIT
rm -f /dev/shm/pointcloud.bin /dev/shm/pcctl.json /dev/shm/pchit.json /dev/shm/pcstatus.json /dev/shm/gnss.json /dev/shm/georef.json /dev/shm/pcpass.bin

if [ -n "$REPLAY" ]; then
  $PY -I tools/replay_shm.py "$REPLAY" --loop > "$LOG/replay.log" 2>&1 & PIDS+=($!)
else
  POSELOG=()
  [ -n "$RECORD" ] && { mkdir -p "$RECORD"; POSELOG=(--pose-log "$RECORD/hmd_500hz.txt"); }
  src/arcgrab --seconds "$((SECONDS_RUN + 10))" --max-exp 3000 "${POSELOG[@]}" 2> "$LOG/arcgrab.log" & PIDS+=($!)
fi
sleep 3
nice -n 8 $PY -I tools/pcserver.py --scale "$SCALE" ${PCSERVER_ARGS:-} > "$LOG/pcserver.log" 2>&1 & PIDS+=($!)
if [ -n "$GNSS" ]; then
  $PY -I tools/gnssd.py --port "$GNSS" > "$LOG/gnssd.log" 2>&1 & PIDS+=($!)
fi
if [ $VIEWER = 1 ]; then
  for i in $(seq 1 "$DELAY"); do pw-play tools/sounds/beep.wav 2>/dev/null & sleep 1; done
  timeout "$SECONDS_RUN" "$GODOT" --path viewer --xr-mode on > "$LOG/viewer.log" 2>&1 & VPID=$!; PIDS+=($VPID)
fi
sleep 5; pw-play tools/sounds/start.wav 2>/dev/null &
if [ -n "$RECORD" ] && [ -z "$REPLAY" ]; then
  mkdir -p "$RECORD/composited"
  ffmpeg -hide_banner -loglevel error -f v4l2 -i /dev/video99 -t "$SECONDS_RUN" -vf fps=2 -q:v 3 "$RECORD/composited/c_%04d.jpg" & PIDS+=($!)
  nice -n 8 $PY -I tools/liverec.py "$RECORD" --seconds "$SECONDS_RUN" --fps 1 > "$LOG/liverec.log" 2>&1 & PIDS+=($!)
fi
echo "session running (${SECONDS_RUN}s max, Ctrl-C to stop); logs in $LOG"
if [ $VIEWER = 1 ]; then wait "$VPID"; else sleep "$SECONDS_RUN"; fi
