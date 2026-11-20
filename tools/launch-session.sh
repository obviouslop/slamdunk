#!/usr/bin/env bash
# Launcher for the .desktop files (no terminal): optional GNSS bridge (Glasgow) + live point-cloud session.
# usage: launch-session.sh [record]      everything is logged to captures/logs/launcher.log
cd "$(dirname "$0")/.."
mkdir -p captures/logs; exec >> captures/logs/launcher.log 2>&1; echo "== $(date) launch-session $*"
ARGS=(--delay 10); GNSS=()
[ "${1:-}" = record ] && ARGS+=(--record)
if [ -x "${GLASGOW:-$HOME/.local/bin/glasgow}" ] && ! pgrep -f "[g]lasgow.* run uart" > /dev/null; then
  setsid ./start-gps.sh > captures/logs/gps.log 2>&1 &
  for _ in $(seq 1 150); do [ -e /dev/shm/gps-nmea ] && break; sleep 0.2; done
  if [ -e /dev/shm/gps-nmea ]; then GNSS=(--gnss /dev/shm/gps-nmea); echo "GNSS on $(readlink /dev/shm/gps-nmea)"
  else echo "no GNSS (see captures/logs/gps.log); continuing without"; fi
fi
tools/pcsession.sh "${ARGS[@]}" "${GNSS[@]}"
pkill -f "[s]tart-gps.sh"; pkill -f "[g]lasgow.* run uart"; pkill -x gpsd
