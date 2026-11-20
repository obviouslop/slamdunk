#!/usr/bin/env bash
# Glasgow UART -> PTY bridge for the GNSS receiver. Prints the PTY path and keeps a stable symlink /dev/shm/gps-nmea to it,
# so tools/gnssd.py (pcsession.sh --gnss /dev/shm/gps-nmea) does not need to know the /dev/pts/N number.
# usage: ./start-gps.sh [--gpsd]     (--gpsd additionally runs gpsd on the PTY, needs sudo; then gnssd must NOT read the PTY too)
cd "$(dirname "$0")"
GLASGOW=${GLASGOW:-$HOME/.local/bin/glasgow}
LINK=/dev/shm/gps-nmea; LOG=captures/logs/glasgow.log; mkdir -p captures/logs
before=$(ls /dev/pts 2>/dev/null | sort)
"$GLASGOW" -v run uart -V 3.3 --rx A1 --tx A2 -b 38400 pty > "$LOG" 2>&1 & GPID=$!
trap 'kill $GPID 2>/dev/null; rm -f "$LINK"' EXIT INT TERM
pty=""
for _ in $(seq 1 150); do                                  # up to 30 s (FPGA bitstream load + applet start)
  kill -0 $GPID 2>/dev/null || { echo "glasgow exited:"; tail -20 "$LOG"; exit 1; }
  pty=$(grep -o '/dev/pts/[0-9]\+' "$LOG" | tail -1)       # 1. glasgow logs the PTY path
  if [ -z "$pty" ]; then                                   # 2. fallback: the newly appeared /dev/pts entry
    n=$(comm -13 <(echo "$before") <(ls /dev/pts | sort) | grep -x '[0-9]\+' | head -1); [ -n "$n" ] && pty=/dev/pts/$n
  fi
  [ -n "$pty" ] && break; sleep 0.2
done
[ -n "$pty" ] || { echo "no PTY found, see $LOG"; exit 1; }
ln -sf "$pty" "$LINK"; echo "GNSS PTY: $pty  (symlink $LINK)"
if [ "${1:-}" = --gpsd ]; then sudo gpsd -N -n -b -D 3 "$pty"; else wait $GPID; fi
