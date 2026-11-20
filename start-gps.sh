#!/usr/bin/env bash
# Glasgow UART -> PTY bridge for the GNSS receiver. Prints the PTY path and keeps a stable symlink /dev/shm/gps-nmea to it,
# so tools/gnssd.py (pcsession.sh --gnss /dev/shm/gps-nmea) does not need to know the /dev/pts/N number.
# GPS_SAVE=1 ./start-gps.sh   stores rate/baud/sentences in the receiver (BBR + flash) once; later starts then take the fast path.
# usage: ./start-gps.sh [--gpsd]     (--gpsd additionally runs gpsd on the PTY, needs sudo; then gnssd must NOT read the PTY too)
cd "$(dirname "$0")"
GLASGOW=${GLASGOW:-$HOME/.local/bin/glasgow}
LINK=/dev/shm/gps-nmea; LOG=captures/logs/glasgow.log; mkdir -p captures/logs
LOG_CFG=captures/logs/gnss_config.log; BAUD=${GPS_BAUD:-57600}; PY=/home/steamos/playground/.venv/bin/python
rm -f "$LINK"; GPID=""; trap '[ -n "$DPID" ] && { kill $DPID 2>/dev/null || sudo -n kill $DPID 2>/dev/null; }; [ -n "$GPID" ] && kill $GPID 2>/dev/null; rm -f "$LINK"' EXIT INT TERM
bridge() {   # bridge BAUD : (re)start the Glasgow UART->PTY bridge at this baud rate; sets $pty
  [ -n "$GPID" ] && { kill $GPID 2>/dev/null; wait $GPID 2>/dev/null; sleep 1; }
  local before; before=$(ls /dev/pts 2>/dev/null | sort); pty=""
  "$GLASGOW" -v run uart -V 3.3 --rx A1 --tx A2 -b "$1" pty > "$LOG" 2>&1 & GPID=$!
  for _ in $(seq 1 150); do                                  # up to 30 s (FPGA bitstream load + applet start)
    kill -0 $GPID 2>/dev/null || { echo "glasgow exited:"; tail -20 "$LOG"; exit 1; }
    pty=$(grep -o '/dev/pts/[0-9]\+' "$LOG" | tail -1)       # 1. glasgow logs the PTY path
    if [ -z "$pty" ]; then                                   # 2. fallback: the newly appeared /dev/pts entry
      n=$(comm -13 <(echo "$before") <(ls /dev/pts | sort) | grep -x '[0-9]\+' | head -1); [ -n "$n" ] && pty=/dev/pts/$n
    fi
    [ -n "$pty" ] && break; sleep 0.2
  done
  [ -n "$pty" ] || { echo "no PTY found, see $LOG"; exit 1; }
}
cfg() { timeout 30 $PY -I tools/gnss_config.py "$pty" --hz "${GPS_HZ:-10}" ${GPS_SAVE:+--save} "$@" >> "$LOG_CFG" 2>&1; }
: > "$LOG_CFG"
# The receiver keeps its RAM settings while powered, so try the target baud first; otherwise configure at the default 38400
# (rate, sentences, then the UART baud switch last) and reopen the bridge at the new rate.
bridge "$BAUD"
cfg --baud "$BAUD"; rc=$?
if [ $rc = 2 ]; then
  echo "receiver not at $BAUD baud, configuring at 38400"; bridge 38400; cfg --baud "$BAUD"; rc=$?; sleep 0.5; [ $rc = 0 ] && bridge "$BAUD"      # only switch the bridge if the receiver took the new baud rate
fi
[ $rc = 0 ] && echo "receiver: ${GPS_HZ:-10} Hz at $BAUD baud" || { echo "receiver config incomplete (rc=$rc), see $LOG_CFG"; cat "$LOG_CFG"; }
ln -sf "$pty" "$LINK"; echo "GNSS PTY: $pty  (symlink $LINK)"     # only now: gnssd must not open a PTY that is about to be recreated
if [ "${1:-}" = --gpsd ]; then
  # gpsd needs no root for a PTY we own; fall back to passwordless sudo (sudoers: steamos ALL=(root) NOPASSWD: /usr/bin/gpsd).
  # It runs in the background so the trap above stops it together with glasgow.
  gpsd -N -n -b -D 3 "$pty" > captures/logs/gpsd.log 2>&1 & DPID=$!; sleep 1
  kill -0 $DPID 2>/dev/null || { sudo -n gpsd -N -n -b -D 3 "$pty" > captures/logs/gpsd.log 2>&1 & DPID=$!; sleep 1; }
  kill -0 $DPID 2>/dev/null || { echo "gpsd failed, see captures/logs/gpsd.log"; exit 1; }
fi
wait $GPID
