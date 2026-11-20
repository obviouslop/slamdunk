#!/usr/bin/env python3
"""Configure a u-blox M10 (MIA-M10Q) over its UART/PTY with UBX-CFG-VALSET (RAM layer only: re-applied at every start).
usage: gnss_config.py PORT [--hz 10] [--model 3] [--msgs GGA,RMC,GSA,GST]
Sets the navigation rate, the dynamic model (3 = pedestrian) and which NMEA sentences are output on UART1 (the others are switched
off so that the rate fits the baud rate: 38400 baud = 3.8 kB/s). Every key is acknowledged (ACK-ACK / ACK-NAK) and the result printed.
Key ids are from the u-blox M10 interface description (as remembered): a NAK means a wrong id for this firmware."""
import argparse, os, select, struct, sys, termios, time

KEYS = dict(RATE_MEAS=(0x30210001, "<H"), RATE_NAV=(0x30210002, "<H"), DYNMODEL=(0x20110021, "<B"))
NMEA = dict(GGA=0x209100bb, GLL=0x209100ca, GSA=0x209100c0, GSV=0x209100c5, RMC=0x209100ac, VTG=0x209100b1, GST=0x209100d4)


def ubx(cls, mid, payload=b""):
    body = bytes([cls, mid]) + struct.pack("<H", len(payload)) + payload
    a = b = 0
    for x in body: a = (a + x) & 255; b = (b + a) & 255
    return b"\xb5\x62" + body + bytes([a, b])


def valset(key, fmt, val, layers=1):
    return ubx(0x06, 0x8A, struct.pack("<BBH", 0, layers, 0) + struct.pack("<I", key) + struct.pack(fmt, val))


def send_wait_ack(fd, msg, cls=0x06, mid=0x8A, timeout=1.5):
    os.write(fd, msg); buf = b""; t0 = time.time()
    while time.time() - t0 < timeout:
        if select.select([fd], [], [], 0.1)[0]:
            buf += os.read(fd, 4096)
            for i in range(len(buf) - 9):
                if buf[i:i + 4] in (b"\xb5\x62\x05\x01", b"\xb5\x62\x05\x00") and buf[i + 6:i + 8] == bytes([cls, mid]):
                    return buf[i + 3] == 1
    return None                                              # no answer


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("port"); ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--model", type=int, default=3); ap.add_argument("--msgs", default="GGA,RMC,GSA,GST")
    a = ap.parse_args()
    fd = os.open(a.port, os.O_RDWR | os.O_NOCTTY)
    try:
        at = termios.tcgetattr(fd); at[0] = at[1] = at[3] = 0; at[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
        termios.tcsetattr(fd, termios.TCSANOW, at)
    except termios.error: pass
    on = set(a.msgs.split(",")); ms = int(round(1000 / a.hz))
    items = [("rate %g Hz" % a.hz, KEYS["RATE_MEAS"], ms), ("nav ratio 1", KEYS["RATE_NAV"], 1), ("dynmodel %d" % a.model, KEYS["DYNMODEL"], a.model)]
    items += [(("%s on" if n in on else "%s off") % n, (k, "<B"), 1 if n in on else 0) for n, k in NMEA.items()]
    bad = 0
    for name, (k, fmt), v in items:
        r = send_wait_ack(fd, valset(k, fmt, v)); bad += r is not True
        print(f"  {name:14s} {'ACK' if r else 'NAK (key not accepted)' if r is False else 'no answer (TX wired? baud?)'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
