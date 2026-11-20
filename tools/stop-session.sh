#!/usr/bin/env bash
# Stops a session started by launch-session.sh (pcsession.sh's cleanup hands the cameras back) and the GNSS bridge.
pkill -INT -f "[p]csession.sh"; sleep 8; pkill -f "[s]tart-gps.sh"; pkill -f "[g]lasgow.* run uart"; exit 0
