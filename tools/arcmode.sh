#!/bin/sh
# Set up both Arcturus capture routes: enable links sensor->csiphy->csid->vfe rdi0 and set formats.
# XRService resets media links/formats on its camera restarts, so run this AFTER taking ownership.
# usage: arcmode.sh [WxH]   default 2464x2464 (XRService default; sensor full mode)
F="fmt:Y10_1X10/${1:-2464x2464}"
M="media-ctl -d /dev/media0"
$M -l '"msm_csiphy3":1 -> "msm_csid0":0 [1]' || exit 1
$M -l '"msm_csid0":1 -> "msm_vfe0_rdi0":0 [1]' || exit 1
$M -l '"msm_csiphy5":1 -> "msm_csid1":0 [1]' || exit 1
$M -l '"msm_csid1":1 -> "msm_vfe1_rdi0":0 [1]' || exit 1
for e in '"arcimx616 0-001a":0' '"msm_csiphy3":0' '"msm_csiphy3":1' '"msm_csid0":0' '"msm_csid0":1' '"msm_vfe0_rdi0":0' \
         '"arcimx616 0-0010":0' '"msm_csiphy5":0' '"msm_csiphy5":1' '"msm_csid1":0' '"msm_csid1":1' '"msm_vfe1_rdi0":0'; do
  $M -V "$e [$F]" || exit 1
done
