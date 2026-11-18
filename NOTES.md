# Point cloud on Steam Frame — recon notes (2026-11-17)

## Hardware / pipeline map
- SoC: Qualcomm (8 cores, A520/A720), 15 GiB RAM, Adreno w/ Vulkan, CAMSS ISP at /dev/media0.
- Sensors on CAMSS:
  | sensor | res | node | consumer |
  |---|---|---|---|
  | og01a1b 4-0060 (slam_right, 180° fisheye, IR) | 1056x1024 Y8 | video13 (vfe4_rdi0) | XRService |
  | og01a1b 4-0036 (slam_left) | 1056x1024 Y8 | video9 (vfe3_rdi0) | XRService |
  | og0ve10 5-0060 (upper cams, 170°) | 640x480 Y8 | video6/7 (vfe2 rdi0/1, interleaved) | XRService |
  | arcimx616 0-001a / 0-0010 (Arcturus color stereo) | 2464x2464 Y10 (raw Bayer?) | video0 (vfe0_rdi0) / video3 (vfe1_rdi0) | color passthrough, only when lit |
- IMU: InvenSense ICM-42688 @1 kHz, not exposed via IIO; consumed by XRService.
- XRService (pid owner of tracking cams) = VIO/SLAM. Sparse map (~1000 pts, ~50 keyframes).
- /dev/video99 = v4l2loopback fed by /opt/steamvr/bin/linuxarm64/v4l2cam (1920x1080 RGB, compositor view).

## Calibration sources
- /persist/xrservice.json — factory rig cal (vicalib): KB fisheye + kb_spline intrinsics,
  extrinsics (mm, plus_x/plus_z/position, parent chain Cam0), IMU biases/scales.
  Covers slam_right, slam_left, upper_left, upper_right. NOT the Arcturus cams.
- /persist/device_config.json — cad_from_cal (cal frame -> head frame), display distortion.
- ~/.config/openvr/config/cv/xrservice/colorPassthroughCalibrationHistory.json — online
  extrinsics: frontCam0FromPassthroughCam0, passthroughCam0FromPassthroughCam1 (≈63 mm baseline).
  Format looks like [tx,ty,tz,qx,qy,qz,qw] (metres). No Arcturus intrinsics found yet.

## OpenVR public camera API
- IVRTrackedCamera: HasCamera=true but NumCameras=0, frame sizes/intrinsics fail,
  GetVideoStreamFrameBuffer -> InvalidHandle. Not usable on Frame.
- !!! HAZARD: Acquire/ReleaseVideoStreamingService on the HMD disconnects the compositor's
  "depth mesh block queues" (vrcompositor.txt) and kills passthrough. Do not call it
  while the user is in the headset. probe/probe.cpp is the culprit — don't rerun.

## XRService depth: /xrservice/roomview3d/* (DECODED)
- Access: IVRBlockQueue_005 (internal; vtable recovered, see src/blockqueue.h, re/vtable.py).
  Connect + WaitAndAcquireReadOnlyBlock(Latest) + ReleaseReadOnlyBlock. Read-only; does NOT
  disturb compositor passthrough (verified). Never call Destroy.
- meshLayout: static uint32 triangle index buffer for a 132-wide vertex grid.
- meshData: 6000512-byte blocks @ ~33 Hz. float32 xyz[132*130] then 76-byte trailer @6000436:
  3x4 pose (NOT the mesh motion; some reference pose), f32 ≈94.255, u32 177, u32 seq,
  u32 0, u32 50688, f64 t (CLOCK_MONOTONIC s, ≈3 ms before delivery).
  "Latest" may return older ring slots and seq=0 empties → dedupe by max seq.
- Depth is computed on the CVP (Qualcomm HW stereo, "DepthFromStereo") — works in IR/dark.
- Frame (re/rv_handeye.py, re/rv_latency.py): world = H_hmd(t-50ms) * [Rx(≈-0.4°,1.3°,0)·diag(-1,1,-1) | c≈(5.3,-1.6,-1.0)cm] * v
  Frame-to-frame static consistency ≈3 cm median (per-frame noise 0.4 cm). Heavily smoothed,
  rubber-sheets across occlusion edges → room-scale structure only, no fine detail.
- src/rvcloud: live voxel fusion → PLY (Standing universe). tools/plyview.py renders views.
- VERDICT (ground-truth test re/rv_groundtruth.py, cued floor/wall/ceiling, IR mode, dim room):
  the dominant plane is ALWAYS perpendicular to mesh +z at 0.7–1.5 m (13–30% inliers), whether
  looking at floor, wall or ceiling; floor at ~1.6 m never appears. The mesh is a regularized,
  view-facing reprojection proxy, not usable as a depth sensor (at least in IR/dark). Fused
  clouds (rvcloud) look nothing like the room. The 3 cm "consistency" was self-consistency of
  the proxy, not accuracy. Untested: behaviour in good light / color passthrough mode.
- Side observation: HMD pitch during the cued run did not follow the floor→wall→ceiling script
  (never beyond ±52°); unresolved whether user motion differed or pose pairing is off.

## Interesting leads
- Compositor consumes "depth mesh blocks" from XRService for passthrough reprojection
  -> XRService already computes scene depth (works in IR). If tappable, could be the
  cheapest path to a point cloud.
- ArcturusVisionCamera app records stereo + per-frame poses (03-beach_poses.csv:
  t; ?; 3x4 left; 3x4 right). It does NOT use V4L2 directly: it loads
  libAV1ExternalCameraProvider.so (export PECreateExternalCameraProviderV1), which leases the
  pipeline from XRService (abstract seqpacket socket @passthrough_camera) and gets frames +
  worldFromAvCam0 poses + calibration. No public header — would need RE.
- Direct RDI route (vendor tool camera-diagnostics/camera-rdi-probe): hold exclusive flock on
  /tmp/passthrough_camera.lock (CameraSwitch holds it; signal via
  /tmp/arcturus_camera_recorder.lock), enable csiphy3/5 -> csid -> vfe0/1_rdi0 links, RAW10
  2464x2464 on /dev/video0 + /dev/video3. Arcturus intrinsics still unknown.
- CameraSwitch auto-switches to IR in low light (Arcturus sensors then idle).

## Arcturus raw capture (tools/arcgrab.py, tools/raw10.py) — 2026-11-17
- Lock handoff works: take /tmp/arcturus_camera_recorder.lock -> CameraSwitch drops
  /tmp/passthrough_camera.lock within 0.5 s -> take it; release both -> CameraSwitch re-takes.
- Routes csiphy3->csid0->vfe0_rdi0 (/dev/video0) and csiphy5->csid1->vfe1_rdi0 (/dev/video3)
  are left enabled by the provider. Capture as Y10P 2464x2464 (stride 3088) works, ~25-30 fps.
- BUT the sensor actually streams lines of 3264 px RAW10 (4080 B/line, no padding); the
  2464x2464 buffer holds ~1864 such rows. Decode with W=3264 continuous (not stride 3088).
  First ~40 rows look like embedded data (not standard Sony tag format; undecoded).
- Driver (built-in, kallsyms): modes arcimx616_mode_data_3820x2464 and 1640x1232_binned,
  arcimx616_private_ioctl (likely exposure/gain), aw9106 LED driver, EEPROM copy/nvmem.
- Images are a real view of the room but ~10 DN above black (black level 64): exposure
  control unknown. Only V4L2 ctrl is 'brightness' (24-bit). Original values 0x003011 /
  0x013011 suggest a packed (value<<16)|reg register-poke interface, NOT exposure.
  !!! I wrote 50000/200000/800000 to it during a sweep (possible writes to regs 0xC350,
  0x0D40, 0x3500), then restored the original ctrl values. Don't touch it until understood.
- Per-module EEPROMs: /sys/bus/nvmem/devices/passthrough_cams_eeprom_{left3,right4}/nvmem
  (8 KiB, root-only) — likely factory lens calibration.

## Arcturus calibration + sensor control (2026-11-17, cont.)
- FACTORY STEREO CAL lives in passthrough_cams_eeprom_left3 as JSON @0x7fa (rig_json_version 3,
  device_serial AV6138B772, hardwareIdentifier "av1"). Extracted: captures/arcturus_calib.json.
  KB fisheye (== OpenCV fisheye k1..k4) per cam, 2464x2464 crops of the 3264-wide line:
  passthrough_right crop x=250, f≈1472.6/1475.4, c≈(1384.4,1236.9)
  passthrough_left  crop x=550, f≈1472.3/1473.7, c≈(1101.2,1221.7)
  baseline 63.98 mm, optical axes toe-out 21.9°, extrinsics in "cad" frame (mm).
  Right EEPROM: binary (probably lens shading). Raw dumps: captures/eeprom_{left,right}.bin.
- Which /dev/video is which side: NOT yet verified (cam0=video0=0-001a, cam1=video3=0-0010).
- Kernel: /boot/Image, no effective KASLR (kallsyms addr - 0xffff800080000000 = file offset).
  arcimx616_set_ctrl: V4L2 brightness = raw register poke: reg = val & 0xffff,
  data = (val>>16) & 0xff. Provider pokes 0x3011 (0 on cam0, 1 on cam1; sync master/slave?).
  My accidental pokes: 0xC350←0x00, 0x0D40←0x03, 0x3500←0x0C (both sensors).
- Mode table (driver setting_3820x2464, 8-byte entries {u32 (data<<16)|reg, u32}):
  full array 6560x4928, 2x2 binning (0x0900=1, 0x0901=0x22), output 3264x2464,
  FRM_LEN 7396, LINE_LEN 3768, COARSE_INT 7348, ANA_GAIN 112, DIG_GAIN 256, RAW10, 4 lanes.
  Table is NOT re-applied on stream start → exposure is whatever XRService AE left (dark).
- Exposure plan: poke 0x0104=1, 0x0202/0x0203 (lines), 0x0204/0x0205 (ana gain), 0x0104=0
  while streaming. Waiting for full EEPROM backup (backup/) before any further sensor writes.

## !!! INCIDENT 2026-11-17 22:03–22:05: Arcturus grabs break head tracking
- Taking /tmp/arcturus_camera_recorder.lock + passthrough lock makes XRService log
  "[ExternalCameraOwnership] External camera ownership released" -> "Restarting cameras from
  tracking thread" -> closes ALL tracking cams -> 3DoF fallback (user saw desktop follow head).
  XRService then segfaulted (22:04:02) and was restarted by vrserver (again 22:05:03).
  XRService log dirs show restarts at 21:49/21:50/21:51 too = every earlier grab.
- Aftermath: upper tracking cams (og0ve10, csid2, VC-interleaved) desync in a loop,
  "escalating to FPGA + upper camera reset" (FPGA GW1N2_CS100H), dmesg "Unexpected CSID CSI2 RX
  status 0x80011 (csid=2)". Front cams OK, 6DoF resumed. (Same desync seen once at 21:01:50
  before any grab, during color->IR switch, so the sync path is fragile anyway.)
- Also: one capture hung (no frames in 30 s, killed) right after the exposure register pokes.
- Possibly relevant: my stray arcimx616 register pokes (0x3500, 0x0D40, 0xC350, plus
  0x0104/0x0202-0x0205) and 0x3011 (sync master/slave?) — Arcturus sensors may share sync with
  the FPGA/upper cams. Unproven.
- RULE: no lock-handoff grabs / sensor pokes while the user depends on tracking. Raw access
  must go through a path XRService cooperates with (provider IPC), or with tracking expendable
  (user seated, warned) and a reboot afterwards.

## Offline route WORKS: Arcturus recorder video + metadata -> fused colour point cloud (2026-11-17)
Input: ~/Downloads/2026-11-07_12-47-24_stereo.mp4 (2x HEVC 2464², 59.94 fps, ~55 Mbit/s each,
72 s, gaps in pts) + timed-metadata track (JSON/frame, recordingMode raw-dual-track-v1).
- tools/avmeta.py -> captures/rec_2026-11-07_meta.npz. Per frame: exposure/gain, KB intrinsics,
  midExpTs, avcam0InWorld (m), avcam1InAvcam0. Pose = plus_x/plus_z/position of child in parent.
  Track 0 = "right" = cam0, track 1 = "left" = cam1. Video packet i <-> metadata record i.
- Camera axes: metadata cam frame = S · OpenCV frame, S0 = [[0,1,0],[-1,0,0],[0,0,1]],
  S1 = [[0,-1,0],[1,0,0],[0,0,1]] (sensors mounted ±90°). Verified twice (stereo + multi-frame).
- Video frames are re-centred vs. factory calibration: principal points offset by ≈(-198,+153)
  (cam0) and (+120,+144) (cam1) px, toe-out 20.4° (+1.7° vs metadata). Fit: re/fit_video_geom.py
  -> captures/stereo_video_calib.json, epipolar median 0.87 px. (Pinning cam0 gives 2.6 px.)
- tools/stereo_depth.py: fisheye stereoRectify (half res 1232², f≈476 px) + SGBM -> cam0-frame
  coloured PLY. ~1.5 s/frame on the Frame CPU. Depth noise ≈0.8 cm @1 m, 3.3 cm @2 m, 7 cm @3 m.
- tools/fuse_recording.py: X_w = W0[f]·S0·X_cv; voxel fusion. 30 frames (1520..2680 step 40,
  ≈20 s) -> 431k voxels @2 cm in 30 s; frames agree ≈3 cm median; roof plane + tile rows
  intact after fusion. World is y-up. Renders: captures/fused/balcony_30f_views.jpg.
- Residual: small rotation video-cam0 vs metadata cam0 (fit suggests ≈(-8.5,-5,0)° but weakly
  constrained; not applied). Needs a recording with bigger rotations to pin down.
- Frame extraction: one ffmpeg pass with select='...*not(mod(n,40))', rename k-th file to n.
- !!! arcimx616 private ioctls (re/kernel/pioctl.s): 0x800456C1 status byte; 0x000056C3 =
  arcimx616_write_reg_dual (arg = val<<16 | reg, writes BOTH sensors via shared i2c client);
  0xC01056C4 = EEPROM READ, 0x401056C5 = EEPROM WRITE (16-byte struct, <=2 KiB, protects
  [0,X) only). NEVER call 0xC5. EEPROM backups: backup/eeprom-20261117-220329/ (+SHA256SUMS).
  No sensor register READ path exists in the driver.

## LIVE CAPTURE WORKS: src/arcgrab -> /dev/shm/arcgrab -> tools/arcshm.py (2026-11-17 22:50)
- src/arcgrab (C++, no OpenCV): take /tmp/arcturus_camera_recorder.lock + passthrough lock,
  wait 1.5 s (XRService restarts its cameras and RESETS media links/formats), run
  tools/arcmode.sh (enable csiphy3->csid0->vfe0_rdi0, csiphy5->csid1->vfe1_rdi0, formats
  2464x2464), STREAMON both, then on the first frame (sensor runtime-active, otherwise
  set_ctrl silently drops register writes!) set exposure/gain + digital crop:
    left  = /dev/video0 = arcimx616 0-001a, crop x=550 ; right = /dev/video3 = 0-0010, crop x=250
    regs (grouped hold 0x0104): 0x0408/9 crop x, 0x040C/D crop w=2464, 0x034C/D X_OUT=2464,
    0x0202/3 integration lines (<=7300 @ ~31 fps), 0x0204/5 analog gain code (<=960 ≈16x),
    0x020E/F digital gain (256 = 1x).
  -> full 2464x2464 frames, RAW10 packed, stride 3088 B, BGGR, ~31 fps per camera, both cams.
  On exit restores crop x=0/w=3264 (XRService's state) before STREAMOFF.
- Without the crop the sensor sends 3264-px lines that overflow the 2464x2464 buffer (only
  ~1864 rows, corrupted band). Setting 3264x2464 or 1640x1232 pad formats -> no frames AND
  XRService segfaults on reclaim (22:39:40, 22:40:50). Keep pads at 2464x2464.
- arcimx616_set_fmt: 1640x1232 exactly -> binned 4x4 mode table, anything else -> full mode;
  only stored, applied at power-on (untested how to trigger). Binned = 3 µm pixels, 4x light.
- Cost: ~0.3 s 3DoF blip at start and end of each session (XRService camera restart). No crash
  with the standard format path.
- Sensor: IMX616, 0.75 µm native pixels (1.5 µm in 2x2-binned full mode) -> needs high gain
  indoors (AE hit 33 ms, 16x analog, up to 16x digital in a lamp-lit room).
- tools/arcshm.py: seqlock reader, unpack, BGGR half-res RGB / gray. tools/qrscan.py:
  cv2 QRCodeDetector multi, 16 ms/frame at 1232². Not yet tested with an actual QR code.
- TODO: AE tuning (target, dgain decay), white balance, pose stamping (OpenVR pose at
  v4l2 ts, CLOCK_MONOTONIC), live stereo, binned mode, systemd service w/ graceful handback.
- QR PoC VERIFIED 2026-11-17 22:55: phone QR "Hello claude" decoded live (cam0 = left), first
  ~1.4 s after start, 90 decodes in 30 s, code ≈130 px wide in the 1232² half-res gray image.
  76 ms/frame with a code in view (16 ms without). Ideas: ROI detect then full-res decode,
  pose-stamped 3D placement of codes. Parked for later; focus = point cloud / SLAM.

## Live calibration + viewer (2026-11-17 23:00–23:30)
- Factory stereo cal fits LIVE frames directly (OpenCV frames, no axis juggling): epipolar
  median 1.11 px (re/live_stereo_check.py). Rectification puts camL on the RIGHT -> livestereo
  mirrors both rectified images for SGBM. Dense vs sparse agree to 1.4 cm (re/live_dense_check.py).
- head_from_camL prior = diag(-1,1,-1) @ cad_from_camL (EEPROM). Hand-eye (liverec1): 0.93° /
  0.68 cm per pair. Joint fit of X+delta (re/calib_world.py, 3D feature residuals) overfits the
  translation (cross-validation: held-out worse). KEEP THE PRIOR. Residual ≈ stereo noise floor.
- Time offset: delta scan vs mid-exposure estimate improves 3.7 cm -> ~2.3-2.7 cm for delta in
  [-50, -130] ms, noisy, no clean minimum. Suspect: GetDeviceToAbsoluteTrackingPose with
  negative prediction extrapolates instead of using history. TODO: high-rate pose sampler
  (prediction 0, own timestamps) + interpolation, then re-calibrate.
- tools/livestereo.py, tools/fuse_live.py (0.78 s/pair, 80 pairs -> 1.9M voxels @2 cm),
  tools/liverec.py (pairs + full-rate poses.jsonl). Recordings: captures/liverec1, liverec2.
- Godot viewer: copy of ~/code/openxr_passthrough in viewer/. Godot 4.7.2 arm64 at
  ~/code/Godot_v4.7.2-stable_linux.arm64. SteamVR OpenXR 2.18.2 supports blend modes [0,2]
  (ALPHA_BLEND = passthrough). GL Compatibility renderer FAILS (xrEnumerateSwapchain RUNTIME_FAILURE
  via zink); use --rendering-driver vulkan --rendering-method mobile -> works (Adreno 750, 72 Hz).
  STAGE reference space == OpenVR Standing -> fused points can be used directly.
  User: tracking felt worse than SteamVR home (likely CPU contention from a concurrent job).

## !!! Capture sessions break CONTROLLER tracking (2026-11-17 23:40)
- XRService "Upper cameras desynced ... escalating to FPGA + upper camera reset" loop STARTED at
  22:37 (first live arcgrab session where exposure writes took effect), continuous while capturing
  (22:37-23:10), quiet 23:11-23:34 (no captures), back at 23:35 with the delay recording, stopped
  23:40:05 after a controller reconnect + FPGA reset ("Sync done fully"). User lost the controller
  laser after captures (last time only a SteamVR restart fixed it). Log hint: "ISP to kernel offset
  has changed ... Camera and controller sync may be affected".
- Hypothesis: Arcturus sensors share frame sync with upper tracking cams via FPGA (VC interleaving);
  our long integration (up to 7300 of 7396 lines) persists after the session. Mitigation added:
  arcgrab restores exposure 1000 lines / gain 0 / dgain 1x (+ crop) on exit. UNVERIFIED.
- To test: one capture with --exp <=4000 cap, then watch `journalctl --user | grep "escalating to FPGA"`.

## Pose/camera timing (delay1, 500 Hz pose history, 2026-11-17 23:36)
- arcgrab now samples HMD pose at ~480 Hz (prediction 0) into a ring buffer and interpolates
  at mid-exposure (+ --pose-offset). --pose-log writes the history.
- re/delay_xcorr.py: around the 6 fastest head snaps, HMD angular speed peaks ~300-350 ms BEFORE
  the camera's -> camera frame content is ~10 frames (~320 ms) older than its V4L2 timestamp
  (buffer recycling / pipeline lag?). Explains swimming in fused clouds. Exact value + stability
  TBD (extend scan to -450 ms; check vs number of V4L2 buffers).
- DELAY MEASURED (delay2, 23:45, exp cap 3000, 4 V4L2 buffers): camera content is 330 ms older
  than t_mid estimate. At delta=-330 ms: mean rotation error 0.44°/frame on fast motion (0.09°
  slow), |omega| correlation 0.996. delay1 agreed (~-320..-330). arcgrab default --pose-offset -0.330.
  Cause unknown (≈10 frames; not explained by 4 buffers). Check with --nbufs 8 some time.
- Exposure cap + exposure reset on exit did NOT prevent the upper-cam desync loop (27 escalations
  within 1 min after delay2). The capture/ownership handover itself is the trigger.
- After captures: controllers connected but pose invalid (TrackingResult 100 Uninitialized) ->
  controller tracking broken until SteamVR restart. src/devstate (read-only) shows device states.
  => Lock-handoff capture is NOT acceptable for a product path. The Arcturus recorder app captures
  without breaking controllers (user recorded with controller bindings) -> its XRService
  "external camera provider" lease (libAV1ExternalCameraProvider.so) is the cooperative route. TODO RE.

## ROOT CAUSE of the "330 ms delay": V4L2 timestamps are +355.7 ms AHEAD (2026-11-17 23:58)
- Direct measurement (ARCGRAB_TSDEBUG=1): ts - CLOCK_MONOTONIC_at_DQBUF = +355.7 ms ±0.5, both cams,
  constant. Images are NOT late; the camss timestamps are in a shifted time base. With the old
  fixed -330 ms offset, t_mid-330 ms landed at ~"now" -> 63% of lookups beyond the newest pose
  sample -> pose_valid=0 (liverec3: only 61/160 pairs valid).
- Fix in src/arcgrab: per camera, offset = min over last 64 frames of (ts - now_at_dequeue);
  ts -= offset (published ts is now real CLOCK_MONOTONIC EOF, ~+4.5 ms dequeue latency);
  pose = history at t_mid (+ --pose-offset, default 0). Verified: all frames pose_valid=1.
- Consistency evidence (liverec3, true 500 Hz history, re/calib_world.py): 3D residual median
  4.90 cm (no correction) -> 0.61-0.64 cm at the equivalent of the corrected timing.
  The earlier "3 cm stereo noise floor" was pose timing error.
- liverec1/liverec2 poses are mis-timed (old method) - don't use them for evaluation.
- Recordings: captures/delay1, delay2 (head snaps, thumbnails + 500 Hz poses), liverec3 (walk).

## NEXT SESSION (after SteamVR restart; captures break controller tracking until restart!)
1. New walk-around recording with fixed timestamps -> fuse_live.py (no --delta) -> verify ~6 mm.
2. Live service: arcgrab -> stereo (livestereo, ~0.4-0.8 s/pair CPU) -> voxel fusion -> publish
   cloud (shm/socket). Consider GPU (Vulkan compute) SGBM later.
3. Godot viewer (viewer/, Vulkan mobile renderer, STAGE space == OpenVR Standing, alpha blend):
   render cloud points over passthrough + in-headset status panel.
4. Cooperative capture (Arcturus recorder's XRService provider lease) to stop breaking
   controller tracking; meanwhile keep sessions few/long and restart SteamVR after.

## 2026-11-18 00:20 findings after reboot
- Upper-camera desync/FPGA-reset loop is PRE-EXISTING (journal per boot: Nov 7 326 + 289
  escalations, Nov 13 293, Nov 10 0); seems tied to colour passthrough via the Arcturus module.
  Our direct grabs amplified it (Nov 17: 1728 escalations, 36 colour-passthrough reconnects).
  After a clean reboot (nothing of ours running) it started 18 s after colour passthrough connected.
- Viewer "mirrored/rotated" cloud: project.godot had openxr/reference_space=2 = LOCAL_FLOOR (from
  the demo), not STAGE. Fixed to 1 (STAGE). Also the test cloud was from the previous universe.
- head_from_camL (EEPROM cad + device_config) == XRService's own (recording metadata
  inv(headliveInFrontcam0)·avcam0InFrontcam0·avcam1InAvcam0) within 1.4° / <1 mm. Geometry OK.
- /dev/video99 ("SteamVR" v4l2loopback, v4l2cam) = COMPOSITED headset view (overlays, dashboard,
  passthrough background), mono, 1920x1080 RGB, ~3.7 fps. Shelved (maybe QR in a pinch).
- Direction: cooperative capture via the Arcturus/XRService external camera provider (user is
  looking for Arcturus docs). Meanwhile develop live pipeline + viewer against recordings.
- Arcturus developers page: "A direct camera access API is not yet available. We're taking developer
  inquiries about your use case." Passthrough = standard OpenXR ALPHA_BLEND (clear to transparent,
  projection layer BLEND_TEXTURE_SOURCE_ALPHA_BIT [+ UNPREMULTIPLIED_ALPHA_BIT]). Their native
  OpenXR/Vulkan sample: third_party/arcturus_sample/openxr-passthrough-cube (MIT; reference only).
- Valve "Steam Frame custom engines" doc: nothing on camera access. Controllers appear as Oculus
  Touch by default; Steam Frame Controller OpenXR interaction profile exists; eye gaze
  XR_EXT_eye_gaze_interaction; FB/META foveation extensions.
- Arcturus app (binary dated Nov 13 20:23) triggered "External camera ownership requested" ->
  "Restarting cameras from tracking thread" at Nov 13 20:26:47 = same handover as our arcgrab.
  Nov 7: desync loop began ~2.5 min after colour passthrough connected, before any recording.

## 2026-11-18 00:30-01:30: first live session in headset + offline improvements
LIVE TEST RESULTS (user in headset, captures/live4, live5):
- Live cloud over passthrough WORKS. Controllers survived full live sessions (arcgrab+server+viewer).
- V4L2 ts offset after reboot was 43.6 ms (was 355.7 ms) -> not constant; online estimation needed.
- Per-camera offset minima broke L/R pairing (7 ms) -> now one shared offset (median L/R 2.6 ms).
- Checked: Godot STAGE == OpenVR Standing exactly; tracking scale image/HMD = 1.035; hand-eye
  0.39° / 1.4 cm; dense vs sparse 1.6-1.9 cm; eye height above reconstructed floor 1.73 m.
  => geometry & scale correct. User's "dwarf" impression = fog/spray + big near squares (rendering).
- Viewer reference space was LOCAL_FLOOR (demo setting 2) -> fixed to STAGE (1).
- SteamVR showed "Resume Game" menu although the app ran (unresolved, cosmetic).
- Arcturus CameraSwitch "Auto night vision" causes its own XRService camera restarts (IR on/off).
OFFLINE WORK (no cameras, no headset):
- Viewer: semi-transparent points (alpha 0.55, no depth write), default 6 mm, near fade 0.25-0.6 m.
  Controls right: trigger hold = paint capture (paint mode), A = auto/paint, B = size;
  left: trigger = alpha, X = show/hide, Y = save PLY (captures/sessions/), grip = reset map.
  Viewer -> server: /dev/shm/pcctl.json at 10 Hz (aim pose STAGE, mode, painting, cmd/cmd_seq).
- tools/fusion.py (Fuser): incremental sorted-key voxel map + free-space carving (project map into
  each frame's rectified depth; seen-through voxels get misses; misses > hits -> removed) + pruning.
  Shared by pcserver (live) and fuse_rec.py (offline). Defaults: voxel 1.5 cm, publish >= 3 hits.
- Stereo: 2x2-binning demosaic + half-res rectification maps: points() 210 -> 119 ms.
  Server end-to-end 1090 -> ~175 ms/pair (~5.7 pairs/s capacity).
- Spray analysis: cross-frame confirmation only ~1/3 regardless of SGBM settings / flying-pixel
  filter -> noise is physics (f=314 px @ scale 0.35, B=64 mm: sigma_Z ~2.8 cm @1.5 m, ~8 cm @2.5 m)
  plus dark/high-gain frames. Levers: more frames (now faster), stricter hits, carving, scale 0.5
  when CPU allows, brighter scenes. Flying-pixel filter kept but off (edge_rel=0).
- tools/pcsession.sh: one-command session (cameras or --replay, optional --record with composited
  view + poses + pairs). Cleans up, hands cameras back, beeps (tools/sounds).
NEXT: in-headset test of the new viewer (transparency, paint mode, save/reset); TSDF-style ray fusion
or GPU stereo if noise/throughput still limit; cooperative camera access if Arcturus answers.

## 2026-11-18 02:30-03:30: live6 review, TSDF, planning tools, export, GNSS (offline only)
User feedback on live6 (8.5 min session, recorded): works, controllers fine; "points move by a different
amount than the camera image when rotating (parallax / FOV mismatch?)"; "points track worse than SteamVR
windows"; wants faster + fewer-but-more-accurate points; SteamVR still starts in "paused / Resume".

Analysis of live6:
* Pose timing re-fitted on live6 (re/calib_world.py): best delta -4.6 ms, extrinsic refit improves the
  3-D residual only 5.83 -> 5.67 cm => our cloud geometry/timing is fine; residual = stereo noise.
* While arcgrab owns the colour cameras the passthrough falls back to the IR tracking cameras (blue-ish
  composited view). SteamVR reprojects passthrough onto a depth *proxy* (the roomview mesh we decoded):
  anything not at the proxy depth shows parallax vs. correctly world-locked content. Our points are
  world-locked, so part of the mismatch is the passthrough, not the cloud. The other part is real:
  spray/fog points at wrong depth slide over the image when the eyes translate -> fixed by TSDF below.
* Godot ran at 72 fps = display rate (Frame refresh 72 Hz) but reloading 400k points took 36-60 ms on the
  main thread every publish -> dropped frames -> the cloud judders vs. compositor-rendered windows.
  Fixed by tiled loading (only changed 1 m tiles, per-frame budget, frustum-culled AABBs).
* Old server: 787 fused frames in 516 s (1.5/s), fusion cost grew with map size (np.insert / carving of
  1.4 M voxels).

New pipeline (all tested on recordings / headless; NOT yet in the headset):
* src/tsdf.cpp (+ tools/tsdf.py, ctypes, auto-built to build/libpc.so): sparse 8^3-block TSDF, 8-byte
  voxels, projective update (one observation per voxel per frame), noise-adaptive band 3 sigma(Z) clamped
  [2 voxels, 12 cm], conservative carving with 3x3-min depth, block GC of spray, zero-crossing extraction
  with min_frames, raycast for picking (15 us).
  Hold-out test on live6 (fuse odd frames, test against even frames, re/holdout_eval.py):
     voxel fuser: floaters (point in front of measured surface > 3 sigma) 20.0 %
     TSDF 2 cm, min 5 frames: floaters 7.8 %, 2.7x fewer points (71k vs 193k for 100 pairs)
  fusion 25-40 ms/frame (was 100-300 ms and growing). Memory ~160 MB per 40k blocks.
* src/raw10.cpp: RAW10 -> 2x2-binned BGR in 4 ms (numpy unpack+demosaic was ~50 ms/pair); bit-exact.
* tools/pcserver.py rewritten: TSDF, min depth 0.35 m (fewer SGBM disparities), cv2 threads 3 (leave cores
  for XR), ~8 pairs/s on replay (was 1.5). Publishes tiles (pcpub PCLD0002), 30 Hz raycast of the right
  controller aim -> /dev/shm/pchit.json, status -> /dev/shm/pcstatus.json. Scan modes auto/paint/off.
* Viewer (viewer/pc_main.gd + cloud_tiles.gd + scene_model.gd): tools SCAN > PAINT > ANCHOR > BOX > PATH
  (right A). Placement on the scanned surface under the ray (green reticle) or free at the ray tip
  (orange, right stick = distance, stick click toggles snap). Boxes: 4 clicks (edge, width, height) or
  3 + grip = 4 cm panel (solar panels); level/plumb snapping within 7 deg; dimensions shown. Paths: clicks
  + grip, length shown. B = undo. Left: trigger alpha, X hide cloud, Y save (scene JSON + PLY with the
  same name in captures/sessions/), stick click pause scanning, grip hold 1 s clear cloud / 3 s also objects.
  Scene autosaved to captures/sessions/autosave_scene.json and reloaded at start (STAGE is persistent).
  Headless tests: godot --headless --path viewer --xr-mode off -s res://tests/test_headless.gd (13 ok).
* Export: tools/scene_export.py scene.json [--enu] -> .glb (cloud POINTS + colours unlit, boxes, path tubes,
  anchor empties; loads in Godot's glTF importer) + anchors CSV. tools/blender_import.py (run in Blender)
  builds editable curves (bevel = conduit diameter), empties, real-size boxes, coloured point mesh
  (UNTESTED: no Blender binary here). Example: captures/fused/live6/live6_demo.glb (full live6 TSDF cloud).
* GNSS: tools/gnssd.py --port /dev/ttyACM0 [--lever x y z]: NMEA GGA(+GST/RMC) over USB serial (termios,
  no pyserial), poses from src/posestream (read-only OpenVR background client, 100 Hz, same line format
  as hmd_500hz.txt), UTC->monotonic via min(t_rx - t_utc), 4-DoF fit ENU = Rz(yaw) M stage + t with outlier
  rejection, ok once horizontal spread >= 3 m. Raw NMEA logged to captures/gnss/ for reprocessing
  (--offline LOG --pose-file). tools/gnss_sim.py: RTK-like simulation on the live6 trajectory -> yaw
  37.007 vs 37.000 deg, absolute positions < 1 mm. Viewer shows fix/sats/hacc/georef rms; saved scenes
  carry the georef; export --enu uses it. tools/pcsession.sh --gnss PORT starts it.
  Hardware notes: any receiver that emits NMEA GGA works; GST gives real accuracy (u-blox: enable GST).
  RTK (F9P etc.) needed for cm-level; with plain GPS (2-5 m) the yaw needs tens of metres of walking.

"Paused / Resume" at start: most likely the SteamVR dashboard is open when the app starts (it is
launched from the desktop/terminal shown inside the dashboard), so SteamVR keeps the dashboard in front
and reports the app as paused until "Resume" closes the dashboard. Not an app bug; launching with a delay
(close the dashboard during the countdown) would avoid it.

Next in headset: tools/pcsession.sh --record captures/live7 ; try tools; compare judder/parallax.
