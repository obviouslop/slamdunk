// posestream: print the HMD pose (OpenVR Standing universe == OpenXR STAGE) as text lines
//   t valid qw qx qy qz px py pz      (t = CLOCK_MONOTONIC seconds; same format as arcgrab --pose-log)
// Read-only OpenVR background client, no cameras. Used by tools/gnssd.py to pair GNSS fixes with poses.
// usage: posestream [--hz 100] [--seconds 0]
// build: g++ -O2 -std=c++17 -o src/posestream src/posestream.cpp -Ithird_party/openvr -L/opt/steamvr/bin/linuxarm64 -lopenvr_api -Wl,-rpath,/opt/steamvr/bin/linuxarm64
#include "openvr.h"
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <unistd.h>

static volatile sig_atomic_t g_stop = 0;
static void mat_to_quat(const float m[3][4], double q[4]) {
  double tr = m[0][0] + m[1][1] + m[2][2];
  if (tr > 0) { double s = std::sqrt(tr + 1.0) * 2; q[0] = 0.25 * s; q[1] = (m[2][1] - m[1][2]) / s; q[2] = (m[0][2] - m[2][0]) / s; q[3] = (m[1][0] - m[0][1]) / s; }
  else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) { double s = std::sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2; q[0] = (m[2][1] - m[1][2]) / s; q[1] = 0.25 * s; q[2] = (m[0][1] + m[1][0]) / s; q[3] = (m[0][2] + m[2][0]) / s; }
  else if (m[1][1] > m[2][2]) { double s = std::sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2; q[0] = (m[0][2] - m[2][0]) / s; q[1] = (m[0][1] + m[1][0]) / s; q[2] = 0.25 * s; q[3] = (m[1][2] + m[2][1]) / s; }
  else { double s = std::sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2; q[0] = (m[1][0] - m[0][1]) / s; q[1] = (m[0][2] + m[2][0]) / s; q[2] = (m[1][2] + m[2][1]) / s; q[3] = 0.25 * s; }
}
int main(int argc, char** argv) {
  double hz = 100, seconds = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--hz") && i + 1 < argc) hz = atof(argv[++i]);
    else if (!strcmp(argv[i], "--seconds") && i + 1 < argc) seconds = atof(argv[++i]);
    else { fprintf(stderr, "usage: posestream [--hz 100] [--seconds 0]\n"); return 1; }
  }
  signal(SIGINT, [](int) { g_stop = 1; }); signal(SIGTERM, [](int) { g_stop = 1; }); signal(SIGPIPE, [](int) { g_stop = 1; });
  vr::EVRInitError ve; vr::IVRSystem* sys = vr::VR_Init(&ve, vr::VRApplication_Background);
  if (!sys) { fprintf(stderr, "OpenVR unavailable: %s\n", vr::VR_GetVRInitErrorAsEnglishDescription(ve)); return 2; }
  timespec t0; clock_gettime(CLOCK_MONOTONIC, &t0);
  double prev[4] = {1, 0, 0, 0};
  while (!g_stop) {
    vr::TrackedDevicePose_t tp;
    sys->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding, 0.0f, &tp, 1);
    timespec tn; clock_gettime(CLOCK_MONOTONIC, &tn);
    double t = tn.tv_sec + tn.tv_nsec * 1e-9;
    if (seconds > 0 && t - (t0.tv_sec + t0.tv_nsec * 1e-9) > seconds) break;
    double q[4]; mat_to_quat(tp.mDeviceToAbsoluteTracking.m, q);
    if (q[0] * prev[0] + q[1] * prev[1] + q[2] * prev[2] + q[3] * prev[3] < 0) for (double& v : q) v = -v;
    memcpy(prev, q, sizeof q);
    int valid = tp.bPoseIsValid && tp.eTrackingResult == vr::TrackingResult_Running_OK;
    const auto& m = tp.mDeviceToAbsoluteTracking.m;
    if (printf("%.6f %d %.7f %.7f %.7f %.7f %.5f %.5f %.5f\n", t, valid, q[0], q[1], q[2], q[3], m[0][3], m[1][3], m[2][3]) < 0) break;
    fflush(stdout);
    usleep(useconds_t(1e6 / hz));
  }
  vr::VR_Shutdown();
  return 0;
}
