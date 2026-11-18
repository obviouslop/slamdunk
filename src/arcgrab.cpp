// arcgrab: live capture from the Arcturus Vision IMX616 stereo pair via CAMSS RDI.
//
// Takes camera ownership once (CameraSwitch lock handoff -> XRService releases the colour
// cameras; expect a short head-tracking hiccup at start and end), streams both cameras,
// runs a simple auto-exposure via sensor register writes (only effective while streaming:
// arcimx616_set_ctrl skips writes when the sensor is not runtime-active), decodes RAW10
// (sensor sends 3264-px lines, 4080 B, no padding) and optionally saves previews / scans QR.
//
// Publishes every frame to /dev/shm/arcgrab (see ShmHdr) for consumers (tools/arcshm.py).
// Bayer: BGGR (verified visually on cropped 2464x2464 output).
// Poses: --pose-log FILE writes the 500 Hz history (t valid qw qx qy qz px py pz), and FILE.controllers the
// left/right controller poses from the same query (t role valid qw qx qy qz px py pz; role 1 = left, 2 = right).
// build: g++ -O2 -std=c++17 -o src/arcgrab src/arcgrab.cpp -Isrc -L$HOME/.local/share/Steam/steamrtarm64 -lopenvr_api -lpthread -Wl,-rpath,$HOME/.local/share/Steam/steamrtarm64
// usage: arcgrab [--seconds S] [--save-every N] [--out DIR] [--exp LINES] [--gain CODE]
//                [--no-ae] [--raw]
//
// Notes / hazards: see ../NOTES.md. Never use private ioctl 0x401056C5 (EEPROM write).
#include <fcntl.h>
#include "openvr.h"
#include <linux/videodev2.h>
#include <poll.h>
#include <sys/file.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <glob.h>
#include <atomic>
#include <mutex>
#include <thread>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

static std::atomic<bool> g_stop{false};

// ---- HMD pose history: sampled at ~500 Hz with prediction 0, stamped with CLOCK_MONOTONIC --------
// Frames are then posed by interpolating this history at their mid-exposure time (asking OpenVR
// for a negative prediction extrapolates instead of using history).
struct PoseSample { double t; double q[4]; double p[3]; bool valid; };   // q = (w,x,y,z)
static void mat_to_quat(const float m[3][4], double q[4]) {
  double tr = m[0][0] + m[1][1] + m[2][2];
  if (tr > 0) { double s = std::sqrt(tr + 1.0) * 2; q[0] = 0.25 * s; q[1] = (m[2][1] - m[1][2]) / s; q[2] = (m[0][2] - m[2][0]) / s; q[3] = (m[1][0] - m[0][1]) / s; }
  else if (m[0][0] > m[1][1] && m[0][0] > m[2][2]) { double s = std::sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2; q[0] = (m[2][1] - m[1][2]) / s; q[1] = 0.25 * s; q[2] = (m[0][1] + m[1][0]) / s; q[3] = (m[0][2] + m[2][0]) / s; }
  else if (m[1][1] > m[2][2]) { double s = std::sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2; q[0] = (m[0][2] - m[2][0]) / s; q[1] = (m[0][1] + m[1][0]) / s; q[2] = 0.25 * s; q[3] = (m[1][2] + m[2][1]) / s; }
  else { double s = std::sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2; q[0] = (m[1][0] - m[0][1]) / s; q[1] = (m[0][2] + m[2][0]) / s; q[2] = (m[1][2] + m[2][1]) / s; q[3] = 0.25 * s; }
}
static void quat_to_mat(const double q[4], const double p[3], float out[12]) {
  double w = q[0], x = q[1], y = q[2], z = q[3];
  double R[3][3] = {{1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)},
                    {2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)},
                    {2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)}};
  for (int r = 0; r < 3; r++) { for (int c = 0; c < 3; c++) out[r * 4 + c] = (float)R[r][c]; out[r * 4 + 3] = (float)p[r]; }
}
struct PoseHistory {
  static constexpr int N = 4096;
  PoseSample buf[N]; long head = 0; std::mutex mu; std::thread th; std::atomic<bool> run{false};
  FILE* log = nullptr;
  FILE* clog = nullptr;     // controllers: <log>.controllers (t role valid qw qx qy qz px py pz), role 1 = left, 2 = right
  void start(vr::IVRSystem* sys, const char* log_path) {
    if (log_path) { log = fopen(log_path, "w"); std::string cp = std::string(log_path) + ".controllers"; clog = fopen(cp.c_str(), "w"); }
    run = true;
    th = std::thread([this, sys] {
      int roles[vr::k_unMaxTrackedDeviceCount] = {0}; long iter = 0;
      while (run) {
        vr::TrackedDevicePose_t tps[vr::k_unMaxTrackedDeviceCount];
        const uint32_t nq = clog ? vr::k_unMaxTrackedDeviceCount : 1;
        sys->GetDeviceToAbsoluteTrackingPose(vr::TrackingUniverseStanding, 0.0f, tps, nq);
        vr::TrackedDevicePose_t& tp = tps[0];
        timespec tn; clock_gettime(CLOCK_MONOTONIC, &tn);
        if (clog) {
          if (iter++ % 250 == 0)       // re-check which devices are controllers twice a second
            for (uint32_t i = 1; i < vr::k_unMaxTrackedDeviceCount; i++)
              roles[i] = sys->GetTrackedDeviceClass(i) == vr::TrackedDeviceClass_Controller ? (int)sys->GetControllerRoleForTrackedDeviceIndex(i) : 0;
          for (uint32_t i = 1; i < nq; i++) {
            if (roles[i] != vr::TrackedControllerRole_LeftHand && roles[i] != vr::TrackedControllerRole_RightHand) continue;
            double q[4]; mat_to_quat(tps[i].mDeviceToAbsoluteTracking.m, q);
            const auto& m = tps[i].mDeviceToAbsoluteTracking.m;
            fprintf(clog, "%.6f %d %d %.7f %.7f %.7f %.7f %.5f %.5f %.5f\n", tn.tv_sec + tn.tv_nsec * 1e-9, roles[i],
                    (int)(tps[i].bPoseIsValid && tps[i].eTrackingResult == vr::TrackingResult_Running_OK), q[0], q[1], q[2], q[3], m[0][3], m[1][3], m[2][3]);
          }
        }
        PoseSample s; s.t = tn.tv_sec + tn.tv_nsec * 1e-9; s.valid = tp.bPoseIsValid && tp.eTrackingResult == vr::TrackingResult_Running_OK;
        mat_to_quat(tp.mDeviceToAbsoluteTracking.m, s.q);
        for (int i = 0; i < 3; i++) s.p[i] = tp.mDeviceToAbsoluteTracking.m[i][3];
        { std::lock_guard<std::mutex> l(mu);
          if (head > 0) { const PoseSample& pr = buf[(head - 1) % N]; double d = pr.q[0]*s.q[0]+pr.q[1]*s.q[1]+pr.q[2]*s.q[2]+pr.q[3]*s.q[3];
            if (d < 0) for (double& v : s.q) v = -v; }      // keep hemisphere continuity
          buf[head % N] = s; head++; }
        if (log) fprintf(log, "%.6f %d %.7f %.7f %.7f %.7f %.5f %.5f %.5f\n", s.t, (int)s.valid, s.q[0], s.q[1], s.q[2], s.q[3], s.p[0], s.p[1], s.p[2]);
        usleep(2000);
      }
    });
  }
  void stop() { run = false; if (th.joinable()) th.join(); if (log) fclose(log); log = nullptr; if (clog) fclose(clog); clog = nullptr; }
  bool at(double t, float out[12]) {   // interpolate (slerp/lerp); false if outside history or invalid
    std::lock_guard<std::mutex> l(mu);
    if (head < 2) return false;
    long lo = std::max(0L, head - N + 1), hi = head - 1;
    if (t < buf[lo % N].t || t > buf[hi % N].t + 0.004) return false;
    if (t >= buf[hi % N].t) { const PoseSample& s = buf[hi % N]; quat_to_mat(s.q, s.p, out); return s.valid; }
    while (hi - lo > 1) { long m = (lo + hi) / 2; if (buf[m % N].t <= t) lo = m; else hi = m; }
    const PoseSample &a = buf[lo % N], &b = buf[hi % N];
    double u = (t - a.t) / std::max(1e-9, b.t - a.t), q[4], p[3];
    double d = a.q[0]*b.q[0]+a.q[1]*b.q[1]+a.q[2]*b.q[2]+a.q[3]*b.q[3], th_ = std::acos(std::min(1.0, std::fabs(d)));
    double wa = 1 - u, wb = u;
    if (th_ > 1e-6) { wa = std::sin((1 - u) * th_) / std::sin(th_); wb = std::sin(u * th_) / std::sin(th_); }
    double n = 0; for (int i = 0; i < 4; i++) { q[i] = wa * a.q[i] + wb * b.q[i]; n += q[i] * q[i]; }
    n = std::sqrt(n); for (double& v : q) v /= n;
    for (int i = 0; i < 3; i++) p[i] = a.p[i] + u * (b.p[i] - a.p[i]);
    quat_to_mat(q, p, out); return a.valid && b.valid;
  }
};
static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static int lock_file(const char* path, double timeout_s) {
  int fd = open(path, O_RDWR | O_CREAT, 0644);
  if (fd < 0) { perror(path); return -1; }
  double t0 = now_s();
  while (flock(fd, LOCK_EX | LOCK_NB) != 0) {
    if (now_s() - t0 > timeout_s) { fprintf(stderr, "timeout locking %s\n", path); close(fd); return -1; }
    usleep(50000);
  }
  return fd;
}

static std::string subdev_for(const std::string& name) {
  glob_t g; std::string res;
  if (glob("/sys/class/video4linux/v4l-subdev*/name", 0, nullptr, &g) == 0) {
    for (size_t i = 0; i < g.gl_pathc; i++) {
      std::ifstream f(g.gl_pathv[i]); std::string n; std::getline(f, n);
      if (n == name) { std::string p = g.gl_pathv[i]; p = p.substr(0, p.rfind('/')); res = "/dev/" + p.substr(p.rfind('/') + 1); break; }
    }
    globfree(&g);
  }
  return res;
}

struct Sensor {
  int fd = -1;
  // 'brightness' ctrl == raw register write: value = (data << 16) | reg   (see NOTES.md)
  bool poke(uint16_t reg, uint8_t val) {
    v4l2_control c{V4L2_CID_BRIGHTNESS, (int)(((uint32_t)val << 16) | reg)};
    return ioctl(fd, VIDIOC_S_CTRL, &c) == 0;
  }
  bool set_exposure(int lines, int gain_code, int dgain = 256) {
    bool ok = poke(0x0104, 1);
    ok &= poke(0x0202, lines >> 8); ok &= poke(0x0203, lines & 0xff);
    ok &= poke(0x0204, gain_code >> 8); ok &= poke(0x0205, gain_code & 0xff);
    ok &= poke(0x020E, dgain >> 8); ok &= poke(0x020F, dgain & 0xff);
    ok &= poke(0x0104, 0);
    return ok;
  }
  // Digital crop of the 3264-wide binned line to the calibrated 2464-px window (factory cropRegion.x)
  bool set_crop(int x, int w) {
    bool ok = poke(0x0104, 1);
    ok &= poke(0x0408, x >> 8); ok &= poke(0x0409, x & 0xff);
    ok &= poke(0x040C, w >> 8); ok &= poke(0x040D, w & 0xff);
    ok &= poke(0x034C, w >> 8); ok &= poke(0x034D, w & 0xff);
    ok &= poke(0x0104, 0);
    return ok;
  }
};

struct Cam {
  std::string name, dev; int fd = -1;
  struct Buf { void* p; size_t len; };
  std::vector<Buf> bufs;
  uint32_t bytesperline = 0, sizeimage = 0;
  bool open_dev(int w, int h, int nb = 4) {
    fd = open(dev.c_str(), O_RDWR | O_NONBLOCK);
    if (fd < 0) { perror(dev.c_str()); return false; }
    v4l2_format f{}; f.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
    f.fmt.pix_mp.width = w; f.fmt.pix_mp.height = h; f.fmt.pix_mp.pixelformat = v4l2_fourcc('Y', '1', '0', 'P');
    f.fmt.pix_mp.num_planes = 1; f.fmt.pix_mp.field = V4L2_FIELD_NONE;
    if (ioctl(fd, VIDIOC_S_FMT, &f)) { perror("S_FMT"); return false; }
    bytesperline = f.fmt.pix_mp.plane_fmt[0].bytesperline; sizeimage = f.fmt.pix_mp.plane_fmt[0].sizeimage;
    v4l2_requestbuffers rb{}; rb.count = nb; rb.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE; rb.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_REQBUFS, &rb)) { perror("REQBUFS"); return false; }
    for (unsigned i = 0; i < rb.count; i++) {
      v4l2_plane pl{}; v4l2_buffer b{}; b.type = rb.type; b.memory = V4L2_MEMORY_MMAP; b.index = i; b.m.planes = &pl; b.length = 1;
      if (ioctl(fd, VIDIOC_QUERYBUF, &b)) { perror("QUERYBUF"); return false; }
      void* p = mmap(nullptr, pl.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, pl.m.mem_offset);
      if (p == MAP_FAILED) { perror("mmap"); return false; }
      bufs.push_back({p, pl.length});
      if (ioctl(fd, VIDIOC_QBUF, &b)) { perror("QBUF"); return false; }
    }
    return true;
  }
  bool start() { int t = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE; return ioctl(fd, VIDIOC_STREAMON, &t) == 0; }
  void stop() { int t = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE; ioctl(fd, VIDIOC_STREAMOFF, &t); for (auto& b : bufs) munmap(b.p, b.len); bufs.clear(); close(fd); fd = -1; }
  // returns buffer index or -1; fills bytesused / timestamp
  uint32_t last_flags = 0;
  int dq(uint32_t& used, double& ts) {
    v4l2_plane pl{}; v4l2_buffer b{}; b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE; b.memory = V4L2_MEMORY_MMAP; b.m.planes = &pl; b.length = 1;
    if (ioctl(fd, VIDIOC_DQBUF, &b)) return -1;
    used = pl.bytesused; ts = b.timestamp.tv_sec + b.timestamp.tv_usec * 1e-6; last_flags = b.flags; return b.index;
  }
  void q(int idx) {
    v4l2_plane pl{}; v4l2_buffer b{}; b.type = V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE; b.memory = V4L2_MEMORY_MMAP; b.index = idx; b.m.planes = &pl; b.length = 1;
    ioctl(fd, VIDIOC_QBUF, &b);
  }
};

static int LINE_PX = 3264, LINE_B = 4080; constexpr int BLACK = 64;   // set from format at runtime

struct Img16 { int rows = 0, cols = 0; std::vector<uint16_t> px; uint16_t* row(int r) { return px.data() + (size_t)r * cols; } };
// RAW10 MIPI-packed (4 px / 5 B) -> 16-bit, continuous 3264-px lines
static Img16 unpack(const uint8_t* src, size_t bytes, int skip_rows) {
  int rows = (int)(bytes / LINE_B);
  Img16 out; out.rows = rows - skip_rows; out.cols = LINE_PX; out.px.resize((size_t)out.rows * out.cols);
  for (int r = skip_rows; r < rows; r++) {
    const uint8_t* s = src + (size_t)r * LINE_B; uint16_t* d = out.row(r - skip_rows);
    for (int g = 0; g < LINE_PX / 4; g++, s += 5, d += 4) {
      uint8_t lo = s[4];
      d[0] = (s[0] << 2) | (lo & 3); d[1] = (s[1] << 2) | ((lo >> 2) & 3);
      d[2] = (s[2] << 2) | ((lo >> 4) & 3); d[3] = (s[3] << 2) | ((lo >> 6) & 3);
    }
  }
  return out;
}

// Shared-memory frame publisher: /dev/shm/arcgrab
//   header (4096 B) + 2 cameras x 2 slots x sizeimage packed RAW10.
//   Per camera: seq (odd while writing), slot, ts (CLOCK_MONOTONIC s), exp, gain, bytes.
struct ShmCam { volatile uint64_t seq; uint32_t slot, exp_lines, gain_code, bytes; double ts; uint64_t frame;
                float pose[12]; uint32_t pose_valid, buf_flags; uint32_t dgain, pad; };   // pose: HMD 3x4 row-major, Standing universe, at ts+pose_offset
struct ShmHdr { char magic[8]; uint32_t version, line_px, line_bytes, black, skip_rows, slot_bytes, ncams, pad; ShmCam cam[2]; };
static ShmHdr* g_shm = nullptr; static uint8_t* g_shm_data = nullptr; static size_t g_slot = 0;
static bool shm_open_pub(size_t slot_bytes) {
  int fd = open("/dev/shm/arcgrab", O_RDWR | O_CREAT | O_TRUNC, 0644); if (fd < 0) return false;
  size_t total = 4096 + 4 * slot_bytes; if (ftruncate(fd, total)) return false;
  void* p = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0); close(fd); if (p == MAP_FAILED) return false;
  g_shm = (ShmHdr*)p; g_shm_data = (uint8_t*)p + 4096; g_slot = slot_bytes; memset(g_shm, 0, sizeof(ShmHdr));
  memcpy(g_shm->magic, "ARCGRAB1", 8); g_shm->version = 2; g_shm->line_px = LINE_PX; g_shm->line_bytes = LINE_B;
  g_shm->black = 64; g_shm->skip_rows = 0; g_shm->slot_bytes = (uint32_t)slot_bytes; g_shm->ncams = 2; return true;
}
static void shm_publish(int k, const uint8_t* src, uint32_t bytes, double ts, int exp, int gain, uint64_t frame,
                        const float* pose, bool pose_valid, uint32_t flags, int dgain) {
  if (!g_shm) return;
  ShmCam& c = g_shm->cam[k]; uint32_t slot = (c.slot + 1) & 1;
  c.seq++; __sync_synchronize();
  memcpy(g_shm_data + (size_t)(k * 2 + slot) * g_slot, src, std::min<size_t>(bytes, g_slot));
  c.slot = slot; c.ts = ts; c.exp_lines = exp; c.gain_code = gain; c.bytes = bytes; c.frame = frame;
  memcpy(c.pose, pose, sizeof c.pose); c.pose_valid = pose_valid; c.buf_flags = flags; c.dgain = dgain;
  __sync_synchronize(); c.seq++;
}

int main(int argc, char** argv) {
  double seconds = 10; int save_every = 0; std::string out = "."; bool ae = true, raw = false;
  int exp_lines = 2000, gain = 512;   // gain code: Sony analog gain = 1024/(1024-code)
  int skip_rows = 48;                 // embedded-data lines at the top
  int fmt_w = 2464, fmt_h = 2464;     // V4L2 format (XRService default); sensor is re-cropped to match
  bool crop = true; int dgain = 256; bool use_pose = true; double pose_offset = 0.0; const char* pose_log = nullptr; int restore_exp = 1000; int max_exp = 3000; int nbufs = 4;
  for (int i = 1; i < argc; i++) {
    std::string a = argv[i];
    if (a == "--seconds") seconds = atof(argv[++i]);
    else if (a == "--save-every") save_every = atoi(argv[++i]);
    else if (a == "--out") out = argv[++i];
    else if (a == "--exp") exp_lines = atoi(argv[++i]);
    else if (a == "--gain") gain = atoi(argv[++i]);
    else if (a == "--no-ae") ae = false;
    else if (a == "--raw") raw = true;
    else if (a == "--no-crop") crop = false;
    else if (a == "--no-pose") use_pose = false;
    else if (a == "--pose-offset") pose_offset = atof(argv[++i]);
    else if (a == "--pose-log") pose_log = argv[++i];
    else if (a == "--restore-exp") restore_exp = atoi(argv[++i]);
    else if (a == "--max-exp") max_exp = atoi(argv[++i]);
    else if (a == "--nbufs") nbufs = atoi(argv[++i]);
    else if (a == "--stride") LINE_B = atoi(argv[++i]);
    else if (a == "--width") fmt_w = atoi(argv[++i]);
    else if (a == "--height") fmt_h = atoi(argv[++i]);
    else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 1; }
  }
  std::string self_dir; { char b[4096]; ssize_t n = readlink("/proc/self/exe", b, sizeof b - 1); if (n > 0) { b[n] = 0; self_dir = b; self_dir = self_dir.substr(0, self_dir.rfind('/')); } }
  signal(SIGINT, [](int) { g_stop = true; }); signal(SIGTERM, [](int) { g_stop = true; });

  int lk_rec = lock_file("/tmp/arcturus_camera_recorder.lock", 5);
  if (lk_rec < 0) return 2;
  int lk_pt = lock_file("/tmp/passthrough_camera.lock", 10);
  if (lk_pt < 0) { close(lk_rec); return 2; }
  fprintf(stderr, "camera ownership taken\n");
  usleep(1500000);  // let XRService finish its camera restart (it resets the media pad formats)
  {   // enable links + formats for our routes AFTER the handover (XRService resets them)
    char cmd[512]; snprintf(cmd, sizeof cmd, "%s/../tools/arcmode.sh %dx%d >/dev/null", self_dir.c_str(), fmt_w, fmt_h);
    if (system(cmd) != 0) fprintf(stderr, "warning: arcmode.sh failed\n");
  }

  // XRService naming: passthrough_left = /dev/video0, passthrough_right = /dev/video3
  Cam cams[2]; cams[0].name = "left"; cams[0].dev = "/dev/video0"; cams[1].name = "right"; cams[1].dev = "/dev/video3";
  Sensor sens[2];
  sens[0].fd = open(subdev_for("arcimx616 0-001a").c_str(), O_RDWR);
  sens[1].fd = open(subdev_for("arcimx616 0-0010").c_str(), O_RDWR);
  int rc = 0;
  for (auto& c : cams) if (!c.open_dev(fmt_w, fmt_h, nbufs)) rc = 3;
  if (!rc) for (auto& c : cams) if (!c.start()) { perror("STREAMON"); rc = 4; }
  if (!rc && fmt_w != 2464) { LINE_PX = fmt_w; LINE_B = cams[0].bytesperline; skip_rows = 0; }   // format matches sensor: use real stride
  if (!rc) fprintf(stderr, "streaming; buffer %u B (bpl %u) -> %u full 3264-px rows\n", cams[0].sizeimage, cams[0].bytesperline, cams[0].sizeimage / LINE_B);

  if (!rc && !shm_open_pub(cams[0].sizeimage)) { fprintf(stderr, "shm failed\n"); }
  vr::IVRSystem* vrsys = nullptr;
  if (use_pose) {
    vr::EVRInitError ve; vrsys = vr::VR_Init(&ve, vr::VRApplication_Background);
    if (!vrsys) fprintf(stderr, "OpenVR unavailable (%s): no poses\n", vr::VR_GetVRInitErrorAsEnglishDescription(ve));
  }
  static PoseHistory hist;
  if (vrsys) hist.start(vrsys, pose_log);
  double t0 = now_s(), last_ae = 0; long frames[2] = {0, 0}; bool exp_set = false; double ts_off[2] = {0, 0};
  while (!rc && !g_stop && now_s() - t0 < seconds) {
    pollfd p[2] = {{cams[0].fd, POLLIN, 0}, {cams[1].fd, POLLIN, 0}};
    int pr = poll(p, 2, 1000);
    if (pr == 0) { fprintf(stderr, "no frames for 1 s\n"); continue; }
    for (int k = 0; k < 2; k++) {
      if (!(p[k].revents & POLLIN)) continue;
      uint32_t used; double ts; int idx = cams[k].dq(used, ts);
      if (idx < 0) continue;
      frames[k]++;
      if (!exp_set) {   // first frame: sensors are active now -> writes take effect
        for (auto& s : sens) s.set_exposure(exp_lines, gain, dgain);
        if (crop) {   // left (0-001a) cropRegion.x=550, right (0-0010) x=250 (EEPROM calibration)
          sens[0].set_crop(550, 2464); sens[1].set_crop(250, 2464);
          LINE_PX = 2464; LINE_B = cams[0].bytesperline; skip_rows = 0;   // cropped lines land at the format stride (3088)
          if (g_shm) { g_shm->line_px = LINE_PX; g_shm->line_bytes = LINE_B; g_shm->skip_rows = 0; }
        }
        exp_set = true; fprintf(stderr, "initial exposure %d lines, gain code %d, crop %d (line %d px / %d B)\n", exp_lines, gain, crop, LINE_PX, LINE_B);
      }
      const uint8_t* src = (const uint8_t*)cams[k].bufs[idx].p;
      float pose[12] = {0}; bool pv = false;
      // V4L2 timestamps on this camss run in a shifted time base: ts - now_at_dequeue = +355.7 ms
      // (constant, both cams, measured 2026-11-17). Track the offset online as the minimum over recent
      // frames (dequeue is always after EOF; min rejects late dequeues) and correct ts with it.
      { timespec tn; clock_gettime(CLOCK_MONOTONIC, &tn); double mono = tn.tv_sec + tn.tv_nsec * 1e-9;
        // one shared offset for both cameras (per-camera minima differed by ~5-10 ms and broke L/R pairing)
        static double offs[128]; static int no = 0;
        offs[no % 128] = ts - mono; no++;
        double off = 1e9; for (int j = 0; j < std::min(no, 128); j++) off = std::min(off, offs[j]);
        ts_off[k] = off; ts -= off; }   // ts is now CLOCK_MONOTONIC end-of-frame (+ a few ms DMA/dequeue latency)
      if (vrsys) {   // HMD pose interpolated from the 500 Hz history at mid-exposure (+ calibrated offset)
        // Mid-exposure ~= EOF - readout/2 - exposure/2 (line time 4.36 us at ~31 fps, 2464 output lines).
        const double line_t = 4.36e-6, t_mid = ts - 2464 * line_t / 2 - exp_lines * line_t / 2;
        pv = hist.at(t_mid + pose_offset, pose);
        if (getenv("ARCGRAB_TSDEBUG") && frames[k] % 15 == 0) {
          timespec tn; clock_gettime(CLOCK_MONOTONIC, &tn); double mono = tn.tv_sec + tn.tv_nsec * 1e-9;
          fprintf(stderr, "TSDEBUG cam%d frame %ld: ts-now %+.1f ms, t_mid-now %+.1f ms, pose_valid %d\n", k, frames[k], (ts - mono) * 1e3, (t_mid - mono) * 1e3, (int)pv);
        }
      }
      shm_publish(k, src, cams[k].sizeimage, ts, exp_lines, gain, frames[k], pose, pv, cams[k].last_flags, dgain);
      bool want_save = save_every && frames[k] % save_every == 0;
      bool want_stats = (k == 0 && now_s() - last_ae > 0.25);
      if (want_save || want_stats) {
        Img16 r16 = unpack(src, cams[k].sizeimage, skip_rows);
        if (want_stats) {
          last_ae = now_s();
          double sum = 0; long n = 0, nsat = 0;
          for (int r = r16.rows / 4; r < 3 * r16.rows / 4; r += 2) { uint16_t* d = r16.row(r);
            for (int x = LINE_PX / 4; x < 3 * LINE_PX / 4; x += 2) { sum += d[x]; nsat += d[x] > 1000; n++; } }
          double mean = sum / n - BLACK, sat = (double)nsat / n;
          fprintf(stderr, "t=%5.1fs fps %.1f/%.1f mean %.0f sat %.2f%% exp %d gain %d dgain %d ts_off %.1f/%.1f ms\n", now_s() - t0,
                  frames[0] / (now_s() - t0), frames[1] / (now_s() - t0), mean, sat * 100, exp_lines, gain, dgain, ts_off[0] * 1e3, ts_off[1] * 1e3);
          if (ae) {   // proportional AE towards mean ~ 200 DN above black; exposure first, then gain
            double target = 200, ratio = std::max(0.25, std::min(4.0, target / std::max(mean, 1.0)));
            if (sat > 0.02) ratio = std::min(ratio, 0.7);
            if (std::abs(ratio - 1) > 0.08) {
              double e = exp_lines * ratio;
              if (e > max_exp) {   // exposure capped (sync safety, motion blur): analog gain, then digital gain
                double need = e / max_exp; e = max_exp;
                if (gain < 960) gain = (int)std::min(960.0, 1024 - (1024 - gain) / std::min(4.0, need));
                else dgain = (int)std::min(4095.0, dgain * std::min(4.0, need));
              } else if (e < 2000) {
                if (dgain > 256) dgain = std::max(256, (int)(dgain / 1.5));
                else if (gain > 0) gain = std::max(0, (int)(1024 - (1024 - gain) * 1.5));
              }
              exp_lines = std::max(16, (int)e);
              for (auto& s : sens) s.set_exposure(exp_lines, gain, dgain);
            }
          }
        }
        if (want_save) {
          char fn[512]; snprintf(fn, sizeof fn, "%s/%s_%06ld", out.c_str(), cams[k].name.c_str(), frames[k]);
          if (raw) { FILE* f = fopen((std::string(fn) + ".y10p").c_str(), "wb"); fwrite(src, 1, cams[k].sizeimage, f); fclose(f); }
          FILE* f = fopen((std::string(fn) + "_raw8.pgm").c_str(), "wb");   // 8-bit mosaic preview (gamma 2.2)
          fprintf(f, "P5\n%d %d\n255\n", r16.cols, r16.rows);
          std::vector<uint8_t> row(r16.cols);
          for (int r = 0; r < r16.rows; r++) { uint16_t* d = r16.row(r);
            for (int x = 0; x < r16.cols; x++) row[x] = (uint8_t)(255.0 * std::pow(std::max(0, d[x] - BLACK) / 959.0, 1 / 2.2));
            fwrite(row.data(), 1, row.size(), f); }
          fclose(f);
        }
      }
      cams[k].q(idx);
    }
  }
  // hand the sensors back the way XRService left them (full-mode crop x=0, 3264 wide)
  if (crop && exp_set) for (auto& s : sens) if (s.fd >= 0) s.set_crop(0, 3264);
  // also hand back a short exposure + unity gains: long integration (near frame length) appears to
  // drag the shared frame sync and leaves XRService's upper tracking cams in a desync/FPGA-reset loop
  if (exp_set) for (auto& s : sens) if (s.fd >= 0) s.set_exposure(restore_exp, 0, 256);
  usleep(100000);
  for (auto& c : cams) if (c.fd >= 0) c.stop();
  for (auto& s : sens) if (s.fd >= 0) close(s.fd);
  if (vrsys) { hist.stop(); vr::VR_Shutdown(); }
  flock(lk_pt, LOCK_UN); close(lk_pt); flock(lk_rec, LOCK_UN); close(lk_rec);
  fprintf(stderr, "released cameras; frames %ld/%ld in %.1fs\n", frames[0], frames[1], now_s() - t0);
  return rc;
}
