// Sparse block TSDF fusion for the live point cloud (tools/tsdf.py wraps it via ctypes).
//
// Projective integration (KinectFusion / voxblox style) of rectified stereo depth maps:
//   * blocks (8^3 voxels) are allocated around measured surfaces (Z +- truncation band)
//   * every voxel of every allocated block in the view frustum is projected into the depth map and
//     updated ONCE per frame:
//       measured surface clearly behind the voxel  -> free space (tsdf += +1)  = carving of spray
//       voxel within the truncation band          -> tsdf += (Zm - z) / band, colour update
//       voxel behind the measured surface         -> occluded, no update
//   * the truncation band follows the stereo noise: band = clamp(3 sigma(Z), trunc_min, trunc_max),
//     sigma(Z) = Z^2 / (f B) * sigma_disp
//   * extraction: zero crossings between face-neighbouring voxels that were both observed as surface
//     in >= min_frames frames -> one point per crossing (sub-voxel interpolated), colour of the voxel
//     closer to the surface.
// Coordinates: world = OpenVR Standing / OpenXR STAGE (metres). Rectified camera frame: x right,
// y down, z forward (OpenCV), pinhole f, cx, cy.
//
// build: g++ -O3 -std=c++17 -fPIC -shared -fopenmp -o build/libtsdf.so src/tsdf.cpp
#include <cmath>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>
#include <algorithm>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace {
constexpr int B = 8;               // voxels per block edge
constexpr int BV = B * B * B;

// 8 bytes per voxel
struct Voxel {
    int16_t ts;      // running weighted mean of the normalised signed distance * 32767 (+ = free side)
    uint16_t wq;     // accumulated weight * WQ (capped)
    uint8_t nsurf;   // number of frames that observed this voxel inside the band (saturates at 255)
    uint8_t r, g, b;
    float tsdf() const { return ts * (1.f / 32767.f); }
};
constexpr float WQ = 512.f;

inline uint64_t bkey(int x, int y, int z) {
    return (uint64_t(uint32_t(x + (1 << 20)) & 0x1FFFFF)) | (uint64_t(uint32_t(y + (1 << 20)) & 0x1FFFFF) << 21) |
           (uint64_t(uint32_t(z + (1 << 20)) & 0x1FFFFF) << 42);
}
inline void bunkey(uint64_t k, int& x, int& y, int& z) {
    x = int(k & 0x1FFFFF) - (1 << 20); y = int((k >> 21) & 0x1FFFFF) - (1 << 20); z = int((k >> 42) & 0x1FFFFF) - (1 << 20);
}
inline int ifloor(float v) { int i = int(v); return i - (v < float(i)); }

struct Map {
    float vox, tmin, tmax, wmax;
    float back_k = 0.f;     // >0: voxels behind the surface by (1, 1+back_k] bands get a 'solid' vote (-1)
    float back_w = 0.5f;    //     with this weight (removes stale back copies left by pose drift)
    float carve_w = 1.f;    // weight of a free-space vote
    int threads = 3;
    std::unordered_map<uint64_t, int> index;   // block key -> block number
    std::vector<uint64_t> keys;                // block number -> key
    std::vector<Voxel> vox_data;               // block number * BV + local
    std::vector<int> born;                     // block number -> frame of allocation
    int frame = 0;
    std::vector<float> out_xyz; std::vector<uint8_t> out_rgb; std::vector<float> out_w;

    int get_or_add(uint64_t k) {
        auto it = index.find(k);
        if (it != index.end()) return it->second;
        int id = int(keys.size());
        index.emplace(k, id); keys.push_back(k); born.push_back(frame);
        vox_data.resize(vox_data.size() + BV, Voxel{0, 0, 0, 0, 0, 0});
        return id;
    }
    const Voxel* find_voxel(int gx, int gy, int gz) const {
        int bx = gx >> 3, by = gy >> 3, bz = gz >> 3;   // arithmetic shift = floor div for B = 8
        auto it = index.find(bkey(bx, by, bz));
        if (it == index.end()) return nullptr;
        return &vox_data[size_t(it->second) * BV + ((gz & 7) * B + (gy & 7)) * B + (gx & 7)];
    }
};
}  // namespace

extern "C" {

// key: 0 = back_k, 1 = back_w, 2 = carve_w
void tsdf_set(void* p, int key, float val) {
    Map* m = static_cast<Map*>(p);
    if (key == 0) m->back_k = val; else if (key == 1) m->back_w = val; else if (key == 2) m->carve_w = val;
}

void* tsdf_new(float voxel, float trunc_min, float trunc_max, float wmax, int threads) {
    Map* m = new Map(); m->vox = voxel; m->tmin = trunc_min; m->tmax = trunc_max; m->wmax = wmax; m->threads = threads;
    m->index.reserve(1 << 16);
    return m;
}
void tsdf_free(void* p) { delete static_cast<Map*>(p); }
void tsdf_reset(void* p) {
    Map* m = static_cast<Map*>(p);
    m->index.clear(); m->keys.clear(); m->born.clear(); m->vox_data.clear(); m->vox_data.shrink_to_fit(); m->frame = 0;
}
int tsdf_nblocks(void* p) { return int(static_cast<Map*>(p)->keys.size()); }

// Z: HxW metric depth along the rectified optical axis (0 = invalid). rgb: HxWx3 uint8 (may be null).
// mask: HxW uint8, 0 = ignore pixel for surface updates/allocation (paint mode), may be null.
// world_from_rect: row-major 4x4. Returns number of voxel updates.
long tsdf_integrate(void* p, const float* Z, const uint8_t* rgb, const uint8_t* mask, int W, int H, float f, float cx, float cy,
                    const double* T, float fB, float sigma_disp, float max_depth, float carve_depth, int alloc_stride) {
    Map* m = static_cast<Map*>(p);
    const float v = m->vox;
    m->frame++;
    float R[9], t[3];
    for (int i = 0; i < 3; i++) { for (int j = 0; j < 3; j++) R[i * 3 + j] = float(T[i * 4 + j]); t[i] = float(T[i * 4 + 3]); }
    auto band = [&](float z) { float s = z * z / fB * sigma_disp; return std::min(m->tmax, std::max(m->tmin, 3.f * s)); };

    // 1) allocate blocks around measured surfaces
    for (int y = 0; y < H; y += alloc_stride)
        for (int x = 0; x < W; x += alloc_stride) {
            float z = Z[y * W + x];
            if (!(z > 0.f) || z > max_depth || (mask && !mask[y * W + x])) continue;
            float bd = band(z);
            float dx = (x - cx) / f, dy = (y - cy) / f;
            float dn = std::sqrt(dx * dx + dy * dy + 1.f);
            for (float s = -bd; s <= bd + 1e-6f; s += bd) {
                float zz = z + s / dn;
                float pc[3] = {dx * zz, dy * zz, zz};
                float pw[3];
                for (int i = 0; i < 3; i++) pw[i] = R[i * 3] * pc[0] + R[i * 3 + 1] * pc[1] + R[i * 3 + 2] * pc[2] + t[i];
                m->get_or_add(bkey(ifloor(pw[0] / (v * B)), ifloor(pw[1] / (v * B)), ifloor(pw[2] / (v * B))));
            }
        }

    // conservative depth for carving: min over 3x3 (do not carve foreground voxels next to depth edges)
    std::vector<float> Zmin(size_t(W) * H, 0.f);
    for (int y = 1; y < H - 1; y++)
        for (int x = 1; x < W - 1; x++) {
            float mn = 1e9f; bool any = false;
            for (int j = -1; j <= 1; j++)
                for (int i = -1; i <= 1; i++) {
                    float z = Z[(y + j) * W + x + i];
                    if (z > 0.f) { mn = std::min(mn, z); any = true; } else { mn = -1.f; }
                }
            Zmin[y * W + x] = (any && mn > 0.f) ? mn : 0.f;   // any invalid neighbour -> no carving
        }

    // 2) visible blocks
    float Rt[9];   // rect_from_world rotation = R^T
    for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) Rt[i * 3 + j] = R[j * 3 + i];
    float tr[3]; for (int i = 0; i < 3; i++) tr[i] = -(Rt[i * 3] * t[0] + Rt[i * 3 + 1] * t[1] + Rt[i * 3 + 2] * t[2]);
    const float bsz = v * B, brad = bsz * 0.8660254f;
    std::vector<int> vis;
    for (size_t b = 0; b < m->keys.size(); b++) {
        int bx, by, bz; bunkey(m->keys[b], bx, by, bz);
        float c[3] = {(bx + 0.5f) * bsz, (by + 0.5f) * bsz, (bz + 0.5f) * bsz};
        float pc[3];
        for (int i = 0; i < 3; i++) pc[i] = Rt[i * 3] * c[0] + Rt[i * 3 + 1] * c[1] + Rt[i * 3 + 2] * c[2] + tr[i];
        if (pc[2] < 0.05f - brad || pc[2] > carve_depth + brad) continue;
        float zc = std::max(pc[2], 0.05f);
        float u = f * pc[0] / zc + cx, w = f * pc[1] / zc + cy, r = f * brad / zc;
        if (u < -r || u > W + r || w < -r || w > H + r) continue;
        vis.push_back(int(b));
    }

    // 3) projective update, one observation per voxel
    long nupd = 0;
#ifdef _OPENMP
    omp_set_num_threads(m->threads);
#endif
#pragma omp parallel for schedule(dynamic, 16) reduction(+ : nupd)
    for (size_t vi = 0; vi < vis.size(); vi++) {
        int b = vis[vi];
        int bx, by, bz; bunkey(m->keys[b], bx, by, bz);
        Voxel* vb = &m->vox_data[size_t(b) * BV];
        for (int lz = 0; lz < B; lz++)
            for (int ly = 0; ly < B; ly++)
                for (int lx = 0; lx < B; lx++) {
                    float c[3] = {(bx * B + lx + 0.5f) * v, (by * B + ly + 0.5f) * v, (bz * B + lz + 0.5f) * v};
                    float pc[3];
                    for (int i = 0; i < 3; i++) pc[i] = Rt[i * 3] * c[0] + Rt[i * 3 + 1] * c[1] + Rt[i * 3 + 2] * c[2] + tr[i];
                    if (pc[2] < 0.1f) continue;
                    int u = int(std::lround(f * pc[0] / pc[2] + cx)), w = int(std::lround(f * pc[1] / pc[2] + cy));
                    if (u < 1 || u >= W - 1 || w < 1 || w >= H - 1) continue;
                    size_t pi = size_t(w) * W + u;
                    float zm = Z[pi];
                    if (!(zm > 0.f)) continue;
                    float dn = std::sqrt(pc[0] * pc[0] + pc[1] * pc[1] + pc[2] * pc[2]) / pc[2];
                    float sd = (zm - pc[2]) * dn;          // metric distance along the ray, + = in front of surface
                    float bd = band(zm);
                    Voxel& vx = vb[(lz * B + ly) * B + lx];
                    float obs, wo;
                    bool surf = false;
                    if (sd > bd) {                          // free space
                        float zc = Zmin[pi];
                        if (!(zc > 0.f) || pc[2] > carve_depth || (zc - pc[2]) * dn <= band(zc)) continue;
                        obs = 1.f; wo = m->carve_w;
                    } else if (sd >= -bd) {                 // inside the band
                        if (zm > max_depth || (mask && !mask[pi])) continue;
                        obs = sd / bd; wo = std::min(1.f, 1.f / (zm * zm)); surf = true;
                    } else if (m->back_k > 0.f && sd >= -bd * (1.f + m->back_k) && zm <= max_depth && !(mask && !mask[pi])) {
                        // just behind the measured surface: solid. Only touches voxels that currently claim to be on
                        // the free side, so a real surface further back (seen by other views) is not eroded.
                        if (vx.wq == 0 || vx.tsdf() <= 0.f) continue;
                        obs = -1.f; wo = m->back_w * std::min(1.f, 1.f / (zm * zm));
                    } else {
                        continue;                           // occluded
                    }
                    float w0 = vx.wq * (1.f / WQ), wn = w0 + wo;
                    float tn = (vx.tsdf() * w0 + obs * wo) / wn;
                    vx.ts = int16_t(std::lround(std::max(-1.f, std::min(1.f, tn)) * 32767.f));
                    vx.wq = uint16_t(std::min(std::min(wn, m->wmax) * WQ, 65535.f));
                    if (surf) {
                        if (vx.nsurf < 255) vx.nsurf++;
                        if (rgb && std::fabs(sd) < 1.5f * v) {
                            const uint8_t* c3 = rgb + pi * 3;
                            float a = 1.f / std::min<float>(vx.nsurf, 16.f);
                            if (vx.nsurf == 1 || (vx.r | vx.g | vx.b) == 0) { vx.r = c3[0]; vx.g = c3[1]; vx.b = c3[2]; }
                            else {
                                vx.r = uint8_t(vx.r + a * (c3[0] - vx.r) + 0.5f); vx.g = uint8_t(vx.g + a * (c3[1] - vx.g) + 0.5f);
                                vx.b = uint8_t(vx.b + a * (c3[2] - vx.b) + 0.5f);
                            }
                        }
                    }
                    nupd++;
                }
    }
    return nupd;
}

// Surface points at zero crossings. Returns the number of points (copy with tsdf_get).
int tsdf_extract(void* p, int min_frames, float min_weight) {
    Map* m = static_cast<Map*>(p);
    const float v = m->vox;
    m->out_xyz.clear(); m->out_rgb.clear(); m->out_w.clear();
    const uint16_t wmin = uint16_t(std::min(min_weight * WQ, 65535.f));
    auto ok = [&](const Voxel* q) { return q && q->nsurf >= min_frames && q->wq >= wmin; };
    for (size_t b = 0; b < m->keys.size(); b++) {
        int bx, by, bz; bunkey(m->keys[b], bx, by, bz);
        const Voxel* vb = &m->vox_data[b * BV];
        for (int lz = 0; lz < B; lz++)
            for (int ly = 0; ly < B; ly++)
                for (int lx = 0; lx < B; lx++) {
                    const Voxel* a = &vb[(lz * B + ly) * B + lx];
                    if (!ok(a)) continue;
                    int gx = bx * B + lx, gy = by * B + ly, gz = bz * B + lz;
                    const int nb[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
                    for (int d = 0; d < 3; d++) {
                        int hx = gx + nb[d][0], hy = gy + nb[d][1], hz = gz + nb[d][2];
                        const Voxel* c = (lx + nb[d][0] < B && ly + nb[d][1] < B && lz + nb[d][2] < B)
                                             ? &vb[((lz + nb[d][2]) * B + ly + nb[d][1]) * B + lx + nb[d][0]]
                                             : m->find_voxel(hx, hy, hz);
                        if (!ok(c)) continue;
                        float ta = a->tsdf(), tc = c->tsdf();
                        if ((ta >= 0.f) == (tc >= 0.f)) continue;
                        float s = ta / (ta - tc);
                        const Voxel* col = std::fabs(ta) <= std::fabs(tc) ? a : c;
                        if ((col->r | col->g | col->b) == 0) col = (col == a) ? c : a;
                        m->out_xyz.push_back((gx + 0.5f + s * nb[d][0]) * v);
                        m->out_xyz.push_back((gy + 0.5f + s * nb[d][1]) * v);
                        m->out_xyz.push_back((gz + 0.5f + s * nb[d][2]) * v);
                        m->out_rgb.push_back(col->r); m->out_rgb.push_back(col->g); m->out_rgb.push_back(col->b);
                        m->out_w.push_back(std::min(a->wq, c->wq) * (1.f / WQ));
                    }
                }
    }
    return int(m->out_w.size());
}

// Drop blocks older than min_age frames in which no voxel was ever seen as surface >= min_nsurf times
// (spray / one-off outliers). Keeps block order (deterministic extraction). Returns blocks removed.
int tsdf_gc(void* p, int min_age, int min_nsurf) {
    Map* m = static_cast<Map*>(p);
    size_t nb = m->keys.size(), out = 0;
    for (size_t b = 0; b < nb; b++) {
        bool keep = m->born[b] > m->frame - min_age;
        const Voxel* vb = &m->vox_data[b * BV];
        for (int i = 0; i < BV && !keep; i++) keep = vb[i].nsurf >= min_nsurf;
        if (!keep) continue;
        if (out != b) {
            m->keys[out] = m->keys[b]; m->born[out] = m->born[b];
            std::memcpy(&m->vox_data[out * BV], vb, sizeof(Voxel) * BV);
        }
        out++;
    }
    int removed = int(nb - out);
    if (removed) {
        m->keys.resize(out); m->born.resize(out); m->vox_data.resize(out * BV);
        m->index.clear();
        for (size_t b = 0; b < out; b++) m->index.emplace(m->keys[b], int(b));
    }
    return removed;
}

// First surface crossing (free -> occupied) along a ray from o in direction d (unit), within [tmin, tmax].
// out[0..2] = point, out[3..5] = outward normal (towards free space), out[6] = distance. Returns 1 on hit.
int tsdf_raycast(void* p, const float* o, const float* d, float tmin, float tmax, int min_frames, float* out) {
    Map* m = static_cast<Map*>(p);
    const float v = m->vox, step = 0.5f * v;
    auto at = [&](float x, float y, float z) { return m->find_voxel(ifloor(x / v), ifloor(y / v), ifloor(z / v)); };
    float tprev = -1.f, sprev = 0.f;
    for (float t = tmin; t <= tmax; t += step) {
        float x = o[0] + d[0] * t, y = o[1] + d[1] * t, z = o[2] + d[2] * t;
        const Voxel* q = at(x, y, z);
        if (!q || q->wq == 0) { tprev = -1.f; continue; }
        float sd = q->tsdf();
        if (sd < 0.f && tprev >= 0.f && sprev >= 0.f && q->nsurf >= min_frames) {
            float th = tprev + (t - tprev) * sprev / (sprev - sd);
            float px = o[0] + d[0] * th, py = o[1] + d[1] * th, pz = o[2] + d[2] * th;
            int gx = ifloor(px / v), gy = ifloor(py / v), gz = ifloor(pz / v);
            auto ts = [&](int a, int b, int c) { const Voxel* r = m->find_voxel(a, b, c); return (r && r->wq) ? r->tsdf() : 0.f; };
            float n[3] = {ts(gx + 1, gy, gz) - ts(gx - 1, gy, gz), ts(gx, gy + 1, gz) - ts(gx, gy - 1, gz), ts(gx, gy, gz + 1) - ts(gx, gy, gz - 1)};
            float nn = std::sqrt(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            if (nn < 1e-6f) { n[0] = -d[0]; n[1] = -d[1]; n[2] = -d[2]; nn = 1.f; }
            out[0] = px; out[1] = py; out[2] = pz;
            out[3] = n[0] / nn; out[4] = n[1] / nn; out[5] = n[2] / nn; out[6] = th;
            return 1;
        }
        tprev = t; sprev = sd;
    }
    return 0;
}

void tsdf_get(void* p, float* xyz, uint8_t* rgb, float* w) {
    Map* m = static_cast<Map*>(p);
    std::memcpy(xyz, m->out_xyz.data(), m->out_xyz.size() * sizeof(float));
    std::memcpy(rgb, m->out_rgb.data(), m->out_rgb.size());
    if (w) std::memcpy(w, m->out_w.data(), m->out_w.size() * sizeof(float));
}

// ---------------------------------------------------------------- drift correction (frame-to-model)
// Trilinear TSDF sample at world point (x,y,z): value (normalised, -1..1) and gradient (per metre).
// Returns false unless all 8 neighbours were observed as surface >= min_nsurf times.
static bool sample(const Map* m, float x, float y, float z, int min_nsurf, float& val, float g[3]) {
    const float v = m->vox;
    float qx = x / v - 0.5f, qy = y / v - 0.5f, qz = z / v - 0.5f;
    int ix = ifloor(qx), iy = ifloor(qy), iz = ifloor(qz);
    float fx = qx - ix, fy = qy - iy, fz = qz - iz;
    float c[8];
    for (int k = 0; k < 8; k++) {
        const Voxel* q = m->find_voxel(ix + (k & 1), iy + ((k >> 1) & 1), iz + ((k >> 2) & 1));
        if (!q || q->wq == 0 || q->nsurf < min_nsurf) return false;
        c[k] = q->tsdf();
    }
    auto L = [](float a, float b, float t) { return a + (b - a) * t; };
    float c00 = L(c[0], c[1], fx), c10 = L(c[2], c[3], fx), c01 = L(c[4], c[5], fx), c11 = L(c[6], c[7], fx);
    float c0 = L(c00, c10, fy), c1 = L(c01, c11, fy);
    val = L(c0, c1, fz);
    // gradient
    float dx0 = L(c[1] - c[0], c[3] - c[2], fy), dx1 = L(c[5] - c[4], c[7] - c[6], fy);
    g[0] = L(dx0, dx1, fz) / v;
    float dy0 = L(c[2] - c[0], c[3] - c[1], fx), dy1 = L(c[6] - c[4], c[7] - c[5], fx);
    g[1] = L(dy0, dy1, fz) / v;
    g[2] = (c1 - c0) / v;
    return true;
}

static void solve6(double A[36], double b[6], double x[6]) {   // Gaussian elimination with partial pivoting
    double M[6][7];
    for (int i = 0; i < 6; i++) { for (int j = 0; j < 6; j++) M[i][j] = A[i * 6 + j]; M[i][6] = b[i]; }
    for (int c = 0; c < 6; c++) {
        int p = c; for (int r = c + 1; r < 6; r++) if (std::fabs(M[r][c]) > std::fabs(M[p][c])) p = r;
        for (int j = 0; j < 7; j++) std::swap(M[c][j], M[p][j]);
        if (std::fabs(M[c][c]) < 1e-12) { for (int i = 0; i < 6; i++) x[i] = 0; return; }
        for (int r = c + 1; r < 6; r++) { double f = M[r][c] / M[c][c]; for (int j = c; j < 7; j++) M[r][j] -= f * M[c][j]; }
    }
    for (int i = 5; i >= 0; i--) { double s = M[i][6]; for (int j = i + 1; j < 6; j++) s -= M[i][j] * x[j]; x[i] = s / M[i][i]; }
}

// Align a depth frame to the map: refine world_from_rect T (3x4 row-major doubles, in/out) by Gauss-Newton on the
// TSDF (point-to-implicit-surface, Huber), damped towards zero motion (prior_rot rad, prior_trans m = 1-sigma).
// out[0] = number of residuals used, out[1] = rms residual (m) after, out[2] = |rotation| (rad), out[3] = |translation| (m)
// Returns 1 if T was updated.
int tsdf_align(void* p, const float* Z, int W, int H, float f, float cx, float cy, double* T, float fB, float sigma_disp,
               float max_depth, int stride, int iters, int min_nsurf, float prior_rot, float prior_trans,
               const double* Tp, float abs_rot, float abs_trans, float* out) {
    // Tp (3x4, optional): raw tracking pose; total correction relative to it is pulled to zero with 1-sigma
    // abs_rot (yaw, rad) / abs_trans (m), so a small systematic bias cannot make the map creep.
    Map* m = static_cast<Map*>(p);
    std::vector<float> P; P.reserve(size_t(W / stride + 1) * (H / stride + 1) * 4);
    for (int y = 0; y < H; y += stride)
        for (int x = 0; x < W; x += stride) {
            float z = Z[y * W + x];
            if (!(z > 0.f) || z > max_depth) continue;
            float s = z * z / fB * sigma_disp, bd = std::min(m->tmax, std::max(m->tmin, 3.f * s));
            P.push_back((x - cx) / f * z); P.push_back((y - cy) / f * z); P.push_back(z); P.push_back(bd);
        }
    size_t n = P.size() / 4;
    double R[9], t[3];
    for (int i = 0; i < 3; i++) { for (int j = 0; j < 3; j++) R[i * 3 + j] = T[i * 4 + j]; t[i] = T[i * 4 + 3]; }
    double tot_w[3] = {0, 0, 0}, tot_v[3] = {0, 0, 0};   // accumulated rotation vector / translation
    double off_v[3] = {0, 0, 0}, off_yaw = 0;           // start pose minus tracking pose (camera centre, yaw)
    if (Tp) {
        for (int i = 0; i < 3; i++) off_v[i] = t[i] - Tp[i * 4 + 3];
        // yaw of R Rp^T about world y: from its (0,2)/(2,2)... use the x-z part of the rotated forward axis
        double a = 0, c = 0;   // sum over axes of sin/cos of the yaw difference
        for (int k = 0; k < 3; k++) {
            double ux = R[0 * 3 + k], uz = R[2 * 3 + k], px = Tp[0 * 4 + k], pz = Tp[2 * 4 + k];
            a += pz * ux - px * uz; c += px * ux + pz * uz;
        }
        off_yaw = std::atan2(a, c);
    }
    int used = 0; double rms = 0;
    for (int it = 0; it < iters; it++) {
        double A[36] = {0}, b[6] = {0}, sr = 0; int cnt = 0;
#ifdef _OPENMP
        omp_set_num_threads(m->threads);
#endif
#pragma omp parallel
        {
            double Al[36] = {0}, bl[6] = {0}, srl = 0; int cl = 0;
#pragma omp for schedule(static)
            for (long k = 0; k < long(n); k++) {
                const float* q = &P[k * 4];
                double pw[3], pc[3];
                for (int i = 0; i < 3; i++) pw[i] = R[i * 3] * q[0] + R[i * 3 + 1] * q[1] + R[i * 3 + 2] * q[2] + t[i];
                float val, g[3];
                if (!sample(m, float(pw[0]), float(pw[1]), float(pw[2]), min_nsurf, val, g)) continue;
                if (std::fabs(val) > 0.9f) continue;                 // saturated: outside the band
                double bd = q[3];
                double r = val * bd;                                  // metric residual (approx.)
                double gm[3] = {g[0] * bd, g[1] * bd, g[2] * bd};     // metric gradient (~unit normal)
                double gn = std::sqrt(gm[0] * gm[0] + gm[1] * gm[1] + gm[2] * gm[2]);
                if (gn < 0.3 || gn > 3.0) continue;
                for (int i = 0; i < 3; i++) pc[i] = pw[i] - t[i];    // relative to the camera centre
                double J[6] = {pc[1] * gm[2] - pc[2] * gm[1], pc[2] * gm[0] - pc[0] * gm[2], pc[0] * gm[1] - pc[1] * gm[0], gm[0], gm[1], gm[2]};
                double hub = 0.02, w = std::fabs(r) < hub ? 1.0 : hub / std::fabs(r);
                w /= (q[2] * q[2]);                                   // depth noise ~ Z^2
                for (int i = 0; i < 6; i++) { bl[i] -= w * J[i] * r; for (int j = 0; j < 6; j++) Al[i * 6 + j] += w * J[i] * J[j]; }
                srl += r * r; cl++;
            }
#pragma omp critical
            { for (int i = 0; i < 36; i++) A[i] += Al[i]; for (int i = 0; i < 6; i++) b[i] += bl[i]; sr += srl; cnt += cl; }
        }
        used = cnt; rms = cnt ? std::sqrt(sr / cnt) : 0;
        if (cnt < 200) return 0;
        // prior: total motion ~ N(0, prior); normalise by the data scale so it acts like `cnt` pseudo-observations
        double scale = 0; for (int i = 0; i < 6; i++) scale += A[i * 6 + i]; scale /= 6.0;
        for (int i = 0; i < 3; i++) {
            double lr = scale * 1e-4 / (prior_rot * prior_rot), lt = scale * 1e-4 / (prior_trans * prior_trans);
            A[i * 6 + i] += lr; b[i] -= lr * tot_w[i];
            A[(i + 3) * 6 + i + 3] += lt; b[i + 3] -= lt * tot_v[i];
        }
        if (Tp) {
            double la = scale * 1e-4 / (abs_trans * abs_trans), lr = scale * 1e-4 / (abs_rot * abs_rot);
            for (int i = 0; i < 3; i++) { A[(i + 3) * 6 + i + 3] += la; b[i + 3] -= la * (tot_v[i] + off_v[i]); }
            A[1 * 6 + 1] += lr; b[1] -= lr * (tot_w[1] + off_yaw);
        }
        // gravity is known from tracking: no roll/pitch correction (4 DoF: yaw about world y + translation)
        A[0 * 6 + 0] += scale * 1e6; A[2 * 6 + 2] += scale * 1e6;
        double x[6]; solve6(A, b, x);
        // apply: p' = Rw (p - t) + t + v  ->  R <- Rw R,  t <- t + v
        double th = std::sqrt(x[0] * x[0] + x[1] * x[1] + x[2] * x[2]);
        double Rw[9] = {1, 0, 0, 0, 1, 0, 0, 0, 1};
        if (th > 1e-12) {
            double k[3] = {x[0] / th, x[1] / th, x[2] / th}, c = std::cos(th), s = std::sin(th), C = 1 - c;
            Rw[0] = c + k[0] * k[0] * C; Rw[1] = k[0] * k[1] * C - k[2] * s; Rw[2] = k[0] * k[2] * C + k[1] * s;
            Rw[3] = k[1] * k[0] * C + k[2] * s; Rw[4] = c + k[1] * k[1] * C; Rw[5] = k[1] * k[2] * C - k[0] * s;
            Rw[6] = k[2] * k[0] * C - k[1] * s; Rw[7] = k[2] * k[1] * C + k[0] * s; Rw[8] = c + k[2] * k[2] * C;
        }
        double Rn[9];
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) Rn[i * 3 + j] = Rw[i * 3] * R[j] + Rw[i * 3 + 1] * R[3 + j] + Rw[i * 3 + 2] * R[6 + j];
        std::memcpy(R, Rn, sizeof R);
        for (int i = 0; i < 3; i++) { t[i] += x[3 + i]; tot_w[i] += x[i]; tot_v[i] += x[3 + i]; }
        if (th < 1e-5 && std::sqrt(x[3] * x[3] + x[4] * x[4] + x[5] * x[5]) < 1e-5) break;
    }
    for (int i = 0; i < 3; i++) { for (int j = 0; j < 3; j++) T[i * 4 + j] = R[i * 3 + j]; T[i * 4 + 3] = t[i]; }
    out[0] = float(used); out[1] = float(rms);
    out[2] = float(std::sqrt(tot_w[0] * tot_w[0] + tot_w[1] * tot_w[1] + tot_w[2] * tot_w[2]));
    out[3] = float(std::sqrt(tot_v[0] * tot_v[0] + tot_v[1] * tot_v[1] + tot_v[2] * tot_v[2]));
    return 1;
}

}  // extern "C"
