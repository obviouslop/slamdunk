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
                        obs = 1.f; wo = 1.f;
                    } else if (sd >= -bd) {                 // inside the band
                        if (zm > max_depth || (mask && !mask[pi])) continue;
                        obs = sd / bd; wo = std::min(1.f, 1.f / (zm * zm)); surf = true;
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

void tsdf_get(void* p, float* xyz, uint8_t* rgb, float* w) {
    Map* m = static_cast<Map*>(p);
    std::memcpy(xyz, m->out_xyz.data(), m->out_xyz.size() * sizeof(float));
    std::memcpy(rgb, m->out_rgb.data(), m->out_rgb.size());
    if (w) std::memcpy(w, m->out_w.data(), m->out_w.size() * sizeof(float));
}

}  // extern "C"
