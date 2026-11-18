// RAW10 (MIPI packed, BGGR) -> 2x2-binned half-resolution BGR float, black level removed.
// Replaces the numpy unpack + LiveStereo._rgb (about 50 ms per pair -> a few ms).
#include <cstdint>
#include <algorithm>
extern "C" void pc_bin_raw10(const uint8_t* buf, int W, int H, int stride, int skip, float black, float* out, int threads) {
    (void)threads;
    const int Wo = W / 2, Ho = H / 2;
#pragma omp parallel for schedule(static) num_threads(threads > 0 ? threads : 1)
    for (int y = 0; y < Ho; y++) {
        const uint8_t* r0 = buf + size_t(skip + 2 * y) * stride;
        const uint8_t* r1 = r0 + stride;
        float* o = out + size_t(y) * Wo * 3;
        for (int g = 0; g < W / 4; g++) {        // 4 pixels per 5 bytes -> 2 output pixels
            const uint8_t* a = r0 + g * 5; const uint8_t* b = r1 + g * 5;
            int a0 = (a[0] << 2) | (a[4] & 3), a1 = (a[1] << 2) | ((a[4] >> 2) & 3), a2 = (a[2] << 2) | ((a[4] >> 4) & 3), a3 = (a[3] << 2) | ((a[4] >> 6) & 3);
            int b0 = (b[0] << 2) | (b[4] & 3), b1 = (b[1] << 2) | ((b[4] >> 2) & 3), b2 = (b[2] << 2) | ((b[4] >> 4) & 3), b3 = (b[3] << 2) | ((b[4] >> 6) & 3);
            float* p = o + g * 6;
            p[0] = std::max(0.f, a0 - black); p[1] = std::max(0.f, 0.5f * (a1 + b0) - black); p[2] = std::max(0.f, b1 - black);
            p[3] = std::max(0.f, a2 - black); p[4] = std::max(0.f, 0.5f * (a3 + b2) - black); p[5] = std::max(0.f, b3 - black);
        }
    }
}
