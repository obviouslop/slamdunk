// Decoder for XRService's /xrservice/roomview3d/meshData block (reverse-engineered, 2026-11).
// Block layout (6000512 B): float32 xyz[17280] @0 ... 76-byte trailer @6000436:
//   float m34[12]   rigid transform (row-major 3x4)
//   float unk94     (≈94.255, constant; FOV deg?)
//   uint32 unk177, uint32 seq, uint32 0, uint32 unk50688
//   double t        seconds (XRService clock?)
#pragma once
#include <cstdint>
#include <cstring>
namespace roomview {
constexpr size_t kBlockSize = 6000512, kTrailer = 6000436, kMaxVerts = 17280;
struct Frame { const float* xyz; float m[3][4]; float unk94; uint32_t unk177, seq, unk50688; double t; };
inline Frame decode(const void* blk){
  const uint8_t* b=(const uint8_t*)blk; Frame f; f.xyz=(const float*)b;
  memcpy(f.m,b+kTrailer,48); memcpy(&f.unk94,b+kTrailer+48,4); memcpy(&f.unk177,b+kTrailer+52,4);
  memcpy(&f.seq,b+kTrailer+56,4); memcpy(&f.unk50688,b+kTrailer+64,4); memcpy(&f.t,b+kTrailer+68,8);
  return f;
}
}
