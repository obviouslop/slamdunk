// IVRBlockQueue_005 — internal SteamVR interface, signatures recovered from
// vrclient.so (CVRBlockQueue_004 thunks forward 1:1 to the _005 impl; arg counts
// verified from scratch-register allocation). See ../re/vtable.py.
#pragma once
#include "openvr.h"
namespace vr {
enum EBlockQueueError { BlockQueueError_None = 0 };
enum EBlockQueueReadType { BlockQueueRead_Latest = 0, BlockQueueRead_New = 1, BlockQueueRead_Next = 2 };
class IVRBlockQueue {
public:
  virtual int Create(PropertyContainerHandle_t* q, const char* path, uint32_t dataSize, uint32_t headerSize, uint32_t count, uint32_t flags) = 0;
  virtual int Connect(PropertyContainerHandle_t* q, const char* path) = 0;
  virtual int Destroy(PropertyContainerHandle_t q) = 0;
  virtual int AcquireWriteOnlyBlock(PropertyContainerHandle_t q, PropertyContainerHandle_t* b, void** buf) = 0;
  virtual int ReleaseWriteOnlyBlock(PropertyContainerHandle_t q, PropertyContainerHandle_t b) = 0;
  virtual int WaitAndAcquireReadOnlyBlock(PropertyContainerHandle_t q, PropertyContainerHandle_t* b, void** buf, EBlockQueueReadType t, uint32_t timeoutMs) = 0;
  virtual int AcquireReadOnlyBlock(PropertyContainerHandle_t q, PropertyContainerHandle_t* b, void** buf, EBlockQueueReadType t) = 0;
  virtual int ReleaseReadOnlyBlock(PropertyContainerHandle_t q, PropertyContainerHandle_t b) = 0;
  virtual int QueueHasReader(PropertyContainerHandle_t q, bool* has) = 0;
};
static const char* const IVRBlockQueue_Version = "IVRBlockQueue_005";
}
