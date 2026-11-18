// Record roomview3d meshData frames (deduped by seq) to a binary file:
// per frame: uint32 seq, double t, float m[12], float xyz[17280*3]  (also HMD raw pose 3x4)
#include "blockqueue.h"
#include "roomview.h"
#include <chrono>
#include <cstdio>
using namespace vr;
int main(int argc,char**argv){
  if(argc<3){ printf("usage: rvrec out.bin seconds\n"); return 1; }
  EVRInitError e; IVRSystem* sys=VR_Init(&e,VRApplication_Background); if(e) return 1;
  auto* bq=(IVRBlockQueue*)VR_GetGenericInterface(IVRBlockQueue_Version,&e);
  PropertyContainerHandle_t q; if(bq->Connect(&q,"/xrservice/roomview3d/meshData")) return 2;
  FILE* out=fopen(argv[1],"wb"); double secs=atof(argv[2]); uint32_t best=0; int n=0;
  auto t0=std::chrono::steady_clock::now();
  while(std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()<secs){
    PropertyContainerHandle_t b; void* buf;
    if(bq->WaitAndAcquireReadOnlyBlock(q,&b,&buf,BlockQueueRead_Latest,100)) continue;
    auto f=roomview::decode(buf);
    if(f.seq>best){ best=f.seq; n++;
      TrackedDevicePose_t hr; sys->GetDeviceToAbsoluteTrackingPose(TrackingUniverseRawAndUncalibrated,0,&hr,1);
      fwrite(&f.seq,4,1,out); fwrite(&f.t,8,1,out); fwrite(f.m,4,12,out); fwrite(hr.mDeviceToAbsoluteTracking.m,4,12,out);
      fwrite(f.xyz,4,roomview::kMaxVerts*3,out); }
    bq->ReleaseReadOnlyBlock(q,b);
  }
  fclose(out); printf("recorded %d frames, last seq %u\n",n,best); VR_Shutdown();
}
