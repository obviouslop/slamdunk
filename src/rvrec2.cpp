// Record roomview frames + HMD (raw universe) poses at look-back offsets relative to frame t.
// per frame: u32 seq, f64 t, f32 trailer[12], f32 hmd[13][12] (offsets 0..-120ms step 10), f32 xyz[17280*3]
#include "blockqueue.h"
#include "roomview.h"
#include <chrono>
#include <ctime>
#include <cstdio>
using namespace vr;
static double mono(){ timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
int main(int argc,char**argv){
  EVRInitError e; IVRSystem* sys=VR_Init(&e,VRApplication_Background); if(e) return 1;
  auto* bq=(IVRBlockQueue*)VR_GetGenericInterface(IVRBlockQueue_Version,&e);
  PropertyContainerHandle_t q; if(bq->Connect(&q,"/xrservice/roomview3d/meshData")) return 2;
  FILE* out=fopen(argv[1],"wb"); double secs=atof(argv[2]); uint32_t best=0; int n=0; double t0=mono();
  while(mono()-t0<secs){
    PropertyContainerHandle_t b; void* buf;
    if(bq->WaitAndAcquireReadOnlyBlock(q,&b,&buf,BlockQueueRead_Latest,100)) continue;
    auto f=roomview::decode(buf);
    if(f.seq>best && f.t>0){ best=f.seq; n++;
      fwrite(&f.seq,4,1,out); fwrite(&f.t,8,1,out); fwrite(f.m,4,12,out);
      double now=mono();
      for(int k=0;k<13;k++){ TrackedDevicePose_t p; sys->GetDeviceToAbsoluteTrackingPose(TrackingUniverseRawAndUncalibrated,(float)(f.t-now-0.010*k),&p,1); fwrite(p.mDeviceToAbsoluteTracking.m,4,12,out); }
      fwrite(f.xyz,4,roomview::kMaxVerts*3,out); }
    bq->ReleaseReadOnlyBlock(q,b);
  }
  fclose(out); printf("recorded %d frames\n",n); VR_Shutdown();
}
