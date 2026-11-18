#include "blockqueue.h"
#include "roomview.h"
#include <ctime>
#include <cstdio>
using namespace vr;
static double clk(clockid_t c){ timespec ts; clock_gettime(c,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
int main(){
  EVRInitError e; IVRSystem* sys=VR_Init(&e,VRApplication_Background); if(e) return 1;
  auto* bq=(IVRBlockQueue*)VR_GetGenericInterface(IVRBlockQueue_Version,&e);
  PropertyContainerHandle_t q; bq->Connect(&q,"/xrservice/roomview3d/meshData"); uint32_t best=0;
  for(int k=0;k<40;){
    PropertyContainerHandle_t b; void* buf;
    if(bq->WaitAndAcquireReadOnlyBlock(q,&b,&buf,BlockQueueRead_Latest,100)) continue;
    auto f=roomview::decode(buf);
    if(f.seq>best){ best=f.seq; k++;
      float sinceVsync; uint64_t fc; sys->GetTimeSinceLastVsync(&sinceVsync,&fc);
      if(k%8==0) printf("t=%.4f  mono-t=%.4f boot-t=%.4f  (mono=%.3f)\n",f.t,clk(CLOCK_MONOTONIC)-f.t,clk(CLOCK_BOOTTIME)-f.t,clk(CLOCK_MONOTONIC)); }
    bq->ReleaseReadOnlyBlock(q,b);
  }
  VR_Shutdown();
}
