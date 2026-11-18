// Poll roomview3d meshData (read-only): rate, timestamps, trailer pose vs HMD pose; dump PLYs.
#include "blockqueue.h"
#include "roomview.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>
#include <thread>
using namespace vr;
static void ply(const char* fn, const std::vector<float>& p){ FILE* f=fopen(fn,"w"); fprintf(f,"ply\nformat ascii 1.0\nelement vertex %zu\nproperty float x\nproperty float y\nproperty float z\nend_header\n",p.size()/3); for(size_t i=0;i<p.size();i+=3) fprintf(f,"%.4f %.4f %.4f\n",p[i],p[i+1],p[i+2]); fclose(f);}
int main(int argc,char**argv){
  const char* out=argc>1?argv[1]:"."; double secs=argc>2?atof(argv[2]):3;
  EVRInitError e; IVRSystem* sys=VR_Init(&e,VRApplication_Background); if(e) return 1;
  auto* bq=(IVRBlockQueue*)VR_GetGenericInterface(IVRBlockQueue_Version,&e);
  PropertyContainerHandle_t q; if(bq->Connect(&q,"/xrservice/roomview3d/meshData")) return 2;
  uint32_t last=0; int n=0; auto t0=std::chrono::steady_clock::now(); double tfirst=0,tlast=0; uint32_t sfirst=0;
  while(std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()<secs){
    PropertyContainerHandle_t b; void* buf;
    if(bq->WaitAndAcquireReadOnlyBlock(q,&b,&buf,BlockQueueRead_Latest,200)) continue;
    auto f=roomview::decode(buf);
    if(f.seq!=last){
      last=f.seq; if(!n){sfirst=f.seq;tfirst=f.t;} tlast=f.t; n++;
      TrackedDevicePose_t hs, hr;
      sys->GetDeviceToAbsoluteTrackingPose(TrackingUniverseStanding,0,&hs,1);
      sys->GetDeviceToAbsoluteTrackingPose(TrackingUniverseRawAndUncalibrated,0,&hr,1);
      int valid=0; for(size_t i=0;i<roomview::kMaxVerts;i++){ const float*p=f.xyz+3*i; if(p[0]||p[1]||p[2]) valid++; }
      if(n<=6 || n%15==0) printf("seq %u t=%.4f valid=%d trailerT=(%.3f %.3f %.3f) hmdStanding=(%.3f %.3f %.3f) hmdRaw=(%.3f %.3f %.3f)\n",
        f.seq,f.t,valid,f.m[0][3],f.m[1][3],f.m[2][3],
        hs.mDeviceToAbsoluteTracking.m[0][3],hs.mDeviceToAbsoluteTracking.m[1][3],hs.mDeviceToAbsoluteTracking.m[2][3],
        hr.mDeviceToAbsoluteTracking.m[0][3],hr.mDeviceToAbsoluteTracking.m[1][3],hr.mDeviceToAbsoluteTracking.m[2][3]);
      if(n==1){
        std::vector<float> raw, xf;
        for(size_t i=0;i<roomview::kMaxVerts;i++){ const float*p=f.xyz+3*i; if(!(p[0]||p[1]||p[2])) continue;
          raw.insert(raw.end(),p,p+3);
          for(int r=0;r<3;r++) xf.push_back(f.m[r][0]*p[0]+f.m[r][1]*p[1]+f.m[r][2]*p[2]+f.m[r][3]); }
        char fn[512]; snprintf(fn,512,"%s/rv_raw.ply",out); ply(fn,raw); snprintf(fn,512,"%s/rv_xf.ply",out); ply(fn,xf);
      }
    }
    bq->ReleaseReadOnlyBlock(q,b);
  }
  printf("frames %d over %.2fs wall; seq %u..%u; data-clock span %.3fs => %.1f Hz\n",n,secs,sfirst,last,tlast-tfirst,(n-1)/(tlast-tfirst));
  VR_Shutdown();
}
