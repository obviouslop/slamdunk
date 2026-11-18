// rvcloud: live point cloud from XRService's RoomView depth mesh (works in IR/dark).
// Fuses frames into a world voxel grid (Standing universe, metres) and writes PLY.
//   usage: rvcloud out.ply [seconds=30] [voxel_m=0.02] [min_hits=3]
// Calibration (re/rv_latency.py, 2026-11-17): pose look-back 50 ms; extrinsic below.
#include "blockqueue.h"
#include "roomview.h"
#include <atomic>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <ctime>
#include <unordered_map>
using namespace vr;
static std::atomic<bool> stop{false};
static double mono(){ timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
constexpr int GW=132, GH=130;          // vertex grid (layout queue row stride 132)
constexpr double kLookback=0.050;
struct Vox{ float x=0,y=0,z=0; uint32_t n=0; uint32_t last=0; uint32_t frames=0; };
int main(int argc,char**argv){
  if(argc<2){ fprintf(stderr,"usage: rvcloud out.ply [seconds] [voxel_m] [min_hits]\n"); return 1; }
  double secs=argc>2?atof(argv[2]):30, vox=argc>3?atof(argv[3]):0.02; uint32_t minHits=argc>4?atoi(argv[4]):3;
  signal(SIGINT,[](int){stop=true;});
  // extrinsic: R = Rx(dR) * diag(-1,1,-1), c  (head frame <- mesh frame)
  const double dr[3]={-0.39*M_PI/180,1.30*M_PI/180,0.0}, c[3]={0.053,-0.016,-0.010};
  double th=sqrt(dr[0]*dr[0]+dr[1]*dr[1]+dr[2]*dr[2]), k[3]={dr[0]/th,dr[1]/th,dr[2]/th}, Rd[3][3];
  { double s=sin(th),cc=1-cos(th); double K[3][3]={{0,-k[2],k[1]},{k[2],0,-k[0]},{-k[1],k[0],0}};
    for(int i=0;i<3;i++)for(int j=0;j<3;j++){ double KK=0; for(int m=0;m<3;m++) KK+=K[i][m]*K[m][j]; Rd[i][j]=(i==j)+s*K[i][j]+cc*KK; } }
  double X[3][3]; const double F[3]={-1,1,-1}; for(int i=0;i<3;i++)for(int j=0;j<3;j++) X[i][j]=Rd[i][j]*F[j];
  EVRInitError e; IVRSystem* sys=VR_Init(&e,VRApplication_Background); if(e){ fprintf(stderr,"VR_Init failed\n"); return 1; }
  auto* bq=(IVRBlockQueue*)VR_GetGenericInterface(IVRBlockQueue_Version,&e);
  PropertyContainerHandle_t q; if(!bq||bq->Connect(&q,"/xrservice/roomview3d/meshData")){ fprintf(stderr,"no roomview queue\n"); return 2; }
  std::unordered_map<uint64_t,Vox> grid; grid.reserve(1<<20);
  auto key=[&](double x,double y,double z){ auto q=[&](double v){ return (uint64_t)(int64_t)floor(v/vox) & 0x1FFFFF; }; return q(x)|q(y)<<21|q(z)<<42; };
  uint32_t best=0, frames=0; double t0=mono();
  static float d[GW*GH]; static bool ok[GW*GH];
  while(!stop && mono()-t0<secs){
    PropertyContainerHandle_t b; void* buf;
    if(bq->WaitAndAcquireReadOnlyBlock(q,&b,&buf,BlockQueueRead_Latest,100)) continue;
    auto f=roomview::decode(buf);
    if(f.seq<=best || f.t<=0){ bq->ReleaseReadOnlyBlock(q,b); continue; }
    best=f.seq; frames++;
    TrackedDevicePose_t p; sys->GetDeviceToAbsoluteTrackingPose(TrackingUniverseStanding,(float)(f.t-kLookback-mono()),&p,1);
    if(!p.bPoseIsValid || p.eTrackingResult!=TrackingResult_Running_OK){ bq->ReleaseReadOnlyBlock(q,b); continue; }
    auto& H=p.mDeviceToAbsoluteTracking.m;
    for(int i=0;i<GW*GH;i++){ const float* v=f.xyz+3*i; d[i]=sqrtf(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]); }
    // reject "rubber sheet" vertices spanning depth discontinuities and far/near junk
    for(int y=0;y<GH;y++) for(int x=0;x<GW;x++){ int i=y*GW+x; float di=d[i]; bool g=di>0.2f && di<4.5f;
      for(int dy=-1;dy<=1&&g;dy++) for(int dx=-1;dx<=1&&g;dx++){ int xx=x+dx,yy=y+dy; if(xx<0||yy<0||xx>=GW||yy>=GH) continue;
        if(fabsf(d[yy*GW+xx]-di) > 0.04f*di) g=false; }
      ok[i]=g; }
    for(int i=0;i<GW*GH;i++){ if(!ok[i]) continue; const float* v=f.xyz+3*i;
      double h[3]; for(int r=0;r<3;r++) h[r]=X[r][0]*v[0]+X[r][1]*v[1]+X[r][2]*v[2]+c[r];
      double w[3]; for(int r=0;r<3;r++) w[r]=H[r][0]*h[0]+H[r][1]*h[1]+H[r][2]*h[2]+H[r][3];
      Vox& vx=grid[key(w[0],w[1],w[2])]; vx.x+=w[0]; vx.y+=w[1]; vx.z+=w[2]; vx.n++;
      if(vx.last!=frames){ vx.last=frames; vx.frames++; } }
    bq->ReleaseReadOnlyBlock(q,b);
    if(frames%100==0) fprintf(stderr,"\r%u frames, %zu voxels",frames,grid.size());
  }
  VR_Shutdown();
  size_t n=0; for(auto& [k,v]:grid) if(v.frames>=minHits) n++;
  FILE* o=fopen(argv[1],"wb");
  fprintf(o,"ply\nformat binary_little_endian 1.0\nelement vertex %zu\nproperty float x\nproperty float y\nproperty float z\nproperty uchar hits\nend_header\n",n);
  for(auto& [k,v]:grid) if(v.frames>=minHits){ float p[3]={v.x/v.n,v.y/v.n,v.z/v.n}; uint8_t h=v.frames>255?255:v.frames; fwrite(p,4,3,o); fwrite(&h,1,1,o); }
  fclose(o); fprintf(stderr,"\n%u frames in %.1fs -> %zu voxels, %zu kept (>=%u frames) -> %s\n",frames,mono()-t0,grid.size(),n,minHits,argv[1]);
}
