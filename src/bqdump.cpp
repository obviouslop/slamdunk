// Read-only dump of an XRService block queue (latest block). Never writes/destroys.
#include "blockqueue.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <unistd.h>
using namespace vr;
static size_t mapping_bytes_after(void* p){  // bytes from p to end of its mapping
  FILE* f=fopen("/proc/self/maps","r"); char line[512]; uintptr_t a=(uintptr_t)p;
  while(fgets(line,sizeof line,f)){ uintptr_t lo,hi; if(sscanf(line,"%lx-%lx",&lo,&hi)==2 && a>=lo && a<hi){ fclose(f); return hi-a; } }
  fclose(f); return 0;
}
int main(int argc,char**argv){
  EVRInitError e; VR_Init(&e, VRApplication_Background);
  if(e){ printf("VR_Init: %s\n",VR_GetVRInitErrorAsEnglishDescription(e)); return 1; }
  auto* bq=(IVRBlockQueue*)VR_GetGenericInterface(IVRBlockQueue_Version,&e);
  if(!bq){ printf("no IVRBlockQueue (%d)\n",e); return 1; }
  int samples = argc>2? atoi(argv[2]) : 1;
  for(const char* path : {"/xrservice/roomview3d/meshLayout","/xrservice/roomview3d/meshData"}){
    PropertyContainerHandle_t q=0; int r=bq->Connect(&q,path);
    printf("%s: connect=%d q=%llu\n",path,r,(unsigned long long)q); if(r) continue;
    for(int s=0;s<samples;s++){
      PropertyContainerHandle_t b=0; void* buf=nullptr;
      r=bq->WaitAndAcquireReadOnlyBlock(q,&b,&buf, s?BlockQueueRead_New:BlockQueueRead_Latest, 500);
      if(r||!buf){ printf("  acquire=%d\n",r); continue; }
      size_t n=mapping_bytes_after(buf);
      std::string out=std::string(argc>1?argv[1]:".")+"/"+(strstr(path,"Layout")?"layout":"data")+std::to_string(s)+".bin";
      size_t cap = n>(64u<<20)?(64u<<20):n;
      FILE* f=fopen(out.c_str(),"wb"); fwrite(buf,1,cap,f); fclose(f);
      printf("  block %llu buf=%p mapping-tail=%zu -> %s\n",(unsigned long long)b,buf,n,out.c_str());
      bq->ReleaseReadOnlyBlock(q,b);
    }
  }
  VR_Shutdown(); return 0;
}
