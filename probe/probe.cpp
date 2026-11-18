// Probe: what does SteamVR's public IVRTrackedCamera expose on this headset?
#include "openvr.h"
#include <cstdio>
#include <vector>
#include <unistd.h>
using namespace vr;
static const char* ft[] = {"Distorted","Undistorted","MaxUndistorted"};
int main(){
  EVRInitError e; IVRSystem* sys = VR_Init(&e, VRApplication_Background);
  if(!sys){ printf("VR_Init failed: %s\n", VR_GetVRInitErrorAsEnglishDescription(e)); return 1; }
  IVRTrackedCamera* cam = VRTrackedCamera();
  if(!cam){ printf("no IVRTrackedCamera\n"); return 1; }
  TrackedDeviceIndex_t d = k_unTrackedDeviceIndex_Hmd;
  bool has=false; auto err = cam->HasCamera(d,&has);
  printf("HasCamera: %d (%s)\n", has, cam->GetCameraErrorNameFromEnum(err));
  ETrackedPropertyError pe;
  int32_t n = sys->GetInt32TrackedDeviceProperty(d, Prop_NumCameras_Int32, &pe);
  printf("NumCameras: %d (perr %d)\n", n, pe);
  int32_t layout = sys->GetInt32TrackedDeviceProperty(d, Prop_CameraFrameLayout_Int32, &pe);
  printf("FrameLayout: 0x%x (perr %d)\n", layout, pe);
  HmdMatrix34_t xf[8]; uint32_t sz = sys->GetArrayTrackedDeviceProperty(d, Prop_CameraToHeadTransforms_Matrix34_Array, k_unHmdMatrix34PropertyTag, xf, sizeof(xf), &pe);
  printf("CameraToHead transforms: %u bytes (perr %d)\n", sz, pe);
  for(uint32_t i=0;i<sz/sizeof(HmdMatrix34_t);i++){ auto&m=xf[i].m; printf("  cam%u t=(%.4f %.4f %.4f) R0=(%.3f %.3f %.3f)\n",i,m[0][3],m[1][3],m[2][3],m[0][0],m[0][1],m[0][2]); }
  int32_t df[8]; sz = sys->GetArrayTrackedDeviceProperty(d, Prop_CameraDistortionFunction_Int32_Array, k_unInt32PropertyTag, df, sizeof(df), &pe);
  double dc[8][k_unMaxDistortionFunctionParameters]; uint32_t sz2 = sys->GetArrayTrackedDeviceProperty(d, Prop_CameraDistortionCoefficients_Float_Array, k_unDoublePropertyTag, dc, sizeof(dc), &pe);
  for(uint32_t i=0;i<sz/4;i++){ printf("  cam%u distfn=%d coeffs:",i,df[i]); for(int k=0;k<6;k++) printf(" %.5f",dc[i][k]); printf("\n"); }
  for(int t=0;t<3;t++){
    uint32_t w=0,h=0,b=0; err = cam->GetCameraFrameSize(d,(EVRTrackedCameraFrameType)t,&w,&h,&b);
    printf("[%s] size %ux%u buf %u (%s)\n", ft[t], w,h,b, cam->GetCameraErrorNameFromEnum(err));
    for(int c=0;c<(n>0?n:2);c++){ HmdVector2_t f,cc; err=cam->GetCameraIntrinsics(d,c,(EVRTrackedCameraFrameType)t,&f,&cc);
      printf("   cam%d f=(%.1f,%.1f) c=(%.1f,%.1f) (%s)\n",c,f.v[0],f.v[1],cc.v[0],cc.v[1],cam->GetCameraErrorNameFromEnum(err)); }
  }
  TrackedCameraHandle_t hc=0; err = cam->AcquireVideoStreamingService(d,&hc);
  printf("Acquire: %s\n", cam->GetCameraErrorNameFromEnum(err));
  uint32_t w,h,b; cam->GetCameraFrameSize(d,VRTrackedCameraFrameType_Distorted,&w,&h,&b);
  std::vector<uint8_t> buf(b?b:1); CameraVideoStreamFrameHeader_t hdr{}; uint32_t last=0; int got=0;
  for(int i=0;i<300 && got<10;i++){ usleep(10000);
    err = cam->GetVideoStreamFrameBuffer(hc,VRTrackedCameraFrameType_Distorted,buf.data(),b,&hdr,sizeof(hdr));
    if(err!=VRTrackedCameraError_None){ if(i%50==0) printf("  frame err %s\n",cam->GetCameraErrorNameFromEnum(err)); continue; }
    if(hdr.nFrameSequence==last) continue; last=hdr.nFrameSequence; got++;
    auto&m=hdr.trackedDevicePose.mDeviceToAbsoluteTracking.m;
    printf("  seq %u %ux%u bpp%u valid%d pos=(%.3f %.3f %.3f) exp=%llu px0=%u\n",hdr.nFrameSequence,hdr.nWidth,hdr.nHeight,hdr.nBytesPerPixel,hdr.trackedDevicePose.bPoseIsValid,m[0][3],m[1][3],m[2][3],(unsigned long long)hdr.ulFrameExposureTime,buf[b/2]);
  }
  if(got){ FILE*f=fopen("frame.raw","wb"); fwrite(buf.data(),1,b,f); fclose(f); printf("saved frame.raw %ux%u bpp%u\n",hdr.nWidth,hdr.nHeight,hdr.nBytesPerPixel);}
  cam->ReleaseVideoStreamingService(hc); VR_Shutdown();
}
