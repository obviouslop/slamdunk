// Read-only: list tracked devices with class, tracking result and pose validity.
#include "openvr.h"
#include <cstdio>
using namespace vr;
int main(){ EVRInitError e; IVRSystem* s=VR_Init(&e,VRApplication_Background); if(!s){printf("init fail %d\n",e);return 1;}
  TrackedDevicePose_t p[k_unMaxTrackedDeviceCount]; s->GetDeviceToAbsoluteTrackingPose(TrackingUniverseStanding,0,p,k_unMaxTrackedDeviceCount);
  const char* cls[]={"invalid","hmd","controller","tracker","reference","display"};
  for(unsigned i=0;i<k_unMaxTrackedDeviceCount;i++){ auto c=s->GetTrackedDeviceClass(i); if(c==TrackedDeviceClass_Invalid) continue;
    char sn[64]=""; s->GetStringTrackedDeviceProperty(i,Prop_SerialNumber_String,sn,64);
    printf("dev %u %-10s %-14s connected=%d valid=%d result=%d pos=(%.2f %.2f %.2f)\n",i,c<6?cls[c]:"?",sn,p[i].bDeviceIsConnected,p[i].bPoseIsValid,p[i].eTrackingResult,
      p[i].mDeviceToAbsoluteTracking.m[0][3],p[i].mDeviceToAbsoluteTracking.m[1][3],p[i].mDeviceToAbsoluteTracking.m[2][3]); }
  VR_Shutdown(); }
