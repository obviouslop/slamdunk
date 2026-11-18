// Query (TRY_FMT only, no streaming) whether the CAMSS RDI node accepts bigger buffers.
#include <fcntl.h>
#include <linux/videodev2.h>
#include <stdio.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>
static void try(int fd,int w,int h,unsigned si){
  struct v4l2_format f; memset(&f,0,sizeof f); f.type=V4L2_BUF_TYPE_VIDEO_CAPTURE_MPLANE;
  f.fmt.pix_mp.width=w; f.fmt.pix_mp.height=h; f.fmt.pix_mp.pixelformat=v4l2_fourcc('Y','1','0','P'); f.fmt.pix_mp.num_planes=1;
  f.fmt.pix_mp.plane_fmt[0].sizeimage=si;
  int r=ioctl(fd,VIDIOC_TRY_FMT,&f);
  printf("req %dx%d si %u -> r=%d got %ux%u bpl %u si %u\n",w,h,si,r,f.fmt.pix_mp.width,f.fmt.pix_mp.height,f.fmt.pix_mp.plane_fmt[0].bytesperline,f.fmt.pix_mp.plane_fmt[0].sizeimage);
}
int main(){ int fd=open("/dev/video0",O_RDWR); if(fd<0){perror("open");return 1;}
  try(fd,1640,1232,0); try(fd,1632,1232,0); try(fd,2464,2464,0); close(fd); }
