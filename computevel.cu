/** \file "computeLRmomenta.cu" : implements the kernel for the "template" procedure
*/
#define __CUDA 1
#include "fargo.h"
#undef __CUDA
#include <stdarg.h>
#include <helper_cuda.h>
#include <cuda.h>

// BLOCK_X : in azimuth
//#define BLOCK_X DEF_BLOCK_X_COMPUTEVEL
#define BLOCK_X 8
// BLOCK_Y : in radius
#define BLOCK_Y 4

#define GET_TAB(u,x,y,pitch) *(u + __mul24(y, pitch) + x)
#define irmed CRadiiStuff[(nr+1)*2+ig]

//__constant__ double CRadiiStuff[8192];
__device__ double CRadiiStuff[32768];


__global__ void kernel_cvel (double *rho,
                             double *vrad,
                             double *vtheta,
                             double *rp,
                             double *rm,
                             double *tp,
                             double *tm,
                             double  omegaframe,
                             int     ns, 
                             int     nr, 
                             int     pitch,
                             double *old_rho, 
                             double *old_tmp, 
                             double *old_tmm) {

  __shared__ double srho[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double orho[(BLOCK_X+2)*(BLOCK_Y+2)]; //shared old rho
  __shared__ double srp[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double stp[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double otp[(BLOCK_X+2)*(BLOCK_Y+2)]; //shared old thetamomp
  int ids, idg;
  //double irmed;
  // jg & ig, g like 'global' (global memory <=> full grid)
  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;
  // js & is, l like 'local' (shared memory <=> local patch)
  int js = threadIdx.x + 1;
  int is = threadIdx.y + 1;
  int jgp = jg+1;
  int jgm = jg-1;
  ids = is*(blockDim.x+2)+js;
  idg = __mul24(ig, pitch) + jg;
  if (jg == ns-1) jgp -= pitch;
  if (jg == 0) jgm += pitch;
  // We perform a coalesced read of 'rho' into the shared memory;
  srho[ids] = rho[idg];
  orho[ids] = old_rho[idg];
  if (js == 1) {
    srho[ids-1] = GET_TAB (rho, jgm, ig, pitch);
    orho[ids-1] = GET_TAB (old_rho, jgm, ig, pitch);
  }
  if ((is == 2) && (ig > 1)) {
    srho[js] = rho[idg-(pitch<<1)];
  }
  // We perform a coalesced read of 'rp' into the shared memory;
  srp[ids] = rp[idg];
  if ((is == 1) && (ig > 0)) {
    srp[js] = rp[idg-pitch];
  }

  // We perform a coalesced read of 'tp' into the shared memory;
  stp[ids] = tp[idg];
  otp[ids] = old_tmp[idg];
  if (js == 1) {
    stp[ids-1] = GET_TAB (tp, jgm, ig, pitch);
    otp[ids-1] = GET_TAB (old_tmp, jgm, ig, pitch);
  }

  __syncthreads ();

  //irmed = CRadiiStuff[(nr+1)*2+ig];

  if (ig > 0)
    vrad[idg] = (srp[ids-blockDim.x-2]+rm[idg])/(srho[ids-blockDim.x-2]+srho[ids]);
  else
    vrad[idg] = 0;//vrad[idg+ns];

  double temp = ((stp[ids-1]-otp[ids-1])+(tm[idg]-old_tmm[idg]))*irmed;
  temp -=  ((srho[ids-1]-orho[ids-1])+(srho[ids]-orho[ids]))*(vtheta[idg]+sqrt(irmed));
  vtheta[idg] += temp/(srho[ids-1]+srho[ids]);
}

extern "C"
void ComputeVelocities_gpu (PolarGrid *Rho, PolarGrid *Vrad, PolarGrid *Vtheta)
{
  int nr, ns;
  nr = Vrad->Nrad;
  ns = Vrad->Nsec;
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid ((ns+block.x-1)/block.x, (nr+block.y-1)/block.y);
  
  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)RadiiStuff, (size_t)(14*(nr+1))*sizeof(double), 0, cudaMemcpyHostToDevice));
  cudaError_t err = cudaGetLastError ();

  double *OldRho       = VradNew->gpu_field;
  double *OldThetaMomP = VradInt->gpu_field;
  double *OldThetaMomM = VthetaInt->gpu_field;

  if ( cudaSuccess != err) {
    fprintf (stderr, "Cuda error in computevel...BEFORE  %s\n", cudaGetErrorString (err));
    exit (-1);
  }

  kernel_cvel <<< grid, block >>> (Rho->gpu_field,
                                   Vrad->gpu_field,
                                   Vtheta->gpu_field,
                                   RadMomP->gpu_field,
                                   RadMomM->gpu_field,
                                   ThetaMomP->gpu_field,
                                   ThetaMomM->gpu_field,
                                   OmegaFrame,
                                   Rho->Nsec, 
                                   Rho->Nrad, 
                                   Rho->pitch/sizeof(double),
                                   OldRho, 
                                   OldThetaMomP, 
                                   OldThetaMomM);
  
  cudaThreadSynchronize();
  getLastCudaError("kernel_cvel execution failed");
  
  //cudaThreadSynchronize();
  //err = cudaGetLastError ();
  //if ( cudaSuccess != err) {
  //  fprintf (stderr, "Cuda error in computevel...AFTER %s\n", cudaGetErrorString (err));
  //  exit (-1);
  //}
}
