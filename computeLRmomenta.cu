/** \file "computeLRmomenta.cu" : implements the kernel for the "template" procedure
*/
#define __CUDA 1
#include "fargo.h"
#undef __CUDA
#include <stdarg.h>
#include <helper_cuda.h>
#include <cuda.h>

// BLOCK_X : in azimuth
//#define BLOCK_X DEF_BLOCK_X_LRMOMENTA
#define BLOCK_X 32
// BLOCK_Y : in radius
#define BLOCK_Y 4

#define GET_TAB(u,x,y,pitch) *(u + __mul24(y, pitch) + x)
#define invrmed CRadiiStuff[(nr+1)*2+ig]
#define rmed CRadiiStuff[(nr+1)*6+ig]

//__constant__ double CRadiiStuff[8192];
__device__ double CRadiiStuff[32768];

__global__ void kernel_clrm (double *rho,
                             double *vrad,
                             double *vtheta,
                             double *rp,
                             double *rm,
                             double *tp,
                             double *tm,
                             double omegaframe,
                             int ns, int nr, int pitch) {
                               
  __shared__ double shared_rho[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double shared_vr[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double shared_vt[(BLOCK_X+2)*(BLOCK_Y+2)];
  int ids, idg;
  double srmed;
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
  shared_rho[ids] = rho[idg];

  // We perform a coalesced read of 'vrad' into the shared memory;
  shared_vr[ids] = vrad[idg];
    // EDGE 4: "TOP EDGE". Be careful not to read anything if in last row..
  if ((is == blockDim.y) && (ig < nr-1))
    shared_vr[js+(blockDim.y+1)*(blockDim.x+2)] = GET_TAB (vrad, jg, ig+1, pitch);

  if (ig == nr-1)
    shared_vr[js+(blockDim.y+1)*(blockDim.x+2)] = 0.0;

  // We perform a coalesced read of 'vtheta' into the shared memory;
  shared_vt[ids] = vtheta[idg];
  // EDGE 2: "RIGHT EDGE"
  if (js == blockDim.x)
    shared_vt[is*(blockDim.x+2)+blockDim.x+1] = GET_TAB (vtheta, jgp, ig, pitch);

  __syncthreads ();

  //rmed = CRadiiStuff[(nr+1)*6+ig];
  srmed = rmed * shared_rho[ids];

  rp[idg] = shared_rho[ids]*shared_vr[ids+blockDim.x+2];
  rm[idg] = shared_rho[ids]*shared_vr[ids];


//  tp[idg] = srmed*(shared_vt[ids+1]+sqrt(1.0/rmed));
//  tm[idg] = srmed*(shared_vt[ids]+sqrt(1.0/rmed));
  const double sqrtinvrmed = sqrt(invrmed);
  tp[idg] = srmed*(shared_vt[ids+1]+sqrtinvrmed);
  tm[idg] = srmed*(shared_vt[ids]+sqrtinvrmed);

}

extern "C"
void ComputeLRMomenta_gpu (PolarGrid *Rho, PolarGrid *Vrad, PolarGrid *Vtheta) {
  int nr, ns;
  nr = Vrad->Nrad;
  ns = Vrad->Nsec;
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid ((ns+block.x-1)/block.x, (nr+block.y-1)/block.y);
  
  cudaMemcpyToSymbol(CRadiiStuff, (void *)RadiiStuff, (size_t)(14*(nr+1))*sizeof(double), 0, cudaMemcpyHostToDevice);
  kernel_clrm <<< grid, block >>> (Rho->gpu_field,
                                   Vrad->gpu_field,
                                   Vtheta->gpu_field,
                                   RadMomP->gpu_field,
                                   RadMomM->gpu_field,
                                   ThetaMomP->gpu_field,
                                   ThetaMomM->gpu_field,
                                   OmegaFrame,
                                   Rho->Nsec, Rho->Nrad, Rho->pitch/sizeof(double));
  
  cudaThreadSynchronize();
  getLastCudaError ("ComputeLRMomenta_gpu: kernel failed");
}
