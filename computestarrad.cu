/** \file "computestarrad.cu" : implements the kernel for the "computestarrad" procedure
*/
#define __CUDA 1
#include "fargo.h"
#undef __CUDA
#include <stdarg.h>
#include <helper_cuda.h>
#include <cuda.h>


// BLOCK_X : in azimuth
//#define BLOCK_X DEF_BLOCK_X_COMPUTESTARRAD
#define BLOCK_X 16
// BLOCK_Y : in radius
#define BLOCK_Y 16

#define GET_TAB(u,x,y,pitch) *(u + __mul24(y, pitch) + x)
#define idrmed  CRadiiStuff[ig]
#define idrmedp CRadiiStuff[ig+1]
#define rmed    CRadiiStuff[(nr+1)*6+ig]
#define rmedp   CRadiiStuff[(nr+1)*6+ig+1]
#define rmedm   CRadiiStuff[(nr+1)*6+ig-1]

//__constant__ double CRadiiStuff[8192];
__device__ double CRadiiStuff[32768];

__global__ void kernel_cpsr (double *qb,
                             double *vr,
                             double *qs,
                             int ns, int nr, int pitch, double dt) {
                               
  __shared__ double sdq[BLOCK_X*(BLOCK_Y+3)];
  __shared__ double sqb[BLOCK_X*(BLOCK_Y+3)];
//  double idrmed, idrmedp, dqm, dqp, rmed, rmedp, rmedm, tqs, v;
  double dqm, dqp, tqs, v;

  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;
  int js = threadIdx.x;
  int is = threadIdx.y+2;
  int idg = __mul24(ig, pitch) + jg;
  int ids = js+is*blockDim.x;
  int idsp = ids+blockDim.x;
  int idsm = ids-blockDim.x;

  // We perform a coalesced read of 'qb' into the shared memory;
  sqb[ids] = qb[idg];
  // EDGE 3: "BOTTOM EDGE". Be careful not to read anything if in first row...
  // We need TWO extra rows on this side
  if ((is == 2) && (ig > 1)) { // First warp reads innermost row
    sqb[js] = qb[idg-(pitch<<1)];
  }
  if ((is == 3) && (ig > 1)) { // Second warp reads subsequent row
    sqb[js+blockDim.x] = qb[idg-(pitch<<1)]; 
  }
  // EDGE 4: "TOP EDGE". Be careful not to read anything if in last row... 
  // We need ONE extra row on this side
  if ((is == blockDim.y) && (ig < nr-2))
    sqb[js+(blockDim.y+2)*blockDim.x] = qb[idg+(pitch<<1)];

  __syncthreads ();
  
  //idrmed  = CRadiiStuff[ig];
  //idrmedp = CRadiiStuff[ig+1];

  sdq[ids] = 0.0;
  if ((ig > 0) && (ig < nr-1)) {
    dqm = (sqb[ids]-sqb[idsm])*idrmed;
    dqp = (sqb[idsp]-sqb[ids])*idrmedp;
    sdq[ids] = 0.0;
    if (dqm*dqp > 0.0)
      sdq[ids] = 2.0*dqp*dqm/(dqp+dqm);
  }
  if ((is == 2) && (ig > 1)) {
    dqm = (sqb[idsm]-sqb[idsm-blockDim.x])*CRadiiStuff[ig-1];
    dqp = (sqb[ids]-sqb[idsm])*idrmed;
      sdq[idsm] = 0.0;
    if (dqm*dqp > 0.0)
      sdq[idsm] = 2.0*dqp*dqm/(dqp+dqm);
  }

  __syncthreads ();
  
  //rmed = CRadiiStuff[(nr+1)*6+ig];
  //rmedp= CRadiiStuff[(nr+1)*6+ig+1];
  //rmedm= CRadiiStuff[(nr+1)*6+ig-1];
  
  tqs = 0.0;
  if (ig > 0) {
    v = vr[idg];
    //tqs = sqb[idsm]+(rmed-rmedm-v*dt)*0.5*sdq[idsm];
    if (v <= 0.0)
      tqs = sqb[ids]-(rmedp-rmed+v*dt)*0.5*sdq[ids];
    else
      tqs = sqb[idsm]+(rmed-rmedm-v*dt)*0.5*sdq[idsm];
  }
  qs[idg] = tqs;
}

extern "C"
void ComputeStarRad_gpu (PolarGrid *Qbase, PolarGrid *Vrad, PolarGrid *QStar, double dt)
{
  int nr, ns;
  nr = Vrad->Nrad;
  ns = Vrad->Nsec;
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid ((ns+block.x-1)/block.x, (nr+block.y-1)/block.y);
  
  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)RadiiStuff, (size_t)(14*(nr+1))*sizeof(double),0, cudaMemcpyHostToDevice));
  
  kernel_cpsr <<< grid, block >>> (Qbase->gpu_field,
                                   Vrad->gpu_field,
                                   QStar->gpu_field,
                                   Vrad->Nsec, Vrad->Nrad,
                                   Vrad->pitch/sizeof(double), dt);
                                   
  cudaThreadSynchronize();
  getLastCudaError("kernel_cpsr execution failed");
}
