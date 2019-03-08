/** \file "computestartheta.cu" : implements the kernel for the "computestartheta" procedure
*/
#define __CUDA 1
#include "fargo.h"
#undef __CUDA
#include <stdarg.h>
#include <helper_cuda.h>
#include <cuda.h>

// BLOCK_X : in azimuth
//#define BLOCK_X DEF_BLOCK_X_COMPUTESTARTHETA
#define BLOCK_X 64
// BLOCK_Y : in radius
#define BLOCK_Y 4

#define GET_TAB(u,x,y,pitch) *(u + __mul24(y, pitch) + x)
#define invdxtheta CRadiiStuff[(nr+1)*2+ig]*(double)ns*0.15915494309189

//__constant__ double CRadiiStuff[8192];
__device__ double CRadiiStuff[32768];

__global__ void kernel_cpst (double *qb,
                             double *vt,
                             double *qs,
                             int     ns, 
                             int     nr, 
                             int     pitch,
                             double  dt, 
                             double  dpns ) {

  __shared__ double sdq[(BLOCK_X+3)*BLOCK_Y];
  __shared__ double sqb[(BLOCK_X+3)*BLOCK_Y];
  //double invdxtheta, dqm, dqp, tqs, ksi, dxtheta;
  double dqm, dqp, tqs, ksi, dxtheta;

  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;
  int js = threadIdx.x+2;
  int is = threadIdx.y;
  int idg = __mul24(ig, pitch) + jg;
  int ids = js+is*(blockDim.x+3);
  int idsp = ids+1;
  int idsm = ids-1;
  int jgmm = jg-2;
  int jgp  = jg+1;
  if (jg == 0) jgmm = ns-2;
  if (jg == 1) jgmm = ns-1;
  if (jg == ns-1) jgp = 0;

  // We perform a coalesced read of 'qb' into the shared memory;
  sqb[ids] = qb[idg];
  // EDGE 1 : "LEFT EDGE". We need TWO extra columns on this side.
  if (js == 2) 
    sqb[ids-2] = GET_TAB (qb, jgmm, ig, pitch);
  if (js == 3) 
    sqb[ids-2] = GET_TAB (qb, jgmm, ig, pitch); //jgmm ! not jgm !
  
  // EDGE 2: "RIGHT EDGE".
  // We need ONE extra column on this side.
  if (js == blockDim.x+1)
    sqb[ids+1] = GET_TAB (qb, jgp, ig, pitch);

  __syncthreads ();

  //invdxtheta = CRadiiStuff[(nr+1)*2+ig]*(double)ns*0.15915494309189f;
  
  dqm = sqb[ids]-sqb[idsm];
  dqp = sqb[idsp]-sqb[ids];
  sdq[ids] = 0.0;
  if (dqm*dqp > 0.0)
    sdq[ids] = dqp*dqm/(dqp+dqm)*invdxtheta;

  if (js == 2) {
    dqm = sqb[idsm]-sqb[idsm-1];
    dqp = sqb[ids]-sqb[idsm];
    sdq[idsm] = 0.0;
    if (dqm*dqp > 0.0)
      sdq[idsm] = dqp*dqm/(dqp+dqm)*invdxtheta;
  }

  __syncthreads ();
  
  dxtheta = dpns * CRadiiStuff[(nr+1)*6+ig];
  
  ksi = vt[idg]*dt;
  //tqs = sqb[idsm]+(dxtheta-ksi)*sdq[idsm];
  if (ksi <= 0.0)
    tqs = sqb[ids]-(dxtheta+ksi)*sdq[ids];
  else
    tqs = sqb[idsm]+(dxtheta-ksi)*sdq[idsm];
  qs[idg] = tqs;
}

extern "C"
void ComputeStarTheta_gpu (PolarGrid *Qbase, PolarGrid *Vtheta, PolarGrid *QStar, double dt)
{
  int nr, ns;
  nr = Vtheta->Nrad;
  ns = Vtheta->Nsec;
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid ((ns+block.x-1)/block.x, (nr+block.y-1)/block.y);
  
  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)RadiiStuff, (size_t)(14*(nr+1))*sizeof(double), 0, cudaMemcpyHostToDevice));

  kernel_cpst <<< grid, block >>> (Qbase->gpu_field,
                                   Vtheta->gpu_field,
                                   QStar->gpu_field,
                                   Vtheta->Nsec, 
                                   Vtheta->Nrad,
                                   Vtheta->pitch/sizeof(double), 
                                   dt,
                                   2.0*M_PI/(double)ns);
  cudaThreadSynchronize();
  getLastCudaError("Kernel cpst execution failed");
}
