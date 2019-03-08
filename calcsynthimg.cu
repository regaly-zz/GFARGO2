/** \file glvortens.cu
*/
#define __CUDA 1
#include "fargo.h"
#undef __CUDA
#include <stdarg.h>
#include <helper_cuda.h>
#include <cuda.h>

// BLOCK_X : in azimuth
//#define BLOCK_X DEF_BLOCK_X_GLVORTENS
#define BLOCK_X 32
// BLOCK_Y : in radius
#define BLOCK_Y 4

__device__  double CRadiiStuff[32768];

__global__ void kernel_synthimg (const double *rho_gr,
                                 const double *rho_sm,
                                 const double *dust_size, 
                                 const double  powindex,
                                       double *brightness,
 	                               const int     pitch) {
                               
  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;
  int idg = __mul24(ig, pitch) + jg;

  temp[idg] = MU/R_SPEC*(adiabatic_index-1.0)*energy[idg]/dens[idg];
}

void CalcSinthImg_gpu (PolarGrid *Rho_gr, PolarGrid *Rho_sm, PolarGrid *DustSize) {
                    
  int nr = Temp->Nrad;
  int ns = Temp->Nsec;

  //dim3 grid;
  dim3 block = dim3(BLOCK_X, BLOCK_Y);
  dim3 grid ((ns+block.x-1)/block.x, (nr+block.y-1)/block.y);
 
  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)RadiiStuff, (size_t)(14*(NRAD+1))*sizeof(double),	0, cudaMemcpyHostToDevice));
  

  kernel_synthimg <<<grid, block>>> (Rho_gr->gpu_field,
                                     Rho_sm->gpu_field,
                                     DustSize->gpu_field,
                                     DUST_SIZE_DISTR,
                                     Work->gpu_field,
                                     Rho_gr->pitch/sizeof(double));

  cudaThreadSynchronize();
  getLastCudaError ("kernel_synthimg failed");
}
