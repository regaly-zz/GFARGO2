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

//__device__  double CRadiiStuff[32768];

__global__ void kernel_dustgasmassratio (double *rho_g,
                                         double *rho_d,
                                         double *ratio,
 	                                       int     pitch) {
                               
  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;
  int idg = __mul24(ig, pitch) + jg;

  ratio[idg] = rho_d[idg]/rho_g[idg];
}

void CalcDustGasMassRatio_gpu (PolarGrid *Rho, PolarGrid *DustRho, PolarGrid *DustGasMassRatio) {
                    
  int nr = Rho->Nrad;
  int ns = Rho->Nsec;

  //dim3 grid;
  dim3 block = dim3(BLOCK_X, BLOCK_Y);
  dim3 grid ((ns+block.x-1)/block.x, (nr+block.y-1)/block.y);
 
//  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)RadiiStuff, (size_t)(14*(NRAD+1))*sizeof(double),	0, cudaMemcpyHostToDevice));
  
  kernel_dustgasmassratio <<<grid, block>>> (Rho->gpu_field,
                                             DustRho->gpu_field,
                                             DustGasMassRatio->gpu_field,
                                             Rho->pitch/sizeof(double));

  cudaThreadSynchronize();
  getLastCudaError ("kernel_dustgasmassratio failed");
}
