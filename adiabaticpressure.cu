/** \file adiabaticpressure.cu: contains a CUDA kernel for calculating adiabatic gas pressure
*/

#define __CUDA 1
#include "fargo.h"
#undef __CUDA
#include <stdarg.h>
#include <helper_cuda.h>
#include <cuda.h>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/extrema.h>

// BLOCK_X : in azimuth
//#define BLOCK_X DEF_BLOCK_X_CALCECC
#define BLOCK_X 16
// BLOCK_Y : in radius
#define BLOCK_Y 16

#define RMED  CRadiiStuff[ig]

__device__ double CRadiiStuff[8096];

__global__ void kernel_adiabaticpressure (double *SurfEcc,
                                          double *Rho,                  
                                          double *Energy,
                                          int nr,
                                          int pitch,
                                          double *Pressure) {

  // jg & ig, g like 'global' (global memory <=> full grid)
  // Below, we recompute x and y for each zone using cos/sin.
  // This method turns out to be faster, on high-end platforms,
  // than a coalesced read of tabulated values.
  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;
  int idg = __mul24(ig, pitch) + jg;
}

extern "C" 
void CalcAdiabaticPresssure (PolarGrid *Rho, PolarGrid *Energy, PolarGrid *Pressure) {
  int nr = Pressure->Nrad;
  int ns = Pressure->Nsec;
   
  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)(RadiiStuff+6*(nr+1)), (size_t)(nr)*sizeof(double)));

  // dsk inner region
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid  ((ns + block.x-1)/block.x, (nr + block.y-1)/block.y);
  kernel_adiabaticpressure <<< grid, block >>> (gpu_Rho,
}