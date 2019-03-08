/** \file bc_open.cu: contains a CUDA kernel for closed inner and outer boundary conditions.

*/
#define __CUDA 1
#include "fargo.h"
#undef __CUDA
#include <stdarg.h>
#include <helper_cuda.h>
#include <cuda.h>

// BLOCK_X : in azimuth
//#define BLOCK_X DEF_BLOCK_X_CLOSEDBC
#define BLOCK_X 16//64
#define BLOCK_Y 1

__global__ void kernel_closedbc_in (double      *vrad,
                                    const int    ns) {

  const int jg = blockDim.x * blockIdx.x + threadIdx.x;
  vrad[jg]    = 0.0;
  vrad[jg+ns] = 0.0;
  return;
}

__global__ void kernel_closedbc_out (double      *vrad,
                                     const int    nr,
                                     const int    ns) {

  const int jg = blockDim.x * blockIdx.x + threadIdx.x;
  vrad[jg+(nr-1)*ns] = 0.0;
  vrad[jg+(nr-2)*ns] = 0.0;
  return;
}


extern "C" void ClosedBoundary_gpu (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, int where) {

 int nr = Vrad->Nrad;
 int ns = Vrad->Nsec;  
  
 dim3 block (BLOCK_X, BLOCK_Y);
 dim3 grid;
 grid.x = (ns+block.x-1)/block.x;
  
 if (where == INNER) {
    int nb_block_y = (1+BLOCK_Y-1)/BLOCK_Y;
    grid.y = nb_block_y;

    kernel_closedbc_in <<< grid, block >>> (Vrad->gpu_field,
                                            ns);
    cudaThreadSynchronize();
    getLastCudaError ("kernel_closedbc_in failed");
 }
 
 if (where == OUTER) {
    int nb_block_y = (1+BLOCK_Y-1)/BLOCK_Y;
    grid.y = nb_block_y;

    kernel_closedbc_out <<< grid, block >>> (Vrad->gpu_field,
                                             nr,
                                             ns);

    cudaThreadSynchronize();
    getLastCudaError ("kernel_closedbc_out failed");
 }
}
