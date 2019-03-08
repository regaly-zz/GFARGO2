/** \file bc_ref.cu: contains a CUDA kernel for reflecting inner and outer boundary condition.
*/
#define __CUDA 1
#include "fargo.h"
#undef __CUDA
#include <stdarg.h>
#include <helper_cuda.h>
#include <cuda.h>

// BLOCK_X : in azimuth
//#define BLOCK_X DEF_BLOCK_X_REFBC
#define BLOCK_X 16//64
#define BLOCK_Y 1

__global__ void kernel_refbc_in (double      *vrad,
                                  double      *vtheta,
                                  double      *rho,
                                  const int    nr,
                                  const int    ns) {
                                    
  const int jg = blockDim.x * blockIdx.x + threadIdx.x;
  
  rho[jg]    = rho[jg+ns];               //SigmaMed0;
  vrad[jg]   = -vrad[jg+ns];
  vtheta[jg] = vtheta[jg+ns];
}


__global__ void kernel_refbc_out (double      *vrad,
                                  double      *vtheta,
                                  double      *rho,
                                  const int    nr,
                                  const int    ns) {
                                    
  const int jg = blockDim.x * blockIdx.x + threadIdx.x;

  rho[jg+(nr-1)*ns]    = rho[jg+(nr-2)*ns];     //SigmaMedNr1
  vrad[jg+(nr-1)*ns]   = -vrad[jg+(nr-2)*ns];  
  vtheta[jg+(nr-1)*ns] = vtheta[jg+(nr-2)*ns];
}


extern "C" void ReflectingBoundary_gpu (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, int where) {
  int nr = Vrad->Nrad;
  int ns = Vrad->Nsec;
  
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid;
  grid.x = (ns+block.x-1)/block.x;
   
  if (where == INNER) {
     int nb_block_y = (nr+BLOCK_Y-1)/BLOCK_Y;
     grid.y = nb_block_y;
   
     kernel_refbc_in <<< grid, block >>> (Vrad->gpu_field,
                                           Vtheta->gpu_field,
                                           Rho->gpu_field,
                                           (int) NRAD,
                                           (int) NSEC);

     cudaThreadSynchronize();   
     getLastCudaError ("kernel kernel_refbc_in failed");
  }
   
  if (where == OUTER) {
     int nb_block_y = (nr+BLOCK_Y-1)/BLOCK_Y;
     grid.y = nb_block_y;
   
     kernel_refbc_out <<< grid, block >>> (Vrad->gpu_field,
                                            Vtheta->gpu_field,
                                            Rho->gpu_field,
                                            (int) NRAD,
                                            (int) NSEC);

     cudaThreadSynchronize();   
     getLastCudaError ("kernel kernel_refbc_out failed");
  }

  
  cudaThreadSynchronize();
}
