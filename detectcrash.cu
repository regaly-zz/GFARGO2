/** \file dampingbc.cu: contains a CUDA kernel for Stockholm's prescription of damping bc's.
*/
#define __CUDA 1
#include "fargo.h"
#undef __CUDA
#include <stdarg.h>
#include <helper_cuda.h>
#include <cuda.h>

// BLOCK_X : in azimuth
//#define BLOCK_X DEF_BLOCK_X_DETECTCRASH
#define BLOCK_X 64
#define BLOCK_Y 1
/* Note that with the above choice of BLOCK_Y, the occupancy is not 1,
   but there is less arithmetic done within the kernels, and the
   performance turns out to be better. */

__global__ void kernel_detectcrash  (double *field, 
                                     double floor_value, 
                                     double pitch) {

  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;
  int idg = __mul24(ig, pitch) + jg;
  
  const double val = field[idg];
  if (val <= floor_value || val != val) {
//    if (jg==0)
//    printf ("err: %i %i %e\n", ig, jg, field[idg]);
    field[idg] = floor_value;
  }
}

extern "C"
bool DetectCrash (PolarGrid *Field, double FloorValue) {
  int nr, ns;

  nr = Field->Nrad;
  ns = Field->Nsec;

  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid ((ns+block.x-1)/block.x, (nr+block.y-1)/block.y);

//  printf ("%s\n",Field->Name);

  kernel_detectcrash <<< grid, block >>> (Field->gpu_field, 
                                          FloorValue, 
                                          Field->pitch/sizeof(double));
  
  cudaThreadSynchronize();
  getLastCudaError ("kernel_detectcrash failed!");

  return NO;
}
