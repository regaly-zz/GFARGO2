/** \file "advectshift.cu" : implements the kernel for the "advectshift" procedure
*/
#define __CUDA 1
#include "fargo.h"
#undef __CUDA
#include <stdarg.h>
#include <helper_cuda.h>
#include <cuda.h>

// BLOCK_X : in azimuth
//#define BLOCK_X DEF_BLOCK_X_ADVECTSHIFT
#define BLOCK_X 16
// BLOCK_Y : in radius
#define BLOCK_Y 2

#define GET_TAB(u,x,y,pitch) *(u + __mul24(y, pitch) + x)

//PolarGrid *WorkShift;

__device__ int Shift[32768];

__global__ void kernel_advsh (double *num, 
                              double *work,
			                        int ns, int pitch) {

  __shared__ double buffer[(6*BLOCK_X)*BLOCK_Y];
  int jg = threadIdx.x + blockIdx.x * blockDim.x * 4;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;
  int js = threadIdx.x;
  int is = threadIdx.y;
  int jgm, jgp;
  int ids = js+is*6*blockDim.x;

  int nshift = Shift[ig];
  buffer[ids +   blockDim.x] = GET_TAB (num, jg,              ig, pitch);
  buffer[ids + 2*blockDim.x] = GET_TAB (num, jg+blockDim.x,   ig, pitch);
  buffer[ids + 3*blockDim.x] = GET_TAB (num, jg+blockDim.x*2, ig, pitch);
  buffer[ids + 4*blockDim.x] = GET_TAB (num, jg+blockDim.x*3, ig, pitch);
  if (nshift > 0) {
    jgm = jg - blockDim.x;
    if (jgm < 0) jgm += ns;
    buffer[ids] = GET_TAB (num, jgm, ig, pitch);
  }
  if (nshift < 0) {
    jgp = jg + 4*blockDim.x;
    if (jgp >= ns) jgp -= ns;
    buffer[ids + 5*blockDim.x] = GET_TAB (num, jgp, ig, pitch);
  }
  __syncthreads ();
  GET_TAB (work, jg, ig, pitch)              = buffer[ids +   blockDim.x - nshift];
  GET_TAB (work, jg+blockDim.x, ig, pitch)   = buffer[ids + 2*blockDim.x - nshift];
  GET_TAB (work, jg+2*blockDim.x, ig, pitch) = buffer[ids + 3*blockDim.x - nshift];
  GET_TAB (work, jg+3*blockDim.x, ig, pitch) = buffer[ids + 4*blockDim.x - nshift];
}

extern "C"
void AdvectSHIFT_gpu (PolarGrid *array, int *Nshift) {
  //static int FirstTime = YES;
  int nr, ns;
  double *temp_gpu_ptr;

  nr = array->Nrad;
  ns = array->Nsec;

  //if (FirstTime) {
  //  WorkShift = CreatePolarGrid (nr, ns, "WorkShift");
  //  FirstTime = NO;
  //}

  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid ((ns+block.x-1)/block.x/4, (nr+block.y-1)/block.y);
  
  checkCudaErrors(cudaMemcpyToSymbol(Shift, (void *)Nshift, (size_t)(nr*sizeof(int)), 0, cudaMemcpyHostToDevice));

  kernel_advsh <<< grid, block >>> (array->gpu_field, 
                                    WorkShift->gpu_field, 
                                    ns, 
                                    array->pitch/sizeof(double));
    
  cudaThreadSynchronize();
  getLastCudaError("Kernel advsh execution failed");
  
  /* Swap gpu arrays to avoid memcpy on device (would halve the performance) */
  temp_gpu_ptr = array->gpu_field;
  array->gpu_field = WorkShift->gpu_field;
  WorkShift->gpu_field = temp_gpu_ptr;
}
