/** \file "partialreduction.cu" : implements the kernel for a summation of an array in azimuth only
(the resulting array is a vector with NRAD elements)
*/
#define __CUDA 1
#include "fargo.h"
#undef __CUDA
#include <stdarg.h>
#include <helper_cuda.h>
#include <cuda.h>

// #define BLOCK_X DEF_BLOCK_X_PARTIALREDUCTION
#define BLOCK_X 64 // cannot be larger - otherwise kernel needs editing
#define BLOCK_Y 4

#define GET_TAB(u,x,y,pitch) *(u + __mul24(y, pitch) + x)

__global__ void kernel_reducesum (double *array2D,
                                  double *buffer,
                                  int pitch,
                                  int size) {
                                    
  __shared__ double sdata[BLOCK_X*BLOCK_Y];
  unsigned int tid = threadIdx.x;
  unsigned int yt  = threadIdx.y*blockDim.x;
  unsigned int jg  = threadIdx.x + blockIdx.x * __mul24(blockDim.x, 2);
  unsigned int ig  = threadIdx.y + blockIdx.y * blockDim.y;

  unsigned int ytid = yt+tid;
  
  sdata[ytid] = 0.0;
  if (jg < size)
    sdata[ytid] = GET_TAB (array2D, jg, ig, pitch);
  if (jg+blockDim.x < size)
    sdata[ytid] += GET_TAB (array2D, jg+blockDim.x, ig, pitch);
  __syncthreads ();
  
  if (tid < 32) {
    volatile double *smem = sdata;
    smem[ytid] += smem[ytid+32];
    smem[ytid] += smem[ytid+16];
    smem[ytid] += smem[ytid+8];
    smem[ytid] += smem[ytid+4];
    smem[ytid] += smem[ytid+2];
    smem[ytid] += smem[ytid+1];
  }

  if (tid == 0)
    GET_TAB (buffer, blockIdx.x, ig, pitch) = sdata[yt];
}

extern "C"
void AzimuthalAverage (PolarGrid *array, double *res) {
  //static int FirstCall = YES;

  int nr, ns, nxbar;

  nr = array->Nrad;
  ns = array->Nsec;

 // if (FirstCall) {
 //   Buffer = CreatePolarGrid (nr, ns, "bufferWorkArray");
 //   FirstCall = NO;
 // }

  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid;

  nxbar = ns;
  grid.x = ((nxbar+block.x-1)/block.x+1)/2;
  grid.y = (nr+block.y-1)/block.y;
//printf ("%d %d\n", grid.x, grid.y);
//exit(-1);
  kernel_reducesum <<< grid, block >>> (array->gpu_field,
                                        Buffer->gpu_field,
                                        array->pitch/sizeof(double),
                                        nxbar);

  cudaThreadSynchronize();
  getLastCudaError ("kernel_reducesum failed. First one");

  //  nxbar = (nxbar+2*BLOCK_X-1)/(2*BLOCK_X);
  nxbar = (nxbar+BLOCK_X-1)/(BLOCK_X);

  while (nxbar > 1) {
    grid.x = ((nxbar+block.x-1)/block.x+1)/2;
    grid.y = (nr+block.y-1)/block.y;
    
    kernel_reducesum <<< grid, block >>> (Buffer->gpu_field,
					                                Buffer->gpu_field,
                                          array->pitch/sizeof(double), nxbar);

    cudaThreadSynchronize();
    getLastCudaError ("kernel_reducesum failed. Second one");
    
    //    nxbar = (nxbar+2*BLOCK_X-1)/(2*BLOCK_X);
    nxbar = (nxbar+BLOCK_X-1)/(BLOCK_X);
  }
  cudaMemcpy2D (res, sizeof(double), Buffer->gpu_field, Buffer->pitch, sizeof(double), nr, cudaMemcpyDeviceToHost);
}
