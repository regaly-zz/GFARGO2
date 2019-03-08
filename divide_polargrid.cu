/** \file "template.cu" : implements the kernel for the "template" procedure
*/
#define __CUDA 1
#include "fargo.h"
#undef __CUDA
#include <stdarg.h>
#include <helper_cuda.h>
#include <cuda.h>

// BLOCK_X : in azimuth
//#define BLOCK_X DEF_BLOCK_X_DIVIDE_POLARGRID
#define BLOCK_X 32
// BLOCK_Y : in radius
#define BLOCK_Y 8

#define GET_TAB(u,x,y,pitch) *(u + __mul24(y, pitch) + x)

__global__ void kernel_divpg (double *num,
                              double *denom,
                              double *res,
                              int pitch) {
                                
  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;

  int idx =  ig * pitch + jg;

  res[idx] =   num[idx] /denom[idx];
}

__global__ void kernel_add (double *Arr1,
                            double *Arr2,
                            double *res,
                            int pitch) {
                                
  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;

  int idx =  ig * pitch + jg;

  res[idx] =   Arr1[idx] + Arr2[idx];
}

__global__ void kernel_add (double *Arr1,
                            double *Arr2,
                            double *Arr3,
                            double *res,
                            int pitch) {
                                
  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;

  int idx =  ig * pitch + jg;

  res[idx] =   Arr1[idx] + Arr2[idx] + Arr3[idx];
}

__global__ void kernel_add (double *Arr1,
                            double *Arr2,
                            double *Arr3,
                            double *Arr4,                            
                            double *res,
                            int pitch) {
                                
  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;

  int idx =  ig * pitch + jg;

  res[idx] =   Arr1[idx] + Arr2[idx] + Arr3[idx] + Arr4[idx];
}


extern "C"
void DivisePolarGrid_gpu (PolarGrid *Num, PolarGrid *Denom, PolarGrid *Res)
{
  int nr, ns;

  nr = Num->Nrad;
  ns = Num->Nsec;
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid ((ns+block.x-1)/block.x, (nr+block.y-1)/block.y);

  kernel_divpg <<< grid, block >>> (Num->gpu_field,
                                    Denom->gpu_field,
                                    Res->gpu_field,
                                    Num->pitch/sizeof(double));
  
  cudaThreadSynchronize();
  getLastCudaError ("Kernel divpg failed");
}


extern "C"
void SumPolarGrid_gpu (int num, PolarGrid **Arr1, PolarGrid *Arr2, PolarGrid *Res) {
  int nr, ns;

  nr = Res->Nrad;
  ns = Res->Nsec;
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid ((ns+block.x-1)/block.x, (nr+block.y-1)/block.y);

  switch (DustBinNum) {
    case 1:   kernel_add <<< grid, block >>> (Arr1[0]->gpu_field,
                                              Arr2->gpu_field,
                                              Res->gpu_field,
                                              Res->pitch/sizeof(double));
              break;

    case 2:   kernel_add <<< grid, block >>> (Arr1[0]->gpu_field,
                                              Arr1[1]->gpu_field,
                                              Arr2->gpu_field,
                                              Res->gpu_field,
                                              Res->pitch/sizeof(double));
              break;
    case 3:   kernel_add <<< grid, block >>> (Arr1[0]->gpu_field,
                                              Arr1[1]->gpu_field,
                                              Arr1[2]->gpu_field,
                                              Arr2->gpu_field,
                                              Res->gpu_field,
                                              Res->pitch/sizeof(double));
               break;
  }
    
  
  cudaThreadSynchronize();
  getLastCudaError ("Kernel divpg failed");
}
