/** \file "computeresidual.cu" : implements the kernel for the "computeresidual" procedure
*/
#define __CUDA 1
#include "fargo.h"
#undef __CUDA
#include <stdarg.h>
#include <helper_cuda.h>
#include <cuda.h>

extern PolarGrid *VthetaRes;
extern int Nshift[MAX1D];

// BLOCK_X : in azimuth
//#define BLOCK_X DEF_BLOCK_X_COMPUTERESIDUAL
#define BLOCK_X 4
// BLOCK_Y : in radius
#define BLOCK_Y 8

#define GET_TAB(u,x,y,pitch) *(u + __mul24(y, pitch) + x)
#define vmed  Vels[ig]
#define vtemp Vels[ig+nr]

__constant__ double Vels[8192];

extern "C" void AzimuthalAverage (PolarGrid *array, double *res);


__global__ void kernel_compr (double *vt,
			                        double *vtr,
                              int nr, int pitch) {
  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;

  //double vmed = Vels[ig];
  //double vtemp = Vels[ig+nr];

  double vazim = GET_TAB(vt, jg, ig, pitch);
  GET_TAB (vtr, jg, ig, pitch) = vazim - vmed;
  GET_TAB (vt, jg, ig, pitch) = vtemp; 
}

extern "C"
void ComputeResiduals_gpu (PolarGrid *Vtheta, double dt, double OmegaFrame)
{
  int nr, ns;
  static int FirstTime = YES;
  static double *Vtot, *Vmed, *Vtemp;
  double DVmed, DVtemp;
  double Ntilde, Nround;

  nr = Vtheta->Nrad;
  ns = Vtheta->Nsec;
  if (FirstTime) {
    Vtot    = (double *)malloc (nr*sizeof(double));
    Vmed    = (double *)malloc (2*nr*sizeof(double));
    Vtemp   = Vmed + nr;
    FirstTime = NO;
  }

  AzimuthalAverage (Vtheta, Vtot);
  
  cudaError_t err = cudaGetLastError ();
  if ( cudaSuccess != err) {
    fprintf (stderr, "pb after azim average in computeresidualsgpu %s\n",cudaGetErrorString (err));
    exit (-1);
  }

  for (int i=0; i < nr; i++) {
    DVmed = (double)(Vtot[i])/(double)ns;
    Vmed[i] = (double)DVmed;
    Ntilde = (DVmed+DInvSqrtRmed[i]-OmegaFrame*DRmed[i])/DRmed[i]*dt*(double)ns/2.0/M_PI;
    Nround = floor(Ntilde+0.5);
    Nshift[i] = (int)Nround;
    DVtemp = (Ntilde-Nround)*Rmed[i]/dt*2.0*M_PI/(double)ns;
    Vtemp[i] = (double)DVtemp;
  }
  checkCudaErrors(cudaMemcpyToSymbol(Vels, (void *)Vmed, (size_t)(2*nr)*sizeof(double), 0, cudaMemcpyHostToDevice));
  
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid ((ns+block.x-1)/block.x, (nr+block.y-1)/block.y);
  
  kernel_compr <<< grid, block >>> (Vtheta->gpu_field,
                                    VthetaRes->gpu_field,
                                    nr,
                                    Vtheta->pitch/sizeof(double));
  
  cudaThreadSynchronize();
  getLastCudaError("Kernel compr execution failed");

}
