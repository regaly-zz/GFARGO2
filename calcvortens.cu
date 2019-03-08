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
#define BLOCK_X 8
// BLOCK_Y : in radius
#define BLOCK_Y 8

#define rmed(i) CRadiiStuff[(nr+1)*6+i]
#define rinf(i) CRadiiStuff[(nr+1)*4+i]

//__constant__ double CRadiiStuff[8192];
__device__  double CRadiiStuff[32768];


__global__ void kernel_calcvortens (double *dens,
			                              int     pitch, 
                                    double *vrad, 
                                    double *vtheta,
                                    double *vort, 
                                    int     nr, 
                                    double  da) {
                               
  __shared__ double sdens[(BLOCK_X+1)*(BLOCK_Y+1)];
  __shared__ double svrad[(BLOCK_X+1)*(BLOCK_Y+1)];
  __shared__ double svtheta[(BLOCK_X+1)*(BLOCK_Y+1)];
  double rmedp, rmedm, vortensity;
  int i, j, m, is, js, ms;
  j = blockIdx.x*blockDim.x + threadIdx.x;
  i = blockIdx.y*blockDim.y + threadIdx.y;

  rmedp = rmed(i);
  if (i > 0)
    rmedm = rmed(i-1);
  else 
    rmedm = 2*rmedp-rmed(i+1);

  m = j+i*pitch;
  js = threadIdx.x+1;
  is = threadIdx.y+1;
  ms = js+is*(BLOCK_X+1);

  sdens[ms]   = dens[m];
  svrad[ms]   = vrad[m];
  svtheta[ms] = vtheta[m];

  __syncthreads ();

  if (is ==1) {
    if (i > 0) {
      sdens[ms-BLOCK_X-1] = dens[m-pitch];
      svtheta[ms-BLOCK_X-1] = vtheta[m-pitch];
    }
    else {
      sdens[ms-BLOCK_X-1] = sdens[ms];
      svtheta[ms-BLOCK_X-1] = svtheta[ms];
    }
  }

  __syncthreads ();

  if (js == 1) {
    if (j > 0) {
      svrad[ms-1] = vrad[m-1];
      sdens[ms-1] = dens[m-1];
    } else {
      svrad[ms-1] = vrad[m-1+pitch];
      sdens[ms-1] = dens[m-1+pitch];
    }
  }

  __syncthreads ();

  if ((is == 1) && (js == 1) && (i > 0) && (j > 0))
    sdens[0] = dens[m-1-pitch];
  if ((is == 1) && (js == 1) && (i == 0) && (j > 0))
    sdens[0] = dens[m-1];
  if ((is == 1) && (js == 1) && (i > 0) && (j == 0))
    sdens[0] = dens[m-1];
  if ((is == 1) && (js == 1) && (i == 0) && (j == 0))
    sdens[0] = dens[m-1+pitch];
  
  vortensity = (-svtheta[ms]*rmedp+svtheta[ms-BLOCK_X-1]*rmedm)*da + (svrad[ms] - svrad[ms-1])*(rmedp-rmedm);
  vortensity /= (.5*da * (rmedp*rmedp-rmedm*rmedm));
  vortensity += .5*sqrt(1./rinf(i));
  vortensity /= (.25*(sdens[ms]+sdens[ms-1]+sdens[ms-BLOCK_X-1]+sdens[ms-BLOCK_X-2]));
  vort[m] = vortensity;
}

void CalcVortens_gpu (PolarGrid *Vortensity, PolarGrid *Rho,
		              PolarGrid *vrad, PolarGrid *vtheta) {
                    
  dim3 grid;
  dim3 block = dim3(BLOCK_X, BLOCK_Y);
  grid.x = NSEC / BLOCK_X;
  grid.y = NRAD / BLOCK_Y;
 
  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)RadiiStuff, (size_t)(14*(NRAD+1))*sizeof(double),	0, cudaMemcpyHostToDevice));
  
  kernel_calcvortens <<<grid, block>>> (Rho->gpu_field,
                                        Rho->pitch/sizeof(double), 
                                        vrad->gpu_field, 
                                        vtheta->gpu_field,
                                        Vortensity->gpu_field,
                                        NRAD, 
                                        2.0*PI/(double)NSEC);

  cudaThreadSynchronize();
  getLastCudaError ("kernel_fillvortens failed");
}
