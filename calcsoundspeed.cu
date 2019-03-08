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

__device__  double CRadiiStuff[32768];

__global__ void kernel_adiabatic_soundspeed (double *energy,
                                             double *dens,
                                             double *cs, 
                                             double  adiabatic_index,
 	                                           int     pitch) {
                               
  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;
  int idg = __mul24(ig, pitch) + jg;

  cs[idg] = sqrt( adiabatic_index*(adiabatic_index-1.0)*energy[idg]/dens[idg]);
}

extern "C"
void CalcSoundSpeed_gpu (PolarGrid *Rho, PolarGrid *Energy, PolarGrid *CS) {
                    
  int nr = CS->Nrad;
  int ns = CS->Nsec;

  //dim3 grid;
  dim3 block = dim3(BLOCK_X, BLOCK_Y);
  dim3 grid ((ns+block.x-1)/block.x, (nr+block.y-1)/block.y);
 
  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)RadiiStuff, (size_t)(14*(NRAD+1))*sizeof(double),	0, cudaMemcpyHostToDevice));
  
  if (Adiabatic) {
    kernel_adiabatic_soundspeed <<<grid, block>>> (Energy->gpu_field,
                                                   Rho->gpu_field,
                                                   CS->gpu_field,
                                                   ADIABATICINDEX,
                                                   CS->pitch/sizeof(double));

    cudaThreadSynchronize();
    getLastCudaError ("kernel_adiabatic_soundspeed failed");

    // azimuthal averaging
    //double csound[nr];
    //AzimuthalAverage (CS, csound);
    //for (int i = 0; i < nr; i++) {
    //  SOUNDSPEED[i] = csound[i]/(double)ns;
    //  CS2[i] = SOUNDSPEED[i]*SOUNDSPEED[i];
    //}
  }
}
