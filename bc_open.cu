/** \file bc_open.cu: contains a CUDA kernel for open inner and outer boundary condition.
*/
#define __CUDA 1
#include "fargo.h"
#undef __CUDA
#include <stdarg.h>
#include <helper_cuda.h>
#include <cuda.h>

// BLOCK_X : in azimuth
//#define BLOCK_X DEF_BLOCK_X_OPENBC
#define BLOCK_X 16//64
#define BLOCK_Y 1


__global__ void kernel_openbc_in (double      *vrad,
                                  double      *vtheta,
                                  double      *rho,
                                  double      *energy,
                                  double      *dust_size,
                                  const int    ns,
                                  const double SigmaMed,
                                  const double VthetaMed0,
                                  const double VthetaMed1,
                                  bool         dust) {
                                    
  const int jg = blockDim.x * blockIdx.x + threadIdx.x;

  // gas: do not allow inflow
  if (!dust) {
    rho[jg] = rho[jg+ns];
    if (energy != NULL)
      energy[jg] = energy[jg+ns];
    
    // do not allow inflow for gas
    if (vrad[jg+ns+ns] > 0.0 || rho[jg+ns] < SigmaMed) {
      vrad[jg+ns] = 0.0;
    }
    else {
      vrad[jg+ns] = vrad[jg+ns+ns];
    }
  }

  // dust: inflow is possible for dust
  else {
    /*
    vrad[jg]    = 0.0;
    vrad[jg+ns] = 0.0;
    */
    //vtheta[jg] = vtheta[jg+ns];
    rho[jg] = rho[jg+ns];
    if (vrad[jg+ns+ns] > 0.0  || rho[jg+ns] < SigmaMed) {
      vrad[jg] = 0.0;
      //vrad[jg+ns] = 0.0;
    }
    else
      vrad[jg+ns] = vrad[jg+ns+ns];
  }
  
  // size of grown dust
  if (dust_size != NULL) {
    dust_size[jg] = dust_size[jg+ns];
  }
}


__global__ void kernel_openbc_out (double      *vrad,
                                   double      *vtheta,
                                   double      *rho,
                                   double      *energy,
                                   double      *dust_size,
                                   const int    nr,
                                   const int    ns,
                                   const double SigmaMed,
                                   bool         dust) {
                                    
  const int jg = blockDim.x * blockIdx.x + threadIdx.x;

  // gas: do not allow inflow
  if (!dust) {
    rho[jg+(nr-1)*ns] = rho[jg+(nr-2)*ns];
    if (energy != NULL)
      energy[jg+(nr-1)*ns] = energy[jg+(nr-2)*ns];
    
    if (vrad[jg+(nr-2)*ns] < 0.0 || rho[jg+(nr-2)*ns] < SigmaMed)
      vrad[jg+(nr-1)*ns] = 0.0;
    else
      vrad[jg+(nr-1)*ns] = vrad[jg+(nr-2)*ns];
  }
  // dust: inflow is possible for dust
  else {    
    rho[jg+(nr-1)*ns] = rho[jg+(nr-2)*ns];
    //rho[jg+(nr-1)*ns] = SigmaMed;//rho[jg+(nr-2)*ns];
    //rho[jg+(nr-2)*ns] = SigmaMed;
    
    if (vrad[jg+(nr-2)*ns] < 0.0) {// || rho[jg+(nr-2)*ns] < SigmaMed) {
      vrad[jg+(nr-1)*ns] = 0.0;
      vrad[jg+(nr-2)*ns] = 0.0;
    }
    else
      vrad[jg+(nr-1)*ns] = vrad[jg+(nr-2)*ns];
  }

  // size of grown dust
  if (dust_size != NULL) {
    dust_size[jg+(nr-1)*ns] = dust_size[jg+(nr-2)*ns];
  }
}


extern "C" void OpenBoundary_gpu (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, PolarGrid *Energy, int where) {
  int nr = Vrad->Nrad;
  int ns = Vrad->Nsec;
  
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid;
  grid.x = (ns+block.x-1)/block.x;
   
  double *energy_gpu_field = NULL;
  if (Adiabatic)
    energy_gpu_field = Energy->gpu_field;
  
  if (where == INNER) {
     int nb_block_y = (1+BLOCK_Y-1)/BLOCK_Y;
     grid.y = nb_block_y;
  
     kernel_openbc_in <<< grid, block >>> (Vrad->gpu_field,
                                           Vtheta->gpu_field,
                                           Rho->gpu_field,
                                           energy_gpu_field,
                                           NULL,
                                           ns,
                                           SigmaMed[0],
                                           GasVelThetaMed[0],
                                           GasVelThetaMed[1],
                                           false);             // dust=false
  
     cudaThreadSynchronize();
     getLastCudaError ("kernel_openbc_in failed");
  }
  if (where == OUTER) {

     int nb_block_y = (1+BLOCK_Y-1)/BLOCK_Y;
     grid.y = nb_block_y;
  
     kernel_openbc_out <<< grid, block >>> (Vrad->gpu_field,
                                            Vtheta->gpu_field,
                                            Rho->gpu_field,
                                            energy_gpu_field,
                                            NULL,
                                            nr,
                                            ns,
                                            SigmaMed[nr-2],
                                            false);             // dust=false
  
     cudaThreadSynchronize();
     getLastCudaError ("kernel_openbc_out failed");
  }
}

extern "C" void OpenBoundaryDust_gpu (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, int DustBin, int where) {
  int nr = Vrad->Nrad;
  int ns = Vrad->Nsec;
   
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid;
  grid.x = (ns+block.x-1)/block.x;

  double *dust_size_gpu_field = NULL;
  if (DustGrowth) {
    dust_size_gpu_field = dust_size->gpu_field;
  }

  if (where == INNER) {
     int nb_block_y = (1+BLOCK_Y-1)/BLOCK_Y;
     grid.y = nb_block_y;
  
     kernel_openbc_in <<< grid, block >>> (Vrad->gpu_field,
                                           Vtheta->gpu_field,
                                           Rho->gpu_field,
                                           NULL,
                                           dust_size_gpu_field,
                                           ns,
                                           SigmaMed[0]*DustMassBin[DustBin],
                                           0,
                                           0,
                                           true);
  
     cudaThreadSynchronize();
     getLastCudaError ("kernel_openbc_in failed");
  }

  if (where == OUTER) {
     int nb_block_y = (1+BLOCK_Y-1)/BLOCK_Y;
     grid.y = nb_block_y;
  
     kernel_openbc_out <<< grid, block >>> (Vrad->gpu_field,
                                            Vtheta->gpu_field,
                                            Rho->gpu_field,
                                            NULL,
                                            dust_size_gpu_field,
                                            nr,
                                            ns,
                                            SigmaMed[nr-2]*DustMassBin[DustBin],
                                            true);
  
     cudaThreadSynchronize();
     getLastCudaError ("kernel_openbc_out failed");
  }

}

