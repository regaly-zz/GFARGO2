/** \file bc_nonref.cu: contains a CUDA kernel for non-reflecting inner and outer boundary conditions.
*/

#define __CUDA 1
#include "fargo.h"
#undef __CUDA
#include <stdarg.h>
#include <helper_cuda.h>
#include <cuda.h>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/extrema.h>

// BLOCK_X : in azimuth
//#define BLOCK_X DEF_BLOCK_X_NONREFBC
#define BLOCK_X 64
#define BLOCK_Y 1
/* Note that with the above choice of BLOCK_Y, the occupancy is not 1,
   but there is less arithmetic done within the kernels, and the
   performance turns out to be better. */

__global__ void kernel_nonrefbc_in1 (double      *vrad,
                                     double      *vtheta,
                                     double      *rho,
                                     const int    nr,
                                     const int    ns,
                                     const double SigmaMed,
                                     const double SoundSpeed,
                                     const int    i_angle) {

  const int jg = blockDim.x * blockIdx.x + threadIdx.x;
  const int l = jg+ns;
  int jp = jg+i_angle;
  
  if (jp >= ns) 
    jp -= ns;
  if (jp < 0) 
    jp += ns;
  
  const int lp = jp;
  rho[lp] = rho[l];		/* copy first ring into ghost ring */
  const double vr_med = -SoundSpeed*(rho[l]-SigmaMed)/SigmaMed;
  vrad[l] = 2.0*vr_med-vrad[l+ns];  
}

__global__ void kernel_nonrefbc_in2 (double      *rho,
                                     const double SigmaMed,
                                     const double mean) {

  const int jg = blockDim.x * blockIdx.x + threadIdx.x;

  rho[jg] += SigmaMed-mean;
}

__global__ void kernel_nonrefbc_out1 (double      *vrad,
                                      double      *vtheta,
                                      double      *rho,
                                      const int    nr,
                                      const int    ns,
                                      const double SigmaMed,
                                      const double SoundSpeed,
                                      const int    i_angle) {

  const int jg = blockDim.x * blockIdx.x + threadIdx.x;
  const int l = jg+(nr-1)*ns;
  int jp = jg+i_angle;
  
  if (jp >= ns) 
    jp -= ns;
  if (jp < 0) 
    jp += ns;
  const int lp = jp+(nr-2)*ns;
  rho[lp] = rho[l];		/* copy first ring into ghost ring */
  const double vr_med = -SoundSpeed*(rho[l]-SigmaMed)/SigmaMed;
  vrad[l] = 2.0*vr_med-vrad[l-ns];
}


__global__ void kernel_nonrefbc_out2 (double      *rho,
                                      const int nr,
                                      const int ns,
                                      const double SigmaMed,
                                      const double mean) {

  const int jg = blockDim.x * blockIdx.x + threadIdx.x;

  rho[jg+(nr-1)*ns] += SigmaMed-mean;
}




extern "C" void NonReflectingBoundary_gpu (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, int where) {
  
  int ns = Rho->Nsec;
  int nr = Rho->Nrad;


  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid;
  grid.x = (ns+block.x-1)/block.x;
 

  if (where == INNER) {
    int nb_block_y = (1+BLOCK_Y-1)/BLOCK_Y;
    grid.y = nb_block_y;

    int i_angle;
    double dangle;

    dangle = (pow(Rinf[1],-1.5)-pow(21.0, -1.5))/(0.5*(SOUNDSPEED[0]+SOUNDSPEED[1]));
    dangle *= (Rmed[1]-Rmed[0]);
    i_angle = (int)(dangle/2.0/M_PI*(double)NSEC+0.5);

    if (i_angle < 0) 
     i_angle = -100;

    kernel_nonrefbc_in1 <<< grid, block >>> (Vrad->gpu_field,
                                             Vtheta->gpu_field,
                                             Rho->gpu_field,
                                             nr,
                                             ns,
                                             SigmaMed[1],
                                             SOUNDSPEED[1],
                                             i_angle);

    cudaThreadSynchronize();
    getLastCudaError ("kernel_nonrefbc_in1 failed");

    const thrust::device_ptr<double> d_rho_mean(Rho->gpu_field);
    double rho_mean = thrust::reduce(d_rho_mean, d_rho_mean + ns, (double) 0, thrust::plus<double>());
    rho_mean /= (double) ns;
    
    kernel_nonrefbc_in2 <<< grid, block >>> (Rho->gpu_field,
                                             SigmaMed[0],
                                             rho_mean);

    cudaThreadSynchronize();
    getLastCudaError ("kernel_nonrefbc_in2 failed");
 }
 
 if (where == OUTER) {
   int nb_block_y = (1+BLOCK_Y-1)/BLOCK_Y;
   grid.y = nb_block_y;

   int i_angle;
   double dangle;

   dangle = (pow(Rinf[nr-2],-1.5)-pow(20.0, -1.5))/(0.5*(SOUNDSPEED[nr-1]+SOUNDSPEED[nr-2]));
   dangle *= (Rmed[nr-1]-Rmed[nr-2]);
   i_angle = (int)(dangle/2.0/M_PI*(double)NSEC+0.5);

   if (i_angle < 0) 
     i_angle = 0;

   kernel_nonrefbc_out1 <<< grid, block >>> (Vrad->gpu_field,
                                             Vtheta->gpu_field,
                                             Rho->gpu_field,
                                             nr,
                                             ns,
                                             SigmaMed[nr-2],
                                             SOUNDSPEED[nr-1],
                                             i_angle);
   cudaThreadSynchronize();
   getLastCudaError ("kernel_nonrefbc_out1 failed");
                                            
   thrust::device_ptr<double> d_rho_mean(Rho->gpu_field);
   double rho_mean = thrust::reduce(d_rho_mean+(nr-1)*ns,  d_rho_mean + nr * ns, (double) 0, thrust::plus<double>());
   rho_mean /= (double) ns;
   
   kernel_nonrefbc_out2 <<< grid, block >>> (Rho->gpu_field,
                                             nr,
                                             ns,
                                             SigmaMed[nr-1],
                                             rho_mean);

   cudaThreadSynchronize();
   getLastCudaError ("kernel_nonrefbc_out2 failed");
 }
}
