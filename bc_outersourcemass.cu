/** \file bc_outersourcemass.cu: contains a CUDA kernel for applying outer source mass.
*/
#define __CUDA 1
#include "fargo.h"
#undef __CUDA
#include <stdarg.h>
#include <helper_cuda.h>
#include <cuda.h>

// BLOCK_X : in azimuth
//#define BLOCK_X DEF_BLOCK_X_OUTERSOURCEMASS
#define BLOCK_X 64
#define BLOCK_Y 1
/* Note that with the above choice of BLOCK_Y, the occupancy is not 1,
   but there is less arithmetic done within the kernels, and the
   performance turns out to be better. */

#define GET_TAB(u,x,y,pitch) *(u + __mul24(y, pitch) + x)

__device__ double __atomicAdd(double* address, 
                            double val) {
                              
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void kernel_outerrhosum (double *rho, 
                                    int nr, 
                                    int pitch, 
                                    double* sum) {

  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = blockIdx.y+nr-1;
  int idg = __mul24(ig, pitch) + jg;

  int tid = threadIdx.x;
  //int i = blockIdx.x*blockDim.x + threadIdx.x;
  
  // Each block loads its elements into shared memory 
  __shared__ double x[BLOCK_X];
//  x[tid] = (idg < 1024) ? rho[idg] : 0; // last block may pad with 0’s 
  x[tid] = rho[idg];
  __syncthreads();
  
  // Build summation tree over elements. 
  for(int s = blockDim.x/2; s > 0; s = s/2) {
    if(tid < s) x[tid] += x[tid + s];
    __syncthreads(); 
  }

  // Thread 0 adds the partial sum to the total sum 
  if(tid == 0) {
    *sum = 0.0;
    __atomicAdd(sum, x[tid]);
  }
}

__global__ void kernel_applyoutersourcemass (double *vrad, 
                                             double *rho, 
                                             int nr, 
                                             int pitch, 
                                             double avg_rho, 
                                             double penul_vr) {

  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = blockIdx.y+nr-1;
  int idg = __mul24(ig, pitch) + jg;

  rho[idg] += avg_rho;
//  vrad[idg] = penul_vr;
  vrad[idg] = 0;
  vrad[idg-768] = 0;
}


extern "C"
void ApplyOuterSourceMass_gpu (PolarGrid *Vrad, PolarGrid *Rho, double dustmass) {
  
  int nr = Rho->Nrad;
  int ns = Rho->Nsec;

  /*
  rho= Rho->Field;
  vr = Vrad->Field;
  i = nr-1;  
  for (j = 0; j < ns; j++) {
    l = j+i*ns;
    average_rho += rho[l];
  }
  */
  static bool FirstTime=YES;
  static double *dev_sum;
  if (FirstTime) {
    checkCudaErrors(cudaMalloc( (void**)&dev_sum, sizeof(double) ));
    FirstTime = NO;
  }
  double sum;
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid;
  grid.x = (ns+block.x-1)/block.x;
  grid.y = (1+block.y-1)/block.y;

  kernel_outerrhosum <<< grid, block >>> (Rho->gpu_field, 
                                          nr, 
                                          Rho->pitch/sizeof(double), 
                                          dev_sum);

  cudaThreadSynchronize();
  getLastCudaError ("kernel outerrhosum failed");

  checkCudaErrors(cudaMemcpy(&sum, dev_sum, sizeof(double), cudaMemcpyDeviceToHost));
  //cudaFree (dev_sum);
  double average_rho = sum/((double) ns);
  average_rho = dustmass*SigmaMed[nr-1]-average_rho;
  /*
  for (j = 0; j < ns; j++) {
    l = j+i*ns;
    rho[l] += average_rho;
  }

  i = nr-1;

  for (j = 0; j < ns; j++) {
    l = j+i*ns;
    vr[l] = penul_vr;
  }
  */
  double penul_vr = IMPOSEDDISKDRIFT*pow((Rinf[nr-1]/1.0),-SIGMASLOPE);;
  kernel_applyoutersourcemass <<< grid, block >>> (Vrad->gpu_field, 
                                                   Rho->gpu_field, 
                                                   nr, 
                                                   Rho->pitch/sizeof(double), 
                                                   average_rho, penul_vr);

  cudaThreadSynchronize();
  getLastCudaError ("kernel applyoutersourcemass failed");
}