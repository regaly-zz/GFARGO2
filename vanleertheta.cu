/** \file "vanleertheta.cu" : implements the kernel for the "VanLeerTheta" procedure
*/
#define __CUDA 1
#include "fargo.h"
#undef __CUDA
#include <stdarg.h>
#include <helper_cuda.h>
#include <cuda.h>

// BLOCK_X : in azimuth
//#define BLOCK_X 16
//#define BLOCK_X DEF_BLOCK_X_VANLEERTHETA
#define BLOCK_X 32
// BLOCK_Y : in radius
#define BLOCK_Y 8

#define GET_TAB(u,x,y,pitch) *(u + __mul24(y, pitch) + x)
#define invsurf      CRadiiStuff[(nr+1)*7+ig]
#define invrmed      CRadiiStuff[(nr+1)*2 + ig]


//__constant__ double CRadiiStuff[8192];
__device__ double CRadiiStuff[32768];

extern PolarGrid *RhoStar, *QRStar, *Work;


__global__ void kernel_vlth (double *rhos,
                             double *qrs,
                             double *vt,
                             double *qb,
                             int ns, 
                             int nr, 
                             int pitch,
                             double dt) {
                               
  __shared__ double srhos[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double svt[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double sqrs[(BLOCK_X+2)*(BLOCK_Y+2)];
  //double dr, invsurf, fluxp, fluxm;

  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;
  // js & is, l like 'local' (shared memory <=> local patch)
  int js = threadIdx.x + 1;
  int is = threadIdx.y + 1;
  int ids = is*(blockDim.x+2)+js;
  int idg = __mul24(ig, pitch) + jg;
  int jgp = jg+1;
  if (jgp == ns) jgp = 0;

  // We perform a coalesced read of 'rhos' into the shared memory;
  srhos[ids] = rhos[idg];
  // EDGE 2: "RIGHT EDGE".
  if (js == blockDim.x)
    srhos[is*(blockDim.x+2)+blockDim.x+1] = GET_TAB (rhos, jgp, ig, pitch);

  // We perform a coalesced read of 'qrs' into the shared memory;
  sqrs[ids] = qrs[idg];
  // EDGE 2: "RIGHT EDGE".
  if (js == blockDim.x)
    sqrs[is*(blockDim.x+2)+blockDim.x+1] = GET_TAB (qrs, jgp, ig, pitch);

  // We perform a coalesced read of 'vt' into the shared memory;
  svt[ids] = vt[idg];
  // EDGE 2: "RIGHT EDGE".
  if (js == blockDim.x)
    svt[is*(blockDim.x+2)+blockDim.x+1] = GET_TAB (vt, jgp, ig, pitch);

  __syncthreads ();

  const double dr = CRadiiStuff[(nr+1)*8+ig] - CRadiiStuff[(nr+1)*4+ig];
//  invsurf = CRadiiStuff[(nr+1)*7+ig];
  
  const double fluxm = sqrs[ids] * srhos[ids] * svt[ids];
  const double fluxp = sqrs[ids+1] * srhos[ids+1] * svt[ids+1];
//  qb[idg] += __dadd_rn(fluxm,-fluxp)*invsurf*dr*dt;
  qb[idg] += (fluxm - fluxp)*invsurf*dr*dt;
}


extern "C"
void VanLeerTheta_gpu_cu (PolarGrid *Vtheta, PolarGrid *Qbase, double dt)
{
  int nr, ns;

  nr = Vtheta->Nrad;
  ns = Vtheta->Nsec;
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid ((ns+block.x-1)/block.x, (nr+block.y-1)/block.y);

  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)RadiiStuff, (size_t)(14*(nr+1))*sizeof(double),	0, cudaMemcpyHostToDevice));

  kernel_vlth <<< grid, block >>> (RhoStar->gpu_field,
                                   QRStar->gpu_field,
                                   Vtheta->gpu_field,
                                   Qbase->gpu_field,
                                   Vtheta->Nsec, 
                                   Vtheta->Nrad,
                                   Vtheta->pitch/sizeof(double),
                                   dt);
  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError ();
  if ( cudaSuccess != err) {
    fprintf (stderr, "Cuda error kernel vlth failed \t%s\n", cudaGetErrorString (err));
    exit (-1);
  }
}


__global__ void kernel_vlthds (double *rhos,
                               double *qrs,
                               double *vt,
                               double *qb,
                               double  invdphi,
                               int     ns, 
                               int     nr, 
                               int     pitch,
                               double  dt) {
                               
  __shared__ double srhos[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double svt[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double sqrs[(BLOCK_X+2)*(BLOCK_Y+2)];
  //double dr, invsurf, fluxp, fluxm;

  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;
  // js & is, l like 'local' (shared memory <=> local patch)
  int js = threadIdx.x + 1;
  int is = threadIdx.y + 1;
  int ids = is*(blockDim.x+2)+js;
  int idg = __mul24(ig, pitch) + jg;
  int jgp = jg+1;
  if (jgp == ns) jgp = 0;

  // We perform a coalesced read of 'rhos' into the shared memory;
  srhos[ids] = rhos[idg];
  // EDGE 2: "RIGHT EDGE".
  if (js == blockDim.x)
    srhos[is*(blockDim.x+2)+blockDim.x+1] = GET_TAB (rhos, jgp, ig, pitch);

  // We perform a coalesced read of 'qrs' into the shared memory;
  sqrs[ids] = qrs[idg];
  // EDGE 2: "RIGHT EDGE".
  if (js == blockDim.x)
    sqrs[is*(blockDim.x+2)+blockDim.x+1] = GET_TAB (qrs, jgp, ig, pitch);

  // We perform a coalesced read of 'vt' into the shared memory;
  svt[ids] = vt[idg];
  // EDGE 2: "RIGHT EDGE".
  if (js == blockDim.x)
    svt[is*(blockDim.x+2)+blockDim.x+1] = GET_TAB (vt, jgp, ig, pitch);

  __syncthreads ();

  const double dr = CRadiiStuff[(nr+1)*8+ig] - CRadiiStuff[(nr+1)*4+ig];
//  invsurf = CRadiiStuff[(nr+1)*7+ig];

  //const double fluxm = sqrs[ids] * svt[ids] * srhos[ids];
  //const double fluxp = sqrs[ids+1] * svt[ids+1] * srhos[ids+1];
  const double fluxm = sqrs[ids] * svt[ids];
  const double fluxp = sqrs[ids+1] * svt[ids+1];
  
  //qb[idg] += ((fluxm - fluxp) - srhos[ids]*sqrs[ids]*(svt[ids] - svt[ids+1])) * invsurf * dr * dt;
  qb[idg] += ((fluxm - fluxp) - sqrs[ids]*(svt[ids] - svt[ids+1])) * invsurf * dr * dt;  
}


extern "C"
void VanLeerThetaDustSize_gpu_cu (PolarGrid *Vtheta, PolarGrid *Qbase, double dt)
{
  int nr, ns;

  nr = Vtheta->Nrad;
  ns = Vtheta->Nsec;
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid ((ns+block.x-1)/block.x, (nr+block.y-1)/block.y);

  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)RadiiStuff, (size_t)(14*(nr+1))*sizeof(double),	0, cudaMemcpyHostToDevice));

  kernel_vlthds <<< grid, block >>> (RhoStar->gpu_field,
                                     QRStar->gpu_field,
                                     Vtheta->gpu_field,
                                     Qbase->gpu_field,
                                     ((double) ns)/2.0/M_PI,
                                     Vtheta->Nsec, 
                                     Vtheta->Nrad,
                                     Vtheta->pitch/sizeof(double),
                                     dt);
  cudaThreadSynchronize();
  getLastCudaError ("kernel_vlthds failed");
}