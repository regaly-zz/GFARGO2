/** \file "vanleerrad.cu" : implements the kernel for the "VanLeerRadial" procedure
*/
#define __CUDA 1
#include "fargo.h"
#undef __CUDA
#include <stdarg.h>
#include <helper_cuda.h>
#include <cuda.h>

// BLOCK_X : in azimuth
//#define BLOCK_X DEF_BLOCK_X_VALLEERAD
#define BLOCK_X 16
// BLOCK_Y : in radius
#define BLOCK_Y 8

#define GET_TAB(u,x,y,pitch) *(u + __mul24(y, pitch) + x)
#define invrmed      CRadiiStuff[(nr+1)*2+ig]
#define rinf         CRadiiStuff[(nr+1)*4+ig]
#define invsurf      CRadiiStuff[(nr+1)*7+ig]
#define rsup         CRadiiStuff[(nr+1)*8+ig]
#define invdiffrsup  CRadiiStuff[(nr+1)*10+ ig]

//__constant__ double CRadiiStuff[8192];
__device__ double CRadiiStuff[32768];

extern PolarGrid *RhoStar, *QRStar, *Work;

extern double StellarAccRate, LostMass;

// double version of atomic add
__device__ double _datomicAdd(double* address, 
                              double val) {
                                
    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}


__global__ void kernel_vlrd (double *rhos,
                             double *qrs,
                             double *vr,
                             double *qb,
                             int ns, 
                             int nr, 
                             int pitch,
                             double dtdtheta) {
             
  __shared__ double srhos[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double svr[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double sqrs[(BLOCK_X+2)*(BLOCK_Y+2)];
  //double rinf, rsup, invsurf, fluxp, fluxm;

  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;
  // js & is, l like 'local' (shared memory <=> local patch)
  int js = threadIdx.x + 1;
  int is = threadIdx.y + 1;
  int ids = is*(blockDim.x+2)+js;
  int idg = __mul24(ig, pitch) + jg;

  // We perform a coalesced read of 'rhos' into the shared memory;
  srhos[ids] = rhos[idg];
  // EDGE 4: "TOP EDGE". Be careful not to read anything if in last row...
  if ((is == blockDim.y) && (ig < nr-1))
    srhos[js+(blockDim.y+1)*(blockDim.x+2)] = GET_TAB (rhos, jg, ig+1, pitch);

  // We perform a coalesced read of 'qrs' into the shared memory;
  sqrs[ids] = qrs[idg];
  // EDGE 4: "TOP EDGE". Be careful not to read anything if in last row...
  if ((is == blockDim.y-1) && (ig < nr-2))
    sqrs[js+(blockDim.y+1)*(blockDim.x+2)] = GET_TAB (qrs, jg, ig+2, pitch);

  // We perform a coalesced read of 'vr' into the shared memory;
  svr[ids] = vr[idg];
  // EDGE 4: "TOP EDGE". Be careful not to read anything if in last row...
  if ((is == blockDim.y-2) && (ig < nr-3))
    svr[js+(blockDim.y+1)*(blockDim.x+2)] = GET_TAB (vr, jg, ig+3, pitch);

  __syncthreads ();

//  rinf = CRadiiStuff[(nr+1)*4+ig];
//  invsurf = CRadiiStuff[(nr+1)*7+ig];
//  rsup = CRadiiStuff[(nr+1)*8+ig];
  
  int idsp = ids+blockDim.x+2;

  double fluxm = 0.0;
  if (ig >= 0)
    fluxm = rinf * sqrs[ids] * srhos[ids] * svr[ids];
  double fluxp = 0.0;
  if (ig < nr-1)
    fluxp = rsup * sqrs[idsp] * srhos[idsp] * svr[idsp];
  qb[idg] += (fluxm-fluxp)*invsurf*dtdtheta;
}


__global__ void kernel_vlrd_lostmass (double *rhos,
                                      double *qrs,
                                      double *vr,
                                      double *qb,
                                      int ns, 
                                      int nr, 
                                      int pitch,
                                      double dtdtheta,
                                      double *gpu_lostmass) {
                                        
  __shared__ double srhos[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double svr[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double sqrs[(BLOCK_X+2)*(BLOCK_Y+2)];
  //double rinf, rsup, invsurf, fluxp, fluxm;

  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;
  // js & is, l like 'local' (shared memory <=> local patch)
  int js = threadIdx.x + 1;
  int is = threadIdx.y + 1;
  int ids = is*(blockDim.x+2)+js;
  int idg = __mul24(ig, pitch) + jg;

  // We perform a coalesced read of 'rhos' into the shared memory;
  srhos[ids] = rhos[idg];
  // EDGE 4: "TOP EDGE". Be careful not to read anything if in last row...
  if ((is == blockDim.y) && (ig < nr-1))
    srhos[js+(blockDim.y+1)*(blockDim.x+2)] = GET_TAB (rhos, jg, ig+1, pitch);

  // We perform a coalesced read of 'qrs' into the shared memory;
  sqrs[ids] = qrs[idg];
  // EDGE 4: "TOP EDGE". Be careful not to read anything if in last row...
  if ((is == blockDim.y-1) && (ig < nr-2))
    sqrs[js+(blockDim.y+1)*(blockDim.x+2)] = GET_TAB (qrs, jg, ig+2, pitch);

  // We perform a coalesced read of 'vr' into the shared memory;
  svr[ids] = vr[idg];
  // EDGE 4: "TOP EDGE". Be careful not to read anything if in last row...
  if ((is == blockDim.y-2) && (ig < nr-3))
    svr[js+(blockDim.y+1)*(blockDim.x+2)] = GET_TAB (vr, jg, ig+3, pitch);

  __syncthreads ();

//  rinf = CRadiiStuff[(nr+1)*4+ig];
//  invsurf = CRadiiStuff[(nr+1)*7+ig];
//  rsup = CRadiiStuff[(nr+1)*8+ig];
  
  int idsp = ids+blockDim.x+2;

  double fluxm = 0.0;
  if (ig >= 0)
    fluxm = rinf * sqrs[ids] * srhos[ids] * svr[ids];
  double fluxp = 0.0;
  if (ig < nr-1)
    fluxp = rsup * sqrs[idsp] * srhos[idsp] * svr[idsp];
  qb[idg] += (fluxm-fluxp)*invsurf*dtdtheta;

  // lost mass caclualtion
  if (ig == 0) {
    //$1
    gpu_lostmass[jg] = (fluxm-fluxp)*dtdtheta;
 }
}


extern "C"
void VanLeerRadial_gpu_cu (PolarGrid *Vrad, PolarGrid *Qbase, double dt, bool calc_lostmass)
{
  int nr, ns;
  nr = Vrad->Nrad;
  ns = Vrad->Nsec;
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid ((ns+block.x-1)/block.x, (nr+block.y-1)/block.y);

  static int FirstTime=YES;
  static double *gpu_lostmass;
  static double *lostmass;

  if (FirstTime) {

    checkCudaErrors(cudaMallocHost ((void **) &lostmass, sizeof(double) * ns));
    checkCudaErrors(cudaMalloc ((void **) &gpu_lostmass, sizeof(double) * ns));
    //checkCudaErrors(cudaMemcpy(gpu_lostmass, lostmass, sizeof(double) * ns, cudaMemcpyHostToDevice));

    FirstTime = NO;
  } 

  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)RadiiStuff, (size_t)(14*(nr+1))*sizeof(double),	0, cudaMemcpyHostToDevice));


  // calculation of lost mass (only for the last calling)
  if (calc_lostmass) {
    kernel_vlrd_lostmass <<< grid, block >>> (RhoStar->gpu_field,
                                              QRStar->gpu_field,
                                              Vrad->gpu_field,
                                              Qbase->gpu_field,
                                              Vrad->Nsec, 
                                              Vrad->Nrad, 
                                              Vrad->pitch/sizeof(double),
                                              dt*2.0*M_PI/(double)ns,
                                              gpu_lostmass);

    // calculate lost mass and stellar accretion (if VanLeerRadial last calling!)
    checkCudaErrors(cudaMemcpy(lostmass, gpu_lostmass, sizeof(double) * ns, cudaMemcpyDeviceToHost));
    double lostMass = 0.0;
    for (int j=0; j < ns; j++)
      lostMass += lostmass[j];
    LostMass += lostMass;
    StellarAccRate = lostMass/dt;    
  }
  //  (for ...)
  else {
    kernel_vlrd <<< grid, block >>> (RhoStar->gpu_field,
                                     QRStar->gpu_field,
                                     Vrad->gpu_field,
                                     Qbase->gpu_field,
                                     Vrad->Nsec, 
                                     Vrad->Nrad, 
                                     Vrad->pitch/sizeof(double),
                                     dt*2.0*M_PI/(double)ns);
  
  }
  
  cudaThreadSynchronize();
  cudaError_t err = cudaGetLastError ();
  if ( cudaSuccess != err) {
    fprintf (stderr, "Cuda error kernel VanLeerRadial failed \t%s\n", cudaGetErrorString (err));
    exit (-1);
  }
}


__global__ void kernel_vlrdds (double *rhos,
                               double *qrs,
                               double *vr,
                               double *qb,
                               int     ns, 
                               int     nr, 
                               int     pitch,
                               double  dtdtheta) {
             
  __shared__ double srhos[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double svr[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double sqrs[(BLOCK_X+2)*(BLOCK_Y+2)];
  //double rinf, rsup, invsurf, fluxp, fluxm;
  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;
  // js & is, l like 'local' (shared memory <=> local patch)
  int js = threadIdx.x + 1;
  int is = threadIdx.y + 1;
  int ids = is*(blockDim.x+2)+js;
  int idg = __mul24(ig, pitch) + jg;

  // We perform a coalesced read of 'rhos' into the shared memory;
  srhos[ids] = rhos[idg];
  // EDGE 4: "TOP EDGE". Be careful not to read anything if in last row...
  if ((is == blockDim.y) && (ig < nr-1))
    srhos[js+(blockDim.y+1)*(blockDim.x+2)] = GET_TAB (rhos, jg, ig+1, pitch);

  // We perform a coalesced read of 'qrs' into the shared memory;
  sqrs[ids] = qrs[idg];
  // EDGE 4: "TOP EDGE". Be careful not to read anything if in last row...
  if ((is == blockDim.y-1) && (ig < nr-2))
    sqrs[js+(blockDim.y+1)*(blockDim.x+2)] = GET_TAB (qrs, jg, ig+2, pitch);

  // We perform a coalesced read of 'vr' into the shared memory;
  svr[ids] = vr[idg];
  // EDGE 4: "TOP EDGE". Be careful not to read anything if in last row...
  if ((is == blockDim.y-2) && (ig < nr-3))
    svr[js+(blockDim.y+1)*(blockDim.x+2)] = GET_TAB (vr, jg, ig+3, pitch);

  __syncthreads ();

//  rinf = CRadiiStuff[(nr+1)*4+ig];
//  invsurf = CRadiiStuff[(nr+1)*7+ig];
//  rsup = CRadiiStuff[(nr+1)*8+ig];
  
  int idsp = ids+blockDim.x+2;

  double fluxm = 0.0;
  double divvm = 0.0; 
  if (ig >= 0) {
    //fluxm = rinf * sqrs[ids] * svr[ids] * srhos[ids];
    fluxm = rinf * sqrs[ids] * svr[ids];
    divvm = rinf * svr[ids];
   }
  double fluxp = 0.0;
  double divvp = 0.0;
  if (ig < nr-1) {
    //fluxp = rsup * sqrs[idsp] * svr[idsp] * srhos[idsp];
    fluxp = rsup * sqrs[idsp] * svr[idsp];
    divvp = rsup * svr[idsp];
  }
  
  //qb[idg] += ((fluxm-fluxp) - srhos[ids]*sqrs[ids]*(divvm-divvp)) * invsurf * dtdtheta;
  qb[idg] += ((fluxm-fluxp) - sqrs[ids]*(divvm-divvp)) * invsurf * dtdtheta;
}


extern "C"
void VanLeerRadialDustSize_gpu_cu (PolarGrid *Vrad, PolarGrid *Qbase, double dt) {
  int nr, ns;
  nr = Vrad->Nrad;
  ns = Vrad->Nsec;
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid ((ns+block.x-1)/block.x, (nr+block.y-1)/block.y);

  static int FirstTime=YES;
  static double *gpu_lostmass;
  static double *lostmass;

  if (FirstTime) {

    checkCudaErrors(cudaMallocHost ((void **) &lostmass, sizeof(double) * ns));
    checkCudaErrors(cudaMalloc ((void **) &gpu_lostmass, sizeof(double) * ns));
    //checkCudaErrors(cudaMemcpy(gpu_lostmass, lostmass, sizeof(double) * ns, cudaMemcpyHostToDevice));

    FirstTime = NO;
  } 

  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)RadiiStuff, (size_t)(14*(nr+1))*sizeof(double),	0, cudaMemcpyHostToDevice));

  kernel_vlrdds <<< grid, block >>> (RhoStar->gpu_field,
                                     QRStar->gpu_field,
                                     Vrad->gpu_field,
                                     Qbase->gpu_field,
                                     Vrad->Nsec, 
                                     Vrad->Nrad, 
                                     Vrad->pitch/sizeof(double),
                                     dt*2.0*M_PI/(double)ns);
    
  cudaThreadSynchronize();
  getLastCudaError ("kernel_vlrds failed");
}


