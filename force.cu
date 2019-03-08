/** \file "force.cu" : implements the kernel for the tidal force calculation
*/
#define __CUDA 1
#include "fargo.h"
#undef __CUDA
#include <stdarg.h>
#include <helper_cuda.h>
#include <cuda.h>

// BLOCK_X : in azimuth
//#define BLOCK_X DEF_BLOCK_X_FORCE
#define BLOCK_X 16
// BLOCK_Y : in radius
#define BLOCK_Y 8

#define GET_TAB(u,x,y,pitch) *(u + __mul24(y, pitch) + x)

// improve legibility
#define rmed CRadiiStuff[(nr+1)*6+ig]
#define surf CRadiiStuff[(nr+1)*9+ig]

//static double ForceX[MAX1D];
//static double ForceY[MAX1D];

//__constant__ double CRadiiStuff[8192];
__device__ double CRadiiStuff[32768];

__global__ void kernel_force (double *Rho,
//                              double *Rho2,
                              double *fx,
                              double *fy,
                              double eps2, 
                              double xp, double yp,
                              int ns, int nr,
                              int pitch, double dphi) {

  // jg & ig, g like 'global' (global memory <=> full grid)
  // Below, we recompute x and y for each zone using cos/sin.
  // This method turns out to be faster, on high-end platforms,
  // than a coalesced read of tabulated values.
  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;
  int idg = __mul24(ig, pitch) + jg;
  double phi= (double)jg*dphi;
  double dx = rmed * cos(phi) - xp;
  double dy = rmed * sin(phi) - yp;
//  double cellmass = surf * (Rho[idg]+Rho2[idg]);
  double cellmass = surf * Rho[idg];
  double dist2 = dx*dx+dy*dy;

  dist2 += eps2;
  double invd3 = 1.0/dist2 * rsqrt(dist2);
  fx[idg] = cellmass*dx*invd3;
  fy[idg] = cellmass*dy*invd3;
}

// Gaussian function (original)
//--------------------------------------------------------------
__global__ void kernel_force_Gauss_cutoff (double *Rho,
                                           double *fx,
                                           double *fy,
                                           double eps2, 
                                           double invrh2,
                                           double xp, double yp,
                                           int ns, int nr,
                                           int pitch, double dphi) {

  // jg & ig, g like 'global' (global memory <=> full grid)
  // Below, we recompute x and y for each zone using cos/sin.
  // This method turns out to be faster, on high-end platforms,
  // than a coalesced read of tabulated values.
  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;
  int idg = __mul24(ig, pitch) + jg;
  double phi= (double)jg*dphi;
  double dx = rmed * cos(phi) - xp;
  double dy = rmed * sin(phi) - yp;
  double cellmass = surf * Rho[idg];
  double dist2 = dx*dx+dy*dy;

  // Gaussian toruq cut-off (original)
  cellmass *= 1.0-exp(-dist2*invrh2);

  dist2 += eps2;
  double invd3 = 1.0/dist2 * rsqrt(dist2);
  fx[idg] = cellmass*dx*invd3;
  fy[idg] = cellmass*dy*invd3;
}
//--------------------------------------------------------------


// Heviside function (Crida et al. 2009)
//---------------------------------------------------------------
__global__ void kernel_force_Heaviside_cutoff (double *Rho,
                                               double *fx,
                                               double *fy,
                                               double eps2, 
                                               double invrh2,
                                               double heaviside_b,
                                               double xp, double yp,
                                               int ns, int nr,
                                               int pitch, double dphi) {

  // jg & ig, g like 'global' (global memory <=> full grid)
  // Below, we recompute x and y for each zone using cos/sin.
  // This method turns out to be faster, on high-end platforms,
  // than a coalesced read of tabulated values.
  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;
  int idg = __mul24(ig, pitch) + jg;
  double phi= (double)jg*dphi;
  double dx = rmed * cos(phi) - xp;
  double dy = rmed * sin(phi) - yp;
  double cellmass = surf * Rho[idg];
  double dist2 = dx*dx+dy*dy;

  // Heaviside torque cut-off function (Crida et al. 2009)
  cellmass *= 1.0/(exp(-10.0*(sqrt(dist2*invrh2)/heaviside_b-1.0))+1.0);
  
  dist2 += eps2;
  double invd3 = 1.0/dist2 * rsqrt(dist2);
  fx[idg] = cellmass*dx*invd3;
  fy[idg] = cellmass*dy*invd3;
}
//---------------------------------------------------------------

extern "C"
Force ComputeForce_gpu (PolarGrid *Rho, double x0, double y0, double smoothing, double mass, int exclude) {
  int nr, ns;
  Force result;
  double fxi = 0.0, fxo = 0.0, fyi = 0.0, fyo = 0.0;
  double a;

  nr = Rho->Nrad;
  ns = Rho->Nsec;

  // planetary Hill radius
  a = sqrt(x0*x0+y0*y0);
  double rh = a*pow(mass/3.0, 1.0/3.0);
  
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid ((ns+block.x-1)/block.x, (nr+block.y-1)/block.y);

  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)RadiiStuff, (size_t)(14*(nr+1))*sizeof(double), 0, cudaMemcpyHostToDevice));  

  // no torque cut-off
  if (exclude == 0) {
    kernel_force <<< grid, block >>> (Rho->gpu_field,
//                                      dust_density[0]->gpu_field,
                                      Work->gpu_field,
                                      TemperInt->gpu_field,
                                      smoothing*smoothing,
                                      x0, 
                                      y0,
                                      ns, 
                                      nr,
                                      Rho->pitch/sizeof(double),
                                      2.0*M_PI/(double)ns);
  }
  // Gaussian torqu cut-off
  else if (exclude == 1) {
    kernel_force_Gauss_cutoff <<< grid, block >>> (Rho->gpu_field,
                                                   Work->gpu_field,
                                                   TemperInt->gpu_field,
                                                   smoothing*smoothing,
                                                   1.0/(rh*rh),
                                                   x0, 
                                                   y0,
                                                   ns, 
                                                   nr,
                                                   Rho->pitch/sizeof(double),
                                                   2.0*M_PI/(double)ns);
  }
  // Heaviside torque cut-off
  else if (exclude == 2) {
    kernel_force_Heaviside_cutoff <<< grid, block >>> (Rho->gpu_field,
                                                       Work->gpu_field,
                                                       TemperInt->gpu_field,
                                                       smoothing*smoothing,
                                                       1.0/(rh*rh),
                                                       HEAVISIDEB,
                                                       x0, 
                                                       y0,
                                                       ns, 
                                                       nr,
                                                       Rho->pitch/sizeof(double),
                                                       2.0*M_PI/(double)ns);
  }
  
  cudaThreadSynchronize();
  getLastCudaError ("ComputeForce_gpu: kernel failed");

  AzimuthalAverage (Work,      ForceX);
  AzimuthalAverage (TemperInt, ForceY);

  getLastCudaError ("grabuge dans les azimuthal average");

  for (int i = 0; i < nr; i++) {
    if (Rmed[i] < a) {
      fxi += G*ForceX[i];
      fyi += G*ForceY[i];
    } else {
      fxo += G*ForceX[i];
      fyo += G*ForceY[i];
    }
  }
  result.fx_inner = fxi;
  result.fy_inner = fyi;
  result.fx_outer = fxo;
  result.fy_outer = fyo;
  result.fx_ex_inner = fxi;
  result.fy_ex_inner = fyi;
  result.fx_ex_outer = fxo;
  result.fy_ex_outer = fyo;
  return result;
}
