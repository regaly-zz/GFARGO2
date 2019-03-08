/** \file calcecc.cu: contains a CUDA kernel for calculating disk eccentricities.
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
//#define BLOCK_X DEF_BLOCK_X_CALCECC
#define BLOCK_X 16
// BLOCK_Y : in radius
#define BLOCK_Y 16

#define rmed  CRadiiStuff[ig]

double *gpu_SurfEcc, *gpu_SurfNormEcc, *gpu_MassNormEcc; // disk eccentricity profiles (unnormalized, surface normalized, mass normalized)

__device__ double CRadiiStuff[16384];

__global__ void kernel_calc_ecc (double *SurfEcc,
                                 double *Rho,                  
                                 double *Vrad,
                                 double *Vtheta,
                                 double omega_frame,
                                 int nr,
                                 int pitch,
                                 double dphi,
                                 double bc_x,
                                 double bc_y,
                                 double bc_r,
                                 double *DiskEcc,
                                 double *SurfNormEcc,
                                 double *MassNormEcc) {

  // jg & ig, g like 'global' (global memory <=> full grid)
  // Below, we recompute x and y for each zone using cos/sin.
  // This method turns out to be faster, on high-end platforms,
  // than a coalesced read of tabulated values.
  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;
  int idg = __mul24(ig, pitch) + jg;

  // create the Cartesian grid from cylindrical
  const double phi= (double)jg*dphi;
  const double x = (rmed) * cos(phi) - bc_x;
  const double y = (rmed) * sin(phi) - bc_y;

  // calculate the velocities in Cartesian system
  const double vx = Vrad[idg]*cos(phi)-(Vtheta[idg]+rsqrt(rmed-bc_r)+0*rmed*omega_frame)*sin(phi);
  const double vy = Vrad[idg]*sin(phi)+(Vtheta[idg]+rsqrt(rmed-bc_r)+0*rmed*omega_frame)*cos(phi);
    
  // calculate the eccentricity    
  const double h = (1/2.)*(vx*vx+vy*vy)-1.0/rmed;
  const double c = x*vy-y*vx;

  const double ecc = sqrt(1.0+2.0*h*c*c);
  DiskEcc[idg]     = ecc;
  SurfNormEcc[idg] = SurfEcc[ig] * ecc;
  MassNormEcc[idg] = SurfEcc[ig] * Rho[idg] * ecc;
}


extern "C" 
void CalcDiskEcc (PolarGrid *Rho, PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *DiskEcc, PlanetarySystem *sys) {
  static bool First = NO;

  int nr = Rho->Nrad;
  int ns = Rho->Nsec;
     
  // initialization
  if (!First) {
    double *h_SurfEcc;
    checkCudaErrors(cudaMallocHost ((void **) &h_SurfEcc, sizeof(double) * nr));
    checkCudaErrors(cudaMalloc ((void **) &gpu_SurfEcc, sizeof(double) * nr));
    for (int i=0; i< nr; i++) {
      h_SurfEcc[i] = Surf[i];
    }  
    checkCudaErrors(cudaMemcpy(gpu_SurfEcc, h_SurfEcc,  sizeof(double) * nr, cudaMemcpyHostToDevice));

    size_t pitch;  
    checkCudaErrors(cudaMallocPitch ((void **) &gpu_SurfNormEcc, &pitch, sizeof(double) * ns, nr));
    checkCudaErrors(cudaMallocPitch ((void **) &gpu_MassNormEcc, &pitch, sizeof(double) * ns, nr));
  
    printf ("Eccentricity calculator is initialized\n");
    
    First = YES;
  }
    
  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)(RadiiStuff+6*(nr+1)), (size_t)(nr)*sizeof(double)));

  double Rpl = sqrt(sys->x[0]*sys->x[0] + sys->y[0]*sys->y[0]);

  // dsk inner region
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid  ((ns + block.x-1)/block.x, (nr + block.y-1)/block.y);
  kernel_calc_ecc <<< grid, block >>> (gpu_SurfEcc,
                                       Rho->gpu_field,
                                       Vrad->gpu_field,
                                       Vtheta->gpu_field,
                                       OmegaFrame,
                                       (int)NRAD,
                                       Vrad->pitch/sizeof(double),
                                       6.28318530717958647688/(double)ns,
                                       //M_PI*(RMAX*RMAX-RMIN*RMIN),
                                       //DiskMass,
                                       0*GasBC_x,
                                       0*GasBC_y,
                                       0*sqrt(GasBC_x*GasBC_x + GasBC_y*GasBC_y),
                                       DiskEcc->gpu_field,
                                       gpu_SurfNormEcc,
                                       gpu_MassNormEcc);

  cudaThreadSynchronize();
  getLastCudaError ("kernel_calc_ecc failed");

  // Dichotomic search for index of planetary radius
  int imin=0;
  int imax=nr-1;
  while (imax-imin > 1) {
    int imed = (imax+imin)/2;
    if (Rmed[imed] > Rpl)
      imax=imed;
    else
      imin=imed;
  }
  int imino = imin;
    
  // wrap raw pointer with a device_ptr
  thrust::device_ptr<double> d_SurfNormEcc(gpu_SurfNormEcc);
  thrust::device_ptr<double> d_MassNormEcc(gpu_MassNormEcc);
  
  // use thrust to find the summs
  DiskEcc_SurfNorm_Inner = thrust::reduce(d_SurfNormEcc, d_SurfNormEcc + imino*ns, (double) 0, thrust::plus<double>());
  DiskEcc_MassNorm_Inner = thrust::reduce(d_MassNormEcc, d_MassNormEcc + imino*ns, (double) 0, thrust::plus<double>());
  DiskEcc_SurfNorm_Outer = thrust::reduce(d_SurfNormEcc+ (imino)*ns, d_SurfNormEcc + nr*ns, (double) 0, thrust::plus<double>());
  DiskEcc_MassNorm_Outer = thrust::reduce(d_MassNormEcc+ (imino)*ns, d_MassNormEcc + nr*ns, (double) 0, thrust::plus<double>());  

  double SurfInner = M_PI*(Rmed[imino]*Rmed[imino]-RMIN*RMIN);
  double SurfOuter = M_PI*(RMAX*RMAX-Rmed[imino]*Rmed[imino]);
  DiskEcc_SurfNorm_Inner /= SurfInner;
  DiskEcc_SurfNorm_Outer /= SurfOuter;
  DiskEcc_MassNorm_Inner /= GasDiskMassInner;
  DiskEcc_MassNorm_Outer /= GasDiskMassOuter;
}