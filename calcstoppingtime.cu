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

// some constants required for stopping time calcuation
#define CONST_SIGMA_CGS                    5.943719364576997e-7      // M_sun/AU^3 in CGS
#define CONST_AU_CGS                       1.49600e13                // AU in CGS 
#define MASS_UNIT                          1.0
#define CONST_MFP_CGS                      3.34859e-9                //

__device__ double calc_tstop (double dens_g,
                              double energy_g,
                              double s, 
                              double r,
#ifndef EPSTEIN_ONLY                               
                              double dv_abs,
#endif
                              double aspect_ratio,
                              double flaring_index, 
                              double bulk_rho,
                              double adiabatic_index,
                              bool   adiabatic,
                              double mass_unit) {
  

  double H;
  if (adiabatic) {
    H = sqrt((adiabatic_index-1.0)*energy_g/dens_g) * pow(r,1.5);
  }
  else {
    H = aspect_ratio * (flaring_index > 0 ? pow(r, 1.0 + flaring_index): r);
  }

  // 3D gas density
  const double rho_g_CGS = CONST_SIGMA_CGS * rsqrt(2.0 * M_PI) * (mass_unit * dens_g / H);

#ifndef EPSTEIN_ONLY
  // smooth transition between Epstein and Stoikes regimes (mfp_CFG is the mean free path of H2 molecules in cm)
  const double f = s / (s + CONST_MFP_CGS/rho_g_CGS);
  const double t_stop =( 1.0/CONST_AU_CGS) * (bulk_rho / rho_g_CGS) * s * 
                        ( (1-f) / (sqrt (8.0/M_PI) * pow (r, -3./2.) * H)          // Epstein regime
                        +    f  * (((8.0/3.0)/0.44) / dv_abs));                    // Stokes regime
#else
  const double t_stop = (1.0/CONST_AU_CGS) * (bulk_rho / rho_g_CGS) * s / (sqrt (8.0/M_PI) * pow (r, -3./2.) * H);
#endif
  return t_stop;
}

__global__ void kernel_stoppingtime (double *vrad_g,
                                     double *vtheta_g,
                                     double *rho_g,
                                     double *vrad_d,
                                     double *vtheta_d,
                                     double *rho_d,
                                     double *tstop) {

  // jg & ig, g like 'global' (global memory <=> full grid)
  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;
  // js & is, l like 'local' (shared memory <=> local patch)
  int jgp = jg+1;
  int jgm = jg-1;
  int idgjp, idgjm;
  if (jg == 0) jgm = ns-1;
  if (jg == ns-1) jgp = 0;
  idgjp = jgp + ig*pitch;
  idgjm = jgm + ig*pitch;
  idg = ig*pitch + jg;

  // calcuate stopping time for dust particels
  if (const_stokes) {
    tstop[idg] = dust_size * pow(rmed,1.5);
  }
  else {
#ifndef EPSTEIN_ONLY
    const double dvr = vrad_g[idg]-vrad_d[idg];
    const double dvt = vtheta_g[idg]-vtheta_d[idg];
    const double dv_abs = sqrt(dvr * dvr + dvt * dvt); 
#endif
    
    tstop[idg] = calc_tstop(rho_g[idg],
                            NULL,
                            dust_size,
                            rmed,
#ifndef EPSTEIN_ONLY
                            dv_abs,
#endif
                            aspect_ratio,
                            flaring_index,
                            dust_bulk_dens,
                            0,
                            false,
                            mass_unit);
  } 
}

void CalcStoppingTime () {
  int nr, ns;
  nr = Vrad->Nrad;
  ns = Vrad->Nsec;
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid ((ns+block.x-1)/block.x, (nr+block.y-1)/block.y);
  
  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)RadiiStuff, (size_t)(14*(nr+1))*sizeof(double),	0, cudaMemcpyHostToDevice));
  kernel_stoppingtime <<< grid, block >>>  (VradGas->gpu_field,
                                            VthetaGas->gpu_field,
                                            RhoGas->gpu_field,
                                            VradDust->gpu_field,
                                            VthetaDust->gpu_field,
                                            RhoDust->gpu_field,
                                            double *tstop);
   cudaThreadSynchronize();
   getLastCudaError ("kernel_stoppingtime failed");
}
