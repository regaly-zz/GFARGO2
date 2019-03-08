/** \file Substep1.cu : implements the kernel for the substep0 procedure
*/
#define __CUDA 1
#include "fargo.h"
#undef __CUDA
#include <stdarg.h>
#include <helper_cuda.h>
#include <cuda.h>

// BLOCK_X : in azimuth
//#define BLOCK_X DEF_BLOCK_X_SUBSTEP1
#define BLOCK_X 8
// BLOCK_Y : in radius
#define BLOCK_Y 8

#define invdr        CRadiiStuff[            ig]
#define cs2          CRadiiStuff[(nr+1)    + ig]
#define cs2m         CRadiiStuff[(nr+1)    + ig - 1]
#define rinf         CRadiiStuff[(nr+1)*4  + ig]
#define invrmed      CRadiiStuff[(nr+1)*2  + ig]
#define invrinf      CRadiiStuff[(nr+1)*3  + ig]
#define rmed         CRadiiStuff[(nr+1)*6  + ig]
#define rsup         CRadiiStuff[(nr+1)*8  + ig]
#define invdiffrsup  CRadiiStuff[(nr+1)*10 + ig]
#define visco        CRadiiStuff[(nr+1)*12 + ig]
#define alphaval     CRadiiStuff[(nr+1)*13 + ig]
#define omega        CRadiiStuff[(nr+1)*14 + ig]

//__constant__ double CRadiiStuff[8192];
__device__ double CRadiiStuff[32768];


#define GET_TAB(u,x,y,pitch) *(u + __mul24(y, pitch) + x)

__device__ double dust_pwl (double p,
                            double a0,
                            double a1) {

  const double pm4 = (double)4.0 - p;
  return (pow (a1, pm4) - pow (a0, pm4))/pm4;
}


__global__ void kernel_calc_dustgrowth (      double *vrad_dgr,
                                              double *vtheta_dgr,
                                        const double *vrad_g,
                                        const double *vtheta_g,
                                              double *rho_dgr,
                                              double *rho_ds,
                                        const double *rho_g,
                                        const double *energy,
                                              double *dust_size,
                                              double *dust_growth_rate,
                                        const double  dust_size_min,
                                        const double  dust_size_star,
                                        const bool    adaptive_alpha,
                                        const double  sigma_thresh,
                                        const double  alpha_smooth,
                                        const double  alpha_active,
                                        const double  alpha_dead,
                                        const bool    adiabatic,
                                        const double  adiabatic_index,
                                        const double  aspect_ratio,
                                        const double  flaring_index,
                                        const double  dust_bulk_rho,
                                        const double  dust_vfrag2,
                                        const double  mass_unit,
                                        const double  dt,
                                        const int     ns, 
                                        const int     nr, 
                                        const int     pitch,
                                        const double  invdphi) {
                                         
  // indices for radial: ig; azimthal: jg; cell: idg
  const int jg = threadIdx.x + blockIdx.x * blockDim.x;
  const int ig = threadIdx.y + blockIdx.y * blockDim.y;
  const int idg = ig*pitch + jg;

  const double sigma = rho_g[idg];
  const double dust_size_old = dust_size[idg];
  
  // get alpha value
  double myalpha;
  if (adaptive_alpha)
    myalpha = (1.0-tanh ((sigma-sigma_thresh) / (sigma_thresh * alpha_smooth * aspect_ratio))) * alpha_active * 0.5 + alpha_dead;
  else
    myalpha = alphaval;
  
  // get sound speed and pressure scale height
  double mycs2, H;
  if (adiabatic) {
    mycs2 = (adiabatic_index-1.0) * energy[idg] / sigma;
//    H = sqrt(mycs2) * pow(rmed, 1.5);
    H = sqrt(mycs2) / omega;
  }
  else {
    mycs2 = cs2;
    H = aspect_ratio * (flaring_index > 0 ? pow(rmed, 1.0 + flaring_index): rmed);
  }

  // 3D gas density
  //const double rho_g_CGS = CONST_RHO_CGS * rsqrt(2.0 * M_PI) * (mass_unit * sigma / H);
  const double rho_g_CGS = CONST_RHO_CGS * CONST_RSQRT2PI * (mass_unit * sigma / H);
  
#ifndef EPSTEIN_ONLY
  const double dvr = vrad_dgr[idg]-vrad_g[idg];
  const double dvt = vtheta_dgr[idg]-vtheta_g[idg];
  const double dv_abs = sqrt(dvr*dvr+dvt*dvt); 
    
  // smooth transition between Epstein and Stoikes regimes (mfp_CFG is the mean free path of H2 molecules in cm)
  const double f = (dust_size[idg] * 0.5) / ((dust_size[idg] * 0.5) + CONST_MFP_CGS/rho_g_CGS);
  const double tstop =( 1.0/CONST_AU_CGS) * (dust_bulk_rho / rho_g_CGS) * dust_size[idg] * 0.5 * 
                      ( (1-f) / (sqrt (8.0/M_PI) * pow (rmed, -3./2.) * H)          // Epstein regime
                      +    f  * (((8.0/3.0)/0.44) / dv_abs));                       // Stokes regime
#else
  //const double tstop = (dust_bulk_rho / rho_g_CGS) * (0.5 * dust_size[idg] / CONST_AU_CGS) / (sqrt (8.0/M_PI) * pow (rmed, -3./2.) * H);
  //const double tstop =  (dust_bulk_rho / rho_g_CGS) * 0.5 * dust_size[idg] / (CONST_AU_CGS * CONST_SQRT8OPI * pow (rmed, -3./2.) * H);
  const double tstop =  (dust_bulk_rho / rho_g_CGS) * dust_size_old / (CONST_AU_CGS * CONST_SQRT8OPI * omega * H);
#endif
  //const double St = tstop * pow(rmed, -1.5);
  const double St = tstop * omega;

  if (St>1)
    return;

//  const double Sc = 1.0 + St;  

  //const double vrel = sqrt(mycs2 * ((8.0/M_PI) + 3.0 * alphaval * St));
//  const double vrel = sqrt(mycs2 * ( CONST_SQRT8OPI + 3.0 * alphaval * St));
  
  
  //const double vturb2 = 3.0 * alphaval * St;
  const double vturb2 = 3.0 * myalpha / (St + 1.0/St); // Birnstiel et al. (2016)
  
 // const double m      = (4./3.) * pow(dust_size_old/2.0,3.0) * M_PI * dust_bulk_rho;
//  const double vth2   = (16.0/M_PI) * CONST_KB_CGS / m * CONST_VEL2_CGS / 1e10;
 // const double vrel   = sqrt(mycs2 * (0*vth2 + vturb2));
   const double vrel   = sqrt(mycs2 * vturb2);
  //const double afrag = ((2.0 * M_PI / 3.0) * sigma * dust_vfrag2) / ((dust_bulk_rho / CONST_RHO_CGS) * alphaval * mycs2);
  const double afrag = CONST_AU_CGS * (CONST_2O3PI * sigma * dust_vfrag2) / ((dust_bulk_rho / CONST_RHO_CGS) * myalpha * mycs2);


//  const double Hd = H * sqrt(alphaval * rsqrt(2.0) * (Sc + sqrt(Sc * Sc + 8.0 * St * (St + sqrt(2.0) * alphaval))) / (Sc * (alphaval * sqrt(2.0) + St)));
  // dust scale-height (Kornet et al. 2001)
  //const double Hd = H * sqrt(myalpha * CONST_RSQRT2 * (Sc + sqrt(Sc * Sc + 8.0 * St * (St + CONST_SQRT2 * myalpha))) / (Sc * (myalpha * CONST_SQRT2 + St)));

  // dust scale-height (Birnstiel et al. 2010)
  //const double Hd = H * min (1.0, sqrt(myalpha / (min (St, 0.5) * (1.0 + St * St))));

  // dust scale-height (Birnstiel et al. 2012)
  const double Hd = H * sqrt(myalpha / St);
    
  // growth
  //const double D = 1e4*CONST_VEL_CGS * (rho_dgr[idg] + rho_ds[idg]) / (sqrt(2.0*M_PI) * Hd) * vrel / (dust_bulk_rho / CONST_RHO_CGS);
  //const double D = CONST_AU_CGS * sqrt(3./(2.*M_PI))*(1.0/(dust_bulk_rho / CONST_RHO_CGS))*(rho_dgr[idg] + rho_ds[idg]) *sqrt(mycs2)*St/H;
  const double D = CONST_AU_CGS * vrel * ((rho_dgr[idg] + rho_ds[idg]) / (dust_bulk_rho / CONST_RHO_CGS) ) / (CONST_SQRT2PI * Hd);
//  if (St>1)
//    printf ("r:%e H:%e v:%e D:%e S:%e\n", rmed, Hd, vrel, D, St);
  
  double dust_size_new = dust_size_old + D * dt;

  // limit the growth by fragmentation barrier
  if (dust_size_new > afrag)
    dust_size_new = afrag;

  dust_size[idg] =  dust_size_new;
      
  // growth rate
  const double tmp = dust_pwl((double) DUST_SIZE_DISTR, dust_size_min, dust_size_star);  
  double Sdt = (rho_dgr[idg] + rho_ds[idg]) *
                     (dust_pwl(DUST_SIZE_DISTR, dust_size_old, dust_size_new) * tmp) /
                     (dust_pwl(DUST_SIZE_DISTR, dust_size_min, dust_size_old) * dust_pwl(DUST_SIZE_DISTR, dust_size_min, dust_size_new));

  if (rho_ds[idg] - Sdt > 1e-10)
    rho_ds[idg] -= Sdt;
  
  rho_dgr[idg] += Sdt;

  if (Sdt > 0) {
    vrad_dgr[idg]   += Sdt * vrad_g[idg]; 
    vtheta_dgr[idg] += Sdt * vtheta_g[idg];
  }
  else {
    vrad_dgr[idg]   += Sdt * vrad_dgr[idg]; 
    vtheta_dgr[idg] += Sdt * vtheta_dgr[idg];    
  }
}


void SubStep4_gpu (PolarGrid *RhoDustGr, PolarGrid *RhoDustSm, PolarGrid *DustSizeGr, PolarGrid *DustGrowthRate, PolarGrid *RhoGas, PolarGrid *Energy, double dt) {

  int nr, ns;
  nr = DustSizeGr->Nrad;
  ns = DustSizeGr->Nsec;
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid ((ns+block.x-1)/block.x, (nr+block.y-1)/block.y);

  double *Energy_gpu_field = NULL;
  if (Adiabatic) {
    Energy_gpu_field = Energy->gpu_field;
  }
  
  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)RadiiStuff, (size_t)(15*(nr+1))*sizeof(double),	0, cudaMemcpyHostToDevice));
  kernel_calc_dustgrowth <<< grid, block >>> (dust_v_rad[0]->gpu_field,
                                              dust_v_theta[0]->gpu_field,
                                              gas_v_rad->gpu_field,
                                              gas_v_theta->gpu_field,
                                              RhoDustGr->gpu_field,
                                              RhoDustSm->gpu_field,
                                              RhoGas->gpu_field,
                                              Energy_gpu_field,
                                              DustSizeGr->gpu_field,
                                              DustGrowthRate->gpu_field,
                                              DUST_MIN_SIZE * CONST_UM_CGS,
                                              DUST_STAR_SIZE * CONST_UM_CGS,
                                              AdaptiveViscosity,
                                              ALPHASIGMATHRESH,
                                              ALPHASMOOTH,
                                              ALPHAVISCOSITY,
                                              ALPHAVISCOSITYDEAD,
                                              Adiabatic,
                                              ADIABATICINDEX,
                                              ASPECTRATIO,
                                              FLARINGINDEX,
                                              DUSTBULKDENS,
                                              DUSTVFRAG*DUSTVFRAG / CONST_VEL2_CGS,
                                              MASSUNIT,
                                              dt,
                                              ns,
                                              nr,
                                              DustSizeGr->pitch/sizeof(double),
                                              (double)ns/2.0/M_PI);

  cudaThreadSynchronize();
  getLastCudaError ("kernel_calc_dutgrowth failed");
}



