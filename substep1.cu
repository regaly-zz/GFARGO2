/** \file Substep1.cu : implements the kernel for the substep1 procedure
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

//__constant__ double CRadiiStuff[8192];
__device__ double CRadiiStuff[32768];

#define invdr    CRadiiStuff[            ig]
#define cs2      CRadiiStuff[(nr+1)    + ig]
#define cs2m     CRadiiStuff[(nr+1)    + ig - 1]
#define rinf     CRadiiStuff[(nr+1)*4  + ig]
#define invrmed  CRadiiStuff[(nr+1)*2  + ig]
#define invrinf  CRadiiStuff[(nr+1)*3  + ig]
#define rmed     CRadiiStuff[(nr+1)*6  + ig]
#define visco    CRadiiStuff[(nr+1)*12 + ig]
#define omega    CRadiiStuff[(nr+1)*14 + ig]



#define GET_TAB(u,x,y,pitch) *(u + __mul24(y, pitch) + x)


// substep1 for gas only
//----------------------------------------------------------------------------------------------------------------------
__global__ void kernel_substep1a (const double *vrad,
                                  const double *vtheta,
                                  const double *pot,
                                  const double *rho,
                                  const double *energy,
                                        double *vradint,
                                        double *vthetaint,
                                  const bool    adiabatic,
                                  const double  adiabatic_index,
                                  const int     ns, 
                                  const int     nr, 
                                  const int     pitch,
                                  const double  dt, 
                                  const double  invdphi) {

  // indices for radial: ig; azimthal: jg; cell: idg
  const int jg = threadIdx.x + blockIdx.x * blockDim.x;
  const int ig = threadIdx.y + blockIdx.y * blockDim.y;
  const int idg = ig * pitch + jg;
  
  int jgp = jg+1;
  int jgm = jg-1;
  int idgjp, idgjm;
  if (jg == 0) 
    jgm = ns-1;
  if (jg == ns-1) 
    jgp = 0;
  idgjp = jgp + ig * pitch;
  idgjm = jgm + ig * pitch;
  
  double vrtemp = 0.0;
  double gradp, vt, vk;
  const double rho_idg    = rho[idg];
  double rho_idgim;
  const double rho_idgjm  = rho[idgjm];
  const double vrad_idg   = vrad[idg];
  const double vtheta_idg = vtheta[idg];

  if (ig > 0) {
    rho_idgim  = rho[idg-pitch];
    if (adiabatic)
      gradp = 2.0 * (adiabatic_index-1.0)*(energy[idg] - energy[idg-pitch]) / (rho_idg + rho_idgim); 
    else
      gradp = 2.0 * ((cs2 * rho_idg - cs2m * rho_idgim) / (rho_idg + rho_idgim));
    gradp += pot[idg] - pot[idg-pitch];
    
    vk = sqrt(invrinf);
    vt = vtheta_idg + vtheta[idgjp] + vtheta[idg-pitch] + vtheta[idgjp-pitch];
    vt = vt*0.25;
    
    vrtemp = vrad_idg + dt * (-gradp * invdr + vt * invrinf * (vt + 2.0 * vk));
  }
  vradint[idg] = vrtemp;

  if (adiabatic)
    gradp = 2.0 * (adiabatic_index - 1.0) * (energy[idg] - energy[idgjm]) / (rho_idg + rho_idgjm); 
  else
    gradp = cs2*2.0*((rho_idg-rho_idgjm) / (rho_idg+rho_idgjm));
  gradp += pot[idg]-pot[idgjm];
  vthetaint[idg] = vtheta_idg - dt*gradp*invdphi*invrmed;
}

void SubStep1_gpu (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, PolarGrid *Energy, double dt,
                   PolarGrid *Vrad_ret, PolarGrid *Vtheta_ret) {
  int nr, ns;
  nr = Vrad->Nrad;
  ns = Vrad->Nsec;
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid ((ns+block.x-1)/block.x, (nr+block.y-1)/block.y);
  
  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)RadiiStuff, (size_t)(14*(nr+1))*sizeof(double),	0, cudaMemcpyHostToDevice));
  double *Energy_gpu_field = NULL;
  if (Adiabatic) {
    Energy_gpu_field = Energy->gpu_field;
  }
  kernel_substep1a <<< grid, block >>> (Vrad->gpu_field,
                                        Vtheta->gpu_field,
                                        Potential->gpu_field,
                                        Rho->gpu_field,
                                        Energy_gpu_field,
                                        Vrad_ret->gpu_field,
                                        Vtheta_ret->gpu_field,
                                        Adiabatic, 
                                        ADIABATICINDEX,
                                        ns,
                                        nr,
                                        Vrad->pitch/sizeof(double), 
                                        dt,
                                        (double)ns/2.0/M_PI);
  cudaThreadSynchronize();
  getLastCudaError ("kernel_substep1a failed");
}
//----------------------------------------------------------------------------------------------------------------------




// calculate stopping time for dust
//----------------------------------------------------------------------------------------------------------------------
__device__ double calc_timestop (const double  sigma,
                                 const double  energy,
                                 const double  s,
                                 const double  r,
                                 const double  Omega,
#ifndef EPSTEIN_ONLY
                                 const double  dv_abs,
#endif
                                 const double  aspect_ratio,
                                 const double  flaring_index, 
                                 const double  bulk_rho,
                                 const double  adiabatic_index,
                                 const bool    adiabatic,
                                 const double  mass_unit) {
  

  double H;  
  if (adiabatic) {
    //H = sqrt((adiabatic_index-1.0)*energy/sigma) * pow(r,1.5);
    H = sqrt((adiabatic_index-1.0)*energy/sigma) / Omega;
  }
  else {
    H = aspect_ratio * (flaring_index > 0 ? pow(r, 1.0 + flaring_index): r);
  }

  // 3D gas density
  //const double rho_g_CGS = CONST_RHO_CGS * rsqrt(2.0 * M_PI) * (mass_unit * sigma_g / H);
  const double rho_g_CGS = CONST_RHO_CGS * CONST_RSQRT2PI * (mass_unit * sigma / H);
    

#ifndef EPSTEIN_ONLY
  // smooth transition between Epstein and Stoikes regimes (mfp_CFG is the mean free path of H2 molecules in cm)
  const double f = s / (s + CONST_MFP_CGS/rho_g_CGS);
  const double t_stop = (bulk_rho / rho_g_CGS) * (s / CONST_AU_CGS) * 
//                        ( (1-f) / (sqrt (8.0/M_PI) * pow (r, -3./2.) * H)          // Epstein regime
//                        ( (1-f) / (CONST_SQRT8OPI * pow (r, -3./2.) * H)          // Epstein regime    
                          ( (1-f) / (CONST_SQRT8OPI * Omega * H)          // Epstein regime    
//                        +    f  * (((8.0/3.0)/0.44) / dv_abs));                    // Stokes regime
                        +    f  * (CONST_8O3o044 / dv_abs));                    // Stokes regime

#else
  //const double t_stop =  (bulk_rho / rho_g_CGS) * (s/CONST_AU_CGS) / (sqrt (8.0/M_PI) * pow (r, -3./2.) * H);
  //const double t_stop =  (bulk_rho / rho_g_CGS) * s / (CONST_AU_CGS * CONST_SQRT8OPI * pow (r, -3./2.) * H);
  const double t_stop =  (bulk_rho / rho_g_CGS) * s / (CONST_AU_CGS * CONST_SQRT8OPI * Omega * H);
#endif
  return t_stop;
}
//----------------------------------------------------------------------------------------------------------------------


// substep1 for dust component (no growth, no bacckreaction)
//----------------------------------------------------------------------------------------------------------------------
__global__ void kernel_substep1b (const double *vrad_g,
                                  const double *vtheta_g,
                                  const double *rho_g,
                                  const double *vrad_d,
                                  const double *vtheta_d,
                                  const double *rho_d,
                                  const double *pot,
                                  const double *energy,
                                        double *vrad_d_int,
                                        double *vtheta_d_int,
                                        double *rho_d_int,
                                  const double *viscosity,
                                  const bool    adiabatic,
                                  const double  adiabatic_index,
                                  const double  aspect_ratio,
                                  const double  flaring_index,
                                  const double  dust_size,
                                  const double  dust_bulk_dens,
                                  const bool    const_stokes,
                                  const double  mass_unit,
                                  const int ns, 
                                  const int nr, 
                                  const int pitch,
                                  const double dt, 
                                  const double invdphi
                                  //double *mywork
                                  ) {

  // indices for radial: ig; azimthal: jg; cell: idg
  const int jg = threadIdx.x + blockIdx.x * blockDim.x;
  const int ig = threadIdx.y + blockIdx.y * blockDim.y;
  const int idg = ig * pitch + jg;
  
  // idgjp & idgjm are azimuthal +1/-1 cell indices
  int jgp = jg+1;
  int jgm = jg-1;
  int idgjp, idgjm;
  if (jg == 0) 
    jgm = ns-1;
  if (jg == ns-1) 
    jgp = 0;
  idgjp = jgp + ig * pitch;
  idgjm = jgm + ig * pitch;

  // calcuate stopping time for dust particels
  double tstop;

  // based on predefined Stokes number
  if (const_stokes) {
    //tstop = dust_size * pow(rmed,1.5);
    tstop = dust_size / omega;
  }
  // based on particle size
  else {

#ifndef EPSTEIN_ONLY
    const double dvr = vrad_d[idg]-vrad_g[idg];
    const double dvt = vtheta_d[idg]-vtheta_g[idg];
    const double dv_abs = sqrt(dvr*dvr+dvt*dvt); 
#endif
    
    tstop = calc_timestop(rho_g[idg],
                          (adiabatic ? energy[idg]:0.0),
                          dust_size,
                          rmed,
                          omega,
#ifndef EPSTEIN_ONLY
                          dv_abs,
#endif
                          aspect_ratio,
                          flaring_index,
                          dust_bulk_dens,
                          adiabatic_index,
                          adiabatic,
                          mass_unit);
  }
    
  // dust feedback requires dust-to-gas massratio
  const double rho_g_idg    = rho_g[idg];
  const double rho_g_idgjm  = rho_g[idgjm];
  const double rho_d_idg    = rho_d[idg]; 
  const double rho_d_idgjm  = rho_d[idgjm];
  const double vrad_g_idg   = vrad_g[idg];
  const double vtheta_g_idg = vtheta_g[idg];
  const double vrad_d_idg   = vrad_d[idg];
  const double vtheta_d_idg = vtheta_d[idg];
  
  double gradpg, gradpd , gradpot, acc_g, acc_d, vt_g, vt_d, vk;
  double vrad_d_temp = 0.0;
  double vtheta_d_temp = 0.0;
  double X, Y;
 
  // Stokes number
  const double St = tstop * omega;
  
  // fully implicit method for small Stokes number
  if (St<10000) {
    // radial velocities
    if (ig > 0) {
      const double rho_g_idgim  = rho_g[idg-pitch];
      //const double rho_d_idgim  = rho_d[idg-pitch];
      if (adiabatic) {
        gradpg = 2.0 * (adiabatic_index-1.0)*(energy[idg] - energy[idg-pitch]) / (rho_g_idg + rho_g_idgim); 
      }
      else {
        gradpg = 2.0 * ((cs2 * rho_g_idg - cs2m * rho_g_idgim) / (rho_g_idg + rho_g_idgim));
      }
      gradpd = 0;//*C_DP* 2.0 * ((cs2 * rho_d_idg - cs2m * rho_d_idgim) / (rho_d_idg + rho_d_idgim));
      gradpot = pot[idg] - pot[idg-pitch];
      vk = sqrt(invrinf);
      vt_g = vtheta_g_idg + vtheta_g[idgjp] + vtheta_g[idg-pitch] + vtheta_g[idgjp-pitch];
      vt_g = vt_g*0.25;
      vt_d = vtheta_d_idg + vtheta_d[idgjp] + vtheta_d[idg-pitch] + vtheta_d[idgjp-pitch];
      vt_d = vt_d*0.25;
      acc_g = -(gradpg + gradpot) * invdr;
      acc_d = -(gradpd + gradpot) * invdr;
    
      X = (vrad_g_idg - vrad_d_idg + dt * (acc_g - acc_d)) / (1.0 + dt / tstop);
      Y = vrad_g_idg + dt * acc_g;
      
      //vrad_g_temp = Y + dt * vt_g * invrinf * (vt_g + 2.0 * vk);
      vrad_d_temp = Y - X + dt * vt_d * invrinf * (vt_d + 2.0 * vk);
    }
    
    //azimuthal velocities
    if (adiabatic) {
      gradpg = 2.0 * (adiabatic_index - 1.0) * (energy[idg] - energy[idgjm]) / (rho_g[idg] + rho_g[idgjm]); 
    }
    else
      gradpg = 2.0 * cs2 * ((rho_g_idg-rho_g_idgjm) / (rho_g_idg + rho_g_idgjm));
    gradpd = 0;//*C_DP * 2.0 * cs2 * ((rho_d_idg-rho_d_idgjm) / (rho_d_idg + rho_d_idgjm));
    gradpot = pot[idg]-pot[idgjm];
    acc_g = -(gradpg + gradpot) * invdphi * invrmed;
    acc_d = -(gradpd + gradpot) * invdphi * invrmed;
    
    X = (vtheta_g_idg - vtheta_d_idg + dt * (acc_g - acc_d)) / (1.0 + dt / tstop);
    Y = vtheta_g_idg + dt * acc_g;
    
    //vtheta_g_temp = Y;
    vtheta_d_temp = Y - X;  
  }
  // fully explicit for high Stokes numbers
  else {
    double fdrag;
    if (ig > 0) {
    
	    const double rho_idgim  = rho_g[idg-pitch];
	    gradpd = pot[idg] - pot[idg-pitch];
    	fdrag = rho_d_idg*(vrad_g_idg-vrad_d_idg)/tstop;
	    vk = sqrt(invrinf);
	    vt_d = vtheta_d_idg + vtheta_d[idgjp] + vtheta_d[idg-pitch] + vtheta_d[idgjp-pitch];
	    vt_d = vt_d*0.25;
    	
	    vrad_d_temp = vrad_d_idg + dt * (-gradpd * invdr + fdrag + vt_d * invrinf * (vt_d + 2.0 * vk));
	  }

	  gradpd = pot[idg]-pot[idgjm];
	  fdrag = rho_d_idg*(vtheta_g_idg-vtheta_d_idg)/tstop;
	  vtheta_d_temp = vtheta_d_idg - dt*(gradpd*invdphi*invrmed+fdrag);

    /*
    // radial directon
    if (ig > 0) {
      const double rho_g_idgim  = rho_g[idg-pitch];
      const double rho_d_idgim  = rho_d[idg-pitch];
      gradpot = pot[idg] - pot[idg-pitch];
      if (adiabatic)
        gradpg = 2.0 * (adiabatic_index-1.0)*(energy[idg] - energy[idg-pitch]) / (rho_g_idg + rho_g_idgim); 
      else
        gradpg = 2.0 * ((cs2 * rho_g_idg - cs2m * rho_g_idgim) / (rho_g_idg + rho_g_idgim));
      gradpd = 0*C_DP* 2.0 * ((cs2 * rho_d_idg - cs2m * rho_d_idgim) / (rho_d_idg + rho_d_idgim));
      vk = sqrt(invrinf);
      vt_g = vtheta_g_idg+vtheta_g[idgjp] + vtheta_g[idg-pitch]+vtheta_g[idgjp-pitch];
      vt_g = vt_g * 0.25;
      vt_d = vtheta_d_idg + vtheta_d[idgjp] + vtheta_d[idg-pitch] + vtheta_d[idgjp-pitch];
      vt_d = vt_d*0.25;
      //vrad_g_temp = vrad_g[idg] + dt * (-(gradpg + gradpot) * invdr + vt_g * invrinf * (vt_g + 2.0 * vk));
        
      //const double dvrad_drag = - (vrad_d_idg-vrad_g_idg)*dt/(tstop);              // 1'st order: unphysical jump in radial velocity at dt~tstop
      const double dvrad_drag = - (vrad_d[idg]-vrad_g[idg])*dt/(tstop+dt/2.0);         // 2'nd order: improoved Euler scheme, see Eq. (6) of Zhu et al (2012)
      vrad_d_temp = vrad_d_idg + dvrad_drag + dt*(-(gradpd + gradpot) * invdr + vt_d * invrinf * (vt_d + 2.0 * vk));
    }
    
    // azimuthal direction
    gradpot = pot[idg]-pot[idgjm];
    if (adiabatic)
      gradpg = 2.0 * (adiabatic_index - 1.0) * (energy[idg] - energy[idgjm]) / (rho_g_idg + rho_g_idgjm); 
    else
      gradpg = 2.0 * cs2 * ((rho_g_idg-rho_g_idgjm) / (rho_g_idg + rho_g_idgjm));
    gradpd = 0*C_DP * 2.0 * cs2 * ((rho_d_idg-rho_d_idgjm) / (rho_d_idg + rho_d_idgjm));
    //vtheta_g_temp = vtheta_g[idg] - dt * (gradpg + gradpot) * invdphi * invrmed;
    
    //const double dvtheta_drag =  - (vtheta_d_idg- vtheta_g_idg)*dt/(tstop);       // 1'st order: unphysical jump in radial velocity at dt~tstop
    const double dvtheta_drag =  - (vtheta_d[idg]- vtheta_g[idg])*dt/(tstop+dt/2.0);  // 2'nd order: improoved Euler scheme, see Eq. (6) of Zhu et al (2012)
    vtheta_d_temp = vtheta_d_idg + dvtheta_drag - dt * (gradpd + gradpot) * invdphi * invrmed;
    */
  
    // radial directon
	/*
    double dvrad_drag = 0.0, dvtheta_drag;
    double const t_exp = exp(-dt / tstop);
    if (ig > 0) {
      
      const double rho_g_idgim  = rho_g[idg-pitch];
      const double rho_d_idgim  = rho_d[idg-pitch];
      gradpot = pot[idg] - pot[idg-pitch];
      
      // radial direction
      gradpot = pot[idg] - pot[idg-pitch];
      if (adiabatic) {
        gradpg = 2.0 * (adiabatic_index-1.0)*(energy[idg] - energy[idg-pitch]) / (rho_g_idg + rho_g_idgim); 
      }
      else {
        gradpg = 2.0 * ((cs2 * rho_g_idg - cs2m * rho_g_idgim) / (rho_g_idg + rho_g_idgim));
      }
      gradpd = 0*C_DP* 2.0 * ((cs2 * rho_d_idg - cs2m * rho_d_idgim) / (rho_d_idg + rho_d_idgim));
      vk = sqrt(invrinf);
      vt_g = vtheta_g_idg+vtheta_g[idgjp] + vtheta_g[idg-pitch]+vtheta_g[idgjp-pitch];
      vt_g = vt_g * 0.25;
      vt_d = vtheta_d_idg + vtheta_d[idgjp] + vtheta_d[idg-pitch] + vtheta_d[idgjp-pitch];
      vt_d = vt_d*0.25;
      dvrad_drag = vrad_g_idg*(1.0-t_exp) + vrad_d_idg*t_exp;
      vrad_d_temp = dvrad_drag + dt*(- (gradpd + gradpot) *invdr + vt_d * invrinf * (vt_d + 2.0 * vk));
    }
    gradpot = pot[idg]-pot[idgjm];
    if (adiabatic)
      gradpg = 2.0 * (adiabatic_index - 1.0) * (energy[idg] - energy[idgjm]) / (rho_g_idg + rho_g_idgjm); 
    else
      gradpg = 2.0 * cs2 * ((rho_g_idg-rho_g_idgjm) / (rho_g_idg + rho_g_idgjm));
      
    // azimuthal direction
    gradpot = pot[idg]-pot[idgjm];
    gradpd = 0*C_DP * 2.0 * cs2 * ((rho_d[idg]-rho_d[idgjm]) / (rho_d[idg] + rho_d[idgjm]));
    dvtheta_drag = vtheta_g_idg*(1.0-t_exp) + vtheta_d_idg*t_exp;
    vtheta_d_temp = dvtheta_drag - dt * (gradpd + gradpot) * invdphi * invrmed;*/
  }

  // finally set the intermediate velocities
  vrad_d_int[idg]   = vrad_d_temp;
  vtheta_d_int[idg] = vtheta_d_temp;

  // dust diffusion and growth correction
  if (ig > 0 && ig < nr-1) {

    // get appropriate viscosity
    double nu;
    if (viscosity != NULL)
      nu = viscosity[idg];
    else
      nu = visco;
    
    // diffussion coefficient, see Eq. (17) of Birnstiel et al. (2010)
    const double Ddt = 0*dt * nu / (1.0 + St * St);
    rho_d_int[idg] = rho_d_idg + Ddt * ((rho_d[idg+pitch] - 2.0 * rho_d_idg + rho_d[idg-pitch]) * invdr * invdr +
                                        (rho_d[idgjp]     - 2.0 * rho_d_idg + rho_d_idgjm)      * invdphi * invrmed * invdphi * invrmed);
  }
}

void SubStep1Dust_gpu (PolarGrid *VradGas, PolarGrid *VthetaGas, PolarGrid *RhoGas, PolarGrid *Energy, 
                       PolarGrid *VradDust, PolarGrid *VthetaDust, PolarGrid *RhoDust,
                       double dust_size, double dt,
                       PolarGrid *VradDust_ret, PolarGrid *VthetaDust_ret, PolarGrid *RhoDust_ret) {

  int nr, ns;
  nr = VradGas->Nrad;
  ns = VradGas->Nsec;
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid ((ns+block.x-1)/block.x, (nr+block.y-1)/block.y);
  
  double *Energy_gpu_field = NULL;
  if (Adiabatic) {
    Energy_gpu_field = Energy->gpu_field;
  }
  
  double *Viscosity_gpu_field = NULL;
  if (Adiabatic || AdaptiveViscosity)
    Viscosity_gpu_field = Viscosity->gpu_field;

  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)RadiiStuff, (size_t)(15*(nr+1))*sizeof(double),	0, cudaMemcpyHostToDevice));
  kernel_substep1b <<< grid, block >>> (VradGas->gpu_field,
                                        VthetaGas->gpu_field,
                                        RhoGas->gpu_field,
                                        VradDust->gpu_field,
                                        VthetaDust->gpu_field,
                                        RhoDust->gpu_field,
                                        Potential->gpu_field,
                                        Energy_gpu_field,
                                        VradDust_ret->gpu_field,
                                        VthetaDust_ret->gpu_field,
                                        Work->gpu_field,
                                        Viscosity_gpu_field,
                                        Adiabatic,
                                        ADIABATICINDEX,
                                        ASPECTRATIO,
                                        FLARINGINDEX,
                                        //(DustConstStokes ? dust_size: dust_size/2.0),
                                        dust_size,
                                        DUSTBULKDENS,
                                        DustConstStokes,
                                        MASSUNIT,
                                        ns,
                                        nr,
                                        VradGas->pitch/sizeof(double), 
                                        dt,
                                        (double)ns/2.0/M_PI
                                        //myWork->gpu_field
                                        );
  
  
  cudaThreadSynchronize();
  getLastCudaError ("kernel_substep1b failed");
  
  // dust density changed due to difffusion must be copied
  ActualiseGas_gpu (RhoDust_ret, Work);
}
//----------------------------------------------------------------------------------------------------------------------


// substep1 for gas and single dust component (no growth, backreaction inlcuded)
//----------------------------------------------------------------------------------------------------------------------
__global__ void kernel_substep1c (const double *vrad_g,
                                  const double *vtheta_g,
                                  const double *rho_g,
                                  const double *vrad_d,
                                  const double *vtheta_d,
                                  const double *rho_d,
                                  const double *pot,
                                  const double *energy,
                                        double *vrad_g_int,
                                        double *vtheta_g_int,
                                        double *vrad_d_int,
                                        double *vtheta_d_int,
                                        double *rho_d_int,
                                  const double *viscosity,
                                  const bool    adiabatic,
                                  const double  adiabatic_index,
                                  const double  aspect_ratio,
                                  const double  flaring_index,
                                  const double  dust_size,
                                  const double  dust_bulk_dens,
                                  const bool    const_stokes,
                                  const double  mass_unit,
                                  const int     ns, 
                                  const int     nr, 
                                  const int     pitch,
                                  const double  dt, 
                                  const double  invdphi) {

  // indices for radial: ig; azimthal: jg; cell: idg
  const int jg = threadIdx.x + blockIdx.x * blockDim.x;
  const int ig = threadIdx.y + blockIdx.y * blockDim.y;
  const int idg = ig * pitch + jg;
  
  // idgjp & idgjm are azimuthal +1/-1 cell indices
  int jgp = jg+1;
  int jgm = jg-1;
  int idgjp, idgjm;
  if (jg == 0) 
    jgm = ns-1;
  if (jg == ns-1) 
    jgp = 0;
  idgjp = jgp + ig * pitch;
  idgjm = jgm + ig * pitch;

  // calcuate stopping time for dust particels
  double tstop;

  // based on predefined Stokes number
  if (const_stokes) {
//    tstop = dust_size * pow(rmed,1.5);
    tstop = dust_size / omega;
  }
  // based on particle size
  else {

#ifndef EPSTEIN_ONLY
    const double dvr = vrad_d[idg]-vrad_g[idg];
    const double dvt = vtheta_d[idg]-vtheta_g[idg];
    const double dv_abs = sqrt(dvr*dvr+dvt*dvt); 
#endif
    
    tstop = calc_timestop(rho_g[idg],
                          (adiabatic ? energy[idg]: 0.0),
                          dust_size,
                          rmed,
                          omega,
#ifndef EPSTEIN_ONLY
                          dv_abs,
#endif
                          aspect_ratio,
                          flaring_index,
                          dust_bulk_dens,
                          adiabatic_index,
                          adiabatic,
                          mass_unit);
  }
    
  // dust feedback requires dust-to-gas massratio
  const double rho_g_idg    = rho_g[idg];
  const double rho_g_idgjm  = rho_g[idgjm];
  const double rho_d_idg    = rho_d[idg]; 
  const double rho_d_idgjm  = rho_d[idgjm];
  const double vrad_g_idg   = vrad_g[idg];
  const double vtheta_g_idg = vtheta_g[idg];
  const double vrad_d_idg   = vrad_d[idg];
  const double vtheta_d_idg = vtheta_d[idg];
  
  const double epsilon = rho_d_idg/rho_g_idg;
  double gradpg, gradpd , gradpot, acc_g, acc_d, vt_g, vt_d, vk;
  double vrad_g_temp = 0.0, vrad_d_temp = 0.0;
  double vtheta_g_temp = 0.0, vtheta_d_temp = 0.0;
  double X, Y;
  
  // Stokes number
  const double St = tstop * omega;

  // radial velocities
  if (ig > 0) {
    const double rho_g_idgim  = rho_g[idg-pitch];
    const double rho_d_idgim  = rho_d[idg-pitch];
    if (adiabatic) {
      gradpg = 2.0 * (adiabatic_index-1.0)*(energy[idg] - energy[idg-pitch]) / (rho_g_idg + rho_g_idgim); 
    }
    else {
      gradpg = 2.0 * ((cs2 * rho_g_idg - cs2m * rho_g_idgim) / (rho_g_idg + rho_g_idgim));
    }
    gradpd = 0*C_DP* 2.0 * ((cs2 * rho_d_idg - cs2m * rho_d_idgim) / (rho_d_idg + rho_d_idgim));
    gradpot = pot[idg] - pot[idg-pitch];
    vk = sqrt(invrinf);
    vt_g = vtheta_g_idg + vtheta_g[idgjp] + vtheta_g[idg-pitch] + vtheta_g[idgjp-pitch];
    vt_g = vt_g*0.25;
    vt_d = vtheta_d_idg + vtheta_d[idgjp] + vtheta_d[idg-pitch] + vtheta_d[idgjp-pitch];
    vt_d = vt_d*0.25;
    acc_g = -(gradpg + gradpot) * invdr;
    acc_d = -(gradpd + gradpot) * invdr;

    X = (vrad_g_idg - vrad_d_idg + dt * (acc_g - acc_d)) / (1.0 + (epsilon + 1.0) * dt / tstop);
    Y = (vrad_g_idg + epsilon * vrad_d_idg) + dt * (acc_g + epsilon * acc_d);
    
    vrad_g_temp = (epsilon * X + Y) / (epsilon + 1.0) + dt * vt_g * invrinf * (vt_g + 2.0 * vk);
    vrad_d_temp = (Y - X) / (epsilon + 1.0) + dt * vt_d * invrinf * (vt_d + 2.0 * vk);
  }
  
  //azimuthal velocities
  if (adiabatic) {
    gradpg = 2.0 * (adiabatic_index - 1.0) * (energy[idg] - energy[idgjm]) / (rho_g[idg] + rho_g[idgjm]); 
  }
  else
    gradpg = 2.0 * cs2 * ((rho_g_idg-rho_g_idgjm) / (rho_g_idg + rho_g_idgjm));
  gradpd = 0*C_DP * 2.0 * cs2 * ((rho_d_idg-rho_d_idgjm) / (rho_d_idg + rho_d_idgjm));
  gradpot = pot[idg]-pot[idgjm];
  acc_g = -(gradpg + gradpot) * invdphi * invrmed;
  acc_d = -(gradpd + gradpot) * invdphi * invrmed;

  X = (vtheta_g_idg - vtheta_d_idg + dt * (acc_g - acc_d)) / (1.0 + (epsilon + 1.0) * dt / tstop);
  Y = (vtheta_g_idg + epsilon * vtheta_d_idg) + dt * (acc_g + epsilon * acc_d);

  vtheta_g_temp = (epsilon * X + Y) / (epsilon + 1.0);
  vtheta_d_temp = (Y - X) / (epsilon + 1.0);
    
  // finally set the intermediate velocities of gas and dust
  vrad_g_int[idg]   = vrad_g_temp;
  vtheta_g_int[idg] = vtheta_g_temp;

  vrad_d_int[idg]   = vrad_d_temp;
  vtheta_d_int[idg] = vtheta_d_temp;
  
  // dust diffusion and growth correction
  if (ig > 0 && ig < nr-1) {

    // get appropriate viscosity
    double nu;
    if (viscosity != NULL)
      nu = viscosity[idg];
    else
      nu = visco;
    
    // diffussion coefficient, see Eq. (17) of Birnstiel et al. (2010)
    const double Ddt = 0*dt * nu / (1.0 + St * St);
    rho_d_int[idg] = rho_d_idg + Ddt * ((rho_d[idg+pitch] - 2.0 * rho_d_idg + rho_d[idg-pitch]) * invdr * invdr +
                                        (rho_d[idgjp]     - 2.0 * rho_d_idg + rho_d_idgjm)     * invdphi * invrmed * invdphi * invrmed);
  }
}

void SubStep1GasDust_gpu (PolarGrid *VradGas, PolarGrid *VthetaGas, PolarGrid *RhoGas, PolarGrid *Energy, 
                          PolarGrid *VradDust, PolarGrid *VthetaDust, PolarGrid *RhoDust,
                          double dust_size, double dt,
                          PolarGrid *VradGas_ret, PolarGrid *VthetaGas_ret,
                          PolarGrid *VradDust_ret, PolarGrid *VthetaDust_ret, PolarGrid *RhoDust_ret) {


  int nr, ns;
  nr = VradGas->Nrad;
  ns = VradGas->Nsec;
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid ((ns+block.x-1)/block.x, (nr+block.y-1)/block.y);
  
  double *Energy_gpu_field = NULL;
  if (Adiabatic) {
    Energy_gpu_field = Energy->gpu_field;
  }
  
  double *Viscosity_gpu_field = NULL;
  if (Adiabatic || AdaptiveViscosity)
    Viscosity_gpu_field = Viscosity->gpu_field;

  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)RadiiStuff, (size_t)(15*(nr+1))*sizeof(double),	0, cudaMemcpyHostToDevice));
  kernel_substep1c <<< grid, block >>> (VradGas->gpu_field,
                                        VthetaGas->gpu_field,
                                        RhoGas->gpu_field,
                                        VradDust->gpu_field,
                                        VthetaDust->gpu_field,
                                        RhoDust->gpu_field,
                                        Potential->gpu_field,
                                        Energy_gpu_field,
                                        VradGas_ret->gpu_field,
                                        VthetaGas_ret->gpu_field,
                                        VradDust_ret->gpu_field,
                                        VthetaDust_ret->gpu_field,
                                        Work->gpu_field,
                                        Viscosity_gpu_field,
                                        Adiabatic,
                                        ADIABATICINDEX,
                                        ASPECTRATIO,
                                        FLARINGINDEX,
                                        //(DustConstStokes ? dust_size: dust_size/2.0),
                                        dust_size,
                                        DUSTBULKDENS,
                                        DustConstStokes,
                                        MASSUNIT,
                                        ns,
                                        nr,
                                        VradGas->pitch/sizeof(double), 
                                        dt,
                                        (double)ns/2.0/M_PI);
  
  
  cudaThreadSynchronize();
  getLastCudaError ("kernel_substep1c failed");

  // dust density changed due to difffusion must be copied
  ActualiseGas_gpu (RhoDust_ret, Work);
}
//----------------------------------------------------------------------------------------------------------------------



// substep1 for gas and single dust compinent (no growth, backreaction inlcuded)
//----------------------------------------------------------------------------------------------------------------------
__global__ void kernel_substep1d (const double *vrad_g,
                                  const double *vtheta_g,
                                  const double *rho_g,
                                  const double *vrad_d,
                                  const double *vtheta_d,
                                  const double *rho_d,
                                  const double *rho_dsm,
                                  const double *pot,
                                  const double *energy,
                                        double *vrad_g_int,
                                        double *vtheta_g_int,
                                        double *vrad_d_int,
                                        double *vtheta_d_int,
                                        double *rho_d_int,
                                        double *vrad_dsm_int,
                                        double *vtheta_dsm_int,
                                        double *rho_dsm_int,
                                  const double *viscosity,
                                  const double *dust_size,
                                        double *dust_size_int,
                                  const double *growth_rate,
                                  const bool    dust_feedback,
                                  const bool    adiabatic,
                                  const double  adiabatic_index,
                                  const double  aspect_ratio,
                                  const double  flaring_index,
                                  const double  dust_bulk_dens,
                                  const double  mass_unit,
                                  const int     ns,
                                  const int     nr,
                                  const int     pitch,
                                  const double  dt,
                                  const double  invdphi) {

  // indices for radial: ig; azimthal: jg; cell: idg
  const int jg = threadIdx.x + blockIdx.x * blockDim.x;
  const int ig = threadIdx.y + blockIdx.y * blockDim.y;
  const int idg = ig * pitch + jg;
  
  // idgjp & idgjm are azimuthal +1/-1 cell indices
  int jgp = jg+1;
  int jgm = jg-1;
  int idgjp, idgjm;
  if (jg == 0) 
    jgm = ns-1;
  if (jg == ns-1) 
    jgp = 0;
  idgjp = jgp + ig * pitch;
  idgjm = jgm + ig * pitch;

  // calcuate stopping time for dust particels
  //double tstop = 1.0/omega;
  double tstop;
#ifndef EPSTEIN_ONLY
    const double dvr = vrad_d[idg]-vrad_g[idg];
    const double dvt = vtheta_d[idg]-vtheta_g[idg];
    const double dv_abs = sqrt(dvr*dvr+dvt*dvt); 
#endif
    
    tstop = calc_timestop(rho_g[idg],
                          (adiabatic ? energy[idg] : 0.0),
                          //dust_size[idg]/2.0,
                          dust_size[idg],
                          rmed,
                          omega,
#ifndef EPSTEIN_ONLY
                          dv_abs,
#endif
                          aspect_ratio,
                          flaring_index,
                          dust_bulk_dens,
                          adiabatic_index,
                          adiabatic,
                          mass_unit);
  
  const double rho_g_idg    = rho_g[idg];
  const double rho_g_idgjm  = rho_g[idgjm];
  const double rho_d_idg    = rho_d[idg]; 
  const double rho_d_idgjm  = rho_d[idgjm];
  const double vrad_g_idg   = vrad_g[idg];
  const double vtheta_g_idg = vtheta_g[idg];
  const double vrad_d_idg   = vrad_d[idg];
  const double vtheta_d_idg = vtheta_d[idg];  
  
  double gradpg, gradpd , gradpot, acc_g, acc_d, vt_g, vt_d, vk;
  double vrad_g_temp = 0.0, vrad_d_temp = 0.0;
  double vtheta_g_temp = 0.0, vtheta_d_temp = 0.0;
  double X, Y;

  // dust feedback requires dust-to-gas massratio
  double epsilon;
  if (dust_feedback) 
   epsilon = rho_d[idg]/rho_g[idg]; 
  else 
   epsilon = 0.0;
  
  // Stokes number
  const double St = tstop * omega;
  
  // fully implicit method for small Stokes number
  if (St < 100000) {
    // radial velocities
    if (ig > 0) {
      const double rho_g_idgim  = rho_g[idg-pitch];
      const double rho_d_idgim  = rho_d[idg-pitch];
      if (adiabatic)
        gradpg = 2.0 * (adiabatic_index-1.0)*(energy[idg] - energy[idg-pitch]) / (rho_g_idg + rho_g_idgim);
      else
        gradpg = 2.0 * ((cs2 * rho_g_idg - cs2m * rho_g_idgim) / (rho_g_idg + rho_g_idgim));
      gradpd = 0;//*C_DP* 2.0 * ((cs2 * rho_d_idg - cs2m * rho_d_idgim) / (rho_d_idg + rho_d_idgim));
      gradpot = pot[idg] - pot[idg-pitch];
      vk = sqrt(invrinf);
      vt_g = vtheta_g_idg + vtheta_g[idgjp] + vtheta_g[idg-pitch] + vtheta_g[idgjp-pitch];
      vt_g = vt_g*0.25;
      vt_d = vtheta_d_idg + vtheta_d[idgjp] + vtheta_d[idg-pitch] + vtheta_d[idgjp-pitch];
      vt_d = vt_d*0.25;
      acc_g = -(gradpg + gradpot) * invdr;
      acc_d = -(gradpd + gradpot) * invdr;
    
      X = (vrad_g_idg - vrad_d_idg + dt * (acc_g - acc_d)) / (1.0 + (epsilon + 1.0) * dt / tstop);
      Y = (vrad_g_idg + epsilon * vrad_d_idg) + dt * (acc_g + epsilon * acc_d);
      
      vrad_g_temp = (epsilon * X + Y) / (epsilon + 1.0) + dt * vt_g * invrinf * (vt_g + 2.0 * vk);
      vrad_d_temp = (Y - X) / (epsilon + 1.0) + dt * vt_d * invrinf * (vt_d + 2.0 * vk);
    }
    
    //azimuthal velocities
    if (adiabatic)
      gradpg = 2.0 * (adiabatic_index - 1.0) * (energy[idg] - energy[idgjm]) / (rho_g_idg + rho_g_idgjm); 
    else
      gradpg = 2.0 * cs2 * ((rho_g_idg-rho_g_idgjm) / (rho_g_idg + rho_g_idgjm));
    gradpd = 0;//*C_DP * 2.0 * cs2 * ((rho_d_idg-rho_d_idgjm) / (rho_d_idg + rho_d_idgjm));
    gradpot = pot[idg]-pot[idgjm];
    acc_g = -(gradpg + gradpot) * invdphi * invrmed;
    acc_d = -(gradpd + gradpot) * invdphi * invrmed;
    
    X = (vtheta_g_idg - vtheta_d_idg + dt * (acc_g - acc_d)) / (1.0 + (epsilon + 1.0) * dt / tstop);
    Y = (vtheta_g_idg + epsilon * vtheta_d_idg) + dt * (acc_g + epsilon * acc_d);
    
    vtheta_g_temp = (epsilon * X + Y) / (epsilon + 1.0);
    vtheta_d_temp = (Y - X) / (epsilon + 1.0);
  }
  // fully explicit for large Stokes numbers
  else {
    const double rho_g_idgim  = rho_g[idg-pitch];
    const double rho_d_idgim  = rho_d[idg-pitch];

    // radial directon
    if (ig > 0) {
      gradpot = pot[idg] - pot[idg-pitch];
      if (adiabatic) {
        gradpg = 2.0 * (adiabatic_index-1.0)*(energy[idg] - energy[idg-pitch]) / (rho_g_idg + rho_g_idgim); 
      }
      else {
        gradpg = 2.0 * ((cs2 * rho_g_idg - cs2m * rho_g_idgim) / (rho_g_idg + rho_g_idgim));
      }
      gradpd = 0*C_DP* 2.0 * ((cs2 * rho_d_idg - cs2m * rho_d_idgim) / (rho_d_idg + rho_d_idgim));
      vk = sqrt(invrinf);
      vt_g = vtheta_g_idg+vtheta_g[idgjp] + vtheta_g[idg-pitch]+vtheta_g[idgjp-pitch];
      vt_g = vt_g * 0.25;
      vt_d = vtheta_d_idg + vtheta_d[idgjp] + vtheta_d[idg-pitch] + vtheta_d[idgjp-pitch];
      vt_d = vt_d*0.25;
      vrad_g_temp = vrad_g_idg + dt * (-(gradpg + gradpot) * invdr + vt_g * invrinf * (vt_g + 2.0 * vk));
        
      //const double dvrad_drag = - (vrad_d[idg]-vrad_g[idg])*dt/(tstop);              // 1'st order: unphysical jump in radial velocity at dt~tstop
      const double dvrad_drag = - (vrad_d_idg-vrad_g_idg)*dt/(tstop+dt/2.0);         // 2'nd order: improoved Euler scheme, see Eq. (6) of Zhu et al (2012)
      vrad_d_temp = vrad_d_idg + dvrad_drag + dt*(-(gradpd + gradpot) * invdr + vt_d * invrinf * (vt_d + 2.0 * vk));
    }
    
    // azimuthal direction
    gradpot = pot[idg]-pot[idgjm];
    if (adiabatic) {
      gradpg = 2.0 * (adiabatic_index - 1.0) * (energy[idg] - energy[idgjm]) / (rho_g_idg + rho_g_idgjm); 
    }
    else
      gradpg = 2.0 * cs2 * ((rho_g_idg-rho_g_idgjm) / (rho_g_idg + rho_g_idgjm));
    gradpd = 0*C_DP * 2.0 * cs2 * ((rho_d_idg-rho_d_idgjm) / (rho_d_idg + rho_d_idgjm));
    vtheta_g_temp = vtheta_g_idg - dt * (gradpg + gradpot) * invdphi * invrmed;
    
    //const double dvtheta_drag =  - (vtheta_d[idg]- vtheta_g[idg])*dt/(tstop);       // 1'st order: unphysical jump in radial velocity at dt~tstop
    const double dvtheta_drag =  - (vtheta_d_idg- vtheta_g_idg)*dt/(tstop+dt/2.0);  // 2'nd order: improoved Euler scheme, see Eq. (6) of Zhu et al (2012)
    vtheta_d_temp = vtheta_d_idg + dvtheta_drag - dt * (gradpd + gradpot) * invdphi * invrmed;
  }

  // finally set the intermediate velocities of gas
  vrad_g_int[idg]   = vrad_g_temp;
  vtheta_g_int[idg] = vtheta_g_temp;
  vrad_d_int[idg]   = vrad_d_temp;
  vtheta_d_int[idg] = vtheta_d_temp;
  /*
  // dust diffusion and growth correction
  if (ig > 0 && ig < nr-1) {

    // get appropriate viscosity
    double nu;
    if (viscosity != NULL)
      nu = viscosity[idg];
    else
      nu = visco;
      
    // diffussion coefficient, see Eq. (17) of Birnstiel et al. (2010)
    const double Ddt = 0;//dt * nu / (1.0 + St * St);
    const double Dsmdt = 0;//dt * nu;
    
    // for dust growth the sign of S(r) depends on the mass, see Eqs. (8) and (9) of Vorobyov et al. (2018) 
    rho_d_int[idg] = rho_d_idg + 0*Sdt + Ddt * ((rho_d[idg+pitch] - 2.0 * rho_d_idg + rho_d[idg-pitch]) * invdr * invdr +
                                              (rho_d[idgjp]     - 2.0 * rho_d_idg + rho_d_idgjm) * invdphi * invrmed * invdphi * invrmed);
    
    double rho_dsm_new = rho_dsm[idg];
  //  if (rho_dsm_new - Sdt > 1e-10)
  //    rho_dsm_new -= Sdt;
      
    rho_dsm_int[idg] = rho_dsm_new + 0*Dsmdt * ((rho_dsm[idg+pitch] - 2.0 * rho_dsm[idg] + rho_dsm[idg-pitch]) * invdr * invdr +
                                              (rho_dsm[idgjp]     - 2.0 * rho_dsm[idg] + rho_dsm[idgjm])     * invdphi * invrmed * invdphi * invrmed);
  
  
  //  dust_size_int[idg] = dust_size[idg] + Dsmdt * ((dust_size[idg+pitch] - 2.0 * dust_size[idg] + dust_size[idg-pitch]) * invdr * invdr +
  //                                             (dust_size[idgjp]     - 2.0 * dust_size[idg] + dust_size[idgjm])     * invdphi * invrmed * invdphi * invrmed);
  
  }
  */
}


void SubStep1GasDustMDGM_gpu (PolarGrid *VradGas, PolarGrid *VthetaGas, PolarGrid *RhoGas, PolarGrid *Energy, 
                              PolarGrid **VradDust, PolarGrid **VthetaDust, PolarGrid **RhoDust,
                              PolarGrid *DustSizeGr, PolarGrid *DustGrowthRate,
                              double dt,
                              PolarGrid *VradGas_ret, PolarGrid *VthetaGas_ret,
                              PolarGrid **VradDust_ret, PolarGrid **VthetaDust_ret, PolarGrid **RhoDust_ret) {

  int nr, ns;
  nr = VradGas->Nrad;
  ns = VradGas->Nsec;
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid ((ns+block.x-1)/block.x, (nr+block.y-1)/block.y);

  double *Energy_gpu_field = NULL;
  if (Adiabatic) {
    Energy_gpu_field = Energy->gpu_field;
  }

  double *Viscosity_gpu_field = NULL;
  if (Adiabatic || AdaptiveViscosity)
    Viscosity_gpu_field = Viscosity->gpu_field;

  // supstep1 for gas and grown dust
  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)RadiiStuff, (size_t)(15*(nr+1))*sizeof(double),	0, cudaMemcpyHostToDevice));
  kernel_substep1d <<< grid, block >>> (VradGas->gpu_field,
                                        VthetaGas->gpu_field,
                                        RhoGas->gpu_field,
                                        VradDust[0]->gpu_field,
                                        VthetaDust[0]->gpu_field,
                                        RhoDust[0]->gpu_field,
                                        RhoDust[1]->gpu_field,
                                        Potential->gpu_field,
                                        Energy_gpu_field,
                                        VradGas_ret->gpu_field,
                                        VthetaGas_ret->gpu_field,
                                        VradDust_ret[0]->gpu_field,
                                        VthetaDust_ret[0]->gpu_field,
                                        Work->gpu_field,
                                        VradDust_ret[1]->gpu_field,
                                        VthetaDust_ret[1]->gpu_field,
                                        tmp1->gpu_field,
                                        Viscosity_gpu_field,
                                        DustSizeGr->gpu_field,
                                        tmp2->gpu_field,
                                        DustGrowthRate->gpu_field,
                                        DustFeedback,
                                        Adiabatic,
                                        ADIABATICINDEX,
                                        ASPECTRATIO,
                                        FLARINGINDEX,
                                        DUSTBULKDENS,
                                        MASSUNIT,
                                        ns,
                                        nr,
                                        VradGas->pitch/sizeof(double), 
                                        dt,
                                        (double)ns/2.0/M_PI);
  cudaThreadSynchronize();
  getLastCudaError ("kernel_substep1d failed");

  // density of grown dust must be copied (changed here due to difffusion and/or growth rate)
  //FARGO_SAFE(ActualiseGas_gpu (RhoDust_ret[0], Work));
  //FARGO_SAFE(ActualiseGas_gpu (RhoDust_ret[1], tmp1));

  //FARGO_SAFE(ActualiseGas_gpu (dust_size, tmp2));
}
//----------------------------------------------------------------------------------------------------------------------