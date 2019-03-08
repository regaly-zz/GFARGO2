/** \file Substep3.cu : implements the kernel for the substep3 procedure
*/
#define __CUDA 1
#include "fargo.h"
#undef __CUDA
#include <stdarg.h>
#include <helper_cuda.h>
#include <cuda.h>

// choose only one of them (see, Stone & Norman 1992)
//#define COOLING_PREDCOR        // predictor corrector (used in FARGO_ADSG)
#define COOLING_IMPLICIT     // implicit version (used in FARGO3D)

// BLOCK_X : in azimuth
//#define BLOCK_X DEF_BLOCK_X_SUBSTEP2
#define BLOCK_X 64
// BLOCK_Y : in radius
#define BLOCK_Y 4

__device__ double CRadiiStuff[32768];


#define invdiffrmed  CRadiiStuff[           ig]
#define cs2          CRadiiStuff[(nr+1)*1 + ig]
#define invrmed      CRadiiStuff[(nr+1)*2 + ig]
#define invrmedm     CRadiiStuff[(nr+1)*2 + ig-1]
#define invrinf      CRadiiStuff[(nr+1)*3 + ig]
#define rinf         CRadiiStuff[(nr+1)*4 + ig]
#define rmed         CRadiiStuff[(nr+1)*6 + ig]
#define rmed_p       CRadiiStuff[(nr+1)*6 + ig +1]
#define rmed_2p      CRadiiStuff[(nr+1)*6 + ig +2]
#define rmedm        CRadiiStuff[(nr+1)*6 + ig-1]
#define rsup         CRadiiStuff[(nr+1)*8 + ig]
#define invdiffrsup  CRadiiStuff[(nr+1)*10+ ig]
#define visco        CRadiiStuff[(nr+1)*12+ ig]
#define visco_p      CRadiiStuff[(nr+1)*12+ ig + 1]
#define visco_2p     CRadiiStuff[(nr+1)*12+ ig + 2]

#define SigmaMed_gpu       CRadiiStuff[(nr+1)*14        + ig]
#define EnergyMed_gpu      CRadiiStuff[(nr+1)*14 + nr   + ig]
#define CoolingTimeMed_gpu CRadiiStuff[(nr+1)*14 + nr*2 + ig]
#define QplusMed_gpu       CRadiiStuff[(nr+1)*14 + nr*3 + ig]

__global__ void kernel_substep3 (double *dens,
                                 double *vrad,
                                 double *vtheta,
                                 double *energy,
                                 double *energynew,
                                 double *viscosity,
                                 double *tau_rr,
                                 double *tau_rp,
                                 double *tau_pp,
                                 double  adiabatic_index,
                                 bool    alpha_viscosity,
                                 bool    cooling,
                                 bool    visc_heating,
                                 int     ns, 
                                 int     nr, 
                                 int     pitch, 
                                 double  invdphi,
                                 double  dt) {
                                   
  // jg & ig, g like 'global' (global memory <=> full grid)
  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;
  int idg   = jg + ig * pitch;
  
  // viscous heating
  double div_v = 0.0;
//  if (ig < nr-1)  
//    div_v = divergence_vel[idg];

  
  int jgp = jg + 1;
  if (jg == ns-1) jgp = 0;
  int idgjp = jgp + ig * pitch; 
  if (ig < nr-1) {
    div_v = ((vrad[idg+pitch]*rsup - vrad[idg]*rinf)*invdiffrsup + (vtheta[idgjp]-vtheta[idg])*invdphi)*invrmed;
  }
  else
    div_v = ((vrad[idg]*rsup - vrad[idg-pitch]*rinf)*invdiffrsup + (vtheta[idgjp]-vtheta[idg])*invdphi)*invrmed;
  
  double qplus = 0;
  if (visc_heating) {
    double nu, nu_p, nu_2p;
    if (alpha_viscosity) {
      nu = viscosity[idg];
      nu_p = viscosity[idg+pitch];
      nu_2p = viscosity[idg+2*pitch];
    }
    else {
      nu = visco;
      nu_p = visco_p;
      nu_2p = visco_2p;
      
    }
    if (ig > 0) {
      qplus = 0.5/nu/dens[idg]*(tau_rr[idg]*tau_rr[idg] + tau_rp[idg]*tau_rp[idg] + tau_pp[idg]*tau_pp[idg] );
      qplus += (2.0/9.0)*nu*dens[idg]*div_v*div_v;
    }
    else {
      int idgp = idg+ns;
      int idg2p = idgp+ns;
      double qpip = 0.5/nu_p/dens[idgp]*(tau_rr[idgp]*tau_rr[idgp] + tau_rp[idgp]*tau_rp[idgp] + tau_pp[idgp]*tau_pp[idgp] );
      qpip += (2.0/9.0)*nu_p*dens[idgp]*div_v*div_v;
      double qpi2p = 0.5/nu_2p/dens[idg2p]*(tau_rr[idg2p]*tau_rr[idg2p] + tau_rp[idg2p]*tau_rp[idg2p] + tau_pp[idg2p]*tau_pp[idg2p] );
      qpi2p += (2.0/9.0)*nu_2p*dens[idg2p]*div_v*div_v;

      qplus = qpip*exp( log(qpip/qpi2p) * log(rmed/rmed_p) / log(rmed_p/rmed_2p));
    }
  }
  
  // cooling
  if (cooling) {
#ifdef COOLING_PREDCOR // implemented in Fargo_ADSG
    double num = EnergyMed_gpu*dt*dens[idg]/SigmaMed_gpu + CoolingTimeMed_gpu*energy[idg] + 0*dt*CoolingTimeMed_gpu*(qplus-QplusMed_gpu*dens[idg]/SigmaMed_gpu);
    double den = dt + CoolingTimeMed_gpu + (adiabatic_index-1.0)*dt*CoolingTimeMed_gpu*div_v;
    energynew[idg] = num/den;
#endif
#ifdef COOLING_IMPLICIT
    const double term = 0.5*dt*(adiabatic_index-1.0)*div_v;
    if (visc_heating)
      qplus -= QplusMed_gpu*dens[idg]/SigmaMed_gpu;

    double qminus = (1.0/CoolingTimeMed_gpu) * (energy[idg] - EnergyMed_gpu*(dens[idg]/SigmaMed_gpu));
    energynew[idg] = (energy[idg]*(1.0-term) + dt * (qplus - qminus))/(1.0+term);
#endif
  }
  else {
#ifdef COOLING_PREDCOR // implemented in Fargo_ADSG
    double num = dt*qplus + energy[idg];
    double den = 1.0+(adiabatic_index-1.0) * dt *div_v;
    energynew[idg] = num/den;
#endif
#ifdef COOLING_IMPLICIT // implemented in Fargo3D
    const double term = 0.5*dt*(adiabatic_index-1.0)*div_v;
    energynew[idg] = (energy[idg]*(1.0-term) + dt*qplus)/(1.0+term);
#endif
  }
}


void SubStep3_gpu (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, PolarGrid *Energy, double dt,
                   PolarGrid *Energy_ret) {
  int nr, ns;
  nr = Rho->Nrad;
  ns = Rho->Nsec;
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid ((ns+block.x-1)/block.x, (nr+block.y-1)/block.y);

  double *TauRR_gpu_field = NULL;
  double *TauRP_gpu_field = NULL;
  double *TauPP_gpu_field = NULL;
  if (Energy != NULL && ViscHeating) {
    TauRR_gpu_field = TauRR->gpu_field;
    TauRP_gpu_field = TauRP->gpu_field;
    TauPP_gpu_field = TauPP->gpu_field;
  }
  
  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *) RadiiStuff,     (size_t)(14*(nr+1))*sizeof(double), 0, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *) SigmaMed,       (size_t)(nr)*sizeof(double), (14*(nr+1)       )*sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *) EnergyMed,      (size_t)(nr)*sizeof(double), (14*(nr+1) +   nr)*sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *) CoolingTimeMed, (size_t)(nr)*sizeof(double), (14*(nr+1) + 2*nr)*sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *) QplusMed,       (size_t)(nr)*sizeof(double), (14*(nr+1) + 3*nr)*sizeof(double), cudaMemcpyHostToDevice));
  
  
  kernel_substep3 <<< grid, block >>> (Rho->gpu_field,
                                       Vrad->gpu_field,
                                       Vtheta->gpu_field,
                                       Energy->gpu_field,
                                       Energy_ret->gpu_field,
                                       Viscosity->gpu_field,
                                       TauRR_gpu_field,
                                       TauRP_gpu_field,
                                       TauPP_gpu_field,
                                       ADIABATICINDEX,
                                       ViscosityAlpha,
                                       Cooling,
                                       ViscHeating,
                                       ns, 
                                       nr,
                                       Energy->pitch/sizeof(double), 
                                       (double)(Rho->Nsec)/2.0/M_PI,
                                       dt);
  cudaThreadSynchronize();
  getLastCudaError ("kernel_substep3 failed");    
  
  /*
  if (Cooling) {
    if (ViscHeating) {
      kernel_substep3_cooling_vischeating <<< grid, block >>> (Rho->gpu_field,
                                                               Vrad->gpu_field,
                                                               Vtheta->gpu_field,
                                                               EnergyInt->gpu_field,
                                                               Energy->gpu_field,
                                                               Viscosity->gpu_field,
                                                               TauRR->gpu_field,
                                                               TauRP->gpu_field,
                                                               TauPP->gpu_field,
                                                               ADIABATICINDEX,
      				                                                 ns, 
                                                               nr,
      				                                                 Energy->pitch/sizeof(double), 
                                                               (double)(Rho->Nsec)/2.0/M_PI,
                                                               dt);
      cudaThreadSynchronize();
      getLastCudaError ("kernel_substep3_cooling_vischeating failed");    
    }
    else {
      kernel_substep3_cooling <<< grid, block >>> (Rho->gpu_field,
                                                   Vrad->gpu_field,
                                                   Vtheta->gpu_field,
                                                   EnergyInt->gpu_field,
                                                   Energy->gpu_field,
                                                   ADIABATICINDEX,
      				                                     ns, 
                                                   nr,
      				                                     Energy->pitch/sizeof(double), 
                                                   (double)(Rho->Nsec)/2.0/M_PI,
                                                   dt);
      cudaThreadSynchronize();
      getLastCudaError ("kernel_substep3_cooling failed");    
    }
  }
  else {
    if (ViscHeating) {
      kernel_substep3_nocooling_vischeating <<< grid, block >>> (Rho->gpu_field,
                                                                 Vrad->gpu_field,
                                                                 Vtheta->gpu_field,
                                                                 EnergyInt->gpu_field,
                                                                 Energy->gpu_field,
                                                                 Viscosity->gpu_field,
                                                                 TauRR->gpu_field,
                                                                 TauRP->gpu_field,
                                                                 TauPP->gpu_field,
                                                                 ADIABATICINDEX,
                                                                 ns, 
                                                                 nr,
                                                                 Energy->pitch/sizeof(double), 
                                                                 (double)(Rho->Nsec)/2.0/M_PI,
                                                                 dt);
      cudaThreadSynchronize();
      getLastCudaError ("kernel_substep3_nocooling failed");    
    }
    else {
      kernel_substep3_nocooling <<< grid, block >>> (Rho->gpu_field,
                                                     Vrad->gpu_field,
                                                     Vtheta->gpu_field,
                                                     EnergyInt->gpu_field,
                                                     Energy->gpu_field,
                                                     ADIABATICINDEX,
                                                     ns, 
                                                     nr,
                                                     Energy->pitch/sizeof(double), 
                                                     (double)(Rho->Nsec)/2.0/M_PI,
                                                     dt);
      cudaThreadSynchronize();
      getLastCudaError ("kernel_substep3_nocooling failed");    
    }
  }*/
}
