/** \file bc_damping.cu: contains a CUDA kernel for Stockholm's prescription of damping boundary conditions.
*/
#define __CUDA 1
#include "fargo.h"
#undef __CUDA
#include <stdarg.h>
#include <helper_cuda.h>
#include <cuda.h>

// BLOCK_X : in azimuth
//#define BLOCK_X DEF_BLOCK_X_DAMPINGBC
#define BLOCK_X 16//64
#define BLOCK_Y 1
/* Note that with the above choice of BLOCK_Y, the occupancy is not 1,
   but there is less arithmetic done within the kernels, and the
   performance turns out to be better. */

/// Improve legibility a bit. These variables are not stored in
/// registers, this turns out to be much faster.

#define RMED    CRadiiStuff[       ig]
#define DENS0   CRadiiStuff[  nr + ig]
#define VRAD0   CRadiiStuff[2*nr + ig]
#define VTHETA0 CRadiiStuff[3*nr + ig]
#define ENERGY0 CRadiiStuff[4*nr + ig]

__device__ double CRadiiStuff[16384];

#define GET_TAB(u,x,y,pitch) *(u + __mul24(y, pitch) + x)

__global__ void kernel_dampbc_in (double *vrad,
                                  double *vtheta,
                                  int     nr, 
                                  int     pitch,
                                  double  dt_tau,
                                  double  rinf, 
                                  double  rmin, 
                                  double  invdr) {

  double ramp, temp;
  int ig = blockIdx.y;
  int jg = blockIdx.x*BLOCK_X + threadIdx.x;

  // beyond damping radius exit
  if (RMED > rinf) 
    return;
  
  ramp = (RMED - rinf)*invdr;
  ramp = ramp*ramp*dt_tau;
  double fact = ramp/__dadd_rn(1.0, ramp);

  // radial velocity
  temp = GET_TAB (vrad,jg,ig,pitch);
  GET_TAB (vrad,jg,ig,pitch) = __dadd_rn (temp, (VRAD0-temp)*fact);

  // azimuthal velocity	
  temp = GET_TAB(vtheta,jg,ig,pitch);
  GET_TAB (vtheta,jg,ig,pitch) =  __dadd_rn (temp, (VTHETA0-temp)*fact);
}


 __global__ void kernel_strongdampbc_in (double *vrad,
                                         double *vtheta,
                                         double *rho,
                                         double *energy,
                                         int     nr, 
                                         int     pitch,
                                         double  dt_tau,
                                         double  rinf, 
                                         double  rmin, 
                                         double  invdr) {
   //  int ig = blockIdx.y*BLOCK_Y+threadIdx.y;
   int ig = blockIdx.y;
   int jg = blockIdx.x*BLOCK_X + threadIdx.x;

   // inside damping radius exit
   if (RMED > rinf) 
     return;
   
   double ramp, temp;
   
   ramp = (RMED - rinf)*invdr;
   ramp = ramp*ramp*dt_tau;

//   double fact = ramp/__dadd_rn(1.0, ramp);
  
   /*
   temp = GET_TAB(rho,jg,ig,pitch);
   GET_TAB (rho,jg,ig,pitch) =  __dadd_rn (temp, (DENS0-temp)*fact);
   temp = GET_TAB (vrad,jg,ig,pitch);
   GET_TAB (vrad,jg,ig,pitch) = __dadd_rn (temp, (VRAD0-temp)*fact);
   temp = GET_TAB(vtheta,jg,ig,pitch);
   GET_TAB (vtheta,jg,ig,pitch) =  __dadd_rn (temp, (VTHETA0-temp)*fact);
   if (adiabatic) {
     temp = GET_TAB(energy,jg,ig,pitch);
     GET_TAB (energy,jg,ig,pitch) =  __dadd_rn (temp, (ENERGY0-temp)*fact);
   }
   */
   const int idg = __mul24(ig, pitch) + jg;
   temp = rho[idg];
   rho[idg] = (temp + ramp*DENS0)/(1.0 + ramp);
   temp = vrad[idg];
   vrad[idg] = (temp + ramp*VRAD0)/(1.0 + ramp);
   temp = vtheta[idg];
   vtheta[idg] = (temp + ramp*VTHETA0)/(1.0 + ramp);
  
   if (energy != NULL) {
     temp = energy[idg];
     energy[idg] = (temp + ramp*ENERGY0)/(1.0 + ramp);
  }
}


__global__ void kernel_dampbc_out (double *vrad,
                                   double *vtheta,
                                   int     nr, 
                                   int     pitch,
                                   int     imin, 
                                   double  dt_tau,
                                   double  rsup, 
                                   double  rmax, 
                                   double  invdr) {
  double ramp, temp;
  //  int ig = nr-1-(blockIdx.y*BLOCK_Y+threadIdx.y);
  int ig = blockIdx.y+imin;
  int jg = blockIdx.x*blockDim.x + threadIdx.x;

  // beyond damping radius
  if (RMED < rsup)
    return;
  
  ramp = __dmul_rz(__dadd_rn(RMED, -rsup), invdr);
  ramp = ramp*ramp*dt_tau;
  double fact = ramp / __dadd_rn(1.0, ramp);

  temp = GET_TAB (vrad,jg,ig,pitch);
  GET_TAB (vrad,jg,ig,pitch) = __dadd_rn (temp, (VRAD0-temp)*fact);
  temp = GET_TAB(vtheta,jg,ig,pitch);
  GET_TAB (vtheta,jg,ig,pitch) =  __dadd_rn (temp, (VTHETA0-temp)*fact);
}

__global__ void kernel_strongdampbc_out (double *vrad,
                                         double *vtheta,
                                         double *rho,
                                         double *energy,
                                         int     nr, 
                                         int     pitch,
                                         int     imin, 
                                         double  dt_tau,
                                         double  rsup, 
                                         double  rmax, 
                                         double  invdr) {
  double ramp, temp;
  //  int ig = nr-1-(blockIdx.y*BLOCK_Y+threadIdx.y);
  int ig = blockIdx.y+imin;
  int jg = blockIdx.x*blockDim.x + threadIdx.x;
  if (RMED < rsup) return;

  ramp = __dmul_rz(__dadd_rn(RMED, -rsup), invdr);
  ramp = ramp*ramp*dt_tau;
  double fact = ramp/__dadd_rn(1.0, ramp);

  temp = GET_TAB(rho,jg,ig,pitch);
  GET_TAB (rho,jg,ig,pitch) =  __dadd_rn (temp, (DENS0-temp)*fact);
  temp = GET_TAB (vrad,jg,ig,pitch);
  GET_TAB (vrad,jg,ig,pitch) = __dadd_rn (temp, (VRAD0-temp)*fact);
  temp = GET_TAB(vtheta,jg,ig,pitch);
  GET_TAB (vtheta,jg,ig,pitch) =  __dadd_rn (temp, (VTHETA0-temp)*fact);

  if (energy != NULL) {
    temp = GET_TAB(energy,jg,ig,pitch);
    GET_TAB (energy,jg,ig,pitch) =  __dadd_rn (temp, (ENERGY0-temp)*fact);
  }
}


extern "C" 
void StockholmBoundary_gpu (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, PolarGrid *Energy, double dt, int where, double R_inf, double R_sup, bool strong) {
  int nr, ns, imin, imax, imed;
  static int FirstTime=YES, imaxi, imino;
  
  int nb_block_y;

  //R_inf = (double)RMIN*1.25;
  //R_sup = (double)RMAX*.84;
  nr = Vrad->Nrad;
  ns = Vrad->Nsec;

  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid;
  grid.x = (ns+block.x-1)/block.x;

  if (FirstTime) {  
    // Dichotomic search of starting index of outer damping BC (on host)
    imin=0;
    imax=nr-1;
    while (imax-imin > 1) {
      imed = (imax+imin)/2;
      if (Rmed[imed] > R_sup)
        imax=imed;
      else
        imin=imed;
    }
    imino = imin;
    
    // Dichotomic search of ending index of inner damping BC (on host)
    imin=0;
    imax=nr-1;
    while (imax-imin > 1) {
      imed = (imax+imin)/2;
      if (Rmed[imed] < R_inf)
        imin=imed;
      else
        imax=imed;
    }
    imaxi = imax;
    FirstTime = NO;
  }
  
  double *energy_gpu = NULL;          // must be defined as for non-adiabatic there is no Energy->gpu_filed!
  if (Adiabatic && Energy != NULL) {
    energy_gpu = Energy->gpu_field;   // get gpu filed for energy
  }

  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)(RadiiStuff+6*(nr+1)), (size_t)(nr)*sizeof(double),	                 0,  cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *) SigmaMed,       (size_t)(nr)*sizeof(double),	  nr*sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *) GasVelRadMed,   (size_t)(nr)*sizeof(double),	2*nr*sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *) GasVelThetaMed, (size_t)(nr)*sizeof(double),	3*nr*sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *) EnergyMed,       (size_t)(nr)*sizeof(double),	4*nr*sizeof(double), cudaMemcpyHostToDevice));
  
  // inner boundary
  if (where == INNER) {
    nb_block_y = (imaxi+1+BLOCK_Y-1)/BLOCK_Y;
    grid.y = nb_block_y;

    // damp density and velcoities
    if (strong) {
      kernel_strongdampbc_in <<< grid, block >>> (Vrad->gpu_field,
                                                  Vtheta->gpu_field,
                                                  Rho->gpu_field,
                                                  energy_gpu,
                                                  nr,
                                                  Vrad->pitch/sizeof(double),
                                                  //10*(dt/(2.0*M_PI*pow((double)RMIN,1.5)/1.0)),
                                                  DAMPING_STRENGTH_IN * dt/(2.0*M_PI*pow((double)RMIN,1.5)),
                                                  R_inf, 
                                                  (double)RMIN, 
                                                  (1.0/(RMIN-R_inf)));
    
      getLastCudaError ("kernel_strongdampbc_in failed");
      cudaThreadSynchronize();    
    }
    // damp only velocities
    else {
      kernel_dampbc_in <<< grid, block >>> (Vrad->gpu_field,
                                            Vtheta->gpu_field,
                                            nr,
                                            Vrad->pitch/sizeof(double),
                                            (dt/(2.0*M_PI*pow((double)RMIN,1.5)/1.0)),
                                            R_inf, 
                                            (double)RMIN, 
                                            (1.0/(RMIN-R_inf)));
  
      getLastCudaError ("kernel dampbc_in failed");
      cudaThreadSynchronize();
    }
  }

  // outer boundary
  if (where == OUTER) {
     nb_block_y = (nr-imino+BLOCK_Y-1)/BLOCK_Y;
     grid.y = nb_block_y;

    // damp density and velcoities
     if (strong) {
       kernel_strongdampbc_out <<< grid, block >>> (Vrad->gpu_field,
                                                    Vtheta->gpu_field,
                                                    Rho->gpu_field,
                                                    energy_gpu,
                                                    nr,
                                                    Vrad->pitch/sizeof(double),
                                                    imino,
                                                    DAMPING_STRENGTH_OUT * dt/(2.0*M_PI*pow((double)RMAX,1.5)),
                                                    R_sup, 
                                                    (double)RMAX, 
                                                    (1.0/(RMAX-R_sup)));

       getLastCudaError ("kernel_strongdampbc_out failed");
       cudaThreadSynchronize();
     }
     // damp only velocities
     else {
       kernel_dampbc_out <<< grid, block >>> (Vrad->gpu_field,
                                              Vtheta->gpu_field,
                                              nr,
                                              Vrad->pitch/sizeof(double),
                                              imino,
                                              (dt/(2.0*M_PI*pow((double)RMAX,1.5)/1.0)),
                                              R_sup, 
                                              (double)RMAX, 
                                              (1.0/(RMAX-R_sup)));
     
       getLastCudaError ("kernel_dampbc_out failed");
       cudaThreadSynchronize();
    }
  }
}


extern "C" 
void StockholmBoundaryDust_gpu (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, int DustBin, double dt, int where) {
  int nr, ns, imin, imax, imed;
  static int FirstTime=YES, imaxi, imino;
  static double R_inf, R_sup;
  int nb_block_y;
  
  //R_inf = (double)RMIN*1.25;
  //R_sup = (double)RMAX*.84;
  
  nr = Vrad->Nrad;
  ns = Vrad->Nsec;

  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid;
  grid.x = (ns+block.x-1)/block.x;

  static double *SigmaMedDust, *VelRadMedDust, *VelThetaMedDust;
  if (FirstTime) {  

    // store initial dust density and velocity components
    SigmaMedDust    = (double*) malloc (NRAD*DustBinNum*sizeof (double));
    VelRadMedDust   = (double*) malloc (NRAD*DustBinNum*sizeof (double));
    VelThetaMedDust = (double*) malloc (NRAD*DustBinNum*sizeof (double));
    for (int ii=0; ii< DustBinNum; ii++) {
      for (int i=0; i<NRAD; i++) {
        SigmaMedDust[ii*NRAD+i]    = dust_density[ii]->Field[i*NSEC];
        VelRadMedDust[ii*NRAD+i]   = dust_v_rad[ii]->Field[i*NSEC];
        VelThetaMedDust[ii*NRAD+i] = dust_v_theta[ii]->Field[i*NSEC];
      }
    }
    
    // set inner outer bondary for wave damping
    R_inf = (double) RMIN*DAMPRMIN;
    R_sup = (double) RMAX*DAMPRMAX;

    // Dichotomic search of starting index of outer damping BC (on host)
    imin=0;
    imax=nr-1;
    while (imax-imin > 1) {
      imed = (imax+imin)/2;
      if (Rmed[imed] > R_sup)
        imax=imed;
      else
        imin=imed;
    }
    imino = imin;
    
    // Dichotomic search of ending index of inner damping BC (on host)
    imin=0;
    imax=nr-1;
    while (imax-imin > 1) {
      imed = (imax+imin)/2;
      if (Rmed[imed] < R_inf)
        imin=imed;
      else
        imax=imed;
    }
    imaxi = imax;
    FirstTime = NO;
  }

  // uppload necessary RadiiStaff and density, velocity components
  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *) (RadiiStuff+6*(nr+1)),            (size_t)(nr)*sizeof(double),	                    0, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *) (SigmaMedDust+DustBin    * NRAD), (size_t)(nr)*sizeof(double),	  NRAD*sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *) (VelRadMedDust+DustBin   * NRAD), (size_t)(nr)*sizeof(double),	2*NRAD*sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *) (VelThetaMedDust+DustBin * NRAD), (size_t)(nr)*sizeof(double),	3*NRAD*sizeof(double), cudaMemcpyHostToDevice));
  
  // inner boundary
  if (where == INNER) {
    nb_block_y = (imaxi+1+BLOCK_Y-1)/BLOCK_Y;
    grid.y = nb_block_y;

    kernel_strongdampbc_in <<< grid, block >>> (Vrad->gpu_field,
                                                Vtheta->gpu_field,
                                                Rho->gpu_field,
                                                NULL,
                                                nr,
                                                Vrad->pitch/sizeof(double),
                                                //10*(dt/(2.0*M_PI*pow((double)RMIN,1.5)/1.0)),
                                                DAMPING_STRENGTH_IN * dt/(2.0*M_PI*pow((double)RMIN,1.5)),
                                                R_inf, 
                                                (double)RMIN, 
                                                (1.0/(RMIN-R_inf)));
    
    getLastCudaError ("kernel_strongdampbc_in failed");
    cudaThreadSynchronize();    
  }

  // outer boundary
  if (where == OUTER) {
     nb_block_y = (nr-imino+BLOCK_Y-1)/BLOCK_Y;
     grid.y = nb_block_y;

    // damp density and velcoities
     kernel_strongdampbc_out <<< grid, block >>> (Vrad->gpu_field,
                                                  Vtheta->gpu_field,
                                                  Rho->gpu_field,
                                                  NULL,
                                                  nr,
                                                  Vrad->pitch/sizeof(double),
                                                  imino,
                                                  DAMPING_STRENGTH_OUT * dt/(2.0*M_PI*pow((double)RMAX,1.5)),
                                                  R_sup, 
                                                  (double)RMAX, 
                                                  (1.0/(RMAX-R_sup)));

     getLastCudaError ("kernel_strongdampbc_out failed");
     cudaThreadSynchronize();
   }
}

