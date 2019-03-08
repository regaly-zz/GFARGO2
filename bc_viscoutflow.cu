/** \file bc_viscoutflow.cu: contains a CUDA kernel for viscous outflow inner and outer boundary condition.
*/
#define __CUDA 1
#include "fargo.h"
#undef __CUDA
#include <stdarg.h>
#include <helper_cuda.h>
#include <cuda.h>

// BLOCK_X : in azimuth
//#define BLOCK_X DEF_BLOCK_X_VISCOUTFLOW
#define BLOCK_X 16//64
#define BLOCK_Y 1


__global__ void kernel_viscoutflowbc_in (double      *vrad,
                                         double      *vtheta,
                                         double      *rho,
                                         const int    nr,
                                         const int    ns,
                                         const double Sigma0,
                                         const double Vrad0,
                                         const double Vtheta0) {

  const int jg = blockDim.x * blockIdx.x + threadIdx.x;
  
  rho[jg]    = Sigma0;  
  if (vrad[jg+ns+ns] > 0.0 || rho[jg+ns] < Sigma0)
    vrad[jg+ns] = 0.0;
  else
    vrad[jg+ns] = Vrad0;
  vtheta[jg] = Vtheta0;
}


__global__ void kernel_viscoutflowbc_out (double      *vrad,
                                          double      *vtheta,
                                          double      *rho,
                                          const int    nr,
                                          const int    ns,
                                          const double SigmaN,
                                          const double VradN,
                                          const double VthetaN) {

  const int jg = blockDim.x * blockIdx.x + threadIdx.x;

  rho[jg+(nr-1)*ns] = rho[jg+(nr-2)*ns];

  if (vrad[jg+(nr-2)*ns] < 0.0 || rho[jg+(nr-2)*ns] < SigmaN)
    vrad[jg+(nr-1)*ns] = 0.0;
  else
    vrad[jg+(nr-1)*ns] = VradN;
  vtheta[jg+(nr-1)*ns] = VthetaN;
}


extern "C" 
void ViscOutflow_gpu (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, int where) {

  int nr = Vrad->Nrad;
  int ns = Vrad->Nsec;
  
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid;
  grid.x = (ns+block.x-1)/block.x;

  // inner boundary 
  if (where == INNER) {
    int nb_block_y = (1+BLOCK_Y-1)/BLOCK_Y;
    grid.y = nb_block_y;

    // initial density keep constant
    double sigma0 = SigmaMed[0];

// self-gravity!!!!


    // viscous inflow into radial direction
    //double vrad0 = -3.0*FViscosity(Rmed[0])/Rmed[0]*(-SIGMASLOPE+0.5); 
    double vrad0;
    if (SIGMACUTOFFRADOUT == 0)
      vrad0 = -3.0*FViscosity(Rmed[0])/Rmed[0]*(-SIGMASLOPE+2.0*FLARINGINDEX+1.0);
    else
      vrad0 = -3.0*FViscosity(Rmed[0])/Rmed[0]*(-SIGMASLOPE+2.0*FLARINGINDEX+1.0+(SIGMASLOPE-2)*pow(Rmed[0]/SIGMACUTOFFRADOUT, 2-SIGMASLOPE)); 

    // should be!!!!!!: -3.0*FViscosity(Rmed[0])/Rmed[0]*(-SIGMASLOPE+1)

    // sub Keplerian velocity into azimuthal direction
    double vtheta0  = sqrt(G/DRmed[0]);
    if (SIGMACUTOFFRADOUT == 0)
      vtheta0 *= sqrt(1.0-pow(ASPECTRATIO,2.0)*pow(DRmed[0],2.0*FLARINGINDEX)*(1.0+SIGMASLOPE-2.0*FLARINGINDEX));
    else
      vtheta0 *= sqrt(1.0+pow(ASPECTRATIO,2.0)*(SIGMASLOPE-2)*pow(DRmed[0],2+2*FLARINGINDEX-SIGMASLOPE)*pow(SIGMACUTOFFRADOUT,SIGMASLOPE-2)- pow(ASPECTRATIO,2.0)*pow(DRmed[0],2*FLARINGINDEX)*(1+SIGMASLOPE-2*FLARINGINDEX));

    vtheta0 -= DInvSqrtRmed[0];//+Rmed[0]*OmegaFrame;

    kernel_viscoutflowbc_in <<< grid, block >>> (Vrad->gpu_field,
                                                 Vtheta->gpu_field,
                                                 Rho->gpu_field,
                                                 (int) NRAD,
                                                 (int) NSEC,
                                                 sigma0,
                                                 vrad0,
                                                 vtheta0);

    cudaThreadSynchronize();
    getLastCudaError ("kernel kernel_viscoutflowbc_in failed");
 }

  // outer boundary
  if (where == OUTER) {
    int nb_block_y = (1+BLOCK_Y-1)/BLOCK_Y;
    grid.y = nb_block_y;

    // initial density keep constant
    double sigmaN = SigmaMed[nr-1];

// self-gravity!!!!
    
    // viscous inflow into radial direction
    //double vradN = -3.0*FViscosity(Rmed[nr-1])/Rmed[nr-1]*(-SIGMASLOPE+0.5);
    double vradN;
    if (SIGMACUTOFFRADOUT == 0)
      vradN = -3.0*FViscosity(Rmed[nr-1])/Rmed[nr-1]*(-SIGMASLOPE+2.0*FLARINGINDEX+1.0);
    else
      vradN = -3.0*FViscosity(Rmed[nr-1])/Rmed[nr-1]*(-SIGMASLOPE+2.0*FLARINGINDEX+1.0+(SIGMASLOPE-2)*pow(Rmed[nr-1]/SIGMACUTOFFRADOUT, 2-SIGMASLOPE));     
    
    // sub Keplerian velocity into azimuthal direction
    double vthetaN  = sqrt(G/DRmed[nr-1]);
    if (SIGMACUTOFFRADOUT == 0)
      vthetaN *= sqrt(1.0-pow(ASPECTRATIO,2.0)*pow(DRmed[nr-1],2.0*FLARINGINDEX)*(1.0+SIGMASLOPE-2.0*FLARINGINDEX));
    else
      vthetaN *= sqrt(1.0+pow(ASPECTRATIO,2.0)*(SIGMASLOPE-2)*pow(DRmed[nr-1],2+2*FLARINGINDEX-SIGMASLOPE)*pow(SIGMACUTOFFRADOUT,SIGMASLOPE-2)- pow(ASPECTRATIO,2.0)*pow(DRmed[nr-1],2*FLARINGINDEX)*(1+SIGMASLOPE-2*FLARINGINDEX));
    vthetaN -= DInvSqrtRmed[nr-1];//+Rmed[nr-1]*OmegaFrame;;

    kernel_viscoutflowbc_out <<< grid, block >>> (Vrad->gpu_field,
                                                  Vtheta->gpu_field,
                                                  Rho->gpu_field,
                                                  (int)NRAD,
                                                  (int)NSEC,
                                                  sigmaN,
                                                  vradN,
                                                  vthetaN);

    cudaThreadSynchronize();
    getLastCudaError ("kernel kernel_viscoutflowbc_out failed");
 }
}
