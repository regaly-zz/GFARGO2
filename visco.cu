/** \file "template.cu" : implements the kernel for the "template" procedure
*/
#define __CUDA 1
#include "fargo.h"
#undef __CUDA
#include <stdarg.h>
#include <helper_cuda.h>
#include <cuda.h>

// BLOCK_X : in azimuth
//#define BLOCK_X DEF_BLOCK_X_VISCO
#define BLOCK_X 64
// BLOCK_Y : in radius
#define BLOCK_Y 4

#define  invdiffrmed  CRadiiStuff[           igp]
#define  cs2          CRadiiStuff[(nr+1)*1 + igp]
#define  invrmed      CRadiiStuff[(nr+1)*2 + igp]
#define  invrmedm     CRadiiStuff[(nr+1)*2 + igp-1]
#define  invrinf      CRadiiStuff[(nr+1)*3 + igp]
#define  rinf         CRadiiStuff[(nr+1)*4 + igp]
#define  rmed         CRadiiStuff[(nr+1)*6 + igp]
#define  rmedm        CRadiiStuff[(nr+1)*6 + igp-1]
#define  rsup         CRadiiStuff[(nr+1)*8 + igp]
#define  invdiffrsup  CRadiiStuff[(nr+1)*10+ igp]
#define  visco        CRadiiStuff[(nr+1)*12+ igp]
#define  alphaval     CRadiiStuff[(nr+1)*13+ igp]
#define  omega        CRadiiStuff[(nr+1)*14+ igp]

#define GET_TAB(u,x,y,pitch) *(u + __mul24(y, pitch) + x)

// [RZS-MOD]
extern double SGAccInnerEdge, SGAccOuterEdge;

//__constant__ double CRadiiStuff[8192];
__device__ double CRadiiStuff[32768];


// calcualte adiabatic alpha viscosity value at each cell
__global__ void kernel_calc_alpha_visco (double *dens,
                                         double *energy,
                                         double *viscosity,
                                         double  alpha,
                                         double  adiabatic_index,
                                         int     ns, 
                                         int     nr, 
                                         int     pitch) {

  const int jg = threadIdx.x + blockIdx.x * blockDim.x;
  const int ig = threadIdx.y + blockIdx.y * blockDim.y;
  const int idg = jg+ig*pitch;
  const int igp = ig;

  const double csa2 = adiabatic_index*(adiabatic_index-1.0)*energy[idg]/dens[idg];
//  viscosity[idg] = alpha * csa2 * pow(rmed, 1.5); // aplha*cs^2/Omega
  viscosity[idg] = alpha * csa2 / omega;
}


// calcualte adiabatic alpha viscosity value at each cell
__global__ void kernel_calc_dze_alpha_visco (double *dens,
                                             double *energy,
                                             double *viscosity,
                                             double  adiabatic_index,
                                             double  viscmod,
                                             double  viscmodr1,
                                             double  viscmoddeltar1,
                                             double  viscmodr2,
                                             double  viscmoddeltar2,
                                             int     ns, 
                                             int     nr, 
                                             int     pitch) {

  const int jg = threadIdx.x + blockIdx.x * blockDim.x;
  const int ig = threadIdx.y + blockIdx.y * blockDim.y;
  const int idg = jg+ig*pitch;
  const int igp = ig;

  const double csa2 = adiabatic_index*(adiabatic_index-1.0)*energy[idg]/dens[idg];
  //viscosity[idg] = alphaval * csa2 * pow(rmed, 1.5); // aplha*cs^2/Omega
  viscosity[idg] = alphaval * csa2 / omega; // aplha*cs^2/Omega
}

// calcualte density dependent alpha viscosity value at each cell
__global__ void kernel_calc_adaptive_alpha_visco (double *dens,
                                                  double *energy,
                                                  double *viscosity,
                                                  double  alpha_active,
                                                  double  alpha_dead,
                                                  double  alpha_smooth,
                                                  double  sigma_thresh,
                                                  double  adiabatic_index,
                                                  double  aspect_ratio,
                                                  int     ns, 
                                                  int     nr, 
                                                  int     pitch,
                                                  bool    adiabatic) {

  const int jg = threadIdx.x + blockIdx.x * blockDim.x;
  const int ig = threadIdx.y + blockIdx.y * blockDim.y;
  const int m = jg+ig*pitch;
  const int igp = ig;
  
  if (adiabatic) {
    const double rho = dens[m];
    const double alpha = (1.0-tanh ((rho-sigma_thresh) / (sigma_thresh * alpha_smooth * aspect_ratio))) * alpha_active * 0.5 + alpha_dead;
    const double mycs2 = adiabatic_index*(adiabatic_index-1.0)*energy[m]/rho;
    //viscosity[m] = alpha * csa2 * pow(rmed, 1.5); //   aplha*cs^2/Omega
    viscosity[m] = alpha * mycs2 / omega; // aplha*cs^2/Omega
  }
  else {
	const double alpha = (1.0-tanh ((dens[m]-sigma_thresh) / (sigma_thresh * alpha_smooth * aspect_ratio))) * alpha_active * 0.5 + alpha_dead;
	viscosity[m] = alpha * cs2 * pow(rmed, 1.5); // aplha*cs^2/Omega    
    /*const double rho = dens[m];
    const double sigma_dead = rho-sigma_thresh;

    //const double alpha = (sigma_dead > 0 ? (alpha_active*sigma_thresh+sigma_dead*alpha_dead)/rho : alpha_active);
    const double alpha = (sigma_dead > 0 ? alpha_dead+alpha_active*exp(1-rho/sigma_thresh) : alpha_active);

    viscosity[m] = alpha * cs2 / omega; // aplha*cs^2/Omega    */
  }
}


// locally isothermal non-adaptive viscosity (density independent)
__global__ void kernel_visco2d (double *vrad,
                                double *vtheta,
                                double *vradnew,
                                double *vthetanew,
                                double *dens,
                                double *viscosity,
                                double *tau_rr,
                                double *tau_rp,
                                double *tau_pp,
                                int     ns, 
                                int     nr, 
                                int     pitch,
                                double  invdphi, 
                                double  dt,
                                double  vtheta_in, 
                                double  vtheta_out,
                                bool    viscosity2d,
                                bool    visc_heating) {

  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;
  int js = threadIdx.x + 1;
  int is = threadIdx.y + 1;
  int jgp = jg+1;
  if (jg == ns-1) jgp = 0;
  int jgm = jg-1;
  if (jg == 0) jgm = ns-1;
  int idg = __mul24(ig, pitch) + jg;
  int ids = __mul24(is, blockDim.x+2) + js;
  int lim, l, lip, ils, igp;

  __shared__ double Trr[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double Tpp[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double Trp[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double div_v[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double rho[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double vr[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double vt[(BLOCK_X+2)*(BLOCK_Y+2)];

  // first get viscosity
  double nu;
  
  // We perform a coalesced read of 'rho', 'vr' and 'vtheta" into the shared memory;
  rho[ids] = dens[idg];
  vr[ids]  = vrad[idg];
  vt[ids]  = vtheta[idg];
  // Some necessary exceptions on the edges:

  // EDGE 1 : "LEFT EDGE"
  if ((is == 2) && (js <= blockDim.y)) {
    // read by second row...
    int it = ig-2+js;
    int jt = jg-js;
    if (jt < 0) jt += ns;
    ils = js*(blockDim.x+2);
    jt = jt+__mul24(it,pitch);
    rho[ils] = dens[jt];
    vr[ils]  = vrad[jt];
    vt[ils]  = vtheta[jt];
  }

  // EDGE 2: "RIGHT EDGE".
  // read by third row...
  if ((is ==3) && (js <= blockDim.y)) {
    int it = ig-3+js;
    int jt = jg-js + blockDim.x+1;
    if (jt > ns-1) jt -= ns;
    ils  = js*(blockDim.x+2)+blockDim.x+1;
    jt = jt+__mul24(it,pitch);
    rho[ils] = dens[jt];
    vr[ils]  = vrad[jt];
    vt[ils]  = vtheta[jt];
  }
  
  // EDGE 3: "BOTTOM EDGE". Be careful not to read anything if in first row...
  if ((is == 1) && (ig > 0)) {
    rho[js] = dens[idg-(int)pitch];
    vr[js]  = vrad[idg-(int)pitch];
    vt[js]  = vtheta[idg-(int)pitch];
  }
  //  EDGE 4: "TOP EDGE". Be careful not to read anything if in last row...
  if ((is == blockDim.y) && (ig < nr-1)) {
    rho[ids+blockDim.x+2] = dens[idg+(int)pitch];
    vr[ids+blockDim.x+2]  = vrad[idg+(int)pitch];
    vt[ids+blockDim.x+2]  = vtheta[idg+(int)pitch];
  }
  if ((is == blockDim.y) && (ig == nr-1)) {
    vr[ids+blockDim.x+2]  = 0.0;
    vt[ids+blockDim.x+2]  = 0.0;
    rho[ids+blockDim.x+2] = 0.0;
  }
  // And now some corners... "Bottom-left" first;
  if ((ig > 0) && (is == 1) && (js == 1)) {
    rho[0] = GET_TAB (dens,   jgm, ig-1, pitch);
    vr[0]  = GET_TAB (vrad,   jgm, ig-1, pitch);
    vt[0]  = GET_TAB (vtheta, jgm, ig-1, pitch);
  }
  // now bottom-right
  if ((ig > 0) && (is == 1) && (js == blockDim.x)) {
    rho[blockDim.x+1] = GET_TAB (dens,   jgp, ig-1, pitch);
    vr[blockDim.x+1]  = GET_TAB (vrad,   jgp, ig-1, pitch);
    vt[blockDim.x+1]  = GET_TAB (vtheta, jgp, ig-1, pitch);
  }
  // now "top-left"... top-right is not needed
  if ((ig < nr-1) && (is == blockDim.y) && (js == 1)) {
    rho[ids+blockDim.x+1] = GET_TAB (dens,   jgm, ig+1, pitch);
    vr[ids+blockDim.x+1]  = GET_TAB (vrad,   jgm, ig+1, pitch);
    vt[ids+blockDim.x+1]  = GET_TAB (vtheta, jgm, ig+1, pitch);
  }

  __syncthreads ();
  
  igp = ig;

  l = ids;
  lip = l + blockDim.x+2;
  lim = l - blockDim.x-2;

  Trr[l] = (vr[lip]-vr[l])*invdiffrsup;
  Tpp[l] = ((vt[l+1]-vt[l])*invdphi+0.5*(vr[lip]+vr[l]))*invrmed;
  div_v[l] = (vr[lip]*rsup-vr[l]*rinf)*invdiffrsup;
  div_v[l] += (vt[l+1]-vt[l])*invdphi;
  div_v[l] *= invrmed;
  if (ig > 0)
    Trp[l] = 0.5*(rinf*((vt[l]+1.0/sqrt(rmed))*invrmed-(vt[lim]+1.0/sqrt(rmedm))*invrmedm)*invdiffrmed+(vr[l]-vr[l-1])*invdphi*invrinf);
  else
    Trp[l] = 0.0;

  if (viscosity2d) {
    nu = viscosity[idg];
    //divergence_vel[idg] = div_v[l];
  }
  else
    nu = visco;

  Trr[l] = 2.0*rho[l]*nu*(Trr[l]-(1.0/3.0)*div_v[l]);
  Tpp[l] = 2.0*rho[l]*nu*(Tpp[l]-(1.0/3.0)*div_v[l]);
  Trp[l] = 0.5*(rho[l]+rho[l-1]+rho[lim]+rho[lim-1])*nu*Trp[l];

  // We need Trr & Tpp in bottom row
  if ((ig > 0) && (is == 1)) {
    igp = ig-1;
    
    l = ids-blockDim.x-2;
    lip = l + blockDim.x+2;
    lim = l - blockDim.x-2;
    
    Trr[l] = (vr[lip]-vr[l])*invdiffrsup;
    Tpp[l] = ((vt[l+1]-vt[l])*invdphi+0.5*(vr[lip]+vr[l]))*invrmed;
    div_v[l] = (vr[lip]*rsup-vr[l]*rinf)*invdiffrsup;
    div_v[l] += (vt[l+1]-vt[l])*invdphi;
    div_v[l] *= invrmed;

    if (viscosity2d) {
      nu = viscosity[idg-ns];
      //divergence_vel[idg] = div_v[l];
    }
    else
      nu = visco;


    Trr[l] = 2.0*rho[l]*nu*(Trr[l]-(1.0/3.0)*div_v[l]);
    Tpp[l] = 2.0*rho[l]*nu*(Tpp[l]-(1.0/3.0)*div_v[l]);
  }
  
  // We need Tpp in left column
  if (js == 1) {
    igp = ig;
    
    l = ids-1;
    lip = l + blockDim.x+2;
    lim = l - blockDim.x-2;
    
    Tpp[l] = ((vt[l+1]-vt[l])*invdphi+0.5*(vr[lip]+vr[l]))*invrmed;
    div_v[l] = (vr[lip]*rsup-vr[l]*rinf)*invdiffrsup;
    div_v[l] += (vt[l+1]-vt[l])*invdphi;
    div_v[l] *= invrmed;
    
    if (viscosity2d) {
      nu = viscosity[idg];
      //divergence_vel[idg] = div_v[l];
    }
    else
      nu = visco;
    
    Tpp[l] = 2.0*rho[l]*nu*(Tpp[l]-(1.0/3.0)*div_v[l]);
  }

  // We need Trp in right column and in top row. Top row first
  if ((ig < nr-1) && (is == blockDim.y)) {
    igp = ig+1;
    
    l = ids+blockDim.x+2;
    lip = l + blockDim.x+2;
    lim = l - blockDim.x-2;

    if (viscosity2d)
      nu = viscosity[idg+ns];
    else
      nu = visco;
    
    Trp[l] = 0.5*(rinf*((vt[l]+1.0/sqrt(rmed))*invrmed-(vt[lim]+1.0/sqrt(rmedm))*invrmedm)*invdiffrmed+(vr[l]-vr[l-1])*invdphi*invrinf);
    Trp[l] = 0.5*(rho[l]+rho[l-1]+rho[lim]+rho[lim-1])*nu*Trp[l];
  }
  // And now right column
  if (js == blockDim.x) {
    igp = ig;
    
    l = ids+1;
    lip = l + blockDim.x+2;
    lim = l - blockDim.x-2;

    if (viscosity2d)
      nu = viscosity[idg];
    else
      nu = visco;
    
    if (ig > 0)
      Trp[l] = 0.5*(rinf*((vt[l]+1.0/sqrt(rmed))*invrmed-(vt[lim]+1.0/sqrt(rmedm))*invrmedm)*invdiffrmed+(vr[l]-vr[l-1])*invdphi*invrinf);
    else
      Trp[l] = 0.0;
    Trp[l] = 0.5*(rho[l]+rho[l-1]+rho[lim]+rho[lim-1])*nu*Trp[l];
  }

  __syncthreads ();

  igp = ig;

  l = ids;
  lip = l + blockDim.x+2;
  lim = l - blockDim.x-2;

  if ((ig > 0) && (ig < nr-1)) {
    vthetanew[idg] = vt[l] + dt*invrmed*((rsup*Trp[lip]-rinf*Trp[l])*invdiffrsup+(Tpp[l]-Tpp[l-1])*invdphi+0.5*(Trp[l]+Trp[lip]))/(0.5*(rho[l]+rho[l-1]));
  }
  if (ig > 0) {
    vradnew[idg] = vr[l] + dt*invrinf*((rmed*Trr[l]-rmedm*Trr[lim])*invdiffrmed+(Trp[l+1]-Trp[l])*invdphi-0.5*(Tpp[l]+Tpp[lim]))/(0.5*(rho[l]+rho[lim]));
  } 
  else {
    vradnew[idg] = 0.0;
  }
  
  if (ig == 0)
    vthetanew[idg] = vtheta_in;
  if (ig == nr-1)
    vthetanew[idg] = vtheta_out;  

  if (visc_heating) {
    // for adiabtic disk we need to store Divergence, TauRR, TauRP, and TauPP
    tau_rr[idg] = Trr[l];
    tau_rp[idg] = Trp[l];
    tau_pp[idg] = Tpp[l];
  }
}


extern "C" 
void ViscousTerms_gpu (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, PolarGrid *Energy, double dt,
                       PolarGrid *Vrad_ret, PolarGrid *Vtheta_ret) {
  
  int nr, ns;
//  double Vtheta_In, Vtheta_Out, OmegaIn, OmegaOut;

  nr = Vrad->Nrad;
  ns = Vrad->Nsec;
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid ((ns+block.x-1)/block.x, (nr+block.y-1)/block.y);
  
  double *Energy_gpu_field = NULL;
  if (Adiabatic) {
    Energy_gpu_field = Energy->gpu_field;
  }
  
  double *Viscosity_gpu_field = NULL;
  if (Adiabatic || AdaptiveViscosity)
    Viscosity_gpu_field = Viscosity->gpu_field;

  double *TauRR_gpu_field = NULL, *TauRP_gpu_field = NULL, *TauPP_gpu_field = NULL; 
  if (ViscHeating) {
    TauRR_gpu_field = TauRR->gpu_field;
    TauRP_gpu_field = TauRP->gpu_field;
    TauPP_gpu_field = TauPP->gpu_field;
  }
  
  // calcualte viscosty
  // for constant kinematic voscosity 
  // adaptive alpha viscosity 
  if (AdaptiveViscosity) {
    checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)RadiiStuff, (size_t)(15*(nr+1))*sizeof(double),0, cudaMemcpyHostToDevice));
    kernel_calc_adaptive_alpha_visco <<< grid, block >>> (Rho->gpu_field,
                                                          Energy_gpu_field,
                                                          Viscosity_gpu_field,
                                                          ALPHAVISCOSITY,
                                                          ALPHAVISCOSITYDEAD,
                                                          ALPHASMOOTH,
                                                          ALPHASIGMATHRESH,
                                                          ADIABATICINDEX,
                                                          ASPECTRATIO,
                                                          ns, 
                                                          nr, 
                                                          Viscosity->pitch/sizeof(double),
                                                          Adiabatic);
  
    cudaThreadSynchronize();
    getLastCudaError ("kernel_calc_adaptive_alpha_visco failed");    
  } 
  // alpha viscosity with stationary dead zone
  else if (ViscosityAlpha && Adiabatic && DeadZone) {
    checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)RadiiStuff, (size_t)(15*(nr+1))*sizeof(double),0, cudaMemcpyHostToDevice));
     kernel_calc_dze_alpha_visco <<< grid, block >>> (Rho->gpu_field,
                                                      Energy_gpu_field,
                                                      Viscosity_gpu_field,
                                                      ADIABATICINDEX,
                                                      DEADZONEALPHA,
                                                      DEADZONERIN,
                                                      DEADZONEDELTARIN,
                                                      DEADZONEROUT,
                                                      DEADZONEDELTAROUT,
                                                      ns, 
                                                      nr, 
                                                      Viscosity->pitch/sizeof(double));
   
    cudaThreadSynchronize();
    getLastCudaError ("kernel_calc_dze_alpha_visco failed");    
  }
  // pure alpha voscosity with adiabatic gas eos
  else if (ViscosityAlpha && Adiabatic) {
    checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)RadiiStuff, (size_t)(15*(nr+1))*sizeof(double),0, cudaMemcpyHostToDevice));
    kernel_calc_alpha_visco <<< grid, block >>> (Rho->gpu_field,
                                                 Energy_gpu_field,
                                                 Viscosity_gpu_field,
                                                 ALPHAVISCOSITY,
                                                 ADIABATICINDEX,
                                                 ns, 
                                                 nr, 
                                                 Viscosity->pitch/sizeof(double));
     
    cudaThreadSynchronize();
    getLastCudaError ("kernel_calc_alpha_visco failed");            
  }
  
  // now we can calcuate viscous terms
  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)RadiiStuff, (size_t)(15*(nr+1))*sizeof(double),0, cudaMemcpyHostToDevice));
  kernel_visco2d <<< grid, block >>> (Vrad->gpu_field,
                                      Vtheta->gpu_field,
                                      tmp1->gpu_field,
                                      tmp2->gpu_field,
                                      Rho->gpu_field,
                                      Viscosity_gpu_field,
                                      TauRR_gpu_field,
                                      TauRP_gpu_field,
                                      TauPP_gpu_field,
                                      Rho->Nsec, 
                                      Rho->Nrad, 
                                      Rho->pitch/sizeof(double),
                                      (double)(Rho->Nsec)/2.0/M_PI,
                                      dt,
                                      GasVelThetaMed[0], 
                                      GasVelThetaMed[nr-1],
                                      (Adiabatic || AdaptiveViscosity),
                                      ViscHeating);
  
  cudaThreadSynchronize();
  getLastCudaError ("kernel_visco2d failed"); 


  FARGO_SAFE(ActualiseGas_gpu (Vrad_ret, tmp1));
  FARGO_SAFE(ActualiseGas_gpu (Vtheta_ret, tmp2));


  // HM
  //double *temp;
//  temp = Vrad->gpu_field;
//  Vrad_ret->gpu_field = tmp1->gpu_field;
//  VradNew->gpu_field = temp;

//  temp = Vtheta->gpu_field;
//  Vtheta_ret->gpu_field = tmp2->gpu_field;
//  VthetaNew->gpu_field = temp;
}



// locally isothermal non-adaptive viscosity (density independent)
__global__ void kernel_visco1d (double *vrad,
                                double *vtheta,
                                double *vradnew,
                                double *vthetanew,
                                double *dens,
                                int     ns, 
                                int     nr, 
                                int     pitch,
                                double  invdphi, 
                                double  dt,
                                double  vtheta_in,
                                double  vtheta_out) {

  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;
  int js = threadIdx.x + 1;
  int is = threadIdx.y + 1;
  int jgp = jg+1;
  if (jg == ns-1) jgp = 0;
  int jgm = jg-1;
  if (jg == 0) jgm = ns-1;
  int idg = __mul24(ig, pitch) + jg;
  int ids = __mul24(is, blockDim.x+2) + js;
  int lim, l, lip, ils, igp;

  __shared__ double Trr[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double Tpp[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double Trp[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double div_v[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double rho[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double vr[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double vt[(BLOCK_X+2)*(BLOCK_Y+2)];

  // first get viscosity
  double nu=1e-6;
  
  // We perform a coalesced read of 'rho', 'vr' and 'vtheta" into the shared memory;
  rho[ids] = dens[idg];
  vr[ids]  = vrad[idg];
  vt[ids]  = vtheta[idg];
  // Some necessary exceptions on the edges:

  // EDGE 1 : "LEFT EDGE"
  if ((is == 2) && (js <= blockDim.y)) {
    // read by second row...
    int it = ig-2+js;
    int jt = jg-js;
    if (jt < 0) jt += ns;
    ils = js*(blockDim.x+2);
    jt = jt+__mul24(it,pitch);
    rho[ils] = dens[jt];
    vr[ils]  = vrad[jt];
    vt[ils]  = vtheta[jt];
  }

  // EDGE 2: "RIGHT EDGE".
  // read by third row...
  if ((is ==3) && (js <= blockDim.y)) {
    int it = ig-3+js;
    int jt = jg-js + blockDim.x+1;
    if (jt > ns-1) jt -= ns;
    ils  = js*(blockDim.x+2)+blockDim.x+1;
    jt = jt+__mul24(it,pitch);
    rho[ils] = dens[jt];
    vr[ils]  = vrad[jt];
    vt[ils]  = vtheta[jt];
  }
  
  // EDGE 3: "BOTTOM EDGE". Be careful not to read anything if in first row...
  if ((is == 1) && (ig > 0)) {
    rho[js] = dens[idg-(int)pitch];
    vr[js]  = vrad[idg-(int)pitch];
    vt[js]  = vtheta[idg-(int)pitch];
  }
  //  EDGE 4: "TOP EDGE". Be careful not to read anything if in last row...
  if ((is == blockDim.y) && (ig < nr-1)) {
    rho[ids+blockDim.x+2] = dens[idg+(int)pitch];
    vr[ids+blockDim.x+2]  = vrad[idg+(int)pitch];
    vt[ids+blockDim.x+2]  = vtheta[idg+(int)pitch];
  }
  if ((is == blockDim.y) && (ig == nr-1)) {
    vr[ids+blockDim.x+2]  = 0.0;
    vt[ids+blockDim.x+2]  = 0.0;
    rho[ids+blockDim.x+2] = 0.0;
  }
  // And now some corners... "Bottom-left" first;
  if ((ig > 0) && (is == 1) && (js == 1)) {
    rho[0] = GET_TAB (dens,   jgm, ig-1, pitch);
    vr[0]  = GET_TAB (vrad,   jgm, ig-1, pitch);
    vt[0]  = GET_TAB (vtheta, jgm, ig-1, pitch);
  }
  // now bottom-right
  if ((ig > 0) && (is == 1) && (js == blockDim.x)) {
    rho[blockDim.x+1] = GET_TAB (dens,   jgp, ig-1, pitch);
    vr[blockDim.x+1]  = GET_TAB (vrad,   jgp, ig-1, pitch);
    vt[blockDim.x+1]  = GET_TAB (vtheta, jgp, ig-1, pitch);
  }
  // now "top-left"... top-right is not needed
  if ((ig < nr-1) && (is == blockDim.y) && (js == 1)) {
    rho[ids+blockDim.x+1] = GET_TAB (dens,   jgm, ig+1, pitch);
    vr[ids+blockDim.x+1]  = GET_TAB (vrad,   jgm, ig+1, pitch);
    vt[ids+blockDim.x+1]  = GET_TAB (vtheta, jgm, ig+1, pitch);
  }

  __syncthreads ();
  
  igp = ig;

  l = ids;
  lip = l + blockDim.x+2;
  lim = l - blockDim.x-2;

  //nu = 1e-7;
  
  Trr[l] = (vr[lip]-vr[l])*invdiffrsup;
  Tpp[l] = ((vt[l+1]-vt[l])*invdphi+0.5*(vr[lip]+vr[l]))*invrmed;
  div_v[l] = (vr[lip]*rsup-vr[l]*rinf)*invdiffrsup;
  div_v[l] += (vt[l+1]-vt[l])*invdphi;
  div_v[l] *= invrmed;
  if (ig > 0)
    Trp[l] = 0.5*(rinf*((vt[l]+1.0/sqrt(rmed))*invrmed-(vt[lim]+1.0/sqrt(rmedm))*invrmedm)*invdiffrmed+(vr[l]-vr[l-1])*invdphi*invrinf);
  else
    Trp[l] = 0.0;

  Trr[l] = 2.0*rho[l]*nu*(Trr[l]-(1.0/3.0)*div_v[l]);
  Tpp[l] = 2.0*rho[l]*nu*(Tpp[l]-(1.0/3.0)*div_v[l]);
  Trp[l] = 0.5*(rho[l]+rho[l-1]+rho[lim]+rho[lim-1])*nu*Trp[l];

  // We need Trr & Tpp in bottom row
  if ((ig > 0) && (is == 1)) {
    igp = ig-1;
    
    l = ids-blockDim.x-2;
    lip = l + blockDim.x+2;
    lim = l - blockDim.x-2;
    
    //nu = 1e-7;
    
    Trr[l] = (vr[lip]-vr[l])*invdiffrsup;
    Tpp[l] = ((vt[l+1]-vt[l])*invdphi+0.5*(vr[lip]+vr[l]))*invrmed;
    div_v[l] = (vr[lip]*rsup-vr[l]*rinf)*invdiffrsup;
    div_v[l] += (vt[l+1]-vt[l])*invdphi;
    div_v[l] *= invrmed;

    Trr[l] = 2.0*rho[l]*nu*(Trr[l]-(1.0/3.0)*div_v[l]);
    Tpp[l] = 2.0*rho[l]*nu*(Tpp[l]-(1.0/3.0)*div_v[l]);
  }
  
  // We need Tpp in left column
  if (js == 1) {
    igp = ig;
    
    l = ids-1;
    lip = l + blockDim.x+2;
    lim = l - blockDim.x-2;
    
    //nu = 1e-7;
    
    Tpp[l] = ((vt[l+1]-vt[l])*invdphi+0.5*(vr[lip]+vr[l]))*invrmed;
    div_v[l] = (vr[lip]*rsup-vr[l]*rinf)*invdiffrsup;
    div_v[l] += (vt[l+1]-vt[l])*invdphi;
    div_v[l] *= invrmed;
    
    Tpp[l] = 2.0*rho[l]*nu*(Tpp[l]-(1.0/3.0)*div_v[l]);
  }

  // We need Trp in right column and in top row. Top row first
  if ((ig < nr-1) && (is == blockDim.y)) {
    igp = ig+1;
    
    l = ids+blockDim.x+2;
    lip = l + blockDim.x+2;
    lim = l - blockDim.x-2;

    //nu = 1e-7;

    Trp[l] = 0.5*(rinf*((vt[l]+1.0/sqrt(rmed))*invrmed-(vt[lim]+1.0/sqrt(rmedm))*invrmedm)*invdiffrmed+(vr[l]-vr[l-1])*invdphi*invrinf);
    Trp[l] = 0.5*(rho[l]+rho[l-1]+rho[lim]+rho[lim-1])*nu*Trp[l];
  }
  // And now right column
  if (js == blockDim.x) {
    igp = ig;
    
    l = ids+1;
    lip = l + blockDim.x+2;
    lim = l - blockDim.x-2;

    //nu = 1e-7;
    
    if (ig > 0)
      Trp[l] = 0.5*(rinf*((vt[l]+1.0/sqrt(rmed))*invrmed-(vt[lim]+1.0/sqrt(rmedm))*invrmedm)*invdiffrmed+(vr[l]-vr[l-1])*invdphi*invrinf);
    else
      Trp[l] = 0.0;
    Trp[l] = 0.5*(rho[l]+rho[l-1]+rho[lim]+rho[lim-1])*nu*Trp[l];
  }

  __syncthreads ();

  igp = ig;

  l = ids;
  lip = l + blockDim.x+2;
  lim = l - blockDim.x-2;

  if ((ig > 1) && (ig < nr-3))
    vthetanew[idg] = vt[l] + dt*invrmed*((rsup*Trp[lip]-rinf*Trp[l])*invdiffrsup+(Tpp[l]-Tpp[l-1])*invdphi+0.5*(Trp[l]+Trp[lip]))/(0.5*(rho[l]+rho[l-1]));
  else
    vthetanew[idg] = 0;

  if (ig > 0)
    vradnew[idg] = vr[l] + dt*invrinf*((rmed*Trr[l]-rmedm*Trr[lim])*invdiffrmed+(Trp[l+1]-Trp[l])*invdphi-0.5*(Tpp[l]+Tpp[lim]))/(0.5*(rho[l]+rho[lim]));
  else
    vradnew[idg] = 0.0;
}




extern "C" 
void ViscousTermsDust_gpu (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, double dt,
                           PolarGrid *Vrad_ret, PolarGrid *Vtheta_ret) {

  int nr, ns;
  
  nr = Vrad->Nrad;
  ns = Vrad->Nsec;
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid ((ns+block.x-1)/block.x, (nr+block.y-1)/block.y);
  
  // now we can calcuate viscous terms
  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)RadiiStuff, (size_t)(15*(nr+1))*sizeof(double),0, cudaMemcpyHostToDevice));
  kernel_visco1d <<< grid, block >>> (Vrad->gpu_field,
                                      Vtheta->gpu_field,
                                      tmp1->gpu_field,
                                      tmp2->gpu_field,
                                      Rho->gpu_field,
                                      Rho->Nsec, 
                                      Rho->Nrad, 
                                      Rho->pitch/sizeof(double),
                                      (double)(Rho->Nsec)/2.0/M_PI,
                                      dt,
                                      GasVelThetaMed[0], 
                                      GasVelThetaMed[nr-1]);
  
  cudaThreadSynchronize();
  getLastCudaError ("kernel_visco1d failed"); 


  FARGO_SAFE(ActualiseGas_gpu (Vrad_ret, tmp1));
  FARGO_SAFE(ActualiseGas_gpu (Vtheta_ret, tmp2));


  // HM
  //double *temp;
//  temp = Vrad->gpu_field;
//  Vrad_ret->gpu_field = tmp1->gpu_field;
//  VradNew->gpu_field = temp;

//  temp = Vtheta->gpu_field;
//  Vtheta_ret->gpu_field = tmp2->gpu_field;
//  VthetaNew->gpu_field = temp;
}

