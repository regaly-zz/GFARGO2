/** \file Substep2.cu : implements the kernel for the substep2 procedure
*/
#define __CUDA 1
#include "fargo.h"
#undef __CUDA
#include <stdarg.h>
#include <helper_cuda.h>
#include <cuda.h>

// BLOCK_X : in azimuth
//#define BLOCK_X DEF_BLOCK_X_SUBSTEP2
#define BLOCK_X 64
// BLOCK_Y : in radius
#define BLOCK_Y 4

//__constant__ double CRadiiStuff[8192];
__device__ double CRadiiStuff[32768];

#define GET_TAB(u,x,y,pitch) *(u + __mul24(y, pitch) + x)
//#define invdxtheta .159154943f * (double)ns * CRadiiStuff[(nr+1)*2+ig]
#define  invdiffrmed  CRadiiStuff[           ig]
#define  invdiffrsup  CRadiiStuff[(nr+1)*10+ ig]

__global__ void kernel_substep2 (const double *vr,
                                 const double *vt,
                                       double *vrnew,
                                       double *vtnew,
                                 const double *rho,
                                 const double *energy,
                                       double *energyint,
                                 const int     ns, 
                                 const int     nr, 
                                 const int     pitch, 
                                 const double  dt) {
                                   
  __shared__ double shared_qr[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double shared_vr[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double shared_qt[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double shared_vt[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double shared_rho[(BLOCK_X+2)*(BLOCK_Y+2)];
  int ids, idg, jgm, jgp, idsm, idsp;
  double dv, invdxtheta;
  //double dv;
  // jg & ig, g like 'global' (global memory <=> full grid)
  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;
  // js & is, l like 'local' (shared memory <=> local patch)
  int js = threadIdx.x + 1;
  int is = threadIdx.y + 1;
  ids = is*(blockDim.x+2)+js;
  idg = __mul24(ig, pitch) + jg;
  jgp = jg+1;
  jgm = jg-1;
  if (jg == ns-1) jgp -= pitch;
  if (jg == 0) jgm += pitch;
  
  // We perform a coalesced read of 'rho' into the shared memory;
  shared_rho[ids] = rho[idg];
  
  // EDGE 1 : "LEFT EDGE"
  if (js == 1) {
    shared_rho[ids-1] = GET_TAB (rho, jgm, ig, pitch);
    //shared_energy[ids-1] = GET_TAB (energy, jgm, ig, pitch);
  }
  // EDGE 3: "BOTTOM EDGE". Be careful not to read anything if in first row...
  if ((is == 1) && (ig > 0)) {
    shared_rho[js] = rho[idg-pitch];
    //shared_energy[js] = energy[idg-pitch];
  }

  if (ig == 0) {
    shared_rho[js] = rho[idg+pitch]; // null gradient of rho at inner edge
    //shared_energy[js] = energy[idg+pitch];
  }
  // We perform a coalesced read of 'vrad' into the shared memory;
  shared_vr[ids] = vr[idg];
  
  // EDGE 3: "BOTTOM EDGE". Be careful not to read anything if in first row...
  if ((is == 2) && (ig > 1)) {
    shared_vr[js] = vr[idg-(pitch<<1)];
  }
  if (ig == 0)
    shared_vr[js] = 0.0;
  
  // EDGE 4: "TOP EDGE". Be careful not to read anything if in last row...
  if (ig == nr-1)
    shared_vr[js+( blockDim.y+1)*(blockDim.x+2)] = 0.0;
  if ((is == blockDim.y) && (ig < nr-1))
    shared_vr[js+(blockDim.y+1)*(blockDim.x+2)] = GET_TAB (vr, jg, ig+1, pitch);
  
  // We perform a coalesced read of 'vt' into the shared memory;
  shared_vt[ids] = vt[idg];
    
  // Some necessary exceptions on the edges:
  
  // EDGE 1 : "LEFT EDGE"
  if (js == 1) 
    shared_vt[ids-1] = GET_TAB (vt, jgm, ig, pitch);
  
  // EDGE 2: "RIGHT EDGE"
  if (js == blockDim.x)
    shared_vt[is*(blockDim.x+2)+blockDim.x+1] = GET_TAB (vt, jgp, ig, pitch);
  
  __syncthreads ();

  idsm = ids-blockDim.x-2;
  idsp = ids+blockDim.x+2;
//  invdxtheta = .159154943 * (double)ns * CRadiiStuff[(nr+1)*2+ig];
  invdxtheta = (1.0/(2.0*M_PI)) * (double)ns * CRadiiStuff[(nr+1)*2+ig]; //invrmed
  
  if (ig == nr-1) 
    shared_vr[idsp] = 0.0;
  shared_qr[idsm] = 0.0;
  dv = shared_vr[ids]-shared_vr[idsm];
  if (dv < 0.0)
    shared_qr[idsm] = CVNR*CVNR*shared_rho[idsm]*dv*dv;

  if (is == blockDim.y) {
    shared_qr[ids] = 0.0;
    dv = shared_vr[idsp]-shared_vr[ids];
    if (dv < 0.0)
      shared_qr[ids] = CVNR*CVNR*shared_rho[ids]*dv*dv;
  }

  shared_qt[ids-1] = 0.0;
  dv = shared_vt[ids]-shared_vt[ids-1];
  if (dv < 0.0)
    shared_qt[ids-1] = CVNR*CVNR*shared_rho[ids-1]*dv*dv;
  if (js == blockDim.x) {
    shared_qt[ids] = 0.0;
    dv = shared_vt[ids+1]-shared_vt[ids];
    if (dv < 0.0)
      shared_qt[ids] = CVNR*CVNR*shared_rho[ids]*dv*dv;
  }
  
  __syncthreads ();

  // update velocity components
  vrnew[idg] = shared_vr[ids]-2.0*dt/(shared_rho[ids]+shared_rho[idsm])*(shared_qr[ids]-shared_qr[idsm])*invdiffrmed;
  vtnew[idg] = shared_vt[ids]-2.0*dt/(shared_rho[ids]+shared_rho[ids-1])*(shared_qt[ids]-shared_qt[ids-1])*invdxtheta;

  // add artifical viscosity to energy only if energy is given
  if (energy != NULL) {
    energyint[idg] =  energy[idg] 
                      - dt*shared_qr[ids]*(shared_vr[idsp]-shared_vr[ids])*invdiffrsup
                      - dt*shared_qt[ids]*(shared_vt[ids+1]-shared_vt[ids])*invdxtheta;
  }
}


void SubStep2_gpu (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, PolarGrid *Energy, double dt,
                   PolarGrid *Vrad_ret, PolarGrid *Vtheta_ret, PolarGrid *Energy_ret) {
  int nr, ns;
  nr = Rho->Nrad;
  ns = Rho->Nsec;
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid ((ns+block.x-1)/block.x, (nr+block.y-1)/block.y);

  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)RadiiStuff, (size_t)(14*(nr+1))*sizeof(double),	0, cudaMemcpyHostToDevice));

  double *Energy_gpu_field = NULL, *Energy_ret_gpu_field = NULL;
  if (Energy != NULL) {
    Energy_gpu_field = Energy->gpu_field;
    Energy_ret_gpu_field = Energy_ret->gpu_field;
  }
  
  kernel_substep2 <<< grid, block >>> (Vrad->gpu_field,
                                       Vtheta->gpu_field,
                                       Vrad_ret->gpu_field,
                                       Vtheta_ret->gpu_field,
                                       Rho->gpu_field,
                                       Energy_gpu_field,
                                       Energy_ret_gpu_field,
                                       ns, 
                                       nr,
                                       Rho->pitch/sizeof(double), 
                                       dt);
  cudaThreadSynchronize();
  getLastCudaError ("kernel_substep2 failed");
}

__global__ void kernel_substep2b (const double *vr,
                                  const double *vt,
                                        double *rho,
                                        //double *vrnew,
                                        //double *vtnew,
                                  const int     ns, 
                                  const int     nr, 
                                  const int     pitch, 
                                  const double  dt) {
                                   
  __shared__ double shared_qr[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double shared_vr[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double shared_qt[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double shared_vt[(BLOCK_X+2)*(BLOCK_Y+2)];
  __shared__ double shared_rho[(BLOCK_X+2)*(BLOCK_Y+2)];
  int ids, idg, jgm, jgp, idsm, idsp;
  double dv, invdxtheta;
  //double dv;
  // jg & ig, g like 'global' (global memory <=> full grid)
  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;
  // js & is, l like 'local' (shared memory <=> local patch)
  int js = threadIdx.x + 1;
  int is = threadIdx.y + 1;
  ids = is*(blockDim.x+2)+js;
  idg = __mul24(ig, pitch) + jg;
  jgp = jg+1;
  jgm = jg-1;
  if (jg == ns-1) jgp -= pitch;
  if (jg == 0) jgm += pitch;
  
  // We perform a coalesced read of 'rho' into the shared memory;
  shared_rho[ids] = rho[idg];
  
  // EDGE 1 : "LEFT EDGE"
  if (js == 1) {
    shared_rho[ids-1] = GET_TAB (rho, jgm, ig, pitch);
    //shared_energy[ids-1] = GET_TAB (energy, jgm, ig, pitch);
  }
  // EDGE 3: "BOTTOM EDGE". Be careful not to read anything if in first row...
  if ((is == 1) && (ig > 0)) {
    shared_rho[js] = rho[idg-pitch];
    //shared_energy[js] = energy[idg-pitch];
  }

  if (ig == 0) {
    shared_rho[js] = rho[idg+pitch]; // null gradient of rho at inner edge
    //shared_energy[js] = energy[idg+pitch];
  }
  // We perform a coalesced read of 'vrad' into the shared memory;
  shared_vr[ids] = vr[idg];
  
  // EDGE 3: "BOTTOM EDGE". Be careful not to read anything if in first row...
  if ((is == 2) && (ig > 1)) {
    shared_vr[js] = vr[idg-(pitch<<1)];
  }
  if (ig == 0)
    shared_vr[js] = 0.0;
  
  // EDGE 4: "TOP EDGE". Be careful not to read anything if in last row...
  if (ig == nr-1)
    shared_vr[js+( blockDim.y+1)*(blockDim.x+2)] = 0.0;
  if ((is == blockDim.y) && (ig < nr-1))
    shared_vr[js+(blockDim.y+1)*(blockDim.x+2)] = GET_TAB (vr, jg, ig+1, pitch);
  
  // We perform a coalesced read of 'vt' into the shared memory;
  shared_vt[ids] = vt[idg];
    
  // Some necessary exceptions on the edges:
  
  // EDGE 1 : "LEFT EDGE"
  if (js == 1) 
    shared_vt[ids-1] = GET_TAB (vt, jgm, ig, pitch);
  
  // EDGE 2: "RIGHT EDGE"
  if (js == blockDim.x)
    shared_vt[is*(blockDim.x+2)+blockDim.x+1] = GET_TAB (vt, jgp, ig, pitch);
  
  __syncthreads ();

  idsm = ids-blockDim.x-2;
  idsp = ids+blockDim.x+2;
//  invdxtheta = .159154943 * (double)ns * CRadiiStuff[(nr+1)*2+ig];
  invdxtheta = (1.0/(2.0*M_PI)) * (double)ns * CRadiiStuff[(nr+1)*2+ig]; //invrmed
  
  if (ig == nr-1) 
    shared_vr[idsp] = 0.0;
  shared_qr[idsm] = 0.0;
  dv = shared_vr[ids]-shared_vr[idsm];
  if (dv < 0.0)
    shared_qr[idsm] = 0.25*CVNR*CVNR*shared_rho[idsm]*dv*dv;

  if (is == blockDim.y) {
    shared_qr[ids] = 0.0;
    dv = shared_vr[idsp]-shared_vr[ids];
    if (dv < 0.0)
      shared_qr[ids] = 0.25*CVNR*CVNR*shared_rho[ids]*dv*dv;
  }

  shared_qt[ids-1] = 0.0;
  dv = shared_vt[ids]-shared_vt[ids-1];
  if (dv < 0.0)
    shared_qt[ids-1] = 0.25*CVNR*CVNR*shared_rho[ids-1]*dv*dv;
  if (js == blockDim.x) {
    shared_qt[ids] = 0.0;
    dv = shared_vt[ids+1]-shared_vt[ids];
    if (dv < 0.0)
      shared_qt[ids] = 0.25*CVNR*CVNR*shared_rho[ids]*dv*dv;
  }
  
  __syncthreads ();

  // update velocity components
 // vrnew[idg] = shared_vr[ids]-2.0*dt/(shared_rho[ids]+shared_rho[idsm])*(shared_qr[ids]-shared_qr[idsm])*invdiffrmed;
 // vtnew[idg] = shared_vt[ids]-2.0*dt/(shared_rho[ids]+shared_rho[ids-1])*(shared_qt[ids]-shared_qt[ids-1])*invdxtheta;

//  rho[idg] =  shared_rho[ids] 
//              - dt*shared_qr[ids]*(shared_vr[idsp]-shared_vr[ids])*invdiffrsup
//              - dt*shared_qt[ids]*(shared_vt[ids+1]-shared_vt[ids])*invdxtheta;
  
  rho[idg] =  shared_rho[ids] 
                  - dt*shared_qr[ids]*(shared_vr[idsp]-shared_vr[ids])*invdiffrsup
                  - dt*shared_qt[ids]*(shared_vt[ids+1]-shared_vt[ids])*invdxtheta;
  
}


void SubStep2Dust_gpu (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, double dt,
                       PolarGrid *Vrad_ret, PolarGrid *Vtheta_ret) {
  int nr, ns;
  nr = Rho->Nrad;
  ns = Rho->Nsec;
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid ((ns+block.x-1)/block.x, (nr+block.y-1)/block.y);

  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)RadiiStuff, (size_t)(14*(nr+1))*sizeof(double),	0, cudaMemcpyHostToDevice));  
  kernel_substep2b <<< grid, block >>> (Vrad->gpu_field,
                                        Vtheta->gpu_field,
                                        dust_size->gpu_field,
                                        //Vrad_ret->gpu_field,
                                        //Vtheta_ret->gpu_field,
                                        ns, 
                                        nr,
                                        Rho->pitch/sizeof(double), 
                                        dt);
  cudaThreadSynchronize();
  getLastCudaError ("kernel_substep2b failed");
}
