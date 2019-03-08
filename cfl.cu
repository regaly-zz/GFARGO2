/** \file "cfl.cu" : implements the kernel for the "conditionCFL" procedure
*/
#define __CUDA 1
#include "fargo.h"
#undef __CUDA
#include <stdarg.h>
#include <helper_cuda.h>
#include <cuda.h>

// BLOCK_X : in azimuth
//#define BLOCK_X DEF_BLOCK_X_CFL
#define BLOCK_X 64
// BLOCK_Y : in radius
#define BLOCK_Y 4

#define GET_TAB(u,x,y,pitch) *(u + __mul24(y, pitch) + x)

/* Here we shall put into CRadiiStuff : viscosity, cs, rinf, rsup, rmed, vmoy */
#define gpu_visc CRadiiStuff[ig]
#define gpu_cs   CRadiiStuff[nr+ig]
#define gpu_rinf CRadiiStuff[nr*2+ig]
#define gpu_rsup CRadiiStuff[nr*3+ig]
#define gpu_rmed CRadiiStuff[nr*4+ig]
#define gpu_vmoy CRadiiStuff[nr*5+ig]

//static PolarGrid *DeltaT;
extern PolarGrid *Buffer;



extern "C" void AzimuthalAverage (PolarGrid *array, double *res);

//__constant__ double CRadiiStuff[16384];
__device__ double CRadiiStuff[16384];

__global__ void kernel_cfl1 (double *vrad,
			                       double *vtheta,
			                       int pitch,
			                       double invns, 
                             double *delta_t,
			                       int nr, 
                             int ns,
                             bool adiabatic,
                             double adiabatic_index,
                             double *energy,
                             double *dens,
                             double  cfl) {

  __shared__ double vr[(BLOCK_X+1)*(BLOCK_Y+1)];
  __shared__ double vt[(BLOCK_X+1)*(BLOCK_Y+1)];
  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;
  int js = threadIdx.x;
  int is = threadIdx.y;
  int jgp = jg+1;
  int ids = is*(blockDim.x+1)+js;
  int idg = __mul24(ig, pitch) + jg;
  if (jg == ns-1) jgp = 0;
  // We perform a coalesced read of 'vrad' into the shared memory (vr);
  vr[ids] = vrad[idg];
  vt[ids] = vtheta[idg];
  // EDGE 2: "RIGHT EDGE". Needed for vtheta. 
  if (js == blockDim.x-1)
    vt[is*(blockDim.x+1)+blockDim.x] = GET_TAB (vtheta, jgp, ig, pitch);
  // EDGE 4: "TOP EDGE". Needed for vrad. 
  if ((is == blockDim.y-1) && (ig < nr-1))
    vr[js+blockDim.y*(blockDim.x+1)] = GET_TAB (vrad, jg, ig+1, pitch);
  if ((is == blockDim.y-1) && (ig == nr-1))
    vr[js+blockDim.y*(blockDim.x+1)] = 0.0;

  __syncthreads ();

  //double visc = CRadiiStuff[ig];
  //double cs   = CRadiiStuff[nr+ig];
  //double rinf   = CRadiiStuff[nr*2+ig];
  //double rsup = CRadiiStuff[nr*3+ig];
  //double rmed = CRadiiStuff[nr*4+ig];
  //double vmoy = CRadiiStuff[nr*5+ig];

  double vtres = vt[ids]-gpu_vmoy;
  double dxr = gpu_rsup-gpu_rinf;
  double dxt = gpu_rmed * 6.28318530717958647688*invns;
  double dxm = dxr;

  if (dxt < dxr) dxm = dxt; // non-divergent test
  
  
  double invdt1;
  if (adiabatic)
    invdt1 = sqrt(adiabatic_index*(adiabatic_index-1.0)*energy[idg]/dens[idg])/dxm;
  else
    invdt1 = gpu_cs/dxm;
  
  double invdt2 = abs(vr[ids])/dxr;
  double invdt3 = abs(vtres)/dxt;
  double dvr = vr[ids+blockDim.x+1]-vr[ids];
  double dvt = vt[ids+1]-vt[ids];
  // Below : possibly divergent
  if (dvr > 0.f) dvr = 0.f;
  if (dvt > 0.f) dvt = 0.f;
  dvr = -dvr;
  dvt = -dvt;
  double invdt4 = 4.0 * CVNR * CVNR * max (dvr/dxr, dvt/dxt);
  double invdt5 = gpu_visc*4.0/(dxm*dxm);
  double dt = cfl/sqrt(invdt1*invdt1+invdt2*invdt2+invdt3*invdt3+invdt4*invdt4+invdt5*invdt5);
  delta_t[idg] = dt;
}


__global__ void kernel_cfl2 (double *array2D,
			                       double *buffer,
			                       int pitch,
			                       int size) {

  __shared__ double sdata[BLOCK_X*BLOCK_Y];
  unsigned int tid = threadIdx.x;
  unsigned int yt  = threadIdx.y*blockDim.x;
  unsigned int jg  = threadIdx.x + blockIdx.x * __mul24(blockDim.x,  2);
  unsigned int ig  = threadIdx.y + blockIdx.y * blockDim.y;

  unsigned int ytid = yt+tid;
  
  sdata[ytid] = 1e30;
  if (jg < size)
    sdata[ytid] = GET_TAB (array2D, jg, ig, pitch);
  if (jg+blockDim.x < size)
    sdata[ytid] = fminf(sdata[ytid], GET_TAB(array2D, jg+blockDim.x, ig, pitch));
  __syncthreads ();
  
  if (tid < 32) {
    volatile double *smem = sdata;
    smem[ytid] = fminf (smem[ytid],smem[ytid+32]);
    smem[ytid] = fminf (smem[ytid],smem[ytid+16]);
    smem[ytid] = fminf (smem[ytid],smem[ytid+8]);
    smem[ytid] = fminf (smem[ytid],smem[ytid+4]);
    smem[ytid] = fminf (smem[ytid],smem[ytid+2]);
    smem[ytid] = fminf (smem[ytid],smem[ytid+1]);
  }

  if (tid == 0)
    GET_TAB (buffer, blockIdx.x, ig, pitch) = sdata[yt];
}

extern "C" int ConditionCFL_gpu (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, PolarGrid *Energy, double deltaT) {
  int nr, ns;
  static int FirstTime = YES;
  static double *rinf, *cs, *visc, *stuff, *rsup, *rmed, *vmoy; // HOST
  double invns, dtmin, dt;

  nr = Vrad->Nrad;
  ns = Vrad->Nsec;

  if (FirstTime) {
    stuff  = (double *)malloc (nr*sizeof(double)*6);
    visc = stuff;
    cs = stuff+nr;
    rinf = stuff+2*nr;
    rsup = stuff+3*nr;
    rmed = stuff+4*nr;
    vmoy = stuff+5*nr;
    for (int i=0; i < nr; i++) {
      rinf[i] = Rinf[i];
      rsup[i] = Rsup[i];
      rmed[i] = Rmed[i];
      cs[i]   = SOUNDSPEED[i];
      visc[i] = FViscosity(Rmed[i]);
    }
    FirstTime = NO;
  }
  /*
  // must be relaculate for adiabatic disc
  if (Adiabatic) {
    for (int i=0; i < nr; i++) {
      cs[i]   = SOUNDSPEED[i];
      visc[i] = FViscosity(Rmed[i]);    
    }
  }*/
  
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid ((ns+block.x-1)/block.x, (nr+block.y-1)/block.y);

  AzimuthalAverage (Vtheta, vmoy);

  invns = 1./(double)ns;
  for (int i=0; i<nr; i++)
    vmoy[i] *= invns;

  dtmin = 1e30;
  for (int i = Zero_or_active; i < MaxMO_or_active; i++) {
    dt = 2.0*M_PI*CFL/(double)NSEC/fabs(vmoy[i]*InvRmed[i]-vmoy[i+1]*InvRmed[i+1]);
    if (dt < dtmin) dtmin = dt;
  }

  double *energy_gpu;                 // must be defined as for non-adiabatic there is no Energy->gpu_filed !
  if (Adiabatic)
    energy_gpu = Energy->gpu_field;   // set gpu field for energy

  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)stuff, (size_t)(6*nr*sizeof(double)),	 0, cudaMemcpyHostToDevice));
  kernel_cfl1 <<< grid, block >>> (Vrad->gpu_field, 
                                   Vtheta->gpu_field, 
                                   Vrad->pitch/sizeof(double), 
                                   1.0/(double)ns, 
                                   DeltaT->gpu_field,
                                   nr, 
                                   ns,
                                   Adiabatic,
                                   ADIABATICINDEX,
                                   energy_gpu,
                                   Rho->gpu_field,
                                   CFL);
  cudaThreadSynchronize();
  getLastCudaError ("kernel cfl failed");

  int nxbar = ns;
  grid.x = ((nxbar+block.x-1)/block.x+1)/2;
  grid.y = (nr+block.y-1)/block.y;
  
  // Below : Buffer is necessarily allocated at this stage, since we already
  // performed a reduction before (see "AzimuthalAverage" above).

  kernel_cfl2 <<< grid, block >>> (DeltaT->gpu_field, 
                                   Buffer->gpu_field, 
                                   DeltaT->pitch/sizeof(double), 
                                   nxbar);
  cudaThreadSynchronize();
  getLastCudaError ("kernel cfl2 failed / 1st");

  nxbar = (nxbar+2*BLOCK_X-1)/(2*BLOCK_X);

  while (nxbar > 1) {
    grid.x = ((nxbar+block.x-1)/block.x+1)/2;
    grid.y = (nr+block.y-1)/block.y;

    kernel_cfl2 <<< grid, block >>> (Buffer->gpu_field,
                                     Buffer->gpu_field,
                                     Buffer->pitch/sizeof(double), 
                                     nxbar);
    cudaThreadSynchronize();
    getLastCudaError("kernel cfl2 failed / 2nd");
    nxbar = (nxbar+2*BLOCK_X-1)/(2*BLOCK_X);

  }
//  D2H (Buffer);
//  for (int kk=0; kk < 10; kk++)
//    printf ("%f\t", Buffer->Field[kk]);
//  exit (1);
  cudaMemcpy2D (vmoy, sizeof(double), Buffer->gpu_field, Buffer->pitch, sizeof(double), nr, cudaMemcpyDeviceToHost);

  // minimal timestep at each radius stored in 'vmoy'
  for (int i = 0; i < nr; i++)
    if (vmoy[i] < dtmin) 
      dtmin = vmoy[i];
  /*
  if (dtmin < .5*dt_before) {
    accident++;
    D2H (Vrad);
    D2H (Vtheta);
    D2H (Rho);
    WriteDiskPolar (Rho,    9999-accident);
    WriteDiskPolar (Vrad,   9999-accident);
    WriteDiskPolar (Vtheta, 9999-accident);
  }
  */
//  dt_before = dtmin;



  if (deltaT < dtmin) 
    dtmin = deltaT;
    
  
  return (int)(ceil(deltaT/dtmin));  
}
