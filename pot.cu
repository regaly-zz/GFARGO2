/** \file "force.cu" : implements the kernel for the tidal force calculation
*/
#define __CUDA 1
#include "fargo.h"
#undef __CUDA
#include <stdarg.h>
#include <helper_cuda.h>
#include <cuda.h>

// BLOCK_X : in azimuth
//#define BLOCK_X DEF_BLOCK_X_POT
#define BLOCK_X 32
// BLOCK_Y : in radius
#define BLOCK_Y 4

#define GET_TAB(u,x,y,pitch) *(u + __mul24(y, pitch) + x)

// improve legibility
#define rmed     CRadiiStuff[5*nb_planets+ig]
#define invrmed  CRadiiStuff[5*nb_planets+nr+ig]
#define xp       CRadiiStuff[nb_planets+k]
#define yp       CRadiiStuff[2*nb_planets+k]
#define mp       CRadiiStuff[k]
#define invd3    CRadiiStuff[3*nb_planets+k]
#define eps2     CRadiiStuff[4*nb_planets+k]
//#define phi      jg*(double)dphi

__device__ double CRadiiStuff[32768];

__global__ void kernel_fillpot (double *pot,
                                int    nr,
                                int    pitch,
                                double dphi,
                                int    nb_planets,
                                double itx,
                                double ity) {

  // jg & ig, g like 'global' (global memory <=> full grid)
  // Below, we recompute x and y for each zone using cos/sin.
  // This method turns out to be faster, on high-end platforms,
  // than a coalesced read of tabulated values.
  double potential = 0.0;
  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;
  double phi = (double)jg*dphi;
  double x = rmed * cos(phi);
  double y = rmed * sin(phi);
  for (int k = 0; k < nb_planets; k++) {
    potential += mp*rsqrt((x-xp)*(x-xp)+(y-yp)*(y-yp)+eps2);  // potential of planets
    potential -= mp*invd3*(x*xp+y*yp);                        // indirect potential of planets ???
  }
  GET_TAB (pot, jg, ig, pitch) = -G*(potential + (itx*x+ity*y)); // potential plus indirect term of disk if defined
}

extern "C"
void FillForcesArrays_gpu (PlanetarySystem *sys)
{
  int nr, ns;//, ii;
  nr = Potential->Nrad;
  ns = Potential->Nsec;
  double Invd3[MAX1D], Eps2[MAX1D], mass[MAX1D], xpl[MAX1D], ypl[MAX1D];
  double xplanet, yplanet, smoothing;//, frac, cs;
  double PlanetDistance, InvPlanetDistance3, RRoche;//, iplanet;
  int NbPlanets = sys->nb;
  ComputeIndirectTerm();
  for (int k = 0; k < NbPlanets; k++) {
    xplanet = (double)sys->x[k];
    yplanet = (double)sys->y[k];
    xpl[k] = xplanet;
    ypl[k] = yplanet;
    mass[k] = (double)((sys->mass[k])*MassTaper);
    PlanetDistance = sqrt(xplanet*xplanet+yplanet*yplanet);
    InvPlanetDistance3 =  1.0/PlanetDistance/PlanetDistance/PlanetDistance;
    RRoche = PlanetDistance*pow((mass[k]/3.0),1.0/3.0);
    if (RocheSmoothing) {
      smoothing = RRoche*ROCHESMOOTHING;
    } 
    else {
      //iplanet = GetGlobalIFrac (PlanetDistance);
      //frac = iplanet-floor(iplanet);
      //ii = (int)iplanet;
      //cs = GLOBAL_SOUNDSPEED[ii]*(1.0-frac)+GLOBAL_SOUNDSPEED[ii+1]*frac;
      //smoothing = cs * PlanetDistance * sqrt(PlanetDistance) * THICKNESSSMOOTHING;
      smoothing=THICKNESSSMOOTHING * AspectRatio(PlanetDistance) * pow(PlanetDistance, 1.0+FLARINGINDEX);
    }
    if (Indirect_Term)
      Invd3[k] = InvPlanetDistance3;
    else
      Invd3[k] = 0.0;
    Eps2[k] = smoothing*smoothing;
  }

  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid ((ns+block.x-1)/block.x, (nr+block.y-1)/block.y);

  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)mass, (size_t)(NbPlanets)*sizeof(double), 0, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)xpl,  (size_t)(NbPlanets)*sizeof(double), NbPlanets*sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)ypl,  (size_t)(NbPlanets)*sizeof(double), NbPlanets*sizeof(double)*2, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)Invd3, (size_t)(NbPlanets)*sizeof(double), NbPlanets*sizeof(double)*3, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)Eps2,  (size_t)(NbPlanets)*sizeof(double), NbPlanets*sizeof(double)*4, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)Rmed,  (size_t)(nr)*sizeof(double), NbPlanets*sizeof(double)*5, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)InvRmed, (size_t)(nr)*sizeof(double), (5*NbPlanets+nr)*sizeof(double), cudaMemcpyHostToDevice));
  
  kernel_fillpot <<< grid, block >>> (Potential->gpu_field,
				                              nr,
				                              Potential->pitch/sizeof(double),
				                              6.28318530717958647688/(double)ns, 
				                              NbPlanets,
				                              IndirectTerm.x,
				                              IndirectTerm.y);

  cudaThreadSynchronize();
  getLastCudaError ("kernel_fillpot failed");
}
