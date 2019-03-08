/** \file Planet.c contains CUDA kernel for planetary accretion.

Accretion of disk material onto the planets, and solver of planetary
orbital elements.  The prescription used for the accretion is the one
designed by W. Kley.

*/

#define __CUDA 1
#include "fargo.h"
#undef __CUDA
#include <stdarg.h>
#include <helper_cuda.h>
#include <cuda.h>

// BLOCK_X : in azimuth
//#define BLOCK_X DEF_BLOCK_X_PLANET
#define BLOCK_X 32
// BLOCK_Y : in radius
#define BLOCK_Y 4

#define surf  CRadiiStuff[(Nr+1)*9 + ig]
#define rmed  CRadiiStuff[(Nr+1)*6 + ig]

__device__ double CRadiiStuff[32768];

// double version of atomic add
__device__ double datomicAdd(double* address, 
                             double val) {

    unsigned long long int* address_as_ull = (unsigned long long int*) address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

__global__ void kernel_placcretion(const int     Nr,
                                   const int     Nphi,
                                   const int     pitch,
                                         double *abs,
                                         double *ord,
                                         double  omega_frame,
                                         double  XPlanet,
                                         double  YPlanet,
                                   const int     i_min,
                                   const int     i_max,
                                   const int     j_min,
                                   const int     j_max,
                                         double  facc1,
                                         double  facc2,
                                         double  frac1,
                                         double  frac2,
                                         double  RRoche,
                                         double *dens,
                                         double *vrad,
                                         double *vtheta,
                                         double *d_dMPlanet,
                                         double *d_dPxPlanet,
                                         double *d_dPyPlanet,
                                         bool    changePlanet){

  /*__shared__ double buffer[3];
  
  buffer[0] = 0.0;
  buffer[1] = 0.0;
  buffer[2] = 0.0;
  
  __syncthreads ();
  */
  const int jg  = blockDim.x * blockIdx.x + threadIdx.x;
  const int ig  = blockDim.y * blockIdx.y + threadIdx.y;
  const int idg = __mul24 (ig, pitch) + jg;
  
  int jf = jg;
  while (jf <  0)
    jf += Nphi;
  while (jf >= Nphi)
    jf -= Nphi;

  const int l = jf + ig * Nphi;
  int lip = l + Nphi;
  int ljp = l + 1;

  if (jf == Nphi-1) 
    ljp = ig*Nphi;
      
  const double xc = abs[l];
  const double yc = ord[l];
  const double dx = XPlanet-xc;
  const double dy = YPlanet-yc;
  const double distance = sqrt(dx*dx+dy*dy);

  //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  const double vtcell=0.5*(vtheta[l]+vtheta[ljp])+0*rmed*omega_frame + rsqrt(rmed); // for GFARGO the local Keplerian speed must be added!
//  const double vtcell=0.5*(vtheta[l]+vtheta[ljp])+XPlanet*sqrt(XPlanet+YPlanet*YPlanet)*omega_frame + rsqrt(rmed);
  const double vrcell=0.5*(vrad[l]+vrad[lip]);
  const double vxcell=(vrcell*xc-vtcell*yc)/rmed;
  const double vycell=(vrcell*yc+vtcell*xc)/rmed;
  
  if (distance < frac1 * RRoche) {
    const double deltaM = facc1*dens[l]*surf;

    // remove density inside Hill sphere
    dens[l] *= (1.0 - facc1);
  
    // change planet mass and momentum if accretion efficiency is grater than zero
    if (changePlanet) {
      // atomic add is required (naive method)
      datomicAdd (d_dMPlanet, deltaM);
      datomicAdd (d_dPxPlanet, deltaM*vxcell);
      datomicAdd (d_dPyPlanet, deltaM*vycell);
    }
  }
  
  
  if (distance < frac2 * RRoche) {
    const double deltaM = facc2*dens[l]*surf;

    // remove density inside Hill sphere
    dens[l] *= (1.0 - facc2);

    // change planet mass and momentum if accretion efficiency is grater than zero
    if (changePlanet) {
      // atomic add is required (naive method)
      datomicAdd (d_dMPlanet, deltaM);
      datomicAdd (d_dPxPlanet, deltaM*vxcell);
      datomicAdd (d_dPyPlanet, deltaM*vycell);
    }
  }
}


extern "C"
void AccreteOntoPlanets_gpu (PolarGrid *Rho, PolarGrid *Vrad, PolarGrid *Vtheta, double dt, PlanetarySystem *sys, double *AccretedMass, bool changePlanet) {
  double RRoche, Rplanet, angle;
  int i_min,i_max, j_min, j_max, ns, nr, k;
  double Xplanet, Yplanet, Mplanet, VXplanet, VYplanet;
  double facc, facc1, facc2, frac1, frac2;
  double PxPlanet, PyPlanet;

  nr     = Rho->Nrad;
  ns     = Rho->Nsec;
  
  static double *d_dMPlanet, *d_dPxPlanet, *d_dPyPlanet;  
  static bool FirstTime=YES;
  if (FirstTime) {
    cudaMalloc(&d_dMPlanet,  sizeof(double));
    cudaMalloc(&d_dPxPlanet, sizeof(double));
    cudaMalloc(&d_dPyPlanet, sizeof(double));
    FirstTime=NO;
  }

  checkCudaErrors(cudaMemcpyToSymbol(CRadiiStuff, (void *)RadiiStuff, (size_t)(14*(nr+1))*sizeof(double),0, cudaMemcpyHostToDevice));
  
  // go through all planets
  for (k=0; k < sys->nb; k++) {
    // accretion only for eta>10^-10
    if (sys->acc[k] != 0) {

      // reset accumulative variables
      double h_dMPlanet = 0.0, h_dPxPlanet = 0.0, h_dPyPlanet = 0.0;      
      cudaMemcpy(d_dMPlanet,  &h_dMPlanet, sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_dPxPlanet, &h_dPxPlanet, sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy(d_dPyPlanet, &h_dPyPlanet, sizeof(double), cudaMemcpyHostToDevice);
      
      // zero the change in planet mass and impulse at the beginning
      //dMplanet = dPxPlanet = dPyPlanet = 0.0;
            
      // get planetary parameter
      Xplanet = sys->x[k];
      Yplanet = sys->y[k];
      VXplanet = sys->vx[k];
      VYplanet = sys->vy[k];
      Mplanet = sys->mass[k]*MassTaper;
      PxPlanet = Mplanet*VXplanet;
      PyPlanet = Mplanet*VYplanet;

      // Roche radius assuming that the central mass is 1.0
      Rplanet = sqrt(Xplanet*Xplanet+Yplanet*Yplanet);

      // Hill-sphere accretion : initialization of W. Kley's parameters
      RRoche = pow((1.0/3.0*Mplanet),(1.0/3.0))*Rplanet; 
        
      // pow(Rplanet, -1.5) must be taken into account
      facc = dt*(abs(sys->acc[k]))*pow(Rplanet, -1.5);
      facc1 = 1.0/3.0*facc;
      facc2 = 2.0/3.0*facc;
      frac1 = 0.75;
      frac2 = 0.45;

      // fix radius accretion
      //RRoche = -sys->acc[k]);
      //facc = dt;
      
      // select the indices in the Roche lobe region
      i_min=0;
      i_max=nr-1;
      while ((Rsup[i_min] < Rplanet-RRoche) && (i_min < nr)) i_min++;
      while ((Rinf[i_max] > Rplanet+RRoche) && (i_max > 0)) i_max--;
      angle = atan2 (Yplanet, Xplanet);
      j_min =(int)((double)ns/2.0/M_PI*(angle - 2.0*RRoche/Rplanet));
      j_max =(int)((double)ns/2.0/M_PI*(angle + 2.0*RRoche/Rplanet));
      
      // kernel calling
      dim3 block (BLOCK_X, BLOCK_Y);
      dim3 grid  ((ns + block.x-1)/block.x, (nr + block.y-1)/block.y);
      kernel_placcretion<<<grid, block>>>(nr, 
                                          ns, 
                                          (Rho->pitch)/sizeof (double),
                                          CellAbscissa->gpu_field, 
                                          CellOrdinate->gpu_field, 
                                          OmegaFrame,
                                          Xplanet, 
                                          Yplanet,
                                          i_min, 
                                          i_max, 
                                          j_min, 
                                          j_max, 
                                          facc1, 
                                          facc2, 
                                          frac1, 
                                          frac2, 
                                          RRoche, 
                                          Rho->gpu_field, 
                                          Vrad->gpu_field, 
                                          Vtheta->gpu_field,
                                          d_dMPlanet, 
                                          d_dPxPlanet, 
                                          d_dPyPlanet,
                                          (changePlanet && (sys->acc[k] > 0)) || MonitorAccretion);

      // download mass and momentum change from device
      cudaMemcpy(&h_dMPlanet,  d_dMPlanet, sizeof(double), cudaMemcpyDeviceToHost); 
      cudaMemcpy(&h_dPxPlanet, d_dPxPlanet, sizeof(double), cudaMemcpyDeviceToHost); 
      cudaMemcpy(&h_dPyPlanet, d_dPyPlanet, sizeof(double), cudaMemcpyDeviceToHost); 

      if (MonitorAccretion)
        *AccretedMass +=  h_dMPlanet;
      
      // donwload net planet values if ...
      if (changePlanet && (sys->acc[k] > 0)) {
      
        // momentum and mass conservation
        Mplanet           += h_dMPlanet;
        sys->acc_rate[k]   = h_dMPlanet/dt;
        PxPlanet          += h_dPxPlanet;
        PyPlanet          += h_dPyPlanet;
      
        // update planetary data (velocity and mass)
        if (sys->FeelDisk[k] == YES) {
	        sys->vx[k] = PxPlanet/Mplanet;
	        sys->vy[k] = PyPlanet/Mplanet;
        }
        sys->mass[k] = Mplanet;
      }
    }
  }
}