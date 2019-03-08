#define __CUDA 1
#include "fargo.h"
#undef __CUDA
#include <stdarg.h>
#include <helper_cuda.h>
#include <cuda.h>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/extrema.h>

// BLOCK_X : in azimuth
//#define BLOCK_X DEF_BLOCK_X_DISKBC
#define BLOCK_X 16
// BLOCK_Y : in radius
#define BLOCK_Y 16

double *gpu_Surf, *gpu_CellMass, *gpu_bcx, *gpu_bcy;   // 

__global__ void kernel_calcbc (double *Surf, 
                               double *Rho, 
                               double *xc, 
                               double *yc,
                               double *cellmass, 
                               double *bcx, 
                               double *bcy,
                               int     ns, 
                               int     nr, 
                               int     pitch) {

  // jg & ig, g like 'global' (global memory <=> full grid)
  // Below, we recompute x and y for each zone using cos/sin.
  // This method turns out to be faster, on high-end platforms,
  // than a coalesced read of tabulated values.
  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;

  //if (ig >= nr || jg >= nsec)
  //  return;

  int idg = __mul24(ig, pitch) + jg;

  const double _cellmass = Surf[ig] * Rho[idg];
  bcx[idg] = _cellmass * xc[idg];
  bcy[idg] = _cellmass * yc[idg];
  cellmass[idg] = _cellmass;
}

__global__ void kernel_calcbc_dust (double *Surf, 
                                    double *Rho, 
                                    double *xc, 
                                    double *yc,
                                    double *cellmass, 
                                    double *bcx, 
                                    double *bcy,
                                    int     ns, 
                                    int     nr, 
                                    int     pitch) {

  // jg & ig, g like 'global' (global memory <=> full grid)
  // Below, we recompute x and y for each zone using cos/sin.
  // This method turns out to be faster, on high-end platforms,
  // than a coalesced read of tabulated values.
  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;

  //if (ig >= nr || jg >= nsec)
  //  return;

  int idg = __mul24(ig, pitch) + jg;

  const double _cellmass = Surf[ig] * Rho[idg];
  bcx[idg] = _cellmass * xc[idg];
  bcy[idg] = _cellmass * yc[idg];
  cellmass[idg] += _cellmass;
}

extern "C" void CalcGasBC (PolarGrid *Rho, PlanetarySystem *sys) {
  static bool First;

  int nr = Rho->Nrad;
  int ns = Rho->Nsec;
  
  if (!First){
    double *h_Surf;
    checkCudaErrors(cudaMallocHost ((void **) &h_Surf, sizeof(double) * nr));
    checkCudaErrors(cudaMalloc ((void **) &gpu_Surf, sizeof(double) * nr));
    for (int i=0; i< nr; i++) {
      h_Surf[i] = Surf[i];
    }  
    checkCudaErrors(cudaMemcpy(gpu_Surf, h_Surf,  sizeof(double) * nr, cudaMemcpyHostToDevice));

    size_t pitch;  
    checkCudaErrors(cudaMallocPitch ((void **) &gpu_CellMass, &pitch, sizeof(double) * ns, nr));
    checkCudaErrors(cudaMallocPitch ((void **) &gpu_bcx, &pitch, sizeof(double) * ns, nr));
    checkCudaErrors(cudaMallocPitch ((void **) &gpu_bcy, &pitch, sizeof(double) * ns, nr));
  
    First = YES;
  }

  int nelements = ns * nr;
  
  double Rpl = sqrt(sys->x[0]*sys->x[0] + sys->y[0]*sys->y[0]);
  
  // Dichotomic search of starting index of outer damping BC (on host)
  int imin=0;
  int imax=nr-1;
  while (imax-imin > 1) {
    int imed = (imax+imin)/2;
    if (Rmed[imed] > Rpl)
      imax=imed;
    else
      imin=imed;
  }
  int imino = imin;
  
  // barycentre kernel calling
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid  ((ns + block.x-1)/block.x, (nr + block.y-1)/block.y);
  kernel_calcbc<<<grid, block>>>(gpu_Surf, 
                                 Rho->gpu_field, 
                                 CellAbscissa->gpu_field, 
                                 CellOrdinate->gpu_field,
                                 gpu_CellMass, 
                                 gpu_bcx, 
                                 gpu_bcy,
                                 ns, 
                                 nr, 
                                (Rho->pitch)/sizeof (double));
    
  // wrap raw pointer with a device_ptr
  thrust::device_ptr<double> d_CellMassInner(gpu_CellMass);
  thrust::device_ptr<double> d_CellMassOuter(gpu_CellMass);
  thrust::device_ptr<double> d_bcx(gpu_bcx);
  thrust::device_ptr<double> d_bcy(gpu_bcy);
      
  // use thrust to find the summs
//  DiskMassInner = thrust::reduce(d_CellMassInner, d_CellMassOuter + imino*ns, (double) 0, thrust::plus<double>());
  GasDiskMassInner = thrust::reduce(d_CellMassInner, d_CellMassInner + imino*ns, (double) 0, thrust::plus<double>());
  GasDiskMassOuter = thrust::reduce(d_CellMassOuter+imino*ns, d_CellMassOuter + nelements, (double) 0, thrust::plus<double>());
  double tot_bcx = thrust::reduce(d_bcx, d_bcx + nelements, (double) 0, thrust::plus<double>());
  double tot_bcy = thrust::reduce(d_bcy, d_bcy + nelements, (double) 0, thrust::plus<double>());
    
  // the result
  
  double TotDiskMass = GasDiskMassInner + GasDiskMassOuter;
  GasDiskBC_x = tot_bcx/TotDiskMass;
  GasDiskBC_y = tot_bcy/TotDiskMass;

  // barycenters of star+planet and total  
  double spl_mass = 1.0;
  StarPlanetBC_x  = 0.0;
  StarPlanetBC_y  = 0.0;
  for (int k = 0; k < sys->nb; k++) {
    spl_mass += sys->mass[k];
    StarPlanetBC_x  += sys->mass[k] * sys->x[k];
    StarPlanetBC_y  += sys->mass[k] * sys->y[k];
  }
  GasBC_x = (StarPlanetBC_x + GasDiskBC_x * TotDiskMass)/(spl_mass+TotDiskMass);
  GasBC_y = (StarPlanetBC_y + GasDiskBC_y * TotDiskMass)/(spl_mass+TotDiskMass);
  StarPlanetBC_x /= spl_mass;
  StarPlanetBC_x /= spl_mass;
}


extern "C" void CalcDustBC (PolarGrid **Rho, PlanetarySystem *sys) {
  static bool First;

  int nr = Rho[0]->Nrad;
  int ns = Rho[0]->Nsec;
  
  if (!First){
    double *h_Surf;
    checkCudaErrors(cudaMallocHost ((void **) &h_Surf, sizeof(double) * nr));
    checkCudaErrors(cudaMalloc ((void **) &gpu_Surf, sizeof(double) * nr));
    for (int i=0; i< nr; i++) {
      h_Surf[i] = Surf[i];
    }  
    checkCudaErrors(cudaMemcpy(gpu_Surf, h_Surf,  sizeof(double) * nr, cudaMemcpyHostToDevice));

    size_t pitch;  
    checkCudaErrors(cudaMallocPitch ((void **) &gpu_CellMass, &pitch, sizeof(double) * ns, nr));
    checkCudaErrors(cudaMallocPitch ((void **) &gpu_bcx, &pitch, sizeof(double) * ns, nr));
    checkCudaErrors(cudaMallocPitch ((void **) &gpu_bcy, &pitch, sizeof(double) * ns, nr));
  
    First = YES;
  }

  int nelements = ns * nr;
  
  double Rpl = sqrt(sys->x[0]*sys->x[0] + sys->y[0]*sys->y[0]);
  
  // Dichotomic search of starting index of outer damping BC (on host)
  int imin=0;
  int imax=nr-1;
  while (imax-imin > 1) {
    int imed = (imax+imin)/2;
    if (Rmed[imed] > Rpl)
      imax=imed;
    else
      imin=imed;
  }
  int imino = imin;
  
  
  // barycentre kernel calling
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid  ((ns + block.x-1)/block.x, (nr + block.y-1)/block.y);
  
  // delete values from dust density grid
  cudaMemset (gpu_CellMass, 0, nelements * sizeof (double));

  for (int i=0; i< DustBinNum; i++) {
    kernel_calcbc_dust<<<grid, block>>>(gpu_Surf, 
                                        Rho[i]->gpu_field, 
                                        CellAbscissa->gpu_field, 
                                        CellOrdinate->gpu_field,
                                        gpu_CellMass, 
                                        gpu_bcx, 
                                        gpu_bcy,
                                        ns, 
                                        nr, 
                                        (Rho[0]->pitch)/sizeof (double));
  }
  // wrap raw pointer with a device_ptr
  thrust::device_ptr<double> d_CellMassInner(gpu_CellMass);
  thrust::device_ptr<double> d_CellMassOuter(gpu_CellMass);
  thrust::device_ptr<double> d_bcx(gpu_bcx);
  thrust::device_ptr<double> d_bcy(gpu_bcy);
      
  // use thrust to find the summs
  //DiskMassInner = thrust::reduce(d_CellMassInner, d_CellMassOuter + imino*ns, (double) 0, thrust::plus<double>());
  DustDiskMassInner = thrust::reduce(d_CellMassInner, d_CellMassInner + imino*ns, (double) 0, thrust::plus<double>());
  DustDiskMassOuter = thrust::reduce(d_CellMassOuter+imino*ns, d_CellMassOuter + nelements, (double) 0, thrust::plus<double>());
  double tot_bcx = thrust::reduce(d_bcx, d_bcx + nelements, (double) 0, thrust::plus<double>());
  double tot_bcy = thrust::reduce(d_bcy, d_bcy + nelements, (double) 0, thrust::plus<double>());
    
  // the result1111
  double TotDiskMass = DustDiskMassInner + DustDiskMassOuter;
  DustDiskBC_x = tot_bcx/TotDiskMass;
  DustDiskBC_y = tot_bcy/TotDiskMass;

  // barycenters of star+planet and total  
  double spl_mass = 1.0;
  StarPlanetBC_x  = 0.0;
  StarPlanetBC_y  = 0.0;
  for (int k = 0; k < sys->nb; k++) {
    spl_mass += sys->mass[k];
    StarPlanetBC_x  += sys->mass[k] * sys->x[k];
    StarPlanetBC_y  += sys->mass[k] * sys->y[k];
  }
  DustBC_x = (StarPlanetBC_x + DustDiskBC_x * TotDiskMass)/(spl_mass+TotDiskMass);
  DustBC_y = (StarPlanetBC_y + DustDiskBC_y * TotDiskMass)/(spl_mass+TotDiskMass);
  StarPlanetBC_x /= spl_mass;
  StarPlanetBC_x /= spl_mass;
}
