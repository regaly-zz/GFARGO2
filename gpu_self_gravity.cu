/*-----------------------------------------------------------------------------
 GPU-based 2D Poission solver for self-gravity

 Implementation of kernels and calling functions

 Written by Zs. Regaly 2016 (www.konkoly.hu/staff/regaly)
 Computational Astrophysics Group of Konkoly Observatory (www.konkoly.hu/CAG/)
------------------------------------------------------------------------------*/
#include <math.h>
#include <stdlib.h>

#define __CUDA 1
#include "fargo.h"
#undef __CUDA
#include <stdarg.h>
#include <helper_cuda.h>
#include <cuda.h>
#include <iostream>
#include <fstream>

#include "gpu_self_gravity.h"
//#include <helper_cuda.h>
//#include "fargo.h"
//#include "common.h"

#ifdef GPU_SG_BENCHMARK
  #include <sys/time.h>
  int BLOCK_X;                                    // BLOCK_X : in azimuth
  int BLOCK_Y;                                    // BLOCK_Y : in radius
#else
  //#define BLOCK_X DEF_BLOCK_X_GPU_SELF_GRAVITY
  #define BLOCK_X 32                              // BLOCK_X : in azimuth
  #define BLOCK_Y 16                              // BLOCK_Y : in radius
#endif

// declaration of global variables
//------------------------------------------------------------------------------
int __gpu_sg_Nr, __gpu_sg_Nphi;                                    // 2D grid dimensions
double *__gpu_sg_h_con1, *__gpu_sg_h_con2, *__gpu_sg_h_con3;       // constants stored in host
double *__gpu_sg_d_con1, *__gpu_sg_d_con2, *__gpu_sg_d_con3;       // constants stored in device

double __gpu_sg_dr, __gpu_sg_dphi, __gpu_sg_eps2;                  //
double *__gpu_sg_h_rmed, *__gpu_sg_h_rii;                          // 
double *__gpu_sg_d_rmed, *__gpu_sg_d_rii;                          // 


cufftDoubleComplex *__gpu_sg_h_A1, *__gpu_sg_h_A2, *__gpu_sg_h_A3; // FFTW complex matrices stored in host
cufftDoubleComplex *__gpu_sg_d_A1, *__gpu_sg_d_A2, *__gpu_sg_d_A3; // FFTW complex matrices stored in host
cufftHandle __gpu_sg_fftw_plan;                                    // FFTW plan

size_t complex_pitch;                                              // pitch for real and complex arrays

bool  __gpu_sg_allocmem = false;

/*
------------------------------------------------------------
  implementtion of kenrnels
------------------------------------------------------------
*/
__global__ void kernel_sg_A2 (const int           Nr,
                              const int           Nphi,
                              const double       *con2,
                              const int           pitch1,
                              const double       *sdens,
                              const int           pitch2,
                              cufftDoubleComplex *A2) {

  const int jg = blockDim.x * blockIdx.x + threadIdx.x;
  const int ig = blockDim.y * blockIdx.y + threadIdx.y;
  const int idg1 = __mul24 (ig, pitch1) + jg;
  const int idg2 = __mul24 (ig, pitch2) + jg;
    
//  if (ig < Nr) 
//    if (jg < Nphi)
      A2[idg2].x = sdens[idg1] * con2[ig];
}

__global__ void kernel_sg_A3 (const int           Nr2,
                              const int           Nphi,
                              double             *con1,
                              const int           pitch,
                              cufftDoubleComplex *A1,
                              cufftDoubleComplex *A3) {

  const int jg = blockDim.x * blockIdx.x + threadIdx.x;
  const int ig = blockDim.y * blockIdx.y + threadIdx.y;
  const int idg = __mul24 (ig, pitch) + jg;
  
//  if (ig < Nr2) 
//    if (jg < Nphi) {
      A3[idg].x = con1[0] * (A1[idg].x * A3[idg].x - A1[idg].y * A3[idg].y);
      A3[idg].y = con1[0] * (A1[idg].x * A3[idg].y + A1[idg].y * A3[idg].x);
 // }
}


__global__ void kernel_sg_pot (const int           Nr,
                               const int           Nphi,
                               const double       *con3,
                               const int           pitch1,
                               cufftDoubleComplex *A1,
                               cufftDoubleComplex *A3,
                               const int           pitch2,
                               double             *pot) {

  const int jg = blockDim.x * blockIdx.x + threadIdx.x;
  const int ig = blockDim.y * blockIdx.y + threadIdx.y;
  const int idg1 = __mul24 (ig, pitch1) + jg;
  const int idg2 = __mul24 (ig, pitch2) + jg;
    
//  if (ig < Nr) 
//    if (jg < Nphi) {
      pot[idg2] += A3[idg1].x / con3[ig];
//  }
}


__global__ void kernel_sg_force (const int           Nr,
                                 const int           Nphi,
                                 const int           pitch1,
                                 const double       *rii,
                                 const double       *rmed,
                                 const double       *sdens,
                                 const double       *pot,
                                 double             *sgacc) {


  const int jg = blockDim.x * blockIdx.x + threadIdx.x; // azimuthal
  const int ig = blockDim.y * blockIdx.y + threadIdx.y; // radial
  
  const int idg0 = __mul24 (ig-1, pitch1) + jg;
  const int idg1 = __mul24 (ig, pitch1) + jg;
  const int idg2 = __mul24 (ig+1, pitch1) + jg;
  
  
  if (ig < Nr-1)// && ig > 0)
//    if (jg < Nphi) {
       //sgacc[idg1] = - ((rmed[ig+1]-rmed[ig])*(pot[idg1]-pot[idg0])+(rmed[ig]-rmed[ig-1])*(pot[idg2]-pot[idg1]))/((rmed[ig+1]-rmed[ig])*(rmed[ig]-rmed[ig-1]));
      
      //sgacc[idg1] -= (pot[idg2] - pot[idg0])/(2.0*(rmed[ig+1]-rmed[ig-1]));
       //sgacc[idg1] /= 3.0;
      sgacc[idg2] = - (pot[idg2] - pot[idg1])/(rmed[ig+1]-rmed[ig]);
//    }
  
  
  if (ig == 0)
    sgacc[idg1] = sgacc[idg2];//- (pot[idg2]-pot[idg1])/(rmed[ig+1]-rmed[ig]);
  
  //if (ig == Nr-1)
  //  sgacc[idg1] = - (pot[idg1]-pot[idg0])/(rmed[ig]-rmed[ig-1]);

//  if (ig == Nr-1)
//    sgacc[idg1] *= 0.5; 

}


__global__ void kernel_sg_add_dens (   const int  Nr,
                                       const int  Nphi,
                                    const int     pitch,
                                    const double *dens_1,
                                    const double *dens_2,
                                          double *dens_res) {
                      
  const int jg = blockDim.x * blockIdx.x + threadIdx.x; // azimuthal
  const int ig = blockDim.y * blockIdx.y + threadIdx.y; // radial
  const int idg = __mul24 (ig, pitch) + jg;                      

  dens_res[idg] = dens_1[idg] + dens_2[idg];
}



/*
------------------------------------------------------------
  implementtion of C calling functions
------------------------------------------------------------*/
extern "C" void gpu_sg_add_dens (size_t real_pitch, double *d_dens1, double *d_dens2, double *d_densres) {
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid1  ((__gpu_sg_Nphi + block.x-1)/block.x, (__gpu_sg_Nr + block.y-1)/block.y);


  kernel_sg_add_dens<<<grid1, block>>>(__gpu_sg_Nr, __gpu_sg_Nphi, real_pitch/sizeof (double), d_dens1, d_dens2, d_densres);
  __checkCudaKernelError ();
}


extern "C" void gpu_sg_init (int Nr, int Nphi, double rmin, double rmax, double eps) {
  // alloca required host and device memory
  gpu_sg_allocmem (Nr, Nphi);
  
  // initializa self gravity solver
  gpu_sg_init_polargrid (Nr, Nphi, rmin, rmax);
  gpu_sg_init_const (eps);
  gpu_sg_init_matrix ();
  printf ("gpu_sg: GPU self-gravity module is initialized\n");
}


void gpu_sg_allocmem (int Nr, int Nphi) {
  // set dimensions
  __gpu_sg_Nr = Nr;
  __gpu_sg_Nphi = Nphi;

  // allocate of 1D host memory for polar grid parameters required for self gravity acceleration
  checkCudaErrors(cudaMallocHost ((void **) &__gpu_sg_h_rmed, sizeof(double) * __gpu_sg_Nr));
  checkCudaErrors(cudaMallocHost ((void **) &__gpu_sg_h_rii, sizeof(double) * (__gpu_sg_Nr+1)));
  
  // allocate of 1D device memory for polar grid parameters required for self gravity acceleration  
  checkCudaErrors(cudaMalloc ((void **) &__gpu_sg_d_rmed, sizeof(double) * __gpu_sg_Nr));
  checkCudaErrors(cudaMalloc ((void **) &__gpu_sg_d_rii, sizeof(double) * (__gpu_sg_Nr+1)));

  // allocation of 1D host memory for constants (with Nr elements)
  checkCudaErrors(cudaMallocHost ((void **) &__gpu_sg_h_con1, sizeof(double) * 1));
  checkCudaErrors(cudaMallocHost ((void **) &__gpu_sg_h_con2, sizeof(double) * __gpu_sg_Nr));
  checkCudaErrors(cudaMallocHost ((void **) &__gpu_sg_h_con3, sizeof(double) * __gpu_sg_Nr));
  
  // allocation of 1D device memory for constants (with Nr elements)
  checkCudaErrors(cudaMalloc ((void **) &__gpu_sg_d_con1, sizeof(double) * 1));
  checkCudaErrors(cudaMalloc ((void **) &__gpu_sg_d_con2, sizeof(double) * __gpu_sg_Nr));
  checkCudaErrors(cudaMalloc ((void **) &__gpu_sg_d_con3, sizeof(double) * __gpu_sg_Nr));

  // allocation of 1D host memory for complex matrices (with 2NrxNphi elements)
  checkCudaErrors(cudaMallocHost ((void **) &__gpu_sg_h_A1, sizeof(cufftDoubleComplex) * __gpu_sg_Nphi * 2 * __gpu_sg_Nr));
  checkCudaErrors(cudaMallocHost ((void **) &__gpu_sg_h_A2, sizeof(cufftDoubleComplex) * __gpu_sg_Nphi * 2 * __gpu_sg_Nr));
  checkCudaErrors(cudaMallocHost ((void **) &__gpu_sg_h_A3, sizeof(cufftDoubleComplex) * __gpu_sg_Nphi * 2 * __gpu_sg_Nr));

  // allocation of 2D pitched device memory for complex matrices (with Nphi, 2Nr elements)
  checkCudaErrors(cudaMallocPitch ((void **) &__gpu_sg_d_A1, &complex_pitch, sizeof(cufftDoubleComplex) * __gpu_sg_Nphi, 2 * __gpu_sg_Nr));
  checkCudaErrors(cudaMallocPitch ((void **) &__gpu_sg_d_A2, &complex_pitch, sizeof(cufftDoubleComplex) * __gpu_sg_Nphi, 2 * __gpu_sg_Nr));
  checkCudaErrors(cudaMallocPitch ((void **) &__gpu_sg_d_A3, &complex_pitch, sizeof(cufftDoubleComplex) * __gpu_sg_Nphi, 2 * __gpu_sg_Nr));
  
  // allocation is completed
  __gpu_sg_allocmem = true;
}


void inline __checkCudaKernelError () {
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) 
      printf("CUDA Error: %s\n", cudaGetErrorString(err));
}


void gpu_sg_init_polargrid (int Nr, int Nphi, double rmin, double rmax) {
  // check memory allocation
  if (!__gpu_sg_allocmem) {
    printf ("GPU memory allocation is required (gpu_sg_allocmem)!\n");
    exit (-1);
  }

  // dr and dphi
  __gpu_sg_dr   = (log (rmax) - log (rmin)) / (double) Nr;
  __gpu_sg_dphi = 2.0 * M_PI / (double) Nphi;

  // radial cell interface coordinates
  //double *rii;
  //rii = new double[Nr];
  for (int i=0; i< Nr+1; i++) {
    __gpu_sg_h_rii[i] = rmin * exp((double) i / (double) Nr * log(rmax / rmin));
  }

  // difeence in radial direction
  for (int i=0; i< Nr; i++) {
    __gpu_sg_h_rmed[i] = (2.0/3.0) * (__gpu_sg_h_rii[i+1]*__gpu_sg_h_rii[i+1]*__gpu_sg_h_rii[i+1]-__gpu_sg_h_rii[i]*__gpu_sg_h_rii[i]*__gpu_sg_h_rii[i]) 
                                   / (__gpu_sg_h_rii[i+1]*__gpu_sg_h_rii[i+1]-__gpu_sg_h_rii[i]*__gpu_sg_h_rii[i]);
  }

  // upload constants to device (with 1D array allocation)
  checkCudaErrors(cudaMemcpy( __gpu_sg_d_rii, __gpu_sg_h_rii,  sizeof(double) * (__gpu_sg_Nr+1), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(__gpu_sg_d_rmed, __gpu_sg_h_rmed, sizeof(double) *     __gpu_sg_Nr, cudaMemcpyHostToDevice));
} 


void gpu_sg_init_const (double eps) {
  
  // check memory allocation
  if (!__gpu_sg_allocmem) {
    printf ("GPU memory allocation is required (gpu_sg_allocmem)!\n");
    exit (-1);
  }

  // setup constants
  __gpu_sg_eps2 = eps * eps;
  __gpu_sg_h_con1[0] = 1.0/(2.0 * (double) __gpu_sg_Nr * (double) __gpu_sg_Nphi);
  for (int i=0; i < __gpu_sg_Nr; i++) {
    __gpu_sg_h_con2[i] = sqrt (__gpu_sg_h_rmed[i] * __gpu_sg_h_rmed[i] * __gpu_sg_h_rmed[i]) * __gpu_sg_dr * __gpu_sg_dphi; 
    __gpu_sg_h_con3[i] = sqrt (__gpu_sg_h_rmed[i]);
  }

  // upload constants to device (with 1D array allocation)
  checkCudaErrors(cudaMemcpy(__gpu_sg_d_con1, __gpu_sg_h_con1, sizeof(double) * 1, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(__gpu_sg_d_con2, __gpu_sg_h_con2, sizeof(double) * __gpu_sg_Nr, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(__gpu_sg_d_con3, __gpu_sg_h_con3, sizeof(double) * __gpu_sg_Nr, cudaMemcpyHostToDevice));
}


void __gpu_sg_allocate_host_array (double ***p, int Ncol, int Nrow) {
  double **array = new double* [Ncol];
  for(int i = 0; i < Ncol; i++)
    array[i] = new double[Nrow];
  
  *p = array;
}


void gpu_sg_init_matrix () {

  // check memory allocation
  if (!__gpu_sg_allocmem) {
    printf ("GPU memory allocation (gpu_init_self_gravity_matrix) is required!\n");
    exit (-1);
  }
  
  // calculate Glm (based on E. Vorobyov's code)
  double **_Glm;
  __gpu_sg_allocate_host_array (&_Glm, __gpu_sg_Nphi, 2 * __gpu_sg_Nr);
  
  double dr   = __gpu_sg_dr;
  double dphi = __gpu_sg_dphi;
  for (int i=0; i < __gpu_sg_Nphi; i++)
    for (int j = 0; j < 2 * __gpu_sg_Nr; j++) {
      if (i == 0 && j - __gpu_sg_Nr == 0)
        _Glm[i][j] = - 2.0 * ((1.0 / dphi) * log ((dphi / dr) + sqrt ((dphi / dr) * (dphi / dr)+1.0)))
                     - 2.0 * ((1.0 / dr)   * log ((dr / dphi) + sqrt ((dr / dphi) * (dr / dphi)+1.0)));
      else
//        _Glm[i][j] = -0.707106781186547/sqrt ((exp ((j-__gpu_sg_Nr) * dr) + exp(-(j-__gpu_sg_Nr) * dr))/2.0 - cos (i * dphi));
        _Glm[i][j] = -0.707106781186547/sqrt (((1.0 + __gpu_sg_eps2) * exp ((j-__gpu_sg_Nr) * dr) + exp(-(j-__gpu_sg_Nr) * dr))/2.0 - cos (i * dphi));
    }

  // setup A1, A2, and A3
  for (int i=0; i < __gpu_sg_Nr; i++)
    for (int j=0; j < __gpu_sg_Nphi; j++) {
      __gpu_sg_h_A1[j + i*__gpu_sg_Nphi].x = _Glm[j][i+__gpu_sg_Nr];
      __gpu_sg_h_A1[j + i*__gpu_sg_Nphi].y = 0;
      __gpu_sg_h_A2[j + i*__gpu_sg_Nphi].x = 0;
      __gpu_sg_h_A2[j + i*__gpu_sg_Nphi].y = 0;
    }
  for (int i=__gpu_sg_Nr; i < 2*__gpu_sg_Nr; i++)
    for (int j=0; j < __gpu_sg_Nphi; j++) {
      __gpu_sg_h_A1[j + i*__gpu_sg_Nphi].x = _Glm[j][i-__gpu_sg_Nr];
      __gpu_sg_h_A1[j + i*__gpu_sg_Nphi].y = 0;
      __gpu_sg_h_A2[j + i*__gpu_sg_Nphi].x = 0;
      __gpu_sg_h_A2[j + i*__gpu_sg_Nphi].y = 0;
    }

  // upload matrices to the device (with 2D array pitched memory allocation)
  checkCudaErrors(cudaMemcpy2D(__gpu_sg_d_A1, complex_pitch, __gpu_sg_h_A1, __gpu_sg_Nphi * sizeof(cufftDoubleComplex), __gpu_sg_Nphi * sizeof(cufftDoubleComplex), 2 * __gpu_sg_Nr, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy2D(__gpu_sg_d_A2, complex_pitch, __gpu_sg_h_A2, __gpu_sg_Nphi * sizeof(cufftDoubleComplex), __gpu_sg_Nphi * sizeof(cufftDoubleComplex), 2 * __gpu_sg_Nr, cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy2D(__gpu_sg_d_A3, complex_pitch, __gpu_sg_h_A3, __gpu_sg_Nphi * sizeof(cufftDoubleComplex), __gpu_sg_Nphi * sizeof(cufftDoubleComplex), 2 * __gpu_sg_Nr, cudaMemcpyHostToDevice));

  // setup 2D FFT plan
  cufftPlan2d(&__gpu_sg_fftw_plan, 2 * __gpu_sg_Nr, __gpu_sg_Nphi, CUFFT_Z2Z);
  cufftExecZ2Z(__gpu_sg_fftw_plan, __gpu_sg_d_A1, __gpu_sg_d_A1, CUFFT_FORWARD);
}


extern "C" void gpu_sg_calc_pot (size_t real_pitch, double *d_sdens, double *d_pot) {
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid1  ((__gpu_sg_Nphi + block.x-1)/block.x, (__gpu_sg_Nr + block.y-1)/block.y);
  dim3 grid2  ((__gpu_sg_Nphi + block.x-1)/block.x, (2 * __gpu_sg_Nr + block.y-1)/block.y);

  // S(r,phi) = r^1.5*Sigma(r,phi)
  kernel_sg_A2<<<grid1, block>>>(__gpu_sg_Nr, __gpu_sg_Nphi, __gpu_sg_d_con2, real_pitch/sizeof (double), d_sdens, complex_pitch/sizeof (cufftDoubleComplex), __gpu_sg_d_A2);
  __checkCudaKernelError ();
  
  // FFT of S(r,phi)
  cufftExecZ2Z(__gpu_sg_fftw_plan, __gpu_sg_d_A2, __gpu_sg_d_A3, CUFFT_FORWARD);
  
  kernel_sg_A3<<<grid2, block>>>(2 * __gpu_sg_Nr, __gpu_sg_Nphi, __gpu_sg_d_con1, complex_pitch/sizeof (cufftDoubleComplex), __gpu_sg_d_A1, __gpu_sg_d_A3);
  __checkCudaKernelError ();
      
  cufftExecZ2Z(__gpu_sg_fftw_plan, __gpu_sg_d_A3, __gpu_sg_d_A3, CUFFT_INVERSE);
  
  kernel_sg_pot<<<grid1, block>>>(__gpu_sg_Nr, __gpu_sg_Nphi, __gpu_sg_d_con3, complex_pitch/sizeof (cufftDoubleComplex), __gpu_sg_d_A1, __gpu_sg_d_A3, real_pitch/sizeof (double), d_pot);
  __checkCudaKernelError ();

  // ???? syncronize threads ???
  cudaThreadSynchronize();
}


extern "C" void gpu_sg_calc_acc (size_t real_pitch, double *d_sdens, double *d_pot, double *d_acc) {
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid1  ((__gpu_sg_Nphi + block.x-1)/block.x, (__gpu_sg_Nr + block.y-1)/block.y);

  kernel_sg_force<<<grid1, block>>>(__gpu_sg_Nr, __gpu_sg_Nphi, real_pitch/sizeof (double), __gpu_sg_d_rii, __gpu_sg_d_rmed, d_sdens, d_pot, d_acc);
  __checkCudaKernelError ();
}



/*----------------------------------------------------------------------
  function for initialization of Poisson solver
----------------------------------------------------------------------*/
/*extern "C" void gpu_sg_init_polargrid (int Nr, int Nphi, double rmin, double rmax, 
                                       double *dr, double *dphi, 
                                       double *rii, double *rmed, double *drii,
                                       double *phii, double *phimed, 
                                       double *surf) {
  // radial cell interface coordinates
  for (int i=0; i< Nr+1; i++) {
    h_rii[i] = rmin * exp((double) i / (double) Nr * log(rmax / rmin));
  }

  // difeence in radial direction
  //for (int i=0; i< Nr; i++) {
  //  drii[i] = (rii[i+1]-rii[i]);
  //}

  // radial cell centre coordinates (naive approach)
  //for (int i=0; i< Nr; i++) {
  //  rmed[i] = (rii[i+1]+rii[i])/2.0;
  //}
  // radial cell centre coordinates (similar to what is used in FARGO)
  for (int i=0; i< Nr; i++) {
    h_rmed[i] = (2.0/3.0) * (rii[i+1]*rii[i+1]*rii[i+1]-rii[i]*rii[i]*rii[i])/(rii[i+1]*rii[i+1]-rii[i]*rii[i]);
  }
  
  // azimuthal cell intreface coordinates
  for (int j=0; j < Nphi+1; j++) {
    phii[j] = (double) j * 2.0 * M_PI/(double) (Nphi);
  }

  // azimuthal cell center coordinates
  phimed[0] = 0.0;
  for (int j=0; j < Nphi; j++) {
    phimed[j] = (phii[j+1] + phii[j])/2.0;
  }
  
  // dr and dphi
  *dr = (log (rmax) - log (rmin)) / (double) Nr;
  *dphi = 2.0 * M_PI / (double) Nphi;

  // cell surface area
  //for (int i=0; i< Nr; i++)
  //  for (int j=0; j< Nphi; j++) {
  //    surf[j+i*Nphi] = (h_rii[i+1]*h_rii[i+1]-h_rii[i]*h_rii[i]) * (*dphi);
      //surf[j+i*Nphi] = (rii[i+1]-rii[i]) * (*dphi);
  //}
  
  printf ("Polargrid is initialized\n");
}*/



/*----------------------------------------------------------------------
  stuff for benchmarking
------------------------------------------------------------------------*/
#ifdef GPU_SG_BENCHMARK
double gpu_sg_get_time() {
  struct timeval Tvalue;
  struct timezone dummy;
  gettimeofday(&Tvalue,&dummy);
  return ((double) Tvalue.tv_sec + 1.e-6*((double) Tvalue.tv_usec));
}

float gpu_run_benchmark (int Nr, int Nphi, double dr, double dphi, double* rmed, double *sdens) {
    
  double *h_pot, *h_sdens;  // potential and sirface mass density stored in host
  double *d_pot, *d_sdens;  // potential and sirface mass density stored in host
  
  size_t d_sdens_pitch, d_pot_pitch;

  // allocate and upload density to device
  checkCudaErrors(cudaMallocHost ((void **) &h_sdens, sizeof(double) * Nphi * Nr));  
  for (int i=0; i < Nr; i++)
    for (int j=0; j < Nphi; j++) {
      h_sdens[j + i*Nphi] = sdens[j + i*Nphi];
    }
  checkCudaErrors(cudaMallocPitch ((void **) &d_sdens, &d_sdens_pitch, sizeof(double) * Nphi, Nr));
  checkCudaErrors(cudaMemcpy2D(d_sdens, d_sdens_pitch, h_sdens, Nphi * sizeof(double), Nphi * sizeof(double), Nr, cudaMemcpyHostToDevice));

  // allocate and upload potential to device
  checkCudaErrors(cudaMallocHost ((void **) &h_pot, sizeof(double) * Nphi * Nr));  
  checkCudaErrors(cudaMallocPitch ((void **) &d_pot, &d_pot_pitch, sizeof(double) * Nphi, Nr));

  // initialization of self-gravity module
  gpu_sg_init (Nr, Nphi, dr, dphi, rmed);

  int max_BLOCK_X=0, max_BLOCK_Y=0;
  float t_gpu, t_min=1e33;
  for (int k =1; k<=128; k*=2) {
    for (int m =1; m<=128; m*=2) {
      if (k*m<=1024) {

        BLOCK_X = k;
        BLOCK_Y = m;
        double t0_gpu = gpu_sg_get_time ();
        for (int i=0; i<10; i++) {
          gpu_sg_calc_pot (d_sdens_pitch, d_sdens, d_pot);
        }
        t_gpu = (gpu_sg_get_time ()-t0_gpu)/10.0; 
        if (t_gpu < t_min) {
          t_min = t_gpu;
          max_BLOCK_X = BLOCK_X;
          max_BLOCK_Y = BLOCK_Y;
        }
        printf ("[%3i %3i]\t t:%f msec\n", BLOCK_X, BLOCK_Y, t_gpu*1000.);
      }
    }
  }
  
  printf("Optimal settings: [BLOCK_X %i BLOCKY %i]\n", t_gpu, max_BLOCK_X, max_BLOCK_Y);
  checkCudaErrors(cudaMemcpy2D(h_pot, Nphi * sizeof(double), d_pot, d_pot_pitch, Nphi * sizeof(double), Nr, cudaMemcpyDeviceToHost));
  disp_matrix (Nr, Nphi, h_pot, "gpu fftw pot", 1.0);
  
  return (t_gpu);
}
#endif
