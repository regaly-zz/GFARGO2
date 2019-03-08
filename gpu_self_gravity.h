/*-----------------------------------------------------------------------------
 GPU-based 2D Poission solver for self-gravity

 Definitions of functions

 Written by Zs. Regaly 2016 (www.konkoly.hu/staff/regaly)
 Computational Astrophysics Group of Konkoly Observatory (www.konkoly.hu/CAG/)
------------------------------------------------------------------------------*/

#ifndef SELF_GRAVITY_H
#define SELF_GRAVITY_H

#include <cufft.h>


#ifdef GPU_SG_BENCHMARK
double gpu_sg_get_time();
extern "C" float gpu_run_benchmark (int Nr, int Nphi, double dr, double dphi, double* rmed, double *sdens);
#endif

// CUDA error checking if defined
void inline __checkCudaKernelError ();

// simple host memory allocation procedurer
void __gpu_sg_allocate_host_array (double ***p, int Ncol, int Nrow);

// allocation of memory
void gpu_sg_allocmem (int Nr, int Nphi);

// initialization of polar grid required for self gravity calculation
extern "C" void gpu_sg_init_polargrid (int Nr, int Nphi, double rmin, double rmax, 
                                       double *dr, double *dphi, 
                                       double *rii, double *rmed, double *drii,
                                       double *phii, double *phimed, 
                                       double *surf);

// initialization of polar grid required for self gravity acceleration
void gpu_sg_init_polargrid (int Nr, int Nphi, double rmin, double rmax);

// initialization of constants
void gpu_sg_init_const (double eps);

// initialization of FFTW matrices
void gpu_sg_init_matrix ();

// function which can be called externally
// initialization of self-gravity solver
extern "C" void gpu_sg_init (int Nr, int Nphi, double rmin, double rmax, double eps);

// function for calculating the potential
extern "C" void gpu_sg_calc_pot (size_t real_pitch, double *sdens,  double *d_pot);

// function for calculateing acceleration due to self-gravity
extern "C" void gpu_sg_calc_acc (size_t real_pitch, double *d_sdens, double *d_pot, double *d_acc);

// function for summing two densities (e.g. gas+dust)
extern "C" void gpu_sg_add_dens (size_t real_pitch, double *d_dens1, double *d_dens2, double *d_densres);

#endif