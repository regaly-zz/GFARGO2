/** \file LowTasks.cu

Contains many low level functions.
*/
#define __CUDA 1
#include "fargo.h"
#undef __CUDA
#include <stdarg.h>
#include <helper_cuda.h>
#include <cuda.h>

// BLOCK_X : in azimuth
//#define BLOCK_X DEF_BLOCK_X_FORCE
#define BLOCK_X 16
// BLOCK_Y : in radius
#define BLOCK_Y 8

PolarGrid *ListOfGrids = NULL;

static long VIDEORAM = 0;

extern "C" void SelectDevice(int number) {
  printf ("GPU device #%i trying to activate...", number);
  fflush (stdout);
  checkCudaErrors(cudaSetDevice(number));
  getLastCudaError ("Device initialization failed!\n");
  printf ("done\n");
  
  if (verbose) {
    cudaDeviceProp myProp;
    cudaGetDeviceProperties (&myProp, number);

    printf ("\nDevice properties\n");
    printf ("------------------------------------------------------------------------------\n");
    printf ("Name:                %s\n", myProp.name);
    printf ("Global Memory:       %0.2f Gbyte\n", (double) myProp.totalGlobalMem /1024.0/1024.0 /1024.0);
    printf ("Shared Memory:       %d byte\n", (int) myProp.sharedMemPerBlock);
    printf ("Const Memory:        %d byte\n", (int) myProp.totalConstMem);
    printf ("Maxthred per block:  %d\n", myProp.maxThreadsPerBlock);
    printf ("Number of cores:     %d\n", myProp.multiProcessorCount);
    printf ("Compute mode:        %d\n", myProp.computeMode);
    printf ("ECC enabled:         %d\n", myProp.ECCEnabled);
    printf ("PCI bus ID:          %d\n", myProp.pciBusID);
    printf ("PCI device ID:       %d\n\n", myProp.pciDeviceID);
  }
}

extern "C" PolarGrid    *CreatePolarGrid(int Nr, int Ns, const char *name) {
  PolarGrid *array;
  double *field;
  double *gpu_field;
  char *string;
  int i, j, l;
  size_t pitch;

  if (verbose)
    //printf (" * %s \t\t dim:[%ix%i]\n", name, Ns, Nr);
    printf (" * %20s dim:[%ix%i]\n", name, Nr, Ns);
  
  array = (PolarGrid *) malloc(sizeof(PolarGrid));
  if (array == NULL) {
    printf("Insufficient memory for PolarGrid creation\n");
    exit (-1);
  }
  field = (double *) malloc(sizeof(double) * (Nr + 1) * Ns);
  if (field == NULL) {
    printf("Insufficient memory for PolarGrid creation\n");
    exit (-1);
  }
  checkCudaErrors(cudaMallocPitch ((void **)&gpu_field, &pitch, Ns*sizeof(double), Nr+1));
  VIDEORAM += sizeof(double)*(Nr+1)*Ns;
  string = (char *) malloc(sizeof(char) * 80);
  if (string == NULL) {
    printf ("Insufficient memory for PolarGrid creation\n");
    exit (-1);
  }
  sprintf(string, "%s", name);
  array->Field = field;
  array->pitch = pitch;
  array->gpu_field = gpu_field;
  array->Name = string;
  array->Nrad = Nr;
  array->Nsec = Ns;
  for (i = 0; i <= Nr; i++) {
    for (j = 0; j < Ns; j++) {
      l = j + i*Ns;
      field[l] = 0.;
    }
  }
  array->next = ListOfGrids;
  ListOfGrids = array;
  H2D (array);
  return array;
}

/*
__global__ void kernel_calc_div (const double *polar_grid0, const double *polar_grid1, const int pitch, double *res) {
  int jg = threadIdx.x + blockIdx.x * blockDim.x;
  int ig = threadIdx.y + blockIdx.y * blockDim.y;
  int idg = __mul24(ig, pitch) + jg;
  
  res[idg] = (polar_grid1[idg]-polar_grid0[idg]);
}
  
extern "C" void CalcDiv_gpu (PolarGrid *polar_grid0, PolarGrid *polar_grid1, PolarGrid *res) {
  dim3 block (BLOCK_X, BLOCK_Y);
  dim3 grid ((NSEC+block.x-1)/block.x, (NRAD+block.y-1)/block.y);
  
  kernel_calc_div <<< grid, block >>> (polar_grid0->gpu_field, polar_grid1->gpu_field, res->pitch/sizeof(double), res->gpu_field);
  
  getLastCudaError("InitGasBiCubicInterpol");
}
*/

extern "C" void ActualiseGas_gpu (PolarGrid* old, PolarGrid* neww) {
  //below : neww, not new, which is a reserved keyword in C++...
  checkCudaErrors (cudaMemcpy2D ((void *)old->gpu_field, old->pitch,
                                 (void *)neww->gpu_field, neww->pitch,
                                 old->Nsec*sizeof(double),
                                 old->Nrad+1,
                                 cudaMemcpyDeviceToDevice));
}

extern "C" void H2D (PolarGrid *pg) {
  //  printf ("Copying %s from host to device\n", pg->Name);
  checkCudaErrors(cudaMemcpy2D (pg->gpu_field, pg->pitch, pg->Field, pg->Nsec*sizeof(double),
                                pg->Nsec*sizeof(double), 
                                pg->Nrad+1, cudaMemcpyHostToDevice));
}

extern "C" void D2H (PolarGrid *pg) {
  //  printf ("Copying %s from device to host\n", pg->Name);
  if (verbose) {
    printf (" * Downloading from GPU %s\n", pg->Name);
  }
  checkCudaErrors(cudaMemcpy2D ((void *)pg->Field, pg->Nsec*sizeof(double), 
                                (void *)pg->gpu_field, pg->pitch,
                                pg->Nsec*sizeof(double), pg->Nrad+1, cudaMemcpyDeviceToHost));
}

extern "C" void H2D_All () {
  PolarGrid *g;
  g = ListOfGrids;

  if (verbose) {
    printf ("\nUploading PolarGrids to GPU\n");
    printf ("------------------------------------------------------------------------------\n");
  }
  while (g != NULL) {
    // add some logging
    if (verbose) {
      printf (" * %s\n", g->Name);
      fflush (stdout);
    }
    H2D (g);
    g = g->next;
  }
}

extern "C" void D2H_All () {
  PolarGrid *g;
  g = ListOfGrids;
  if (verbose) {
    printf ("\nDownloading PolarGrids from GPU\n");
    printf ("------------------------------------------------------------------------------\n");
  }
  while (g != NULL) {
    // [RZS-MOD]
    // add some logging
    printf (" * %s\n", g->Name);
    fflush (stdout);
    D2H (g);
    g = g->next;
  }
  if (verbose)
    printf ("Finished\n");
}

void PrintVideoRAMUsage () {
  if (verbose) {
    printf ("\nGPU ram usage\n");
    printf ("------------------------------------------------------------------------------\n");
    printf ("Memory allocated      :%.3f Mbytes\n", (double)VIDEORAM/1024.f/1024.f);
  }
}
