#define __CUDA 1
#include "fargo.h"
#undef __CUDA
#include <stdarg.h>
#include <helper_cuda.h>
#include <cuda.h>

#ifdef FARGO_INTEGRATION

// BLOCK_X : in azimuth
//#define BLOCK_X DEF_BLOCK_X_GBLINEARINTERPOL
#define BLOCK_X 16
// BLOCK_Y : in radius
#define BLOCK_Y 16
#include "global_ex.h"

extern PolarGrid *gas_density, *gas_v_rad, *gas_v_theta, *CoarseDust, *DustDens;//, *DustMass;
extern double *gpu_surf;

bool bInitGasBiCubicInterpol = false;

int GasBiCubicInterpol_Nsec = 0;
int GasBiCubicInterpol_Nrad = 0;
int GasBiCubicInterpol_UpScalingRad = 0;
int GasBiCubicInterpol_UpScalingAzim = 0;

float GasBiCubicInterpol_cx;    
float GasBiCubicInterpol_cy;    
float GasBiCubicInterpol_scale_x;
float GasBiCubicInterpol_scale_y;
float GasBiCubicInterpol_tx;    
float GasBiCubicInterpol_ty;    

bool bInitDustBiCubicInterpol = false;
int DustBiCubicInterpol_Nsec = 0;
int DustBiCubicInterpol_Nrad = 0;
int DustBiCubicInterpol_UpScaling = 0;

float DustBiCubicInterpol_cx;    
float DustBiCubicInterpol_cy;    
float DustBiCubicInterpol_scale; 
float DustBiCubicInterpol_tx;    
float DustBiCubicInterpol_ty;    

double *GPU_GasBiCubicInterpol_VelKep;

// coarse array (must be float for image array!)
float *coarse_gas_array, *coarse_dust_array;


// the texture memory
texture<float, 2, cudaReadModeElementType > texGas;
texture<float, 2, cudaReadModeElementType > texDust;

// CUDA image Array
cudaArray *d_imageGasArray = 0;
cudaArray *d_imageDustArray = 0;

// 
__global__ void dev_copyDouble2Float(float* array, double *arrayd, uint width, uint height) { 
  uint const x = blockIdx.x * blockDim.x + threadIdx.x;
  uint const y = blockIdx.y * blockDim.y + threadIdx.y;
  
  if ((x < width) && (y < height)) {
    if (x == 0)
      array[y*width + x] = (float) arrayd[y*(width-2) + width - 3];
    else if (x == width - 1)
      array[y*width + x] = (float) arrayd[y*(width-2)];
    else
      array[y*width + x] = (float) arrayd[y*(width-2) + x - 1];    
  }
}

__global__ void dev_copyDouble2Float_add(float* array, double *arrayd, double* vel_kep, uint width, uint height) { 
  uint const x = blockIdx.x * blockDim.x + threadIdx.x;
  uint const y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x < width) && (y < height)) {
    if (x == 0)
      array[y*width + x] = (float) arrayd[y*(width-2) + width - 3] + vel_kep[y];
    else if (x == width - 1)
      array[y*width + x] = (float) arrayd[y*(width-2)] + vel_kep[y];
    else
      array[y*width + x] = (float) arrayd[y*(width-2) + x - 1] + vel_kep[y];
  }
}

// 
__global__ void dev_CalcDustDens(double* out, double *in1, double *in2, uint width, uint height) { 
  uint const x = blockIdx.x * blockDim.x + threadIdx.x;
  uint const y = blockIdx.y * blockDim.y + threadIdx.y;
  uint const i = y*width + x;
  
  if ((x < width) && (y < height)) {
    out[i] = in1[i]/in2[y];    
  }
}

// w0, w1, w2, and w3 are the four cubic B-spline basis functions
__host__ __device__ float w0(float a) {
    //    return (1.0f/6.0f)*(-a*a*a + 3.0f*a*a - 3.0f*a + 1.0f);
 return (1.0f/6.0f)*(a*(a*(-a + 3.0f) - 3.0f) + 1.0f);   // optimized
}

__host__ __device__ float w1(float a) {
    //    return (1.0f/6.0f)*(3.0f*a*a*a - 6.0f*a*a + 4.0f);
 return (1.0f/6.0f)*(a*a*(3.0f*a - 6.0f) + 4.0f);
}

__host__ __device__ float w2(float a) {
    //    return (1.0f/6.0f)*(-3.0f*a*a*a + 3.0f*a*a + 3.0f*a + 1.0f);
 return (1.0f/6.0f)*(a*(a*(-3.0f*a + 3.0f) + 3.0f) + 1.0f);
}

__host__ __device__ float w3(float a) {
 return (1.0f/6.0f)*(a*a*a);
}

// filter 4 values using cubic splines
template<class T> __device__ T cubicFilter(float x, T c0, T c1, T c2, T c3) {
    T r;
    r  = c0 * w0(x);
    r += c1 * w1(x);
    r += c2 * w2(x);
    r += c3 * w3(x);
    return r;
}



// slow but precise bicubic lookup using 16 texture lookups
// texture data type, return type
template<class T, class R> __device__ R tex2DBicubic(const texture<T, 2, cudaReadModeElementType > texref, float x, float y) {
  x -= 0.5f;
  y -= 0.5f;
  float px = floor(x);
  float py = floor(y);
  float fx = x - px;
  float fy = y - py;

  return cubicFilter<R>(fy,
                        cubicFilter<R>(fx, tex2D(texref, px-1, py-1), tex2D(texref, px, py-1), tex2D(texref, px+1, py-1), tex2D(texref, px+2,py-1)),
                        cubicFilter<R>(fx, tex2D(texref, px-1, py),   tex2D(texref, px, py),   tex2D(texref, px+1, py),   tex2D(texref, px+2, py)),
                        cubicFilter<R>(fx, tex2D(texref, px-1, py+1), tex2D(texref, px, py+1), tex2D(texref, px+1, py+1), tex2D(texref, px+2, py+1)),
                        cubicFilter<R>(fx, tex2D(texref, px-1, py+2), tex2D(texref, px, py+2), tex2D(texref, px+1, py+2), tex2D(texref, px+2, py+2))
                       );
}

// render image using bicubic texture lookup
__global__ void dev_GasBiCubicInterpol (double *d_output, uint width, uint height, float tx, float ty, float scale_x, float scale_y, float cx, float cy) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint i = y * width + x;

    float u = (x-cx)*scale_x+cx + tx;
    float v = (y-cy)*scale_y+cy + ty;

    if ((x < width) && (y < height))
      d_output[i] = (double )tex2DBicubic<float, float>(texGas, u, v);
}

// render image using bicubic texture lookup
__global__ void dev_DustBiCubicInterpol (double *d_output, uint width, uint height, float tx, float ty, float scale, float cx, float cy) {
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    uint i = y * width + x;

    float u = (x-cx)*scale+cx + tx;
    float v = (y-cy)*scale+cy + ty;

    if ((x < width) && (y < height))
      d_output[i] = (double )tex2DBicubic<float, float>(texDust, u, v);
}


extern "C" void InitGasBiCubicInterpol (int nsec, int nrad, int upscaling_rad, int upscaling_azim) {

  // upscaling > 1 then setup bicubic interpolation
  if (upscaling_rad >= 2 || upscaling_azim >= 2) {
    GasBiCubicInterpol_Nrad = nrad;
    GasBiCubicInterpol_Nsec = nsec;
    GasBiCubicInterpol_UpScalingRad = upscaling_rad;
    GasBiCubicInterpol_UpScalingAzim = upscaling_azim;
        
    // allocate coarse temporary arrays for coarse (float) array
    checkCudaErrors(cudaMalloc(&coarse_gas_array, (GasBiCubicInterpol_Nsec+2)*GasBiCubicInterpol_Nrad*sizeof(float)));
    
    // allocate array for fine grid arrays
    size_t pitch;
    checkCudaErrors(cudaMallocPitch((void **)&fine_gas_density, &pitch, GasBiCubicInterpol_UpScalingAzim*GasBiCubicInterpol_Nsec*sizeof(double), GasBiCubicInterpol_UpScalingRad*GasBiCubicInterpol_Nrad));
    checkCudaErrors(cudaMallocPitch((void **)&fine_gas_v_rad,   &pitch, GasBiCubicInterpol_UpScalingAzim*GasBiCubicInterpol_Nsec*sizeof(double), GasBiCubicInterpol_UpScalingRad*GasBiCubicInterpol_Nrad));
    checkCudaErrors(cudaMallocPitch((void **)&fine_gas_v_theta, &pitch, GasBiCubicInterpol_UpScalingAzim*GasBiCubicInterpol_Nsec*sizeof(double), GasBiCubicInterpol_UpScalingRad*GasBiCubicInterpol_Nrad));
    checkCudaErrors(cudaMallocPitch((void **)&fine_gas_dv_rad, &pitch, GasBiCubicInterpol_UpScalingAzim*GasBiCubicInterpol_Nsec*sizeof(double), GasBiCubicInterpol_UpScalingRad*GasBiCubicInterpol_Nrad));
    checkCudaErrors(cudaMallocPitch((void **)&fine_gas_dv_theta, &pitch, GasBiCubicInterpol_UpScalingAzim*GasBiCubicInterpol_Nsec*sizeof(double), GasBiCubicInterpol_UpScalingRad*GasBiCubicInterpol_Nrad));        
    // allocate image array
    checkCudaErrors(cudaMallocArray(&d_imageGasArray, &texGas.channelDesc, GasBiCubicInterpol_Nsec + 2, GasBiCubicInterpol_Nrad));

    double *GasBiCubicInterpol_VelKep;
    checkCudaErrors(cudaMallocHost ((void **) &GasBiCubicInterpol_VelKep, sizeof(double) * NRAD));
    checkCudaErrors(cudaMalloc ((void **) &GPU_GasBiCubicInterpol_VelKep, sizeof(double) * NRAD));
    for (int i=0; i< NRAD; i++) {
      GasBiCubicInterpol_VelKep[i] = rsqrt (Rmed[i]);
    }  
    checkCudaErrors(cudaMemcpy(GPU_GasBiCubicInterpol_VelKep, GasBiCubicInterpol_VelKep,  sizeof(double) * NRAD, cudaMemcpyHostToDevice));

    
    // set texture parameters
    texGas.addressMode[0] = cudaAddressModeClamp;
    texGas.addressMode[1] = cudaAddressModeClamp;
    texGas.filterMode     = cudaFilterModePoint;
    texGas.normalized     = false;
    
    // setup scaling  parameters    
    checkCudaErrors(cudaBindTextureToArray(texGas, d_imageGasArray)); 
    GasBiCubicInterpol_cx      = (float) (GasBiCubicInterpol_Nsec + 2)* 0.5f;
    GasBiCubicInterpol_cy      = (float) GasBiCubicInterpol_Nrad * 0.5f;
    GasBiCubicInterpol_scale_x = 1.0/((float)GasBiCubicInterpol_UpScalingAzim);
    GasBiCubicInterpol_scale_y = 1.0/((float)GasBiCubicInterpol_UpScalingRad);
//    GasBiCubicInterpol_tx    = -(GasBiCubicInterpol_Nsec*GasBiCubicInterpol_UpScaling * 0.5f - GasBiCubicInterpol_cx)*GasBiCubicInterpol_scale;
//    GasBiCubicInterpol_ty    = -(GasBiCubicInterpol_Nrad*GasBiCubicInterpol_UpScaling * 0.5f - GasBiCubicInterpol_cy)*GasBiCubicInterpol_scale;
    GasBiCubicInterpol_tx    = -(GasBiCubicInterpol_Nsec*GasBiCubicInterpol_UpScalingAzim * 0.5f - GasBiCubicInterpol_cx)*GasBiCubicInterpol_scale_x;
    GasBiCubicInterpol_ty    = -(GasBiCubicInterpol_Nrad* GasBiCubicInterpol_UpScalingRad * 0.5f - GasBiCubicInterpol_cy)*GasBiCubicInterpol_scale_y;
    
    // check error
    getLastCudaError("InitGasBiCubicInterpol");

    // interpolation is ready
    bInitGasBiCubicInterpol = true;
  }
  // no bicubic interpolation
  else 
    bInitGasBiCubicInterpol = false;
}



extern "C" void GasBiCubicInterpol () {

  // bicubic interpolation
  if (bInitGasBiCubicInterpol) {
    dim3 blockSize(BLOCK_X, BLOCK_Y);
    dim3 gridSizeCoarse(ceil((float)(GasBiCubicInterpol_Nsec+2) / blockSize.x), GasBiCubicInterpol_Nrad / blockSize.y);
    dim3 gridSizeFine(GasBiCubicInterpol_UpScalingAzim*GasBiCubicInterpol_Nsec / blockSize.x, GasBiCubicInterpol_UpScalingRad*GasBiCubicInterpol_Nrad / blockSize.y);
    int size = (GasBiCubicInterpol_Nsec+2)*GasBiCubicInterpol_Nrad*sizeof(float);

    // bicubic interpolation of density (copy density to coarse grid and bicubic interpolate onto fine array)
    dev_copyDouble2Float<<<gridSizeCoarse, blockSize>>>(coarse_gas_array, gas_density->gpu_field, GasBiCubicInterpol_Nsec+2, GasBiCubicInterpol_Nrad);
    getLastCudaError("copyDouble2Float kernel failed");
    checkCudaErrors(cudaMemcpyToArray(d_imageGasArray, 0, 0, coarse_gas_array, size, cudaMemcpyDeviceToDevice));    
    //checkCudaErrors(cudaBindTextureToArray(texGas, d_imageGasArray));  /// NEEDED??????????
    dev_GasBiCubicInterpol<<<gridSizeFine, blockSize>>>(fine_gas_density, GasBiCubicInterpol_UpScalingAzim*GasBiCubicInterpol_Nsec, GasBiCubicInterpol_UpScalingRad*GasBiCubicInterpol_Nrad, 
                                                        GasBiCubicInterpol_tx, GasBiCubicInterpol_ty, GasBiCubicInterpol_scale_x, GasBiCubicInterpol_scale_y, GasBiCubicInterpol_cx, GasBiCubicInterpol_cy);
    getLastCudaError("GasBiCubicInterpol kernel failed");
    
    // bicubic interpolation of radial velocity (copy radial velocity to coarse grid and bicubic interpolate onto fine array)
    dev_copyDouble2Float<<<gridSizeCoarse, blockSize>>>(coarse_gas_array, gas_v_rad->gpu_field, GasBiCubicInterpol_Nsec+2, GasBiCubicInterpol_Nrad);
    checkCudaErrors(cudaMemcpyToArray(d_imageGasArray, 0, 0, coarse_gas_array, size, cudaMemcpyDeviceToDevice));    
    //checkCudaErrors(cudaBindTextureToArray(texGas, d_imageGasArray));  /// NEEDED??????????
    dev_GasBiCubicInterpol<<<gridSizeFine, blockSize>>>(fine_gas_v_rad, GasBiCubicInterpol_UpScalingAzim*GasBiCubicInterpol_Nsec, GasBiCubicInterpol_UpScalingRad*GasBiCubicInterpol_Nrad,
                                                        GasBiCubicInterpol_tx, GasBiCubicInterpol_ty, GasBiCubicInterpol_scale_x, GasBiCubicInterpol_scale_y, GasBiCubicInterpol_cx, GasBiCubicInterpol_cy);
    getLastCudaError("GasBiCubicInterpol kernel failed");
    
    // bicubic interpolation of azimuthal velocity (copy azimuthal velocity to coarse grid and bicubic interpolate onto fine array)
    //dev_copyDouble2Float<<<gridSizeCoarse, blockSize>>>(coarse_gas_array, gas_v_theta->gpu_field, GasBiCubicInterpol_Nsec+2, GasBiCubicInterpol_Nrad);
    dev_copyDouble2Float_add<<<gridSizeCoarse, blockSize>>>(coarse_gas_array, gas_v_theta->gpu_field, GPU_GasBiCubicInterpol_VelKep, GasBiCubicInterpol_Nsec+2, GasBiCubicInterpol_Nrad);
    checkCudaErrors(cudaMemcpyToArray(d_imageGasArray, 0, 0, coarse_gas_array, size, cudaMemcpyDeviceToDevice));    
    //checkCudaErrors(cudaBindTextureToArray(texGas, d_imageGasArray));  /// NEEDED??????????
    dev_GasBiCubicInterpol<<<gridSizeFine, blockSize>>>(fine_gas_v_theta, GasBiCubicInterpol_UpScalingAzim*GasBiCubicInterpol_Nsec, GasBiCubicInterpol_UpScalingRad*GasBiCubicInterpol_Nrad,
                                                     GasBiCubicInterpol_tx, GasBiCubicInterpol_ty, GasBiCubicInterpol_scale_x, GasBiCubicInterpol_scale_y, GasBiCubicInterpol_cx, GasBiCubicInterpol_cy);
    getLastCudaError("GasBiCubicInterpol kernel failed");
    
    // bicubic interpolation of radial velocity (copy radial velocity to coarse grid and bicubic interpolate onto fine array)
    dev_copyDouble2Float<<<gridSizeCoarse, blockSize>>>(coarse_gas_array, gas_dv_rad->gpu_field, GasBiCubicInterpol_Nsec+2, GasBiCubicInterpol_Nrad);
    checkCudaErrors(cudaMemcpyToArray(d_imageGasArray, 0, 0, coarse_gas_array, size, cudaMemcpyDeviceToDevice));    
    //checkCudaErrors(cudaBindTextureToArray(texGas, d_imageGasArray));  /// NEEDED??????????
    dev_GasBiCubicInterpol<<<gridSizeFine, blockSize>>>(fine_gas_dv_rad, GasBiCubicInterpol_UpScalingAzim*GasBiCubicInterpol_Nsec, GasBiCubicInterpol_UpScalingRad*GasBiCubicInterpol_Nrad,
                                                        GasBiCubicInterpol_tx, GasBiCubicInterpol_ty, GasBiCubicInterpol_scale_x, GasBiCubicInterpol_scale_y, GasBiCubicInterpol_cx, GasBiCubicInterpol_cy);
    getLastCudaError("GasBiCubicInterpol kernel failed");

    // bicubic interpolation of radial velocity (copy radial velocity to coarse grid and bicubic interpolate onto fine array)
    dev_copyDouble2Float<<<gridSizeCoarse, blockSize>>>(coarse_gas_array, gas_dv_theta->gpu_field, GasBiCubicInterpol_Nsec+2, GasBiCubicInterpol_Nrad);
    checkCudaErrors(cudaMemcpyToArray(d_imageGasArray, 0, 0, coarse_gas_array, size, cudaMemcpyDeviceToDevice));    
    //checkCudaErrors(cudaBindTextureToArray(texGas, d_imageGasArray));  /// NEEDED??????????
    dev_GasBiCubicInterpol<<<gridSizeFine, blockSize>>>(fine_gas_dv_theta, GasBiCubicInterpol_UpScalingAzim*GasBiCubicInterpol_Nsec, GasBiCubicInterpol_UpScalingRad*GasBiCubicInterpol_Nrad,
                                                        GasBiCubicInterpol_tx, GasBiCubicInterpol_ty, GasBiCubicInterpol_scale_x, GasBiCubicInterpol_scale_y, GasBiCubicInterpol_cx, GasBiCubicInterpol_cy);
    getLastCudaError("GasBiCubicInterpol kernel failed");

  }
  // no bicubic interpolation
  else {
    fine_gas_density = gas_density->gpu_field;
    fine_gas_v_rad   = gas_v_rad->gpu_field;
    fine_gas_v_theta = gas_v_theta->gpu_field;
  //  fine_gradpdivsigma = GradPdivSigma->gpu_field;
  }
}


extern "C" void InitDustBiCubicInterpol (int nsec, int nrad, int upscaling) {

  // upscaling => 1 then setup bicubic interpolation
  if (upscaling >=2 ) {
  
    DustBiCubicInterpol_Nrad = nrad;
    DustBiCubicInterpol_Nsec = nsec;
    DustBiCubicInterpol_UpScaling = upscaling;
  
    // allocate coarse temporary arrays for coarse (float) array
    checkCudaErrors(cudaMalloc(&coarse_dust_array, (DustBiCubicInterpol_Nsec+2)*DustBiCubicInterpol_Nrad*sizeof(float)));
          
    // allocate image array
    checkCudaErrors(cudaMallocArray(&d_imageDustArray, &texDust.channelDesc, DustBiCubicInterpol_Nsec+2, DustBiCubicInterpol_Nrad));
  
    // set texture parameters
    texDust.addressMode[0] = cudaAddressModeClamp;
    texDust.addressMode[1] = cudaAddressModeClamp;
    texDust.filterMode     = cudaFilterModePoint;
    texDust.normalized     = false;
  
    // setup scaling  parameters
    checkCudaErrors(cudaBindTextureToArray(texDust, d_imageDustArray)); 
    DustBiCubicInterpol_cx    =  (DustBiCubicInterpol_Nsec + 2) * 0.5f;
    DustBiCubicInterpol_cy    =  (DustBiCubicInterpol_Nrad) * 0.5f;  
    DustBiCubicInterpol_scale =  1.0/(float) DustBiCubicInterpol_UpScaling;
    DustBiCubicInterpol_tx    = -(DustBiCubicInterpol_Nsec*DustBiCubicInterpol_UpScaling * 0.5f - DustBiCubicInterpol_cx)*DustBiCubicInterpol_scale;
    DustBiCubicInterpol_ty    = -(DustBiCubicInterpol_Nrad*DustBiCubicInterpol_UpScaling * 0.5f - DustBiCubicInterpol_cy)*DustBiCubicInterpol_scale;
      
    // check error
    getLastCudaError("InitGasBiCubicInterpol");
    
    // interpolation is ready
    bInitDustBiCubicInterpol = true;
  }
  // no bicubic interpolation
  else {
    // allocate coarse temporary arrays for coarse (float) array
    //checkCudaErrors(cudaMalloc(&coarse_dust_array, nsec*nrad*sizeof(double)));

    //CoarseDust->gpu_field = DustDens->gpu_field;
    
    // interpolation is turned off
    bInitDustBiCubicInterpol = false; 
  }
}


extern "C" void DustBiCubicInterpol () {
  return;
  if (bInitDustBiCubicInterpol) {

    dim3 blockSize(BLOCK_X, BLOCK_Y);
    dim3 gridSizeFine(DustBiCubicInterpol_UpScaling*DustBiCubicInterpol_Nsec / blockSize.x, DustBiCubicInterpol_UpScaling*DustBiCubicInterpol_Nrad / blockSize.y);

    dim3 gridSizeCoarse(ceil((float)(DustBiCubicInterpol_Nsec+2) / blockSize.x), DustBiCubicInterpol_Nrad / blockSize.y);
    int size = (DustBiCubicInterpol_Nsec+2)*DustBiCubicInterpol_Nrad*sizeof(float);
  
    // bicubic interpolation of density (copy density to coarse grid and bicubic interpolate onto fine array)
    dev_copyDouble2Float<<<gridSizeCoarse, blockSize>>>(coarse_dust_array, CoarseDust->gpu_field, DustBiCubicInterpol_Nsec+2, DustBiCubicInterpol_Nrad);
    getLastCudaError("copyDouble2Float kernel failed");
    checkCudaErrors(cudaMemcpyToArray(d_imageDustArray, 0, 0, coarse_dust_array, size, cudaMemcpyDeviceToDevice));    
    checkCudaErrors(cudaBindTextureToArray(texDust, d_imageDustArray)); 
    dev_DustBiCubicInterpol<<<gridSizeFine, blockSize>>>(DustDens->gpu_field, DustBiCubicInterpol_UpScaling*DustBiCubicInterpol_Nsec, DustBiCubicInterpol_UpScaling*DustBiCubicInterpol_Nrad, 
                                                         DustBiCubicInterpol_tx, DustBiCubicInterpol_ty, DustBiCubicInterpol_scale, DustBiCubicInterpol_cx, DustBiCubicInterpol_cy);
    getLastCudaError("DustBiCubicInterpol kernel failed");
  }
  // no bicubic interpolation
  else
      DustDens->gpu_field = CoarseDust->gpu_field;
  
  
    //DustMass = CoarseDust;
//    DustDens = CoarseDust;

//  cudaDeviceSynchronize ();
//  dev_CalcDustDens<<<gridSizeFine, blockSize>>>(DustDens->gpu_field, DustMass->gpu_field, gpu_surf,
//                                                DustBiCubicInterpol_UpScaling*DustBiCubicInterpol_Nsec, DustBiCubicInterpol_UpScaling*DustBiCubicInterpol_Nrad);
//  getLastCudaError("CalcDustDens kernel failed");
}

#endif

