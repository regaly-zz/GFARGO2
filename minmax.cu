#define HUGENUM 1e33

// calculate the minimum and maxiumum simulaneously
// requires ? FLOPS
template <unsigned int blockSize, bool nIsPow2>
__device__ __forceinline__ void dev_reduction_minmax_template (const int      nj,
                                                               const double  *in,
                                                                     double2 *out) {
  extern __shared__ double2 sMinMax [];

  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockSize * 2 + threadIdx.x;
  unsigned int gridSize = blockSize * 2 * gridDim.x;

  double pMax = 0;
  double pMin = HUGENUM;
  while (i < nj) {
    if (nIsPow2 || i + blockSize < nj) {
      pMin = fmin(pMin, fmin ((in[i+blockSize] < 0 ? HUGENUM : in[i+blockSize]), (in[i] < 0 ? HUGENUM : in[i])));
      pMax = fmax(pMax, fmax ((in[i+blockSize] < 0 ? 0 : in[i+blockSize]), (in[i] < 0 ? 0 : in[i])));

    }
    else {
      pMin = fmax(pMin, (in[i] < 0 ? HUGENUM : in[i]));
      pMax = fmax(pMax, (in[i] < 0 ? 0 : in[i]));
    }

    i += gridSize;
  }
  sMinMax[tid].x = pMin;
  sMinMax[tid].y = pMax;
  __syncthreads ();

  if (blockSize >= 512) {if (tid < 256) {sMinMax[tid].x = fmin(sMinMax[tid].x, sMinMax[tid + 256].x); sMinMax[tid].y = fmax(sMinMax[tid].y, sMinMax[tid + 256].y); } __syncthreads ();}
  if (blockSize >= 256) {if (tid < 128) {sMinMax[tid].x = fmin(sMinMax[tid].x, sMinMax[tid + 128].x); sMinMax[tid].y = fmax(sMinMax[tid].y, sMinMax[tid + 128].y);} __syncthreads ();}
  if (blockSize >= 128) {if (tid <  64) {sMinMax[tid].x = fmin(sMinMax[tid].x, sMinMax[tid +  64].x); sMinMax[tid].y = fmax(sMinMax[tid].y, sMinMax[tid +  64].y);} __syncthreads ();}

  if (tid < 32) {
    volatile double2* vsMinMax = sMinMax;
    if (blockSize >= 64) {vsMinMax[tid].x = fmin(sMinMax[tid].x, vsMinMax[tid + 32].x); vsMinMax[tid].y = fmax(sMinMax[tid].y, vsMinMax[tid + 32].y);}
    if (blockSize >= 32) {vsMinMax[tid].x = fmin(sMinMax[tid].x, vsMinMax[tid + 16].x); vsMinMax[tid].y = fmax(sMinMax[tid].y, vsMinMax[tid + 16].y);}
    if (blockSize >= 16) {vsMinMax[tid].x = fmin(sMinMax[tid].x, vsMinMax[tid +  8].x); vsMinMax[tid].y = fmax(sMinMax[tid].y, vsMinMax[tid +  8].y);}
    if (blockSize >=  8) {vsMinMax[tid].x = fmin(sMinMax[tid].x, vsMinMax[tid +  4].x); vsMinMax[tid].y = fmax(sMinMax[tid].y, vsMinMax[tid +  4].y);}
    if (blockSize >=  4) {vsMinMax[tid].x = fmin(sMinMax[tid].x, vsMinMax[tid +  2].x); vsMinMax[tid].y = fmax(sMinMax[tid].y, vsMinMax[tid +  2].y);}
    if (blockSize >=  2) {vsMinMax[tid].x = fmin(sMinMax[tid].x, vsMinMax[tid +  1].x); vsMinMax[tid].y = fmax(sMinMax[tid].y, vsMinMax[tid +  1].y);}
  }
  if (tid == 0) {
    out[blockIdx.x].x = sMinMax[0].x;
    out[blockIdx.x].y = sMinMax[0].y;
  }
}


// find maximum time step
extern "C" __global__ void dev_minmax (const int      blockSize,
                                       const int      nj,
                                       const double  *in,
                                             double2 *out) {

  if ((nj & (nj-1)) == 0)
    switch (blockSize) {
      case 512: 
        dev_reduction_minmax_template<512, true> (nj, in, out); break;
      case 256: 
        dev_reduction_minmax_template<256, true> (nj, in, out); break;
      case 128: 
        dev_reduction_minmax_template<128, true> (nj, in, out); break;
      case  64: 
        dev_reduction_minmax_template< 64, true> (nj, in, out); break;
      case  32: 
        dev_reduction_minmax_template< 32, true> (nj, in, out); break;
      case  16: 
        dev_reduction_minmax_template< 16, true> (nj, in, out); break;
      case   8: 
        dev_reduction_minmax_template<  8, true> (nj, in, out); break;
      case   4: 
        dev_reduction_minmax_template<  4, true> (nj, in, out); break;
      case   2: 
        dev_reduction_minmax_template<  2, true> (nj, in, out); break;
      case   1: 
        dev_reduction_minmax_template<  1, true> (nj, in, out); break;
    }
  else
    switch (blockSize) {
      case 512: 
        dev_reduction_minmax_template<512, false> (nj, in, out); break;
      case 256:                                        
        dev_reduction_minmax_template<256, false> (nj, in, out); break;
      case 128:                                        
        dev_reduction_minmax_template<128, false> (nj, in, out); break;
      case  64:                                        
        dev_reduction_minmax_template< 64, false> (nj, in, out); break;
      case  32:                                        
        dev_reduction_minmax_template< 32, false> (nj, in, out); break;
      case  16:                                        
        dev_reduction_minmax_template< 16, false> (nj, in, out); break;
      case   8:                                        
        dev_reduction_minmax_template<  8, false> (nj, in, out); break;
      case   4:                                        
        dev_reduction_minmax_template<  4, false> (nj, in, out); break;
      case   2:                                        
        dev_reduction_minmax_template<  2, false> (nj, in, out); break;
      case   1:                                        
        dev_reduction_minmax_template<  1, false> (nj, in, out); break;
  }
}

#define MAX_NTHREADS 256

// calculate closest number that is power of 2
unsigned int nextPow2(unsigned int x) {
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

void get_num_blocks_and_threads(int n, int &blocks, int &threads) {
  //get device capability, to avoid block/grid size excceed the upbound
  cudaDeviceProp prop;
  int device;
  cudaGetDevice(&device);
  cudaGetDeviceProperties(&prop, device);
  
  threads = (n < MAX_NTHREADS*2) ? nextPow2((n + 1)/ 2) : MAX_NTHREADS;
  blocks = (n + (threads * 2 - 1)) / (threads * 2);
  
  if ((float) threads * blocks > (float) prop.maxGridSize[0] * prop.maxThreadsPerBlock) {
      printf("n is too large, please choose a smaller number!\n");
  }
  
  if (blocks > prop.maxGridSize[0]) {
      printf("Grid size <%d> excceeds the device capability <%d>, set block size as %d (original %d)\n",
             blocks, prop.maxGridSize[0], threads * 2, threads);
  
      blocks /= 2;
      threads *= 2;
  }
}

double2 *gpu_minmax;
bool allocated;

void get_minmax_gpu (int nrad, int nphi, double *gpu_field, double *min, double *max) {

  // Nmax must be integer times the MAX_THREAD number
  int Nmax = ceil((float) nrad * nphi / (float) MAX_NTHREADS) * MAX_NTHREADS;
  
  // declare argument index, shared memory size and block numbers
  int sMemSize, numThreads, numBlocks;
  
  // get initial block and thread number
  get_num_blocks_and_threads(Nmax, numBlocks, numThreads);
  
  dim3 grid;
  dim3 block = dim3(numBlocks, numThreads);
  grid.x = numBlocks;
  grid.y = numThreads;

  // set shared memory (since totenergy is double2 it must be 2 times numThreads)
  sMemSize = 2 * (numThreads <= 32 ? 32: numThreads) * sizeof (double);

  // set execution configuration and start the kernel
  int nmax = nrad * nphi;
  get_num_blocks_and_threads <<<grid, block, sMemSize>>> (numThreads, nmax, gpu_field, gpu_minmax);
  // min partial block mins on GPU
  int s = numBlocks;
  while (s > 1) {
    
    // get next block and thread number
    get_num_blocks_and_threads(s, numBlocks, numThreads);
  
    // in this step timestep calculation is required
    int new_s = (s + (numThreads*2-1)) / (numThreads*2);
  
    int sMemSize = 2 * (numThreads <= 32 ? 32: numThreads) * sizeof (double);
  
    // set execution configuration and start the kernel
    get_num_blocks_and_threads <<<grid, block, sMemSize>>> (numThreads, s, gpu_field, gpu_minmax);
  
    s = new_s;
  }
  *min = gpu_minmax->x;
  *max = gpu_minmax->y;
} 