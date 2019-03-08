/** \file glcuda.cu : implements the filling of the PBO for OpenGL interoperability
*/
#define __CUDA 1
#include "fargo.h"
#undef __CUDA
#include <stdarg.h>
#include <helper_cuda.h>
#include <cuda.h>

// BLOCK_X : in azimuth
//#define BLOCK_X DEF_BLOCK_X_GLCUDA
#define BLOCK_X 8
// BLOCK_Y : in radius
#define BLOCK_Y 8

extern double minftd, maxftd;
extern int    CartesianView, Xsize, Ysize;
extern int    CenterPlanet;
extern double Zoom;
extern bool   LogGrid;
extern bool   LogDisplay;
__constant__ unsigned int colormap[256];

__global__ void clear_rgba_kernel (unsigned int *plot, unsigned int  clBackground, int pitch) {
  const int j = blockIdx.x*blockDim.x + threadIdx.x;
  const int i = blockIdx.y*blockDim.y + threadIdx.y;
  
  plot[j+i*pitch] = clBackground;
}


__global__ void clear_flowpattern_rgba_kernel (unsigned int *plot, 
                                               int           pitch) {
  const int j = blockIdx.x*blockDim.x + threadIdx.x;
  const int i = blockIdx.y*blockDim.y + threadIdx.y;
  
  const int m = j+i*pitch;

  // get rgb color tripplet of the given pixel
  unsigned int rgb = plot[m];
  unsigned int blue  = (rgb >> 16) & 0x0FF;
  unsigned int green = (rgb >> 8)  & 0x0FF;
  unsigned int red   =  rgb        & 0x0FF; 

  if (rgb > 10) {
    red     = (red   > 10 ? red  -10:0);
    green   = (green > 10 ? green-10:0);
    blue    = (blue  > 10 ? blue -10:0);
    plot[m] = red+256*green+256*256*blue;
  }
  else
    plot[m] = 0;
}


__global__ void kernel_get_rgba_polar(double       *field, 
                                      unsigned int *plot,
                                      int           pitch, 
                                      double        min, 
                                      double        max, 
                                      double        disp_gamma, 
                                      double        top, 
                                      int           shift) {

  const int j = blockIdx.x*blockDim.x + threadIdx.x;
  const int i = blockIdx.y*blockDim.y + threadIdx.y;
  
  const int m = j+i*pitch;
  const double data = field[m];

  float frac = (disp_gamma != 1.0 ? pow((data-min)/(max*top-min),disp_gamma) : (data-min)/(max*top-min));
  if (frac < 0.0) frac = 0.0;
  else if (frac > 1.0) frac = 1.0;
  
  const int sm = (j-shift >= 0 ? j-shift+i*pitch : pitch-shift+j+i*pitch);
  plot[sm] = colormap[(int)(frac * 255.0)];
}

__global__ void kernel_get_rgba_polar_log(double       *field, 
                                          unsigned int *plot,
                                          int           pitch, 
                                          double        min, 
                                          double        max, 
                                          double        disp_gamma, 
                                          double        top, 
                                          int           shift) {

  const int j = blockIdx.x*blockDim.x + threadIdx.x;
  const int i = blockIdx.y*blockDim.y + threadIdx.y;
  
  const int m = j+i*pitch;
  const double data = log10(field[m]);

  float frac = (disp_gamma != 1.0 ? pow((data-min)/(max*top-min),disp_gamma) : (data-min)/(max*top-min));
  if (frac < 0.0) frac = 0.0; 
  else if (frac > 1.0) frac = 1.0;
  
  const int sm = (j-shift >= 0 ? j-shift+i*pitch : pitch-shift+j+i*pitch);
  plot[sm] = colormap[(int)(frac * 255.0)];
}


__global__  void kernel_add_planet_polar (bool loggrid,
                                          double pl_x, 
                                          double pl_y,
                                          unsigned int *plot,
                                          int xsize,
                                          double rmin, 
                                          double rmax,
                                          int nrad,
                                          int pitch, 
                                          int shift) {

  if (threadIdx.y > 0)
    return;
  
  // get pixel coordinate of particles for polar plot
  const double r = sqrt (pl_x * pl_x + pl_y * pl_y);
  double theta = atan2(pl_y ,pl_x) + 3.14159265;
  if (theta < 0.)
    theta += 2.*3.14159265;

  // index of plot array at a given particle position (arithmetic grid)
  //const int m = (int) ((r-rmin)/(rmax-rmin)*nrad)*pitch + (int) (theta/2.f/3.14159265f*(float)pitch);
  const int i = (loggrid == true ? (int) ((log(r)-log(rmin))/(log(rmax)-log(rmin))*nrad) : (int) ((r-rmin)/(rmax-rmin)*nrad));
  const int j = (int) (theta/2./3.14159265*(double)pitch);
  const int m = i * pitch + j;  

  // plotting planets
  if (m > 0 && m < nrad*pitch) {
//  if (m-2-xsize > 0 && m+2+xsize < nrad*pitch) {    
  	// cyan color for planet
    const unsigned int c0 = 150 * 256 + 150*256*256;
    const unsigned int c1 = 180 * 256 + 180256*256;
    const unsigned int c2 = 128 + 255 * 256 + 255*256*256;
                            plot[m-1-2*xsize] = c2; plot[m-2*xsize] = c2;  plot[m+1-2*xsize] = c2; 
    plot[m-2-xsize]   = c2; plot[m-1-xsize]   = c1; plot[m-xsize]   = c1;  plot[m+1-xsize]   = c1; plot[m+2-xsize]   = c2;
    plot[m-2]         = c2; plot[m-1]         = c1; plot[m]         = c0;  plot[m+1]         = c1; plot[m+2]         = c2;
    plot[m-2+xsize]   = c2; plot[m-1+xsize]   = c1; plot[m+xsize]   = c1;  plot[m+1+xsize]   = c1; plot[m+2+xsize]   = c2;
                            plot[m-1+2*xsize] = c2; plot[m+2*xsize] = c2;  plot[m+1+2*xsize] = c2;
  }
}


__global__ void kernel_get_rgba_cart (bool          loggrid, 
                                      double       *field, 
                                      unsigned int *plot, 
                                      int           pitch, 
                                      double        min, 
                                      double        max,
                                      int           xsize, 
                                      int           ysize, 
                                      double        rmin, 
                                      double        rmax, 
                                      double        log_rmin, 
                                      double        log_rmax,
                                      int           xc, 
                                      int           yc, 
                                      int           nrad, 
                                      double        center_x, 
                                      double        center_y, 
                                      double        zoom, 
                                      double        ar, 
                                      double        disp_gamma, 
                                      double        top, 
                                      double        frame_rotate) {

  const int j = blockIdx.x*blockDim.x + threadIdx.x;
  const int i = blockIdx.y*blockDim.y + threadIdx.y;
    
  const int mm = j+i*xsize;
  const double x = (double)(j-xc)/(double)xc*zoom+center_x;
  const double y = (double)(i-yc)/(double)yc*zoom*ar+center_y;

  double theta = atan2(y,x) + frame_rotate;
  if (theta < 0.f) 
    theta += 2.0*M_PI;

  double r = sqrt (x*x+y*y);
  double frac;
  int m;
  if ((r < rmin) || (r > rmax)) 
    frac = 0.0;
  else {
    m = (loggrid ? (int) ((log(r)-log_rmin)/(log_rmax-log_rmin)*nrad)*pitch + (int) (theta/2.0/M_PI*(double)pitch)
                 : (int) ((r-rmin)/(rmax-rmin)*nrad)*pitch + (int) (theta/2.0/M_PI*(double)pitch));
  //  const int ii = m / pitch;
    double data = 0.0;
//    if (ii > 1 && ii < nrad-2)
      data = field[m];
    frac = (disp_gamma != 1.0 ? pow((data-min)/(max*top-min),disp_gamma) : (data-min)/(max*top-min));
    if (frac < 0.0) frac = 0.0;
    if (frac > 1.0) frac = 1.0;
  }
  plot[mm] = colormap[(int)(frac * 255.0)]; 
}

__global__ void kernel_get_rgba_cart_log (bool          loggrid, 
                                          double       *field, 
                                          unsigned int *plot, 
                                          int           pitch, 
                                          double        min, 
                                          double        max,
                                          int           xsize, 
                                          int           ysize, 
                                          double        rmin, 
                                          double        rmax, 
                                          double        log_rmin, 
                                          double        log_rmax,
                                          int           xc, 
                                          int           yc, 
                                          int           nrad, 
                                          double        center_x, 
                                          double        center_y, 
                                          double        zoom, 
                                          double        ar, 
                                          double        disp_gamma, 
                                          double        top, 
                                          double        frame_rotate) {

  const int j = blockIdx.x*blockDim.x + threadIdx.x;
  const int i = blockIdx.y*blockDim.y + threadIdx.y;

  const int mm = j+i*xsize;
  const double x = (double)(j-xc)/(double)xc*zoom+center_x;
  const double y = (double)(i-yc)/(double)yc*zoom*ar+center_y;
  
  double theta = atan2(y,x) + frame_rotate;
  if (theta < 0.f) 
    theta += 2.f*3.14159265;

  double r = sqrt (x*x+y*y);
  double frac;
  int m;
  if ((r < rmin) || (r > rmax)) 
    frac = 0.0;
  else {
    m = (loggrid ? (int) ((log(r)-log_rmin)/(log_rmax-log_rmin)*nrad)*pitch + (int) (theta/2.0/M_PI*(double)pitch)
                 : (int) ((r-rmin)/(rmax-rmin)*nrad)*pitch + (int) (theta/2.0/M_PI*(double)pitch));
    const int ii = (int) m / pitch;
    double data = 0.0;
    if (ii > 1 && ii < nrad-2)
      data = log10(field[m]);
    frac = (disp_gamma != 1.0 ? pow((data-min)/(max*top-min),disp_gamma) : (data-min)/(max*top-min));
    if (frac < 0.0) frac = 0.0;
    if (frac > 1.0) frac = 1.0;
  }
  plot[mm] = colormap[(int)(frac * 255.0)];
}


__global__  void kernel_add_planet_cart (bool loggrid,
                                         double pl_x, 
                                         double pl_y,
                                         unsigned int *plot,
                                         int pitch, 
                                         int xsize, 
                                         int ysize,
                                         int xc, 
                                         int yc, 
                                         int nrad,
                                         double center_x, 
                                         double center_y, 
                                         double zoom, 
                                         double ar,
                                         double frame_rotate,
                                         double rhill) {

  //if (threadIdx.y > 0)
  //  return;

  //const double3 pos = make_double3 (pos_j[k].x-central_start.x, pos_j[k].y-central_start.y, 0.0);
  const double3 pos = make_double3 (pl_x, pl_y, 0.0);
  const int j = (int) ((pos.x-center_x) / zoom * xc + xc);
  const int i = (int) ((pos.y-center_y) / zoom * ar * yc + yc);

  const int n = ysize/2*xsize+xsize/2;
  if (threadIdx.y == 0 && n-1-2*xsize > 0 && n+1+2*xsize < xsize*ysize) {
 
  	
    // yellow color for star
    const unsigned int c0 = 255 + 255 * 256 + 20*256*256;
    const unsigned int c1 = 255 + 255 * 256 + 20*256*256;
    const unsigned int c2 = 255 + 255 * 256 + 20*256*256;
                            plot[n-1-2*xsize] = c2; plot[n-2*xsize] = c2;  plot[n+1-2*xsize] = c2; 
    plot[n-2-xsize]   = c2; plot[n-1-xsize]   = c1; plot[n-xsize]   = c1;  plot[n+1-xsize]   = c1; plot[n+2-xsize]   = c2;
    plot[n-2]         = c2; plot[n-1]         = c1; plot[n]         = c0;  plot[n+1]         = c1; plot[n+2]         = c2;
    plot[n-2+xsize]   = c2; plot[n-1+xsize]   = c1; plot[n+xsize]   = c1;  plot[n+1+xsize]   = c1; plot[n+2+xsize]   = c2;
                            plot[n-1+2*xsize] = c2; plot[n+2*xsize] = c2;  plot[n+1+2*xsize] = c2;
  }

  if (i<0 || j > ysize || i > xsize || j < 0)
    return;

  const int m = (j+i*xsize);  
  
  // plotting planets
//  if (m >0 && m < xsize*ysize) {
  if (threadIdx.y == 0 && m-1-2*xsize > 0 && m+1+2*xsize < xsize*ysize) {
 
  	
    // cyan color for planet
    const unsigned int c0 = 150 * 256 + 150*256*256;
    const unsigned int c1 = 180 * 256 + 180256*256;
    const unsigned int c2 = 128 + 255 * 256 + 255*256*256;
                            plot[m-1-2*xsize] = c2; plot[m-2*xsize] = c2;  plot[m+1-2*xsize] = c2; 
    plot[m-2-xsize]   = c2; plot[m-1-xsize]   = c1; plot[m-xsize]   = c1;  plot[m+1-xsize]   = c1; plot[m+2-xsize]   = c2;
    plot[m-2]         = c2; plot[m-1]         = c1; plot[m]         = c0;  plot[m+1]         = c1; plot[m+2]         = c2;
    plot[m-2+xsize]   = c2; plot[m-1+xsize]   = c1; plot[m+xsize]   = c1;  plot[m+1+xsize]   = c1; plot[m+2+xsize]   = c2;
                            plot[m-1+2*xsize] = c2; plot[m+2*xsize] = c2;  plot[m+1+2*xsize] = c2;
  }
  
  // plot hill sphere
  const int di = rhill / zoom *xc;  
  const int dj = rhill / zoom * ar * yc;
  const int jj = blockIdx.x*blockDim.x + threadIdx.x;
  const int ii = blockIdx.y*blockDim.y + threadIdx.y;

  if ((ii-i)*(ii-i) + (jj-j)*(jj-j) <= di*di + dj*dj && (ii-i)*(ii-i) + (jj-j)*(jj-j) >= (di-1)*(di-1) + (dj-1)*(dj-1)) {
    const int mm = (jj+ii*xsize);
    if (mm >0 && mm < xsize*ysize)
      plot[mm] = 255+255*256+255*256*256;
  }
}
    
                                            
void reset_rgba (int mode, unsigned int *plot, unsigned int clBackground) {

  if (mode == 0) {
    //clear_rgba_kernel <<<grid, block>>> (plot, clBackground, Xsize);
    cudaMemset (plot, 0, Xsize * Ysize * sizeof (int));  
  }
  else if (mode == 1) {
    dim3 grid;
    dim3 block = dim3(BLOCK_X, BLOCK_Y);
    grid.x = Xsize / BLOCK_X;
    grid.y = Ysize / BLOCK_Y;

    clear_flowpattern_rgba_kernel <<<grid, block>>> (plot, Xsize);
  }
}


void get_rgba(PolarGrid *FTD, unsigned int *plot, unsigned int *cmap, PlanetarySystem *sys, double ar, float fDispGamma, float fTop, double FrameRotate) {
  double xcenter=0.0, ycenter=0.0;
  dim3 grid;
  dim3 block = dim3(BLOCK_X, BLOCK_Y);
  grid.x = Xsize / BLOCK_X;
  grid.y = Ysize / BLOCK_Y;

  dim3 _grid;
  dim3 _block = dim3(1, 1);
  _grid.x = 1; //Xsize / BLOCK_X;
  _grid.y = 1;//Ysize / BLOCK_Y;

  checkCudaErrors(cudaMemcpyToSymbol(colormap, (void *)cmap, (size_t)(256*sizeof(unsigned int)), 0, cudaMemcpyHostToDevice));
  
  // Cartesian view
  if (CartesianView) {
    int nsec = (float) (FTD->pitch/sizeof(double));
    int shift = round ((FrameRotate / 2.0 / M_PI) * (float) nsec) + nsec / 2;
    shift = (shift >= nsec ? shift - nsec : shift);

    if (!LogDisplay) {
      kernel_get_rgba_polar <<<grid, block>>> (FTD->gpu_field, plot,
                                               FTD->pitch/sizeof(double),
                                               minftd, 
                                               maxftd, 
                                               fDispGamma, 
                                               fTop, 
                                               shift);
      
      cudaThreadSynchronize();
      getLastCudaError ("kernel_get_rgba_polar failed.");
    }
    else {
      kernel_get_rgba_polar_log <<<grid, block>>> (FTD->gpu_field, plot,
                                                   FTD->pitch/sizeof(double),
                                                   log10(minftd), 
                                                   log10(maxftd), 
                                                   fDispGamma, 
                                                   fTop, 
                                                   shift);
      
      cudaThreadSynchronize();
      getLastCudaError ("kernel_get_rgba_polar_log failed.");
    }
      

    kernel_add_planet_polar <<<_grid, _block>>> (LogGrid,
                                                 sys->x[0], 
                                                 sys->y[0],
                                                 plot,
                                                 Xsize,
                                                 RMIN, 
                                                 RMAX,
                                                 NRAD,
                                                 FTD->pitch/sizeof(double),
                                                 shift);
    cudaThreadSynchronize();
    getLastCudaError ("kernel_add_planet_polar failed.");
  }
  
  // Polar view
  else {
    // if centering on planet
    if (CenterPlanet == YES) {
      xcenter = sys->x[0];
      ycenter = sys->y[0];
    }
    
    if (!LogDisplay) {
      kernel_get_rgba_cart <<<grid, block>>> (LogGrid,
                                              FTD->gpu_field, 
                                              plot, 
                                              FTD->pitch/sizeof(double),
                                              minftd, 
                                              maxftd, 
                                              Xsize, 
                                              Ysize,
                                              RMIN, 
                                              RMAX, 
                                              log(RMIN), 
                                              log(RMAX), 
                                              Xsize/2, 
                                              Ysize/2, 
                                              NRAD,
                                              xcenter, 
                                              ycenter, 
                                              1.0/Zoom, 
                                              ar, 
                                              fDispGamma, 
                                              fTop,
                                              FrameRotate);
      cudaThreadSynchronize();
      getLastCudaError ("kernel_get_rgba_cart failed."); 
    }
    else {
      kernel_get_rgba_cart_log <<<grid, block>>> (LogGrid,
                                                  FTD->gpu_field, 
                                                  plot, 
                                                  FTD->pitch/sizeof(double),
                                                  log10(minftd), 
                                                  log10(maxftd), 
                                                  Xsize, 
                                                  Ysize,
                                                  RMIN, 
                                                  RMAX, 
                                                  log(RMIN), 
                                                  log(RMAX), 
                                                  Xsize/2, 
                                                  Ysize/2, 
                                                  NRAD,
                                                  xcenter, 
                                                  ycenter, 
                                                  1.0/Zoom, 
                                                  ar, 
                                                  fDispGamma, 
                                                  fTop,
                                                  FrameRotate);
      cudaThreadSynchronize();
      getLastCudaError ("kernel_get_rgba_cart_log failed."); 
          
    }                                           
    
    double rhill = sqrt((sys->x[0])*(sys->x[0])+(sys->y[0])*(sys->y[0]))*pow(sys->mass[0]*MassTaper/3.0,1./3.);
    kernel_add_planet_cart <<<grid, block>>> (LogGrid,
                                                sys->x[0], 
                                                sys->y[0],
                                                plot,
                                                FTD->pitch/sizeof(double),
                                                Xsize, 
                                                Ysize,
                                                Xsize/2,
                                                Ysize/2,
                                                NRAD,
                                                xcenter,
                                                ycenter,
                                                1.0/Zoom, 
                                                ar,
                                                FrameRotate,
                                                rhill);
    cudaThreadSynchronize();
    getLastCudaError ("kernel_add_planet_cart failed.");
  }  
}

/*

void get_fine_rgba(double *FTD, unsigned int *plot, unsigned int *cmap, PlanetarySystem *sys, double ar, float fDispGamma, float fTop, double FrameRotate) {
  double xcenter=0.0, ycenter=0.0;
  dim3 grid;
  dim3 block = dim3(BLOCK_X, BLOCK_Y);
  grid.x = Xsize / BLOCK_X;
  grid.y = Ysize / BLOCK_Y;

  checkCudaErrors(cudaMemcpyToSymbol(colormap, (void *)cmap,(size_t)(256*sizeof(unsigned int)), 0, cudaMemcpyHostToDevice));
  
  // Cartesian view
  if (CartesianView) {
    int nsec = NSEC;// *GASOVERSAMPAZIM;
    int shift = round ((FrameRotate / 2.0 / M_PI) * (float) nsec*GASOVERSAMPAZIM) + nsec*GASOVERSAMPAZIM / 2;
    shift = (shift >= nsec*GASOVERSAMPAZIM ? shift - nsec*GASOVERSAMPAZIM : shift);
    get_rgba_polar_kernel <<<grid, block>>> (FTD, 
                                             plot,
                                             NSEC*GASOVERSAMPAZIM,
                                             minftd, 
                                             maxftd, 
                                             fDispGamma, 
                                             fTop, 
                                             shift);
    cudaThreadSynchronize();
    getLastCudaError ("get_rgba_polar_kernel failed.");
  } 
  // Polar view
  else {
    // if centering on planet
    if (CenterPlanet == YES) {
      xcenter = sys->x[0];
      ycenter = sys->y[0];
    }
    get_rgba_cart_kernel <<<grid, block>>> (LogGrid,
                                            FTD,
                                            plot,
                                            NSEC*GASOVERSAMPAZIM,
                                            minftd, 
                                            maxftd, 
                                            Xsize, 
                                            Ysize,
                                            RMIN, 
                                            RMAX, 
                                            log(RMIN), 
                                            log(RMAX), 
                                            Xsize/2, 
                                            Ysize/2,
                                            NRAD*GASOVERSAMPRAD, 
                                            xcenter, 
                                            ycenter, 
                                            1.0/Zoom, 
                                            ar, 
                                            fDispGamma, 
                                            fTop,
                                            FrameRotate);
    cudaThreadSynchronize();
    getLastCudaError ("get_rgba_cart_kernel failed.");
  }
}
*/





