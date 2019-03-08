#define __CUDA 1
#include "fargo.h"
#undef __CUDA
//#include "glcmap.h"
#include <GL/glew.h>

#include <stdarg.h>
#ifdef __APPLE
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <helper_cuda.h>
#include <thrust/device_ptr.h>
#include <thrust/fill.h>
#include <thrust/extrema.h>

#define GL_WINSIZE_X     WINSIZE
#define GL_WINSIZE_Y     WINSIZE

// BLOCK_X : in azimuth
// #define BLOCK_X DEF_BLOCK_X_GLEXT
#define BLOCK_X 8
// BLOCK_Y : in radius
#define BLOCK_Y 8



//#define DEBUGGING 1








// [RZS-MOD]
// read my own colormaps
#include "my_glcmap"
unsigned int cmap[256];
char simdir[2048];


//PolarGrid *D2GMass;

FILE *avconv = NULL;
GLubyte *pixels;
bool bFine = false;

GLuint gl_PBO, gl_Tex, win;
unsigned int *plot_rgba;
//extern PolarGrid *FTD; // Field to display
int    CartesianView = 1, Xsize, Ysize;

double minftd = 2e-4, maxftd = 2e-3;
int    DustGridType = 0;
int    color_idx = 0;
double Zoom=0.05;
float  DispGamma = 1.0, Top=1.0;
int    CenterPlanet=1;
double FrameRotate = 0.0;

static int Update = 1;


static int palette_nb = 6, old_palette_nb=6;

#ifdef FARGO_INTEGRATION
#include "../HIPERION-v1.7/interface_extfuncs.h"
double *FineFTD;
bool FlowPattern   = NO;
double FrameRadius = 0.0;
//double dust_sizecm[4] = {1e-4,1e-1,1e2,1e5};
//double dust_sizecm[2] = {1e-3,1e-1};
extern int iDustBinNum;                   // inherited from HIPERION ????
extern double *dDustSize;
#endif


bool Lock                = NO;
bool LogDisplay          = NO;
bool DustParticles       = NO;
bool DustParticlesonGrid = NO;
//bool Dust2GassMassRatio  = NO;

bool CalcVortensity = NO;
bool CalcTemperature = NO;
bool CalcDiskHeight = NO;
bool CalcSoundSpeed = NO;
bool CalcDustGasRatio = NO;
bool CalcDustSize = NO;

void reset_rgba (int mode, unsigned int *plot, unsigned int clBackground);
unsigned int ColBackground = 0;
void get_rgba(PolarGrid *FTD, unsigned int *plot, unsigned int *cmap, PlanetarySystem *sys, double ar, float fDispGamma, float fTop, double FrameRotate);
void get_fine_rgba(double *FTD, unsigned int *plot, unsigned int *cmap, PlanetarySystem *sys, double ar, float fDispGamma, float fTop, double FrameRotate);
void DrawField ();
void resize(int w, int h);
void keyCB (unsigned char key, int x, int y);
void keySpecCB (int key, int x, int y);

int xview, yview;
char winTitle[1024], winTitleMessage[1024], avconv_str[1024], ppm_filename[1024], movie_filename[1024];
int av_count = 0;
int nframes = 0;
double delta_rad = 0.0, Shift = 0;

// RZS [MOD]
// function to get wall-clock timings
//----------------------------------
#include <sys/time.h>
double get_time2 () {
  struct timeval Tvalue;
  struct timezone dummy;

  gettimeofday(&Tvalue,&dummy);
  return ((double) Tvalue.tv_sec +
          1.e-6*((double) Tvalue.tv_usec));
}
//----------------------------------

void getminmax (PolarGrid *var)
{
  int i, j, m, nr, ns;
  double min=1e30, max=-1e30;
  double *field;
  D2H (var);
  field = var->Field;
  nr = var->Nrad;
  ns = var->Nsec;
  for (i = 0; i < nr; i++) {
    for (j = 0; j < ns; j++) {
      m = j+i*ns;
      if (min > field[m]) min = field[m];
      if (max < field[m]) max = field[m];
    }
  }
  minftd = min;
  maxftd = max;
}

// RZS [MOD]
// GPU-based minmax finding with thrust
//-------------------------------------
void get_minmax_gpu(PolarGrid* FTD, double* min, double *max) {
    
  // wrap raw pointer with a device_ptr
  thrust::device_ptr<double> d_ftd(FTD->gpu_field);
  
  // use thrust to find the maximum element
  int nelements = (FTD->Nrad) * (FTD->Nsec);
  thrust::device_ptr<double> d_ptr_max_r = thrust::max_element(d_ftd, d_ftd + nelements);
  thrust::device_ptr<double> d_ptr_min_r = thrust::min_element(d_ftd, d_ftd + nelements);

  // copy the max element from device memory to host memory
  cudaMemcpy((void*)max, (void*)d_ptr_max_r.get(), sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy((void*)min, (void*)d_ptr_min_r.get(), sizeof(double), cudaMemcpyDeviceToHost);
}

void get_minmax_fine_gpu(double* FTD, double* min, double *max) {
    
  // wrap raw pointer with a device_ptr
  thrust::device_ptr<double> d_ftd(FTD);
  
  // use thrust to find the maximum element
  int nelements = NRAD * GASOVERSAMPRAD * NSEC * GASOVERSAMPAZIM;
  thrust::device_ptr<double> d_ptr_max_r = thrust::max_element(d_ftd, d_ftd + nelements);
  thrust::device_ptr<double> d_ptr_min_r = thrust::min_element(d_ftd, d_ftd + nelements);

  // copy the max element from device memory to host memory
  cudaMemcpy((void*)max, (void*)d_ptr_max_r.get(), sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy((void*)min, (void*)d_ptr_min_r.get(), sizeof(double), cudaMemcpyDeviceToHost);
}
//-------------------------------------


__global__ void fill_d2gm (int pitch, int nr, 
                           double* d2gm, const double *gas_dens, const double *dust_dens) {
                           
  const int j = blockIdx.x*blockDim.x + threadIdx.x;
  const int i = blockIdx.y*blockDim.y + threadIdx.y;                             
  const int m = j+i*pitch;
  
  d2gm[m] = dust_dens[m]/gas_dens[m];
}

/*
void EvalD2GM (PolarGrid *D2GMass, PolarGrid *gas_dens, PolarGrid *dust_dens) {
  dim3 grid;
  dim3 block = dim3(BLOCK_X, BLOCK_Y);
  grid.x = NSEC / BLOCK_X;
  grid.y = NRAD / BLOCK_Y;
  //int nsec = (float) (FTD->pitch/sizeof(double));
  fill_d2gm <<<grid, block>>> (FTD->pitch/sizeof(double), 0, D2GMass->gpu_field, gas_dens->gpu_field, dust_dens->gpu_field);
}*/

void screenshot_ppm(const char *filename, unsigned int width, unsigned int height, GLubyte **pixels) {
    size_t i, j, cur;
    const size_t format_nchannels = 3;
    FILE *f = fopen(filename, "w");
    fprintf(f, "P3\n%d %d\n%d\n", width, height, 255);
    *pixels = (GLubyte*) realloc(*pixels, format_nchannels * sizeof(GLubyte) * width * height);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, *pixels);
    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            cur = format_nchannels * ((height - i - 1) * width + j);
            fprintf(f, "%3d %3d %3d ", (*pixels)[cur], (*pixels)[cur + 1], (*pixels)[cur + 2]);
        }
        fprintf(f, "\n");
    }
    fclose(f);
}

void load_palette () {
  if (palette_nb > 6)
    palette_nb = 1;
  switch (palette_nb) {
    case 0:
      memcpy (cmap, cmap0, 256*sizeof(unsigned int));
    break;
    case 1:
      memcpy (cmap, cmap1, 256*sizeof(unsigned int));
    break;
    case 2:
      memcpy (cmap, cmap2, 256*sizeof(unsigned int));
      break;
    case 3:
      memcpy (cmap, cmap3, 256*sizeof(unsigned int));
    break;
    case 4:
      memcpy (cmap, cmap4, 256*sizeof(unsigned int));
    break;
    case 5:
      memcpy (cmap, cmap5, 256*sizeof(unsigned int));
    break;
    case 6:
      memcpy (cmap, cmap6, 256*sizeof(unsigned int));
    break;
/*    case 7:
      memcpy (cmap, cmap7, 256*sizeof(unsigned int));
    break;
    case 8:
      memcpy (cmap, cmap8, 256*sizeof(unsigned int));
    break;
  */}
}

void InitDisplay (int *argc, char **argv) {
  size_t pitch;
  memcpy (cmap, cmap1, 256*sizeof(unsigned int));  
  Xsize = NSEC;
  Ysize = NRAD;
  xview = GL_WINSIZE_X;
  yview = GL_WINSIZE_Y;
  glutInitWindowSize(xview, yview);  
  glutInitWindowPosition(30, 30);
  glutInit(argc, argv);
  glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);

  if (strcmp(OUTPUTDIR, "./") == 0)
    getcwd (simdir, 2048);
  else
    snprintf(simdir, 2048, "%s", OUTPUTDIR);
  snprintf(winTitle, 1023, "GFARGO:%s - gas density", simdir);    
  win =  glutCreateWindow(winTitle);

  //Vortens = CreatePolarGrid (NRAD, NSEC, "Vortensity");
  //if (DustGrid)
  //  D2GMass = CreatePolarGrid (NRAD, NSEC, "D2GMass");
  
  //if (Adiabatic)
  //  Temperature = CreatePolarGrid (NRAD, NSEC, "Temperature");
    
  // Check for OpenGL extension support 
  if (verbose)
    printf("Loading OPENGL extensions: %s\n", glewGetErrorString(glewInit()));
  else
    glewGetErrorString(glewInit());
  if(!glewIsSupported("GL_VERSION_2_0 " 
                      "GL_ARB_pixel_buffer_object "
                      "GL_EXT_framebuffer_object ")){
    fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
    fflush(stderr);
    return;
  }
  
  // Set up view
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glMatrixMode(GL_PROJECTION);
  glLoadIdentity();
  glOrtho(0,Xsize,0.,Ysize, -200.0, 200.0);
  
  // Create texture which we use to display the result and bind to gl_Tex
  glEnable(GL_TEXTURE_2D);
  glGenTextures(1, &gl_Tex);                     // Generate 2D texture
  glBindTexture(GL_TEXTURE_2D, gl_Tex);          // bind to gl_Tex
  
  // texture properties:
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, Xsize, Ysize, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  pitch = NSEC*sizeof(double);
  
  // Create pixel buffer object and bind to gl_PBO. We store the data we want to
  // plot in memory on the graphics card - in a "pixel buffer". We can then 
  // copy this to the texture defined above and send it to the screen
  glGenBuffers(1, &gl_PBO);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
  glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, pitch*Ysize, NULL, GL_STREAM_COPY);
  checkCudaErrors( cudaGLRegisterBufferObject(gl_PBO) );
  
  // define glut functions
  glutDisplayFunc(DrawField);
  glutReshapeFunc(resize);
  glutIdleFunc(DrawField);
  glutKeyboardFunc(keyCB);
  glutSpecialFunc(keySpecCB);
  
  load_palette ();
  
#ifdef DEBUGGING
  Paused = 1;
#endif
}

void DisplayLoadDensity () {
  keyCB ('c', 0, 0);
  keyCB ('d', 0, 0);
  keyCB ('s', 0, 0);
  Zoom = 1.0/RMAX;
}

#include <unistd.h>
static int frameNum = 0;
void DrawField () {
  static int FirstTime = YES, wx, wy;
  static double previous, actual;
  //static double ts, te;
  double ar;

  // hydro calculation, if not movie generation is requested (option -m)
  if (!CreatingMovieOnly)
    Loop ();

  // just red snapshot file and display it
  else {
     ReadfromFile (gas_density, "gas_dens", frameNum);
     H2D (gas_density);
     frameNum++;
     usleep (100000);
  }
  
#ifdef DEBUGGING
  Paused = 1;
#endif
  
  // update window only if its is not turned off
  //if (Update == -1) 
  //  return;

  wx = glutGet (GLUT_WINDOW_WIDTH);
  wy = glutGet (GLUT_WINDOW_HEIGHT);
  ar = (double)wx/(double)wy;
  actual = clock();

  if ((((actual - previous)/CLOCKS_PER_SEC) > (1./RefreshRate)) || (FirstTime == YES) || CreatingMovieOnly) {
    if (Update >= 0) {

      FirstTime = NO;

#ifdef DEBUGGING
      get_minmax_gpu (FTD, &minftd, &maxftd);
#endif

      // some additional values must be calculated if requested
      if (CalcVortensity == YES) 
        CalcVortens_gpu (gas_density, gas_v_rad, gas_v_theta, Work);
      if (Adiabatic) {
        if(CalcTemperature)
          CalcTemp_gpu (gas_density, gas_energy, Work);
        else if (CalcDiskHeight)
          CalcDiskHeight_gpu (gas_density, gas_energy, Work);
        else if (CalcSoundSpeed)
          CalcSoundSpeed_gpu (gas_density, gas_energy, Work);
      }
      if (CalcDustGasRatio && DustGrid) 
        CalcDustGasMassRatio_gpu(gas_density, dust_density[color_idx], Work);
      if (CalcDustSize)
        CalcDustSize_gpu(dust_size, dust_density[0], Work);
#ifdef FARGO_INTEGRATON      
//      if (Dust2GassMassRatio == YES)
//        EvalD2GM (D2GMass, gas_density, DustDens);
#endif      
      // Apparently there is no need to do a cudaMalloc of plot_rgba, the following function does the job
      checkCudaErrors(cudaGLMapBufferObject((void**)&plot_rgba, gl_PBO));


      //get_minmax_gpu (FTD, &minftd, &maxftd);
      
      // [RZS-MOD]
      // lock view to the planet 0
      //--------------------------------------
      if (Lock) {
        double lock_rad = sqrt(sys->y[0]*sys->y[0] + sys->x[0]*sys->x[0]);
        FrameRotate = atan2 (sys->y[0] , sys->x[0]);
        FrameRotate = (FrameRotate < 0.0 ? FrameRotate + 2.0 * M_PI: FrameRotate);
        FrameRotate -= (pow(lock_rad*(1.0+delta_rad), -1.5) - pow(lock_rad, -1.5))*PhysicalTime - Shift;
        FrameRotate = fmod (FrameRotate, 2.0*M_PI);
      }
      else
        FrameRotate = 0.0;
      //--------------------------------------
            
      // plot filed
      if (FTD != NULL) {
#ifdef FARGO_INTEGRATION
        if (bFine)
          get_fine_rgba(FineFTD, plot_rgba, cmap, sys, ar, DispGamma, Top, FrameRotate);
        else
#endif          
        get_rgba(FTD, plot_rgba, cmap, sys, ar, DispGamma, Top, FrameRotate);
      }
      
      // [RZS-MOD]
      // displaying dust particles
      //--------------------------
#ifdef FARGO_INTEGRATION
      if (DustParticles) {
        double xcenter = 0.0, ycenter = 0.0;
        if (CenterPlanet == YES) {
          xcenter = sys->x[0];
          ycenter = sys->y[0]; 
        }
        if (FTD == NULL)
          field (FlowPattern, plot_rgba, ColBackground);
        if (CartesianView)
          HIPERION_DisplayDust (0, NRAD, gas_density->pitch/sizeof(double), 0, 0, 0, 0, RMIN, RMAX, FrameRotate, color_idx, plot_rgba);
        else
          HIPERION_DisplayDust (1, Xsize, Ysize, ar, Zoom, Xsize/2, Ysize/2, xcenter, ycenter, FrameRotate, color_idx, plot_rgba);
      }
#endif
      //--------------------------
      checkCudaErrors(cudaGLUnmapBufferObject(gl_PBO));
      
      // Copy the pixel buffer to the texture, ready to display
      glTexSubImage2D(GL_TEXTURE_2D,0,0,0,Xsize,Ysize,GL_RGBA,GL_UNSIGNED_BYTE,0);

      // Render one quad to the screen and colour it using our texture
      // i.e. plot our plotvar data to the screen
      glClear(GL_COLOR_BUFFER_BIT);
      glBegin(GL_QUADS);
      glTexCoord2f (0.0, 0.0);
      glVertex3f (0.0, 0.0, 0.0);
      glTexCoord2f (1.0, 0.0);
      glVertex3f (Xsize, 0.0, 0.0);
      glTexCoord2f (1.0, 1.0);
      glVertex3f (Xsize, Ysize, 0.0);
      glTexCoord2f (0.0, 1.0);
      glVertex3f (0.0, Ysize, 0.0);
      glEnd();
      glutSwapBuffers();
      
      /* save */
      if (avconv) {
        glReadPixels(0, 0, Xsize, Ysize, GL_RGB, GL_UNSIGNED_BYTE, pixels);
        fwrite(pixels ,Xsize*Ysize*3 , 1, avconv);
        //add_frame_tomovie (&pixels);
      }
      
      // refesh rate dpending on Update or not
      if (Update==0) 
        RefreshRate = 1; 
      else 
        RefreshRate = 50;
    }
    previous = actual;
  }    
}


// GLUT special key functions
void keySpecCB (int key, int x, int y) {

  switch (key) {
    case GLUT_KEY_UP:
      delta_rad += 0.01;
    break;
    case GLUT_KEY_DOWN:
      delta_rad -= 0.01;
    break;
    case GLUT_KEY_LEFT:
      Shift -= 0.1;
      if (Shift > 2.0*M_PI)
        Shift = 0; 
    break;
    case GLUT_KEY_RIGHT:
      Shift += 0.1;
      if (Shift < 0)
        Shift = 0; 
    break;
  }
}

// GLUT normal key functions
void keyCB (unsigned char key, int x, int y) {
  
  size_t pitch;
  static int fullscreen = 0, px, py, wx, wy;

#ifdef DEBUGGING
  if (key == ' ') {
    Paused = 0;
  }
#else
  // stop simulation
  if (key == ' ') {
    Paused = 1 - Paused;
  }
#endif
  
  // display control functions
  //-------------------------------------------------------------------------------------------------
  // zoom in
  if (key == '+')  {
    Zoom *= 1.4;
    if (DustParticles)
      reset_rgba (0, plot_rgba, ColBackground);
  }
  // zoom out
  if (key == '-') {
    Zoom /= 1.4;
    if (DustParticles)
      reset_rgba (0, plot_rgba, ColBackground);
  }
  // centering on planet 0
  if (key == 's') {
    CenterPlanet = 1-CenterPlanet;
    if (DustParticles)
      reset_rgba (0, plot_rgba, ColBackground);
  }
  // full screen
  if (key == 'f') {
    // if video is creating do not change anithing
    if (avconv)
      return;
    fullscreen = 1-fullscreen;
    if (fullscreen == 1) {
      px = glutGet (GLUT_WINDOW_X);
      py = glutGet (GLUT_WINDOW_Y);
      wx = glutGet (GLUT_WINDOW_WIDTH);
      wy = glutGet (GLUT_WINDOW_HEIGHT);
      glutFullScreen();
    }
    if (fullscreen == 0) {
      glutReshapeWindow(wx, wy);
      glutPositionWindow(px, py);
    }
  }

  // temporarily turn off real time displaying
  if (key == 'w') {  
    Update = !Update;
    if (Update == 0) {
      old_palette_nb = palette_nb;
      palette_nb = 0;
    }
    else if (Update == 1) {
      palette_nb = old_palette_nb;
      if (FTD != NULL)
        get_minmax_gpu (FTD, &minftd, &maxftd);
    }
    load_palette ();
  }
  // exit simulation
  if (key == 27) {
    printf ("\n");
    // close movie if it is creating
    if (avconv) {
      pclose(avconv);
      free (pixels);
    }
    exit (0);
  }
  // change view type between polar and cartesian
  if (key == 'c') {
    // if video is creating do not change anithing
    if (avconv)
      return;
    // no full screen for  
    if (fullscreen == 1) 
      return;
    CartesianView = 1-CartesianView;
    // cartesian grid
    if (CartesianView == 1) {
      Xsize = NSEC;
      Ysize = NRAD;
      xview = NSEC;
      yview = NRAD;
    } 
    // polar grid
    else {
      Ysize = xview = WINSIZE;
      Xsize = yview = WINSIZE;
    }
    glutReshapeWindow (xview, yview);
    glutPostRedisplay();
    checkCudaErrors(cudaGLUnregisterBufferObject (gl_PBO));
    glDeleteBuffers (1, &gl_PBO);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, Xsize, Ysize, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    pitch = Xsize*sizeof(double);
    glGenBuffers(1, &gl_PBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, gl_PBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER_ARB, pitch*Ysize, NULL, GL_STREAM_COPY);
    checkCudaErrors( cudaGLRegisterBufferObject(gl_PBO) );
    glFlush ();
  }
  // change color table for hydro 
  if (key == 'p') {
    palette_nb++;
    load_palette ();
  } 
  // lock planet
  if (key == 'l') {
    Lock = !Lock;
  }
  // logarithmic filed
  if (key == 'L') {
    LogDisplay = !LogDisplay;
  }
  // increase gamma
  if (key == '1') {
    DispGamma /= 1.25;
    if (DispGamma < 0.0001) DispGamma = 0.0001;
  }
  // decrease gamma
  if (key == '2') {
    DispGamma *= 1.25;
    if (DispGamma > 1000) DispGamma = 1000;
  }
  // decrease maximum value to plot
  if (key == '3') {
    Top /= 1.25;
    if (Top < 1e-6) Top = 1e-6;
  }
  // increase maximum value to plot
  if (key == '4') {
    Top *= 1.25;
    if (Top > 1) Top = 1;
  }
  // reset max, min and gamma
  if (key == '0') {
    DispGamma = 1.0;
    Top = 1.0;
    
    // recalcalculate requested value
    if (CalcVortensity == YES) 
      CalcVortens_gpu (gas_density, gas_v_rad, gas_v_theta, Work);
    if (Adiabatic) {
      if(CalcTemperature)
        CalcTemp_gpu (gas_density, gas_energy, Work);
      else if (CalcDiskHeight)
        CalcDiskHeight_gpu (gas_density, gas_energy, Work);
      else if (CalcSoundSpeed)
        CalcSoundSpeed_gpu (gas_density, gas_energy, Work);
    }
    if (CalcDustGasRatio && DustGrid)
      CalcDustGasMassRatio_gpu(gas_density, dust_density[color_idx], Work);
    if (CalcDustSize)
      CalcDustSize_gpu(dust_size, dust_density[0], Work);
    
    if (FTD != NULL) {
#ifdef FARGO_INTEGRATION
      if (bFine)
        get_minmax_fine_gpu (FineFTD, &minftd, &maxftd);
      else
#endif
        get_minmax_gpu (FTD, &minftd, &maxftd);

      /*
      if (minftd == maxftd) {
        minftd = SIGMA0 / 3.0;
        maxftd = SIGMA0 * 3.0;
      }
      */

    }
    printf ("\n%s min/max: %e/%e\n", FTD->Name, minftd, maxftd);
  }
  // change background color
  if (key == 'I') {
    // to black
    if (ColBackground == 255+256*255+256*256*255)
      ColBackground = 0;
    //to white
    else if (ColBackground == 0)
      ColBackground = 255+256*255+256*256*255; 
  }  
  //-------------------------------------------------------------------------------------------------


  // select gas field to display
  //-------------------------------------------------------------------------------------------------
  // gas surface mass density
  if (key == 'W') {
    bFine = false;
    FTD = myWork;
    DustGridType = 0;
    CalcVortensity = NO;
    CalcTemperature = NO;
    CalcDiskHeight = NO;
    DustParticlesonGrid = NO;
    DustParticles = NO;
//    Dust2GassMassRatio = NO;
    CalcSoundSpeed = NO;
    CalcDustGasRatio = NO;
    get_minmax_gpu (FTD, &minftd, &maxftd);
    if (minftd == maxftd) {
      minftd = SIGMA0 / 3.0;
      maxftd = SIGMA0 * 3.0;
    }
    DispGamma = 1.0; Top = 1.0;
    snprintf(winTitle, 1023, "GFARGO:%s - ???????? %s", simdir, winTitleMessage);
    glutSetWindowTitle (winTitle);
  }  

  // gas surface mass density
  if (key == 'd') {
    bFine = false;
    FTD = gas_density;
    DustGridType = 0;
    CalcVortensity = NO;
    CalcTemperature = NO;
    CalcDiskHeight = NO;
    DustParticlesonGrid = NO;
    DustParticles = NO;
//    Dust2GassMassRatio = NO;
    CalcSoundSpeed = NO;
    CalcDustGasRatio = NO;
    CalcDustSize = NO;
    get_minmax_gpu (FTD, &minftd, &maxftd);
    if (minftd == maxftd) {
      minftd = SIGMA0 / 3.0;
      maxftd = SIGMA0 * 3.0;
    }
    DispGamma = 1.0; Top = 1.0;
    snprintf(winTitle, 1023, "GFARGO:%s - gas density %s", simdir, winTitleMessage);
    glutSetWindowTitle (winTitle);
  }  
  // gas radial velocity
  if (key == 'r') {
    bFine = false;
    FTD = gas_v_rad;
    DustGridType = 0;
    CalcVortensity = NO;
    CalcTemperature = NO;    
    CalcDiskHeight = NO;
    DustParticlesonGrid = NO;
    DustParticles = NO;
//    Dust2GassMassRatio = NO;
    CalcSoundSpeed = NO;
    CalcDustGasRatio = NO;
    CalcDustSize = NO;
    get_minmax_gpu (FTD, &minftd, &maxftd);
    DispGamma = 1.0; Top = 1.0;
    snprintf(winTitle, 1023, "GFARGO:%s - gas radial velocity %s", simdir, winTitleMessage);
    glutSetWindowTitle (winTitle);
  }
  // gas azimuthal velocity field
  if (key == 't') {
    bFine = false;
    FTD = gas_v_theta;
    DustGridType = 0;
    CalcVortensity = NO;
    CalcTemperature = NO;    
    CalcDiskHeight = NO;
    DustParticlesonGrid = NO;
    DustParticles = NO;
//    Dust2GassMassRatio = NO;    
    CalcSoundSpeed = NO;
    CalcDustGasRatio = NO;    
    CalcDustSize = NO;
    DispGamma = 1.0; Top = 1.0;
    get_minmax_gpu (FTD, &minftd, &maxftd);
    snprintf(winTitle, 1023, "GFARGO:%s - gas azimuthal velocity %s", simdir, winTitleMessage);
    glutSetWindowTitle (winTitle);
  }
  // gas vortensity
  /*if (key == 'v') {
    bFine = false;
    FTD = Work;
    DustGridType = 0;
    CalcVortensity = YES;
    CalcTemperature = NO;
    CalcDiskHeight = NO;
    DustParticlesonGrid = NO;
    DustParticles = NO;
//    Dust2GassMassRatio = NO;    
    CalcSoundSpeed = NO;
    CalcVortens_gpu (gas_density, gas_v_rad, gas_v_theta, Work);
    get_minmax_gpu (FTD, &minftd, &maxftd);
    snprintf(winTitle, 1023, "GFARGO:%s - gas vortensity %s", simdir, winTitleMessage);
    glutSetWindowTitle (winTitle);
  }*/  
  
  // gas disk eccentricity
  if (key == 'e') {
    bFine = false;
    FTD = disk_ecc;
    DustGridType = 0;
    CalcVortensity = NO;
    CalcTemperature = NO;
    CalcDiskHeight = NO;
    DustParticlesonGrid = NO;
    DustParticles = NO;
//    Dust2GassMassRatio = NO;    
    CalcSoundSpeed = NO;
    CalcDustGasRatio = NO;
    CalcDustSize = NO;
    DispGamma = 1.0; Top = 1.0;
    get_minmax_gpu (FTD, &minftd, &maxftd);
    snprintf(winTitle, 1023, "GFARGO:%s - gas disk eccentricity %s", simdir, winTitleMessage);
    glutSetWindowTitle (winTitle);
  }
  // gas disk eccentricity
  if (key == 'P') {
    bFine = false;
    FTD = Potential;
    DustGridType = 0;
    CalcVortensity = NO;
    CalcTemperature = NO;
    CalcDiskHeight = NO;
    DustParticlesonGrid = NO;
    DustParticles = NO;
//    Dust2GassMassRatio = NO;
    CalcSoundSpeed = NO;
    CalcDustGasRatio = NO; 
    CalcDustSize = NO;
    DispGamma = 1.0; Top = 1.0;
    get_minmax_gpu (FTD, &minftd, &maxftd);
    snprintf(winTitle, 1023, "GFARGO:%s - disk grap pot %s", simdir, winTitleMessage);
    glutSetWindowTitle (winTitle);
  }

  // select adiabatic gas field to display
  //-------------------------------------------------------------------------------------------------
  if (Adiabatic || AdaptiveViscosity) {
    // viscosity field (for adaptive or adiabatic disks)
    if (key == 'a') {
        bFine = false;
        FTD = Viscosity;
        DustGridType = 0;
        CalcVortensity = NO;
        CalcTemperature = NO;      
        CalcDiskHeight = NO;
        DustParticlesonGrid = NO;
        DustParticles = NO;
//        Dust2GassMassRatio = NO;    
        CalcSoundSpeed = NO;
        CalcDustGasRatio = NO;
        CalcDustSize = NO;
        DispGamma = 1.0; Top = 1.0;
        get_minmax_gpu (FTD, &minftd, &maxftd);
        snprintf(winTitle, 1023, "GFARGO:%s - viscosity %s", simdir, winTitleMessage);
        glutSetWindowTitle (winTitle);
    }
  }
  if (Adiabatic) {
    // gas internal energy
    if (key == 'q') {
      bFine = false;
      FTD = gas_energy;
      DustGridType = 0;
      CalcVortensity = NO;
      CalcTemperature = NO;
      CalcDiskHeight = NO;
      DustParticlesonGrid = NO;
      DustParticles = NO;
//      Dust2GassMassRatio = NO;
      CalcSoundSpeed = NO;
      CalcDustGasRatio = NO;
      CalcDustSize = NO;
      DispGamma = 1.0; Top = 1.0;
      get_minmax_gpu (FTD, &minftd, &maxftd);
      snprintf(winTitle, 1023, "GFARGO:%s - gas specific energy %s", simdir, winTitleMessage);
      glutSetWindowTitle (winTitle);
    }    
    // gas temperature
    if (key == 'Q') {
      bFine = false;
      FTD = Work;
      CalcTemperature = YES;
      DustGridType = 0;
      CalcVortensity = NO;
      CalcDiskHeight = NO;
      DustParticlesonGrid = NO;
      DustParticles = NO;
      //Dust2GassMassRatio = NO;
      CalcSoundSpeed = NO;
      CalcDustGasRatio = NO;
      CalcDustSize = NO;
      CalcTemp_gpu (gas_density, gas_energy, Work);
      DispGamma = 1.0; Top = 1.0;
      get_minmax_gpu (FTD, &minftd, &maxftd);
      snprintf(winTitle, 1023, "GFARGO:%s - gas temperature %s", simdir, winTitleMessage);
      glutSetWindowTitle (winTitle); 
    }  
    // disk adiabatic height
    if (key == 'h') {
      bFine = false;
      FTD = Work;
      CalcDiskHeight = YES;
      DustGridType = 0;
      CalcVortensity = NO;
      CalcTemperature = NO;
      DustParticlesonGrid = NO;
      DustParticles = NO;
      //Dust2GassMassRatio = NO;
      CalcSoundSpeed = NO;
      CalcDustGasRatio = NO;
      CalcDustSize = NO;
      CalcDiskHeight_gpu (gas_density, gas_energy, Work);
      DispGamma = 1.0; Top = 1.0;
      get_minmax_gpu (FTD, &minftd, &maxftd);
      snprintf(winTitle, 1023, "GFARGO:%s - disk height %s", simdir, winTitleMessage);
      glutSetWindowTitle (winTitle); 
    }    
    // disk soundspeed
    if (key == 'H') {
      bFine = false;
      FTD = Work;
      CalcSoundSpeed = YES;
      CalcDiskHeight = NO;
      DustGridType = 0;
      CalcVortensity = NO;
      CalcTemperature = NO;
      DustParticlesonGrid = NO;
      DustParticles = NO;
      //Dust2GassMassRatio = NO;
      CalcDustGasRatio = NO;
      CalcDustSize = NO;
      CalcSoundSpeed_gpu (gas_density, gas_energy, Work);
      DispGamma = 1.0; Top = 1.0;
      get_minmax_gpu (FTD, &minftd, &maxftd);
      snprintf(winTitle, 1023, "GFARGO:%s - disk cs %s", simdir, winTitleMessage);
      glutSetWindowTitle (winTitle); 
    }    
  }
  
  // select dust field to display
  //-------------------------------------------------------------------------------------------------
  // dust surface mass density
  if (DustGrid) {
    if (key == 'x') {
      bFine = false;
      FTD = dust_density[color_idx];
      CalcVortensity = NO;
      CalcTemperature = NO;
      CalcDiskHeight = NO;
      DustGridType = 1;
      DustParticlesonGrid = NO;
      DustParticles = NO;
      CalcDustGasRatio = NO;
      DispGamma = 1.0; Top = 1.0;      
      CalcDustSize = NO;
      get_minmax_gpu (FTD, &minftd, &maxftd);
      if (DustConstStokes)
        sprintf (winTitle, "GFARGO:%s - dust dens [St=%0.2e] %s", simdir, DustSizeBin[color_idx], winTitleMessage);
      else if (DustGrowth)
        if (color_idx == 0)
          sprintf (winTitle, "GFARGO:%s - grown dust dens %s", simdir, winTitleMessage);
        else
          sprintf (winTitle, "GFARGO:%s - small dust dens %s", simdir, winTitleMessage);
      else
        sprintf (winTitle, "GFARGO:%s - dust dens [s=%0.2e cm] %s", simdir, DustSizeBin[color_idx], winTitleMessage);
      glutSetWindowTitle (winTitle);
    }

    // dust radial velocity
    if (key == 'y') {
      bFine = false;
      FTD = dust_v_rad[color_idx];
      DustGridType = 2;
      CalcVortensity = NO;
      CalcTemperature = NO;
      CalcDiskHeight = NO;
      DustParticlesonGrid = NO;
      DustParticles = NO;
      CalcDustGasRatio = NO;
      CalcDustSize = NO;
      DispGamma = 1.0; Top = 1.0;
      get_minmax_gpu (FTD, &minftd, &maxftd);
      sprintf (winTitle, "GFARGO:%s - dust vrad [%0.2e cm] %s", simdir, DustSizeBin[color_idx], winTitleMessage);
      glutSetWindowTitle (winTitle);
    }

    // dust azimuthal velocity
    if (key == 'z') {
      bFine = false;
      FTD = dust_v_theta[color_idx];
      DustGridType = 3;
      CalcVortensity = NO;
      CalcTemperature = NO;
      CalcDiskHeight = NO;
      DustParticlesonGrid = NO;
      DustParticles = NO;
      CalcDustGasRatio = NO;
      CalcDustSize = NO;
      DispGamma = 1.0; Top = 1.0;
      get_minmax_gpu (FTD, &minftd, &maxftd);
      sprintf (winTitle, "GFARGO:%s - dust vth [%0.2e cm] %s", simdir, DustSizeBin[color_idx], winTitleMessage);
      glutSetWindowTitle (winTitle);
    }

    // grown dust size
    if (DustGrowth)
    if (key == 'b') {
      bFine = false;
      FTD = dust_size;
      //FTD = Work;
      DustGridType = 4;
      CalcVortensity = NO;
      CalcTemperature = NO;
      CalcDiskHeight = NO;
      DustParticlesonGrid = NO;
      DustParticles = NO;
      CalcDustGasRatio = NO;
      CalcDustSize = NO;
      //CalcDustSize = YES;
      //CalcDustSize_gpu(dust_size, dust_density[0], Work);
  
      DispGamma = 1.0; Top = 1.0;
      get_minmax_gpu (FTD, &minftd, &maxftd);
      sprintf (winTitle, "GFARGO:%s - dust size %s", simdir, winTitleMessage);
      glutSetWindowTitle (winTitle);
    }
    
    // dust-to-gas mass ratio
    if (key == 'G') {
      bFine = false;
      FTD = Work;
      DustGridType = 5;
      CalcVortensity = NO;
      CalcTemperature = NO;
      CalcDiskHeight = NO;
      DustParticlesonGrid = NO;
      DustParticles = NO;
      CalcDustGasRatio = YES;
      CalcDustSize = NO;
      CalcDustGasMassRatio_gpu(gas_density, dust_density[color_idx], Work);
      DispGamma = 1.0; Top = 1.0;
      get_minmax_gpu (FTD, &minftd, &maxftd);
      sprintf (winTitle, "GFARGO:%s - Md/Mg [%0.2e cm] %s", simdir, DustSizeBin[color_idx], winTitleMessage);
      glutSetWindowTitle (winTitle);
    }
    
    // change color palette for dust representation
    if (key == ']') {
      if (color_idx < DustBinNum-1) 
        color_idx++;
      else
        color_idx=0;
      switch (DustGridType) {
        case 1: keyCB ('x', 0,0); break;
        case 2: keyCB ('y', 0,0); break;
        case 3: keyCB ('z', 0,0); break;
        case 4: keyCB ('b', 0,0); break;
        case 5: keyCB ('G', 0,0); break;        
      }
    }
    // change color palette for dust representation
    if (key == '[') {
      if (color_idx > 0) 
        color_idx--;
      else
        color_idx=DustBinNum-1;
      switch (DustGridType) {
        case 1: keyCB ('x', 0,0); break;
        case 2: keyCB ('y', 0,0); break;
        case 3: keyCB ('z', 0,0); break;
        case 4: keyCB ('b', 0,0); break;
        case 5: keyCB ('G', 0,0); break;        
      }
    }
  }
  //-------------------------------------------------------------------------------------------------

#ifdef FARGO_INTEGRATION
  // HIPERION integration
  //-------------------------------------------------------------------------------------------------
  // interpolated surface mass density on the fine grid
  if (key == 'D') {
    bFine = true;
    FineFTD = fine_gas_density;
    DustGridType = 0;
    CalcVortensity = NO;
    CalcTemperature = NO;
    CalcDiskHeight = NO;
    DustParticlesonGrid = NO;
    DustParticles = NO;
    //Dust2GassMassRatio = NO;
    CalcDustGasRatio = NO;    
    DispGamma = 1.0; Top = 1.0;
    get_minmax_fine_gpu (FineFTD, &minftd, &maxftd);

    if (minftd == maxftd) {
      minftd = SIGMA0 / 3.0;
      maxftd = SIGMA0 * 3.0;
    }
    snprintf(winTitle, 1023, "GFARGO:%s - gas density (ovrs) %s", simdir, winTitleMessage);
    glutSetWindowTitle (winTitle);
  }
  // interpolated radial gas velocity on the fine grid
  if (key == 'R') {
    bFine = true;
    FineFTD = fine_gas_dv_rad;
    DustGridType = 0;
    CalcVortensity = NO;
    CalcTemperature = NO;    
    CalcDiskHeight = NO;
    DustParticlesonGrid = NO;
    DustParticles = NO;
    //Dust2GassMassRatio = NO;
    CalcDustGasRatio = NO;    
    DispGamma = 1.0; Top = 1.0;
    get_minmax_fine_gpu (FineFTD, &minftd, &maxftd);
    snprintf(winTitle, 1023, "GFARGO:%s - gas radial velocity (ovrs) %s", simdir, winTitleMessage);
    glutSetWindowTitle (winTitle);
  }
  // interpolated azimuthal gas velocity field on the fine grid
  if (key == 'T') {
    bFine = true;
    FineFTD = fine_gas_dv_theta;
    DustGridType = 0;
    CalcVortensity = NO;
    CalcTemperature = NO;    
    CalcDiskHeight = NO;
    DustParticlesonGrid = NO;
    DustParticles = NO;
    //Dust2GassMassRatio = NO;
    CalcDustGasRatio = NO;    
    DispGamma = 1.0; Top = 1.0;
    get_minmax_fine_gpu (FineFTD, &minftd, &maxftd);
    snprintf(winTitle, 1023, "GFARGO:%s - gas azimuthal velocity (ovrs) %s", simdir, winTitleMessage);
    glutSetWindowTitle (winTitle);
  }
  // flow pattern of dust
  if (key == 'n') {
    FlowPattern = !FlowPattern;
    if (!FlowPattern)
      field (0, plot_rgba, ColBackground);
  }
  // dust distribution
  if (key == 'b') {
    Vortensity = NO;
    CalcVortensity = NO;
    CalcTemperature = NO;
    CalcDiskHeight = NO;
    DustParticles = YES;
    DustParticlesonGrid = NO;
    //Dust2GassMassRatio = NO;
    CalcDustGasRatio = NO;    
    DispGamma = 1.0; Top = 1.0;
    sprintf (winTitle, "GFARGO:%s - dust [%0.2e cm] %s", simdir, dDustSize[color_idx], winTitleMessage);
    glutSetWindowTitle (winTitle);
    field (0, plot_rgba,ColBackground);
    FTD = NULL;
  }
  // turn off
  if (key == 'B') {
    CalcVortensity = NO;
    CalcTemperature = NO;
    CalcDiskHeight = NO;
    //Dust2GassMassRatio = NO;
    CalcDustGasRatio = NO;
    if (FTD != NULL)
      DustParticles = !DustParticles;
    palette_nb = 0;
    load_palette ();

  }
  // change color palette for dust representation
  if (key == ']') {
    if (color_idx < iDustBinNum-1) 
      color_idx++;
    else
      color_idx=0;
    if (DustParticles) {
      sprintf (winTitle, "GFARGO:%s - dust [%0.2e cm] %s", simdir, dDustSize[color_idx], winTitleMessage);
      glutSetWindowTitle (winTitle);
    }
  }
  if (key == '[') {
    if (color_idx > 0) 
      color_idx--;
    else
      color_idx=iDustBinNum-1;
    if (DustParticles) {
      sprintf (winTitle, "GFARGO:%s - dust [%0.2e cm] %s", simdir, dDustSize[color_idx], winTitleMessage);
      glutSetWindowTitle (winTitle);
    }
  }
#ifdef DUST_FEEDBACK  
  if (key == 'g') {
    DustParticles = NO;
    CalcVortensity = NO;
    CalcTemperature = NO;
    CalcDiskHeight = NO;
    DustParticlesonGrid = YES;
    //Dust2GassMassRatio = NO;
    CalcDustGasRatio = NO;    
    FTD = DustDens;
    get_minmax_gpu (FTD, &minftd, &maxftd);  
    DispGamma = 1.0; Top = 1.0;
    sprintf (winTitle, "GFARGO:%s - dust on grid [%0.2e cm] %s", simdir, dDustSize[color_idx], winTitleMessage);
    glutSetWindowTitle (winTitle);
  }

  /*if (key == 'G') {
    CalcVortensity = NO;
    CalcTemperature = NO;
    CalcDiskHeight = NO;
    DustParticles = NO;
    //Dust2GassMassRatio = YES;
    FTD = D2GMass;
    get_minmax_gpu (FTD, &minftd, &maxftd);    

    printf ("\n%e %e\n", minftd, maxftd);
    sprintf (winTitle, "GFARGO:%s - dust-to-gas mass ratio %s", simdir, winTitleMessage);
    glutSetWindowTitle (winTitle);
  */
  }
#endif
  //-------------------------------------------------------------------------------------------------
#endif
  

  // screenshots & movie
  //-------------------------------------------------------------------------------------------------
  // initiate movie
  if (key == 'M') {
    av_count++;
    snprintf(movie_filename, 1023, "GFARGO_%i.mov", av_count);
    //snprintf (avconv_str, 1024, "avconv -y -f rawvideo -qscale 10 -s %ix%i -pix_fmt rgb24 -r 25 -i - -vf vflip -an -b:v 20M %s", Xsize, Ysize, movie_filename);
    snprintf (avconv_str, 1024, "avconv -y -f rawvideo -s %ix%i -pix_fmt rgb24 -r 25 -i - -vf vflip -an -b:v 20M %s", Xsize, Ysize, movie_filename);
    avconv = popen(avconv_str, "w");
    pixels = (GLubyte *) malloc(3 * 768 * 768);
    snprintf(winTitleMessage, 1023, "(Snapshot: %s)", movie_filename);
    snprintf(winTitle, 1023, "%s %s", winTitle, winTitleMessage);
    glutSetWindowTitle (winTitle);
  }
  // stop movie
  if (key == 'm') {
    // close movie if it is opened
    if (avconv) {
      pclose(avconv);
      //ffmpeg_encoder_finish();
      free (pixels);
      avconv = NULL;
      glutSetWindowTitle (winTitle);
      snprintf(winTitleMessage, 1023, "Movie finished");
    }
  }
  // create a snapshot png
  if (key == '>') {
    snprintf(ppm_filename, 1023, "./hydro_snapshot_%d.ppm", nframes);
    screenshot_ppm(ppm_filename, Xsize, Ysize, &pixels);
    nframes++;
    printf ("\nSnapshot is taken to <%s>\n", ppm_filename);
  }
  //-------------------------------------------------------------------------------------------------
}

void resize(int w, int h) {
   glViewport (0, 0, w, h); 
   glMatrixMode (GL_PROJECTION); 
   glLoadIdentity ();
   glOrtho (0., Xsize, 0., Ysize, -200. ,200.); 
   glMatrixMode (GL_MODELVIEW); 
   glLoadIdentity ();
}

void StartMainLoop () {
  glutMainLoop();
}

