/** \file main.c

Main file of the distribution. Manages the call to initialization
functions, then the main loop.

*/

#include "fargo.h"
#include <time.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
//#include <helper_cuda.h>

#include "gpu_self_gravity.h"

// [RZS-MOD]
// global variables anf function declaration for HIPERION
//-------------------------------------------------------
#ifdef FARGO_INTEGRATION
// load global variable's deinitions
#include "../HIPERION-v1.7/globals.h"
#include "../HIPERION-v1.7/interface_extfuncs.h"
PolarGrid *CoarseDust, *DustDens, *gasdust_dens;
double *GPU_rmed, *gpu_rii, *gpu_surf, *gpu_alpha;
#endif
//-------------------------------------------------------



int main(int argc, char *argv[]) {
  int  device=0;            // default GPU device is 0
  bool disable = NO;        // 
  
  //char HIPERION_initfile[1024];
  //snprintf (HIPERION_initfile, 1024, "");
  
  if (argc == 1) PrintUsage (argv[0]);
  strcpy (ParameterFile, "");

  // interpretting initial arguments
  for (int i = 1; i < argc; i++) {
    if (*(argv[i]) == '-') {
      if (strspn (argv[i], "-sSecwrFAdDGovtpfamMzibBET0123456789") != strlen (argv[i]))
        PrintUsage (argv[0]);

      if (strchr (argv[i], 'G'))
        NoGasAccretion = YES;

      if (strchr (argv[i], 'A'))
        MonitorAccretion = YES;

      if (strchr (argv[i], 'm'))
        CreatingMovieOnly = YES;

      // do nothing ?
      if (strchr (argv[i], '0')) {
        disable = YES;
        printf ("\n");
        printf ("========================================\n");
        printf (" Just read and interpret parameter file \n");
        printf ("========================================\n");
      }

      if (strchr (argv[i], '1')) {
        OnlyInit = YES;
        printf ("\n");
        printf ("========================================\n");
        printf (" Only initialize and save initial state \n");
        printf ("========================================\n");
      }
      // verbose iteration
      if (strchr (argv[i], 'v'))
        verbose = YES;
      
      // timing info during snapshot creation
      if (strchr (argv[i], 't'))
        TimeInfo = YES;
      
      // slopy CFL setting (?)
      if (strchr (argv[i], 'c'))
        SloppyCFL = YES;
      
      // profiling the computation
      if (strchr (argv[i], 'p'))
        Profiling = YES;

      // ????? debugging
      //if (strchr (argv[i], 'd'))
	     // debug = YES;

      // [RZS-mod]
      // real-time displaying calculating on GPU
      if (strchr (argv[i], 'w'))
        Window = YES;
      
      // set initial centrifugal ballance
      if (strchr (argv[i], 'b'))
        CentrifugalBalance = YES;

      // monitoring total mass and momentum of the disk
      if (strchr (argv[i], 'a'))
	      MonitorIntegral = YES;
      
      // ??????? fake sequential
      if (strchr (argv[i], 'z'))
	      FakeSequential = YES;
      
      // ???????
      if (strchr (argv[i], 'i'))
	      StoreSigma = YES;

      // ??????
      if ((argv[i][1] >= '1') && (argv[i][1] <= '9')) {
	      GotoNextOutput = YES;
	      StillWriteOneOutput = (int)(argv[i][1]-'0');
      }
      
      // restart simulation at a given frame
      if (strchr (argv[i], 's')) {
	      Restart = YES;
	      i++;
	      NbRestart = atoi(argv[i]);
	      if ((NbRestart < 0)) {
	         masterprint ("Incorrect restart number!\n");
	         PrintUsage (argv[0]);
	      }
      }

      // set refresh rate for real-time displaying
      if (strchr (argv[i], 'r')) {
	      i++;
	      RefreshRate = atof(argv[i]);
	      if (RefreshRate <= 0.) {
	         masterprint ("Incorrect refresh rate");
	         PrintUsage (argv[0]);
	      }
      }

      // set scaling factor ()
      if (strchr (argv[i], 'f')) {
         i++;
         ScalingFactor = atof(argv[i]);
         if ((ScalingFactor <= 0)) {
            masterprint ("Incorrect scaling factor\n");
            PrintUsage (argv[0]);
         }
      }

      // monitoring baricentre
      if (strchr (argv[i], 'B'))
        MonitorBC = YES;

      // monitoring torques
      if (strchr (argv[i], 'T'))
        MonitorTorque = YES;

      // monitoring eccentricity and mass 
      if (strchr (argv[i], 'E'))
        MonitorDiskEcc = YES;

      // set output dir to ...
      if (strchr (argv[i], 'o')) {
	      OverridesOutputdir = YES;
	      i++;
	      sprintf (NewOutputdir, "%s", argv[i]);
      } 

      // [RZS-MOD]
      // barycentric initialization
      //----------------------------------------------------------
//      if (strchr (argv[i], 'B')) {
//        BaryCentric = YES;
//      }
      //----------------------------------------------------------

      // [RZS-MOD]
      // automatic restarting (required for NIIF HPC SLURM requeu)
      //----------------------------------------------------------
      if (strchr (argv[i], 'S')) {
        AutoRestart = YES;
      }
      //----------------------------------------------------------
      
      // [RZS-MOD]
      // select GPU device
      //-------------------------------------
      if (strchr (argv[i], 'D')) {
	      i++;
	      device = atoi(argv[i]);
      }
      //-------------------------------------
 
      // [RZS-MOD]
      // set termination at semi-major axis given of planet0
      if (strchr (argv[i], 'R')) {
        i++;
        MinSemiMajorPlanet = atof(argv[i]);
      }
    }

    else {
      // [RZS-MOD]
      // Define HIPERION init file
      //-------------------------------------
#ifdef FARGO_INTEGRATION
      strncpy (HIPERION_initfile, argv[i], 256);
      i++;
      //-------------------------------------
#endif
      strcpy (ParameterFile, argv[i]);
    }
  }
  
  //-------------------------------------
  if ((StoreSigma) && !(Restart)) {
    mastererr ("You cannot use tabulated surface density\n");
    mastererr ("in a non-restart run. Aborted\n");
    prs_exit (0);
  }
  if (ParameterFile[0] == 0) {
    printf ("Parameter file missing\n");
    PrintUsage (argv[0]);
  }

  // read init parameters and create output directory if neccessary
  ReadVariables (ParameterFile);

  // make output directory if it is not exist
  MakeDir (OUTPUTDIR);

  // [RZS-MOD]
  // automatic restarting (useful for SLURM)
  if (AutoRestart) {
    int curr_frame = ReadCurrentFrameNum ();
    printf ("Last saved frame: %i\n", curr_frame);
    if (curr_frame >= 0) {
      NbRestart = curr_frame;
      Restart = YES;
    }
    else {
    }
  }

  // [RZS-MOD]
  // Initialization of GFARGO's NINTERM
  //-------------------------------------
#ifdef FARGO_INTEGRATION
  // check several GFARGO parameters which are now handled by HIPERION
  if (NTOT >= 0) {
    printf ("GFARGO's NTOT must be set to negative to be consistent with HIPERION which controls it\n");
    exit (-1);
  }
  if (NINTERM >= 0) {
    printf ("GFARGO's NINTERM must be set to negative to be consistent with HIPERION which controls it\n");
    exit (-1);
  }
  if ((*FRAME != 'F') && (*FRAME != 'f')) {
    printf ("GFARGO's Frame must be set to FIXED to be consistent with HIPERION\n");
    exit (-1);
  }
  
  MonitorBC = YES;
#endif
  //-------------------------------------

  // init and display planetary system
  sys = InitPlanetarySystem (PLANETCONFIG);
//  sys = InitPlanetarySystemBin (PLANETCONFIG);
//  cacc = InitCoreAccretion (COREACCCONFIG, sys->nb, sys->mass);
//  ListCoreAccretion (cacc);

  // [RZS-MOD]
  // skiping domain split but setting GLOBALNRAD is required
  //-------------------------------------
  //SplitDomain ();
	IMIN  = 0;
  GLOBALNRAD = NRAD;
  
  if (verbose) {
    TellEverything ();
    ListPlanets(sys);
  }

  if (disable == YES) {
    printf ("==================================\n");
    printf (" Simulation is disabled, Exiting! \n");
    printf ("==================================\n");
    exit (0);
  }

  printf ("====================================== \n");
  printf (" Simulation is started from frame %i \n", NbRestart);
  printf ("====================================== \n\n");


  // [RZS-MOD]
  // Initialization of HIPERION
  //-------------------------------------
#ifdef FARGO_INTEGRATION
  // HIPERION initizlization file must be defined
  printf ("GFARGO-HIPERION running on device: %d\n", device);
  if (strcmp (HIPERION_initfile, "") == 0) {
    printf ("HIPERION init file must be set! Exiting\n");
    exit (-1);
  }
  
  // initialize argc and argv[] for HIPERION
  int HIPERION_argc;
  if (!Restart)
    HIPERION_argc = 6;
  else
    HIPERION_argc = 8;
  char **HIPERION_argv =  new char* [HIPERION_argc];
  for (int argc_i=0; argc_i<HIPERION_argc; argc_i++)
    HIPERION_argv[argc_i] = new char[1024];
  snprintf (HIPERION_argv[0], 1024, "hiperion");
  snprintf (HIPERION_argv[1], 1024, "%s", HIPERION_initfile);
  snprintf (HIPERION_argv[2], 1024, "-D");
  snprintf (HIPERION_argv[3], 1024, "%i", device);
  snprintf (HIPERION_argv[4], 1024, "-s");
  snprintf (HIPERION_argv[5], 1024, "%i", AutoRestart);
  if (Restart) {
    snprintf (HIPERION_argv[6], 1024, "-I");
    snprintf (HIPERION_argv[7], 1024, "snapshot_%06d", NbRestart);
  }
  printf ("HIPERION calling sequence: ");
  for (int argc_i=0; argc_i<HIPERION_argc; argc_i++)
    printf ("%s ", HIPERION_argv[argc_i]);
  printf("\n");
  HIPERION_Init (&HIPERION_argc, HIPERION_argv);
  IterCount = NbRestart;
#else
  SelectDevice (device);
#endif

  // dump source files to output directory
  //--------------------------------------
  //MakeDir (OUTPUTDIR);
  //DumpSources (argc, argv);
  //--------------------------------------
  
  // init OPENGL display
  if (Window)
    InitDisplay (&argc, argv);

  if (verbose) {
    printf ("\nAllocating PolaGrids\n");
    printf ("------------------------------------------------------------------------------\n");
  }
  gas_density = CreatePolarGrid(NRAD, NSEC, "gas_dens");
  gas_v_rad   = CreatePolarGrid(NRAD, NSEC, "gas_vrad");
  gas_v_theta = CreatePolarGrid(NRAD, NSEC, "gas_vtheta");
  disk_ecc    = CreatePolarGrid(NRAD, NSEC, "diskecc");
  gas_label   = CreatePolarGrid(NRAD, NSEC, "label");
  
  if (Adiabatic)
    gas_energy = CreatePolarGrid(NRAD, NSEC, "gas_energy");

  // grid based dust
  if (DustGrid) {
    dust_density = new PolarGrid* [DustBinNum];
    dust_v_rad   = new PolarGrid* [DustBinNum];
    dust_v_theta = new PolarGrid* [DustBinNum];
    char name[1024];
    for (int i=0; i< DustBinNum; i++) {
      sprintf (name, "dust_dens_s%d", i);
      dust_density[i]  = CreatePolarGrid(NRAD, NSEC, name);
      sprintf (name, "dust_vrad_s%d", i);
      dust_v_rad[i]    = CreatePolarGrid(NRAD, NSEC, name);
      sprintf (name, "dust_vtheta_s%d", i);
      dust_v_theta[i]  = CreatePolarGrid(NRAD, NSEC, name);
    }
  }

  OmegaFrame = OMEGAFRAME;
  if (Corotating == YES) 
    OmegaFrame = GetPsysInfo (sys, FREQUENCY);

  // Initialization of self-gravity potential calculator
  if (SelfGravity)
    gpu_sg_init (NRAD, NSEC, RMIN, RMAX, THICKNESSSMOOTHING*ASPECTRATIO);
    
  // [RZS-MOD]
  // Barycentric or normal initialization
  //-------------------------------------
  //if (BaryCentric)
  //  InitializationBC (sys, gas_density, gas_v_rad, gas_v_theta, gas_label);
  //else
    Initialization (gas_density, gas_v_rad, gas_v_theta, gas_energy, gas_label,
                    dust_density, dust_v_rad, dust_v_theta);
  //-------------------------------------
  
  InitComputeAccel ();
  if (Restart == YES) {
#ifndef FARGO_INTEGRATION
    IterCount     = NbRestart * NINTERM;
#endif
    RestartPlanetarySystem (NbRestart, sys);
    LostMass      = GetfromPlanetFile (NbRestart, 7, 0); // 0 refers to planet #0
    PhysicalTime  = GetfromPlanetFile (NbRestart, 8, 0);
    OmegaFrame    = GetfromPlanetFile (NbRestart, 9, 0);
  } 
  // We initialize 'planet[i].dat' file
  else {
    EmptyPlanetSystemFile (sys);
  }
  PhysicalTimeInitial = PhysicalTime;
  MultiplyPolarGridbyConstant (gas_density, ScalingFactor);
  
  // uploading all arrays to GPU
  H2D_All ();  
  

  // [RZS-MOD]
  // initialization of FARGO & HIPERION integration 
  //-----------------------------------------------------------
#ifdef FARGO_INTEGRATION
  

//  FineDust   = CreatePolarGrid(NRAD, NSEC, "Dust");
//  CoarseDust = CreatePolarGrid(NRAD/CUBICOVERSAMP, NSEC/CUBICOVERSAMP, "Dust");
  
  InitGasBiCubicInterpol (NSEC, NRAD, GASOVERSAMPRAD, GASOVERSAMPAZIM);
//  InitDustBiCubicInterpol (NSEC/CUBICOVERSAMP, NRAD/CUBICOVERSAMP, CUBICOVERSAMP);


  DustDens   = CreatePolarGrid(NRAD, NSEC, "DustDens");
  //DustMass   = CreatePolarGrid(NRAD, NSEC, "DustMass");

  
  if (DUSTCUBICUNDERSAMP >= 1) {
    CoarseDust = CreatePolarGrid(NRAD/DUSTCUBICUNDERSAMP, NSEC/DUSTCUBICUNDERSAMP, "CoarseDust");
    InitDustBiCubicInterpol (NSEC/DUSTCUBICUNDERSAMP, NRAD/DUSTCUBICUNDERSAMP, DUSTCUBICUNDERSAMP);
  }
  
  // create gas+dust density polar grid if dust feedback is turned on
  if (DustFeedBackGrav) {
    gasdust_dens = CreatePolarGrid(NRAD, NSEC, "GasDust");
  }
  
  
  
//  else {
//    CoarseDust = CreatePolarGrid(NRAD, NSEC, "CoarseDust");
//    InitDustBiCubicInterpol (NSEC, NRAD, 1);
//  }

  double *h_Rmed;
  checkCudaErrors(cudaMallocHost ((void **) &h_Rmed, sizeof(double) * NRAD));
  checkCudaErrors(cudaMalloc ((void **) &GPU_rmed, sizeof(double) * NRAD));
  for (int i=0; i< NRAD; i++) {
    h_Rmed[i] = Rmed[i];
  }  
  checkCudaErrors(cudaMemcpy(GPU_rmed, h_Rmed,  sizeof(double) * NRAD, cudaMemcpyHostToDevice));
  
  
  

  checkCudaErrors(cudaMalloc ((void **) &gpu_rii, sizeof(double) * (NRAD+1)));
  checkCudaErrors(cudaMemcpy(gpu_rii, Rinf, sizeof(double) * (NRAD+1), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc ((void **) &gpu_surf, sizeof(double) * (NRAD)));
  checkCudaErrors(cudaMemcpy(gpu_surf, Surf, sizeof(double) * (NRAD), cudaMemcpyHostToDevice));

  checkCudaErrors(cudaMalloc ((void **) &gpu_alpha, sizeof(double) * (NRAD)));
  checkCudaErrors(cudaMemcpy(gpu_alpha, alphaval, sizeof(double) * (NRAD), cudaMemcpyHostToDevice));
    
  HIPERION_InitFargoInterface (LogGrid, 
                               NRAD, NSEC, GASOVERSAMPRAD, GASOVERSAMPAZIM, DUSTCUBICUNDERSAMP, DustFeedBackDrag,
                               RMIN, RMAX,
                               gpu_alpha, FLARINGINDEX, ASPECTRATIO, SIGMASLOPE, SIGMA0,
                               GPU_rmed, gpu_rii, gpu_surf,
                               gas_density->pitch,
                               CoarseDust->gpu_field); 

#endif
  //----------------------------------------------------------- 
  
  PrintVideoRAMUsage ();

  if (verbose) {
    printf ("\n");
    printf ("==================================\n");
    printf ("        Simulation begins!        \n");
    printf ("==================================\n");
  }


  if (Window) {
    
    // initialization of OpenGL window
    //-------------------------------------
    DisplayLoadDensity ();
    //-------------------------------------
    
    StartMainLoop ();
  }
  else {
    while (1)
      Loop();
  }
  return 0;
}
