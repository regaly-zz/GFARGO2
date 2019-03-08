/** \file Interpret.c

Contains the functions required to read the parameter file,
and functions that provide runtime information. The function
var() associates a string to a global variable. The function
ReadVariables() reads the content of a parameter file.
In addition, this file contains a function that prints the
command line usage to the standard output, a function that
provides verbose information about the setup (if the -v switch
is set on the command line), and functions that act as a
chronometer (if the -t switch is set on the command line).

The code is modified by Zs. Regaly 2017
*/

#include "fargo.h"
#define MAXVARIABLES 500
extern bool verbose;
   
extern int      begin_i;

static Param    VariableSet[MAXVARIABLES];
static int      VariableIndex = 0;
static int	    FirstStep = YES;
static clock_t  First, Preceeding, Current, FirstUser, CurrentUser, PreceedingUser;
static long	    Ticks;

void var(char *name, void *ptr, int type, int necessary, char *deflt) {
  double valuer;
  int    valuei;
  double temp;
  sscanf (deflt, "%lf", &temp);
  valuer = (double) (temp);
  valuei = (int) valuer;
  strcpy(VariableSet[VariableIndex].name, name);
  VariableSet[VariableIndex].variable = (char *) ptr;
  VariableSet[VariableIndex].type = type;
  VariableSet[VariableIndex].necessary = necessary;
  VariableSet[VariableIndex].read = NO;
  if (necessary == NO) {
    if (type == INT) {
      *((int *) ptr) = valuei;
    } else if (type == REAL) {
      *((double *) ptr) = valuer;
    } else if (type == STRING) {
      strcpy ((char *) ptr, deflt);
    }
  }
  VariableIndex++;
}

void ErrorinParfile () {
  printf ("\n");
  printf ("==================================\n");
  printf ("    Error in parfameter file!     \n");
  printf ("==================================\n\n");
}


void SetDustSizeBins (char *strDustBins) {
  
  if (*strDustBins == '\0') {
    ErrorinParfile ();
    printf ("DustSizeBin is undefined!\n");
    exit (1);
  }

  // get the number of particle size bins
  char str[1024];
  strcpy (str, strDustBins);
  DustBinNum = 1;
  for (int ii = 0; str[ii] != '\0'; ii++) {
    if (str[ii] == ',')
       DustBinNum++;
  }
  if (DustBinNum == 0) {
    ErrorinParfile ();
    printf ("DustSizeBins is undefined!");
    exit (1);
  }
  DustSizeBin = new double [DustBinNum];

  // read particle sizes
  char *token;
  token = strtok(str, ",");
  int ii = 0;
  while (token != NULL) {
      DustSizeBin[ii] = atof (token);
      token = strtok (NULL, ",");
      ii++;
  }
}

void SetDustMassBins (char *strDustBins) {
  
  if (*strDustBins == '\0') {
    ErrorinParfile ();
    printf ("DustMassBin is undefined!\n");
    exit (1);
  }

  // get the number of particle size bins
  char str[1024];
  strcpy (str, strDustBins);
  int DustMassBinNum = 1;
  for (int ii = 0; str[ii] != '\0'; ii++) {
    if (str[ii] == ',')
       DustMassBinNum++;
  }
  if (DustMassBinNum != DustBinNum) {
    ErrorinParfile ();
    printf ("DustGasMassRatios and DustSizeBin have different number of parameters!\n");
    exit (1);
  }
    
  DustMassBin = new double [DustMassBinNum];

  // read particle sizes
  char *token;
  token = strtok(str, ",");
  int ii = 0;
  while (token != NULL) {
      DustMassBin[ii] = atof (token);
      token = strtok (NULL, ",");
      ii++;
  }
}

void ReadVariables(char *filename) {
  char            nm[300], s[350],stringval[290];
  char           *s1;
  double	  temp;
  double          valuer;
  int             found, valuei, success;//, type;
  int   i;
  int            *ptri;
  double         *ptrr;
  FILE           *input;

  InitVariables();
  input = fopen(filename, "r");
  if (input == NULL) {
    ErrorinParfile ();
    printf ("Unable to read '%s'. Program stopped.\n",filename);
    exit(1);
  }
  mastererr ("Reading parameters file '%s'\n", filename);
  while (fgets(s, 349, input) != NULL) {
    success = sscanf(s, "%s ", nm);
    if ((nm[0] != '#') && (success == 1)) {	// # begins a comment line
      s1 = s + strlen(nm);
      sscanf(s1 + strspn(s1, "\t :=>_"), "%lf", &temp);
      sscanf(s1 + strspn(s1, "\t :=>_"), "%289s ", stringval);
      valuer = (double) temp;
      valuei = (int) temp;
      for (i = 0; i < (int) strlen(nm); i++) {
        nm[i] = (char) toupper(nm[i]);
      }
      found = NO;
      for (i = 0; i < VariableIndex; i++) {
        if (strcmp(nm, VariableSet[i].name) == 0) {
          if (VariableSet[i].read == YES) {
            printf("Warning : %s defined more than once.\n", nm);
          }
          found = YES;
          VariableSet[i].read = YES;
          ptri = (int *) (VariableSet[i].variable);
          ptrr = (double *) (VariableSet[i].variable);
          if (VariableSet[i].type == INT) {
            *ptri = valuei;
          } else if (VariableSet[i].type == REAL) {
            *ptrr = valuer;
          } else if (VariableSet[i].type == STRING) {
            strcpy (VariableSet[i].variable, stringval);
          }
        }
      }
      if (found == NO) {
        ErrorinParfile ();
        printf("Warning : variable %s defined but non-existent in code.\n", nm);
        exit (1);
      }
    }
  }

  found = NO;
  for (i = 0; i < VariableIndex; i++) {
    if ((VariableSet[i].read == NO) && (VariableSet[i].necessary == YES)) {
      if (found == NO) {
        ErrorinParfile ();
        printf("Fatal error : undefined mandatory variable(s):\n");
        found = YES;
      }
      mastererr("%s\n", VariableSet[i].name);
    }
    if (found == YES)
      exit(1);

  }
  found = NO;
/*  if (verbose)
  for (i = 0; i < VariableIndex; i++) {
    if (VariableSet[i].read == NO) {
      if (found == NO) {
        mastererr("Secondary variables omitted:\n");
        found = YES;
      }
      // print secondary variables
      if ((type = VariableSet[i].type) == REAL)
        mastererr("%-25s\tDefault Value : %.5g\n", VariableSet[i].name, *((real *) VariableSet[i].variable));
      if (type == INT)
        mastererr("%-25s\tDefault Value : %d\n", VariableSet[i].name, *((int *) VariableSet[i].variable));
      if (type == STRING)
        mastererr("%-25s\tDefault Value : %s\n", VariableSet[i].name, VariableSet[i].variable);
    }
  }*/

  if ((NSEC % 64)  != 0) {
    ErrorinParfile ();
    mastererr ("In this version of gfargo2, NSEC (the number of zones in azimuth)\n");
    mastererr ("needs to be a multiple of 64. I suggest %d or %d, ", (NSEC / 64)*64, ((NSEC / 64)+1)*64);
    mastererr ("instead of %d.\n", NSEC);
    mastererr ("Please edit the parameter files and retry running the code.\n");
    exit (1);
  }
  if ((NRAD % 16)  != 0) {
    ErrorinParfile ();
    mastererr ("In this version of gfargo2, NRAD (the number of zones in radius)\n");
    mastererr ("needs to be a multiple of 16. I suggest %d or %d, ", (NRAD / 16)*16, ((NRAD / 16)+1)*16);
    mastererr ("instead of %d.\n", NRAD);
    mastererr ("Please edit the parameter files and retry running the code.\n");
    prs_exit (1);
  }
  if ((*ADVLABEL == 'y') || (*ADVLABEL == 'Y')) AdvecteLabel = YES;
  if ((*OUTERSOURCEMASS == 'y') || (*OUTERSOURCEMASS == 'Y')) OuterSourceMass = YES;
  if ((*TRANSPORT == 's') || (*TRANSPORT == 'S')) FastTransport = NO;
  //if ((*OPENINNERBOUNDARY == 'O') || (*OPENINNERBOUNDARY == 'o')) OpenInner = YES;
  //if ((*OPENINNERBOUNDARY == 'N') || (*OPENINNERBOUNDARY == 'n')) NonReflecting = YES;
  //if ((*OPENINNERBOUNDARY == 'S') || (*OPENINNERBOUNDARY == 's')) Stockholm = YES;

  if ((*INNERBOUNDARY == 'O') || (*INNERBOUNDARY == 'o')) OpenInner = YES;
  if ((*OUTERBOUNDARY == 'O') || (*OUTERBOUNDARY == 'o')) OpenOuter = YES;
  if ((*INNERBOUNDARY == 'D') || (*INNERBOUNDARY == 'd')) DampingInner = YES;
  if ((*OUTERBOUNDARY == 'D') || (*OUTERBOUNDARY == 'd')) DampingOuter = YES;
  if ((*INNERBOUNDARY == 'S') || (*INNERBOUNDARY == 's')) StrongDampingInner = YES;
  if ((*OUTERBOUNDARY == 'S') || (*OUTERBOUNDARY == 's')) StrongDampingOuter = YES;
  if ((*INNERBOUNDARY == 'N') || (*INNERBOUNDARY == 'n')) NonReflectingInner = YES;
  if ((*OUTERBOUNDARY == 'N') || (*OUTERBOUNDARY == 'n')) NonReflectingOuter = YES;
  if ((*INNERBOUNDARY == 'R') || (*INNERBOUNDARY == 'r')) RigidWallInner = YES;
  if ((*OUTERBOUNDARY == 'R') || (*OUTERBOUNDARY == 'r')) RigidWallOuter = YES;
  if ((*INNERBOUNDARY == 'V') || (*INNERBOUNDARY == 'v')) ViscOutflowInner = YES;
  if ((*OUTERBOUNDARY == 'V') || (*OUTERBOUNDARY == 'v')) ViscOutflowOuter = YES;
  if ((*INNERBOUNDARY == 'C') || (*INNERBOUNDARY == 'c')) ClosedInner = YES;
  if ((*OUTERBOUNDARY == 'C') || (*OUTERBOUNDARY == 'c')) ClosedOuter = YES;
  if ((*INNERBOUNDARY == 'E') || (*INNERBOUNDARY == 'e')) EmptyInner = YES;
  if ((*OUTERBOUNDARY == 'E') || (*OUTERBOUNDARY == 'e')) EmptyOuter = YES;
  

  if ((*GRIDSPACING == 'L') || (*GRIDSPACING == 'l')) LogGrid = YES;
  if ((*DISK == 'N') || (*DISK == 'n')) IsDisk = NO;
  if ((*FRAME == 'C') || (*FRAME == 'c')) Corotating = YES;
  if ((*FRAME == 'G') || (*FRAME == 'g')) {
    Corotating = YES;
    GuidingCenter = YES;
  }
  
  if (RELEASEDATE > 0.0) {
    if (RELEASERADIUS <= 0.0) {
      ErrorinParfile ();
      printf ("Please specify ReleaseRadius or unset ReleaseDate.\n");
      exit (1);
    }
  }
    
  // which snapshot to be saved
  if ((*WRITEDENSITY == 'N') || (*WRITEDENSITY == 'n')) Write_Density = NO;
  if ((*WRITEVELOCITY == 'N') || (*WRITEVELOCITY == 'n')) Write_Velocity = NO;
  if ((*WRITEENERGY == 'N') || (*WRITEENERGY == 'n')) Write_Energy = NO;
  if ((*WRITETEMP == 'Y') || (*WRITETEMP == 'y')) Write_Temperature = YES;
  if ((*WRITEDISKHEIGHT == 'Y') || (*WRITEDISKHEIGHT == 'y')) Write_DiskHeight = YES;
  if ((*WRITEPOTENTIAL == 'Y') || (*WRITEPOTENTIAL == 'y')) Write_Potential = YES;    
  if ((*WRITESOUNDSPEED == 'Y') || (*WRITESOUNDSPEED == 'y')) Write_SoundSpeed = YES;    
  
  if ((*INDIRECTTERM == 'N') || (*INDIRECTTERM == 'n')) Indirect_Term = NO;
  if ((*EXCLUDEHILL == 'N') || (*EXCLUDEHILL == 'n')) ExcludeHill = 0; // no Hill exclusion
  if ((*EXCLUDEHILL == 'G') || (*EXCLUDEHILL == 'g')) ExcludeHill = 1; // Gaussian Hill exclusiopn
  if ((*EXCLUDEHILL == 'H') || (*EXCLUDEHILL == 'h')) ExcludeHill = 2; // Heaviside Hill exclusion
  if ((*EXCLUDEHILL == 'Y') || (*EXCLUDEHILL == 'y')) {
    ErrorinParfile ();
    printf ("Please specify Hill exclusion method (NO, GAUSSIAN or HEAVISIDE)\n");
    exit (1);
  }
  if ((ALPHAVISCOSITY != 0.0) && (VISCOSITY != 0.0)) {
    ErrorinParfile ();
    printf ("You cannot use at the same time: Viscosity and AlphaViscosity.\n");
    printf ("Edit the parameter file so as to remove one of these variables and run again.\n");
    exit (1);
  }
  if (ALPHAVISCOSITY != 0.0) {
    ViscosityAlpha = YES;
  }
  if ((*ADAPTIVEVISCOSITY == 'Y') || (*ADAPTIVEVISCOSITY == 'y')) {
    ViscosityAlpha = YES;
    AdaptiveViscosity = YES;
    if (ALPHAVISCOSITY <= 0) {
      ErrorinParfile ();
      printf ("AdaptiveViscosity requires AlphaViscosity value to be defined!\n");
      exit (1);
    }
    if (ALPHAVISCOSITYDEAD <= 0) {
      ErrorinParfile ();
      printf ("AdaptiveViscosity requires AlphaViscosityDead value to be defined\n");
      exit (-1);
    }
    if (ALPHASMOOTH <= 0) {
      ErrorinParfile ();
      printf ("AdaptiveViscosity requires AlphaSmooth value to be defined\n");
      exit (1);
    }
    if (ALPHASIGMATHRESH <= 0) {
      ErrorinParfile ();
      printf ("AdaptiveViscosity requires AlphaSigmaThresh value to be defined\n");
      exit (1);
    }
  }
  if ((THICKNESSSMOOTHING != 0.0) && (ROCHESMOOTHING != 0.0)) {
    ErrorinParfile ();
    printf ("You cannot use at the same time: ThicknessSmoothing and RocheSmoothing.\n");
    printf ("Edit the parameter file so as to remove one of these variables and run again.\n");
    exit (1);
  }
  if ((THICKNESSSMOOTHING <= 0.0) && (ROCHESMOOTHING <= 0.0)) {
    ErrorinParfile ();
    printf ("A non-vanishing potential smoothing length is required.\n");
    printf ("Please use either of the following variables: ThicknessSmoothing or RocheSmoothing.\n");
    printf ("Edit the parameter file so as to set one of these variables and run again.\n");
    exit (1);
  }
  if (ROCHESMOOTHING != 0.0) {
    RocheSmoothing = YES;
    printf ("Planet potential smoothing scales with their Hill sphere.\n");
  }
  if (OverridesOutputdir == YES) {
    sprintf (OUTPUTDIR, "%s", NewOutputdir);
  }
  /* Add a trailing slash to OUTPUTDIR if needed */
  if (*(OUTPUTDIR+strlen(OUTPUTDIR)-1) != '/')
    strcat (OUTPUTDIR, "/");
  
#ifdef FARGO_INTEGRATION
  // RZS-MOD
  // self-gravity
  //-------------
  if ((*DUSTFEEDBACK == 'Y') || (*DUSTFEEDBAC == 'y'))
    DustFeedback = YES;

  if ((*DUSTFEEDBACKGRAV == 'Y') || (*DUSTFEEDBACKGRAV == 'y'))
    DustFeedBackGrav = YES;
#endif
  
  if ((*SELFGRAVITY == 'Y') || (*SELFGRAVITY == 'y')) {
    if (LogGrid) {
      SelfGravity = YES;
    }
    else {
      ErrorinParfile ();
      printf ("For self-gravitating disk logarithmic grid is required!\n");
      exit (1);
    }
    //if (ExcludeHill > 0) {
    //  masterprint ("For self-gravitating disk ExcludeHill must be turned off!\n");
    //  prs_exit (1);
    //}
  }
  
  // dust related parameters
  if ((*DUSTGRID == 'Y') || (*DUSTGRID == 'y')) {
    DustGrid = true;
    SetDustSizeBins (DUSTSIZEBIN);
    SetDustMassBins (DUSTMASSBIN);
  }
  else {
    DustGrid = NO;
    DustBinNum = 0;
  }
  if ((*DUSTCONSTSTOKES == 'Y') || (*DUSTCONSTSTOKES == 'y')) {
    DustConstStokes = YES;
  }
  if ((*DUSTSELFGRAVITY == 'Y') || (*DUSTSELFGRAVITY == 'y')) {
    if ((*SELFGRAVITY == 'N') || (*SELFGRAVITY == 'n')) {
      ErrorinParfile ();
      printf ("For dust self-gravity disk self-gravity must be turned on!\n");
      exit (1);
    }
    DustSelfGravity = YES;
  }

  if ((*DUSTGROWTH == 'Y') || (*DUSTGROWTH == 'y')) {
    DustGrowth = YES;
    if (DustConstStokes) {
      ErrorinParfile ();
      printf ("DustGrowth excludes DustConstSokes, please set DustConstSokes=NO!\n");
      exit (1);
    }
    if (DustBinNum != 2) {
      ErrorinParfile ();
      printf ("DustGrowth requres two dust size bins!\n");
      exit (1);
    }
    if (!ViscosityAlpha) {
      ErrorinParfile ();
      printf ("DustGrowth requres AlphaViscosity!\n");
      exit (1);
    }
    if (DUSTVFRAG == 0) {
      ErrorinParfile ();
      printf ("DustVFrag must be set!\n");
      exit (1);      
    }
  }
    
  if ((*DUSTFEEDBACK == 'Y') || (*DUSTFEEDBACK == 'y')) {
    DustFeedback = YES;
    if (DustBinNum != 1 && !DustGrowth) {
      ErrorinParfile ();
      printf ("DustFeedback without DustGrowth only work for single dust bin!\n");
      exit (1);
    }
  }
  
  
  // adiabatic gas related parameters
  if ((*ADIABATIC == 'Y') || (*ADIABATIC == 'y')) {
    Adiabatic = YES;
    if (ADIABATICINDEX <= 0.0) {
      ErrorinParfile ();
      printf ("For adiabatic gas AdiabaticIndex must be defined!\n");
      exit (1);
    }
    if ((*COOLING == 'Y') || (*COOLING == 'y')) {
      Cooling = YES;
      if (COOLINGTIME0 <= 0.0) {
        ErrorinParfile ();
        printf ("Cooling is defined, CoolingTIme0 should be gretaer than zero!\n");
        exit (1);
      }
    }
    else
      if (COOLINGTIME0 > 0) {
        ErrorinParfile ();
        printf ("CoolingTime0 is defined, but Cooling is turned off!\n");
        exit (1);
      }
  }
  
  if ((*VISCHEATING == 'Y') || (*VISCHEATING == 'y')) {
    if (Adiabatic)
      ViscHeating = YES;
    else {
      ErrorinParfile ();
      printf ("ViscHeating is defined, disk must be Adiabatic!\n");
      exit (1);
    } 
  }
  
  if ((*DEADZONE == 'Y') || (*DEADZONE == 'y')) {
    DeadZone = YES;
    if (AdaptiveViscosity) {
      ErrorinParfile ();
      printf ("DeadZone and AdaptiveViscosity are mutually exclusive!\n");
      exit (1);
    }
  }

  if ((*NEBULA == 'Y') || (*NEBULA == 'y')) {
    ModelNebula = YES;
    if (NEBULAMASS == 0) {
      ErrorinParfile ();
      printf ("Nebula is defined, therefore NebulaMass must be defined!\n");
      exit (1);
    }
    if (NEBULAEXT  == 0) {
      ErrorinParfile ();
      printf ("Nebula is defined, therefore NebulaRad must be defined!\n");
      exit (1);
    }
    if (NEBULAPOSX  == 0) {
      ErrorinParfile ();
      printf ("Nebula is defined, therefore NebulaPosX must be defined!\n");
      exit (1);
    }
    if (NEBULAPOSY  == 0) {
      ErrorinParfile ();
      printf ("Nebula is defined, therefore NebulaPosY must be defined!\n");
      exit (1);
    }
  }
  
  // MassTaper must be non-zero
  if (MASSTAPER == 0)
    MASSTAPER = 1e-10;
  
   if ((*ZEROVELRAD == 'Y') || (*ZEROVELRAD == 'y'))
     ZeroVelrad = true;
}


void PrintUsage (char *execname) {
  mastererr("Usage : %s [-Dabcdimnptvz] [-(0-9)] [-r fps] [-s number] [-f scaling] parameters file\n\n", execname);
  mastererr("-D : Define the GPU device to use (see nvidia-smi output for available devices)\n");
  mastererr("-B : Baricentric initialization\n");
  mastererr("-T : Monitor planetary torques\n");
  mastererr("-E : Monitor disk eccentricity\n");
  mastererr("-G : No gas accretion allowed\n");
  mastererr("-a : Monitor mass and angular momentum at each timestep\n");
  mastererr("-b : Adjust azimuthal velocity to impose strict centrifugal balance at t=0\n");
  mastererr("-c : Sloppy CFL condition (checked at each DT, not at each timestep)\n");
  mastererr("-f : Scale density array by 'scaling'. Useful to increase/decrease\n");
  mastererr("     disk surface density after a restart, for instance.            \n");
  mastererr("-w : enable window (OpenGL builts only)\n");
  mastererr("-i : tabulate Sigma profile as given by restart files\n");
  mastererr("-n : Disable simulation, the program just reads parameters file\n");
  mastererr("-o : Overrides output directory of input file.\n");
  mastererr("-p : Give profiling information at each time step\n");
  mastererr("-r : Refresh rate, in frames per second.\n");
  mastererr("-s : Restart simulation, taking defined [number] snapshots as initial conditions\n");
  mastererr("-S : Restart simulation from the latest snapshots (using lastframe.dat)\n");
  mastererr("-t : Monitor CPU time usage at each time step\n");
  mastererr("-v : Verbose mode, tells everything about parameters\n");
  mastererr("-0 : only write initial (or restart) HD meshes,\n");
//  mastererr("-(0-9) : only write initial (or restart) HD meshes,\n");
//  mastererr("     proceed to the next nth output and exit\n");
//  mastererr("     This option must stand alone on one switch (-va -4 is legal, -v4a is not)\n");
  prs_exit (1);
}

double TellNbOrbits (double time) {
  return time/2.0/M_PI*sqrt(G*1.0/1.0/1.0/1.0);
}

double TellNbOutputs (double time) {
  return (time/DT/NINTERM);
}

void TellEverything () {
  double temp;
  
  
  printf ("\nSimulation \"%s\"\n", SIMNAME);
  printf ("------------------------------------------------------------------------------\n");
  printf ("Parametre file        : %s\n", ParameterFile);
  printf ("Planet config file    : %s\n", PLANETCONFIG);
  printf ("Output directory      : %s\n", OUTPUTDIR);

  printf ("\nDisk properties\n");
  printf ("------------------------------------------------------------------------------\n");
  if (SIGMASLOPE != 2)
    temp = SIGMA0 * 2.0 * M_PI * (pow(RMAX, 2-SIGMASLOPE) - pow (RMIN, 2-SIGMASLOPE));
  else
    temp = SIGMA0 * 2.0 * M_PI * (log (RMIN) - log(RMAX));
  printf ("Disk Mass             : %g\n", temp);
  printf ("Aspect ratio          : %g\n", ASPECTRATIO);
  printf ("Sigma slope           : %g\n", SIGMASLOPE);
  if (SIGMACUTOFFRADIN != 0)
    printf ("Inner cutoff radius   : %g\n", SIGMACUTOFFRADIN);
  if (SIGMACUTOFFRADOUT != 0)
    printf ("Outer cutoff width    : %g\n", SIGMACUTOFFRADOUT);
  
  if(Adiabatic) {
    printf ("Disk thermodynamics   : politropic\n");
    printf (" * Adiabatic index    : %0.3g\n", ADIABATICINDEX);    
    printf (" * Cooling            : %s\n", (Cooling ? "Yes":"No"));
    if (Cooling)
      printf (" * Cooling time       : %g\n", COOLINGTIME0);
  }
  else {
    printf ("Disk thermodynamics   : locally isothermal\n");
  }
  if(ViscosityAlpha) {
    printf ("Viscosity alpha       : %0.3g\n", ALPHAVISCOSITY);
    if (AdaptiveViscosity) {
      printf ("Adaptive alpha-viscosity is defined\n");
      printf (" * Alpha dead         : %0.3g\n", ALPHAVISCOSITYDEAD);
      printf (" * Alpha Smoothing len: %0.3g\n", ALPHASMOOTH);
      printf (" * Alpha Sigma thresh : %0.3g\n", ALPHASIGMATHRESH);
    }
  }
  else {
    printf ("Viscosity nu          : %0.3g\n", VISCOSITY);
  }
  if (DeadZone) {
    printf ("Dead zone model       : %s\n", (DeadZone ? "Yes" : "No"));
    printf (" * R inner            : %.3f\n", DEADZONERIN);
    printf (" * DeltaR inner       : %.3f\n", DEADZONEDELTARIN);
    printf (" * R ouer             : %.3f\n", DEADZONEROUT);
    printf (" * DeltaR outer       : %.3f\n", DEADZONEDELTAROUT);    
    printf (" * dead zone alpha    : %.1e\n", DEADZONEALPHA);
  }
  printf ("Self-gravitating disk : %s\n", (SelfGravity ? "Yes":"No"));
  printf ("Exclude Hill mode     : %s\n", (ExcludeHill < 1 ? "None" : (ExcludeHill > 1 ? "Hevieside" : "Gaussian")));
  if (ExcludeHill == 2)
      printf ("Heaviside width       : %0.3g\n", HEAVISIDEB);
  printf ("Centrifugal ballance  : %s\n", (CentrifugalBalance ? "Yes" : "No"));
  printf ("Scaling factor        : %g\n", ScalingFactor);

  printf ("\nGrid properties\n");
  printf ("------------------------------------------------------------------------------\n");
  if (LogGrid)
    printf("Radial grid type      : logarithmic\n");
  else
    printf("Radial grid type      : arithmetic\n");
  printf ("Inner radius          : %g\n", RMIN);
  printf ("Outer radius          : %g\n", RMAX);
  printf ("Number of rings       : %d\n", NRAD);
  printf ("Number of sectors     : %d\n", NSEC);
  printf ("Total cells           : %d\n", NRAD*NSEC);
//  printf ("Gas oversampling      : %d[R]/%d[PHI]\n", GASOVERSAMPRAD, GASOVERSAMPAZIM);
//  printf ("Dust undersampling    : %d\n", DUSTCUBICUNDERSAMP);
  printf ("Inner Boundary cond.  : ");
  if (OpenInner)   printf ("OPEN\n");
  if (DampingInner) printf ("DAMPING\n");
  if (StrongDampingInner) printf ("STRONGDAMPING\n");
  if (NonReflectingInner) printf ("NONREFLECTING\n");
  if (RigidWallInner) printf ("RIGIDWALL\n");
  if (ViscOutflowInner) printf ("VISCOUSOOUTFLOW\n");
  printf ("Outer Boundary cond.  : ");
  if (OpenOuter)   printf ("OPEN\n");
  if (DampingOuter) printf ("DAMPING\n");
  if (StrongDampingOuter) printf ("STRONGDAMPING\n");
  if (NonReflectingOuter) printf ("NONREFLECTING\n");
  if (RigidWallOuter) printf ("RIGIDWALL\n");
  if (ViscOutflowOuter) printf ("VISCOUSOOUTFLOW\n");

  printf ("\nDisk physical conditions\n");
  printf ("------------------------------------------------------------------------------\n");
  printf ("VKep at inner edge    : %.3g\n", sqrt(G*1.0*(1.-0.0)/RMIN));
  printf ("VKep at outer edge    : %.3g\n", sqrt(G*1.0/RMAX));
  temp=SIGMA0*M_PI*(1.0-RMIN*RMIN);
  printf ("Mass inner to r=1.0   : %g \n", temp);
  temp=SIGMA0*M_PI*(RMAX*RMAX-1.0);
  printf ("Mass outer to r=1.0   : %g \n", temp);
  printf ("Travelling time for acoustic density waves\n");
  temp = 2.0/3.0/ASPECTRATIO*(pow(RMAX,1.5)-pow(RMIN,1.5));
  printf (" * from Rmin to Rmax  : %.2g = %.2f orbits ~ %.1f outputs\n", temp, TellNbOrbits(temp), TellNbOutputs(temp));
  temp = 2.0/3.0/ASPECTRATIO*(pow(RMAX,1.5)-pow(1.0,1.5));
  printf (" * from r=1.0 to Rmax : %.2g = %.2f orbits ~ %.1f outputs\n", temp, TellNbOrbits(temp), TellNbOutputs(temp));
  temp = 2.0/3.0/ASPECTRATIO*(pow(1.0,1.5)-pow(RMIN,1.5));
  printf (" * from r=1.0 to Rmin : %.2g = %.2f orbits ~ %.1f outputs\n", temp, TellNbOrbits(temp), TellNbOutputs(temp));
  temp = 2.0*M_PI*sqrt(RMIN*RMIN*RMIN/G/1.0);
  printf ("Orbital time at Rmin  : %.3g ~ %.2f outputs\n", temp, TellNbOutputs(temp));
  temp = 2.0*M_PI*sqrt(RMAX*RMAX*RMAX/G/1.0);
  printf ("Orbital time at Rmax  : %.3g ~ %.2f outputs\n", temp, TellNbOutputs(temp));
  printf ("Sound speed\n");
  printf (" * At unit radius     : %.3g\n", ASPECTRATIO*sqrt(G*1.0));
  printf (" * At outer edge      : %.3g\n", ASPECTRATIO*sqrt(G*1.0/RMAX));
  printf (" * At inner edge      : %.3g\n", ASPECTRATIO*sqrt(G*1.0/RMIN));
#ifdef FARGO_INTEGRATION
  printf ("Integrtaoin with HIPERION\n");
  printf (" * dust drag feedback : %s\n", (DUSTFEEDBACKDRAG? "Yes":"No"));
  printf (" * dust grav feedback : %s\n", (DUSTFEEDBACKGRAV?"Yes":"No"")");  
#endif

  if (DustGrid) {
    printf ("\nDust module is activated\n");
    printf ("------------------------------------------------------------------------------\n");  
    printf ("Dust growth model     : %s\n", (DustGrowth ? "Mono Dispersed": "None"));
    printf ("Dust bulk density     : %0.3e (gm/cm^3)\n", DUSTBULKDENS);
    if (DustGrowth) {
      printf ("Dust vel. frag.       : %0.3e (cm/s)\n", DUSTVFRAG);
    }
    printf ("Dust feedback         : %s\n", (DustFeedback ? "Yes": "No"));
    printf ("Number of dust bins   : %i\n", DustBinNum);
    if (DustBinNum) {
      for (int i=0; i< DustBinNum; i++) {
        if (DustConstStokes) 
          printf (" * #%i dust St         : %0.3e \n", i+1, DustSizeBin[i]);
        else
          printf (" * #%i dust size       : %0.3e (diameter in cm)\n", i+1, DustSizeBin[i]);
        printf (" * #%i Mdust/Mgas      : %0.3e \n", i+1, DustMassBin[i]);
      }
      
    }
  }

  printf ("\nSimulation termianation conditions\n");
  printf ("------------------------------------------------------------------------------\n");
  printf (" * planet distance    : a_pl < %f\n", MinSemiMajorPlanet);
  printf (" * physical time      : t > %gyr (%.3f orbits)\n", NTOT*DT / 2.0 / M_PI, TellNbOrbits(NTOT*DT));

  printf ("\nOutput properties\n");
  printf ("------------------------------------------------------------------------------\n");
  printf ("Number of outputs     : %d\n", NTOT/NINTERM);
  printf ("Snapshot creation at  : %.3fyr (%.3f orbits)\n", NINTERM*DT / 2.0 / M_PI, TellNbOrbits(NINTERM*DT));
  temp = NRAD*NSEC*sizeof(double)/ 1024.0 / 1024.0;
  printf ("At each output #i, the following files are written\n");
  printf (" * gas_dens_[i].dat   : %0.2f Mbytes\n", temp);
  printf (" * gas_vrad_[i].dat   : %0.2f Mbytes\n", temp);
  printf (" * gas_vtheta_[i].dat : %0.2f Mbytes\n", temp);
  int num_snapshots = 3;
  if (Adiabatic == YES) {
    printf (" * gas_energy_[i].dat : %0.2f Mbytes\n", temp);
    num_snapshots++;
  }
  if (AdvecteLabel == YES) {
    printf (" * gaslabel_[i].dat   : %0.2f Mbytes\n", temp);
    num_snapshots++;
  }
  if (DustGrid == YES) {
    for (int i=0; i<DustBinNum; i++) {
      printf (" * dust_dens_s%i_[i].dat: %0.2f Mbytes\n",i, temp);
      printf (" * dust_velrad_s%i_[i].dat  : %0.2f Mbytes\n",i, temp);
      printf (" * dust_veltheta_s%i_[i].dat: %0.2f Mbytes\n",i, temp);
      num_snapshots += 3;
    }
  }
  temp = num_snapshots*NRAD*NSEC*sizeof(double);
  if (AdvecteLabel == YES)
    temp *= 4.0/3.0;
  temp *= (double)NTOT/(double)NINTERM;
  temp /= 1024.0*1024.0*1024.0;
  printf ("Total data generated  : ~%.2f Gbytes\n", temp);

//  printf ("Check (eg by issuing a 'df' command) that you have enough disk space,\n");
//  printf ("otherwise you will get a system full and the code will stop.\n\n");

  fflush (stdout);
}

void GiveTimeInfo (int number) {
  struct tms buffer;
  double total, last, mean, totalu;
  Current = times (&buffer);
  CurrentUser = buffer.tms_utime;
  if (FirstStep == YES) {
    First = Current;
    FirstUser = CurrentUser;
    if (verbose) {
      printf ("\nTime counters \n");
      printf ("------------------------------------------------------------------------------\n");
      printf ("Initialized\n");
    }
    FirstStep = NO;
    Ticks = sysconf (_SC_CLK_TCK);
  }
  else {
    if (verbose) {
      printf ("\nTime counters\n");
      printf ("------------------------------------------------------------------------------\n");
    }
    total = (double)(Current - First)/Ticks;
    totalu= (double)(CurrentUser-FirstUser)/Ticks;
    last  = (double)(CurrentUser - PreceedingUser)/Ticks;
    number -= begin_i/NINTERM;
    mean  = totalu / number;
    fprintf (stderr, "Total Real Time elapsed          : %.3f s\n", total);
    fprintf (stderr, "Total CPU Time of process        : %.3f s (%.1f %%)\n", totalu, 100.*totalu/total);
    fprintf (stderr, "CPU Time since last time step    : %.3f s\n", last);
    fprintf (stderr, "Mean CPU Time between time steps : %.3f s\n", mean);
    fprintf (stderr, "CPU Load on last time step       : %.1f %%\n\n", (double)(CurrentUser-PreceedingUser)/(double)(Current-Preceeding)*100.);

  }	
  PreceedingUser = CurrentUser;
  Preceeding = Current;
}

void InitSpecificTime (bool profiling, TimeProcess *process_name, char *title) {
  struct tms buffer;
  if (profiling == NO) 
    return;
  Ticks = sysconf (_SC_CLK_TCK);
  times (&buffer);
  process_name->clicks = buffer.tms_utime;
  strcpy (process_name->name, title);
}

void GiveSpecificTime (bool profiling, TimeProcess process_name) {
  struct tms buffer;
  long ticks;
  double t;
  if (profiling == NO) return;
  Ticks = sysconf (_SC_CLK_TCK);
  times (&buffer);
  ticks = buffer.tms_utime - process_name.clicks;
  t = (double)ticks / (double)Ticks;
  fprintf (stderr, "Time spent in %s : %.3f s\n", process_name.name, t);
}

