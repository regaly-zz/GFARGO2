/** \file SourceEuler.c 

Contains routines used by the hydrodynamical loop. More specifically,
it contains the main loop itself and all the source term substeps
(with the exception of the evaluation of the viscous force). The
transport substep is treated elsewhere. */

#include "fargo.h"
#include "gpu_self_gravity.h"
#include <time.h>

void SetRadiiStuff () {
  InvDiffRmed  = RadiiStuff + (NRAD+1)*0;
  CS2          = RadiiStuff + (NRAD+1)*1;
  InvRmed      = RadiiStuff + (NRAD+1)*2;
  InvRinf      = RadiiStuff + (NRAD+1)*3;
  Rinf         = RadiiStuff + (NRAD+1)*4;
  Radii        = RadiiStuff + (NRAD+1)*5;
  Rmed         = RadiiStuff + (NRAD+1)*6;
  InvSurf      = RadiiStuff + (NRAD+1)*7;
  Rsup         = RadiiStuff + (NRAD+1)*8;
  Surf         = RadiiStuff + (NRAD+1)*9;
  InvDiffRsup  = RadiiStuff + (NRAD+1)*10;
  SOUNDSPEED   = RadiiStuff + (NRAD+1)*11;
  viscoval     = RadiiStuff + (NRAD+1)*12;
  alphaval     = RadiiStuff + (NRAD+1)*13;
  omega        = RadiiStuff + (NRAD+1)*14;
  
  DRadii        = (double*)malloc((NRAD+1)*sizeof(double));
  DRmed         = (double*)malloc((NRAD+1)*sizeof(double));
  DInvSqrtRmed  = (double*)malloc((NRAD+1)*sizeof(double));
}
 
void FillPolar1DArrays () {
  FILE *input, *output;
  int i,ii;
  double drrsep;
  double temporary;
  char InputName[256], OutputName[256];
  SetRadiiStuff ();
  drrsep = (RMAX-RMIN)/(double)GLOBALNRAD;
  sprintf (InputName, "%s%s", OUTPUTDIR, "radii.dat");
  sprintf (OutputName, "%s%s", OUTPUTDIR, "used_rad.dat");
  input = fopen (InputName, "r");
  if (input == NULL) {
    if (verbose)
      mastererr ("Warning : no `radii.dat' file found. Using default.\n");
    if (LogGrid == YES) {
      for (i = 0; i <= GLOBALNRAD; i++) {
	      DRadii[i] = RMIN*exp((double)i/(double)GLOBALNRAD*log(RMAX/RMIN));
        Radii[i] = (double)DRadii[i];
      }
    } 
    else {
      for (i = 0; i <= GLOBALNRAD; i++) {
	      DRadii[i] = RMIN+drrsep*(double)(i);
	      Radii[i] = (double)DRadii[i];
      }
    }
  } 
  else {
    mastererr ("Reading 'radii.dat' file.\n");
    for (i = 0; i <= GLOBALNRAD; i++) {
      fscanf (input, "%lf", &temporary);
      Radii[i] = (double)temporary;
      DRadii[i] = temporary;
    }
  }
  for (i = 0; i < GLOBALNRAD; i++) {
    GlobalRmed[i] = 2.0/3.0*(Radii[i+1]*Radii[i+1]*Radii[i+1]-Radii[i]*Radii[i]*Radii[i]);
    GlobalRmed[i] = GlobalRmed[i] / (Radii[i+1]*Radii[i+1]-Radii[i]*Radii[i]);
    //GlobalRmed[i] = 0.5*(Radii[i+1]+Radii[i]);
  }
  for (i = 0; i < NRAD; i++) {
    ii = i+IMIN;
    Rinf[i]         = Radii[ii];
    Rsup[i]         = Radii[ii+1];
  

  //  Rmed[i]         = 0.5*(Radii[ii+1]+Radii[ii]);
  //    DRmed[i]         = 0.5*(DRadii[ii+1]+DRadii[ii]);
    
    
    Rmed[i]         = 2.0/3.0*(Rsup[i]*Rsup[i]*Rsup[i]-Rinf[i]*Rinf[i]*Rinf[i]);
    Rmed[i]         = Rmed[i] / (Rsup[i]*Rsup[i]-Rinf[i]*Rinf[i]);
    Surf[i]         = M_PI*(Rsup[i]*Rsup[i]-Rinf[i]*Rinf[i])/(double)NSEC;
    InvRmed[i]      = 1.0/Rmed[i];
    InvSurf[i]      = 1.0/Surf[i];
    InvDiffRsup[i]  = 1.0/(Rsup[i]-Rinf[i]);
    InvRinf[i]      = 1.0/Rinf[i];
    viscoval[i]     = FViscosity (Rmed[i]);
    alphaval[i]     = AlphaValue (Rmed[i]);
    omega[i]        = pow(Rmed[i], -3.0/2.0);

    DRmed[i]        = 2.0/3.0*(DRadii[ii+1]*DRadii[ii+1]*DRadii[ii+1]-DRadii[ii]*DRadii[ii]*DRadii[ii]);
    DRmed[i]        = DRmed[i] / (DRadii[ii+1]*DRadii[ii+1]-DRadii[i]*DRadii[i]);  
    DInvSqrtRmed[i] = 1.0/sqrt(Rmed[i]);

  }

  Rinf[NRAD]=Radii[NRAD+IMIN];
  for (i = 1; i < NRAD; i++) {
    InvDiffRmed[i] = 1.0/(Rmed[i]-Rmed[i-1]);
  }
  //if (CPU_Master) {
    output = fopen (OutputName, "w");
    if (output == NULL) {
      mastererr ("Can't write %s.\nProgram stopped.\n", OutputName);
      prs_exit (1);
    }
    for (i = 0; i <= GLOBALNRAD; i++) {
      fprintf (output, "%.18g\n", Radii[i]);
    }
    fclose (output);
    //}
  if (input != NULL) fclose (input);
}

void InitPolarGrids () {
  if (verbose) {
    printf ("\nAllocating PolarGrids required for gas\n");
    printf ("------------------------------------------------------------------------------\n");
  }
  
  DeltaT = CreatePolarGrid (NRAD, NSEC, "DeltaT");
  Buffer = CreatePolarGrid (NRAD, NSEC, "WorkArray");
  WorkShift = CreatePolarGrid (NRAD, NSEC, "WorkShift");
  
  RhoStar      = CreatePolarGrid(NRAD, NSEC, "RhoStar");
  RhoInt       = CreatePolarGrid(NRAD, NSEC, "RhoInt");
  VradNew      = CreatePolarGrid(NRAD, NSEC, "VradNew");
  VradInt      = CreatePolarGrid(NRAD, NSEC, "VradInt");
  VthetaNew    = CreatePolarGrid(NRAD, NSEC, "VthetaNew");
  VthetaInt    = CreatePolarGrid(NRAD, NSEC, "VthetaInt");
  TemperInt    = CreatePolarGrid(NRAD, NSEC, "TemperInt");
  Potential    = CreatePolarGrid(NRAD, NSEC, "Potential");
  
  tmp1    = CreatePolarGrid(NRAD, NSEC, "tmp1");
  tmp2    = CreatePolarGrid(NRAD, NSEC, "tmp2");  
  if (SelfGravity)
    SGAcc = CreatePolarGrid(NRAD, NSEC, "SGAcc");
  if (Adiabatic) {
    SoundSpeed    = CreatePolarGrid(NRAD, NSEC, "SoundSpeed");
    EnergyInt     = CreatePolarGrid(NRAD, NSEC, "EnergyInt");
  }
  if (Adiabatic || AdaptiveViscosity)
      Viscosity     = CreatePolarGrid(NRAD, NSEC, "Viscosity");
  if (ViscHeating) {
    TauRR         = CreatePolarGrid(NRAD, NSEC, "TauRR");
    TauRP         = CreatePolarGrid(NRAD, NSEC, "TauRP");
    TauPP         = CreatePolarGrid(NRAD, NSEC, "TauPP");
  }
  
  myWork = CreatePolarGrid(NRAD, NSEC, "myWork");
}


void InitDustPolarGrids () {
  if (verbose) {
    printf ("\nAllocating PolarGrids required for dust\n");
    printf ("------------------------------------------------------------------------------\n");
  }

  // first create dust intermediate velocities
  VradDustInt   = new PolarGrid* [DustBinNum];
  VthetaDustInt = new PolarGrid* [DustBinNum];
  char name[1024];
  for (int i=0; i< DustBinNum; i++) {
    sprintf (name, "VradDustInt_s%d", i);
    VradDustInt[i]  = CreatePolarGrid(NRAD, NSEC, name);
    sprintf (name, "VthetaDustInt_s%d", i);
    VthetaDustInt[i]    = CreatePolarGrid(NRAD, NSEC, name);
  }

  if (DustGrowth) {
    dust_size = CreatePolarGrid(NRAD, NSEC, "dust_size");
    dust_growth_rate = CreatePolarGrid(NRAD, NSEC, "dust_growth_rate");
  }
  
  tmp3    = CreatePolarGrid(NRAD, NSEC, "tmp3");
  tmp4    = CreatePolarGrid(NRAD, NSEC, "tmp4");
}

void InitDust (double dust_mass, double dust_size_bin, PolarGrid *GasRho, PolarGrid *DustRho, PolarGrid *DustVrad, PolarGrid *DustVtheta) {
  const int nr  = GasRho->Nrad;
  const int ns  = GasRho->Nsec;
  
  static bool First = true;
  for (int i = 0; i < nr; i++) {
    for (int j = 0; j < ns; j++) {
      const int l = j+i*ns;  
      DustRho->Field[l]    = dust_mass*GasRho->Field[l]; // constant dust-to-gass mass ratio
      double vrad, vtheta;
      DustDragVel (dust_size_bin, Rmed[i], SigmaMed[i], &vrad, &vtheta);
      DustVrad->Field[l]   = vrad;                        // circular orbit around the star
      DustVtheta->Field[l] = vtheta;                      // azimuthal velocity is Keplerian (vth=vk - vk =0)
    
      if (DustGrowth && First) {
        dust_size->Field[l] = dust_size_bin;// * DustRho->Field[l];
        //dust_size->Field[l] = (1.+j/512.)/1e3;//dust_size_bin/2.0;
        
//        if (i<200)
//          dust_size->Field[l] = 0.1;
//        else
//          dust_size->Field[l] = 1;
        dust_growth_rate->Field[l] = 0.0;
      }
    }
  }
  First = false;
}

void InitEuler (PolarGrid *Rho, PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Energy,
                PolarGrid **DustRho, PolarGrid **DustVrad, PolarGrid **DustVtheta) {
  
  // init 1D arrays
  FillPolar1DArrays ();

  // init neccessary PolarGrids
  InitPolarGrids ();
  InitTransport ();
  
  // init sigma
  if (ModelNebula)
    FillNebulaSigma ();
  else
    FillSigma ();

  // int self-gravity module
  if (SelfGravity) {
    PolarGrid *FakeRho;
    FakeRho = CreatePolarGrid(NRAD, NSEC, "FakeRho");
    int i, j;
    for (i = 0; i< NRAD;i++)
      for (j = 0; j< NSEC;j++)
        FakeRho->Field[j+i*NSEC] = SigmaMed[i];
    H2D (FakeRho);
    gpu_sg_calc_pot (FakeRho->pitch, FakeRho->gpu_field, Potential->gpu_field);
    gpu_sg_calc_acc (FakeRho->pitch, FakeRho->gpu_field, Potential->gpu_field, SGAcc->gpu_field);
    D2H (SGAcc);

    SGAccInnerEdge = SGAcc->Field[0]; 
    SGAccOuterEdge = SGAcc->Field[(NRAD-1)*NSEC];
    
    // update planetary velocity if simulation is not restarted (-g_sg/r)
    if (!Restart) {
      for (int i=0; i < sys->nb; i++) {
        sys->vy[i] = (double) sqrt((1.0+sys->mass[i])/sys->x[i] -  SGAcc->Field[i*NSEC]/sys->x[i]) *	sqrt(1.0-ECCENTRICITY*ECCENTRICITY)/(1.0+ECCENTRICITY);
      }
    }
  }

  // init sounspeed and velocities
  FillSoundSpeed ();
  FillViscosity ();
  
  if (ModelNebula)
    FillNebulaVelocities();
  else
    FillVelocites();
  
  if (Adiabatic) {
    if (ModelNebula) {
      FillNebulaEnergy();
      FillNebulaCoolingTime();
    }
    else {
      FillEnergy ();
      FillCoolingTime();
    }
    FillQplus();
    //InitGasEnergy (Energy);
  }
  
  // init dust
  if (DustGrid) {
    Pl0AccretedMass = (double*) malloc(DustBinNum*sizeof(double));
    InitDustPolarGrids ();
    for (int i=0; i< DustBinNum; i++) {
      Pl0AccretedMass[i] = 0.0;
      InitDust (DustMassBin[i], DustSizeBin[i], Rho, DustRho[i], DustVrad[i], DustVtheta[i]);
    }
  }
}

/*
void InitEulerBC (PlanetarySystem *sys, PolarGrid *Rho, PolarGrid *Vr, PolarGrid *Vt) {
  FillPolar1DArrays ();
  FillSigma ();
  InitTransport ();
  RhoStar      = CreatePolarGrid(NRAD, NSEC, "RhoStar");
  RhoInt       = CreatePolarGrid(NRAD, NSEC, "RhoInt");
  VradNew      = CreatePolarGrid(NRAD, NSEC, "VradNew");
  VradInt      = CreatePolarGrid(NRAD, NSEC, "VradInt");
  VthetaNew    = CreatePolarGrid(NRAD, NSEC, "VthetaNew");
  VthetaInt    = CreatePolarGrid(NRAD, NSEC, "VthetaInt");
  TemperInt    = CreatePolarGrid(NRAD, NSEC, "TemperInt");
  Potential    = CreatePolarGrid(NRAD, NSEC, "Potential");
  
  // [RZS-MOD]
  // for self-gravity
  //-----------------
  PolarGrid *SGAcc = NULL;
  if (SelfGravity) {
    SGAcc      = CreatePolarGrid(NRAD, NSEC, "SGAcc");
    PolarGrid *FakeRho;
    FakeRho = CreatePolarGrid(NRAD, NSEC, "FakeRho");
    int i, j;
    for (i = 0; i< NRAD;i++)
      for (j = 0; j< NSEC;j++)
        FakeRho->Field[j+i*NSEC] = SigmaMed[i];
    H2D (FakeRho);
    gpu_sg_calc_pot (FakeRho->pitch, FakeRho->gpu_field, Potential->gpu_field);
    gpu_sg_calc_acc (FakeRho->pitch, FakeRho->gpu_field, Potential->gpu_field, SGAcc->gpu_field);
    D2H (SGAcc);

    SGAccInnerEdge = SGAcc->Field[0]; 
    SGAccOuterEdge = SGAcc->Field[(NRAD-1)*NSEC];
    
    // update planetary velocity if simulation is not restarted (-g_sg/r)
    if (!Restart) {
      for (int i=0; i < sys->nb; i++) {
        sys->vy[i] = (double) sqrt((1.0+sys->mass[i])/sys->x[i] -  SGAcc->Field[i*NSEC]/sys->x[i]) *	sqrt(1.0-ECCENTRICITY*ECCENTRICITY)/(1.0+ECCENTRICITY);
      }
    }
  }
  //-----------------
  InitGasBC (sys, Rho, Vr, Vt, SGAcc);
}
*/

double min2 (double a,double b) {
  if (b < a) return b;
  return a;
}

double max2 (double a,double b) {
  if (b > a) return b;
  return a;
}
