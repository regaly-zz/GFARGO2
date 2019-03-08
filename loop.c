#include "fargo.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
// [RZS-MOD]
// some function declaration used for HIPERION
//--------------------------------------------
#include "gpu_self_gravity.h"

#ifdef FARGO_INTEGRATION
#include "../HIPERION-v1.7/interface_extfuncs.h"
#endif

int           InnerLoopCount = 0;
static int    InnerOutputCounter=0;
//static double timeCRASH;  
//static int    AlreadyCrashed = 0, 
static int    GasTimeStepsCFL;

bool bOutputGenerated;

// RZS [MOD]
//Function to get wall-clock timings
//----------------------------------
#include <sys/time.h>
double get_time() {
  struct timeval Tvalue;
  struct timezone dummy;

  gettimeofday(&Tvalue,&dummy);
  return ((double) Tvalue.tv_sec +
          1.e-6*((double) Tvalue.tv_usec));
}

#define PROFILINGITNUM 10
//----------------------------------
void Loop () {
  static int OuterLoop = YES;
  double dt=0.0;
  static double dtemp=0.0;
  double OmegaNew;
  int gastimestepcfl;
  bool Crashed=NO;
  static int count = 0;
  double time_GFARGO_start, time_GFARGO_end;
  static double time_GFARGO = 0.0;
  static int ProfilingCounter = 0;
#ifdef FARGO_INTEGRATION
  int NBody_stepNum = 1;
  double time_HIPERION_start, time_HIPERION_end;
  static double time_HIPERION = 0.0;
#endif

  if (Paused) 
    return;  
  
  // outer loop
  if (OuterLoop == YES) {
    InnerOutputCounter++;
    
    // Monitoring
    //-----------------------------------------------------------------------------------
    if (MonitorBC || MonitorDiskEcc || MonitorTorque) {
      CalcGasBC (gas_density, sys);
    
      if (DustGrid)
        CalcDustBC (dust_density, sys);
    }
    if (MonitorDiskEcc)
      CalcDiskEcc (gas_density, gas_v_rad, gas_v_theta, disk_ecc, sys);
    //-----------------------------------------------------------------------------------

    if (InnerOutputCounter == 1) {
      InnerOutputCounter = 0;

      // write big planets file at every time step
      //-----------------------------------------------------------------------------------
      WriteBigPlanetSystemFile (sys, PhysicalTime);
      //-----------------------------------------------------------------------------------
            
      // if Monitor BC is set then write barycentre.dat file at every timestep
      //-----------------------------------------------------------------------------------
      WriteBigFiles (sys, gas_density, dust_density, PhysicalTime);
      //-----------------------------------------------------------------------------------
    }
    
    // Outputs are done here
    // save HIPERION snapshots and initiate GFARGO saving
    //-----------------------------------------------------------------------------------
#ifdef FARGO_INTEGRATION

    int iCounter = HIPERION_SaveSnasphot (false);

    // #1 outer loop
    if (iCounter >= 0 || IterCount == 0) {


      // saving snapshots and determine dt
      if (iCounter >= 0) 
        TimeStep = iCounter;
#else

      if (NINTERM * (TimeStep = (IterCount / NINTERM)) == IterCount) {
#endif
    //-----------------------------------------------------------------------------------
        
        // outputs
        //-----------------------------------------------------------------------------------
        SendOutput (TimeStep);
        WriteTorques (sys, gas_density, dust_density, TimeStep);
        WritePlanetSystemFile (sys, TimeStep);        
        
        // if Only initialization or ... then just simple exit
        if ((OnlyInit) || ((GotoNextOutput) && (!StillWriteOneOutput))) {
	        exit (1);
        }
        
        StillWriteOneOutput--;
        WriteCurrentFrameNum (TimeStep);
        //-----------------------------------------------------------------------------------
                
        // time monitoring is done here
        //-----------------------------------------------------------------------------------
        if (TimeInfo == YES)
	        GiveTimeInfo (TimeStep);
        //-----------------------------------------------------------------------------------
        
        bOutputGenerated = true;
      }
      else 
        bOutputGenerated = false;

      // hydrodynamical Part
      //is it necessary here???????
      //FillForcesArrays_gpu (sys);

      InitSpecificTime (Profiling, &t_Hydro, (char*) "Eulerian Hydro algorithms");

      if (Adiabatic)
        CalcSoundSpeed_gpu (gas_density, gas_energy, SoundSpeed);

      // determine CFL time-step
      //-----------------------------------------------------------------------------------      
      gastimestepcfl = 1;
      if (IsDisk == YES) {
        if (SloppyCFL == YES)  {
          gastimestepcfl = ConditionCFL_gpu (gas_v_rad, gas_v_theta, gas_density, gas_energy, (double)(DT-dtemp));
        }
        GasTimeStepsCFL = gastimestepcfl;        
      }      
      dt = DT / (double) GasTimeStepsCFL;
      dtemp = 0.0;
      //-----------------------------------------------------------------------------------
    }
      
  // #2 inner loop
  if (dtemp < 0.999999999*DT) {

    // Mass tapering
    //-----------------------------------------------------------------------------------
    OuterLoop = NO;
    MassTaper = PhysicalTime/MASSTAPER;
    MassTaper = (MassTaper > 1.0 ? 1.0 : pow(sin(MassTaper*M_PI/2.0),2.0));
    //-----------------------------------------------------------------------------------
      
    if (IsDisk == YES) {
      
      // CFL calculation
      //-----------------------------------------------------------------------------------
      if (SloppyCFL == NO) {
          //gastimestepcfl = 1;
          gastimestepcfl = ConditionCFL_gpu (gas_v_rad, gas_v_theta, gas_density, gas_energy, (double)(DT-dtemp));
          dt = (DT-dtemp) / gastimestepcfl;
      }
      //-----------------------------------------------------------------------------------
      
      // planetary accretion on device
      //-----------------------------------------------------------------------------------
      if (!NoGasAccretion) {
        AccreteOntoPlanets_gpu (gas_density, gas_v_rad, gas_v_theta, dt, sys, &(Pl0AccretedMass[0]), (MassTaper < 1.0 ? 0 : 1));
      }
      if (DustGrid)
        for (int ii=0; ii < DustBinNum; ii++)
          AccreteOntoPlanets_gpu (dust_density[ii], dust_v_rad[ii], dust_v_theta[ii], dt, sys, &(Pl0AccretedMass[1+ii]), (MassTaper < 1.0 ? 0 : 1));
      //-----------------------------------------------------------------------------------
    }
    dtemp += dt;
    
    DiskOnPrimaryAcceleration.x = 0.0;
    DiskOnPrimaryAcceleration.y = 0.0;
    if (Corotating == YES) 
      GetPsysInfo (sys, MARK);
    
    // Runge-Kutta solver for planet system
          //-----------------------------------------------------------------------------------
    if (IsDisk == YES) {
      DiskOnPrimaryAcceleration = ComputeAccel (gas_density, 0.0, 0.0, 0.0, 0.0);
      FillForcesArrays_gpu (sys);
      AdvanceSystemFromDisk (gas_density, sys, dt);
      if (DustGrid)
        for (int ii=0; ii < DustBinNum; ii++)
        AdvanceSystemFromDisk (dust_density[ii], sys, dt);

      //AdvanceSystemFromDiskRZS (gas_density, sys, dt);
    }  
    // advance planetary system
    AdvanceSystemRK5 (sys, dt);
    //AdvanceSystemRK5RZS (gas_density, sys, dt);
    //-----------------------------------------------------------------------------------

    // frame rotation
      //-----------------------------------------------------------------------------------
    if (Corotating == YES) {
      OmegaNew = GetPsysInfo(sys, GET) / dt;
      //domega = OmegaNew-OmegaFrame;
      OmegaFrame = OmegaNew;
    }
    RotatePsys (sys, OmegaFrame*dt);
      //-----------------------------------------------------------------------------------

    time_GFARGO_start = get_time ();

    // Hydrodynamic solution
    //===============================================
    if (IsDisk == YES) {

      // add self-gravity potential
      //-----------------------------------------------------------------------------------
      if (SelfGravity) {
#ifdef FARGO_INTEGRATION
        if (DustFeedBackGrav) {
          gpu_sg_add_dens (gas_density->pitch, gas_density->gpu_field, CoarseDust->gpu_field, gasdust_dens->gpu_field);
          gpu_sg_calc_pot (gasdust_dens->pitch, gasdust_dens->gpu_field, Potential->gpu_field);
        }
#endif  
        // if dust is included add dust mass to gas mass  
        if (DustGrid && DustSelfGravity) {
            SumPolarGrid_gpu (DustBinNum, dust_density, gas_density, Work);
            gpu_sg_calc_pot (Work->pitch, Work->gpu_field, Potential->gpu_field);
        }
        // only gas mass is included
        else        
          gpu_sg_calc_pot (gas_density->pitch, gas_density->gpu_field, Potential->gpu_field);
      }
      //-----------------------------------------------------------------------------------
      
      
      // apply boundary conditions
      //-----------------------------------------------------------------------------------
      //FARGO_SAFE (ApplyBoundaryCondition(gas_v_rad, gas_v_theta, gas_density, gas_energy, dt));
      //if (DustGrid)
      //  for (int ii=0; ii < DustBinNum; ii++)
      //    FARGO_SAFE (ApplyBoundaryConditionDust(dust_v_rad[ii], dust_v_theta[ii], dust_density[ii], DustMassBin[ii], dt));
      //-----------------------------------------------------------------------------------
               
      
      // Testing for negative values
      //-----------------------------------------------------------------------------------
      Crashed = DetectCrash (gas_density, DENSITYFLOOR);            // test for negative density values
      //if (Adiabatic)
        //   Crashed |= DetectCrash (gas_energy, DENSITYFLOOR);          // test for negative energy values
      if (DustGrid)
        for (int ii=0; ii < DustBinNum; ii++)
          Crashed |= DetectCrash (dust_density[ii], DENSITYFLOOR);  // test for negative dust density values
      
      if (DustGrowth)
        DetectCrash (dust_size, 5e-4);
      //-----------------------------------------------------------------------------------

      // #1 calculate pressure and potential source terms
      //---------------------------------------------------------------------------------------      
      // substep1 for gas and dust without feedback and growth model
      if (DustGrid && !DustFeedback && !DustGrowth) {
        // (gas_v_rad, gas_v_theta -> VradInt, VthetaInt)
        FARGO_SAFE (SubStep1_gpu (gas_v_rad, gas_v_theta, gas_density, gas_energy, dt,
                                  VradInt, VthetaInt));

        // dust_v_rad[ii], dust_v_theta[ii], dust_density[ii] -> VradDustInt[ii], VthetaDustInt[ii], dust_density[ii]
        // dust diffusion arel also calcualted here
        for (int ii = 0; ii < DustBinNum; ii++)
          FARGO_SAFE (SubStep1Dust_gpu (gas_v_rad, gas_v_theta, gas_density, gas_energy,
                                        dust_v_rad[ii], dust_v_theta[ii], dust_density[ii],
                                        DustSizeBin[ii], dt,
                                        VradDustInt[ii], VthetaDustInt[ii], dust_density[ii]));
      }
      // combined substep1 for gas and dust with backreacteion and without growth model
      // works ony for a single dust component
      else if (DustGrid && DustFeedback && !DustGrowth) {
        FARGO_SAFE (SubStep1GasDust_gpu (gas_v_rad, gas_v_theta, gas_density, gas_energy,
                                         dust_v_rad[0], dust_v_theta[0], dust_density[0],
                                         DustSizeBin[0], dt,
                                         VradInt, VthetaInt,
                                         VradDustInt[0], VthetaDustInt[0], dust_density[0]));
      }
      // combined substep1 for gas and dust with monodispersed growth model
      // works onyl for two components (grown dust and small dust) model
      else if (DustGrid && DustGrowth) {
        FARGO_SAFE (SubStep1GasDustMDGM_gpu (gas_v_rad, gas_v_theta, gas_density, gas_energy,
                                             dust_v_rad, dust_v_theta, dust_density,
                                             dust_size, dust_growth_rate,
                                             dt,
                                             VradInt, VthetaInt,
                                             VradDustInt, VthetaDustInt, dust_density));
      }
      // substep1 only for gas, no dust model
      // (gas_v_rad, gas_v_theta -> VradInt, VthetaInt)
      else  {
        FARGO_SAFE (SubStep1_gpu (gas_v_rad, gas_v_theta, gas_density, gas_energy, dt,
                                  VradInt, VthetaInt));
      }
      //---------------------------------------------------------------------------------------

      // #2 calcualte viscous terms
      // (VradInt, VthetaInt -> VradInt, VthetaInt)
      //-----------------------------------------------------------------------------------
      if ((VISCOSITY > 1e-15) || (ALPHAVISCOSITY > 1e-15))
         FARGO_SAFE (ViscousTerms_gpu (VradInt, VthetaInt, gas_density, gas_energy, dt,
                                       VradInt, VthetaInt));
      //-----------------------------------------------------------------------------------
        
      // #3 artificial viscosity for gas
      //  (VradInt, VthetaInt -> gas_v_rad, gas_v_theta)
      //  (gas_energy -> gas_energy for adiabatic disc)
      //-----------------------------------------------------------------------------------
      FARGO_SAFE (SubStep2_gpu (VradInt, VthetaInt, gas_density, gas_energy, dt,
                                gas_v_rad, gas_v_theta, gas_energy));
      //-----------------------------------------------------------------------------------

      // #4 dust velocity actualized
      //-----------------------------------------------------------------------------------
      if (DustGrid) {
        for (int ii=0; ii < DustBinNum; ii++) {
          if (DustGrowth && ii == 1) {
            ActualiseGas_gpu (dust_v_rad[ii], gas_v_rad);
            ActualiseGas_gpu (dust_v_theta[ii], gas_v_theta);
          }
          else {

            if (DustGrowth) {
              ActualiseGas_gpu (dust_v_rad[ii], VradDustInt[ii]);
              ActualiseGas_gpu (dust_v_theta[ii], VthetaDustInt[ii]);
            }
            else {
              if (DustConstStokes && DustSizeBin[ii]>=10) {
                FARGO_SAFE (ViscousTermsDust_gpu (VradDustInt[ii], VthetaDustInt[ii], dust_density[ii], dt,
                                                  dust_v_rad[ii], dust_v_theta[ii]));
              }
              else {
                ActualiseGas_gpu (dust_v_rad[ii], VradDustInt[ii]);
                ActualiseGas_gpu (dust_v_theta[ii], VthetaDustInt[ii]);
              }
                //FARGO_SAFE (SubStep2Dust_gpu (VradDustInt[ii], VthetaDustInt[ii], dust_density[ii], dt,
                //                              dust_v_rad[ii], dust_v_theta[ii]));
            }
          }
        }
      }
      //-----------------------------------------------------------------------------------
     //FARGO_SAFE (SubStep2Dust_gpu (VradDustInt[0], VthetaDustInt[0], dust_size, dt,
     //                              dust_v_rad[0], dust_v_theta[0]));
     //
      
      // #5 calculate source terms for energ
      // (gas_energy -> gas_energy)
      //-----------------------------------------------------------------------------------
      if (Adiabatic)
        FARGO_SAFE (SubStep3_gpu (gas_v_rad, gas_v_theta, gas_density, gas_energy, dt,
                                  gas_energy));
      //-----------------------------------------------------------------------------------

      // #8 dust growth
      //-----------------------------------------------------------------------------------
      if (DustGrowth) {
        FARGO_SAFE (SubStep4_gpu (dust_density[0], dust_density[1], dust_size, dust_growth_rate, gas_density, gas_energy, dt));
      }
      //-----------------------------------------------------------------------------------

      // #6 gas transport
      //-----------------------------------------------------------------------------------
      FARGO_SAFE (ApplyBoundaryCondition (gas_v_rad, gas_v_theta, gas_density, gas_energy, dt));
      Transport (gas_density, gas_v_rad, gas_v_theta, gas_energy, NULL, dt);
      FARGO_SAFE (ApplyBoundaryCondition (gas_v_rad, gas_v_theta, gas_density, gas_energy, dt));
      //-----------------------------------------------------------------------------------
      
      // #7 dust transport
      //-----------------------------------------------------------------------------------
      if (DustGrid) {
        // dust growth model
        if (DustGrowth) {
          for (int ii = 0; ii < DustBinNum; ii++) {
            FARGO_SAFE (ApplyBoundaryConditionDust (dust_v_rad[ii], dust_v_theta[ii], dust_density[ii], ii, dt));
            if (ii == 0)
              Transport (dust_density[ii], dust_v_rad[ii], dust_v_theta[ii], NULL, dust_size, dt);
            else
              Transport (dust_density[ii], dust_v_rad[ii], dust_v_theta[ii], NULL, NULL, dt, false);
            FARGO_SAFE (ApplyBoundaryConditionDust (dust_v_rad[ii], dust_v_theta[ii], dust_density[ii], ii, dt));
          }
        }
        // non growing dust model
        else {
          for (int ii = 0; ii < DustBinNum; ii++) {
            FARGO_SAFE (ApplyBoundaryConditionDust (dust_v_rad[ii], dust_v_theta[ii], dust_density[ii], ii, dt));
            Transport (dust_density[ii], dust_v_rad[ii], dust_v_theta[ii], NULL, NULL, dt);
            FARGO_SAFE (ApplyBoundaryConditionDust (dust_v_rad[ii], dust_v_theta[ii], dust_density[ii], ii, dt));
          }
        }
      }
      //-----------------------------------------------------------------------------------

    }
      
    // [RZS-MOD]
    // calculate gas density on a fine grid
    //-------------------------------------    
#ifdef FARGO_INTEGRATION
    if (PhysicalTime > HIPRELEASEDATE) {
      GasBiCubicInterpol ();
    }
#ifdef DUST_FEEDBACK
      DustBiCubicInterpol ();
#endif
#endif
    //-------------------------------------
      
    // end of hydrosteps
    if (Profiling)
      cudaDeviceSynchronize ();

    time_GFARGO_end = get_time ();
    time_GFARGO += time_GFARGO_end - time_GFARGO_start;

    // [RZS-MOD]
    // HIPERION iteration of dust particles
    //-------------------------------------
#ifdef FARGO_INTEGRATION
    // HIPERION iteration only after release date
    time_HIPERION_start = get_time ();
  
    if (PhysicalTime > HIPRELEASEDATE)  {
      HIPERION_SendParticle2DPositionToDevice (sys->nb, sys->mass,
                                               sys->x,  sys->y,
                                               sys->vx, sys->vy,
                                               BC_x, BC_y,
                                               MassTaper);
                                                   
      HIPERION_ActualizeFargoInterface (gas_density->gpu_field, gas_v_rad->gpu_field, gas_v_theta->gpu_field, DustDens->gpu_field,
                                        fine_gas_density, fine_gas_v_rad, fine_gas_v_theta, fine_gas_dv_rad, fine_gas_dv_theta);     


      //for (int substep_count = 0; substep_count < NBODY_SUBDT; substep_count++) 
      //NBody_stepNum =  HIPERION_Iterate (false, true, dt);
      NBody_stepNum =  HIPERION_Iterate (false, true, dt);
    }    
    if (Profiling)
        cudaDeviceSynchronize ();
    time_HIPERION_end = get_time ();
    time_HIPERION += time_HIPERION_end - time_HIPERION_start;
#endif
    //-------------------------------------
    
    InnerLoopCount ++;
    PhysicalTime += dt;
    
    // [RZS-MOD]
    // stdout logging stuff
    //-------------------------------------
    if (Profiling) {
      ProfilingCounter ++;
      if (ProfilingCounter==PROFILINGITNUM) {
#ifdef FARGO_INTEGRATION
      printf ("\rt: %10.5e yr  dt: %10.5e yr DT/dt: %4d GFARGO: %5.2f msec HIPERION (x%i): %5.2f msec", PhysicalTime / 2.0 / M_PI, dt/2.0 / M_PI, (int)(DT/dt), (time_GFARGO)*1000.0/10./(double)PROFILINGITNUM, NBody_stepNum, (time_HIPERION)*1000.0); 
      fflush (stdout);
      time_HIPERION = 0.0;
#else
      printf ("\rt: %10.5e yr  dt: %10.5e yr Nsubs: %4d IT: %5.2f msec", 
              PhysicalTime / 2.0 / M_PI, 
              dt/2.0 / M_PI, 
              InnerLoopCount,
              (double) ((time_GFARGO)*1000.0/(double)PROFILINGITNUM)); 
      fflush (stdout);
      }
    }
    else {
      printf (".");
      fflush (stdout);
      count++;
    }
#endif
    if (ProfilingCounter == PROFILINGITNUM) {
      ProfilingCounter = 0;
      time_GFARGO = 0.0;
    }
  }
  else {    
    OuterLoop = YES;
    InnerLoopCount = 0;
  }
    
  // outer loop again
  if (OuterLoop == YES) {
    if (!Profiling) {
      printf ("%i \n", count);  
      count = 0;
    }
//    else
//      GiveSpecificTime (Profiling, t_Hydro);
    SolveOrbits (sys);

    if (MonitorIntegral == YES) {
      D2H (gas_density);
      D2H (gas_v_theta);
      masterprint ("Gas Momentum   : %.18g\n", GasMomentum (gas_density, gas_v_theta));
      masterprint ("Gas total Mass : %.18g\n", GasTotalMass (gas_density));
    }
    
    // increment iteration counter
    IterCount++;  
    
    // [RZS-MOD]
    // simulation end is controlled by HIPERION
    //-----------------------------------------
#ifdef FARGO_INTEGRATION
    if (HIPERION_CheckTermination ()) {
        printf ("\nSimulation successfully finished.\n\n");
        prs_end (sys, 0);
    }
#else
    if (IterCount > NTOT) {
      printf ("\nSimulation successfully finished (time criterium).\n\n");
      GiveTimeInfo (TimeStep);
      prs_end (sys, 0);
    }
    if (TerminateDuetoPlanet) {
      printf ("\nTerminate simulation (planet reached minimal sem-major axis).\n\n");
      GiveTimeInfo (TimeStep);
      prs_end (sys, 0);
    }
#endif
    //-----------------------------------------
  }
}
