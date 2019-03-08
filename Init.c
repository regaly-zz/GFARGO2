/** \file Init.c

Contains the functions needed to initialize the hydrodynamics arrays.
These can be initialized by reading a given output (in the case of a
restart) or by calling a function, InitEuler (), which contains
analytic prescription for the different hydrodynamics fields. Note
that this function InitEuler() is located in SourceEuler.c, which
itself calls _InitGas()_, in the file Pframeforce.c.
Also, note that the present file contains InitLabel(), which sets
the initial value of a passive scalar.
*/

#include "fargo.h"

void ReadfromFile (PolarGrid *array, const char *fileprefix, int filenumber) {
  int nr,ns,c;//, foo=0;
  double *field;
  char name[256];
  FILE *input;
  
  /* Simultaneous read access to the same file have been observed to give wrong results. */
  /* A sequential reading is imposed below. */
				/* If current CPU has a predecessor, wait for a message from him */
  //if (CPU_Rank > 0) 
    //MPI_Recv (&foo, 1, MPI_INT, CPU_Rank-1, 10, MPI_COMM_WORLD, &fargostat);
  
  // [RZS-MOD]
  // snapshot name changed to gas*_XXXXXX.dat
  //sprintf (name, "%s%s%d.dat", OUTPUTDIR, fileprefix, filenumber);
  sprintf (name, "%s%s_%06d.dat", OUTPUTDIR, fileprefix, filenumber);

  input = fopen (name, (char*) "r");
  if (input == NULL) {
    fprintf (stderr, "WARNING ! Can't read %s. Exiting.\n", name); 
    //if (CPU_Rank < CPU_Number-1) 
      //MPI_Send (&foo, 1, MPI_INT, CPU_Rank+1, 10, MPI_COMM_WORLD);
    //return;
    exit (-1);
  }
  field = array->Field;
  nr = array->Nrad;
  ns = array->Nsec;
  for (c = 0; c < IMIN; c++) {
    fread (field, sizeof(double), ns, input); /* Can't read at once in order not to overflow 'field' */
  }
  fread (field, sizeof(double), nr*ns, input);
  fclose (input);
  
  printf ("\t%s_%06d.dat\n", fileprefix, filenumber);
  /* Next CPU is waiting. Tell it to start now by sending the message that it expects */
  //if (CPU_Rank < CPU_Number-1) 
    //MPI_Send (&foo, 1, MPI_INT, CPU_Rank+1, 10, MPI_COMM_WORLD);
  //MPI_Barrier (MPI_COMM_WORLD);	/* previous CPUs do not touch anything meanwhile */
}

void InitLabel (PolarGrid *array) {
  int nr,ns,i,j,l;
  double *field;
  field = array->Field;
  nr = array->Nrad;
  ns = array->Nsec;
  for (i = 0; i <= nr; i++) {
    for (j = 0; j < ns; j++) {
      l = j+i*ns;
      field[l] = (Rmed[i]-RMIN)/(RMAX-RMIN);
    }
  }
}

void Initialization (PolarGrid *gas_density, PolarGrid *gas_v_rad, PolarGrid *gas_v_theta, PolarGrid *gas_energy, 
                     PolarGrid *gas_label,
                     PolarGrid **dust_density, PolarGrid **dust_v_rad, PolarGrid **dust_v_theta) {

   if (verbose) {
     printf ("\nInitializatoin\n");
     printf ("------------------------------------------------------------------------------\n");
   }

  ReadPrevDim ();
    
  InitEuler (gas_density, gas_v_rad, gas_v_theta, gas_energy,
             dust_density, dust_v_rad, dust_v_theta);
             
  InitLabel (gas_label);

  if (Restart == YES) {
    CheckRebin (NbRestart);
    //MPI_Barrier (MPI_COMM_WORLD); /* Don't start reading before master has finished rebining... */
				  /* It shouldn't be a problem though since a sequential read is */
                                  /* imposed in the ReadfromFile function below */
    mastererr ("Reading snapshots:\n");
    fflush (stderr);
    ReadfromFile (gas_density, "gas_dens", NbRestart);
    ReadfromFile (gas_v_rad,   "gas_vrad", NbRestart);
    ReadfromFile (gas_v_theta, "gas_vtheta", NbRestart);

    if (Adiabatic)
      ReadfromFile (gas_energy, "gas_energy", NbRestart);

    if (DustGrid) {
      char name[1024];
      for (int i = 0; i < DustBinNum; i++) {
        sprintf (name, "dust_dens_s%d", i);
        ReadfromFile (dust_density[i], name, NbRestart);
        sprintf (name, "dust_vrad_s%d", i);
        ReadfromFile (dust_v_rad[i],   name, NbRestart);
        sprintf (name, "dust_vtheta_s%d", i);
        ReadfromFile (dust_v_theta[i], name, NbRestart);
      }
      if (DustGrowth)
        ReadfromFile (dust_size, "dust_size", NbRestart);
    }

    if (AdvecteLabel == YES)
      ReadfromFile (gas_label,   "gaslabel", NbRestart);
    if (StoreSigma) 
      RefillSigma (gas_density);
  }
  WriteDim (); 
}

/*
void InitializationBC (PlanetarySystem *sys, PolarGrid *gas_density, PolarGrid *gas_v_rad, PolarGrid *gas_v_theta, PolarGrid *gas_label) {
  ReadPrevDim ();
  InitEulerBC (sys, gas_density, gas_v_rad, gas_v_theta);
  InitLabel (gas_label);
  printf ("BC Initialization started\n");
  if (Restart == YES) {
    CheckRebin (NbRestart);
    //MPI_Barrier (MPI_COMM_WORLD); // Don't start reading before master has finished rebining... 
				  // It shouldn't be a problem though since a sequential read is 
                                  // imposed in the ReadfromFile function below
    mastererr ("Reading restart files...");
    fflush (stderr);
    ReadfromFile (gas_density, "gasdens", NbRestart);
    ReadfromFile (gas_v_rad,   "gasvrad", NbRestart);
    ReadfromFile (gas_v_theta, "gasvtheta", NbRestart);
    if (AdvecteLabel == YES)
      ReadfromFile (gas_label,   "gaslabel", NbRestart);
    if (StoreSigma) 
      RefillSigma (gas_density);
    fprintf (stderr, "done\n");
    fflush (stderr);
  }
  WriteDim (); 
  printf ("Finished!\n");
}*/
