/** \file Output.c

Contains most of the functions that write the output files.
In addition to the writing of hydrodynamics files (handled by
SendOutput ()), this file also contains the functions that update
the planet.dat and bigplanet.dat files, and the functions that
seek information about the planets at a restart.
*/

#include "fargo.h"

static double   Xplanet, Yplanet, VXplanet, VYplanet, MplanetVirtual;

void EmptyPlanetSystemFile (PlanetarySystem *sys) {
  FILE *output;
  char name[256];
  int i, n;
  n = sys->nb;
//  if (!CPU_Master) 
//    return;
  for (i = 0; i < n; i++) {
    sprintf (name, "%splanet%d.dat", OUTPUTDIR, i);
    output = fopenp (name, (char*)  "w"); /* This empties the file */
    fclose (output);
  }
}

void WritePlanetFile (int TimeStep, int n) {
  FILE *output;
  char name[256];
//  if (!CPU_Master) 
//    return;
  printf (" * Updating 'planet%d.dat'...", n);
  fflush (stdout);
  sprintf (name, "%splanet%d.dat", OUTPUTDIR, n);
  output = fopenp (name, (char*) "a");
//  fprintf (output, "%d\t%#.18g\t%#.18g\t%#.18g\t%#.18g\t%#.18g\t%#.18g\t%#.18g\t%#.18g\n",
  fprintf (output, "%d\t%.18e\t%.18e\t%.18e\t%.18e\t%.18e\t%.18e\t%.18e\t%.18e\n",
	                  TimeStep, Xplanet, Yplanet, VXplanet, VYplanet, MplanetVirtual, LostMass, PhysicalTime, OmegaFrame);
  fclose (output);
  printf ("done\n\n");
  fflush (stdout);
}

void WritePlanetSystemFile (PlanetarySystem *sys, int t) {
  int i, n;
  n = sys->nb;
  for (i = 0; i < n; i++) {
    Xplanet = sys->x[i];
    Yplanet = sys->y[i];
    VXplanet = sys->vx[i];
    VYplanet = sys->vy[i];
    MplanetVirtual = sys->mass[i]*MassTaper;
    AccRate = sys->acc_rate[i];
    WritePlanetFile (t, i);
  }
}
   

void WriteBigPlanetFile (double Time, int n) {
  FILE *output;
  char name[256];
//  if (!CPU_Master) 
//    return;
  sprintf (name, "%sbigplanet%d.dat", OUTPUTDIR, n);
  output = fopenp (name, (char*) "a");
//  fprintf (output, "%#.18g\t%#.18g\t%#.18g\t%#.18g\t%#.18g\t%#.18g\t%#.18g\t%#.18g\t%#.18g\n", Time, Xplanet, Yplanet, VXplanet, VYplanet, MplanetVirtual, LostMass, PhysicalTime, OmegaFrame);
   fprintf (output, "%#.18e\t%#.18e\t%#.18e\t%#.18e\t%#.18e\t%#.18e\t%#.18e\t%#.18e\t%#.18e\t%#.18e\n",\
 	   Time, Xplanet, Yplanet, VXplanet, VYplanet, MplanetVirtual, AccRate, StellarAccRate, PhysicalTime, OmegaFrame);     
  fclose (output);
}

void WriteBigPlanetSystemFile (PlanetarySystem *sys, double t) {
  int i, n;
  n = sys->nb;
  for (i = 0; i < n; i++) {
    Xplanet = sys->x[i];
    Yplanet = sys->y[i];
    VXplanet = sys->vx[i];
    VYplanet = sys->vy[i];
    MplanetVirtual = sys->mass[i]*MassTaper;
    AccRate = sys->acc_rate[i];
    WriteBigPlanetFile (t, i);
  }
}

double GetfromPlanetFile (int TimeStep, int column, int n) {
  FILE *input;
  char name[256];
  char testline[256];
  int time_step;
  char *pt;
  double value;
  sprintf (name, "%splanet%d.dat", OUTPUTDIR, n);
  input = fopen (name, (char*) "r");
  if (input == NULL) {
    mastererr ("Can't read 'planet%d.dat' file. Aborting restart.\n",n);
    prs_exit (1);
  }
  if (column < 2) {
    mastererr ("Invalid column number in 'planet%d.dat'. Aborting restart.\n",n);
    prs_exit (1);
  }
  do {
    pt = fgets (testline, 255, input);
    sscanf (testline, "%d", &time_step );
  } while ((time_step != TimeStep) && (pt != NULL));
  if (pt == NULL) {
    mastererr ("Can't read entry %d in 'planet%d.dat' file. Aborting restart.\n", TimeStep,n);
    prs_exit (1);
  }
  fclose (input);
  pt = testline;    
  while (column > 1) {
    pt += strspn(pt, "eE0123456789-+.");
    pt += strspn(pt, "\t :=>_");
    column--;
  }
  sscanf (pt, "%lf", &value);
  return value;
}

void RestartPlanetarySystem (int timestep, PlanetarySystem *sys) {
  int k;
  for (k = 0; k < sys->nb; k++) {
    sys->x[k] = GetfromPlanetFile (timestep, 2, k);
    sys->y[k] = GetfromPlanetFile (timestep, 3, k);
    sys->vx[k] = GetfromPlanetFile (timestep, 4, k);
    sys->vy[k] = GetfromPlanetFile (timestep, 5, k);
    sys->mass[k] = GetfromPlanetFile (timestep, 6, k);
  }
}

void WriteDiskPolar(PolarGrid *array, int number, char *ext_name) {
  int   Nr, Ns;
  FILE *dump;
  char file_name[80];
  double *ptr;
  ptr = array->Field;
  if (!ext_name) {
    sprintf (file_name, "%s%s_%06d.dat", OUTPUTDIR, array->Name, number);
    printf (" * Writing '%s_%06d.dat'\n", array->Name, number);
  }
  else {
    sprintf (file_name, "%s%s_%06d.dat", OUTPUTDIR, ext_name, number);
    printf (" * Writing '%s_%06d.dat'\n", ext_name, number);
  }
  Nr = array->Nrad;
  Ns = array->Nsec;
  dump = fopenp (file_name, (char*) "w");
  fflush (stdout);
  fwrite (ptr, sizeof(double), Nr*Ns, dump);
  fclose (dump);
}

void WriteDim () {
  char filename[200];
  FILE 	*dim;
//  if (!CPU_Master) return;
  sprintf (filename, "%sdims.dat", OUTPUTDIR);
  dim = fopenp (filename, (char*) "w");
  fprintf (dim,"%d\t%d\t\t%d\t%f\t%f\t%d\t%d\t%d\n",
	         0, 0, 0, RMIN, RMAX, NTOT/NINTERM, GLOBALNRAD, NSEC);
  fclose (dim);
}


void SendOutput (int index) {

  if (verbose) {
    printf ("\n\nOUTPUT %d \n", index);
    printf ("------------------------------------------------------------------------------\n");
  }

  // download necessary PolarGrids from GPU
  D2H (gas_density);
  D2H (gas_v_rad);
  D2H (gas_v_theta);
  if (MonitorDiskEcc)
    D2H (disk_ecc);
  if (SelfGravity)
    D2H (Potential);
  if (Adiabatic)
    D2H (gas_energy);
  if (DustGrid) {
    for (int ii=0; ii < DustBinNum; ii++) {
      D2H (dust_density[ii]);
      D2H (dust_v_rad[ii]);
      D2H (dust_v_theta[ii]);
    }
    if (DustGrowth)
      D2H (dust_size);
  }
  // writing PolarGrid as snapshots
  if (IsDisk == YES) {
    if (Write_Density == YES) 
      WriteDiskPolar (gas_density, index);
  
    if (Write_Velocity == YES) {
      WriteDiskPolar (gas_v_rad, index);
      WriteDiskPolar (gas_v_theta, index);
    }

    if (AdvecteLabel == YES) 
      WriteDiskPolar (gas_label, index);
    //MPI_Barrier (MPI_COMM_WORLD);
    //if (Merge && (CPU_Number > 1)) merge (local_index);

    if (Write_Potential && SelfGravity)
      WriteDiskPolar (Potential, index);

    if (Write_Energy && Adiabatic)
      WriteDiskPolar (gas_energy, index);

    if (Write_Temperature && Adiabatic) {
      CalcTemp_gpu (gas_density, gas_energy, Work);
      D2H (Work);
      WriteDiskPolar (Work, index, "gas_temp");      
    }

    if (Write_DiskHeight && Adiabatic) {
      CalcDiskHeight_gpu (gas_density, gas_energy, Work);
      D2H (Work);
      WriteDiskPolar (Work, index, "gas_height");
    }
    
    if (Write_SoundSpeed && Adiabatic) {
      CalcSoundSpeed_gpu (gas_density, gas_energy, Work);
      D2H (Work);
      WriteDiskPolar (Work, index, "gas_cs");
    }

    if (MonitorDiskEcc)
      WriteDiskPolar (disk_ecc, index);

    if (DustGrid) {
      for (int ii=0; ii < DustBinNum; ii++) {
        WriteDiskPolar (dust_density[ii], index);
        WriteDiskPolar (dust_v_rad[ii],   index);
        WriteDiskPolar (dust_v_theta[ii], index);        
      }
      if (DustGrowth)
        WriteDiskPolar (dust_size,   index);
    }
      
#ifdef FARGO_INTEGRATION    
      WriteDiskPolar (DustDens, index);
#endif
  }
}

#define CURRENTFRAME_FILE "lastframe.dat"
void WriteCurrentFrameNum (int index) {
  FILE *output;
  char fname[256];
  
  sprintf (fname, "%s%s", OUTPUTDIR, CURRENTFRAME_FILE);
  output = fopen (fname, (char*) "w");
  fprintf (output, "%i\n", index);
  fclose (output);
}

int ReadCurrentFrameNum () {
  FILE *input;
  char fname[256];
  char testline[256];
  int index;
 
  sprintf (fname, "%s%s", OUTPUTDIR, CURRENTFRAME_FILE);
  input = fopen (fname, (char*) "r");
  if (input == NULL) {
    index = -1;
    printf ("Erro: file does not exist: %s\n", fname);
    //prs_exit (1);
  }
  else {
    fgets (testline, 255, input);
    sscanf (testline, "%d", &index);
    fclose (input);
  }
  return index;
}

void WriteBigFiles (PlanetarySystem *sys, PolarGrid *Rho, PolarGrid **DustRho, double phys_time) {

  // barycenters
  //double bcx, bcy, disk_mass;
  //CalcDiskBC (Rho, &disk_mass, &bcx, &bcy);
  //double spl_mass = 1.0;
  //double spl_bcx  = 0.0;
  //double spl_bcy  = 0.0;
  //for (int k = 0; k < sys->nb; k++) {
  //  spl_mass += sys->mass[k];
  //  spl_bcx  += sys->mass[k] * sys->x[k];
  //  spl_bcy  += sys->mass[k] * sys->y[k];
  //}
  //double spld_bcx = (spl_bcx + bcx * disk_mass)/(spl_mass+disk_mass);
  //double spld_bcy = (spl_bcy + bcy * disk_mass)/(spl_mass+disk_mass);
  //spl_bcx /= spl_mass;
  //spl_bcy /= spl_mass;
  
  FILE *out;
  char filename[MAX1D];  

  if (MonitorBC) {
    sprintf (filename, "%smon_gasbc.dat", OUTPUTDIR);
    out = fopenp (filename, (char*) "a");
    fprintf (out, "%.18e\t%.18e\t%.18e\t%.18e\t%.18e\t%.18e\t%.18e\t%.18e\n", phys_time, GasDiskMassInner+GasDiskMassOuter, GasDiskBC_x, GasDiskBC_y, StarPlanetBC_x, StarPlanetBC_y, GasBC_x, GasBC_y);
    fclose (out);
    
    if (DustGrid) {
      sprintf (filename, "%smon_dustbc.dat", OUTPUTDIR);
      out = fopenp (filename, (char*) "a");
      fprintf (out, "%.18e\t%.18e\t%.18e\t%.18e\t%.18e\t%.18e\t%.18e\t%.18e\n", phys_time, DustDiskMassInner+DustDiskMassOuter, DustDiskBC_x, DustDiskBC_y, StarPlanetBC_x, StarPlanetBC_y, DustBC_x, DustBC_y);
      fclose (out);
    }
  }

  // disk eccentricity
  if (MonitorDiskEcc) {
    sprintf (filename, "%smon_ecc.dat", OUTPUTDIR);
    out = fopenp (filename, (char*) "a");
    fprintf (out, "%.18e\t%.18e\t%.18e\t%.18e\t%.18e\t%.18e\t%.18e\t%.18e\t%.18e\n", phys_time, GasDiskMassInner, GasDiskMassOuter, DiskEcc_SurfNorm_Inner, DiskEcc_SurfNorm_Outer, DiskEcc_MassNorm_Inner, DiskEcc_MassNorm_Outer, StellarAccRate, LostMass);
    fclose (out);  
  }
  
  if (MonitorAccretion && DustGrid) {
    sprintf (filename, "%smon_accretion.dat", OUTPUTDIR);
    out = fopenp (filename, (char*) "a");
    fprintf (out, "%.18e", phys_time);
    for (int i=0; i< DustBinNum+1; i++)
      fprintf (out, "\t%.18e", Pl0AccretedMass[i]);
    fprintf (out, "\n");
    fclose (out);
  }
  
  // torques in barycentric system
  if (MonitorTorque) {
    int ii;
    double x, y, r, m, smoothing, iplanet, cs, frac;
    Force fc, fce;
    for (int i = 0; i < sys->nb; i++) {
      // planet's coordinate in barycentric system
      x = sys->x[i];
      y = sys->y[i];
      r = sqrt(x*x+y*y);
      m = sys->mass[i];
      if (RocheSmoothing) {
        smoothing = r*pow(m/3.0,1./3.)*ROCHESMOOTHING;
      } 
      else {
        iplanet = GetGlobalIFrac (r);
        frac = iplanet-floor(iplanet);
        ii = (int)iplanet;
        cs = GLOBAL_SOUNDSPEED[ii]*(1.0-frac)+GLOBAL_SOUNDSPEED[ii+1]*frac;
        smoothing = cs * r * sqrt(r) * THICKNESSSMOOTHING;
        smoothing=THICKNESSSMOOTHING * AspectRatio(r) * pow(r, 1.0+FLARINGINDEX);
      }
      r += smoothing;
      fc = ComputeForce_gpu (Rho, x, y, smoothing, m, 0);
      fce= ComputeForce_gpu (Rho, x, y, smoothing, m, ExcludeHill);
      
      double _x = x-GasBC_x;
      double _y = y-GasBC_y;

      sprintf (filename, "%smon_trq%d.dat", OUTPUTDIR, i);
      out = fopenp (filename, (char*) "a");
      fprintf (out, "%.18e\t%.18e\t%.18e\t%.18e\t%.18e\t%.18e\t%.18e\t%.18e\n", 
               phys_time,
               _x*fc.fy_inner     - _y*fc.fx_inner,      // inner disk torque
               _x*fc.fy_outer     - _y*fc.fx_outer,      // outer disk torque
               _x*fce.fy_ex_inner - _y*fce.fx_ex_inner,  // inner disk torque with Hill exclusion (!!)
               _x*fce.fy_ex_outer - _y*fce.fx_ex_outer,  // outer disk torque with Hill exclusion (!!)
               (_x * y - _y * x) * m / (r * r *r),       // stellar torque
               2.0/(m * r * pow(r, -1.5)),
               pow(m/ASPECTRATIO, 2.0) * Sigma(r) * pow(r, 4.0) * pow(r, -3.0));              // gamma0
      fclose (out);

      // dust torque (time, G0, G0', Gs0, Gs1, ...
      if (DustGrid) {
        // save dust torque profiles
        sprintf (filename, "%smon_tot_trq%d.dat", OUTPUTDIR, i);
        out = fopenp (filename, (char*) "a");
        fprintf (out, "%.18e\t%.18e", phys_time, _x*fc.fy_inner - _y*fc.fx_inner + _x*fc.fy_outer - _y*fc.fx_outer);
      
        fprintf (out, "\t%.18e\t%.18e", 2.0/(m * r * pow(r, -1.5)), pow(m/ASPECTRATIO, 2.0) * Sigma(r) * pow(r, 4.0) * pow(r, -3.0));
        for (int j=0; j < DustBinNum; j++) {
          fc = ComputeForce_gpu (DustRho[j], x, y, smoothing, m, 0);
          fce= ComputeForce_gpu (DustRho[j], x, y, smoothing, m, ExcludeHill);
          fprintf (out, "\t%.18e", _x*fc.fy_inner - _y*fc.fx_inner + _x*fc.fy_outer - _y*fc.fx_outer); 
        }
        fprintf (out, "\n");
        fclose (out);
      }
    }
  }
}

void WriteTorques (PlanetarySystem *sys, PolarGrid *Rho, PolarGrid **DustRho, int time_step) {
  FILE *out;
  char filename[MAX1D];  
  
  // torques in barycentric system
  if (MonitorTorque) {
    int ii;
    double x, y, r, m, smoothing, iplanet, cs, frac;
    Force fc, fce;
    for (int i = 0; i < sys->nb; i++) {
      // planet's coordinate in barycentric system
      x = sys->x[i];
      y = sys->y[i];
      r = sqrt(x*x+y*y);
      m = sys->mass[i];
      if (RocheSmoothing) {
        smoothing = r*pow(m/3.0,1./3.)*ROCHESMOOTHING;
      } 
      else {
        iplanet = GetGlobalIFrac (r);
        frac = iplanet-floor(iplanet);
        ii = (int)iplanet;
        cs = GLOBAL_SOUNDSPEED[ii]*(1.0-frac)+GLOBAL_SOUNDSPEED[ii+1]*frac;
        smoothing = cs * r * sqrt(r) * THICKNESSSMOOTHING;
        smoothing=THICKNESSSMOOTHING * AspectRatio(r) * pow(r, 1.0+FLARINGINDEX);
      }
      r += smoothing;
      fc = ComputeForce_gpu (Rho, x, y, smoothing, m, 0);
      fce= ComputeForce_gpu (Rho, x, y, smoothing, m, ExcludeHill);

      double _x = x-GasBC_x;
      double _y = y-GasBC_y;

      // save gas torque profiles
      sprintf (filename, "%smon_gas_trqprof%d_%d.dat", OUTPUTDIR, i, time_step);
      out = fopenp (filename, (char*) "w");
      for (int k=0; k< NRAD; k++) {
        fprintf (out, "%.18e\t%.18e\n",
                         Rmed[k], (_x*ForceY[k] - _y*ForceX[k]) * (2.0/(m * r * pow(r, -1.5))));
//                         Rmed[k], (_x*ForceY[k] - _y*ForceX[k]) * (m * m) * ASPECTRATIO * Sigma(r) * pow(r, 4.0));
      }
      fclose (out);
      masterprint ("Writing 'mon_gas_trqprof%d_%d.dat'\n", i, time_step);      

      // dust torque
      if (DustGrid) {
        for (int j=0; j < DustBinNum; j++) {
          fc = ComputeForce_gpu (DustRho[j], x, y, smoothing, m, 0);
          fce= ComputeForce_gpu (DustRho[j], x, y, smoothing, m, ExcludeHill);      
          
          // save dust torque profiles
          sprintf (filename, "%smon_ds%d_trqprof%d_%d.dat", OUTPUTDIR, j, i, time_step);
          out = fopenp (filename, (char*) "w");
          for (int k=0; k< NRAD; k++) {
            fprintf (out, "%.18e\t%.18e\n",
                           Rmed[k], (_x*ForceY[k] - _y*ForceX[k]) * (2.0/(m * r * pow(r, -1.5))));
          }
          fclose (out);
          masterprint ("Writing 'mon_ds%d_trqprof%d_%d.dat'\n", j, i, time_step);
        }
      }
    }
  }
}
