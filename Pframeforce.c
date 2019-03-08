/** \file Pframeforce.c

Functions that evaluate the %force between the planets and the disk.
The FillForcesArrays() function is ill-named: it rather fill an array
of the potential, that is later use to derive the force acting on the
disk at every zone.  The name of this file is due to the fact that we
work in the frame centered on the primary (which is therefore not
inertial). Old versions of fargo also featured the file Gframeforce.c,
in which we worked in the frame centered on the center of gravity of
the system.  The present file also contains the functions necessary to
update the planets' position and velocities (taking into account, or
not, the indirect term, ie the term arising from the fact that the
frame is not inertial), as well as the function that initializes the
hydrodynamics fields with analytic prescription.

*/

#include "fargo.h"
#include <stdio.h>
#include <time.h>

static double q0[MAX1D], q1[MAX1D], PlanetMasses[MAX1D];
//static double vt_int[MAX1D], vt_cent[MAX1D];

void ComputeIndirectTerm () {
  IndirectTerm.x = -DiskOnPrimaryAcceleration.x;
  IndirectTerm.y = -DiskOnPrimaryAcceleration.y; 
  if (Indirect_Term == NO) {
    IndirectTerm.x = 0.0;
    IndirectTerm.y = 0.0;
  }
}

/* Below : work in non-rotating frame */
/* centered on the primary */
void AdvanceSystemFromDisk (PolarGrid *Rho, PlanetarySystem *sys, double dt) {
  int NbPlanets, k;//, ii;
  Pair gamma;
  double x,y,r,m,smoothing; //,iplanet, frac, cs;
  NbPlanets = sys->nb;
  for (k = 0; k < NbPlanets; k++) {
    if (sys->FeelDisk[k] == YES) {      
      m=(double)sys->mass[k];
      x=(double)sys->x[k];
      y=(double)sys->y[k];
      r=sqrt(x*x+y*y);
      if (RocheSmoothing) {
	      smoothing = r*pow(m/3.0,1./3.)*ROCHESMOOTHING;
      } 
      else {
	      //iplanet = GetGlobalIFrac (r);
	      //frac = iplanet-floor(iplanet);
	      //ii = (int)iplanet;
	      //cs = GLOBAL_SOUNDSPEED[ii]*(1.0-frac)+GLOBAL_SOUNDSPEED[ii+1]*frac;
        //smoothing = cs * r * sqrt(r) * THICKNESSSMOOTHING;
        smoothing=THICKNESSSMOOTHING * AspectRatio(r) * pow(r, 1.0+FLARINGINDEX);
      }
      gamma = ComputeAccel (Rho, x, y, smoothing, m);
      sys->vx[k] += (double)dt * (double)gamma.x;
      sys->vy[k] += (double)dt * (double)gamma.y;
      sys->vx[k] += (double)dt * (double)IndirectTerm.x;
      sys->vy[k] += (double)dt * (double)IndirectTerm.y;
    }
  }
}

void AdvanceSystemFromDiskRZS (PolarGrid *Rho, PlanetarySystem *sys, double dt) {
  int NbPlanets, k;//, ii;
  //Pair gamma;
  //real x,y,r,m, iplanet, frac, cs, smoothing;
  NbPlanets = sys->nb;
  for (k = 0; k < NbPlanets; k++) {
    if (sys->FeelDisk[k] == YES) {
/*      m=(real)sys->mass[k];
      x=(real)sys->x[k];
      y=(real)sys->y[k];
      r=sqrt(x*x+y*y);
      if (RocheSmoothing) {
	      smoothing = r*pow(m/3.0,1./3.)*ROCHESMOOTHING;
      } 
      else {
	      iplanet = GetGlobalIFrac (r);
	      frac = iplanet-floor(iplanet);
	      ii = (int)iplanet;
	      cs = GLOBAL_SOUNDSPEED[ii]*(1.0-frac)+GLOBAL_SOUNDSPEED[ii+1]*frac;
        smoothing = cs * r * sqrt(r) * THICKNESSSMOOTHING;
      }
      gamma = ComputeAccel (Rho, x, y, smoothing, m);
      sys->vx[k] += (double)dt * (double)gamma.x;
     sys->vy[k] += (double)dt * (double)gamma.y;*/
      sys->vx[k] += (double)dt * (double)IndirectTerm.x;
      sys->vy[k] += (double)dt * (double)IndirectTerm.y;
    }
  }
}


void AdvanceSystemRK5 (PlanetarySystem *sys, double dt) {
  int i, n;
  bool *feelothers;
  double dtheta, omega, rdot, x, y, r, new_r, vx, vy, theta, denom;
  n = sys->nb;
  for (i = 0; i < n; i++) {
    q0[i] = sys->x[i];
    q0[i+n] = sys->y[i];
    q0[i+2*n] = sys->vx[i];
    q0[i+3*n] = sys->vy[i];
    PlanetMasses[i] = sys->mass[i];
  }
  feelothers = sys->FeelOthers;
  RungeKunta (q0, dt, PlanetMasses, q1, n, feelothers);
  for (i = 1-(PhysicalTime >= RELEASEDATE); i < sys->nb; i++) {
    sys->x[i] = q1[i];
    sys->y[i] = q1[i+n];
    sys->vx[i] = q1[i+2*n];
    sys->vy[i] = q1[i+3*n];
  }
  
  // Keplerian orbit (no migration, no interaction between planets)
  if (PhysicalTime < RELEASEDATE) {
    x = sys->x[0];
    y = sys->y[0];
    r = sqrt(x*x+y*y);
    theta = atan2(y,x);
    rdot = (RELEASERADIUS-r)/(RELEASEDATE-PhysicalTime);
    omega = sqrt((1.+sys->mass[0])/r/r/r);
    new_r = r + rdot*dt;
    denom = r-new_r;
    if (denom != 0.0) {
      dtheta = 2.*dt*r*omega/denom*(sqrt(r/new_r)-1.);
    } else {
      dtheta = omega*dt;
    }
    vx = rdot;
    vy = new_r*sqrt((1.+sys->mass[0])/new_r/new_r/new_r);
    sys->x[0] = new_r*cos(dtheta+theta);
    sys->y[0] = new_r*sin(dtheta+theta);
    sys->vx[0]= vx*cos(dtheta+theta)-vy*sin(dtheta+theta); 
    sys->vy[0]= vx*sin(dtheta+theta)+vy*cos(dtheta+theta); 
  }
}

void AdvanceSystemRK5RZS (PolarGrid *Rho, PlanetarySystem *sys, double dt) {
  int i, n;
  bool *feelothers;
  double dtheta, omega, rdot, x, y, r, new_r, vx, vy, theta, denom;
  n = sys->nb;
  for (i = 0; i < n; i++) {
    q0[i] = sys->x[i];
    q0[i+n] = sys->y[i];
    q0[i+2*n] = sys->vx[i];
    q0[i+3*n] = sys->vy[i];
    PlanetMasses[i] = sys->mass[i];
  }
  feelothers = sys->FeelOthers;
  RungeKuntaRZS (Rho, q0, dt, PlanetMasses, q1, n, feelothers);
  for (i = 1-(PhysicalTime >= RELEASEDATE); i < sys->nb; i++) {
    sys->x[i] = q1[i];
    sys->y[i] = q1[i+n];
    sys->vx[i] = q1[i+2*n];
    sys->vy[i] = q1[i+3*n];
  }
  
  // Keplerian orbit (no migration, no interaction between planets)
  if (PhysicalTime < RELEASEDATE) {
    x = sys->x[0];
    y = sys->y[0];
    r = sqrt(x*x+y*y);
    theta = atan2(y,x);
    rdot = (RELEASERADIUS-r)/(RELEASEDATE-PhysicalTime);
    omega = sqrt((1.+sys->mass[0])/r/r/r);
    new_r = r + rdot*dt;
    denom = r-new_r;
    if (denom != 0.0) {
      dtheta = 2.*dt*r*omega/denom*(sqrt(r/new_r)-1.);
    } else {
      dtheta = omega*dt;
    }
    vx = rdot;
    vy = new_r*sqrt((1.+sys->mass[0])/new_r/new_r/new_r);
    sys->x[0] = new_r*cos(dtheta+theta);
    sys->y[0] = new_r*sin(dtheta+theta);
    sys->vx[0]= vx*cos(dtheta+theta)-vy*sin(dtheta+theta); 
    sys->vy[0]= vx*sin(dtheta+theta)+vy*cos(dtheta+theta); 
  }
}

void SolveOrbits (PlanetarySystem *sys) {
  int i, n;
  double x,y,vx,vy;
  n = sys->nb;
  for (i = 0; i < n; i++) {
    x = sys->x[i];
    y = sys->y[i];
    vx = sys->vx[i];
    vy = sys->vy[i];
    FindOrbitalElements (x,y,vx,vy,1.0+sys->mass[i],i);
  }
} 

double ConstructSequence (double *u, double *v, int n) {
  int i;
  double lapl=0.0;
  for (i = 1; i < n; i++)
    u[i] = 2.0*v[i]-u[i-1];
  for (i = 1; i < n-1; i++) {
    lapl += fabs(u[i+1]+u[i-1]-2.0*u[i]);
  }
  return lapl;
}



/*
void InitGas (PolarGrid *Rho, PolarGrid *Vr, PolarGrid *Vt, PolarGrid *SGAcc) {

  int i, j, l, nr, ns;
  real *dens, *vr, *vt;
  double temporary;
  FILE *CS;
  char csfile[512];
  double  r, rg, omega, ri, vtemp;
  real viscosity, t1, t2, r1, r2;
  dens= Rho->Field;
  vr  = Vr->Field;
  vt  = Vt->Field;
  nr  = Rho->Nrad;
  ns  = Rho->Nsec;
  
	// #1 set sound speed
	sprintf (csfile, "%s%s", OUTPUTDIR, "soundspeed.dat");
  CS = fopen (csfile, "r");
  if (CS == NULL) {
    for (i = 0; i < nr; i++) {
      SOUNDSPEED[i] = AspectRatio(Rmed[i]) * sqrt(G*1.0/Rmed[i]) * pow(Rmed[i], FLARINGINDEX);
      CS2[i] = pow(SOUNDSPEED[i],2.0);
    }
    for (i = 0; i < GLOBALNRAD; i++) {
      GLOBAL_SOUNDSPEED[i] = AspectRatio(GlobalRmed[i]) * sqrt(G*1.0/GlobalRmed[i]) * pow(GlobalRmed[i], FLARINGINDEX);
    }
  } 
	else {
    masterprint ("Reading soundspeed.dat file\n");
    for (i = 0; i < GLOBALNRAD; i++) {
      fscanf (CS, "%lf", &temporary);
      GLOBAL_SOUNDSPEED[i] = (real)temporary;
    }
    for (i = 0; i < nr; i++) {
      SOUNDSPEED[i] = GLOBAL_SOUNDSPEED[i+IMIN];
    }
  }
  
	// #2 azimuthal speed
	for (i = 1; i < GLOBALNRAD; i++) {
    vt_int[i]=(GLOBAL_SOUNDSPEED[i]*GLOBAL_SOUNDSPEED[i]*Sigma(GlobalRmed[i])-\
	             GLOBAL_SOUNDSPEED[i-1]*GLOBAL_SOUNDSPEED[i-1]*Sigma(GlobalRmed[i-1]))/\
               (0.5*(Sigma(GlobalRmed[i])+Sigma(GlobalRmed[i-1])))/(GlobalRmed[i]-GlobalRmed[i-1])+\
               G*(1.0/GlobalRmed[i-1]-1.0/GlobalRmed[i])/(GlobalRmed[i]-GlobalRmed[i-1]);
    vt_int[i] = sqrt(vt_int[i]*Radii[i])-Radii[i]*OmegaFrame;
  }
  t1 = vt_cent[0] = vt_int[1]+0.75*(vt_int[1]-vt_int[2]);
  r1 = ConstructSequence (vt_cent, vt_int, GLOBALNRAD);
  vt_cent[0] += 0.25*(vt_int[1]-vt_int[2]);
  t2 = vt_cent[0];
  r2 = ConstructSequence (vt_cent, vt_int, GLOBALNRAD);
  t1 = t1-r1/(r2-r1)*(t2-t1);
  vt_cent[0] = t1;
  ConstructSequence (vt_cent, vt_int, GLOBALNRAD);
  vt_cent[GLOBALNRAD]=vt_cent[GLOBALNRAD-1];
  
	for (i = 0; i <= nr; i++) {
    if (i == nr) {
      r = DRmed[nr-1];
      ri= DRadii[nr-1+IMIN];
    }
    else {
      r = DRmed[i];
      ri= DRadii[i+IMIN];
    }
    viscosity = FViscosity (r);
    for (j = 0; j < ns; j++) {
      l = j+i*ns;
      rg = r;
      omega = sqrt(G*1.0/rg/rg/rg);
      vtemp = omega*r*sqrt(1.0-pow(ASPECTRATIO,2.0)*pow(r,2.0*FLARINGINDEX)*(1.+SIGMASLOPE-2.0*FLARINGINDEX));
      vtemp -= DInvSqrtRmed[i];
      vt[l] = (double)vtemp;
      if (CentrifugalBalance)
				vt[l] = vt_cent[i+IMIN]-sqrt(1.0/r);
      if (i == nr) 
				vr[l] = 0.0;
      else {
				vr[l] = IMPOSEDDISKDRIFT*SIGMA0/SigmaInf[i]/ri;
				if (ViscosityAlpha) {
					vr[l] -= 3.0*viscosity/r*(-SIGMASLOPE+2.0*FLARINGINDEX+1.0);
				} else {
					vr[l] -= 3.0*viscosity/r*(-SIGMASLOPE+.5);
				}
      }
      dens[l] = SigmaMed[i];
    }
  }
  for (j = 0; j < ns; j++) {
    vr[j] = vr[j+ns*nr] = 0.0;
  }
}


*/
/*
void InitGasEnergy (PolarGrid *Energy) {
  int i, j, l, nr, ns;
  double *energy;
  energy = Energy->Field;
  nr = Energy->Nrad;
  ns = Energy->Nsec;
  for (i = 0; i < nr; i++) {
    for (j = 0; j < ns; j++) {
      l = j+i*ns;
      energy[l] = EnergyMed[i];
    }
  }
}

void InitGas (PolarGrid *Rho, PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Energy, PolarGrid *SGAcc) {
  int i, j, l, nr, ns;
  double *dens, *vr, *vt, *energy = NULL;
  //double temporary;
  //FILE *CS;
  //char csfile[512];
  double  r, rg, omega, ri, vtemp;  
  double viscosity, t1, t2, r1, r2;

  nr  = Rho->Nrad;
  ns  = Rho->Nsec;

  dens = Rho->Field;
  vr  = Vrad->Field;
  vt  = Vtheta->Field;

  if (Adiabatic)
    energy = Energy->Field;
  
  // self gravity part
  //----------------------------
  double sgacc = 0.0, *sgacc_field = NULL;
  if (SelfGravity)
    sgacc_field = SGAcc->Field;
  //-----------------------------

  // set density and energy if adiabatic
  for (i = 0; i < nr; i++) {
    for (j = 0; j < ns; j++) {
      l = j+i*ns;
      dens[l] = SigmaMed[i];
      if (Adiabatic)
          energy[l] = EnergyMed[i];
    }
  }

	// #1 set sound speed
  //sprintf (csfile, "%s%s", OUTPUTDIR, "soundspeed.dat");
  //CS = fopen (csfile, "r");
  //if (CS == NULL) {
  //  for (i = 0; i < nr; i++) {
  //    //SOUNDSPEED[i] = AspectRatio(Rmed[i]) * sqrt(G*1.0/Rmed[i]) * pow(Rmed[i], FLARINGINDEX);
  //    SOUNDSPEED[i] = SoundSpeed (Rmed[i], EnergyMed[i], SigmaMed[i]);
  //    CS2[i] = pow(SOUNDSPEED[i],2.0);
  //  }
  //  for (i = 0; i < GLOBALNRAD; i++) {
  //    //GLOBAL_SOUNDSPEED[i] = AspectRatio(GlobalRmed[i]) * sqrt(G*1.0/GlobalRmed[i]) * pow(GlobalRmed[i], FLARINGINDEX);
  //    GLOBAL_SOUNDSPEED[i] = SoundSpeed (Rmed[i], EnergyMed[i], SigmaMed[i]);
  //  }
  //} 
  //else {
  //  masterprint ("Reading soundspeed.dat file\n");
  //  for (i = 0; i < GLOBALNRAD; i++) {
  //    fscanf (CS, "%lf", &temporary);
  //    GLOBAL_SOUNDSPEED[i] = (double) temporary;
  //  }
  //  for (i = 0; i < nr; i++) {
  //    SOUNDSPEED[i] = GLOBAL_SOUNDSPEED[i+IMIN];
  //  }
  //}

  // #2 azimuthal speed needed for centrifugal ballance
  for (i = 1; i < GLOBALNRAD; i++) {
    vt_int[i]=(GLOBAL_SOUNDSPEED[i]*GLOBAL_SOUNDSPEED[i]*Sigma(GlobalRmed[i])-\
               GLOBAL_SOUNDSPEED[i-1]*GLOBAL_SOUNDSPEED[i-1]*Sigma(GlobalRmed[i-1]))/\
               (0.5*(Sigma(GlobalRmed[i])+Sigma(GlobalRmed[i-1])))/(GlobalRmed[i]-GlobalRmed[i-1])+\
               G*(1.0/GlobalRmed[i-1]-1.0/GlobalRmed[i])/(GlobalRmed[i]-GlobalRmed[i-1]);

    // [RZS-MOD]
    // ?????????
    //------------------------------------------
    if (SelfGravity) {
      sgacc = sgacc_field[i*ns];
      vt_int[i] = sqrt(vt_int[i]*Radii[i]-sgacc*Radii[i])-Radii[i]*OmegaFrame;  //-Radii[i]*OmegaFrame; // [RZS-CHECK] must be removed otherwise bad initiation!
    }
    else
      vt_int[i] = sqrt(vt_int[i]*Radii[i])-Radii[i]*OmegaFrame;
    //------------------------------------------
  }
  t1 = vt_cent[0] = vt_int[1]+.75*(vt_int[1]-vt_int[2]);
  r1 = ConstructSequence (vt_cent, vt_int, GLOBALNRAD);
  vt_cent[0] += .25*(vt_int[1]-vt_int[2]);
  t2 = vt_cent[0];
  r2 = ConstructSequence (vt_cent, vt_int, GLOBALNRAD);
  t1 = t1-r1/(r2-r1)*(t2-t1);
  vt_cent[0] = t1;
  ConstructSequence (vt_cent, vt_int, GLOBALNRAD);
  vt_cent[GLOBALNRAD]=vt_cent[GLOBALNRAD-1];
  
  // #3 set radial and azimuthal velocity
  for (i = 0; i < GLOBALNRAD; i++) {
    
    //if (i==nr-1) {
    //  r = DRmed[i];
    //  ri= DRadii[i+IMIN];
    //  if (SelfGravity) {
    //    double a = (sgacc_field[(nr-2)*ns]-sgacc_field[(nr-3)*ns])/(DRmed[nr-2]-DRmed[nr-3]);
    //    sgacc= 2*(a * r + sgacc_field[(nr-3)*ns] - a * DRmed[nr-3]);
    //    sgacc = sgacc_field[(i-1)*ns];
    //  }
    //}
    //else {
      
	 //	
	 //	
	 //	r = DRmed[i];
   //  ri= DRadii[i+IMIN];
   //  //rmed = Rmed[i];
   //  // [RZS-MOD]
   //  if (SelfGravity) {
   //      sgacc= sgacc_field[i*ns];
   //  }
   //  //}
   //
    
    if (i == nr) {
      r = DRmed[nr-1];
      ri= DRadii[nr-1+IMIN];
		  if (SelfGravity)
      	sgacc=sgacc_field[(nr-1)*ns];
    }
    else {
      r = DRmed[i];
      ri= DRadii[i+IMIN];
		  if (SelfGravity) 
      	sgacc= sgacc_field[i*ns];
    }
    
    viscosity = FViscosity (r);
    for (j = 0; j < ns; j++) {
      l = j+i*ns;
      rg = r;
      omega = sqrt(G*1.0/rg/rg/rg);
      // [RZS-MOD]
      // add acceleration due to self-gravity
      //-------------------------------------

			//
      // omega = sqrt(G*1.0/rg/rg/rg);
      // vtemp = omega*r*sqrt(1.0-pow(ASPECTRATIO,2.0)*pow(r,2.0*FLARINGINDEX)*(1.+SIGMASLOPE-2.0*FLARINGINDEX));
      // vtemp -= DInvSqrtRmed[i];
			
      // azimuthal velocity
      if (SelfGravity) {
          if (SIGMACUTOFFRADOUT==0)
            vtemp = r*sqrt(omega*omega*(1.0-pow(ASPECTRATIO,2.0) * pow(r,2.0*FLARINGINDEX)* (1.0+SIGMASLOPE-2.0*FLARINGINDEX)) - sgacc/r);
          else
            vtemp = r*sqrt(omega*omega*(1.0+pow(ASPECTRATIO,2.0)*(SIGMASLOPE-2)*pow(r,2+2*FLARINGINDEX-SIGMASLOPE)*pow(SIGMACUTOFFRADOUT,SIGMASLOPE-2)- pow(ASPECTRATIO,2.0)*pow(r,2*FLARINGINDEX)*(1+SIGMASLOPE-2*FLARINGINDEX)) - sgacc/r);
      }
      else {
        if (SIGMACUTOFFRADOUT==0)
          vtemp = omega * r * sqrt(1.0-pow(ASPECTRATIO,2.0) * pow(r,2.0*FLARINGINDEX) * (1.0+SIGMASLOPE-2.0*FLARINGINDEX));                
        else
          vtemp = omega * r * sqrt(1.0+pow(ASPECTRATIO,2.0)*(SIGMASLOPE-2)*pow(r,2+2*FLARINGINDEX-SIGMASLOPE)*pow(SIGMACUTOFFRADOUT,SIGMASLOPE-2)- pow(ASPECTRATIO,2.0)*pow(r,2*FLARINGINDEX)*(1+SIGMASLOPE-2*FLARINGINDEX));
      }
      //-------------------------------------
      vtemp -= DInvSqrtRmed[i];//+r*OmegaFrame;      
      vt[l] = vtemp;
        
      if (CentrifugalBalance)
 	      vt[l] = vt_cent[i+IMIN]-sqrt(1.0/r);
      
      // radial velocity
      if (i == nr) 
  	    vr[l] = 0.0;
      else {
	      vr[l] = IMPOSEDDISKDRIFT*SIGMA0/SigmaInf[i]/ri;
         
        // alpha-viscosity
	      if (ViscosityAlpha) {
          if (SIGMACUTOFFRADOUT==0)
            vr[l] -= 3.0*viscosity/r*(-SIGMASLOPE+2.0*FLARINGINDEX+1.0);
          else
            vr[l] -= 3.0*FViscosity(r)/r*(-SIGMASLOPE+2.0*FLARINGINDEX+1.0+(SIGMASLOPE-2)*pow(r/SIGMACUTOFFRADOUT, 2-SIGMASLOPE));
	      }
         
        // constant viscosity 
        else {
          if (SIGMACUTOFFRADOUT==0)
            vr[l] -= 3.0*viscosity/r*(-SIGMASLOPE+.5);
           else  
            vr[l] -= 3.0*FViscosity(r)/r*(-SIGMASLOPE+0.5+(SIGMASLOPE-2)*pow(r/SIGMACUTOFFRADOUT, 2-SIGMASLOPE));
	      }
      }
      vr[l] = 0.0;
      
    }
  }  
  if (verbose)
    masterprint ("Stellocentric disk is initialized\n");
}
*/
/*
// [RZS-MOD]
// initial condition in barycentric system
//-----------------------------------------
void InitGasBC (PlanetarySystem *sys, PolarGrid *Rho, PolarGrid *Vr, PolarGrid *Vt, PolarGrid *SGAcc) {
  int i, j, l, nr, ns;
  double *dens, *vr, *vt;
  double temporary;
  FILE *CS;
  char csfile[512];
  double  r=0.0, omega, ri, vtemp;  
  double viscosity, t1, t2, r1, r2;
  dens= Rho->Field;

  // self gravity part
  //----------------------------
  double sgacc = 0.0, *sgacc_field = NULL;
  if (SelfGravity)
    sgacc_field = SGAcc->Field;
  //-----------------------------
  vr  = Vr->Field;
  vt  = Vt->Field;
  nr  = Rho->Nrad;
  ns  = Rho->Nsec;
  sprintf (csfile, "%s%s", OUTPUTDIR, "soundspeed.dat");
  CS = fopen (csfile, "r");
  if (CS == NULL) {
    for (i = 0; i < nr; i++) {
      SOUNDSPEED[i] = AspectRatio(Rmed[i]) * sqrt(G*1.0/Rmed[i]) * pow(Rmed[i], FLARINGINDEX);
      CS2[i] = pow(SOUNDSPEED[i],2.0);
    }
    for (i = 0; i < GLOBALNRAD; i++) {
      GLOBAL_SOUNDSPEED[i] = AspectRatio(GlobalRmed[i]) * sqrt(G*1.0/GlobalRmed[i]) * pow(GlobalRmed[i], FLARINGINDEX);
    }
  } 
  else {
    masterprint ("Reading soundspeed.dat file\n");
    for (i = 0; i < GLOBALNRAD; i++) {
      fscanf (CS, "%lf", &temporary);
      GLOBAL_SOUNDSPEED[i] = (double)temporary;
    }
    for (i = 0; i < nr; i++) {
      SOUNDSPEED[i] = GLOBAL_SOUNDSPEED[i+IMIN];
    }
  }

  for (i = 1; i < GLOBALNRAD; i++) {
    vt_int[i]=(GLOBAL_SOUNDSPEED[i]*GLOBAL_SOUNDSPEED[i]*Sigma(GlobalRmed[i])-\
               GLOBAL_SOUNDSPEED[i-1]*GLOBAL_SOUNDSPEED[i-1]*Sigma(GlobalRmed[i-1]))/\
               (.5*(Sigma(GlobalRmed[i])+Sigma(GlobalRmed[i-1])))/(GlobalRmed[i]-GlobalRmed[i-1])+\
               G*(1.0/GlobalRmed[i-1]-1.0/GlobalRmed[i])/(GlobalRmed[i]-GlobalRmed[i-1]);

    // [RZS-MOD]
    // ?????????
    //------------------------------------------
    if (SelfGravity) {
      sgacc = sgacc_field[i*ns];
      vt_int[i] = sqrt(vt_int[i]*Radii[i]-sgacc*Radii[i])-Radii[i]*OmegaFrame;  //-Radii[i]*OmegaFrame; // [RZS-CHECK] must be removed otherwise bad initiation!
    }
    else
      vt_int[i] = sqrt(vt_int[i]*Radii[i])-Radii[i]*OmegaFrame;
    //------------------------------------------
  }
  t1 = vt_cent[0] = vt_int[1]+.75*(vt_int[1]-vt_int[2]);
  r1 = ConstructSequence (vt_cent, vt_int, GLOBALNRAD);
  vt_cent[0] += .25*(vt_int[1]-vt_int[2]);
  t2 = vt_cent[0];
  r2 = ConstructSequence (vt_cent, vt_int, GLOBALNRAD);
  t1 = t1-r1/(r2-r1)*(t2-t1);
  vt_cent[0] = t1;
  ConstructSequence (vt_cent, vt_int, GLOBALNRAD);
  vt_cent[GLOBALNRAD]=vt_cent[GLOBALNRAD-1];

  double xbc, ybc;
  FindPlanetStarBC (sys, &xbc, &ybc);
  
  //xbc = 0.0;
  //ybc = 0.0;
    
  double dphi = 2.0 * M_PI/(double) ns;
  for (i = 0; i < nr; i++) {
    
    if (SelfGravity) {
      sgacc= sgacc_field[i*ns];
    }
    
    viscosity = FViscosity (r);
    for (j = 0; j < ns; j++) {

      // calculate distance from the baricentre based on law of cosines
      double phi = dphi * (double) j + dphi/2.0;
      double rbc  = sqrt(DRmed[i]*DRmed[i] + xbc*xbc - 2.0*DRmed[i]*xbc*cos(phi));
      double rbci = sqrt(DRadii[i]*DRadii[i] + xbc*xbc - 2.0*DRadii[i]*xbc*cos(phi));
      r = rbc;
      ri = rbci;
            
      l = j+i*ns;

      omega = sqrt(G*1.0/r/r/r);

      // [RZS-MOD]
      // add acceleration due to self-gravity
      //-------------------------------------
      if (SelfGravity) {
        if (SIGMACUTOFFRAD==0)
          vtemp = r*sqrt(omega*omega*(1.0-pow(ASPECTRATIO,2.0) * pow(r,2.0*FLARINGINDEX)* (1.+SIGMASLOPE-2.0*FLARINGINDEX)) - sgacc/r);
        else  
          vtemp = r*sqrt(omega*omega*(1.0+pow(ASPECTRATIO,2.0)*(SIGMASLOPE-2)*pow(r,2+2*FLARINGINDEX-SIGMASLOPE)*pow(SIGMACUTOFFRAD,SIGMASLOPE-2)- pow(ASPECTRATIO,2.0)*pow(r,2*FLARINGINDEX)*(1+SIGMASLOPE-2*FLARINGINDEX)) - sgacc/r);
      }
      else {
        if (SIGMACUTOFFRAD==0)
          vtemp = omega * r * sqrt(1.0-pow(ASPECTRATIO,2.0) * pow(r,2.0*FLARINGINDEX) * (1.+SIGMASLOPE-2.0*FLARINGINDEX));          
        else
          vtemp = omega * r * sqrt(1.0+pow(ASPECTRATIO,2.0)*(SIGMASLOPE-2)*pow(r,2+2*FLARINGINDEX-SIGMASLOPE)*pow(SIGMACUTOFFRAD,SIGMASLOPE-2)- pow(ASPECTRATIO,2.0)*pow(r,2*FLARINGINDEX)*(1+SIGMASLOPE-2*FLARINGINDEX));
      }
      //-------------------------------------
      vtemp -= DInvSqrtRmed[i];
      vt[l] = (double)vtemp;

      if (CentrifugalBalance)
 	      vt[l] = vt_cent[i+IMIN]-sqrt(1.0/r);
      
      if (i == nr) 
	      vr[l] = 0.0;
      else {
	      vr[l] = IMPOSEDDISKDRIFT*SIGMA0/SigmaInf[i]/ri;
	      if (ViscosityAlpha) {
          if (SIGMACUTOFFRAD==0)
  	        vr[l] -= 3.0*viscosity/r*(-SIGMASLOPE+2.0*FLARINGINDEX+1.0);
          else  
            vr[l] -= 3.0*FViscosity(r)/r*(-SIGMASLOPE+2.0*FLARINGINDEX+1.0+(SIGMASLOPE-2)*pow(r/SIGMACUTOFFRAD, 2-SIGMASLOPE));
	      } 
        else {
          if (SIGMACUTOFFRAD==0)
            vr[l] -= 3.0*viscosity/r*(-SIGMASLOPE+.5);
          else  
            vr[l] -= 3.0*FViscosity(r)/r*(-SIGMASLOPE+0.5+(SIGMASLOPE-2)*pow(r/SIGMACUTOFFRAD, 2-SIGMASLOPE));
	      }
      }
      vr[l] = 0.0;

      // take into account the baricentre
      //double sin_alpha = (xbc/r)*sin (phi);
      //double cos_alpha = sqrt (1.0-sin_alpha*sin_alpha);
      
      //vt[l] = vt[l]*cos_alpha+vr[l]*sin_alpha;
      //vr[l] = -vt[l]*sin_alpha+vr[l]*cos_alpha;      
      
      // density
      dens[l] = Sigma(rbc);
    }
  }
  masterprint ("Baricentric disk is initialized\n");
}
//----------------------------------------------
*/