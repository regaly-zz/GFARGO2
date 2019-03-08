/** \file RungeKunta.c

A fifth-order Runge-Kunta integrator for the N-body problem. This
N-body problem consists of the central star and N-1 planets.

*/

#include "fargo.h"

static double k1[MAX1D], k2[MAX1D], k3[MAX1D], k4[MAX1D], k5[MAX1D], k6[MAX1D];
static double Dist[MAX1D];

void DerivMotionRK5 (double *q_init, double *masses, double *deriv, int n, double dt, bool *feelothers) {
  double *x,*y,*vx,*vy, dist;
  double *derivx, *derivy, *derivvx, *derivvy;
  int i, j;
  x = q_init;
  y = x+n;
  vx = y+n;
  vy = vx+n;
  derivx = deriv;
  derivy = deriv+n;
  derivvx = derivy+n;
  derivvy = derivvx+n;
  for (i = 0; i < n; i++) 
    Dist[i] = sqrt(x[i]*x[i]+y[i]*y[i]);
  for (i = 0; i < n; i++) {
    derivx[i] = vx[i];
    derivy[i] = vy[i];

    // force due to the central star
    derivvx[i] = -G*1.0/Dist[i]/Dist[i]/Dist[i]*x[i];
    derivvy[i] = -G*1.0/Dist[i]/Dist[i]/Dist[i]*y[i];
    //
    for (j = 0; j < n; j++) {
      // indirect force due to other planets
      if (Indirect_Term == YES) {
	      derivvx[i] -= G*masses[j]/Dist[j]/Dist[j]/Dist[j]*x[j];
	      derivvy[i] -= G*masses[j]/Dist[j]/Dist[j]/Dist[j]*y[j];
      }
      // force due to other planets
      if ((j != i) && (feelothers[i] == YES)) {
	      dist = (x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j]);
	      dist = sqrt(dist);
	      derivvx[i] += G*masses[j]/dist/dist/dist*(x[j]-x[i]);
	      derivvy[i] += G*masses[j]/dist/dist/dist*(y[j]-y[i]);
      }
    }
  }
    
  for (i = 0; i < 4*n; i++)
    deriv[i] *= dt;
}


void DerivMotionRK5RZS (PolarGrid *Rho, double *q_init, double *masses, double *deriv, int n, double dt, bool *feelothers) {
  Pair gamma;
  double smoothing, iplanet, frac, cs;
  int ii;
  
  double *x,*y,*vx,*vy, dist;
  double *derivx, *derivy, *derivvx, *derivvy;
  int i, j;
  x = q_init;
  y = x+n;
  vx = y+n;
  vy = vx+n;
  derivx = deriv;
  derivy = deriv+n;
  derivvx = derivy+n;
  derivvy = derivvx+n;
  for (i = 0; i < n; i++) 
    Dist[i] = sqrt(x[i]*x[i]+y[i]*y[i]);
  for (i = 0; i < n; i++) {
    derivx[i] = vx[i];
    derivy[i] = vy[i];

    // force due to the central star
    derivvx[i] = -G*1.0/Dist[i]/Dist[i]/Dist[i]*x[i];
    derivvy[i] = -G*1.0/Dist[i]/Dist[i]/Dist[i]*y[i];

    // force due to disk
    if (RocheSmoothing) {
      smoothing = Dist[i]*pow(masses[i]/3.0,1./3.)*ROCHESMOOTHING;
    } 
    else {
      iplanet = GetGlobalIFrac (Dist[i]);
      frac = iplanet-floor(iplanet);
      ii = (int)iplanet;
      cs = GLOBAL_SOUNDSPEED[ii]*(1.0-frac)+GLOBAL_SOUNDSPEED[ii+1]*frac;
      smoothing = cs * Dist[i] * sqrt(Dist[i]) * THICKNESSSMOOTHING;
    }
    gamma = ComputeAccel (Rho, x[i], y[i], smoothing, masses[i]);
    derivvx[i] += (double)gamma.x;
    derivvy[i] += (double)gamma.y;

    // disk indirect term
    derivvx[i] += (double)IndirectTerm.x;
    derivvy[i] += (double)IndirectTerm.y;
    
    //
    for (j = 0; j < n; j++) {
      // indirect force due to other planets
      if (Indirect_Term == YES) {
	      derivvx[i] -= G*masses[j]/Dist[j]/Dist[j]/Dist[j]*x[j];
	      derivvy[i] -= G*masses[j]/Dist[j]/Dist[j]/Dist[j]*y[j];
      }
      // force due to other planets
      if ((j != i) && (feelothers[i] == YES)) {
	      dist = (x[i]-x[j])*(x[i]-x[j])+(y[i]-y[j])*(y[i]-y[j]);
	      dist = sqrt(dist);
	      derivvx[i] += G*masses[j]/dist/dist/dist*(x[j]-x[i]);
	      derivvy[i] += G*masses[j]/dist/dist/dist*(y[j]-y[i]);
      }
    }
  }
    
  for (i = 0; i < 4*n; i++)
    deriv[i] *= dt;
}


void TranslatePlanetRK5 (double *qold, double c1, double c2, double c3, double c4, double c5, double *qnew, int n) {
  int i;
  for (i = 0; i < 4*n; i++)
    qnew[i] = qold[i]+c1*k1[i]+c2*k2[i]+c3*k3[i]+c4*k4[i]+c5*k5[i];
}

void RungeKunta (double *q0, double dt, double *masses, double *q1, int n, bool *feelothers) {
  int i;
  double timestep;
  timestep = dt;
  for (i = 0; i < n*4; i++) {
    k1[i] = k2[i] = k3[i] = k4[i] = k5[i] = k6[i];
  }
  DerivMotionRK5 (q0, masses, k1, n, timestep, feelothers);
  TranslatePlanetRK5 (q0, 0.2, 0.0, 0.0, 0.0, 0.0, q1, n);
  DerivMotionRK5 (q1, masses, k2, n, timestep, feelothers);
  TranslatePlanetRK5 (q0, 0.075, 0.225, 0.0, 0.0, 0.0, q1, n);
  DerivMotionRK5 (q1, masses, k3, n, timestep, feelothers);
  TranslatePlanetRK5 (q0, 0.3, -0.9, 1.2, 0.0, 0.0, q1, n);
  DerivMotionRK5 (q1, masses, k4, n, timestep, feelothers);
  TranslatePlanetRK5 (q0, -11.0/54.0, 2.5, -70.0/27.0, 35.0/27.0, 0.0, q1, n);
  DerivMotionRK5 (q1, masses, k5, n, timestep, feelothers);
  TranslatePlanetRK5 (q0, 1631.0/55296.0, 175.0/512.0, 575.0/13824.0, 44275.0/110592.0, 253.0/4096.0, q1, n);
  DerivMotionRK5 (q1, masses, k6, n, timestep, feelothers);
  for (i = 0; i < 4*n; i++) {
    q1[i]=q0[i]+37.0/378.0*k1[i]+250.0/621.0*k3[i]+125.0/594.0*k4[i]+512.0/1771.0*k6[i];
  }
}

void RungeKuntaRZS (PolarGrid *Rho, double *q0, double dt, double *masses, double *q1, int n, bool *feelothers) {
  int i;
  double timestep;
  timestep = dt;
  for (i = 0; i < n*4; i++) {
    k1[i] = k2[i] = k3[i] = k4[i] = k5[i] = k6[i];
  }
  DerivMotionRK5RZS (Rho, q0, masses, k1, n, timestep, feelothers);
  TranslatePlanetRK5 (q0, 0.2, 0.0, 0.0, 0.0, 0.0, q1, n);

  DerivMotionRK5RZS (Rho, q1, masses, k2, n, timestep, feelothers);
  TranslatePlanetRK5 (q0, 0.075, 0.225, 0.0, 0.0, 0.0, q1, n);

  DerivMotionRK5RZS (Rho, q1, masses, k3, n, timestep, feelothers);
  TranslatePlanetRK5 (q0, 0.3, -0.9, 1.2, 0.0, 0.0, q1, n);

  DerivMotionRK5RZS (Rho, q1, masses, k4, n, timestep, feelothers);
  TranslatePlanetRK5 (q0, -11.0/54.0, 2.5, -70.0/27.0, 35.0/27.0, 0.0, q1, n);

  DerivMotionRK5RZS (Rho, q1, masses, k5, n, timestep, feelothers);
  TranslatePlanetRK5 (q0, 1631.0/55296.0, 175.0/512.0, 575.0/13824.0, 44275.0/110592.0, 253.0/4096.0, q1, n);

  DerivMotionRK5RZS (Rho, q1, masses, k6, n, timestep, feelothers);
  for (i = 0; i < 4*n; i++) {
    q1[i]=q0[i]+37.0/378.0*k1[i]+250.0/621.0*k3[i]+125.0/594.0*k4[i]+512.0/1771.0*k6[i];
  }
}


 
