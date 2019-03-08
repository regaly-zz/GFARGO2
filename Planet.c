/** \file Planet.c

Accretion of disk material onto the planets, and solver of planetary
orbital elements.  The prescription used for the accretion is the one
designed by W. Kley.

*/

#include "fargo.h"

void AccreteOntoPlanets (PolarGrid *Rho, PolarGrid *Vrad, PolarGrid *Vtheta, double dt, PlanetarySystem *sys) {
  double RRoche, Rplanet, distance, dx, dy, deltaM, angle;
  int i_min,i_max, j_min, j_max, i, j, l, jf, ns, nr, lip, ljp, k;
  double Xplanet, Yplanet, Mplanet, VXplanet, VYplanet;
  double facc, facc1, facc2, frac1, frac2; /* We adopt the same notations as W. Kley */
  double *dens, *abs, *ord, *vrad, *vtheta;
  double PxPlanet, PyPlanet, vrcell, vtcell, vxcell, vycell, xc, yc;
  double dPxPlanet, dPyPlanet, dMplanet;
  nr     = Rho->Nrad;
  ns     = Rho->Nsec;
  dens   = Rho->Field;
  abs    = CellAbscissa->Field;
  ord    = CellOrdinate->Field;
  vrad   = Vrad->Field;
  vtheta = Vtheta->Field;
  for (k=0; k < sys->nb; k++) {
    // accretion only for eta>10^-10
    if (sys->acc[k] > 1e-10) {
      
      // zero the change in planet mass and impulse at the beginning
      dMplanet = dPxPlanet = dPyPlanet = 0.0;
            
      // get planetary parameter
      Xplanet = sys->x[k];
      Yplanet = sys->y[k];
      VXplanet = sys->vx[k];
      VYplanet = sys->vy[k];
      Mplanet = sys->mass[k];
      Rplanet = sqrt(Xplanet*Xplanet+Yplanet*Yplanet);
      
      // initialization of W. Kley's parameters
      facc = dt*(sys->acc[k])*(sys->acc[k])*pow(Rplanet, -1.5);;
      facc1 = 1.0/3.0*facc;
      facc2 = 2.0/3.0*facc;
      frac1 = 0.75;
      frac2 = 0.45;

      // Roche radius assuming that the central mass is 1.0
      RRoche = pow((1.0/3.0*Mplanet),(1.0/3.0))*Rplanet; 

      // select the indices in the Roche lobe region
      i_min=0;
      i_max=nr-1;
      while ((Rsup[i_min] < Rplanet-RRoche) && (i_min < nr)) i_min++;
      while ((Rinf[i_max] > Rplanet+RRoche) && (i_max > 0)) i_max--;
      angle = atan2 (Yplanet, Xplanet);
      j_min =(int)((double)ns/2.0/M_PI*(angle - 2.0*RRoche/Rplanet));
      j_max =(int)((double)ns/2.0/M_PI*(angle + 2.0*RRoche/Rplanet));

      PxPlanet = Mplanet*VXplanet;
      PyPlanet = Mplanet*VYplanet;

      for (i = i_min; i <= i_max; i++) {
        for (j = j_min; j <= j_max; j++) {
          jf = j;
          while (jf <  0)  jf += ns;
          while (jf >= ns) jf -= ns;
          l   = jf+i*ns;
          lip = l+ns;
          ljp = l+1;
          if (jf == ns-1) ljp = i*ns;
          xc = abs[l];
          yc = ord[l];
          dx = Xplanet-xc;
          dy = Yplanet-yc;
          distance = sqrt(dx*dx+dy*dy);
          vtcell=0.5*(vtheta[l]+vtheta[ljp])+Rmed[i]*OmegaFrame + 1/sqrt(Rmed[i]);
          vrcell=0.5*(vrad[l]+vrad[lip]);
          vxcell=(vrcell*xc-vtcell*yc)/Rmed[i];
          vycell=(vrcell*yc+vtcell*xc)/Rmed[i];
          if (distance < frac1*RRoche) {
            deltaM = facc1*dens[l]*Surf[i];
            //if (i < Zero_or_active) deltaM = 0.0;
            //if (i >= Max_or_active) deltaM = 0.0;
            dens[l] *= (1.0 - facc1);
  
            dPxPlanet    += deltaM*vxcell;
  
            dPyPlanet    += deltaM*vycell;
  
            dMplanet     += deltaM;
          }
          if (distance < frac2*RRoche) {
            deltaM = facc2*dens[l]*Surf[i];
            //if (i < Zero_or_active) deltaM = 0.0;
            //if (i >= Max_or_active) deltaM = 0.0;
            dens[l] *= (1.0 - facc2);
  
            dPxPlanet    += deltaM*vxcell;
  
            dPyPlanet    += deltaM*vycell;
  
            dMplanet     += deltaM;
          }
        }
      }
      
      PxPlanet += dPxPlanet;
      PyPlanet += dPyPlanet;
      Mplanet  += dMplanet;
      
      if (sys->FeelDisk[k] == YES) {
	      sys->vx[k] = PxPlanet/Mplanet;
	      sys->vy[k] = PyPlanet/Mplanet;
      }
      sys->mass[k] = Mplanet;
      
      printf ("\n%g %g\n",dMplanet,sys->mass[k]);
    }
  }
}


void FindOrbitalElements (double x, double y, double vx, double vy, double m, int n) {
    static double angle[12] = {0,0,0,0,0,0,0,0,0,0,0,0};
  double Ax, Ay, e, d, h, a, E, M, V;
  double PerihelionPA;
  FILE *output;
  char name[256];
  //if (CPU_Rank != CPU_Number-1) return;
  sprintf (name, "%sorbit%d.dat", OUTPUTDIR, n);
  output = fopenp (name, (char*) "a");
  h = x*vy-y*vx;
  d = sqrt(x*x+y*y);
  Ax = x*vy*vy-y*vx*vy-G*m*x/d;
  Ay = y*vx*vx-x*vx*vy-G*m*y/d;
  e = sqrt(Ax*Ax+Ay*Ay)/G/m;
  a = h*h/G/m/(1-e*e);
  // [RZS-MOD]
  // check whether planet reached the minimum sem-major axis distance
  if (a < MinSemiMajorPlanet)
    TerminateDuetoPlanet = true;
    
  if (e != 0.0) {
    E = acos((1.0-d/a)/e);
  } else {
    E = 0.0;
  }
  if ((x*y*(vy*vy-vx*vx)+vx*vy*(x*x-y*y)) < 0) E= -E;
  M = E-e*sin(E);
  if (e != 0.0) {
    V = acos ((a*(1.0-e*e)/d-1.0)/e);
  } else {
    V = 0.0;
  }
  if (E < 0.0) V = -V;
  if (e != 0.0) {
    PerihelionPA=atan2(Ay,Ax);
  } else {
    PerihelionPA=atan2(y,x);
  }
  
  // [RZS-MOD]
  // add planetary mass and number of orbits to the output
  //------------------------------------------------------
  double theta = atan2(y,x);
  if (theta < 0.0) 
    theta += 2.0*M_PI;
  angle[n] += theta;
  fprintf (output, "%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\t%.12g\n", PhysicalTime, e, a, M, V, PerihelionPA, m-1.0,angle[n]);
  //------------------------------------------------------
  fclose (output);
}
 
