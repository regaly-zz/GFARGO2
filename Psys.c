/** \file Psys.c

Contains the functions that set up the planetary system configuration.
In addition, the last two functions allow to track the first planet
(number 0) of the planetary system, in order to perform a calculation
in the frame corotating either with this planet or with its
guiding-center.

*/

#include "fargo.h"

static double Xplanet, Yplanet;

int FindNumberOfPlanets (char *filename) {
  FILE *input;
  char s[512];
  int Counter=0;
  input = fopen (filename, "r");
  if (input == NULL) {
    fprintf (stderr, "Error : can't find '%s'.\n", filename);
    prs_exit (1);
  }
  while (fgets(s, 510, input) != NULL) {
    if (isalpha(s[0]))
      Counter++;
  }
  fclose (input);
  return Counter;
}

PlanetarySystem *AllocPlanetSystem (int nb) {
  double *mass, *x, *y, *vx, *vy, *acc, *acc_rate;
  bool *feeldisk, *feelothers;
  int i;
  PlanetarySystem *sys;
  sys  = (PlanetarySystem *)malloc (sizeof(PlanetarySystem));
  if (sys == NULL) {
    fprintf (stderr, "Not enough memory.\n");
    prs_exit (1);
  }
  x         = (double *)malloc (sizeof(double)*(nb+1));
  y         = (double *)malloc (sizeof(double)*(nb+1));
  vy        = (double *)malloc (sizeof(double)*(nb+1));
  vx        = (double *)malloc (sizeof(double)*(nb+1));
  mass      = (double *)malloc (sizeof(double)*(nb+1));
  acc       = (double *)malloc (sizeof(double)*(nb+1));
  acc_rate  = (double *)malloc (sizeof(double)*(nb+1));
  if ((x == NULL) || (y == NULL) || (vx == NULL) || (vy == NULL) || (acc == NULL) || (mass == NULL)) {
    fprintf (stderr, "Not enough memory.\n");
    prs_exit (1);
  }
  feeldisk   = (bool *)malloc (sizeof(double)*(nb+1));
  feelothers = (bool *)malloc (sizeof(double)*(nb+1));
  if ((feeldisk == NULL) || (feelothers == NULL)) {
    fprintf (stderr, "Not enough memory.\n");
    prs_exit (1);
  }
  sys->x = x;
  sys->y = y;
  sys->vx= vx;
  sys->vy= vy;
  sys->acc=acc;
  sys->mass = mass;
  sys->FeelDisk = feeldisk;
  sys->FeelOthers = feelothers;
  sys->acc_rate = acc_rate;
  for (i = 0; i < nb; i++) {
    x[i] = y[i] = vx[i] = vy[i] = mass[i] = acc[i] = acc_rate[i]  = 0.0;
    feeldisk[i] = feelothers[i] = YES;
  }
  return sys;
}
/*
CoreAccretion *AllocCoreAccretion (int nb) {
  double *critmass, *removemat;
  boolean *evolve;
  void **growthfunc;
  int i;
  char buffer[512];
  CoreAccretion *cacc;
  cacc  = (CoreAccretion *)malloc (sizeof(CoreAccretion));
  if (cacc == NULL) {
    fprintf (stderr, "Not enough memory.\n");
    prs_exit (1);
  }
  critmass    = (double *)malloc (sizeof(double)*(nb+1));
  removemat    = (double *)malloc (sizeof(double)*(nb+1));
  cacc  = (double *)malloc (sizeof(double)*(nb+1));
  if ((critmass == NULL) || (removemat == NULL)) {
    fprintf (stderr, "Not enough memory.\n");
    prs_exit (1);
  }
  evolve   = (boolean *)malloc (sizeof(double)*(nb+1));
  growthfunc = (void *)malloc (sizeof(void)*(nb+1));
  if ((evolve == NULL) || (growthfunc == NULL)) {
    fprintf (stderr, "Not enough memory.\n");
    prs_exit (1);
  }
  cacc->CriticalMass = critmass;
  cacc->Evolve = evolve;
  cacc->RemoveMaterial= removemat;
  cacc->GrowthFunction = growthfunc;
  for (i = 0; i < nb; i++) {
    critmass[i] = removemat[i] = 0.0;
    evolve[i] = NO;
    strcpy(buffer,"t");
    growthfunc[i] = evaluator_create(buffer);
    assert(growthfunc[i]);
  }
  return cacc;
}
*/

void FreePlanetary (PlanetarySystem *sys) {
  free (sys->x);
  free (sys->vx);
  free (sys->y);
  free (sys->vy);
  free (sys->mass);
  free (sys->acc);
  free (sys->FeelOthers);
  free (sys->FeelDisk);
  free (sys);
}

PlanetarySystem *InitPlanetarySystem (char *filename) {
  FILE *input;
  char s[512], nm[512], test1[512], test2[512], *s1;
  PlanetarySystem *sys;
  int i=0, nb;
  double mass, dist, accret;
  bool feeldis, feelothers;
  
  char dirfilename[512];
  if (OverridesOutputdir)
    sprintf (dirfilename, "%s/%s", NewOutputdir, filename);
  else
    sprintf (dirfilename, "%s", filename);
  
  nb = FindNumberOfPlanets (dirfilename);
//  if (CPU_Master)
//    printf ("%d planet(s) found.\n", nb);
  sys = AllocPlanetSystem (nb);
  input = fopen (dirfilename, "r");
  sys->nb = nb;
  while (fgets(s, 510, input) != NULL) {
    sscanf(s, "%s ", nm);
    if (isalpha(s[0])) {
      s1 = s + strlen(nm);
      sscanf(s1 + strspn(s1, "\t :=>_"), "%lf %lf %lf %s %s", &dist, &mass, &accret, test1, test2);
      sys->mass[i] = (double)mass;
      feeldis = feelothers = YES;
      if (tolower(*test1) == 'n') feeldis = NO;
      if (tolower(*test2) == 'n') feelothers = NO;
      sys->x[i] = (double) dist * (1.0+ECCENTRICITY);
      sys->y[i] = 0.0;
      sys->vy[i] = (double) sqrt((1.0+mass)/dist) *	sqrt(1.0-ECCENTRICITY*ECCENTRICITY)/(1.0+ECCENTRICITY);
      sys->vx[i] = 0.0;//-0.0000000001*sys->vy[i];
      sys->acc[i] = accret;
      sys->FeelDisk[i] = feeldis;
      sys->FeelOthers[i] = feelothers;
      i++;
    }
  }
  return sys;
}



PlanetarySystem *InitPlanetarySystemBin (char *filename) {
  FILE *input;
  char s[512], nm[512], test1[512], test2[512], *s1;
  PlanetarySystem *sys;
  int i=0, nb;
  double mass, dist, accret;
  bool feeldis, feelothers;
  
  char dirfilename[512];
  if (OverridesOutputdir)
    sprintf (dirfilename, "%s/%s", NewOutputdir, filename);
  else
    sprintf (dirfilename, "%s", filename);
  
  nb = FindNumberOfPlanets (dirfilename);
  sys = AllocPlanetSystem (nb);
  input = fopen (dirfilename, "r");
  sys->nb = nb;
  while (fgets(s, 510, input) != NULL) {
    sscanf(s, "%s ", nm);
    if (isalpha(s[0])) {
      s1 = s + strlen(nm);
      sscanf(s1 + strspn(s1, "\t :=>_"), "%lf %lf %lf %s %s", &dist, &mass, &accret, test1, test2);
      sys->mass[i] = (double)mass;
      feeldis = feelothers = YES;
      if (tolower(*test1) == 'n') feeldis = NO;
      if (tolower(*test2) == 'n') feelothers = NO;
      sys->x[i] = (double) dist * (1.0+ECCENTRICITY);
      sys->y[i] = 0.0;
      sys->vy[i] = (double) sqrt((1.0+mass)/dist) *	sqrt(1.0-ECCENTRICITY*ECCENTRICITY)/(1.0+ECCENTRICITY);
      sys->vx[i] = 0.0;
      sys->acc[i] = accret;
      sys->FeelDisk[i] = feeldis;
      sys->FeelOthers[i] = feelothers;
      i++;
    }
  }

  double _dist = 0.3;
  sys->vx[0] += sqrt((sys->mass[0]+sys->mass[1])/dist);
  sys->vx[1] -= sqrt((sys->mass[0]+sys->mass[1])/dist);
  sys->y[0] += _dist/2.0;
  sys->y[1] -= _dist/2.0;
    
  return sys;


}





void FindPlanetStarBC (PlanetarySystem *sys, double *xbc, double *ybc) {
  int nb = sys->nb;

  double total_mass = 1.0;
  for (int i=0; i< nb; i++) {
    total_mass += sys->mass[i];
  }  
  
  *xbc = 0.0;
  *ybc = 0.0;
  for (int i=0; i< nb; i++) {
    *xbc += sys->x[i]*sys->mass[i]/total_mass;
    *ybc += sys->y[i]*sys->mass[i]/total_mass;
  }

  printf ("\nBarycentre: xbc=%g ybc=%g\n", *xbc, *ybc);
}


/*
CoreAccretion *InitCoreAccretion (filename, sys_nb)
char *filename;
{
  FILE *input;
  char s[512], nm[512], growthfunc_buffer[512], *s1, test1[512];
  CoreAccretion *cacc;
  int i=0, nb;
  double critmass, removemat;
  boolean evolve;

  cacc = AllocCoreAccretion (sys_nb);
  nb = FindNumberOfPlanets (filename);
  if (CPU_Master)
    printf ("%d planet(s) found.\n", nb);
  if (nb != sys_nb) {
    printf ("Parameter mismach! No core accretion is set!");
    cacc->Evolve = NO;
    return cacc;
  }

  input = fopen (filename, "r");
  cacc->nb = nb;
  while (fgets(s, 510, input) != NULL) {
    sscanf(s, "%s ", nm);
    if (isalpha(s[0])) {
      s1 = s + strlen(nm);
      sscanf(s1 + strspn(s1, "\t :=>_"), "%lf %lf %s %lf", test1, &critmass, growthfunc_buffer, &removemat);
      cacc->CriticalMass[i] = (double)critmass;
      cacc->RemoveMaterial[i] = (double)removemat;
      evolve = YES;
      if (tolower(*test1) == 'n') evolve = NO;
      cacc->Evolve[i] = evolve;
      cacc->GrowthFunction[i] = evaluator_create(growthfunc_buffer);
      i++;
    }
  }
  return cacc;
}
*/

void ListPlanets (PlanetarySystem *sys) {
  int nb;
  int i;
  nb = sys->nb;
//  if (!CPU_Master) return;

  printf ("\nPlanets general properties\n");
  printf ("------------------------------------------------------------------------------\n");
  printf ("Mass tapering         : %f\n", MASSTAPER);
  printf ("Release date          : %f\n", RELEASEDATE);
  printf ("Release radius        : %f\n\n", RELEASERADIUS);
  for (i = 0; i < nb; i++) {
    printf ("Planet number %d\n", i);
    printf ("------------------------------------------------------------------------------\n");
    printf (" * mass               : %.10f\n", sys->mass[i]);
    printf (" * position           : x = %.10f\ty = %.10f\n", sys->x[i],sys->y[i]);
    printf (" * vlocity            : vx = %.10f\tvy = %.10f\n", sys->vx[i],sys->vy[i]);
    if (sys->acc[i] == 0.0)
      printf (" * panet is non-accreting\n");
    else
      printf (" * accretion time = %.10f\n", 1.0/(sys->acc[i]));
    if (sys->FeelDisk[i] == YES) {
      printf (" * planet feels the disk potential\n");
    } else {
      printf (" * doesn't feel the disk potential\n");
    }
    if (sys->FeelOthers[i] == YES) {
      printf (" * feels the other planets potential\n");
    } else {
      printf (" * doesn't feel the other planets potential\n");
    }
    printf ("\n");
  }
  printf ("\n");
}

double GetPsysInfo (PlanetarySystem *sys, int action) {
  double d1,d2,cross;
  double x,y, vx, vy, m, h, d, Ax, Ay, e, a, E, M;
  double xc, yc, vxc, vyc, omega;
  double arg, PerihelionPA;
  xc = x = sys->x[0];
  yc = y = sys->y[0];
  vxc = vx= sys->vx[0];
  vyc = vy= sys->vy[0];
  m = sys->mass[0]+1.;
  h = x*vy-y*vx;
  d = sqrt(x*x+y*y);
  Ax = x*vy*vy-y*vx*vy-G*m*x/d;
  Ay = y*vx*vx-x*vx*vy-G*m*y/d;
  e = sqrt(Ax*Ax+Ay*Ay)/m;
  a = h*h/G/m/(1.-e*e);
  if (e == 0.0) {
    arg = 1.0;
  } else {
    arg = (1.0-d/a)/e;
  }
  if (fabs(arg) >= 1.0) 
    E = M_PI*(1.-arg/fabs(arg))/2.;
  else
    E = acos((1.0-d/a)/e);
  if ((x*y*(vy*vy-vx*vx)+vx*vy*(x*x-y*y)) < 0) E= -E;
  M = E-e*sin(E);
  omega = sqrt(m/a/a/a);
  PerihelionPA=atan2(Ay,Ax);
  if (GuidingCenter == YES) {
    xc = a*cos(M+PerihelionPA);
    yc = a*sin(M+PerihelionPA);
    vxc = -a*omega*sin(M+PerihelionPA);
    vyc =  a*omega*cos(M+PerihelionPA);
  } 
  if (e < 1e-8) {		/* works in simple and double precision */
    xc = x;
    yc = y;
    vxc = vx;
    vyc = vy;
  }
  switch (action) {
    case MARK: 
      Xplanet = xc;
      Yplanet = yc;
      return 0.0;
    break;
    case GET:
      x = xc;
      y = yc;
      vx = vxc;
      vy = vyc;
      d2 = sqrt(x*x+y*y);
      d1 = sqrt(Xplanet*Xplanet+Yplanet*Yplanet);
      cross = Xplanet*y-x*Yplanet;
      Xplanet = x;
      Yplanet = y;
      return asin(cross/(d1*d2));
    break;
    case FREQUENCY:
      return omega;
    break;
  }
  return 0.0;
}

void RotatePsys (PlanetarySystem *sys, double angle) {
  int nb;
  int i;
  double sint, cost, xt, yt;
  nb = sys->nb;
  sint = sin(angle);
  cost = cos(angle);
  for (i = 0; i < nb; i++) {
    xt = sys->x[i];
    yt = sys->y[i];
    sys->x[i] = xt*cost+yt*sint;
    sys->y[i] = -xt*sint+yt*cost;
    xt = sys->vx[i];
    yt = sys->vy[i];
    sys->vx[i] = xt*cost+yt*sint;
    sys->vy[i] = -xt*sint+yt*cost;
  }
}
 
