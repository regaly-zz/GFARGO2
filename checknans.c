/** \file checknanas.c: contains a CUDA kernel for calculating disk eccentricities.
*/
#include "fargo.h"

extern PolarGrid *ListOfGrids;

extern PolarGrid *gas_v_theta;

int CheckNansPolar (PolarGrid *g) {
  int nr, ns, i;
  nr = g->Nrad;
  ns = g->Nsec;
  for (i = 0; i < nr*ns; i++)
    if (isnan(g->Field[i])) {
      return i;
    }
  return -1;
}

void CheckNans (char *string) {
  static int count=0;
  int l;
  PolarGrid *g;
  D2H_All ();
  g = ListOfGrids;
  while (g != NULL) {
    if ((l = CheckNansPolar (g)) >= 0) {
      printf ("Found NaNs in grid %s at location (%d, %d)\n", g->Name, l%(g->Nsec), l/(g->Nsec));
      printf ("position: after call to %s\n", string);
      WriteDiskPolar (g, 1111+count);
      count++;
      if (count == 5) exit(1);
    }
    g = g->next;
  }
}

