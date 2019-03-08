/** \file LowTasks.c

Contains many low level short functions.  The name of these functions
should be self-explanatory in most cases.  The prefix 'prs_' stands
for 'personal'. The prefix 'master' means that only the process 0
executes the function [note that the architecture is not of the kind
master/slaves, all processes perform similar tasks, but a minor
number of tasks (like output of information on the standard output) do
not need to be performed by several processes.] The function fopenp()
is an upper layer of fopen(), which should be used only in the case
of writing or appending a file (and not reading a file). It tries to
create the output directory if it does not exist, and it issues an
error message if it fails, so that the calling function does not
need to worry about these details.
*/

#include "fargo.h"
#include <stdarg.h>

double GetGlobalIFrac (double r) {
  int i=0;
  double ifrac;
  if (r < GlobalRmed[0]) return 0.0;
  if (r > GlobalRmed[GLOBALNRAD-1]) return (double)GLOBALNRAD-1.0;
  while (GlobalRmed[i] <= r) i++;
  ifrac = (double)i+(r-GlobalRmed[i-1])/(GlobalRmed[i]-GlobalRmed[i-1])-1.0;
  return ifrac;
}

void prs_exit (int numb) {
  //MPI_Finalize ();
  exit (numb);
}

void prs_end (PlanetarySystem *sys, int numb) {
  free (sys);
  //MPI_Finalize ();
  exit (numb);
}

void masterprint (const char *mytemplate, ...)
{
  va_list ap;
//  if (!CPU_Master) return;
  va_start (ap, mytemplate);
  vfprintf (stdout, mytemplate, ap);
  va_end (ap);
}

void mastererr (const char *mytemplate, ...)
{
  va_list ap;
//  if (!CPU_Master) return;
  va_start (ap, mytemplate);
  vfprintf (stderr, mytemplate, ap);
  va_end (ap);
}

void message (char *msg) {
	fprintf (stdout, "%s", msg);
}

void MultiplyPolarGridbyConstant (PolarGrid *arraysrc, double constant) {
  int i, nr, ns;
  double *fieldsrc;

  nr = arraysrc->Nrad;
  ns = arraysrc->Nsec;

  fieldsrc  =  arraysrc->Field;
#pragma omp parallel for
  for (i = 0; i < (nr+1)*ns; i++) {
    fieldsrc[i] *= constant;
  }
}

void DumpSources (int argc, char *argv[]) {
  char CommandLine[1024];
  char filecom[1024];
  int i;
  FILE *COM;
  //if (!CPU_Master) return;
  sprintf (filecom, "%srun.commandline", OUTPUTDIR);
  COM = fopenp (filecom, (char*) "w");
  for (i = 0; i < argc; i++) {
    fprintf (COM, "%s ",argv[i]);
  }
  fclose (COM);
  sprintf (CommandLine, "cp .source.tar.bz2 %ssrc.tar.bz2", OUTPUTDIR);
  system (CommandLine);
}

void MakeDir (char *string) {
  //int foo=0;
  char command[MAX1D];
  DIR *dir;
  /* Each processor tries to create the directory, sequentially */
  /* Silent if directory exists */
  //if (CPU_Rank) 
  //  MPI_Recv (&foo, 1, MPI_INT, CPU_Rank-1, 53, MPI_COMM_WORLD, &fargostat);
  dir = opendir (string);
  if (dir) {
    closedir (dir);
  } 
  else {
    fprintf (stdout, "Creating directory %s\n", string);
    sprintf (command, "mkdir -p %s", string);
    system (command);
  }
  //if (CPU_Rank < CPU_Number-1) 
  //  MPI_Send (&foo, 1, MPI_INT, CPU_Rank+1, 53, MPI_COMM_WORLD);
}


FILE *fopenp (char *string, char *mode) {
  FILE *f;
  f = fopen (string, mode);
  if (f == NULL) {
    /* This should be redundant with the call to MakeDir () at the
       beginning, from main.c; this is not a problem however */
    printf ("Could not open %s\n", string);
    printf ("Trying to create %s\n", OUTPUTDIR);
    MakeDir (OUTPUTDIR);
    f = fopen (string, "w");	/* "w" instead of mode: at this stage we know the file does not exist */
    if (f == NULL) {
      fprintf (stdout, "I still cannot open %s.\n", string);
      fprintf (stdout, "You should check that the permissions are correctly set.\n");
      fprintf (stdout, "Run aborted\n");
      prs_exit (1);
    }
  }
  return f;
}
