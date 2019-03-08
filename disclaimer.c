#include "fargo.h"

void Disclaimer () {
  char string[1024];
  char filename[1024];
  FILE *f;
  string[0] = 0;
  sprintf (filename, "%s/.gfargo_disclaimer", getenv("HOME"));
  f = fopen (filename, "r");
  if (f != NULL) {
    fclose (f);
    return;
  }
  do {
    masterprint ("  \nDISCLAIMER\n\n");
    masterprint ("  The present software is provided 'as is' without any warranty of any kind,\n");
    masterprint ("  even the implied warranty of merchantability or fitness for a particular purpose.\n");
    masterprint ("  The use of GPUs for general purpose programming is notoriously unsecure. The\n");
    masterprint ("  content of the video RAM is not protected and can be accessed by any user of the\n");
    masterprint ("  system. The video RAM is not reset at the end of the job. The developers\n");
    masterprint ("  of the present software or their employers shall not be liable for loss or\n");
    masterprint ("  theft of data resulting from the use of this software, and not even for damage\n");
    masterprint ("  to the video card or any other part of the computer.\n\n");
    masterprint ("I HAVE READ AND UNDERSTOOD ALL THE ABOVE (yes/no) ");
    fscanf (stdin, "%s", string);
  } while ((strncmp(string, "yes",3) !=0) && (strncmp(string, "no", 2) != 0));
  if (strncmp(string, "no", 2) == 0) prs_exit (1);
  system ("touch $HOME/.gfargo_disclaimer");
}
