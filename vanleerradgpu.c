#include "fargo.h"

extern PolarGrid *RhoInt, *Work, *QRStar;

void VanLeerRadial_gpu (PolarGrid *Vrad, PolarGrid *Qbase, double dt, bool calc_lostmass) {
  FARGO_SAFE (DivisePolarGrid_gpu (Qbase, RhoInt, Work));
  FARGO_SAFE (ComputeStarRad_gpu (Work, Vrad, QRStar, dt));
  FARGO_SAFE (VanLeerRadial_gpu_cu (Vrad, Qbase, dt, calc_lostmass));
}

void VanLeerRadialDustSize_gpu (PolarGrid *Vrad, PolarGrid *Qbase, double dt) {
  //DivisePolarGrid_gpu (Qbase, RhoInt, Work);         // X/Sigma
  //ComputeStarRad_gpu (Work, Vrad, QRStar, dt);     // (X/Sigma)*
  FARGO_SAFE (ComputeStarRad_gpu (Qbase, Vrad, QRStar, dt));      // ar)*
  FARGO_SAFE (VanLeerRadialDustSize_gpu_cu (Vrad, Qbase, dt));
}
