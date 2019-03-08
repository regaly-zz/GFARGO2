#include "fargo.h"

extern PolarGrid *RhoInt, *Work, *QRStar;

void VanLeerTheta_gpu (PolarGrid *Vtheta, PolarGrid *Qbase, double dt) {
  FARGO_SAFE (DivisePolarGrid_gpu (Qbase, RhoInt, Work));
  FARGO_SAFE (ComputeStarTheta_gpu (Work, Vtheta, QRStar, dt));
  FARGO_SAFE (VanLeerTheta_gpu_cu (Vtheta, Qbase, dt));
}

void VanLeerThetaDustSize_gpu (PolarGrid *Vtheta, PolarGrid *Qbase, double dt) {
  //DivisePolarGrid_gpu (Qbase, RhoInt, Work);
  //ComputeStarTheta_gpu (Work, Vtheta, QRStar, dt);
  FARGO_SAFE (ComputeStarTheta_gpu (Qbase, Vtheta, QRStar, dt)); 
  FARGO_SAFE (VanLeerThetaDustSize_gpu_cu (Vtheta, Qbase, dt));
}