/** \file TransportEuler.c

Functions that handle the transport substep of a hydrodynamical time
step.  The FARGO algorithm is implemented here. The transport is
performed in a manner similar to what is done for the ZEUS code (Stone
& Norman, 1992), except for the momenta transport (we define a left
and right momentum for each zone, which we declare zone centered; we
then transport then normally, and deduce the new velocity in each zone
by a proper averaging).

*/

#include "fargo.h"

#include <time.h>

int Nshift[MAX1D];
int MyNshift[MAX1D];

void InitTransport () {
  RadMomP      = CreatePolarGrid(NRAD, NSEC, "RadMomP");
  RadMomM      = CreatePolarGrid(NRAD, NSEC, "RadMomM");
  ThetaMomP    = CreatePolarGrid(NRAD, NSEC, "ThetaMomP");
  ThetaMomM    = CreatePolarGrid(NRAD, NSEC, "ThetaMomM");
  Work         = CreatePolarGrid(NRAD, NSEC, "WorkGrid");
  QRStar       = CreatePolarGrid(NRAD, NSEC, "QRStar");
  ExtLabel     = CreatePolarGrid(NRAD, NSEC, "ExtLabel");
  VthetaRes    = CreatePolarGrid(NRAD, NSEC, "VThetaRes");
  Elongations  = CreatePolarGrid(NRAD, NSEC, "Elongations");
  Qder         = CreatePolarGrid(NRAD, NSEC, "slope");
}


// Transport of gas and dust
//-----------------------------------------------------------------------------------
void OneWindRad (PolarGrid *Rho, PolarGrid *Vrad, PolarGrid *Energy, PolarGrid *DustSize, double dt, bool advect_vel) {
  FARGO_SAFE (ComputeStarRad_gpu (Rho, Vrad, RhoStar, dt));
  ActualiseGas_gpu (RhoInt, Rho);
  if (advect_vel) {
    VanLeerRadial_gpu (Vrad, RadMomP, dt); 
    VanLeerRadial_gpu (Vrad, RadMomM, dt);  
    VanLeerRadial_gpu (Vrad, ThetaMomP, dt);   
    VanLeerRadial_gpu (Vrad, ThetaMomM, dt);
  }
  if (Energy != NULL)
    VanLeerRadial_gpu (Vrad, Energy, dt);
  
  VanLeerRadial_gpu (Vrad, Rho, dt, true); /* MUST be the last line */

  if (DustSize != NULL)
    VanLeerRadialDustSize_gpu (Vrad, DustSize, dt);
}

void QuantitiesAdvection (PolarGrid * Rho, PolarGrid *Vtheta, PolarGrid *Energy, PolarGrid *DustSize, double dt, bool advect_vel) {
  ComputeStarTheta_gpu (Rho, Vtheta, RhoStar, dt);
  ActualiseGas_gpu (RhoInt, Rho);
  if (advect_vel) {
    VanLeerTheta_gpu (Vtheta, RadMomP, dt);
    VanLeerTheta_gpu (Vtheta, RadMomM, dt);
    VanLeerTheta_gpu (Vtheta, ThetaMomP, dt);
    VanLeerTheta_gpu (Vtheta, ThetaMomM, dt);
  }
  if (Energy != NULL)
    VanLeerTheta_gpu (Vtheta, Energy, dt);
  VanLeerTheta_gpu (Vtheta, Rho, dt);         /* MUST be the last line */

  if (DustSize != NULL)
    VanLeerThetaDustSize_gpu (Vtheta, DustSize, dt);
}

void OneWindTheta (PolarGrid *Rho, PolarGrid *Vtheta, PolarGrid *Energy, PolarGrid *DustSize, double dt, bool advect_vel) {
  FARGO_SAFE (ComputeResiduals_gpu (Vtheta, dt, OmegaFrame));   /* Constant residual is in Vtheta from now on */
  QuantitiesAdvection (Rho, VthetaRes, Energy, DustSize, dt, advect_vel);
  if (FastTransport == YES) {                                   /* Useless in standard transport as Vtheta is zero */
    QuantitiesAdvection (Rho, Vtheta, Energy, DustSize, dt, advect_vel); /* Uniform Transport here */
    if (advect_vel) {
      FARGO_SAFE (AdvectSHIFT_gpu (RadMomP, Nshift));
      FARGO_SAFE (AdvectSHIFT_gpu (RadMomM, Nshift));
      FARGO_SAFE (AdvectSHIFT_gpu (ThetaMomP, Nshift));
      FARGO_SAFE (AdvectSHIFT_gpu (ThetaMomM, Nshift));
    }
    if (Energy != NULL)
      FARGO_SAFE (AdvectSHIFT_gpu (Energy, Nshift));
    FARGO_SAFE (AdvectSHIFT_gpu (Rho, Nshift));
    if (DustSize != NULL)
      FARGO_SAFE (AdvectSHIFT_gpu (DustSize, Nshift));
  }
}

void Transport (PolarGrid *Rho, PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Energy, PolarGrid *DustSize, double dt, bool advect_vel) {
  // Compute momentum and angular monetum 
  // updated: RadMomP, RadMomM, ThetaMomP, ThetaMomM
  if (advect_vel) {
    FARGO_SAFE (ComputeLRMomenta_gpu (Rho, Vrad, Vtheta));

    /* We need to keep track of Rho, Vtheta, ThetaMomP, ThetaMomM at this stage */
    /* We copy them into arrays not used in the transport step */
    ActualiseGas_gpu (VradNew,   Rho);
    ActualiseGas_gpu (VthetaNew, Vtheta);
    ActualiseGas_gpu (VradInt,   ThetaMomP);
    ActualiseGas_gpu (VthetaInt, ThetaMomM);
  }
  
  OneWindRad (Rho, Vrad, Energy, DustSize, dt, advect_vel);
  OneWindTheta (Rho, Vtheta, Energy, DustSize, dt, advect_vel);

  /* We recover the value of Vtheta that we stored above in VthetaNew
     and calculate velocities from monenta */
  if (advect_vel) {
    ActualiseGas_gpu (Vtheta, VthetaNew);
    FARGO_SAFE (ComputeVelocities_gpu (Rho, Vrad, Vtheta));
  }
}
//-----------------------------------------------------------------------------------




