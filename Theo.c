/** \file Theo.c

A few functions that manipulate the surface density profile.

*/

#include "fargo.h"

// surface mass density
//--------------------------------------------------------------------------------------------
double Sigma(double r) {
  // This is *not* a steady state
  // profile, if a cavity is defined. It first needs 
  // to relax towards steady state, on a viscous time scale 
  double cavity=1.0;
  if (r < CAVITYRADIUS) 
    cavity = 1.0/CAVITYRATIO;
  
  double sigma = cavity * ScalingFactor * SIGMA0 * pow(r,-SIGMASLOPE);
  
  // exponential inner cut-off
  if (SIGMACUTOFFRADIN > 0)
    sigma /= 1+exp(-(r-SIGMACUTOFFRADIN)/(0.1*SIGMACUTOFFRADIN));

  // exponential outer cut-off
  if (SIGMACUTOFFRADOUT > 0)
    sigma /= 1+exp((r-SIGMACUTOFFRADOUT)/(0.1*SIGMACUTOFFRADOUT));

  // check density floor
  if (sigma > DENSITYFLOOR)
    return sigma;
  else
    return DENSITYFLOOR;
}

void FillSigma() {
  for (int i = 0; i < NRAD; i++) {
    SigmaMed[i] = Sigma(Rmed[i]);
    SigmaInf[i] = Sigma(Rinf[i]);
  }
  for (int i=0; i< NRAD; i++)
    for (int j=0; j< NSEC; j++)
      gas_density->Field[i*NSEC+j] = SigmaMed[i];
}
//--------------------------------------------------------------------------------------------

// velocity components
//--------------------------------------------------------------------------------------------
double VelRad (double r) {
  double vel;
  
  //
  //// alpha-viscosity
  //if (ViscosityAlpha) {
  //  if (SIGMACUTOFFRAD==0)
  //    vel = -3.0*viscosity/r*(-SIGMASLOPE+2.0*FLARINGINDEX+1.0);
  //  else
  //    vel = -3.0*viscosity/r*(-SIGMASLOPE+2.0*FLARINGINDEX+1.0+(SIGMASLOPE-2)*pow(r/SIGMACUTOFFRAD, 2-SIGMASLOPE));
  //}
  //   
  //// constant viscosity 
  //else {
  //  if (SIGMACUTOFFRAD==0)
  //    vel = -3.0*viscosity/r*(-SIGMASLOPE+.5);
  //  else  
  //    vel = -3.0*viscosity/r*(-SIGMASLOPE+0.5+(SIGMASLOPE-2)*pow(r/SIGMACUTOFFRAD, 2-SIGMASLOPE));
  //}
  // vrad = (3/r)d(r^1/2 d(nu*sigma*r^1/2)/dr)/dr
  if (ZeroVelrad)
    vel = 0.0;
  else {
    double viscosity = FViscosity (r);
    if (ViscosityAlpha) {
      vel = (3.0*(-1.0+SIGMASLOPE-2.0*FLARINGINDEX)/r)*viscosity;
    }
    else {
      vel = -3.0*viscosity/r*(-SIGMASLOPE+0.5);
    } 
  }
  return vel;
}

double VelTheta (double r, double sgacc) {
  double omega = sqrt(G*1.0/r/r/r);
  double vel;

  if (SelfGravity) {
    if (SIGMACUTOFFRADOUT==0)
      vel = r*sqrt(omega*omega*(1.0-pow(ASPECTRATIO,2.0) * pow(r,2.0*FLARINGINDEX)* (1.0+SIGMASLOPE-2.0*FLARINGINDEX)) - sgacc/r);
    else
      vel = r*sqrt(omega*omega*(1.0+pow(ASPECTRATIO,2.0)*(SIGMASLOPE-2)*pow(r,2+2*FLARINGINDEX-SIGMASLOPE)*pow(SIGMACUTOFFRADOUT,SIGMASLOPE-2)- pow(ASPECTRATIO,2.0)*pow(r,2*FLARINGINDEX)*(1+SIGMASLOPE-2*FLARINGINDEX)) - sgacc/r);
  }
  else {
    if (SIGMACUTOFFRADOUT==0)
      vel = omega * r * sqrt(1.0-pow(ASPECTRATIO,2.0) * pow(r,2.0*FLARINGINDEX) * (1.0+SIGMASLOPE-2.0*FLARINGINDEX));
    else
      vel = omega * r * sqrt(1.0+pow(ASPECTRATIO,2.0)*(SIGMASLOPE-2)*pow(r,2+2*FLARINGINDEX-SIGMASLOPE)*pow(SIGMACUTOFFRADOUT,SIGMASLOPE-2)- pow(ASPECTRATIO,2.0)*pow(r,2*FLARINGINDEX)*(1+SIGMASLOPE-2*FLARINGINDEX));
  }
  vel = omega * r * sqrt(1.0-pow(ASPECTRATIO,2.0) * pow(r,2.0*FLARINGINDEX) * (1.0+SIGMASLOPE-2.0*FLARINGINDEX));
  
  return vel;
}

void FillVelocites() {
  for (int i = 0; i < NRAD; i++) {
    // radial velocity
    double velrad = IMPOSEDDISKDRIFT*SIGMA0/SigmaInf[i]/Radii[i];
    GasVelRadMed[i] = velrad + VelRad(Rmed[i]);

    // azimuthal velocity
    double sgacc = 0.0;
    if (SelfGravity) {
      if (i == NRAD)
        sgacc = SGAcc->Field[(NRAD-1)*NSEC];
      else
        sgacc = SGAcc->Field[i*NSEC];
    }
    GasVelThetaMed[i] = VelTheta(Rmed[i], sgacc) - DInvSqrtRmed[i];
  }

  for (int i=0; i< NRAD; i++) {
    for (int j=0; j< NSEC; j++) {
      if (i >= 0)
        gas_v_rad->Field[i*NSEC+j] = GasVelRadMed[i];
      else
        gas_v_rad->Field[i*NSEC+j] = 0.0;
      gas_v_theta->Field[i*NSEC+j] = GasVelThetaMed[i];
    }
  }
}
//--------------------------------------------------------------------------------------------


// sound speed
//--------------------------------------------------------------------------------------------
double SoundSpeedAdiabatic (double r, double energy, double dens) {
  double cs = sqrt( ADIABATICINDEX*(ADIABATICINDEX-1.0)*energy/dens);
  return cs;
}

double SoundSpeedLocIso (double r) {
  double cs = AspectRatio(r) * sqrt(G*1.0/r) * pow(r, FLARINGINDEX);
  return cs;
}


void FillSoundSpeed () {
  for (int i = 0; i < NRAD; i++) {
    if (Adiabatic)
      SOUNDSPEED[i] = SoundSpeedAdiabatic(Rmed[i], EnergyMed[i], SigmaMed[i]);///sqrt(ADIABATICINDEX);
    else
      SOUNDSPEED[i] = SoundSpeedLocIso(Rmed[i]);      

    CS2[i] = SOUNDSPEED[i]*SOUNDSPEED[i];
    GLOBAL_SOUNDSPEED[i] = SOUNDSPEED[i];
  }
}
//--------------------------------------------------------------------------------------------


// energy
//--------------------------------------------------------------------------------------------
double Energy(double r) {
  double energy0 = 0.0;
  if (ADIABATICINDEX <= 1.0) {
    printf ("The adiabatic index must be larger than 1 to initialize the gas internal energy.\n");
    exit (1);
  }
  else
    //energy0 = R_SPEC/MU/(ADIABATICINDEX-1.0)*SIGMA0*pow(ASPECTRATIO,2.0)*pow(r,-SIGMASLOPE-1.0+2.0*FLARINGINDEX);
    // e0 = cs_ad^2*sigma/(gamma*(gamma-1)), csad=gamma^1/2 * cs_ad
    energy0 = R_SPEC/MU/(ADIABATICINDEX-1.0) * Sigma(r) * ASPECTRATIO * ASPECTRATIO * pow(r, -1.0 + 2.0*FLARINGINDEX);
  return energy0;
}

void FillEnergy() {
  for (int i = 0; i < NRAD; i++) {
    EnergyMed[i] = Energy(Rmed[i]);
  }
  for (int i=0; i< NRAD; i++)
    for (int j=0; j< NSEC; j++)
      gas_energy->Field[i*NSEC+j] = EnergyMed[i];
}
//--------------------------------------------------------------------------------------------


// cooling time
//--------------------------------------------------------------------------------------------
double CoolingTime(double r) {
  //double ct0 = COOLINGTIME0*pow(r,2.0+2.0*FLARINGINDEX);
  double ct0 = COOLINGTIME0*pow(r, 3./2.);
  return ct0;
}

void FillCoolingTime() {
  for (int i = 0; i < NRAD; i++)
    CoolingTimeMed[i] = CoolingTime(Rmed[i]);
}
//--------------------------------------------------------------------------------------------


// viscous heating term
//--------------------------------------------------------------------------------------------
double Qplusinit(double r) {
  //qp0 = 2.25*FViscosity(r)*SIGMA0*pow(r,-SIGMASLOPE-3.0);
  double qp0 = 2.25*FViscosity(r)*Sigma(r)*pow(r,-3.0);
  return qp0;
}

void FillQplus() {
  for (int i = 0; i < NRAD; i++)
    QplusMed[i] = Qplusinit(Rmed[i]);
}
//--------------------------------------------------------------------------------------------


// viscosity
//--------------------------------------------------------------------------------------------
double FViscosity (double rad) {
  double viscosity = VISCOSITY;
  int i = 0;
  if (ViscosityAlpha) {
    while (GlobalRmed[i] < rad)
      i++;
      //viscosity = ALPHAVISCOSITY*GLOBAL_SOUNDSPEED[i]*GLOBAL_SOUNDSPEED[i]*pow(rad, 1.5);
    viscosity = SOUNDSPEED[i]*SOUNDSPEED[i]*pow(rad, 1.5);

    // get alpha value
    viscosity *=  AlphaValue (rad);
  }

  double rmin = CAVITYRADIUS-CAVITYWIDTH*ASPECTRATIO;
  double rmax = CAVITYRADIUS+CAVITYWIDTH*ASPECTRATIO;
  double scale = 1.0+(PhysicalTime-PhysicalTimeInitial)*LAMBDADOUBLING;
  rmin *= scale;
  rmax *= scale;
  if (rad < rmin) viscosity *= CAVITYRATIO;
  if ((rad >= rmin) && (rad <= rmax)) {
    viscosity *= exp((rmax-rad)/(rmax-rmin)*log(CAVITYRATIO));
  }
   
  return viscosity;
}

// static viscosity transition makeing with spatially varying alpha value
// applied only after mass tapLering
double AlphaValue (double rad) {
  double alpha = ALPHAVISCOSITY;

  int i = 0;
  if (DeadZone) {// && MassTaper >= 1.0) {
    while (GlobalRmed[i] < rad) 
      i++;
    alpha *= (1.0-0.5*(1.0-DEADZONEALPHA/ALPHAVISCOSITY)*(tanh((rad-DEADZONERIN)/DEADZONEDELTARIN)-tanh((rad-DEADZONEROUT)/DEADZONEDELTAROUT)));
  }      
  return alpha;
}
  
double AspectRatio (double rad) {
  double aspectratio;//, rmin, rmax, scale;
  aspectratio = ASPECTRATIO;

  // snowline
  //if (SNOWLINETEMPRED != 0) {
  //  aspectratio *= 1.0-0.5*(1.0-SNOWLINETEMPRED)*(tanh((rad-SNOWLINER)/SNOWLINEDELTAR)+1.0);
  //}
  /*
  rmin = TRANSITIONRADIUS-TRANSITIONWIDTH*ASPECTRATIO;
  rmax = TRANSITIONRADIUS+TRANSITIONWIDTH*ASPECTRATIO;
  scale = 1.0+(PhysicalTime-PhysicalTimeInitial)*LAMBDADOUBLING;
  rmin *= scale;
  rmax *= scale;
  if (rad < rmin) aspectratio *= TRANSITIONRATIO;
  if ((rad >= rmin) && (rad <= rmax)) {
    aspectratio *= exp((rmax-rad)/(rmax-rmin)*log(TRANSITIONRATIO));
  }
  */
  return aspectratio;
}

void FillViscosity () {
   for (int i = 0; i < NRAD; i++) {
     viscoval[i] = FViscosity (Rmed[i]);
     alphaval[i] = AlphaValue (Rmed[i]);
   }
}
//--------------------------------------------------------------------------------------------


// dust velocity set by drag force
//--------------------------------------------------------------------------------------------
void DustDragVel (double s, double r, double sigma_g, double *vrad, double *vtheta) {
  double St;
  
  if (DustConstStokes) {
    St= s;

    const double eta = (1.0 + SIGMASLOPE) * pow(ASPECTRATIO, 2.0);
    *vrad = -eta / (St + 1.0 / St) * pow(r, -0.5);
    const double vkep = pow(r, -0.5);
    *vtheta = pow(1.0 - eta, 0.5) * pow(r, -0.5) - 0.5 * St * (*vrad) - vkep;
  }
  else {
    *vrad = 0.0;
    *vtheta = 0.0;
  }
}
//--------------------------------------------------------------------------------------------


  
  
  
void RefillSigma (PolarGrid *Surfdens) {
  int i, j, nr, ns, l;
  double *field;
  double moy;
  nr = Surfdens->Nrad;
  ns = Surfdens->Nsec;
  field = Surfdens->Field;
  for (i = 0; i < nr; i++) {
    moy = 0.0;
    for (j = 0; j < ns; j++) {
      l = j+i*ns;
      moy += field[l];
    }
    moy /= (double)ns;
    SigmaMed[i] = moy;
  }
  SigmaInf[0] = SigmaMed[0];
  for (i = 1; i < nr; i++) {
    SigmaInf[i] = (SigmaMed[i-1]*(Rmed[i]-Rinf[i])+\
      SigmaMed[i]*(Rinf[i]-Rmed[i-1]))/\
      (Rmed[i]-Rmed[i-1]);
  }
}
