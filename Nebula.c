/** \file Nebula.c

A few functions that add a spherical nebual
Ir requre adiabatic model

*/

#include "fargo.h"

double NebulaSigma(double x, double y, double dens0, double sigma_ext, double c_x, double c_y) {
  double dens = dens0 * exp(-pow((x-c_x)/sigma_ext,2.0)) * exp(-pow((y-c_y)/sigma_ext, 2.0));
  return dens;
}

void FillNebulaSigma () {
  double dphi = 2.0 * M_PI/(double) NSEC;
  double dens0 = NEBULAMASS/(NEBULAEXT * NEBULAEXT * M_PI);
  for (int i=0; i< NRAD; i++) {
    for (int j=0; j< NSEC; j++) {
      int m = i*NSEC+j;
      double Phimed = j * dphi + dphi/2.0;
      double x = Rmed[i] * cos(Phimed);
      double y = Rmed[i] * sin(Phimed);
      if (j == 0)
        SigmaMed[i] = SIGMA0*pow(Rmed[i], -SIGMASLOPE);
      gas_density->Field[m] = 0*SigmaMed[i];
      //if (pow(x-NEBULAPOSX, 2.0)+pow(y-NEBULAPOSY, 2.0) < pow(4*NEBULAEXT, 2.0))
        gas_density->Field[m] += NebulaSigma (x, y, dens0, NEBULAEXT, NEBULAPOSX, NEBULAPOSY);
    }
  }
}

double NebulaVelRad (double r, double phi, double x, double y, double sigma_ext, double c_x, double c_y, double vel_x, double vel_y, double vel_unit) {
  double vel = cos(phi) * vel_x/vel_unit+sin(phi) * vel_y/vel_unit;
  //vel *= exp(-pow((x-c_x)/sigma_ext,2.0)) * exp(-pow((y-c_y)/sigma_ext, 2.0));
  return vel;
}

double NebulaVelTheta (double r, double phi, double x, double y, double sigma_ext, double c_x, double c_y, double vel_x, double vel_y, double vel_unit) {
  double vel = - sin(phi) * vel_x/vel_unit + cos(phi) * vel_y/vel_unit;
  //vel *= exp(-pow((x-c_x)/sigma_ext,2.0)) * exp(-pow((y-c_y)/sigma_ext, 2.0));
  return vel;
}

#define NEBULAVELUNIT  1.49600e13/(365.24*24.*60.*60.)*(2.*M_PI)/1e5
void FillNebulaVelocities () {

  double dphi = 2.0 * M_PI/(double) NSEC;
  for (int i=0; i< NRAD; i++) {
    for (int j=0; j< NSEC; j++) {
      double Phimed = j * dphi + dphi/2.0;
      double x = Rmed[i] * cos(Phimed);
      double y = Rmed[i] * sin(Phimed);
      double omega = sqrt(1.0/pow(Rmed[i],3.0));
      int m = i*NSEC+j;
      gas_v_rad->Field[m] = 0.0;
      //gas_v_theta->Field[m] = omega * Rmed[i] * sqrt(1.0-pow(ASPECTRATIO,2.0) * pow(Rmed[i],2.0*FLARINGINDEX) * (1.0+SIGMASLOPE-2.0*FLARINGINDEX));
      //gas_v_theta->Field[m] -= DInvSqrtRmed[i];
      if (j == 0) {
        GasVelRadMed[i] = gas_v_rad->Field[m];
        GasVelThetaMed[i] = gas_v_theta->Field[m];
      }
      
      //if (pow(x-NEBULAPOSX, 2.0)+pow(y-NEBULAPOSY, 2.0) < pow(4*NEBULAEXT, 2.0)) 
      {
        double phi = j * dphi + dphi/2.0;
        gas_v_rad->Field[m]   = NebulaVelRad   (Rmed[i], phi, x, y, 2*NEBULAEXT, NEBULAPOSX, NEBULAPOSY, NEBULAVELX, NEBULAVELY, NEBULAVELUNIT);
        gas_v_theta->Field[m] = NebulaVelTheta (Rmed[i], phi, x, y, 2*NEBULAEXT, NEBULAPOSX, NEBULAPOSY, NEBULAVELX, NEBULAVELY, NEBULAVELUNIT)-DInvSqrtRmed[i];
      }
    }
  }
}


double NebulaEnergy(double r, double sigma) {
  double energy0 = 1.0 / (ADIABATICINDEX-1.0) * sigma;
  return energy0;
}



void FillNebulaEnergy() {
  double dphi = 2.0 * M_PI/(double) NSEC;
  double dens0 = NEBULAMASS/(NEBULAEXT * NEBULAEXT * M_PI);
  for (int i=0; i< NRAD; i++) {
    for (int j=0; j< NSEC; j++) {
      double Phimed = j * dphi + dphi/2.0;
      double x = Rmed[i] * cos(Phimed);
      double y = Rmed[i] * sin(Phimed);
      int m = i*NSEC+j;
      gas_energy->Field[m] = 0*1.0/(ADIABATICINDEX-1.0) * SIGMA0 * pow(ASPECTRATIO,2.0) * pow(Rmed[i],-SIGMASLOPE-1.0+2.0*FLARINGINDEX);
      if (j ==0)
        EnergyMed[i] = gas_energy->Field[m];
      //if (pow(x-NEBULAPOSX, 2.0)+pow(y-NEBULAPOSY, 2.0) < pow(4*NEBULAEXT, 2.0)) {
        gas_energy->Field[m] += NEBULATEMP * NebulaEnergy(Rmed[i], pow(gas_density->Field[m],2)/dens0);
        //}
    }
  }
}

// cooling time
//--------------------------------------------------------------------------------------------
double NebulaCoolingTime(double r) {
  double ct0 = COOLINGTIME0*pow(r,2.0+2.0*FLARINGINDEX);
  return ct0;
}

void FillNebulaCoolingTime() {
  for (int i = 0; i < NRAD; i++)
    CoolingTimeMed[i] = NebulaCoolingTime(Rmed[i]);
}
//--------------------------------------------------------------------------------------------
