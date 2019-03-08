/** \file proto.h

Declaration of all the functions of the FARGO code

*/

#ifndef __CUDA
void masterprint (const char *mytemplate, ...);
void mastererr (const char *mytemplate, ...);
#endif
#ifdef __cplusplus
extern "C" {
#endif
  void AzimuthalAverage (PolarGrid *array, double *res);
  void prs_exit (int numb);
  void SelectDevice(int number);
  //void Disclaimer ();
  void Loop ();
  void StartMainLoop ();
  void InitDisplay (int *argc, char **argv);
  void DisplayLoadDensity ();
  void prs_end (PlanetarySystem *sys, int numb);
  void prs_error(const char *string);
  void message (char *msg);
  PolarGrid    *CreatePolarGrid(int Nr, int Ns, const char *name);
  void MultiplyPolarGridbyConstant (PolarGrid *arraysrc, double constant);
  void DumpSources (int argc, char *argv[]);
  Force ComputeForce ();
  void UpdateLog (PlanetarySystem *psys, PolarGrid *Rho, int outputnb, double time);
  void ReadfromFile (PolarGrid *array, const char *fileprefix, int filenumber);
  void InitLabel (PolarGrid *array);
  void Initialization (PolarGrid *gas_density, PolarGrid *gas_v_rad, PolarGrid *gas_v_theta, PolarGrid *gas_energy, 
                       PolarGrid *gas_label,
                       PolarGrid **dust_density, PolarGrid **dust_v_rad, PolarGrid **dust_v_theta);
  void var(char *name, void *ptr, int type, int necessary, char *deflt);
  void ReadVariables(char *filename);
  void PrintUsage (char *execname);
  double TellNbOrbits (double time);
  double TellNbOutputs (double time);
  void TellEverything ();
  void GiveTimeInfo (int number);
  void InitSpecificTime (bool profiling, TimeProcess *process_name, char *title);
  void GiveSpecificTime (bool profiling, TimeProcess process_name);
  void EmptyPlanetSystemFile (PlanetarySystem *sys);
  void WritePlanetFile (int TimeStep, int n);
  void WritePlanetSystemFile (PlanetarySystem *sys, int t);
  void WriteBigPlanetFile (double Time, int n);
  void WriteBigPlanetSystemFile (PlanetarySystem *sys, double t);
  double GetfromPlanetFile (int TimeStep, int column, int n);
  void RestartPlanetarySystem (int timestep, PlanetarySystem *sys);
  void WriteDiskPolar(PolarGrid *array, int number, char *ext_name = NULL);
  void WriteDim ();
  void SendOutput (int index);
  void ComputeIndirectTerm ();
  void AdvanceSystemFromDisk (PolarGrid *Rho, PlanetarySystem *sys, double dt);
  void AdvanceSystemFromDiskRZS (PolarGrid *Rho, PlanetarySystem *sys, double dt);
  void AdvanceSystemRK5 (PlanetarySystem *sys, double dt);
  void AdvanceSystemRK5RZS (PlanetarySystem *sys, double dt);
  void SolveOrbits (PlanetarySystem *sys);
  double ConstructSequence (double *u, double *v, int n);
  void InitGas (PolarGrid *Rho, PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Energy, PolarGrid *SGAcc);
  void InitGasEnergy (PolarGrid *Energy);
  void InitDust (double dust_mass, PolarGrid *GasRho, PolarGrid *DustRho, PolarGrid *DustVrad, PolarGrid *DustVtheta);
  void AccreteOntoPlanets (PolarGrid *Rho, PolarGrid *Vrad, PolarGrid *Vtheta, double dt, PlanetarySystem *sys);
  //void EvolveCoreAccretions (double t, PlanetarySystem *sys, CoreAccretion *cacc, PolarGrid *Rho);
  void FindOrbitalElements (double x, double y, double vx, double vy, double m, int n);
  int FindNumberOfPlanets (char *filename);
  PlanetarySystem *AllocPlanetSystem (int nb);
  void FreePlanetary (PlanetarySystem *sys);
  PlanetarySystem *InitPlanetarySystem (char *filename);
  PlanetarySystem *InitPlanetarySystemBin(char *filename);
  void ListPlanets (PlanetarySystem *sys);
  double GetPsysInfo (PlanetarySystem *sys, int action);
  void RotatePsys (PlanetarySystem *sys, double angle);
  int FindNumberOfCores (char *filename);
  //CoreAccretion *AllocCoreAccretion (int nb);
  //CoreAccretion *InitCoreAccretion (char *filename, int sys_nb, double *sys_mass);
  //void ListCoreAccretion (CoreAccretion *cacc);
  void DerivMotionRK5 (double *q_init, double *masses, double *deriv, int n, double dt, bool *feelothers);
  void DerivMotionRK5RZS (PolarGrid *Rho, double *q_init, double *masses, double *deriv, int n, double dt, bool *feelothers);
  void TranslatePlanetRK5 (double *qold, double c1, double c2, double c3, double c4, double c5, double *qnew, int n);
  void RungeKunta (double *q0, double dt, double *masses, double *q1, int n, bool *feelothers);
  void RungeKuntaRZS (PolarGrid *Rho, double *q0, double dt, double *masses, double *q1, int n, bool *feelothers);
  double GasTotalMass (PolarGrid *array);
  double GasMomentum (PolarGrid *Density, PolarGrid *Vtheta);
  void DivisePolarGrid ();
  void InitComputeAccel ();
  Pair ComputeAccel (PolarGrid *Rho, double x, double y, double rsmoothing, double mass);
  void OpenBoundary ();
  void NonReflectingBoundary ();
//  void ApplyOuterSourceMass ();
  void ApplyOuterSourceMass_gpu (PolarGrid *Vrad, PolarGrid *Rho, double dustmass);
  
  void ApplyBoundaryCondition (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, PolarGrid *Energy, double dt);
  void ApplyBoundaryConditionDust (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, int DustBin, double dt);
  void CorrectVtheta ();
  bool DetectCrash (PolarGrid *Field, double FloorValue);
  void FillPolar1DArrays ();
  void InitEuler (PolarGrid  *Rho, PolarGrid *Vrad,  PolarGrid *Vtheta, PolarGrid *Energy,
                  PolarGrid **DustRho, PolarGrid **DustVrad, PolarGrid **DustVtheta);
  double min2 (double a,double b);
  double max2 (double a,double b);
  void ActualiseGas ();
  void ActualiseGas_gpu (PolarGrid *a, PolarGrid *b);
//  void AlgoGas ();
  void SubStep4_gpu (PolarGrid *RhoDustGr, PolarGrid *RhoDustSm, PolarGrid *DustSizeGr, PolarGrid *DustGrowthRate, PolarGrid *RhoGas, PolarGrid *Energy, double dt);

  void SubStep1_gpu (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, PolarGrid *Energy, double dt,
                     PolarGrid *Vrad_ret, PolarGrid *Vtheta_ret);
  void SubStep1Dust_gpu (PolarGrid *VradGas, PolarGrid *VthetaGas, PolarGrid *RhoGas, PolarGrid *Energy, 
                         PolarGrid *VradDust, PolarGrid *VthetaDust, PolarGrid *RhoDust,
                         double dust_size, double dt,
                         PolarGrid *VradDust_ret, PolarGrid *VthetaDust_ret, PolarGrid *RhoDust_ret);
  void SubStep1GasDust_gpu (PolarGrid *VradGas, PolarGrid *VthetaGas, PolarGrid *RhoGas, PolarGrid *Energy, 
                            PolarGrid *VradDust, PolarGrid *VthetaDust, PolarGrid *RhoDust,
                            double dust_size, double dt,
                            PolarGrid *VradGas_ret, PolarGrid *VthetaGas_ret,
                            PolarGrid *VradDust_ret, PolarGrid *VthetaDust_ret, PolarGrid *RhoDust_ret);
  void SubStep1GasDustMDGM_gpu (PolarGrid *VradGas, PolarGrid *VthetaGas, PolarGrid *RhoGas, PolarGrid *Energy, 
                                PolarGrid **VradDust, PolarGrid **VthetaDust, PolarGrid **RhoDust,
                                PolarGrid *DustSizeGr, PolarGrid *DustGrowthRate,
                                double dt,
                                PolarGrid *VradGas_ret, PolarGrid *VthetaGas_ret,
                                PolarGrid **VradDust_ret, PolarGrid **VthetaDust_ret, PolarGrid **RhoDust_ret);


  void SubStep2_gpu (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, PolarGrid *Energy, double dt,
                     PolarGrid *Vrad_ret, PolarGrid *Vtheta_ret, PolarGrid *Energy_ret);
  void SubStep2Dust_gpu (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, double dt,
                         PolarGrid *Vrad_ret, PolarGrid *Vtheta_ret);
  void SubStep3_gpu (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, PolarGrid *Energy, double dt,
                     PolarGrid *Energy_ret);
                     
  void ViscousTerms_gpu (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, PolarGrid *Energy, double dt,
                         PolarGrid *Vrad_ret, PolarGrid *Vtheta_ret);
  void ViscousTermsDust_gpu (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, double dt,
                             PolarGrid *Vrad_ret, PolarGrid *Vtheta_ret);
  void CorrectDustSize_gpu (PolarGrid *VradDustGr, PolarGrid *VthetaDustGr, PolarGrid *DustSizeOld, PolarGrid *DustSize, double dt);
                     
  int ConditionCFL ();
  double Sigma(double r);
  void FillSigma();
  void FillVelocites();
  void FillViscosity (); 
  void FillSoundSpeed ();
  void FillEnergy();
  void FillCoolingTime();
  void FillQplus();
  void RefillSigma (PolarGrid *Surfdens);
  void Transport (PolarGrid *Rho, PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Energy, PolarGrid *Label, double dt, bool advect_vel = true);
  void InitTransport ();
  void ComputeThetaElongations ();
  void ComputeAverageThetaVelocities ();
  void ComputeResiduals ();
  void ComputeConstantResidual ();
  void AdvectSHIFT ();
  void ComputeStarTheta (PolarGrid *Qbase, PolarGrid *Vtheta, PolarGrid *QStar, double dt);
  void InitViscosity ();
  void ViscousTerms ();
  void AllocateComm ();
  //void CommunicateBoundaries (PolarGrid *Density, PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Label);
  void handfpe();
  void setfpe ();
  void merge (int nb);
  void ReadPrevDim ();
  void CheckRebin (int nb);
  void SplitDomain ();
  void InitVariables();
  double FViscosity (double rad);
  double AspectRatio (double rad);
  Force ComputeForceStockholm ();
  void UpdateLogStockholm ();
  Pair MassInOut ();
  void StockholmBoundary ();
  void MakeDir (char *string);
  FILE *fopenp (char *string, char *mode);
  void D2H (PolarGrid *a);
  void H2D (PolarGrid *a);
  void SetRadiiStuff ();
  void DivisePolarGrid_gpu (PolarGrid *Num, PolarGrid *Denom, PolarGrid *Res);

  void ComputeLRMomenta_gpu (PolarGrid *Rho, PolarGrid *Vrad, PolarGrid *Vtheta);
  void ComputeVelocities_gpu (PolarGrid *Rho, PolarGrid *Vrad, PolarGrid *Vtheta);
  
  void VanLeerRadial_gpu (PolarGrid *Vrad, PolarGrid *Qbase, double dt, bool calc_lostmass = 0);
  void VanLeerRadialDustSize_gpu (PolarGrid *Vrad, PolarGrid *Qbase, double dt);
  void VanLeerTheta_gpu (PolarGrid *Vtheta, PolarGrid *Qbase, double dt);
  void VanLeerThetaDustSize_gpu (PolarGrid *Vtheta, PolarGrid *Qbase, double dt);
    
  void ComputeStarRad_gpu (PolarGrid *Qbase, PolarGrid *Vrad, PolarGrid *QStar, double dt);
  void ComputeStarTheta_gpu (PolarGrid *Qbase, PolarGrid *Vtheta, PolarGrid *QStar, double dt);
  void AdvectSHIFT_gpu (PolarGrid *array, int *Nshift);
  void ComputeResiduals_gpu (PolarGrid *Vtheta, double dt, double o);
  void H2D_All ();
  void D2H_All ();

  void VanLeerTheta_gpu_cu (PolarGrid *Vtheta, PolarGrid *Qbase, double dt);
  void VanLeerThetaDustSize_gpu_cu (PolarGrid *Vtheta, PolarGrid *Qbase, double dt);

  void VanLeerRadial_gpu_cu (PolarGrid *Vrad, PolarGrid *Qbase, double dt, bool calc_lostmass = false);
  void VanLeerRadialDustSize_gpu_cu (PolarGrid *Vrad, PolarGrid *Qbase, double dt);

  int ConditionCFL_gpu (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, PolarGrid *Energy, double deltaT);

  
  void CalcViscousTerms_gpu (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, PolarGrid *Energy, double dt);
  Force ComputeForce_gpu (PolarGrid *Rho, double x0, double y0, double smoothing, double mass, int exclude);
  void StockholmBoundary_gpu (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, PolarGrid *Energy, double dt, int where, double R_inf, double R_sup, bool Strong);
  void StockholmBoundaryDust_gpu (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, int DustBin, double dt, int where);
  void OpenBoundary_gpu (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, PolarGrid *Energy, int where);
  void OpenBoundaryDust_gpu (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, int DustBin, int where);
  void ClosedBoundary_gpu (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, int where);
  void TestBoundary_gpu (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, int where);
  void ViscOutflow_gpu (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, int where);
  void ReflectingBoundary_gpu (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, int where);
  void NonReflectingBoundary_gpu (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, int where);
  double GetGlobalIFrac (double r);
  void CorrectVtheta_gpu (PolarGrid *Vtheta, double domega);
  void FillForcesArrays_gpu (PlanetarySystem *sys);
  void FillForcesArrays (PlanetarySystem *sys);
  int CheckNansPolar (PolarGrid *g);
  void CheckNans (char *string);
  void PrintVideoRAMUsage ();
  
  // planetary accretion on gpu
  void AccreteOntoPlanets_gpu (PolarGrid *Rho, PolarGrid *Vrad, PolarGrid *Vtheta, double dt, PlanetarySystem *sys, double *AccretedMass, bool changePlanet);
  void WriteCurrentFrameNum (int index);
  int  ReadCurrentFrameNum ();
  void InitializationBC (PlanetarySystem *sys, PolarGrid *gas_density, PolarGrid *gas_v_rad, PolarGrid *gas_v_theta, PolarGrid *gas_label);
  void InitEulerBC (PlanetarySystem *sys, PolarGrid *Rho, PolarGrid *Vr, PolarGrid *Vt);
  void FindPlanetStarBC (PlanetarySystem *sys, double *xbc, double *ybc);
  void InitGasBC (PlanetarySystem *sysm, PolarGrid *Rho, PolarGrid *Vr, PolarGrid *Vt, PolarGrid *SGAcc);
  void CalcGasBC (PolarGrid *Rho, PlanetarySystem *sys);
  void CalcDustBC (PolarGrid **Rho, PlanetarySystem *sys);
  void WriteBigFiles (PlanetarySystem *sys, PolarGrid *Rho, PolarGrid **DustRho, double phys_time);
  void WriteTorques (PlanetarySystem *sys, PolarGrid *Rho, PolarGrid **DustRho, int time_step);
  void CalcDiskEcc (PolarGrid *Rho, PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *DiskEcc, PlanetarySystem *sys);
  double AlphaValue (double rad);
  void SumPolarGrid_gpu (int num, PolarGrid **dust_density, PolarGrid *gas_density, PolarGrid *Work);

  void CalcVortens_gpu (PolarGrid *Vortensity, PolarGrid *dens, PolarGrid *vrad, PolarGrid *vtheta);
  void CalcTemp_gpu (PolarGrid *temp, PolarGrid *energy, PolarGrid *dens);
  void CalcDiskHeight_gpu (PolarGrid *Rho, PolarGrid *Energy, PolarGrid *DiskHeight);
  void CalcSoundSpeed_gpu (PolarGrid *Rho, PolarGrid *Energy, PolarGrid *CS);
  void CalcDustGasMassRatio_gpu (PolarGrid *Rho, PolarGrid *DustRho, PolarGrid *DustGasMassRatio);
  void CalcDustSize_gpu (PolarGrid *DustSizeGr, PolarGrid *Rho, PolarGrid *DustSize);
  
  // some nebula related functions
  void FillNebulaSigma ();
  void FillNebulaVelocities ();
  void FillNebulaEnergy();
  void FillNebulaCoolingTime();
  void DustDragVel (double s, double r, double sigma_g, double *vrad, double *vtheta);

#ifdef FARGO_INTEGRATION
  void InitGasBiCubicInterpol (int nsec, int nrad, int upscaling_rad, int upscaling_azim);
  void GasBiCubicInterpol ();
  void InitDustBiCubicInterpol (int nsec, int nrad, int upscaling);
  void DustBiCubicInterpol ();

#endif
    
#ifdef __cplusplus
}
#endif

