//int CPU_Rank;
//int CPU_Number;
//bool CPU_Master;
int IMIN;
int IMAX;
int Zero_or_active;
int Max_or_active;
int One_or_active;
int MaxMO_or_active;
int GLOBALNRAD;


bool NoGasAccretion = NO;                                        // turning off gas accretion

char ParameterFile[1024];

//--------------------------------------------------------------------------------------------------------------------------------------------------------------
// double arrays
//--------------------------------------------------------------------------------------------------------------------------------------------------------------

double RadiiStuff[MAX1D*16];                                     // radii stuff contains 1D arrays of grid related vectors listed below 
double *Rinf, *Rsup, *Rmed, *Surf;                               // cell left interface, right interfa, cecenter, and surface area
double *DRmed, *DInvSqrtRmed, *DRadii;                           // cell center, 1/sqrt(cell center), left cell interface
double *InvRmed, *InvSurf, *InvDiffRmed, *viscoval, *alphaval;   // 
double *InvDiffRsup, *InvRinf, *Radii, GlobalRmed[MAX1D];        //
double *Energy0Med;
double DGlobalRmed[MAX1D];                                       //
double SigmaMed[MAX1D], SigmaInf[MAX1D], EnergyMed[MAX1D];       // initial surface mass density at cell center, left cell interface, and energy at cell center
double  GasVelRadMed[MAX1D], GasVelThetaMed[MAX1D];
double  DustVelRadMed[MAX1D], DustVelThetaMed[MAX1D];
double CoolingTimeMed[MAX1D], QplusMed[MAX1D];                   // cooling time (beta cooling) and QPlus for adiabatic gas
double ForceX[MAX1D];
double ForceY[MAX1D];
double *SOUNDSPEED, *CS2;                                        // initial soundspeed and square of sound speed
double GLOBAL_SOUNDSPEED[MAX1D];                                 // ???
double MassTaper;                                                // mass tapering value
double OmegaFrame;                                               // frame rotation angular speed
double PhysicalTime=0.0, PhysicalTimeInitial;                    // time of simulation, initial time when simulation was started
double  SGAccInnerEdge, SGAccOuterEdge;                          // sg
int    TimeStep = 0;                                             // number of steps calcualted
double TotDiskMass;
double *omega;

//--------------------------------------------------------------------------------------------------------------------------------------------------------------
// boolean switches
//--------------------------------------------------------------------------------------------------------------------------------------------------------------
bool    LogGrid;                                                 // logarithmic or arithmetic grid grid
bool   AdvecteLabel, FakeSequential, MonitorIntegral, OnlyInit;
bool	 GotoNextOutput;
bool   StoreSigma;
bool   ViscosityAlpha;
bool   RocheSmoothing;
bool    CentrifugalBalance;
bool    SloppyCFL;
bool    BaryCentric = NO;
bool    MonitorBC = NO, MonitorTorque = NO, MonitorDiskEcc = NO;  // monitoring disk bary center, planet torques, and disk eccentricity
bool    MonitorAccretion = NO;
int     ExcludeHill = 0;                                          // planetary toruqe Hill exclusion (0: none, 1 Gaussian, 2 Heaviside)
bool    SelfGravity;
bool    Indirect_Term = YES;                                      // inlcude indirect term caused by the disk (planetary ind. term is always taken into account)
bool    DustGrid;                                                 // simulate dust as pressureless fluid
bool    DustConstStokes;
bool    DustSelfGravity;                                          // calculate dust self-gravity
bool    DustFeedback = NO;
bool    FastTransport = YES;                                       // use FARGO algorithm
///MPI_Status fargostat;
bool    IsDisk = YES, Corotating = NO;
bool  Profiling = NO, TimeInfo = YES;
bool    ZeroVelrad = NO;
bool    OverridesOutputdir;
char    NewOutputdir[1024];

double  MinSemiMajorPlanet = 0;
bool    TerminateDuetoPlanet = false;
double  GasDiskMassInner, GasDiskMassOuter;
double  GasDiskBC_x, GasDiskBC_y;
double  GasBC_x, GasBC_y;
double  DustDiskMassInner, DustDiskMassOuter;
double  DustDiskBC_x, DustDiskBC_y;
double  DustBC_x, DustBC_y;
double  StarPlanetBC_x, StarPlanetBC_y;
double  DiskEcc_SurfNorm_Inner, DiskEcc_SurfNorm_Outer, DiskEcc_MassNorm_Inner, DiskEcc_MassNorm_Outer;
double *DustSizeBin, *DustMassBin;
double *Pl0AccretedMass;
int     DustBinNum;
bool    Write_Density = YES, Write_Velocity = YES;                            // density and velociy snapsots
bool    Write_Energy = YES, Write_Temperature = NO, Write_DiskHeight = NO;    // energy, temperature and disk height snappshots for adiabatic disk
bool    Write_Potential = NO;                                                 // gravitational potental for self-gravitating disk
bool    Write_SoundSpeed = NO;                                                // sound speed

//--------------------------------------------------------------------------------------------------------------------------------------------------------------
//  switches for boundary conditions
//--------------------------------------------------------------------------------------------------------------------------------------------------------------
bool    OpenInner, OpenOuter;                                      // open
bool    DampingInner, DampingOuter;                                // damping (only velocities are damped)
bool    StrongDampingInner, StrongDampingOuter;                    // strong damping (velocities and density are also damped)
bool    NonReflectingInner, NonReflectingOuter;                    // non-reflecting
bool    RigidWallInner, RigidWallOuter;                            // rigid inner wall (like reflecting)
bool    ViscOutflowInner, ViscOutflowOuter;                        // viscous outflow
bool    ClosedInner, ClosedOuter;                                  // closed (no inflow, no outflow)
bool    EmptyInner, EmptyOuter;
bool    OuterSourceMass = NO;
bool    GuidingCenter = NO;


bool    AdaptiveViscosity = NO;
bool    DeadZone = NO;
bool    Adiabatic = NO;                                            // solve energy equation for adiabatic gas EOS
bool    Cooling = NO;                                              // calculate cooling
bool    ViscHeating = NO;
bool    ModelNebula = NO;
double  ScalingFactor = 1.0;
bool    Restart = NO, AutoRestart = NO;
int     begin_i = 0, NbRestart = 0;
double  LostMass, AccRate, StellarAccRate;
bool    CreatingMovieOnly = NO;
bool    DustGrowth = NO;
//--------------------------------------------------------------------------------------------------------------------------------------------------------------
// polar grids used by GPU
//--------------------------------------------------------------------------------------------------------------------------------------------------------------
PolarGrid *CellAbscissa, *CellOrdinate;                            // cell 
PolarGrid *RhoStar, *RhoInt, *Potential;                           // 
PolarGrid *gas_density, *gas_v_rad, *gas_v_theta, *gas_label;      // gas density, radial and azimuthal velocity, and gas label
PolarGrid *VradInt, *VthetaInt;
PolarGrid *EnergyInt;
PolarGrid *VradNew, *VthetaNew;              // intermediate and new gas velocities
PolarGrid *gas_energy, *SoundSpeed, *Viscosity;                 // adiabatic 
PolarGrid *TauRR, *TauRP, *TauPP;
PolarGrid *Work, *TemperInt, *QRStar, *Elongations, *Qder;                   // ?
PolarGrid *RadMomP, *RadMomM, *ThetaMomP, *ThetaMomM, *ExtLabel, *VthetaRes; // ?
PolarGrid **dust_density, **dust_v_rad, **dust_v_theta;                      // dust densities, radial and azimuthal velocities for each dust bin
PolarGrid **VradDustInt, **VthetaDustInt;
PolarGrid  *disk_ecc;                                                        // disk eccentricity
PolarGrid *FTD;                                                              // field to display by OpenGl
PolarGrid *Buffer;
PolarGrid *SGAcc;
PolarGrid *dust_size, *dust_growth_rate;

PolarGrid *tmp1, *tmp2;
PolarGrid *tmp3, *tmp4;
PolarGrid *DeltaT;
PolarGrid *WorkShift;
PolarGrid *myWork;
//--------------------------------------------------------------------------------------------------------------------------------------------------------------
// planetary system
//--------------------------------------------------------------------------------------------------------------------------------------------------------------
PlanetarySystem *sys;                                                        // structure used for planetary system
Pair DiskOnPrimaryAcceleration;
Pair IndirectTerm;

//--------------------------------------------------------------------------------------------------------------------------------------------------------------
// OpenGL related switches
//--------------------------------------------------------------------------------------------------------------------------------------------------------------
double  RefreshRate = 50.0;
bool    Window = NO;
bool       Paused = NO;

int     IterCount, StillWriteOneOutput;

TimeProcess      t_Hydro;
bool    verbose = NO;
