/** \file global_ex.h

This file is created      
automatically during      
compilation from global.h. Do not edit. 
See perl script           
"varparser.pl" for details

\file global.h

Declares all global variables.
Used to construct automatically
the file global_ex.h. The file
global.h cannot contain any comment,
as it would not be parsed correctly
by varparser.pl
*/                          

extern int IMIN;
extern int IMAX;
extern int Zero_or_active;
extern int Max_or_active;
extern int One_or_active;
extern int MaxMO_or_active;
extern int GLOBALNRAD;
extern bool NoGasAccretion;                                        // turning off gas accretion
extern char ParameterFile[1024];
extern double RadiiStuff[MAX1D*16];                                     // radii stuff contains 1D arrays of grid related vectors listed below 
extern double *Rinf, *Rsup, *Rmed, *Surf;                               // cell left interface, right interfa, cecenter, and surface area
extern double *DRmed, *DInvSqrtRmed, *DRadii;                           // cell center, 1/sqrt(cell center), left cell interface
extern double *InvRmed, *InvSurf, *InvDiffRmed, *viscoval, *alphaval;   // 
extern double *InvDiffRsup, *InvRinf, *Radii, GlobalRmed[MAX1D];        //
extern double *Energy0Med;
extern double DGlobalRmed[MAX1D];                                       //
extern double SigmaMed[MAX1D], SigmaInf[MAX1D], EnergyMed[MAX1D];       // initial surface mass density at cell center, left cell interface, and energy at cell center
extern double  GasVelRadMed[MAX1D], GasVelThetaMed[MAX1D];
extern double  DustVelRadMed[MAX1D], DustVelThetaMed[MAX1D];
extern double CoolingTimeMed[MAX1D], QplusMed[MAX1D];                   // cooling time (beta cooling) and QPlus for adiabatic gas
extern double ForceX[MAX1D];
extern double ForceY[MAX1D];
extern double *SOUNDSPEED, *CS2;                                        // initial soundspeed and square of sound speed
extern double GLOBAL_SOUNDSPEED[MAX1D];                                 // ???
extern double MassTaper;                                                // mass tapering value
extern double OmegaFrame;                                               // frame rotation angular speed
extern double PhysicalTime, PhysicalTimeInitial;                    // time of simulation, initial time when simulation was started
extern double  SGAccInnerEdge, SGAccOuterEdge;                          // sg
extern int    TimeStep;                                             // number of steps calcualted
extern double TotDiskMass;
extern double *omega;
extern bool    LogGrid;                                                 // logarithmic or arithmetic grid grid
extern bool   AdvecteLabel, FakeSequential, MonitorIntegral, OnlyInit;
extern bool	 GotoNextOutput;
extern bool   StoreSigma;
extern bool   ViscosityAlpha;
extern bool   RocheSmoothing;
extern bool    CentrifugalBalance;
extern bool    SloppyCFL;
extern bool    BaryCentric;
extern bool    MonitorBC, MonitorTorque, MonitorDiskEcc;  // monitoring disk bary center, planet torques, and disk eccentricity
extern bool    MonitorAccretion;
extern int     ExcludeHill;                                          // planetary toruqe Hill exclusion (0: none, 1 Gaussian, 2 Heaviside)
extern bool    SelfGravity;
extern bool    Indirect_Term;                                      // inlcude indirect term caused by the disk (planetary ind. term is always taken into account)
extern bool    DustGrid;                                                 // simulate dust as pressureless fluid
extern bool    DustConstStokes;
extern bool    DustSelfGravity;                                          // calculate dust self-gravity
extern bool    DustFeedback;
extern bool    FastTransport;                                       // use FARGO algorithm
extern bool    IsDisk, Corotating;
extern bool  Profiling, TimeInfo;
extern bool    ZeroVelrad;
extern bool    OverridesOutputdir;
extern char    NewOutputdir[1024];
extern double  MinSemiMajorPlanet;
extern bool    TerminateDuetoPlanet;
extern double  GasDiskMassInner, GasDiskMassOuter;
extern double  GasDiskBC_x, GasDiskBC_y;
extern double  GasBC_x, GasBC_y;
extern double  DustDiskMassInner, DustDiskMassOuter;
extern double  DustDiskBC_x, DustDiskBC_y;
extern double  DustBC_x, DustBC_y;
extern double  StarPlanetBC_x, StarPlanetBC_y;
extern double  DiskEcc_SurfNorm_Inner, DiskEcc_SurfNorm_Outer, DiskEcc_MassNorm_Inner, DiskEcc_MassNorm_Outer;
extern double *DustSizeBin, *DustMassBin;
extern double *Pl0AccretedMass;
extern int     DustBinNum;
extern bool    Write_Density, Write_Velocity;                            // density and velociy snapsots
extern bool    Write_Energy, Write_Temperature, Write_DiskHeight;    // energy, temperature and disk height snappshots for adiabatic disk
extern bool    Write_Potential;                                                 // gravitational potental for self-gravitating disk
extern bool    Write_SoundSpeed;                                                // sound speed
extern bool    OpenInner, OpenOuter;                                      // open
extern bool    DampingInner, DampingOuter;                                // damping (only velocities are damped)
extern bool    StrongDampingInner, StrongDampingOuter;                    // strong damping (velocities and density are also damped)
extern bool    NonReflectingInner, NonReflectingOuter;                    // non-reflecting
extern bool    RigidWallInner, RigidWallOuter;                            // rigid inner wall (like reflecting)
extern bool    ViscOutflowInner, ViscOutflowOuter;                        // viscous outflow
extern bool    ClosedInner, ClosedOuter;                                  // closed (no inflow, no outflow)
extern bool    EmptyInner, EmptyOuter;
extern bool    OuterSourceMass;
extern bool    GuidingCenter;
extern bool    AdaptiveViscosity;
extern bool    DeadZone;
extern bool    Adiabatic;                                            // solve energy equation for adiabatic gas EOS
extern bool    Cooling;                                              // calculate cooling
extern bool    ViscHeating;
extern bool    ModelNebula;
extern double  ScalingFactor;
extern bool    Restart, AutoRestart;
extern int     begin_i, NbRestart;
extern double  LostMass, AccRate, StellarAccRate;
extern bool    CreatingMovieOnly;
extern bool    DustGrowth;
extern PolarGrid *CellAbscissa, *CellOrdinate;                            // cell 
extern PolarGrid *RhoStar, *RhoInt, *Potential;                           // 
extern PolarGrid *gas_density, *gas_v_rad, *gas_v_theta, *gas_label;      // gas density, radial and azimuthal velocity, and gas label
extern PolarGrid *VradInt, *VthetaInt;
extern PolarGrid *EnergyInt;
extern PolarGrid *VradNew, *VthetaNew;              // intermediate and new gas velocities
extern PolarGrid *gas_energy, *SoundSpeed, *Viscosity;                 // adiabatic 
extern PolarGrid *TauRR, *TauRP, *TauPP;
extern PolarGrid *Work, *TemperInt, *QRStar, *Elongations, *Qder;                   // ?
extern PolarGrid *RadMomP, *RadMomM, *ThetaMomP, *ThetaMomM, *ExtLabel, *VthetaRes; // ?
extern PolarGrid **dust_density, **dust_v_rad, **dust_v_theta;                      // dust densities, radial and azimuthal velocities for each dust bin
extern PolarGrid **VradDustInt, **VthetaDustInt;
extern PolarGrid  *disk_ecc;                                                        // disk eccentricity
extern PolarGrid *FTD;                                                              // field to display by OpenGl
extern PolarGrid *Buffer;
extern PolarGrid *SGAcc;
extern PolarGrid *dust_size, *dust_growth_rate;
extern PolarGrid *tmp1, *tmp2;
extern PolarGrid *tmp3, *tmp4;
extern PolarGrid *DeltaT;
extern PolarGrid *WorkShift;
extern PolarGrid *myWork;
extern PlanetarySystem *sys;                                                        // structure used for planetary system
extern Pair DiskOnPrimaryAcceleration;
extern Pair IndirectTerm;
extern double  RefreshRate;
extern bool    Window;
extern bool       Paused;
extern int     IterCount, StillWriteOneOutput;
extern TimeProcess      t_Hydro;
extern bool    verbose;
