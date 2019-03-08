/** \file fondam.h

Contains fondamental constants used thorough the code.
*/

#define	G	            1.0
#define	PI	          3.14159265358979323844
#define CPUOVERLAP    5	            // Zeus-like overlap kernel. 2:transport; 2: source, 1:viscous stress

#define CVNR          1.41       	  // Shocks are spread over CVNR zones:
                                    // von Neumann-Richtmyer viscosity constant
                                    // Beware of misprint in Stone and Norman's
                                    // paper : use C2^2 instead of C2

#define MU            1.0           // Mean molecular weight
#define R_SPEC        1.0           // Universal Gas Constant in code units

#define C_DP 1e-3                   // artificial dust pressure magnitude

#define EPSTEIN_ONLY                                                 // only Epstein regime is taken into account

#define DAMPING_STRENGTH_IN   30.0         // strength for inner damping boundary condition
#define DAMPING_STRENGTH_OUT  30.0         // strength for outer damping boundary condition

// some constants required for stopping time calcuation
#define CONST_MASS_CGS                     1.9900001e+33             // Msun in CGS
#define CONST_RHO_CGS                      5.943719364576997e-7      // Msun/AU^3 in CGS
#define CONST_AU_CGS                       1.49600e13                // AU in CGS 
#define CONST_KB_CGS                       1.3806580e-16             // Boltzmann constant in CGS

#define MASS_UNIT                          1.0
#define CONST_MFP_CGS                      3.34859e-9                //
#define CONST_VEL_CGS                      7.5442268265680483e+04    // AU/year in CGS
#define CONST_VEL2_CGS                     5.6915358410709000e+09    // (AU/year)^2 in CGS
#define DUST_MIN_SIZE                      0.1/2.0   // 0.1 um minimum dust size
#define DUST_STAR_SIZE                     10.0/2.0   // 1un dust soze *
#define CONST_UM_CGS                       1e-4  
#define DUST_SIZE_DISTR                    3.5    // dust size distribution power index (e.g. Dohnanyi (19??) )

#define CONST_SQRT8OPI                     sqrt (8.0/M_PI)
#define CONST_8O3o044                      ((8.0/3.0)/0.44)
#define CONST_RSQRT2PI                     1.0/sqrt(2.0 * M_PI)
#define CONST_2O3PI                        2.0 / (3.0 * M_PI)
#define CONST_SQRT2                        sqrt(2.0)
#define CONST_RSQRT2                       1.0/sqrt(2.0)
#define CONST_SQRT2PI                      sqrt(2.0*M_PI)
#define CONST_SQRT16OPI                    sqrt(16.0/M_PI)