/** \file SideEuler.c

Total mass and angular momentum monitoring, and boundary conditions.
In addition, this file contains a few low-level functions that
manipulate PolarGrid 's or initialize the forces evaluation.

*/

#include "fargo.h"

double GasTotalMass (PolarGrid *array) {
   int i,j,ns;
   double *density, total = 0.0, fulltotal=0.0;
   ns = array->Nsec;
   density = array->Field;
   //if (FakeSequential && (CPU_Rank > 0)) 
     //MPI_Recv (&total, 1, MPI_DOUBLE, CPU_Rank-1, 0, MPI_COMM_WORLD, &fargostat);
   for (i = Zero_or_active; i < Max_or_active; i++) {
     for (j = 0; j < ns; j++) {
       total += Surf[i]*density[j+i*ns];
     }
   }
   if (FakeSequential) {
     //if (CPU_Rank < CPU_Number-1)
       //MPI_Send (&total, 1, MPI_DOUBLE, CPU_Rank+1, 0, MPI_COMM_WORLD);
   }
   //else
     //MPI_Allreduce (&total, &fulltotal, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   if (FakeSequential) {
     //MPI_Bcast (&total, 1, MPI_DOUBLE, CPU_Number-1, MPI_COMM_WORLD);
     fulltotal = total;
   }
   return fulltotal;
}

double GasMomentum (PolarGrid *Density, PolarGrid *Vtheta) {
   int i,j,ns;
   double *density, *vtheta, total = 0.0, fulltotal=0.0;
   ns = Density->Nsec;
   density = Density->Field;
   vtheta = Vtheta->Field;
   //if (FakeSequential && (CPU_Rank > 0)) 
     //MPI_Recv (&total, 1, MPI_DOUBLE, CPU_Rank-1, 2, MPI_COMM_WORLD, &fargostat);
   for (i = Zero_or_active; i < Max_or_active; i++) {
     for (j = 1; j < ns; j++) {
       total += Surf[i]*(density[j+i*ns]+density[j-1+i*ns])*Rmed[i]*(vtheta[j+i*ns]+OmegaFrame*Rmed[i]);
     }
     total += Surf[i]*(density[i*ns]+density[i*ns+ns-1])*Rmed[i]*(vtheta[i*ns]+OmegaFrame*Rmed[i]);
   }
   if (FakeSequential) {
     //if (CPU_Rank < CPU_Number-1)
       //MPI_Send (&total, 1, MPI_DOUBLE, CPU_Rank+1, 2, MPI_COMM_WORLD);
   }
   //else
     //MPI_Allreduce (&total, &fulltotal, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   if (FakeSequential) {
     //MPI_Bcast (&total, 1, MPI_DOUBLE, CPU_Number-1, MPI_COMM_WORLD);
     fulltotal = total;
   }
   return 0.5*fulltotal;
}

void DivisePolarGrid (PolarGrid *Num, PolarGrid *Denom, PolarGrid *Res) {
  int i,j,l,nr,ns;
  double *num, *denom, *res;
  num = Num->Field;
  denom=Denom->Field;
  res = Res->Field;
  ns = Res->Nrad;
  nr = Res->Nsec;
#pragma omp parallel for private(j,l)
  for (i = 0; i <= nr; i++) {
    for (j = 0; j < ns; j++) {
      l = j+ns*i;
      res[l] = num[l]/(denom[l]+1e-20);
    }
  }
}

void InitComputeAccel () {
  int i, j, l, nr, ns;
  double *abs, *ord;
  CellAbscissa = CreatePolarGrid (NRAD,NSEC, "abscissa");
  CellOrdinate = CreatePolarGrid (NRAD,NSEC, "ordinate");
  nr = CellAbscissa->Nrad;
  ns = CellAbscissa->Nsec;
  abs = CellAbscissa->Field;
  ord = CellOrdinate->Field;
  for (i = 0; i < nr; i++) {
    for (j = 0; j < ns; j++) {
       l = j+i*ns;
       abs[l] = Rmed[i] * cos(2.0*M_PI*(double)j/(double)ns);
       ord[l] = Rmed[i] * sin(2.0*M_PI*(double)j/(double)ns);
    }
  }
}
  
Pair ComputeAccel (PolarGrid *Rho, double x, double y, double rsmoothing, double mass) {
  Pair acceleration;
  Force force;
  if (ExcludeHill > 0) {
    force = ComputeForce_gpu (Rho, x, y, rsmoothing, mass, ExcludeHill);
    acceleration.x = force.fx_ex_inner+force.fx_ex_outer;
    acceleration.y = force.fy_ex_inner+force.fy_ex_outer;
  } 
  else {
    force = ComputeForce_gpu (Rho, x, y, rsmoothing, mass, NO);
    acceleration.x = force.fx_inner+force.fx_outer;
    acceleration.y = force.fy_inner+force.fy_outer;
  }
  return acceleration;
}

void ApplyBoundaryCondition (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, PolarGrid *Energy, double dt) {
   double R_inf, R_sup;

  // do nothing
  if (EmptyInner == YES || EmptyOuter == YES)
    return; 

  // DAMPING (wave killing) boundary conditions
  // damps only the velocity components
  if (DampingInner == YES) {
    R_inf = (double)RMIN*DAMPRMIN;
    R_sup = (double)RMAX*DAMPRMAX;
    StockholmBoundary_gpu (Vrad, Vtheta, Rho, Energy, dt, INNER, R_inf, R_sup, false);
  }
  if (DampingOuter == YES) {
    R_inf = (double)RMIN*DAMPRMIN;
    R_sup = (double)RMAX*DAMPRMAX;
    StockholmBoundary_gpu (Vrad, Vtheta, Rho, Energy, dt, OUTER, R_inf, R_sup, false);
  }

  // STRONGDAMPING (wave killing boundary) conditions
  // damps the density, energy, and velocity components too
  if (StrongDampingInner == YES) {
    R_inf = (double)RMIN*DAMPRMIN;
    R_sup = (double)RMAX*DAMPRMAX;
    StockholmBoundary_gpu (Vrad, Vtheta, Rho,  Energy, dt, INNER, R_inf, R_sup, true);
  }
  if (StrongDampingOuter == YES) {
    R_inf = (double)RMIN*DAMPRMIN;
    R_sup = (double)RMAX*DAMPRMAX;
    StockholmBoundary_gpu (Vrad, Vtheta, Rho,  Energy, dt, OUTER, R_inf, R_sup, true);
  }
      
  // OPEN boundary conditions
  if (OpenInner == YES) {
    OpenBoundary_gpu (Vrad, Vtheta, Rho, Energy, INNER);
  }
  if (OpenOuter == YES) {
    OpenBoundary_gpu (Vrad, Vtheta, Rho, Energy, OUTER);
  }

  // REFLECTING boundary conditions
  if (RigidWallInner == YES) {
    ReflectingBoundary_gpu (Vrad, Vtheta, Rho, INNER);
  }
  if (RigidWallOuter == YES) {
    ReflectingBoundary_gpu (Vrad, Vtheta, Rho, OUTER);
  }
  
  // VISCOUSOUTFLOW conditions
  if (ViscOutflowInner == YES) {
    ViscOutflow_gpu (Vrad, Vtheta, Rho, INNER);
  }
  if (ViscOutflowOuter == YES) {
    ViscOutflow_gpu (Vrad, Vtheta, Rho, OUTER);
  }

  // CLOSED conditions
  if (ClosedInner == YES) {
    ClosedBoundary_gpu (Vrad, Vtheta, Rho, INNER);
  }
  if (ClosedOuter == YES) {
    ClosedBoundary_gpu (Vrad, Vtheta, Rho, OUTER);
  }

  // NONREFLECTING boundary conditions
  if (NonReflectingInner == YES) {
    NonReflectingBoundary_gpu (Vrad, Vtheta, Rho, INNER);
  }
  if (NonReflectingOuter == YES) {
    NonReflectingBoundary_gpu (Vrad, Vtheta, Rho, OUTER);
  }
  /*
  // outer soure mass
  // it is not implemented yet!!!!
  if (OuterSourceMass == YES) {
     ApplyOuterSourceMass_gpu(Vrad, Rho);
  }
  */
}
 
void ApplyBoundaryConditionDust (PolarGrid *Vrad, PolarGrid *Vtheta, PolarGrid *Rho, int DustBin, double dt) {

  // special open for dust at inner edge
  if (OpenInner == YES)
    OpenBoundaryDust_gpu (Vrad, Vtheta, Rho, DustBin, INNER);

  // special open for dust at inner edge
  if (OpenOuter == YES)
    OpenBoundaryDust_gpu (Vrad, Vtheta, Rho, DustBin, OUTER);

  // apply outer source mass or special open for dust at outer edge
  //if (OuterSourceMass == YES)
  //  ApplyOuterSourceMass_gpu(Vrad, Rho, dustmass);
  //else
//  if (OpenOuter == YES) 
 //   OpenBoundaryDust_gpu (Vrad, Vtheta, Rho, DustBin, OUTER);

  /*
  if (OpenInner == YES)
    OpenBoundaryDust_gpu (Vrad, Vtheta, Rho, dustmass, INNER);
  
  if (OpenOuter == YES) 
    OpenBoundaryDust_gpu (Vrad, Vtheta, Rho, dustmass, OUTER);

  // closed conditions
  if (ClosedInner == YES) {
    ClosedBoundary_gpu (Vrad, Vtheta, Rho, INNER);
  }
  if (ClosedOuter == YES) {
    ClosedBoundary_gpu (Vrad, Vtheta, Rho, OUTER);
  }
  */
   // StrongDamping or wave killing boundary conditions (Stockholm conditions) only at the inner boundary
  if (StrongDampingInner == YES) {
    StockholmBoundaryDust_gpu (Vrad, Vtheta, Rho, DustBin, dt, INNER);
  }
  
  // STRONGDAMPING for dust
  if (StrongDampingOuter == YES) {
    StockholmBoundaryDust_gpu (Vrad, Vtheta, Rho, DustBin, dt, OUTER);
  }
} 
