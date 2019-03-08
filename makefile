#/*----------------------------------------------------------------------------
# GFARGO 
#
# makefile
#
# Written by Zs. Regaly 2016 (www.konkoly.hu/staff/regaly)
# Computational Astrophysics Group of Konkoly Observatory (www.konkoly.hu/CAG/)
#------------------------------------------------------------------------------*/

verbose = 0

# CUDA path (Linux)
ifeq ($(NIIF),1)
CUDA_TK        = /opt/nce/packages/global/cuda/8.0.61
else
CUDA_TK        = /usr/local/cuda
endif

CUDA_SDK_PATH  = $(CUDA_TK)/samples

# change EXENAME to 
EXENAME        = gfargo2

# C/C++ compiler and flags for Linux architecture
CXX       = /usr/bin/g++
#CXX_FLAGS = -O3 -Wall -m64 -Wno-unknown-pragmas -Wno-format-zero-length -Wno-write-strings -I$(CUDA_TK)/include -I$(CUDA_TK)/include
CXX_FLAGS = -O3 -Wall -m64 -I$(CUDA_TK)/include -I$(CUDA_TK)/include 

# new version required this
CXX_FLAGS += -Wno-unknown-pragmas -Wno-write-strings -Wno-format-extra-args
CXX_FLAGS += -no-pie -fno-pie

# NVCC settings
NVCC              = $(CUDA_TK)/bin/nvcc
NVCC_FLAGS        = -I $(CUDA_TK)/include -I $(CUDA_SDK_PATH)/common/inc
NVCC_FLAGS       += --compiler-options -fno-strict-aliasing 
NVCC_FLAGS       += -Wno-deprecated-gpu-targets -Wno-deprecated-declarations 



# add GPU architecture to NVCC setting
ifeq ($(GPU_V100),1)
	GPU_ARCHITECTURES += -gencode arch=compute_70,code=sm_70
endif
ifeq ($(GPU_K80),1)
	GPU_ARCHITECTURES += -gencode arch=compute_37,code=sm_37
endif
ifeq ($(GPU_K40),1)
	GPU_ARCHITECTURES += -gencode arch=compute_35,code=sm_35
endif
ifeq ($(GPU_K20),1)
	GPU_ARCHITECTURES += -gencode arch=compute_30,code=sm_30
endif
ifeq ($(GPU_T2075),1)
	GPU_ARCHITECTURES += -gencode arch=compute_20,code=sm_20
endif
ifeq ($(GPU_ALL),1)
	GPU_ARCHITECTURES = -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37
endif
NVCC_FLAGS       += $(GPU_ARCHITECTURES)


MAINOBJ     = LowTasks.o SideEuler.o Output.o Init.o main.o Theo.o Nebula.o\
              Interpret.o SourceEuler.o TransportEuler.o\
              Planet.o RungeKunta.o Psys.o \
              rebin.o\
              var.o Pframeforce.o\
              vanleerradgpu.o vanleerthetagpu.o loop.o\
              checknans.o

CUDAFILES   = LowTasks.cu substep1.cu substep2.cu substep3.cu substep4.cu \
              divide_polargrid.cu computeLRmomenta.cu computevel.cu \
              vanleerrad.cu vanleertheta.cu computestarrad.cu \
              computestartheta.cu partialreduction.cu advectshift.cu \
              computeresidual.cu cfl.cu visco.cu force.cu  \
              pot.cu \
              bc_damping.cu bc_open.cu bc_nonref.cu bc_ref.cu bc_viscoutflow.cu bc_closed.cu bc_outersourcemass.cu \
              detectcrash.cu \
              gpu_self_gravity.cu Planet.cu gBilinearInterpol.cu \
              calcbc.cu calcecc.cu  calcvortens.cu calctemp.cu calcdiskheight.cu calcsoundspeed.cu calcdustgasmratio.cu calcdustsize.cu

# shell command
SHELL		=  /bin/sh

include .window

# Integration with FARGO
ifeq ($(FARGO_INTEGRATION),1)
  CXX_FLAGS  += -DFARGO_INTEGRATION
  NVCC_FLAGS += -DFARGO_INTEGRATION
endif

ifeq ($(DUST_FEEDBACK),1)
  CXX_FLAGS  += -DDUST_FEEDBACK
  NVCC_FLAGS += -DDUST_FEEDBACK
endif

#ifeq ($(OPT_KERNEL_THREADS),1)
#  CXX_FLAGS  += -DOPT_KERNEL_THREADS
#  NVCC_FLAGS += -DOPT_KERNEL_THREADS
#endif


#
#
#
#--------------------No Changes Needed after this line------------------------
#
#
#

ifneq ($(WINDOW),opengl)
MAINOBJ  += gldummy.o
endif
#DUMMY	 = mpi_dummy.o
LD_LIBS  = -L$(LIBMATH_EVAL)/lib -lm
AUTOINCL = param.h param_noex.h global_ex.h

INCLUDE = *.h

OPTIONS		= $(OPT) $(OPTSEQ)
OBJ		= $(MAINOBJ)

all: conditionalrebuild $(AUTOINCL) $(OBJ) $(CUDAOBJ) $(EXENAME)
#ifeq ($(BUILD),parallel)
#	@echo ""
#	@echo "NOTE"
#	@echo ""
#	@echo "Parallel build deprecated in this beta version of gfargo"
#	@echo "Building sequential executable instead"
#endif
	@echo ""
	@echo "NOTE"
	@echo "GPU architectures included:" $(GPU_ARCHITECTURES)
	@echo ""
	
.PHONY: conditionalrebuild
ifneq ($(WINDOW),$(OLDWINDOW))
conditionalrebuild: clean
	@echo "OLDWINDOW = $(WINDOW)" > .window
	@echo "WINDOW = $(WINDOW)" >> .window
else
conditionalrebuild:
endif

.oldconfig:
.config:
.window:
.oldwindow:

archive : $(SRC) $(INCL) makefile varparser.pl	
	@tar cf ../.source.tar *.c
	@tar rf ../.source.tar *.h
	@tar rf ../.source.tar makefile
	@tar rf ../.source.tar varparser.pl
	@bzip2 -9 -f ../.source.tar

#para:
#	@make BUILD=parallel

#seq:
#	@make BUILD=sequential

win:
	@make WINDOW=opengl
	@echo "Using OpenGL for real time rendering"

nw:
	@make WINDOW=none
	@echo "No real time rendering"
	
$(AUTOINCL) : var.c global.h makefile varparser.pl
	@./varparser.pl

$(OBJ): fargo.h fondam.h param.h param_noex.h types.h makefile

.PHONY: clean mrproper package

mrproper:
	@echo "Cleaning everything"
	@rm -f *.o *.cuo *~ *.s *.il $(AUTOINCL) $(EXENAME) ../core.*\
	*.tex *.dvi *.pdf *.ps *.log *.aux *.lint $(ARCHIVE)\
	$(ARCHIVECOMP)
	@find . | egrep "#" | xargs rm -f

clean:
	@echo "Tidying up"
	@rm -f *.o *~ *.s *.il *.cuo

ifeq ($(verbose), 1)
        VERBOSE :=
else
        VERBOSE := @
endif

%.o  : %.c
	@echo "Compiling C file $*.c"
	$(VERBOSE)$(CXX) $*.c -c $(CXX_FLAGS)

ifeq ($(WINDOW),opengl)
  CUDAFILES += glext.cu glcuda.cu
endif

CUDAOBJ     = $(patsubst %.cu,%.cuo, $(CUDAFILES))
#CUDALIBS    = -L$(CUDA_TK)/lib64 -L$(CUDA_SDK_PATH)/common/lib/linux/x86_64  -lcufft -lcuda -lcudart
CUDALIBS    = -L$(CUDA_TK)/lib64 -L$(CUDA_SDK_PATH)/common/lib/linux/x86_64  -lcufft -lcudart

%.cuo : %.cu
	@echo "Compiling CUDA file $<"
	$(VERBOSE)$(NVCC) $(NVCC_FLAGS) -o $@ -c $<

$(CUDAOBJ) : $(INCLUDE)

OPENGLOPT_	=
ifeq ($(WINDOW), opengl)
	OPENGLOPT +=  -lGL -lGLU -lGLEW -lglut
endif

ifndef FARGO_INTEGRATION
$(EXENAME): $(OBJ) $(CUDAOBJ)
	@echo "Building executable for $(EXENAME)..."
	$(VERBOSE)g++ -fPIC $(OBJ) $(CUDAOBJ) $(CXX_FLAGS) -o $(EXENAME) $(LD_LIBS) $(CUDALIBS) -DUNIX $(OPENGLOPT)
	@echo "Done"
else
$(EXENAME): $(OBJ) $(CUDAOBJ)
	@echo "Building modules for $(EXENAME)"
endif
