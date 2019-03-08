//
//
//

#ifdef OPT_KERNEL_THREADS

#define DEF_BLOCK_X_ADVECTSHIFT       16*2
#define DEF_BLOCK_X_CALCECC           16*2
#define DEF_BLOCK_X_CFL               64*4
#define DEF_BLOCK_X_CLOSEDBC          64*4
#define DEF_BLOCK_X_LRMOMENTA         32
#define DEF_BLOCK_X_COMPUTERESIDUAL   4
#define DEF_BLOCK_X_COMPUTESTARRAD    16
#define DEF_BLOCK_X_COMPUTESTARTHETA  64*4
#define DEF_BLOCK_X_COMPUTEVEL        8
#define DEF_BLOCK_X_DAMPINGBC         64*4
#define DEF_BLOCK_X_DETECTCRASH       64*4
#define DEF_BLOCK_X_DISKBC            16
#define DEF_BLOCK_X_DIVIDE_POLARGRID  32
#define DEF_BLOCK_X_FORCE             16
#define DEF_BLOCK_X_GBLINEARINTERPOL  16
#define DEF_BLOCK_X_GLCUDA            8
#define DEF_BLOCK_X_GLVORTENS         8
#define DEF_BLOCK_X_GPU_SELF_GRAVITY  32
#define DEF_BLOCK_X_NONREFBC          64*4
#define DEF_BLOCK_X_OPENBC            64*4
#define DEF_BLOCK_X_OUTERSOURCEMASS   64*4
#define DEF_BLOCK_X_PARTIALREDUCTION  64 // cannot be larger - otherwise kernel needs editing
#define DEF_BLOCK_X_PLANET            32*4
#define DEF_BLOCK_X_POT               32*4
#define DEF_BLOCK_X_REFBC             64
#define DEF_BLOCK_X_SUBSTEP1          8
#define DEF_BLOCK_X_SUBSTEP2          64*2
#define DEF_BLOCK_X_TESTBC            64*4
#define DEF_BLOCK_X_VALLEERAD         16
#define DEF_BLOCK_X_VANLEERTHETA      32
#define DEF_BLOCK_X_VISCO_ADAPTIVE    64*2
#define DEF_BLOCK_X_VISCO             64*2
#define DEF_BLOCK_X_VISCOUTFLOW       64*4


#else

#define DEF_BLOCK_X_ADVECTSHIFT       16
#define DEF_BLOCK_X_CALCECC           16
#define DEF_BLOCK_X_CFL               64
#define DEF_BLOCK_X_CLOSEDBC          64
#define DEF_BLOCK_X_LRMOMENTA         32
#define DEF_BLOCK_X_COMPUTERESIDUAL   4
#define DEF_BLOCK_X_COMPUTESTARRAD    16
#define DEF_BLOCK_X_COMPUTESTARTHETA  64
#define DEF_BLOCK_X_COMPUTEVEL        8
#define DEF_BLOCK_X_DAMPINGBC         64
#define DEF_BLOCK_X_DETECTCRASH       64
#define DEF_BLOCK_X_DISKBC            16
#define DEF_BLOCK_X_DIVIDE_POLARGRID  32
#define DEF_BLOCK_X_FORCE             16
#define DEF_BLOCK_X_GBLINEARINTERPOL  16
#define DEF_BLOCK_X_GLCUDA            8
#define DEF_BLOCK_X_GLVORTENS         8
#define DEF_BLOCK_X_GPU_SELF_GRAVITY  32
#define DEF_BLOCK_X_NONREFBC          64
#define DEF_BLOCK_X_OPENBC            64
#define DEF_BLOCK_X_OUTERSOURCEMASS   64
#define DEF_BLOCK_X_PARTIALREDUCTION  64 // cannot be larger - otherwise kernel needs editing
#define DEF_BLOCK_X_PLANET            32
#define DEF_BLOCK_X_POT               32
#define DEF_BLOCK_X_REFBC             64
#define DEF_BLOCK_X_SUBSTEP1          8
#define DEF_BLOCK_X_SUBSTEP2          64
#define DEF_BLOCK_X_TESTBC            64
#define DEF_BLOCK_X_VALLEERAD         16
#define DEF_BLOCK_X_VANLEERTHETA      32
#define DEF_BLOCK_X_VISCO_ADAPTIVE    64
#define DEF_BLOCK_X_VISCO             64
#define DEF_BLOCK_X_VISCOUTFLOW       64

#endif
