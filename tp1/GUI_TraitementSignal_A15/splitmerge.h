/*
 * MATLAB Compiler: 4.8 (R2008a)
 * Date: Tue Apr 29 14:26:53 2014
 * Arguments: "-B" "macro_default" "-l" "splitmerge.m" 
 */

#ifndef __splitmerge_h
#define __splitmerge_h 1

#if defined(__cplusplus) && !defined(mclmcrrt_h) && defined(__linux__)
#  pragma implementation "mclmcrrt.h"
#endif
#include "mclmcrrt.h"
#ifdef __cplusplus
extern "C" {
#endif

#if defined(__SUNPRO_CC)
/* Solaris shared libraries use __global, rather than mapfiles
 * to define the API exported from a shared library. __global is
 * only necessary when building the library -- files including
 * this header file to use the library do not need the __global
 * declaration; hence the EXPORTING_<library> logic.
 */

#ifdef EXPORTING_splitmerge
#define PUBLIC_splitmerge_C_API __global
#else
#define PUBLIC_splitmerge_C_API /* No import statement needed. */
#endif

#define LIB_splitmerge_C_API PUBLIC_splitmerge_C_API

#elif defined(_HPUX_SOURCE)

#ifdef EXPORTING_splitmerge
#define PUBLIC_splitmerge_C_API __declspec(dllexport)
#else
#define PUBLIC_splitmerge_C_API __declspec(dllimport)
#endif

#define LIB_splitmerge_C_API PUBLIC_splitmerge_C_API


#else

#define LIB_splitmerge_C_API

#endif

/* This symbol is defined in shared libraries. Define it here
 * (to nothing) in case this isn't a shared library. 
 */
#ifndef LIB_splitmerge_C_API 
#define LIB_splitmerge_C_API /* No special import/export declaration */
#endif

extern LIB_splitmerge_C_API 
bool MW_CALL_CONV splitmergeInitializeWithHandlers(mclOutputHandlerFcn error_handler,
                                                   mclOutputHandlerFcn print_handler);

extern LIB_splitmerge_C_API 
bool MW_CALL_CONV splitmergeInitialize(void);

extern LIB_splitmerge_C_API 
void MW_CALL_CONV splitmergeTerminate(void);



extern LIB_splitmerge_C_API 
void MW_CALL_CONV splitmergePrintStackTrace(void);


extern LIB_splitmerge_C_API 
bool MW_CALL_CONV mlxSplitmerge(int nlhs, mxArray *plhs[],
                                int nrhs, mxArray *prhs[]);


extern LIB_splitmerge_C_API bool MW_CALL_CONV mlfSplitmerge(int nargout
                                                            , mxArray** g
                                                            , mxArray* f
                                                            , mxArray* mindim
                                                            , mxArray* fun);

#ifdef __cplusplus
}
#endif

#endif
