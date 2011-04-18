// -*- C++ -*-

/*! \file
 * \brief Primary include file for QDP
 *
 * No other file should be included by the user
 */

/*! \mainpage  QDP
 *
 * \section Description
 *
 * QDP is a C++ data-parallel interface for Lattice field theory.
 * The QDP interface provides an environment somewhat similar to 
 * Fortran 90 - namely data-parallel operations (operator/infix form)
 * which can be applied on lattice wide objects. The interface provides a
 * level of abstraction such that high-level user code written using the
 * API can be run unchanged on a single processor workstation or a
 * collection of multiprocessor nodes with parallel communications.
 * Architectural dependencies are hidden below the interface. A variety
 * of types for the site elements are provided. To achieve good
 * performance, overlapping communication and computation primitives are
 * provided.
 */

/*! \namespace QDP
 * \brief Primary namespace holding all QDP types, operations and objects
 */

#ifndef QDP_INCLUDE
#define QDP_INCLUDE

/* Get local configuration options (ARCH_SCALAR/PARSCALAR, Nd, Nc, Ns) */
#include <cudp_config.h>
#include "cudp_precision.h"


// Under gcc, set some attributes
#if defined(__GNUC__)
// gcc
#define QDP_ALIGN8   __attribute__ ((aligned (8)))
#define QDP_ALIGN16  __attribute__ ((aligned (16)))
#define QDP_INLINE   __attribute__ ((always_inline))
// The attributes in QDP_CONST is buggering g++-3.4 
//#define QDP_CONST    __attribute__ ((const,pure))
#define QDP_CONST
#define QDP_CINLINE  __attribute__ ((always_inline,const,pure))
#else
// default
#define QDP_ALIGN8
#define QDP_ALIGN16
#define QDP_INLINE
#define QDP_CONST
#define QDP_CINLINE
#endif

#define QDP_USE_SSE   0
#define QDP_USE_SSE2  0


#define QDP_ALIGNMENT_SIZE 16

#include <cstdio>
#include <cstdlib>
//#include <ostream>
//#include <iostream>

using namespace std;   // I do not like this - fix later

// using std::iostream;
// using std::ostream;
// END OF YUKKINESS


// Basic includes
#define PETE_MAKE_EMPTY_CONSTRUCTORS
#define PETE_USER_DEFINED_EXPRESSION
namespace QDP {
#include <cuPETE/cuPETE.h>
}

#include "cudp_init.h"
#include "cudp_forward.h"
#include "cudp_multi.h"

#include "cudp_params.h"
#include "cudp_layout.h"
// #include "cudp_filebuf.h"
// #include "cudp_io.h"
// #include "cudp_stdio.h"
// #include "cudp_xmlio.h"
// #include "cudp_qdpio.h"
// #include "cudp_binx.h"
#include "cudp_subset.h"
#include "cudp_map.h"
// #include "cudp_stopwatch.h"

#include "cudp_traits.h"
#include "cudp_qdpexpr.h"
#include "cudp_qdptype.h"
#include "cudp_qdpsubtype.h"

namespace QDP {
#include "cuQDPOperators.h"
}

// Include the allocator stuff here, before QDP_outer
#include "cudp_allocator.h"

#include "cudp_newops.h"
#include "cudp_optops.h"
// #include "cudp_profile.h"
#include "cudp_simpleword.h"
#include "cudp_reality.h"
//#include "cudp_inner.h"
#include "cudp_primitive.h"
#include "cudp_outer.h"
// #include "cudp_outersubtype.h"
#include "cudp_defs.h"
#include "cudp_globalfuncs.h"
#include "cudp_specializations.h"

//#include "cudp_special.h"
#include "cudp_random.h"

// Include threading code here if applicable
//#include "cudp_dispatch.h"

namespace ThreadReductions { 
 
}

#if defined(ARCH_SCALAR)
// Architectural specific code to a single node/single proc box
#warning "Using scalar architecture"
#include "cudp_scalar_specific.h"

// Include SSE code here if applicable
#if QDP_USE_SSE == 1
#include "cudp_scalarsite_sse.h"
#elif QDP_USE_BAGEL_QDP == 1
// USE_BAGEL_QDP
#include "cudp_scalarsite_bagel_qdp.h"
#else
// Use Generics only
//#include "cudp_scalarsite_generic.h"
#endif

#elif defined(ARCH_PARSCALAR)
// Architectural specific code to a parallel/single proc box
#warning "Using parallel scalar architecture"
#include "cudp_parscalar_specific.h"

// Include optimized code here if applicable
#if QDP_USE_SSE == 1
#include "cudp_scalarsite_sse.h"
#elif QDP_USE_BAGEL_QDP == 1
// Use BAGEL_QDP 
#include "cudp_scalarsite_bagel_qdp.h"
#else
// Use generics
#include "cudp_scalarsite_generic.h"
#endif

#elif defined(ARCH_SCALARVEC)
// Architectural specific code to a single node/single proc box 
// with vector extension
#warning "Using scalar architecture with vector extensions"
#include "cudp_scalarvec_specific.h"

// Include optimized code here if applicable
#if QDP_USE_SSE == 1
#include "cudp_scalarvecsite_sse.h"
#endif

#elif defined(ARCH_PARSCALARVEC)
// Architectural specific code to a parallel/single proc box
// with vector extension
#warning "Using parallel scalar architecture with vector extensions"
#include "cudp_parscalarvec_specific.h"

// Include optimized code here if applicable
#if QDP_USE_SSE == 1
#include "cudp_scalarvecsite_sse.h"
#endif

#else
#error "Unknown architecture ARCH"
#endif

//#include "cudp_flopcount.h"

#endif  // QDP_INCLUDE
