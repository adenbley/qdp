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
#include <qdp_config.h>
#include "qdp_precision.h"

// GNU specific stuff
#if defined(__GNUC__)
// Under g++, enforce using V3 or greater
#if __GNUC__ < 3
#error "QDP++ requires g++ 3.0 or higher. This version of the g++ compiler is not supported"
#endif
#endif

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

#if (QDP_USE_SSE == 1 || QDP_USE_SSE2 == 1) && ! defined(__GNUC__)
// SSE requires GNUC
#undef QDP_USE_SSE
#undef QDP_USE_SSE2

#define QDP_USE_SSE   0
#define QDP_USE_SSE2  0
#endif

// Commented this out and set QDP_ALIGNMENT_SIZE to be 16 all the time
// This is a minimal waste of space and should allow an SSE dslash
// to be used even if the QDP itself is not compiled with SSE.
#if 0
// Alignment size: SSE requires a larger alignment
// This should probably move under more compiler specific info
#if QDP_USE_SSE == 1
#define QDP_ALIGNMENT_SIZE  16
#else
#define QDP_ALIGNMENT_SIZE  8
#endif

#else
#define QDP_ALIGNMENT_SIZE 16
#endif
// YUKKY - Eventually get rid of these includes
#include <cstdio>
#include <cstdlib>
#include <ostream>
#include <iostream>

using namespace std;   // I do not like this - fix later

using std::iostream;
using std::ostream;
// END OF YUKKINESS


// Basic includes
#define PETE_USER_DEFINED_EXPRESSION
namespace QDP {
#include <PETE/PETE.h>
}

#include "qdp_init.h"
#include "qdp_forward.h"
#include "qdp_multi.h"

#include "qdp_params.h"
#include "qdp_layout.h"
#include "qdp_io.h"
#include "qdp_qlimewriter.h"
#include "qdp_stdio.h"
#include "qdp_xmlio.h"
#include "qdp_qdpio.h"
#include "qdp_subset.h"
#include "qdp_map.h"
#include "qdp_stopwatch.h"

#include "qdp_traits.h"
#include "qdp_qdpexpr.h"
#include "qdp_qdptype.h"
#include "qdp_qdpsubtype.h"

namespace QDP {
#include "QDPOperators.h"
}

// Include the allocator stuff here, before QDP_outer
#include "qdp_allocator.h"

#include "qdp_newops.h"
#include "qdp_optops.h"
#include "qdp_profile.h"
//#include "qdp_word.h"
#include "qdp_simpleword.h"
#include "qdp_reality.h"
#include "qdp_inner.h"
#include "qdp_primitive.h"
#include "qdp_outer.h"
#include "qdp_outersubtype.h"

#include "qdp_pretty.h"

// Replaces previous ifdef structure. Structure moved into the header file
#include "qdp_defs.h"
#include "qdp_globalfuncs.h"
#include "qdp_specializations.h"

//#include "qdp_special.h"
#include "qdp_random.h"

#ifdef BUILD_CUDP
#include "qdp_cudp_just.h"
#endif

// Include threading code here if applicable
#include "qdp_dispatch.h"

namespace ThreadReductions { 
 
}

#if defined(ARCH_SCALAR)
// Architectural specific code to a single node/single proc box
#warning "Using scalar architecture"
#include "qdp_scalar_specific.h"

// Include SSE code here if applicable
#if QDP_USE_SSE == 1
#include "qdp_scalarsite_sse.h"
#elif QDP_USE_BAGEL_QDP == 1
// USE_BAGEL_QDP
#include "qdp_scalarsite_bagel_qdp.h"
#else
// Use Generics only
#include "qdp_scalarsite_generic.h"
#endif

#elif defined(ARCH_PARSCALAR)
// Architectural specific code to a parallel/single proc box
#warning "Using parallel scalar architecture"
#include "qdp_parscalar_specific.h"

// Include optimized code here if applicable
#if QDP_USE_SSE == 1
#include "qdp_scalarsite_sse.h"
#elif QDP_USE_BAGEL_QDP == 1
// Use BAGEL_QDP 
#include "qdp_scalarsite_bagel_qdp.h"
#else
// Use generics
#include "qdp_scalarsite_generic.h"
#endif

#elif defined(ARCH_SCALARVEC)
// Architectural specific code to a single node/single proc box 
// with vector extension
#warning "Using scalar architecture with vector extensions"
#include "qdp_scalarvec_specific.h"

// Include optimized code here if applicable
#if QDP_USE_SSE == 1
#include "qdp_scalarvecsite_sse.h"
#endif

#elif defined(ARCH_PARSCALARVEC)
// Architectural specific code to a parallel/single proc box
// with vector extension
#warning "Using parallel scalar architecture with vector extensions"
#include "qdp_parscalarvec_specific.h"

// Include optimized code here if applicable
#if QDP_USE_SSE == 1
#include "qdp_scalarvecsite_sse.h"
#endif

#else
#error "Unknown architecture ARCH"
#endif

#include "qdp_flopcount.h"

#endif  // QDP_INCLUDE
