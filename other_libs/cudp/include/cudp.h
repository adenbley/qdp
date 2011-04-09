// -*- C++ -*-
// $Id: qdp.h,v 1.63 2009/04/17 00:46:36 bjoo Exp $

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

#include <qdp_config.h>
#include "qdp_precision.h"


#define PETE_USER_DEFINED_EXPRESSION 

const int Nd = QDP_ND;
const int Nc = QDP_NC;
const int Ns = QDP_NS;



#include "cudp_precision.h"

#define QDP_ALIGN8   __attribute__ ((aligned (8)))
#define QDP_ALIGN16  __attribute__ ((aligned (16)))
#define QDP_INLINE   __attribute__ ((always_inline))
#define QDP_CONST
#define QDP_CINLINE  __attribute__ ((always_inline,const,pure))
#define QDP_ALIGNMENT_SIZE  128

// #include<iostream>
// #include<cstdlib>

// #include<cstring>

#include<stdio.h>
#include<stdlib.h>
#include<string.h>

//#include<new>
#include<cmath>
#include<assert.h>




using namespace std;   // I do not like this - fix later

namespace QDP {
#include <PETE/cuPETE.h>
}

#include "cudp_util.h"
// #include "cudp_init.h"
#include "cudp_forward.h"
#include "cudp_multi.h"


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
#include "QDPOperators.h"
}



//#include "cudp_cell.h"

#include "cudp_allocator.h"
#include "cudp_newops.h"
#include "cudp_optops.h"
// #include "cudp_profile.h"
#include "cudp_simpleword.h"
#include "cudp_reality.h"
#include "cudp_inner.h"
#include "cudp_primitive.h"
#include "cudp_outer.h"
// #include "cudp_outersubtype.h"
#include "cudp_defs.h"
#include "cudp_globalfuncs.h"
#include "cudp_specializations.h"
#include "cudp_random.h"
#include "cudp_dispatch.h"



#include "cudp_scalar_specific.h"



//#include "cudp_flopcount.h"



#endif  // QDP_INCLUDE
