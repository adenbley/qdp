// -*- C++ -*- 
/*! file 
 * \brief QDP Typedef switcharoo 
 *
 * This file acts as a switchbox for including cudp_scalarsite_defs.h
 * or cudp_scalarvecsite_defs.h for common datatypes depending on the 
 * parallel architecture chosen.
 *
 * The purpose of this file is so that this switching 
 *   i) Done in a single place only  AND
 *  ii) Other QDP++ header files can include this file with impunity 
 *      (it'll have an include guard)
 */

#ifndef QDP_DEFS_H
#define QDP_DEFS_H

#include <cudp_config.h>
#include "cudp_precision.h"

#if defined(ARCH_SCALAR) || defined(ARCH_PARSCALAR)
#include "cudp_scalarsite_defs.h"

#elif defined(ARCH_SCALARVEC) || defined(ARCH_PARSCALARVEC)
#include "cudp_scalarvecsite_defs.h"

#else
#error "Unknown architecture ARCH"
#endif

#endif
