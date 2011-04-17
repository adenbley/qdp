// -*- C++ -*-
// $Id: cudp_primitive.h,v 1.5 2007/06/10 14:32:09 edwards Exp $

/*! \file
 * \brief Primitive classes
 *
 * Primitives are the various types on the fibers at the lattice sites
 */


#ifndef QDP_PRIMITIVE_H
#define QDP_PRIMITIVE_H


/*! \defgroup fiber Fiber only types and operations
 * \ingroup fiberbundle
 *
 * Primitives are the various types on the fibers at the lattice sites.
 *
 * The primitive indices, including Reality (also known as complex or real),
 * is represented as a tensor product over various vector spaces. Different
 * kinds of object can transform in those vector spaces, like Scalar, Vector, and
 * Matrix.
 */

#include "cudp_primscalar.h"
#include "cudp_primmatrix.h"
#include "cudp_primvector.h"
#include "cudp_primseed.h"
#include "cudp_primcolormat.h"
#include "cudp_primcolorvec.h"
#include "cudp_primgamma.h"
#include "cudp_primspinmat.h"
#include "cudp_primspinvec.h"

#endif
