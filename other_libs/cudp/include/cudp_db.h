// -*- C++ -*-
/*! @file
 * @brief Support for filedb
 */

#ifndef QDP_DB_H
#define QDP_DB_H

#ifdef BUILD_FILEDB

// FileDB is used, so define a real set of classes
#include "cudp_db_imp.h"

#else

// FileDB is NOT used, so define stubs of classes
#include "cudp_db_stub.h"
#endif

#endif
