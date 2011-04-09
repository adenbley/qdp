// -*- C++ -*-
// $Id: cudp_default_allocator.h,v 1.7 2007/06/10 14:32:08 edwards Exp $

/*! \file
 * \brief Default memory allocator for QDP
 *
 */

#ifndef QDP_DEFAULT_ALLOCATOR
#define QDP_DEFAULT_ALLOCATOR

#include "cudp_allocator.h"
//#include "cudp_stdio.h"
//#include "cudp_singleton.h"
//#include <string>
//#include <map>

namespace QDP
{
  namespace Allocator
  {

    void* allocate(size_t n_bytes);
    void free( void *mem);
    const size_t& getPoolSize();

    // Specialise allocator to the default case
    // class QDPDefaultAllocator {
    // private:
    //   // Disallow Copies
    //   QDPDefaultAllocator(const QDPDefaultAllocator& c) {}

    //   // Disallow assignments (copies by another name)
    //   void operator=(const QDPDefaultAllocator& c) {}

    //   // Disallow creation / destruction by anyone except 
    //   // the singleton CreateUsingNew policy which is a "friend"
    //   // I don't like friends but this follows Alexandrescu's advice
    //   // on p154 of Modern C++ Design (A. Alexandrescu)
    //   QDPDefaultAllocator() {}
    //   ~QDPDefaultAllocator() {}

    // public:

    //   volatile void*
    //   allocate(size_t n_bytes);
    //   //allocate(size_t n_bytes,const MemoryPoolHint& mem_pool_hint);

    //   void 
    //   free(volatile void *mem);

    // };

    // typedef QDPDefaultAllocator theQDPAllocator;

  } // namespace Allocator
} // namespace QDP

#endif
