// -*- C++ -*-
//
// QDP data parallel interface
//
// Outer lattice routines specific to a scalar platform 

#ifndef QDP_SCALAR_SPECIFIC_H
#define QDP_SCALAR_SPECIFIC_H


#include "cudp_iface.h"
#include "cudp_newtags.h"



namespace QDP {

// Use separate defs here. This will cause subroutine calls under g++

//-----------------------------------------------------------------------------
// Layout stuff specific to a scalar architecture
namespace Layout
{
  //! coord[mu]  <- mu  : fill with lattice coord in mu direction
    __device__
  LatticeInteger latticeCoordinate(int mu);
}


//-----------------------------------------------------------------------------
// Internal ops designed to look like those in parscalar
// These dummy routines exist just to make code more portable
namespace Internal
{
  //! Dummy array sum accross all nodes
  template<class T>
  __device__ inline void globalSumArray(T* dest, int n) {}

  //! Dummy global sum on a multi1d
  template<class T>
  __device__ inline void globalSumArray(multi1d<T>& dest) {}

  //! Dummy global sum on a multi2d
  template<class T>
  __device__ inline void globalSumArray(multi2d<T>& dest) {}

  //! Dummy sum across all nodes
  template<class T>
  __device__ inline void globalSum(T& dest) {}

  //! Dummy broadcast from primary node to all other nodes
  template<class T>
  __device__ inline void broadcast(T& dest) {}

  //! Dummy broadcast a string from primary node to all other nodes
  //__device__ inline void broadcast_str(std::string& dest) {}

  //! Dummy broadcast from primary node to all other nodes
  __device__ inline void broadcast(void* dest, size_t nbytes) {}
}

/////////////////////////////////////////////////////////
// Threading evaluate with openmp and qmt implementation
//
// by Xu Guo, EPCC, 16 June 2008
/////////////////////////////////////////////////////////

//! user argument for the evaluate function:
// "OLattice Op Scalar(Expression(source)) under an Subset"
//
template<class T, class T1, class Op, class RHS>
struct u_arg{
        OLattice<T>& d;
        const QDPExpr<RHS,OScalar<T1> >& r;
        const Op& op;
        const int *tab;
    __device__
  u_arg( OLattice<T>& d_,
	 const QDPExpr<RHS, OScalar<T1> >& r_,
	 const Op& op_,
	 const int *tab_ ) : d(d_), r(r_), op(op_), tab(tab_) {}
   };

//! user function for the evaluate function:
// "OLattice Op Scalar(Expression(source)) under an Subset"
//
template<class T, class T1, class Op, class RHS>
    __device__
void ev_userfunc(int lo, int hi, int myId, u_arg<T,T1,Op,RHS> *a)
{
   OLattice<T>& dest = a->d;
   const QDPExpr<RHS,OScalar<T1> >&rhs = a->r;
   const int* tab = a->tab;
   const Op& op= a->op;

      
   for(int j=lo; j < hi; ++j)
   {
     int i = tab[j];
     op(dest.elem(i), forEach(rhs, EvalLeaf1(0), OpCombine()));
   }
}


//! user argument for the evaluate function:
// "OLattice Op OLattice(Expression(source)) under an Subset"
//
template<class T, class T1, class Op, class RHS>
struct user_arg{
        OLattice<T>& d;
        const QDPExpr<RHS,OLattice<T1> >& r;
        const Op& op;
        const int *tab;
    __device__
  user_arg(OLattice<T>& d_,
	   const QDPExpr<RHS,OLattice<T1> >& r_,
	   const Op& op_,
	   const int *tab_) : d(d_), r(r_), op(op_), tab(tab_) {}

   };

//! user function for the evaluate function:
// "OLattice Op OLattice(Expression(source)) under an Subset"
//
template<class T, class T1, class Op, class RHS>
    __device__
void evaluate_userfunc(int lo, int hi, int myId, user_arg<T,T1,Op,RHS> *a)
{

   OLattice<T>& dest = a->d;
   const QDPExpr<RHS,OLattice<T1> >&rhs = a->r;
   const int* tab = a->tab;
   const Op& op= a->op;

      
   for(int j=lo; j < hi; ++j)
   {
     int i = tab[j];
     op(dest.elem(i), forEach(rhs, EvalLeaf1(i), OpCombine()));
   }
}

//! include the header file for dispatch
//#include "cudp_dispatch.h"

//-----------------------------------------------------------------------------
//! OLattice Op Scalar(Expression(source)) under an Subset
/*! 
 * OLattice Op Expression, where Op is some kind of binary operation 
 * involving the destination 
 */
template<class T, class T1, class Op, class RHS>
__device__ inline
void evaluate(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs,
	      const Subset& s)
{
//  //cerr << "In evaluateUnorderedSubet(olattice,oscalar)\n";

// #if defined(QDP_USE_PROFILING)   
//   #warning "SCALAR: Using PROPFILING"
//   static QDPProfile_t prof(dest, op, rhs);
//   prof.time -= getClockTime();
// #endif

//   int numSiteTable = s.numSiteTable();
  
//   u_arg<T,T1,Op,RHS> a(dest, rhs, op, s.siteTable().slice());

//   dispatch_to_threads< u_arg<T,T1,Op,RHS> >(numSiteTable, a, ev_userfunc);
 
//   ///////////////////
//   // Original code
//   //////////////////
//   //const int *tab = s.siteTable().slice();
//   //for(int j=0; j < s.numSiteTable(); ++j) 
//   //{
//   //int i = tab[j];
// //    fprintf(stderr,"eval(olattice,oscalar): site %d\n",i);
// //    op(dest.elem(i), forEach(rhs, ElemLeaf(), OpCombine()));
//   //op(dest.elem(i), forEach(rhs, EvalLeaf1(0), OpCombine()));
//   //}

// #if defined(QDP_USE_PROFILING)   
//   prof.time += getClockTime();
//   prof.count++;
//   prof.print();
// #endif
}



//! OLattice Op OLattice(Expression(source)) under an Subset
/*! 
 * OLattice Op Expression, where Op is some kind of binary operation 
 * involving the destination 
 */
template<class T, class T1, class Op, class RHS>
__device__ inline
void evaluate(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs,
	      const Subset& s)
{
  //int linearIndex = (blockIdx.x * blockDim.x + threadIdx.x)*s.threadsite;
  // if (s.hasOrderedRep())
  //   i = linearIndex + s.start();
  // else {
  //   const int *tab = s.siteTable().slice();
  //   i = tab[ linearIndex ];
  // }

  // if (s.Nthread != blockDim.x)
  //   printf("think again!\n");

  int idx_start = 
    blockDim.x * s.threadsite * blockIdx.x + 
    blockDim.x * s.threadsite * gridDim.x * blockIdx.y;

  for (int i = 0 ; i < s.threadsite ; ++i) {
    int idx = idx_start + threadIdx.x + i * blockDim.x;
    if (idx < s.totalsite)   // necessary only if Nblock = .. +1
      op(dest.elem(idx), forEach(rhs, EvalLeaf1(idx), OpCombine()));
    // else
    //   printf("out of range\n");
  }

  // for (int i=0;i<s.threadsite;++i) {
  //   int idx=linearIndex+i+s.start();
  //   if (idx < s.totalsite)
  //     op(dest.elem(idx), forEach(rhs, EvalLeaf1(idx), OpCombine()));
  //   else
  //     printf("out of range\n");
  // }

  // int i;
  // int linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
  // if (s.hasOrderedRep())
  //   i = linearIndex + s.start();
  // else {
  //   const int *tab = s.siteTable().slice();
  //   i = tab[ linearIndex ];
  // }
  // op(dest.elem(i), forEach(rhs, EvalLeaf1(i), OpCombine()));


  //   int numSiteTable = s.numSiteTable();
  //   user_arg<T,T1,Op,RHS> a(dest, rhs, op, s.siteTable().slice());
  //   dispatch_to_threads<user_arg<T,T1,Op,RHS> >(numSiteTable, a, evaluate_userfunc);


  //   ////////////////////
  //   // Original code
  //   ///////////////////

  // const int *tab = s.siteTable().slice();
  // for(int j=0; j < s.numSiteTable(); ++j)
  //   {
  //     int i = tab[j];
  //     //fprintf(stderr,"eval(olattice,olattice): site %d\n",i);
  //     op(dest.elem(i), forEach(rhs, EvalLeaf1(i), OpCombine()));
  //   }


}



//-----------------------------------------------------------------------------
//! dest = (mask) ? s1 : dest
template<class T1, class T2> 
    __device__
void 
copymask(OSubLattice<T2> d, const OLattice<T1>& mask, const OLattice<T2>& s1) 
{
  OLattice<T2>& dest = d.field();
  const Subset& s = d.subset();

  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    copymask(dest.elem(i), mask.elem(i), s1.elem(i));
  }
}


//! dest = (mask) ? s1 : dest
template<class T1, class T2> 
    __device__
void 
copymask(OLattice<T2>& dest, const OLattice<T1>& mask, const OLattice<T2>& s1) 
{
  const int vvol = Layout::vol();
  for(int i=0; i < vvol; ++i) 
    copymask(dest.elem(i), mask.elem(i), s1.elem(i));
}



//-----------------------------------------------------------------------------
// Random numbers
namespace RNG
{
  extern Seed ran_seed;
  extern Seed ran_mult;
  extern Seed ran_mult_n;
  extern LatticeSeed *lattice_ran_mult;
}


//! dest  = random  
/*! This implementation is correct for no inner grid */
template<class T>
    __device__
void 
random(OScalar<T>& d)
{
  Seed seed = RNG::ran_seed;
  Seed skewed_seed = RNG::ran_seed * RNG::ran_mult;

  fill_random(d.elem(), seed, skewed_seed, RNG::ran_mult);

  RNG::ran_seed = seed;  // The seed from any site is the same as the new global seed
}


//! dest  = random    under a subset
template<class T>
    __device__
void 
random(OLattice<T>& d, const Subset& s)
{
  Seed seed;
  Seed skewed_seed;

  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    seed = RNG::ran_seed;
    skewed_seed.elem() = RNG::ran_seed.elem() * RNG::lattice_ran_mult->elem(i);
    fill_random(d.elem(i), seed, skewed_seed, RNG::ran_mult_n);
  }

  RNG::ran_seed = seed;  // The seed from any site is the same as the new global seed
}




//! dest  = random   under a subset
template<class T>
    __device__
void random(OSubLattice<T> dd)
{
  OLattice<T>& d = dd.field();
  const Subset& s = dd.subset();

  random(d,s);
}


//! dest  = random  
template<class T>
    __device__
void random(OLattice<T>& d)
{
  random(d,all);
}


//! dest  = gaussian   under a subset
template<class T>
    __device__
void gaussian(OLattice<T>& d, const Subset& s)
{
  OLattice<T>  r1, r2;

  random(r1,s);
  random(r2,s);

  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    fill_gaussian(d.elem(i), r1.elem(i), r2.elem(i));
  }
}



//! dest  = gaussian   under a subset
template<class T>
    __device__
void gaussian(OSubLattice<T> dd)
{
  OLattice<T>& d = dd.field();
  const Subset& s = dd.subset();

  gaussian(d,s);
}


//! dest  = gaussian
template<class T>
    __device__
void gaussian(OLattice<T>& d)
{
  gaussian(d,all);
}



//-----------------------------------------------------------------------------
// Broadcast operations
//! dest  = 0 
template<class T> 
    __device__
void zero_rep(OLattice<T>& dest, const Subset& s) 
{
  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    zero_rep(dest.elem(i));
  }
}



//! dest  = 0 
template<class T, class S>
    __device__
void zero_rep(OSubLattice<T> dd) 
{
  OLattice<T>& d = dd.field();
  const Subset& s = dd.subset();
  
  zero_rep(d,s);
}


//! dest  = 0 
template<class T> 
    __device__
void zero_rep(OLattice<T>& dest) 
{
  const int vvol = Layout::vol();
  for(int i=0; i < vvol; ++i) 
    zero_rep(dest.elem(i));
}



//-----------------------------------------------
// Global sums
//! OScalar = sum(OScalar) under an explicit subset
/*!
 * Allow a global sum that sums over the lattice, but returns an object
 * of the same primitive type. E.g., contract only over lattice indices
 */
template<class RHS, class T>
typename UnaryReturn<OScalar<T>, FnSum>::Type_t
    __device__
sum(const QDPExpr<RHS,OScalar<T> >& s1, const Subset& s)
{
  typename UnaryReturn<OScalar<T>, FnSum>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  evaluate(d,OpAssign(),s1,all);   // since OScalar, no global sum needed

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}


//! OScalar = sum(OScalar)
/*!
 * Allow a global sum that sums over the lattice, but returns an object
 * of the same primitive type. E.g., contract only over lattice indices
 */
template<class RHS, class T>
typename UnaryReturn<OScalar<T>, FnSum>::Type_t
    __device__
sum(const QDPExpr<RHS,OScalar<T> >& s1)
{
  typename UnaryReturn<OScalar<T>, FnSum>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  evaluate(d,OpAssign(),s1,all);

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}



//! OScalar = sum(OLattice)  under an explicit subset
/*!
 * Allow a global sum that sums over the lattice, but returns an object
 * of the same primitive type. E.g., contract only over lattice indices
 */
template<class RHS, class T>
typename UnaryReturn<OLattice<T>, FnSum>::Type_t
    __device__
sum(const QDPExpr<RHS,OLattice<T> >& s1, const Subset& s)
{
  typename UnaryReturn<OLattice<T>, FnSum>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  // Must initialize to zero since we do not know if the loop will be entered
  zero_rep(d.elem());

  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    d.elem() += forEach(s1, EvalLeaf1(i), OpCombine());   // SINGLE NODE VERSION FOR NOW
  }

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}


//! OScalar = sum(OLattice)
/*!
 * Allow a global sum that sums over the lattice, but returns an object
 * of the same primitive type. E.g., contract only over lattice indices
 */
template<class RHS, class T>
typename UnaryReturn<OLattice<T>, FnSum>::Type_t
    __device__
sum(const QDPExpr<RHS,OLattice<T> >& s1)
{
  typename UnaryReturn<OLattice<T>, FnSum>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  // Loop always entered - could unroll
  zero_rep(d.elem());

  const int vvol = Layout::vol();
  for(int i=0; i < vvol; ++i) 
    d.elem() += forEach(s1, EvalLeaf1(i), OpCombine());   // SINGLE NODE VERSION FOR NOW

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}


//-----------------------------------------------------------------------------
// Multiple global sums 
//! multi1d<OScalar> dest  = sumMulti(OScalar,Set) 
/*!
 * Compute the global sum on multiple subsets specified by Set 
 *
 * This implementation is specific to a purely olattice like
 * types. The scalar input value is replicated to all the
 * slices
 */
template<class RHS, class T>
typename UnaryReturn<OScalar<T>, FnSum>::Type_t
    __device__
sumMulti(const QDPExpr<RHS,OScalar<T> >& s1, const Set& ss)
{
  typename UnaryReturn<OScalar<T>, FnSumMulti>::Type_t  dest(ss.numSubsets());

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(dest[0], OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  // lazy - evaluate repeatedly
  for(int i=0; i < ss.numSubsets(); ++i)
    evaluate(dest[i],OpAssign(),s1,all);

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return dest;
}


//! multi1d<OScalar> dest  = sumMulti(OLattice,Set) 
/*!
 * Compute the global sum on multiple subsets specified by Set 
 *
 * This is a very simple implementation. There is no need for
 * anything fancier unless global sums are just so extraordinarily
 * slow. Otherwise, generalized sums happen so infrequently the slow
 * version is fine.
 */
  template<class RHS, class T>
  struct SumMultiOLatticeThreadArgs {
    const multi1d<int>& lat_color;
    const QDPExpr<RHS,OLattice<T> >& s;
    multi1d<typename UnaryReturn<OLattice<T>, FnSumMulti>::Type_t>& dest;
    SumMultiOLatticeThreadArgs(const multi1d<int>& lat_color_,
			       const QDPExpr<RHS,OLattice<T> >& s_,
			       multi1d<typename UnaryReturn<OLattice<T>, FnSumMulti>::Type_t>& dest_) : lat_color(lat_color_), s(s_), dest(dest_) {}

  };

  template<class RHS, class T>
    __device__
  void sumMultiKernel(int lo, int hi, int my_id, SumMultiOLatticeThreadArgs<RHS,T>* a)
  {
    const multi1d<int>& lat_color = a->lat_color;
    const  QDPExpr<RHS,OLattice<T> >& s=a->s;
    multi1d<typename UnaryReturn<OLattice<T>, FnSumMulti>::Type_t>& dest=a->dest;
    for(int i=lo; i < hi; ++i) { 
      int j = lat_color[i];
      (dest[my_id])[j].elem() += forEach(s, EvalLeaf1(i), OpCombine());   // SINGLE NODE VERSION FOR NOW
    }
  }



#if 0
template<class RHS, class T>
typename UnaryReturn<OLattice<T>, FnSumMulti>::Type_t
    __device__
sumMulti(const QDPExpr<RHS,OLattice<T> >& s1, const Set& ss)
{
  printf("sumMulti not yet implemented\n");
  typename UnaryReturn<OLattice<T>, FnSumMulti>::Type_t  dest(ss.numSubsets());

// #if defined(QDP_USE_PROFILING)   
//   static QDPProfile_t prof(dest[0], OpAssign(), FnSum(), s1);
//   prof.time -= getClockTime();
// #endif

//   multi1d< typename UnaryReturn<OLattice<T>, FnSumMulti>::Type_t > pdest(qdpNumThreads());

//   // Initialize result with zero
//   for(int thread=0; thread < qdpNumThreads(); ++thread) {
//     pdest[thread].resize(ss.numSubsets());

//     for(int k=0; k < ss.numSubsets(); ++k) {
//       zero_rep(pdest[thread][k]);
//     }
//   }

//   // Loop over all sites and accumulate based on the coloring 
//   const multi1d<int>& lat_color =  ss.latticeColoring();
//   SumMultiOLatticeThreadArgs<RHS,T> args(lat_color,s1,pdest);

//   const int vvol = Layout::vol();
//   dispatch_to_threads(vvol, args, sumMultiKernel<RHS,T>);

//   for(int k=0; k< ss.numSubsets(); ++k) { 
//     dest[k] = pdest[0][k];
//   }

//   for(int thread=1; thread < qdpNumThreads(); thread++) { 
//     for(int k=0; k< ss.numSubsets(); ++k) { 
//       dest[k] += pdest[thread][k];
//     }
//   }
  
// #if 0
//   for(int i=0; i < vvol; ++i) 
//   {
//     int j = lat_color[i];
//     dest[j].elem() += forEach(s1, EvalLeaf1(i), OpCombine());   // SINGLE NODE VERSION FOR NOW
//   }
// #endif

// #if defined(QDP_USE_PROFILING)   
//   prof.time += getClockTime();
//   prof.count++;
//   prof.print();
// #endif

  return dest;
}
#endif


#if 1
// Original code
template<class RHS, class T>
typename UnaryReturn<OLattice<T>, FnSumMulti>::Type_t
__device__
sumMulti(const QDPExpr<RHS,OLattice<T> >& s1, const Set& ss)
{
  typename UnaryReturn<OLattice<T>, FnSumMulti>::Type_t  dest(ss.numSubsets());

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(dest[0], OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  // Initialize result with zero
  for(int k=0; k < ss.numSubsets(); ++k)
    zero_rep(dest[k]);

  // Loop over all sites and accumulate based on the coloring 
  const multi1d<int>& lat_color =  ss.latticeColoring();

  const int vvol = Layout::vol();
  for(int i=0; i < vvol; ++i) 
  {
    int j = lat_color[i];
    dest[j].elem() += forEach(s1, EvalLeaf1(i), OpCombine());   // SINGLE NODE VERSION FOR NOW
  }

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return dest;
}
#endif

//-----------------------------------------------------------------------------
// Multiple global sums 
//! multi2d<OScalar> dest  = sumMulti(multi1d<OScalar>,Set) 
/*!
 * Compute the global sum on multiple subsets specified by Set 
 *
 * This implementation is specific to a purely olattice like
 * types. The scalar input value is replicated to all the
 * slices
 */
template<class T>
multi2d<typename UnaryReturn<OScalar<T>, FnSum>::Type_t>
    __device__
sumMulti(const multi1d< OScalar<T> >& s1, const Set& ss)
{
  multi2d<typename UnaryReturn<OScalar<T>, FnSum>::Type_t>  dest(s1.size(),ss.numSubsets());

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(dest(0,0), OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  // lazy - evaluate repeatedly
  for(int i=0; i < dest.size1(); ++i)
    for(int j=0; j < dest.size2(); ++j)
      dest(j,i) = s1[j];

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return dest;
}


//! multi2d<OScalar> dest  = sumMulti(multi1d<OLattice>,Set) 
/*!
 * Compute the global sum on multiple subsets specified by Set 
 *
 * This is a very simple implementation. There is no need for
 * anything fancier unless global sums are just so extraordinarily
 * slow. Otherwise, generalized sums happen so infrequently the slow
 * version is fine.
 */
template<class T>
multi2d<typename UnaryReturn<OLattice<T>, FnSum>::Type_t>
    __device__
sumMulti(const multi1d< OLattice<T> >& s1, const Set& ss)
{
  multi2d<typename UnaryReturn<OLattice<T>, FnSum>::Type_t>  dest(s1.size(),ss.numSubsets());

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(dest(0,0), OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  // Initialize result with zero
  for(int i=0; i < dest.size1(); ++i)
    for(int j=0; j < dest.size2(); ++j)
      zero_rep(dest(j,i));

  // Loop over all sites and accumulate based on the coloring 
  const multi1d<int>& lat_color =  ss.latticeColoring();

  const int vvol = Layout::vol();
  for(int k=0; k < s1.size(); ++k)
  {
    const OLattice<T>& ss1 = s1[k];

    for(int i=0; i < vvol; ++i) 
    {
      int j = lat_color[i];
      dest(k,j).elem() += ss1.elem(i);
    }
  }

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return dest;
}


//-----------------------------------------------------------------------------
//! OScalar = norm2(trace(adj(multi1d<source>)*multi1d<source>))
/*!
 * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
 *
 * Sum over the lattice
 * Allow a global sum that sums over all indices
 */
template<class T>
__device__ inline typename UnaryReturn<OScalar<T>, FnNorm2>::Type_t
norm2(const multi1d< OScalar<T> >& s1)
{
  typename UnaryReturn<OScalar<T>, FnNorm2>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnNorm2(), s1[0]);
  prof.time -= getClockTime();
#endif

  // Possibly loop entered
  zero_rep(d.elem());

  for(int n=0; n < s1.size(); ++n)
  {
    OScalar<T>& ss1 = s1[n];
    d.elem() += localNorm2(ss1.elem());
  }

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}

//! OScalar = sum(OScalar)  under an explicit subset
/*! Discards subset */
template<class T>
__device__ inline typename UnaryReturn<OScalar<T>, FnNorm2>::Type_t
norm2(const multi1d< OScalar<T> >& s1, const Subset& s)
{
  return norm2(s1);
}



//! OScalar = norm2(multi1d<OLattice>) under an explicit subset
/*!
 * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
 *
 * Sum over the lattice
 * Allow a global sum that sums over all indices
 */
template<class T>
__device__ inline typename UnaryReturn<OLattice<T>, FnNorm2>::Type_t
norm2(const multi1d< OLattice<T> >& s1, const Subset& s)
{
  typename UnaryReturn<OLattice<T>, FnNorm2>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnNorm2(), s1[0]);
  prof.time -= getClockTime();
#endif

  // Possibly loop entered
  zero_rep(d.elem());

  const int *tab = s.siteTable().slice();
  for(int n=0; n < s1.size(); ++n)
  {
    const OLattice<T>& ss1 = s1[n];
    for(int j=0; j < s.numSiteTable(); ++j) 
    {
      int i = tab[j];
      d.elem() += localNorm2(ss1.elem(i));
    }
  }

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}



//! OScalar = norm2(multi1d<OLattice>)
/*!
 * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
 *
 * Sum over the lattice
 * Allow a global sum that sums over all indices
 */
template<class T>
__device__ inline typename UnaryReturn<OLattice<T>, FnNorm2>::Type_t
norm2(const multi1d< OLattice<T> >& s1)
{
  return norm2(s1,all);
}



//-----------------------------------------------------------------------------
//! OScalar = innerProduct(multi1d<source1>,multi1d<source2>))
/*!
 * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
 *
 * Sum over the lattice
 * Allow a global sum that sums over all indices
 */
template<class T1, class T2>
__device__ inline typename BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProduct>::Type_t
innerProduct(const multi1d< OScalar<T1> >& s1, const multi1d< OScalar<T2> >& s2)
{
  typename BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProduct>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnInnerProduct(), s1[0]);
  prof.time -= getClockTime();
#endif

  // Possibly loop entered
  zero_rep(d.elem());

  for(int n=0; n < s1.size(); ++n)
  {
    OScalar<T1>& ss1 = s1[n];
    OScalar<T2>& ss2 = s2[n];
    d.elem() += localInnerProduct(ss1.elem(),ss2.elem());
  }

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}

//! OScalar = sum(OScalar)  under an explicit subset
/*! Discards subset */
template<class T1, class T2>
__device__ inline typename BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProduct>::Type_t
innerProduct(const multi1d< OScalar<T1> >& s1, const multi1d< OScalar<T2> >& s2,
	     const Subset& s)
{
  return innerProduct(s1,s2);
}


//! OScalar = innerProduct(multi1d<OLattice>,multi1d<OLattice>) under an explicit subset
/*!
 * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
 *
 * Sum over the lattice
 * Allow a global sum that sums over all indices
 */
template<class T1, class T2>
__device__ inline typename BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProduct>::Type_t
innerProduct(const multi1d< OLattice<T1> >& s1, const multi1d< OLattice<T2> >& s2,
	     const Subset& s)
{
  typename BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProduct>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnInnerProduct(), s1[0]);
  prof.time -= getClockTime();
#endif

  // Possibly loop entered
  zero_rep(d.elem());

  const int *tab = s.siteTable().slice();
  for(int n=0; n < s1.size(); ++n)
  {
    const OLattice<T1>& ss1 = s1[n];
    const OLattice<T2>& ss2 = s2[n];
    for(int j=0; j < s.numSiteTable(); ++j) 
    {
      int i = tab[j];
      d.elem() += localInnerProduct(ss1.elem(i),ss2.elem(i));
    }
  }

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}


//! OScalar = innerProduct(multi1d<OLattice>,multi1d<OLattice>)
/*!
 * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
 *
 * Sum over the lattice
 * Allow a global sum that sums over all indices
 */
template<class T1, class T2>
__device__ inline typename BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProduct>::Type_t
innerProduct(const multi1d< OLattice<T1> >& s1, const multi1d< OLattice<T2> >& s2)
{
  return innerProduct(s1,s2,all);
}



//-----------------------------------------------------------------------------
//! OScalar = innerProductReal(multi1d<source1>,multi1d<source2>))
/*!
 * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
 *
 * Sum over the lattice
 * Allow a global sum that sums over all indices
 */
template<class T1, class T2>
__device__ inline typename BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProductReal>::Type_t
innerProductReal(const multi1d< OScalar<T1> >& s1, const multi1d< OScalar<T2> >& s2)
{
  typename BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProductReal>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnInnerProductReal(), s1[0]);
  prof.time -= getClockTime();
#endif

  // Possibly loop entered
  zero_rep(d.elem());

  for(int n=0; n < s1.size(); ++n)
  {
    OScalar<T1>& ss1 = s1[n];
    OScalar<T2>& ss2 = s2[n];
    d.elem() += localInnerProductReal(ss1.elem(),ss2.elem());
  }

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}

//! OScalar = sum(OScalar)  under an explicit subset
/*! Discards subset */
template<class T1, class T2>
__device__ inline typename BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProductReal>::Type_t
innerProductReal(const multi1d< OScalar<T1> >& s1, const multi1d< OScalar<T2> >& s2,
		 const Subset& s)
{
  return innerProductReal(s1,s2);
}



//! OScalar = innerProductReal(multi1d<OLattice>,multi1d<OLattice>) under an explicit subset
/*!
 * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
 *
 * Sum over the lattice
 * Allow a global sum that sums over all indices
 */
template<class T1, class T2>
__device__ inline typename BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProductReal>::Type_t
innerProductReal(const multi1d< OLattice<T1> >& s1, const multi1d< OLattice<T2> >& s2,
		 const Subset& s)
{
  typename BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProductReal>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnInnerProductReal(), s1[0]);
  prof.time -= getClockTime();
#endif

  // Possibly loop entered
  zero_rep(d.elem());

  const int *tab = s.siteTable().slice();
  for(int n=0; n < s1.size(); ++n)
  {
    const OLattice<T1>& ss1 = s1[n];
    const OLattice<T2>& ss2 = s2[n];
    for(int j=0; j < s.numSiteTable(); ++j) 
    {
      int i = tab[j];
      d.elem() += localInnerProductReal(ss1.elem(i),ss2.elem(i));
    }
  }

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}


//! OScalar = innerProductReal(multi1d<OLattice>,multi1d<OLattice>)
/*!
 * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
 *
 * Sum over the lattice
 * Allow a global sum that sums over all indices
 */
template<class T1, class T2>
__device__ inline typename BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProductReal>::Type_t
innerProductReal(const multi1d< OLattice<T1> >& s1, const multi1d< OLattice<T2> >& s2)
{
  return innerProductReal(s1,s2,all);
}


//-----------------------------------------------
// Global max and min
// NOTE: there are no subset version of these operations. It is very problematic
// and QMP does not support them.
//! OScalar = globalMax(OScalar)
/*!
 * Find the maximum an object has across the lattice
 */
template<class RHS, class T>
typename UnaryReturn<OScalar<T>, FnGlobalMax>::Type_t
globalMax(const QDPExpr<RHS,OScalar<T> >& s1)
{
  typename UnaryReturn<OScalar<T>, FnGlobalMax>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnGlobalMax(), s1);
  prof.time -= getClockTime();
#endif

  evaluate(d,OpAssign(),s1,all);   // since OScalar, no global max needed

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}



//! OScalar = globalMax(OLattice)
/*!
 * Find the maximum an object has across the lattice
 */
template<class RHS, class T>
typename UnaryReturn<OLattice<T>, FnGlobalMax>::Type_t
    __device__
globalMax(const QDPExpr<RHS,OLattice<T> >& s1)
{
  typename UnaryReturn<OLattice<T>, FnGlobalMax>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnGlobalMax(), s1);
  prof.time -= getClockTime();
#endif

  // Loop always entered so unroll
  d.elem() = forEach(s1, EvalLeaf1(0), OpCombine());   // SINGLE NODE VERSION FOR NOW

  const int vvol = Layout::vol();
  for(int i=1; i < vvol; ++i) 
  {
    typename UnaryReturn<T, FnGlobalMax>::Type_t  dd = 
      forEach(s1, EvalLeaf1(i), OpCombine());   // SINGLE NODE VERSION FOR NOW

    if (toBool(dd > d.elem()))
      d.elem() = dd;
  }

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}


//! OScalar = globalMin(OScalar)
/*!
 * Find the minimum an object has across the lattice
 */
template<class RHS, class T>
typename UnaryReturn<OScalar<T>, FnGlobalMin>::Type_t
    __device__
globalMin(const QDPExpr<RHS,OScalar<T> >& s1)
{
  typename UnaryReturn<OScalar<T>, FnGlobalMin>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnGlobalMin(), s1);
  prof.time -= getClockTime();
#endif

  evaluate(d,OpAssign(),s1,all);   // since OScalar, no global min needed

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}



//! OScalar = globalMin(OLattice)
/*!
 * Find the minimum of an object under a subset of the lattice
 */
template<class RHS, class T>
typename UnaryReturn<OLattice<T>, FnGlobalMin>::Type_t
    __device__
globalMin(const QDPExpr<RHS,OLattice<T> >& s1)
{
  typename UnaryReturn<OLattice<T>, FnGlobalMin>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnGlobalMin(), s1);
  prof.time -= getClockTime();
#endif

  // Loop always entered so unroll
  d.elem() = forEach(s1, EvalLeaf1(0), OpCombine());   // SINGLE NODE VERSION FOR NOW

  const int vvol = Layout::vol();
  for(int i=1; i < vvol; ++i) 
  {
    typename UnaryReturn<T, FnGlobalMin>::Type_t  dd = 
      forEach(s1, EvalLeaf1(i), OpCombine());   // SINGLE NODE VERSION FOR NOW

    if (toBool(dd < d.elem()))
      d.elem() = dd;
  }

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}


//-----------------------------------------------------------------------------
// Peek and poke at individual sites. This is very architecture specific
// NOTE: these two routines assume there is no underlying inner grid

//! Extract site element
/*! @ingroup group1
  @param l  source to examine
  @param coord Nd lattice coordinates to examine
  @return single site object of the same primitive type
  @ingroup group1
  @relates QDPType */
template<class T1>
__device__ inline OScalar<T1>
peekSite(const OScalar<T1>& l, const multi1d<int>& coord)
{
  return l;
}

//! Extract site element
/*! @ingroup group1
  @param l  source to examine
  @param coord Nd lattice coordinates to examine
  @return single site object of the same primitive type
  @ingroup group1
  @relates QDPType */
template<class RHS, class T1>
__device__ inline OScalar<T1>
peekSite(const QDPExpr<RHS,OScalar<T1> > & l, const multi1d<int>& coord)
{
  // For now, simply evaluate the expression and then call the function
  typedef OScalar<T1> C1;
  
  return peekSite(C1(l), coord);
}


//! Extract site element
/*! @ingroup group1
  @param l  source to examine
  @param coord Nd lattice coordinates to examine
  @return single site object of the same primitive type
  @ingroup group1
  @relates QDPType */
template<class T1>
__device__ inline OScalar<T1>
peekSite(const OLattice<T1>& l, const multi1d<int>& coord)
{
  OScalar<T1> dest;

  dest.elem() = l.elem(Layout::linearSiteIndex(coord));
  return dest;
}

//! Extract site element
/*! @ingroup group1
  @param l  source to examine
  @param coord Nd lattice coordinates to examine
  @return single site object of the same primitive type
  @ingroup group1
  @relates QDPType */
template<class RHS, class T1>
__device__ inline OScalar<T1>
peekSite(const QDPExpr<RHS,OLattice<T1> > & l, const multi1d<int>& coord)
{
  // For now, simply evaluate the expression and then call the function
  typedef OLattice<T1> C1;
  
  return peekSite(C1(l), coord);
}


//! Insert site element
/*! @ingroup group1
  @param l  target to update
  @param r  source to insert
  @param coord Nd lattice coordinates where to insert
  @return object of the same primitive type but of promoted lattice type
  @ingroup group1
  @relates QDPType */
template<class T1>
__device__ inline OLattice<T1>&
pokeSite(OLattice<T1>& l, const OScalar<T1>& r, const multi1d<int>& coord)
{
  l.elem(Layout::linearSiteIndex(coord)) = r.elem();
  return l;
}


//! Copy data values from field src to array dest
/*! @ingroup group1
  @param dest  target to update
  @param src   QDP source to insert
  @param s     subset
  @ingroup group1
  @relates QDPType */
template<class T>
__device__ inline void 
QDP_extract(multi1d<OScalar<T> >& dest, const OLattice<T>& src, const Subset& s)
{
  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    dest[i].elem() = src.elem(i);
  }
}

//! Inserts data values from site array src.
/*! @ingroup group1
  @param dest  QDP target to update
  @param src   source to insert
  @param s     subset
  @ingroup group1
  @relates QDPType */
template<class T>
__device__ inline void 
QDP_insert(OLattice<T>& dest, const multi1d<OScalar<T> >& src, const Subset& s)
{
  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    dest.elem(i) = src[i].elem();
  }
}


//-----------------------------------------------------------------------------
// Forward declaration
//struct FnMap;
// This is the PETE version of a map, namely return an expression
struct FnMap
{
  PETE_EMPTY_CONSTRUCTORS(FnMap)

  __device__
  FnMap(){}

  mutable int *goff;
  __device__
  FnMap(int *goffsets): goff(goffsets)
    {
//    fprintf(stderr,"FnMap(): goff=0x%x\n",goff);
    }
  
  template<class T>
  __device__ inline typename UnaryReturn<T, FnMap>::Type_t
  operator()(const T &a) const
  {
    return (a);
  }


  __device__
  void unpackNode(void * ptr) const {
#ifdef __CUDA_ARCH__
    goff = (int*)(ptr);
#ifdef GPU_DEBUG
    if (blockIdx.x * blockDim.x + threadIdx.x == 0)
      printf("FnMap::unpackNode %llx \n",ptr);
#endif
#endif
  }




};




//! General permutation map class for communications
class Map
{
public:
  //! Constructor - does nothing really
    __device__
  Map() {}

  //! Destructor
    __device__
  ~Map() {}

  //! Constructor from a function object
    __device__
  Map(const MapFunc& fn) {make(fn);}

  //! Actual constructor from a function object
  /*! The semantics are   source_site = func(dest_site) */
    __device__
  void make(const MapFunc& func);

  //! Function call operator for a shift
  /*! 
   * map(source)
   *
   * Implements:  dest(x) = s1(x+offsets)
   *
   * Shifts on a OLattice are non-trivial.
   * Notice, there may be an ILattice underneath which requires shift args.
   * This routine is very architecture dependent.
   */
  template<class T1,class C1>
  __device__ inline typename MakeReturn<UnaryNode<FnMap,
						  typename CreateLeaf<QDPType<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPType<T1,C1> & l)
  {
    typedef UnaryNode<FnMap,
      typename CreateLeaf<QDPType<T1,C1> >::Leaf_t> Tree_t;
    return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(goffsets.slice()),
     					      CreateLeaf<QDPType<T1,C1> >::make(l)));
  }


  template<class T1,class C1>
  __device__ inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPExpr<T1,C1> & l)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(goffsets.slice()),
	CreateLeaf<QDPExpr<T1,C1> >::make(l)));
    }



public:
  //! Accessor to offsets
    __device__
  const multi1d<int>& Offsets() const {return goffsets;}

private:
  //! Hide copy constructor
    __device__
  Map(const Map&) {}

  //! Hide operator=
    __device__
  void operator=(const Map&) {}

private:
  //! Offset table used for communications. 
  const multi1d<int> goffsets;
};





#if defined(QDP_USE_PROFILING)   
template <>
struct TagVisitor<FnMap, PrintTag> : public ParenPrinter<FnMap>
{ 
  static void visit(FnMap op, PrintTag t) 
    { t.os_m << "shift"; }
};
#endif


// Specialization of ForEach deals with maps. 
template<class A, class CTag>
struct ForEach<UnaryNode<FnMap, A>, EvalLeaf1, CTag>
{
  typedef typename ForEach<A, EvalLeaf1, CTag>::Type_t TypeA_t;
  typedef typename Combine1<TypeA_t, FnMap, CTag>::Type_t Type_t;
  __device__ inline static
  Type_t apply(const UnaryNode<FnMap, A> &expr, const EvalLeaf1 &f, 
    const CTag &c) 
  {
    EvalLeaf1 ff(expr.operation().goff[f.val1()]);
//  fprintf(stderr,"ForEach<Unary<FnMap>>: site = %d, new = %d\n",f.val1(),ff.val1());

    return Combine1<TypeA_t, FnMap, CTag>::
      combine(ForEach<A, EvalLeaf1, CTag>::apply(expr.child(), ff, c),
              expr.operation(), c);
  }
};



template<class A, class CTag>
struct ForEach<UnaryNode<FnMap, A>, FlattenTag, CTag>
{
  typedef typename ForEach<A, FlattenTag, CTag>::Type_t TypeA_t;
  typedef typename Combine1<TypeA_t, FnMap, CTag>::Type_t Type_t;
  __device__
  inline static
  Type_t apply(const UnaryNode<FnMap, A> &expr, const FlattenTag &f, 
	       const CTag &c) 
  {
#ifdef __CUDA_ARCH__
#ifdef GPU_DEBUG
    if (f.count_node >= f.numberNodes) {
      printf("Oops: f.count >= f.numberNodes!\n");
    }
#endif

    expr.operation().unpackNode( f.nodeDataArray[ f.count_node ].pointer );
#ifdef GPU_DEBUG
    if (blockIdx.x * blockDim.x + threadIdx.x == 0)
      printf("Flatten: FnMap     : %d %llx %d\n",f.count_node,f.nodeDataArray[ f.count_node ].pointer);
#endif
    f.count_node++;

    return Combine1<TypeA_t, FnMap, CTag>::
      combine(ForEach<A, FlattenTag, CTag>::apply(expr.child(), f, c),
	      expr.operation(), c);
#endif
  }
};





//-----------------------------------------------------------------------------
//! Array of general permutation map class for communications
class ArrayMap
{
public:
  //! Constructor - does nothing really
    __device__
  ArrayMap() {}

  //! Destructor
    __device__
  ~ArrayMap() {}

  //! Constructor from a function object
    __device__
  ArrayMap(const ArrayMapFunc& fn) {make(fn);}

  //! Actual constructor from a function object
  /*! The semantics are   source_site = func(dest_site,sign) */
    __device__
  void make(const ArrayMapFunc& func);

  //! Function call operator for a shift
  /*! 
   * map(source,dir)
   *
   * Implements:  dest(x) = source(map(x,dir))
   *
   * Shifts on a OLattice are non-trivial.
   * Notice, there may be an ILattice underneath which requires shift args.
   * This routine is very architecture dependent.
   */
  template<class T1,class C1>
  __device__ inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPType<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPType<T1,C1> & l, int dir)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPType<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(mapsa[dir].Offsets().slice()),
	CreateLeaf<QDPType<T1,C1> >::make(l)));
    }


  template<class T1,class C1>
  __device__ inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPExpr<T1,C1> & l, int dir)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(mapsa[dir].Offsets().slice()),
	CreateLeaf<QDPExpr<T1,C1> >::make(l)));
    }


private:
  //! Hide copy constructor
    __device__
  ArrayMap(const ArrayMap&) {}

  //! Hide operator=
    __device__
  void operator=(const ArrayMap&) {}

private:
  multi1d<Map> mapsa;
  
};


//-----------------------------------------------------------------------------
//! Bi-directional version of general permutation map class for communications
class BiDirectionalMap
{
public:
  //! Constructor - does nothing really
    __device__
  BiDirectionalMap() {}

  //! Destructor
    __device__
  ~BiDirectionalMap() {}

  //! Constructor from a function object
    __device__
  BiDirectionalMap(const MapFunc& fn) {make(fn);}

  //! Actual constructor from a function object
  /*! The semantics are   source_site = func(dest_site,sign) */
    __device__
  void make(const MapFunc& func);

  //! Function call operator for a shift
  /*! 
   * map(source,isign)
   *
   * Implements:  dest(x) = source(map(x,isign))
   *
   * Shifts on a OLattice are non-trivial.
   * Notice, there may be an ILattice underneath which requires shift args.
   * This routine is very architecture dependent.
   */
  template<class T1,class C1>
  __device__ inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPType<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPType<T1,C1> & l, int isign)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPType<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(bimaps[(isign+1)>>1].Offsets().slice()),
	CreateLeaf<QDPType<T1,C1> >::make(l)));
    }


  template<class T1,class C1>
  __device__ inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPExpr<T1,C1> & l, int isign)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(bimaps[(isign+1)>>1].Offsets().slice()),
	CreateLeaf<QDPExpr<T1,C1> >::make(l)));
    }


private:
  //! Hide copy constructor
    __device__
  BiDirectionalMap(const BiDirectionalMap&) {}

  //! Hide operator=
    __device__
  void operator=(const BiDirectionalMap&) {}

private:
  multi1d<Map> bimaps;
  
};


//-----------------------------------------------------------------------------
//! Bi-directional version of general permutation map class for communications
class ArrayBiDirectionalMap
{
public:
  //! Constructor - does nothing really
    __device__
  ArrayBiDirectionalMap() {}

  //! Destructor
    __device__
  ~ArrayBiDirectionalMap() {}

  //! Constructor from a function object
    __device__
  ArrayBiDirectionalMap(const ArrayMapFunc& fn) {make(fn);}

  //! Actual constructor from a function object
  /*! The semantics are   source_site = func(dest_site,sign) */
    __device__
  void make(const ArrayMapFunc& func);

  //! Function call operator for a shift
  /*! 
   * map(source,isign,dir)
   *
   * Implements:  dest(x) = source(map(x,isign,dir))
   *
   * Shifts on a OLattice are non-trivial.
   * Notice, there may be an ILattice underneath which requires shift args.
   * This routine is very architecture dependent.
   */
  template<class T1,class C1>
  __device__ inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPType<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPType<T1,C1> & l, int isign, int dir)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPType<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(bimapsa((isign+1)>>1,dir).Offsets().slice()),
	CreateLeaf<QDPType<T1,C1> >::make(l)));
    }


  template<class T1,class C1>
  __device__ inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPExpr<T1,C1> & l, int isign, int dir)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(bimapsa((isign+1)>>1,dir).Offsets().slice()),
	CreateLeaf<QDPExpr<T1,C1> >::make(l)));
    }


private:
  //! Hide copy constructor
    __device__
  ArrayBiDirectionalMap(const ArrayBiDirectionalMap&) {}

  //! Hide operator=
    __device__
  void operator=(const ArrayBiDirectionalMap&) {}

private:
  multi2d<Map> bimapsa;
  
};


//-----------------------------------------------------------------------------
// Input and output of various flavors that are architecture specific

//! Decompose a lexicographic site into coordinates
    __device__
multi1d<int> crtesn(int ipos, const multi1d<int>& latt_size);



} // namespace QDP

#endif
