// -*- C++ -*-
// $Id: cudp_random.h,v 1.6 2007/06/10 14:32:09 edwards Exp $
//
// QDP data parallel interface
//
// Random number support

#ifndef QDP_RANDOM_H
#define QDP_RANDOM_H

namespace QDP {


//! Random number generator namespace
/*!
 * A collection of routines and data for supporting random numbers
 * 
 * It is a linear congruential with modulus m = 2**47, increment c = 0,
 * and multiplier a = (2**36)*m3 + (2**24)*m2 + (2**12)*m1 + m0.  
 */

namespace RNG
{
  //! Default initialization of the RNG
  /*! Uses arbitrary internal seed to initialize the RNG */
  __device__
  void initDefaultRNG(void);

  //! Initialize the internals of the RNG
  __device__
  void initRNG(void);

  //! Initialize the RNG seed
  /*!
   * Seeds are big-ints
   */
  __device__
  void setrn(const Seed& lseed);

  //! Recover the current RNG seed
  /*!
   * Seeds are big-ints
   */
  __device__
  void savern(Seed& lseed);


  //! Internal seed multiplier
  __device__
  float sranf(Seed& seed, Seed&, const Seed&);

  //! Internal seed multiplier
  __device__
  void sranf(float* d, int N, Seed& seed, ILatticeSeed&, const Seed&);
}

//! dest  = random
template<class T1, class T2>
inline void
fill_random(float& d, T1& seed, T2& skewed_seed, const T1& seed_mult)
{
  d = float(RNG::sranf(seed, skewed_seed, seed_mult));
}

//! dest  = random
template<class T1, class T2>
__device__ inline void
fill_random(double& d, T1& seed, T2& skewed_seed, const T1& seed_mult)
{
  d = double(RNG::sranf(seed, skewed_seed, seed_mult));
}


//! dest  = random
template<class T1, class T2, int N>
__device__ inline void
fill_random(float* d, T1& seed, T2& skewed_seed, const T1& seed_mult)
{
  RNG::sranf(d, N, seed, skewed_seed, seed_mult);
}


//! dest  = random
template<class T1, class T2, int N>
__device__ inline void
fill_random(double* d, T1& seed, T2& skewed_seed, const T1& seed_mult)
{
  float dd[N];
  RNG::sranf(dd, N, seed, skewed_seed, seed_mult);
  for(int i=0; i < N; ++i)
    d[i] = float(dd[i]);
}

} // namespace QDP

#endif
