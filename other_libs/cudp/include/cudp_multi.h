// -*- C++ -*-
// $Id: cudp_multi.h,v 1.20 2009/06/18 19:30:15 edwards Exp $

/*! @file
 * @brief Multi-dimensional arrays
 * 
 * Support for reference semantic multi-dimensional arrays
 */

#ifndef MULTI_INCLUDE
#define MULTI_INCLUDE

#include "cudp_config.h"
namespace QDP {

/*! @defgroup multi  Multi-dimensional arrays
 *
 * Container classes that provide 1D, 2D, 3D and 4D multidimensional
 * array semantics.
 *
 * @{
 */

//! Container for a multi-dimensional 1D array
template<class T> class multi1d
{
public:
  // Basic cosntructor. Null (0x0) array_pointer, no copymem, no fast memory
  __device__
  multi1d() {F=0;n1=0;copymem=false;fast_mem_hint=false;}

  // Placement constructor. Copy pointer, copymem=true, fast_mem_hint is false
  __device__
  multi1d(T *f, int ns1) {F=f; n1=ns1;copymem=true; fast_mem_hint=false;}

  // Explicit constructor, copymem is false, fast_mem is false, call resize
  __device__
  explicit multi1d(int ns1) {copymem=false;F=0;fast_mem_hint=false;resize(ns1);}
  // Destructor
  __device__
  ~multi1d() {
    // If not created with placement, delete array
    if (! copymem) {
      delete[] F;
    }
  }


  //! Copy constructor
  // Copy from s, into slow memory
  __device__
  multi1d(const multi1d& s): copymem(false), fast_mem_hint(false), n1(s.n1), F(0)
    {
      resize(n1);

      for(int i=0; i < n1; ++i)
	F[i] = s.F[i];
    }

  //! Resize routine, call a templated resize, using *this to disambiguate
  // template type
  __device__
  void resize(int ns1) { resize(*this, ns1); }


  //! Size of array
  __device__
  int size() const {return n1;}
  __device__
  int size1() const {return n1;}

  //! Equal operator uses underlying = of T
  /*! Default = */
  __device__
  multi1d& operator=(const multi1d& s1)
    {
      if (size() != s1.size())   // a simple check avoids resizing always
	resize(s1.size());

      for(int i=0; i < n1; ++i)
	F[i] = s1.F[i];
      return *this;
    }

  // fw
  template< typename T2>
  __device__
  multi1d& operator=(const multi1d<T2>& s1)
    {
      if (size() != s1.size())   
	resize(s1.size());

      for(int i=0; i < n1; ++i)
	F[i] = s1[i];
      return *this;
    }

  //! Equal operator uses underlying = of T
  template<class T1>
  __device__
  multi1d<T>& operator=(const T1& s1)
    {
      if (F == 0)
      {
	//cerr << "multi1d: left hand side not initialized in =" << endl;
	exit(1);
      }

      for(int i=0; i < n1; ++i)
	F[i] = s1;
      return *this;
    }

  //! Set equal to a old-style C 1-D array
  __device__
  multi1d<T>& operator=(const T* s1)
    {
      if (F == 0)
      {
	//cerr << "multi1d: left hand side not initialized in =" << endl;
	exit(1);
      }

      for(int i=0; i < n1; ++i)
	F[i] = s1[i];
      return *this;
    }

  //! Add-replace on each element
  /*! Uses underlying += */
  __device__
  multi1d<T>& operator+=(const multi1d<T>& s1)
    {
      if (size() != s1.size())
      {
	//cerr << "multi1d: Sizes incompatible in +=" << endl;
	exit(1);
      }

      for(int i=0; i < n1; ++i)
	F[i] += s1.F[i];
      return *this;
    }

  //! Add-replace on each element
  /*! Uses underlying += */
  __device__
  multi1d<T>& operator+=(const T& s1)
    {
      if (F == 0)
      {
	//cerr << "multi1d: left hand side not initialized in +=" << endl;
	exit(1);
      }

      for(int i=0; i < n1; ++i)
	F[i] += s1;
      return *this;
    }

  //! Subtract-replace on each element
  /*! Uses underlying -= */
  __device__
  multi1d<T>& operator-=(const multi1d<T>& s1)
    {
      if (size() != s1.size())
      {
	//cerr << "multi1d: Sizes incompatible in -=" << endl;
	exit(1);
      }

      for(int i=0; i < n1; ++i)
	F[i] -= s1.F[i];
      return *this;
    }

  //! Subtract-replace on each element
  /*! Uses underlying -= */
  __device__
  multi1d<T>& operator-=(const T& s1)
    {
      if (F == 0)
      {
	//cerr << "multi1d: left hand side not initialized in -=" << endl;
	exit(1);
      }

      for(int i=0; i < n1; ++i)
	F[i] -= s1;
      return *this;
    }

  //! Mult-replace on each element
  /*! Uses underlying *= */
  __device__
  multi1d<T>& operator*=(const multi1d<T>& s1)
    {
      if (size() != s1.size())
      {
	//cerr << "multi1d: Sizes incompatible in *=" << endl;
	exit(1);
      }

      for(int i=0; i < n1; ++i)
	F[i] *= s1.F[i];
      return *this;
    }

  //! Mult-replace on each element
  /*! Uses underlying *= */
  __device__
  multi1d<T>& operator*=(const T& s1)
    {
      if (F == 0)
      {
	//cerr << "multi1d: left hand side not initialized in *=" << endl;
	exit(1);
      }

      for(int i=0; i < n1; ++i)
	F[i] *= s1;
      return *this;
    }

  //! Divide-replace on each element
  /*! Uses underlying /= */
  __device__
  multi1d<T>& operator/=(const multi1d<T>& s1)
    {
      if (size() != s1.size())
      {
	//cerr << "multi1d: Sizes incompatible in /=" << endl;
	exit(1);
      }

      for(int i=0; i < n1; ++i)
	F[i] /= s1.F[i];
      return *this;
    }

  //! Divide-replace on each element
  /*! Uses underlying /= */
  __device__
  multi1d<T>& operator/=(const T& s1)
    {
      if (F == 0)
      {
	//cerr << "multi1d: left hand side not initialized in /=" << endl;
	exit(1);
      }

      for(int i=0; i < n1; ++i)
	F[i] /= s1;
      return *this;
    }

  //! Return ref to a column slice
  __device__
  T* slice() const {return F;}

  //! Return ref to an element
  __device__
  T& operator()(int i) {return F[i];}

  //! Return const ref to an element
  __device__
  const T& operator()(int i) const {return F[i];}

  //! Return ref to an element
  __device__
  T& operator[](int i) {return F[i];}

  //! Return const ref to an element
  __device__
  const T& operator[](int i) const {return F[i];}

  //! moveToFastMemoryHint for the whole multi1d if the 
  //! internal type T supports it. Calls templated
  //! moveToFastMemoryHint, using *this to disambiguate type
  __device__ inline void moveToFastMemoryHint(bool copy=false) {
    moveToFastMemoryHint(*this, copy);
  }

  //! revertFromFastMemoryHint for the whole multi1d if the 
  //! internal type T supports it. Calls templated revertFrom
  //! fast memory hint, using *this to disambiguate template type.
  __device__ inline void revertFromFastMemoryHint(bool copy=false) { 
    revertFromFastMemoryHint(*this, copy);
  }

private:
  //! Resize for most types
  template<typename I>
  __device__
  void resize(multi1d<I>& disambiguator, int ns1)
  {
    if(copymem) {
      //cerr <<"multi1d: invalid resize of a copy of memory" << endl;
      exit(1);
    }
    delete [] F;
    n1=ns1;
    F = new T[n1];
    if ( F == 0x0 ) { 
      QDP_error_exit("Unable to allocate memory in multi1d::resize(%d)\n",ns1);
    }
  }

  //! Catchall case for things that dont support Fast Memory Hints
  //! does nothing and should be __device__ inlined away.
  template<typename I>
  __device__ inline void moveToFastMemoryHint(multi1d<I>& disambiguator, bool copy=false) {}

  //! Catchall case for things that dont support Fast Memory Hints
  //! does nothing and should be __device__ inlined away
  template<typename I>
  __device__ inline void revertFromFastMemoryHint(multi1d<I>& disambiguator, bool copy=false) {}

#ifdef QDP_USE_QCDOC
  //! Special resize for multi1d<OLattice> which may have a resize hint
  //! in effect
  template<typename I>
  __device__
  void resize(multi1d<OLattice<I> >& disambiguator, int ns1)
  {
     if(copymem) {
       //cerr <<"multi1d: invalid resize of a copy of memory" << endl;
       exit(1);
     }
     delete[] F; 
     n1=ns1;      
     F = new T[n1];
     if ( F == 0x0 ) { 
       QDP_error_exit("Unable to allocate memory in multi1d::resize()\n");
     }
     // If fast mem hint is in effect, call moveToFastMemoryHint
     // On elements
     if( fast_mem_hint ) { 
       for(int i=0; i < ns1; i++) { 
	 F[i].moveToFastMemoryHint(false);
       }
     }
  }
#endif

#ifdef QDP_USE_QCDOC
  //! Specialised case for multi1d<OLattice> Objects, convenience function to 
  //! revert from fastMemory for the whole array.
  template<typename I>
  __device__ inline void revertFromFastMemoryHint(multi1d<OLattice<I> >& disambiguator, bool copy=false ) { 
    if( fast_mem_hint ) { 
      for(int i=0; i < n1; i++) { 
	F[i].revertFromFastMemoryHint(copy);
      }
      fast_mem_hint=false;
    }
    else {}
  }
#endif

#ifdef QDP_USE_QCDOC
  //! Specialised case for multi1d<OLattice> Objects, convenience function 
  //! to moveToFastMemoryHint for the whole array. Enables fast_mem_hint
  //! so that subsequent resizes will repeat the hint (useful in some 
  //! places where there is an indiscriminate resize on the object.
  template<typename I>
  __device__ inline void moveToFastMemoryHint(multi1d< OLattice<I> >& disambiguator, bool copy=false) {
    if( ! fast_mem_hint ) { 
      for(int i=0; i < n1; i++) { 
	F[i].moveToFastMemoryHint(copy);
      }
      fast_mem_hint=true;
    }
    else {}
  }
#endif

  bool copymem;
  bool fast_mem_hint;
  int n1;
  T *F;

public:
  void setF(void *ptr) {
    F = (T*)(ptr);
  }

};



//---------------------------------------------------------------
// Comparisons/recombinations

//! Concatenate two Array's
template<typename T>
__device__ inline multi1d<T> concat(const multi1d<T>& l, const multi1d<T>& r)
{
  multi1d<int> nz(l.size() + r.size());
  int j = 0;
  for(int i=0; i < l.size(); ++i)
    nz[j++] = l[i];
  
  for(int i=0; i < r.size(); ++i)
    nz[j++] = r[i];

  return nz;
}

//! Check if two Array's are the same
template<typename T>
__device__ inline bool operator==(const multi1d<T>& n1, const multi1d<T>& n2)
{
  if (n1.size() == 0 || n1.size() != n2.size())
    return false;
    
  for(int i=0; i < n1.size(); ++i)
    if (n2[i] != n1[i])
      return false;

  return true;
}
  
//! Check if two Array's are no the same
template<typename T>
__device__ inline bool operator!=(const multi1d<T>& n1, const multi1d<T>& n2)
{
  return ! (n1 == n2);
}

//! a < b
/*! This definition follows that of string comparison */
template<typename T>
__device__ inline bool operator<(const multi1d<T>& a, const multi1d<T>& b)
{
  int  len = (a.size() < b.size()) ? a.size() : b.size();

  for(int i=0; i < len; ++i)
  {
    if (a[i] != b[i])
      return (a[i] < b[i]) ? true : false;
  }
    
  return (a.size() == b.size()) ? false : (a.size() < b.size()) ? true : false;
}

//! a > b
/*! This definition follows that of string comparison */
template<typename T>
__device__ inline bool operator>(const multi1d<T>& a, const multi1d<T>& b)
{
  int  len = (a.size() < b.size()) ? a.size() : b.size();

  for(int i=0; i < len; ++i)
  {
    if (a[i] != b[i])
      return (a[i] > b[i]) ? true : false;
  }
    
  return (a.size() == b.size()) ? false : (a.size() > b.size()) ? true : false;
}

//! a <= b
/*! This definition follows that of string comparison */
template<typename T>
__device__ inline bool operator<=(const multi1d<T>& a, const multi1d<T>& b)
{
  return (a < b) || (a == b);
}

//! a >= b
/*! This definition follows that of string comparison */
template<typename T>
__device__ inline bool operator>=(const multi1d<T>& a, const multi1d<T>& b)
{
  return (a > b) || (a == b);
}


//---------------------------------------------------------------
// Basic math support
//
//! add Arrays
template< typename T> 
__device__ inline
multi1d<T> operator+(const multi1d<T>& a, const multi1d<T>& b)
{
  multi1d<T> c(a); 
  c+=b;
  return c;
}
  
//! subtract Arrays
template< typename T> 
__device__ inline
multi1d<T> operator-(const multi1d<T>& a, const multi1d<T>& b)
{
  multi1d<T> c(a); 
  c-=b;
  return c;
}
  
//! multiply Arrays
template< typename T> 
__device__ inline
multi1d<T> operator*(const multi1d<T>& a, const multi1d<T>& b)
{
  multi1d<T> c(a); 
  c*=b;
  return c;
}
  
//!divide Arrays
template< typename T> 
__device__ inline
multi1d<T> operator/(const multi1d<T>& a, const multi1d<T>& b)
{
  multi1d<T> c(a); 
  c/=b;
  return c;
}

//! scalar + Array
template< typename T> 
__device__ inline
multi1d<T> operator+(const T& s, const multi1d<T>& a)
{
  multi1d<T> c(a); 
  c+=s;
  return c;
}

//! Array + scalar
template< typename T> 
__device__ inline
multi1d<T> operator+(const multi1d<T>& a, const T& s)
{
  multi1d<T> c(a); 
  c+=s;
  return c;
}
  
//! scalar - Array
template< typename T> 
__device__ inline
multi1d<T> operator-(const T& s, const multi1d<T>& a)
{
  multi1d<T> c(-a); 
  c+=s;
  return c;
}
//! Array - scalar
template< typename T> 
__device__ inline
multi1d<T> operator-(const multi1d<T>& a, const T& s)
{
  multi1d<T> c(a); 
  c-=s;
  return c;
}

//! scalar * Array
template< typename T> 
__device__ inline
multi1d<T> operator*(const T& s, const multi1d<T>& a)
{
  multi1d<T> c(a); 
  c*=s;
  return c;
}

//! Array * scalar
template< typename T> 
__device__ inline
multi1d<T> operator*(const multi1d<T>& a, const T& s)
{
  multi1d<T> c(a); 
  c*=s;
  return c;
}

//! scalar / Array
template< typename T> 
__device__ inline
multi1d<T> operator/(const T& s, const multi1d<T>& a)
{
  multi1d<T> c(a.size());
  c = s;
  c/= a;
  return c;
}

//! Array / scalar
template< typename T> 
__device__ inline
multi1d<T> operator/(const multi1d<T>& a, const T& s)
{
  multi1d<T> c(a); 
  c/=s;
  return c;
}

//! sqrt
template< typename T> 
__device__ inline
multi1d<T> sqrt(const multi1d<T>& a)
{
  multi1d<T> c(a.size()); 
  for(int i(0);i<a.size();i++)
  {
    T tt;
    tt = a[i];
    c[i] = sqrt(a[i]);
  }
  return c;
}

//! log
template< typename T> 
__device__ inline
multi1d<T> log(const multi1d<T>& a)
{
  multi1d<T> c(a.size()); 
  for(int i(0);i<a.size();i++)
  {
    T tt;
    tt = a[i];
    c[i] = log(a[i]);
  }
  return c;
}

//! sin
template< typename T> 
__device__ inline
multi1d<T> sin(const multi1d<T>& a)
{
  multi1d<T> c(a.size()); 
  for(int i(0);i<a.size();i++)
  {
    T tt;
    tt = a[i];
    c[i] = sin(a[i]);
  }
  return c;
}


//! cos
template< typename T> 
__device__ inline
multi1d<T> cos(const multi1d<T>& a)
{
  multi1d<T> c(a.size()); 
  for(int i(0);i<a.size();i++)
  {
    T tt;
    tt = a[i];
    c[i] = cos(a[i]);
  }
  return c;
}

//! tan
template< typename T> 
__device__ inline
multi1d<T> tan(const multi1d<T>& a)
{
  multi1d<T> c(a.size()); 
  for(int i(0);i<a.size();i++)
  {
    T tt;
    tt = a[i];
    c[i] = tan(a[i]);
  }
  return c;
}

//! asin
template< typename T> 
__device__ inline
multi1d<T> asin(const multi1d<T>& a)
{
  multi1d<T> c(a.size()); 
  for(int i(0);i<a.size();i++)
  {
    T tt;
    tt = a[i];
    c[i] = asin(a[i]);
  }
  return c;
}


//! acos
template< typename T> 
__device__ inline
multi1d<T> acos(const multi1d<T>& a)
{
  multi1d<T> c(a.size()); 
  for(int i(0);i<a.size();i++)
  {
    T tt;
    tt = a[i];
    c[i] = acos(a[i]);
  }
  return c;
}

//! atan
template< typename T> 
__device__ inline
multi1d<T> atan(const multi1d<T>& a)
{
  multi1d<T> c(a.size()); 
  for(int i(0);i<a.size();i++)
  {
    T tt;
    tt = a[i];
    c[i] = atan(a[i]);
  }
  return c;
}



//! norm2 of an array
template< typename T> 
__device__ inline
T norm2(const multi1d<T>& a)
{
  T nn = a[0]*a[0];  // assumes at least 1 element
  for(int i=1; i < a.size(); ++i)
    nn += a[i]*a[i];

  return nn;
}




//------------------------------------------------------------------------------------------
//! Container for a multi-dimensional 2D array
template<class T> class multi2d
{
public:
  multi2d() {F=0;n1=n2=sz=0;copymem=false;}
  multi2d(T *f, int ns2, int ns1) {F=f; n1=ns1; n2=ns2; sz=n1*n2; copymem=true;}
  explicit multi2d(int ns2, int ns1) {copymem=false;F=0;resize(ns2,ns1);}
  ~multi2d() {if (! copymem) {delete[] F;}}

  //! Copy constructor
  multi2d(const multi2d& s): copymem(false), n1(s.n1), n2(s.n2), sz(s.sz), F(0)
    {
      resize(n2,n1);

      for(int i=0; i < sz; ++i)
	F[i] = s.F[i];
    }

  //! Allocate mem for the array
  void resize(int ns2, int ns1) {
    if(copymem) {
      //cerr <<"multi2d: invalid resize of a copy of memory" << endl;
      exit(1);
    }  
    delete[] F; 
    n1=ns1; 
    n2=ns2;  
    sz=n1*n2; 
    F = new T[sz];
    if( F == 0x0 ) { 
      QDP_error_exit("Unable to new memory in multi2d::resize(%d,%d)\n",ns2,ns1);
    }
  }

  //! Size of array
  int size1() const {return n1;}
  int size2() const {return n2;}

  //! Equal operator uses underlying = of T
  multi2d<T>& operator=(const multi2d<T>& s1)
    {
      resize(s1.size2(), s1.size1());   // always resize

      for(int i=0; i < sz; ++i)
	F[i] = s1.F[i];
      return *this;
    }

  //! Equal operator uses underlying = of T
  template<class T1>
  multi2d<T>& operator=(const T1& s1)
    {
      if (F == 0)
      {
	//cerr << "multi2d: left hand side not initialized in =" << endl;
	exit(1);
      }

      for(int i=0; i < sz; ++i)
	F[i] = s1;
      return *this;
    }

  //! Return ref to a row slice
  const T* slice(int j) const {return F+n1*j;}

  //! Return ref to an element
  T& operator()(int j, int i) {return F[i+n1*j];}

  //! Return const ref to an element
  const T& operator()(int j, int i) const {return F[i+n1*j];}

  //! Return ref to an element
  multi1d<T> operator[](int j) {return multi1d<T>(F+j*n1,n1);}

  //! Return const ref to an element
  const multi1d<T> operator[](int j) const {return multi1d<T>(F+j*n1,n1);}

private:
  bool copymem;
  int n1;
  int n2;
  int sz;
  T *F;
};



//------------------------------------------------------------------------------------------
//! Container for a multi-dimensional 3D array
template<class T> class multi3d
{
public:
  multi3d() {F=0;n1=n2=n3=sz=0;copymem=false;}
  multi3d(T *f, int ns3, int ns2, int ns1) {F=f; n1=ns1; n2=ns2; n3=ns3; sz=n1*n2*n3; copymem=true;}
  explicit multi3d(int ns3, int ns2, int ns1) {copymem=false;F=0;resize(ns3,ns2,ns1);}
  ~multi3d() {if (! copymem) {delete[] F;}}

  //! Copy constructor
  multi3d(const multi3d& s): copymem(false), n1(s.n1), n2(s.n2), n3(s.n3), sz(s.sz), F(0)
    {
      resize(n3,n2,n1);

      for(int i=0; i < sz; ++i)
	F[i] = s.F[i];
    }

  //! Allocate mem for the array 
  void resize(int ns3, int ns2, int ns1) 
  {
    if(copymem) {
      //cerr <<"multi3d: invalid resize of a copy of memory" << endl;
      exit(1);
    }

    // Only delete if the array is not NULL. If it is NULL
    // deleting may be bad
    if ( F != 0x0 ) {
      delete[] F; 
    }

    n1=ns1; n2=ns2; n3=ns3; sz=n1*n2*n3; F = new T[sz];
    if( F == 0x0 ) { 
      QDP_error_exit("Unable to new memory in multi3d::resize(%d,%d,%d)\n",ns3,ns2,ns1);
    }
  }

  //! Size of array
  int size1() const {return n1;}
  int size2() const {return n2;}
  int size3() const {return n3;}

  //! Equal operator uses underlying = of T
  multi3d<T>& operator=(const multi3d<T>& s1)
    {
      resize(s1.size3(), s1.size2(), s1.size1());

      for(int i=0; i < sz; ++i)
	F[i] = s1.F[i];
      return *this;
    }

  //! Equal operator uses underlying = of T
  template<class T1>
  multi3d<T>& operator=(const T1& s1)
    {
      if (F == 0)
      {
	//cerr << "multi3d: left hand side not initialized in =" << endl;
	exit(1);
      }

      for(int i=0; i < sz; ++i)
	F[i] = s1;
      return *this;
    }

  //! Return ref to a column slice
  const T* slice(int k, int j) const {return F+n1*(j+n2*(k));}

  //! Return ref to an element
  T& operator()(int k, int j, int i) {return F[i+n1*(j+n2*(k))];}

  //! Return const ref to an element
  const T& operator()(int k, int j, int i) const {return F[i+n1*(j+n2*(k))];}

  //! Return ref to an element
  multi2d<T> operator[](int k) {return multi2d<T>(F+n1*n2*k,n2,n1);}

  //! Return const ref to an element
  const multi2d<T> operator[](int k) const {return multi2d<T>(F+n1*n2*k,n2,n1);}

private:
  bool copymem;
  int n1;
  int n2;
  int n3;
  int sz;
  T *F;
};


//! Container for a generic N dimensional array
template<class T> class multiNd
{
public:
  multiNd() {F=0;}
  explicit multiNd(const multi1d<int>& _nz) {F=0;resize(_nz);}
  ~multiNd() {delete[] F;}

  //! Copy constructor
  multiNd(const multiNd& s): nz(s.nz), sz(s.sz), F(0)
    {
      resize(nz);

      for(int i=0; i < sz; ++i)
	F[i] = s.F[i];
    }

  //! Allocate mem for the array
  void resize(const multi1d<int>& _nz) 
    {
      delete[] F; 
      nz = _nz;
      sz = nz[0];
      for(int i=1; i < nz.size(); ++i)
	sz *= nz[i];
      F = new T[sz];
      if ( F==0x0 ) { 
	QDP_error_exit("Unable to new memory in multiNd::resize()\n");
      }
    }

  //! Size of i-th array index. Indices run from left to right in operator() 
  /*! Note, the last/right index is the fastest varying index */
  int size(int i) const {return nz[i];}

  //! Size of an array containing sizes of each index.
  /*! Note, the last/right index is the fastest varying index */
  const multi1d<int>& size() const {return nz;}

  //! Number of elements in the array
  /*! The number of elements is the product of the sizes */
  int numElem() const {return sz;}

  //! Equal operator uses underlying = of T
  multiNd<T>& operator=(const multiNd<T>& s1)
    {
      resize(s1.size());

      for(int i=0; i < sz; ++i)
	F[i] = s1.F[i];
      return *this;
    }

  //! Equal operator uses underlying = of T
  template<class T1>
  multiNd<T>& operator=(const T1& s1)
    {
      if (F == 0)
      {
	//cerr << "multiNd: left hand side not initialized in =" << endl;
	exit(1);
      }

      for(int i=0; i < sz; ++i)
	F[i] = s1;
      return *this;
    }

  //! Return ref to an element
  T& operator()(int i)
    {
      if (nz.size() != 1)
      {
	//cerr << "multiNd: improper rank of array indices" << endl;
	exit(1);
      }

      return F[i];
    }

  //! Return const ref to an element
  const T& operator()(int i) const
    {
      if (nz.size() != 1)
      {
	//cerr << "multiNd: improper rank of array indices" << endl;
	exit(1);
      }

      return F[i];
    }

  //! Return ref to an element
  T& operator()(int j, int i)
    {
      if (nz.size() != 2)
      {
	//cerr << "multiNd: improper rank of array indices" << endl;
	exit(1);
      }

      return F[i+nz[0]*j];
    }

  //! Return const ref to an element
  const T& operator()(int j, int i) const
    {
      if (nz.size() != 2)
      {
	//cerr << "multiNd: improper rank of array indices" << endl;
	exit(1);
      }

      return F[i+nz[0]*j];
    }

  //! Return ref to an element
  T& operator()(int k, int j, int i) 
    {
      if (nz.size() != 3)
      {
	//cerr << "multiNd: improper rank of array indices" << endl;
	exit(1);
      }

      return F[i+nz[0]*(j+nz[1]*(k))];
    }

  //! Return const ref to an element
  const T& operator()(int k, int j, int i) const
    {
      if (nz.size() != 3)
      {
	//cerr << "multiNd: improper rank of array indices" << endl;
	exit(1);
      }

      return F[i+nz[0]*(j+nz[1]*(k))];
    }

  //! Return ref to an element
  T& operator()(int l, int k, int j, int i) 
    {
      if (nz.size() != 4)
      {
	//cerr << "multiNd: improper rank of array indices" << endl;
	exit(1);
      }

      return F[i+nz[0]*(j+nz[1]*(k+nz[2]*l))];
    }

  //! Return const ref to an element
  const T& operator()(int l, int k, int j, int i) const
    {
      if (nz.size() != 4)
      {
	//cerr << "multiNd: improper rank of array indices" << endl;
	exit(1);
      }

      return F[i+nz[0]*(j+nz[1]*(k+nz[2]*l))];
    }

  //! Return ref to an element via indices packed in a multi1d array
  T& operator[](const multi1d<int>& ind)
    {
      if (ind.size() != nz.size())
      {
	//cerr << "multiNd: improper rank of array indices" << endl;
	exit(1);
      }

      int off = ind[0];
      for(int i=1; i < nz.size(); ++i)
	off = off*nz[i] + ind[i];

      return F[off];
    }

  //! Return ref to an element via indices packed in a multi1d array
  const T& operator[](const multi1d<int>& ind) const
    {
      if (ind.size() != nz.size())
      {
	//cerr << "multiNd: improper rank of array indices" << endl;
	exit(1);
      }

      int off = ind[0];
      for(int i=1; i < nz.size(); ++i)
	off = off*nz[i] + ind[i];

      return F[off];
    }

  //! Return ref to an element with index flattened over indices
  /*! Right index is fastest varying */
  T& getElem(int off)
    {
      if (off < 0 || off >= sz)
      {
	//cerr << "multiNd: index out of bounds" << endl;
	exit(1);
      }

      return F[off];
    }

  //! Return const-ref to an element with index flattened over indices
  /*! Right index is fastest varying */
  const T& getElem(int off) const
    {
      if (off < 0 || off >= sz)
      {
	//cerr << "multiNd: index out of bounds" << endl;
	exit(1);
      }

      return F[off];
    }

private:
  multi1d<int> nz;
  int sz;
  T *F;
};

/*! @} */  // end of group multi

}

#endif
