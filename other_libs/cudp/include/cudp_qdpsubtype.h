// -*- C++ -*-
/*! @file
 * @brief QDPType after a subset
 *
 * Subclass of QDPType used for subset operations
 */

#ifndef QDP_QDPSUBTYPE_H
#define QDP_QDPSUBTYPE_H

namespace QDP {


//! QDPSubType - type representing a field living on a subset
/*! 
 * This class is meant to be an auxilliary class used only for
 * things like lvalues - left hand side of expressions, arguments
 * to calls that modify the source (like RNG), etc.
 */
template<class T, class C> 
class QDPSubType
{
  //! This is a type name like OSubLattice<T> or OSubScalar<T>
  typedef typename QDPSubTypeTrait<C>::Type_t CC;

public:
  //! Default constructor 
  __device__
  QDPSubType() {}

  //! Copy constructor
  __device__
  QDPSubType(const QDPSubType&) {}

  //! Destructor
  __device__
  ~QDPSubType() {}


  //---------------------------------------------------------
  // Operators

  __device__ inline
  void assign(const typename WordType<C>::Type_t& rhs)
    {
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(field(),OpAssign(),PETE_identity(Scalar_t(rhs)),subset());
    }

  __device__ inline
  void assign(const Zero&)
    {
      zero_rep(field(),subset());
    }

  template<class T1,class C1>
  __device__ inline
  void assign(const QDPType<T1,C1>& rhs)
    {
      evaluate(field(),OpAssign(),PETE_identity(rhs),subset());
    }

  template<class T1,class C1>
  __device__ inline
  void assign(const QDPExpr<T1,C1>& rhs)
    {
      evaluate(field(),OpAssign(),rhs,subset());
    }

  __device__ inline
  void operator+=(const typename WordType<C>::Type_t& rhs)
    {
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(field(),OpAddAssign(),PETE_identity(Scalar_t(rhs)),subset());
    }

  template<class T1,class C1>
  __device__ inline
  void operator+=(const QDPType<T1,C1>& rhs)
    {
      evaluate(field(),OpAddAssign(),PETE_identity(rhs),subset());
    }

  template<class T1,class C1>
  __device__ inline
  void operator+=(const QDPExpr<T1,C1>& rhs)
    {
      evaluate(field(),OpAddAssign(),rhs,subset());
    }


  __device__ inline
  void operator-=(const typename WordType<C>::Type_t& rhs)
    {
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(field(),OpSubtractAssign(),PETE_identity(Scalar_t(rhs)),subset());
    }

  template<class T1,class C1>
  __device__ inline
  void operator-=(const QDPType<T1,C1>& rhs)
    {
      evaluate(field(),OpSubtractAssign(),PETE_identity(rhs),subset());
    }

  template<class T1,class C1>
  __device__ inline
  void operator-=(const QDPExpr<T1,C1>& rhs)
    {
      evaluate(field(),OpSubtractAssign(),rhs,subset());
    }


  __device__ inline
  void operator*=(const typename WordType<C>::Type_t& rhs)
    {
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(field(),OpMultiplyAssign(),PETE_identity(Scalar_t(rhs)),subset());
    }

  template<class T1,class C1>
  __device__ inline
  void operator*=(const QDPType<T1,C1>& rhs)
    {
      evaluate(field(),OpMultiplyAssign(),PETE_identity(rhs),subset());
    }

  template<class T1,class C1>
  __device__ inline
  void operator*=(const QDPExpr<T1,C1>& rhs)
    {
      evaluate(field(),OpMultiplyAssign(),rhs,subset());
    }


  __device__ inline
  void operator/=(const typename WordType<C>::Type_t& rhs)
    {
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(field(),OpDivideAssign(),PETE_identity(Scalar_t(rhs)),subset());
    }

  template<class T1,class C1>
  __device__ inline
  void operator/=(const QDPType<T1,C1>& rhs)
    {
      evaluate(field(),OpDivideAssign(),PETE_identity(rhs),subset());
    }

  template<class T1,class C1>
  __device__ inline
  void operator/=(const QDPExpr<T1,C1>& rhs)
    {
      evaluate(field(),OpDivideAssign(),rhs,subset());
    }


  __device__ inline
  void operator%=(const typename WordType<C>::Type_t& rhs)
    {
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(field(),OpModAssign(),PETE_identity(Scalar_t(rhs)),subset());
    }

  template<class T1,class C1>
  __device__ inline
  void operator%=(const QDPType<T1,C1>& rhs)
    {
      evaluate(field(),OpModAssign(),PETE_identity(rhs),subset());
    }

  template<class T1,class C1>
  __device__ inline
  void operator%=(const QDPExpr<T1,C1>& rhs)
    {
      evaluate(field(),OpModAssign(),rhs,subset());
    }


  __device__ inline
  void operator|=(const typename WordType<C>::Type_t& rhs)
    {
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(field(),OpBitwiseOrAssign(),PETE_identity(Scalar_t(rhs)),subset());
    }

  template<class T1,class C1>
  __device__ inline
  void operator|=(const QDPType<T1,C1>& rhs)
    {
      evaluate(field(),OpBitwiseOrAssign(),PETE_identity(rhs),subset());
    }

  template<class T1,class C1>
  __device__ inline
  void operator|=(const QDPExpr<T1,C1>& rhs)
    {
      evaluate(field(),OpBitwiseOrAssign(),PETE_identity(rhs),subset());
    }


  __device__ inline
  void operator&=(const typename WordType<C>::Type_t& rhs)
    {
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(field(),OpBitwiseAndAssign(),PETE_identity(Scalar_t(rhs)),subset());
    }

  template<class T1,class C1>
  __device__ inline
  void operator&=(const QDPType<T1,C1>& rhs)
    {
      evaluate(field(),OpBitwiseAndAssign(),PETE_identity(rhs),subset());
    }

  template<class T1,class C1>
  __device__ inline
  void operator&=(const QDPExpr<T1,C1>& rhs)
    {
      evaluate(field(),OpBitwiseAndAssign(),rhs,subset());
    }


  __device__ inline
  void operator^=(const typename WordType<C>::Type_t& rhs)
    {
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(field(),OpBitwiseXorAssign(),PETE_identity(Scalar_t(rhs)));
    }

  template<class T1,class C1>
  __device__ inline
  void operator^=(const QDPType<T1,C1>& rhs)
    {
      evaluate(field(),OpBitwiseXorAssign(),PETE_identity(rhs),subset());
    }

  template<class T1,class C1>
  __device__ inline
  void operator^=(const QDPExpr<T1,C1>& rhs)
    {
      evaluate(field(),OpBitwiseXorAssign(),rhs,subset());
    }


  __device__ inline
  void operator<<=(const typename WordType<C>::Type_t& rhs)
    {
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(field(),OpLeftShiftAssign(),PETE_identity(Scalar_t(rhs)),subset());
    }

  template<class T1,class C1>
  __device__ inline
  void operator<<=(const QDPType<T1,C1>& rhs)
    {
      evaluate(field(),OpLeftShiftAssign(),PETE_identity(rhs),subset());
    }

  template<class T1,class C1>
  __device__ inline
  void operator<<=(const QDPExpr<T1,C1>& rhs)
    {
      evaluate(field(),OpLeftShiftAssign(),rhs,subset());
    }


  __device__ inline
  void operator>>=(const typename WordType<C>::Type_t& rhs)
    {
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(field(),OpRightShiftAssign(),PETE_identity(Scalar_t(rhs)),subset());
    }

  template<class T1,class C1>
  __device__ inline
  void operator>>=(const QDPType<T1,C1>& rhs)
    {
      evaluate(field(),OpRightShiftAssign(),PETE_identity(rhs),subset());
    }

  template<class T1,class C1>
  __device__ inline
  void operator>>=(const QDPExpr<T1,C1>& rhs)
    {
      evaluate(field(),OpRightShiftAssign(),rhs,subset());
    }

private:
  //! Hide default operator=
  __device__ inline
  C& operator=(const QDPSubType& rhs) {return static_cast<CC*>(this)->field();}


public:
  __device__
  C& field() {return static_cast<CC*>(this)->field();}
  __device__
  const Subset& subset() const {return static_cast<const CC*>(this)->subset();}

};


} // namespace QDP

#endif



