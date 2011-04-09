// -*- C++ -*-
// $Id: cudp_qdpsubtype.h,v 1.3 2007/06/10 14:32:09 edwards Exp $

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
template<class T, class C, class S> 
class QDPSubType
{
  //! This is a type name like OSubLattice<T,S> or OSubScalar<T,S>
  typedef typename QDPSubTypeTrait<C,S>::Type_t CC;

public:
  //! Default constructor 
  QDPSubType() {}

  //! Copy constructor
  QDPSubType(const QDPSubType&) {}

  //! Destructor
  ~QDPSubType() {}


  //---------------------------------------------------------
  // Operators

  inline
  void assign(const typename WordType<C>::Type_t& rhs)
    {
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(field(),OpAssign(),PETE_identity(Scalar_t(rhs)),subset());
    }

  inline
  void assign(const Zero&)
    {
      zero_rep(field(),subset());
    }

  template<class T1,class C1>
  inline
  void assign(const QDPType<T1,C1>& rhs)
    {
      evaluate(field(),OpAssign(),PETE_identity(rhs),subset());
    }

  template<class T1,class C1>
  inline
  void assign(const QDPExpr<T1,C1>& rhs)
    {
      evaluate(field(),OpAssign(),rhs,subset());
    }

  inline
  void operator+=(const typename WordType<C>::Type_t& rhs)
    {
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(field(),OpAddAssign(),PETE_identity(Scalar_t(rhs)),subset());
    }

  template<class T1,class C1>
  inline
  void operator+=(const QDPType<T1,C1>& rhs)
    {
      evaluate(field(),OpAddAssign(),PETE_identity(rhs),subset());
    }

  template<class T1,class C1>
  inline
  void operator+=(const QDPExpr<T1,C1>& rhs)
    {
      evaluate(field(),OpAddAssign(),rhs,subset());
    }


  inline
  void operator-=(const typename WordType<C>::Type_t& rhs)
    {
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(field(),OpSubtractAssign(),PETE_identity(Scalar_t(rhs)),subset());
    }

  template<class T1,class C1>
  inline
  void operator-=(const QDPType<T1,C1>& rhs)
    {
      evaluate(field(),OpSubtractAssign(),PETE_identity(rhs),subset());
    }

  template<class T1,class C1>
  inline
  void operator-=(const QDPExpr<T1,C1>& rhs)
    {
      evaluate(field(),OpSubtractAssign(),rhs,subset());
    }


  inline
  void operator*=(const typename WordType<C>::Type_t& rhs)
    {
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(field(),OpMultiplyAssign(),PETE_identity(Scalar_t(rhs)),subset());
    }

  template<class T1,class C1>
  inline
  void operator*=(const QDPType<T1,C1>& rhs)
    {
      evaluate(field(),OpMultiplyAssign(),PETE_identity(rhs),subset());
    }

  template<class T1,class C1>
  inline
  void operator*=(const QDPExpr<T1,C1>& rhs)
    {
      evaluate(field(),OpMultiplyAssign(),rhs,subset());
    }


  inline
  void operator/=(const typename WordType<C>::Type_t& rhs)
    {
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(field(),OpDivideAssign(),PETE_identity(Scalar_t(rhs)),subset());
    }

  template<class T1,class C1>
  inline
  void operator/=(const QDPType<T1,C1>& rhs)
    {
      evaluate(field(),OpDivideAssign(),PETE_identity(rhs),subset());
    }

  template<class T1,class C1>
  inline
  void operator/=(const QDPExpr<T1,C1>& rhs)
    {
      evaluate(field(),OpDivideAssign(),rhs,subset());
    }


  inline
  void operator%=(const typename WordType<C>::Type_t& rhs)
    {
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(field(),OpModAssign(),PETE_identity(Scalar_t(rhs)),subset());
    }

  template<class T1,class C1>
  inline
  void operator%=(const QDPType<T1,C1>& rhs)
    {
      evaluate(field(),OpModAssign(),PETE_identity(rhs),subset());
    }

  template<class T1,class C1>
  inline
  void operator%=(const QDPExpr<T1,C1>& rhs)
    {
      evaluate(field(),OpModAssign(),rhs,subset());
    }


  inline
  void operator|=(const typename WordType<C>::Type_t& rhs)
    {
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(field(),OpBitwiseOrAssign(),PETE_identity(Scalar_t(rhs)),subset());
    }

  template<class T1,class C1>
  inline
  void operator|=(const QDPType<T1,C1>& rhs)
    {
      evaluate(field(),OpBitwiseOrAssign(),PETE_identity(rhs),subset());
    }

  template<class T1,class C1>
  inline
  void operator|=(const QDPExpr<T1,C1>& rhs)
    {
      evaluate(field(),OpBitwiseOrAssign(),PETE_identity(rhs),subset());
    }


  inline
  void operator&=(const typename WordType<C>::Type_t& rhs)
    {
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(field(),OpBitwiseAndAssign(),PETE_identity(Scalar_t(rhs)),subset());
    }

  template<class T1,class C1>
  inline
  void operator&=(const QDPType<T1,C1>& rhs)
    {
      evaluate(field(),OpBitwiseAndAssign(),PETE_identity(rhs),subset());
    }

  template<class T1,class C1>
  inline
  void operator&=(const QDPExpr<T1,C1>& rhs)
    {
      evaluate(field(),OpBitwiseAndAssign(),rhs,subset());
    }


  inline
  void operator^=(const typename WordType<C>::Type_t& rhs)
    {
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(field(),OpBitwiseXorAssign(),PETE_identity(Scalar_t(rhs)));
    }

  template<class T1,class C1>
  inline
  void operator^=(const QDPType<T1,C1>& rhs)
    {
      evaluate(field(),OpBitwiseXorAssign(),PETE_identity(rhs),subset());
    }

  template<class T1,class C1>
  inline
  void operator^=(const QDPExpr<T1,C1>& rhs)
    {
      evaluate(field(),OpBitwiseXorAssign(),rhs,subset());
    }


  inline
  void operator<<=(const typename WordType<C>::Type_t& rhs)
    {
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(field(),OpLeftShiftAssign(),PETE_identity(Scalar_t(rhs)),subset());
    }

  template<class T1,class C1>
  inline
  void operator<<=(const QDPType<T1,C1>& rhs)
    {
      evaluate(field(),OpLeftShiftAssign(),PETE_identity(rhs),subset());
    }

  template<class T1,class C1>
  inline
  void operator<<=(const QDPExpr<T1,C1>& rhs)
    {
      evaluate(field(),OpLeftShiftAssign(),rhs,subset());
    }


  inline
  void operator>>=(const typename WordType<C>::Type_t& rhs)
    {
      typedef typename SimpleScalar<typename WordType<C>::Type_t>::Type_t  Scalar_t;
      evaluate(field(),OpRightShiftAssign(),PETE_identity(Scalar_t(rhs)),subset());
    }

  template<class T1,class C1>
  inline
  void operator>>=(const QDPType<T1,C1>& rhs)
    {
      evaluate(field(),OpRightShiftAssign(),PETE_identity(rhs),subset());
    }

  template<class T1,class C1>
  inline
  void operator>>=(const QDPExpr<T1,C1>& rhs)
    {
      evaluate(field(),OpRightShiftAssign(),rhs,subset());
    }

private:
  //! Hide default operator=
  inline
  C& operator=(const QDPSubType& rhs) {}


public:
  C& field() {return static_cast<CC*>(this)->field();}
  const S& subset() const {return static_cast<const CC*>(this)->subset();}

};


} // namespace QDP

#endif



