// -*- C++ -*-
// $Id: cudp_outer.h,v 1.54 2009/10/16 10:25:00 edwards Exp $

#ifndef QDP_OUTER_H
#define QDP_OUTER_H

#include "cudp_config.h"
#include "cudp_allocator.h"

/*! \file
 * \brief Outer grid classes
 */

namespace QDP {

/*! \defgroup fiberbundle Fiberbundle types and operations
 *
 * A fiberbundle is a base space (here a lattice or scalar) with a fiber
 * at each site. Mathematically, we represent a fiber as a tensor product
 * over various vector spaces that have some indices.
 * QDP constructs types via a nested composition of lattice and fiber types.
 */


/*! \addtogroup oscalar Outer grid scalar
 * \ingroup fiberbundle
 *
 * Outer grid scalar means sites are not the slowest varying index. There can
 * still be an Inner grid.
 *
 * @{
 */

//! Outer grid Scalar class */
/*! All outer lattices are of OScalar or OLattice type */
template<class T>
class OScalar: public QDPType<T, OScalar<T> >
{
public:

  OScalar() {}
  ~OScalar() {}

  //---------------------------------------------------------
  //! construct dest = const
  OScalar(const typename WordType<T>::Type_t& rhs)
    {
      typedef typename InternalScalar<T>::Type_t  Scalar_t;
      elem() = Scalar_t(rhs);
    }


  //! construct dest = 0
  OScalar(const Zero& rhs)
    {
      this->assign(rhs);
    }


  //! conversion by constructor  OScalar<T> = OScalar<T1>
  template<class T1>
  OScalar(const OScalar<T1>& rhs)
    {
      this->assign(rhs);
    }


  //! conversion by constructor  OScalar = Expr
  template<class RHS, class T1>
  OScalar(const QDPExpr<RHS, OScalar<T1> >& rhs)
    {
      this->assign(rhs);
    }


  //---------------------------------------------------------
  // Operators
  // NOTE: all this->assignment-like operators except operator= are
  // inherited from QDPType

  inline
  OScalar& operator=(const typename WordType<T>::Type_t& rhs)
    {
      return this->assign(rhs);
    }

  inline
  OScalar& operator=(const Zero& rhs)
    {
      return this->assign(rhs);
    }

  template<class T1,class C1>
  inline
  OScalar& operator=(const QDPType<T1,C1>& rhs)
    {
      return this->assign(rhs);
    }

  template<class T1,class C1>
  inline
  OScalar& operator=(const QDPExpr<T1,C1>& rhs)
    {
      return this->assign(rhs);
    }


  //! Use this for default operator=
  inline
  OScalar& operator=(const OScalar& rhs)
    {
      return this->assign(rhs);
    }


  //---------------------------------------------------------
  // Subsets
  OSubScalar<T,Subset> operator[](const Subset& s) 
    {return OSubScalar<T,Subset>(*this,s);}

  //---------------------------------------------------------
  //! Deep copy constructor
  OScalar(const OScalar& a): F(a.F) {/*fprintf(stderr,"copy OScalar\n");*/}


public:
  inline T& elem() {return F;}
  inline const T& elem() const {return F;}

  inline T& elem(int i) {return F;}  // The indexing is a nop
  inline const T& elem(int i) const {return F;}  // The indexing is a nop

private:
  T F;
};


// //! Ascii input
// /*! Treat all istreams here like all nodes can read. To use specialized ones
//  *  that can broadcast, use TextReader */
// template<class T>
// istream& operator>>(istream& s, OScalar<T>& d)
// {
//   return s >> d.elem();
// }

// //! Ascii output
// /*! Treat all ostreams here like all nodes can write. To use specialized ones
//  *  that can broadcast, use TextReader */
// template<class T>
// inline
// ostream& operator<<(ostream& s, const OScalar<T>& d)
// {
//   return s << d.elem();
// }

// //! Text output
// template<class RHS, class T1>
// ostream& operator<<(ostream& s, const QDPExpr<RHS, OScalar<T1> >& l)
// {
//   typedef OScalar<T1> C1;
//   return s << C1(l);
// }

// //! Text input
// template<class T>
// TextReader& operator>>(TextReader& txt, OScalar<T>& d)
// {
//   return txt >> d.elem();
// }

// //! Text input
// template<class T>
// StandardInputStream& operator>>(StandardInputStream& is, OScalar<T>& d)
// {
//   return is >> d.elem();
// }

// //! Text output
// template<class T>
// inline
// TextWriter& operator<<(TextWriter& txt, const OScalar<T>& d)
// {
//   return txt << d.elem();
// }

// //! Text output
// template<class T>
// StandardOutputStream& operator<<(StandardOutputStream& s, const OScalar<T>& d)
// {
//   return s << d.elem();
// }

// //! Text output
// template<class RHS, class T1>
// StandardOutputStream& operator<<(StandardOutputStream& s, const QDPExpr<RHS, OScalar<T1> >& l)
// {
//   typedef OScalar<T1> C1;
//   return s << C1(l);
// }

// //! XML output
// /*! Supports also having an inner grid */
// template<class T>
// inline
// XMLWriter& operator<<(XMLWriter& xml, const OScalar<T>& d)
// {
//   return xml << d.elem();
// }

// //! XML input
// /*! Supports also having an inner grid */
// template<class T>
// inline
// void read(XMLReader& xml, const string& path, OScalar<T>& d)
// {
//   read(xml, path, d.elem());
// }


/*! @} */  // end of group oscalar


//! OScalar Op OScalar(Expression(source))
/*! 
 * OScalar Op Expression, where Op is some kind of binary operation 
 * involving the destination 
 */
// template<class T, class T1, class Op, class RHS>
// inline
// __device__
// void evaluate(OScalar<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs,
// 	      const Subset& s)
// {
//   // Subset is not used at this level. It may be needed, though, within an inner operation
//   op(dest.elem(), forEach(rhs, ElemLeaf(), OpCombine()));
// }


//-------------------------------------------------------------------------------------
/*! \addtogroup olattice Lattice outer grid 
 * \ingroup fiberbundle
 *
 * Outer grid lattice means sites are slowest varying index. There can
 * still be an Inner grid.
 *
 * @{
 */

//! Outer grid Lattice type
/*! All outer lattices are of OScalar or OLattice type */
template<class T> 
class OLattice: public QDPType<T, OLattice<T> >
{
public:
  __device__
  OLattice() {}

  typedef T Type_t;

  // __device__
  // OLattice( const unsigned int & _e , const int & _buffer_fold ): elements(_e)
  //   {
  //     assert( _buffer_fold == 2 || _buffer_fold == 3 || _buffer_fold == 4 );
  //     alloc_mem( _buffer_fold );
  //   }

  __device__
  ~OLattice();


  //---------------------------------------------------------
  //! conversion by constructor  OLattice<T> = OScalar<T1>
  template<class T1>
  __device__
  OLattice(const OScalar<T1>& rhs)
    {
      // alloc_mem("construct from OScalar");
      // this->assign(rhs);
    }


  //! conversion by constructor  OLattice<T> = OLattice<T1>
  template<class T1>
  __device__
  OLattice(const OLattice<T1>& rhs)
    {
      // alloc_mem("construct from OLattice");
      // this->assign(rhs);
    }


  //! conversion by constructor  OLattice = Expr
  template<class RHS, class T1>
  __device__
  OLattice(const QDPExpr<RHS, OLattice<T1> >& rhs)
    {
      // alloc_mem("construct from expr");
      // this->assign(rhs);
    }


  //! construct OLattice = const
  __device__
  OLattice(const typename WordType<T>::Type_t& rhs)
    {
      // alloc_mem("construct from const");

      // typedef OScalar<typename InternalScalar<T>::Type_t>  Scalar_t;
      // this->assign(Scalar_t(rhs));
    }


  //! construct OLattice = 0
  __device__
  OLattice(const Zero& rhs)
    {
      // alloc_mem("construct from zero");
      // this->assign(rhs);
    }

  //---------------------------------------------------------
  // Operators
  // NOTE: all assignment-like operators except operator= are
  // inherited from QDPType

  inline
  __device__
  OLattice& operator=(const typename WordType<T>::Type_t& rhs)
    {
      return this->assign(rhs);
    }

  inline
  __device__
  OLattice& operator=(const Zero& rhs)
    {
      return this->assign(rhs);
    }

  template<class T1,class C1>
  inline
  __device__
  OLattice& operator=(const QDPType<T1,C1>& rhs)
    {
      return this->assign(rhs);
    }

  template<class T1,class C1>
  inline
  __device__
  OLattice& operator=(const QDPExpr<T1,C1>& rhs)
    {
      return this->assign(rhs);
    }

  inline
  __device__
  OLattice& operator=(const OLattice& rhs)
    {
      return this->assign(rhs);
    }



  __device__
  OSubLattice<T,Subset> operator[](const Subset& s) 
    {return OSubLattice<T,Subset>(*this,s);}


  __device__
  OLattice(const OLattice& rhs)
    {
      // alloc_mem("copy");
      // this->assign(rhs);
    }


public:
  __device__
  inline T* getF() const {return F;}

  
public:
  __device__
  inline  T& elem(int i) {return F[i];}
  __device__
  inline const T& elem(int i) const {return F[i];}


private:

  __device__
  void alloc_mem(const int& buffer_fold ) 
  {
  }

  __device__
  void free_mem() 
  {
  }



private:
  T *F; 

};



template<class T> 
__device__
OLattice<T>::~OLattice()     
{
}








//-----------------------------------------------------------------------------
// We need to specialize CreateLeaf<T> for our class, so that operators
// know what to stick in the leaves of the expression tree.
//-----------------------------------------------------------------------------

template<class T>
struct CreateLeaf<OScalar<T> >
{
//  typedef OScalar<T> Leaf_t;
  typedef Reference<OScalar<T> > Leaf_t;
  inline static
  Leaf_t make(const OScalar<T> &a) { return Leaf_t(a); }
};

template<class T>
struct CreateLeaf<OLattice<T> >
{
//  typedef OLattice<T> Leaf_t;
  typedef Reference<OLattice<T> > Leaf_t;
  inline static
  Leaf_t make(const OLattice<T> &a) { return Leaf_t(a); }
};

//-----------------------------------------------------------------------------
// Specialization of LeafFunctor class for applying the EvalLeaf1
// tag to a OScalar and OLattice. The apply method simply returns the array
// evaluated at the point.
//-----------------------------------------------------------------------------

// Empty leaf functor tag
struct ElemLeaf
{
  inline ElemLeaf() { }
};

template<class T>
struct LeafFunctor<OScalar<T>, ElemLeaf>
{
//  typedef T Type_t;
  typedef Reference<T> Type_t;
  inline static Type_t apply(const OScalar<T> &a, const ElemLeaf &f)
    {return Type_t(a.elem());}
};

template<class T>
struct LeafFunctor<OScalar<T>, EvalLeaf1>
{
//  typedef T Type_t;
  typedef Reference<T> Type_t;
  inline static Type_t apply(const OScalar<T> &a, const EvalLeaf1 &f)
    {return Type_t(a.elem());}
};

template<class T>
struct LeafFunctor<OLattice<T>, EvalLeaf1>
{
//  typedef T Type_t;
  typedef Reference<T> Type_t;
  inline static Type_t apply(const OLattice<T> &a, const EvalLeaf1 &f)
    {return Type_t(a.elem(f.val1()));}
};


//-----------------------------------------------------------------------------
// Traits classes to support operations of simple scalars (floating constants, 
// etc.) on QDPTypes
//-----------------------------------------------------------------------------

template<class T>
struct WordType<OScalar<T> > 
{
  typedef typename WordType<T>::Type_t  Type_t;
};

template<class T>
struct WordType<OLattice<T> > 
{
  typedef typename WordType<T>::Type_t  Type_t;
};

template<class T> 
struct SinglePrecType<OScalar<T> >
{
  typedef OScalar<typename SinglePrecType<T>::Type_t> Type_t;
};

template<class T>
struct DoublePrecType<OScalar<T> >
{
  typedef OScalar<typename DoublePrecType<T>::Type_t> Type_t;
};

template<class T> 
struct SinglePrecType<OLattice<T> >
{
  typedef OLattice<typename SinglePrecType<T>::Type_t> Type_t;
};

template<class T>
struct DoublePrecType<OLattice<T> >
{
  typedef OLattice<typename DoublePrecType<T>::Type_t> Type_t;
};

// Internally used scalars
template<class T>
struct InternalScalar<OScalar<T> > {
  typedef OScalar<typename InternalScalar<T>::Type_t>  Type_t;
};

template<class T>
struct InternalScalar<OLattice<T> > {
  typedef OScalar<typename InternalScalar<T>::Type_t>  Type_t;
};


// Trait to make a primitive scalar leaving grid along
template<class T>
struct PrimitiveScalar<OScalar<T> > {
  typedef OScalar<typename PrimitiveScalar<T>::Type_t>  Type_t;
};

template<class T>
struct PrimitiveScalar<OLattice<T> > {
  typedef OLattice<typename PrimitiveScalar<T>::Type_t>  Type_t;
};


// Trait to make a lattice scalar leaving primitive indices alone
template<class T>
struct LatticeScalar<OScalar<T> > {
  typedef OScalar<typename LatticeScalar<T>::Type_t>  Type_t;
};

template<class T>
struct LatticeScalar<OLattice<T> > {
  typedef OScalar<typename LatticeScalar<T>::Type_t>  Type_t;
};


// Internally used real scalars
template<class T>
struct RealScalar<OScalar<T> > {
  typedef OScalar<typename RealScalar<T>::Type_t>  Type_t;
};

template<class T>
struct RealScalar<OLattice<T> > {
  typedef OScalar<typename RealScalar<T>::Type_t>  Type_t;
};



//-----------------------------------------------------------------------------
// Traits classes to support return types
//-----------------------------------------------------------------------------

// Default unary(OScalar) -> OScalar
template<class T1, class Op>
struct UnaryReturn<OScalar<T1>, Op> {
  typedef OScalar<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
};

// Default unary(OLattice) -> OLattice
template<class T1, class Op>
struct UnaryReturn<OLattice<T1>, Op> {
  typedef OLattice<typename UnaryReturn<T1, Op>::Type_t>  Type_t;
};

// Default binary(OScalar,OScalar) -> OScalar
template<class T1, class T2, class Op>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, Op> {
  typedef OScalar<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};

// Currently, the only trinary operator is ``where'', so return 
// based on T2 and T3
// Default trinary(OScalar,OScalar,OScalar) -> OScalar
template<class T1, class T2, class T3, class Op>
struct TrinaryReturn<OScalar<T1>, OScalar<T2>, OScalar<T3>, Op> {
  typedef OScalar<typename BinaryReturn<T2, T3, Op>::Type_t>  Type_t;
};

// Default binary(OLattice,OLattice) -> OLattice
template<class T1, class T2, class Op>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, Op> {
  typedef OLattice<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};

// Default binary(OScalar,OLattice) -> OLattice
template<class T1, class T2, class Op>
struct BinaryReturn<OScalar<T1>, OLattice<T2>, Op> {
  typedef OLattice<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};

// Default binary(OLattice,OScalar) -> OLattice
template<class T1, class T2, class Op>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, Op> {
  typedef OLattice<typename BinaryReturn<T1, T2, Op>::Type_t>  Type_t;
};


// Default trinary(OLattice,OLattice,OLattice) -> OLattice
template<class T1, class T2, class T3, class Op>
struct TrinaryReturn<OLattice<T1>, OLattice<T2>, OLattice<T3>, Op> {
  typedef OLattice<typename TrinaryReturn<T1, T2, T3, Op>::Type_t>  Type_t;
};


// Default trinary(OLattice,OScalar,OLattice) -> OLattice
template<class T1, class T2, class T3, class Op>
struct TrinaryReturn<OLattice<T1>, OScalar<T2>, OLattice<T3>, Op> {
  typedef OLattice<typename TrinaryReturn<T1, T2, T3, Op>::Type_t>  Type_t;
};

// Default trinary(OLattice,OLattice,OScalar) -> OLattice
template<class T1, class T2, class T3, class Op>
struct TrinaryReturn<OLattice<T1>, OLattice<T2>, OScalar<T3>, Op> {
  typedef OLattice<typename TrinaryReturn<T1, T2, T3, Op>::Type_t>  Type_t;
};

// Default trinary(OScalar,OLattice,OLattice) -> OLattice
template<class T1, class T2, class T3, class Op>
struct TrinaryReturn<OScalar<T1>, OLattice<T2>, OLattice<T3>, Op> {
  typedef OLattice<typename TrinaryReturn<T1, T2, T3, Op>::Type_t>  Type_t;
};


// Default trinary(OScalar,OScalar,OLattice) -> OLattice
template<class T1, class T2, class T3, class Op>
struct TrinaryReturn<OScalar<T1>, OScalar<T2>, OLattice<T3>, Op> {
  typedef OLattice<typename TrinaryReturn<T1, T2, T3, Op>::Type_t>  Type_t;
};

// Default trinary(OSscalar,OLattice,OScalar) -> OLattice
template<class T1, class T2, class T3, class Op>
struct TrinaryReturn<OScalar<T1>, OLattice<T2>, OScalar<T3>, Op> {
  typedef OLattice<typename TrinaryReturn<T1, T2, T3, Op>::Type_t>  Type_t;
};

// Default trinary(OLattice,OScalar,OScalar) -> OLattice
template<class T1, class T2, class T3, class Op>
struct TrinaryReturn<OLattice<T1>, OScalar<T2>, OScalar<T3>, Op> {
  typedef OLattice<typename TrinaryReturn<T1, T2, T3, Op>::Type_t>  Type_t;
};




// Specific OScalar cases
// Global operations
template<class T>
struct UnaryReturn<OScalar<T>, FnPeekSite> {
  typedef OScalar<typename UnaryReturn<T, FnPeekSite>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<OScalar<T>, FnSum> {
  typedef OScalar<typename UnaryReturn<T, FnSum>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<OScalar<T>, FnSumMulti > {
  typedef multi1d<OScalar<typename UnaryReturn<T, FnSumMulti>::Type_t> >  Type_t;
};

template<class T>
struct UnaryReturn<OScalar<T>, FnNorm2 > {
  typedef OScalar<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<OScalar<T>, FnGlobalMax> {
  typedef OScalar<typename UnaryReturn<T, FnGlobalMax>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<OScalar<T>, FnGlobalMin> {
  typedef OScalar<typename UnaryReturn<T, FnGlobalMin>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProduct > {
  typedef OScalar<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProductReal > {
  typedef OScalar<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<OScalar<T>, FnLocalNorm2 > {
  typedef OScalar<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, FnLocalInnerProduct > {
  typedef OScalar<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, FnLocalInnerProductReal > {
  typedef OScalar<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};


// Gamma algebra
template<int N, int m, class T2, class OpGammaConstMultiply>
struct BinaryReturn<GammaConst<N,m>, OScalar<T2>, OpGammaConstMultiply> {
  typedef OScalar<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, int m, class OpMultiplyGammaConst>
struct BinaryReturn<OScalar<T2>, GammaConst<N,m>, OpMultiplyGammaConst> {
  typedef OScalar<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpGammaTypeMultiply>
struct BinaryReturn<GammaType<N>, OScalar<T2>, OpGammaTypeMultiply> {
  typedef OScalar<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpMultiplyGammaType>
struct BinaryReturn<OScalar<T2>, GammaType<N>, OpMultiplyGammaType> {
  typedef OScalar<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

// Gamma algebra
template<int N, int m, class T2, class OpGammaConstDPMultiply>
struct BinaryReturn<GammaConstDP<N,m>, OScalar<T2>, OpGammaConstDPMultiply> {
  typedef OScalar<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, int m, class OpMultiplyGammaConstDP>
struct BinaryReturn<OScalar<T2>, GammaConstDP<N,m>, OpMultiplyGammaConstDP> {
  typedef OScalar<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpGammaTypeDPMultiply>
struct BinaryReturn<GammaTypeDP<N>, OScalar<T2>, OpGammaTypeDPMultiply> {
  typedef OScalar<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpMultiplyGammaTypeDP>
struct BinaryReturn<OScalar<T2>, GammaTypeDP<N>, OpMultiplyGammaTypeDP> {
  typedef OScalar<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};



// Local operations
template<class T>
struct UnaryReturn<OScalar<T>, OpNot > {
  typedef OScalar<typename UnaryReturn<T, OpNot>::Type_t>  Type_t;
};


#if 0
template<class T1, class T2>
struct UnaryReturn<OScalar<T2>, OpCast<T1> > {
  typedef OScalar<typename UnaryReturn<T, OpCast>::Type_t>  Type_t;
//  typedef T1 Type_t;
};
#endif

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpLT > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpLT>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpLE > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpLE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpGT > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpGT>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpGE > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpGE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpEQ > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpEQ>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpNE > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpNE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpAnd > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpAnd>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpOr > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpOr>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpLeftShift > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpLeftShift>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpRightShift > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpRightShift>::Type_t>  Type_t;
};
 

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpAddAssign > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpAddAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpSubtractAssign > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpSubtractAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpMultiplyAssign > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpMultiplyAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpDivideAssign > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpDivideAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpModAssign > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpModAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpBitwiseOrAssign > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpBitwiseOrAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpBitwiseAndAssign > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpBitwiseAndAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpBitwiseXorAssign > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpBitwiseXorAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpLeftShiftAssign > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpLeftShiftAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OScalar<T2>, OpRightShiftAssign > {
  typedef OScalar<typename BinaryReturn<T1, T2, OpRightShiftAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2, class T3>
struct TrinaryReturn<OScalar<T1>, OScalar<T2>, OScalar<T3>, FnColorContract> {
  typedef OScalar<typename TrinaryReturn<T1, T2, T3, FnColorContract>::Type_t>  Type_t;
};

// Specific OLattice cases
// Global operations
template<class T>
struct UnaryReturn<OLattice<T>, FnGetSite> {
  typedef OScalar<typename UnaryReturn<T, FnGetSite>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<OLattice<T>, FnPeekSite> {
  typedef OScalar<typename UnaryReturn<T, FnPeekSite>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<OLattice<T>, FnSum > {
  typedef OScalar<typename UnaryReturn<T, FnSum>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<OLattice<T>, FnGlobalMax> {
  typedef OScalar<typename UnaryReturn<T, FnGlobalMax>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<OLattice<T>, FnGlobalMin> {
  typedef OScalar<typename UnaryReturn<T, FnGlobalMin>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<OLattice<T>, FnSumMulti > {
  typedef multi1d<OScalar<typename UnaryReturn<T, FnSumMulti>::Type_t> >  Type_t;
};

template<class T>
struct UnaryReturn<OLattice<T>, FnNorm2 > {
  typedef OScalar<typename UnaryReturn<T, FnNorm2>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProduct > {
  typedef OScalar<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProductReal > {
  typedef OScalar<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T>
struct UnaryReturn<OLattice<T>, FnLocalNorm2 > {
  typedef OLattice<typename UnaryReturn<T, FnLocalNorm2>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, FnLocalInnerProduct > {
  typedef OLattice<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, FnLocalInnerProductReal > {
  typedef OLattice<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};


// Gamma algebra
template<int N, int m, class T2, class OpGammaConstMultiply>
struct BinaryReturn<GammaConst<N,m>, OLattice<T2>, OpGammaConstMultiply> {
  typedef OLattice<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, int m, class OpMultiplyGammaConst>
struct BinaryReturn<OLattice<T2>, GammaConst<N,m>, OpMultiplyGammaConst> {
  typedef OLattice<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpGammaTypeMultiply>
struct BinaryReturn<GammaType<N>, OLattice<T2>, OpGammaTypeMultiply> {
  typedef OLattice<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpMultiplyGammaType>
struct BinaryReturn<OLattice<T2>, GammaType<N>, OpMultiplyGammaType> {
  typedef OLattice<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};


// Gamma algebra
template<int N, int m, class T2, class OpGammaConstDPMultiply>
struct BinaryReturn<GammaConstDP<N,m>, OLattice<T2>, OpGammaConstDPMultiply> {
  typedef OLattice<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, int m, class OpMultiplyGammaConstDP>
struct BinaryReturn<OLattice<T2>, GammaConstDP<N,m>, OpMultiplyGammaConstDP> {
  typedef OLattice<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpGammaTypeDPMultiply>
struct BinaryReturn<GammaTypeDP<N>, OLattice<T2>, OpGammaTypeDPMultiply> {
  typedef OLattice<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};

template<class T2, int N, class OpMultiplyGammaTypeDP>
struct BinaryReturn<OLattice<T2>, GammaTypeDP<N>, OpMultiplyGammaTypeDP> {
  typedef OLattice<typename UnaryReturn<T2, OpUnaryPlus>::Type_t>  Type_t;
};



// Local operations
template<class T>
struct UnaryReturn<OLattice<T>, OpNot > {
  typedef OLattice<typename UnaryReturn<T, OpNot>::Type_t>  Type_t;
};


#if 0
template<class T1, class T2>
struct UnaryReturn<OLattice<T2>, OpCast<T1> > {
  typedef OLattice<typename UnaryReturn<T, OpCast>::Type_t>  Type_t;
//  typedef T1 Type_t;
};
#endif

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpLT > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpLT>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpLE > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpLE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpGT > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpGT>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpGE > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpGE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpEQ > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpEQ>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpNE > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpNE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpAnd > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpAnd>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpOr > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpOr>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpLeftShift > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpLeftShift>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpRightShift > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpRightShift>::Type_t>  Type_t;
};
 

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpAddAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpAddAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpSubtractAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpSubtractAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpMultiplyAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpMultiplyAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpDivideAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpDivideAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpModAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpModAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpBitwiseOrAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpBitwiseOrAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpBitwiseAndAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpBitwiseAndAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpBitwiseXorAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpBitwiseXorAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpLeftShiftAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpLeftShiftAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OLattice<T2>, OpRightShiftAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpRightShiftAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2, class T3>
struct TrinaryReturn<OLattice<T1>, OLattice<T2>, OLattice<T3>, FnColorContract> {
  typedef OLattice<typename TrinaryReturn<T1, T2, T3, FnColorContract>::Type_t>  Type_t;
};


// Mixed OLattice & OScalar cases
// Global operations
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, FnInnerProduct > {
  typedef OScalar<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, FnInnerProductReal > {
  typedef OScalar<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OLattice<T2>, FnInnerProduct > {
  typedef OScalar<typename BinaryReturn<T1, T2, FnInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OLattice<T2>, FnInnerProductReal > {
  typedef OScalar<typename BinaryReturn<T1, T2, FnInnerProductReal>::Type_t>  Type_t;
};


template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, FnLocalInnerProduct > {
  typedef OLattice<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, FnLocalInnerProductReal > {
  typedef OLattice<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OLattice<T2>, FnLocalInnerProduct > {
  typedef OLattice<typename BinaryReturn<T1, T2, FnLocalInnerProduct>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OLattice<T2>, FnLocalInnerProductReal > {
  typedef OLattice<typename BinaryReturn<T1, T2, FnLocalInnerProductReal>::Type_t>  Type_t;
};


// Local operations
template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpLT > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpLT>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpLE > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpLE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpGT > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpGT>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpGE > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpGE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpEQ > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpEQ>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpNE > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpNE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpAnd > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpAnd>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpOr > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpOr>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpLeftShift > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpLeftShift>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpRightShift > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpRightShift>::Type_t>  Type_t;
};
 

template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpAddAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpAddAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpSubtractAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpSubtractAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpMultiplyAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpMultiplyAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpDivideAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpDivideAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpModAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpModAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpBitwiseOrAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpBitwiseOrAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpBitwiseAndAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpBitwiseAndAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpBitwiseXorAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpBitwiseXorAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpLeftShiftAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpLeftShiftAssign>::Type_t>  &Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OLattice<T1>, OScalar<T2>, OpRightShiftAssign > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpRightShiftAssign>::Type_t>  &Type_t;
};
 

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OLattice<T2>, OpLT > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpLT>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OLattice<T2>, OpLE > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpLE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OLattice<T2>, OpGT > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpGT>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OLattice<T2>, OpGE > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpGE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OLattice<T2>, OpEQ > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpEQ>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OLattice<T2>, OpNE > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpNE>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OLattice<T2>, OpAnd > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpAnd>::Type_t>  Type_t;
};

template<class T1, class T2 >
struct BinaryReturn<OScalar<T1>, OLattice<T2>, OpOr > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpOr>::Type_t>  Type_t;
};

template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OLattice<T2>, OpLeftShift > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpLeftShift>::Type_t>  Type_t;
};
 
template<class T1, class T2>
struct BinaryReturn<OScalar<T1>, OLattice<T2>, OpRightShift > {
  typedef OLattice<typename BinaryReturn<T1, T2, OpRightShift>::Type_t>  Type_t;
};
 




//-----------------------------------------------------------------------------
// Scalar Operations
//-----------------------------------------------------------------------------

/*! \addtogroup oscalar */
/*! @{ */

//! QDP Wordtype to primitive wordtype
template<class T> 
inline typename WordType< OScalar<T> >::Type_t
toWordType(const OScalar<T>& s) 
{
  return toWordType(s.elem());
}

//! dest [some type] = source [some type]
/*! Portable (internal) way of returning a single site */
template<class T>
struct UnaryReturn<OScalar<T>, FnGetSite> {
  typedef OScalar<typename UnaryReturn<T, FnGetSite>::Type_t>  Type_t;
};

template<class T>
inline typename UnaryReturn<OScalar<T>, FnGetSite>::Type_t
getSite(const OScalar<T>& s1, int innersite)
{
  typename UnaryReturn<OScalar<T>, FnGetSite>::Type_t  d;

  d.elem() = getSite(s1.elem(), innersite);
  return d;
}


//! dest = 0
template<class T> 
void zero_rep(OScalar<T>& dest) 
{
  zero_rep(dest.elem());
}

//! dest = (mask) ? s1 : dest
template<class T1, class T2> 
void copymask(OScalar<T2>& dest, const OScalar<T1>& mask, const OScalar<T2>& s1) 
{
  copymask(dest.elem(), mask.elem(), s1.elem());
}

//! dest [some type] = source [some type]
template<class T, class T1>
void cast_rep(T& d, const OScalar<T1>& s1)
{
  cast_rep(d, s1.elem());
}

//! dest [some type] = source [some type]
template<class T, class T1>
void recast_rep(OScalar<T>& d, const OScalar<T1>& s1)
{
  cast_rep(d.elem(), s1.elem());
}


//-----------------------------------------------------------------------------
// Random numbers
//! dest  = random  
/*! Implementation is in the specific files */
template<class T>
void random(OScalar<T>& d);


//! dest  = gaussian
template<class T, class S>
void gaussian(OSubScalar<T,S>& d)
{
  OScalar<T>  r1, r2;

  random(OSubScalar<T,S>(r1,d.subset()));
  random(OSubScalar<T,S>(r2,d.subset()));

  fill_gaussian(d.elem(), r1.elem(), r2.elem());
}



/*! @} */  // end of group oscalar

} // namespace QDP

#endif
