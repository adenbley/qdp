// -*- C++ -*-

/*! @file
 * @brief Additional operations on QDPTypes
 */

#ifndef QDP_NEWOPS_H
#define QDP_NEWOPS_H

#include "cudp_iface.h"

namespace QDP {

//-----------------------------------------------------------------------------
// Operator tags that are only used for type resolution
//-----------------------------------------------------------------------------

struct FnSpinProject
{
  PETE_EMPTY_CONSTRUCTORS(FnSpinProject)
};

struct FnSpinReconstruct
{
  PETE_EMPTY_CONSTRUCTORS(FnSpinReconstruct)
};

struct FnQuarkContractXX
{
  PETE_EMPTY_CONSTRUCTORS(FnQuarkContractXX)
};

struct FnSum
{
  PETE_EMPTY_CONSTRUCTORS(FnSum)
};

struct FnGlobalMax
{
  PETE_EMPTY_CONSTRUCTORS(FnGlobalMax)
};

struct FnGlobalMin
{
  PETE_EMPTY_CONSTRUCTORS(FnGlobalMin)
};

struct FnNorm2
{
  PETE_EMPTY_CONSTRUCTORS(FnNorm2)
};

struct FnInnerProduct
{
  PETE_EMPTY_CONSTRUCTORS(FnInnerProduct)
};

struct FnInnerProductReal
{
  PETE_EMPTY_CONSTRUCTORS(FnInnerProductReal)
};

struct FnSumMulti
{
  PETE_EMPTY_CONSTRUCTORS(FnSumMulti)
};

struct FnNorm2Multi
{
  PETE_EMPTY_CONSTRUCTORS(FnNorm2Multi)
};

struct FnInnerProductMulti
{
  PETE_EMPTY_CONSTRUCTORS(FnInnerProductMulti)
};

struct FnInnerProductRealMulti
{
  PETE_EMPTY_CONSTRUCTORS(FnInnerProductRealMulti)
};


//-----------------------------------------------------------------------------
// Operators and tags for accessing elements of a QDP object
//-----------------------------------------------------------------------------

struct FnGetSite
{
  PETE_EMPTY_CONSTRUCTORS(FnGetSite)
};

struct FnPeekSite
{
  PETE_EMPTY_CONSTRUCTORS(FnPeekSite)
};

struct FnPokeSite
{
  PETE_EMPTY_CONSTRUCTORS(FnPokeSite)
};


//! Structure for extracting color matrix components
struct FnPeekColorMatrix
{
  PETE_EMPTY_CONSTRUCTORS(FnPeekColorMatrix)

  __device__
  FnPeekColorMatrix(int _row, int _col): row(_row), col(_col) {}
  
  template<class T>
  __device__ inline typename UnaryReturn<T, FnPeekColorMatrix>::Type_t
  operator()(const T &a) const
  {
    return (peekColor(a,row,col));
  }

  __device__
  void unpackNode(FlattenTag::NodeData & nodeData) const {
#ifdef __CUDA_ARCH__
    row = nodeData.row;
    col = nodeData.col;
    if (blockIdx.x * blockDim.x + threadIdx.x == 0)
      printf("FnPeekSpinMatrix::unpackNode %d %d \n",row,col);
#endif
  }

private:
  mutable int row, col;
};


//! Extract color matrix components
/*! @ingroup group1
  @relates QDPType */
template<class T1,class C1>
__device__ inline typename MakeReturn<UnaryNode<FnPeekColorMatrix,
  typename CreateLeaf<QDPType<T1,C1> >::Leaf_t>,
  typename UnaryReturn<C1,FnPeekColorMatrix >::Type_t >::Expression_t
peekColor(const QDPType<T1,C1> & l, int row, int col)
{
  typedef UnaryNode<FnPeekColorMatrix,
    typename CreateLeaf<QDPType<T1,C1> >::Leaf_t> Tree_t;
  typedef typename UnaryReturn<C1,FnPeekColorMatrix >::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(FnPeekColorMatrix(row,col),
    CreateLeaf<QDPType<T1,C1> >::make(l)));
}


template<class T1,class C1>
__device__ inline typename MakeReturn<UnaryNode<FnPeekColorMatrix,
  typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
peekColor(const QDPExpr<T1,C1> & l, int row, int col)
{
  typedef UnaryNode<FnPeekColorMatrix,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
  typedef typename UnaryReturn<C1,FnPeekColorMatrix >::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(FnPeekColorMatrix(row,col),
    CreateLeaf<QDPExpr<T1,C1> >::make(l)));
}


//! Structure for extracting color vector components
struct FnPeekColorVector
{
  PETE_EMPTY_CONSTRUCTORS(FnPeekColorVector)

  __device__
  FnPeekColorVector(int _row): row(_row) {}
  
  template<class T>
  __device__ inline typename UnaryReturn<T, FnPeekColorVector>::Type_t
  operator()(const T &a) const
  {
    return (peekColor(a,row));
  }

  __device__
  void unpackNode(FlattenTag::NodeData & nodeData) const {
#ifdef __CUDA_ARCH__
    row = nodeData.row;
    if (blockIdx.x * blockDim.x + threadIdx.x == 0)
      printf("FnPeekSpinMatrix::unpackNode %d  \n",row);
#endif
  }

private:
  mutable int row;
};


//! Extract color vector components
/*! @ingroup group1
  @relates QDPType */
template<class T1,class C1>
__device__ inline typename MakeReturn<UnaryNode<FnPeekColorVector,
  typename CreateLeaf<QDPType<T1,C1> >::Leaf_t>,
  typename UnaryReturn<C1,FnPeekColorVector >::Type_t >::Expression_t
peekColor(const QDPType<T1,C1> & l, int row)
{
  typedef UnaryNode<FnPeekColorVector,
    typename CreateLeaf<QDPType<T1,C1> >::Leaf_t> Tree_t;
  typedef typename UnaryReturn<C1,FnPeekColorVector >::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(FnPeekColorVector(row),
    CreateLeaf<QDPType<T1,C1> >::make(l)));
}


template<class T1,class C1>
__device__ inline typename MakeReturn<UnaryNode<FnPeekColorVector,
  typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
peekColor(const QDPExpr<T1,C1> & l, int row)
{
  typedef UnaryNode<FnPeekColorVector,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
  typedef typename UnaryReturn<C1,FnPeekColorVector >::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(FnPeekColorVector(row),
    CreateLeaf<QDPExpr<T1,C1> >::make(l)));
}


//! Structure for extracting spin matrix components
struct FnPeekSpinMatrix
{
  PETE_EMPTY_CONSTRUCTORS(FnPeekSpinMatrix)

  __device__
  FnPeekSpinMatrix(int _row, int _col): row(_row), col(_col) {
#ifdef __CUDA_ARCH__
    if (blockIdx.x * blockDim.x + threadIdx.x == 0)
      printf("FnPeekSpinMatrix::FnPeekSpinMatrix %d %d \n",row,col);
#endif
  }
  
  template<class T>
  __device__ inline typename UnaryReturn<T, FnPeekSpinMatrix>::Type_t
  operator()(const T &a) const
  {
#ifdef __CUDA_ARCH__
    if (blockIdx.x * blockDim.x + threadIdx.x == 0)
      printf("FnPeekSpinMatrix::operator() %d %d \n",row,col);
#endif
    return (peekSpin(a,row,col));
  }

  __device__
  void unpackNode(FlattenTag::NodeData & nodeData) const {
#ifdef __CUDA_ARCH__
    row = nodeData.row;
    col = nodeData.col;
    if (blockIdx.x * blockDim.x + threadIdx.x == 0)
      printf("FnPeekSpinMatrix::unpackNode %d %d \n",row,col);
#endif
  }

private:
  mutable int row;
  mutable int col;
};

//! Extract spin matrix components
/*! @ingroup group1
  @relates QDPType */
template<class T1,class C1>
__device__ inline typename MakeReturn<UnaryNode<FnPeekSpinMatrix,
  typename CreateLeaf<QDPType<T1,C1> >::Leaf_t>,
  typename UnaryReturn<C1,FnPeekSpinMatrix >::Type_t >::Expression_t
peekSpin(const QDPType<T1,C1> & l, int row, int col)
{
  typedef UnaryNode<FnPeekSpinMatrix,
    typename CreateLeaf<QDPType<T1,C1> >::Leaf_t> Tree_t;
  typedef typename UnaryReturn<C1,FnPeekSpinMatrix >::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(FnPeekSpinMatrix(row,col),
    CreateLeaf<QDPType<T1,C1> >::make(l)));
}


template<class T1,class C1>
__device__ inline typename MakeReturn<UnaryNode<FnPeekSpinMatrix,
  typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
peekSpin(const QDPExpr<T1,C1> & l, int row, int col)
{
  typedef UnaryNode<FnPeekSpinMatrix,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
  typedef typename UnaryReturn<C1,FnPeekSpinMatrix >::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(FnPeekSpinMatrix(row,col),
    CreateLeaf<QDPExpr<T1,C1> >::make(l)));
}


//! Structure for extracting spin vector components
struct FnPeekSpinVector
{
  PETE_EMPTY_CONSTRUCTORS(FnPeekSpinVector)

  __device__
  FnPeekSpinVector(int _row): row(_row) {}
  
  template<class T>
  __device__ inline typename UnaryReturn<T, FnPeekSpinVector>::Type_t
  operator()(const T &a) const
  {
    return (peekSpin(a,row));
  }

  __device__
  void unpackNode(FlattenTag::NodeData & nodeData) const {
#ifdef __CUDA_ARCH__
    row = nodeData.row;
    if (blockIdx.x * blockDim.x + threadIdx.x == 0)
      printf("FnPeekSpinMatrix::unpackNode %d  \n",row);
#endif
  }

private:
  mutable int row;
};


//! Extract spin vector components
/*! @ingroup group1
  @relates QDPType */
template<class T1,class C1>
__device__ inline typename MakeReturn<UnaryNode<FnPeekSpinVector,
  typename CreateLeaf<QDPType<T1,C1> >::Leaf_t>,
  typename UnaryReturn<C1,FnPeekSpinVector >::Type_t >::Expression_t
peekSpin(const QDPType<T1,C1> & l, int row)
{
  typedef UnaryNode<FnPeekSpinVector,
    typename CreateLeaf<QDPType<T1,C1> >::Leaf_t> Tree_t;
  typedef typename UnaryReturn<C1,FnPeekSpinVector >::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(FnPeekSpinVector(row),
    CreateLeaf<QDPType<T1,C1> >::make(l)));
}


template<class T1,class C1>
__device__ inline typename MakeReturn<UnaryNode<FnPeekSpinVector,
  typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
peekSpin(const QDPExpr<T1,C1> & l, int row)
{
  typedef UnaryNode<FnPeekSpinVector,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
  typedef typename UnaryReturn<C1,FnPeekSpinVector >::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(FnPeekSpinVector(row),
    CreateLeaf<QDPExpr<T1,C1> >::make(l)));
}



//---------------------------------------
//! Structure for inserting color matrix components
struct FnPokeColorMatrix
{
  PETE_EMPTY_CONSTRUCTORS(FnPokeColorMatrix)

  __device__
  FnPokeColorMatrix(int _row, int _col): row(_row), col(_col) {}
  
  template<class T1, class T2>
  __device__ inline typename BinaryReturn<T1, T2, FnPokeColorMatrix>::Type_t
  operator()(const T1 &a, const T2 &b) const
  {
    pokeColor(const_cast<T1&>(a),b,row,col);
    return const_cast<T1&>(a);
  }

  __device__
  void unpackNode(FlattenTag::NodeData & nodeData) const {
#ifdef __CUDA_ARCH__
    row = nodeData.row;
    col = nodeData.col;
    if (blockIdx.x * blockDim.x + threadIdx.x == 0)
      printf("FnPeekSpinMatrix::unpackNode %d %d \n",row,col);
#endif
  }

private:
  mutable int row, col;
};


//! Insert color matrix components
/*! @ingroup group1
  @param l  target to update
  @param r  source
  @param row  row of color matrix
  @param col  column of color matrix
  @return updated target
  @ingroup group1
  @relates QDPType */
template<class T1,class C1,class T2,class C2>
__device__ inline C1& 
pokeColor(QDPType<T1,C1> & l, const QDPType<T2,C2>& r, int row, int col)
{
  C1& ll = static_cast<C1&>(l);
  evaluate(ll,FnPokeColorMatrix(row,col),PETE_identity(r),all);
  return ll;
}


template<class T1,class C1,class T2,class C2>
__device__ inline C1& 
pokeColor(QDPType<T1,C1> & l, const QDPExpr<T2,C2>& r, int row, int col)
{
  C1& ll = static_cast<C1&>(l);
  evaluate(ll,FnPokeColorMatrix(row,col),r,all);
  return ll;
}


template<class T1,class C1,class T2,class C2>
__device__ inline C1
pokeColor(const QDPSubType<T1,C1>& l, const QDPType<T2,C2>& r, int row, int col)
{
  C1& ll = const_cast<QDPSubType<T1,C1>&>(l).field();
  const Subset& s = l.subset();

  evaluate(ll,FnPokeColorMatrix(row,col),PETE_identity(r),s);
  return ll;
}


template<class T1,class C1,class T2,class C2>
__device__ inline C1
pokeColor(const QDPSubType<T1,C1>& l, const QDPExpr<T2,C2>& r, int row, int col)
{
  C1& ll = const_cast<QDPSubType<T1,C1>&>(l).field();
  const Subset& s = l.subset();

  evaluate(ll,FnPokeColorMatrix(row,col),r,s);
  return ll;
}


//! Structure for inserting color vector components
struct FnPokeColorVector
{
  PETE_EMPTY_CONSTRUCTORS(FnPokeColorVector)

  __device__
  FnPokeColorVector(int _row): row(_row) {}
  
  template<class T1, class T2>
  __device__ inline typename BinaryReturn<T1, T2, FnPokeColorVector>::Type_t
  operator()(const T1 &a, const T2 &b) const
  {
    pokeColor(const_cast<T1&>(a),b,row);
    return const_cast<T1&>(a);
  }

  __device__
  void unpackNode(FlattenTag::NodeData & nodeData) const {
#ifdef __CUDA_ARCH__
    row = nodeData.row;
    if (blockIdx.x * blockDim.x + threadIdx.x == 0)
      printf("FnPeekSpinMatrix::unpackNode %d %d \n",row);
#endif
  }

private:
  mutable int row;
};



//! Insert color vector components
/*! @ingroup group1
  @param l  target to update
  @param r  source
  @param row  row of color vector
  @return updated target
  @ingroup group1
  @relates QDPType */
template<class T1,class C1,class T2,class C2>
__device__ inline C1& 
pokeColor(QDPType<T1,C1>& l, const QDPType<T2,C2>& r, int row)
{
  C1& ll = static_cast<C1&>(l);
  evaluate(ll,FnPokeColorVector(row),PETE_identity(r),all);
  return ll;
}

template<class T1,class C1,class T2,class C2>
__device__ inline C1& 
pokeColor(QDPType<T1,C1>& l, const QDPExpr<T2,C2>& r, int row)
{
  C1& ll = static_cast<C1&>(l);
  evaluate(ll,FnPokeColorVector(row),r,all);
  return ll;
}


template<class T1,class C1,class T2,class C2>
__device__ inline C1
pokeColor(const QDPSubType<T1,C1>& l, const QDPType<T2,C2>& r, int row)
{
  C1& ll = const_cast<QDPSubType<T1,C1>&>(l).field();
  const Subset& s = l.subset();

  evaluate(ll,FnPokeColorVector(row),PETE_identity(r),s);
  return ll;
}


template<class T1,class C1,class T2,class C2>
__device__ inline C1
pokeColor(const QDPSubType<T1,C1>& l, const QDPExpr<T2,C2>& r, int row)
{
  C1& ll = const_cast<QDPSubType<T1,C1>&>(l).field();
  const Subset& s = l.subset();

  evaluate(ll,FnPokeColorVector(row),r,s);
  return ll;
}


//! Structure for inserting spin matrix components
struct FnPokeSpinMatrix
{
  PETE_EMPTY_CONSTRUCTORS(FnPokeSpinMatrix)

  __device__
  FnPokeSpinMatrix(int _row, int _col): row(_row), col(_col) {}
  
  template<class T1, class T2>
  __device__ inline typename BinaryReturn<T1, T2, FnPokeSpinMatrix>::Type_t
  operator()(const T1 &a, const T2 &b) const
  {
    pokeSpin(const_cast<T1&>(a),b,row,col);
    return const_cast<T1&>(a);
  }

  __device__
  void unpackNode(FlattenTag::NodeData & nodeData) const {
#ifdef __CUDA_ARCH__
    row = nodeData.row;
    col = nodeData.col;
    if (blockIdx.x * blockDim.x + threadIdx.x == 0)
      printf("FnPeekSpinMatrix::unpackNode %d %d \n",row,col);
#endif
  }

private:
  mutable int row, col;
};

//! Insert spin matrix components
/*! @ingroup group1
  @param l  target to update
  @param r  source
  @param row  row of spin matrix
  @param col  column of spin matrix
  @return updated target
  @ingroup group1
  @relates QDPType */
template<class T1,class C1,class T2,class C2>
__device__ inline C1& 
pokeSpin(QDPType<T1,C1> & l, const QDPType<T2,C2>& r, int row, int col)
{
  C1& ll = static_cast<C1&>(l);
  evaluate(ll,FnPokeSpinMatrix(row,col),PETE_identity(r),all);
  return ll;
}

template<class T1,class C1,class T2,class C2>
__device__ inline C1& 
pokeSpin(QDPType<T1,C1> & l, const QDPExpr<T2,C2>& r, int row, int col)
{
  C1& ll = static_cast<C1&>(l);
  evaluate(ll,FnPokeSpinMatrix(row,col),r,all);
  return ll;
}


template<class T1,class C1,class T2,class C2>
__device__ inline C1
pokeSpin(const QDPSubType<T1,C1>& l, const QDPType<T2,C2>& r, int row, int col)
{
  C1& ll = const_cast<QDPSubType<T1,C1>&>(l).field();
  const Subset& s = l.subset();

  evaluate(ll,FnPokeSpinMatrix(row,col),PETE_identity(r),s);
  return ll;
}


template<class T1,class C1,class T2,class C2>
__device__ inline C1
pokeSpin(const QDPSubType<T1,C1>& l, const QDPExpr<T2,C2>& r, int row, int col)
{
  C1& ll = const_cast<QDPSubType<T1,C1>&>(l).field();
  const Subset& s = l.subset();

  evaluate(ll,FnPokeSpinMatrix(row,col),r,s);
  return ll;
}



//! Structure for inserting spin vector components
struct FnPokeSpinVector
{
  PETE_EMPTY_CONSTRUCTORS(FnPokeSpinVector)

  __device__
  FnPokeSpinVector(int _row): row(_row) {}
  
  template<class T1, class T2>
  __device__ inline typename BinaryReturn<T1, T2, FnPokeSpinVector>::Type_t
  operator()(const T1 &a, const T2 &b) const
  {
    pokeSpin(const_cast<T1&>(a),b,row);
    return const_cast<T1&>(a);
  }

  __device__
  void unpackNode(FlattenTag::NodeData & nodeData) const {
#ifdef __CUDA_ARCH__
    row = nodeData.row;
    if (blockIdx.x * blockDim.x + threadIdx.x == 0)
      printf("FnPeekSpinMatrix::unpackNode %d \n",row);
#endif
  }

private:
  mutable int row;
};


//! Insert spin vector components
/*! @ingroup group1
  @param l  target to update
  @param r  source
  @param row  row of spin vector
  @return updated target
  @ingroup group1
  @relates QDPType */
template<class T1,class C1,class T2,class C2>
__device__ inline C1& 
pokeSpin(QDPType<T1,C1>& l, const QDPType<T2,C2>& r, int row)
{
  C1& ll = static_cast<C1&>(l);
  evaluate(ll,FnPokeSpinVector(row),PETE_identity(r),all);
  return ll;
}

template<class T1,class C1,class T2,class C2>
__device__ inline C1& 
pokeSpin(QDPType<T1,C1>& l, const QDPExpr<T2,C2>& r, int row)
{
  C1& ll = static_cast<C1&>(l);
  evaluate(ll,FnPokeSpinVector(row),r,all);
  return ll;
}


template<class T1,class C1,class T2,class C2>
__device__ inline C1
pokeSpin(const QDPSubType<T1,C1>& l, const QDPType<T2,C2>& r, int row)
{
  C1& ll = const_cast<QDPSubType<T1,C1>&>(l).field();
  const Subset& s = l.subset();

  evaluate(ll,FnPokeSpinVector(row),PETE_identity(r),s);
  return ll;
}


template<class T1,class C1,class T2,class C2>
__device__ inline C1
pokeSpin(const QDPSubType<T1,C1>& l, const QDPExpr<T2,C2>& r, int row)
{
  C1& ll = const_cast<QDPSubType<T1,C1>&>(l).field();
  const Subset& s = l.subset();

  evaluate(ll,FnPokeSpinVector(row),r,s);
  return ll;
}



//-----------------------------------------------------------------------------
// Additional operator tags 
//-----------------------------------------------------------------------------

struct OpGammaConstMultiply
{
  PETE_EMPTY_CONSTRUCTORS(OpGammaConstMultiply)
  template<class T1, class T2>
  __device__ inline typename BinaryReturn<T1, T2, OpGammaConstMultiply >::Type_t
  operator()(const T1 &a, const T2 &b) const
  {
    return (a * b);
  }
};


struct OpMultiplyGammaConst
{
  PETE_EMPTY_CONSTRUCTORS(OpMultiplyGammaConst)
  template<class T1, class T2>
  __device__ inline typename BinaryReturn<T1, T2, OpMultiplyGammaConst >::Type_t
  operator()(const T1 &a, const T2 &b) const
  {
    return (a * b);
  }
};


// Member function definition in primgamma.h
struct OpGammaTypeMultiply
{
  PETE_EMPTY_CONSTRUCTORS(OpGammaTypeMultiply)
  template<int N, class T>
  __device__ inline T
  operator()(const GammaType<N>& a, const T &b) const;
};


// Member function definition in primgamma.h
struct OpMultiplyGammaType
{
  PETE_EMPTY_CONSTRUCTORS(OpMultiplyGammaType)
  template<class T, int N>
  __device__ inline T
  operator()(const T &a, const GammaType<N>& b) const;
};


//-----------------------------------------------------------------------------
// Additional operator tags 
//-----------------------------------------------------------------------------

struct OpGammaConstDPMultiply
{
  PETE_EMPTY_CONSTRUCTORS(OpGammaConstDPMultiply)
  template<class T1, class T2>
  __device__ inline typename BinaryReturn<T1, T2, OpGammaConstDPMultiply >::Type_t
  operator()(const T1 &a, const T2 &b) const
  {
    return (a * b);
  }
};


struct OpMultiplyGammaConstDP
{
  PETE_EMPTY_CONSTRUCTORS(OpMultiplyGammaConstDP)
  template<class T1, class T2>
  __device__ inline typename BinaryReturn<T1, T2, OpMultiplyGammaConstDP >::Type_t
  operator()(const T1 &a, const T2 &b) const
  {
    return (a * b);
  }
};


// Member function definition in primgamma.h
struct OpGammaTypeDPMultiply
{
  PETE_EMPTY_CONSTRUCTORS(OpGammaTypeDPMultiply)
  template<int N, class T>
  __device__ inline T
  operator()(const GammaTypeDP<N>& a, const T &b) const;
};


// Member function definition in primgamma.h
struct OpMultiplyGammaTypeDP
{
  PETE_EMPTY_CONSTRUCTORS(OpMultiplyGammaTypeDP)
  template<class T, int N>
  __device__ inline T
  operator()(const T &a, const GammaTypeDP<N>& b) const;
};





template<class A, class CTag, class FnTag>
struct ForEach_Base;


template<class A, class CTag, class FnTag>
struct ForEach_Base<UnaryNode<FnTag, A>, FlattenTag , CTag>
{
  typedef typename ForEach<A, FlattenTag, CTag>::Type_t TypeA_t;
  typedef typename Combine1<TypeA_t, FnTag, CTag>::Type_t Type_t;
  __device__
  inline static
  Type_t apply(const UnaryNode<FnTag, A> &expr, const FlattenTag &f, 
	       const CTag &c)
  {
#ifdef __CUDA_ARCH__

    if (blockIdx.x * blockDim.x + threadIdx.x == 0)
      printf("ForEach_Base: node %d\n" , f.count_node );

    if (f.count_node >= f.numberNodes) {
      if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	printf("Oops: f.count >= f.numberLeafs!\n");
    }

    expr.operation().unpackNode( f.nodeDataArray[ f.count_node ] );

    f.count_node++;

    return Combine1<TypeA_t, FnTag, CTag>::
      combine(ForEach<A, FlattenTag, CTag>::apply(expr.child(), f, c),
	      expr.operation(), c);
#endif
  }
};


template<class A, class CTag>
struct ForEach<UnaryNode<FnPeekSpinMatrix,A>,FlattenTag,CTag>:ForEach_Base<UnaryNode<FnPeekSpinMatrix,A>,FlattenTag,CTag>{};
template<class A, class CTag>
struct ForEach<UnaryNode<FnPeekSpinVector,A>,FlattenTag,CTag>:ForEach_Base<UnaryNode<FnPeekSpinVector,A>,FlattenTag,CTag>{};
template<class A, class CTag>
struct ForEach<UnaryNode<FnPeekColorMatrix,A>,FlattenTag,CTag>:ForEach_Base<UnaryNode<FnPeekColorMatrix,A>,FlattenTag,CTag>{};
template<class A, class CTag>
struct ForEach<UnaryNode<FnPeekColorVector,A>,FlattenTag,CTag>:ForEach_Base<UnaryNode<FnPeekColorVector,A>,FlattenTag,CTag>{};
template<class A, class CTag>
struct ForEach<UnaryNode<FnPokeSpinMatrix,A>,FlattenTag,CTag>:ForEach_Base<UnaryNode<FnPokeSpinMatrix,A>,FlattenTag,CTag>{};
template<class A, class CTag>
struct ForEach<UnaryNode<FnPokeSpinVector,A>,FlattenTag,CTag>:ForEach_Base<UnaryNode<FnPokeSpinVector,A>,FlattenTag,CTag>{};
template<class A, class CTag>
struct ForEach<UnaryNode<FnPokeColorMatrix,A>,FlattenTag,CTag>:ForEach_Base<UnaryNode<FnPokeColorMatrix,A>,FlattenTag,CTag>{};
template<class A, class CTag>
struct ForEach<UnaryNode<FnPokeColorVector,A>,FlattenTag,CTag>:ForEach_Base<UnaryNode<FnPokeColorVector,A>,FlattenTag,CTag>{};












//-----------------------------------------------------------------------------
// Leaf stuff
//-----------------------------------------------------------------------------

template<int N>
struct CreateLeaf<GammaType<N> >
{
  typedef GammaType<N> Input_t;
  typedef Input_t Leaf_t;
//  typedef Reference<Input_t> Leaf_t;

  __device__ inline static
  Leaf_t make(const Input_t& a) { return Leaf_t(a); }
};

template<int N>
struct LeafFunctor<GammaType<N>, ElemLeaf>
{
  typedef GammaType<N> Type_t;
  __device__ inline static Type_t apply(const GammaType<N> &a, const ElemLeaf &f)
    {return a;}
};

template<int N>
struct LeafFunctor<GammaType<N>, EvalLeaf1>
{
  typedef GammaType<N> Type_t;
  __device__ inline static Type_t apply(const GammaType<N> &a, const EvalLeaf1 &f)
    {return a;}
};


template<int N, int m>
struct CreateLeaf<GammaConst<N,m> >
{
  typedef GammaConst<N,m> Input_t;
  typedef Input_t Leaf_t;

  __device__ inline static
  Leaf_t make(const Input_t& a) { return Leaf_t(a); }
};

template<int N, int m>
struct LeafFunctor<GammaConst<N,m>, ElemLeaf>
{
  typedef GammaConst<N,m> Type_t;
  __device__ inline static Type_t apply(const GammaConst<N,m> &a, const ElemLeaf &f)
    {return a;}
};

template<int N, int m>
struct LeafFunctor<GammaConst<N,m>, EvalLeaf1>
{
  typedef GammaConst<N,m> Type_t;
  __device__ inline static Type_t apply(const GammaConst<N,m> &a, const EvalLeaf1 &f)
    {return a;}
};



//-----------------------------------------------------------------------------
// Leaf stuff
//-----------------------------------------------------------------------------

template<int N>
struct CreateLeaf<GammaTypeDP<N> >
{
  typedef GammaTypeDP<N> Input_t;
  typedef Input_t Leaf_t;
//  typedef Reference<Input_t> Leaf_t;

  __device__ inline static
  Leaf_t make(const Input_t& a) { return Leaf_t(a); }
};

template<int N>
struct LeafFunctor<GammaTypeDP<N>, ElemLeaf>
{
  typedef GammaTypeDP<N> Type_t;
  __device__ inline static Type_t apply(const GammaTypeDP<N> &a, const ElemLeaf &f)
    {return a;}
};

template<int N>
struct LeafFunctor<GammaTypeDP<N>, EvalLeaf1>
{
  typedef GammaTypeDP<N> Type_t;
  __device__ inline static Type_t apply(const GammaTypeDP<N> &a, const EvalLeaf1 &f)
    {return a;}
};


template<int N, int m>
struct CreateLeaf<GammaConstDP<N,m> >
{
  typedef GammaConstDP<N,m> Input_t;
  typedef Input_t Leaf_t;

  __device__ inline static
  Leaf_t make(const Input_t& a) { return Leaf_t(a); }
};

template<int N, int m>
struct LeafFunctor<GammaConstDP<N,m>, ElemLeaf>
{
  typedef GammaConstDP<N,m> Type_t;
  __device__ inline static Type_t apply(const GammaConstDP<N,m> &a, const ElemLeaf &f)
    {return a;}
};

template<int N, int m>
struct LeafFunctor<GammaConstDP<N,m>, EvalLeaf1>
{
  typedef GammaConstDP<N,m> Type_t;
  __device__ inline static Type_t apply(const GammaConstDP<N,m> &a, const EvalLeaf1 &f)
    {return a;}
};



//-----------------------------------------------------------------------------
// Additional operators
//-----------------------------------------------------------------------------

//! GammaConst * QDPType
/*! @ingroup group1 */
template<int N,int m,class T2,class C2>
__device__ inline typename MakeReturn<BinaryNode<OpGammaConstMultiply,
  typename CreateLeaf<GammaConst<N,m> >::Leaf_t,
  typename CreateLeaf<QDPType<T2,C2> >::Leaf_t>,
  typename BinaryReturn<GammaConst<N,m>,C2,OpGammaConstMultiply>::Type_t >::Expression_t
operator*(const GammaConst<N,m> & l,const QDPType<T2,C2> & r)
{
  typedef BinaryNode<OpGammaConstMultiply,
    typename CreateLeaf<GammaConst<N,m> >::Leaf_t,
    typename CreateLeaf<QDPType<T2,C2> >::Leaf_t> Tree_t;
  typedef typename BinaryReturn<GammaConst<N,m>,C2,OpGammaConstMultiply>::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(
    CreateLeaf<GammaConst<N,m> >::make(l),
    CreateLeaf<QDPType<T2,C2> >::make(r)));
}

//! GammaConst * QDPExpr
template<int N,int m,class T2,class C2>
__device__ inline typename MakeReturn<BinaryNode<OpGammaConstMultiply,
  typename CreateLeaf<GammaConst<N,m> >::Leaf_t,
  typename CreateLeaf<QDPExpr<T2,C2> >::Leaf_t>,
  typename BinaryReturn<GammaConst<N,m>,C2,OpGammaConstMultiply>::Type_t >::Expression_t
operator*(const GammaConst<N,m> & l,const QDPExpr<T2,C2> & r)
{
  typedef BinaryNode<OpGammaConstMultiply,
    typename CreateLeaf<GammaConst<N,m> >::Leaf_t,
    typename CreateLeaf<QDPExpr<T2,C2> >::Leaf_t> Tree_t;
  typedef typename BinaryReturn<GammaConst<N,m>,C2,OpGammaConstMultiply>::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(
    CreateLeaf<GammaConst<N,m> >::make(l),
    CreateLeaf<QDPExpr<T2,C2> >::make(r)));
}

//! QDPType * GammaConst
/*! @ingroup group1 */
template<class T1,class C1,int N,int m>
__device__ inline typename MakeReturn<BinaryNode<OpMultiplyGammaConst,
  typename CreateLeaf<QDPType<T1,C1> >::Leaf_t,
  typename CreateLeaf<GammaConst<N,m> >::Leaf_t>,
  typename BinaryReturn<C1,GammaConst<N,m>,OpMultiplyGammaConst>::Type_t >::Expression_t
operator*(const QDPType<T1,C1> & l,const GammaConst<N,m> & r)
{
  typedef BinaryNode<OpMultiplyGammaConst,
    typename CreateLeaf<QDPType<T1,C1> >::Leaf_t,
    typename CreateLeaf<GammaConst<N,m> >::Leaf_t> Tree_t;
  typedef typename BinaryReturn<C1,GammaConst<N,m>,OpMultiplyGammaConst>::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(
    CreateLeaf<QDPType<T1,C1> >::make(l),
    CreateLeaf<GammaConst<N,m> >::make(r)));
}

//! QDPExpr * GammaConst
template<class T1,class C1,int N,int m>
__device__ inline typename MakeReturn<BinaryNode<OpMultiplyGammaConst,
  typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t,
  typename CreateLeaf<GammaConst<N,m> >::Leaf_t>,
  typename BinaryReturn<C1,GammaConst<N,m>,OpMultiplyGammaConst>::Type_t >::Expression_t
operator*(const QDPExpr<T1,C1> & l,const GammaConst<N,m> & r)
{
  typedef BinaryNode<OpMultiplyGammaConst,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t,
    typename CreateLeaf<GammaConst<N,m> >::Leaf_t> Tree_t;
  typedef typename BinaryReturn<C1,GammaConst<N,m>,OpMultiplyGammaConst>::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(
    CreateLeaf<QDPExpr<T1,C1> >::make(l),
    CreateLeaf<GammaConst<N,m> >::make(r)));
}


//! GammaType * QDPType
/*! @ingroup group1 */
template<int N,class T2,class C2>
__device__ inline typename MakeReturn<BinaryNode<OpGammaTypeMultiply,
  typename CreateLeaf<GammaType<N> >::Leaf_t,
  typename CreateLeaf<QDPType<T2,C2> >::Leaf_t>,
  typename BinaryReturn<GammaType<N>,C2,OpGammaTypeMultiply>::Type_t >::Expression_t
operator*(const GammaType<N> & l,const QDPType<T2,C2> & r)
{
  typedef BinaryNode<OpGammaTypeMultiply,
    typename CreateLeaf<GammaType<N> >::Leaf_t,
    typename CreateLeaf<QDPType<T2,C2> >::Leaf_t> Tree_t;
  typedef typename BinaryReturn<GammaType<N>,C2,OpGammaTypeMultiply>::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(
    CreateLeaf<GammaType<N> >::make(l),
    CreateLeaf<QDPType<T2,C2> >::make(r)));
}

//! GammaType * QDPExpr
template<int N,class T2,class C2>
__device__ inline typename MakeReturn<BinaryNode<OpGammaTypeMultiply,
  typename CreateLeaf<GammaType<N> >::Leaf_t,
  typename CreateLeaf<QDPExpr<T2,C2> >::Leaf_t>,
  typename BinaryReturn<GammaType<N>,C2,OpGammaTypeMultiply>::Type_t >::Expression_t
operator*(const GammaType<N> & l,const QDPExpr<T2,C2> & r)
{
  typedef BinaryNode<OpGammaTypeMultiply,
    typename CreateLeaf<GammaType<N> >::Leaf_t,
    typename CreateLeaf<QDPExpr<T2,C2> >::Leaf_t> Tree_t;
  typedef typename BinaryReturn<GammaType<N>,C2,OpGammaTypeMultiply>::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(
    CreateLeaf<GammaType<N> >::make(l),
    CreateLeaf<QDPExpr<T2,C2> >::make(r)));
}

//! QDPType * GammaType
/*! @ingroup group1 */
template<class T1,class C1,int N>
__device__ inline typename MakeReturn<BinaryNode<OpMultiplyGammaType,
  typename CreateLeaf<QDPType<T1,C1> >::Leaf_t,
  typename CreateLeaf<GammaType<N> >::Leaf_t>,
  typename BinaryReturn<C1,GammaType<N>,OpMultiplyGammaType>::Type_t >::Expression_t
operator*(const QDPType<T1,C1> & l,const GammaType<N> & r)
{
  typedef BinaryNode<OpMultiplyGammaType,
    typename CreateLeaf<QDPType<T1,C1> >::Leaf_t,
    typename CreateLeaf<GammaType<N> >::Leaf_t> Tree_t;
  typedef typename BinaryReturn<C1,GammaType<N>,OpMultiplyGammaType>::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(
    CreateLeaf<QDPType<T1,C1> >::make(l),
    CreateLeaf<GammaType<N> >::make(r)));
}

//! QDPExpr * GammaType
template<class T1,class C1,int N>
__device__ inline typename MakeReturn<BinaryNode<OpMultiplyGammaType,
  typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t,
  typename CreateLeaf<GammaType<N> >::Leaf_t>,
  typename BinaryReturn<C1,GammaType<N>,OpMultiplyGammaType>::Type_t >::Expression_t
operator*(const QDPExpr<T1,C1> & l,const GammaType<N> & r)
{
  typedef BinaryNode<OpMultiplyGammaType,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t,
    typename CreateLeaf<GammaType<N> >::Leaf_t> Tree_t;
  typedef typename BinaryReturn<C1,GammaType<N>,OpMultiplyGammaType>::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(
    CreateLeaf<QDPExpr<T1,C1> >::make(l),
    CreateLeaf<GammaType<N> >::make(r)));
}


//-----------------------------------------------------------------------------
// Additional operators
//-----------------------------------------------------------------------------

//! GammaConstDP * QDPType
/*! @ingroup group1 */
template<int N,int m,class T2,class C2>
__device__ inline typename MakeReturn<BinaryNode<OpGammaConstDPMultiply,
  typename CreateLeaf<GammaConstDP<N,m> >::Leaf_t,
  typename CreateLeaf<QDPType<T2,C2> >::Leaf_t>,
  typename BinaryReturn<GammaConstDP<N,m>,C2,OpGammaConstDPMultiply>::Type_t >::Expression_t
operator*(const GammaConstDP<N,m> & l,const QDPType<T2,C2> & r)
{
  typedef BinaryNode<OpGammaConstDPMultiply,
    typename CreateLeaf<GammaConstDP<N,m> >::Leaf_t,
    typename CreateLeaf<QDPType<T2,C2> >::Leaf_t> Tree_t;
  typedef typename BinaryReturn<GammaConstDP<N,m>,C2,OpGammaConstDPMultiply>::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(
    CreateLeaf<GammaConstDP<N,m> >::make(l),
    CreateLeaf<QDPType<T2,C2> >::make(r)));
}

//! GammaConstDP * QDPExpr
template<int N,int m,class T2,class C2>
__device__ inline typename MakeReturn<BinaryNode<OpGammaConstDPMultiply,
  typename CreateLeaf<GammaConstDP<N,m> >::Leaf_t,
  typename CreateLeaf<QDPExpr<T2,C2> >::Leaf_t>,
  typename BinaryReturn<GammaConstDP<N,m>,C2,OpGammaConstDPMultiply>::Type_t >::Expression_t
operator*(const GammaConstDP<N,m> & l,const QDPExpr<T2,C2> & r)
{
  typedef BinaryNode<OpGammaConstDPMultiply,
    typename CreateLeaf<GammaConstDP<N,m> >::Leaf_t,
    typename CreateLeaf<QDPExpr<T2,C2> >::Leaf_t> Tree_t;
  typedef typename BinaryReturn<GammaConstDP<N,m>,C2,OpGammaConstDPMultiply>::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(
    CreateLeaf<GammaConstDP<N,m> >::make(l),
    CreateLeaf<QDPExpr<T2,C2> >::make(r)));
}

//! QDPType * GammaConstDP
/*! @ingroup group1 */
template<class T1,class C1,int N,int m>
__device__ inline typename MakeReturn<BinaryNode<OpMultiplyGammaConstDP,
  typename CreateLeaf<QDPType<T1,C1> >::Leaf_t,
  typename CreateLeaf<GammaConstDP<N,m> >::Leaf_t>,
  typename BinaryReturn<C1,GammaConstDP<N,m>,OpMultiplyGammaConstDP>::Type_t >::Expression_t
operator*(const QDPType<T1,C1> & l,const GammaConstDP<N,m> & r)
{
  typedef BinaryNode<OpMultiplyGammaConstDP,
    typename CreateLeaf<QDPType<T1,C1> >::Leaf_t,
    typename CreateLeaf<GammaConstDP<N,m> >::Leaf_t> Tree_t;
  typedef typename BinaryReturn<C1,GammaConstDP<N,m>,OpMultiplyGammaConstDP>::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(
    CreateLeaf<QDPType<T1,C1> >::make(l),
    CreateLeaf<GammaConstDP<N,m> >::make(r)));
}

//! QDPExpr * GammaConstDP
template<class T1,class C1,int N,int m>
__device__ inline typename MakeReturn<BinaryNode<OpMultiplyGammaConstDP,
  typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t,
  typename CreateLeaf<GammaConstDP<N,m> >::Leaf_t>,
  typename BinaryReturn<C1,GammaConstDP<N,m>,OpMultiplyGammaConstDP>::Type_t >::Expression_t
operator*(const QDPExpr<T1,C1> & l,const GammaConstDP<N,m> & r)
{
  typedef BinaryNode<OpMultiplyGammaConstDP,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t,
    typename CreateLeaf<GammaConstDP<N,m> >::Leaf_t> Tree_t;
  typedef typename BinaryReturn<C1,GammaConstDP<N,m>,OpMultiplyGammaConstDP>::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(
    CreateLeaf<QDPExpr<T1,C1> >::make(l),
    CreateLeaf<GammaConstDP<N,m> >::make(r)));
}


//! GammaTypeDP * QDPType
/*! @ingroup group1 */
template<int N,class T2,class C2>
__device__ inline typename MakeReturn<BinaryNode<OpGammaTypeDPMultiply,
  typename CreateLeaf<GammaTypeDP<N> >::Leaf_t,
  typename CreateLeaf<QDPType<T2,C2> >::Leaf_t>,
  typename BinaryReturn<GammaTypeDP<N>,C2,OpGammaTypeDPMultiply>::Type_t >::Expression_t
operator*(const GammaTypeDP<N> & l,const QDPType<T2,C2> & r)
{
  typedef BinaryNode<OpGammaTypeDPMultiply,
    typename CreateLeaf<GammaTypeDP<N> >::Leaf_t,
    typename CreateLeaf<QDPType<T2,C2> >::Leaf_t> Tree_t;
  typedef typename BinaryReturn<GammaTypeDP<N>,C2,OpGammaTypeDPMultiply>::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(
    CreateLeaf<GammaTypeDP<N> >::make(l),
    CreateLeaf<QDPType<T2,C2> >::make(r)));
}

//! GammaTypeDP * QDPExpr
template<int N,class T2,class C2>
__device__ inline typename MakeReturn<BinaryNode<OpGammaTypeDPMultiply,
  typename CreateLeaf<GammaTypeDP<N> >::Leaf_t,
  typename CreateLeaf<QDPExpr<T2,C2> >::Leaf_t>,
  typename BinaryReturn<GammaTypeDP<N>,C2,OpGammaTypeDPMultiply>::Type_t >::Expression_t
operator*(const GammaTypeDP<N> & l,const QDPExpr<T2,C2> & r)
{
  typedef BinaryNode<OpGammaTypeDPMultiply,
    typename CreateLeaf<GammaTypeDP<N> >::Leaf_t,
    typename CreateLeaf<QDPExpr<T2,C2> >::Leaf_t> Tree_t;
  typedef typename BinaryReturn<GammaTypeDP<N>,C2,OpGammaTypeDPMultiply>::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(
    CreateLeaf<GammaTypeDP<N> >::make(l),
    CreateLeaf<QDPExpr<T2,C2> >::make(r)));
}

//! QDPType * GammaTypeDP
/*! @ingroup group1 */
template<class T1,class C1,int N>
__device__ inline typename MakeReturn<BinaryNode<OpMultiplyGammaTypeDP,
  typename CreateLeaf<QDPType<T1,C1> >::Leaf_t,
  typename CreateLeaf<GammaTypeDP<N> >::Leaf_t>,
  typename BinaryReturn<C1,GammaTypeDP<N>,OpMultiplyGammaTypeDP>::Type_t >::Expression_t
operator*(const QDPType<T1,C1> & l,const GammaTypeDP<N> & r)
{
  typedef BinaryNode<OpMultiplyGammaTypeDP,
    typename CreateLeaf<QDPType<T1,C1> >::Leaf_t,
    typename CreateLeaf<GammaTypeDP<N> >::Leaf_t> Tree_t;
  typedef typename BinaryReturn<C1,GammaTypeDP<N>,OpMultiplyGammaTypeDP>::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(
    CreateLeaf<QDPType<T1,C1> >::make(l),
    CreateLeaf<GammaTypeDP<N> >::make(r)));
}

//! QDPExpr * GammaTypeDP
template<class T1,class C1,int N>
__device__ inline typename MakeReturn<BinaryNode<OpMultiplyGammaTypeDP,
  typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t,
  typename CreateLeaf<GammaTypeDP<N> >::Leaf_t>,
  typename BinaryReturn<C1,GammaTypeDP<N>,OpMultiplyGammaTypeDP>::Type_t >::Expression_t
operator*(const QDPExpr<T1,C1> & l,const GammaTypeDP<N> & r)
{
  typedef BinaryNode<OpMultiplyGammaTypeDP,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t,
    typename CreateLeaf<GammaTypeDP<N> >::Leaf_t> Tree_t;
  typedef typename BinaryReturn<C1,GammaTypeDP<N>,OpMultiplyGammaTypeDP>::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(
    CreateLeaf<QDPExpr<T1,C1> >::make(l),
    CreateLeaf<GammaTypeDP<N> >::make(r)));
}


//-----------------------------------------------------------------------------
// Additional operators
//-----------------------------------------------------------------------------

// Explicit casts
template<class T1,class T2,class C2>
__device__ inline typename MakeReturn<UnaryNode<OpCast<T1>,
  typename CreateLeaf<QDPType<T2,C2> >::Leaf_t>,
  typename UnaryReturn<C2,OpCast<T1> >::Type_t>::Expression_t
peteCast(const T1&, const QDPType<T2,C2>& l)
{
  typedef UnaryNode<OpCast<T1>,
    typename CreateLeaf<QDPType<T2,C2> >::Leaf_t> Tree_t;
  typedef typename UnaryReturn<C2,OpCast<T1> >::Type_t Container_t;
  return MakeReturn<Tree_t,Container_t>::make(Tree_t(
    CreateLeaf<QDPType<T2,C2> >::make(l)));
}

} // namespace QDP

#endif