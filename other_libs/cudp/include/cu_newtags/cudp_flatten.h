#ifndef QDP_TAG_FLATTEN_H
#define QDP_TAG_FLATTEN_H

namespace QDP {

  template<class T, class C>
    struct LeafFunctor<QDPType<T,C>, FlattenTag>
    {
      typedef int Type_t;
      __device__
	static Type_t apply(const QDPType<T,C> &s, const FlattenTag &f)
      { 
#ifdef __CUDA_ARCH__
	return LeafFunctor<C,FlattenTag>::apply(static_cast<const C&>(s),f);
#endif
      }
    };

  template<class T>
    struct LeafFunctor<OLattice<T>, FlattenTag>
    {
      typedef int Type_t;
      __device__ inline static Type_t apply(const OLattice<T> &a, const FlattenTag &f)
      {
#ifdef __CUDA_ARCH__
	OLattice<T>& b = const_cast<OLattice<T>&>(a);

#ifdef GPU_DEBUG
	if (f.count_leaf >= f.numberLeafs) {
	  printf("Oops: f.count >= f.numberLeafs!\n");
	}
#endif

	b.setF( f.leafDataArray[ f.count_leaf ].pointer );
#ifdef GPU_DEBUG
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	  printf("Flatten: OLattice     : %d %llx %d\n",f.count_leaf,f.leafDataArray[ f.count_leaf ].pointer,f.leafDataArray[ f.count_leaf ].misc );
#endif
	f.count_leaf++;

	return 0;
#endif
      }
    };



  template<class T>
    struct LeafFunctor<OScalar<T>, FlattenTag>
    {
      typedef int Type_t;
      __device__ inline static Type_t apply(const OScalar<T> &a, const FlattenTag &f)
      {
#ifdef __CUDA_ARCH__
	OScalar<T>& b = const_cast<OScalar<T>&>(a);

#ifdef GPU_DEBUG
	if (f.count_leaf >= f.numberLeafs) {
	  printf("Oops: f.count >= f.numberLeafs (OScalar)!\n");
	}
#endif

	b.setF( f.leafDataArray[ f.count_leaf ].pointer );
#ifdef GPU_DEBUG
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	  printf("Flatten: OScalar      : %d %llx %d\n",f.count_leaf,f.leafDataArray[ f.count_leaf ].pointer,f.leafDataArray[ f.count_leaf ].misc );
#endif
	f.count_leaf++;

	return 0;
#endif
      }
    };



  template<int N, int m>
    struct LeafFunctor<GammaConst<N, m>, FlattenTag>
    {
      typedef int Type_t;
      __device__ 
	inline static Type_t apply(const GammaConst<N, m> &a, const FlattenTag &f)
      {
#ifdef __CUDA_ARCH__
#ifdef GPU_DEBUG
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	  printf("Flatten: GammaConst   : %d %llx %d\n",f.count_leaf,f.leafDataArray[ f.count_leaf ].pointer,f.leafDataArray[ f.count_leaf ].misc );
#endif
	f.count_leaf++;
#endif
	return 0;
      }
    };



  template<int N>
    struct LeafFunctor<GammaType<N>, FlattenTag>
    {
      typedef int Type_t;
      __device__ 
	inline static Type_t apply(const GammaType<N> &a, const FlattenTag &f)
      {
#ifdef __CUDA_ARCH__
	a.setElem( f.leafDataArray[ f.count_leaf ].misc );
#ifdef GPU_DEBUG
	if (blockIdx.x * blockDim.x + threadIdx.x == 0)
	  printf("Flatten: GammaType<N> : %d %llx %d\n",f.count_leaf,f.leafDataArray[ f.count_leaf ].pointer,f.leafDataArray[ f.count_leaf ].misc );
#endif
	f.count_leaf++;
#endif
	return 0;
      }
    };




}
#endif

