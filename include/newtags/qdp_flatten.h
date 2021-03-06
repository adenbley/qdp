#ifndef QDP_TAG_FLATTEN_H
#define QDP_TAG_FLATTEN_H

namespace QDP {

  template<class T, class C>
    struct LeafFunctor<QDPType<T,C>, FlattenTag>
    {
      typedef int Type_t;
      static Type_t apply(const QDPType<T,C> &s, const FlattenTag &f)
      { 
	return LeafFunctor<C,FlattenTag>::apply(static_cast<const C&>(s),f);
      }
    };


  template<class T>
    struct LeafFunctor<OLattice<T>, FlattenTag>
    {
      typedef int Type_t;
      inline static Type_t apply(const OLattice<T> &a, const FlattenTag &f)
      {
	FlattenTag::LeafData leafData;
	leafData.pointer = (void *)( a.Fd ); 
	leafData.misc = 0;
	f.listLeaf.push_front(leafData);
#ifdef GPU_DEBUG
	cout << "Flatten: OLattice:" << f.listLeaf.size()-1 << " " << leafData.pointer << " " << leafData.misc << endl;
#endif
	return 0;
      }
    };


  template<class T>
    struct LeafFunctor<OScalar<T>, FlattenTag>
    {
      typedef int Type_t;
      inline static Type_t apply(const OScalar<T> &a, const FlattenTag &f)
      {
	FlattenTag::LeafData leafData;
	leafData.pointer = (void *)( a.Fd ); 
	leafData.misc = 0;
	f.listLeaf.push_front(leafData);
#ifdef GPU_DEBUG
	cout << "Flatten: OScalar:" << f.listLeaf.size()-1 << " " << leafData.pointer << " " << leafData.misc << endl;
#endif
	return 0;
      }
    };


  template<int N, int m>
    struct LeafFunctor<GammaConst<N, m>, FlattenTag>
    {
      typedef int Type_t;
      inline static Type_t apply(const GammaConst<N, m> &a, const FlattenTag &f)
      {
	FlattenTag::LeafData leafData;
	leafData.pointer = NULL;
	leafData.misc = 0;
	f.listLeaf.push_front(leafData);
#ifdef GPU_DEBUG
	cout << "Flatten: GammaConst:" << f.listLeaf.size()-1 << " " << leafData.pointer << " " << leafData.misc << endl;
#endif
	return 0;
      }
    };


  template<int N>
    struct LeafFunctor<GammaType<N>, FlattenTag>
    {
      typedef int Type_t;
      inline static Type_t apply(const GammaType<N> &a, const FlattenTag &f)
      {
	FlattenTag::LeafData leafData;
	leafData.pointer = NULL;
	leafData.misc = a.elem();
	f.listLeaf.push_front(leafData);

#ifdef GPU_DEBUG
	cout << "Flatten: GammaType:" << f.listLeaf.size()-1 << " " << leafData.pointer << " " << leafData.misc << endl;
#endif

	return 0;
      }
    };

}
#endif

