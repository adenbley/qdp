#ifndef QDP_TAG_ON_DEVICE_H
#define QDP_TAG_ON_DEVICE_H


namespace QDP {

  struct OnDeviceTag {
  OnDeviceTag(): count(0),count_lattice(0) {}
    mutable int count;
    mutable int count_lattice;
  };


  template<class T, class C>
    struct LeafFunctor<QDPType<T,C>, OnDeviceTag>
    {
      typedef bool Type_t;
      static Type_t apply(const QDPType<T,C> &s, const OnDeviceTag &f)
      { 
	return LeafFunctor<C,OnDeviceTag>::apply(static_cast<const C&>(s),f);
      }
    };


  template<int N, int m>
    struct LeafFunctor<GammaConst<N, m>, OnDeviceTag>
    {
      typedef bool Type_t;
      inline static Type_t apply(const GammaConst<N, m> &a, const OnDeviceTag &f)
      {
	return true;
      }
    };


  template<int N>
    struct LeafFunctor<GammaType<N>, OnDeviceTag>
    {
      typedef bool Type_t;
      inline static Type_t apply(const GammaType<N> &a, const OnDeviceTag &f)
      {
	return true;
      }
    };


  template<class T>
    struct LeafFunctor<OLattice<T>, OnDeviceTag>
    {
      typedef bool Type_t;
      inline static Type_t apply(const OLattice<T> &a, const OnDeviceTag &f)
      {
	f.count_lattice++;
	if (a.onDevice())
	  f.count++;
	return a.onDevice();
      }
    };


  template<class T>
    struct LeafFunctor<OScalar<T>, OnDeviceTag>
    {
      typedef bool Type_t;
      inline static Type_t apply(const OScalar<T> &a, const OnDeviceTag &f)
      {
	return true;
      }
    };

}

#endif


