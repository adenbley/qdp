#ifndef QDP_TAG_SCALAR_TO_DEVICE
#define QDP_TAG_SCALAR_TO_DEVICE

namespace QDP {


  struct OScalarToDeviceTag {
  OScalarToDeviceTag(bool _toDevice):count(0),toDevice(_toDevice) {}
    mutable int count;
    bool toDevice;
  };


  template<class T, class C>
    struct LeafFunctor<QDPType<T,C>, OScalarToDeviceTag>
    {
      typedef int Type_t;
      static Type_t apply(const QDPType<T,C> &s, const OScalarToDeviceTag &f)
      { 
	return LeafFunctor<C,OScalarToDeviceTag>::apply(static_cast<const C&>(s),f);
      }
    };


  template<class T>
    struct LeafFunctor<OScalar<T>, OScalarToDeviceTag>
    {
      typedef int Type_t;
      inline static Type_t apply(const OScalar<T> &a, const OScalarToDeviceTag &f)
      {
	if (f.toDevice) {
#ifdef GPU_DEBUG
	  cout << "copy OScalar to device" << endl;
#endif
	  a.getHostMem();
	  a.getDeviceMem();
	  a.copyToDevice();
	} else {
#ifdef GPU_DEBUG
	  cout << "free OScalar host and device memory" << endl;
#endif
	  a.freeHostMem();
	  a.freeDeviceMem();
	}
	f.count++;
	return 0;
      }
    };


  template<class T>
    struct LeafFunctor<OLattice<T>, OScalarToDeviceTag>
    {
      typedef int Type_t;
      inline static Type_t apply(const OLattice<T> &a, const OScalarToDeviceTag &f)
      {
	return 0;
      }
    };


  template<int N, int m>
    struct LeafFunctor<GammaConst<N, m>, OScalarToDeviceTag>
    {
      typedef int Type_t;
      inline static Type_t apply(const GammaConst<N, m> &a, const OScalarToDeviceTag &f)
      {
	return 0;
      }
    };


  template<int N>
    struct LeafFunctor<GammaType<N>, OScalarToDeviceTag>
    {
      typedef int Type_t;
      inline static Type_t apply(const GammaType<N> &a, const OScalarToDeviceTag &f)
      {
	return 0;
      }
    };


}


#endif


