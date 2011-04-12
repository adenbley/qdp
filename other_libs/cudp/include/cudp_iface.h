#ifndef __CUQDP_IFACE
#define __CUQDP_IFACE


#ifndef __CUDA_ARCH__
#warning cuda arch not defined 
#else
#warning cuda arch defined 
#endif

#ifndef __CUDA_ARCH__
#include<list>
#endif

struct FlattenTag {
  struct LeafData {
    void * pointer;
  };
#ifndef __CUDA_ARCH__
  typedef list<LeafData> ListLeaf;
  mutable ListLeaf listLeaf;
#else
  __device__ FlattenTag(): count(0) {}
  int          numberLeafs;
  LeafData    *leafDataArray;
  mutable int  count;
#endif
};



struct IfaceCudp {
  void       *dest;
  void       *opMeta;
  size_t      opMetaSize;
  int         numberLeafs;
  FlattenTag::LeafData *leafDataArray;
};


#endif
