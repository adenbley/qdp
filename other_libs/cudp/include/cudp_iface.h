#ifndef __CUQDP_IFACE
#define __CUQDP_IFACE

#include<list>

struct FlattenTag {
  struct LeafData {
    void * pointer;
  };
  typedef list<LeafData> ListLeaf;
  mutable ListLeaf listLeaf;
};


struct IfaceCudp {
  void       *dest;
  void       *opMeta;
  size_t      opMetaSize;
  int         numberLeafs;
  FlattenTag::LeafData *leafDataArray;
};


#endif
