#ifndef __CUQDP_IFACE
#define __CUQDP_IFACE

struct FlattenTag {
  inline
  FlattenTag() : iadr(0), totalsize(0) {}
  const static  size_t maxleaf = 20;
  mutable void * adr[maxleaf];
  mutable unsigned int size[maxleaf];
  mutable unsigned int custom[maxleaf];
  mutable unsigned int leaftype[maxleaf];
  mutable unsigned int iadr;
  mutable unsigned int totalsize;
};

struct CUDA_iface_eval {
  void       *dest;
  FlattenTag flatten;
  void       *opMeta;
  size_t      opMetaSize;
};

#endif
