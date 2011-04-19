#ifndef __CUQDP_IFACE
#define __CUQDP_IFACE



#ifndef __CUDA_ARCH__
#include<list>
#endif

struct FlattenTag {
  struct LeafData {
    void * pointer;
    int  misc;
  };
  struct NodeData {
    void * pointer;
  };
#ifndef __CUDA_ARCH__
  typedef std::string NodeDataString;
  typedef list<LeafData>       ListLeaf;
  typedef list<NodeDataString> ListNode;
  mutable ListNode listNode;
  mutable ListLeaf listLeaf;
#else
  __device__ FlattenTag(): count_leaf(0),count_node(0) {}
  int          numberLeafs;
  int          numberNodes;
  LeafData    *leafDataArray;
  NodeData    *nodeDataArray;
  mutable int  count_leaf;
  mutable int  count_node;
#endif
};



struct IfaceCudp {
  void       *dest;
  void       *opMeta;
  size_t      opMetaSize;
  int         numberLeafs;
  int         numberNodes;
  FlattenTag::LeafData *leafDataArray;
  FlattenTag::NodeData *nodeDataArray;
};


#endif
