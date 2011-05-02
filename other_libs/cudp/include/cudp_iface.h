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
    int row,col;
  };
#ifndef __CUDA_ARCH__
  //typedef std::string NodeDataString;
  //typedef list<NodeDataString> ListNode;
  typedef list<LeafData> ListLeaf;
  typedef list<NodeData> ListNode;
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
  int totalsite;
  int threadsite;
  int Nthread;
  int Nblock_x;
  int Nblock_y;
  int         numSiteTable;
  bool        hasOrderedRep;
  int         start;
  int         end;
  void       *siteTable;
  void       *dest;
  int         numberLeafs;
  int         numberNodes;
  FlattenTag::LeafData *leafDataArray;
  FlattenTag::NodeData *nodeDataArray;
};


#endif
