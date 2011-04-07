#ifndef QDP_PRETTY_INCLUDE
#define QDP_PRETTY_INCLUDE

#include "map"

namespace QDP {

  std::map<string,int>& getPretty();

  int getPrettyLevel();
  void setPrettyLevel(int n);

  string getPrettyFn();
  void setPrettyFn(string fn);


} 

#endif 
