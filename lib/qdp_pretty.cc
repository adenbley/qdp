#include "qdp.h"



namespace QDP {

  static int pretty_level = 0;       // 0-original    1-write_pretty functions     2-use_pretty_functions
  static string pretty_filename;
  
  static map<string,int> pretty;

  map<string,int>& 
  getPretty()
  {
    return pretty;
  }
  

  int
  getPrettyLevel() 
  {
    return pretty_level;
  }

  string
  getPrettyFn() 
  {
    return pretty_filename;
  }


  void
  setPrettyLevel(int n)
  {
    pretty_level=n;
  }

  void
  setPrettyFn(string fn)
  {
    pretty_filename=fn;
  }



} 


