// -*- C++ -*-

/*! @file
 * @brief 
 *
 * 
 */

#ifndef QDP_CUDPJUST_H
#define QDP_CUDPJUST_H

#include<map>

namespace QDP {


  class CudpJust
  {
  public:
    typedef void (*Kernel)(void *);
    typedef char* Pretty;
    struct SharedLibEntry {
      Kernel kernel;
      Pretty pretty;
    };
    typedef map<string,SharedLibEntry> MapFunction;
    typedef list<void *> ListHandle;
    typedef string Filename;

    void setPath(string _path) { path = _path; }
    void operator()(string pretty,void * data);
    void closeAllShared();
    void loadAllShared();

  private:
    bool hasEnding (std::string const &fullString, std::string const &ending);
    int getdir(string dir, list<string> &files);
    SharedLibEntry loadShared(Filename filename);
    Filename buildFunction(string pretty);
    MapFunction mapFunction;
    ListHandle listHandle;
    string path;
  };

    
  extern CudpJust theCudpJust;



}

#endif
