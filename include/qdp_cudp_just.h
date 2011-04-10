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
    typedef void (*CudpFunction)(void *);
    typedef map<string,CudpFunction> MapFunction;
    typedef list<void *> ListHandle;
    typedef string Filename;

  public:
    void operator()(string pretty,void * data);
    void closeAllShared();

  private:
    CudpFunction loadShared(Filename filename);
    Filename buildFunction(string pretty);
    MapFunction mapFunction;
    ListHandle listHandle;
  };

    
  extern CudpJust theCudpJust;



}

#endif
