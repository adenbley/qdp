#ifndef QDP_THREADMETRIC_H
#define QDP_THREADMETRIC_H

//#include "qdp.h"

namespace QDP {

  class CudaThreadMetric {
  public:
  CudaThreadMetric(): threadsite(1), Nthread(32) {}
    int threadsite;
    int Nthread;
  private:
    
  };

  extern CudaThreadMetric theCudaThreadMetric;

}

#endif
