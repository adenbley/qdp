#ifndef QDP_DEVICE_STORAGE_H
#define QDP_DEVICE_STORAGE_H

#include "qdp.h"

namespace QDP {

  class DeviceStorage {
  public:    
  DeviceStorage(): listCount() {}

    const static int max_storage = 16;

    void * getDevicePointer( void * hostPtr , size_t size );
    void freeAll();

  private:
    void * listHostPtr[max_storage];
    void * listDevPtr[max_storage];
    int    listCount;
  };

  extern DeviceStorage theDeviceStorage; 

}

#endif

