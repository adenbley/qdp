#include "qdp_device_storage.h"

namespace QDP {


  void * DeviceStorage::getDevicePointer( void * hostPtr , size_t size )
  {
    for (int i=0;i<listCount;++i)
      if (hostPtr == listHostPtr[i])
	return listDevPtr[i];
    if (listCount >= max_storage) {
      cout << "DeviceStorage::getDevicePointer increase max_storage!" << endl;
      exit(1);
    }
    QDPCUDA::getDeviceMem( (void **)(&listDevPtr[listCount]) , size);
    QDPCUDA::copyToDevice( listDevPtr[listCount] , hostPtr , size);
    listHostPtr[listCount]=hostPtr;
    return listDevPtr[listCount];
  }

  void DeviceStorage::freeAll()
  {
    for (int i=0;i<listCount;++i)
      QDPCUDA::freeDeviceMem( listDevPtr[i] );
    listCount=0;
  }


  DeviceStorage theDeviceStorage; 


}

