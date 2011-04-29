
#include "qdp.h"

namespace QDP {


  void * DeviceStorage::getDevicePointer( void * hostPtr , size_t size )
  {
    cout << "DeviceStorage::getDevicePointer(" << hostPtr << "," << size << ")" << endl;
    for (int i=0;i<listCount;++i) {
      if (hostPtr == listHostPtr[i]) {
	cout << "found returning " << listDevPtr[i] << endl;
	return listDevPtr[i];
      }
    }
    if (listCount >= max_storage) {
      cout << "DeviceStorage::getDevicePointer increase max_storage!" << endl;
      exit(1);
    }
    QDPCUDA::getDeviceMem( (void **)(&listDevPtr[listCount]) , size);
    QDPCUDA::copyToDevice( listDevPtr[listCount] , hostPtr , size);
    listHostPtr[listCount]=hostPtr;
    cout << "not found, storage allocated, returning " << listDevPtr[listCount] << endl;
    return listDevPtr[listCount++];
  }

  void DeviceStorage::freeAll()
  {
    for (int i=0;i<listCount;++i)
      QDPCUDA::freeDeviceMem( listDevPtr[i] );
    listCount=0;
  }

   DeviceStorage theDeviceStorage; 


}


