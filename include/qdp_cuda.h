#ifndef QDP_CUDA_H
#define QDP_CUDA_H

namespace QDPCUDA {

  void copyToHost(void *dest , void *src , size_t size);
  void copyToDevice(void *dest , void *src , size_t size);
  void freeDeviceMem(void *mem);
  void getDeviceMem(void **mem , size_t size);
  void freeHostMem(void *mem);
  void getHostMem(void **mem , size_t size);

}

#endif
