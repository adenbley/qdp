#ifndef QDP_CUDA_H
#define QDP_CUDA_H


namespace QDPCUDA {

  extern size_t total_device_memory;

  void hostRegister(void * ptr , size_t size, unsigned int flags);
  void hostUnregister(void * ptr );
  void copyHostToHost(void *dest , void const *src , size_t size);
  void copyToHost(void *dest , void const *src , size_t size);
  void copyToDevice(void *dest , void const *src , size_t size);
  void freeDeviceMem(void *mem);
  void getDeviceMem(void **mem , size_t size);
  void freeHostMem(void *mem);
  void getHostMem(void **mem , size_t size);

}

#endif
