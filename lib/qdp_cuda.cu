#include <iostream>

#include "qdp_cuda.h"

using namespace std;

namespace QDPCUDA {

  size_t total_device_memory = 0;
  
  void getDeviceMem(void **mem , size_t size)
  {
    cudaError_t ret;
    ret = cudaMalloc(mem,size);
#ifdef GPU_DEBUG
    cout << "cudaMalloc     " << size << " : " << *mem << " " << string(cudaGetErrorString(ret)) << endl;
#endif
    if (ret == cudaSuccess)
      total_device_memory += size;
  }
  void freeDeviceMem(void *mem)
  {
    cudaError_t ret;
    ret = cudaFree(mem);
#ifdef GPU_DEBUG
    cout << "cudaFree   : " << string(cudaGetErrorString(ret)) << endl;
#endif
  }
  void getHostMem(void **mem , size_t size)
  {
    cudaError_t ret;
    ret = cudaHostAlloc(mem,size,cudaHostAllocDefault);
#ifdef GPU_DEBUG
    cout << "cudaHostMalloc " << size << " : " << string(cudaGetErrorString(ret)) << endl;
#endif
  }
  void freeHostMem(void *mem)
  {
    cudaError_t ret;
    ret = cudaFreeHost(mem);
#ifdef GPU_DEBUG
    cout << "cudaFreeHost   : " << string(cudaGetErrorString(ret)) << endl;
#endif
  }

  void copyToDevice(void *dest , void const *src , size_t size)
  {
    cudaError_t ret;
    ret = cudaMemcpy(dest,src,size,cudaMemcpyHostToDevice);
#ifdef GPU_DEBUG
    cout << "cudaMemcpy to device: " << string(cudaGetErrorString(ret)) << endl;
#endif
  }
  void copyToHost(void *dest , void const *src , size_t size)
  {
    cudaError_t ret;
    ret = cudaMemcpy(dest,src,size,cudaMemcpyDeviceToHost);
#ifdef GPU_DEBUG
    cout << "cudaMemcpy to host: " << string(cudaGetErrorString(ret)) << endl;
#endif
  }
  void copyHostToHost(void *dest , void const *src , size_t size)
  {
    cudaError_t ret;
    ret = cudaMemcpy(dest,src,size,cudaMemcpyHostToHost);
#ifdef GPU_DEBUG
    cout << "cudaMemcpy host to host: " << string(cudaGetErrorString(ret)) << endl;
#endif
  }

}

