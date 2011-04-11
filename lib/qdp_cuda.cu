#include <iostream>

#include "qdp_cuda.h"

using namespace std;

namespace QDPCUDA {

  
  void getDeviceMem(void **mem , size_t size)
  {
    cudaError_t ret;
    ret = cudaMalloc(mem,size);
    cout << "cudaMalloc     " << size << " : " << *mem << " " << string(cudaGetErrorString(ret)) << endl;
  }
  void freeDeviceMem(void *mem)
  {
    cudaError_t ret;
    ret = cudaFree(mem);
    cout << "cudaFree   : " << string(cudaGetErrorString(ret)) << endl;
  }
  void getHostMem(void **mem , size_t size)
  {
    cudaError_t ret;
    ret = cudaHostAlloc(mem,size,cudaHostAllocDefault);
    cout << "cudaHostMalloc " << size << " : " << string(cudaGetErrorString(ret)) << endl;
  }
  void freeHostMem(void *mem)
  {
    cudaError_t ret;
    ret = cudaFreeHost(mem);
    cout << "cudaFreeHost   : " << string(cudaGetErrorString(ret)) << endl;
  }

  void copyToDevice(void *dest , void *src , size_t size)
  {
    cudaError_t ret;
    ret = cudaMemcpy(dest,src,size,cudaMemcpyHostToDevice);
    cout << "cudaMemcpy to device: " << string(cudaGetErrorString(ret)) << endl;
  }
  void copyToHost(void *dest , void *src , size_t size)
  {
    cudaError_t ret;
    ret = cudaMemcpy(dest,src,size,cudaMemcpyDeviceToHost);
    cout << "cudaMemcpy to host: " << string(cudaGetErrorString(ret)) << endl;
  }

}

