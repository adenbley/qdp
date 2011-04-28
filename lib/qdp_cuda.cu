#include <iostream>

#include "malloc.h"
#include "qdp_cuda.h"

using namespace std;

namespace QDPCUDA {


void inline cudp_check_error(std::string msg,cudaError_t& ret)
{
#ifdef GPU_DEBUG
    cout << msg << endl;
    if (ret != cudaSuccess) {
	cout << string(cudaGetErrorString(ret)) << endl;
	exit(1);
    }
#else
    if (ret != cudaSuccess) {
	cout << msg << endl;
	cout << string(cudaGetErrorString(ret)) << endl;
	exit(1);
    }
#endif
}


  void hostRegister(void * ptr , size_t size, unsigned int flags)
  {
    cudaError_t ret;
    ret = cudaHostRegister(ptr, size, flags);
    cudp_check_error("hostRegister",ret);
  }
  
  void hostUnregister(void * ptr )
  {
    cudaError_t ret;
    ret = cudaHostUnregister(ptr);
    cudp_check_error("hostUnregister",ret);
  }
  
  void getDeviceMem(void **mem , size_t size)
  {
    cudaError_t ret;
    ret = cudaMalloc(mem,size);
    cudp_check_error("cudaMalloc",ret);
  }

  void freeDeviceMem(void *mem)
  {
    cudaError_t ret;
    ret = cudaFree(mem);
    cudp_check_error("cudaFree",ret);
  }

  void getHostMem(void **mem , size_t size)
  {
    cudaError_t ret;
    ret = cudaHostAlloc(mem,size,cudaHostAllocDefault);
    cudp_check_error("cudaHostMalloc",ret);
  }

  void freeHostMem(void *mem)
  {
    cudaError_t ret;
    ret = cudaFreeHost(mem);
    cudp_check_error("cudaFreeHost",ret);
  }

  void copyToDevice(void *dest , void const *src , size_t size)
  {
    cudaError_t ret;
    ret = cudaMemcpy(dest,src,size,cudaMemcpyHostToDevice);
    cudp_check_error("cudaMemcpy to device",ret);
  }

  void copyToHost(void *dest , void const *src , size_t size)
  {
    cudaError_t ret;
    ret = cudaMemcpy(dest,src,size,cudaMemcpyDeviceToHost);
    cudp_check_error("cudaMemcpy to host",ret);
  }

  void copyHostToHost(void *dest , void const *src , size_t size)
  {
    cudaError_t ret;
    ret = cudaMemcpy(dest,src,size,cudaMemcpyHostToHost);
    cudp_check_error("cudaMemcpy host to host",ret);
  }



}

