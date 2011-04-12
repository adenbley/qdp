#include "qdp.h"
#include "stdlib.h"
#include <dlfcn.h>


#include "cudp_iface.h"


using namespace QDP;

namespace QDP 
{

  CudpJust theCudpJust;

  void CudpJust::operator()(string pretty,void * data)
  {
    MapFunction::const_iterator call;
    call = mapFunction.find(pretty);
    if (call != mapFunction.end()) {
      cout << "cudp function already compiled" << endl;
      (*call).second(data);
    } else {
      cout << "building cudp function..." << endl;
      Filename filename = buildFunction(pretty);
      CudpFunction cudpFunction = loadShared(filename);

      // Sanity check
      for(MapFunction::iterator iter = mapFunction.begin() ; iter != mapFunction.end() ; ++iter ) 
	if (iter->second == cudpFunction)
	  QDP_error_exit("Same memory address found for another cudp function. Giving up!\n");

      mapFunction.insert(MapFunction::value_type(pretty,cudpFunction));
      cudpFunction(data);
    }
  }

  CudpJust::Filename CudpJust::buildFunction(string pretty)
  {
    char temp[]="/tmp/cudp_XXXXXX";
    mktemp(temp);
    if (!temp) {
      QDP_error_exit("error while creating temporary file /tmp/cudp_...\n");
    }
    cout << "building cudp function using temporary file: " << string(temp) << endl;
    string file_cu = string(temp) + ".cu";
    string file_o  = string(temp) + ".o";

    string gen;
    gen = "$QDP_INSTALL/bin/cudp_codegen.pl " + string(temp) + ".cu";
    cout << gen << endl;
    FILE * fileGenGpu;
    fileGenGpu=popen(gen.c_str(),"w");
    if (!fileGenGpu) {
      QDP_error_exit("error while calling PERL\n");
    }
    fprintf(fileGenGpu,"%s\n",pretty.c_str());
    pclose(fileGenGpu);

    int ret;
    gen = "nvcc -v -arch=sm_20 --compiler-options -fPIC,-shared -link " + file_cu + " -I$QDP_INSTALL/include -o " + file_o;
    cout << gen << endl;
    ret=system(gen.c_str());
    if (ret) {
      cout << "return value = " << ret << endl;
      QDP_error_exit("Nvcc error\n");
    }

    return file_o;
  }


  CudpJust::CudpFunction CudpJust::loadShared(CudpJust::Filename filename)
  {
    void *handle;
    handle = dlopen( filename.c_str() ,  RTLD_LAZY);

    if (!handle) {
      cout << string(dlerror()) << endl;
      QDP_error_exit("dlopen error\n");
    } else {
      cout << "LSB shared object loaded successfully" << endl;
    }

    listHandle.push_back(handle);

    void (*fptr)(void *);
    char *err;
    dlerror(); /* clear error code */
    *(void **)(&fptr) = dlsym(handle, "function_host");
    if ((err = dlerror()) != NULL) {
      cout << string(err) << endl;
      QDP_error_exit("dlsym error\n");
    } 

    cout << "symbol found" << endl;
    return fptr;
  }

  void CudpJust::closeAllShared()
  {
    cout << "cudp: closing " << listHandle.size() << " opened shared objects." << endl;
    for(ListHandle::iterator iter = listHandle.begin() ; iter != listHandle.end() ; ++iter ) 
      {
	dlclose(*iter);
      }
  }

}

