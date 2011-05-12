#include "qdp.h"
#include "stdlib.h"
#include <dlfcn.h>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>


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
#ifdef GPU_DEBUG
      cout << "cudp function already compiled" << endl;
#endif
      (*call).second.kernel(data);
    } else {
#ifdef GPU_DEBUG
      cout << "building cudp function..." << endl;
#endif
      Filename filename = buildFunction(pretty);
      SharedLibEntry entry = loadShared(filename);

      // Sanity check
      for(MapFunction::iterator iter = mapFunction.begin() ; iter != mapFunction.end() ; ++iter ) 
	if (iter->second.kernel == entry.kernel)
	  QDP_error_exit("Same memory address found for another cudp function. Giving up!\n");

      mapFunction.insert(MapFunction::value_type(pretty,entry));
      entry.kernel(data);
    }
  }

  CudpJust::Filename CudpJust::buildFunction(string pretty)
  {
    char temp[]="/tmp/cudp_XXXXXX";
    char * tempfn = &temp[5];
    mktemp(temp);
    if (!temp) {
      QDP_error_exit("error while creating temporary file /tmp/cudp_...\n");
    }
    cout << "building cudp function using temporary file: " << string(temp) << endl;
    string file_cu = string(temp) + ".cu";
    string file_o  = path + string(tempfn) + ".o";

    string gen;
    gen = "$QDP_INSTALL/bin/cudp_codegen.pl " + file_cu;
    cout << gen << endl;
    FILE * fileGenGpu;
    fileGenGpu=popen(gen.c_str(),"w");
    if (!fileGenGpu) {
      QDP_error_exit("error while calling PERL\n");
    }
    fprintf(fileGenGpu,"%s\n",pretty.c_str());
    pclose(fileGenGpu);

    int ret;
    gen = string(CUDA_DIR) + "/bin/nvcc -v ";
    gen = gen + "-arch=" + string(QDP_GPUARCH);
    gen = gen + " -m64 --compiler-options -fPIC,-shared -link ";
    gen = gen + file_cu + " -I$QDP_INSTALL/include -o " + file_o;
    cout << gen << endl;
    ret=system(gen.c_str());
    if (ret) {
      cout << "return value = " << ret << endl;
      QDP_error_exit("Nvcc error\n");
    }

    return file_o;
  }


  CudpJust::SharedLibEntry CudpJust::loadShared(CudpJust::Filename filename)
  {
    CudpJust::SharedLibEntry entry;

    void *handle;
    handle = dlopen( filename.c_str() ,  RTLD_LAZY);

    if (!handle) {
      cout << string(dlerror()) << endl;
      QDP_error_exit("dlopen error\n");
    } else {
      cout << "LSB shared object loaded successfully" << endl;
    }

    listHandle.push_back(handle);

    {
      void (*fptr)(void *);
      char *err;
      dlerror(); /* clear error code */
      *(void **)(&fptr) = dlsym(handle, "function_host");
      if ((err = dlerror()) != NULL) {
	cout << string(err) << endl;
	QDP_error_exit("dlsym error\n");
      }
      entry.kernel = fptr;
    }

    {
      char *fptr;
      char *err;
      dlerror(); /* clear error code */
      *(void **)(&fptr) = dlsym(handle, "pretty");
      if ((err = dlerror()) != NULL) {
	cout << string(err) << endl;
	QDP_error_exit("dlsym error\n");
      }
      entry.pretty = fptr;
    }

#ifdef GPU_DEBUG
    cout << "symbol found" << endl;
#endif
    return entry;
  }

  void CudpJust::closeAllShared()
  {
    cout << "cudp: closing " << listHandle.size() << " opened shared objects." << endl;
    for(ListHandle::iterator iter = listHandle.begin() ; iter != listHandle.end() ; ++iter ) 
      {
	dlclose(*iter);
      }
  }


  bool CudpJust::hasEnding (std::string const &fullString, std::string const &ending)
  {
    if (fullString.length() > ending.length()) {
      return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
      return false;
    }
  }


  int CudpJust::getdir(string dir, list<string> &files)
  {
    DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(dir.c_str())) == NULL) {
      cout << "Error(" << errno << ") opening " << dir << endl;
      return errno;
    }

    cout << "kernels found:" << endl;
    while ((dirp = readdir(dp)) != NULL) {
      if (dirp->d_type == DT_REG) {
	if (hasEnding(string(dirp->d_name),".o")) {
	    cout << dir + "/" + string(dirp->d_name) << "\n";
	    files.push_back(dir+"/"+string(dirp->d_name));
	}
      }
    }
    closedir(dp);
    return 0;
  }



  void CudpJust::loadAllShared()
  {
    list<Filename> filenames;
    if (!getdir( path , filenames )) {

      for(list<Filename>::iterator iter = filenames.begin() ; iter != filenames.end() ; ++iter ) {
	cout << "loading " << *iter << endl;
	SharedLibEntry entry = loadShared( *iter );
	mapFunction.insert(MapFunction::value_type(string(entry.pretty),entry));
      }
    }
  }


}

