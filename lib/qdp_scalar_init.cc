
/*! @file
 * @brief Scalar init routines
 * 
 * Initialization routines for scalar implementation
 */


#include "qdp.h"

#if defined(QDP_USE_QMT_THREADS)
#include <qmt.h>
#endif

namespace QDP {

namespace ThreadReductions {
  REAL64* norm2_results;
  REAL64* innerProd_results;
}

//! Private flag for status
static bool isInit = false;

//! Turn on the machine
void QDP_initialize(int *argc, char ***argv) 
{
  if (isInit)
    QDP_error_exit("QDP already inited");

  Layout::init();   // setup extremely basic functionality in Layout

  isInit = true;

  //
  // add qmt inilisisation
  //
#ifdef QDP_USE_QMT_THREADS
    
  // Initialize threads
  cout << "QDP uses qmt threading: Initializing threads..." ;
  int thread_status = qmt_init();
  if( thread_status == 0 ) { 
    cout << "Success. We have " << qdpNumThreads() << " threads \n"; 
  }
  else { 
    cout << "Failure... qmt_init() returned " << thread_status << endl;
    QDP_abort(1);
  }
#else 
#ifdef QDP_USE_OMP_THREADS
  cout << "QDP uses OpenMP threading. We have " << qdpNumThreads() << " threads \n";
#endif
#endif


// Alloc space for reductions
  ThreadReductions::norm2_results = new REAL64 [ qdpNumThreads() ];
  if( ThreadReductions::norm2_results == 0x0 ) { 
    cout << "Failure... space for norm2 results failed "  << endl;
    QDP_abort(1);
  }

  ThreadReductions::innerProd_results = new REAL64 [ 2*qdpNumThreads() ];
  if( ThreadReductions::innerProd_results == 0x0 ) { 
    cout << "Failure... space for innerProd results failed "  << endl;
    QDP_abort(1);
  }

  // initialize the global streams
  QDPIO::cin.init(&std::cin);
  QDPIO::cout.init(&std::cout);
  QDPIO::cerr.init(&std::cerr);

  //
  // Process command line
  //

  // Look for help
  bool help_flag = false;
  for (int i=0; i<*argc; i++) 
  {
    if (strcmp((*argv)[i], "-h")==0)
      help_flag = true;
  }

  if (help_flag) 
  {
    fprintf(stderr,"Usage:    %s options\n",(*argv)[0]);
    fprintf(stderr,"options:\n");
    fprintf(stderr,"    -h        help\n");
#if defined(QDP_USE_PROFILING)   
    fprintf(stderr,"    -p        %%d [%d] profile level\n", 
	    getProfileLevel());
#endif
    fprintf(stderr,"    -pretty-level found 1 no exe/exe\n");
    fprintf(stderr,"                        2 orig/SPU\n");
    fprintf(stderr,"              not found 16 no exe/exe\n");
    fprintf(stderr,"                        32 no write/write\n");
    fprintf(stderr,"                        256 do nothing if -1\n");
    fprintf(stderr,"                        512 orig shift/spu shift\n");
    fprintf(stderr,"    -pretty-filename    pr.lst is written whether found or not\n");
    fprintf(stderr,"    -nspu               <INT>\n");
    fprintf(stderr,"    -spuloops           <INT> default=1 (1st not timed)\n");
    fprintf(stderr,"    -lattice            XxYxZxT\n");


    exit(1);
  }

  for (int i=1; i<*argc; i++) 
  {
#if defined(QDP_USE_PROFILING)   
    if (strcmp((*argv)[i], "-p")==0) 
    {
      int lev;
      sscanf((*argv)[++i], "%d", &lev);
      setProgramProfileLevel(lev);
    }
#endif
    if (strcmp((*argv)[i], "-pretty-filename")==0) 
    {
      char fn[2048];
      sscanf((*argv)[++i], "%s", fn);
      setPrettyFn( string(fn) );
    }

    if (strcmp((*argv)[i], "-pretty-level")==0) 
    {
      int lev;
      sscanf((*argv)[++i], "%d", &lev);
      setPrettyLevel(lev);
    }

    // if (strcmp((*argv)[i], "-nspu")==0) 
    // {
    //   int lev;
    //   sscanf((*argv)[++i], "%d", &lev);
    //   setNSPU(lev);
    // }

    // if (strcmp((*argv)[i], "-spuloops")==0) 
    // {
    //   int loops;
    //   sscanf((*argv)[++i], "%d", &loops);
    //   setSPUloops(loops);
    // }

    // if (strcmp((*argv)[i], "-lattice")==0) 
    // {
    //   int x,y,z,t;
    //   sscanf((*argv)[++i], "%dx%dx%dx%d", &x,&y,&z,&t);
    //   setLattice(x,y,z,t);
    // }


    if (i >= *argc) 
    {
      QDP_error_exit("missing argument at the end");
    }
  }

  initProfile(__FILE__, __func__, __LINE__);
}

//! Is the machine initialized?
bool QDP_isInitialized() {return isInit;}

//! Turn off the machine
void QDP_finalize()
{
#ifdef BUILD_CUDP
  theCudpJust.closeAllShared();
#endif

  if ( ThreadReductions::norm2_results != 0x0 ) { 
   delete [] ThreadReductions::norm2_results;
  }

  if ( ThreadReductions::innerProd_results != 0x0 ) { 
   delete [] ThreadReductions::innerProd_results;
  }

   
  // finalise qmt
  //
#if defined(QMT_USE_QMT_THREADS)

 

    // Finalize threads
    cout << "QDP use qmt threading: Finalizing threads" << endl;
    qmt_finalize();
#endif 

  printProfile();

  isInit = false;
}

//! Panic button
void QDP_abort(int status)
{
  QDP_finalize(); 
  exit(status);
}

//! Resumes QDP communications
void QDP_resume() {}

//! Suspends QDP communications
void QDP_suspend() {}


} // namespace QDP;
