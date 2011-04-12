// -*- C++ -*-
// $Id: cudp_scalar_specific.h,v 1.38 2009/07/14 20:08:41 bjoo Exp $
//
// QDP data parallel interface
//
// Outer lattice routines specific to a scalar platform 

#ifndef QDP_SCALAR_SPECIFIC_H
#define QDP_SCALAR_SPECIFIC_H

#include "cudp_iface.h"
#include <iostream>

namespace QDP {


  template<class T, class C>
  struct LeafFunctor<QDPType<T,C>, FlattenTag>
  {
    typedef int Type_t;
    __device__
    static Type_t apply(const QDPType<T,C> &s, const FlattenTag &f)
    { 
#ifdef __CUDA_ARCH__
      return LeafFunctor<C,FlattenTag>::apply(static_cast<const C&>(s),f);
#endif
    }
  };

  template<class T>
  struct LeafFunctor<OLattice<T>, FlattenTag>
  {
    //typedef Reference<T> Type_t;
    typedef int Type_t;
    __device__
    inline static Type_t apply(const OLattice<T> &a, const FlattenTag &f)
    {
#ifdef __CUDA_ARCH__
      OLattice<T>& b = const_cast<OLattice<T>&>(a);

      if (f.count >= f.numberLeafs) {
	printf("Oops: f.count >= f.numberLeafs!\n");
      }

      b.setF( f.leafDataArray[ f.count ].pointer );
      if (threadIdx.x==0)
        printf("device: %d %llx\n",f.count,f.leafDataArray[ f.count ].pointer );
      f.count++;

      return 0;
#endif
    }
  };


//   template<class T>
//   struct LeafFunctor<OScalar<T>, FlattenTag>
//   {
//     //typedef Reference<T> Type_t;
//     typedef int Type_t;
//     inline static Type_t apply(const OScalar<T> &a, const FlattenTag &f)
//     {
//       if (f.iadr >= f.maxleaf) {
// 	printf("LeafFunctor<OLattice<T>, FlattenTag>::apply too many leafs\n");
// 	exit(1);
//       }
//       f.adr[ f.iadr ] = (size_t)( &a.elem() );
//       f.size[ f.iadr ] = sizeof( a.elem() );
//       f.leaftype[ f.iadr ] = 1;
//       f.custom[ f.iadr ] = 0;
//       f.iadr++;
//       //cout << "ppu: im OScalar... not yet implemented" << endl;
//       //exit(0);
//     }
//   };


  // template<>
  // struct LeafFunctor<OScalar<PScalar<PScalar<RScalar<int> > > >, FlattenTag>
  // {
  //   //typedef Reference<T> Type_t;
  //   typedef int Type_t;
  //   inline static Type_t apply(const OScalar<PScalar<PScalar<RScalar<int> > > > &a, const FlattenTag &f)
  //     {
  //       cout << "ppu: im OScalar" << endl;

  //       union {
  // 	unsigned int ui;
  // 	int si;
  //       } tmp;

  //       tmp.si = a.elem().elem().elem().elem();

  //       f.size[ f.iadr ] = tmp.ui;
  //       f.iadr++;
  //       printf("LeafFunctor<OScalar<PScalar<PScalar<RScalar<int> > > >, FlattenTag>::apply\n");
  //     }
  // };



//   template<int N, int m>
//   struct LeafFunctor<GammaConst<N, m>, FlattenTag>
//   {
//     //typedef Reference<T> Type_t;
//     typedef int Type_t;
//     inline static Type_t apply(const GammaConst<N, m> &a, const FlattenTag &f)
//     {
// #if defined(SPU_DEBUG)
//       printf("LeafFunctor<GammaConst<N, m>,FlattenTag>::apply \n");
// #endif
//       //return Type_t();
//     }
//   };


//   template<int N>
//   struct LeafFunctor<GammaType<N>, FlattenTag>
//   {
//     //typedef Reference<T> Type_t;
//     typedef int Type_t;
//     inline static Type_t apply(const GammaType<N> &a, const FlattenTag &f)
//     {
//       if (f.iadr >= f.maxleaf) {
// 	printf("LeafFunctor<OLattice<T>, FlattenTag>::apply too many leafs\n");
// 	exit(1);
//       }
//       f.adr[f.iadr]=0;
//       f.size[f.iadr]=0;
//       f.leaftype[ f.iadr ] = 2;
//       f.custom[ f.iadr ] = a.elem();
//       f.iadr++;
// #if defined(SPU_DEBUG)
//       printf("LeafFunctor<GammaType<N>,FlattenTag>::apply size = %d %d\n",sizeof(GammaType<N>),sizeof(a) );
// #endif
//       //return Type_t();
//     }
//   };












// Use separate defs here. This will cause subroutine calls under g++

//-----------------------------------------------------------------------------
// Layout stuff specific to a scalar architecture
namespace Layout
{
  //! coord[mu]  <- mu  : fill with lattice coord in mu direction
  LatticeInteger latticeCoordinate(int mu);
}


//-----------------------------------------------------------------------------
// Internal ops designed to look like those in parscalar
// These dummy routines exist just to make code more portable
namespace Internal
{
  //! Dummy array sum accross all nodes
  template<class T>
  inline void globalSumArray(T* dest, int n) {}

  //! Dummy global sum on a multi1d
  template<class T>
  inline void globalSumArray(multi1d<T>& dest) {}

  //! Dummy global sum on a multi2d
  template<class T>
  inline void globalSumArray(multi2d<T>& dest) {}

  //! Dummy sum across all nodes
  template<class T>
  inline void globalSum(T& dest) {}

  //! Dummy broadcast from primary node to all other nodes
  template<class T>
  inline void broadcast(T& dest) {}

  //! Dummy broadcast a string from primary node to all other nodes
  //inline void broadcast_str(std::string& dest) {}

  //! Dummy broadcast from primary node to all other nodes
  inline void broadcast(void* dest, size_t nbytes) {}
}

/////////////////////////////////////////////////////////
// Threading evaluate with openmp and qmt implementation
//
// by Xu Guo, EPCC, 16 June 2008
/////////////////////////////////////////////////////////

//! user argument for the evaluate function:
// "OLattice Op Scalar(Expression(source)) under an Subset"
//
template<class T, class T1, class Op, class RHS>
struct u_arg{
        OLattice<T>& d;
        const QDPExpr<RHS,OScalar<T1> >& r;
        const Op& op;
        const int *tab;
  u_arg( OLattice<T>& d_,
	 const QDPExpr<RHS, OScalar<T1> >& r_,
	 const Op& op_,
	 const int *tab_ ) : d(d_), r(r_), op(op_), tab(tab_) {}
   };

//! user function for the evaluate function:
// "OLattice Op Scalar(Expression(source)) under an Subset"
//
template<class T, class T1, class Op, class RHS>
void ev_userfunc(int lo, int hi, int myId, u_arg<T,T1,Op,RHS> *a)
{
   OLattice<T>& dest = a->d;
   const QDPExpr<RHS,OScalar<T1> >&rhs = a->r;
   const int* tab = a->tab;
   const Op& op= a->op;

      
   for(int j=lo; j < hi; ++j)
   {
     int i = tab[j];
     op(dest.elem(i), forEach(rhs, EvalLeaf1(0), OpCombine()));
   }
}


//! user argument for the evaluate function:
// "OLattice Op OLattice(Expression(source)) under an Subset"
//
template<class T, class T1, class Op, class RHS>
struct user_arg{
        OLattice<T>& d;
        const QDPExpr<RHS,OLattice<T1> >& r;
        const Op& op;
        const int *tab;
  user_arg(OLattice<T>& d_,
	   const QDPExpr<RHS,OLattice<T1> >& r_,
	   const Op& op_,
	   const int *tab_) : d(d_), r(r_), op(op_), tab(tab_) {}

   };

//! user function for the evaluate function:
// "OLattice Op OLattice(Expression(source)) under an Subset"
//
template<class T, class T1, class Op, class RHS>
void evaluate_userfunc(int lo, int hi, int myId, user_arg<T,T1,Op,RHS> *a)
{

   OLattice<T>& dest = a->d;
   const QDPExpr<RHS,OLattice<T1> >&rhs = a->r;
   const int* tab = a->tab;
   const Op& op= a->op;

      
   for(int j=lo; j < hi; ++j)
   {
     int i = tab[j];
     op(dest.elem(i), forEach(rhs, EvalLeaf1(i), OpCombine()));
   }
}

//! include the header file for dispatch
//#include "cudp_dispatch.h"

//-----------------------------------------------------------------------------
//! OLattice Op Scalar(Expression(source)) under an Subset
/*! 
 * OLattice Op Expression, where Op is some kind of binary operation 
 * involving the destination 
 */
template<class T, class T1, class Op, class RHS>
//inline
void evaluate(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OScalar<T1> >& rhs,
	      const Subset& s)
{
//  //cerr << "In evaluateUnorderedSubet(olattice,oscalar)\n";

// #if defined(QDP_USE_PROFILING)   
//   #warning "SCALAR: Using PROPFILING"
//   static QDPProfile_t prof(dest, op, rhs);
//   prof.time -= getClockTime();
// #endif

//   int numSiteTable = s.numSiteTable();
  
//   u_arg<T,T1,Op,RHS> a(dest, rhs, op, s.siteTable().slice());

//   dispatch_to_threads< u_arg<T,T1,Op,RHS> >(numSiteTable, a, ev_userfunc);
 
//   ///////////////////
//   // Original code
//   //////////////////
//   //const int *tab = s.siteTable().slice();
//   //for(int j=0; j < s.numSiteTable(); ++j) 
//   //{
//   //int i = tab[j];
// //    fprintf(stderr,"eval(olattice,oscalar): site %d\n",i);
// //    op(dest.elem(i), forEach(rhs, ElemLeaf(), OpCombine()));
//   //op(dest.elem(i), forEach(rhs, EvalLeaf1(0), OpCombine()));
//   //}

// #if defined(QDP_USE_PROFILING)   
//   prof.time += getClockTime();
//   prof.count++;
//   prof.print();
// #endif
}



  // template <unsigned I,unsigned elements,unsigned i>
  // struct repeat {
  //   template <typename Op , typename OLa , typename Rhs >
  //   repeat(const Op& op,OLa& dest,const Rhs& rhs,unsigned icalc)
  //   {
  //     op( dest.elem( icalc * elements + i ), forEach(rhs, EvalLeaf1(  icalc * elements + i ), OpCombine() ) );
  //     repeat<I-1,elements,i+1>(op,dest,rhs,icalc); 
  //   }
  // };
  // template <unsigned elements,unsigned i>
  // struct repeat<0,elements,i> {
  //   template <typename Op , typename OLa , typename Rhs >
  //   repeat(const Op& op,OLa& dest,const Rhs& rhs,unsigned icalc) {}
  // };







// template<typename O,typename D,typename R>
// struct foobar
// {
//   foobar( const O& op,D& dest , const R& rhs, const unsigned int& elements, int icalc_d, int icalc_rhs, int start):
//     op(op),rhs(rhs),dest(dest),elements(elements),icalc_d(icalc_d),icalc_rhs(icalc_rhs),start(start){}
//   inline void operator()(int i) {
//     op( dest.elem( icalc_d * elements + i + start ), forEach(rhs, EvalLeaf1(  icalc_rhs * elements + i + start ), OpCombine() ) );
//   }
//   const O& op;
//   const R& rhs;
//   D& dest;
//   int elements;
//   int icalc_d;
//   int icalc_rhs;
//   int start;
// };





  template< class T, class T1, class Op, class RHS >
  //inline
  __device__
  void evaluate(QDP::OLattice<T>& dest, 
		const Op& op, 
		const QDP::QDPExpr<RHS, QDP::OLattice<T1> >& rhs, 
		const QDP::Subset& s )
  {
    //dest.elem(0).elem(0,0).elem(0,0).real() = rhs.expression().child().elem(0).elem(0,0).elem(0,0).real();
    if (threadIdx.x==0)
      op(dest.elem(0), forEach(rhs, EvalLeaf1(0), OpCombine()));
    // T tmp=forEach(rhs, EvalLeaf1(0), OpCombine());
    // if (threadIdx.x==0)
    //   printf("%f\n", tmp.elem(0,0).elem(0,0).real() );
    // dest.elem(0)=tmp;//forEach(rhs, EvalLeaf1(0), OpCombine());

    // General form of loop structure
    //const int *tab = s.siteTable().slice();
    //    for(int j=0; j < s.numSiteTable(); ++j) 
      {
	//int i = tab[j];
	//fprintf(stderr,"eval(olattice,olattice): site %d\n",i);
	//op(dest.elem(j), forEach(rhs, EvalLeaf1(j), OpCombine()));
      }

  }

//   int tagid=0;

//   Op& varop = const_cast< Op& >(op);
//   bool fetchDest=varop.getReadAccessMode();
//   varop.recvInfo();

// #if defined(SPU_DEBUG)
//   printf("spu: generic evaluate entered %lu %u!\n",sizeof(T),elements);
//   printf("spu: __PRETTY_FUNCTION__\n%s\n",__PRETTY_FUNCTION__ );
//   printf("spu: op fetch dest %s\n",fetchDest?"true":"false");
// #endif


// //   const InfoOp * ptrIO = dynamic_cast<const InfoOp *>(&op);
// //   if (ptrIO)
// //     {
// // #if defined(SPU_DEBUG)
// //       printf("spu: Op is InfoOp\n");
// // #endif
// //       InfoOp * ptrIOvar = const_cast<InfoOp *>(ptrIO);
// //       ptrIOvar->recvInfo();
// //       fetchDest=ptrIOvar->getFetchDest();
// //     } 
// //   else 
// //     {
// //       const PureOp * ptrPO = dynamic_cast<const PureOp *>(&op);
// //       if (ptrPO)
// // 	fetchDest=ptrPO->getFetchDest();
// //     }

//   //printf("spu fetch dest %s\n",fetchDest?"true":"false");

//   unsigned int destHI,destLO,destLOsave;
//   destHI = spu_readch( SPU_RdInMbox );
//   destLOsave = spu_readch( SPU_RdInMbox );
// #if defined(SPU_DEBUG)
//   printf("spu: dest: got 64bit address %u %u\n",destHI,destLOsave);
// #endif

//   unsigned int adrHI[maxleaf],adrLO[maxleaf],delta[maxleaf],custom[maxleaf],leaftype[maxleaf];
//   unsigned int adrLOsave[maxleaf],size[maxleaf];
//   unsigned int leafcount = spu_readch( SPU_RdInMbox );

// #if defined(SPU_DEBUG)
//   printf("spu: leafcount = %u (max=%lu)\n",leafcount,maxleaf);
// #endif

//   for ( unsigned int i = 0 ; i < leafcount ; ++i ) {
//     adrHI[i] = spu_readch( SPU_RdInMbox );
//     adrLOsave[i] = spu_readch( SPU_RdInMbox );
//     size[i] = spu_readch( SPU_RdInMbox );
//     custom[i] = spu_readch( SPU_RdInMbox );
//     leaftype[i] = spu_readch( SPU_RdInMbox );
//     delta[i] = size[i] * elements;
// #if defined(SPU_DEBUG)
//     printf("spu: leaf: got 64bit address %u %u size=%u custom=%u leaftype=%u\n",adrHI[i],adrLOsave[i],size[i],custom[i],leaftype[i]);
// #endif
//   }



//   unsigned int hasOrder = spu_readch( SPU_RdInMbox );
//   unsigned int start,SPUsites;
//   unsigned int tabHI, tabLO;
//   unsigned int benchmark_loops=0;
//   if (hasOrder) {
//     start    = spu_readch( SPU_RdInMbox );  
//     SPUsites = spu_readch( SPU_RdInMbox );
//     benchmark_loops = spu_readch( SPU_RdInMbox );
//   } else {
//     tabHI = spu_readch( SPU_RdInMbox );
//     tabLO = spu_readch( SPU_RdInMbox );
//     start    = spu_readch( SPU_RdInMbox );  
//     SPUsites = spu_readch( SPU_RdInMbox );
//   }

//   destLOsave += sizeof(T)*start;
//   for ( unsigned int i = 0 ; i < leafcount ; ++i )
//     if (leaftype[i] == 0)
//       adrLOsave[i] += size[i] * start;

//   forEach(rhs, DMAGetOScalarTag( tagid , adrHI , adrLOsave , size , leafcount ), NullCombine());


//   // unsigned int indexStart = spu_readch( SPU_RdInMbox );
//   // unsigned int indexTotal = spu_readch( SPU_RdInMbox );

//   //printf("hasOrder=%u\n",hasOrder);
  
//   if( hasOrder && (SPUsites % elements == 0) ) {
//     unsigned int loopcount = SPUsites / elements;   // divide by number of sites to do in one loop (now 16, since dest<T> sizeof(T) == 1 is ok)
// #if defined(SPU_DEBUG)
//     printf("spu: hasOrder and indexTotal mod NUMBER ==> using aligned access.\n");
//     printf("SPUsites = %u\n",SPUsites);
//     printf("loopcount = %u\n",loopcount);
//     printf("elements = %u\n",elements);
// #endif



//     //printf("spu: from %u total %u\n",indexStart,indexTotal);
//     //printf("spu: now %u\n",indexTotal);

//     // ------
//     unsigned int DECR=0;
//     // ------

//     size_t totalbytes = sizeof( T ) * elements;

//     for ( unsigned int iii = 0 ; iii < benchmark_loops ; iii++ ) {

//       for ( unsigned int i = 0 ; i < leafcount ; ++i ) {
// 	adrLO[i] = adrLOsave[i];
//       }
//       destLO = destLOsave;
      
//       if (iii==1) {
// 	DECR = 1 << 29;
// 	spu_writech( SPU_WrDec , DECR );
// 	spu_writech( SPU_WrEventMask , MFC_DECREMENTER_EVENT );
// 	DECR = spu_readch( SPU_RdDec );
//       }

//       size_t adr;
//       unsigned int icalc_d=0;
//       unsigned int icalc_rhs=0;

//       DMAGetTag dmatag( elements , tagid , adrHI , adrLO , delta , custom , leafcount , elements );

//       for (unsigned int loop = 0 ; loop < loopcount+2 ; ++loop ) {

// #if defined(SPU_DEBUG)
// 	printf("spu: loop = %u\n",loop);
// #endif

// 	unsigned int ifetch_d   = (icalc_d+1)%3; 
// 	unsigned int istore_d   = (icalc_d-1+3)%3;


// 	if ( loop < loopcount ) {
// 	  dmatag.resetILeaf();
// 	  forEach(rhs, dmatag , NullCombine());
// 	  dmatag.switchBuffer();

// 	  if (fetchDest) {
// 	    adr = (size_t)(&dest.elem(ifetch_d*elements));

// 	    //assert( adr%16==0 && destLO%16==0 && totalbytes%16==0 );

// 	    // if ( totalbytes <= 16*1024 )
// 	    //   spu_mfcdma64((void *)(adr), destHI , destLO , totalbytes, tagid , MFC_GET_CMD );
// 	    // else
// 	    get_large_region( (void *)(adr), destHI , destLO , totalbytes , tagid, MFC_GETL_CMD );
// 	  }
// 	}



// 	if (loop >= 2) {
// 	  adr = (size_t)(&dest.elem(istore_d*elements));

// 	  // printf("put adr=%lu(%lu) %u(%u) %u(%u)\n",
// 	  // 	 (size_t)(adr),(size_t)(adr)%128,
// 	  // 	 destLO - totalbytes - totalbytes , (destLO - totalbytes - totalbytes)%128,
// 	  // 	 totalbytes,(totalbytes)%128);
// 	  //assert( adr%16==0 && destLO%16==0 && totalbytes%16==0 );

// 	  // if ( totalbytes <= 16*1024 )
// 	  //   spu_mfcdma64((void *)(adr), destHI , destLO - totalbytes - totalbytes , totalbytes, tagid , MFC_PUT_CMD );
// 	  // else
// 	  get_large_region( (void *)(adr), destHI , destLO - totalbytes - totalbytes , totalbytes , tagid, MFC_PUTL_CMD );
// 	}


// 	unsigned int dest_buffer = icalc_d * elements;
// 	unsigned int rhs_buffer = icalc_rhs * elements;
// 	if ( ( loop >= 1 ) && ( loop < loopcount + 1 ) ) {
// 	  for ( unsigned int i = 0 ; i < elements ; i++ )
// 	    op( dest.elem( dest_buffer + i ), forEach(rhs, EvalLeaf1(  rhs_buffer + i ), OpCombine() ) );
// 	}

// 	// const int nmax   = 2;
// 	// typedef boost::mpl::if_< boost::mpl::less< boost::mpl::int_< elements > , boost::mpl::int_< nmax > > , boost::mpl::int_< elements > , boost::mpl::int_< nmax > > nmod;
// 	// if ( ( loop >= 1 ) && ( loop < loopcount + 1 ) ) {
// 	//   int edone = 0;
// 	//   while( edone < elements ) {
// 	//     foobar<Op,QDP::OLattice<T>,QDP::QDPExpr<RHS, QDP::OLattice<T1> > > f(op,dest,rhs,elements,icalc_d,icalc_rhs,edone);
// 	//     boost::mpl::for_each< boost::mpl::range_c<int, 0 , nmod::type::value > >( f );
// 	//     edone += nmod::type::value;
// 	//   }
// 	// }


// 	spu_writech(MFC_WrTagMask, 1 << tagid );
// 	(void)spu_mfcstat(MFC_TAG_UPDATE_ALL);
	
// 	icalc_d++;
// 	icalc_d%=3;

// 	icalc_rhs++;
// 	icalc_rhs%=2;

// 	destLO += totalbytes;
	
//       }
//     }

//     unsigned int aktdecr;
//     aktdecr = spu_readch( SPU_RdDec );
//     if (benchmark_loops==1)
//       spu_writech( SPU_WrOutMbox , 0 );
//     else
//       spu_writech( SPU_WrOutMbox , (DECR-aktdecr) / (benchmark_loops-1) );
    

//     spu_writech( SPU_WrOutMbox , 123 );

//   } else {
// #if defined(SPU_DEBUG)
//     printf("spu: SubSet not linear.\n");
//     exit(1);
// #endif
//   }








// //! OLattice Op OLattice(Expression(source)) under an Subset
// /*! 
//  * OLattice Op Expression, where Op is some kind of binary operation 
//  * involving the destination 
//  */
// template<class T, class T1, class Op, class RHS>
// //inline
// void evaluate(OLattice<T>& dest, const Op& op, const QDPExpr<RHS,OLattice<T1> >& rhs,
// 	      const Subset& s)
// {
// //  //cerr << "In evaluateSubset(olattice,olattice)" << endl;

// #if defined(QDP_USE_PROFILING)   
//   #warning "LATTICE: Using PROPFILING"
//   static QDPProfile_t prof(dest, op, rhs);
//   prof.time -= getClockTime();
// #endif

//   int numSiteTable = s.numSiteTable();

//   user_arg<T,T1,Op,RHS> a(dest, rhs, op, s.siteTable().slice());

//   dispatch_to_threads<user_arg<T,T1,Op,RHS> >(numSiteTable, a, evaluate_userfunc);

//   ////////////////////
//   // Original code
//   ///////////////////

//   // General form of loop structure
//   //const int *tab = s.siteTable().slice();
//   //for(int j=0; j < s.numSiteTable(); ++j) 
//   //{
//   //int i = tab[j];
// //    fprintf(stderr,"eval(olattice,olattice): site %d\n",i);
//   //op(dest.elem(i), forEach(rhs, EvalLeaf1(i), OpCombine()));
//   //}

// #if defined(QDP_USE_PROFILING)   
//   prof.time += getClockTime();
//   prof.count++;
//   prof.print();
// #endif
// }



//-----------------------------------------------------------------------------
//! dest = (mask) ? s1 : dest
// template<class T1, class T2> 
// void 
// copymask(OSubLattice<T2,Subset> d, const OLattice<T1>& mask, const OLattice<T2>& s1) 
// {
//   OLattice<T2>& dest = d.field();
//   const Subset& s = d.subset();

//   const int *tab = s.siteTable().slice();
//   for(int j=0; j < s.numSiteTable(); ++j) 
//   {
//     int i = tab[j];
//     copymask(dest.elem(i), mask.elem(i), s1.elem(i));
//   }
// }


// //! dest = (mask) ? s1 : dest
// template<class T1, class T2> 
// void 
// copymask(OLattice<T2>& dest, const OLattice<T1>& mask, const OLattice<T2>& s1) 
// {
//   const int vvol = Layout::vol();
//   for(int i=0; i < vvol; ++i) 
//     copymask(dest.elem(i), mask.elem(i), s1.elem(i));
// }



//-----------------------------------------------------------------------------
// Random numbers
namespace RNG
{
  extern Seed ran_seed;
  extern Seed ran_mult;
  extern Seed ran_mult_n;
  extern LatticeSeed *lattice_ran_mult;
}


//! dest  = random  
/*! This implementation is correct for no inner grid */
template<class T>
__device__
void 
random(OScalar<T>& d)
{
  Seed seed = RNG::ran_seed;
  Seed skewed_seed = RNG::ran_seed * RNG::ran_mult;

  fill_random(d.elem(), seed, skewed_seed, RNG::ran_mult);

  RNG::ran_seed = seed;  // The seed from any site is the same as the new global seed
}


//! dest  = random    under a subset
template<class T>
__device__
void 
random(OLattice<T>& d, const Subset& s)
{
  Seed seed;
  Seed skewed_seed;

  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    seed = RNG::ran_seed;
    skewed_seed.elem() = RNG::ran_seed.elem() * RNG::lattice_ran_mult->elem(i);
    fill_random(d.elem(i), seed, skewed_seed, RNG::ran_mult_n);
  }

  RNG::ran_seed = seed;  // The seed from any site is the same as the new global seed
}




//! dest  = random   under a subset
template<class T, class S>
__device__
void random(const OSubLattice<T,S>& dd)
{
  OLattice<T>& d = const_cast<OSubLattice<T,S>&>(dd).field();
  const S& s = dd.subset();

  random(d,s);
}


//! dest  = random  
template<class T>
__device__
void random(OLattice<T>& d)
{
  random(d,all);
}


//! dest  = gaussian   under a subset
template<class T>
__device__
void gaussian(OLattice<T>& d, const Subset& s)
{
  OLattice<T>  r1, r2;

  random(r1,s);
  random(r2,s);

  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    fill_gaussian(d.elem(i), r1.elem(i), r2.elem(i));
  }
}



//! dest  = gaussian   under a subset
template<class T, class S>
__device__
void gaussian(const OSubLattice<T,S>& dd)
{
  OLattice<T>& d = const_cast<OSubLattice<T,S>&>(dd).field();
  const S& s = dd.subset();

  gaussian(d,s);
}


//! dest  = gaussian
template<class T>
__device__
void gaussian(OLattice<T>& d)
{
  gaussian(d,all);
}



//-----------------------------------------------------------------------------
// Broadcast operations
//! dest  = 0 
template<class T> 
__device__
void zero_rep(OLattice<T>& dest, const Subset& s) 
{
  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    zero_rep(dest.elem(i));
  }
}



//! dest  = 0 
template<class T, class S>
__device__
void zero_rep(OSubLattice<T,S> dd) 
{
  OLattice<T>& d = dd.field();
  const S& s = dd.subset();
  
  zero_rep(d,s);
}


//! dest  = 0 
template<class T> 
__device__
void zero_rep(OLattice<T>& dest) 
{
  const int vvol = Layout::vol();
  for(int i=0; i < vvol; ++i) 
    zero_rep(dest.elem(i));
}



//-----------------------------------------------
// Global sums
//! OScalar = sum(OScalar) under an explicit subset
/*!
 * Allow a global sum that sums over the lattice, but returns an object
 * of the same primitive type. E.g., contract only over lattice indices
 */
template<class RHS, class T>
__device__
typename UnaryReturn<OScalar<T>, FnSum>::Type_t
sum(const QDPExpr<RHS,OScalar<T> >& s1, const Subset& s)
{
  typename UnaryReturn<OScalar<T>, FnSum>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  evaluate(d,OpAssign(),s1,all);   // since OScalar, no global sum needed

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}


//! OScalar = sum(OScalar)
/*!
 * Allow a global sum that sums over the lattice, but returns an object
 * of the same primitive type. E.g., contract only over lattice indices
 */
template<class RHS, class T>
__device__
typename UnaryReturn<OScalar<T>, FnSum>::Type_t
sum(const QDPExpr<RHS,OScalar<T> >& s1)
{
  typename UnaryReturn<OScalar<T>, FnSum>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  evaluate(d,OpAssign(),s1,all);

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}



//! OScalar = sum(OLattice)  under an explicit subset
/*!
 * Allow a global sum that sums over the lattice, but returns an object
 * of the same primitive type. E.g., contract only over lattice indices
 */
template<class RHS, class T>
__device__
typename UnaryReturn<OLattice<T>, FnSum>::Type_t
sum(const QDPExpr<RHS,OLattice<T> >& s1, const Subset& s)
{
  typename UnaryReturn<OLattice<T>, FnSum>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  // Must initialize to zero since we do not know if the loop will be entered
  zero_rep(d.elem());

  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    d.elem() += forEach(s1, EvalLeaf1(i), OpCombine());   // SINGLE NODE VERSION FOR NOW
  }

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}


//! OScalar = sum(OLattice)
/*!
 * Allow a global sum that sums over the lattice, but returns an object
 * of the same primitive type. E.g., contract only over lattice indices
 */
template<class RHS, class T>
__device__
typename UnaryReturn<OLattice<T>, FnSum>::Type_t
sum(const QDPExpr<RHS,OLattice<T> >& s1)
{
  typename UnaryReturn<OLattice<T>, FnSum>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  // Loop always entered - could unroll
  zero_rep(d.elem());

  const int vvol = Layout::vol();
  for(int i=0; i < vvol; ++i) 
    d.elem() += forEach(s1, EvalLeaf1(i), OpCombine());   // SINGLE NODE VERSION FOR NOW

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}


//-----------------------------------------------------------------------------
// Multiple global sums 
//! multi1d<OScalar> dest  = sumMulti(OScalar,Set) 
/*!
 * Compute the global sum on multiple subsets specified by Set 
 *
 * This implementation is specific to a purely olattice like
 * types. The scalar input value is replicated to all the
 * slices
 */
template<class RHS, class T>
__device__
typename UnaryReturn<OScalar<T>, FnSum>::Type_t
sumMulti(const QDPExpr<RHS,OScalar<T> >& s1, const Set& ss)
{
  typename UnaryReturn<OScalar<T>, FnSumMulti>::Type_t  dest(ss.numSubsets());

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(dest[0], OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  // lazy - evaluate repeatedly
  for(int i=0; i < ss.numSubsets(); ++i)
    evaluate(dest[i],OpAssign(),s1,all);

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return dest;
}


//! multi1d<OScalar> dest  = sumMulti(OLattice,Set) 
/*!
 * Compute the global sum on multiple subsets specified by Set 
 *
 * This is a very simple implementation. There is no need for
 * anything fancier unless global sums are just so extraordinarily
 * slow. Otherwise, generalized sums happen so infrequently the slow
 * version is fine.
 */
template<class RHS, class T>
__device__
typename UnaryReturn<OLattice<T>, FnSumMulti>::Type_t
sumMulti(const QDPExpr<RHS,OLattice<T> >& s1, const Set& ss)
{
  typename UnaryReturn<OLattice<T>, FnSumMulti>::Type_t  dest(ss.numSubsets());

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(dest[0], OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  // Initialize result with zero
  for(int k=0; k < ss.numSubsets(); ++k)
    zero_rep(dest[k]);

  // Loop over all sites and accumulate based on the coloring 
  const multi1d<int>& lat_color =  ss.latticeColoring();

  const int vvol = Layout::vol();
  for(int i=0; i < vvol; ++i) 
  {
    int j = lat_color[i];
    dest[j].elem() += forEach(s1, EvalLeaf1(i), OpCombine());   // SINGLE NODE VERSION FOR NOW
  }

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return dest;
}


//-----------------------------------------------------------------------------
// Multiple global sums 
//! multi2d<OScalar> dest  = sumMulti(multi1d<OScalar>,Set) 
/*!
 * Compute the global sum on multiple subsets specified by Set 
 *
 * This implementation is specific to a purely olattice like
 * types. The scalar input value is replicated to all the
 * slices
 */
template<class T>
__device__
multi2d<typename UnaryReturn<OScalar<T>, FnSum>::Type_t>
sumMulti(const multi1d< OScalar<T> >& s1, const Set& ss)
{
  multi2d<typename UnaryReturn<OScalar<T>, FnSum>::Type_t>  dest(s1.size(),ss.numSubsets());

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(dest(0,0), OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  // lazy - evaluate repeatedly
  for(int i=0; i < dest.size1(); ++i)
    for(int j=0; j < dest.size2(); ++j)
      dest(j,i) = s1[j];

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return dest;
}


//! multi2d<OScalar> dest  = sumMulti(multi1d<OLattice>,Set) 
/*!
 * Compute the global sum on multiple subsets specified by Set 
 *
 * This is a very simple implementation. There is no need for
 * anything fancier unless global sums are just so extraordinarily
 * slow. Otherwise, generalized sums happen so infrequently the slow
 * version is fine.
 */
template<class T>
__device__
multi2d<typename UnaryReturn<OLattice<T>, FnSum>::Type_t>
sumMulti(const multi1d< OLattice<T> >& s1, const Set& ss)
{
  multi2d<typename UnaryReturn<OLattice<T>, FnSum>::Type_t>  dest(s1.size(),ss.numSubsets());

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(dest(0,0), OpAssign(), FnSum(), s1);
  prof.time -= getClockTime();
#endif

  // Initialize result with zero
  for(int i=0; i < dest.size1(); ++i)
    for(int j=0; j < dest.size2(); ++j)
      zero_rep(dest(j,i));

  // Loop over all sites and accumulate based on the coloring 
  const multi1d<int>& lat_color =  ss.latticeColoring();

  const int vvol = Layout::vol();
  for(int k=0; k < s1.size(); ++k)
  {
    const OLattice<T>& ss1 = s1[k];

    for(int i=0; i < vvol; ++i) 
    {
      int j = lat_color[i];
      dest(k,j).elem() += ss1.elem(i);
    }
  }

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return dest;
}


//-----------------------------------------------------------------------------
//! OScalar = norm2(trace(adj(multi1d<source>)*multi1d<source>))
/*!
 * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
 *
 * Sum over the lattice
 * Allow a global sum that sums over all indices
 */
template<class T>
__device__
inline typename UnaryReturn<OScalar<T>, FnNorm2>::Type_t
norm2(const multi1d< OScalar<T> >& s1)
{
  typename UnaryReturn<OScalar<T>, FnNorm2>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnNorm2(), s1[0]);
  prof.time -= getClockTime();
#endif

  // Possibly loop entered
  zero_rep(d.elem());

  for(int n=0; n < s1.size(); ++n)
  {
    OScalar<T>& ss1 = s1[n];
    d.elem() += localNorm2(ss1.elem());
  }

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}

//! OScalar = sum(OScalar)  under an explicit subset
/*! Discards subset */
template<class T>
__device__
inline typename UnaryReturn<OScalar<T>, FnNorm2>::Type_t
norm2(const multi1d< OScalar<T> >& s1, const Subset& s)
{
  return norm2(s1);
}



//! OScalar = norm2(multi1d<OLattice>) under an explicit subset
/*!
 * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
 *
 * Sum over the lattice
 * Allow a global sum that sums over all indices
 */
template<class T>
__device__
inline typename UnaryReturn<OLattice<T>, FnNorm2>::Type_t
norm2(const multi1d< OLattice<T> >& s1, const Subset& s)
{
  typename UnaryReturn<OLattice<T>, FnNorm2>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnNorm2(), s1[0]);
  prof.time -= getClockTime();
#endif

  // Possibly loop entered
  zero_rep(d.elem());

  const int *tab = s.siteTable().slice();
  for(int n=0; n < s1.size(); ++n)
  {
    const OLattice<T>& ss1 = s1[n];
    for(int j=0; j < s.numSiteTable(); ++j) 
    {
      int i = tab[j];
      d.elem() += localNorm2(ss1.elem(i));
    }
  }

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}



//! OScalar = norm2(multi1d<OLattice>)
/*!
 * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
 *
 * Sum over the lattice
 * Allow a global sum that sums over all indices
 */
template<class T>
__device__
inline typename UnaryReturn<OLattice<T>, FnNorm2>::Type_t
norm2(const multi1d< OLattice<T> >& s1)
{
  return norm2(s1,all);
}



//-----------------------------------------------------------------------------
//! OScalar = innerProduct(multi1d<source1>,multi1d<source2>))
/*!
 * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
 *
 * Sum over the lattice
 * Allow a global sum that sums over all indices
 */
template<class T1, class T2>
__device__
inline typename BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProduct>::Type_t
innerProduct(const multi1d< OScalar<T1> >& s1, const multi1d< OScalar<T2> >& s2)
{
  typename BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProduct>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnInnerProduct(), s1[0]);
  prof.time -= getClockTime();
#endif

  // Possibly loop entered
  zero_rep(d.elem());

  for(int n=0; n < s1.size(); ++n)
  {
    OScalar<T1>& ss1 = s1[n];
    OScalar<T2>& ss2 = s2[n];
    d.elem() += localInnerProduct(ss1.elem(),ss2.elem());
  }

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}

//! OScalar = sum(OScalar)  under an explicit subset
/*! Discards subset */
template<class T1, class T2>
__device__
inline typename BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProduct>::Type_t
innerProduct(const multi1d< OScalar<T1> >& s1, const multi1d< OScalar<T2> >& s2,
	     const Subset& s)
{
  return innerProduct(s1,s2);
}


//! OScalar = innerProduct(multi1d<OLattice>,multi1d<OLattice>) under an explicit subset
/*!
 * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
 *
 * Sum over the lattice
 * Allow a global sum that sums over all indices
 */
template<class T1, class T2>
__device__
inline typename BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProduct>::Type_t
innerProduct(const multi1d< OLattice<T1> >& s1, const multi1d< OLattice<T2> >& s2,
	     const Subset& s)
{
  typename BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProduct>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnInnerProduct(), s1[0]);
  prof.time -= getClockTime();
#endif

  // Possibly loop entered
  zero_rep(d.elem());

  const int *tab = s.siteTable().slice();
  for(int n=0; n < s1.size(); ++n)
  {
    const OLattice<T1>& ss1 = s1[n];
    const OLattice<T2>& ss2 = s2[n];
    for(int j=0; j < s.numSiteTable(); ++j) 
    {
      int i = tab[j];
      d.elem() += localInnerProduct(ss1.elem(i),ss2.elem(i));
    }
  }

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}


//! OScalar = innerProduct(multi1d<OLattice>,multi1d<OLattice>)
/*!
 * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
 *
 * Sum over the lattice
 * Allow a global sum that sums over all indices
 */
template<class T1, class T2>
__device__
inline typename BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProduct>::Type_t
innerProduct(const multi1d< OLattice<T1> >& s1, const multi1d< OLattice<T2> >& s2)
{
  return innerProduct(s1,s2,all);
}



//-----------------------------------------------------------------------------
//! OScalar = innerProductReal(multi1d<source1>,multi1d<source2>))
/*!
 * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
 *
 * Sum over the lattice
 * Allow a global sum that sums over all indices
 */
template<class T1, class T2>
__device__
inline typename BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProductReal>::Type_t
innerProductReal(const multi1d< OScalar<T1> >& s1, const multi1d< OScalar<T2> >& s2)
{
  typename BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProductReal>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnInnerProductReal(), s1[0]);
  prof.time -= getClockTime();
#endif

  // Possibly loop entered
  zero_rep(d.elem());

  for(int n=0; n < s1.size(); ++n)
  {
    OScalar<T1>& ss1 = s1[n];
    OScalar<T2>& ss2 = s2[n];
    d.elem() += localInnerProductReal(ss1.elem(),ss2.elem());
  }

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}

//! OScalar = sum(OScalar)  under an explicit subset
/*! Discards subset */
template<class T1, class T2>
__device__
inline typename BinaryReturn<OScalar<T1>, OScalar<T2>, FnInnerProductReal>::Type_t
innerProductReal(const multi1d< OScalar<T1> >& s1, const multi1d< OScalar<T2> >& s2,
		 const Subset& s)
{
  return innerProductReal(s1,s2);
}



//! OScalar = innerProductReal(multi1d<OLattice>,multi1d<OLattice>) under an explicit subset
/*!
 * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
 *
 * Sum over the lattice
 * Allow a global sum that sums over all indices
 */
template<class T1, class T2>
__device__
inline typename BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProductReal>::Type_t
innerProductReal(const multi1d< OLattice<T1> >& s1, const multi1d< OLattice<T2> >& s2,
		 const Subset& s)
{
  typename BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProductReal>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnInnerProductReal(), s1[0]);
  prof.time -= getClockTime();
#endif

  // Possibly loop entered
  zero_rep(d.elem());

  const int *tab = s.siteTable().slice();
  for(int n=0; n < s1.size(); ++n)
  {
    const OLattice<T1>& ss1 = s1[n];
    const OLattice<T2>& ss2 = s2[n];
    for(int j=0; j < s.numSiteTable(); ++j) 
    {
      int i = tab[j];
      d.elem() += localInnerProductReal(ss1.elem(i),ss2.elem(i));
    }
  }

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}


//! OScalar = innerProductReal(multi1d<OLattice>,multi1d<OLattice>)
/*!
 * return  \sum_{multi1d} \sum_x(trace(adj(multi1d<source>)*multi1d<source>))
 *
 * Sum over the lattice
 * Allow a global sum that sums over all indices
 */
template<class T1, class T2>
__device__
inline typename BinaryReturn<OLattice<T1>, OLattice<T2>, FnInnerProductReal>::Type_t
innerProductReal(const multi1d< OLattice<T1> >& s1, const multi1d< OLattice<T2> >& s2)
{
  return innerProductReal(s1,s2,all);
}


//-----------------------------------------------
// Global max and min
// NOTE: there are no subset version of these operations. It is very problematic
// and QMP does not support them.
//! OScalar = globalMax(OScalar)
/*!
 * Find the maximum an object has across the lattice
 */
template<class RHS, class T>
__device__
typename UnaryReturn<OScalar<T>, FnGlobalMax>::Type_t
globalMax(const QDPExpr<RHS,OScalar<T> >& s1)
{
  typename UnaryReturn<OScalar<T>, FnGlobalMax>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnGlobalMax(), s1);
  prof.time -= getClockTime();
#endif

  evaluate(d,OpAssign(),s1,all);   // since OScalar, no global max needed

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}



//! OScalar = globalMax(OLattice)
/*!
 * Find the maximum an object has across the lattice
 */
template<class RHS, class T>
__device__
typename UnaryReturn<OLattice<T>, FnGlobalMax>::Type_t
globalMax(const QDPExpr<RHS,OLattice<T> >& s1)
{
  typename UnaryReturn<OLattice<T>, FnGlobalMax>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnGlobalMax(), s1);
  prof.time -= getClockTime();
#endif

  // Loop always entered so unroll
  d.elem() = forEach(s1, EvalLeaf1(0), OpCombine());   // SINGLE NODE VERSION FOR NOW

  const int vvol = Layout::vol();
  for(int i=1; i < vvol; ++i) 
  {
    typename UnaryReturn<T, FnGlobalMax>::Type_t  dd = 
      forEach(s1, EvalLeaf1(i), OpCombine());   // SINGLE NODE VERSION FOR NOW

    if (toBool(dd > d.elem()))
      d.elem() = dd;
  }

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}


//! OScalar = globalMin(OScalar)
/*!
 * Find the minimum an object has across the lattice
 */
template<class RHS, class T>
__device__
typename UnaryReturn<OScalar<T>, FnGlobalMin>::Type_t
globalMin(const QDPExpr<RHS,OScalar<T> >& s1)
{
  typename UnaryReturn<OScalar<T>, FnGlobalMin>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnGlobalMin(), s1);
  prof.time -= getClockTime();
#endif

  evaluate(d,OpAssign(),s1,all);   // since OScalar, no global min needed

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}



//! OScalar = globalMin(OLattice)
/*!
 * Find the minimum of an object under a subset of the lattice
 */
template<class RHS, class T>
__device__
typename UnaryReturn<OLattice<T>, FnGlobalMin>::Type_t
globalMin(const QDPExpr<RHS,OLattice<T> >& s1)
{
  typename UnaryReturn<OLattice<T>, FnGlobalMin>::Type_t  d;

#if defined(QDP_USE_PROFILING)   
  static QDPProfile_t prof(d, OpAssign(), FnGlobalMin(), s1);
  prof.time -= getClockTime();
#endif

  // Loop always entered so unroll
  d.elem() = forEach(s1, EvalLeaf1(0), OpCombine());   // SINGLE NODE VERSION FOR NOW

  const int vvol = Layout::vol();
  for(int i=1; i < vvol; ++i) 
  {
    typename UnaryReturn<T, FnGlobalMin>::Type_t  dd = 
      forEach(s1, EvalLeaf1(i), OpCombine());   // SINGLE NODE VERSION FOR NOW

    if (toBool(dd < d.elem()))
      d.elem() = dd;
  }

#if defined(QDP_USE_PROFILING)   
  prof.time += getClockTime();
  prof.count++;
  prof.print();
#endif

  return d;
}


//-----------------------------------------------------------------------------
// Peek and poke at individual sites. This is very architecture specific
// NOTE: these two routines assume there is no underlying inner grid

//! Extract site element
/*! @ingroup group1
  @param l  source to examine
  @param coord Nd lattice coordinates to examine
  @return single site object of the same primitive type
  @ingroup group1
  @relates QDPType */
template<class T1>
__device__
inline OScalar<T1>
peekSite(const OScalar<T1>& l, const multi1d<int>& coord)
{
  return l;
}

//! Extract site element
/*! @ingroup group1
  @param l  source to examine
  @param coord Nd lattice coordinates to examine
  @return single site object of the same primitive type
  @ingroup group1
  @relates QDPType */
template<class RHS, class T1>
__device__
inline OScalar<T1>
peekSite(const QDPExpr<RHS,OScalar<T1> > & l, const multi1d<int>& coord)
{
  // For now, simply evaluate the expression and then call the function
  typedef OScalar<T1> C1;
  
  return peekSite(C1(l), coord);
}


//! Extract site element
/*! @ingroup group1
  @param l  source to examine
  @param coord Nd lattice coordinates to examine
  @return single site object of the same primitive type
  @ingroup group1
  @relates QDPType */
template<class T1>
__device__
inline OScalar<T1>
peekSite(const OLattice<T1>& l, const multi1d<int>& coord)
{
  OScalar<T1> dest;

  dest.elem() = l.elem(Layout::linearSiteIndex(coord));
  return dest;
}

//! Extract site element
/*! @ingroup group1
  @param l  source to examine
  @param coord Nd lattice coordinates to examine
  @return single site object of the same primitive type
  @ingroup group1
  @relates QDPType */
template<class RHS, class T1>
__device__
inline OScalar<T1>
peekSite(const QDPExpr<RHS,OLattice<T1> > & l, const multi1d<int>& coord)
{
  // For now, simply evaluate the expression and then call the function
  typedef OLattice<T1> C1;
  
  return peekSite(C1(l), coord);
}


//! Insert site element
/*! @ingroup group1
  @param l  target to update
  @param r  source to insert
  @param coord Nd lattice coordinates where to insert
  @return object of the same primitive type but of promoted lattice type
  @ingroup group1
  @relates QDPType */
template<class T1>
__device__
inline OLattice<T1>&
pokeSite(OLattice<T1>& l, const OScalar<T1>& r, const multi1d<int>& coord)
{
  l.elem(Layout::linearSiteIndex(coord)) = r.elem();
  return l;
}


//! Copy data values from field src to array dest
/*! @ingroup group1
  @param dest  target to update
  @param src   QDP source to insert
  @param s     subset
  @ingroup group1
  @relates QDPType */
template<class T>
__device__
inline void 
QDP_extract(multi1d<OScalar<T> >& dest, const OLattice<T>& src, const Subset& s)
{
  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    dest[i].elem() = src.elem(i);
  }
}

//! Inserts data values from site array src.
/*! @ingroup group1
  @param dest  QDP target to update
  @param src   source to insert
  @param s     subset
  @ingroup group1
  @relates QDPType */
template<class T>
__device__
inline void 
QDP_insert(OLattice<T>& dest, const multi1d<OScalar<T> >& src, const Subset& s)
{
  const int *tab = s.siteTable().slice();
  for(int j=0; j < s.numSiteTable(); ++j) 
  {
    int i = tab[j];
    dest.elem(i) = src[i].elem();
  }
}


//-----------------------------------------------------------------------------
// Forward declaration
//struct FnMap;
// This is the PETE version of a map, namely return an expression
struct FnMap
{
  PETE_EMPTY_CONSTRUCTORS(FnMap)

  const int *goff;
__device__
  FnMap(const int *goffsets): goff(goffsets)
    {
//    fprintf(stderr,"FnMap(): goff=0x%x\n",goff);
    }
  
  template<class T>
__device__
  inline typename UnaryReturn<T, FnMap>::Type_t
  operator()(const T &a) const
  {
    return (a);
  }
};




//! General permutation map class for communications
class Map
{
public:
  //! Constructor - does nothing really
__device__
  Map() {}

  //! Destructor
__device__
  ~Map() {}

  //! Constructor from a function object
__device__
  Map(const MapFunc& fn) {make(fn);}

  //! Actual constructor from a function object
  /*! The semantics are   source_site = func(dest_site) */
__device__
  void make(const MapFunc& func);

  //! Function call operator for a shift
  /*! 
   * map(source)
   *
   * Implements:  dest(x) = s1(x+offsets)
   *
   * Shifts on a OLattice are non-trivial.
   * Notice, there may be an ILattice underneath which requires shift args.
   * This routine is very architecture dependent.
   */
  template<class T1,class C1>
  __device__
  inline typename MakeReturn<UnaryNode<FnMap,
				       typename CreateLeaf<QDPType<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPType<T1,C1> & l)
  {
    typedef UnaryNode<FnMap,
      typename CreateLeaf<QDPType<T1,C1> >::Leaf_t> Tree_t;
    return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(goffsets.slice()),
					      CreateLeaf<QDPType<T1,C1> >::make(l)));
  }


  template<class T1,class C1>
__device__
  inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPExpr<T1,C1> & l)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(goffsets.slice()),
	CreateLeaf<QDPExpr<T1,C1> >::make(l)));
    }


public:
  //! Accessor to offsets
__device__
  const multi1d<int>& Offsets() const {return goffsets;}

private:
  //! Hide copy constructor
__device__
  Map(const Map&) {}

  //! Hide operator=
__device__
  void operator=(const Map&) {}

private:
  //! Offset table used for communications. 
  multi1d<int> goffsets;
};





#if defined(QDP_USE_PROFILING)   
template <>
struct TagVisitor<FnMap, PrintTag> : public ParenPrinter<FnMap>
{ 
__device__
  static void visit(FnMap op, PrintTag t) 
    { t.os_m << "shift"; }
};
#endif


// Specialization of ForEach deals with maps. 
template<class A, class CTag>
struct ForEach<UnaryNode<FnMap, A>, EvalLeaf1, CTag>
{
  typedef typename ForEach<A, EvalLeaf1, CTag>::Type_t TypeA_t;
  typedef typename Combine1<TypeA_t, FnMap, CTag>::Type_t Type_t;
  inline static
__device__
  Type_t apply(const UnaryNode<FnMap, A> &expr, const EvalLeaf1 &f, 
    const CTag &c) 
  {
    EvalLeaf1 ff(expr.operation().goff[f.val1()]);
//  fprintf(stderr,"ForEach<Unary<FnMap>>: site = %d, new = %d\n",f.val1(),ff.val1());

    return Combine1<TypeA_t, FnMap, CTag>::
      combine(ForEach<A, EvalLeaf1, CTag>::apply(expr.child(), ff, c),
              expr.operation(), c);
  }
};


//-----------------------------------------------------------------------------
//! Array of general permutation map class for communications
class ArrayMap
{
public:
  //! Constructor - does nothing really
__device__
  ArrayMap() {}

  //! Destructor
__device__
  ~ArrayMap() {}

  //! Constructor from a function object
__device__
  ArrayMap(const ArrayMapFunc& fn) {make(fn);}

  //! Actual constructor from a function object
  /*! The semantics are   source_site = func(dest_site,sign) */
__device__
  void make(const ArrayMapFunc& func);

  //! Function call operator for a shift
  /*! 
   * map(source,dir)
   *
   * Implements:  dest(x) = source(map(x,dir))
   *
   * Shifts on a OLattice are non-trivial.
   * Notice, there may be an ILattice underneath which requires shift args.
   * This routine is very architecture dependent.
   */
  template<class T1,class C1>
__device__
  inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPType<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPType<T1,C1> & l, int dir)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPType<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(mapsa[dir].Offsets().slice()),
	CreateLeaf<QDPType<T1,C1> >::make(l)));
    }


  template<class T1,class C1>
__device__
  inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPExpr<T1,C1> & l, int dir)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(mapsa[dir].Offsets().slice()),
	CreateLeaf<QDPExpr<T1,C1> >::make(l)));
    }


private:
  //! Hide copy constructor
__device__
  ArrayMap(const ArrayMap&) {}

  //! Hide operator=
__device__
  void operator=(const ArrayMap&) {}

private:
  multi1d<Map> mapsa;
  
};


//-----------------------------------------------------------------------------
//! Bi-directional version of general permutation map class for communications
class BiDirectionalMap
{
public:
  //! Constructor - does nothing really
__device__
  BiDirectionalMap() {}

  //! Destructor
__device__
  ~BiDirectionalMap() {}

  //! Constructor from a function object
__device__
  BiDirectionalMap(const MapFunc& fn) {make(fn);}

  //! Actual constructor from a function object
  /*! The semantics are   source_site = func(dest_site,sign) */
__device__
  void make(const MapFunc& func);

  //! Function call operator for a shift
  /*! 
   * map(source,isign)
   *
   * Implements:  dest(x) = source(map(x,isign))
   *
   * Shifts on a OLattice are non-trivial.
   * Notice, there may be an ILattice underneath which requires shift args.
   * This routine is very architecture dependent.
   */
  template<class T1,class C1>
__device__
  inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPType<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPType<T1,C1> & l, int isign)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPType<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(bimaps[(isign+1)>>1].Offsets().slice()),
	CreateLeaf<QDPType<T1,C1> >::make(l)));
    }


  template<class T1,class C1>
__device__
  inline typename MakeReturn<UnaryNode<FnMap,
    typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
  operator()(const QDPExpr<T1,C1> & l, int isign)
    {
      typedef UnaryNode<FnMap,
	typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
      return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(bimaps[(isign+1)>>1].Offsets().slice()),
	CreateLeaf<QDPExpr<T1,C1> >::make(l)));
    }


private:
  //! Hide copy constructor
__device__
  BiDirectionalMap(const BiDirectionalMap&) {}

  //! Hide operator=
__device__
  void operator=(const BiDirectionalMap&) {}

private:
  multi1d<Map> bimaps;
  
};


//-----------------------------------------------------------------------------
//! Bi-directional version of general permutation map class for communications
class ArrayBiDirectionalMap
{
public:
  //! Constructor - does nothing really
__device__
  ArrayBiDirectionalMap() {}

  //! Destructor
__device__
  ~ArrayBiDirectionalMap() {}

  //! Constructor from a function object
__device__
  ArrayBiDirectionalMap(const ArrayMapFunc& fn) {make(fn);}

  //! Actual constructor from a function object
  /*! The semantics are   source_site = func(dest_site,sign) */
__device__
  void make(const ArrayMapFunc& func);

  //! Function call operator for a shift
  /*! 
   * map(source,isign,dir)
   *
   * Implements:  dest(x) = source(map(x,isign,dir))
   *
   * Shifts on a OLattice are non-trivial.
   * Notice, there may be an ILattice underneath which requires shift args.
   * This routine is very architecture dependent.
   */
  // template<class T1,class C1>
  // inline typename MakeReturn<UnaryNode<FnMap,
  //   typename CreateLeaf<QDPType<T1,C1> >::Leaf_t>, C1>::Expression_t
  // operator()(const QDPType<T1,C1> & l, int isign, int dir)
  //   {
  //     typedef UnaryNode<FnMap,
  // 	typename CreateLeaf<QDPType<T1,C1> >::Leaf_t> Tree_t;
  //     return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(bimapsa((isign+1)>>1,dir).Offsets().slice()),
  // 	CreateLeaf<QDPType<T1,C1> >::make(l)));
  //   }


  // template<class T1,class C1>
  // inline typename MakeReturn<UnaryNode<FnMap,
  //   typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t>, C1>::Expression_t
  // operator()(const QDPExpr<T1,C1> & l, int isign, int dir)
  //   {
  //     typedef UnaryNode<FnMap,
  // 	typename CreateLeaf<QDPExpr<T1,C1> >::Leaf_t> Tree_t;
  //     return MakeReturn<Tree_t,C1>::make(Tree_t(FnMap(bimapsa((isign+1)>>1,dir).Offsets().slice()),
  // 	CreateLeaf<QDPExpr<T1,C1> >::make(l)));
  //   }


  // fw: taken from parscalar
  template<class T1>
__device__
  OLattice<T1>
  operator()(const OLattice<T1> & l, int isign, int dir)
  {
#if defined(SPU_DEBUG)
    printf("parscalar version...\n");
#endif
    return bimapsa((isign+1)>>1,dir)(l);
  }
  template<class RHS, class T1>
__device__
  OLattice<T1>
  operator()(const QDPExpr<RHS,OLattice<T1> > & l, int isign, int dir)
  {
#if defined(SPU_DEBUG)
    printf("parscalar version...\n");
#endif
    return bimapsa((isign+1)>>1,dir)(l);
  }




private:
  //! Hide copy constructor
__device__
  ArrayBiDirectionalMap(const ArrayBiDirectionalMap&) {}

  //! Hide operator=
__device__
  void operator=(const ArrayBiDirectionalMap&) {}

private:
  multi2d<Map> bimapsa;
  
};


//-----------------------------------------------------------------------------
// Input and output of various flavors that are architecture specific

//! Decompose a lexicographic site into coordinates
__device__
multi1d<int> crtesn(int ipos, const multi1d<int>& latt_size);

// //! XML output
// template<class T>  
// XMLWriter& operator<<(XMLWriter& xml, const OLattice<T>& d)
// {
//   xml.openTag("OLattice");

//   XMLWriterAPI::AttributeList alist;

//   const int vvol = Layout::vol();
//   for(int site=0; site < vvol; ++site) 
//   { 
//     multi1d<int> coord = crtesn(site, Layout::lattSize());
//     std::ostringstream os;
//     os << coord[0];
//     for(int i=1; i < coord.size(); ++i)
//       os << " " << coord[i];

//     alist.clear();
//     alist.push_back(XMLWriterAPI::Attribute("site", site));
//     alist.push_back(XMLWriterAPI::Attribute("coord", os.str()));

//     xml.openTag("elem", alist);
//     xml << d.elem(Layout::linearSiteIndex(site));
//     xml.closeTag();
//   }

//   xml.closeTag(); // OLattice

//   return xml;
// }



// //! Binary output
// /*! Assumes no inner grid */
// template<class T>
// inline
// void write(BinaryWriter& bin, const OScalar<T>& d)
// {
//   if (Layout::primaryNode()) 
//     bin.writeArray((const char *)&(d.elem()), 
// 		   sizeof(typename WordType<T>::Type_t), 
// 		   sizeof(T) / sizeof(typename WordType<T>::Type_t));
// }

// //! Binary output
// /*! Assumes no inner grid */
// template<class T>  
// void write(BinaryWriter& bin, const OLattice<T>& d)
// {
//   const int vvol = Layout::vol();
//   for(int site=0; site < vvol; ++site) 
//   {
//     int i = Layout::linearSiteIndex(site);
//     bin.writeArray((const char*)&(d.elem(i)), 
// 		   sizeof(typename WordType<T>::Type_t), 
// 		   sizeof(T) / sizeof(typename WordType<T>::Type_t));
//   }
// }

// //! Write a single site of a lattice quantity at coord
// /*! Assumes no inner grid */
// template<class T>  
// void write(BinaryWriter& bin, const OLattice<T>& d, const multi1d<int>& coord)
// {
//   int i = Layout::linearSiteIndex(coord);
//   bin.writeArray((const char*)&(d.elem(i)), 
// 		 sizeof(typename WordType<T>::Type_t), 
// 		 sizeof(T) / sizeof(typename WordType<T>::Type_t));
// }

// //! Binary input
// /*! Assumes no inner grid */
// template<class T>
// void read(BinaryReader& bin, OScalar<T>& d)
// {
//   bin.readArray((char*)&(d.elem()), 
// 		sizeof(typename WordType<T>::Type_t), 
// 		sizeof(T) / sizeof(typename WordType<T>::Type_t)); 
// }

// //! Binary input
// /*! Assumes no inner grid */
// template<class T>  
// void read(BinaryReader& bin, OLattice<T>& d)
// {
//   const int vvol = Layout::vol();
//   for(int site=0; site < vvol; ++site) 
//   {
//     int i = Layout::linearSiteIndex(site);
//     bin.readArray((char*)&(d.elem(i)), 
// 		  sizeof(typename WordType<T>::Type_t), 
// 		  sizeof(T) / sizeof(typename WordType<T>::Type_t));
//   }
// }

// //! Read a single site and place it at coord
// /*! Assumes no inner grid */
// template<class T>  
// void read(BinaryReader& bin, OLattice<T>& d, const multi1d<int>& coord)
// {
//   int i = Layout::linearSiteIndex(coord);
//   bin.readArray((char*)&(d.elem(i)), 
// 		sizeof(typename WordType<T>::Type_t), 
// 		sizeof(T) / sizeof(typename WordType<T>::Type_t));
// }

} // namespace QDP

#endif
