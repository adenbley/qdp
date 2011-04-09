// -*- C++ -*-
// $Id: cudp_primspinvec.h,v 1.10 2009/10/16 10:25:00 edwards Exp $

/*! \file
 * \brief Primitive Spin Vector
 */



#ifndef QDP_PRIMITIVE_CELL64_H
#define QDP_PRIMITIVE_CELL64_H

namespace QDP {



  template <class T, template<class,int> class C> class PMatrix<T,3,C>
  {
  public:
    PMatrix() {}
    ~PMatrix() {}

    typedef C<T,3>  CC;


    template<class T1>
    inline
    CC& assign(const PScalar<T1>& rhs)
    {
      zero_rep(elem(0,1));
      zero_rep(elem(0,2));
      zero_rep(elem(1,0));
      zero_rep(elem(1,2));
      zero_rep(elem(2,0));
      zero_rep(elem(2,1));
      elem(0,0) = rhs.elem();
      elem(1,1) = rhs.elem();
      elem(2,2) = rhs.elem();
      return static_cast<CC&>(*this);
    }

    //! PMatrix = PMatrix
    /*! Set equal to another PMatrix */
    template<class T1>
    inline
    CC& assign(const C<T1,3>& rhs) 
    {
      elem(0,0) = rhs.elem(0,0);
      elem(0,1) = rhs.elem(0,1);
      elem(0,2) = rhs.elem(0,2);
      elem(1,0) = rhs.elem(1,0);
      elem(1,1) = rhs.elem(1,1);
      elem(1,2) = rhs.elem(1,2);
      elem(2,0) = rhs.elem(2,0);
      elem(2,1) = rhs.elem(2,1);
      elem(2,2) = rhs.elem(2,2);
      return static_cast<CC&>(*this);
    }


    template<class T1>
    inline
    CC& operator=(const C<T1,3>& rhs) 
    {
      elem(0,0) = rhs.elem(0,0);
      elem(0,1) = rhs.elem(0,1);
      elem(0,2) = rhs.elem(0,2);
      elem(1,0) = rhs.elem(1,0);
      elem(1,1) = rhs.elem(1,1);
      elem(1,2) = rhs.elem(1,2);
      elem(2,0) = rhs.elem(2,0);
      elem(2,1) = rhs.elem(2,1);
      elem(2,2) = rhs.elem(2,2);
      return static_cast<CC&>(*this);
    }



    inline
    PMatrix& operator=(const PMatrix& rhs) 
    {
      elem(0,0) = rhs.elem(0,0);
      elem(0,1) = rhs.elem(0,1);
      elem(0,2) = rhs.elem(0,2);
      elem(1,0) = rhs.elem(1,0);
      elem(1,1) = rhs.elem(1,1);
      elem(1,2) = rhs.elem(1,2);
      elem(2,0) = rhs.elem(2,0);
      elem(2,1) = rhs.elem(2,1);
      elem(2,2) = rhs.elem(2,2);
      return *this;
    }


    //! PMatrix += PMatrix
    template<class T1>
    inline
    CC& operator+=(const C<T1,3>& rhs) 
    {
      elem(0,0) += rhs.elem(0,0);
      elem(0,1) += rhs.elem(0,1);
      elem(0,2) += rhs.elem(0,2);
      elem(1,0) += rhs.elem(1,0);
      elem(1,1) += rhs.elem(1,1);
      elem(1,2) += rhs.elem(1,2);
      elem(2,0) += rhs.elem(2,0);
      elem(2,1) += rhs.elem(2,1);
      elem(2,2) += rhs.elem(2,2);
      return static_cast<CC&>(*this);
    }

    //! PMatrix -= PMatrix
    template<class T1>
    inline
    CC& operator-=(const C<T1,3>& rhs) 
    {
      elem(0,0) -= rhs.elem(0,0);
      elem(0,1) -= rhs.elem(0,1);
      elem(0,2) -= rhs.elem(0,2);
      elem(1,0) -= rhs.elem(1,0);
      elem(1,1) -= rhs.elem(1,1);
      elem(1,2) -= rhs.elem(1,2);
      elem(2,0) -= rhs.elem(2,0);
      elem(2,1) -= rhs.elem(2,1);
      elem(2,2) -= rhs.elem(2,2);
      return static_cast<CC&>(*this);
    }

    //! PMatrix += PScalar
    template<class T1>
    inline
    CC& operator+=(const PScalar<T1>& rhs) 
    {
      elem(0,0) += rhs.elem();
      elem(1,1) += rhs.elem();
      elem(2,2) += rhs.elem();
      return static_cast<CC&>(*this);
    }

    //! PMatrix -= PScalar
    template<class T1>
    inline
    CC& operator-=(const PScalar<T1>& rhs) 
    {
      elem(0,0) -= rhs.elem();
      elem(1,1) -= rhs.elem();
      elem(2,2) -= rhs.elem();
      return static_cast<CC&>(*this);
    }

    //! PMatrix *= PScalar
    template<class T1>
    inline
    CC& operator*=(const PScalar<T1>& rhs) 
    {
      elem(0,0) *= rhs.elem();
      elem(0,1) *= rhs.elem();
      elem(0,2) *= rhs.elem();
      elem(1,0) *= rhs.elem();
      elem(1,1) *= rhs.elem();
      elem(1,2) *= rhs.elem();
      elem(2,0) *= rhs.elem();
      elem(2,1) *= rhs.elem();
      elem(2,2) *= rhs.elem();
      return static_cast<CC&>(*this);
    }

    //! PMatrix /= PScalar
    template<class T1>
    inline
    CC& operator/=(const PScalar<T1>& rhs) 
    {
      elem(0,0) /= rhs.elem();
      elem(0,1) /= rhs.elem();
      elem(0,2) /= rhs.elem();
      elem(1,0) /= rhs.elem();
      elem(1,1) /= rhs.elem();
      elem(1,2) /= rhs.elem();
      elem(2,0) /= rhs.elem();
      elem(2,1) /= rhs.elem();
      elem(2,2) /= rhs.elem();
      return static_cast<CC&>(*this);
    }


    // PMatrix(const PMatrix& rhs)
    //   {
    //     elem(0,0) = rhs.elem(0,0);
    //     elem(0,1) = rhs.elem(0,1);
    //     elem(0,2) = rhs.elem(0,2);
    //     elem(1,0) = rhs.elem(1,0);
    //     elem(1,1) = rhs.elem(1,1);
    //     elem(1,2) = rhs.elem(1,2);
    //     elem(2,0) = rhs.elem(2,0);
    //     elem(2,1) = rhs.elem(2,1);
    //     elem(2,2) = rhs.elem(2,2);
    //     // for(int i=0; i < 3*3; ++i)
    //     // 	F[i] = a.F[i];
    //   }

  public:
    inline T& elem(int i, int j) {return F[j+3*i];}
    const inline T& elem(int i, int j) const {return F[j+3*i];}

  private:
    T F[3*3];
  };





  template <class T> class PColorMatrix<T,3> : public PMatrix<T, 3, PColorMatrix>
  {
  public:
    //! PColorMatrix = PScalar
    /*! Fill with primitive scalar */
    template<class T1>
    inline
    PColorMatrix& operator=(const PScalar<T1>& rhs)
    {
      assign(rhs);
      return *this;
    }

    //! PColorMatrix = PColorMatrix
    /*! Set equal to another PMatrix */
    template<class T1>
    inline
    PColorMatrix& operator=(const PColorMatrix<T1,3>& rhs) 
    {
      assign(rhs);
      return *this;
    }


  };










  template <class T, template<class,int> class C> class PMatrix<T,4,C>
  {
  public:
    PMatrix() {}
    ~PMatrix() {}

    typedef C<T,4>  CC;


    template<class T1>
    inline
    CC& assign(const PScalar<T1>& rhs)
    {
      zero_rep(elem(0,1));
      zero_rep(elem(0,2));
      zero_rep(elem(0,3));

      zero_rep(elem(1,0));
      zero_rep(elem(1,2));
      zero_rep(elem(1,3));

      zero_rep(elem(2,0));
      zero_rep(elem(2,1));
      zero_rep(elem(2,3));

      zero_rep(elem(3,0));
      zero_rep(elem(3,1));
      zero_rep(elem(3,2));

      elem(0,0) = rhs.elem();
      elem(1,1) = rhs.elem();
      elem(2,2) = rhs.elem();
      elem(3,3) = rhs.elem();
      return static_cast<CC&>(*this);
    }

    //! PMatrix = PMatrix
    /*! Set equal to another PMatrix */
    template<class T1>
    inline
    CC& assign(const C<T1,4>& rhs) 
    {
      elem(0,0) = rhs.elem(0,0);
      elem(0,1) = rhs.elem(0,1);
      elem(0,2) = rhs.elem(0,2);
      elem(0,3) = rhs.elem(0,3);

      elem(1,0) = rhs.elem(1,0);
      elem(1,1) = rhs.elem(1,1);
      elem(1,2) = rhs.elem(1,2);
      elem(1,3) = rhs.elem(1,3);

      elem(2,0) = rhs.elem(2,0);
      elem(2,1) = rhs.elem(2,1);
      elem(2,2) = rhs.elem(2,2);
      elem(2,3) = rhs.elem(2,3);

      elem(3,0) = rhs.elem(3,0);
      elem(3,1) = rhs.elem(3,1);
      elem(3,2) = rhs.elem(3,2);
      elem(3,3) = rhs.elem(3,3);
      return static_cast<CC&>(*this);
    }


    template<class T1>
    inline
    CC& operator=(const C<T1,4>& rhs) 
    {
      elem(0,0) = rhs.elem(0,0);
      elem(0,1) = rhs.elem(0,1);
      elem(0,2) = rhs.elem(0,2);
      elem(0,3) = rhs.elem(0,3);

      elem(1,0) = rhs.elem(1,0);
      elem(1,1) = rhs.elem(1,1);
      elem(1,2) = rhs.elem(1,2);
      elem(1,3) = rhs.elem(1,3);

      elem(2,0) = rhs.elem(2,0);
      elem(2,1) = rhs.elem(2,1);
      elem(2,2) = rhs.elem(2,2);
      elem(2,3) = rhs.elem(2,3);

      elem(3,0) = rhs.elem(3,0);
      elem(3,1) = rhs.elem(3,1);
      elem(3,2) = rhs.elem(3,2);
      elem(3,3) = rhs.elem(3,3);
      return static_cast<CC&>(*this);
    }


    inline
    PMatrix& operator=(const PMatrix& rhs) 
    {
      elem(0,0) = rhs.elem(0,0);
      elem(0,1) = rhs.elem(0,1);
      elem(0,2) = rhs.elem(0,2);
      elem(0,3) = rhs.elem(0,3);

      elem(1,0) = rhs.elem(1,0);
      elem(1,1) = rhs.elem(1,1);
      elem(1,2) = rhs.elem(1,2);
      elem(1,3) = rhs.elem(1,3);

      elem(2,0) = rhs.elem(2,0);
      elem(2,1) = rhs.elem(2,1);
      elem(2,2) = rhs.elem(2,2);
      elem(2,3) = rhs.elem(2,3);

      elem(3,0) = rhs.elem(3,0);
      elem(3,1) = rhs.elem(3,1);
      elem(3,2) = rhs.elem(3,2);
      elem(3,3) = rhs.elem(3,3);
      return *this;
    }


    //! PMatrix += PMatrix
    template<class T1>
    inline
    CC& operator+=(const C<T1,4>& rhs) 
    {
      elem(0,0) += rhs.elem(0,0);
      elem(0,1) += rhs.elem(0,1);
      elem(0,2) += rhs.elem(0,2);
      elem(0,3) += rhs.elem(0,3);

      elem(1,0) += rhs.elem(1,0);
      elem(1,1) += rhs.elem(1,1);
      elem(1,2) += rhs.elem(1,2);
      elem(1,3) += rhs.elem(1,3);

      elem(2,0) += rhs.elem(2,0);
      elem(2,1) += rhs.elem(2,1);
      elem(2,2) += rhs.elem(2,2);
      elem(2,3) += rhs.elem(2,3);

      elem(3,0) += rhs.elem(3,0);
      elem(3,1) += rhs.elem(3,1);
      elem(3,2) += rhs.elem(3,2);
      elem(3,3) += rhs.elem(3,3);

      return static_cast<CC&>(*this);
    }

    //! PMatrix -= PMatrix
    template<class T1>
    inline
    CC& operator-=(const C<T1,4>& rhs) 
    {
      elem(0,0) -= rhs.elem(0,0);
      elem(0,1) -= rhs.elem(0,1);
      elem(0,2) -= rhs.elem(0,2);
      elem(0,3) -= rhs.elem(0,3);

      elem(1,0) -= rhs.elem(1,0);
      elem(1,1) -= rhs.elem(1,1);
      elem(1,2) -= rhs.elem(1,2);
      elem(1,3) -= rhs.elem(1,3);

      elem(2,0) -= rhs.elem(2,0);
      elem(2,1) -= rhs.elem(2,1);
      elem(2,2) -= rhs.elem(2,2);
      elem(2,3) -= rhs.elem(2,3);

      elem(3,0) -= rhs.elem(3,0);
      elem(3,1) -= rhs.elem(3,1);
      elem(3,2) -= rhs.elem(3,2);
      elem(3,3) -= rhs.elem(3,3);

      return static_cast<CC&>(*this);
    }

    //! PMatrix += PScalar
    template<class T1>
    inline
    CC& operator+=(const PScalar<T1>& rhs) 
    {
      elem(0,0) += rhs.elem();
      elem(1,1) += rhs.elem();
      elem(2,2) += rhs.elem();
      elem(3,3) += rhs.elem();
      return static_cast<CC&>(*this);
    }

    //! PMatrix -= PScalar
    template<class T1>
    inline
    CC& operator-=(const PScalar<T1>& rhs) 
    {
      elem(0,0) -= rhs.elem();
      elem(1,1) -= rhs.elem();
      elem(2,2) -= rhs.elem();
      elem(3,3) -= rhs.elem();
      return static_cast<CC&>(*this);
    }

    //! PMatrix *= PScalar
    template<class T1>
    inline
    CC& operator*=(const PScalar<T1>& rhs) 
    {
      elem(0,0) *= rhs.elem();
      elem(0,1) *= rhs.elem();
      elem(0,2) *= rhs.elem();
      elem(0,3) *= rhs.elem();

      elem(1,0) *= rhs.elem();
      elem(1,1) *= rhs.elem();
      elem(1,2) *= rhs.elem();
      elem(1,3) *= rhs.elem();

      elem(2,0) *= rhs.elem();
      elem(2,1) *= rhs.elem();
      elem(2,2) *= rhs.elem();
      elem(2,3) *= rhs.elem();

      elem(3,0) *= rhs.elem();
      elem(3,1) *= rhs.elem();
      elem(3,2) *= rhs.elem();
      elem(3,3) *= rhs.elem();

      return static_cast<CC&>(*this);
    }

    //! PMatrix /= PScalar
    template<class T1>
    inline
    CC& operator/=(const PScalar<T1>& rhs) 
    {
      elem(0,0) /= rhs.elem();
      elem(0,1) /= rhs.elem();
      elem(0,2) /= rhs.elem();
      elem(0,3) /= rhs.elem();

      elem(1,0) /= rhs.elem();
      elem(1,1) /= rhs.elem();
      elem(1,2) /= rhs.elem();
      elem(1,3) /= rhs.elem();

      elem(2,0) /= rhs.elem();
      elem(2,1) /= rhs.elem();
      elem(2,2) /= rhs.elem();
      elem(2,3) /= rhs.elem();

      elem(3,0) /= rhs.elem();
      elem(3,1) /= rhs.elem();
      elem(3,2) /= rhs.elem();
      elem(3,3) /= rhs.elem();

      return static_cast<CC&>(*this);
    }


    // PMatrix(const PMatrix& rhs)
    //   {
    //     elem(0,0) = rhs.elem(0,0);
    //     elem(0,1) = rhs.elem(0,1);
    //     elem(0,2) = rhs.elem(0,2);
    //     elem(1,0) = rhs.elem(1,0);
    //     elem(1,1) = rhs.elem(1,1);
    //     elem(1,2) = rhs.elem(1,2);
    //     elem(2,0) = rhs.elem(2,0);
    //     elem(2,1) = rhs.elem(2,1);
    //     elem(2,2) = rhs.elem(2,2);
    //     // for(int i=0; i < 3*3; ++i)
    //     // 	F[i] = a.F[i];
    //   }

  public:
    inline T& elem(int i, int j) {return F[j+4*i];}
    const inline T& elem(int i, int j) const {return F[j+4*i];}

  private:
    T F[4*4];
  };





  template <class T> class PSpinMatrix<T,4> : public PMatrix<T, 4, PSpinMatrix>
  {
  public:

    template<class T1>
    inline
    PSpinMatrix& operator=(const PScalar<T1>& rhs)
    {
      assign(rhs);
      return *this;
    }

    template<class T1>
    inline
    PSpinMatrix& operator=(const PSpinMatrix<T1,3>& rhs) 
    {
      assign(rhs);
      return *this;
    }


  };














  template<>
  class RComplex<REAL64>
  {
  public:
    RComplex() {}
    ~RComplex() {}

    // template<class T1, class T2>
    // RComplex(const RScalar<T1>& _re, const RScalar<T2>& _im): re(_re.elem()), im(_im.elem()) {}

    template<class T1, class T2>
    inline
    RComplex(const T1& _re, const T2& _im) 
    {
      F = (vector double){ _re , _im };
    }

    template<class T1>
    inline
    RComplex& operator=(const RScalar<T1>& rhs) 
    {
      F = (vector double){ rhs.elem() , 0.0 };
      return *this;
    }


    inline
    RComplex& operator=(const RComplex<REAL32>& rhs) 
    {
      F = (vector double){ rhs.real() , rhs.imag() };
      return *this;
    }

    inline
    RComplex& operator=(const RComplex<REAL64>& rhs) 
    {
      F = rhs.F;
      return *this;
    }

    // template<class T1>
    // inline
    // RComplex& operator=(const RComplex<T1>& rhs) 
    // {
    //   F = (vector double){ rhs.real() , rhs.imag() };
    //   return *this;
    // }




    // inline
    // RComplex<REAL64>& operator=(const RComplex<float>& rhs)
    // {
    //   const vector unsigned char swap  = { 0,1,2,3 , 0,1,2,3 , 4,5,6,7 , 4,5,6,7 };
    //   F = spu_extend( spu_shuffle( rhs.fl , rhs.fl , swap ) );
    //   return *this;
    // }





    // template<class T1>
    // inline
    // RComplex& operator+=(const RScalar<T1>& rhs) 
    // {
    //   real() += rhs.elem();
    //   return *this;
    // }


    // template<class T1>
    // inline
    // RComplex& operator-=(const RScalar<T1>& rhs) 
    // {
    //   real() -= rhs.elem();
    //   return *this;
    // }


    template<class T1>
    inline
    RComplex& operator*=(const RScalar<T1>& rhs) 
    {
      vector double d = (vector double){ rhs.elem() , rhs.elem() };
      F = spu_mul( F , d );
      return *this;
    }


    // template<class T1>
    // inline
    // RComplex& operator/=(const RScalar<T1>& rhs) 
    // {
    //   real() /= rhs.elem();
    //   imag() /= rhs.elem();
    //   return *this;
    // }



    // inline
    // RComplex<REAL64>& operator=(const RComplex& rhs) 
    // {
    //   F = rhs.F;
    //   return *this;
    // }


    inline
    RComplex& operator+=(const RComplex<REAL64>& rhs) 
    {
      F = spu_add( F , rhs.F );
      return *this;
    }

    // template<class T1>
    // inline
    // RComplex& operator+=(const RComplex<T1>& rhs) 
    // {
    //   vector double d = (vector double){ rhs.real() , rhs.imag() };
    //   F = spu_add( F , d );
    //   return *this;
    // }



    // template<>
    // inline
    // RComplex& operator+=(const RComplex<float>& rhs) 
    // {
    //   F = spu_add( F , rhs.F );
    //   return *this;
    // }

    inline
    RComplex& operator-=(const RComplex<REAL64>& rhs) 
    {
      F = spu_sub( F , rhs.F );
      return *this;
    }

    // template<class T1>
    // inline
    // RComplex& operator-=(const RComplex<T1>& rhs) 
    // {
    //   vector double d = (vector double){ rhs.real() , rhs.imag() };
    //   F = spu_add( F , d );
    //   return *this;
    // }


    template<class T1>
    inline
    RComplex& operator*=(const RComplex<T1>& rhs)
    {
      RComplex d;
      d = *this * rhs;
      F = d.F;
      return *this;
    }


    // template<class T1>
    // inline
    // RComplex& operator/=(const RComplex<T1>& rhs) 
    // {
    //   RComplex<T> d;
    //   d = *this / rhs;

    //   real() = d.real();
    //   imag() = d.imag();
    //   return *this;
    // }



  public:
    inline RComplex(const RComplex& a) { F=a.F; }
    inline RComplex(const vector double& va) { F=va; }
    inline RComplex(      vector double& va) { F=va; }

    //inline vector double & elem() { return F; }
    inline REAL64 real() { return spu_extract(F,0); }
    inline const REAL64 real() const {return spu_extract(F,0);}

    inline REAL64 imag() {return spu_extract(F,1);}
    inline const REAL64 imag() const {return spu_extract(F,1);}

    //private:
    vector double F;
  };   // possibly force alignment



  template<class T1, class T2, template<class,int> class C>
  inline typename BinaryReturn<PMatrix<T1,3,C>, PMatrix<T2,3,C>, OpMultiply>::Type_t
  operator*(const PMatrix<T1,3,C>& l, const PMatrix<T2,3,C>& r)
  {
    typename BinaryReturn<PMatrix<T1,3,C>, PMatrix<T2,3,C>, OpMultiply>::Type_t  d;

    d.elem(0,0) = l.elem(0,0)*r.elem(0,0) + l.elem(0,1)*r.elem(1,0) + l.elem(0,2)*r.elem(2,0);
    d.elem(0,1) = l.elem(0,0)*r.elem(0,1) + l.elem(0,1)*r.elem(1,1) + l.elem(0,2)*r.elem(2,1);
    d.elem(0,2) = l.elem(0,0)*r.elem(0,2) + l.elem(0,1)*r.elem(1,2) + l.elem(0,2)*r.elem(2,2);

    d.elem(1,0) = l.elem(1,0)*r.elem(0,0) + l.elem(1,1)*r.elem(1,0) + l.elem(1,2)*r.elem(2,0);
    d.elem(1,1) = l.elem(1,0)*r.elem(0,1) + l.elem(1,1)*r.elem(1,1) + l.elem(1,2)*r.elem(2,1);
    d.elem(1,2) = l.elem(1,0)*r.elem(0,2) + l.elem(1,1)*r.elem(1,2) + l.elem(1,2)*r.elem(2,2);

    d.elem(2,0) = l.elem(2,0)*r.elem(0,0) + l.elem(2,1)*r.elem(1,0) + l.elem(2,2)*r.elem(2,0);
    d.elem(2,1) = l.elem(2,0)*r.elem(0,1) + l.elem(2,1)*r.elem(1,1) + l.elem(2,2)*r.elem(2,1);
    d.elem(2,2) = l.elem(2,0)*r.elem(0,2) + l.elem(2,1)*r.elem(1,2) + l.elem(2,2)*r.elem(2,2);

    // for(int i=0; i < N; ++i)
    //   for(int j=0; j < N; ++j)
    //   {
    //     d.elem(i,j) = l.elem(i,0) * r.elem(0,j);
    //     for(int k=1; k < N; ++k)
    // 	d.elem(i,j) += l.elem(i,k) * r.elem(k,j);
    //   }

    return d;
  }



  template<class T1, class T2, template<class,int> class C>
  inline typename BinaryReturn<PMatrix<T1,3,C>, PMatrix<T2,3,C>, OpAdd>::Type_t
  operator+(const PMatrix<T1,3,C>& l, const PMatrix<T2,3,C>& r)
  {
    typename BinaryReturn<PMatrix<T1,3,C>, PMatrix<T2,3,C>, OpAdd>::Type_t  d;

    d.elem(0,0) = l.elem(0,0) + r.elem(0,0);
    d.elem(1,0) = l.elem(1,0) + r.elem(1,0);
    d.elem(2,0) = l.elem(2,0) + r.elem(2,0);
    d.elem(0,1) = l.elem(0,1) + r.elem(0,1);
    d.elem(1,1) = l.elem(1,1) + r.elem(1,1);
    d.elem(2,1) = l.elem(2,1) + r.elem(2,1);
    d.elem(0,2) = l.elem(0,2) + r.elem(0,2);
    d.elem(1,2) = l.elem(1,2) + r.elem(1,2);
    d.elem(2,2) = l.elem(2,2) + r.elem(2,2);
    
    return d;
  }




  template<class T1, class T2, template<class,int> class C>
  inline typename BinaryReturn<PMatrix<T1,3,C>, PScalar<T2>, OpMultiply>::Type_t
  operator*(const PMatrix<T1,3,C>& l, const PScalar<T2>& r)
  {
    typename BinaryReturn<PMatrix<T1,3,C>, PScalar<T2>, OpMultiply>::Type_t  d;
    
    d.elem(0,0) = l.elem(0,0) * r.elem();
    d.elem(1,0) = l.elem(1,0) * r.elem();
    d.elem(2,0) = l.elem(2,0) * r.elem();
    d.elem(0,1) = l.elem(0,1) * r.elem();
    d.elem(1,1) = l.elem(1,1) * r.elem();
    d.elem(2,1) = l.elem(2,1) * r.elem();
    d.elem(0,2) = l.elem(0,2) * r.elem();
    d.elem(1,2) = l.elem(1,2) * r.elem();
    d.elem(2,2) = l.elem(2,2) * r.elem();

    return d;
  }


  template<class T1, class T2, template<class,int> class C>
  inline typename BinaryReturn<PScalar<T2>,PMatrix<T1,3,C>, OpMultiply>::Type_t
  operator*(const PScalar<T2>& l, const PMatrix<T1,3,C>& r)
  {
    typename BinaryReturn<PScalar<T2>, PMatrix<T1,3,C>, OpMultiply>::Type_t  d;
    
    d.elem(0,0) = l.elem() * r.elem(0,0);
    d.elem(1,0) = l.elem() * r.elem(1,0);
    d.elem(2,0) = l.elem() * r.elem(2,0);
    d.elem(0,1) = l.elem() * r.elem(0,1);
    d.elem(1,1) = l.elem() * r.elem(1,1);
    d.elem(2,1) = l.elem() * r.elem(2,1);
    d.elem(0,2) = l.elem() * r.elem(0,2);
    d.elem(1,2) = l.elem() * r.elem(1,2);
    d.elem(2,2) = l.elem() * r.elem(2,2);

    return d;
  }



  template<class T1, class T2, template<class,int> class C>
  inline typename BinaryReturn<PMatrix<T1,3,C>, PMatrix<T2,3,C>, FnLocalInnerProduct>::Type_t
  localInnerProduct(const PMatrix<T1,3,C>& s1, const PMatrix<T2,3,C>& s2)
  {
    typename BinaryReturn<PMatrix<T1,3,C>, PMatrix<T2,3,C>, FnLocalInnerProduct>::Type_t  d;

    d.elem() = 
      localInnerProduct(s1.elem(0,0), s2.elem(0,0)) + 
      localInnerProduct(s1.elem(1,0), s2.elem(1,0)) + 
      localInnerProduct(s1.elem(2,0), s2.elem(2,0)) + 
      localInnerProduct(s1.elem(0,1), s2.elem(0,1)) + 
      localInnerProduct(s1.elem(1,1), s2.elem(1,1)) + 
      localInnerProduct(s1.elem(2,1), s2.elem(2,1)) + 
      localInnerProduct(s1.elem(0,2), s2.elem(0,2)) + 
      localInnerProduct(s1.elem(1,2), s2.elem(1,2)) + 
      localInnerProduct(s1.elem(2,2), s2.elem(2,2));

    return d;
  }

  template<class T1, class T2, template<class,int> class C>
  inline typename BinaryReturn<PMatrix<T1,4,C>, PMatrix<T2,4,C>, FnLocalInnerProduct>::Type_t
  localInnerProduct(const PMatrix<T1,4,C>& s1, const PMatrix<T2,4,C>& s2)
  {
    typename BinaryReturn<PMatrix<T1,4,C>, PMatrix<T2,4,C>, FnLocalInnerProduct>::Type_t  d;

    d.elem() = 
      localInnerProduct(s1.elem(0,0), s2.elem(0,0)) + 
      localInnerProduct(s1.elem(1,0), s2.elem(1,0)) + 
      localInnerProduct(s1.elem(2,0), s2.elem(2,0)) + 
      localInnerProduct(s1.elem(3,0), s2.elem(3,0)) + 
      localInnerProduct(s1.elem(0,1), s2.elem(0,1)) + 
      localInnerProduct(s1.elem(1,1), s2.elem(1,1)) + 
      localInnerProduct(s1.elem(2,1), s2.elem(2,1)) + 
      localInnerProduct(s1.elem(3,1), s2.elem(3,1)) + 
      localInnerProduct(s1.elem(0,2), s2.elem(0,2)) + 
      localInnerProduct(s1.elem(1,2), s2.elem(1,2)) + 
      localInnerProduct(s1.elem(2,2), s2.elem(2,2)) + 
      localInnerProduct(s1.elem(3,2), s2.elem(3,2)) + 
      localInnerProduct(s1.elem(0,3), s2.elem(0,3)) + 
      localInnerProduct(s1.elem(1,3), s2.elem(1,3)) + 
      localInnerProduct(s1.elem(2,3), s2.elem(2,3)) + 
      localInnerProduct(s1.elem(3,3), s2.elem(3,3));

    return d;
  }

template<>
inline BinaryReturn<RComplex<REAL64>, RComplex<REAL64>, FnLocalInnerProduct>::Type_t
localInnerProduct(const RComplex<REAL64>& l, const RComplex<REAL64>& r)
{
  typedef BinaryReturn<RComplex<REAL64>, RComplex<REAL64>, FnLocalInnerProduct>::Type_t  Ret_t;

  typedef vector double T;
  const vector unsigned char swap  = { 8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7 };
  const vector unsigned char alter = { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 
				       0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F  };

  T c = spu_mul( l.F , r.F );
  c   = spu_add( c , spu_shuffle( c , c , swap ) );
  T d = spu_mul( r.F , spu_shuffle( l.F , l.F , swap ) );
  d   = spu_sub( d , spu_shuffle( d , d , swap ) );

  return Ret_t( spu_shuffle( c , d , alter ) );

  // return Ret_t(l.real()*r.real() + l.imag()*r.imag(),
  // 		 l.real()*r.imag() - l.imag()*r.real());

  // // op*
  // return Ret_t(l.real()*r.real() - l.imag()*r.imag(),
  // 		 l.real()*r.imag() + l.imag()*r.real());

}



template<class T1, class T2, template<class,int> class C>
inline typename BinaryReturn<PMatrix<T1,4,C>, PMatrix<T2,4,C>, OpMultiply>::Type_t
operator*(const PMatrix<T1,4,C>& l, const PMatrix<T2,4,C>& r)
{
  typename BinaryReturn<PMatrix<T1,4,C>, PMatrix<T2,4,C>, OpMultiply>::Type_t  d;

  d.elem(0,0) = l.elem(0,0)*r.elem(0,0) + l.elem(0,1)*r.elem(1,0) + l.elem(0,2)*r.elem(2,0) + l.elem(0,3)*r.elem(3,0);
  d.elem(0,1) = l.elem(0,0)*r.elem(0,1) + l.elem(0,1)*r.elem(1,1) + l.elem(0,2)*r.elem(2,1) + l.elem(0,3)*r.elem(3,1);
  d.elem(0,2) = l.elem(0,0)*r.elem(0,2) + l.elem(0,1)*r.elem(1,2) + l.elem(0,2)*r.elem(2,2) + l.elem(0,3)*r.elem(3,2);
  d.elem(0,3) = l.elem(0,0)*r.elem(0,3) + l.elem(0,1)*r.elem(1,3) + l.elem(0,2)*r.elem(2,3) + l.elem(0,3)*r.elem(3,3);

  d.elem(1,0) = l.elem(1,0)*r.elem(0,0) + l.elem(1,1)*r.elem(1,0) + l.elem(1,2)*r.elem(2,0) + l.elem(1,3)*r.elem(3,0);
  d.elem(1,1) = l.elem(1,0)*r.elem(0,1) + l.elem(1,1)*r.elem(1,1) + l.elem(1,2)*r.elem(2,1) + l.elem(1,3)*r.elem(3,1);
  d.elem(1,2) = l.elem(1,0)*r.elem(0,2) + l.elem(1,1)*r.elem(1,2) + l.elem(1,2)*r.elem(2,2) + l.elem(1,3)*r.elem(3,2);
  d.elem(1,3) = l.elem(1,0)*r.elem(0,3) + l.elem(1,1)*r.elem(1,3) + l.elem(1,2)*r.elem(2,3) + l.elem(1,3)*r.elem(3,3);

  d.elem(2,0) = l.elem(2,0)*r.elem(0,0) + l.elem(2,1)*r.elem(1,0) + l.elem(2,2)*r.elem(2,0) + l.elem(2,3)*r.elem(3,0);
  d.elem(2,1) = l.elem(2,0)*r.elem(0,1) + l.elem(2,1)*r.elem(1,1) + l.elem(2,2)*r.elem(2,1) + l.elem(2,3)*r.elem(3,1);
  d.elem(2,2) = l.elem(2,0)*r.elem(0,2) + l.elem(2,1)*r.elem(1,2) + l.elem(2,2)*r.elem(2,2) + l.elem(2,3)*r.elem(3,2);
  d.elem(2,3) = l.elem(2,0)*r.elem(0,3) + l.elem(2,1)*r.elem(1,3) + l.elem(2,2)*r.elem(2,3) + l.elem(2,3)*r.elem(3,3);

  d.elem(3,0) = l.elem(3,0)*r.elem(0,0) + l.elem(3,1)*r.elem(1,0) + l.elem(3,2)*r.elem(2,0) + l.elem(3,3)*r.elem(3,0);
  d.elem(3,1) = l.elem(3,0)*r.elem(0,1) + l.elem(3,1)*r.elem(1,1) + l.elem(3,2)*r.elem(2,1) + l.elem(3,3)*r.elem(3,1);
  d.elem(3,2) = l.elem(3,0)*r.elem(0,2) + l.elem(3,1)*r.elem(1,2) + l.elem(3,2)*r.elem(2,2) + l.elem(3,3)*r.elem(3,2);
  d.elem(3,3) = l.elem(3,0)*r.elem(0,3) + l.elem(3,1)*r.elem(1,3) + l.elem(3,2)*r.elem(2,3) + l.elem(3,3)*r.elem(3,3);

  // for(int i=0; i < N; ++i)
  //   for(int j=0; j < N; ++j)
  //   {
  //     d.elem(i,j) = l.elem(i,0) * r.elem(0,j);
  //     for(int k=1; k < N; ++k)
  // 	d.elem(i,j) += l.elem(i,k) * r.elem(k,j);
  //   }

  return d;
}


  template<class T1, class T2, template<class,int> class C>
  inline typename BinaryReturn<PMatrix<T1,3,C>, PMatrix<T2,3,C>, OpAdjMultiply>::Type_t
  adjMultiply(const PMatrix<T1,3,C>& l, const PMatrix<T2,3,C>& r)
  {
    typename BinaryReturn<PMatrix<T1,3,C>, PMatrix<T2,3,C>, OpAdjMultiply>::Type_t  d;

    d.elem(0,0) = adjMultiply(l.elem(0,0),r.elem(0,0)) + adjMultiply(l.elem(1,0),r.elem(1,0)) + adjMultiply(l.elem(2,0),r.elem(2,0));
    d.elem(0,1) = adjMultiply(l.elem(0,0),r.elem(0,1)) + adjMultiply(l.elem(1,0),r.elem(1,1)) + adjMultiply(l.elem(2,0),r.elem(2,1));
    d.elem(0,2) = adjMultiply(l.elem(0,0),r.elem(0,2)) + adjMultiply(l.elem(1,0),r.elem(1,2)) + adjMultiply(l.elem(2,0),r.elem(2,2));

    d.elem(1,0) = adjMultiply(l.elem(0,1),r.elem(0,0)) + adjMultiply(l.elem(1,1),r.elem(1,0)) + adjMultiply(l.elem(2,1),r.elem(2,0));
    d.elem(1,1) = adjMultiply(l.elem(0,1),r.elem(0,1)) + adjMultiply(l.elem(1,1),r.elem(1,1)) + adjMultiply(l.elem(2,1),r.elem(2,1));
    d.elem(1,2) = adjMultiply(l.elem(0,1),r.elem(0,2)) + adjMultiply(l.elem(1,1),r.elem(1,2)) + adjMultiply(l.elem(2,1),r.elem(2,2));

    d.elem(2,0) = adjMultiply(l.elem(0,2),r.elem(0,0)) + adjMultiply(l.elem(1,2),r.elem(1,0)) + adjMultiply(l.elem(2,2),r.elem(2,0));
    d.elem(2,1) = adjMultiply(l.elem(0,2),r.elem(0,1)) + adjMultiply(l.elem(1,2),r.elem(1,1)) + adjMultiply(l.elem(2,2),r.elem(2,1));
    d.elem(2,2) = adjMultiply(l.elem(0,2),r.elem(0,2)) + adjMultiply(l.elem(1,2),r.elem(1,2)) + adjMultiply(l.elem(2,2),r.elem(2,2));

    // for(int i=0; i < N; ++i)
    //   for(int j=0; j < N; ++j)
    //   {
    //     d.elem(i,j) = adjMultiply(l.elem(0,i), r.elem(0,j));
    //     for(int k=1; k < N; ++k)
    // 	d.elem(i,j) += adjMultiply(l.elem(k,i), r.elem(k,j));
    //   }

    return d;
  }


  template<class T1, class T2, template<class,int> class C>
  inline typename BinaryReturn<PMatrix<T1,3,C>, PMatrix<T2,3,C>, OpMultiplyAdj>::Type_t
  multiplyAdj(const PMatrix<T1,3,C>& l, const PMatrix<T2,3,C>& r)
  {
    typename BinaryReturn<PMatrix<T1,3,C>, PMatrix<T2,3,C>, OpMultiplyAdj>::Type_t  d;

    d.elem(0,0) = multiplyAdj(l.elem(0,0),r.elem(0,0)) + multiplyAdj(l.elem(0,1),r.elem(0,1)) + multiplyAdj(l.elem(0,2),r.elem(0,2));
    d.elem(0,1) = multiplyAdj(l.elem(0,0),r.elem(1,0)) + multiplyAdj(l.elem(0,1),r.elem(1,1)) + multiplyAdj(l.elem(0,2),r.elem(1,2));
    d.elem(0,2) = multiplyAdj(l.elem(0,0),r.elem(2,0)) + multiplyAdj(l.elem(0,1),r.elem(2,1)) + multiplyAdj(l.elem(0,2),r.elem(2,2));

    d.elem(1,0) = multiplyAdj(l.elem(1,0),r.elem(0,0)) + multiplyAdj(l.elem(1,1),r.elem(0,1)) + multiplyAdj(l.elem(1,2),r.elem(0,2));
    d.elem(1,1) = multiplyAdj(l.elem(1,0),r.elem(1,0)) + multiplyAdj(l.elem(1,1),r.elem(1,1)) + multiplyAdj(l.elem(1,2),r.elem(1,2));
    d.elem(1,2) = multiplyAdj(l.elem(1,0),r.elem(2,0)) + multiplyAdj(l.elem(1,1),r.elem(2,1)) + multiplyAdj(l.elem(1,2),r.elem(2,2));

    d.elem(2,0) = multiplyAdj(l.elem(2,0),r.elem(0,0)) + multiplyAdj(l.elem(2,1),r.elem(0,1)) + multiplyAdj(l.elem(2,2),r.elem(0,2));
    d.elem(2,1) = multiplyAdj(l.elem(2,0),r.elem(1,0)) + multiplyAdj(l.elem(2,1),r.elem(1,1)) + multiplyAdj(l.elem(2,2),r.elem(1,2));
    d.elem(2,2) = multiplyAdj(l.elem(2,0),r.elem(2,0)) + multiplyAdj(l.elem(2,1),r.elem(2,1)) + multiplyAdj(l.elem(2,2),r.elem(2,2));

    return d;
  }


  // // fw kill
  // template<class T1, class T2, template<class,int> class C>
  // inline typename BinaryReturn<PMatrix<T1,3,C>, PMatrix<T2,3,C>, OpAdd>::Type_t
  // operator+(const PMatrix<T1,3,C>& l, const PMatrix<T2,3,C>& r)
  // {
  //   typename BinaryReturn<PMatrix<T1,3,C>, PMatrix<T2,3,C>, OpAdd>::Type_t  d;

  //   d.elem(0,0) = l.elem(0,0) + r.elem(0,0);
  //   d.elem(0,1) = l.elem(0,1) + r.elem(0,1);
  //   d.elem(0,2) = l.elem(0,2) + r.elem(0,2);

  //   d.elem(1,0) = l.elem(1,0) + r.elem(1,0);
  //   d.elem(1,1) = l.elem(1,1) + r.elem(1,1);
  //   d.elem(1,2) = l.elem(1,2) + r.elem(1,2);

  //   d.elem(2,0) = l.elem(2,0) + r.elem(2,0);
  //   d.elem(2,1) = l.elem(2,1) + r.elem(2,1);
  //   d.elem(2,2) = l.elem(2,2) + r.elem(2,2);

  //   return d;
  // }


  // template<class T1, class T2, template<class,int> class C>
  // inline typename BinaryReturn<PMatrix<T1,4,C>, PMatrix<T2,4,C>, OpAdd>::Type_t
  // operator+(const PMatrix<T1,4,C>& l, const PMatrix<T2,4,C>& r)
  // {
  //   typename BinaryReturn<PMatrix<T1,4,C>, PMatrix<T2,4,C>, OpAdd>::Type_t  d;

  //   d.elem(0,0) = l.elem(0,0) + r.elem(0,0);
  //   d.elem(0,1) = l.elem(0,1) + r.elem(0,1);
  //   d.elem(0,2) = l.elem(0,2) + r.elem(0,2);
  //   d.elem(0,3) = l.elem(0,3) + r.elem(0,3);

  //   d.elem(1,0) = l.elem(1,0) + r.elem(1,0);
  //   d.elem(1,1) = l.elem(1,1) + r.elem(1,1);
  //   d.elem(1,2) = l.elem(1,2) + r.elem(1,2);
  //   d.elem(1,3) = l.elem(1,3) + r.elem(1,3);

  //   d.elem(2,0) = l.elem(2,0) + r.elem(2,0);
  //   d.elem(2,1) = l.elem(2,1) + r.elem(2,1);
  //   d.elem(2,2) = l.elem(2,2) + r.elem(2,2);
  //   d.elem(2,3) = l.elem(2,3) + r.elem(2,3);

  //   d.elem(3,0) = l.elem(3,0) + r.elem(3,0);
  //   d.elem(3,1) = l.elem(3,1) + r.elem(3,1);
  //   d.elem(3,2) = l.elem(3,2) + r.elem(3,2);
  //   d.elem(3,3) = l.elem(3,3) + r.elem(3,3);

  //   return d;
  // }










  // template<class T1, class T2, template<class,int> class C>
  // inline typename BinaryReturn<PMatrix<T1,3,C>, PScalar<T2>, OpMultiply>::Type_t
  // operator*(const PMatrix<T1,3,C>& l, const PScalar<T2>& r)
  // {
  //   typename BinaryReturn<PMatrix<T1,3,C>, PScalar<T2>, OpMultiply>::Type_t  d;

  //   d.elem(0,0) = l.elem(0,0) * r.elem();
  //   d.elem(0,1) = l.elem(0,1) * r.elem();
  //   d.elem(0,2) = l.elem(0,2) * r.elem();

  //   d.elem(1,0) = l.elem(1,0) * r.elem();
  //   d.elem(1,1) = l.elem(1,1) * r.elem();
  //   d.elem(1,2) = l.elem(1,2) * r.elem();

  //   d.elem(2,0) = l.elem(2,0) * r.elem();
  //   d.elem(2,1) = l.elem(2,1) * r.elem();
  //   d.elem(2,2) = l.elem(2,2) * r.elem();

  //   return d;
  // }


  // template<class T1, class T2, template<class,int> class C>
  // inline typename BinaryReturn<PScalar<T1>, PMatrix<T2,3,C>, OpMultiply>::Type_t
  // operator*(const PScalar<T1>& l, const PMatrix<T2,3,C>& r)
  // {
  //   typename BinaryReturn<PScalar<T1>, PMatrix<T2,3,C>, OpMultiply>::Type_t  d;

  //   d.elem(0,0) =l.elem() * r.elem(0,0);
  //   d.elem(0,1) =l.elem() * r.elem(0,1);
  //   d.elem(0,2) =l.elem() * r.elem(0,2);

  //   d.elem(1,0) =l.elem() * r.elem(1,0);
  //   d.elem(1,1) =l.elem() * r.elem(1,1);
  //   d.elem(1,2) =l.elem() * r.elem(1,2);

  //   d.elem(2,0) =l.elem() * r.elem(2,0);
  //   d.elem(2,1) =l.elem() * r.elem(2,1);
  //   d.elem(2,2) =l.elem() * r.elem(2,2);

  //   return d;
  // }


  // template<class T1, class T2, template<class,int> class C>
  // inline typename BinaryReturn<PScalar<T1>, PMatrix<T2,4,C>, OpMultiply>::Type_t
  // operator*(const PScalar<T1>& l, const PMatrix<T2,4,C>& r)
  // {
  //   typename BinaryReturn<PScalar<T1>, PMatrix<T2,4,C>, OpMultiply>::Type_t  d;

  //   d.elem(0,0) =l.elem() * r.elem(0,0);
  //   d.elem(0,1) =l.elem() * r.elem(0,1);
  //   d.elem(0,2) =l.elem() * r.elem(0,2);
  //   d.elem(0,3) =l.elem() * r.elem(0,3);

  //   d.elem(1,0) =l.elem() * r.elem(1,0);
  //   d.elem(1,1) =l.elem() * r.elem(1,1);
  //   d.elem(1,2) =l.elem() * r.elem(1,2);
  //   d.elem(1,3) =l.elem() * r.elem(1,3);

  //   d.elem(2,0) =l.elem() * r.elem(2,0);
  //   d.elem(2,1) =l.elem() * r.elem(2,1);
  //   d.elem(2,2) =l.elem() * r.elem(2,2);
  //   d.elem(2,3) =l.elem() * r.elem(2,3);

  //   d.elem(3,0) =l.elem() * r.elem(3,0);
  //   d.elem(3,1) =l.elem() * r.elem(3,1);
  //   d.elem(3,2) =l.elem() * r.elem(3,2);
  //   d.elem(3,3) =l.elem() * r.elem(3,3);

  //   return d;
  // }





  // template<class T1, class T2, template<class,int> class C>
  // inline typename BinaryReturn<PVector<T1,3,C>, PScalar<T2>, OpMultiply>::Type_t
  // operator*(const PVector<T1,3,C>& l, const PScalar<T2>& r)
  // {
  //   typename BinaryReturn<PVector<T1,3,C>, PScalar<T2>, OpMultiply>::Type_t  d;

  //   d.elem(0) = l.elem(0) * r.elem();
  //   d.elem(1) = l.elem(1) * r.elem();
  //   d.elem(2) = l.elem(2) * r.elem();

  //   return d;
  // }

  // template<class T1, class T2, template<class,int> class C>
  // inline typename BinaryReturn<PScalar<T1>, PVector<T2,3,C>, OpMultiply>::Type_t
  // operator*(const PScalar<T1>& l, const PVector<T2,3,C>& r)
  // {
  //   typename BinaryReturn<PScalar<T1>, PVector<T2,3,C>, OpMultiply>::Type_t  d;

  //   d.elem(0) = r.elem(0) * l.elem();
  //   d.elem(1) = r.elem(1) * l.elem();
  //   d.elem(2) = r.elem(2) * l.elem();

  //   return d;
  // }


  template<class T1, class T2, template<class,int> class C1, template<class,int> class C2>
  inline typename BinaryReturn<PMatrix<T1,3,C1>, PVector<T2,3,C2>, OpMultiply>::Type_t
  operator*(const PMatrix<T1,3,C1>& l, const PVector<T2,3,C2>& r)
  {
    typename BinaryReturn<PMatrix<T1,3,C1>, PVector<T2,3,C2>, OpMultiply>::Type_t  d;

    d.elem(0) = l.elem(0,0)*r.elem(0) + l.elem(0,1)*r.elem(1) + l.elem(0,2)*r.elem(2);
    d.elem(1) = l.elem(1,0)*r.elem(0) + l.elem(1,1)*r.elem(1) + l.elem(1,2)*r.elem(2);
    d.elem(2) = l.elem(2,0)*r.elem(0) + l.elem(2,1)*r.elem(1) + l.elem(2,2)*r.elem(2);

    return d;
  }

  template<class T1, class T2, template<class,int> class C1, template<class,int> class C2>
  inline typename BinaryReturn<PMatrix<T1,3,C1>, PVector<T2,3,C2>, OpAdjMultiply>::Type_t
  adjMultiply(const PMatrix<T1,3,C1>& l, const PVector<T2,3,C2>& r)
  {
    typename BinaryReturn<PMatrix<T1,3,C1>, PVector<T2,3,C2>, OpAdjMultiply>::Type_t  d;

    d.elem(0) = adjMultiply(l.elem(0,0),r.elem(0)) + adjMultiply(l.elem(1,0),r.elem(1)) + adjMultiply(l.elem(2,0),r.elem(2));
    d.elem(1) = adjMultiply(l.elem(0,1),r.elem(0)) + adjMultiply(l.elem(1,1),r.elem(1)) + adjMultiply(l.elem(2,1),r.elem(2));
    d.elem(2) = adjMultiply(l.elem(0,2),r.elem(0)) + adjMultiply(l.elem(1,2),r.elem(1)) + adjMultiply(l.elem(2,2),r.elem(2));

    return d;
  }

  template<class T1, class T2, template<class,int> class C1, template<class,int> class C2>
  inline typename BinaryReturn<PMatrix<T1,4,C1>, PVector<T2,4,C2>, OpAdjMultiply>::Type_t
  adjMultiply(const PMatrix<T1,4,C1>& l, const PVector<T2,4,C2>& r)
  {
    typename BinaryReturn<PMatrix<T1,4,C1>, PVector<T2,4,C2>, OpAdjMultiply>::Type_t  d;

    d.elem(0) = adjMultiply(l.elem(0,0),r.elem(0)) + adjMultiply(l.elem(1,0),r.elem(1)) + adjMultiply(l.elem(2,0),r.elem(2)) + adjMultiply(l.elem(3,0),r.elem(3));
    d.elem(1) = adjMultiply(l.elem(0,1),r.elem(0)) + adjMultiply(l.elem(1,1),r.elem(1)) + adjMultiply(l.elem(2,1),r.elem(2)) + adjMultiply(l.elem(3,1),r.elem(3));
    d.elem(2) = adjMultiply(l.elem(0,2),r.elem(0)) + adjMultiply(l.elem(1,2),r.elem(1)) + adjMultiply(l.elem(2,2),r.elem(2)) + adjMultiply(l.elem(3,2),r.elem(3));
    d.elem(3) = adjMultiply(l.elem(0,3),r.elem(0)) + adjMultiply(l.elem(1,3),r.elem(1)) + adjMultiply(l.elem(2,3),r.elem(2)) + adjMultiply(l.elem(3,3),r.elem(3));

    return d;
  }

  template<class T1, class T2, template<class,int> class C1, template<class,int> class C2>
  inline typename BinaryReturn<PMatrix<T1,4,C1>, PVector<T2,4,C2>, OpMultiply>::Type_t
  operator*(const PMatrix<T1,4,C1>& l, const PVector<T2,4,C2>& r)
  {
    typename BinaryReturn<PMatrix<T1,4,C1>, PVector<T2,4,C2>, OpMultiply>::Type_t  d;

    d.elem(0) = (l.elem(0*0)*r.elem(0)) + (l.elem(0*1)*r.elem(1)) + (l.elem(0*2)*r.elem(2)) + (l.elem(0*3)*r.elem(3));
    d.elem(1) = (l.elem(1*0)*r.elem(0)) + (l.elem(1*1)*r.elem(1)) + (l.elem(1*2)*r.elem(2)) + (l.elem(1*3)*r.elem(3));
    d.elem(2) = (l.elem(2*0)*r.elem(0)) + (l.elem(2*1)*r.elem(1)) + (l.elem(2*2)*r.elem(2)) + (l.elem(2*3)*r.elem(3));
    d.elem(3) = (l.elem(3*0)*r.elem(0)) + (l.elem(3*1)*r.elem(1)) + (l.elem(3*2)*r.elem(2)) + (l.elem(3*3)*r.elem(3));

    return d;
  }


  // // fw kill
  // template<class T1, class T2, template<class,int> class C>
  // inline typename BinaryReturn<PScalar<T1>, PMatrix<T2,3,C>, OpAdjMultiply>::Type_t
  // adjMultiply(const PScalar<T1>& l, const PMatrix<T2,3,C>& r)
  // {
  //   typename BinaryReturn<PScalar<T1>, PMatrix<T2,3,C>, OpAdjMultiply>::Type_t  d;

  //   d.elem(0,0) = adjMultiply(l.elem(), r.elem(0,0));
  //   d.elem(0,1) = adjMultiply(l.elem(), r.elem(0,1));
  //   d.elem(0,2) = adjMultiply(l.elem(), r.elem(0,2));

  //   d.elem(1,0) = adjMultiply(l.elem(), r.elem(1,0));
  //   d.elem(1,1) = adjMultiply(l.elem(), r.elem(1,1));
  //   d.elem(1,2) = adjMultiply(l.elem(), r.elem(1,2));

  //   d.elem(2,0) = adjMultiply(l.elem(), r.elem(2,0));
  //   d.elem(2,1) = adjMultiply(l.elem(), r.elem(2,1));
  //   d.elem(2,2) = adjMultiply(l.elem(), r.elem(2,2));

  //   return d;
  // }



  // template<class T1, class T2, template<class,int> class C>
  // inline typename BinaryReturn<PScalar<T1>, PVector<T2,3,C>, OpAdjMultiply>::Type_t
  // adjMultiply(const PScalar<T1>& l, const PVector<T2,3,C>& r)
  // {
  //   typename BinaryReturn<PScalar<T1>, PVector<T2,3,C>, OpAdjMultiply>::Type_t  d;

  //   d.elem(0) = adjMultiply(l.elem(), r.elem(0));
  //   d.elem(1) = adjMultiply(l.elem(), r.elem(1));
  //   d.elem(2) = adjMultiply(l.elem(), r.elem(2));

  //   return d;
  // }








template<>
inline UnaryReturn<RComplex<REAL64>, FnTimesI>::Type_t
timesI(const RComplex<REAL64>& s1)
{
  typedef UnaryReturn<RComplex<REAL64>, FnTimesI>::Type_t  Ret_t;

  const vector unsigned char swap  = { 8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7 };
  const vector double help         = { -1.0 , 1.0 };

  Ret_t d( spu_mul( spu_shuffle( s1.F , s1.F , swap ) , help ) );
  return d;
}

template<>
inline UnaryReturn<RComplex<REAL64>, FnTimesMinusI>::Type_t 
timesMinusI(const RComplex<REAL64>& s1)
{
  typedef UnaryReturn<RComplex<REAL64>, FnTimesMinusI>::Type_t  Ret_t;
  const vector unsigned char swap  = { 8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7 };
  const vector double help         = { 1.0 , -1.0 };

  Ret_t d( spu_mul( spu_shuffle( s1.F , s1.F , swap ) , help ) );
  return d;
}



  // template<class T, template<class,int> class C>
  // inline typename UnaryReturn<PMatrix<T,3,C>, FnTimesI>::Type_t
  // timesI(const PMatrix<T,3,C>& s1)
  // {
  //   typename UnaryReturn<PMatrix<T,3,C>, FnTimesI>::Type_t  d;

  //   d.elem(0,0) = timesI(s1.elem(0,0));
  //   d.elem(0,1) = timesI(s1.elem(0,1));
  //   d.elem(0,2) = timesI(s1.elem(0,2));

  //   d.elem(1,0) = timesI(s1.elem(1,0));
  //   d.elem(1,1) = timesI(s1.elem(1,1));
  //   d.elem(1,2) = timesI(s1.elem(1,2));

  //   d.elem(2,0) = timesI(s1.elem(2,0));
  //   d.elem(2,1) = timesI(s1.elem(2,1));
  //   d.elem(2,2) = timesI(s1.elem(2,2));

  //   return d;
  // }


  // template<class T, template<class,int> class C>
  // inline typename UnaryReturn<PMatrix<T,3,C>, FnTimesMinusI>::Type_t
  // timesMinusI(const PMatrix<T,3,C>& s1)
  // {
  //   typename UnaryReturn<PMatrix<T,3,C>, FnTimesMinusI>::Type_t  d;

  //   d.elem(0,0) = timesMinusI(s1.elem(0,0));
  //   d.elem(0,1) = timesMinusI(s1.elem(0,1));
  //   d.elem(0,2) = timesMinusI(s1.elem(0,2));

  //   d.elem(1,0) = timesMinusI(s1.elem(1,0));
  //   d.elem(1,1) = timesMinusI(s1.elem(1,1));
  //   d.elem(1,2) = timesMinusI(s1.elem(1,2));

  //   d.elem(2,0) = timesMinusI(s1.elem(2,0));
  //   d.elem(2,1) = timesMinusI(s1.elem(2,1));
  //   d.elem(2,2) = timesMinusI(s1.elem(2,2));

  //   return d;
  // }




  //! RComplex = RComplex * RComplex
  template<>
  inline BinaryReturn<RComplex<REAL64>, RComplex<REAL64>, OpMultiply>::Type_t
  operator*(const RComplex<REAL64>& l, const RComplex<REAL64>& r) 
  {
    typedef BinaryReturn<RComplex<REAL64>, RComplex<REAL64>, OpMultiply>::Type_t  Ret_t;

    typedef vector double T;
    const vector unsigned char swap  = { 8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7 };
    const vector unsigned char alter = { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 
					 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F  };

    T c = spu_mul( l.F , r.F );
    c   = spu_sub( c , spu_shuffle( c , c , swap ) );
    T d = spu_mul( r.F , spu_shuffle( l.F , l.F , swap ) );
    d   = spu_add( d , spu_shuffle( d , d , swap ) );

    return Ret_t( spu_shuffle( c , d , alter ) );

    // return Ret_t(l.real()*r.real() - l.imag()*r.imag(),
    // 		 l.real()*r.imag() + l.imag()*r.real());
  }



  //! RComplex =  RComplex + RComplex
template<>
inline BinaryReturn<RComplex<REAL64>, RComplex<REAL64>, OpAdd>::Type_t
operator+(const RComplex<REAL64>& l, const RComplex<REAL64>& r) 
{
  typedef BinaryReturn<RComplex<REAL64>, RComplex<REAL64>, OpAdd>::Type_t  Ret_t;
  return Ret_t( spu_add( l.F , r.F ) );
}








  //! RComplex =  RComplex - RComplex
  template<>
  inline BinaryReturn<RComplex<REAL64>, RComplex<REAL64>, OpSubtract>::Type_t
  operator-(const RComplex<REAL64>& l, const RComplex<REAL64>& r) 
  {
    typedef BinaryReturn<RComplex<REAL64>, RComplex<REAL64>, OpSubtract>::Type_t  Ret_t;
    return Ret_t( spu_sub( l.F , r.F ) );
  }


  template<>
  inline UnaryReturn<RComplex<REAL64>, OpUnaryMinus>::Type_t
  operator-(const RComplex<REAL64>& l)
  {
    typedef UnaryReturn<RComplex<REAL64>, OpUnaryMinus>::Type_t  Ret_t;
    const vector double zero = { 0.0 , 0.0 };
    return Ret_t( spu_sub( zero , l.F ) );
  }


  // Optimized  adj(RComplex)*RComplex
  template<>
  inline BinaryReturn<RComplex<REAL64>, RComplex<REAL64>, OpAdjMultiply>::Type_t
  adjMultiply(const RComplex<REAL64>& l, const RComplex<REAL64>& r)
  {
    typedef BinaryReturn<RComplex<REAL64>, RComplex<REAL64>, OpAdjMultiply>::Type_t  Ret_t;

    typedef vector double T;
    const vector unsigned char swap  = { 8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7 };
    const vector unsigned char alter = { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 
					 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F  };

    T c = spu_mul( l.F , r.F );
    c   = spu_add( c , spu_shuffle( c , c , swap ) );
    T d = spu_mul( r.F , spu_shuffle( l.F , l.F , swap ) );
    d   = spu_sub( d , spu_shuffle( d , d , swap ) );
    return Ret_t( spu_shuffle( c , d , alter ) );

    // return Ret_t(l.real()*r.real() + l.imag()*r.imag(),
    // 		 l.real()*r.imag() - l.imag()*r.real());
  }


  template<>
  inline BinaryReturn<RComplex<REAL64>, RComplex<REAL64>, OpMultiplyAdj>::Type_t
  multiplyAdj(const RComplex<REAL64>& l, const RComplex<REAL64>& r)
  {
    typedef BinaryReturn<RComplex<REAL64>, RComplex<REAL64>, OpMultiplyAdj>::Type_t  Ret_t;

    typedef vector double T;
    const vector unsigned char swap  = { 8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7 };
    const vector unsigned char alter = { 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 
					 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F  };

    T c = spu_mul( l.F , r.F );
    c   = spu_add( c , spu_shuffle( c , c , swap ) );
    T d = spu_mul( r.F , spu_shuffle( l.F , l.F , swap ) );
    d   = spu_sub( spu_shuffle( d , d , swap ) , d );
    return Ret_t( spu_shuffle( c , d , alter ) );

    // return Ret_t(l.real()*r.real() + l.imag()*r.imag(),
    // 		    l.imag()*r.real() - l.real()*r.imag());
  }



  // Adjoint
  template<>
  inline UnaryReturn<RComplex<REAL64>, FnAdjoint>::Type_t
  adj(const RComplex<REAL64>& l)
  {
    typedef UnaryReturn<RComplex<REAL64>, FnAdjoint>::Type_t  Ret_t;

    const vector double help  = { 1.0 , -1.0 };

    return Ret_t( spu_mul( l.F , help ) );
  }




  // // // Assignment is different
  // template<>
  // struct BinaryReturn<PSpinVector< PColorVector< RComplex<REAL>, Nc > , Ns >, PSpinVector< PColorVector< RComplex<REAL>, Nc > , Ns >, OpAssign > {
  //   typedef PSpinVector<PColorVector< RComplex<REAL>, Nc>,Ns> &Type_t;
  // };




} // namespace QDP

#endif
