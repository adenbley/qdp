// -*- C++ -*-
/*! @file
 * @brief Flop counters
 *
 * Flop counters
 */

#ifndef QDP_BYTECOUNT_H
#define QDP_BYTECOUNT_H

namespace QDP
{

  //---------------------------------------------------------------------
  /*! @defgroup qdpbytes Byte Counting Mechanism
   *
   * \ingroup qdp
   *
   * This is a basic byte counter class. It is cleared on instantiation
   * but can be cleared at any time by the user. It has functions
   * to add various kinds of bytecount and to retreive the total
   * accumulated bytecount.
   * @{
   */

  //! Basic Byte Counter Clas
  class ByteCounter {
  public:
    //! Constructor - zeroes bytecount
    ByteCounter(void) : count(0), sitesOnNode((unsigned long long)Layout::sitesOnNode()) {}

    //! Destructor - kills object. No cleanup needed
    ~ByteCounter() {} 

    //! Copy Constructor
    ByteCounter(const ByteCounter& c) : count(c.count), sitesOnNode(c.sitesOnNode) {}

    //! Explicit zero method. Clears bytecounts
    inline void reset(void) { 
      count = 0;
    }

    //! Method to add raw number of bytes (eg from Level 3 operators)
    inline void addBytes(unsigned long long bytes) { 
      count += bytes;
    }

    //! Method to add per site byte count. Count is multiplied by sitesOnNode()
    inline void addSiteBytes(unsigned long  long bytes) { 
      count += (bytes * sitesOnNode);
    }

    //! Method to add per site byte count for a subset of sites. Count is multiplied by the site table size of the subset (ie number of sites in a subset)
    inline void addSiteBytes(unsigned long bytes, const Subset& s) {
      count += (bytes * (unsigned long long)(s.numSiteTable()));
    }

    //! Method to retrieve accumulated bytecount
    inline unsigned long long getBytes(void) const { 
      return count;
    }

    //! Report bytepage
    inline void report(const std::string& name, 
		       const double& time_in_seconds) {

      double mbytes_per_cpu = (double)count/((double)(1000*1000)*time_in_seconds);
      double mbytes_overall = mbytes_per_cpu;
      Internal::globalSum(mbytes_overall);
      double gbytes_overall = mbytes_overall/(double)(1000);
      double tbytes_overall = gbytes_overall/(double)(1000);

      QDPIO::cout <<"QDP:ByteCount:" << name << " Performance/CPU: t=" << time_in_seconds << "(s) Bytes=" << (double)count << " => " << mbytes_per_cpu << " Mbytes/cpu." << endl;
      QDPIO::cout << "QDP:ByteCount:"  << name <<" Total performance:  " << mbytes_overall << " Mbytes = " << gbytes_overall << " Gbytes = " << tbytes_overall << " Tbytes" << endl;
    }



    inline double gbytes_overall(const double& time_in_seconds) {
      double mbytes_per_cpu = (double)count/((double)(1000*1000)*time_in_seconds);
      double mbytes_overall = mbytes_per_cpu;
      Internal::globalSum(mbytes_overall);
      double gbytes_overall = mbytes_overall/(double)(1000);
      double tbytes_overall = gbytes_overall/(double)(1000);
      return gbytes_overall;
    }



  private:
    unsigned long long count;
    const  unsigned long long sitesOnNode;
  };

  /*! @} */  // end of group 

} // namespace QDP

#endif
