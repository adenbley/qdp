// -*- C++ -*-

/*! @file
 * @brief IO support
 */

#ifndef QDP_QLIMEWRITER_H
#define QDP_QLIMEWRITER_H

#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include "qdp_byteorder.h"

extern "C"
{
#include "lime.h"
}

namespace QDP
{

  // #include "defs.h"
  // #include <string>
  // #include <cstdio>
  // #include "common/logger.h"



  //! Interface to Lime
  class QLimeWriter
  {
  private:
    FILE*		_fp;			//!< File pointer
    LimeWriter*		_w;			//!< Pointer to Lime writer handler

    typedef enum { stateInit, stateHdr, stateData } State;

    State		_state;			//!< Current state

    string		_recordName;		//!< Record name
    uint64_t		_recordSize;		//!< Record size
    int			_recordMBFlag;		//!< Message begin flag
    int			_recordMEFlag;		//!< Message end flag

    void _writeHeader(void);

  public:
    QLimeWriter(const char*);
    ~QLimeWriter();
    
    void setRecordName(const char*);
    void setRecordSize(uint64_t);
    void setRecordMBFlag(int);
    void setRecordMEFlag(int);
    void setRecordHeader(const char*, uint64_t, int mBFlag = 0, int mEFlag = 0);

    void write(void*, uint64_t&);

    void endRecord(void) { _state = stateInit; }

    const string& recordName(void) const { return _recordName; }
    const int recordSize(void) const { return _recordSize; }
  };

  /*! @} */   // end of group io
} // namespace QDP

#endif
