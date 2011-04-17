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
#include <climits>
#include "cudp_byteorder.h"

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


  enum QLimeReturn {
    QLIME_UNDEF = INT_MAX,
    QLIME_SUCCESS = LIME_SUCCESS,
    QLIME_EOF = LIME_EOF,
  };


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



  class QLimeReader
  {
  private:
    FILE*		_fp;			//!< File pointer
    LimeReader*		_r;			//!< Pointer to Lime reader handler

  public:
    QLimeReader(const char*);
    ~QLimeReader();

    bool read(void*, uint64_t&) const;
    bool seek(off_t, int) const;

    const char* recordName(void) const;
    off_t recordSize(void) const { return limeReaderBytes(_r); }	//!< Record size in bytes

    QLimeReturn nextRecord(void) const;

    bool isMessageBegin(void) const { return (limeReaderMBFlag(_r) != 0); }
    bool isMessageEnd(void) const { return (limeReaderMEFlag(_r) != 0); }

    bool eof(void) const { return (feof(_r->fp) != 0); }
  };





  /*! @} */   // end of group io
} // namespace QDP

#endif
