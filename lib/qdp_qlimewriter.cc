//#include "defs.h"
//#include "common/dcap-overload.h"


#include "qdp.h"
#include "qdp_byteorder.h"

namespace QDP
{

  QLimeWriter::QLimeWriter(const char* file)
  {
    if (! Layout::primaryNode()) return;

    _fp = std::fopen(file, "w");
    if (_fp == NULL)
      QDP_error_exit("failed to open \"%s\" for writing\n", file);
    
    _w = limeCreateWriter(_fp);
    
    _recordName   = string("");
    _recordSize   = 0;
    _recordMBFlag = 0;
    _recordMEFlag = 0;
    
    _state = stateInit;
    
  }



  QLimeWriter::~QLimeWriter(void)
  {
    if (! Layout::primaryNode()) return;

    limeDestroyWriter(_w);

    if (_fp != NULL)
      DCAP(fclose)(_fp);
  }




  void QLimeWriter::setRecordName(const char* name)
  {
    if (! Layout::primaryNode()) return;

    if (_state != stateInit)
      QDP_error_exit("Writing record data not completed\n");

    _recordName = string(name);

    _state = stateHdr;
  }

  void QLimeWriter::setRecordSize(uint64_t size)
  {
    if (! Layout::primaryNode()) return;

    if (_state != stateInit)
      QDP_error_exit("Writing record data not completed\n");

    _recordSize = size;

    _state = stateHdr;
  }


  void QLimeWriter::setRecordMBFlag(int flag)
  {
    if (! Layout::primaryNode()) return;

    if (_state != stateInit)
      QDP_error_exit("Writing record data not completed\n");

    _recordMBFlag = flag;

    _state = stateHdr;
  }


  void QLimeWriter::setRecordMEFlag(int flag)
  {
    if (! Layout::primaryNode()) return;

    if (_state != stateInit)
      QDP_error_exit("Writing record data not completed\n");

    _recordMEFlag = flag;

    _state = stateHdr;
  }


  void QLimeWriter::setRecordHeader(const char* name, uint64_t size, int mBFlag, int mEFlag)
  {
    if (! Layout::primaryNode()) return;

    if (_state != stateInit)
      QDP_error_exit("Writing record data not completed\n");

    _recordName   = string(name);
    _recordSize   = size;
    _recordMBFlag = mBFlag;
    _recordMEFlag = mEFlag;

    _state = stateHdr;
  }




  void QLimeWriter::_writeHeader(void)
  {
    if (! Layout::primaryNode()) return;

    if (_state != stateHdr)
      QDP_error_exit("Record header not defined\n");
    if (_recordName == "")
      QDP_error_exit("No record name defined\n");
    if (_recordSize == 0)
      QDP_error_exit("No record size defined\n");

    LimeRecordHeader *h;
    h = limeCreateHeader(_recordMBFlag, _recordMEFlag, (char*) _recordName.c_str(), _recordSize);

    limeWriteRecordHeader(h, _w);

    limeDestroyHeader(h);

    _state = stateData;
  }




  void QLimeWriter::write(void* buf, uint64_t& size)
  {
    if (! Layout::primaryNode()) return;

    if (_state == stateInit)
      QDP_error_exit("Record header not defined\n");
    if (_state == stateHdr)
      _writeHeader();

    uint64_t sizeWritten = size;
    limeWriteRecordData(buf, &sizeWritten, _w);
    if (sizeWritten != size)
      QDP_error_exit("Failed to write all data\n");
  }


}

