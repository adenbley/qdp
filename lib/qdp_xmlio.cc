// $Id: qdp_xmlio.cc,v 1.4 2003-05-22 18:24:36 edwards Exp $
//
/*! @file
 * @brief XML IO support
 */

#include "qdp.h"

QDP_BEGIN_NAMESPACE(QDP);

using std::string;

//--------------------------------------------------------------------------------
// XML classes
// XML reader class
XMLReader::XMLReader() {iop=false;}

XMLReader::XMLReader(const std::string& filename)
{
  iop=false;
  open(filename);
}

XMLReader::XMLReader(std::istream& is)
{
  iop=false;
  open(is);
}

XMLReader::XMLReader(const XMLMetaWriter& mw)
{
  iop=false;
  open(mw);
}

void XMLReader::open(const string& filename)
{
  if (Layout::primaryNode())
    BasicXPathReader::open(filename);

  iop = true;
}

void XMLReader::open(std::istream& is)
{
  if (Layout::primaryNode())
    BasicXPathReader::open(is);

  iop = true;
}

void XMLReader::open(const XMLMetaWriter& mw)
{
  if (Layout::primaryNode())
    BasicXPathReader::open(const_cast<XMLMetaWriter&>(mw).str());

  iop = true;
}

void XMLReader::close()
{
  if (is_open()) 
  {
    if (Layout::primaryNode()) 
      BasicXPathReader::close();

    iop = false;
  }
}

bool XMLReader::is_open() {return iop;}

XMLReader::~XMLReader() {close();}


// Overloaded Reader Functions
void XMLReader::get(const std::string& xpath, string& result)
{
  if (Layout::primaryNode())
    BasicXPathReader::get(xpath, result);
}
void XMLReader::get(const std::string& xpath, int& result)
{
  if (Layout::primaryNode())
    BasicXPathReader::get(xpath, result);
}
void XMLReader::get(const std::string& xpath, unsigned int& result)
{
  if (Layout::primaryNode())
    BasicXPathReader::get(xpath, result);
}
void XMLReader::get(const std::string& xpath, short int& result)
{
  if (Layout::primaryNode())
    BasicXPathReader::get(xpath, result);
}
void XMLReader::get(const std::string& xpath, unsigned short int& result)
{
  if (Layout::primaryNode())
    BasicXPathReader::get(xpath, result);
}
void XMLReader::get(const std::string& xpath, long int& result)
{
  if (Layout::primaryNode())
    BasicXPathReader::get(xpath, result);
}
void XMLReader::get(const std::string& xpath, unsigned long int& result)
{
  if (Layout::primaryNode())
    BasicXPathReader::get(xpath, result);
}
void XMLReader::get(const std::string& xpath, float& result)
{
  if (Layout::primaryNode())
    BasicXPathReader::get(xpath, result);
}
void XMLReader::get(const std::string& xpath, double& result)
{
  if (Layout::primaryNode())
    BasicXPathReader::get(xpath, result);
}
void XMLReader::get(const std::string& xpath, bool& result)
{
  if (Layout::primaryNode())
    BasicXPathReader::get(xpath, result);
}
   
void XMLReader::print(ostream& os)
{
  if (Layout::primaryNode())
    BasicXPathReader::print(os);
}
   
void XMLReader::printRoot(ostream& os)
{
  if (Layout::primaryNode())
    BasicXPathReader::printRoot(os);
}
   

// Overloaded Reader Functions
void read(XMLReader& xml, const std::string& xpath, string& result)
{
  xml.get(xpath, result);
}
void read(XMLReader& xml, const std::string& xpath, int& result)
{
  xml.get(xpath, result);
}
void read(XMLReader& xml, const std::string& xpath, unsigned int& result)
{
  xml.get(xpath, result);
}
void read(XMLReader& xml, const std::string& xpath, short int& result)
{
  xml.get(xpath, result);
}
void read(XMLReader& xml, const std::string& xpath, unsigned short int& result)
{
  xml.get(xpath, result);
}
void read(XMLReader& xml, const std::string& xpath, long int& result)
{
  xml.get(xpath, result);
}
void read(XMLReader& xml, const std::string& xpath, unsigned long int& result)
{
  xml.get(xpath, result);
}
void read(XMLReader& xml, const std::string& xpath, float& result)
{
  xml.get(xpath, result);
}
void read(XMLReader& xml, const std::string& xpath, double& result)
{
  xml.get(xpath, result);
}
void read(XMLReader& xml, const std::string& xpath, bool& result)
{
  xml.get(xpath, result);
}
   


//--------------------------------------------------------------------------------
// XML writer base class
XMLWriter::XMLWriter()
{
}

XMLWriter::~XMLWriter()
{
}

void XMLWriter::openTag(const string& tagname)
{
  if (Layout::primaryNode())
    XMLSimpleWriter::openTag(tagname);
}

void XMLWriter::openTag(const string& nsprefix, const string& tagname)
{
  if (Layout::primaryNode())
    XMLSimpleWriter::openTag(nsprefix,tagname);
}

void XMLWriter::openTag(const string& tagname, XMLWriterAPI::AttributeList& al)
{
  if (Layout::primaryNode())
    XMLSimpleWriter::openTag(tagname,al);
}

void XMLWriter::openTag(const string& nsprefix, 
			const string& tagname, 
			XMLWriterAPI::AttributeList& al)
{
  if (Layout::primaryNode())
    XMLSimpleWriter::openTag(nsprefix,tagname,al);
}

void XMLWriter::closeTag()
{
  if (Layout::primaryNode())
    XMLSimpleWriter::closeTag();
}

void XMLWriter::emptyTag(const string& tagname)
{
  if (Layout::primaryNode())
    XMLSimpleWriter::emptyTag(tagname);
}
void XMLWriter::emptyTag(const string& tagname,  XMLWriterAPI::AttributeList& al)
{
  if (Layout::primaryNode())
    XMLSimpleWriter::emptyTag(tagname,al);
}

void XMLWriter::emptyTag(const string& nsprefix, 
			 const string& tagname, 
			 XMLWriterAPI::AttributeList& al)
{
  if (Layout::primaryNode())
    XMLSimpleWriter::emptyTag(nsprefix,tagname,al);
}


// Overloaded Writer Functions
void XMLWriter::write(const string& output)
{
  if (Layout::primaryNode())
    XMLSimpleWriter::write(output);
}
void XMLWriter::write(const int& output)
{
  if (Layout::primaryNode())
    XMLSimpleWriter::write(output);
}
void XMLWriter::write(const unsigned int& output)
{
  if (Layout::primaryNode())
    XMLSimpleWriter::write(output);
}
void XMLWriter::write(const short int& output)
{
  if (Layout::primaryNode())
    XMLSimpleWriter::write(output);
}
void XMLWriter::write(const unsigned short int& output)
{
  if (Layout::primaryNode())
    XMLSimpleWriter::write(output);
}
void XMLWriter::write(const long int& output)
{
  if (Layout::primaryNode())
    XMLSimpleWriter::write(output);
}
void XMLWriter::write(const unsigned long int& output)
{
  if (Layout::primaryNode())
    XMLSimpleWriter::write(output);
}
void XMLWriter::write(const float& output)
{
  if (Layout::primaryNode())
    XMLSimpleWriter::write(output);
}
void XMLWriter::write(const double& output)
{
  if (Layout::primaryNode())
    XMLSimpleWriter::write(output);
}
void XMLWriter::write(const bool& output)
{
  if (Layout::primaryNode())
    XMLSimpleWriter::write(output);
}
   
// Write XML string
void XMLWriter::writeXML(const string& output)
{
  if (Layout::primaryNode())
    XMLSimpleWriter::writeXML(output);
}


// Push a group name
void push(XMLWriter& xml, const string& s) {xml.openTag(s);}

// Pop a group name
void pop(XMLWriter& xml) {xml.closeTag();}

// Write something from a reader
void write(XMLWriter& xml, const std::string& s, const XMLReader& d)
{
  xml.openTag(s);
  xml << d;
  xml.closeTag();
}

XMLWriter& operator<<(XMLWriter& xml, const XMLReader& d)
{
  ostringstream os;
  const_cast<XMLReader&>(d).printRoot(os);
  xml.writeXML(os.str());
}

// Write something from a XMLMetaWriter
void write(XMLWriter& xml, const std::string& s, const XMLMetaWriter& d)
{
  xml.openTag(s);
  xml << d;
  xml.closeTag();
}

XMLWriter& operator<<(XMLWriter& xml, const XMLMetaWriter& d)
{
  xml.writeXML(const_cast<XMLMetaWriter&>(d).printRoot());
}

// Time to build a telephone book of basic primitives
void write(XMLWriter& xml, const string& s, const string& d)
{
  xml.openTag(s);
  xml.write(d);
  xml.closeTag();
}

//! Write an int
void write(XMLWriter& xml, const string& s, const int& d)
{
  xml.openTag(s);
  xml.write(d);
  xml.closeTag();
}

//! Write a float
void write(XMLWriter& xml, const string& s, const float& d)
{
  xml.openTag(s);
  xml.write(d);
  xml.closeTag();
}

//! Write a double
void write(XMLWriter& xml, const string& s, const double& d)
{
  xml.openTag(s);
  xml.write(d);
  xml.openTag(s);
}

//! Write a bool
void write(XMLWriter& xml, const string& s, const bool& d)
{
  xml.openTag(s);
  xml.write(d);
  xml.openTag(s);
}

// Versions that do not print a name
XMLWriter& operator<<(XMLWriter& xml, const string& d) {xml.write(d);return xml;}
XMLWriter& operator<<(XMLWriter& xml, const int& d) {xml.write(d);return xml;}
XMLWriter& operator<<(XMLWriter& xml, const float& d) {xml.write(d);return xml;}
XMLWriter& operator<<(XMLWriter& xml, const double& d) {xml.write(d);return xml;}
XMLWriter& operator<<(XMLWriter& xml, const bool& d) {xml.write(d);return xml;}




//--------------------------------------------------------------------------------
// Metadata writer class
XMLMetaWriter::XMLMetaWriter() {indent_level=0;}

string XMLMetaWriter::str()
{
  ostringstream s;
  
  if (Layout::primaryNode()) 
  {
    writePrologue(s);
    s << output_stream.str();
  }
    
  return s.str();
}

string XMLMetaWriter::printRoot() {return output_stream.str();}

XMLMetaWriter::~XMLMetaWriter() {}


//--------------------------------------------------------------------------------
// Metadata writer class
XMLDataWriter::XMLDataWriter() {indent_level=0;iop=false;}

void XMLDataWriter::close()
{
  if (is_open()) 
  {
    if (Layout::primaryNode()) 
      output_stream.close();

    iop = false;
  }
}

bool XMLDataWriter::is_open() {return iop;}

XMLDataWriter::~XMLDataWriter() {close();}


QDP_END_NAMESPACE();
