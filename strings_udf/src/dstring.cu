/*
 * Copyright (c) 2021, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdio>
#include <cstring>
#include "dstring.cuh"

typedef unsigned char BYTE;


// returns the number of bytes used to represent that char
__host__ __device__ static unsigned int bytes_in_char_byte(BYTE byte)
{
    unsigned int count = 1;
    // no if-statements means no divergence
    count += (int)((byte & 0xF0) == 0xF0);
    count += (int)((byte & 0xE0) == 0xE0);
    count += (int)((byte & 0xC0) == 0xC0);
    count -= (int)((byte & 0xC0) == 0x80);
    return count;
}


// utility for methods allowing for multiple delimiters (e.g. strip)
__device__ static bool has_one_of( const char* tgts, Char chr )
{
    Char tchr = 0;
    unsigned int cw = dstring::char_to_Char(tgts,tchr);
    while( tchr )
    {
        if( tchr==chr )
            return true;
        tgts += cw;
        cw = dstring::char_to_Char(tgts,tchr);
    }
    return false;
}

__device__ static unsigned int string_length( const char* str )
{
    if( !str )
        return 0;
    unsigned int bytes = 0;
    while(*str++)
        ++bytes;
    return bytes;
}

__device__ char* dstring::allocate(unsigned int bytes)
{
    char* data = (char*)malloc(bytes+1);
    if( !data )
        printf("!out of malloc data! mem-size request: %d bytes\n",bytes);
    data[bytes] = 0;
    return data;
}

__device__ void dstring::deallocate(char* data)
{
    if( (m_flags & 1)==0 && data )
        free(data);
}

__device__ dstring::dstring() : m_data(nullptr), m_bytes(0), m_flags(0)
{
    //printf("empty ctor\n");
}

__device__ dstring::dstring(const char* data, unsigned int bytes)
    : m_data(nullptr), m_bytes(bytes), m_flags(1)
{
    m_data = const_cast<char*>(data);
    //printf("const ctor(%p,%d)\n", m_data, m_bytes);
}

__device__ dstring::dstring(const char* data)
    : m_data(nullptr), m_flags(1)
{
    m_data = const_cast<char*>(data);
    m_bytes = string_length(data);
    //printf("const ctor(%p)\n", m_data);
}

__device__ dstring::dstring(char* data, unsigned int bytes)
    : m_data(nullptr), m_bytes(bytes), m_flags(0)
{
    m_data = allocate(bytes);
    memcpy(m_data,data,bytes);
    //printf("ctor(%p,%d)\n", m_data,m_bytes);
}

__device__ dstring::dstring(char* data)
    : m_data(nullptr), m_bytes(0), m_flags(0)
{
    if( data )
    {
        m_bytes = string_length(data);
        m_data = allocate(m_bytes);
        memcpy(m_data,data,m_bytes);
    }
    //printf("ctor(%p)\n", m_data);
}

__device__ dstring::dstring(const dstring& src)
{
    m_bytes = src.m_bytes;
    m_flags = 0;
    m_data = allocate(m_bytes);
    memcpy(m_data,src.m_data,m_bytes);
    //printf("copy ctor(%p)\n", m_data);
}

__device__ dstring::dstring(dstring&& src)
{
    m_bytes = src.m_bytes;
    m_flags = src.m_flags;
    m_data = src.m_data;
    src.m_bytes = 0;
    src.m_flags = 0;
    src.m_data = nullptr;
    //printf("move ctor(%p)\n", m_data);
}

__device__ dstring::~dstring()
{
    //printf("dtor(%p):%d\n", m_data,m_flags);;
    deallocate(m_data);
}

__device__ dstring& dstring::operator=(const dstring& src)
{
    //printf("cp=(%p):%d/(%p):%d\n", m_data,m_flags,src.m_data,src.m_flags);
    if( &src == this )
        return *this;
    m_bytes = src.m_bytes;
    deallocate(m_data);
    if( src.m_flags )
        m_data = src.m_data;
    else
    {
        m_data = allocate(m_bytes);
        memcpy(m_data,src.m_data,m_bytes);
    }
    m_flags = src.m_flags;
    return *this;
}

__device__ dstring& dstring::operator=(dstring&& src)
{
    //printf("mv=(%p):%d/(%p):%d\n", m_data,m_flags,src.m_data,src.m_flags);
    if( &src == this )
        return *this;
    m_bytes = src.m_bytes;
    deallocate(m_data);
    if( src.m_flags )
        m_data = src.m_data;
    else
    {
        m_data = allocate(m_bytes);
        memcpy(m_data,src.m_data,m_bytes);
    }
    m_flags = src.m_flags;
    src.m_data = nullptr;
    src.m_bytes = 0;
    src.m_flags = 0;
    return *this;
}

//
__device__ unsigned int dstring::size() const
{
    return m_bytes;
}

__device__ unsigned int dstring::length() const
{
    return m_bytes;
}

__device__ unsigned int dstring::chars_count() const
{
    return chars_in_string(m_data,m_bytes);
}

__device__ char* dstring::data()
{
    return m_data;
}

__device__ const char* dstring::data() const
{
    return m_data;
}

__device__ bool dstring::empty() const
{
    return m_bytes == 0;
}

__device__ bool dstring::is_null() const
{
    return m_data == nullptr;
}

// the custom iterator knows about UTF8 encoding
__device__ dstring::iterator::iterator(const dstring& str, unsigned int initPos)
    : p(0), cpos(0), offset(0)
{
    p = str.data();
    cpos = initPos;
    offset = str.byte_offset_for(cpos);
}

__device__ dstring::iterator::iterator(const dstring::iterator& mit)
    : p(mit.p), cpos(mit.cpos), offset(mit.offset)
{}

__device__ dstring::iterator& dstring::iterator::operator++()
{
    offset += bytes_in_char_byte((BYTE)p[offset]);
    ++cpos;
    return *this;
}

// what is the int parm for?
__device__ dstring::iterator dstring::iterator::operator++(int)
{
    iterator tmp(*this);
    operator++();
    return tmp;
}

__device__ bool dstring::iterator::operator==(const dstring::iterator& rhs) const
{
    return (p == rhs.p) && (cpos == rhs.cpos);
}

__device__ bool dstring::iterator::operator!=(const dstring::iterator& rhs) const
{
    return (p != rhs.p) || (cpos != rhs.cpos);
}

// unsigned int can hold 1-4 bytes for the UTF8 char
__device__ Char dstring::iterator::operator*() const
{
    Char chr = 0;
    char_to_Char(p + offset, chr);
    return chr;
}

__device__ unsigned int dstring::iterator::position() const
{
    return cpos;
}

__device__ unsigned int dstring::iterator::byte_offset() const
{
    return offset;
}

__device__ dstring::iterator dstring::begin() const
{
    return iterator(*this, 0);
}

__device__ dstring::iterator dstring::end() const
{
    return iterator(*this, chars_count());
}

__device__ Char dstring::at(unsigned int pos) const
{
    unsigned int offset = byte_offset_for(pos);
    if(offset >= m_bytes)
        return 0;
    Char chr = 0;
    char_to_Char(data() + offset, chr);
    return chr;
}

__device__ Char dstring::operator[](unsigned int pos) const
{
    return at(pos);
}

__device__ unsigned int dstring::byte_offset_for(unsigned int pos) const
{
    unsigned int offset = 0;
    const char* sptr = m_data;
    const char* eptr = sptr + m_bytes;
    while( (pos > 0) && (sptr < eptr) )
    {
        unsigned int charbytes = bytes_in_char_byte((BYTE)*sptr++);
        if( charbytes )
            --pos;
        offset += charbytes;
    }
    return offset;
}

// 0	They compare equal
// <0	Either the value of the first character of this string that does not match is lower in the arg string,
//      or all compared characters match but the arg string is shorter.
// >0	Either the value of the first character of this string that does not match is greater in the arg string,
//      or all compared characters match but the arg string is longer.
__device__ int dstring::compare(const dstring& in) const
{
    return compare(in.data(), in.size());
}

__device__ int dstring::compare(const char* data, unsigned int bytes) const
{
    const unsigned char* ptr1 = reinterpret_cast<const unsigned char*>(this->data());
    if(!ptr1)
        return -1;
    const unsigned char* ptr2 = reinterpret_cast<const unsigned char*>(data);
    if(!ptr2)
        return 1;
    unsigned int len1 = size();
    unsigned int len2 = bytes;
    unsigned int idx;
    for(idx = 0; (idx < len1) && (idx < len2); ++idx)
    {
        if(*ptr1 != *ptr2)
            return (int)*ptr1 - (int)*ptr2;
        ptr1++;
        ptr2++;
    }
    if(idx < len1)
        return 1;
    if(idx < len2)
        return -1;
    return 0;
}

__device__ bool dstring::operator==(const dstring& rhs) const
{
    return compare(rhs) == 0;
}

__device__ bool dstring::operator!=(const dstring& rhs) const
{
    return compare(rhs) != 0;
}

__device__ bool dstring::operator<(const dstring& rhs) const
{
    return compare(rhs) < 0;
}

__device__ bool dstring::operator>(const dstring& rhs) const
{
    return compare(rhs) > 0;
}

__device__ bool dstring::operator<=(const dstring& rhs) const
{
    int rc = compare(rhs);
    return (rc == 0) || (rc < 0);
}

__device__ bool dstring::operator>=(const dstring& rhs) const
{
    int rc = compare(rhs);
    return (rc == 0) || (rc > 0);
}

__device__ int dstring::find(const dstring& str, unsigned int pos, int count) const
{
    return find(str.data(), str.size(), pos, count);
}

__device__ int dstring::find(const char* str, unsigned int bytes, unsigned int pos, int count) const
{
    char* sptr = (char*)data();
    if(!str || !bytes)
        return -1;
    unsigned int nchars = chars_count();
    if(count < 0)
        count = nchars;
    int end = (int)pos + count;
    if(end < 0 || end > nchars)
       end = nchars;
    int spos = (int)byte_offset_for(pos);
    int epos = (int)byte_offset_for((unsigned int)end);

    int len2 = (int)bytes;
    int len1 = (epos - spos) - (int)len2 + 1;

    char* ptr1 = sptr + spos;
    char* ptr2 = (char*)str;
    for(int idx=0; idx < len1; ++idx)
    {
        bool match = true;
        for( int jdx=0; match && (jdx < len2); ++jdx )
            match = (ptr1[jdx] == ptr2[jdx]);
        if( match )
            return (int)char_offset(idx+spos);
        ptr1++;
    }
    return -1;
}

// maybe get rid of this one
__device__ int dstring::find(Char chr, unsigned int pos, int count) const
{
    unsigned int sz = size();
    unsigned int nchars = chars_count();
    if(count < 0)
        count = nchars;
    int end = (int)pos + count;
    if(end < 0 || end > nchars)
        end = nchars;
    if(pos > end || chr == 0 || sz == 0)
        return -1;
    int spos = (int)byte_offset_for(pos);
    int epos = (int)byte_offset_for((unsigned int)end);
    //
    int chsz = (int)bytes_in_char(chr);
    char* sptr = (char*)data();
    char* ptr = sptr + spos;
    int len = (epos - spos) - chsz;
    for(int idx = 0; idx <= len; ++idx)
    {
        Char ch = 0;
        char_to_Char(ptr++, ch);
        if(chr == ch)
            return (int)chars_in_string(sptr, idx + spos);
    }
    return -1;
}

__device__ int dstring::rfind(const dstring& str, unsigned int pos, int count) const
{
    return rfind(str.data(), str.size(), pos, count);
}

__device__ int dstring::rfind(const char* str, unsigned int bytes, unsigned int pos, int count) const
{
    char* sptr = (char*)data();
    if(!str || !bytes)
        return -1;
    unsigned int sz = size();
    unsigned int nchars = chars_count();
    int end = (int)pos + count;
    if(end < 0 || end > nchars)
        end = nchars;
    int spos = (int)byte_offset_for(pos);
    int epos = (int)byte_offset_for(end);

    int len2 = (int)bytes;
    int len1 = (epos - spos) - len2 + 1;

    char* ptr1 = sptr + epos - len2;
    char* ptr2 = (char*)str;
    for(int idx=0; idx < len1; ++idx)
    {
        bool match = true;
        for(int jdx=0; match && (jdx < len2); ++jdx)
            match = (ptr1[jdx] == ptr2[jdx]);
        if(match)
            return (int)char_offset(epos - len2 - idx);
        ptr1--; // go backwards
    }
    return -1;
}

__device__ int dstring::rfind(Char chr, unsigned int pos, int count) const
{
    unsigned int sz = size();
    unsigned int nchars = chars_count();
    if(count < 0)
        count = nchars;
    int end = (int)pos + count;
    if(end < 0 || end > nchars)
        end = nchars;
    if(pos > end || chr == 0 || sz == 0)
        return -1;
    int spos = (int)byte_offset_for(pos);
    int epos = (int)byte_offset_for(end);

    int chsz = (int)bytes_in_char(chr);
    char* sptr = (char*)data();
    char* ptr = sptr + epos - 1;
    int len = (epos - spos) - chsz;
    for(int idx = 0; idx < len; ++idx)
    {
        Char ch = 0;
        char_to_Char(ptr--, ch);
        if(chr == ch)
            return (int)chars_in_string(sptr, epos - idx - 1);
    }
    return -1;
}

// this is useful since operator+= takes only 1 argument
__device__ dstring& dstring::append(const char* str, unsigned int bytes)
{
    unsigned int sz = size();
    unsigned int nsz = sz + bytes;
    char* ndata = allocate(nsz);
    memcpy(ndata, data(), sz);
    memcpy(ndata+sz, str, bytes);
    deallocate(m_data);
    m_data = ndata;
    m_bytes = nsz;
    m_flags = 0;
    return *this;
}

__device__ dstring& dstring::append(const char* str)
{
    return append(str,string_length(str));
}

__device__ dstring& dstring::append(Char chr, unsigned int count)
{
    unsigned int sz = size();
    unsigned int bytes = bytes_in_char(chr);
    unsigned int nsz = sz + (count * bytes);
    char* ndata = allocate(nsz);
    memcpy(ndata, data(), sz);
    char* sptr = ndata + sz;
    for(unsigned idx = 0; idx < count; ++idx)
    {
        Char_to_char(chr, sptr);
        sptr += bytes;
    }
    deallocate(m_data);
    m_data = ndata;
    m_bytes = nsz;
    m_flags = 0;
    return *this;
}

__device__ dstring& dstring::append(const dstring& in)
{
    return append(in.data(), in.size());
}


__device__ dstring& dstring::operator+=(const dstring& in)
{
    return append(in);
}

__device__ dstring& dstring::operator+=(Char chr)
{
    return append(chr);
}

// operators can only take one argument
__device__ dstring& dstring::operator+=(const char* str)
{
    return append(str);
}

__device__ dstring& dstring::insert(unsigned int pos, const char* str, unsigned int bytes )
{
    unsigned int sz = size();
    unsigned int spos = byte_offset_for(pos);
    if(spos > sz)
        return *this;
    unsigned int nsz = sz + bytes;
    char* ndata = allocate(nsz);
    memcpy(ndata, data(), spos);                    // left
    memcpy(ndata+spos, str, bytes);                 // middle
    memcpy(ndata+spos+bytes, data()+spos, sz-spos); // right
    deallocate(m_data);
    m_data = ndata;
    m_bytes = nsz;
    m_flags = 0;
    return *this;
}

__device__ dstring& dstring::insert(unsigned int pos, const char* str )
{
    return insert(pos,str,string_length(str));
}

__device__ dstring& dstring::insert(unsigned int pos, dstring& in)
{
    return insert(pos, in.data(), in.size());
}

__device__ dstring& dstring::insert(unsigned int pos, unsigned int count, Char chr)
{
    unsigned int sz = size();
    unsigned int spos = byte_offset_for(pos);
    if(spos > sz)
        return *this;
    unsigned int bytes = bytes_in_char(chr) * count;
    unsigned int nsz = sz + bytes;
    char* ndata = allocate(nsz);
    memcpy(ndata, data(), spos); // left
    char* sptr = ndata + spos;
    for(unsigned idx = 0; idx < count; ++idx)
    {
        Char_to_char(chr, sptr); // middle
        sptr += bytes;
    }
    memcpy(sptr, data()+spos, sz-spos); // right
    deallocate(m_data);
    m_data = ndata;
    m_bytes = nsz;
    m_flags = 0;
    return *this;
}

// parameters are character position values
__device__ dstring dstring::substr(unsigned int pos, unsigned int length) const
{
    unsigned int spos = byte_offset_for(pos);
    unsigned int epos = byte_offset_for(pos + length);
    if( epos > size() )
        epos = size();
    if(spos >= epos)
        return dstring("",0);
    length = epos - spos; // converts length to bytes
    return dstring(data()+spos,length);
}

// replace specified section with the given string
__device__ dstring& dstring::replace(unsigned int pos, unsigned int length, const char* str, unsigned int bytes)
{
    unsigned int spos = byte_offset_for(pos);
    unsigned int epos = byte_offset_for(pos + length);
    unsigned int sz = size();
    if( epos > sz )
        epos = sz;
    if(spos >= epos)
        return *this;
    //
    unsigned int nsz = spos + bytes + (sz - epos);
    char* ndata = allocate(nsz);
    memcpy(ndata, data(), spos);                    // left
    memcpy(ndata+spos, str, bytes);                 // middle
    memcpy(ndata+spos+bytes, data()+epos, sz-epos); // right
    deallocate(m_data);
    m_data = ndata;
    m_bytes = nsz;
    m_flags = 0;
    return *this;
}

__device__ dstring& dstring::replace(unsigned int pos, unsigned int length, const char* str)
{
    return replace(pos,length,str,string_length(str));
}

__device__ dstring& dstring::replace(unsigned int pos, unsigned int length, const dstring& in)
{
    return replace(pos, length, in.data(), in.size());
}

__device__ dstring& dstring::replace(unsigned int pos, unsigned int length, unsigned int count, Char chr)
{
    unsigned int spos = byte_offset_for(pos);
    unsigned int epos = byte_offset_for(pos + length);
    unsigned int sz = size();
    if( epos > sz )
        epos = sz;
    if(spos >= epos)
        return *this;
    //
    unsigned int bytes = bytes_in_char(chr);
    unsigned int nsz = spos + (bytes*count) + (sz - epos);
    char* ndata = allocate(nsz);
    memcpy(ndata, data(), spos); // left
    char* sptr = ndata + spos;
    for(unsigned idx = 0; idx < count; ++idx)
    {
        Char_to_char(chr, sptr); // middle
        sptr += bytes;
    }
    memcpy(sptr, data()+epos, sz-epos); // right
    deallocate(m_data);
    m_data = ndata;
    m_bytes = nsz;
    m_flags = 0;
    return *this;
}

__device__ unsigned int dstring::split(const char* delim, int count, dstring* strs) const
{
    const char* sptr = data();
    unsigned int sz = size();
    if(sz == 0)
    {
        if(strs && count)
            strs[0] = *this;
        return 1;
    }

    unsigned int bytes = string_length(delim);
    unsigned int delimCount = 0;
    int pos = find(delim, bytes);
    while(pos >= 0)
    {
        ++delimCount;
        pos = find(delim, bytes, (unsigned int)pos + bytes);
    }

    unsigned int strsCount = delimCount + 1;
    unsigned int rtn = strsCount;
    if((count > 0) && (rtn > count))
        rtn = count;
    if(!strs)
        return rtn;
    //
    if(strsCount < count)
        count = strsCount;
    //
    unsigned int dchars = (bytes ? chars_in_string(delim,bytes) : 1);
    unsigned int nchars = chars_count();
    unsigned int spos = 0, sidx = 0;
    int epos = find(delim, bytes);
    while(epos >= 0)
    {
        if(sidx >= (count - 1)) // add this to the while clause
            break;
        int len = (unsigned int)epos - spos;
        strs[sidx++] = substr(spos, len);
        spos = epos + dchars;
        epos = find(delim, bytes, spos);
    }
    if((spos <= nchars) && (sidx < count))
        strs[sidx] = substr(spos, nchars - spos);
    //
    return rtn;
}


__device__ unsigned int dstring::rsplit(const char* delim, int count, dstring* strs) const
{
    const char* sptr = data();
    unsigned int sz = size();
    if(sz == 0)
    {
        if(strs && count)
            strs[0] = *this;
        return 1;
    }

    unsigned int bytes = string_length(delim);
    unsigned int delimCount = 0;
    int pos = find(delim, bytes);
    while(pos >= 0)
    {
        ++delimCount;
        pos = find(delim, bytes, (unsigned int)pos + bytes);
    }

    unsigned int strsCount = delimCount + 1;
    unsigned int rtn = strsCount;
    if((count > 0) && (rtn > count))
        rtn = count;
    if(!strs)
        return rtn;
    //
    if(strsCount < count)
        count = strsCount;
    //
    unsigned int dchars = (bytes ? chars_in_string(delim,bytes) : 1);
    int epos = (int)chars_count(); // end pos is not inclusive
    int sidx = count - 1;          // index for strs array
    int spos = rfind(delim, bytes);
    while(spos >= 0)
    {
        if(sidx <= 0)
            break;
        //int spos = pos + (int)bytes;
        int len = epos - spos - dchars;
        strs[sidx--] = substr((unsigned int)spos+dchars, (unsigned int)len);
        epos = spos;
        spos = rfind(delim, bytes, 0, (unsigned int)epos);
    }
    if(epos >= 0)
        strs[0] = substr(0, epos);
    //
    return rtn;
}


__device__ dstring& dstring::strip(const char* tgts)
{
    if(!tgts)
        tgts = " \n\t";
    unsigned int sz = size();
    unsigned int nchars = chars_count();
    // count the leading chars
    unsigned int lcount = 0;
    char* sptr = data();      // point to beginning
    char* eptr = data() + sz; // point to the end
    while( sptr < eptr )
    {
        Char ch = 0;
        unsigned int cw = char_to_Char(sptr, ch);
        if( !has_one_of(tgts,ch) )
            break;
        sptr += cw;
        lcount += cw;
    }
    // count trailing bytes
    unsigned int rcount = 0;
    while( sptr < eptr )
    {
        while(bytes_in_char_byte((BYTE)*(--eptr)) == 0)
            ; // skip over 'extra' bytes
        Char ch = 0;
        unsigned int cw = char_to_Char(eptr, ch);
        if( !has_one_of(tgts,ch) )
            break;
        rcount += cw;
    }
    unsigned int nsz = sz - rcount - lcount;
    char* ndata = allocate(nsz);
    memcpy(ndata, sptr, nsz);
    deallocate(m_data);
    m_data = ndata;
    m_bytes = nsz;
    m_flags = 0;
    return *this;
}

__device__ dstring& dstring::lstrip(const char* tgts)
{
    if(!tgts)
        tgts = " \n\t";
    unsigned int sz = size();
    unsigned int count = 0;
    char* sptr = data();      // point to beginning
    char* eptr = data() + sz; // point to the end
    while( sptr < eptr )
    {
        Char ch = 0;
        unsigned int cw = char_to_Char(sptr, ch);
        if( !has_one_of(tgts,ch) )
            break;
        sptr += cw;
        count += cw;
    }
    unsigned int nsz = sz - count;
    char* ndata = allocate(nsz);
    memcpy(ndata, sptr, nsz);
    deallocate(m_data);
    m_data = ndata;
    m_bytes = nsz;
    m_flags = 0;
    return *this;
}

__device__ dstring& dstring::rstrip(const char* tgts)
{
    if(!tgts)
        tgts = " \n\t";
    unsigned int sz = size();
    unsigned int count = 0;
    char* sptr = data();      // point to beginning
    char* eptr = data() + sz; // point to the end
    while( sptr < eptr )
    {
        while(bytes_in_char_byte((BYTE)*(--eptr)) == 0)
            ; // skip over 'extra' bytes
        Char ch = 0;
        unsigned int cw = char_to_Char(eptr, ch);
        if( !has_one_of(tgts,ch) )
            break;
        count += cw;
    }
    unsigned int nsz = sz - count;
    char* ndata = allocate(nsz);
    memcpy(ndata, sptr, nsz);
    deallocate(m_data);
    m_data = ndata;
    m_bytes = nsz;
    m_flags = 0;
    return *this;
}


// ascii only right now
__device__ dstring& dstring::upper()
{
    char* sptr = data();
    char* eptr = data() + size();
    while( sptr < eptr )
    {
        char ch = *sptr;
        if( ch >= 'a' && ch <= 'z' )
            ch = ch - 'a' + 'A';
        *sptr++ = ch;
    }
    return *this;
}

// ascii only right now
__device__ dstring& dstring::lower()
{
    char* sptr = data();
    char* eptr = data() + size();
    while( sptr < eptr )
    {
        char ch = *sptr;
        if( ch >= 'A' && ch <= 'Z' )
            ch = ch - 'A' + 'a';
        *sptr++ = ch;
    }
    return *this;
}

__device__ dstring& dstring::center( unsigned int width, Char fill)
{
    unsigned int nchars = chars_count();
    if( width <= nchars )
        return *this;
    unsigned int fill_size = width - nchars;
    unsigned int left = fill_size/2;
    unsigned int right = fill_size - left;
    return insert(0,left,fill).append(fill,right);
}

__device__ dstring& dstring::ljust( unsigned int width, Char fill)
{
    unsigned int nchars = chars_count();
    if( width <= nchars )
        return *this;
    return append( fill, width-nchars );
}

__device__ dstring& dstring::rjust( unsigned int width, Char fill)
{
    unsigned int nchars = chars_count();
    if( width <= nchars )
        return *this;
    return insert(0,width-nchars,fill);
}

__device__ dstring& dstring::join( const dstring* strings, unsigned int count )
{
    // compute size
    unsigned int nsz = 0;
    for( unsigned int idx=0; idx < count; ++idx )
    {
        nsz += strings[idx].size();
        nsz += size();
    }
    if( nsz > 0 )
        nsz -= size();
    if( nsz==0 )
        return *this;
    char* ndata = (char*)allocate(nsz);
    char* ptr = ndata;
    for( unsigned int idx=0; idx < count-1; ++idx )
    {
        unsigned int bytes = strings[idx].size();
        memcpy(ptr, strings[idx].data(), bytes );
        ptr += bytes;
        memcpy(ptr, data(), size());
        ptr += size();
    }
    memcpy(ptr, strings[count-1].data(), strings[count-1].size() );
    deallocate(m_data);
    m_data = ndata;
    m_bytes = nsz;
    m_flags = 0;
    return *this;
}

// ascii only right now
__device__ bool dstring::is_alnum() const
{
    unsigned int sz = size();
    if( sz==0 )
        return false;
    const char* sptr = data();
    const char* eptr = sptr + sz;
    while( sptr < eptr )
    {
        char ch = *sptr++;
        if( ((ch < '0') || (ch > 'z')) ||
            ((ch > '9') && (ch < 'A')) ||
            ((ch > 'Z') && (ch < 'a')) )
            return false;
    }
    return true;
}

// ascii only right now
__device__ bool dstring::is_alpha() const
{
    unsigned int sz = size();
    if( sz==0 )
        return false;
    const char* sptr = data();
    const char* eptr = sptr + sz;
    while( sptr < eptr )
    {
        char ch = *sptr++;
        if( ((ch < 'A') || (ch > 'z')) ||
            ((ch > 'Z') && (ch < 'a')) )
            return false;
    }
    return true;
}

// ascii only right now
__device__ bool dstring::is_float() const
{
    unsigned int sz = size();
    if( sz==0 )
        return false;
    const char* sptr = data();
    const char* eptr = sptr + sz;
    while( sptr < eptr )
    {
        char ch = *sptr++;
        if( ((ch < '0') || (ch > '9')) &&
            (ch != '-') && (ch != '+') && (ch != '.') )
            return false;
    }
    return true;
}

// ascii only right now
__device__ bool dstring::is_lower() const
{
    unsigned int sz = size();
    if( sz==0 )
        return false;
    const char* sptr = data();
    const char* eptr = sptr + sz;
    while( sptr < eptr )
    {
        char ch = *sptr++;
        if( (ch >= 'A') && (ch <= 'Z') )
            return false;
    }
    return true;
}

// ascii only right now
__device__ bool dstring::is_integer() const
{
    unsigned int sz = size();
    if( sz==0 )
        return false;
    const char* sptr = data();
    const char* eptr = sptr + sz;
    while( sptr < eptr )
    {
        char ch = *sptr++;
        if( (ch < '0') || (ch > '9') )
            return false;
    }
    return true;
}

// ascii only right now
__device__ bool dstring::is_space() const
{
    unsigned int sz = size();
    if( sz==0 )
        return false;
    const char* sptr = data();
    const char* eptr = sptr + sz;
    while( sptr < eptr )
    {
        char ch = *sptr++;
        if( ch > ' ' )
            return false;
    }
    return true;
}

// ascii only right now
__device__ bool dstring::is_upper() const
{
    unsigned int sz = size();
    if( sz==0 )
        return false;
    const char* sptr = data();
    const char* eptr = sptr + sz;
    while( sptr < eptr )
    {
        char ch = *sptr++;
        if( (ch >= 'a') && (ch <= 'z') )
            return false;
    }
    return true;
}

//
__device__ bool dstring::starts_with(const char* str, unsigned int bytes) const
{
    if(bytes > size())
        return false;
    char* sptr = (char*)data();
    for(unsigned int idx = 0; idx < bytes; ++idx)
    {
        if(*sptr++ != *str++)
            return false;
    }
    return true;
}

__device__ bool dstring::starts_with(const char* str) const
{
    return starts_with(str,string_length(str));
}

__device__ bool dstring::starts_with(dstring& in) const
{
    return starts_with(in.data(), in.size());
}

__device__ bool dstring::ends_with(const char* str, unsigned int bytes) const
{
    unsigned int sz = size();
    if(bytes > sz)
        return false;
    char* sptr = (char*)data() + sz - bytes; // point to the end
    for(unsigned int idx = 0; idx < bytes; ++idx)
    {
        if(*sptr++ != *str++)
            return false;
    }
    return true;
}

__device__ bool dstring::ends_with(const char* str) const
{
    return ends_with(str,string_length(str));
}

__device__ bool dstring::ends_with(dstring& str) const
{
    unsigned int sz = size();
    unsigned int bytes = str.size();
    if(bytes > sz)
        return false;
    return find(str, sz - bytes) >= 0;
}

__host__ __device__ unsigned int dstring::bytes_in_char(Char chr)
{
    unsigned int count = 1;
    // no if-statements means no divergence
    count += (int)((chr & (unsigned)0x0000FF00) > 0);
    count += (int)((chr & (unsigned)0x00FF0000) > 0);
    count += (int)((chr & (unsigned)0xFF000000) > 0);
    return count;
}

__host__ __device__ unsigned int dstring::char_to_Char(const char* pSrc, Char &chr)
{
    unsigned int chwidth = bytes_in_char_byte((BYTE)*pSrc);
    chr = (Char)(*pSrc++) & 0xFF;
    if(chwidth > 1)
    {
        chr = chr << 8;
        chr |= ((Char)(*pSrc++) & 0xFF); // << 8;
        if(chwidth > 2)
        {
            chr = chr << 8;
            chr |= ((Char)(*pSrc++) & 0xFF); // << 16;
            if(chwidth > 3)
            {
                chr = chr << 8;
                chr |= ((Char)(*pSrc++) & 0xFF); // << 24;
            }
        }
    }
    return chwidth;
}

__host__ __device__ unsigned int dstring::Char_to_char(Char chr, char* dst)
{
    unsigned int chwidth = bytes_in_char(chr);
    for(unsigned int idx = 0; idx < chwidth; ++idx)
    {
        dst[chwidth - idx - 1] = (char)chr & 0xFF;
        chr = chr >> 8;
    }
    return chwidth;
}

// counts the number of character in the first bytes of the given char array
__host__ __device__ unsigned int dstring::chars_in_string(const char* str, unsigned int bytes)
{
    if( (str==0) || (bytes==0) )
        return 0;
    //
    unsigned int nchars = 0;
    for(unsigned int idx = 0; idx < bytes; ++idx)
        nchars += (unsigned int)(((BYTE)str[idx] & 0xC0) != 0x80);
    return nchars;
}

__device__ unsigned int dstring::char_offset(unsigned int bytepos) const
{
    return chars_in_string(data(), bytepos);
}