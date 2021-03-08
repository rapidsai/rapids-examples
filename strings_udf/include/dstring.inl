/*
 * Copyright (c) 2020, NVIDIA CORPORATION.
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

namespace {

using BYTE = uint8_t;

// utility for methods that allow for multiple delimiters (e.g. strip)
__device__ inline static bool has_one_of(char const* tgts, char_utf8 chr)
{
  char_utf8 tchr = 0;
  size_type cw   = utf8::to_char_utf8(tgts, tchr);
  while (tchr) {
    if (tchr == chr) return true;
    tgts += cw;
    cw = utf8::to_char_utf8(tgts, tchr);
  }
  return false;
}

__device__ inline static size_type string_length(char const* str)
{
  if (!str) return 0;
  size_type bytes = 0;
  while (*str++) ++bytes;
  return bytes;
}

}  // namespace

__device__ inline char* dstring::allocate(size_type bytes)
{
  char* data = (char*)malloc(bytes + 1);
  if (!data) printf("!out of malloc data! mem-size request: %d bytes\n", bytes);
  data[bytes] = 0;
  return data;
}

__device__ inline void dstring::deallocate(char* data)
{
  if (data) free(data);
}

__device__ inline dstring::dstring() : m_data(nullptr), m_bytes(0), m_size(0) {}

__device__ inline dstring::dstring(char const* data, size_type bytes)
  : m_bytes(bytes), m_size(bytes)
{
  m_data = allocate(bytes);
  memcpy(m_data, data, bytes);
}

__device__ inline dstring::dstring(char const* data)
{
  m_bytes = m_size = string_length(data);
  m_data           = allocate(m_size);
  memcpy(m_data, data, m_bytes);
}

__device__ inline dstring::dstring(dstring const& src)
{
  m_bytes = src.m_bytes;
  m_size  = m_bytes;
  m_data  = allocate(m_size);
  memcpy(m_data, src.m_data, m_size);
}

__device__ inline dstring::dstring(dstring&& src)
{
  m_bytes     = src.m_bytes;
  m_size      = src.m_size;
  m_data      = src.m_data;
  src.m_bytes = 0;
  src.m_size  = 0;
  src.m_data  = nullptr;
}

__device__ inline dstring::dstring(dstring_view const& str)
  : m_bytes(str.size_bytes()), m_size(str.size_bytes())
{
  m_data = allocate(m_size);
  memcpy(m_data, str.data(), m_bytes);
}

__device__ inline dstring::~dstring() { deallocate(m_data); }

__device__ inline dstring& dstring::operator=(dstring const& src)
{
  if (&src == this) return *this;
  m_bytes = src.m_bytes;
  deallocate(m_data);
  m_size = m_bytes;
  m_data = allocate(m_size);
  memcpy(m_data, src.m_data, m_size);
  return *this;
}

__device__ inline dstring& dstring::operator=(dstring_view const& src)
{
  m_size         = src.size_bytes();
  char* new_data = allocate(m_size);
  memcpy(new_data, src.data(), m_size);
  deallocate(m_data);
  m_data  = new_data;
  m_bytes = m_size;
  return *this;
}

__device__ inline dstring& dstring::operator=(dstring&& src)
{
  if (&src == this) return *this;
  m_bytes = src.m_bytes;
  deallocate(m_data);
  m_size = m_bytes;
  m_data = allocate(m_size);
  memcpy(m_data, src.m_data, m_size);
  src.m_data  = nullptr;
  src.m_bytes = 0;
  src.m_size  = 0;
  return *this;
}

//
__device__ inline size_type dstring::size_bytes() const { return m_bytes; }

__device__ inline size_type dstring::length() const
{
  return utf8::characters_in_string(m_data, m_bytes);
}

__device__ inline char* dstring::data() { return m_data; }

__device__ inline char const* dstring::data() const { return m_data; }

__device__ inline bool dstring::empty() const { return m_bytes == 0; }

__device__ inline bool dstring::is_null() const { return m_data == nullptr; }

__device__ inline dstring_view::const_iterator dstring::begin() const
{
  return dstring_view::const_iterator(*this, 0);
}

__device__ inline dstring_view::const_iterator dstring::end() const
{
  return dstring_view::const_iterator(*this, length());
}

__device__ inline char_utf8 dstring::at(size_type pos) const
{
  size_type offset = byte_offset(pos);
  if (offset >= m_bytes) return 0;
  char_utf8 chr = 0;
  utf8::to_char_utf8(data() + offset, chr);
  return chr;
}

__device__ inline char_utf8 dstring::operator[](size_type pos) const { return at(pos); }

__device__ inline size_type dstring::byte_offset(size_type pos) const
{
  size_type offset = 0;
  char const* sptr = m_data;
  char const* eptr = sptr + m_bytes;
  while ((pos > 0) && (sptr < eptr)) {
    size_type charbytes = utf8::bytes_in_utf8_byte((BYTE)*sptr++);
    if (charbytes) --pos;
    offset += charbytes;
  }
  return offset;
}

__device__ inline int dstring::compare(dstring_view const& in) const
{
  return compare(in.data(), in.size_bytes());
}

__device__ inline int dstring::compare(char const* data, size_type nbytes) const
{
  const unsigned char* ptr1 = reinterpret_cast<const unsigned char*>(this->data());
  if (!ptr1) return -1;
  const unsigned char* ptr2 = reinterpret_cast<const unsigned char*>(data);
  if (!ptr2) return 1;
  size_type len1 = size_bytes();
  size_type len2 = nbytes;
  size_type idx;
  for (idx = 0; (idx < len1) && (idx < len2); ++idx) {
    if (*ptr1 != *ptr2) return (int)*ptr1 - (int)*ptr2;
    ptr1++;
    ptr2++;
  }
  if (idx < len1) return 1;
  if (idx < len2) return -1;
  return 0;
}

__device__ inline bool dstring::operator==(dstring_view const& rhs) const
{
  return compare(rhs) == 0;
}

__device__ inline bool dstring::operator!=(dstring_view const& rhs) const
{
  return compare(rhs) != 0;
}

__device__ inline bool dstring::operator<(dstring_view const& rhs) const
{
  return compare(rhs) < 0;
}

__device__ inline bool dstring::operator>(dstring_view const& rhs) const
{
  return compare(rhs) > 0;
}

__device__ inline bool dstring::operator<=(dstring_view const& rhs) const
{
  int rc = compare(rhs);
  return (rc == 0) || (rc < 0);
}

__device__ inline bool dstring::operator>=(dstring_view const& rhs) const
{
  int rc = compare(rhs);
  return (rc == 0) || (rc > 0);
}

__device__ inline int dstring::find(dstring_view const& str, size_type pos, int count) const
{
  return find(str.data(), str.size_bytes(), pos, count);
}

__device__ inline int dstring::find(char const* str,
                                    size_type bytes,
                                    size_type pos,
                                    int count) const
{
  char* sptr = (char*)data();
  if (!str || !bytes) return -1;
  size_type nchars = length();
  if (count < 0) count = nchars;
  int end = (int)pos + count;
  if (end < 0 || end > nchars) end = nchars;
  int spos = (int)byte_offset(pos);
  int epos = (int)byte_offset((size_type)end);

  int len2 = (int)bytes;
  int len1 = (epos - spos) - (int)len2 + 1;

  char* ptr1 = sptr + spos;
  char* ptr2 = (char*)str;
  for (int idx = 0; idx < len1; ++idx) {
    bool match = true;
    for (int jdx = 0; match && (jdx < len2); ++jdx) match = (ptr1[jdx] == ptr2[jdx]);
    if (match) return (int)char_offset(idx + spos);
    ptr1++;
  }
  return -1;
}

// maybe get rid of this one
__device__ inline int dstring::find(char_utf8 chr, size_type pos, int count) const
{
  size_type sz     = size_bytes();
  size_type nchars = length();
  if (count < 0) count = nchars;
  int end = (int)pos + count;
  if (end < 0 || end > nchars) end = nchars;
  if (pos > end || chr == 0 || sz == 0) return -1;
  int spos = (int)byte_offset(pos);
  int epos = (int)byte_offset((size_type)end);
  //
  int chsz   = (int)utf8::bytes_in_char_utf8(chr);
  char* sptr = (char*)data();
  char* ptr  = sptr + spos;
  int len    = (epos - spos) - chsz;
  for (int idx = 0; idx <= len; ++idx) {
    char_utf8 ch = 0;
    utf8::to_char_utf8(ptr++, ch);
    if (chr == ch) return (int)utf8::characters_in_string(sptr, idx + spos);
  }
  return -1;
}

__device__ inline int dstring::rfind(dstring_view const& str, size_type pos, int count) const
{
  return rfind(str.data(), str.size_bytes(), pos, count);
}

__device__ inline int dstring::rfind(char const* str,
                                     size_type nbytes,
                                     size_type pos,
                                     int count) const
{
  char* sptr = (char*)data();
  if (!str || !nbytes) return -1;
  size_type sz     = size_bytes();
  size_type nchars = length();
  int end          = (int)pos + count;
  if (end < 0 || end > nchars) end = nchars;
  int spos = (int)byte_offset(pos);
  int epos = (int)byte_offset(end);

  int len2 = (int)nbytes;
  int len1 = (epos - spos) - len2 + 1;

  char* ptr1 = sptr + epos - len2;
  char* ptr2 = (char*)str;
  for (int idx = 0; idx < len1; ++idx) {
    bool match = true;
    for (int jdx = 0; match && (jdx < len2); ++jdx) match = (ptr1[jdx] == ptr2[jdx]);
    if (match) return (int)char_offset(epos - len2 - idx);
    ptr1--;  // go backwards
  }
  return -1;
}

__device__ inline int dstring::rfind(char_utf8 chr, size_type pos, int count) const
{
  size_type sz     = size_bytes();
  size_type nchars = length();
  if (count < 0) count = nchars;
  int end = (int)pos + count;
  if (end < 0 || end > nchars) end = nchars;
  if (pos > end || chr == 0 || sz == 0) return -1;
  int spos = (int)byte_offset(pos);
  int epos = (int)byte_offset(end);

  int chsz   = (int)utf8::bytes_in_char_utf8(chr);
  char* sptr = (char*)data();
  char* ptr  = sptr + epos - 1;
  int len    = (epos - spos) - chsz;
  for (int idx = 0; idx < len; ++idx) {
    char_utf8 ch = 0;
    utf8::to_char_utf8(ptr--, ch);
    if (chr == ch) return (int)utf8::characters_in_string(sptr, epos - idx - 1);
  }
  return -1;
}

// this is useful since operator+= takes only 1 argument
__device__ inline dstring& dstring::append(char const* str, size_type in_bytes)
{
  size_type nbytes = size_bytes() + in_bytes;
  if (nbytes > m_size) {
    m_size         = 2 * nbytes;
    char* new_data = allocate(m_size);
    memcpy(new_data, m_data, size_bytes());
    deallocate(m_data);
    m_data = new_data;
  }
  memcpy(m_data + size_bytes(), str, in_bytes);
  m_bytes = nbytes;
  return *this;
}

__device__ inline dstring& dstring::append(char const* str)
{
  return append(str, string_length(str));
}

__device__ inline dstring& dstring::append(char_utf8 chr, size_type count)
{
  size_type char_bytes = utf8::bytes_in_char_utf8(chr);
  size_type nbytes     = size_bytes() + (count * char_bytes);
  if (nbytes > m_size) {
    m_size         = 2 * nbytes;
    char* new_data = allocate(m_size);
    memcpy(new_data, m_data, size_bytes());
    deallocate(m_data);
    m_data = new_data;
  }
  char* out_ptr = m_data + size_bytes();
  for (unsigned idx = 0; idx < count; ++idx) {
    utf8::from_char_utf8(chr, out_ptr);
    out_ptr += char_bytes;
  }
  m_bytes = nbytes;
  return *this;
}

__device__ inline dstring& dstring::append(dstring_view const& in)
{
  return append(in.data(), in.size_bytes());
}

__device__ inline dstring& dstring::operator+=(dstring_view const& in) { return append(in); }

__device__ inline dstring& dstring::operator+=(char_utf8 chr) { return append(chr); }

// operators can only take one argument
__device__ inline dstring& dstring::operator+=(char const* str) { return append(str); }

__device__ inline dstring& dstring::insert(size_type pos, char const* str, size_type in_bytes)
{
  size_type spos = byte_offset(pos);
  if (spos > size_bytes()) return *this;
  size_type nbytes = size_bytes() + in_bytes;
  if (nbytes > m_size) {
    m_size         = 2 * nbytes;
    char* new_data = allocate(m_size);
    memcpy(new_data, m_data, size_bytes());
    deallocate(m_data);
    m_data = new_data;
  }
  auto epos = nbytes;
  while (epos > (spos + in_bytes)) {
    --epos;
    m_data[epos] = m_data[epos - in_bytes];
  }
  memcpy(m_data + spos, str, in_bytes);
  m_bytes = nbytes;
  return *this;
}

__device__ inline dstring& dstring::insert(size_type pos, char const* str)
{
  return insert(pos, str, string_length(str));
}

__device__ inline dstring& dstring::insert(size_type pos, dstring_view const& in)
{
  return insert(pos, in.data(), in.size_bytes());
}

__device__ inline dstring& dstring::insert(size_type pos, size_type count, char_utf8 chr)
{
  size_type spos = byte_offset(pos);
  if (spos > size_bytes()) return *this;
  auto char_bytes  = utf8::bytes_in_char_utf8(chr);
  auto in_bytes    = char_bytes * count;
  size_type nbytes = size_bytes() + in_bytes;
  if (nbytes > m_size) {
    m_size         = 2 * nbytes;
    char* new_data = allocate(m_size);
    memcpy(new_data, m_data, size_bytes());
    deallocate(m_data);
    m_data = new_data;
  }
  auto epos = nbytes;
  while (epos > (spos + in_bytes)) {
    --epos;
    m_data[epos] = m_data[epos - in_bytes];
  }
  char* out_ptr = m_data + spos;
  for (unsigned idx = 0; idx < count; ++idx) {
    utf8::from_char_utf8(chr, out_ptr);  // middle
    out_ptr += char_bytes;
  }
  m_bytes = nbytes;
  return *this;
}

// parameters are character position values
__device__ inline dstring dstring::substr(size_type pos, size_type length) const
{
  size_type spos = byte_offset(pos);
  size_type epos = byte_offset(pos + length);
  if (epos > size_bytes()) epos = size_bytes();
  if (spos >= epos) return dstring("", 0);
  length = epos - spos;  // converts length to bytes
  return dstring(data() + spos, length);
}

// replace specified section with the given string
__device__ inline dstring& dstring::replace(size_type pos,
                                            size_type length,
                                            char const* str,
                                            size_type in_bytes)
{
  size_type spos = byte_offset(pos);
  size_type epos = byte_offset(pos + length);
  size_type sz   = size_bytes();
  if (epos > sz) epos = sz;
  if (spos >= epos) return *this;
  //
  size_type nbytes = spos + in_bytes + (size_bytes() - epos);
  if (nbytes > m_size) {
    m_size         = 2 * nbytes;
    char* new_data = allocate(m_size);
    memcpy(new_data, m_data, size_bytes());
    deallocate(m_data);
    m_data = new_data;
  }
  auto cpos = nbytes;
  while (cpos > (epos + in_bytes)) {
    --cpos;
    m_data[cpos] = m_data[cpos - in_bytes];
  }
  memcpy(m_data + spos, str, in_bytes);
  m_bytes = nbytes;
  return *this;
}

__device__ inline dstring& dstring::replace(size_type pos, size_type length, char const* str)
{
  return replace(pos, length, str, string_length(str));
}

__device__ inline dstring& dstring::replace(size_type pos, size_type length, dstring_view const& in)
{
  return replace(pos, length, in.data(), in.size_bytes());
}

__device__ inline dstring& dstring::replace(size_type pos,
                                            size_type length,
                                            size_type count,
                                            char_utf8 chr)
{
  size_type spos = byte_offset(pos);
  size_type epos = byte_offset(pos + length);
  size_type sz   = size_bytes();
  if (epos > sz) epos = sz;
  if (spos >= epos) return *this;
  //
  auto const char_bytes = utf8::bytes_in_char_utf8(chr);
  auto const in_bytes   = char_bytes * count;
  size_type nbytes      = spos + in_bytes + (size_bytes() - epos);
  if (nbytes > m_size) {
    m_size         = 2 * nbytes;
    char* new_data = allocate(m_size);
    memcpy(new_data, m_data, size_bytes());
    deallocate(m_data);
    new_data = m_data;
  }
  auto cpos = nbytes;
  while (cpos > (epos + in_bytes)) {
    --cpos;
    m_data[cpos] = m_data[cpos - in_bytes];
  }
  char* out_ptr = m_data + spos;
  for (unsigned idx = 0; idx < count; ++idx) {
    utf8::from_char_utf8(chr, out_ptr);
    out_ptr += char_bytes;
  }
  m_bytes = nbytes;
  return *this;
}

__device__ inline size_type dstring::split(char const* delim, int count, dstring* strs) const
{
  char const* sptr = data();
  size_type sz     = size_bytes();
  if (sz == 0) {
    if (strs && count) strs[0] = *this;
    return 1;
  }

  size_type nbytes     = string_length(delim);
  size_type delimCount = 0;
  int pos              = find(delim, nbytes);
  while (pos >= 0) {
    ++delimCount;
    pos = find(delim, nbytes, (size_type)pos + nbytes);
  }

  size_type strsCount = delimCount + 1;
  size_type rtn       = strsCount;
  if ((count > 0) && (rtn > count)) rtn = count;
  if (!strs) return rtn;
  //
  if (strsCount < count) count = strsCount;
  //
  size_type dchars = (nbytes ? utf8::characters_in_string(delim, nbytes) : 1);
  size_type nchars = length();
  size_type spos = 0, sidx = 0;
  int epos = find(delim, nbytes);
  while (epos >= 0) {
    if (sidx >= (count - 1))  // add this to the while clause
      break;
    int len      = (size_type)epos - spos;
    strs[sidx++] = substr(spos, len);
    spos         = epos + dchars;
    epos         = find(delim, nbytes, spos);
  }
  if ((spos <= nchars) && (sidx < count)) strs[sidx] = substr(spos, nchars - spos);
  //
  return rtn;
}

__device__ inline size_type dstring::rsplit(char const* delim, int count, dstring* strs) const
{
  char const* sptr = data();
  size_type sz     = size_bytes();
  if (sz == 0) {
    if (strs && count) strs[0] = *this;
    return 1;
  }

  size_type nbytes     = string_length(delim);
  size_type delimCount = 0;
  int pos              = find(delim, nbytes);
  while (pos >= 0) {
    ++delimCount;
    pos = find(delim, nbytes, (size_type)pos + nbytes);
  }

  size_type strsCount = delimCount + 1;
  size_type rtn       = strsCount;
  if ((count > 0) && (rtn > count)) rtn = count;
  if (!strs) return rtn;
  //
  if (strsCount < count) count = strsCount;
  //
  size_type dchars = (nbytes ? utf8::characters_in_string(delim, nbytes) : 1);
  int epos         = (int)length();  // end pos is not inclusive
  int sidx         = count - 1;      // index for strs array
  int spos         = rfind(delim, nbytes);
  while (spos >= 0) {
    if (sidx <= 0) break;
    // int spos = pos + (int)bytes;
    int len      = epos - spos - dchars;
    strs[sidx--] = substr((size_type)spos + dchars, (size_type)len);
    epos         = spos;
    spos         = rfind(delim, nbytes, 0, (size_type)epos);
  }
  if (epos >= 0) strs[0] = substr(0, epos);
  //
  return rtn;
}

__device__ inline dstring& dstring::strip(char const* tgts)
{
  if (!tgts) tgts = " \n\t";
  size_type nchars = length();
  // count the leading chars
  size_type lcount = 0;
  char* sptr       = data();                 // point to beginning
  char* eptr       = data() + size_bytes();  // point to the end
  while (sptr < eptr) {
    char_utf8 ch = 0;
    size_type cw = utf8::to_char_utf8(sptr, ch);
    if (!has_one_of(tgts, ch)) break;
    sptr += cw;
    lcount += cw;
  }
  // count trailing bytes
  size_type rcount = 0;
  while (sptr < eptr) {
    while (utf8::bytes_in_utf8_byte((BYTE) * (--eptr)) == 0)
      ;  // skip over 'extra' bytes
    char_utf8 ch = 0;
    size_type cw = utf8::to_char_utf8(eptr, ch);
    if (!has_one_of(tgts, ch)) break;
    rcount += cw;
  }
  size_type nbytes = size_bytes() - rcount - lcount;
  if (lcount) memcpy(m_data, sptr, nbytes);
  m_bytes = nbytes;
  return *this;
}

__device__ inline dstring& dstring::lstrip(char const* tgts)
{
  if (!tgts) tgts = " \n\t";
  size_type count = 0;
  char* sptr      = data();                 // point to beginning
  char* eptr      = data() + size_bytes();  // point to the end
  while (sptr < eptr) {
    char_utf8 ch = 0;
    size_type cw = utf8::to_char_utf8(sptr, ch);
    if (!has_one_of(tgts, ch)) break;
    sptr += cw;
    count += cw;
  }
  m_bytes = size_bytes() - count;
  if (count) memcpy(m_data, sptr, m_bytes);
  return *this;
}

__device__ inline dstring& dstring::rstrip(char const* tgts)
{
  if (!tgts) tgts = " \n\t";
  size_type count = 0;
  char* sptr      = data();                 // point to beginning
  char* eptr      = data() + size_bytes();  // point to the end
  while (sptr < eptr) {
    while (utf8::bytes_in_utf8_byte((BYTE) * (--eptr)) == 0)
      ;  // skip over 'extra' bytes
    char_utf8 ch = 0;
    size_type cw = utf8::to_char_utf8(eptr, ch);
    if (!has_one_of(tgts, ch)) break;
    count += cw;
  }
  m_bytes = size_bytes() - count;
  return *this;
}

// ascii only right now
__device__ inline dstring& dstring::upper()
{
  char* sptr = data();
  char* eptr = data() + size_bytes();
  while (sptr < eptr) {
    char ch = *sptr;
    if (ch >= 'a' && ch <= 'z') ch = ch - 'a' + 'A';
    *sptr++ = ch;
  }
  return *this;
}

// ascii only right now
__device__ inline dstring& dstring::lower()
{
  char* sptr = data();
  char* eptr = data() + size_bytes();
  while (sptr < eptr) {
    char ch = *sptr;
    if (ch >= 'A' && ch <= 'Z') ch = ch - 'A' + 'a';
    *sptr++ = ch;
  }
  return *this;
}

__device__ inline dstring& dstring::join(dstring_view const* strings, size_type count)
{
  // compute size
  size_type nbytes = 0;
  for (size_type idx = 0; idx < count; ++idx) {
    nbytes += strings[idx].size_bytes();
    nbytes += size_bytes();
  }
  if (nbytes > 0) nbytes -= size_bytes();
  if (nbytes == 0) return *this;
  if (nbytes > m_size) {
    m_size         = 2 * nbytes;
    char* new_data = (char*)allocate(m_size);
    memcpy(new_data, m_data, size_bytes());
    deallocate(m_data);
    m_data = new_data;
  }
  // move delimiter to the end
  for (unsigned idx = 0; idx < size_bytes(); ++idx) {
    m_data[nbytes - 1 - idx] = m_data[size_bytes() - 1 - idx];
  }

  char* ptr = m_data;
  for (size_type idx = 0; idx < count - 1; ++idx) {
    size_type str_bytes = strings[idx].size_bytes();
    memcpy(ptr, strings[idx].data(), str_bytes);
    ptr += str_bytes;
    memcpy(ptr, m_data + nbytes - size_bytes(), size_bytes());
    ptr += size_bytes();
  }
  memcpy(ptr, strings[count - 1].data(), strings[count - 1].size_bytes());
  m_bytes = nbytes;
  return *this;
}

__device__ inline size_type dstring::char_offset(size_type bytepos) const
{
  return utf8::characters_in_string(data(), bytepos);
}

//
__device__ inline bool starts_with(dstring_view const& dstr, char const* str, size_type nbytes)
{
  if (nbytes > dstr.size_bytes()) return false;
  auto sptr = dstr.data();
  for (size_type idx = 0; idx < nbytes; ++idx) {
    if (*sptr++ != *str++) return false;
  }
  return true;
}

__device__ inline bool starts_with(dstring_view const& dstr, char const* str)
{
  return starts_with(dstr, str, string_length(str));
}

__device__ inline bool starts_with(dstring_view const& dstr, dstring_view const& in)
{
  return starts_with(dstr, in.data(), in.size_bytes());
}

__device__ inline bool ends_with(dstring_view const& dstr, char const* str, size_type nbytes)
{
  size_type sz = dstr.size_bytes();
  if (nbytes > sz) return false;
  auto sptr = dstr.data() + sz - nbytes;  // point to the end
  for (size_type idx = 0; idx < nbytes; ++idx) {
    if (*sptr++ != *str++) return false;
  }
  return true;
}

__device__ inline bool ends_with(dstring_view const& dstr, char const* str)
{
  return ends_with(dstr, str, string_length(str));
}

__device__ inline bool ends_with(dstring_view const& dstr, dstring const& str)
{
  size_type sz     = dstr.size_bytes();
  size_type nbytes = str.size_bytes();
  if (nbytes > sz) return false;
  return dstr.find(str, sz - nbytes) >= 0;
}