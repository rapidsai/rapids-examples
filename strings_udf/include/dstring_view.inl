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

// number of characters in a string computed on-demand
// the _length member is initialized to this value as a place-holder
constexpr size_type UNK_STRING_LENGTH{-1};
// the byte-width of the characters in a string is computed on-demand
// the _char_width member is initialized to this value as a place-holder
constexpr int8_t UNK_CHAR_WIDTH{-1};
// this value is assigned to the _char_width member if the string
// contains characters of different widths
constexpr int8_t VAR_CHAR_WIDTH{0};

}  // namespace

__host__ __device__ inline dstring_view::dstring_view()
  : _data(""), _bytes(0), _length(0), _char_width(0)
{
}

__device__ inline dstring_view::dstring_view(dstring const& dstr)
  : _data(dstr.data()),
    _bytes(dstr.size_bytes()),
    _length(UNK_STRING_LENGTH),
    _char_width(UNK_CHAR_WIDTH)
{
}

__host__ __device__ inline dstring_view::dstring_view(const char* data, size_type bytes)
  : _data(data), _bytes(bytes), _length(UNK_STRING_LENGTH), _char_width(UNK_CHAR_WIDTH)
{
}

//
__host__ __device__ inline size_type dstring_view::size_bytes() const { return _bytes; }

__device__ inline size_type dstring_view::length() const
{
  if (_length == UNK_STRING_LENGTH) _length = utf8::characters_in_string(_data, _bytes);
  if (_length && (_char_width == UNK_CHAR_WIDTH)) {
    BYTE const* bytes = reinterpret_cast<BYTE const*>(data());
    _char_width       = utf8::bytes_in_utf8_byte(*bytes);
    // see if they are all the same width
    for (size_type idx = 0; idx < size_bytes(); ++idx) {
      if (utf8::bytes_in_utf8_byte(bytes[idx]) != _char_width) {
        _char_width = VAR_CHAR_WIDTH;
        break;
      }
    }
  }
  return _length;
}

__host__ __device__ inline const char* dstring_view::data() const { return _data; }

__host__ __device__ inline bool dstring_view::empty() const { return _bytes == 0; }

// this custom iterator knows about UTF8 encoding
__device__ inline dstring_view::const_iterator::const_iterator(dstring_view const& str,
                                                               size_type pos)
  : p{str.data()}, bytes{str.size_bytes()}, char_pos{pos}, byte_pos{str.byte_offset(pos)}
{
}

__device__ inline dstring_view::const_iterator& dstring_view::const_iterator::operator++()
{
  if (byte_pos < bytes) byte_pos += utf8::bytes_in_utf8_byte((BYTE)p[byte_pos]);
  ++char_pos;
  return *this;
}

__device__ inline dstring_view::const_iterator dstring_view::const_iterator::operator++(int)
{
  dstring_view::const_iterator tmp(*this);
  operator++();
  return tmp;
}

__device__ inline dstring_view::const_iterator dstring_view::const_iterator::operator+(
  dstring_view::const_iterator::difference_type offset)
{
  const_iterator tmp(*this);
  size_type adjust = abs(offset);
  while (adjust-- > 0) offset > 0 ? ++tmp : --tmp;
  return tmp;
}

__device__ inline dstring_view::const_iterator& dstring_view::const_iterator::operator+=(
  dstring_view::const_iterator::difference_type offset)
{
  size_type adjust = abs(offset);
  while (adjust-- > 0) offset > 0 ? operator++() : operator--();
  return *this;
}

__device__ inline dstring_view::const_iterator& dstring_view::const_iterator::operator--()
{
  if (byte_pos > 0)
    while (utf8::bytes_in_utf8_byte((BYTE)p[--byte_pos]) == 0)
      ;
  --char_pos;
  return *this;
}

__device__ inline dstring_view::const_iterator dstring_view::const_iterator::operator--(int)
{
  dstring_view::const_iterator tmp(*this);
  operator--();
  return tmp;
}

__device__ inline dstring_view::const_iterator& dstring_view::const_iterator::operator-=(
  dstring_view::const_iterator::difference_type offset)
{
  size_type adjust = abs(offset);
  while (adjust-- > 0) offset > 0 ? operator--() : operator++();
  return *this;
}

__device__ inline dstring_view::const_iterator dstring_view::const_iterator::operator-(
  dstring_view::const_iterator::difference_type offset)
{
  const_iterator tmp(*this);
  size_type adjust = abs(offset);
  while (adjust-- > 0) offset > 0 ? --tmp : ++tmp;
  return tmp;
}

__device__ inline bool dstring_view::const_iterator::operator==(
  dstring_view::const_iterator const& rhs) const
{
  return (p == rhs.p) && (char_pos == rhs.char_pos);
}

__device__ inline bool dstring_view::const_iterator::operator!=(
  dstring_view::const_iterator const& rhs) const
{
  return (p != rhs.p) || (char_pos != rhs.char_pos);
}

__device__ inline bool dstring_view::const_iterator::operator<(
  dstring_view::const_iterator const& rhs) const
{
  return (p == rhs.p) && (char_pos < rhs.char_pos);
}

__device__ inline bool dstring_view::const_iterator::operator<=(
  dstring_view::const_iterator const& rhs) const
{
  return (p == rhs.p) && (char_pos <= rhs.char_pos);
}

__device__ inline bool dstring_view::const_iterator::operator>(
  dstring_view::const_iterator const& rhs) const
{
  return (p == rhs.p) && (char_pos > rhs.char_pos);
}

__device__ inline bool dstring_view::const_iterator::operator>=(
  dstring_view::const_iterator const& rhs) const
{
  return (p == rhs.p) && (char_pos >= rhs.char_pos);
}

__device__ inline char_utf8 dstring_view::const_iterator::operator*() const
{
  char_utf8 chr = 0;
  utf8::to_char_utf8(p + byte_offset(), chr);
  return chr;
}

__device__ inline size_type dstring_view::const_iterator::position() const { return char_pos; }

__device__ inline size_type dstring_view::const_iterator::byte_offset() const { return byte_pos; }

__device__ inline dstring_view::const_iterator dstring_view::begin() const
{
  return const_iterator(*this, 0);
}

__device__ inline dstring_view::const_iterator dstring_view::end() const
{
  return const_iterator(*this, length());
}

__device__ inline char_utf8 dstring_view::operator[](size_type pos) const
{
  size_type offset = byte_offset(pos);
  if (offset >= _bytes) return 0;
  char_utf8 chr = 0;
  utf8::to_char_utf8(data() + offset, chr);
  return chr;
}

__device__ inline size_type dstring_view::byte_offset(size_type pos) const
{
  size_type offset = 0;
  const char* sptr = _data;
  const char* eptr = sptr + _bytes;
  if (_char_width > 0) return pos * _char_width;
  while ((pos > 0) && (sptr < eptr)) {
    size_type charbytes = utf8::bytes_in_utf8_byte((BYTE)*sptr++);
    if (charbytes) --pos;
    offset += charbytes;
  }
  return offset;
}

__device__ inline int dstring_view::compare(dstring_view const& in) const
{
  return compare(in.data(), in.size_bytes());
}

__device__ inline int dstring_view::compare(const char* data, size_type bytes) const
{
  size_type const len1      = size_bytes();
  const unsigned char* ptr1 = reinterpret_cast<const unsigned char*>(this->data());
  const unsigned char* ptr2 = reinterpret_cast<const unsigned char*>(data);
  size_type idx             = 0;
  for (; (idx < len1) && (idx < bytes); ++idx) {
    if (*ptr1 != *ptr2) return (int)*ptr1 - (int)*ptr2;
    ++ptr1;
    ++ptr2;
  }
  if (idx < len1) return 1;
  if (idx < bytes) return -1;
  return 0;
}

__device__ inline bool dstring_view::operator==(dstring_view const& rhs) const
{
  return compare(rhs) == 0;
}

__device__ inline bool dstring_view::operator!=(dstring_view const& rhs) const
{
  return compare(rhs) != 0;
}

__device__ inline bool dstring_view::operator<(dstring_view const& rhs) const
{
  return compare(rhs) < 0;
}

__device__ inline bool dstring_view::operator>(dstring_view const& rhs) const
{
  return compare(rhs) > 0;
}

__device__ inline bool dstring_view::operator<=(dstring_view const& rhs) const
{
  int rc = compare(rhs);
  return (rc == 0) || (rc < 0);
}

__device__ inline bool dstring_view::operator>=(dstring_view const& rhs) const
{
  int rc = compare(rhs);
  return (rc == 0) || (rc > 0);
}

__device__ inline size_type dstring_view::find(dstring_view const& str,
                                               size_type pos,
                                               size_type count) const
{
  return find(str.data(), str.size_bytes(), pos, count);
}

__device__ inline size_type dstring_view::find(const char* str,
                                               size_type bytes,
                                               size_type pos,
                                               size_type count) const
{
  const char* sptr = data();
  if (!str || !bytes) return -1;
  size_type nchars = length();
  if (count < 0) count = nchars;
  size_type end = pos + count;
  if (end < 0 || end > nchars) end = nchars;
  size_type spos = byte_offset(pos);
  size_type epos = byte_offset(end);

  size_type len2 = bytes;
  size_type len1 = (epos - spos) - len2 + 1;

  const char* ptr1 = sptr + spos;
  const char* ptr2 = str;
  for (size_type idx = 0; idx < len1; ++idx) {
    bool match = true;
    for (size_type jdx = 0; match && (jdx < len2); ++jdx) match = (ptr1[jdx] == ptr2[jdx]);
    if (match) return character_offset(idx + spos);
    ptr1++;
  }
  return -1;
}

__device__ inline size_type dstring_view::find(char_utf8 chr, size_type pos, size_type count) const
{
  char str[sizeof(char_utf8)];
  size_type chwidth = utf8::from_char_utf8(chr, str);
  return find(str, chwidth, pos, count);
}

__device__ inline size_type dstring_view::rfind(dstring_view const& str,
                                                size_type pos,
                                                size_type count) const
{
  return rfind(str.data(), str.size_bytes(), pos, count);
}

__device__ inline size_type dstring_view::rfind(const char* str,
                                                size_type bytes,
                                                size_type pos,
                                                size_type count) const
{
  const char* sptr = data();
  if (!str || !bytes) return -1;
  size_type nchars = length();
  size_type end    = pos + count;
  if (end < 0 || end > nchars) end = nchars;
  size_type spos = byte_offset(pos);
  size_type epos = byte_offset(end);

  size_type len2 = bytes;
  size_type len1 = (epos - spos) - len2 + 1;

  const char* ptr1 = sptr + epos - len2;
  const char* ptr2 = str;
  for (int idx = 0; idx < len1; ++idx) {
    bool match = true;
    for (size_type jdx = 0; match && (jdx < len2); ++jdx) match = (ptr1[jdx] == ptr2[jdx]);
    if (match) return character_offset(epos - len2 - idx);
    ptr1--;  // go backwards
  }
  return -1;
}

__device__ inline size_type dstring_view::rfind(char_utf8 chr, size_type pos, size_type count) const
{
  char str[sizeof(char_utf8)];
  size_type chwidth = utf8::from_char_utf8(chr, str);
  return rfind(str, chwidth, pos, count);
}

// parameters are character position values
__device__ inline dstring_view dstring_view::substr(size_type pos, size_type length) const
{
  size_type spos = byte_offset(pos);
  size_type epos = byte_offset(pos + length);
  if (epos > size_bytes()) epos = size_bytes();
  if (spos >= epos) return dstring_view("", 0);
  return dstring_view(data() + spos, epos - spos);
}

__device__ inline size_type dstring_view::character_offset(size_type bytepos) const
{
  if (_char_width > 0) return bytepos / _char_width;
  return utf8::characters_in_string(data(), bytepos);
}