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
#pragma once

#include "dstring_view.hpp"

class dstring {
 public:
  /**
   * @brief Create an empty string.
   */
  CUDA_DEVICE_CALLABLE dstring();

  /**
   * @brief Cast to dstring_view operator
   */
  CUDA_DEVICE_CALLABLE operator dstring_view() const { return dstring_view(m_data, m_bytes); }

  /**
   * @brief Create a string pointing to existing device memory.
   *
   * The given memory is copied into the instance returned.
   *
   * @param data Device pointer to UTF-8 encoded string
   * @param bytes Number of bytes in the string
   */
  CUDA_DEVICE_CALLABLE dstring(const char* data, size_type bytes);

  /**
   * @brief Create a string object from a null-terminated array.
   *
   * The given memory is copied into the instance returned.
   *
   * @param data Device pointer to UTF-8 encoded null-terminated
   *             character array.
   */
  CUDA_DEVICE_CALLABLE dstring(const char* data);

  CUDA_DEVICE_CALLABLE dstring(dstring_view const&);
  CUDA_DEVICE_CALLABLE dstring(dstring const&);
  CUDA_DEVICE_CALLABLE dstring(dstring&&);
  CUDA_DEVICE_CALLABLE ~dstring();

  CUDA_DEVICE_CALLABLE dstring& operator=(dstring const&);
  CUDA_DEVICE_CALLABLE dstring& operator=(dstring&&);
  CUDA_DEVICE_CALLABLE dstring& operator=(dstring_view const&);

  /**
   * @brief Return the number of bytes in this string.
   */
  CUDA_DEVICE_CALLABLE size_type size_bytes() const;

  /**
   * @brief Return the number of characters in this string.
   */
  CUDA_DEVICE_CALLABLE size_type length() const;

  /**
   * @brief Return the internal pointer to the character array for this object.
   */
  CUDA_DEVICE_CALLABLE char* data();
  CUDA_DEVICE_CALLABLE const char* data() const;

  /**
   * @brief Returns true if there are no characters in this string.
   */
  CUDA_DEVICE_CALLABLE bool empty() const;

  /**
   * @brief Returns true if `data()==nullptr`
   *
   * This is experimental and may be removed in the futre.
   */
  CUDA_DEVICE_CALLABLE bool is_null() const;

  /**
   * @brief Returns an iterator that can be used to navigate through
   *        the UTF-8 characters in this string.
   *
   * This returns a `dstring_view::const_iterator` which is read-only.
   */
  CUDA_DEVICE_CALLABLE dstring_view::const_iterator begin() const;
  CUDA_DEVICE_CALLABLE dstring_view::const_iterator end() const;

  /**
   * @brief Returns the character at the specified position.
   */
  CUDA_DEVICE_CALLABLE char_utf8 at(size_type pos) const;

  /**
   * @brief Returns the character at the specified index.
   *
   * Note this is read-only right now.
   */
  CUDA_DEVICE_CALLABLE char_utf8 operator[](size_type pos) const;

  /**
   * @brief Return the byte offset for a given character position.
   */
  CUDA_DEVICE_CALLABLE size_type byte_offset(size_type pos) const;

  /**
   * @brief Comparing target string with this string. Each character is compared
   * as a UTF-8 code-point value.
   *
   * @param str Target string to compare with this string.
   * @return 0  If they compare equal.
   *         <0 Either the value of the first character of this string that does
   *            not match is lower in the arg string, or all compared characters
   *            match but the arg string is shorter.
   *         >0 Either the value of the first character of this string that does
   *            not match is greater in the arg string, or all compared characters
   *            match but the arg string is longer.
   */
  CUDA_DEVICE_CALLABLE int compare(dstring_view const& str) const;

  /**
   * @brief Comparing target string with this string. Each character is compared
   * as a UTF-8 code-point value.
   *
   * @param str Target string to compare with this string.
   * @param bytes Number of bytes in str.
   * @return 0  If they compare equal.
   *         <0 Either the value of the first character of this string that does
   *            not match is lower in the arg string, or all compared characters
   *            match but the arg string is shorter.
   *         >0 Either the value of the first character of this string that does
   *            not match is greater in the arg string, or all compared characters
   *            match but the arg string is longer.
   */
  CUDA_DEVICE_CALLABLE int compare(const char* str, size_type bytes) const;

  /**
   * @brief Returns true if rhs matches this string exactly.
   */
  CUDA_DEVICE_CALLABLE bool operator==(dstring_view const& rhs) const;

  /**
   * @brief Returns true if rhs does not match this string.
   */
  CUDA_DEVICE_CALLABLE bool operator!=(dstring_view const& rhs) const;

  /**
   * @brief Returns true if this string is ordered before rhs.
   */
  CUDA_DEVICE_CALLABLE bool operator<(dstring_view const& rhs) const;

  /**
   * @brief Returns true if rhs is ordered before this string.
   */
  CUDA_DEVICE_CALLABLE bool operator>(dstring_view const& rhs) const;

  /**
   * @brief Returns true if this string matches or is ordered before rhs.
   */
  CUDA_DEVICE_CALLABLE bool operator<=(dstring_view const& rhs) const;

  /**
   * @brief Returns true if rhs matches or is ordered before this string.
   */
  CUDA_DEVICE_CALLABLE bool operator>=(dstring_view const& rhs) const;

  /**
   * @brief Returns the character position of the first occurrence where the
   * argument str is found in this string within the character range [pos,pos+n).
   *
   * @param str Target string to search within this string.
   * @param pos Character position to start search within this string.
   * @param count Number of characters from pos to include in the search.
   *              Specify -1 to indicate to the end of the string.
   * @return -1 if str is not found in this string.
   */
  CUDA_DEVICE_CALLABLE int find(dstring_view const& str, size_type pos = 0, int count = -1) const;
  CUDA_DEVICE_CALLABLE int find(const char* str,
                                size_type bytes,
                                size_type pos = 0,
                                int count     = -1) const;
  CUDA_DEVICE_CALLABLE int find(char_utf8 chr, size_type pos = 0, int count = -1) const;

  /**
   * @brief Returns the character position of the last occurrence where the
   * argument str is found in this string within the character range [pos,pos+n).
   *
   * @param str Target string to search within this string.
   * @param pos Character position to start search within this string.
   * @param count Number of characters from pos to include in the search.
   *              Specify -1 to indicate to the end of the string.
   * @return -1 if arg string is not found in this string.
   */
  CUDA_DEVICE_CALLABLE int rfind(dstring_view const& str, size_type pos = 0, int count = -1) const;
  CUDA_DEVICE_CALLABLE int rfind(const char* str,
                                 size_type bytes,
                                 size_type pos = 0,
                                 int count     = -1) const;
  CUDA_DEVICE_CALLABLE int rfind(char_utf8 chr, size_type pos = 0, int count = -1) const;

  /**
   * @brief Append a string to the end of this string.
   *
   * This string is modified.
   */
  CUDA_DEVICE_CALLABLE dstring& operator+=(dstring_view const& str);

  /**
   * @brief Append a character to the end of this string.
   *
   * This string is modified.
   */
  CUDA_DEVICE_CALLABLE dstring& operator+=(char_utf8 chr);

  /**
   * @brief Append a null-terminated device memory character array
   * to the end of this string.
   *
   * This string is modified.
   */
  CUDA_DEVICE_CALLABLE dstring& operator+=(const char* str);

  /**
   * @brief Append a null-terminated character array to the end of this string.
   *
   * This string is modified.
   */
  CUDA_DEVICE_CALLABLE dstring& append(const char* str);

  /**
   * @brief Append a character array to the end of this string.
   *
   * This string is modified.
   */
  CUDA_DEVICE_CALLABLE dstring& append(const char* str, size_type bytes);

  /**
   * @brief Append a string to the end of this string.
   *
   * This string is modified.
   */
  CUDA_DEVICE_CALLABLE dstring& append(dstring_view const& str);

  /**
   * @brief Append a character to the end of this string
   * a specified number of times.
   *
   * This string is modified.
   */
  CUDA_DEVICE_CALLABLE dstring& append(char_utf8 chr, size_type count = 1);

  /**
   * @brief Insert the given string into the character position specified.
   *
   * This string is modified.
   */
  CUDA_DEVICE_CALLABLE dstring& insert(size_type pos, const char* data);

  /**
   * @brief Insert a string into the character position specified.
   *
   * This string is modified.
   */
  CUDA_DEVICE_CALLABLE dstring& insert(size_type pos, dstring_view const& str);
  CUDA_DEVICE_CALLABLE dstring& insert(size_type pos, const char* data, size_type bytes);

  /**
   * @brief Insert a character one or more times into the character position specified.
   *
   * This string is modified.
   */
  CUDA_DEVICE_CALLABLE dstring& insert(size_type pos, size_type count, char_utf8 chr);

  /**
   * @brief Returns a substring of this string.
   *
   * @param start Character position to start the substring
   * @param length Number of characters for the substring.
   *               If can be greater than the number of available characters.
   * @return New string with the specified characters
   */
  CUDA_DEVICE_CALLABLE dstring substr(size_type start, size_type length) const;

  /**
   * @brief Replace the given range of characters with a given string.
   *
   * This string is modified.
   */
  CUDA_DEVICE_CALLABLE dstring& replace(size_type pos, size_type length, const dstring_view& str);
  CUDA_DEVICE_CALLABLE dstring& replace(size_type pos, size_type length, const char* data);
  CUDA_DEVICE_CALLABLE dstring& replace(size_type pos,
                                        size_type length,
                                        const char* data,
                                        size_type bytes);

  /**
   * @brief Replace the given range of characters with a character one or more times.
   *
   * This string is modified.
   */
  CUDA_DEVICE_CALLABLE dstring& replace(size_type pos,
                                        size_type length,
                                        size_type count,
                                        char_utf8 chr);

  /**
   * @brief Tokenizes a string around a given delimiter up to a maximum count.
   *
   * @param delim Null-terminated array identifying where to split this string.
   * @param count Maximum number of tokens to return.
   * @param strs Tokens are added to this array and must be large enough to
   *             hold the split results. Set to `nullptr` to get the number
   *             of tokens first.
   * @return The number of tokens.
   */
  CUDA_DEVICE_CALLABLE size_type split(const char* delim, int count, dstring* strs) const;
  CUDA_DEVICE_CALLABLE size_type rsplit(const char* delim, int count, dstring* strs) const;
  CUDA_DEVICE_CALLABLE dstring& join(dstring_view const* strings, size_type count);

  /**
   * @brief Remove the specified characters from the beginning or end of this string.
   *
   * This string is modified.
   */
  CUDA_DEVICE_CALLABLE dstring& strip(const char* tgts = nullptr);
  CUDA_DEVICE_CALLABLE dstring& lstrip(const char* tgts = nullptr);
  CUDA_DEVICE_CALLABLE dstring& rstrip(const char* tgts = nullptr);

  /**
   * @brief Convert this string to all upper-case characters.
   *
   * This only supports ASCII right now.
   * This string is modified.
   */
  CUDA_DEVICE_CALLABLE dstring& upper();

  /**
   * @brief Convert this string to all lower-case characters.
   *
   * This only supports ASCII right now.
   * This string is modified.
   */
  CUDA_DEVICE_CALLABLE dstring& lower();

 private:
  char* m_data{};
  size_type m_bytes{};
  size_type m_size{};

  // utilities
  CUDA_DEVICE_CALLABLE char* allocate(size_type bytes);
  CUDA_DEVICE_CALLABLE void deallocate(char* data);
  CUDA_DEVICE_CALLABLE size_type char_offset(size_type bytepos) const;
};

CUDA_DEVICE_CALLABLE bool starts_with(dstring_view const& dstr, const char* tgt);
CUDA_DEVICE_CALLABLE bool starts_with(dstring_view const& dstr, const char* tgt, size_type bytes);
CUDA_DEVICE_CALLABLE bool starts_with(dstring_view const& dstr, dstring_view const& tgt);
CUDA_DEVICE_CALLABLE bool ends_with(dstring_view const& dstr, const char* tgt);
CUDA_DEVICE_CALLABLE bool ends_with(dstring_view const& dstr, const char* tgt, size_type bytes);
CUDA_DEVICE_CALLABLE bool ends_with(dstring_view const& dstr, dstring_view const& tgt);
