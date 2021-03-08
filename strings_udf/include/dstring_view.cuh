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

 #pragma once

 #include <iterator>
 #include "utf8.cuh"
 
 class dstring;
 
 /**
  * @brief Read-only view of device memory containing UTF-8 encoded string.
  */
 class dstring_view {
  public:
   /**
    * @brief Create an empty string.
    */
   __host__ __device__ dstring_view();
 
   /**
    * @brief Create string view from string.
    */
   __device__ dstring_view(dstring const& dstr);
 
   /**
    * @brief Create string view from existing device memory.
    *
    * The memory provided must exist for the lifetime of the
    * object returned.
    *
    * @param data Device pointer to UTF-8 encoded string
    * @param bytes Number of bytes in the string
    */
   __host__ __device__ dstring_view(const char* data, size_type bytes);
 
   dstring_view(const dstring_view&) = default;
   dstring_view(dstring_view&&)      = default;
   ~dstring_view()                   = default;
   dstring_view& operator=(const dstring_view&) = default;
   dstring_view& operator=(dstring_view&&) = default;
 
   /**
    * @brief Return the number of bytes in this string
    */
   __host__ __device__ size_type size_bytes() const;
 
   /**
    * @brief Return the number of characters in this string
    */
   __device__ size_type length() const;
 
   /**
    * @brief Return a pointer to the internal device array
    */
   __host__ __device__ const char* data() const;
 
   /**
    * @brief Return true if string has no characters
    */
   __host__ __device__ bool empty() const;
 
   /**
    * @brief Handy iterator for navigating through UTF-8 encoded characters.
    */
   class const_iterator {
    public:
     using difference_type   = ptrdiff_t;
     using value_type        = char_utf8;
     using reference         = char_utf8&;
     using pointer           = char_utf8*;
     using iterator_category = std::input_iterator_tag;
     __device__ const_iterator(const dstring_view& str, size_type pos);
     const_iterator(const const_iterator& mit) = default;
     const_iterator(const_iterator&& mit)      = default;
     const_iterator& operator=(const const_iterator&) = default;
     const_iterator& operator=(const_iterator&&) = default;
     __device__ const_iterator& operator++();
     __device__ const_iterator operator++(int);
     __device__ const_iterator& operator+=(difference_type);
     __device__ const_iterator operator+(difference_type);
     __device__ const_iterator& operator--();
     __device__ const_iterator operator--(int);
     __device__ const_iterator& operator-=(difference_type);
     __device__ const_iterator operator-(difference_type);
     __device__ bool operator==(const const_iterator&) const;
     __device__ bool operator!=(const const_iterator&) const;
     __device__ bool operator<(const const_iterator&) const;
     __device__ bool operator<=(const const_iterator&) const;
     __device__ bool operator>(const const_iterator&) const;
     __device__ bool operator>=(const const_iterator&) const;
     __device__ char_utf8 operator*() const;
     __device__ size_type position() const;
     __device__ size_type byte_offset() const;
 
    private:
     const char* p{};
     size_type bytes{};
     size_type char_pos{};
     size_type byte_pos{};
   };
 
   /**
    * @brief Return new iterator pointing to the beginning of this string
    */
   __device__ const_iterator begin() const;
 
   /**
    * @brief Return new iterator pointing past the end of this string
    */
   __device__ const_iterator end() const;
 
   /**
    * @brief Return single UTF-8 character at the given character position
    *
    * @param pos Character position
    */
   __device__ char_utf8 operator[](size_type pos) const;
 
   /**
    * @brief Return the byte offset from data() for a given character position
    *
    * @param pos Character position
    */
   __device__ size_type byte_offset(size_type pos) const;
 
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
   __device__ int compare(const dstring_view& str) const;
 
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
   __device__ int compare(const char* str, size_type bytes) const;
 
   /**
    * @brief Returns true if rhs matches this string exactly.
    */
   __device__ bool operator==(const dstring_view& rhs) const;
   /**
    * @brief Returns true if rhs does not match this string.
    */
   __device__ bool operator!=(const dstring_view& rhs) const;
   /**
    * @brief Returns true if this string is ordered before rhs.
    */
   __device__ bool operator<(const dstring_view& rhs) const;
   /**
    * @brief Returns true if rhs is ordered before this string.
    */
   __device__ bool operator>(const dstring_view& rhs) const;
   /**
    * @brief Returns true if this string matches or is ordered before rhs.
    */
   __device__ bool operator<=(const dstring_view& rhs) const;
   /**
    * @brief Returns true if rhs matches or is ordered before this string.
    */
   __device__ bool operator>=(const dstring_view& rhs) const;
 
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
   __device__ size_type find(const dstring_view& str, size_type pos = 0, size_type count = -1) const;
   /**
    * @brief Returns the character position of the first occurrence where the
    * array str is found in this string within the character range [pos,pos+n).
    *
    * @param str Target array to search within this string.
    * @param bytes Number of bytes in str.
    * @param pos Character position to start search within this string.
    * @param count Number of characters from pos to include in the search.
    *              Specify -1 to indicate to the end of the string.
    * @return -1 if arg string is not found in this string.
    */
   __device__ size_type find(const char* str,
                             size_type bytes,
                             size_type pos   = 0,
                             size_type count = -1) const;
   /**
    * @brief Returns the character position of the first occurrence where
    * character is found in this string within the character range [pos,pos+n).
    *
    * @param character Single encoded character.
    * @param pos Character position to start search within this string.
    * @param count Number of characters from pos to include in the search.
    *              Specify -1 to indicate to the end of the string.
    * @return -1 if arg string is not found in this string.
    */
   __device__ size_type find(char_utf8 character, size_type pos = 0, size_type count = -1) const;
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
   __device__ size_type rfind(const dstring_view& str,
                              size_type pos   = 0,
                              size_type count = -1) const;
   /**
    * @brief Returns the character position of the last occurrence where the
    * array str is found in this string within the character range [pos,pos+n).
    *
    * @param str Target string to search with this string.
    * @param bytes Number of bytes in str.
    * @param pos Character position to start search within this string.
    * @param count Number of characters from pos to include in the search.
    *              Specify -1 to indicate to the end of the string.
    * @return -1 if arg string is not found in this string.
    */
   __device__ size_type rfind(const char* str,
                              size_type bytes,
                              size_type pos   = 0,
                              size_type count = -1) const;
   /**
    * @brief Returns the character position of the last occurrence where
    * character is found in this string within the character range [pos,pos+n).
    *
    * @param character Single encoded character.
    * @param pos Character position to start search within this string.
    * @param count Number of characters from pos to include in the search.
    *              Specify -1 to indicate to the end of the string.
    * @return -1 if arg string is not found in this string.
    */
   __device__ size_type rfind(char_utf8 character, size_type pos = 0, size_type count = -1) const;
 
   /**
    * @brief Return a sub-string of this string. The original string and device
    * memory must still be maintained for the lifetime of the returned instance.
    *
    * @param start Character position to start the sub-string.
    * @param length Number of characters from start to include in the sub-string.
    * @return New instance pointing to a subset of the characters within this instance.
    */
   __device__ dstring_view substr(size_type start, size_type length) const;
 
  private:
   const char* _data{};           ///< Pointer to device memory contain char array for this string
   size_type _bytes{};            ///< Number of bytes in _data for this string
   mutable size_type _length{};   ///< Number of characters in this string (computed)
   mutable int8_t _char_width{};  ///< Number of bytes per character if uniform width (computed)
 
   /**
    * @brief Return the character position of the given byte offset.
    *
    * @param bytepos Byte position from start of _data.
    * @return The character position for the specified byte.
    */
   __device__ size_type character_offset(size_type bytepos) const;
 };