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

 #include <cstdint>
 
 using char_utf8 = uint32_t;  //<< UTF-8 characters are 1-4 bytes
 using size_type = int32_t;
 
 namespace utf8 {
 
 /**
  * @brief Returns the number of bytes in the specified character.
  *
  * @param character Single character
  * @return Number of bytes
  */
 __host__ __device__ size_type bytes_in_char_utf8(char_utf8 character)
 {
   size_type count = 1;
   count += static_cast<size_type>((character & (unsigned)0x0000FF00) > 0);
   count += static_cast<size_type>((character & (unsigned)0x00FF0000) > 0);
   count += static_cast<size_type>((character & (unsigned)0xFF000000) > 0);
   return count;
 }
 
 /**
  * @brief Returns the number of bytes used to represent the provided byte.
  * This could be 0 to 4 bytes. 0 is returned for intermediate bytes within a
  * single character. For example, for the two-byte 0xC3A8 single character,
  * the first byte would return 2 and the second byte would return 0.
  *
  * @param byte Byte from an encoded character.
  * @return Number of bytes.
  */
 __host__ __device__ size_type bytes_in_utf8_byte(unsigned char byte)
 {
   size_type count = 1;
   count += static_cast<size_type>((byte & 0xF0) == 0xF0);  // 4-byte character prefix
   count += static_cast<size_type>((byte & 0xE0) == 0xE0);  // 3-byte character prefix
   count += static_cast<size_type>((byte & 0xC0) == 0xC0);  // 2-byte character prefix
   count -= static_cast<size_type>((byte & 0xC0) == 0x80);  // intermediate byte
   return count;
 }
 
 /**
  * @brief Convert a char array into a char_utf8 value.
  *
  * @param src String containing encoded char bytes.
  * @param[out] character Single char_utf8 value.
  * @return The number of bytes in the character
  */
 __host__ __device__ size_type to_char_utf8(const char* src, char_utf8& character)
 {
   size_type const chwidth = utf8::bytes_in_utf8_byte((unsigned char)*src);
   character               = (char_utf8)(*src++) & 0xFF;
   if (chwidth > 1) {
     character = character << 8;
     character |= ((char_utf8)(*src++) & 0xFF);  // << 8;
     if (chwidth > 2) {
       character = character << 8;
       character |= ((char_utf8)(*src++) & 0xFF);  // << 16;
       if (chwidth > 3) {
         character = character << 8;
         character |= ((char_utf8)(*src++) & 0xFF);  // << 24;
       }
     }
   }
   return chwidth;
 }
 
 /**
  * @brief Place a char_utf8 value into a char array.
  *
  * @param character Single character
  * @param[out] dst Allocated char array with enough space to hold the encoded characer.
  * @return The number of bytes in the character
  */
 __host__ __device__ size_type from_char_utf8(char_utf8 character, char* dst)
 {
   size_type const chwidth = bytes_in_char_utf8(character);
   for (size_type idx = 0; idx < chwidth; ++idx) {
     dst[chwidth - idx - 1] = (char)character & 0xFF;
     character              = character >> 8;
   }
   return chwidth;
 }
 
 /**
  * @brief Return the number of UTF-8 characters in this provided char array.
  *
  * @param str String with encoded char bytes.
  * @param bytes Number of bytes in str.
  * @return The number of characters in the array.
  */
 __host__ __device__ size_type characters_in_string(const char* str, size_type bytes)
 {
   if ((str == 0) || (bytes == 0)) return 0;
   unsigned int nchars = 0;
   for (size_type idx = 0; idx < bytes; ++idx)
     nchars += (unsigned int)(((unsigned char)str[idx] & 0xC0) != 0x80);
   return (size_type)nchars;
 }
 
 /**
  * @brief This will return true if passed the first byte of a UTF-8 character.
  *
  * @param byte Any byte from a valid UTF-8 character
  * @return true if this the first byte of the character
  */
 constexpr bool is_begin_utf8_char(unsigned char byte)
 {
   // The (0xC0 & 0x80) bit pattern identifies a continuation byte of a character.
   return (byte & 0xC0) != 0x80;
 }
 
 }  // namespace utf8