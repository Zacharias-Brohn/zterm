//! SIMD-optimized string operations based on Kitty's implementation.
//!
//! This module provides high-performance SIMD-accelerated operations:
//! - UTF-8 decoder (16-byte SSE or 32-byte AVX2 chunks)
//! - Byte search functions (find_either_of_two_bytes, find_c0_control)
//! - XOR masking for WebSocket frames
//!
//! The UTF-8 algorithm is based on the blog post:
//! https://woboq.com/blog/utf-8-processing-using-simd.html
//! and Kitty's implementation in simd-string-impl.h

// Allow unsafe operations within unsafe functions without additional blocks.
// This code is ported from C and follows the same patterns as Kitty's implementation.
#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[cfg(target_arch = "x86")]
use std::arch::x86::*;

// ============================================================================
// SIMD Feature Detection and Dispatch
// ============================================================================

/// Cached SIMD capability flags for runtime dispatch.
#[derive(Clone, Copy)]
pub struct SimdCapabilities {
    pub has_sse41: bool,
    pub has_ssse3: bool,
    pub has_avx2: bool,
}

impl SimdCapabilities {
    /// Detect SIMD capabilities at runtime.
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    pub fn detect() -> Self {
        Self {
            has_sse41: is_x86_feature_detected!("sse4.1"),
            has_ssse3: is_x86_feature_detected!("ssse3"),
            has_avx2: is_x86_feature_detected!("avx2"),
        }
    }
    
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86")))]
    pub fn detect() -> Self {
        Self {
            has_sse41: false,
            has_ssse3: false,
            has_avx2: false,
        }
    }
}

// Global cached capabilities (initialized on first use)
static SIMD_CAPS: std::sync::OnceLock<SimdCapabilities> = std::sync::OnceLock::new();

/// Get cached SIMD capabilities.
pub fn simd_caps() -> &'static SimdCapabilities {
    SIMD_CAPS.get_or_init(SimdCapabilities::detect)
}

// ============================================================================
// Byte Search Functions (like Kitty's find_either_of_two_bytes)
// ============================================================================

/// Find the first occurrence of a single byte in the haystack.
/// Returns the index of the first match, or None if not found.
///
/// This is a SIMD-accelerated alternative to `memchr::memchr`.
#[inline]
pub fn find_byte(haystack: &[u8], needle: u8) -> Option<usize> {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        let caps = simd_caps();
        if caps.has_avx2 {
            // SAFETY: We checked for AVX2 support
            return unsafe { find_byte_avx2(haystack, needle) };
        }
        if caps.has_sse41 {
            // SAFETY: We checked for SSE4.1 support
            return unsafe { find_byte_sse(haystack, needle) };
        }
    }
    haystack.iter().position(|&b| b == needle)
}

/// SSE implementation of find_byte.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "sse2", enable = "sse4.1")]
unsafe fn find_byte_sse(haystack: &[u8], needle: u8) -> Option<usize> {
    let needle_vec = _mm_set1_epi8(needle as i8);
    let mut offset = 0;
    let len = haystack.len();
    let ptr = haystack.as_ptr();
    
    while offset + 16 <= len {
        let chunk = _mm_loadu_si128(ptr.add(offset) as *const __m128i);
        let cmp = _mm_cmpeq_epi8(chunk, needle_vec);
        let mask = _mm_movemask_epi8(cmp) as u32;
        if mask != 0 {
            return Some(offset + mask.trailing_zeros() as usize);
        }
        offset += 16;
    }
    
    for i in offset..len {
        if *ptr.add(i) == needle {
            return Some(i);
        }
    }
    None
}

/// AVX2 implementation of find_byte.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
unsafe fn find_byte_avx2(haystack: &[u8], needle: u8) -> Option<usize> {
    let needle_vec = _mm256_set1_epi8(needle as i8);
    let mut offset = 0;
    let len = haystack.len();
    let ptr = haystack.as_ptr();
    
    while offset + 32 <= len {
        let chunk = _mm256_loadu_si256(ptr.add(offset) as *const __m256i);
        let cmp = _mm256_cmpeq_epi8(chunk, needle_vec);
        let mask = _mm256_movemask_epi8(cmp) as u32;
        if mask != 0 {
            return Some(offset + mask.trailing_zeros() as usize);
        }
        offset += 32;
    }
    
    // Handle remainder with SSE
    while offset + 16 <= len {
        let chunk = _mm_loadu_si128(ptr.add(offset) as *const __m128i);
        let needle_vec_128 = _mm_set1_epi8(needle as i8);
        let cmp = _mm_cmpeq_epi8(chunk, needle_vec_128);
        let mask = _mm_movemask_epi8(cmp) as u32;
        if mask != 0 {
            return Some(offset + mask.trailing_zeros() as usize);
        }
        offset += 16;
    }
    
    for i in offset..len {
        if *ptr.add(i) == needle {
            return Some(i);
        }
    }
    None
}

/// Find the first occurrence of either byte `a` or byte `b` in the haystack.
/// Returns the index of the first match, or None if not found.
/// 
/// This is equivalent to Kitty's `find_either_of_two_bytes` function.
#[inline]
pub fn find_either_of_two_bytes(haystack: &[u8], a: u8, b: u8) -> Option<usize> {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        let caps = simd_caps();
        if caps.has_avx2 {
            // SAFETY: We checked for AVX2 support
            return unsafe { find_either_of_two_bytes_avx2(haystack, a, b) };
        }
        if caps.has_sse41 {
            // SAFETY: We checked for SSE4.1 support
            return unsafe { find_either_of_two_bytes_sse(haystack, a, b) };
        }
    }
    find_either_of_two_bytes_scalar(haystack, a, b)
}

/// Scalar fallback for find_either_of_two_bytes.
#[inline]
fn find_either_of_two_bytes_scalar(haystack: &[u8], a: u8, b: u8) -> Option<usize> {
    haystack.iter().position(|&byte| byte == a || byte == b)
}

/// SSE implementation of find_either_of_two_bytes.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "sse2", enable = "sse4.1")]
unsafe fn find_either_of_two_bytes_sse(haystack: &[u8], a: u8, b: u8) -> Option<usize> {
    let a_vec = _mm_set1_epi8(a as i8);
    let b_vec = _mm_set1_epi8(b as i8);
    
    let mut offset = 0;
    let len = haystack.len();
    let ptr = haystack.as_ptr();
    
    // Process 16 bytes at a time
    while offset + 16 <= len {
        let chunk = _mm_loadu_si128(ptr.add(offset) as *const __m128i);
        let cmp_a = _mm_cmpeq_epi8(chunk, a_vec);
        let cmp_b = _mm_cmpeq_epi8(chunk, b_vec);
        let combined = _mm_or_si128(cmp_a, cmp_b);
        let mask = _mm_movemask_epi8(combined) as u32;
        if mask != 0 {
            return Some(offset + mask.trailing_zeros() as usize);
        }
        offset += 16;
    }
    
    // Handle remainder with scalar
    for i in offset..len {
        let byte = *ptr.add(i);
        if byte == a || byte == b {
            return Some(i);
        }
    }
    None
}

/// AVX2 implementation of find_either_of_two_bytes.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
unsafe fn find_either_of_two_bytes_avx2(haystack: &[u8], a: u8, b: u8) -> Option<usize> {
    let a_vec = _mm256_set1_epi8(a as i8);
    let b_vec = _mm256_set1_epi8(b as i8);
    
    let mut offset = 0;
    let len = haystack.len();
    let ptr = haystack.as_ptr();
    
    // Process 32 bytes at a time
    while offset + 32 <= len {
        let chunk = _mm256_loadu_si256(ptr.add(offset) as *const __m256i);
        let cmp_a = _mm256_cmpeq_epi8(chunk, a_vec);
        let cmp_b = _mm256_cmpeq_epi8(chunk, b_vec);
        let combined = _mm256_or_si256(cmp_a, cmp_b);
        let mask = _mm256_movemask_epi8(combined) as u32;
        if mask != 0 {
            return Some(offset + mask.trailing_zeros() as usize);
        }
        offset += 32;
    }
    
    // Handle remainder with SSE (16 bytes)
    while offset + 16 <= len {
        let chunk = _mm_loadu_si128(ptr.add(offset) as *const __m128i);
        let a_vec_128 = _mm_set1_epi8(a as i8);
        let b_vec_128 = _mm_set1_epi8(b as i8);
        let cmp_a = _mm_cmpeq_epi8(chunk, a_vec_128);
        let cmp_b = _mm_cmpeq_epi8(chunk, b_vec_128);
        let combined = _mm_or_si128(cmp_a, cmp_b);
        let mask = _mm_movemask_epi8(combined) as u32;
        if mask != 0 {
            return Some(offset + mask.trailing_zeros() as usize);
        }
        offset += 16;
    }
    
    // Handle remainder with scalar
    for i in offset..len {
        let byte = *ptr.add(i);
        if byte == a || byte == b {
            return Some(i);
        }
    }
    None
}

// ============================================================================
// C0 Control Character Detection (like Kitty's IndexC0)
// ============================================================================

/// Find the first C0 control character (byte < 0x20 or byte == 0x7F).
/// Returns the index of the first match, or None if not found.
///
/// This is equivalent to Kitty's `IndexC0` function in the simdstring package.
#[inline]
pub fn find_c0_control(haystack: &[u8]) -> Option<usize> {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        let caps = simd_caps();
        if caps.has_avx2 {
            // SAFETY: We checked for AVX2 support
            return unsafe { find_c0_control_avx2(haystack) };
        }
        if caps.has_sse41 {
            // SAFETY: We checked for SSE4.1 support
            return unsafe { find_c0_control_sse(haystack) };
        }
    }
    find_c0_control_scalar(haystack)
}

/// Scalar fallback for find_c0_control.
#[inline]
fn find_c0_control_scalar(haystack: &[u8]) -> Option<usize> {
    haystack.iter().position(|&byte| byte < 0x20 || byte == 0x7F)
}

/// SSE implementation of find_c0_control.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "sse2", enable = "sse4.1")]
unsafe fn find_c0_control_sse(haystack: &[u8]) -> Option<usize> {
    // C0 control chars are: 0x00-0x1F and 0x7F
    // Strategy: (byte < 0x20) || (byte == 0x7F)
    // For the < 0x20 check, we use saturating subtraction: if byte < 0x20, then (byte - 0x20) saturates to 0
    // Or we can use: byte + 0x80 (wrapping) < 0x20 + 0x80 = 0xA0 in unsigned = -96 in signed
    let threshold = _mm_set1_epi8(-96i8); // 0x20 - 0x80 = -96 in signed
    let bias = _mm_set1_epi8(-128i8); // 0x80 as i8
    let del = _mm_set1_epi8(0x7F);
    
    let mut offset = 0;
    let len = haystack.len();
    let ptr = haystack.as_ptr();
    
    while offset + 16 <= len {
        let chunk = _mm_loadu_si128(ptr.add(offset) as *const __m128i);
        // Convert to signed range for comparison: chunk_signed = chunk + 0x80 (wrapping)
        // This maps 0x00 -> 0x80 (-128), 0x7F -> 0xFF (-1), 0x80 -> 0x00, etc.
        let chunk_signed = _mm_add_epi8(chunk, bias);
        // Check chunk_signed < threshold (equivalent to chunk < 0x20)
        let lt_20 = _mm_cmplt_epi8(chunk_signed, threshold);
        // Check chunk == 0x7F
        let eq_7f = _mm_cmpeq_epi8(chunk, del);
        // Combine
        let combined = _mm_or_si128(lt_20, eq_7f);
        let mask = _mm_movemask_epi8(combined) as u32;
        if mask != 0 {
            return Some(offset + mask.trailing_zeros() as usize);
        }
        offset += 16;
    }
    
    // Handle remainder
    for i in offset..len {
        let byte = *ptr.add(i);
        if byte < 0x20 || byte == 0x7F {
            return Some(i);
        }
    }
    None
}

/// AVX2 implementation of find_c0_control.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
unsafe fn find_c0_control_avx2(haystack: &[u8]) -> Option<usize> {
    let threshold = _mm256_set1_epi8(-96i8); // 0x20 - 0x80 = -96 in signed
    let bias = _mm256_set1_epi8(-128i8); // 0x80 as i8
    let del = _mm256_set1_epi8(0x7F);
    
    let mut offset = 0;
    let len = haystack.len();
    let ptr = haystack.as_ptr();
    
    while offset + 32 <= len {
        let chunk = _mm256_loadu_si256(ptr.add(offset) as *const __m256i);
        let chunk_signed = _mm256_add_epi8(chunk, bias);
        let lt_20 = _mm256_cmpgt_epi8(threshold, chunk_signed);
        let eq_7f = _mm256_cmpeq_epi8(chunk, del);
        let combined = _mm256_or_si256(lt_20, eq_7f);
        let mask = _mm256_movemask_epi8(combined) as u32;
        if mask != 0 {
            return Some(offset + mask.trailing_zeros() as usize);
        }
        offset += 32;
    }
    
    // Handle remainder with SSE path
    while offset + 16 <= len {
        let chunk = _mm_loadu_si128(ptr.add(offset) as *const __m128i);
        let threshold_128 = _mm_set1_epi8(-96i8);
        let bias_128 = _mm_set1_epi8(-128i8);
        let del_128 = _mm_set1_epi8(0x7F);
        let chunk_signed = _mm_add_epi8(chunk, bias_128);
        let lt_20 = _mm_cmplt_epi8(chunk_signed, threshold_128);
        let eq_7f = _mm_cmpeq_epi8(chunk, del_128);
        let combined = _mm_or_si128(lt_20, eq_7f);
        let mask = _mm_movemask_epi8(combined) as u32;
        if mask != 0 {
            return Some(offset + mask.trailing_zeros() as usize);
        }
        offset += 16;
    }
    
    // Handle remainder
    for i in offset..len {
        let byte = *ptr.add(i);
        if byte < 0x20 || byte == 0x7F {
            return Some(i);
        }
    }
    None
}

// ============================================================================
// XOR Data for WebSocket Masking (like Kitty's xor_data64)
// ============================================================================

/// XOR data with a 4-byte mask (WebSocket frame masking).
/// The mask is applied cyclically starting from the given offset.
/// Returns the new mask offset after processing.
///
/// This is equivalent to Kitty's `xor_data64` function but optimized for
/// the standard 4-byte WebSocket mask.
#[inline]
pub fn xor_mask(data: &mut [u8], mask: [u8; 4], start_offset: usize) -> usize {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        let caps = simd_caps();
        if caps.has_avx2 && data.len() >= 32 {
            // SAFETY: We checked for AVX2 support
            return unsafe { xor_mask_avx2(data, mask, start_offset) };
        }
        if caps.has_sse41 && data.len() >= 16 {
            // SAFETY: We checked for SSE4.1 support
            return unsafe { xor_mask_sse(data, mask, start_offset) };
        }
    }
    xor_mask_scalar(data, mask, start_offset)
}

/// Scalar fallback for xor_mask.
#[inline]
fn xor_mask_scalar(data: &mut [u8], mask: [u8; 4], start_offset: usize) -> usize {
    let mut offset = start_offset;
    for byte in data.iter_mut() {
        *byte ^= mask[offset & 3];
        offset += 1;
    }
    offset & 3
}

/// SSE implementation of xor_mask.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "sse2")]
unsafe fn xor_mask_sse(data: &mut [u8], mask: [u8; 4], start_offset: usize) -> usize {
    let len = data.len();
    let ptr = data.as_mut_ptr();
    let mut pos = 0;
    let mut offset = start_offset;
    
    // Handle unaligned prefix to get to mask-aligned position
    while pos < len && (offset & 3) != 0 {
        *ptr.add(pos) ^= mask[offset & 3];
        pos += 1;
        offset += 1;
    }
    
    // Create 16-byte mask vector (repeat 4-byte mask 4 times)
    let mask_vec = _mm_set_epi8(
        mask[3] as i8, mask[2] as i8, mask[1] as i8, mask[0] as i8,
        mask[3] as i8, mask[2] as i8, mask[1] as i8, mask[0] as i8,
        mask[3] as i8, mask[2] as i8, mask[1] as i8, mask[0] as i8,
        mask[3] as i8, mask[2] as i8, mask[1] as i8, mask[0] as i8,
    );
    
    // Process 16 bytes at a time
    while pos + 16 <= len {
        let chunk = _mm_loadu_si128(ptr.add(pos) as *const __m128i);
        let xored = _mm_xor_si128(chunk, mask_vec);
        _mm_storeu_si128(ptr.add(pos) as *mut __m128i, xored);
        pos += 16;
        offset += 16;
    }
    
    // Handle remainder
    while pos < len {
        *ptr.add(pos) ^= mask[offset & 3];
        pos += 1;
        offset += 1;
    }
    
    offset & 3
}

/// AVX2 implementation of xor_mask.
#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2")]
unsafe fn xor_mask_avx2(data: &mut [u8], mask: [u8; 4], start_offset: usize) -> usize {
    let len = data.len();
    let ptr = data.as_mut_ptr();
    let mut pos = 0;
    let mut offset = start_offset;
    
    // Handle unaligned prefix
    while pos < len && (offset & 3) != 0 {
        *ptr.add(pos) ^= mask[offset & 3];
        pos += 1;
        offset += 1;
    }
    
    // Create 32-byte mask vector (repeat 4-byte mask 8 times)
    let mask_vec = _mm256_set_epi8(
        mask[3] as i8, mask[2] as i8, mask[1] as i8, mask[0] as i8,
        mask[3] as i8, mask[2] as i8, mask[1] as i8, mask[0] as i8,
        mask[3] as i8, mask[2] as i8, mask[1] as i8, mask[0] as i8,
        mask[3] as i8, mask[2] as i8, mask[1] as i8, mask[0] as i8,
        mask[3] as i8, mask[2] as i8, mask[1] as i8, mask[0] as i8,
        mask[3] as i8, mask[2] as i8, mask[1] as i8, mask[0] as i8,
        mask[3] as i8, mask[2] as i8, mask[1] as i8, mask[0] as i8,
        mask[3] as i8, mask[2] as i8, mask[1] as i8, mask[0] as i8,
    );
    
    // Process 32 bytes at a time
    while pos + 32 <= len {
        let chunk = _mm256_loadu_si256(ptr.add(pos) as *const __m256i);
        let xored = _mm256_xor_si256(chunk, mask_vec);
        _mm256_storeu_si256(ptr.add(pos) as *mut __m256i, xored);
        pos += 32;
        offset += 32;
    }
    
    // Process 16 bytes if remaining
    while pos + 16 <= len {
        let mask_vec_128 = _mm_set_epi8(
            mask[3] as i8, mask[2] as i8, mask[1] as i8, mask[0] as i8,
            mask[3] as i8, mask[2] as i8, mask[1] as i8, mask[0] as i8,
            mask[3] as i8, mask[2] as i8, mask[1] as i8, mask[0] as i8,
            mask[3] as i8, mask[2] as i8, mask[1] as i8, mask[0] as i8,
        );
        let chunk = _mm_loadu_si128(ptr.add(pos) as *const __m128i);
        let xored = _mm_xor_si128(chunk, mask_vec_128);
        _mm_storeu_si128(ptr.add(pos) as *mut __m128i, xored);
        pos += 16;
        offset += 16;
    }
    
    // Handle remainder
    while pos < len {
        *ptr.add(pos) ^= mask[offset & 3];
        pos += 1;
        offset += 1;
    }
    
    offset & 3
}

// ============================================================================
// UTF-8 Decoder State and Tables
// ============================================================================

/// UTF-8 decoder state for handling sequences that span chunks.
#[derive(Debug, Default, Clone)]
pub struct Utf8State {
    pub cur: u8,
    pub prev: u8,
    pub codep: u32,
}

const UTF8_ACCEPT: u8 = 0;
const UTF8_REJECT: u8 = 12;

/// UTF-8 state transition table (Bjoern Hoehrmann's DFA).
static UTF8_DECODE_TABLE: [u8; 364] = [
    // Character class lookup (0-255)
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,  0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,  9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,9,
    7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,  7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,7,
    8,8,2,2,2,2,2,2,2,2,2,2,2,2,2,2,  2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,
    10,3,3,3,3,3,3,3,3,3,3,3,3,4,3,3, 11,6,6,6,5,8,8,8,8,8,8,8,8,8,8,8,
    // State transition table
     0,12,24,36,60,96,84,12,12,12,48,72, 12,12,12,12,12,12,12,12,12,12,12,12,
    12, 0,12,12,12,12,12, 0,12, 0,12,12, 12,24,12,12,12,12,12,24,12,24,12,12,
    12,12,12,12,12,12,12,24,12,12,12,12, 12,24,12,12,12,12,12,12,12,24,12,12,
    12,12,12,12,12,12,12,36,12,36,12,12, 12,36,12,12,12,12,12,36,12,36,12,12,
    12,36,12,12,12,12,12,12,12,12,12,12,
];

/// Decode a single UTF-8 byte using DFA.
#[inline(always)]
fn decode_utf8_byte(state: &mut u8, codep: &mut u32, byte: u8) -> u8 {
    let char_class = UTF8_DECODE_TABLE[byte as usize];
    *codep = if *state == UTF8_ACCEPT {
        (0xFF >> char_class) as u32 & byte as u32
    } else {
        (byte as u32 & 0x3F) | (*codep << 6)
    };
    *state = UTF8_DECODE_TABLE[256 + *state as usize + char_class as usize];
    *state
}

/// SIMD UTF-8 decoder.
/// 
/// Processes input in 16-byte (SSE) or 32-byte (AVX2) chunks, using SIMD for:
/// - Fast ESC (0x1B) detection
/// - Pure ASCII fast path
/// - Parallel UTF-8 validation and decoding
#[derive(Debug, Default)]
pub struct SimdUtf8Decoder {
    pub state: Utf8State,
}

impl SimdUtf8Decoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.state = Utf8State::default();
    }

    /// Decode UTF-8 bytes until ESC is found.
    /// Returns (bytes_consumed, found_esc).
    /// 
    /// Output codepoints are written to the output buffer as u32 values.
    /// Uses AVX2 (32 bytes at a time) if available, otherwise SSE (16 bytes).
    #[inline]
    pub fn decode_to_esc(&mut self, src: &[u8], output: &mut Vec<u32>) -> (usize, bool) {
        output.clear();
        if src.is_empty() {
            return (0, false);
        }
        output.reserve(src.len());
        
        #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
        {
            let caps = simd_caps();
            // TODO: AVX2 decoder would go here when implemented
            // For now, AVX2 is used for the byte search functions
            if caps.has_sse41 && caps.has_ssse3 {
                // SAFETY: We checked for required SIMD support
                return unsafe { self.decode_to_esc_simd(src, output) };
            }
        }
        
        // Fallback to scalar
        self.decode_to_esc_scalar(src, output)
    }

    /// Scalar fallback decoder.
    fn decode_to_esc_scalar(&mut self, src: &[u8], output: &mut Vec<u32>) -> (usize, bool) {
        let mut pos = 0;
        
        while pos < src.len() {
            let byte = src[pos];
            
            if byte == 0x1B {
                if self.state.cur != UTF8_ACCEPT {
                    output.push(0xFFFD);
                    self.state = Utf8State::default();
                }
                return (pos + 1, true);
            }
            
            pos += 1;
            self.state.prev = self.state.cur;
            
            match decode_utf8_byte(&mut self.state.cur, &mut self.state.codep, byte) {
                UTF8_ACCEPT => {
                    output.push(self.state.codep);
                }
                UTF8_REJECT => {
                    output.push(0xFFFD);
                    let was_accept = self.state.prev == UTF8_ACCEPT;
                    self.state = Utf8State::default();
                    if !was_accept {
                        pos -= 1;
                    }
                }
                _ => {}
            }
        }
        
        (pos, false)
    }

    /// SIMD decoder - processes 16 bytes at a time.
    /// Based on Kitty's simd-string-impl.h
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "sse2", enable = "ssse3", enable = "sse4.1")]
    unsafe fn decode_to_esc_simd(&mut self, src: &[u8], output: &mut Vec<u32>) -> (usize, bool) {
        let mut num_consumed: usize = 0;
        
        // Finish any trailing sequence from previous call
        if self.state.cur != UTF8_ACCEPT {
            num_consumed = self.scalar_decode_to_accept(src, output);
            if num_consumed >= src.len() {
                return (num_consumed, false);
            }
        }
        
        // SIMD constants
        let esc_vec = _mm_set1_epi8(0x1Bu8 as i8);
        let zero = _mm_setzero_si128();
        let one = _mm_set1_epi8(1);
        let two = _mm_set1_epi8(2);
        let three = _mm_set1_epi8(3);
        let four = _mm_set1_epi8(4);
        let numbered = _mm_set_epi8(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
        
        let limit = src.as_ptr().add(src.len());
        let mut p = src.as_ptr().add(num_consumed);
        let mut sentinel_found = false;
        
        while p < limit && !sentinel_found {
            let remaining = limit.offset_from(p) as usize;
            let mut chunk_src_sz = remaining.min(16);
            
            // Load chunk (potentially partial)
            let mut vec = if chunk_src_sz == 16 {
                _mm_loadu_si128(p as *const __m128i)
            } else {
                // Partial load - zero-extend
                let mut buf = [0u8; 16];
                std::ptr::copy_nonoverlapping(p, buf.as_mut_ptr(), chunk_src_sz);
                _mm_loadu_si128(buf.as_ptr() as *const __m128i)
            };
            
            let start_of_current_chunk = p;
            p = p.add(chunk_src_sz);
            
            // Check for ESC
            let esc_cmp = _mm_cmpeq_epi8(vec, esc_vec);
            let num_bytes_to_first_esc = Self::bytes_to_first_match(esc_cmp);
            
            if num_bytes_to_first_esc >= 0 && (num_bytes_to_first_esc as usize) < chunk_src_sz {
                sentinel_found = true;
                chunk_src_sz = num_bytes_to_first_esc as usize;
                num_consumed += chunk_src_sz + 1; // +1 for ESC
                if chunk_src_sz == 0 {
                    continue;
                }
            } else {
                num_consumed += chunk_src_sz;
            }
            
            // Zero out bytes past chunk_src_sz
            if chunk_src_sz < 16 {
                vec = Self::zero_last_n_bytes(vec, 16 - chunk_src_sz);
            }
            
            // Check for trailing incomplete sequence
            let mut num_trailing_bytes = 0usize;
            let mut check_for_trailing = !sentinel_found;
            
            'classification: loop {
                // Check if pure ASCII (no high bits set)
                let ascii_mask = _mm_movemask_epi8(vec);
                if ascii_mask == 0 {
                    // Pure ASCII - fast output
                    Self::output_plain_ascii(vec, chunk_src_sz, output);
                    
                    // Handle trailing bytes
                    if num_trailing_bytes > 0 && p < limit {
                        p = p.sub(num_trailing_bytes);
                    }
                    break 'classification;
                }
                
                // Classify bytes by whether they start 2, 3, or 4 byte sequences
                let state_80 = _mm_set1_epi8(0x80u8 as i8);
                let vec_signed = _mm_add_epi8(vec, state_80);
                
                // state now has 0x80 on all bytes
                let mut state = state_80;
                
                // 2-byte sequence starters (0xC0-0xDF, but 0xC0-0xC1 invalid)
                let c2_start = _mm_cmplt_epi8(_mm_set1_epi8((0xC0 - 1 - 0x80) as i8), vec_signed);
                state = _mm_blendv_epi8(state, _mm_set1_epi8(0xC2u8 as i8), c2_start);
                
                // 3-byte sequence starters (0xE0-0xEF)
                let e3_start = _mm_cmplt_epi8(_mm_set1_epi8((0xE0 - 1 - 0x80) as i8), vec_signed);
                state = _mm_blendv_epi8(state, _mm_set1_epi8(0xE3u8 as i8), e3_start);
                
                // 4-byte sequence starters (0xF0-0xFF, but 0xF5+ invalid)
                let f4_start = _mm_cmplt_epi8(_mm_set1_epi8((0xF0 - 1 - 0x80) as i8), vec_signed);
                state = _mm_blendv_epi8(state, _mm_set1_epi8(0xF4u8 as i8), f4_start);
                
                // mask = upper 5 bits of state (indicates byte type)
                let mask = _mm_and_si128(state, _mm_set1_epi8(0xF8u8 as i8));
                // count = lower 3 bits of state (sequence length)
                let count = _mm_and_si128(state, _mm_set1_epi8(0x07));
                
                // Propagate counts: count[i] = remaining bytes in sequence at position i
                // count_subs1[i] = count[i] - 1, saturating
                let count_subs1 = _mm_subs_epu8(count, one);
                // counts[i] = count[i] + count_subs1[i-1]
                let mut counts = _mm_add_epi8(count, _mm_srli_si128(count_subs1, 1));
                // counts[i] += counts_subs2[i-2] (for 3 and 4 byte sequences)
                counts = _mm_add_epi8(counts, _mm_srli_si128(_mm_subs_epu8(counts, two), 2));
                
                // Check for trailing incomplete sequence
                if check_for_trailing {
                    let last_byte_idx = _mm_set1_epi8((chunk_src_sz - 1) as i8);
                    let at_last_byte = _mm_cmpeq_epi8(numbered, last_byte_idx);
                    let counts_at_last = _mm_and_si128(counts, at_last_byte);
                    let has_trailing = _mm_cmplt_epi8(one, counts_at_last);
                    
                    if _mm_testz_si128(has_trailing, has_trailing) == 0 {
                        // We have a trailing incomplete sequence
                        check_for_trailing = false;
                        
                        let last_byte = *start_of_current_chunk.add(chunk_src_sz - 1);
                        if last_byte >= 0xC0 {
                            num_trailing_bytes = 1;
                        } else if chunk_src_sz > 1 && *start_of_current_chunk.add(chunk_src_sz - 2) >= 0xE0 {
                            num_trailing_bytes = 2;
                        } else if chunk_src_sz > 2 && *start_of_current_chunk.add(chunk_src_sz - 3) >= 0xF0 {
                            num_trailing_bytes = 3;
                        }
                        
                        chunk_src_sz -= num_trailing_bytes;
                        num_consumed -= num_trailing_bytes;
                        
                        if chunk_src_sz == 0 {
                            // Fall back to scalar for trailing bytes
                            let slice = std::slice::from_raw_parts(
                                start_of_current_chunk,
                                num_trailing_bytes
                            );
                            self.scalar_decode_all(slice, output);
                            num_consumed += num_trailing_bytes;
                            break 'classification;
                        }
                        
                        vec = Self::zero_last_n_bytes(vec, 16 - chunk_src_sz);
                        continue 'classification;
                    }
                }
                
                // Validation: ASCII bytes should have counts[i] == 0
                let count_gt_zero = _mm_cmpgt_epi8(counts, zero);
                let count_mask = _mm_movemask_epi8(count_gt_zero);
                if ascii_mask != count_mask {
                    // Invalid UTF-8 - fall back to scalar
                    let slice = std::slice::from_raw_parts(
                        start_of_current_chunk, 
                        chunk_src_sz + num_trailing_bytes
                    );
                    self.scalar_decode_all(slice, output);
                    num_consumed += num_trailing_bytes;
                    break 'classification;
                }
                
                // Build chunk_is_invalid vector
                let mut chunk_invalid = zero;
                
                // Validate 2-byte starters: 0xC0, 0xC1 are invalid
                chunk_invalid = _mm_or_si128(chunk_invalid,
                    _mm_and_si128(c2_start, _mm_cmplt_epi8(vec, _mm_set1_epi8(0xC2u8 as i8))));
                
                // Validate 4-byte starters: 0xF5+ are invalid
                chunk_invalid = _mm_or_si128(chunk_invalid,
                    _mm_and_si128(f4_start, _mm_cmpgt_epi8(vec, _mm_set1_epi8(0xF4u8 as i8))));
                
                // Validate continuation bytes don't have starter bytes
                let cont_has_starter = _mm_andnot_si128(
                    _mm_cmplt_epi8(vec, _mm_set1_epi8(0xC0u8 as i8)),
                    _mm_cmpgt_epi8(counts, count)
                );
                chunk_invalid = _mm_or_si128(chunk_invalid, cont_has_starter);
                
                // Validate E0 second bytes (must be >= 0xA0)
                let e0_starters = _mm_cmpeq_epi8(vec, _mm_set1_epi8(0xE0u8 as i8));
                let e0_followers = _mm_srli_si128(e0_starters, 1);
                let e0_invalid = _mm_and_si128(e0_followers, 
                    _mm_cmplt_epi8(_mm_and_si128(e0_followers, vec), _mm_set1_epi8(0xA0u8 as i8)));
                chunk_invalid = _mm_or_si128(chunk_invalid, e0_invalid);
                
                // Validate ED second bytes (must be < 0xA0, i.e. <= 0x9F)
                let ed_starters = _mm_cmpeq_epi8(vec, _mm_set1_epi8(0xEDu8 as i8));
                let ed_followers = _mm_srli_si128(ed_starters, 1);
                let ed_invalid = _mm_and_si128(ed_followers,
                    _mm_cmpgt_epi8(_mm_and_si128(ed_followers, vec), _mm_set1_epi8(0x9Fu8 as i8)));
                chunk_invalid = _mm_or_si128(chunk_invalid, ed_invalid);
                
                // Validate F0 second bytes (must be >= 0x90)
                let f0_starters = _mm_cmpeq_epi8(vec, _mm_set1_epi8(0xF0u8 as i8));
                let f0_followers = _mm_srli_si128(f0_starters, 1);
                let f0_invalid = _mm_and_si128(f0_followers,
                    _mm_cmplt_epi8(_mm_and_si128(f0_followers, vec), _mm_set1_epi8(0x90u8 as i8)));
                chunk_invalid = _mm_or_si128(chunk_invalid, f0_invalid);
                
                // Validate F4 second bytes (must be < 0x90, i.e. <= 0x8F)
                let f4_starters = _mm_cmpeq_epi8(vec, _mm_set1_epi8(0xF4u8 as i8));
                let f4_followers = _mm_srli_si128(f4_starters, 1);
                let f4_invalid = _mm_and_si128(f4_followers,
                    _mm_cmpgt_epi8(_mm_and_si128(f4_followers, vec), _mm_set1_epi8(0x8Fu8 as i8)));
                chunk_invalid = _mm_or_si128(chunk_invalid, f4_invalid);
                
                // If invalid, fall back to scalar
                if _mm_testz_si128(chunk_invalid, chunk_invalid) == 0 {
                    let slice = std::slice::from_raw_parts(
                        start_of_current_chunk, 
                        chunk_src_sz + num_trailing_bytes
                    );
                    self.scalar_decode_all(slice, output);
                    num_consumed += num_trailing_bytes;
                    break 'classification;
                }
                
                // Mask control bits to get payload only
                vec = _mm_andnot_si128(mask, vec);
                
                // Build output vectors
                let vec_non_ascii = _mm_andnot_si128(_mm_cmpeq_epi8(counts, zero), vec);
                
                // output1: lowest byte of each codepoint
                // For count==1 positions: OR with shifted bits from count==2 position
                let count1_locs = _mm_cmpeq_epi8(counts, one);
                let shifted_6 = _mm_and_si128(
                    _mm_slli_epi16(_mm_srli_si128(vec_non_ascii, 1), 6),
                    _mm_set1_epi8(0xC0u8 as i8)
                );
                let output1 = _mm_blendv_epi8(vec, _mm_or_si128(vec, shifted_6), count1_locs);
                
                // output2: middle byte (for 3 and 4 byte sequences)
                let count2_locs = _mm_cmpeq_epi8(counts, two);
                let count3_locs = _mm_cmpeq_epi8(counts, three);
                let mut output2 = _mm_and_si128(vec, count2_locs);
                output2 = _mm_srli_epi32(output2, 2); // bits 5,4,3,2
                let shifted_4 = _mm_and_si128(
                    _mm_set1_epi8(0xF0u8 as i8),
                    _mm_slli_epi16(_mm_srli_si128(_mm_and_si128(count3_locs, vec_non_ascii), 1), 4)
                );
                output2 = _mm_or_si128(output2, shifted_4);
                output2 = _mm_and_si128(output2, count2_locs);
                output2 = _mm_srli_si128(output2, 1);
                
                // output3: highest byte (for 4 byte sequences)
                let count4_locs = _mm_cmpeq_epi8(counts, four);
                let mut output3 = _mm_and_si128(three, _mm_srli_epi32(vec, 4)); // bits 5,6 from count==3
                let shifted_2 = _mm_and_si128(
                    _mm_set1_epi8(0xFCu8 as i8),
                    _mm_slli_epi16(_mm_srli_si128(_mm_and_si128(count4_locs, vec_non_ascii), 1), 2)
                );
                output3 = _mm_or_si128(output3, shifted_2);
                output3 = _mm_and_si128(output3, count3_locs);
                output3 = _mm_srli_si128(output3, 2);
                
                // Shuffle to remove continuation bytes
                // shifts = number of bytes to skip for each position
                let mut shifts = count_subs1;
                // Propagate shifts: shifts[i] += shifts[i-1] + shifts[i-2] + ...
                shifts = _mm_add_epi8(shifts, _mm_srli_si128(shifts, 1));
                shifts = _mm_add_epi8(shifts, _mm_srli_si128(shifts, 2));
                shifts = _mm_add_epi8(shifts, _mm_srli_si128(shifts, 4));
                shifts = _mm_add_epi8(shifts, _mm_srli_si128(shifts, 8));
                
                // Zero shifts for discarded continuation bytes (where counts >= 2)
                shifts = _mm_and_si128(shifts, _mm_cmplt_epi8(counts, two));
                
                // Move shifts leftward based on bit patterns
                // This is Kitty's move() macro
                shifts = Self::move_shifts_by_1(shifts);
                shifts = Self::move_shifts_by_2(shifts);
                shifts = Self::move_shifts_by_4(shifts);
                shifts = Self::move_shifts_by_8(shifts);
                
                // Add byte numbers to create shuffle mask
                shifts = _mm_add_epi8(shifts, numbered);
                
                // Shuffle the output vectors
                let output1 = _mm_shuffle_epi8(output1, shifts);
                let output2 = _mm_shuffle_epi8(output2, shifts);
                let output3 = _mm_shuffle_epi8(output3, shifts);
                
                // Count discarded bytes to get codepoint count
                let num_discarded = Self::sum_bytes(count_subs1);
                let num_codepoints = chunk_src_sz - num_discarded;
                
                // Output unicode codepoints
                Self::output_unicode(output1, output2, output3, num_codepoints, output);
                
                // Handle trailing bytes
                if num_trailing_bytes > 0 && p < limit {
                    p = p.sub(num_trailing_bytes);
                }
                
                break 'classification;
            }
        }
        
        (num_consumed, sentinel_found)
    }
    
    /// move() macro from Kitty: move shifts leftward based on bit pattern
    /// move(shifts, one_byte, 1)
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "sse2", enable = "ssse3", enable = "sse4.1")]
    #[inline]
    unsafe fn move_shifts_by_1(shifts: __m128i) -> __m128i {
        // blendv_epi8(shifts, shift_left_by_one_byte(shifts), 
        //             shift_left_by_one_byte(shift_left_by_bits16(shifts, 7)))
        let selector = _mm_slli_si128(_mm_slli_epi16(shifts, 7), 1);
        let shifted = _mm_slli_si128(shifts, 1);
        _mm_blendv_epi8(shifts, shifted, selector)
    }
    
    /// move(shifts, two_bytes, 2)
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "sse2", enable = "ssse3", enable = "sse4.1")]
    #[inline]
    unsafe fn move_shifts_by_2(shifts: __m128i) -> __m128i {
        let selector = _mm_slli_si128(_mm_slli_epi16(shifts, 6), 2);
        let shifted = _mm_slli_si128(shifts, 2);
        _mm_blendv_epi8(shifts, shifted, selector)
    }
    
    /// move(shifts, four_bytes, 3)
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "sse2", enable = "ssse3", enable = "sse4.1")]
    #[inline]
    unsafe fn move_shifts_by_4(shifts: __m128i) -> __m128i {
        let selector = _mm_slli_si128(_mm_slli_epi16(shifts, 5), 4);
        let shifted = _mm_slli_si128(shifts, 4);
        _mm_blendv_epi8(shifts, shifted, selector)
    }
    
    /// move(shifts, eight_bytes, 4)
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "sse2", enable = "ssse3", enable = "sse4.1")]
    #[inline]
    unsafe fn move_shifts_by_8(shifts: __m128i) -> __m128i {
        let selector = _mm_slli_si128(_mm_slli_epi16(shifts, 4), 8);
        let shifted = _mm_slli_si128(shifts, 8);
        _mm_blendv_epi8(shifts, shifted, selector)
    }
    
    /// Find first matching byte position, returns -1 if none found
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "sse2", enable = "sse4.1")]
    #[inline]
    unsafe fn bytes_to_first_match(cmp_result: __m128i) -> i32 {
        if _mm_testz_si128(cmp_result, cmp_result) != 0 {
            -1
        } else {
            _mm_movemask_epi8(cmp_result).trailing_zeros() as i32
        }
    }
    
    /// Zero the last n bytes of the vector.
    /// E.g., zero_last_n_bytes(vec, 3) zeros bytes at indices 13, 14, 15.
    /// This matches Kitty's implementation which uses shift_left_by_bytes (actually _mm_srli_si128).
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "sse2")]
    #[inline]
    unsafe fn zero_last_n_bytes(vec: __m128i, n: usize) -> __m128i {
        // Kitty's approach: shift all-ones "left" (toward lower indices) by n bytes
        // This uses _mm_srli_si128 which shifts bytes toward index 0, zeros enter at high indices
        // Result: mask with FF at indices 0..15-n and 00 at indices 16-n..15
        let all_ones = _mm_set1_epi8(-1);
        let mask = match n {
            0 => all_ones,
            1 => _mm_srli_si128(all_ones, 1),
            2 => _mm_srli_si128(all_ones, 2),
            3 => _mm_srli_si128(all_ones, 3),
            4 => _mm_srli_si128(all_ones, 4),
            5 => _mm_srli_si128(all_ones, 5),
            6 => _mm_srli_si128(all_ones, 6),
            7 => _mm_srli_si128(all_ones, 7),
            8 => _mm_srli_si128(all_ones, 8),
            9 => _mm_srli_si128(all_ones, 9),
            10 => _mm_srli_si128(all_ones, 10),
            11 => _mm_srli_si128(all_ones, 11),
            12 => _mm_srli_si128(all_ones, 12),
            13 => _mm_srli_si128(all_ones, 13),
            14 => _mm_srli_si128(all_ones, 14),
            15 => _mm_srli_si128(all_ones, 15),
            _ => _mm_setzero_si128(),
        };
        _mm_and_si128(mask, vec)
    }
    
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "sse2")]
    #[inline]
    unsafe fn sum_bytes(vec: __m128i) -> usize {
        let sum = _mm_sad_epu8(vec, _mm_setzero_si128());
        let lower = _mm_cvtsi128_si32(sum) as usize;
        let upper = _mm_cvtsi128_si32(_mm_srli_si128(sum, 8)) as usize;
        lower + upper
    }
    
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "sse2", enable = "sse4.1")]
    #[inline]
    unsafe fn output_plain_ascii(vec: __m128i, src_sz: usize, output: &mut Vec<u32>) {
        output.reserve(src_sz);
        
        // Process 4 bytes at a time
        let mut v = vec;
        let mut remaining = src_sz;
        
        while remaining > 0 {
            let unpacked = _mm_cvtepu8_epi32(v);
            let to_write = remaining.min(4);
            
            let mut buf = [0u32; 4];
            _mm_storeu_si128(buf.as_mut_ptr() as *mut __m128i, unpacked);
            output.extend_from_slice(&buf[..to_write]);
            
            remaining = remaining.saturating_sub(4);
            v = _mm_srli_si128(v, 4);
        }
    }
    
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    #[target_feature(enable = "sse2", enable = "sse4.1")]
    #[inline]
    unsafe fn output_unicode(
        output1: __m128i,
        output2: __m128i, 
        output3: __m128i,
        num_codepoints: usize,
        output: &mut Vec<u32>
    ) {
        output.reserve(num_codepoints);
        
        let mut o1 = output1;
        let mut o2 = output2;
        let mut o3 = output3;
        let mut remaining = num_codepoints;
        
        while remaining > 0 {
            // Unpack lowest 4 bytes to 4 u32s
            let unpacked1 = _mm_cvtepu8_epi32(o1);
            // Shift right by 1 byte, then unpack - this puts bytes in position 1 of each u32 (bits 8-15)
            let unpacked2 = _mm_cvtepu8_epi32(_mm_srli_si128(o2, 0));
            let unpacked2 = _mm_slli_epi32(unpacked2, 8);
            // Shift right by 2 bytes for output3 - puts bytes in position 2 (bits 16-23)
            let unpacked3 = _mm_cvtepu8_epi32(_mm_srli_si128(o3, 0));
            let unpacked3 = _mm_slli_epi32(unpacked3, 16);
            
            let unpacked = _mm_or_si128(_mm_or_si128(unpacked1, unpacked2), unpacked3);
            
            let to_write = remaining.min(4);
            let mut buf = [0u32; 4];
            _mm_storeu_si128(buf.as_mut_ptr() as *mut __m128i, unpacked);
            output.extend_from_slice(&buf[..to_write]);
            
            remaining = remaining.saturating_sub(4);
            o1 = _mm_srli_si128(o1, 4);
            o2 = _mm_srli_si128(o2, 4);
            o3 = _mm_srli_si128(o3, 4);
        }
    }
    
    /// Scalar decode until state is ACCEPT.
    fn scalar_decode_to_accept(&mut self, src: &[u8], output: &mut Vec<u32>) -> usize {
        let mut pos = 0;
        while pos < src.len() && self.state.cur != UTF8_ACCEPT {
            let byte = src[pos];
            if byte == 0x1B {
                output.push(0xFFFD);
                self.state = Utf8State::default();
                return pos;
            }
            pos += 1;
            self.state.prev = self.state.cur;
            match decode_utf8_byte(&mut self.state.cur, &mut self.state.codep, byte) {
                UTF8_ACCEPT => output.push(self.state.codep),
                UTF8_REJECT => {
                    output.push(0xFFFD);
                    let was_accept = self.state.prev == UTF8_ACCEPT;
                    self.state = Utf8State::default();
                    if !was_accept {
                        pos -= 1;
                    }
                }
                _ => {}
            }
        }
        pos
    }
    
    /// Scalar decode all bytes.
    fn scalar_decode_all(&mut self, src: &[u8], output: &mut Vec<u32>) -> usize {
        let mut pos = 0;
        while pos < src.len() {
            let byte = src[pos];
            if byte == 0x1B {
                if self.state.cur != UTF8_ACCEPT {
                    output.push(0xFFFD);
                    self.state = Utf8State::default();
                }
                return pos;
            }
            pos += 1;
            self.state.prev = self.state.cur;
            match decode_utf8_byte(&mut self.state.cur, &mut self.state.codep, byte) {
                UTF8_ACCEPT => output.push(self.state.codep),
                UTF8_REJECT => {
                    output.push(0xFFFD);
                    let was_accept = self.state.prev == UTF8_ACCEPT;
                    self.state = Utf8State::default();
                    if !was_accept {
                        pos -= 1;
                    }
                }
                _ => {}
            }
        }
        pos
    }
}

/// Convert u32 codepoints to chars.
/// SAFETY: Caller must ensure all codepoints are valid Unicode.
#[inline]
pub fn codepoints_to_chars(codepoints: &[u32], chars: &mut Vec<char>) {
    chars.clear();
    chars.reserve(codepoints.len());
    for &cp in codepoints {
        // SAFETY: The SIMD decoder validates UTF-8, so codepoints are valid
        if cp <= 0x10FFFF && !(0xD800..=0xDFFF).contains(&cp) {
            chars.push(unsafe { char::from_u32_unchecked(cp) });
        } else {
            chars.push('\u{FFFD}');
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_ascii() {
        let mut decoder = SimdUtf8Decoder::new();
        let mut output = Vec::new();
        let input = b"Hello, World!";
        let (consumed, found_esc) = decoder.decode_to_esc(input, &mut output);
        assert_eq!(consumed, 13);
        assert!(!found_esc);
        let chars: Vec<char> = output.iter().filter_map(|&cp| char::from_u32(cp)).collect();
        assert_eq!(chars.iter().collect::<String>(), "Hello, World!");
    }
    
    #[test]
    fn test_with_esc() {
        let mut decoder = SimdUtf8Decoder::new();
        let mut output = Vec::new();
        let input = b"Hello\x1b[0m";
        let (consumed, found_esc) = decoder.decode_to_esc(input, &mut output);
        assert_eq!(consumed, 6); // "Hello" + ESC
        assert!(found_esc);
        let chars: Vec<char> = output.iter().filter_map(|&cp| char::from_u32(cp)).collect();
        assert_eq!(chars.iter().collect::<String>(), "Hello");
    }
    
    #[test]
    fn test_utf8_2byte() {
        let mut decoder = SimdUtf8Decoder::new();
        let mut output = Vec::new();
        let input = "café".as_bytes();
        let (consumed, found_esc) = decoder.decode_to_esc(input, &mut output);
        assert_eq!(consumed, 5); // c, a, f, é (2 bytes)
        assert!(!found_esc);
        let chars: Vec<char> = output.iter().filter_map(|&cp| char::from_u32(cp)).collect();
        assert_eq!(chars.iter().collect::<String>(), "café");
    }
    
    #[test]
    fn test_utf8_3byte() {
        let mut decoder = SimdUtf8Decoder::new();
        let mut output = Vec::new();
        let input = "日本語".as_bytes();
        let (consumed, found_esc) = decoder.decode_to_esc(input, &mut output);
        assert_eq!(consumed, 9); // 3 chars * 3 bytes
        assert!(!found_esc);
        let chars: Vec<char> = output.iter().filter_map(|&cp| char::from_u32(cp)).collect();
        assert_eq!(chars.iter().collect::<String>(), "日本語");
    }
    
    #[test]
    fn test_utf8_4byte() {
        let mut decoder = SimdUtf8Decoder::new();
        let mut output = Vec::new();
        let input = "🎉🚀".as_bytes();
        let (consumed, found_esc) = decoder.decode_to_esc(input, &mut output);
        assert_eq!(consumed, 8); // 2 chars * 4 bytes
        assert!(!found_esc);
        let chars: Vec<char> = output.iter().filter_map(|&cp| char::from_u32(cp)).collect();
        assert_eq!(chars.iter().collect::<String>(), "🎉🚀");
    }
    
    #[test]
    fn test_invalid_utf8() {
        let mut decoder = SimdUtf8Decoder::new();
        let mut output = Vec::new();
        let input = b"\xff\xfe";
        let (consumed, _) = decoder.decode_to_esc(input, &mut output);
        assert_eq!(consumed, 2);
        // Should have replacement characters
        assert!(output.iter().any(|&cp| cp == 0xFFFD));
    }
    
    // ========================================================================
    // Tests for find_byte
    // ========================================================================
    
    #[test]
    fn test_find_byte_first() {
        let haystack = b"hello world";
        assert_eq!(find_byte(haystack, b'h'), Some(0));
    }
    
    #[test]
    fn test_find_byte_middle() {
        let haystack = b"hello world";
        assert_eq!(find_byte(haystack, b'w'), Some(6));
    }
    
    #[test]
    fn test_find_byte_not_found() {
        let haystack = b"hello world";
        assert_eq!(find_byte(haystack, b'x'), None);
    }
    
    #[test]
    fn test_find_byte_long() {
        // Test with > 32 bytes to exercise AVX2 path
        let mut haystack = vec![b'a'; 100];
        haystack[75] = b'Z';
        assert_eq!(find_byte(&haystack, b'Z'), Some(75));
    }
    
    // ========================================================================
    // Tests for find_either_of_two_bytes
    // ========================================================================
    
    #[test]
    fn test_find_either_of_two_bytes_first() {
        let haystack = b"hello world";
        assert_eq!(find_either_of_two_bytes(haystack, b'h', b'x'), Some(0));
    }
    
    #[test]
    fn test_find_either_of_two_bytes_second() {
        let haystack = b"hello world";
        assert_eq!(find_either_of_two_bytes(haystack, b'x', b'h'), Some(0));
    }
    
    #[test]
    fn test_find_either_of_two_bytes_middle() {
        let haystack = b"hello world";
        assert_eq!(find_either_of_two_bytes(haystack, b'w', b'o'), Some(4)); // first 'o' at index 4
    }
    
    #[test]
    fn test_find_either_of_two_bytes_not_found() {
        let haystack = b"hello world";
        assert_eq!(find_either_of_two_bytes(haystack, b'x', b'y'), None);
    }
    
    #[test]
    fn test_find_either_of_two_bytes_esc() {
        let haystack = b"hello\x1bworld";
        assert_eq!(find_either_of_two_bytes(haystack, 0x1B, b'\n'), Some(5));
    }
    
    #[test]
    fn test_find_either_of_two_bytes_long() {
        // Test with > 32 bytes to exercise AVX2 path
        let mut haystack = vec![b'a'; 100];
        haystack[50] = b'X';
        assert_eq!(find_either_of_two_bytes(&haystack, b'X', b'Y'), Some(50));
    }
    
    #[test]
    fn test_find_either_of_two_bytes_empty() {
        let haystack = b"";
        assert_eq!(find_either_of_two_bytes(haystack, b'a', b'b'), None);
    }
    
    // ========================================================================
    // Tests for find_c0_control
    // ========================================================================
    
    #[test]
    fn test_find_c0_control_newline() {
        let haystack = b"hello\nworld";
        assert_eq!(find_c0_control(haystack), Some(5));
    }
    
    #[test]
    fn test_find_c0_control_tab() {
        let haystack = b"hello\tworld";
        assert_eq!(find_c0_control(haystack), Some(5));
    }
    
    #[test]
    fn test_find_c0_control_del() {
        let haystack = b"hello\x7fworld";
        assert_eq!(find_c0_control(haystack), Some(5));
    }
    
    #[test]
    fn test_find_c0_control_bell() {
        let haystack = b"hello\x07world";
        assert_eq!(find_c0_control(haystack), Some(5));
    }
    
    #[test]
    fn test_find_c0_control_esc() {
        let haystack = b"hello\x1bworld";
        assert_eq!(find_c0_control(haystack), Some(5));
    }
    
    #[test]
    fn test_find_c0_control_none() {
        let haystack = b"hello world!";
        assert_eq!(find_c0_control(haystack), None);
    }
    
    #[test]
    fn test_find_c0_control_long() {
        // Test with > 32 bytes to exercise AVX2 path
        let mut haystack = vec![b'a'; 100];
        haystack[60] = b'\n';
        assert_eq!(find_c0_control(&haystack), Some(60));
    }
    
    #[test]
    fn test_find_c0_control_at_start() {
        let haystack = b"\x00hello";
        assert_eq!(find_c0_control(haystack), Some(0));
    }
    
    // ========================================================================
    // Tests for xor_mask
    // ========================================================================
    
    #[test]
    fn test_xor_mask_basic() {
        let mut data = vec![0u8; 8];
        let mask = [0x12, 0x34, 0x56, 0x78];
        xor_mask(&mut data, mask, 0);
        assert_eq!(data, vec![0x12, 0x34, 0x56, 0x78, 0x12, 0x34, 0x56, 0x78]);
    }
    
    #[test]
    fn test_xor_mask_offset() {
        let mut data = vec![0u8; 8];
        let mask = [0x12, 0x34, 0x56, 0x78];
        xor_mask(&mut data, mask, 1);
        // Starting at offset 1: 0x34, 0x56, 0x78, 0x12, 0x34, ...
        assert_eq!(data, vec![0x34, 0x56, 0x78, 0x12, 0x34, 0x56, 0x78, 0x12]);
    }
    
    #[test]
    fn test_xor_mask_roundtrip() {
        let original = b"Hello, World!".to_vec();
        let mut data = original.clone();
        let mask = [0xAB, 0xCD, 0xEF, 0x01];
        
        // XOR once
        xor_mask(&mut data, mask, 0);
        assert_ne!(data, original);
        
        // XOR again to get back original
        xor_mask(&mut data, mask, 0);
        assert_eq!(data, original);
    }
    
    #[test]
    fn test_xor_mask_long() {
        // Test with > 32 bytes to exercise AVX2 path
        let mut data = vec![0xFFu8; 100];
        let mask = [0x12, 0x34, 0x56, 0x78];
        xor_mask(&mut data, mask, 0);
        
        // Verify pattern
        for (i, &byte) in data.iter().enumerate() {
            assert_eq!(byte, 0xFF ^ mask[i % 4]);
        }
    }
    
    #[test]
    fn test_xor_mask_empty() {
        let mut data: Vec<u8> = vec![];
        let mask = [0x12, 0x34, 0x56, 0x78];
        let result = xor_mask(&mut data, mask, 0);
        assert_eq!(result, 0);
    }
}
