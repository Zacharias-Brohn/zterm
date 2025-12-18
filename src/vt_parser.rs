//! VT Parser - A high-performance terminal escape sequence parser.
//! 
//! Based on Kitty's vt-parser.c design, this parser uses explicit state tracking
//! to enable fast-path processing of normal text while correctly handling
//! escape sequences.
//!
//! Key design principles from Kitty:
//! 1. UTF-8 decode until ESC sentinel is found (not byte-by-byte parsing)
//! 2. Pass decoded codepoints to the text handler, not raw bytes
//! 3. Control characters (LF, CR, TAB, BS, etc.) are handled inline in text drawing
//! 4. Only ESC triggers state machine transitions
//! 5. Use SIMD-accelerated byte search for finding escape sequence terminators

/// Maximum number of CSI parameters.
pub const MAX_CSI_PARAMS: usize = 256;

/// Maximum length of an OSC string (same as escape length - no separate limit needed).
/// Kitty doesn't have a separate OSC limit, just the overall escape sequence limit.
const MAX_OSC_LEN: usize = 262144; // 256KB, same as MAX_ESCAPE_LEN

/// Maximum length of an escape sequence before we give up.
const MAX_ESCAPE_LEN: usize = 262144; // 256KB like Kitty

/// Replacement character for invalid UTF-8.
const REPLACEMENT_CHAR: char = '\u{FFFD}';

/// UTF-8 decoder states (DFA-based, like Kitty uses).
const UTF8_ACCEPT: u8 = 0;
const UTF8_REJECT: u8 = 12;

/// UTF-8 state transition and character class tables.
/// Based on Bjoern Hoehrmann's DFA decoder.
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
#[inline]
fn decode_utf8(state: &mut u8, codep: &mut u32, byte: u8) -> u8 {
    let char_class = UTF8_DECODE_TABLE[byte as usize];
    *codep = if *state == UTF8_ACCEPT {
        (0xFF >> char_class) as u32 & byte as u32
    } else {
        (byte as u32 & 0x3F) | (*codep << 6)
    };
    *state = UTF8_DECODE_TABLE[256 + *state as usize + char_class as usize];
    *state
}

/// UTF-8 decoder that decodes until ESC (0x1B) is found.
/// Returns (output_chars, bytes_consumed, found_esc).
#[derive(Debug, Default)]
pub struct Utf8Decoder {
    state: u8,
    codep: u32,
}

impl Utf8Decoder {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn reset(&mut self) {
        self.state = UTF8_ACCEPT;
        self.codep = 0;
    }

    /// Decode UTF-8 bytes until ESC is found.
    /// Outputs decoded codepoints to the output buffer.
    /// Returns (bytes_consumed, found_esc).
    #[inline]
    pub fn decode_to_esc(&mut self, src: &[u8], output: &mut Vec<char>) -> (usize, bool) {
        output.clear();
        // Pre-allocate capacity to avoid reallocations during decode.
        // Worst case: one char per byte (ASCII). Kitty does the same.
        output.reserve(src.len());
        let mut consumed = 0;
        
        for &byte in src {
            consumed += 1;
            
            if byte == 0x1B {
                // ESC found - emit replacement if we were in the middle of a sequence
                if self.state != UTF8_ACCEPT {
                    output.push(REPLACEMENT_CHAR);
                }
                self.reset();
                return (consumed, true);
            }
            
            let prev_state = self.state;
            match decode_utf8(&mut self.state, &mut self.codep, byte) {
                UTF8_ACCEPT => {
                    // SAFETY: The DFA decoder guarantees valid Unicode codepoints when
                    // state is ACCEPT. This is the same guarantee that Kitty relies on.
                    // Using unchecked avoids a redundant validity check in the hot path.
                    let c = unsafe { char::from_u32_unchecked(self.codep) };
                    output.push(c);
                }
                UTF8_REJECT => {
                    // Invalid UTF-8 sequence
                    output.push(REPLACEMENT_CHAR);
                    self.state = UTF8_ACCEPT;
                    // If previous state was accept, we consumed a bad lead byte
                    // Otherwise, re-process this byte as a potential new sequence start
                    if prev_state != UTF8_ACCEPT {
                        consumed -= 1;
                        continue;
                    }
                }
                _ => {
                    // Continue accumulating multi-byte sequence
                }
            }
        }
        
        (consumed, false)
    }
}

/// Parser state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    /// Normal text processing mode.
    Normal,
    /// Just saw ESC, waiting for next character.
    Escape,
    /// ESC seen, waiting for second char of two-char sequence (e.g., ESC ( B).
    EscapeIntermediate(u8),
    /// Processing CSI sequence (ESC [).
    Csi,
    /// Processing OSC sequence (ESC ]).
    Osc,
    /// Processing DCS sequence (ESC P).
    Dcs,
    /// Processing APC sequence (ESC _).
    Apc,
    /// Processing PM sequence (ESC ^).
    Pm,
    /// Processing SOS sequence (ESC X).
    Sos,
}

impl Default for State {
    fn default() -> Self {
        State::Normal
    }
}

/// CSI parsing sub-state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum CsiState {
    #[default]
    Start,
    Body,
    PostSecondary,
}

/// Parsed CSI sequence data.
#[derive(Debug, Clone)]
pub struct CsiParams {
    /// Collected parameters.
    pub params: [i32; MAX_CSI_PARAMS],
    /// Which parameters are sub-parameters (colon-separated).
    pub is_sub_param: [bool; MAX_CSI_PARAMS],
    /// Number of collected parameters.
    pub num_params: usize,
    /// Primary modifier (e.g., '?' in CSI ? Ps h).
    pub primary: u8,
    /// Secondary modifier (e.g., '$' in CSI Ps $ p).
    pub secondary: u8,
    /// Final character (e.g., 'm' in CSI 1 m).
    pub final_char: u8,
    /// Whether the sequence is valid.
    pub is_valid: bool,
    // Internal parsing state
    state: CsiState,
    accumulator: i64,
    multiplier: i32,
    num_digits: usize,
}

impl Default for CsiParams {
    fn default() -> Self {
        Self {
            params: [0; MAX_CSI_PARAMS],
            is_sub_param: [false; MAX_CSI_PARAMS],
            num_params: 0,
            primary: 0,
            secondary: 0,
            final_char: 0,
            is_valid: false,
            state: CsiState::Start,
            accumulator: 0,
            multiplier: 1,
            num_digits: 0,
        }
    }
}

impl CsiParams {
    /// Reset for a new CSI sequence.
    /// Note: We don't zero the params/is_sub_param arrays since they're written before being read.
    /// This avoids zeroing 1280 bytes on every CSI sequence.
    #[inline]
    pub fn reset(&mut self) {
        // Don't zero arrays - individual elements are written before being read
        // self.params = [0; MAX_CSI_PARAMS];      // Skip - saves 1024 bytes memset
        // self.is_sub_param = [false; MAX_CSI_PARAMS]; // Skip - saves 256 bytes memset
        self.num_params = 0;
        self.primary = 0;
        self.secondary = 0;
        self.final_char = 0;
        self.is_valid = false;
        self.state = CsiState::Start;
        self.accumulator = 0;
        self.multiplier = 1;
        self.num_digits = 0;
    }

    /// Get parameter at index, or default value if not present.
    #[inline]
    pub fn get(&self, index: usize, default: i32) -> i32 {
        if index < self.num_params && self.params[index] != 0 {
            self.params[index]
        } else {
            default
        }
    }

    /// Add a digit to the current parameter.
    #[inline]
    fn add_digit(&mut self, digit: u8) {
        self.accumulator = self.accumulator.saturating_mul(10).saturating_add((digit - b'0') as i64);
        self.num_digits += 1;
    }

    /// Commit the current parameter.
    fn commit_param(&mut self) -> bool {
        if self.num_params >= MAX_CSI_PARAMS {
            return false;
        }
        let value = (self.accumulator as i32).saturating_mul(self.multiplier);
        self.params[self.num_params] = value;
        self.num_params += 1;
        self.accumulator = 0;
        self.multiplier = 1;
        self.num_digits = 0;
        true
    }
}

/// VT Parser with Kitty-style state tracking.
#[derive(Debug)]
pub struct Parser {
    /// Current parser state.
    pub state: State,
    /// CSI parameters being collected.
    pub csi: CsiParams,
    /// UTF-8 decoder for text.
    utf8: Utf8Decoder,
    /// Decoded character buffer (reused to avoid allocation).
    char_buf: Vec<char>,
    /// OSC string buffer.
    osc_buffer: Vec<u8>,
    /// DCS/APC/PM/SOS string buffer.
    string_buffer: Vec<u8>,
    /// Intermediate byte for two-char escape sequences.
    intermediate: u8,
    /// Number of bytes consumed in current escape sequence (for max length check).
    escape_len: usize,
}

impl Default for Parser {
    fn default() -> Self {
        Self {
            state: State::Normal,
            csi: CsiParams::default(),
            utf8: Utf8Decoder::new(),
            // Pre-allocate to match typical read buffer sizes (1MB) to avoid reallocation
            char_buf: Vec::with_capacity(1024 * 1024),
            osc_buffer: Vec::new(),
            string_buffer: Vec::new(),
            intermediate: 0,
            escape_len: 0,
        }
    }
}

impl Parser {
    /// Create a new parser.
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if parser is in normal (ground) state.
    #[inline]
    pub fn is_normal(&self) -> bool {
        self.state == State::Normal
    }

    /// Reset parser to normal state.
    pub fn reset(&mut self) {
        self.state = State::Normal;
        self.csi.reset();
        self.utf8.reset();
        self.char_buf.clear();
        self.osc_buffer.clear();
        self.string_buffer.clear();
        self.intermediate = 0;
        self.escape_len = 0;
    }

    /// Process a buffer of bytes, calling the handler for each action.
    /// Returns the number of bytes consumed.
    pub fn parse<H: Handler>(&mut self, bytes: &[u8], handler: &mut H) -> usize {
        let mut pos = 0;
        
        while pos < bytes.len() {
            match self.state {
                State::Normal => {
                    // Fast path: UTF-8 decode until ESC
                    let (consumed, found_esc) = self.utf8.decode_to_esc(&bytes[pos..], &mut self.char_buf);
                    
                    // Process decoded characters (text + control chars)
                    if !self.char_buf.is_empty() {
                        handler.text(&self.char_buf);
                    }
                    
                    pos += consumed;
                    
                    if found_esc {
                        self.state = State::Escape;
                        self.escape_len = 0;
                    }
                }
                State::Escape => {
                    pos += self.consume_escape(bytes, pos, handler);
                }
                State::EscapeIntermediate(_) => {
                    pos += self.consume_escape_intermediate(bytes, pos, handler);
                }
                State::Csi => {
                    pos += self.consume_csi(bytes, pos, handler);
                }
                State::Osc => {
                    pos += self.consume_osc(bytes, pos, handler);
                }
                State::Dcs | State::Apc | State::Pm | State::Sos => {
                    pos += self.consume_string_command(bytes, pos, handler);
                }
            }
        }
        
        pos
    }

    /// Process bytes after ESC.
    fn consume_escape<H: Handler>(&mut self, bytes: &[u8], pos: usize, handler: &mut H) -> usize {
        if pos >= bytes.len() {
            return 0;
        }
        
        let ch = bytes[pos];
        self.escape_len += 1;
        
        match ch {
            // CSI: ESC [
            b'[' => {
                self.state = State::Csi;
                self.csi.reset();
                1
            }
            // OSC: ESC ]
            b']' => {
                self.state = State::Osc;
                self.osc_buffer.clear();
                1
            }
            // DCS: ESC P
            b'P' => {
                self.state = State::Dcs;
                self.string_buffer.clear();
                1
            }
            // APC: ESC _
            b'_' => {
                self.state = State::Apc;
                self.string_buffer.clear();
                1
            }
            // PM: ESC ^
            b'^' => {
                self.state = State::Pm;
                self.string_buffer.clear();
                1
            }
            // SOS: ESC X
            b'X' => {
                self.state = State::Sos;
                self.string_buffer.clear();
                1
            }
            // Two-char sequences: ESC ( ESC ) ESC # ESC % ESC SP etc.
            b'(' | b')' | b'*' | b'+' | b'-' | b'.' | b'/' | b'%' | b'#' | b' ' => {
                self.state = State::EscapeIntermediate(ch);
                self.intermediate = ch;
                1
            }
            // Single-char escape sequences
            b'7' => {
                // DECSC - Save cursor
                handler.save_cursor();
                self.state = State::Normal;
                1
            }
            b'8' => {
                // DECRC - Restore cursor
                handler.restore_cursor();
                self.state = State::Normal;
                1
            }
            b'c' => {
                // RIS - Full reset
                handler.reset();
                self.state = State::Normal;
                1
            }
            b'D' => {
                // IND - Index (move down, scroll if needed)
                handler.index();
                self.state = State::Normal;
                1
            }
            b'E' => {
                // NEL - Next line
                handler.newline();
                self.state = State::Normal;
                1
            }
            b'H' => {
                // HTS - Horizontal tab set
                handler.set_tab_stop();
                self.state = State::Normal;
                1
            }
            b'M' => {
                // RI - Reverse index
                handler.reverse_index();
                self.state = State::Normal;
                1
            }
            b'=' => {
                // DECKPAM - Application keypad mode
                handler.set_keypad_mode(true);
                self.state = State::Normal;
                1
            }
            b'>' => {
                // DECKPNM - Normal keypad mode
                handler.set_keypad_mode(false);
                self.state = State::Normal;
                1
            }
            b'\\' => {
                // ST - String terminator (ignore if not in string mode)
                self.state = State::Normal;
                1
            }
            _ => {
                // Unknown escape sequence, ignore and return to normal
                log::debug!("Unknown escape sequence: ESC {:02x}", ch);
                self.state = State::Normal;
                1
            }
        }
    }

    /// Process second byte of two-char escape sequence.
    fn consume_escape_intermediate<H: Handler>(&mut self, bytes: &[u8], pos: usize, handler: &mut H) -> usize {
        if pos >= bytes.len() {
            return 0;
        }
        
        let ch = bytes[pos];
        let intermediate = self.intermediate;
        self.escape_len += 1;
        self.state = State::Normal;
        
        match intermediate {
            b'(' | b')' => {
                // Designate character set G0/G1
                let set = if intermediate == b'(' { 0 } else { 1 };
                handler.designate_charset(set, ch);
            }
            b'#' => {
                if ch == b'8' {
                    // DECALN - Screen alignment test
                    handler.screen_alignment();
                }
            }
            b'%' => {
                // Character set selection (we always use UTF-8)
            }
            b' ' => {
                // S7C1T / S8C1T - we ignore these
            }
            _ => {}
        }
        
        1
    }

    /// Process CSI sequence bytes.
    fn consume_csi<H: Handler>(&mut self, bytes: &[u8], pos: usize, handler: &mut H) -> usize {
        let mut consumed = 0;
        
        while pos + consumed < bytes.len() {
            let ch = bytes[pos + consumed];
            consumed += 1;
            self.escape_len += 1;
            
            // Check for max length
            if self.escape_len > MAX_ESCAPE_LEN {
                log::debug!("CSI sequence too long, aborting");
                self.state = State::Normal;
                return consumed;
            }
            
            match self.csi.state {
                CsiState::Start => {
                    match ch {
                        // Control characters embedded in CSI - handle them
                        0x00..=0x1F => {
                            // Handle control chars (except ESC which would be weird here)
                            if ch != 0x1B {
                                handler.control(ch);
                            }
                        }
                        b';' => {
                            // Empty parameter = 0
                            self.csi.params[self.csi.num_params] = 0;
                            self.csi.num_params += 1;
                            self.csi.state = CsiState::Body;
                        }
                        b'0'..=b'9' => {
                            self.csi.add_digit(ch);
                            self.csi.state = CsiState::Body;
                        }
                        b'?' | b'>' | b'<' | b'=' => {
                            self.csi.primary = ch;
                            self.csi.state = CsiState::Body;
                        }
                        b' ' | b'\'' | b'"' | b'!' | b'$' | b'#' | b'*' => {
                            self.csi.secondary = ch;
                            self.csi.state = CsiState::PostSecondary;
                        }
                        b'-' => {
                            self.csi.multiplier = -1;
                            self.csi.num_digits = 1;
                            self.csi.state = CsiState::Body;
                        }
                        // Final byte
                        b'@'..=b'~' => {
                            self.csi.final_char = ch;
                            self.csi.is_valid = true;
                            self.dispatch_csi(handler);
                            self.state = State::Normal;
                            return consumed;
                        }
                        _ => {
                            log::debug!("Invalid CSI character: {:02x}", ch);
                            self.state = State::Normal;
                            return consumed;
                        }
                    }
                }
                CsiState::Body => {
                    match ch {
                        0x00..=0x1F => {
                            if ch != 0x1B {
                                handler.control(ch);
                            }
                        }
                        b'0'..=b'9' => {
                            self.csi.add_digit(ch);
                        }
                        b';' => {
                            if self.csi.num_digits == 0 {
                                self.csi.num_digits = 1; // Empty = 0
                            }
                            if !self.csi.commit_param() {
                                self.state = State::Normal;
                                return consumed;
                            }
                            self.csi.is_sub_param[self.csi.num_params] = false;
                        }
                        b':' => {
                            if !self.csi.commit_param() {
                                self.state = State::Normal;
                                return consumed;
                            }
                            self.csi.is_sub_param[self.csi.num_params] = true;
                        }
                        b' ' | b'\'' | b'"' | b'!' | b'$' | b'#' | b'*' => {
                            if !self.csi.commit_param() {
                                self.state = State::Normal;
                                return consumed;
                            }
                            self.csi.secondary = ch;
                            self.csi.state = CsiState::PostSecondary;
                        }
                        b'-' if self.csi.num_digits == 0 => {
                            self.csi.multiplier = -1;
                            self.csi.num_digits = 1;
                        }
                        // Final byte
                        b'@'..=b'~' => {
                            if self.csi.num_digits > 0 || self.csi.num_params > 0 {
                                self.csi.commit_param();
                            }
                            self.csi.final_char = ch;
                            self.csi.is_valid = true;
                            self.dispatch_csi(handler);
                            self.state = State::Normal;
                            return consumed;
                        }
                        _ => {
                            log::debug!("Invalid CSI body character: {:02x}", ch);
                            self.state = State::Normal;
                            return consumed;
                        }
                    }
                }
                CsiState::PostSecondary => {
                    match ch {
                        0x00..=0x1F => {
                            if ch != 0x1B {
                                handler.control(ch);
                            }
                        }
                        // Final byte
                        b'@'..=b'~' => {
                            self.csi.final_char = ch;
                            self.csi.is_valid = true;
                            self.dispatch_csi(handler);
                            self.state = State::Normal;
                            return consumed;
                        }
                        _ => {
                            log::debug!("Invalid CSI post-secondary character: {:02x}", ch);
                            self.state = State::Normal;
                            return consumed;
                        }
                    }
                }
            }
        }
        
        consumed
    }

    /// Dispatch a complete CSI sequence to the handler.
    fn dispatch_csi<H: Handler>(&mut self, handler: &mut H) {
        handler.csi(&self.csi);
    }

    /// Process OSC sequence bytes using SIMD-accelerated terminator search.
    /// Like Kitty's find_st_terminator + accumulate_st_terminated_esc_code.
    fn consume_osc<H: Handler>(&mut self, bytes: &[u8], pos: usize, handler: &mut H) -> usize {
        let remaining = &bytes[pos..];
        
        // Use SIMD-accelerated search to find BEL (0x07), ESC (0x1B), or C1 ST (0x9C)
        // memchr2 finds either of two bytes; we check ESC specially for ESC \ sequence
        // First, try to find BEL or C1 ST (the simple terminators)
        if let Some(term_pos) = memchr::memchr3(0x07, 0x1B, 0x9C, remaining) {
            let terminator = remaining[term_pos];
            
            // Check max length before accepting
            if self.escape_len + term_pos > MAX_ESCAPE_LEN || self.osc_buffer.len() + term_pos > MAX_OSC_LEN {
                log::debug!("OSC sequence too long, aborting");
                self.state = State::Normal;
                return remaining.len();
            }
            
            match terminator {
                0x07 => {
                    // BEL terminator - copy data in bulk and dispatch
                    self.osc_buffer.extend_from_slice(&remaining[..term_pos]);
                    handler.osc(&self.osc_buffer);
                    self.state = State::Normal;
                    self.escape_len += term_pos + 1;
                    return term_pos + 1;
                }
                0x9C => {
                    // C1 ST terminator - copy data in bulk and dispatch
                    self.osc_buffer.extend_from_slice(&remaining[..term_pos]);
                    handler.osc(&self.osc_buffer);
                    self.state = State::Normal;
                    self.escape_len += term_pos + 1;
                    return term_pos + 1;
                }
                0x1B => {
                    // ESC found - check if followed by \ for ST
                    if term_pos + 1 < remaining.len() && remaining[term_pos + 1] == b'\\' {
                        // ESC \ (ST) terminator
                        self.osc_buffer.extend_from_slice(&remaining[..term_pos]);
                        handler.osc(&self.osc_buffer);
                        self.state = State::Normal;
                        self.escape_len += term_pos + 2;
                        return term_pos + 2;
                    } else if term_pos + 1 < remaining.len() {
                        // ESC not followed by \ - this is a new escape sequence
                        // Copy everything before ESC and transition to Escape state
                        self.osc_buffer.extend_from_slice(&remaining[..term_pos]);
                        handler.osc(&self.osc_buffer);
                        self.state = State::Escape;
                        self.escape_len += term_pos + 1;
                        return term_pos + 1;
                    } else {
                        // ESC at end of buffer, need more data
                        // Copy everything before ESC, keep ESC for next parse
                        self.osc_buffer.extend_from_slice(&remaining[..term_pos]);
                        self.escape_len += term_pos;
                        return term_pos;
                    }
                }
                _ => unreachable!(),
            }
        } else {
            // No terminator found - check max length
            if self.escape_len + remaining.len() > MAX_ESCAPE_LEN || self.osc_buffer.len() + remaining.len() > MAX_OSC_LEN {
                log::debug!("OSC sequence too long, aborting");
                self.state = State::Normal;
                return remaining.len();
            }
            
            // Buffer all remaining bytes for next parse call
            self.osc_buffer.extend_from_slice(remaining);
            self.escape_len += remaining.len();
            return remaining.len();
        }
    }

    /// Process DCS/APC/PM/SOS sequence bytes using SIMD-accelerated terminator search.
    /// Like Kitty's find_st_terminator + accumulate_st_terminated_esc_code.
    fn consume_string_command<H: Handler>(&mut self, bytes: &[u8], pos: usize, handler: &mut H) -> usize {
        let remaining = &bytes[pos..];
        
        // Use SIMD-accelerated search to find ESC (0x1B) or C1 ST (0x9C)
        if let Some(term_pos) = memchr::memchr2(0x1B, 0x9C, remaining) {
            let terminator = remaining[term_pos];
            
            // Check max length before accepting
            if self.escape_len + term_pos > MAX_ESCAPE_LEN {
                log::debug!("String command too long, aborting");
                self.state = State::Normal;
                return remaining.len();
            }
            
            match terminator {
                0x9C => {
                    // C1 ST terminator - copy data in bulk and dispatch
                    self.string_buffer.extend_from_slice(&remaining[..term_pos]);
                    match self.state {
                        State::Dcs => handler.dcs(&self.string_buffer),
                        State::Apc => handler.apc(&self.string_buffer),
                        State::Pm => handler.pm(&self.string_buffer),
                        State::Sos => handler.sos(&self.string_buffer),
                        _ => {}
                    }
                    self.state = State::Normal;
                    self.escape_len += term_pos + 1;
                    return term_pos + 1;
                }
                0x1B => {
                    // ESC found - check if followed by \ for ST
                    if term_pos + 1 < remaining.len() && remaining[term_pos + 1] == b'\\' {
                        // ESC \ (ST) terminator
                        self.string_buffer.extend_from_slice(&remaining[..term_pos]);
                        match self.state {
                            State::Dcs => handler.dcs(&self.string_buffer),
                            State::Apc => handler.apc(&self.string_buffer),
                            State::Pm => handler.pm(&self.string_buffer),
                            State::Sos => handler.sos(&self.string_buffer),
                            _ => {}
                        }
                        self.state = State::Normal;
                        self.escape_len += term_pos + 2;
                        return term_pos + 2;
                    } else if term_pos + 1 < remaining.len() {
                        // ESC not followed by \ - include ESC in data and continue
                        // (Unlike OSC, string commands include raw ESC that isn't ST)
                        self.string_buffer.extend_from_slice(&remaining[..=term_pos]);
                        self.escape_len += term_pos + 1;
                        // Continue searching from after this ESC
                        let consumed = term_pos + 1;
                        return consumed + self.consume_string_command(bytes, pos + consumed, handler);
                    } else {
                        // ESC at end of buffer, need more data
                        // Copy everything before ESC, keep ESC for next parse
                        self.string_buffer.extend_from_slice(&remaining[..term_pos]);
                        self.escape_len += term_pos;
                        return term_pos;
                    }
                }
                _ => unreachable!(),
            }
        } else {
            // No terminator found - check max length
            if self.escape_len + remaining.len() > MAX_ESCAPE_LEN {
                log::debug!("String command too long, aborting");
                self.state = State::Normal;
                return remaining.len();
            }
            
            // Buffer all remaining bytes for next parse call
            self.string_buffer.extend_from_slice(remaining);
            self.escape_len += remaining.len();
            return remaining.len();
        }
    }
}

/// Handler trait for responding to parsed escape sequences.
/// 
/// Unlike the vte crate's Perform trait, this trait receives decoded characters
/// (not bytes) for text, and control characters are expected to be handled
/// inline in the text() method (like Kitty does).
pub trait Handler {
    /// Handle a chunk of decoded text (Unicode codepoints).
    /// 
    /// This includes control characters (0x00-0x1F except ESC).
    /// The handler should process control chars like:
    /// - LF (0x0A), VT (0x0B), FF (0x0C): line feed
    /// - CR (0x0D): carriage return
    /// - HT (0x09): tab
    /// - BS (0x08): backspace
    /// - BEL (0x07): bell
    /// 
    /// ESC is never passed to this method - it triggers state transitions.
    fn text(&mut self, chars: &[char]);
    
    /// Handle a single control character embedded in a CSI/OSC sequence.
    /// This is called for control chars (0x00-0x1F) that appear inside
    /// escape sequences, which should still be processed.
    fn control(&mut self, byte: u8);
    
    /// Handle a complete CSI sequence.
    fn csi(&mut self, params: &CsiParams);
    
    /// Handle a complete OSC sequence.
    fn osc(&mut self, data: &[u8]);
    
    /// Handle a DCS sequence.
    fn dcs(&mut self, _data: &[u8]) {}
    
    /// Handle an APC sequence.
    fn apc(&mut self, _data: &[u8]) {}
    
    /// Handle a PM sequence.
    fn pm(&mut self, _data: &[u8]) {}
    
    /// Handle a SOS sequence.
    fn sos(&mut self, _data: &[u8]) {}
    
    /// Save cursor position (DECSC).
    fn save_cursor(&mut self) {}
    
    /// Restore cursor position (DECRC).
    fn restore_cursor(&mut self) {}
    
    /// Full terminal reset (RIS).
    fn reset(&mut self) {}
    
    /// Index - move cursor down, scroll if at bottom (IND).
    fn index(&mut self) {}
    
    /// Newline - carriage return + line feed (NEL).
    fn newline(&mut self) {}
    
    /// Reverse index - move cursor up, scroll if at top (RI).
    fn reverse_index(&mut self) {}
    
    /// Set tab stop at current position (HTS).
    fn set_tab_stop(&mut self) {}
    
    /// Set keypad application/normal mode.
    fn set_keypad_mode(&mut self, _application: bool) {}
    
    /// Designate character set.
    fn designate_charset(&mut self, _set: u8, _charset: u8) {}
    
    /// Screen alignment test (DECALN).
    fn screen_alignment(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    
    struct TestHandler {
        text_chunks: Vec<Vec<char>>,
        csi_count: usize,
        osc_count: usize,
        control_chars: Vec<u8>,
    }
    
    impl TestHandler {
        fn new() -> Self {
            Self {
                text_chunks: Vec::new(),
                csi_count: 0,
                osc_count: 0,
                control_chars: Vec::new(),
            }
        }
    }
    
    impl Handler for TestHandler {
        fn text(&mut self, chars: &[char]) {
            self.text_chunks.push(chars.to_vec());
        }
        
        fn control(&mut self, byte: u8) {
            self.control_chars.push(byte);
        }
        
        fn csi(&mut self, _params: &CsiParams) {
            self.csi_count += 1;
        }
        
        fn osc(&mut self, _data: &[u8]) {
            self.osc_count += 1;
        }
    }
    
    #[test]
    fn test_plain_text() {
        let mut parser = Parser::new();
        let mut handler = TestHandler::new();
        
        parser.parse(b"Hello, World!", &mut handler);
        
        assert_eq!(handler.text_chunks.len(), 1);
        let text: String = handler.text_chunks[0].iter().collect();
        assert_eq!(text, "Hello, World!");
    }
    
    #[test]
    fn test_utf8_text() {
        let mut parser = Parser::new();
        let mut handler = TestHandler::new();
        
        parser.parse("Hello, 世界!".as_bytes(), &mut handler);
        
        assert_eq!(handler.text_chunks.len(), 1);
        let text: String = handler.text_chunks[0].iter().collect();
        assert_eq!(text, "Hello, 世界!");
    }
    
    #[test]
    fn test_control_chars_in_text() {
        let mut parser = Parser::new();
        let mut handler = TestHandler::new();
        
        // Text with LF and CR
        parser.parse(b"Hello\nWorld\r!", &mut handler);
        
        assert_eq!(handler.text_chunks.len(), 1);
        let text: String = handler.text_chunks[0].iter().collect();
        assert_eq!(text, "Hello\nWorld\r!");
    }
    
    #[test]
    fn test_csi_sequence() {
        let mut parser = Parser::new();
        let mut handler = TestHandler::new();
        
        // ESC [ 1 ; 2 m (SGR bold + dim)
        parser.parse(b"\x1b[1;2m", &mut handler);
        
        assert_eq!(handler.csi_count, 1);
    }
    
    #[test]
    fn test_mixed_text_and_csi() {
        let mut parser = Parser::new();
        let mut handler = TestHandler::new();
        
        parser.parse(b"Hello\x1b[1mWorld", &mut handler);
        
        assert_eq!(handler.text_chunks.len(), 2);
        let text1: String = handler.text_chunks[0].iter().collect();
        let text2: String = handler.text_chunks[1].iter().collect();
        assert_eq!(text1, "Hello");
        assert_eq!(text2, "World");
        assert_eq!(handler.csi_count, 1);
    }
    
    #[test]
    fn test_osc_sequence() {
        let mut parser = Parser::new();
        let mut handler = TestHandler::new();
        
        // OSC 0 ; title BEL
        parser.parse(b"\x1b]0;My Title\x07", &mut handler);
        
        assert_eq!(handler.osc_count, 1);
    }
    
    #[test]
    fn test_csi_with_subparams() {
        let mut parser = Parser::new();
        let mut handler = TestHandler::new();
        
        // CSI 38:2:255:128:64 m (RGB foreground with colon separators)
        parser.parse(b"\x1b[38:2:255:128:64m", &mut handler);
        
        assert_eq!(handler.csi_count, 1);
    }
}
