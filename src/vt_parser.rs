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
//! 5. Buffer is integrated into parser - I/O writes directly here
//! 6. Lock is released during parsing - I/O can continue while main parses

use std::sync::Mutex;
use crate::simd_utf8::SimdUtf8Decoder;

/// Buffer size - 1MB like Kitty
pub const BUF_SIZE: usize = 1024 * 1024;

/// Maximum number of CSI parameters.
pub const MAX_CSI_PARAMS: usize = 256;

/// Maximum length of an OSC string (same as escape length - no separate limit needed).
/// Kitty doesn't have a separate OSC limit, just the overall escape sequence limit.
const MAX_OSC_LEN: usize = 262144; // 256KB, same as MAX_ESCAPE_LEN

/// Maximum length of an escape sequence before we give up.
const MAX_ESCAPE_LEN: usize = 262144; // 256KB like Kitty

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

/// Digit multipliers for reverse-order accumulation (like Kitty).
/// Digits are accumulated with multipliers, then divided at commit time.
/// This avoids a multiply on every digit, using a table lookup instead.
static DIGIT_MULTIPLIERS: [i64; 16] = [
    10_000_000_000_000_000,
    1_000_000_000_000_000,
    100_000_000_000_000,
    10_000_000_000_000,
    1_000_000_000_000,
    100_000_000_000,
    10_000_000_000,
    1_000_000_000,
    100_000_000,
    10_000_000,
    1_000_000,
    100_000,
    10_000,
    1_000,
    100,
    10,
];

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
    /// Uses Kitty's reverse-order accumulation with lookup table.
    #[inline(always)]
    fn add_digit(&mut self, digit: u8) {
        // Like Kitty: accumulate with multipliers, divide at commit
        if self.num_digits < DIGIT_MULTIPLIERS.len() {
            self.accumulator += (digit - b'0') as i64 * DIGIT_MULTIPLIERS[self.num_digits];
            self.num_digits += 1;
        }
    }

    /// Commit the current parameter.
    #[inline]
    fn commit_param(&mut self) -> bool {
        if self.num_params >= MAX_CSI_PARAMS {
            return false;
        }
        // Convert reverse-order accumulator to final value
        // Like Kitty: accumulator / digit_multipliers[num_digits - 1]
        let value = if self.num_digits == 0 {
            0
        } else {
            // Division converts from reverse-order accumulation
            (self.accumulator / DIGIT_MULTIPLIERS[self.num_digits - 1]) as i32 * self.multiplier
        };
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
    /// UTF-8 decoder for text (SIMD-optimized).
    utf8: SimdUtf8Decoder,
    /// Decoded codepoint buffer (reused to avoid allocation).
    codepoint_buf: Vec<u32>,
    /// OSC string buffer.
    osc_buffer: Vec<u8>,
    /// DCS/APC/PM/SOS string buffer.
    string_buffer: Vec<u8>,
    /// Number of bytes consumed in current escape sequence (for max length check).
    escape_len: usize,
}

impl Default for Parser {
    fn default() -> Self {
        Self {
            state: State::Normal,
            csi: CsiParams::default(),
            utf8: SimdUtf8Decoder::new(),
            // Pre-allocate to match typical read buffer sizes (1MB) to avoid reallocation
            codepoint_buf: Vec::with_capacity(1024 * 1024),
            osc_buffer: Vec::new(),
            string_buffer: Vec::new(),
            escape_len: 0,
        }
    }
}

/// Shared buffer state for I/O thread communication.
/// This tracks read/write positions like Kitty's PS struct.
struct BufferState {
    /// Read tracking (like Kitty's read struct):
    /// - pos: current parse position (advances as we parse)
    /// - consumed: bytes that can be discarded (complete sequences only)
    /// - sz: total valid bytes in buffer
    read_pos: usize,
    read_consumed: usize,
    read_sz: usize,
    /// Write tracking: pending = bytes written by I/O but not yet visible to reader
    write_pending: usize,
}

/// Kitty-style shared parser with integrated 1MB buffer.
/// 
/// Like Kitty's PS struct, this owns the buffer AND all parser state.
/// I/O thread writes directly to this buffer, main thread parses in-place.
/// 
/// Critical: Lock is RELEASED during parsing so I/O can continue writing.
pub struct SharedParser {
    /// The 1MB buffer - I/O writes to end, main reads from front
    buf: std::cell::UnsafeCell<Box<[u8; BUF_SIZE]>>,
    /// Buffer state protected by mutex
    state: Mutex<BufferState>,
    /// Eventfd for waking I/O thread when space available
    wakeup_fd: i32,
    
    // ========== Parser state (main thread only, not behind mutex) ==========
    // These are copies of read_pos/read_sz/read_consumed for use while lock is released
    /// Current parse position (main thread working copy)
    parse_pos: std::cell::UnsafeCell<usize>,
    /// Total valid bytes (main thread working copy)  
    parse_sz: std::cell::UnsafeCell<usize>,
    /// Bytes that can be discarded (main thread working copy)
    parse_consumed: std::cell::UnsafeCell<usize>,
    /// Current parser state
    vte_state: std::cell::UnsafeCell<State>,
    /// CSI parameters being collected
    csi: std::cell::UnsafeCell<CsiParams>,
    /// UTF-8 decoder for text (SIMD-optimized)
    utf8: std::cell::UnsafeCell<SimdUtf8Decoder>,
    /// Decoded codepoint buffer (reused to avoid allocation)
    codepoint_buf: std::cell::UnsafeCell<Vec<u32>>,
    /// OSC string buffer
    osc_buffer: std::cell::UnsafeCell<Vec<u8>>,
    /// DCS/APC/PM/SOS string buffer
    string_buffer: std::cell::UnsafeCell<Vec<u8>>,
    /// Number of bytes consumed in current escape sequence (for max length check)
    escape_len: std::cell::UnsafeCell<usize>,
}

// SAFETY: I/O thread only writes to buf[read_sz+write_pending..], main thread
// only reads buf[read_pos..read_sz]. Parser state is only used by main thread.
unsafe impl Sync for SharedParser {}
unsafe impl Send for SharedParser {}

impl SharedParser {
    /// Create a new shared parser with integrated buffer.
    pub fn new() -> Self {
        let wakeup_fd = unsafe { libc::eventfd(0, libc::EFD_NONBLOCK | libc::EFD_CLOEXEC) };
        if wakeup_fd < 0 {
            panic!("Failed to create eventfd: {}", std::io::Error::last_os_error());
        }
        
        Self {
            buf: std::cell::UnsafeCell::new(Box::new([0u8; BUF_SIZE])),
            state: Mutex::new(BufferState {
                read_pos: 0,
                read_consumed: 0,
                read_sz: 0,
                write_pending: 0,
            }),
            wakeup_fd,
            // Parser state - working copies for use while lock is released
            parse_pos: std::cell::UnsafeCell::new(0),
            parse_sz: std::cell::UnsafeCell::new(0),
            parse_consumed: std::cell::UnsafeCell::new(0),
            vte_state: std::cell::UnsafeCell::new(State::Normal),
            csi: std::cell::UnsafeCell::new(CsiParams::default()),
            utf8: std::cell::UnsafeCell::new(SimdUtf8Decoder::new()),
            codepoint_buf: std::cell::UnsafeCell::new(Vec::with_capacity(BUF_SIZE)),
            osc_buffer: std::cell::UnsafeCell::new(Vec::new()),
            string_buffer: std::cell::UnsafeCell::new(Vec::new()),
            escape_len: std::cell::UnsafeCell::new(0),
        }
    }
    
    /// Get the wakeup fd for I/O thread to poll on.
    pub fn wakeup_fd(&self) -> i32 {
        self.wakeup_fd
    }
    
    // ========== I/O Thread API ==========
    
    /// Check if there's space for writing. Called by I/O thread.
    pub fn has_space(&self) -> bool {
        let state = self.state.lock().unwrap();
        state.read_sz + state.write_pending < BUF_SIZE
    }
    
    /// Get write buffer for I/O thread. Returns (ptr, available_bytes).
    /// Caller MUST call commit_write() after writing.
    pub fn create_write_buffer(&self) -> (*mut u8, usize) {
        let state = self.state.lock().unwrap();
        let write_offset = state.read_sz + state.write_pending;
        let available = BUF_SIZE.saturating_sub(write_offset);
        
        if available == 0 {
            return (std::ptr::null_mut(), 0);
        }
        
        // SAFETY: I/O writes past read_sz+write_pending
        let ptr = unsafe { (*self.buf.get()).as_mut_ptr().add(write_offset) };
        (ptr, available)
    }
    
    /// Commit bytes written by I/O thread.
    pub fn commit_write(&self, len: usize) {
        let mut state = self.state.lock().unwrap();
        state.write_pending += len;
    }
    
    /// Read from PTY fd into buffer. Returns bytes read, -1 for error.
    pub fn read_from_fd(&self, fd: i32) -> isize {
        let (ptr, available) = self.create_write_buffer();
        if available == 0 {
            return 0;
        }
        
        let result = unsafe { libc::read(fd, ptr as *mut libc::c_void, available) };
        
        if result > 0 {
            self.commit_write(result as usize);
        }
        result
    }
    
    /// Drain the wakeup eventfd.
    pub fn drain_wakeup(&self) {
        let mut buf = 0u64;
        unsafe {
            libc::read(self.wakeup_fd, &mut buf as *mut u64 as *mut libc::c_void, 8);
        }
    }
    
    // ========== Main Thread API ==========
    
    /// Run a parse pass. This is the Kitty-style run_worker():
    /// 1. Lock, make pending visible
    /// 2. UNLOCK during actual parsing (consume_input)
    /// 3. Re-lock, add new pending, check for more data, repeat
    /// 4. Final compaction and wake I/O if space created
    /// 
    /// Returns true if any data was parsed.
    pub fn run_parse_pass<H: Handler>(&self, handler: &mut H) -> bool {
        let mut parsed_any = false;
        
        // Lock for initial bookkeeping
        let mut state = self.state.lock().unwrap();
        
        // Make pending writes visible (like Kitty: self->read.sz += self->write.pending)
        state.read_sz += state.write_pending;
        state.write_pending = 0;
        
        // Check if there's data to parse (like Kitty: read.pos < read.sz)
        let has_pending_input = state.read_pos < state.read_sz;
        if !has_pending_input {
            return false;
        }
        
        let initial_pos = state.read_pos;
        let initial_sz = state.read_sz;
        
        // Track if buffer was ever full during this parse pass (for wakeup decision)
        // Like Kitty: pd->write_space_created = self->read.sz >= BUF_SZ (checked BEFORE compaction)
        let mut buffer_was_ever_full = state.read_sz >= BUF_SIZE;
        
        // Reset consumed counter for this parse pass (like Kitty: self->read.consumed = 0)
        state.read_consumed = 0;
        
        // Copy positions to UnsafeCell fields for use while lock is released
        unsafe {
            *self.parse_pos.get() = state.read_pos;
            *self.parse_sz.get() = state.read_sz;
            *self.parse_consumed.get() = state.read_pos; // consumed starts at current pos
        }
        
        // Parse loop - release lock during parsing!
        // Like Kitty's do { ... } while (self->read.pos < self->read.sz)
        let mut loop_count = 0;
        loop {
            let parse_pos = unsafe { *self.parse_pos.get() };
            let parse_sz = unsafe { *self.parse_sz.get() };
            
            if parse_pos >= parse_sz {
                break;
            }
            
            // RELEASE LOCK during parsing - I/O can continue writing!
            drop(state);
            
            // Parse the data - consume_input updates parse_pos and parse_consumed
            self.consume_input(handler);
            parsed_any = true;
            loop_count += 1;
            
            // Re-acquire lock
            state = self.state.lock().unwrap();
            
            // CRITICAL: Like Kitty line 1518, add new pending data INSIDE the loop
            // This allows us to process data that arrived while we were parsing
            state.read_sz += state.write_pending;
            state.write_pending = 0;
            
            // Update buffer_was_ever_full if buffer is now full
            if state.read_sz >= BUF_SIZE {
                buffer_was_ever_full = true;
            }
            
            // Update state with new positions from parsing
            state.read_pos = unsafe { *self.parse_pos.get() };
            state.read_consumed = unsafe { *self.parse_consumed.get() };
            
            // Update parse_sz to include new data for next iteration
            unsafe {
                *self.parse_sz.get() = state.read_sz;
            }
            
            // If no more unparsed data, we're done (like Kitty: while read.pos < read.sz)
            if state.read_pos >= state.read_sz {
                break;
            }
        }
        
        let bytes_parsed = state.read_pos.saturating_sub(initial_pos);
        if bytes_parsed > 0 || loop_count > 1 {
            log::debug!("[PARSE] initial_pos={} initial_sz={} final_pos={} final_sz={} loops={} bytes={}",
                initial_pos, initial_sz, state.read_pos, state.read_sz, loop_count, bytes_parsed);
        }
        
        // Compaction - remove consumed bytes (like Kitty)
        if state.read_consumed > 0 {
            let old_sz = state.read_sz;
            
            // Like Kitty: pos -= consumed, sz -= consumed, memmove
            state.read_pos = state.read_pos.saturating_sub(state.read_consumed);
            state.read_sz = state.read_sz.saturating_sub(state.read_consumed);
            
            // memmove remaining data to front
            if state.read_sz > 0 {
                unsafe {
                    let buf = &mut *self.buf.get();
                    std::ptr::copy(
                        buf.as_ptr().add(state.read_consumed),
                        buf.as_mut_ptr(),
                        state.read_sz,
                    );
                }
            }
            
            let consumed = state.read_consumed;
            state.read_consumed = 0;
            
            // Wake I/O thread if buffer was ever full during this pass and we freed space
            // Like Kitty: if (pd.write_space_created) wakeup_io_loop()
            if buffer_was_ever_full && state.read_sz < BUF_SIZE {
                log::debug!("[PARSE] Waking I/O: was_full={} old_sz={} new_sz={} consumed={}", 
                    buffer_was_ever_full, old_sz, state.read_sz, consumed);
                drop(state);
                let val = 1u64;
                unsafe {
                    libc::write(self.wakeup_fd, &val as *const u64 as *const libc::c_void, 8);
                }
                return parsed_any;
            }
        } else if buffer_was_ever_full {
            // Buffer was full but nothing consumed - stuck in partial sequence?
            log::warn!("[PARSE] Buffer was full but read_consumed=0! read_pos={} read_sz={}", 
                state.read_pos, state.read_sz);
        }
        
        drop(state);
        parsed_any
    }
    
    /// Check if there's pending data (for tick scheduling).
    pub fn has_pending_data(&self) -> bool {
        let state = self.state.lock().unwrap();
        state.read_pos < state.read_sz || state.write_pending > 0
    }
    
    // ========== Internal parsing methods (main thread only) ==========
    
    /// Main parsing dispatch - like Kitty's consume_input().
    /// Reads from buf[parse_pos..parse_sz] and updates positions.
    /// 
    /// IMPORTANT: Unlike the previous implementation, this now loops internally
    /// until the buffer is exhausted or we're waiting for more data in an incomplete
    /// escape sequence. This reduces per-CSI overhead from 3 function calls to 1.
    fn consume_input<H: Handler>(&self, handler: &mut H) {
        #[cfg(feature = "render_timing")]
        let start = std::time::Instant::now();
        
        // Get mutable access to parser state (SAFETY: only main thread calls this)
        let parse_pos = unsafe { &mut *self.parse_pos.get() };
        let parse_sz = unsafe { *self.parse_sz.get() };
        let parse_consumed = unsafe { &mut *self.parse_consumed.get() };
        let vte_state = unsafe { &mut *self.vte_state.get() };
        let csi = unsafe { &mut *self.csi.get() };
        let utf8 = unsafe { &mut *self.utf8.get() };
        let codepoint_buf = unsafe { &mut *self.codepoint_buf.get() };
        let osc_buffer = unsafe { &mut *self.osc_buffer.get() };
        let string_buffer = unsafe { &mut *self.string_buffer.get() };
        let escape_len = unsafe { &mut *self.escape_len.get() };
        let buf = unsafe { &*self.buf.get() };
        
        // Loop until buffer exhausted or waiting for more data
        while *parse_pos < parse_sz {
            match *vte_state {
                State::Normal => {
                    // Like Kitty: consume_normal(self); self->read.consumed = self->read.pos;
                    Self::consume_normal_impl(handler, buf, parse_pos, parse_sz, utf8, codepoint_buf, vte_state, escape_len);
                    *parse_consumed = *parse_pos;
                    // consume_normal_impl sets vte_state to Escape if ESC found, so loop continues
                }
                State::Escape => {
                    // Like Kitty: if (consume_esc(self)) { self->read.consumed = self->read.pos; }
                    if Self::consume_escape_impl(handler, buf, parse_pos, parse_sz, *parse_consumed, vte_state, csi, osc_buffer, string_buffer, escape_len) {
                        *parse_consumed = *parse_pos;
                        // State changed, continue loop
                    } else {
                        // Need more data for escape sequence
                        break;
                    }
                }
                State::EscapeIntermediate(_) => {
                    if Self::consume_escape_intermediate_impl(handler, buf, parse_pos, parse_sz, vte_state) {
                        *parse_consumed = *parse_pos;
                    } else {
                        break;
                    }
                }
                State::Csi => {
                    // Like Kitty: if (consume_csi(self)) { self->read.consumed = self->read.pos; dispatch; SET_STATE(NORMAL); }
                    if Self::consume_csi_impl(handler, buf, parse_pos, parse_sz, *parse_consumed, csi, escape_len) {
                        *parse_consumed = *parse_pos;
                        if csi.is_valid {
                            handler.csi(csi);
                        }
                        *vte_state = State::Normal;
                        // Continue loop to process more data
                    } else {
                        // Need more data for CSI sequence
                        break;
                    }
                }
                State::Osc => {
                    if Self::consume_osc_impl(handler, buf, parse_pos, parse_sz, vte_state, osc_buffer, escape_len) {
                        *parse_consumed = *parse_pos;
                        *vte_state = State::Normal;
                    } else {
                        break;
                    }
                }
                State::Dcs | State::Apc | State::Pm | State::Sos => {
                    if Self::consume_string_impl(handler, buf, parse_pos, parse_sz, vte_state, string_buffer, escape_len) {
                        *parse_consumed = *parse_pos;
                        *vte_state = State::Normal;
                    } else {
                        break;
                    }
                }
            }
        }
        
        #[cfg(feature = "render_timing")]
        handler.add_vt_parser_ns(start.elapsed().as_nanos() as u64);
    }
    
    /// Consume normal text - like Kitty's consume_normal().
    /// UTF-8 decodes until ESC is found using SIMD-optimized decoder.
    #[inline]
    fn consume_normal_impl<H: Handler>(
        handler: &mut H,
        buf: &[u8; BUF_SIZE],
        parse_pos: &mut usize,
        parse_sz: usize,
        utf8: &mut SimdUtf8Decoder,
        codepoint_buf: &mut Vec<u32>,
        vte_state: &mut State,
        escape_len: &mut usize,
    ) {
        loop {
            if *parse_pos >= parse_sz {
                break;
            }
            
            let remaining = &buf[*parse_pos..parse_sz];
            let (consumed, found_esc) = utf8.decode_to_esc(remaining, codepoint_buf);
            *parse_pos += consumed;
            
            if !codepoint_buf.is_empty() {
                handler.text(codepoint_buf);
            }
            
            if found_esc {
                *vte_state = State::Escape;
                *escape_len = 0;
                break;
            }
        }
    }
    
    /// Consume escape sequence start - like Kitty's consume_esc().
    /// Returns true if sequence is complete (consumed = pos).
    #[inline]
    fn consume_escape_impl<H: Handler>(
        handler: &mut H,
        buf: &[u8; BUF_SIZE],
        parse_pos: &mut usize,
        parse_sz: usize,
        parse_consumed: usize,
        vte_state: &mut State,
        csi: &mut CsiParams,
        osc_buffer: &mut Vec<u8>,
        string_buffer: &mut Vec<u8>,
        escape_len: &mut usize,
    ) -> bool {
        if *parse_pos >= parse_sz {
            return false;
        }
        
        let ch = buf[*parse_pos];
        *parse_pos += 1;
        *escape_len += 1;
        
        // Like Kitty: is_first_char = read.pos - read.consumed == 1
        let is_first_char = *parse_pos - parse_consumed == 1;
        
        if is_first_char {
            match ch {
                b'[' => { *vte_state = State::Csi; csi.reset(); }
                b']' => { *vte_state = State::Osc; osc_buffer.clear(); }
                b'P' => { *vte_state = State::Dcs; string_buffer.clear(); }
                b'_' => { *vte_state = State::Apc; string_buffer.clear(); }
                b'^' => { *vte_state = State::Pm; string_buffer.clear(); }
                b'X' => { *vte_state = State::Sos; string_buffer.clear(); }
                // Two-char sequences - need another char
                b'(' | b')' | b'*' | b'+' | b'-' | b'.' | b'/' | b'%' | b'#' | b' ' => {
                    *vte_state = State::EscapeIntermediate(ch);
                    return false; // Need more chars
                }
                // Single-char escape sequences
                b'7' => { handler.save_cursor(); *vte_state = State::Normal; }
                b'8' => { handler.restore_cursor(); *vte_state = State::Normal; }
                b'c' => { handler.reset(); *vte_state = State::Normal; }
                b'D' => { handler.index(); *vte_state = State::Normal; }
                b'E' => { handler.newline(); *vte_state = State::Normal; }
                b'H' => { handler.set_tab_stop(); *vte_state = State::Normal; }
                b'M' => { handler.reverse_index(); *vte_state = State::Normal; }
                b'=' => { handler.set_keypad_mode(true); *vte_state = State::Normal; }
                b'>' => { handler.set_keypad_mode(false); *vte_state = State::Normal; }
                b'\\' => { *vte_state = State::Normal; } // ST
                _ => {
                    log::debug!("Unknown escape sequence: ESC {:02x}", ch);
                    *vte_state = State::Normal;
                }
            }
            return true;
        } else {
            // Second char of two-char sequence - like Kitty's else branch
            let prev_ch = buf[*parse_pos - 2];
            *vte_state = State::Normal;
            
            match prev_ch {
                b'(' | b')' => {
                    let set = if prev_ch == b'(' { 0 } else { 1 };
                    handler.designate_charset(set, ch);
                }
                b'#' => {
                    if ch == b'8' {
                        handler.screen_alignment();
                    }
                }
                _ => {}
            }
            return true;
        }
    }
    
    /// Consume second byte of two-char escape sequence.
    fn consume_escape_intermediate_impl<H: Handler>(
        handler: &mut H,
        buf: &[u8; BUF_SIZE],
        parse_pos: &mut usize,
        parse_sz: usize,
        vte_state: &mut State,
    ) -> bool {
        if *parse_pos >= parse_sz {
            return false;
        }
        
        let ch = buf[*parse_pos];
        *parse_pos += 1;
        
        let intermediate = match *vte_state {
            State::EscapeIntermediate(i) => i,
            _ => { *vte_state = State::Normal; return true; }
        };
        
        *vte_state = State::Normal;
        
        match intermediate {
            b'(' | b')' => {
                let set = if intermediate == b'(' { 0 } else { 1 };
                handler.designate_charset(set, ch);
            }
            b'#' => {
                if ch == b'8' {
                    handler.screen_alignment();
                }
            }
            _ => {}
        }
        
        true
    }
    
    /// Consume CSI sequence - like Kitty's csi_parse_loop().
    /// Returns true when sequence is complete.
    #[inline]
    fn consume_csi_impl<H: Handler>(
        handler: &mut H,
        buf: &[u8; BUF_SIZE],
        parse_pos: &mut usize,
        parse_sz: usize,
        parse_consumed: usize,
        csi: &mut CsiParams,
        escape_len: &mut usize,
    ) -> bool {
        while *parse_pos < parse_sz {
            let ch = buf[*parse_pos];
            *parse_pos += 1;
            *escape_len += 1;
            
            // Handle embedded control characters
            if ch <= 0x1F && ch != 0x1B {
                handler.control(ch);
                continue;
            }
            
            match csi.state {
                CsiState::Start => {
                    match ch {
                        b';' => {
                            csi.params[csi.num_params] = 0;
                            csi.num_params += 1;
                            csi.state = CsiState::Body;
                        }
                        b'0'..=b'9' => {
                            csi.add_digit(ch);
                            csi.state = CsiState::Body;
                        }
                        b'?' | b'>' | b'<' | b'=' => {
                            csi.primary = ch;
                            csi.state = CsiState::Body;
                        }
                        b'-' => {
                            csi.multiplier = -1;
                            csi.num_digits = 1;
                            csi.state = CsiState::Body;
                        }
                        b' ' | b'\'' | b'"' | b'!' | b'$' | b'#' | b'*' => {
                            csi.secondary = ch;
                            csi.state = CsiState::PostSecondary;
                        }
                        b'@'..=b'~' => {
                            csi.final_char = ch;
                            csi.is_valid = true;
                            return true;
                        }
                        _ => {
                            log::debug!("Invalid CSI character: {:02x}", ch);
                            return true;
                        }
                    }
                }
                CsiState::Body => {
                    match ch {
                        b'0'..=b'9' => {
                            csi.add_digit(ch);
                        }
                        b';' => {
                            if csi.num_digits == 0 {
                                csi.num_digits = 1;
                            }
                            if !csi.commit_param() {
                                return true;
                            }
                            csi.is_sub_param[csi.num_params] = false;
                        }
                        b':' => {
                            if !csi.commit_param() {
                                return true;
                            }
                            csi.is_sub_param[csi.num_params] = true;
                        }
                        b'-' if csi.num_digits == 0 => {
                            csi.multiplier = -1;
                            csi.num_digits = 1;
                        }
                        b' ' | b'\'' | b'"' | b'!' | b'$' | b'#' | b'*' => {
                            if !csi.commit_param() {
                                return true;
                            }
                            csi.secondary = ch;
                            csi.state = CsiState::PostSecondary;
                        }
                        b'@'..=b'~' => {
                            if csi.num_digits > 0 || csi.num_params > 0 {
                                csi.commit_param();
                            }
                            csi.final_char = ch;
                            csi.is_valid = true;
                            return true;
                        }
                        _ => {
                            log::debug!("Invalid CSI body character: {:02x}", ch);
                            return true;
                        }
                    }
                }
                CsiState::PostSecondary => {
                    match ch {
                        b'@'..=b'~' => {
                            csi.final_char = ch;
                            csi.is_valid = true;
                            return true;
                        }
                        _ => {
                            log::debug!("Invalid CSI post-secondary character: {:02x}", ch);
                            return true;
                        }
                    }
                }
            }
        }
        
        // Check max length
        if *parse_pos - parse_consumed > MAX_ESCAPE_LEN {
            log::debug!("CSI escape too long, ignoring");
            return true;
        }
        
        false
    }
    
    /// Consume OSC sequence.
    fn consume_osc_impl<H: Handler>(
        handler: &mut H,
        buf: &[u8; BUF_SIZE],
        parse_pos: &mut usize,
        parse_sz: usize,
        vte_state: &mut State,
        osc_buffer: &mut Vec<u8>,
        escape_len: &mut usize,
    ) -> bool {
        while *parse_pos < parse_sz {
            let ch = buf[*parse_pos];
            
            match ch {
                0x07 => {
                    // BEL terminator
                    *parse_pos += 1;
                    handler.osc(osc_buffer);
                    return true;
                }
                0x9C => {
                    // C1 ST terminator
                    *parse_pos += 1;
                    handler.osc(osc_buffer);
                    return true;
                }
                0x1B => {
                    // Check for ESC \
                    if *parse_pos + 1 < parse_sz && buf[*parse_pos + 1] == b'\\' {
                        *parse_pos += 2;
                        handler.osc(osc_buffer);
                        return true;
                    } else if *parse_pos + 1 < parse_sz {
                        // ESC followed by something else - abort OSC, start new escape
                        *parse_pos += 1;
                        handler.osc(osc_buffer);
                        *vte_state = State::Escape;
                        *escape_len = 0;
                        return false;
                    } else {
                        // ESC at end of buffer - need more data
                        return false;
                    }
                }
                _ => {
                    osc_buffer.push(ch);
                    *parse_pos += 1;
                    *escape_len += 1;
                }
            }
            
            if *escape_len > MAX_ESCAPE_LEN {
                log::debug!("OSC sequence too long, aborting");
                return true;
            }
        }
        
        false
    }
    
    /// Consume DCS/APC/PM/SOS string sequence.
    fn consume_string_impl<H: Handler>(
        handler: &mut H,
        buf: &[u8; BUF_SIZE],
        parse_pos: &mut usize,
        parse_sz: usize,
        vte_state: &mut State,
        string_buffer: &mut Vec<u8>,
        escape_len: &mut usize,
    ) -> bool {
        while *parse_pos < parse_sz {
            let ch = buf[*parse_pos];
            
            match ch {
                0x9C => {
                    // C1 ST terminator
                    *parse_pos += 1;
                    Self::dispatch_string_command(handler, vte_state, string_buffer);
                    return true;
                }
                0x1B => {
                    // Check for ESC \
                    if *parse_pos + 1 < parse_sz && buf[*parse_pos + 1] == b'\\' {
                        *parse_pos += 2;
                        Self::dispatch_string_command(handler, vte_state, string_buffer);
                        return true;
                    } else if *parse_pos + 1 < parse_sz {
                        // ESC not followed by \ - include in buffer
                        string_buffer.push(ch);
                        *parse_pos += 1;
                        *escape_len += 1;
                    } else {
                        // ESC at end of buffer - need more data
                        return false;
                    }
                }
                _ => {
                    string_buffer.push(ch);
                    *parse_pos += 1;
                    *escape_len += 1;
                }
            }
            
            if *escape_len > MAX_ESCAPE_LEN {
                log::debug!("String command too long, aborting");
                return true;
            }
        }
        
        false
    }
    
    /// Dispatch string command to handler.
    fn dispatch_string_command<H: Handler>(
        handler: &mut H,
        vte_state: &State,
        string_buffer: &[u8],
    ) {
        match vte_state {
            State::Dcs => handler.dcs(string_buffer),
            State::Apc => handler.apc(string_buffer),
            State::Pm => handler.pm(string_buffer),
            State::Sos => handler.sos(string_buffer),
            _ => {}
        }
    }
}

impl Drop for SharedParser {
    fn drop(&mut self) {
        unsafe {
            libc::close(self.wakeup_fd);
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
        self.codepoint_buf.clear();
        self.osc_buffer.clear();
        self.string_buffer.clear();
        self.escape_len = 0;
    }

    /// Process a buffer of bytes, calling the handler for each action.
    /// Returns the number of bytes consumed.
    pub fn parse<H: Handler>(&mut self, bytes: &[u8], handler: &mut H) -> usize {
        let mut pos = 0;
        
        while pos < bytes.len() {
            match self.state {
                State::Normal => {
                    // Fast path: UTF-8 decode until ESC using SIMD
                    let (consumed, found_esc) = self.utf8.decode_to_esc(&bytes[pos..], &mut self.codepoint_buf);
                    
                    // Process decoded codepoints (text + control chars)
                    if !self.codepoint_buf.is_empty() {
                        handler.text(&self.codepoint_buf);
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
        // Extract intermediate from state enum (eliminates redundant self.intermediate field)
        let intermediate = match self.state {
            State::EscapeIntermediate(i) => i,
            _ => return 0, // Should never happen
        };
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
            
            // Handle control characters embedded in CSI (common to all states)
            if ch <= 0x1F && ch != 0x1B {
                handler.control(ch);
                continue;
            }
            
            match self.csi.state {
                CsiState::Start => {
                    match ch {
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

    /// Dispatch the string command to the appropriate handler method.
    #[inline]
    fn dispatch_string_command<H: Handler>(&self, handler: &mut H) {
        match self.state {
            State::Dcs => handler.dcs(&self.string_buffer),
            State::Apc => handler.apc(&self.string_buffer),
            State::Pm => handler.pm(&self.string_buffer),
            State::Sos => handler.sos(&self.string_buffer),
            _ => unreachable!("dispatch_string_command called in invalid state"),
        }
    }

    /// Process DCS/APC/PM/SOS sequence bytes using SIMD-accelerated terminator search.
    /// Like Kitty's find_st_terminator + accumulate_st_terminated_esc_code.
    /// Uses iterative approach to avoid stack overflow on malformed input.
    fn consume_string_command<H: Handler>(&mut self, bytes: &[u8], pos: usize, handler: &mut H) -> usize {
        let mut current_pos = pos;
        let mut total_consumed = 0;
        
        loop {
            let remaining = &bytes[current_pos..];
            
            // Use SIMD-accelerated search to find ESC (0x1B) or C1 ST (0x9C)
            if let Some(term_pos) = memchr::memchr2(0x1B, 0x9C, remaining) {
                let terminator = remaining[term_pos];
                
                // Check max length before accepting
                if self.escape_len + term_pos > MAX_ESCAPE_LEN {
                    log::debug!("String command too long, aborting");
                    self.state = State::Normal;
                    return total_consumed + remaining.len();
                }
                
                match terminator {
                    0x9C => {
                        // C1 ST terminator - copy data in bulk and dispatch
                        self.string_buffer.extend_from_slice(&remaining[..term_pos]);
                        self.dispatch_string_command(handler);
                        self.state = State::Normal;
                        self.escape_len += term_pos + 1;
                        return total_consumed + term_pos + 1;
                    }
                    0x1B => {
                        // ESC found - check if followed by \ for ST
                        if term_pos + 1 < remaining.len() && remaining[term_pos + 1] == b'\\' {
                            // ESC \ (ST) terminator
                            self.string_buffer.extend_from_slice(&remaining[..term_pos]);
                            self.dispatch_string_command(handler);
                            self.state = State::Normal;
                            self.escape_len += term_pos + 2;
                            return total_consumed + term_pos + 2;
                        } else if term_pos + 1 < remaining.len() {
                            // ESC not followed by \ - include ESC in data and continue
                            // (Unlike OSC, string commands include raw ESC that isn't ST)
                            self.string_buffer.extend_from_slice(&remaining[..=term_pos]);
                            self.escape_len += term_pos + 1;
                            // Continue searching from after this ESC (iterative, not recursive)
                            let consumed = term_pos + 1;
                            total_consumed += consumed;
                            current_pos += consumed;
                            continue;
                        } else {
                            // ESC at end of buffer, need more data
                            // Copy everything before ESC, keep ESC for next parse
                            self.string_buffer.extend_from_slice(&remaining[..term_pos]);
                            self.escape_len += term_pos;
                            return total_consumed + term_pos;
                        }
                    }
                    _ => unreachable!(),
                }
            } else {
                // No terminator found - check max length
                if self.escape_len + remaining.len() > MAX_ESCAPE_LEN {
                    log::debug!("String command too long, aborting");
                    self.state = State::Normal;
                    return total_consumed + remaining.len();
                }
                
                // Buffer all remaining bytes for next parse call
                self.string_buffer.extend_from_slice(remaining);
                self.escape_len += remaining.len();
                return total_consumed + remaining.len();
            }
        }
    }
}

/// Handler trait for responding to parsed escape sequences.
/// 
/// Unlike the vte crate's Perform trait, this trait receives decoded characters
/// (not bytes) for text, and control characters are expected to be handled
/// inline in the text() method (like Kitty does).
pub trait Handler {
    /// Handle a chunk of decoded text (Unicode codepoints as u32).
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
    /// 
    /// Codepoints are passed as u32 for efficiency (avoiding char validation).
    /// All codepoints are guaranteed to be valid Unicode (validated during UTF-8 decode).
    fn text(&mut self, codepoints: &[u32]);
    
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
    
    /// Add VT parser time (for performance tracking).
    /// Called by the parser to report time spent in consume_input.
    fn add_vt_parser_ns(&mut self, _ns: u64) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    
    struct TestHandler {
        text_chunks: Vec<Vec<u32>>,
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
        fn text(&mut self, codepoints: &[u32]) {
            self.text_chunks.push(codepoints.to_vec());
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
        let text: String = handler.text_chunks[0].iter().filter_map(|&cp| char::from_u32(cp)).collect();
        assert_eq!(text, "Hello, World!");
    }
    
    #[test]
    fn test_utf8_text() {
        let mut parser = Parser::new();
        let mut handler = TestHandler::new();
        
        parser.parse("Hello, 世界!".as_bytes(), &mut handler);
        
        assert_eq!(handler.text_chunks.len(), 1);
        let text: String = handler.text_chunks[0].iter().filter_map(|&cp| char::from_u32(cp)).collect();
        assert_eq!(text, "Hello, 世界!");
    }
    
    #[test]
    fn test_control_chars_in_text() {
        let mut parser = Parser::new();
        let mut handler = TestHandler::new();
        
        // Text with LF and CR
        parser.parse(b"Hello\nWorld\r!", &mut handler);
        
        assert_eq!(handler.text_chunks.len(), 1);
        let text: String = handler.text_chunks[0].iter().filter_map(|&cp| char::from_u32(cp)).collect();
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
        let text1: String = handler.text_chunks[0].iter().filter_map(|&cp| char::from_u32(cp)).collect();
        let text2: String = handler.text_chunks[1].iter().filter_map(|&cp| char::from_u32(cp)).collect();
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
