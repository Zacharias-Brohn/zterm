//! Kitty keyboard protocol implementation.
//!
//! This module implements the progressive keyboard enhancement protocol
//! as specified at: https://sw.kovidgoyal.net/kitty/keyboard-protocol/

use bitflags::bitflags;

bitflags! {
    /// Keyboard enhancement flags for the Kitty keyboard protocol.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
    pub struct KeyboardFlags: u8 {
        /// Disambiguate escape codes (report Esc, alt+key, ctrl+key using CSI u).
        const DISAMBIGUATE = 0b00001;
        /// Report key repeat and release events.
        const REPORT_EVENTS = 0b00010;
        /// Report alternate keys (shifted key, base layout key).
        const REPORT_ALTERNATES = 0b00100;
        /// Report all keys as escape codes (including text-generating keys).
        const REPORT_ALL_KEYS = 0b01000;
        /// Report associated text with key events.
        const REPORT_TEXT = 0b10000;
    }
}

/// Key event types for the keyboard protocol.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyEventType {
    Press = 1,
    Repeat = 2,
    Release = 3,
}

/// Modifier flags for key events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Modifiers {
    pub shift: bool,
    pub alt: bool,
    pub ctrl: bool,
    pub super_key: bool,
    pub hyper: bool,
    pub meta: bool,
    pub caps_lock: bool,
    pub num_lock: bool,
}

impl Modifiers {
    /// Encodes modifiers as a decimal number (1 + bitfield).
    /// Returns None if no modifiers are active.
    pub fn encode(&self) -> Option<u8> {
        let mut bits: u8 = 0;
        if self.shift {
            bits |= 1;
        }
        if self.alt {
            bits |= 2;
        }
        if self.ctrl {
            bits |= 4;
        }
        if self.super_key {
            bits |= 8;
        }
        if self.hyper {
            bits |= 16;
        }
        if self.meta {
            bits |= 32;
        }
        if self.caps_lock {
            bits |= 64;
        }
        if self.num_lock {
            bits |= 128;
        }

        if bits == 0 {
            None
        } else {
            Some(1 + bits)
        }
    }

    /// Returns true if any modifier is active.
    pub fn any(&self) -> bool {
        self.shift
            || self.alt
            || self.ctrl
            || self.super_key
            || self.hyper
            || self.meta
            || self.caps_lock
            || self.num_lock
    }
}

/// Functional key codes from the Kitty keyboard protocol.
/// These are Unicode Private Use Area codepoints (57344 - 63743).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum FunctionalKey {
    Escape = 27,
    Enter = 13,
    Tab = 9,
    Backspace = 127,
    Insert = 57348,
    Delete = 57349,
    Left = 57350,
    Right = 57351,
    Up = 57352,
    Down = 57353,
    PageUp = 57354,
    PageDown = 57355,
    Home = 57356,
    End = 57357,
    CapsLock = 57358,
    ScrollLock = 57359,
    NumLock = 57360,
    PrintScreen = 57361,
    Pause = 57362,
    Menu = 57363,
    F1 = 57364,
    F2 = 57365,
    F3 = 57366,
    F4 = 57367,
    F5 = 57368,
    F6 = 57369,
    F7 = 57370,
    F8 = 57371,
    F9 = 57372,
    F10 = 57373,
    F11 = 57374,
    F12 = 57375,
    F13 = 57376,
    F14 = 57377,
    F15 = 57378,
    F16 = 57379,
    F17 = 57380,
    F18 = 57381,
    F19 = 57382,
    F20 = 57383,
    F21 = 57384,
    F22 = 57385,
    F23 = 57386,
    F24 = 57387,
    F25 = 57388,
    // Keypad keys
    KpDecimal = 57409,
    KpDivide = 57410,
    KpMultiply = 57411,
    KpSubtract = 57412,
    KpAdd = 57413,
    KpEnter = 57414,
    KpEqual = 57415,
    KpSeparator = 57416,
    KpLeft = 57417,
    KpRight = 57418,
    KpUp = 57419,
    KpDown = 57420,
    KpPageUp = 57421,
    KpPageDown = 57422,
    KpHome = 57423,
    KpEnd = 57424,
    KpInsert = 57425,
    KpDelete = 57426,
    KpBegin = 57427,
    // Media keys
    MediaPlay = 57428,
    MediaPause = 57429,
    MediaPlayPause = 57430,
    MediaReverse = 57431,
    MediaStop = 57432,
    MediaFastForward = 57433,
    MediaRewind = 57434,
    MediaTrackNext = 57435,
    MediaTrackPrevious = 57436,
    MediaRecord = 57437,
    LowerVolume = 57438,
    RaiseVolume = 57439,
    MuteVolume = 57440,
    // Modifier keys
    LeftShift = 57441,
    LeftControl = 57442,
    LeftAlt = 57443,
    LeftSuper = 57444,
    LeftHyper = 57445,
    LeftMeta = 57446,
    RightShift = 57447,
    RightControl = 57448,
    RightAlt = 57449,
    RightSuper = 57450,
    RightHyper = 57451,
    RightMeta = 57452,
    IsoLevel3Shift = 57453,
    IsoLevel5Shift = 57454,
}

/// Keyboard protocol state.
#[derive(Debug, Clone)]
pub struct KeyboardState {
    /// Current enhancement flags.
    flags: KeyboardFlags,
    /// Stack of pushed flag states (for push/pop).
    stack: Vec<KeyboardFlags>,
}

impl Default for KeyboardState {
    fn default() -> Self {
        Self::new()
    }
}

impl KeyboardState {
    /// Maximum stack size to prevent DoS.
    const MAX_STACK_SIZE: usize = 16;

    pub fn new() -> Self {
        Self {
            flags: KeyboardFlags::empty(),
            stack: Vec::new(),
        }
    }

    /// Gets the current keyboard enhancement flags.
    pub fn flags(&self) -> KeyboardFlags {
        self.flags
    }

    /// Sets keyboard flags using the specified mode.
    /// mode 1: set all flags to the given value
    /// mode 2: set bits that are set in flags, leave others unchanged
    /// mode 3: reset bits that are set in flags, leave others unchanged
    pub fn set_flags(&mut self, flags: u8, mode: u8) {
        let new_flags = KeyboardFlags::from_bits_truncate(flags);
        match mode {
            1 => self.flags = new_flags,
            2 => self.flags |= new_flags,
            3 => self.flags &= !new_flags,
            _ => self.flags = new_flags, // Default to mode 1
        }
    }

    /// Pushes current flags onto the stack and optionally sets new flags.
    pub fn push(&mut self, flags: Option<u8>) {
        // Evict oldest entry if stack is full
        if self.stack.len() >= Self::MAX_STACK_SIZE {
            self.stack.remove(0);
        }
        self.stack.push(self.flags);
        if let Some(f) = flags {
            self.flags = KeyboardFlags::from_bits_truncate(f);
        }
    }

    /// Pops entries from the stack.
    pub fn pop(&mut self, count: usize) {
        let count = count.max(1);
        for _ in 0..count {
            if let Some(flags) = self.stack.pop() {
                self.flags = flags;
            } else {
                // Stack is empty, reset all flags
                self.flags = KeyboardFlags::empty();
                break;
            }
        }
    }

    /// Returns whether the DISAMBIGUATE flag is set.
    pub fn disambiguate(&self) -> bool {
        self.flags.contains(KeyboardFlags::DISAMBIGUATE)
    }

    /// Returns whether the REPORT_EVENTS flag is set.
    pub fn report_events(&self) -> bool {
        self.flags.contains(KeyboardFlags::REPORT_EVENTS)
    }

    /// Returns whether the REPORT_ALTERNATES flag is set.
    pub fn report_alternates(&self) -> bool {
        self.flags.contains(KeyboardFlags::REPORT_ALTERNATES)
    }

    /// Returns whether the REPORT_ALL_KEYS flag is set.
    pub fn report_all_keys(&self) -> bool {
        self.flags.contains(KeyboardFlags::REPORT_ALL_KEYS)
    }

    /// Returns whether the REPORT_TEXT flag is set.
    pub fn report_text(&self) -> bool {
        self.flags.contains(KeyboardFlags::REPORT_TEXT)
    }
}

/// Encodes a key event according to the Kitty keyboard protocol.
pub struct KeyEncoder<'a> {
    state: &'a KeyboardState,
    /// Whether application cursor keys mode (DECCKM) is enabled.
    /// When true, arrow keys send SS3 format (ESC O letter).
    /// When false, arrow keys send CSI format (ESC [ letter).
    application_cursor_keys: bool,
}

impl<'a> KeyEncoder<'a> {
    pub fn new(state: &'a KeyboardState) -> Self {
        Self {
            state,
            application_cursor_keys: false,
        }
    }

    /// Creates a new KeyEncoder with application cursor keys mode setting.
    pub fn with_cursor_mode(
        state: &'a KeyboardState,
        application_cursor_keys: bool,
    ) -> Self {
        Self {
            state,
            application_cursor_keys,
        }
    }

    /// Encodes a functional key press to bytes.
    pub fn encode_functional(
        &self,
        key: FunctionalKey,
        modifiers: Modifiers,
        event_type: KeyEventType,
    ) -> Vec<u8> {
        let key_code = key as u32;

        // Special handling for legacy keys in legacy mode
        if self.state.flags().is_empty() {
            return self.encode_legacy_functional(key, modifiers);
        }

        self.encode_csi_u(key_code, modifiers, event_type, None)
    }

    /// Encodes a Unicode character key press.
    pub fn encode_char(
        &self,
        c: char,
        modifiers: Modifiers,
        event_type: KeyEventType,
    ) -> Vec<u8> {
        let key_code = c as u32;

        // In legacy mode without REPORT_ALL_KEYS, just send the character
        // (with legacy ctrl/alt handling)
        if !self.state.report_all_keys() {
            return self.encode_legacy_text(c, modifiers);
        }

        // With REPORT_ALL_KEYS, encode as CSI u
        let text = if self.state.report_text() {
            Some(c)
        } else {
            None
        };

        self.encode_csi_u(key_code, modifiers, event_type, text)
    }

    /// Encodes a key event as CSI u format.
    fn encode_csi_u(
        &self,
        key_code: u32,
        modifiers: Modifiers,
        event_type: KeyEventType,
        text: Option<char>,
    ) -> Vec<u8> {
        let mut result = Vec::with_capacity(16);
        result.extend_from_slice(b"\x1b[");
        result.extend_from_slice(key_code.to_string().as_bytes());

        let mod_value = modifiers.encode();
        let has_event_type =
            self.state.report_events() && event_type != KeyEventType::Press;

        if mod_value.is_some() || has_event_type || text.is_some() {
            result.push(b';');
            if let Some(m) = mod_value {
                result.extend_from_slice(m.to_string().as_bytes());
            } else if has_event_type {
                result.push(b'1'); // Default modifier value
            }

            if has_event_type {
                result.push(b':');
                result.extend_from_slice(
                    (event_type as u8).to_string().as_bytes(),
                );
            }
        }

        if let Some(text_char) = text {
            result.push(b';');
            result.extend_from_slice((text_char as u32).to_string().as_bytes());
        }

        result.push(b'u');
        result
    }

    /// Encodes functional keys in legacy mode.
    fn encode_legacy_functional(
        &self,
        key: FunctionalKey,
        modifiers: Modifiers,
    ) -> Vec<u8> {
        let mod_param = modifiers.encode();

        match key {
            FunctionalKey::Escape => {
                if modifiers.alt {
                    vec![0x1b, 0x1b]
                } else {
                    vec![0x1b]
                }
            }
            FunctionalKey::Enter => {
                if modifiers.alt {
                    vec![0x1b, 0x0d]
                } else {
                    vec![0x0d]
                }
            }
            FunctionalKey::Tab => {
                if modifiers.shift && !modifiers.alt && !modifiers.ctrl {
                    // Shift+Tab -> CSI Z
                    vec![0x1b, b'[', b'Z']
                } else if modifiers.alt {
                    vec![0x1b, 0x09]
                } else {
                    vec![0x09]
                }
            }
            FunctionalKey::Backspace => {
                if modifiers.ctrl {
                    if modifiers.alt {
                        vec![0x1b, 0x08]
                    } else {
                        vec![0x08]
                    }
                } else if modifiers.alt {
                    vec![0x1b, 0x7f]
                } else {
                    vec![0x7f]
                }
            }
            // Arrow keys
            FunctionalKey::Up => self.encode_arrow(b'A', mod_param),
            FunctionalKey::Down => self.encode_arrow(b'B', mod_param),
            FunctionalKey::Right => self.encode_arrow(b'C', mod_param),
            FunctionalKey::Left => self.encode_arrow(b'D', mod_param),
            FunctionalKey::Home => self.encode_arrow(b'H', mod_param),
            FunctionalKey::End => self.encode_arrow(b'F', mod_param),
            // Function keys F1-F4 (SS3 in legacy mode without modifiers)
            FunctionalKey::F1 => self.encode_f1_f4(b'P', mod_param),
            FunctionalKey::F2 => self.encode_f1_f4(b'Q', mod_param),
            FunctionalKey::F3 => self.encode_f1_f4(b'R', mod_param),
            FunctionalKey::F4 => self.encode_f1_f4(b'S', mod_param),
            // Function keys F5-F12 (CSI number ~)
            FunctionalKey::F5 => self.encode_tilde(15, mod_param),
            FunctionalKey::F6 => self.encode_tilde(17, mod_param),
            FunctionalKey::F7 => self.encode_tilde(18, mod_param),
            FunctionalKey::F8 => self.encode_tilde(19, mod_param),
            FunctionalKey::F9 => self.encode_tilde(20, mod_param),
            FunctionalKey::F10 => self.encode_tilde(21, mod_param),
            FunctionalKey::F11 => self.encode_tilde(23, mod_param),
            FunctionalKey::F12 => self.encode_tilde(24, mod_param),
            // Navigation keys
            FunctionalKey::Insert => self.encode_tilde(2, mod_param),
            FunctionalKey::Delete => self.encode_tilde(3, mod_param),
            FunctionalKey::PageUp => self.encode_tilde(5, mod_param),
            FunctionalKey::PageDown => self.encode_tilde(6, mod_param),
            // Other functional keys - encode as CSI u
            _ => {
                let key_code = key as u32;
                self.encode_csi_u(
                    key_code,
                    modifiers,
                    KeyEventType::Press,
                    None,
                )
            }
        }
    }

    /// Encodes arrow/home/end keys based on DECCKM mode:
    /// - Normal mode (application_cursor_keys=false): CSI letter (ESC [ letter)
    /// - Application mode (application_cursor_keys=true): SS3 letter (ESC O letter)
    /// With modifiers, always use CSI 1;mod letter format.
    fn encode_arrow(&self, letter: u8, mod_param: Option<u8>) -> Vec<u8> {
        if let Some(m) = mod_param {
            // With modifiers: CSI 1;mod letter
            let mut result = vec![0x1b, b'[', b'1', b';'];
            result.extend_from_slice(m.to_string().as_bytes());
            result.push(letter);
            result
        } else if self.application_cursor_keys {
            // Application cursor mode: SS3 letter (ESC O letter)
            vec![0x1b, b'O', letter]
        } else {
            // Normal cursor mode: CSI letter (ESC [ letter)
            vec![0x1b, b'[', letter]
        }
    }

    /// Encodes F1-F4: SS3 letter (no mods) or CSI 1;mod letter (with mods).
    fn encode_f1_f4(&self, letter: u8, mod_param: Option<u8>) -> Vec<u8> {
        if let Some(m) = mod_param {
            let mut result = vec![0x1b, b'[', b'1', b';'];
            result.extend_from_slice(m.to_string().as_bytes());
            result.push(letter);
            result
        } else {
            vec![0x1b, b'O', letter]
        }
    }

    /// Encodes CSI number ; modifier ~ format.
    fn encode_tilde(&self, number: u8, mod_param: Option<u8>) -> Vec<u8> {
        let mut result = vec![0x1b, b'['];
        result.extend_from_slice(number.to_string().as_bytes());
        if let Some(m) = mod_param {
            result.push(b';');
            result.extend_from_slice(m.to_string().as_bytes());
        }
        result.push(b'~');
        result
    }

    /// Encodes text keys in legacy mode.
    fn encode_legacy_text(&self, c: char, modifiers: Modifiers) -> Vec<u8> {
        // For plain text without modifiers, just send UTF-8
        if !modifiers.any() {
            let mut buf = [0u8; 4];
            let s = c.encode_utf8(&mut buf);
            return s.as_bytes().to_vec();
        }

        // Handle ctrl modifier for ASCII keys
        if modifiers.ctrl && !modifiers.shift && c.is_ascii_lowercase() {
            let ctrl_code = (c as u8) - b'a' + 1;
            if modifiers.alt {
                return vec![0x1b, ctrl_code];
            } else {
                return vec![ctrl_code];
            }
        }

        // Handle ctrl+space
        if modifiers.ctrl && c == ' ' {
            if modifiers.alt {
                return vec![0x1b, 0x00];
            } else {
                return vec![0x00];
            }
        }

        // Handle alt modifier alone
        if modifiers.alt && !modifiers.ctrl {
            let mut buf = [0u8; 4];
            let s = c.encode_utf8(&mut buf);
            let mut result = vec![0x1b];
            result.extend_from_slice(s.as_bytes());
            return result;
        }

        // Handle shift (just send the shifted character)
        if modifiers.shift && !modifiers.ctrl && !modifiers.alt {
            let shifted = c.to_uppercase().next().unwrap_or(c);
            let mut buf = [0u8; 4];
            let s = shifted.encode_utf8(&mut buf);
            return s.as_bytes().to_vec();
        }

        // For complex modifier combinations, use CSI u encoding even in "legacy" mode
        // This provides better compatibility than dropping the key
        let key_code = c as u32;
        self.encode_csi_u(key_code, modifiers, KeyEventType::Press, None)
    }
}

/// Generates the response for a keyboard mode query (CSI ? u).
pub fn query_response(flags: KeyboardFlags) -> Vec<u8> {
    let mut result = vec![0x1b, b'[', b'?'];
    result.extend_from_slice(flags.bits().to_string().as_bytes());
    result.push(b'u');
    result
}
