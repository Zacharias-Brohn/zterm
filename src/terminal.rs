//! Terminal state management and escape sequence handling.

use crate::keyboard::{query_response, KeyboardState};
use crate::vt_parser::{CsiParams, Handler, Parser};

/// A single cell in the terminal grid.
#[derive(Clone, Copy, Debug)]
pub struct Cell {
    pub character: char,
    pub fg_color: Color,
    pub bg_color: Color,
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            character: ' ',
            fg_color: Color::Default,
            bg_color: Color::Default,
            bold: false,
            italic: false,
            underline: false,
        }
    }
}

/// Terminal colors.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Color {
    #[default]
    Default,
    Rgb(u8, u8, u8),
    Indexed(u8),
}

/// Cursor shape styles (DECSCUSR).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum CursorShape {
    /// Blinking block (and default)
    #[default]
    BlinkingBlock,
    /// Steady block
    SteadyBlock,
    /// Blinking underline
    BlinkingUnderline,
    /// Steady underline
    SteadyUnderline,
    /// Blinking bar (beam)
    BlinkingBar,
    /// Steady bar (beam)
    SteadyBar,
}

/// Mouse tracking mode - determines what mouse events are reported to the application.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum MouseTrackingMode {
    /// No mouse tracking (terminal handles selection).
    #[default]
    None,
    /// X10 compatibility mode - only report button press (mode 9).
    X10,
    /// Normal tracking mode - report button press and release (mode 1000).
    Normal,
    /// Button-event tracking - report press, release, and motion while button pressed (mode 1002).
    ButtonEvent,
    /// Any-event tracking - report all motion events (mode 1003).
    AnyEvent,
}

/// Mouse encoding format - how mouse events are encoded.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum MouseEncoding {
    /// Default X10 encoding (limited to 223 rows/cols).
    #[default]
    X10,
    /// UTF-8 encoding (mode 1005) - deprecated, rarely used.
    Utf8,
    /// SGR extended encoding (mode 1006) - most common modern format.
    Sgr,
    /// URXVT encoding (mode 1015) - rarely used.
    Urxvt,
}

/// Color palette with 256 colors + default fg/bg.
#[derive(Clone)]
pub struct ColorPalette {
    /// 256 indexed colors (ANSI 0-15 + 216 color cube + 24 grayscale).
    pub colors: [[u8; 3]; 256],
    /// Default foreground color.
    pub default_fg: [u8; 3],
    /// Default background color.
    pub default_bg: [u8; 3],
}

impl Default for ColorPalette {
    fn default() -> Self {
        let mut colors = [[0u8; 3]; 256];
        
        // Standard ANSI colors (0-7)
        colors[0] = [0, 0, 0];       // Black
        colors[1] = [204, 0, 0];     // Red
        colors[2] = [0, 204, 0];     // Green
        colors[3] = [204, 204, 0];   // Yellow
        colors[4] = [0, 0, 204];     // Blue
        colors[5] = [204, 0, 204];   // Magenta
        colors[6] = [0, 204, 204];   // Cyan
        colors[7] = [204, 204, 204]; // White
        
        // Bright ANSI colors (8-15)
        colors[8] = [102, 102, 102];  // Bright Black (Gray)
        colors[9] = [255, 0, 0];      // Bright Red
        colors[10] = [0, 255, 0];     // Bright Green
        colors[11] = [255, 255, 0];   // Bright Yellow
        colors[12] = [0, 0, 255];     // Bright Blue
        colors[13] = [255, 0, 255];   // Bright Magenta
        colors[14] = [0, 255, 255];   // Bright Cyan
        colors[15] = [255, 255, 255]; // Bright White
        
        // 216 color cube (16-231)
        for r in 0..6 {
            for g in 0..6 {
                for b in 0..6 {
                    let idx = 16 + r * 36 + g * 6 + b;
                    let to_val = |c: usize| if c == 0 { 0 } else { (55 + c * 40) as u8 };
                    colors[idx] = [to_val(r), to_val(g), to_val(b)];
                }
            }
        }
        
        // 24 grayscale colors (232-255)
        for i in 0..24 {
            let gray = (8 + i * 10) as u8;
            colors[232 + i] = [gray, gray, gray];
        }
        
        Self {
            colors,
            default_fg: [230, 230, 230], // Light gray
            default_bg: [26, 26, 26],    // Dark gray
        }
    }
}

/// Packed color value for GPU transfer (Kitty-style encoding).
/// Layout: type in low 8 bits, RGB value in upper 24 bits.
/// - Type 0: Default (use color table entries 256/257 for fg/bg)
/// - Type 1: Indexed (index in bits 8-15, look up in color table)
/// - Type 2: RGB (R in bits 8-15, G in bits 16-23, B in bits 24-31)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct PackedColor(pub u32);

impl PackedColor {
    /// Color type: default (resolved from color table)
    pub const TYPE_DEFAULT: u8 = 0;
    /// Color type: indexed (look up in 256-color palette)
    pub const TYPE_INDEXED: u8 = 1;
    /// Color type: direct RGB
    pub const TYPE_RGB: u8 = 2;

    /// Create a default color (resolved at render time from palette).
    #[inline]
    pub const fn default_color() -> Self {
        Self(Self::TYPE_DEFAULT as u32)
    }

    /// Create an indexed color (0-255 palette index).
    #[inline]
    pub const fn indexed(index: u8) -> Self {
        Self(Self::TYPE_INDEXED as u32 | ((index as u32) << 8))
    }

    /// Create a direct RGB color.
    #[inline]
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self(Self::TYPE_RGB as u32 | ((r as u32) << 8) | ((g as u32) << 16) | ((b as u32) << 24))
    }

    /// Get the color type.
    #[inline]
    pub const fn color_type(self) -> u8 {
        (self.0 & 0xFF) as u8
    }

    /// Get the index for indexed colors.
    #[inline]
    pub const fn index(self) -> u8 {
        ((self.0 >> 8) & 0xFF) as u8
    }

    /// Get RGB components for RGB colors.
    #[inline]
    pub const fn rgb_components(self) -> (u8, u8, u8) {
        (
            ((self.0 >> 8) & 0xFF) as u8,
            ((self.0 >> 16) & 0xFF) as u8,
            ((self.0 >> 24) & 0xFF) as u8,
        )
    }
}

impl From<Color> for PackedColor {
    fn from(color: Color) -> Self {
        match color {
            Color::Default => Self::default_color(),
            Color::Indexed(idx) => Self::indexed(idx),
            Color::Rgb(r, g, b) => Self::rgb(r, g, b),
        }
    }
}

impl From<&Color> for PackedColor {
    fn from(color: &Color) -> Self {
        (*color).into()
    }
}

/// Packed cell attributes for GPU transfer (Kitty-style).
/// Layout (32-bit bitfield):
/// - bits 0-2: decoration (underline style, 0=none, 1=single, 2=double, 3=curly, etc.)
/// - bit 3: bold
/// - bit 4: italic
/// - bit 5: reverse
/// - bit 6: strike
/// - bit 7: dim
/// - bits 8-31: reserved for future use
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(transparent)]
pub struct CellAttrs(pub u32);

impl CellAttrs {
    pub const DECORATION_MASK: u32 = 0b111;
    pub const BOLD_BIT: u32 = 1 << 3;
    pub const ITALIC_BIT: u32 = 1 << 4;
    pub const REVERSE_BIT: u32 = 1 << 5;
    pub const STRIKE_BIT: u32 = 1 << 6;
    pub const DIM_BIT: u32 = 1 << 7;

    /// Decoration values
    pub const DECO_NONE: u32 = 0;
    pub const DECO_SINGLE: u32 = 1;
    pub const DECO_DOUBLE: u32 = 2;
    pub const DECO_CURLY: u32 = 3;
    pub const DECO_DOTTED: u32 = 4;
    pub const DECO_DASHED: u32 = 5;

    #[inline]
    pub const fn new() -> Self {
        Self(0)
    }

    #[inline]
    pub const fn with_underline(self, style: u32) -> Self {
        Self((self.0 & !Self::DECORATION_MASK) | (style & Self::DECORATION_MASK))
    }

    #[inline]
    pub const fn with_bold(self, bold: bool) -> Self {
        if bold {
            Self(self.0 | Self::BOLD_BIT)
        } else {
            Self(self.0 & !Self::BOLD_BIT)
        }
    }

    #[inline]
    pub const fn with_italic(self, italic: bool) -> Self {
        if italic {
            Self(self.0 | Self::ITALIC_BIT)
        } else {
            Self(self.0 & !Self::ITALIC_BIT)
        }
    }

    #[inline]
    pub const fn with_reverse(self, reverse: bool) -> Self {
        if reverse {
            Self(self.0 | Self::REVERSE_BIT)
        } else {
            Self(self.0 & !Self::REVERSE_BIT)
        }
    }

    #[inline]
    pub const fn with_strike(self, strike: bool) -> Self {
        if strike {
            Self(self.0 | Self::STRIKE_BIT)
        } else {
            Self(self.0 & !Self::STRIKE_BIT)
        }
    }

    #[inline]
    pub const fn with_dim(self, dim: bool) -> Self {
        if dim {
            Self(self.0 | Self::DIM_BIT)
        } else {
            Self(self.0 & !Self::DIM_BIT)
        }
    }

    #[inline]
    pub const fn decoration(self) -> u32 {
        self.0 & Self::DECORATION_MASK
    }

    #[inline]
    pub const fn is_bold(self) -> bool {
        (self.0 & Self::BOLD_BIT) != 0
    }

    #[inline]
    pub const fn is_italic(self) -> bool {
        (self.0 & Self::ITALIC_BIT) != 0
    }

    #[inline]
    pub const fn is_reverse(self) -> bool {
        (self.0 & Self::REVERSE_BIT) != 0
    }

    #[inline]
    pub const fn is_strike(self) -> bool {
        (self.0 & Self::STRIKE_BIT) != 0
    }

    #[inline]
    pub const fn is_dim(self) -> bool {
        (self.0 & Self::DIM_BIT) != 0
    }
}

/// GPU cell data for instanced rendering (Kitty-style).
/// 
/// This struct is uploaded directly to the GPU for each cell.
/// The shader uses instanced rendering where each cell is one instance.
/// 
/// Layout: 20 bytes total
/// - fg: 4 bytes (packed color)
/// - bg: 4 bytes (packed color)  
/// - decoration_fg: 4 bytes (packed color for underline/strikethrough)
/// - sprite_idx: 4 bytes (glyph atlas index, bit 31 = colored glyph flag)
/// - attrs: 4 bytes (packed attributes)
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GPUCell {
    /// Foreground color (packed)
    pub fg: u32,
    /// Background color (packed)
    pub bg: u32,
    /// Decoration color for underline/strikethrough (packed)
    pub decoration_fg: u32,
    /// Sprite index in glyph atlas (bit 31 = colored glyph flag)
    pub sprite_idx: u32,
    /// Packed attributes (bold, italic, underline style, etc.)
    pub attrs: u32,
}

impl GPUCell {
    /// Flag indicating this glyph is colored (e.g., emoji) and should not be tinted
    pub const COLORED_GLYPH_FLAG: u32 = 1 << 31;
    /// Sprite index indicating no glyph (space/empty)
    pub const NO_GLYPH: u32 = 0;

    /// Create an empty cell (space with default colors)
    #[inline]
    pub const fn empty() -> Self {
        Self {
            fg: PackedColor::TYPE_DEFAULT as u32,
            bg: PackedColor::TYPE_DEFAULT as u32,
            decoration_fg: PackedColor::TYPE_DEFAULT as u32,
            sprite_idx: Self::NO_GLYPH,
            attrs: 0,
        }
    }

    /// Create a GPUCell from terminal Cell and a sprite index
    #[inline]
    pub fn from_cell(cell: &Cell, sprite_idx: u32) -> Self {
        let fg = PackedColor::from(&cell.fg_color);
        let bg = PackedColor::from(&cell.bg_color);
        
        let mut attrs = CellAttrs::new();
        if cell.bold {
            attrs = attrs.with_bold(true);
        }
        if cell.italic {
            attrs = attrs.with_italic(true);
        }
        if cell.underline {
            attrs = attrs.with_underline(CellAttrs::DECO_SINGLE);
        }

        Self {
            fg: fg.0,
            bg: bg.0,
            decoration_fg: fg.0, // Use fg color for decoration by default
            sprite_idx,
            attrs: attrs.0,
        }
    }

    /// Set the sprite index
    #[inline]
    pub fn with_sprite(mut self, idx: u32) -> Self {
        self.sprite_idx = idx;
        self
    }

    /// Mark this glyph as colored (emoji)
    #[inline]
    pub fn with_colored_glyph(mut self) -> Self {
        self.sprite_idx |= Self::COLORED_GLYPH_FLAG;
        self
    }

    /// Get the sprite index (without the colored flag)
    #[inline]
    pub const fn get_sprite_idx(self) -> u32 {
        self.sprite_idx & !Self::COLORED_GLYPH_FLAG
    }

    /// Check if this is a colored glyph
    #[inline]
    pub const fn is_colored_glyph(self) -> bool {
        (self.sprite_idx & Self::COLORED_GLYPH_FLAG) != 0
    }
}

impl ColorPalette {
    /// Parse a color specification like "#RRGGBB" or "rgb:RR/GG/BB".
    pub fn parse_color_spec(spec: &str) -> Option<[u8; 3]> {
        let spec = spec.trim();
        
        if let Some(hex) = spec.strip_prefix('#') {
            // #RRGGBB format
            if hex.len() == 6 {
                let r = u8::from_str_radix(&hex[0..2], 16).ok()?;
                let g = u8::from_str_radix(&hex[2..4], 16).ok()?;
                let b = u8::from_str_radix(&hex[4..6], 16).ok()?;
                return Some([r, g, b]);
            }
        } else if let Some(rgb) = spec.strip_prefix("rgb:") {
            // rgb:RR/GG/BB or rgb:RRRR/GGGG/BBBB format
            let parts: Vec<&str> = rgb.split('/').collect();
            if parts.len() == 3 {
                let parse_component = |s: &str| -> Option<u8> {
                    let val = u16::from_str_radix(s, 16).ok()?;
                    // Scale to 8-bit if it's a 16-bit value
                    Some(if s.len() > 2 { (val >> 8) as u8 } else { val as u8 })
                };
                let r = parse_component(parts[0])?;
                let g = parse_component(parts[1])?;
                let b = parse_component(parts[2])?;
                return Some([r, g, b]);
            }
        }
        
        None
    }
    
    /// Get RGBA for a color, using the palette for indexed colors.
    pub fn to_rgba(&self, color: &Color) -> [f32; 4] {
        match color {
            Color::Default => {
                let [r, g, b] = self.default_fg;
                [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0]
            }
            Color::Rgb(r, g, b) => [*r as f32 / 255.0, *g as f32 / 255.0, *b as f32 / 255.0, 1.0],
            Color::Indexed(idx) => {
                let [r, g, b] = self.colors[*idx as usize];
                [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0]
            }
        }
    }
    
    /// Get RGBA for background, using palette default_bg for Color::Default.
    pub fn to_rgba_bg(&self, color: &Color) -> [f32; 4] {
        match color {
            Color::Default => {
                let [r, g, b] = self.default_bg;
                [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0]
            }
            Color::Rgb(r, g, b) => [*r as f32 / 255.0, *g as f32 / 255.0, *b as f32 / 255.0, 1.0],
            Color::Indexed(idx) => {
                let [r, g, b] = self.colors[*idx as usize];
                [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0]
            }
        }
    }
}

/// Saved cursor state for DECSC/DECRC.
#[derive(Clone, Debug, Default)]
struct SavedCursor {
    col: usize,
    row: usize,
    fg: Color,
    bg: Color,
    bold: bool,
    italic: bool,
    underline: bool,
}

/// Alternate screen buffer state.
#[derive(Clone)]
struct AlternateScreen {
    grid: Vec<Vec<Cell>>,
    line_map: Vec<usize>,
    cursor_col: usize,
    cursor_row: usize,
    saved_cursor: SavedCursor,
    scroll_top: usize,
    scroll_bottom: usize,
}

/// Timing stats for performance debugging.
#[derive(Debug, Default)]
pub struct ProcessingStats {
    /// Total time spent in scroll_up operations (nanoseconds).
    pub scroll_up_ns: u64,
    /// Number of scroll_up calls.
    pub scroll_up_count: u32,
    /// Total time spent in scrollback operations (nanoseconds).
    pub scrollback_ns: u64,
    /// Time in VecDeque pop_front.
    pub pop_front_ns: u64,
    /// Time in VecDeque push_back.
    pub push_back_ns: u64,
    /// Time in mem::swap.
    pub swap_ns: u64,
    /// Total time spent in line clearing (nanoseconds).
    pub clear_line_ns: u64,
    /// Total time spent in text handler (nanoseconds).
    pub text_handler_ns: u64,
    /// Number of characters processed.
    pub chars_processed: u32,
}

impl ProcessingStats {
    pub fn reset(&mut self) {
        *self = Self::default();
    }
    
    pub fn log_if_slow(&self, threshold_ms: u64) {
        let total_ms = (self.scroll_up_ns + self.text_handler_ns) / 1_000_000;
        if total_ms >= threshold_ms {
            log::info!(
                "TIMING: scroll_up={:.2}ms ({}x), scrollback={:.2}ms [pop={:.2}ms swap={:.2}ms push={:.2}ms], clear={:.2}ms, text={:.2}ms, chars={}",
                self.scroll_up_ns as f64 / 1_000_000.0,
                self.scroll_up_count,
                self.scrollback_ns as f64 / 1_000_000.0,
                self.pop_front_ns as f64 / 1_000_000.0,
                self.swap_ns as f64 / 1_000_000.0,
                self.push_back_ns as f64 / 1_000_000.0,
                self.clear_line_ns as f64 / 1_000_000.0,
                self.text_handler_ns as f64 / 1_000_000.0,
                self.chars_processed,
            );
        }
    }
}

/// Kitty-style ring buffer for scrollback history.
/// 
/// Pre-allocates all lines upfront to avoid allocation during scrolling.
/// Uses modulo arithmetic for O(1) operations with no memory allocation or
/// pointer chasing - just simple index arithmetic like Kitty's historybuf.
/// 
/// Key insight from Kitty: When the buffer is full, instead of pop_front + push_back
/// (which involves linked-list-style pointer updates in VecDeque), we just:
/// 1. Calculate the insertion slot with modulo arithmetic
/// 2. Increment the start pointer (also with modulo)
/// 
/// This eliminates all per-scroll overhead that was causing timing variance.
pub struct ScrollbackBuffer {
    /// Pre-allocated line storage. All lines are allocated upfront.
    lines: Vec<Vec<Cell>>,
    /// Index of the oldest line (start of valid data).
    start: usize,
    /// Number of valid lines currently stored.
    count: usize,
    /// Maximum capacity (same as lines.len()).
    capacity: usize,
}

impl ScrollbackBuffer {
    /// Creates a new scrollback buffer with the given capacity and column width.
    /// All lines are pre-allocated to avoid any allocation during scrolling.
    pub fn new(capacity: usize, cols: usize) -> Self {
        // Pre-allocate all lines upfront
        let lines = if capacity > 0 {
            (0..capacity).map(|_| vec![Cell::default(); cols]).collect()
        } else {
            Vec::new()
        };
        
        Self {
            lines,
            start: 0,
            count: 0,
            capacity,
        }
    }
    
    /// Returns the number of lines currently stored.
    #[inline]
    pub fn len(&self) -> usize {
        self.count
    }
    
    /// Returns true if the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
    
    /// Returns true if the buffer is at capacity.
    #[inline]
    pub fn is_full(&self) -> bool {
        self.count == self.capacity
    }
    
    /// Push a line into the buffer, returning a mutable reference to write into.
    /// 
    /// If the buffer is full, the oldest line is overwritten and its slot is returned
    /// for reuse (the caller can swap content into it).
    /// 
    /// This is the key operation - it's O(1) with just modulo arithmetic, no allocation.
    #[inline]
    pub fn push(&mut self) -> &mut Vec<Cell> {
        if self.capacity == 0 {
            // Shouldn't happen in normal use, but handle gracefully
            panic!("Cannot push to zero-capacity scrollback buffer");
        }
        
        // Calculate insertion index: (start + count) % capacity
        // This is where the new line goes
        let idx = (self.start + self.count) % self.capacity;
        
        if self.count == self.capacity {
            // Buffer is full - we're overwriting the oldest line
            // Advance start to point to the new oldest line
            self.start = (self.start + 1) % self.capacity;
            // count stays the same
        } else {
            // Buffer not full yet - just increment count
            self.count += 1;
        }
        
        &mut self.lines[idx]
    }
    
    /// Get a line by logical index (0 = oldest, count-1 = newest).
    /// Returns None if index is out of bounds.
    #[inline]
    pub fn get(&self, index: usize) -> Option<&Vec<Cell>> {
        if index >= self.count {
            return None;
        }
        // Map logical index to physical index
        let physical_idx = (self.start + index) % self.capacity;
        Some(&self.lines[physical_idx])
    }
    
    /// Clear all lines from the buffer.
    /// Note: This doesn't deallocate - lines stay allocated for reuse.
    #[inline]
    pub fn clear(&mut self) {
        self.start = 0;
        self.count = 0;
        // Lines remain allocated but logically empty
    }
}

/// The terminal grid state.
pub struct Terminal {
    /// Grid of cells (row-major order).
    /// Access via line_map for correct visual ordering.
    pub grid: Vec<Vec<Cell>>,
    /// Maps visual row index to actual grid row index.
    /// This allows O(1) scrolling by rotating indices instead of moving cells.
    line_map: Vec<usize>,
    /// Number of columns.
    pub cols: usize,
    /// Number of rows.
    pub rows: usize,
    /// Current cursor column (0-indexed).
    pub cursor_col: usize,
    /// Current cursor row (0-indexed).
    pub cursor_row: usize,
    /// Cursor visibility.
    pub cursor_visible: bool,
    /// Cursor shape (block, underline, bar).
    pub cursor_shape: CursorShape,
    /// Current foreground color for new text.
    pub current_fg: Color,
    /// Current background color for new text.
    pub current_bg: Color,
    /// Current bold state.
    pub current_bold: bool,
    /// Current italic state.
    pub current_italic: bool,
    /// Current underline state.
    pub current_underline: bool,
    /// Whether the terminal content has changed.
    pub dirty: bool,
    /// Bitmap of dirty lines - bit N is set if line N needs redrawing.
    /// Supports up to 256 lines (4 x u64).
    pub dirty_lines: [u64; 4],
    /// Scroll region top (0-indexed, inclusive).
    scroll_top: usize,
    /// Scroll region bottom (0-indexed, inclusive).
    scroll_bottom: usize,
    /// Kitty keyboard protocol state.
    pub keyboard: KeyboardState,
    /// Response queue (bytes to send back to PTY).
    response_queue: Vec<u8>,
    /// Color palette (can be modified by OSC sequences).
    pub palette: ColorPalette,
    /// Scrollback buffer (lines that scrolled off the top).
    /// Uses a Kitty-style ring buffer for O(1) operations with no allocation.
    pub scrollback: ScrollbackBuffer,
    /// Current scroll offset (0 = viewing live terminal, >0 = viewing history).
    pub scroll_offset: usize,
    /// Mouse tracking mode (what events to report to application).
    pub mouse_tracking: MouseTrackingMode,
    /// Mouse encoding format (how to encode mouse events).
    pub mouse_encoding: MouseEncoding,
    /// Saved cursor state (DECSC/DECRC).
    saved_cursor: SavedCursor,
    /// Alternate screen buffer (for fullscreen apps like vim, less).
    alternate_screen: Option<AlternateScreen>,
    /// Whether we're currently using the alternate screen.
    pub using_alternate_screen: bool,
    /// Application cursor keys mode (DECCKM) - arrows send ESC O instead of ESC [.
    pub application_cursor_keys: bool,
    /// Auto-wrap mode (DECAWM) - wrap at end of line.
    auto_wrap: bool,
    /// Bracketed paste mode - wrap pasted text with escape sequences.
    pub bracketed_paste: bool,
    /// Focus event reporting mode.
    pub focus_reporting: bool,
    /// Synchronized output mode (for reducing flicker).
    synchronized_output: bool,
    /// Pool of pre-allocated empty lines to avoid allocation during scrolling.
    /// When we need a new line, we pop from this pool instead of allocating.
    line_pool: Vec<Vec<Cell>>,
    /// VT parser for escape sequence handling.
    parser: Option<Parser>,
    /// Performance timing stats (for debugging).
    pub stats: ProcessingStats,
}

impl Terminal {
    /// Default scrollback limit (10,000 lines for better cache performance).
    pub const DEFAULT_SCROLLBACK_LIMIT: usize = 10_000;
    
    /// Size of the line pool for recycling allocations.
    /// This avoids allocation during the first N scrolls before scrollback is full.
    const LINE_POOL_SIZE: usize = 64;

    /// Creates a new terminal with the given dimensions and scrollback limit.
    pub fn new(cols: usize, rows: usize, scrollback_limit: usize) -> Self {
        let grid = vec![vec![Cell::default(); cols]; rows];
        let line_map: Vec<usize> = (0..rows).collect();
        
        // Pre-allocate a pool of empty lines to avoid allocation during scrolling
        let line_pool: Vec<Vec<Cell>> = (0..Self::LINE_POOL_SIZE)
            .map(|_| vec![Cell::default(); cols])
            .collect();

        Self {
            grid,
            line_map,
            cols,
            rows,
            cursor_col: 0,
            cursor_row: 0,
            cursor_visible: true,
            cursor_shape: CursorShape::default(),
            current_fg: Color::Default,
            current_bg: Color::Default,
            current_bold: false,
            current_italic: false,
            current_underline: false,
            dirty: true,
            dirty_lines: [!0u64; 4], // All lines dirty initially
            scroll_top: 0,
            scroll_bottom: rows.saturating_sub(1),
            keyboard: KeyboardState::new(),
            response_queue: Vec::new(),
            palette: ColorPalette::default(),
            scrollback: ScrollbackBuffer::new(scrollback_limit, cols),
            scroll_offset: 0,
            mouse_tracking: MouseTrackingMode::default(),
            mouse_encoding: MouseEncoding::default(),
            saved_cursor: SavedCursor::default(),
            alternate_screen: None,
            using_alternate_screen: false,
            application_cursor_keys: false,
            auto_wrap: true, // Auto-wrap is on by default
            bracketed_paste: false,
            focus_reporting: false,
            synchronized_output: false,
            line_pool,
            parser: Some(Parser::new()),
            stats: ProcessingStats::default(),
        }
    }
    
    /// Return a line to the pool for reuse (if pool isn't full).
    #[allow(dead_code)]
    #[inline]
    fn return_line_to_pool(&mut self, line: Vec<Cell>) {
        if self.line_pool.len() < Self::LINE_POOL_SIZE {
            self.line_pool.push(line);
        }
        // Otherwise, let the line be dropped
    }
    
    /// Mark a specific line as dirty (needs redrawing).
    #[inline]
    pub fn mark_line_dirty(&mut self, line: usize) {
        if line < 256 {
            let word = line / 64;
            let bit = line % 64;
            self.dirty_lines[word] |= 1u64 << bit;
        }
    }
    
    /// Mark all lines as dirty.
    #[inline]
    pub fn mark_all_lines_dirty(&mut self) {
        self.dirty_lines = [!0u64; 4];
    }
    
    /// Check if a line is dirty.
    #[inline]
    pub fn is_line_dirty(&self, line: usize) -> bool {
        if line < 256 {
            let word = line / 64;
            let bit = line % 64;
            (self.dirty_lines[word] & (1u64 << bit)) != 0
        } else {
            true // Lines beyond 256 are always considered dirty
        }
    }
    
    /// Clear all dirty line flags.
    #[inline]
    pub fn clear_dirty_lines(&mut self) {
        self.dirty_lines = [0u64; 4];
    }
    
    /// Get the dirty lines bitmap (for passing to shm).
    #[inline]
    pub fn get_dirty_lines(&self) -> u64 {
        // Return first 64 lines worth of dirty bits (most common case)
        self.dirty_lines[0]
    }
    
    /// Get the actual grid row index for a visual row.
    #[inline]
    pub fn grid_row(&self, visual_row: usize) -> usize {
        self.line_map[visual_row]
    }
    
    /// Get a reference to a row by visual index.
    #[inline]
    pub fn row(&self, visual_row: usize) -> &Vec<Cell> {
        &self.grid[self.line_map[visual_row]]
    }
    
    /// Get a mutable reference to a row by visual index.
    #[inline]
    pub fn row_mut(&mut self, visual_row: usize) -> &mut Vec<Cell> {
        let idx = self.line_map[visual_row];
        &mut self.grid[idx]
    }
    
    /// Clear a row (by actual grid index, not visual).
    #[inline]
    fn clear_grid_row(&mut self, grid_row: usize) {
        let blank = self.blank_cell();
        self.grid[grid_row].fill(blank);
    }

    /// Create a blank cell with the current background color (BCE - Background Color Erase).
    #[inline]
    fn blank_cell(&self) -> Cell {
        Cell {
            character: ' ',
            fg_color: Color::Default,
            bg_color: self.current_bg,
            bold: false,
            italic: false,
            underline: false,
        }
    }

    /// Takes any pending response bytes to send to the PTY.
    pub fn take_response(&mut self) -> Option<Vec<u8>> {
        if self.response_queue.is_empty() {
            None
        } else {
            Some(std::mem::take(&mut self.response_queue))
        }
    }

    /// Processes raw bytes from the PTY using the internal VT parser.
    /// Uses Kitty-style architecture: UTF-8 decode until ESC, then parse escape sequences.
    pub fn process(&mut self, bytes: &[u8]) {
        // We need to temporarily take ownership of the parser to satisfy the borrow checker,
        // since parse() needs &mut self for both parser and handler (Terminal).
        // Use Option::take to avoid creating a new default parser each time.
        if let Some(mut parser) = self.parser.take() {
            parser.parse(bytes, self);
            self.parser = Some(parser);
        }
        self.dirty = true;
    }

    /// Resizes the terminal grid.
    pub fn resize(&mut self, cols: usize, rows: usize) {
        if cols == self.cols && rows == self.rows {
            return;
        }

        // Create new grid
        let mut new_grid = vec![vec![Cell::default(); cols]; rows];

        // Copy existing content using line_map for correct visual ordering
        for visual_row in 0..rows.min(self.rows) {
            let old_grid_row = self.line_map[visual_row];
            for col in 0..cols.min(self.cols) {
                new_grid[visual_row][col] = self.grid[old_grid_row][col].clone();
            }
        }

        self.grid = new_grid;
        // Reset line_map to identity (0, 1, 2, ...)
        self.line_map = (0..rows).collect();
        self.cols = cols;
        self.rows = rows;

        // Reset scroll region to full screen
        self.scroll_top = 0;
        self.scroll_bottom = rows.saturating_sub(1);

        // Adjust cursor position
        self.cursor_col = self.cursor_col.min(cols.saturating_sub(1));
        self.cursor_row = self.cursor_row.min(rows.saturating_sub(1));
        self.dirty = true;
        self.mark_all_lines_dirty();
    }

    /// Switch to alternate screen buffer.
    fn enter_alternate_screen(&mut self) {
        if self.using_alternate_screen {
            return; // Already in alternate screen
        }

        // Save main screen state
        self.alternate_screen = Some(AlternateScreen {
            grid: self.grid.clone(),
            line_map: self.line_map.clone(),
            cursor_col: self.cursor_col,
            cursor_row: self.cursor_row,
            saved_cursor: self.saved_cursor.clone(),
            scroll_top: self.scroll_top,
            scroll_bottom: self.scroll_bottom,
        });

        // Clear the screen for alternate buffer
        self.grid = vec![vec![Cell::default(); self.cols]; self.rows];
        self.line_map = (0..self.rows).collect();
        self.cursor_col = 0;
        self.cursor_row = 0;
        // Reset scroll region to full screen for alternate buffer
        self.scroll_top = 0;
        self.scroll_bottom = self.rows.saturating_sub(1);
        // Reset scroll offset (can't view scrollback in alternate screen)
        self.scroll_offset = 0;
        self.using_alternate_screen = true;
        self.mark_all_lines_dirty();
        self.dirty = true;
        log::debug!(
            "Entered alternate screen buffer: rows={}, cols={}, scroll_region={}-{}, dirty_lines={:016x}{:016x}{:016x}{:016x}",
            self.rows, self.cols, self.scroll_top, self.scroll_bottom,
            self.dirty_lines[3], self.dirty_lines[2], self.dirty_lines[1], self.dirty_lines[0]
        );
    }

    /// Switch back to main screen buffer.
    fn leave_alternate_screen(&mut self) {
        if !self.using_alternate_screen {
            return; // Not in alternate screen
        }

        if let Some(saved) = self.alternate_screen.take() {
            self.grid = saved.grid;
            self.line_map = saved.line_map;
            self.cursor_col = saved.cursor_col;
            self.cursor_row = saved.cursor_row;
            self.saved_cursor = saved.saved_cursor;
            self.scroll_top = saved.scroll_top;
            self.scroll_bottom = saved.scroll_bottom;
        }

        self.using_alternate_screen = false;
        self.mark_all_lines_dirty();
        log::debug!("Left alternate screen buffer");
    }

    /// Scrolls the scroll region up by n lines.
    /// Uses line_map rotation for O(n) instead of O(n*cols) cell copying.
    fn scroll_up(&mut self, n: usize) {
        let region_size = self.scroll_bottom - self.scroll_top + 1;
        let n = n.min(region_size);
        
        self.stats.scroll_up_count += n as u32;
        
        for _ in 0..n {
            // Save the top line's grid index before rotation
            let recycled_grid_row = self.line_map[self.scroll_top];
            
            // Save to scrollback only if scrolling from the very top of the screen
            // AND not in alternate screen mode (alternate screen never uses scrollback)
            // AND scrollback is enabled (capacity > 0)
            if self.scroll_top == 0 && !self.using_alternate_screen && self.scrollback.capacity > 0 {
                // Get a slot in the ring buffer - this is O(1) with just modulo arithmetic
                // If buffer is full, this overwrites the oldest line (perfect for our swap)
                let dest = self.scrollback.push();
                // Swap grid row content into scrollback slot
                // The scrollback slot's old content (if any) moves to the grid row
                std::mem::swap(&mut self.grid[recycled_grid_row], dest);
                // Clear the grid row (now contains old scrollback data or empty)
                self.clear_grid_row(recycled_grid_row);
            } else {
                // Not saving to scrollback - just clear the line
                self.clear_grid_row(recycled_grid_row);
            }
            
            // Rotate line_map: shift all indices up within scroll region using memmove
            self.line_map.copy_within(self.scroll_top + 1..=self.scroll_bottom, self.scroll_top);
            self.line_map[self.scroll_bottom] = recycled_grid_row;
        }
        
        // Mark all lines dirty with a single bitmask operation instead of loop
        self.mark_region_dirty(self.scroll_top, self.scroll_bottom);
    }
    
    /// Mark a range of lines as dirty efficiently.
    #[inline]
    fn mark_region_dirty(&mut self, start: usize, end: usize) {
        // For small regions (< 64 lines), this is faster than individual calls
        for line in start..=end.min(255) {
            let word = line / 64;
            let bit = line % 64;
            self.dirty_lines[word] |= 1u64 << bit;
        }
    }

    /// Scrolls the scroll region down by n lines.
    /// Uses line_map rotation for O(n) instead of O(n*cols) cell copying.
    fn scroll_down(&mut self, n: usize) {
        let region_size = self.scroll_bottom - self.scroll_top + 1;
        let n = n.min(region_size);
        
        for _ in 0..n {
            // Save the bottom line's grid index before rotation
            let recycled_grid_row = self.line_map[self.scroll_bottom];
            
            // Rotate line_map: shift all indices down within scroll region using memmove
            self.line_map.copy_within(self.scroll_top..self.scroll_bottom, self.scroll_top + 1);
            self.line_map[self.scroll_top] = recycled_grid_row;
            
            // Clear the recycled line (now at visual top of scroll region)
            self.clear_grid_row(recycled_grid_row);
        }
        
        // Mark all lines in the scroll region as dirty.
        self.mark_region_dirty(self.scroll_top, self.scroll_bottom);
    }

    /// Scrolls the viewport up (into scrollback history) by n lines.
    /// Returns the new scroll offset.
    /// Note: Scrollback is disabled in alternate screen mode.
    pub fn scroll_viewport_up(&mut self, n: usize) -> usize {
        // Alternate screen has no scrollback
        if self.using_alternate_screen {
            return 0;
        }
        let max_offset = self.scrollback.len();
        let new_offset = (self.scroll_offset + n).min(max_offset);
        if new_offset != self.scroll_offset {
            self.scroll_offset = new_offset;
            self.dirty = true;
            self.mark_all_lines_dirty(); // All visible content changes when scrolling
        }
        self.scroll_offset
    }

    /// Scrolls the viewport down (toward live terminal) by n lines.
    /// Returns the new scroll offset.
    /// Note: Scrollback is disabled in alternate screen mode.
    pub fn scroll_viewport_down(&mut self, n: usize) -> usize {
        // Alternate screen has no scrollback
        if self.using_alternate_screen {
            return 0;
        }
        let new_offset = self.scroll_offset.saturating_sub(n);
        if new_offset != self.scroll_offset {
            self.scroll_offset = new_offset;
            self.dirty = true;
            self.mark_all_lines_dirty(); // All visible content changes when scrolling
        }
        self.scroll_offset
    }

    /// Resets viewport to show live terminal (scroll_offset = 0).
    pub fn reset_scroll(&mut self) {
        if self.scroll_offset != 0 {
            self.scroll_offset = 0;
            self.dirty = true;
            self.mark_all_lines_dirty(); // All visible content changes when scrolling
        }
    }
    
    /// Scroll the viewport by the given number of lines.
    /// Positive values scroll up (into history), negative values scroll down (toward live).
    pub fn scroll(&mut self, lines: i32) {
        if lines > 0 {
            self.scroll_viewport_up(lines as usize);
        } else if lines < 0 {
            self.scroll_viewport_down((-lines) as usize);
        }
    }
    
    /// Encode a mouse event based on current tracking mode and encoding.
    /// Returns the escape sequence to send to the application, or empty vec if no tracking.
    pub fn encode_mouse(
        &self,
        button: u8,
        col: u16,
        row: u16,
        pressed: bool,
        is_motion: bool,
        modifiers: u8,
    ) -> Vec<u8> {
        // Check if we should report this event based on tracking mode
        match self.mouse_tracking {
            MouseTrackingMode::None => return Vec::new(),
            MouseTrackingMode::X10 => {
                // X10 only reports button press, not release or motion
                if !pressed || is_motion {
                    return Vec::new();
                }
            }
            MouseTrackingMode::Normal => {
                // Normal reports press and release, not motion
                if is_motion {
                    return Vec::new();
                }
            }
            MouseTrackingMode::ButtonEvent => {
                // Button-event reports press, release, and motion while button pressed
                // The is_motion flag indicates motion events
            }
            MouseTrackingMode::AnyEvent => {
                // Any-event reports all motion
            }
        }
        
        // Build the button code
        // Bits 0-1: button (0=left, 1=middle, 2=right, 3=release)
        // Bit 2: shift
        // Bit 3: meta/alt
        // Bit 4: control
        // Bits 5-6: 00=press, 01=motion with button, 10=scroll
        let mut cb = button;
        
        // Handle release
        if !pressed && !is_motion {
            // For SGR encoding, we keep the button number
            // For X10/UTF-8 encoding, release is button 3
            if self.mouse_encoding != MouseEncoding::Sgr {
                cb = 3;
            }
        }
        
        // Add modifiers
        cb |= modifiers << 2;
        
        // Add motion flag
        if is_motion {
            cb |= 32;
        }
        
        // Convert to 1-based coordinates
        let col = col.saturating_add(1);
        let row = row.saturating_add(1);
        
        match self.mouse_encoding {
            MouseEncoding::X10 => {
                // X10 encoding: ESC [ M Cb Cx Cy
                // Each value is encoded as a byte with 32 added
                // Limited to 223 columns/rows
                let cb = (cb + 32).min(255);
                let cx = ((col as u8).min(223) + 32).min(255);
                let cy = ((row as u8).min(223) + 32).min(255);
                vec![0x1b, b'[', b'M', cb, cx, cy]
            }
            MouseEncoding::Utf8 => {
                // UTF-8 encoding: ESC [ M Cb Cx Cy
                // Values > 127 are UTF-8 encoded
                // This is deprecated and rarely used
                let cb = cb + 32;
                let cx = (col as u8).saturating_add(32);
                let cy = (row as u8).saturating_add(32);
                vec![0x1b, b'[', b'M', cb, cx, cy]
            }
            MouseEncoding::Sgr => {
                // SGR encoding: ESC [ < Cb ; Cx ; Cy M/m
                // M for press, m for release
                // Most modern and recommended format
                let suffix = if pressed { b'M' } else { b'm' };
                format!("\x1b[<{};{};{}{}", cb, col, row, suffix as char).into_bytes()
            }
            MouseEncoding::Urxvt => {
                // URXVT encoding: ESC [ Cb ; Cx ; Cy M
                // Similar to SGR but uses decimal with offset
                let cb = cb + 32;
                format!("\x1b[{};{};{}M", cb, col, row).into_bytes()
            }
        }
    }

    /// Returns the visible rows accounting for scroll offset.
    /// This combines scrollback lines with the current grid.
    pub fn visible_rows(&self) -> Vec<&Vec<Cell>> {
        let mut rows = Vec::with_capacity(self.rows);
        
        if self.scroll_offset == 0 {
            // No scrollback viewing, just return the grid via line_map
            for visual_row in 0..self.rows {
                rows.push(&self.grid[self.line_map[visual_row]]);
            }
        } else {
            // We're viewing scrollback
            // scroll_offset = how many lines back we're looking
            let scrollback_len = self.scrollback.len();
            
            for i in 0..self.rows {
                // Calculate which line to show
                // If scroll_offset = 5, we want to show 5 lines from scrollback at the top
                let lines_from_scrollback = self.scroll_offset.min(self.rows);
                
                if i < lines_from_scrollback {
                    // This row comes from scrollback
                    // Use ring buffer's get() method with logical index
                    let scrollback_idx = scrollback_len - self.scroll_offset + i;
                    if let Some(line) = self.scrollback.get(scrollback_idx) {
                        rows.push(line);
                    } else {
                        // Shouldn't happen, but fall back to grid
                        rows.push(&self.grid[self.line_map[i]]);
                    }
                } else {
                    // This row comes from the grid
                    let grid_visual_idx = i - lines_from_scrollback;
                    if grid_visual_idx < self.rows {
                        rows.push(&self.grid[self.line_map[grid_visual_idx]]);
                    }
                }
            }
        }
        
        rows
    }

    /// Inserts n blank lines at the cursor position, scrolling lines below down.
    /// Uses line_map rotation for efficiency.
    fn insert_lines(&mut self, n: usize) {
        if self.cursor_row < self.scroll_top || self.cursor_row > self.scroll_bottom {
            return;
        }
        let n = n.min(self.scroll_bottom - self.cursor_row + 1);
        
        for _ in 0..n {
            // Save the bottom line's grid index before rotation
            let recycled_grid_row = self.line_map[self.scroll_bottom];
            
            // Rotate line_map: shift lines from cursor to bottom down by 1
            // The bottom line becomes the new line at cursor position
            for i in (self.cursor_row + 1..=self.scroll_bottom).rev() {
                self.line_map[i] = self.line_map[i - 1];
            }
            self.line_map[self.cursor_row] = recycled_grid_row;
            
            // Clear the recycled line (now at cursor position)
            self.clear_grid_row(recycled_grid_row);
            
            // Mark affected lines dirty
            for line in self.cursor_row..=self.scroll_bottom {
                self.mark_line_dirty(line);
            }
        }
    }

    /// Deletes n lines at the cursor position, scrolling lines below up.
    /// Uses line_map rotation for efficiency.
    fn delete_lines(&mut self, n: usize) {
        if self.cursor_row < self.scroll_top || self.cursor_row > self.scroll_bottom {
            return;
        }
        let n = n.min(self.scroll_bottom - self.cursor_row + 1);
        
        for _ in 0..n {
            // Save the line at cursor's grid index before rotation
            let recycled_grid_row = self.line_map[self.cursor_row];
            
            // Rotate line_map: shift lines from cursor to bottom up by 1
            // The cursor line becomes the new bottom line
            for i in self.cursor_row..self.scroll_bottom {
                self.line_map[i] = self.line_map[i + 1];
            }
            self.line_map[self.scroll_bottom] = recycled_grid_row;
            
            // Clear the recycled line (now at bottom of scroll region)
            self.clear_grid_row(recycled_grid_row);
            
            // Mark affected lines dirty
            for line in self.cursor_row..=self.scroll_bottom {
                self.mark_line_dirty(line);
            }
        }
    }

    /// Inserts n blank characters at the cursor, shifting existing chars right.
    fn insert_characters(&mut self, n: usize) {
        let grid_row = self.line_map[self.cursor_row];
        let blank = self.blank_cell();
        let row = &mut self.grid[grid_row];
        let n = n.min(self.cols - self.cursor_col);
        // Remove n characters from the end
        for _ in 0..n {
            row.pop();
        }
        // Insert n blank characters at cursor position
        for _ in 0..n {
            row.insert(self.cursor_col, blank);
        }
        self.mark_line_dirty(self.cursor_row);
    }

    /// Deletes n characters at the cursor, shifting remaining chars left.
    fn delete_characters(&mut self, n: usize) {
        let grid_row = self.line_map[self.cursor_row];
        let blank = self.blank_cell();
        let row = &mut self.grid[grid_row];
        let n = n.min(self.cols - self.cursor_col);
        // Remove n characters at cursor position
        for _ in 0..n {
            if self.cursor_col < row.len() {
                row.remove(self.cursor_col);
            }
        }
        // Pad with blank characters at the end
        while row.len() < self.cols {
            row.push(blank);
        }
        self.mark_line_dirty(self.cursor_row);
    }

    /// Erases n characters at the cursor (replaces with spaces, doesn't shift).
    fn erase_characters(&mut self, n: usize) {
        let grid_row = self.line_map[self.cursor_row];
        let n = n.min(self.cols - self.cursor_col);
        let blank = self.blank_cell();
        for i in 0..n {
            if self.cursor_col + i < self.cols {
                self.grid[grid_row][self.cursor_col + i] = blank;
            }
        }
        self.mark_line_dirty(self.cursor_row);
    }

    /// Clears the current line from cursor to end.
    fn clear_line_from_cursor(&mut self) {
        let grid_row = self.line_map[self.cursor_row];
        let blank = self.blank_cell();
        for col in self.cursor_col..self.cols {
            self.grid[grid_row][col] = blank;
        }
        self.mark_line_dirty(self.cursor_row);
    }

    /// Clears the entire screen, pushing current content to scrollback first (main screen only).
    fn clear_screen(&mut self) {
        // Push all visible lines to scrollback before clearing
        // This preserves the content in history so the user can scroll back to see it
        // BUT: Only do this for main screen, not alternate screen
        // AND only if scrollback is enabled
        if !self.using_alternate_screen && self.scrollback.capacity > 0 {
            for visual_row in 0..self.rows {
                let grid_row = self.line_map[visual_row];
                // Get a slot in the ring buffer and swap content into it
                let dest = self.scrollback.push();
                std::mem::swap(&mut self.grid[grid_row], dest);
            }
        }
        
        // Now clear the grid with BCE
        let blank = self.blank_cell();
        for row in &mut self.grid {
            row.fill(blank);
        }
        self.mark_all_lines_dirty();
        self.cursor_col = 0;
        self.cursor_row = 0;
    }
}

impl Handler for Terminal {
    /// Handle a chunk of decoded text (Unicode codepoints).
    /// This includes control characters (0x00-0x1F except ESC).
    fn text(&mut self, chars: &[char]) {
        // Cache the current line to avoid repeated line_map lookups
        let mut cached_row = self.cursor_row;
        let mut grid_row = self.line_map[cached_row];
        
        for &c in chars {
            match c {
                // Bell
                '\x07' => {
                    // BEL - ignore for now (could trigger visual bell)
                }
                // Backspace
                '\x08' => {
                    if self.cursor_col > 0 {
                        self.cursor_col -= 1;
                    }
                }
                // Tab
                '\x09' => {
                    let next_tab = (self.cursor_col / 8 + 1) * 8;
                    self.cursor_col = next_tab.min(self.cols - 1);
                }
                // Line feed, Vertical tab, Form feed
                '\x0A' | '\x0B' | '\x0C' => {
                    self.cursor_row += 1;
                    if self.cursor_row > self.scroll_bottom {
                        self.scroll_up(1);
                        self.cursor_row = self.scroll_bottom;
                    }
                    // Update cache after line change
                    cached_row = self.cursor_row;
                    grid_row = self.line_map[cached_row];
                }
                // Carriage return
                '\x0D' => {
                    self.cursor_col = 0;
                }
                // Printable characters (including all Unicode)
                c if c >= ' ' => {
                    // Handle wrap
                    if self.cursor_col >= self.cols {
                        if self.auto_wrap {
                            self.cursor_col = 0;
                            self.cursor_row += 1;
                            if self.cursor_row > self.scroll_bottom {
                                self.scroll_up(1);
                                self.cursor_row = self.scroll_bottom;
                            }
                            // Update cache after line change
                            cached_row = self.cursor_row;
                            grid_row = self.line_map[cached_row];
                        } else {
                            self.cursor_col = self.cols - 1;
                        }
                    }
                    
                    // Write character directly using cached grid_row
                    self.grid[grid_row][self.cursor_col] = Cell {
                        character: c,
                        fg_color: self.current_fg,
                        bg_color: self.current_bg,
                        bold: self.current_bold,
                        italic: self.current_italic,
                        underline: self.current_underline,
                    };
                    self.cursor_col += 1;
                }
                // Other control chars - ignore
                _ => {}
            }
        }
        
        // Mark all lines dirty at the end (we touched many lines)
        self.mark_all_lines_dirty();
    }

    /// Handle control characters embedded in escape sequences.
    fn control(&mut self, byte: u8) {
        match byte {
            0x08 => {
                if self.cursor_col > 0 {
                    self.cursor_col -= 1;
                }
            }
            0x09 => {
                let next_tab = (self.cursor_col / 8 + 1) * 8;
                self.cursor_col = next_tab.min(self.cols - 1);
            }
            0x0A | 0x0B | 0x0C => {
                self.cursor_row += 1;
                if self.cursor_row > self.scroll_bottom {
                    self.scroll_up(1);
                    self.cursor_row = self.scroll_bottom;
                }
            }
            0x0D => {
                self.cursor_col = 0;
            }
            _ => {}
        }
    }

    /// Handle a complete OSC sequence.
    fn osc(&mut self, data: &[u8]) {
        // Parse OSC format: "number;content" or "number;arg;content"
        // Split on ';'
        let parts: Vec<&[u8]> = data.splitn(3, |&b| b == b';').collect();
        if parts.is_empty() {
            return;
        }

        // First part is the OSC number
        let osc_num = match std::str::from_utf8(parts[0]) {
            Ok(s) => s.parse::<u32>().unwrap_or(u32::MAX),
            Err(_) => return,
        };

        match osc_num {
            // OSC 0, 1, 2 - Set window title (ignore for now)
            0 | 1 | 2 => {}
            // OSC 4 - Set/query indexed color
            4 => {
                // Format: OSC 4;index;color ST
                if parts.len() >= 3 {
                    if let Ok(index_str) = std::str::from_utf8(parts[1]) {
                        if let Ok(index) = index_str.parse::<u8>() {
                            if let Ok(color_spec) = std::str::from_utf8(parts[2]) {
                                if let Some(rgb) = ColorPalette::parse_color_spec(color_spec) {
                                    self.palette.colors[index as usize] = rgb;
                                    log::debug!("OSC 4: Set color {} to {:?}", index, rgb);
                                }
                            }
                        }
                    }
                }
            }
            // OSC 10 - Set/query default foreground color
            10 => {
                if parts.len() >= 2 {
                    if let Ok(color_spec) = std::str::from_utf8(parts[1]) {
                        if let Some(rgb) = ColorPalette::parse_color_spec(color_spec) {
                            self.palette.default_fg = rgb;
                            log::debug!("OSC 10: Set default foreground to {:?}", rgb);
                        }
                    }
                }
            }
            // OSC 11 - Set/query default background color
            11 => {
                if parts.len() >= 2 {
                    if let Ok(color_spec) = std::str::from_utf8(parts[1]) {
                        if let Some(rgb) = ColorPalette::parse_color_spec(color_spec) {
                            self.palette.default_bg = rgb;
                            log::debug!("OSC 11: Set default background to {:?}", rgb);
                        }
                    }
                }
            }
            _ => {
                log::debug!("Unhandled OSC {}", osc_num);
            }
        }
    }

    /// Handle a complete CSI sequence.
    fn csi(&mut self, params: &CsiParams) {
        let action = params.final_char as char;
        let primary = params.primary;
        let secondary = params.secondary;

        match action {
            // Cursor Up
            'A' => {
                let n = params.get(0, 1).max(1) as usize;
                self.cursor_row = self.cursor_row.saturating_sub(n);
            }
            // Cursor Down
            'B' => {
                let n = params.get(0, 1).max(1) as usize;
                self.cursor_row = (self.cursor_row + n).min(self.rows - 1);
            }
            // Cursor Forward
            'C' => {
                let n = params.get(0, 1).max(1) as usize;
                self.cursor_col = (self.cursor_col + n).min(self.cols - 1);
            }
            // Cursor Back
            'D' => {
                let n = params.get(0, 1).max(1) as usize;
                self.cursor_col = self.cursor_col.saturating_sub(n);
            }
            // Cursor Next Line (CNL)
            'E' => {
                let n = params.get(0, 1).max(1) as usize;
                self.cursor_col = 0;
                self.cursor_row = (self.cursor_row + n).min(self.rows - 1);
            }
            // Cursor Previous Line (CPL)
            'F' => {
                let n = params.get(0, 1).max(1) as usize;
                self.cursor_col = 0;
                self.cursor_row = self.cursor_row.saturating_sub(n);
            }
            // Cursor Horizontal Absolute (CHA)
            'G' => {
                let col = params.get(0, 1).max(1) as usize;
                self.cursor_col = (col - 1).min(self.cols - 1);
            }
            // Cursor Position
            'H' | 'f' => {
                let row = params.get(0, 1).max(1) as usize;
                let col = params.get(1, 1).max(1) as usize;
                self.cursor_row = (row - 1).min(self.rows - 1);
                self.cursor_col = (col - 1).min(self.cols - 1);
            }
            // Erase in Display
            'J' => {
                let mode = params.get(0, 0);
                let blank = self.blank_cell();
                match mode {
                    0 => {
                        // Clear from cursor to end of screen
                        self.clear_line_from_cursor();
                        for visual_row in (self.cursor_row + 1)..self.rows {
                            let grid_row = self.line_map[visual_row];
                            self.grid[grid_row].fill(blank);
                            self.mark_line_dirty(visual_row);
                        }
                    }
                    1 => {
                        // Clear from start to cursor
                        for visual_row in 0..self.cursor_row {
                            let grid_row = self.line_map[visual_row];
                            self.grid[grid_row].fill(blank);
                            self.mark_line_dirty(visual_row);
                        }
                        let cursor_grid_row = self.line_map[self.cursor_row];
                        for col in 0..=self.cursor_col {
                            self.grid[cursor_grid_row][col] = blank;
                        }
                        self.mark_line_dirty(self.cursor_row);
                    }
                    2 | 3 => {
                        // Clear entire screen
                        self.clear_screen();
                    }
                    _ => {}
                }
            }
            // Erase in Line
            'K' => {
                let mode = params.get(0, 0);
                let blank = self.blank_cell();
                match mode {
                    0 => self.clear_line_from_cursor(),
                    1 => {
                        let grid_row = self.line_map[self.cursor_row];
                        for col in 0..=self.cursor_col {
                            self.grid[grid_row][col] = blank;
                        }
                        self.mark_line_dirty(self.cursor_row);
                    }
                    2 => {
                        let grid_row = self.line_map[self.cursor_row];
                        self.grid[grid_row].fill(blank);
                        self.mark_line_dirty(self.cursor_row);
                    }
                    _ => {}
                }
            }
            // Insert Lines (IL)
            'L' => {
                let n = params.get(0, 1).max(1) as usize;
                self.insert_lines(n);
            }
            // Delete Lines (DL)
            'M' => {
                let n = params.get(0, 1).max(1) as usize;
                self.delete_lines(n);
            }
            // Delete Characters (DCH)
            'P' => {
                let n = params.get(0, 1).max(1) as usize;
                self.delete_characters(n);
            }
            // Scroll Up (SU)
            'S' => {
                let n = params.get(0, 1).max(1) as usize;
                self.scroll_up(n);
            }
            // Scroll Down (SD)
            'T' => {
                let n = params.get(0, 1).max(1) as usize;
                self.scroll_down(n);
            }
            // Erase Characters (ECH)
            'X' => {
                let n = params.get(0, 1).max(1) as usize;
                self.erase_characters(n);
            }
            // Insert Characters (ICH)
            '@' => {
                let n = params.get(0, 1).max(1) as usize;
                self.insert_characters(n);
            }
            // Repeat preceding character (REP)
            'b' => {
                let n = params.get(0, 1).max(1) as usize;
                if self.cursor_col > 0 {
                    let grid_row = self.line_map[self.cursor_row];
                    let last_char = self.grid[grid_row][self.cursor_col - 1].character;
                    for _ in 0..n {
                        self.print_char(last_char);
                    }
                }
            }
            // Device Attributes (DA)
            'c' => {
                if primary == 0 || primary == b'?' {
                    // Primary DA - respond as VT220
                    self.response_queue.extend_from_slice(b"\x1b[?62;c");
                } else if primary == b'>' {
                    // Secondary DA - respond with terminal version
                    self.response_queue.extend_from_slice(b"\x1b[>0;0;0c");
                }
            }
            // Vertical Position Absolute (VPA)
            'd' => {
                let row = params.get(0, 1).max(1) as usize;
                self.cursor_row = (row - 1).min(self.rows - 1);
            }
            // SGR (Select Graphic Rendition)
            'm' => {
                self.handle_sgr(params);
            }
            // Device Status Report (DSR)
            'n' => {
                let param = params.get(0, 0);
                match param {
                    5 => {
                        // Status report - respond with "OK"
                        self.response_queue.extend_from_slice(b"\x1b[0n");
                    }
                    6 => {
                        // Cursor position report
                        let response = format!("\x1b[{};{}R", self.cursor_row + 1, self.cursor_col + 1);
                        self.response_queue.extend_from_slice(response.as_bytes());
                    }
                    _ => {}
                }
            }
            // DECSCUSR - Set Cursor Style (CSI Ps SP q)
            // Also handle CSI q (no space) as reset to default
            'q' => {
                if secondary == b' ' || secondary == 0 {
                    let style = params.get(0, 0);
                    self.cursor_shape = match style {
                        0 | 1 => CursorShape::BlinkingBlock,
                        2 => CursorShape::SteadyBlock,
                        3 => CursorShape::BlinkingUnderline,
                        4 => CursorShape::SteadyUnderline,
                        5 => CursorShape::BlinkingBar,
                        6 => CursorShape::SteadyBar,
                        _ => CursorShape::BlinkingBlock,
                    };
                }
            }
            // Set Scrolling Region (DECSTBM)
            'r' => {
                let top = params.get(0, 1).max(1) as usize;
                let bottom = params.get(1, self.rows as i32).max(1) as usize;
                self.scroll_top = (top - 1).min(self.rows - 1);
                self.scroll_bottom = (bottom - 1).min(self.rows - 1);
                if self.scroll_top > self.scroll_bottom {
                    std::mem::swap(&mut self.scroll_top, &mut self.scroll_bottom);
                }
                // Move cursor to home position
                self.cursor_row = 0;
                self.cursor_col = 0;
            }
            // Window manipulation (CSI Ps t)
            't' => {
                let ps = params.get(0, 0);
                match ps {
                    22 | 23 => {
                        // Save/restore window title - ignore
                    }
                    _ => {
                        log::trace!("Window manipulation: ps={}", ps);
                    }
                }
            }
            // Kitty keyboard protocol
            'u' => {
                self.handle_keyboard_protocol_csi(params);
            }
            // DEC Private Mode Set (CSI ? Ps h)
            'h' if primary == b'?' => {
                self.handle_dec_private_mode_set(params);
            }
            // DEC Private Mode Reset (CSI ? Ps l)
            'l' if primary == b'?' => {
                self.handle_dec_private_mode_reset(params);
            }
            _ => {
                log::debug!(
                    "Unhandled CSI: action='{}' primary={} secondary={} params={:?}",
                    action, primary, secondary, &params.params[..params.num_params]
                );
            }
        }
    }

    fn save_cursor(&mut self) {
        self.saved_cursor = SavedCursor {
            col: self.cursor_col,
            row: self.cursor_row,
            fg: self.current_fg,
            bg: self.current_bg,
            bold: self.current_bold,
            italic: self.current_italic,
            underline: self.current_underline,
        };
        log::debug!("ESC 7: Cursor saved at ({}, {})", self.cursor_col, self.cursor_row);
    }

    fn restore_cursor(&mut self) {
        self.cursor_col = self.saved_cursor.col.min(self.cols.saturating_sub(1));
        self.cursor_row = self.saved_cursor.row.min(self.rows.saturating_sub(1));
        self.current_fg = self.saved_cursor.fg;
        self.current_bg = self.saved_cursor.bg;
        self.current_bold = self.saved_cursor.bold;
        self.current_italic = self.saved_cursor.italic;
        self.current_underline = self.saved_cursor.underline;
        log::debug!("ESC 8: Cursor restored to ({}, {})", self.cursor_col, self.cursor_row);
    }

    fn reset(&mut self) {
        self.current_fg = Color::Default;
        self.current_bg = Color::Default;
        self.current_bold = false;
        self.current_italic = false;
        self.current_underline = false;
        self.cursor_col = 0;
        self.cursor_row = 0;
        self.cursor_visible = true;
        self.cursor_shape = CursorShape::default();
        self.scroll_top = 0;
        self.scroll_bottom = self.rows.saturating_sub(1);
        self.mouse_tracking = MouseTrackingMode::None;
        self.mouse_encoding = MouseEncoding::X10;
        self.application_cursor_keys = false;
        self.auto_wrap = true;
        self.bracketed_paste = false;
        self.focus_reporting = false;
        self.synchronized_output = false;
        if self.using_alternate_screen {
            self.leave_alternate_screen();
        }
        for row in &mut self.grid {
            for cell in row {
                *cell = Cell::default();
            }
        }
        self.mark_all_lines_dirty();
        log::debug!("ESC c: Full terminal reset");
    }

    fn index(&mut self) {
        if self.cursor_row >= self.scroll_bottom {
            self.scroll_up(1);
        } else {
            self.cursor_row += 1;
        }
    }

    fn newline(&mut self) {
        self.cursor_col = 0;
        if self.cursor_row >= self.scroll_bottom {
            self.scroll_up(1);
        } else {
            self.cursor_row += 1;
        }
    }

    fn reverse_index(&mut self) {
        if self.cursor_row <= self.scroll_top {
            self.scroll_down(1);
        } else {
            self.cursor_row -= 1;
        }
    }

    fn set_tab_stop(&mut self) {
        // HTS - default tab stops every 8 columns
    }

    fn set_keypad_mode(&mut self, application: bool) {
        if application {
            log::debug!("ESC =: Application keypad mode");
        } else {
            log::debug!("ESC >: Normal keypad mode");
        }
    }

    fn designate_charset(&mut self, _set: u8, _charset: u8) {
        // UTF-8 internally, no-op
    }

    fn screen_alignment(&mut self) {
        for visual_row in 0..self.rows {
            let grid_row = self.line_map[visual_row];
            for cell in &mut self.grid[grid_row] {
                *cell = Cell {
                    character: 'E',
                    fg_color: Color::Default,
                    bg_color: Color::Default,
                    bold: false,
                    italic: false,
                    underline: false,
                };
            }
            self.mark_line_dirty(visual_row);
        }
    }
}

impl Terminal {
    /// Print a single character at the cursor position.
    #[inline]
    fn print_char(&mut self, c: char) {
        if self.cursor_col >= self.cols {
            if self.auto_wrap {
                self.cursor_col = 0;
                self.cursor_row += 1;
                if self.cursor_row > self.scroll_bottom {
                    self.scroll_up(1);
                    self.cursor_row = self.scroll_bottom;
                }
            } else {
                self.cursor_col = self.cols - 1;
            }
        }

        let grid_row = self.line_map[self.cursor_row];
        self.grid[grid_row][self.cursor_col] = Cell {
            character: c,
            fg_color: self.current_fg,
            bg_color: self.current_bg,
            bold: self.current_bold,
            italic: self.current_italic,
            underline: self.current_underline,
        };
        self.mark_line_dirty(self.cursor_row);
        self.cursor_col += 1;
    }

    /// Handle SGR (Select Graphic Rendition) parameters.
    fn handle_sgr(&mut self, params: &CsiParams) {
        if params.num_params == 0 {
            self.current_fg = Color::Default;
            self.current_bg = Color::Default;
            self.current_bold = false;
            self.current_italic = false;
            self.current_underline = false;
            return;
        }

        let mut i = 0;
        while i < params.num_params {
            let code = params.params[i];

            match code {
                0 => {
                    self.current_fg = Color::Default;
                    self.current_bg = Color::Default;
                    self.current_bold = false;
                    self.current_italic = false;
                    self.current_underline = false;
                }
                1 => self.current_bold = true,
                3 => self.current_italic = true,
                4 => self.current_underline = true,
                7 => std::mem::swap(&mut self.current_fg, &mut self.current_bg),
                22 => self.current_bold = false,
                23 => self.current_italic = false,
                24 => self.current_underline = false,
                27 => std::mem::swap(&mut self.current_fg, &mut self.current_bg),
                30..=37 => self.current_fg = Color::Indexed((code - 30) as u8),
                38 => {
                    // Extended foreground color
                    if i + 1 < params.num_params && params.is_sub_param[i + 1] {
                        let mode = params.params[i + 1];
                        if mode == 5 && i + 2 < params.num_params {
                            self.current_fg = Color::Indexed(params.params[i + 2] as u8);
                            i += 2;
                        } else if mode == 2 && i + 4 < params.num_params {
                            self.current_fg = Color::Rgb(
                                params.params[i + 2] as u8,
                                params.params[i + 3] as u8,
                                params.params[i + 4] as u8,
                            );
                            i += 4;
                        }
                    } else if i + 2 < params.num_params {
                        let mode = params.params[i + 1];
                        if mode == 5 {
                            self.current_fg = Color::Indexed(params.params[i + 2] as u8);
                            i += 2;
                        } else if mode == 2 && i + 4 < params.num_params {
                            self.current_fg = Color::Rgb(
                                params.params[i + 2] as u8,
                                params.params[i + 3] as u8,
                                params.params[i + 4] as u8,
                            );
                            i += 4;
                        }
                    }
                }
                39 => self.current_fg = Color::Default,
                40..=47 => self.current_bg = Color::Indexed((code - 40) as u8),
                48 => {
                    // Extended background color
                    if i + 1 < params.num_params && params.is_sub_param[i + 1] {
                        let mode = params.params[i + 1];
                        if mode == 5 && i + 2 < params.num_params {
                            self.current_bg = Color::Indexed(params.params[i + 2] as u8);
                            i += 2;
                        } else if mode == 2 && i + 4 < params.num_params {
                            self.current_bg = Color::Rgb(
                                params.params[i + 2] as u8,
                                params.params[i + 3] as u8,
                                params.params[i + 4] as u8,
                            );
                            i += 4;
                        }
                    } else if i + 2 < params.num_params {
                        let mode = params.params[i + 1];
                        if mode == 5 {
                            self.current_bg = Color::Indexed(params.params[i + 2] as u8);
                            i += 2;
                        } else if mode == 2 && i + 4 < params.num_params {
                            self.current_bg = Color::Rgb(
                                params.params[i + 2] as u8,
                                params.params[i + 3] as u8,
                                params.params[i + 4] as u8,
                            );
                            i += 4;
                        }
                    }
                }
                49 => self.current_bg = Color::Default,
                90..=97 => self.current_fg = Color::Indexed((code - 90 + 8) as u8),
                100..=107 => self.current_bg = Color::Indexed((code - 100 + 8) as u8),
                _ => {}
            }
            i += 1;
        }
    }

    /// Handle Kitty keyboard protocol CSI sequences.
    fn handle_keyboard_protocol_csi(&mut self, params: &CsiParams) {
        match params.primary {
            b'?' => {
                let response = query_response(self.keyboard.flags());
                self.response_queue.extend(response);
            }
            b'=' => {
                let flags = params.get(0, 0) as u8;
                let mode = params.get(1, 1) as u8;
                self.keyboard.set_flags(flags, mode);
                log::debug!("Keyboard flags set to {:?} (mode {})", self.keyboard.flags(), mode);
            }
            b'>' => {
                let flags = if params.num_params == 0 {
                    None
                } else {
                    Some(params.params[0] as u8)
                };
                self.keyboard.push(flags);
                log::debug!("Keyboard flags pushed: {:?}", self.keyboard.flags());
            }
            b'<' => {
                let count = params.get(0, 1) as usize;
                self.keyboard.pop(count);
                log::debug!("Keyboard flags popped: {:?}", self.keyboard.flags());
            }
            _ => {}
        }
    }

    /// Handle DEC private mode set (CSI ? Ps h).
    fn handle_dec_private_mode_set(&mut self, params: &CsiParams) {
        for i in 0..params.num_params {
            match params.params[i] {
                1 => {
                    self.application_cursor_keys = true;
                    log::debug!("DECCKM: Application cursor keys enabled");
                }
                7 => {
                    self.auto_wrap = true;
                    log::debug!("DECAWM: Auto-wrap enabled");
                }
                9 => {
                    self.mouse_tracking = MouseTrackingMode::X10;
                    log::debug!("Mouse tracking: X10 mode enabled");
                }
                25 => {
                    self.cursor_visible = true;
                    log::debug!("DECTCEM: cursor visible");
                }
                47 => self.enter_alternate_screen(),
                1000 => {
                    self.mouse_tracking = MouseTrackingMode::Normal;
                    log::debug!("Mouse tracking: Normal mode enabled");
                }
                1002 => {
                    self.mouse_tracking = MouseTrackingMode::ButtonEvent;
                    log::debug!("Mouse tracking: Button-event mode enabled");
                }
                1003 => {
                    self.mouse_tracking = MouseTrackingMode::AnyEvent;
                    log::debug!("Mouse tracking: Any-event mode enabled");
                }
                1004 => {
                    self.focus_reporting = true;
                    log::debug!("Focus event reporting enabled");
                }
                1005 => {
                    self.mouse_encoding = MouseEncoding::Utf8;
                    log::debug!("Mouse encoding: UTF-8");
                }
                1006 => {
                    self.mouse_encoding = MouseEncoding::Sgr;
                    log::debug!("Mouse encoding: SGR");
                }
                1015 => {
                    self.mouse_encoding = MouseEncoding::Urxvt;
                    log::debug!("Mouse encoding: URXVT");
                }
                1047 => self.enter_alternate_screen(),
                1048 => Handler::save_cursor(self),
                1049 => {
                    Handler::save_cursor(self);
                    self.enter_alternate_screen();
                }
                2004 => {
                    self.bracketed_paste = true;
                    log::debug!("Bracketed paste mode enabled");
                }
                2026 => {
                    self.synchronized_output = true;
                    log::trace!("Synchronized output enabled");
                }
                _ => log::debug!("Unhandled DEC private mode set: {}", params.params[i]),
            }
        }
    }

    /// Handle DEC private mode reset (CSI ? Ps l).
    fn handle_dec_private_mode_reset(&mut self, params: &CsiParams) {
        for i in 0..params.num_params {
            match params.params[i] {
                1 => {
                    self.application_cursor_keys = false;
                    log::debug!("DECCKM: Normal cursor keys enabled");
                }
                7 => {
                    self.auto_wrap = false;
                    log::debug!("DECAWM: Auto-wrap disabled");
                }
                9 => {
                    if self.mouse_tracking == MouseTrackingMode::X10 {
                        self.mouse_tracking = MouseTrackingMode::None;
                        log::debug!("Mouse tracking: X10 mode disabled");
                    }
                }
                25 => {
                    self.cursor_visible = false;
                    log::debug!("DECTCEM: cursor hidden");
                }
                47 => self.leave_alternate_screen(),
                1000 => {
                    if self.mouse_tracking == MouseTrackingMode::Normal {
                        self.mouse_tracking = MouseTrackingMode::None;
                        log::debug!("Mouse tracking: Normal mode disabled");
                    }
                }
                1002 => {
                    if self.mouse_tracking == MouseTrackingMode::ButtonEvent {
                        self.mouse_tracking = MouseTrackingMode::None;
                        log::debug!("Mouse tracking: Button-event mode disabled");
                    }
                }
                1003 => {
                    if self.mouse_tracking == MouseTrackingMode::AnyEvent {
                        self.mouse_tracking = MouseTrackingMode::None;
                        log::debug!("Mouse tracking: Any-event mode disabled");
                    }
                }
                1004 => {
                    self.focus_reporting = false;
                    log::debug!("Focus event reporting disabled");
                }
                1005 => {
                    if self.mouse_encoding == MouseEncoding::Utf8 {
                        self.mouse_encoding = MouseEncoding::X10;
                        log::debug!("Mouse encoding: reset to X10");
                    }
                }
                1006 => {
                    if self.mouse_encoding == MouseEncoding::Sgr {
                        self.mouse_encoding = MouseEncoding::X10;
                        log::debug!("Mouse encoding: reset to X10");
                    }
                }
                1015 => {
                    if self.mouse_encoding == MouseEncoding::Urxvt {
                        self.mouse_encoding = MouseEncoding::X10;
                        log::debug!("Mouse encoding: reset to X10");
                    }
                }
                1047 => self.leave_alternate_screen(),
                1048 => Handler::restore_cursor(self),
                1049 => {
                    self.leave_alternate_screen();
                    Handler::restore_cursor(self);
                }
                2004 => {
                    self.bracketed_paste = false;
                    log::debug!("Bracketed paste mode disabled");
                }
                2026 => {
                    self.synchronized_output = false;
                    log::trace!("Synchronized output disabled");
                }
                _ => log::debug!("Unhandled DEC private mode reset: {}", params.params[i]),
            }
        }
    }
}
