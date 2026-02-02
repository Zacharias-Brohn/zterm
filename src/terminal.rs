//! Terminal state management and escape sequence handling.

use crate::graphics::{GraphicsCommand, ImageStorage};
use crate::keyboard::{query_response, KeyboardState};
use crate::vt_parser::{CsiParams, Handler};
use unicode_width::UnicodeWidthChar;

/// Commands that the terminal can send to the application.
/// These are triggered by special escape sequences from programs like Neovim.
#[derive(Clone, Debug, PartialEq)]
pub enum TerminalCommand {
    /// Navigate to a neighboring pane in the given direction.
    /// Triggered by OSC 51;navigate;<direction> ST
    NavigatePane(Direction),
    /// Set custom statusline content for this pane.
    /// Triggered by OSC 51;statusline;<content> ST
    /// Empty content clears the statusline (restores default).
    SetStatusline(Option<String>),
    /// Set clipboard content via OSC 52.
    /// Triggered by OSC 52;c;<base64-data> ST
    SetClipboard(String),
}

/// Direction for pane navigation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Direction {
    Up,
    Down,
    Left,
    Right,
}

/// A single cell in the terminal grid.
#[derive(Clone, Copy, Debug)]
pub struct Cell {
    pub character: char,
    pub fg_color: Color,
    pub bg_color: Color,
    pub bold: bool,
    pub italic: bool,
    /// Underline style: 0=none, 1=single, 2=double, 3=curly, 4=dotted, 5=dashed
    pub underline_style: u8,
    /// Strikethrough decoration
    pub strikethrough: bool,
    /// If true, this cell is the continuation of a wide (double-width) character.
    /// The actual character is stored in the previous cell.
    pub wide_continuation: bool,
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            character: ' ',
            fg_color: Color::Default,
            bg_color: Color::Default,
            bold: false,
            italic: false,
            underline_style: 0,
            strikethrough: false,
            wide_continuation: false,
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
        colors[0] = [0, 0, 0]; // Black
        colors[1] = [204, 0, 0]; // Red
        colors[2] = [0, 204, 0]; // Green
        colors[3] = [204, 204, 0]; // Yellow
        colors[4] = [0, 0, 204]; // Blue
        colors[5] = [204, 0, 204]; // Magenta
        colors[6] = [0, 204, 204]; // Cyan
        colors[7] = [204, 204, 204]; // White

        // Bright ANSI colors (8-15)
        colors[8] = [102, 102, 102]; // Bright Black (Gray)
        colors[9] = [255, 0, 0]; // Bright Red
        colors[10] = [0, 255, 0]; // Bright Green
        colors[11] = [255, 255, 0]; // Bright Yellow
        colors[12] = [0, 0, 255]; // Bright Blue
        colors[13] = [255, 0, 255]; // Bright Magenta
        colors[14] = [0, 255, 255]; // Bright Cyan
        colors[15] = [255, 255, 255]; // Bright White

        // 216 color cube (16-231)
        for r in 0..6 {
            for g in 0..6 {
                for b in 0..6 {
                    let idx = 16 + r * 36 + g * 6 + b;
                    let to_val =
                        |c: usize| if c == 0 { 0 } else { (55 + c * 40) as u8 };
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
                    Some(if s.len() > 2 {
                        (val >> 8) as u8
                    } else {
                        val as u8
                    })
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
            Color::Rgb(r, g, b) => {
                [*r as f32 / 255.0, *g as f32 / 255.0, *b as f32 / 255.0, 1.0]
            }
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
            Color::Rgb(r, g, b) => {
                [*r as f32 / 255.0, *g as f32 / 255.0, *b as f32 / 255.0, 1.0]
            }
            Color::Indexed(idx) => {
                let [r, g, b] = self.colors[*idx as usize];
                [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0]
            }
        }
    }
}

/// Saved cursor state for DECSC/DECRC.
/// Per ECMA-48 and DEC VT standards, DECSC saves:
/// - Cursor position (col, row)
/// - Character attributes (fg, bg, bold, italic, underline, strikethrough)
/// - Origin mode (DECOM) - affects cursor positioning relative to scroll region
/// - Auto-wrap mode (DECAWM) - affects line wrapping behavior
#[derive(Clone, Debug, Default)]
struct SavedCursor {
    col: usize,
    row: usize,
    fg: Color,
    bg: Color,
    bold: bool,
    italic: bool,
    underline_style: u8,
    strikethrough: bool,
    origin_mode: bool,
    auto_wrap: bool,
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
/// Only populated when the `render_timing` feature is enabled.
#[derive(Debug, Default)]
pub struct ProcessingStats {
    #[cfg(feature = "render_timing")]
    /// Total time spent in scroll_up operations (nanoseconds).
    pub scroll_up_ns: u64,
    #[cfg(feature = "render_timing")]
    /// Number of scroll_up calls.
    pub scroll_up_count: u32,
    #[cfg(feature = "render_timing")]
    /// Total time spent in scrollback operations (nanoseconds).
    pub scrollback_ns: u64,
    #[cfg(feature = "render_timing")]
    /// Time in VecDeque pop_front.
    pub pop_front_ns: u64,
    #[cfg(feature = "render_timing")]
    /// Time in VecDeque push_back.
    pub push_back_ns: u64,
    #[cfg(feature = "render_timing")]
    /// Time in mem::swap.
    pub swap_ns: u64,
    #[cfg(feature = "render_timing")]
    /// Total time spent in line clearing (nanoseconds).
    pub clear_line_ns: u64,
    #[cfg(feature = "render_timing")]
    /// Total time spent in text handler (nanoseconds).
    pub text_handler_ns: u64,
    #[cfg(feature = "render_timing")]
    /// Total time spent in CSI handler (nanoseconds).
    pub csi_handler_ns: u64,
    #[cfg(feature = "render_timing")]
    /// Number of CSI sequences processed.
    pub csi_count: u32,
    #[cfg(feature = "render_timing")]
    /// Number of characters processed.
    pub chars_processed: u32,
    #[cfg(feature = "render_timing")]
    /// Total time spent in VT parser (consume_input) - nanoseconds.
    pub vt_parser_ns: u64,
    #[cfg(feature = "render_timing")]
    /// Number of consume_input calls.
    pub consume_input_count: u32,
}

impl ProcessingStats {
    #[cfg(feature = "render_timing")]
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    #[cfg(not(feature = "render_timing"))]
    pub fn reset(&mut self) {}

    #[cfg(feature = "render_timing")]
    pub fn log_if_slow(&self, threshold_ms: u64) {
        let total_ms =
            (self.scroll_up_ns + self.text_handler_ns + self.csi_handler_ns)
                / 1_000_000;
        if total_ms >= threshold_ms {
            let vt_only_ns = self
                .vt_parser_ns
                .saturating_sub(self.text_handler_ns + self.csi_handler_ns);
            log::info!(
                "[PARSE_DETAIL] text={:.2}ms ({}chars) csi={:.2}ms ({}x) vt_only={:.2}ms ({}calls) scroll={:.2}ms ({}x)",
                self.text_handler_ns as f64 / 1_000_000.0,
                self.chars_processed,
                self.csi_handler_ns as f64 / 1_000_000.0,
                self.csi_count,
                vt_only_ns as f64 / 1_000_000.0,
                self.consume_input_count,
                self.scroll_up_ns as f64 / 1_000_000.0,
                self.scroll_up_count,
            );
        }
    }

    #[cfg(not(feature = "render_timing"))]
    pub fn log_if_slow(&self, _threshold_ms: u64) {}
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
    /// Creates a new scrollback buffer with the given capacity.
    /// Lines are allocated lazily as needed to avoid slow startup.
    pub fn new(capacity: usize) -> Self {
        // Don't pre-allocate lines - allocate them lazily as content is added
        // This avoids allocating and zeroing potentially 20MB+ of memory at startup
        let lines = Vec::with_capacity(capacity.min(1024)); // Start with reasonable capacity

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
    /// Lines are allocated lazily on first use to avoid slow startup.
    #[inline]
    pub fn push(&mut self, cols: usize) -> &mut Vec<Cell> {
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
            // Buffer not full yet - allocate new line if needed
            if idx >= self.lines.len() {
                // Grow the lines vector and allocate the new line
                self.lines.push(vec![Cell::default(); cols]);
            }
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
    /// Current underline style (0=none, 1=single, 2=double, 3=curly, 4=dotted, 5=dashed).
    pub current_underline_style: u8,
    /// Current strikethrough state.
    pub current_strikethrough: bool,
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
    /// Origin mode (DECOM) - cursor positioning relative to scroll region.
    origin_mode: bool,
    /// Bracketed paste mode - wrap pasted text with escape sequences.
    pub bracketed_paste: bool,
    /// Focus event reporting mode.
    pub focus_reporting: bool,
    /// Synchronized output mode (for reducing flicker).
    synchronized_output: bool,
    /// Performance timing stats (for debugging).
    pub stats: ProcessingStats,
    /// Command queue for terminal-to-application communication.
    /// Commands are added by OSC handlers and consumed by the application.
    command_queue: Vec<TerminalCommand>,
    /// Image storage for Kitty graphics protocol.
    pub image_storage: ImageStorage,
    /// Cell width in pixels (for image sizing).
    pub cell_width: f32,
    /// Cell height in pixels (for image sizing).
    pub cell_height: f32,
}

impl Terminal {
    /// Default scrollback limit (10,000 lines for better cache performance).
    pub const DEFAULT_SCROLLBACK_LIMIT: usize = 10_000;

    /// Creates a new terminal with the given dimensions and scrollback limit.
    pub fn new(cols: usize, rows: usize, scrollback_limit: usize) -> Self {
        log::info!(
            "Terminal::new: cols={}, rows={}, scroll_bottom={}",
            cols,
            rows,
            rows.saturating_sub(1)
        );
        let grid = vec![vec![Cell::default(); cols]; rows];
        let line_map: Vec<usize> = (0..rows).collect();

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
            current_underline_style: 0,
            current_strikethrough: false,
            dirty: true,
            dirty_lines: [!0u64; 4], // All lines dirty initially
            scroll_top: 0,
            scroll_bottom: rows.saturating_sub(1),
            keyboard: KeyboardState::new(),
            response_queue: Vec::new(),
            palette: ColorPalette::default(),
            scrollback: ScrollbackBuffer::new(scrollback_limit),
            scroll_offset: 0,
            mouse_tracking: MouseTrackingMode::default(),
            mouse_encoding: MouseEncoding::default(),
            saved_cursor: SavedCursor::default(),
            alternate_screen: None,
            using_alternate_screen: false,
            application_cursor_keys: false,
            auto_wrap: true,
            origin_mode: false,
            bracketed_paste: false,
            focus_reporting: false,
            synchronized_output: false,
            stats: ProcessingStats::default(),
            command_queue: Vec::new(),
            image_storage: ImageStorage::new(),
            cell_width: 10.0, // Default, will be set by renderer
            cell_height: 20.0, // Default, will be set by renderer
        }
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

    /// Take all pending commands from the queue.
    /// Returns an empty Vec if no commands are pending.
    #[inline]
    pub fn take_commands(&mut self) -> Vec<TerminalCommand> {
        std::mem::take(&mut self.command_queue)
    }

    /// Get the dirty lines bitmap (for passing to shm).
    #[inline]
    pub fn get_dirty_lines(&self) -> u64 {
        // Return first 64 lines worth of dirty bits (most common case)
        self.dirty_lines[0]
    }

    /// Check if synchronized output mode is active (rendering should be suppressed).
    /// This is set by CSI 2026 or DCS pending mode (=1s/=2s).
    #[inline]
    pub fn is_synchronized(&self) -> bool {
        self.synchronized_output
    }

    /// Advance cursor to next row, scrolling if necessary.
    /// This is the common pattern: increment row, scroll if past scroll_bottom.
    #[inline]
    fn advance_row(&mut self) {
        self.cursor_row += 1;
        if self.cursor_row > self.scroll_bottom {
            self.scroll_up(1);
            self.cursor_row = self.scroll_bottom;
        }
    }

    /// Create a cell with current text attributes.
    #[inline]
    fn make_cell(&self, character: char, wide_continuation: bool) -> Cell {
        Cell {
            character,
            fg_color: self.current_fg,
            bg_color: self.current_bg,
            bold: self.current_bold,
            italic: self.current_italic,
            underline_style: self.current_underline_style,
            strikethrough: self.current_strikethrough,
            wide_continuation,
        }
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
        let row = &mut self.grid[grid_row];
        // Ensure row has correct width (may differ after swap with scrollback post-resize)
        row.resize(self.cols, blank);
        row.fill(blank);
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
            underline_style: 0,
            strikethrough: false,
            wide_continuation: false,
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

    /// Mark terminal as dirty (needs redraw). Called after parsing.
    pub fn mark_dirty(&mut self) {
        self.dirty = true;
    }

    /// Resizes the terminal grid.
    pub fn resize(&mut self, cols: usize, rows: usize) {
        if cols == self.cols && rows == self.rows {
            return;
        }

        log::info!(
            "Terminal::resize: {}x{} -> {}x{}",
            self.cols,
            self.rows,
            cols,
            rows
        );

        let old_cols = self.cols;
        let old_rows = self.rows;

        // Create new grid
        let mut new_grid = vec![vec![Cell::default(); cols]; rows];

        // Copy existing content using line_map for correct visual ordering
        for visual_row in 0..rows.min(self.rows) {
            let old_grid_row = self.line_map[visual_row];
            // Use actual row length - may differ from self.cols after scrollback swap
            let old_row_len = self.grid[old_grid_row].len();
            for col in 0..cols.min(old_row_len) {
                new_grid[visual_row][col] =
                    self.grid[old_grid_row][col].clone();
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

        // Also resize the saved alternate screen if it exists
        if let Some(ref mut saved) = self.alternate_screen {
            let mut new_saved_grid = vec![vec![Cell::default(); cols]; rows];
            for visual_row in 0..rows.min(old_rows) {
                let old_grid_row = saved
                    .line_map
                    .get(visual_row)
                    .copied()
                    .unwrap_or(visual_row);
                if old_grid_row < saved.grid.len() {
                    for col in 0..cols.min(old_cols) {
                        if col < saved.grid[old_grid_row].len() {
                            new_saved_grid[visual_row][col] =
                                saved.grid[old_grid_row][col].clone();
                        }
                    }
                }
            }
            saved.grid = new_saved_grid;
            saved.line_map = (0..rows).collect();
            saved.cursor_col = saved.cursor_col.min(cols.saturating_sub(1));
            saved.cursor_row = saved.cursor_row.min(rows.saturating_sub(1));
            saved.scroll_top = 0;
            saved.scroll_bottom = rows.saturating_sub(1);
        }

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
            self.saved_cursor = saved.saved_cursor;
            self.scroll_top = saved.scroll_top;
            self.scroll_bottom = saved.scroll_bottom;
            // Clamp cursor positions to current grid dimensions (defensive)
            self.cursor_col = saved.cursor_col.min(self.cols.saturating_sub(1));
            self.cursor_row = saved.cursor_row.min(self.rows.saturating_sub(1));
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

        #[cfg(feature = "render_timing")]
        {
            self.stats.scroll_up_count += n as u32;
        }

        for _ in 0..n {
            // Save the top line's grid index before rotation
            let recycled_grid_row = self.line_map[self.scroll_top];

            // Save to scrollback only if scrolling from the very top of the screen
            // AND not in alternate screen mode (alternate screen never uses scrollback)
            // AND scrollback is enabled (capacity > 0)
            if self.scroll_top == 0
                && !self.using_alternate_screen
                && self.scrollback.capacity > 0
            {
                // Get a slot in the ring buffer - this is O(1) with just modulo arithmetic
                // If buffer is full, this overwrites the oldest line (perfect for our swap)
                let cols = self.cols;
                let dest = self.scrollback.push(cols);
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
            self.line_map.copy_within(
                self.scroll_top + 1..=self.scroll_bottom,
                self.scroll_top,
            );
            self.line_map[self.scroll_bottom] = recycled_grid_row;
        }

        // Mark all lines dirty with a single bitmask operation instead of loop
        self.mark_region_dirty(self.scroll_top, self.scroll_bottom);
    }

    /// Mark a range of lines as dirty efficiently using bitmask operations.
    #[inline]
    fn mark_region_dirty(&mut self, start: usize, end: usize) {
        let end = end.min(255);
        if start > end {
            return;
        }

        // Process each 64-bit word that overlaps with [start, end]
        let start_word = start / 64;
        let end_word = end / 64;

        for word_idx in start_word..=end_word.min(3) {
            let word_start = word_idx * 64;
            let word_end = word_start + 63;

            // Calculate bit range within this word
            let bit_start = if start > word_start {
                start - word_start
            } else {
                0
            };
            let bit_end = if end < word_end { end - word_start } else { 63 };

            // Create mask for bits [bit_start, bit_end]
            // mask = ((1 << (bit_end - bit_start + 1)) - 1) << bit_start
            let num_bits = bit_end - bit_start + 1;
            let mask = if num_bits >= 64 {
                !0u64
            } else {
                ((1u64 << num_bits) - 1) << bit_start
            };

            self.dirty_lines[word_idx] |= mask;
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
            self.line_map.copy_within(
                self.scroll_top..self.scroll_bottom,
                self.scroll_top + 1,
            );
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
                format!("\x1b[<{};{};{}{}", cb, col, row, suffix as char)
                    .into_bytes()
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
                    let scrollback_idx =
                        scrollback_len - self.scroll_offset + i;
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

    /// Get a single visible row by index without allocation.
    /// Returns None if row_idx is out of bounds.
    #[inline]
    pub fn get_visible_row(&self, row_idx: usize) -> Option<&Vec<Cell>> {
        if row_idx >= self.rows {
            return None;
        }

        if self.scroll_offset == 0 {
            // No scrollback viewing, just return from grid via line_map
            Some(&self.grid[self.line_map[row_idx]])
        } else {
            // We're viewing scrollback
            let scrollback_len = self.scrollback.len();
            let lines_from_scrollback = self.scroll_offset.min(self.rows);

            if row_idx < lines_from_scrollback {
                // This row comes from scrollback
                let scrollback_idx =
                    scrollback_len - self.scroll_offset + row_idx;
                self.scrollback
                    .get(scrollback_idx)
                    .or_else(|| Some(&self.grid[self.line_map[row_idx]]))
            } else {
                // This row comes from the grid
                let grid_visual_idx = row_idx - lines_from_scrollback;
                if grid_visual_idx < self.rows {
                    Some(&self.grid[self.line_map[grid_visual_idx]])
                } else {
                    None
                }
            }
        }
    }

    /// Inserts n blank lines at the cursor position, scrolling lines below down.
    /// Uses line_map rotation for efficiency.
    fn insert_lines(&mut self, n: usize) {
        if self.cursor_row < self.scroll_top
            || self.cursor_row > self.scroll_bottom
        {
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
        }

        // Mark affected lines dirty once after all rotations
        for line in self.cursor_row..=self.scroll_bottom {
            self.mark_line_dirty(line);
        }
    }

    /// Deletes n lines at the cursor position, scrolling lines below up.
    /// Uses line_map rotation for efficiency.
    fn delete_lines(&mut self, n: usize) {
        if self.cursor_row < self.scroll_top
            || self.cursor_row > self.scroll_bottom
        {
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
        }

        // Mark affected lines dirty once after all rotations
        for line in self.cursor_row..=self.scroll_bottom {
            self.mark_line_dirty(line);
        }
    }

    /// Inserts n blank characters at the cursor, shifting existing chars right.
    fn insert_characters(&mut self, n: usize) {
        let grid_row = self.line_map[self.cursor_row];
        let blank = self.blank_cell();
        let row = &mut self.grid[grid_row];
        let n = n.min(self.cols - self.cursor_col);
        // Truncate n characters from the end
        row.truncate(self.cols - n);
        // Insert n blank characters at cursor position (single O(cols) operation)
        row.splice(
            self.cursor_col..self.cursor_col,
            std::iter::repeat(blank).take(n),
        );
        self.mark_line_dirty(self.cursor_row);
    }

    /// Deletes n characters at the cursor, shifting remaining chars left.
    fn delete_characters(&mut self, n: usize) {
        let grid_row = self.line_map[self.cursor_row];
        let blank = self.blank_cell();
        let row = &mut self.grid[grid_row];
        let n = n.min(self.cols - self.cursor_col);
        let end = (self.cursor_col + n).min(row.len());
        // Remove n characters at cursor position (single O(cols) operation)
        row.drain(self.cursor_col..end);
        // Pad with blank characters at the end
        row.resize(self.cols, blank);
        self.mark_line_dirty(self.cursor_row);
    }

    /// Erases n characters at the cursor (replaces with spaces, doesn't shift).
    fn erase_characters(&mut self, n: usize) {
        let grid_row = self.line_map[self.cursor_row];
        let n = n.min(self.cols - self.cursor_col);
        let blank = self.blank_cell();
        // Fill range with blanks (bounds already guaranteed by min above)
        self.grid[grid_row][self.cursor_col..self.cursor_col + n].fill(blank);
        self.mark_line_dirty(self.cursor_row);
    }

    /// Clears the current line from cursor to end.
    #[inline]
    fn clear_line_from_cursor(&mut self) {
        let grid_row = self.line_map[self.cursor_row];
        let blank = self.blank_cell();
        // Use slice fill for efficiency
        self.grid[grid_row][self.cursor_col..].fill(blank);
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
                let cols = self.cols;
                let dest = self.scrollback.push(cols);
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
    /// Handle a chunk of decoded text (Unicode codepoints as u32).
    /// This includes control characters (0x00-0x1F except ESC).
    fn text(&mut self, codepoints: &[u32]) {
        // DEBUG: Detect CSI sequence content appearing as text (indicates parser bug)
        // Look for patterns like "38;2;128" or "4;64;64m" - these are SGR parameters
        if codepoints.len() >= 3 {
            let has_semicolon = codepoints.iter().any(|&c| c == 0x3B); // ';'
            let has_m = codepoints.iter().any(|&c| c == 0x6D); // 'm'
            let mostly_digits = codepoints
                .iter()
                .filter(|&&c| c >= 0x30 && c <= 0x39)
                .count()
                > codepoints.len() / 2;
            if has_semicolon && mostly_digits {
                let text: String = codepoints
                    .iter()
                    .filter_map(|&c| char::from_u32(c))
                    .collect();
                log::error!("DEBUG CSI LEAK: text handler received CSI-like content: {:?} at ({}, {})", 
                    text, self.cursor_col, self.cursor_row);
            }
        }

        #[cfg(feature = "render_timing")]
        let start = std::time::Instant::now();

        // Cache the current line to avoid repeated line_map lookups
        let mut cached_row = self.cursor_row;
        let mut grid_row = self.line_map[cached_row];

        // Mark the initial line as dirty (like Kitty's init_text_loop_line)
        self.mark_line_dirty(cached_row);

        for &cp in codepoints {
            // Fast path for ASCII control characters and printable ASCII
            // These are the most common cases, so check them first using u32 directly
            match cp {
                // Bell
                0x07 => {
                    // BEL - ignore for now (could trigger visual bell)
                }
                // Backspace
                0x08 => {
                    if self.cursor_col > 0 {
                        self.cursor_col -= 1;
                    }
                }
                // Tab
                0x09 => {
                    let next_tab = (self.cursor_col / 8 + 1) * 8;
                    self.cursor_col = next_tab.min(self.cols - 1);
                }
                // Line feed, Vertical tab, Form feed
                0x0A | 0x0B | 0x0C => {
                    let old_row = self.cursor_row;
                    self.cursor_row += 1;
                    if self.cursor_row > self.scroll_bottom {
                        self.scroll_up(1);
                        self.cursor_row = self.scroll_bottom;
                        log::trace!(
                            "LF: scrolled at row {}, now at scroll_bottom {}",
                            old_row,
                            self.cursor_row
                        );
                    }
                    // Update cache after line change
                    cached_row = self.cursor_row;
                    grid_row = self.line_map[cached_row];
                    // Mark the new line as dirty
                    self.mark_line_dirty(cached_row);
                }
                // Carriage return
                0x0D => {
                    self.cursor_col = 0;
                }
                // Fast path for printable ASCII (0x20-0x7E) - like Kitty
                // ASCII is always width 1, never zero-width, never wide
                cp if cp >= 0x20 && cp <= 0x7E => {
                    // Handle wrap
                    if self.cursor_col >= self.cols {
                        if self.auto_wrap {
                            self.cursor_col = 0;
                            self.advance_row();
                            cached_row = self.cursor_row;
                            grid_row = self.line_map[cached_row];
                            self.mark_line_dirty(cached_row);
                        } else {
                            self.cursor_col = self.cols - 1;
                        }
                    }

                    // Write character directly - no wide char handling needed for ASCII
                    // SAFETY: cp is in 0x20..=0x7E which are valid ASCII chars
                    let c = unsafe { char::from_u32_unchecked(cp) };
                    self.grid[grid_row][self.cursor_col] =
                        self.make_cell(c, false);
                    self.cursor_col += 1;
                }
                // Slow path for non-ASCII printable characters (including all Unicode)
                // Delegates to print_char() which handles wide characters, wrapping, etc.
                cp if cp > 0x7E => {
                    // Convert to char, using replacement character for invalid codepoints
                    let c = char::from_u32(cp).unwrap_or('\u{FFFD}');
                    self.print_char(c);
                    // Update cached values since print_char may have scrolled or wrapped
                    if cached_row != self.cursor_row {
                        cached_row = self.cursor_row;
                        grid_row = self.line_map[cached_row];
                    }
                }
                // Other control chars - ignore
                _ => {}
            }
        }
        // Dirty lines are marked incrementally above - no need for mark_all_lines_dirty()

        #[cfg(feature = "render_timing")]
        {
            self.stats.text_handler_ns += start.elapsed().as_nanos() as u64;
            self.stats.chars_processed += codepoints.len() as u32;
        }
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
                self.advance_row();
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
                            if let Ok(color_spec) =
                                std::str::from_utf8(parts[2])
                            {
                                if let Some(rgb) =
                                    ColorPalette::parse_color_spec(color_spec)
                                {
                                    self.palette.colors[index as usize] = rgb;
                                    log::debug!(
                                        "OSC 4: Set color {} to {:?}",
                                        index,
                                        rgb
                                    );
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
                        if let Some(rgb) =
                            ColorPalette::parse_color_spec(color_spec)
                        {
                            self.palette.default_fg = rgb;
                            log::debug!(
                                "OSC 10: Set default foreground to {:?}",
                                rgb
                            );
                        }
                    }
                }
            }
            // OSC 11 - Set/query default background color
            11 => {
                if parts.len() >= 2 {
                    if let Ok(color_spec) = std::str::from_utf8(parts[1]) {
                        if let Some(rgb) =
                            ColorPalette::parse_color_spec(color_spec)
                        {
                            self.palette.default_bg = rgb;
                            log::debug!(
                                "OSC 11: Set default background to {:?}",
                                rgb
                            );
                        }
                    }
                }
            }
            // OSC 51 - ZTerm custom commands
            // Format: OSC 51;command;args ST
            // Currently supported:
            //   OSC 51;navigate;up/down/left/right ST - Navigate to neighboring pane
            //   OSC 51;statusline;<content> ST - Set custom statusline (empty to clear)
            51 => {
                if parts.len() >= 2 {
                    if let Ok(command) = std::str::from_utf8(parts[1]) {
                        match command {
                            "navigate" => {
                                if parts.len() >= 3 {
                                    if let Ok(direction_str) =
                                        std::str::from_utf8(parts[2])
                                    {
                                        let direction = match direction_str {
                                            "up" => Some(Direction::Up),
                                            "down" => Some(Direction::Down),
                                            "left" => Some(Direction::Left),
                                            "right" => Some(Direction::Right),
                                            _ => None,
                                        };
                                        if let Some(dir) = direction {
                                            log::debug!(
                                                "OSC 51: Navigate {:?}",
                                                dir
                                            );
                                            self.command_queue.push(
                                                TerminalCommand::NavigatePane(
                                                    dir,
                                                ),
                                            );
                                        }
                                    }
                                }
                            }
                            "statusline" => {
                                // OSC 51;statusline;<content> ST
                                // If content is empty or missing, clear the statusline
                                // Content may be base64-encoded (prefixed with "b64:") to avoid
                                // escape sequence interpretation issues in the terminal
                                let prefix = b"51;statusline;";
                                let raw_content = if data.len() > prefix.len()
                                    && data.starts_with(prefix)
                                {
                                    std::str::from_utf8(&data[prefix.len()..])
                                        .ok()
                                        .map(|s| s.to_string())
                                } else if parts.len() >= 3 {
                                    std::str::from_utf8(parts[2])
                                        .ok()
                                        .map(|s| s.to_string())
                                } else {
                                    None
                                };

                                // Decode base64 if prefixed with "b64:"
                                let content = raw_content.and_then(|s| {
                                    if let Some(encoded) = s.strip_prefix("b64:") {
                                        use base64::Engine;
                                        base64::engine::general_purpose::STANDARD
                                            .decode(encoded)
                                            .ok()
                                            .and_then(|bytes| String::from_utf8(bytes).ok())
                                    } else {
                                        Some(s)
                                    }
                                });

                                let statusline =
                                    content.filter(|s| !s.is_empty());
                                log::info!(
                                    "OSC 51: Set statusline: {:?}",
                                    statusline
                                        .as_ref()
                                        .map(|s| format!("{} bytes", s.len()))
                                );
                                self.command_queue.push(
                                    TerminalCommand::SetStatusline(statusline),
                                );
                            }
                            _ => {
                                log::debug!(
                                    "OSC 51: Unknown command '{}'",
                                    command
                                );
                            }
                        }
                    }
                }
            }
            // OSC 52 - Clipboard operations
            // Format: OSC 52;Pc;Pd ST
            // Pc = clipboard type ('c' for clipboard, 'p' for primary, 's' for selection)
            // Pd = base64-encoded data to set, or '?' to query
            52 => {
                if parts.len() >= 3 {
                    if let Ok(data_str) = std::str::from_utf8(parts[2]) {
                        if data_str == "?" {
                            log::debug!(
                                "OSC 52: Query clipboard (not implemented)"
                            );
                        } else {
                            use base64::Engine;
                            if let Ok(decoded) =
                                base64::engine::general_purpose::STANDARD
                                    .decode(data_str)
                            {
                                if let Ok(text) = String::from_utf8(decoded) {
                                    log::debug!(
                                        "OSC 52: Set clipboard ({} bytes)",
                                        text.len()
                                    );
                                    self.command_queue.push(
                                        TerminalCommand::SetClipboard(text),
                                    );
                                }
                            }
                        }
                    }
                }
            }
            _ => {
                log::debug!("Unhandled OSC {}", osc_num);
            }
        }
    }

    /// Handle an APC (Application Program Command) sequence.
    /// Used for Kitty graphics protocol.
    fn apc(&mut self, data: &[u8]) {
        self.handle_apc(data);
    }

    /// Handle a DCS (Device Control String) sequence.
    /// Used for pending mode (synchronized output via DCS).
    fn dcs(&mut self, data: &[u8]) {
        // DCS pending mode: =1s to start, =2s to stop
        // This is an alternative to CSI 2026 for synchronized output
        if data.len() >= 3 && data[0] == b'=' && data[2] == b's' {
            match data[1] {
                b'1' => {
                    // Start pending mode (pause rendering)
                    if self.synchronized_output {
                        log::warn!("Pending mode start requested while already in pending mode");
                    }
                    self.synchronized_output = true;
                    log::trace!("DCS pending mode started (=1s)");
                }
                b'2' => {
                    // Stop pending mode (resume rendering)
                    if !self.synchronized_output {
                        log::warn!("Pending mode stop requested while not in pending mode");
                    }
                    self.synchronized_output = false;
                    self.dirty = true; // Force a redraw
                    log::trace!("DCS pending mode stopped (=2s)");
                }
                _ => {
                    log::debug!("Unknown DCS pending mode command: {:?}", data);
                }
            }
        } else {
            log::debug!(
                "Unhandled DCS sequence: {:?}",
                std::str::from_utf8(data).unwrap_or("<invalid utf8>")
            );
        }
    }

    /// Handle a complete CSI sequence.
    #[inline]
    fn csi(&mut self, params: &CsiParams) {
        #[cfg(feature = "render_timing")]
        let start = std::time::Instant::now();

        let action = params.final_char as char;
        let primary = params.primary;
        let secondary = params.secondary;

        match action {
            'A' => {
                let n = params.get(0, 1).max(1) as usize;
                let min_row =
                    if self.origin_mode { self.scroll_top } else { 0 };
                self.cursor_row =
                    self.cursor_row.saturating_sub(n).max(min_row);
            }
            'B' => {
                let n = params.get(0, 1).max(1) as usize;
                let max_row = if self.origin_mode {
                    self.scroll_bottom
                } else {
                    self.rows - 1
                };
                self.cursor_row = (self.cursor_row + n).min(max_row);
            }
            // Cursor Forward
            'C' => {
                let n = params.get(0, 1).max(1) as usize;
                let old_col = self.cursor_col;
                self.cursor_col = (self.cursor_col + n).min(self.cols - 1);
                log::trace!(
                    "CSI C: cursor forward {} from col {} to {}",
                    n,
                    old_col,
                    self.cursor_col
                );
            }
            // Cursor Back
            'D' => {
                let n = params.get(0, 1).max(1) as usize;
                self.cursor_col = self.cursor_col.saturating_sub(n);
            }
            'E' => {
                let n = params.get(0, 1).max(1) as usize;
                let max_row = if self.origin_mode {
                    self.scroll_bottom
                } else {
                    self.rows - 1
                };
                self.cursor_col = 0;
                self.cursor_row = (self.cursor_row + n).min(max_row);
            }
            'F' => {
                let n = params.get(0, 1).max(1) as usize;
                let min_row =
                    if self.origin_mode { self.scroll_top } else { 0 };
                self.cursor_col = 0;
                self.cursor_row =
                    self.cursor_row.saturating_sub(n).max(min_row);
            }
            // Cursor Horizontal Absolute (CHA)
            'G' => {
                let col = params.get(0, 1).max(1) as usize;
                let old_col = self.cursor_col;
                self.cursor_col = (col - 1).min(self.cols - 1);
                log::trace!(
                    "CSI G: cursor to col {} (was {})",
                    self.cursor_col,
                    old_col
                );
            }
            // Cursor Position
            'H' | 'f' => {
                let row = params.get(0, 1).max(1) as usize;
                let col = params.get(1, 1).max(1) as usize;
                if self.origin_mode {
                    let abs_row =
                        (self.scroll_top + row - 1).min(self.scroll_bottom);
                    self.cursor_row = abs_row;
                } else {
                    self.cursor_row = (row - 1).min(self.rows - 1);
                }
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
                        log::warn!(
                            "DEBUG EL1 (erase to cursor): row={} cursor_col={}",
                            self.cursor_row,
                            self.cursor_col
                        );
                        let grid_row = self.line_map[self.cursor_row];
                        for col in 0..=self.cursor_col {
                            self.grid[grid_row][col] = blank;
                        }
                        self.mark_line_dirty(self.cursor_row);
                    }
                    2 => {
                        log::warn!(
                            "DEBUG EL2 (erase whole line): row={}",
                            self.cursor_row
                        );
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
            // Optimized like Kitty: batch writes for ASCII, avoid per-char overhead
            'b' => {
                let n = (params.get(0, 1).max(1) as usize).min(65535); // Like Kitty's CSI_REP_MAX_REPETITIONS
                if self.cursor_col > 0 && n > 0 {
                    let grid_row = self.line_map[self.cursor_row];
                    let last_char =
                        self.grid[grid_row][self.cursor_col - 1].character;
                    let last_cp = last_char as u32;

                    // Fast path for ASCII: direct grid write, no width lookup
                    if last_cp >= 0x20 && last_cp <= 0x7E {
                        let cell = self.make_cell(last_char, false);
                        self.mark_line_dirty(self.cursor_row);

                        for _ in 0..n {
                            // Handle wrap
                            if self.cursor_col >= self.cols {
                                if self.auto_wrap {
                                    self.cursor_col = 0;
                                    self.advance_row();
                                    self.mark_line_dirty(self.cursor_row);
                                } else {
                                    self.cursor_col = self.cols - 1;
                                }
                            }
                            // Direct write - recompute grid_row in case of scroll
                            let gr = self.line_map[self.cursor_row];
                            self.grid[gr][self.cursor_col] = cell.clone();
                            self.cursor_col += 1;
                        }
                    } else {
                        // Slow path for non-ASCII: use print_char for proper width handling
                        for _ in 0..n {
                            self.print_char(last_char);
                        }
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
                if self.origin_mode {
                    let abs_row =
                        (self.scroll_top + row - 1).min(self.scroll_bottom);
                    self.cursor_row = abs_row;
                } else {
                    self.cursor_row = (row - 1).min(self.rows - 1);
                }
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
                        let response = format!(
                            "\x1b[{};{}R",
                            self.cursor_row + 1,
                            self.cursor_col + 1
                        );
                        self.response_queue
                            .extend_from_slice(response.as_bytes());
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
                    std::mem::swap(
                        &mut self.scroll_top,
                        &mut self.scroll_bottom,
                    );
                }
                // Move cursor to home position (respects origin mode)
                self.cursor_row =
                    if self.origin_mode { self.scroll_top } else { 0 };
                self.cursor_col = 0;
            }
            // Window manipulation (CSI Ps t) - XTWINOPS
            't' => {
                let ps = params.get(0, 0);
                match ps {
                    14 => {
                        // Report text area size in pixels: CSI 4 ; height ; width t
                        let pixel_height =
                            (self.rows as f32 * self.cell_height) as u32;
                        let pixel_width =
                            (self.cols as f32 * self.cell_width) as u32;
                        let response =
                            format!("\x1b[4;{};{}t", pixel_height, pixel_width);
                        self.response_queue
                            .extend_from_slice(response.as_bytes());
                        log::debug!(
                            "XTWINOPS 14: Reported text area size {}x{} pixels",
                            pixel_width,
                            pixel_height
                        );
                    }
                    16 => {
                        // Report cell size in pixels: CSI 6 ; height ; width t
                        let cell_h = self.cell_height as u32;
                        let cell_w = self.cell_width as u32;
                        let response = format!("\x1b[6;{};{}t", cell_h, cell_w);
                        self.response_queue
                            .extend_from_slice(response.as_bytes());
                        log::debug!(
                            "XTWINOPS 16: Reported cell size {}x{} pixels",
                            cell_w,
                            cell_h
                        );
                    }
                    18 => {
                        // Report text area size in characters: CSI 8 ; rows ; cols t
                        let response =
                            format!("\x1b[8;{};{}t", self.rows, self.cols);
                        self.response_queue
                            .extend_from_slice(response.as_bytes());
                        log::debug!(
                            "XTWINOPS 18: Reported text area size {}x{} chars",
                            self.cols,
                            self.rows
                        );
                    }
                    22 | 23 => {
                        // Save/restore window title - ignore
                    }
                    _ => {
                        log::trace!("Window manipulation: ps={}", ps);
                    }
                }
            }
            // ANSI Save Cursor (CSI s) - DECSLRM uses CSI ? s which has primary='?'
            's' if primary == 0 => {
                self.save_cursor();
            }
            // CSI u: ANSI restore cursor (no params) vs Kitty keyboard protocol (with params)
            'u' => {
                if primary == 0 && params.num_params == 0 {
                    self.restore_cursor();
                } else {
                    self.handle_keyboard_protocol_csi(params);
                }
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

        #[cfg(feature = "render_timing")]
        {
            self.stats.csi_handler_ns += start.elapsed().as_nanos() as u64;
            self.stats.csi_count += 1;
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
            underline_style: self.current_underline_style,
            strikethrough: self.current_strikethrough,
            origin_mode: self.origin_mode,
            auto_wrap: self.auto_wrap,
        };
        log::debug!(
            "ESC 7: Cursor saved at ({}, {}), origin_mode={}, auto_wrap={}",
            self.cursor_col,
            self.cursor_row,
            self.origin_mode,
            self.auto_wrap
        );
    }

    fn restore_cursor(&mut self) {
        self.cursor_col =
            self.saved_cursor.col.min(self.cols.saturating_sub(1));
        self.cursor_row =
            self.saved_cursor.row.min(self.rows.saturating_sub(1));
        self.current_fg = self.saved_cursor.fg;
        self.current_bg = self.saved_cursor.bg;
        self.current_bold = self.saved_cursor.bold;
        self.current_italic = self.saved_cursor.italic;
        self.current_underline_style = self.saved_cursor.underline_style;
        self.current_strikethrough = self.saved_cursor.strikethrough;
        self.origin_mode = self.saved_cursor.origin_mode;
        self.auto_wrap = self.saved_cursor.auto_wrap;
        log::debug!(
            "ESC 8: Cursor restored to ({}, {}), origin_mode={}, auto_wrap={}",
            self.cursor_col,
            self.cursor_row,
            self.origin_mode,
            self.auto_wrap
        );
    }

    fn reset(&mut self) {
        self.current_fg = Color::Default;
        self.current_bg = Color::Default;
        self.current_bold = false;
        self.current_italic = false;
        self.current_underline_style = 0;
        self.current_strikethrough = false;
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
        self.origin_mode = false;
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
                    underline_style: 0,
                    strikethrough: false,
                    wide_continuation: false,
                };
            }
            self.mark_line_dirty(visual_row);
        }
    }

    #[cfg(feature = "render_timing")]
    fn add_vt_parser_ns(&mut self, ns: u64) {
        self.stats.vt_parser_ns += ns;
        self.stats.consume_input_count += 1;
    }

    #[cfg(not(feature = "render_timing"))]
    fn add_vt_parser_ns(&mut self, _ns: u64) {}
}

impl Terminal {
    /// Print a single character at the cursor position.
    /// Handles double-width characters (emoji, CJK) by occupying two cells.
    #[inline]
    fn print_char(&mut self, c: char) {
        // Determine character width using Unicode Standard Annex #11
        // Width 2 = double-width (emoji, CJK, etc.)
        // Width 1 = normal width
        // Width 0 = combining/non-spacing marks (handled separately)
        let char_width = c.width().unwrap_or(1);

        // Skip zero-width characters (combining marks, etc.)
        if char_width == 0 {
            // TODO: Handle combining characters by attaching to previous cell
            return;
        }

        // Check if we need to wrap before printing
        if self.cursor_col >= self.cols {
            if self.auto_wrap {
                self.cursor_col = 0;
                self.advance_row();
            } else {
                self.cursor_col = self.cols - 1;
            }
        }

        // For double-width characters, check if there's room
        // If at the last column, we need to wrap first
        if char_width == 2 && self.cursor_col == self.cols - 1 {
            if self.auto_wrap {
                // Write a space in the last column and wrap
                let grid_row = self.line_map[self.cursor_row];
                self.grid[grid_row][self.cursor_col] = Cell::default();
                self.cursor_col = 0;
                self.advance_row();
            } else {
                // Can't fit, don't print
                return;
            }
        }

        let grid_row = self.line_map[self.cursor_row];

        // If we're overwriting a wide character's continuation cell,
        // we need to clear the first cell of that wide character
        if self.grid[grid_row][self.cursor_col].wide_continuation
            && self.cursor_col > 0
        {
            self.grid[grid_row][self.cursor_col - 1] = Cell::default();
        }

        // If we're overwriting the first cell of a wide character,
        // we need to clear its continuation cell
        if char_width == 1
            && self.cursor_col + 1 < self.cols
            && self.grid[grid_row][self.cursor_col + 1].wide_continuation
        {
            self.grid[grid_row][self.cursor_col + 1] = Cell::default();
        }

        // Write the character to the first cell
        self.grid[grid_row][self.cursor_col] = self.make_cell(c, false);
        self.mark_line_dirty(self.cursor_row);
        self.cursor_col += 1;

        // For double-width characters, write a continuation marker to the second cell
        if char_width == 2 && self.cursor_col < self.cols {
            // If the next cell is the first cell of another wide character,
            // clear its continuation cell
            if self.cursor_col + 1 < self.cols
                && self.grid[grid_row][self.cursor_col + 1].wide_continuation
            {
                self.grid[grid_row][self.cursor_col + 1] = Cell::default();
            }

            self.grid[grid_row][self.cursor_col] = self.make_cell(' ', true);
            self.cursor_col += 1;
        }
    }

    /// Parse extended color (SGR 38/48) and return the color and number of params consumed.
    /// Returns (Color, params_consumed) or None if parsing failed.
    ///
    /// SAFETY: Caller must ensure i < params.num_params
    #[inline(always)]
    fn parse_extended_color(
        params: &CsiParams,
        i: usize,
    ) -> Option<(Color, usize)> {
        let num = params.num_params;
        let p = &params.params;
        let is_sub = &params.is_sub_param;

        // Check for sub-parameter format (38:2:r:g:b or 38:5:idx)
        if i + 1 < num && is_sub[i + 1] {
            let mode = p[i + 1];
            if mode == 5 && i + 2 < num {
                return Some((Color::Indexed(p[i + 2] as u8), 2));
            } else if mode == 2 && i + 4 < num {
                return Some((
                    Color::Rgb(p[i + 2] as u8, p[i + 3] as u8, p[i + 4] as u8),
                    4,
                ));
            }
        } else if i + 2 < num {
            // Regular format (38;2;r;g;b or 38;5;idx)
            let mode = p[i + 1];
            if mode == 5 {
                return Some((Color::Indexed(p[i + 2] as u8), 2));
            } else if mode == 2 && i + 4 < num {
                return Some((
                    Color::Rgb(p[i + 2] as u8, p[i + 3] as u8, p[i + 4] as u8),
                    4,
                ));
            }
        }
        None
    }

    /// Handle SGR (Select Graphic Rendition) parameters.
    /// This is a hot path - called for every color/style change in terminal output.
    #[inline(always)]
    fn handle_sgr(&mut self, params: &CsiParams) {
        let num = params.num_params;

        // Fast path: SGR 0 (reset) with no params or explicit 0
        if num == 0 {
            self.reset_sgr_attributes();
            return;
        }

        let p = &params.params;
        let is_sub = &params.is_sub_param;
        let mut i = 0;

        while i < num {
            // SAFETY: i < num <= MAX_CSI_PARAMS, so index is always valid
            let code = p[i];

            match code {
                0 => self.reset_sgr_attributes(),
                1 => self.current_bold = true,
                // 2 => dim (not currently rendered)
                3 => self.current_italic = true,
                4 => {
                    // Check for sub-parameter (4:x format for underline style)
                    if i + 1 < num && is_sub[i + 1] {
                        // 0=none, 1=single, 2=double, 3=curly, 4=dotted, 5=dashed
                        self.current_underline_style = (p[i + 1] as u8).min(5);
                        i += 1;
                    } else {
                        // Plain SGR 4 = single underline
                        self.current_underline_style = 1;
                    }
                }
                7 => std::mem::swap(&mut self.current_fg, &mut self.current_bg),
                9 => self.current_strikethrough = true,
                21 => self.current_underline_style = 2, // Double underline
                22 => self.current_bold = false,
                23 => self.current_italic = false,
                24 => self.current_underline_style = 0,
                27 => {
                    std::mem::swap(&mut self.current_fg, &mut self.current_bg)
                }
                29 => self.current_strikethrough = false,
                // Standard foreground colors (30-37)
                30..=37 => self.current_fg = Color::Indexed((code - 30) as u8),
                38 => {
                    // Extended foreground color
                    if let Some((color, consumed)) =
                        Self::parse_extended_color(params, i)
                    {
                        self.current_fg = color;
                        i += consumed;
                    }
                }
                39 => self.current_fg = Color::Default,
                // Standard background colors (40-47)
                40..=47 => self.current_bg = Color::Indexed((code - 40) as u8),
                48 => {
                    // Extended background color
                    if let Some((color, consumed)) =
                        Self::parse_extended_color(params, i)
                    {
                        self.current_bg = color;
                        i += consumed;
                    }
                }
                49 => self.current_bg = Color::Default,
                // Bright foreground colors (90-97)
                90..=97 => {
                    self.current_fg = Color::Indexed((code - 90 + 8) as u8)
                }
                // Bright background colors (100-107)
                100..=107 => {
                    self.current_bg = Color::Indexed((code - 100 + 8) as u8)
                }
                _ => {}
            }
            i += 1;
        }
    }

    /// Reset all SGR attributes to defaults.
    #[inline(always)]
    fn reset_sgr_attributes(&mut self) {
        self.current_fg = Color::Default;
        self.current_bg = Color::Default;
        self.current_bold = false;
        self.current_italic = false;
        self.current_underline_style = 0;
        self.current_strikethrough = false;
    }

    /// Handle Kitty keyboard protocol CSI sequences.
    #[inline]
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
                log::debug!(
                    "Keyboard flags set to {:?} (mode {})",
                    self.keyboard.flags(),
                    mode
                );
            }
            b'>' => {
                let flags = if params.num_params == 0 {
                    None
                } else {
                    Some(params.params[0] as u8)
                };
                self.keyboard.push(flags);
                log::debug!(
                    "Keyboard flags pushed: {:?}",
                    self.keyboard.flags()
                );
            }
            b'<' => {
                let count = params.get(0, 1) as usize;
                self.keyboard.pop(count);
                log::debug!(
                    "Keyboard flags popped: {:?}",
                    self.keyboard.flags()
                );
            }
            _ => {}
        }
    }

    /// Handle DEC private mode set (CSI ? Ps h).
    #[inline]
    fn handle_dec_private_mode_set(&mut self, params: &CsiParams) {
        for i in 0..params.num_params {
            match params.params[i] {
                1 => {
                    self.application_cursor_keys = true;
                    log::debug!("DECCKM: Application cursor keys enabled");
                }
                6 => {
                    self.origin_mode = true;
                    self.cursor_row = self.scroll_top;
                    self.cursor_col = 0;
                    log::debug!(
                        "DECOM: Origin mode enabled, cursor at ({}, {})",
                        self.cursor_col,
                        self.cursor_row
                    );
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
                _ => log::debug!(
                    "Unhandled DEC private mode set: {}",
                    params.params[i]
                ),
            }
        }
    }

    /// Handle DEC private mode reset (CSI ? Ps l).
    #[inline]
    fn handle_dec_private_mode_reset(&mut self, params: &CsiParams) {
        for i in 0..params.num_params {
            match params.params[i] {
                1 => {
                    self.application_cursor_keys = false;
                    log::debug!("DECCKM: Normal cursor keys enabled");
                }
                6 => {
                    self.origin_mode = false;
                    self.cursor_row = 0;
                    self.cursor_col = 0;
                    log::debug!(
                        "DECOM: Origin mode disabled, cursor at (0, 0)"
                    );
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
                        log::debug!(
                            "Mouse tracking: Button-event mode disabled"
                        );
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
                _ => log::debug!(
                    "Unhandled DEC private mode reset: {}",
                    params.params[i]
                ),
            }
        }
    }

    /// Set cell dimensions (called by renderer after font metrics are calculated).
    pub fn set_cell_size(&mut self, width: f32, height: f32) {
        self.cell_width = width;
        self.cell_height = height;
    }

    /// Handle an APC (Application Program Command) sequence.
    /// This is used for the Kitty graphics protocol.
    fn handle_apc(&mut self, data: &[u8]) {
        // Kitty graphics protocol: APC starts with 'G'
        if let Some(cmd) = GraphicsCommand::parse(data) {
            log::debug!(
                "Graphics command: action={:?} format={:?} id={:?} size={}x{:?} C={} U={}",
                cmd.action,
                cmd.format,
                cmd.image_id,
                cmd.width.unwrap_or(0),
                cmd.height,
                cmd.cursor_movement,
                cmd.unicode_placeholder
            );

            // Convert cursor_row to absolute row (accounting for scrollback)
            // This allows images to scroll with terminal content
            let absolute_row = self.scrollback.len() + self.cursor_row;

            // Process the command
            let (response, placement_result) =
                self.image_storage.process_command(
                    cmd,
                    self.cursor_col,
                    absolute_row,
                    self.cell_width,
                    self.cell_height,
                );

            // Queue the response to send back to the application
            if let Some(resp) = response {
                self.response_queue.extend_from_slice(resp.as_bytes());
            }

            // Move cursor after image placement per Kitty protocol spec:
            // "After placing an image on the screen the cursor must be moved to the
            // right by the number of cols in the image placement rectangle and down
            // by the number of rows in the image placement rectangle."
            // However, if C=1 was specified, don't move the cursor.
            if let Some(placement) = placement_result {
                if !placement.suppress_cursor_move
                    && !placement.virtual_placement
                {
                    // Move cursor down by (rows - 1) since we're already on the first row
                    // Then set cursor to the column after the image
                    let new_row =
                        self.cursor_row + placement.rows.saturating_sub(1);
                    if new_row >= self.rows {
                        // Need to scroll
                        let scroll_amount = new_row - self.rows + 1;
                        self.scroll_up(scroll_amount);
                        self.cursor_row = self.rows - 1;
                    } else {
                        self.cursor_row = new_row;
                    }
                    // Move cursor to after the image (or stay at column 0 of next line)
                    // Per protocol, cursor ends at the last row of the image
                    log::debug!(
                        "Cursor moved after image placement: row={} (moved {} rows)",
                        self.cursor_row, placement.rows.saturating_sub(1)
                    );
                }
            }
        }
    }
}
