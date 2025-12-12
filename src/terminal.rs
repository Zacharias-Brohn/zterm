//! Terminal state management and escape sequence handling.

use crate::keyboard::{query_response, KeyboardState};
use vte::{Params, Parser, Perform};

/// A single cell in the terminal grid.
#[derive(Clone, Debug)]
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
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Color {
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

/// Color palette with 256 colors + default fg/bg.
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

/// The terminal grid state.
pub struct Terminal {
    /// Grid of cells (row-major order).
    pub grid: Vec<Vec<Cell>>,
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
    pub scrollback: Vec<Vec<Cell>>,
    /// Maximum number of lines to keep in scrollback.
    pub scrollback_limit: usize,
    /// Current scroll offset (0 = viewing live terminal, >0 = viewing history).
    pub scroll_offset: usize,
}

impl Terminal {
    /// Default scrollback limit (10,000 lines).
    const DEFAULT_SCROLLBACK_LIMIT: usize = 10_000;

    /// Creates a new terminal with the given dimensions.
    pub fn new(cols: usize, rows: usize) -> Self {
        let grid = vec![vec![Cell::default(); cols]; rows];

        Self {
            grid,
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
            scroll_top: 0,
            scroll_bottom: rows.saturating_sub(1),
            keyboard: KeyboardState::new(),
            response_queue: Vec::new(),
            palette: ColorPalette::default(),
            scrollback: Vec::new(),
            scrollback_limit: Self::DEFAULT_SCROLLBACK_LIMIT,
            scroll_offset: 0,
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

    /// Processes raw bytes from the PTY using the provided parser.
    pub fn process(&mut self, bytes: &[u8], parser: &mut Parser) {
        for byte in bytes {
            parser.advance(self, *byte);
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

        // Copy existing content
        for row in 0..rows.min(self.rows) {
            for col in 0..cols.min(self.cols) {
                new_grid[row][col] = self.grid[row][col].clone();
            }
        }

        self.grid = new_grid;
        self.cols = cols;
        self.rows = rows;

        // Reset scroll region to full screen
        self.scroll_top = 0;
        self.scroll_bottom = rows.saturating_sub(1);

        // Adjust cursor position
        self.cursor_col = self.cursor_col.min(cols.saturating_sub(1));
        self.cursor_row = self.cursor_row.min(rows.saturating_sub(1));
        self.dirty = true;
    }

    /// Scrolls the scroll region up by n lines.
    fn scroll_up(&mut self, n: usize) {
        let n = n.min(self.scroll_bottom - self.scroll_top + 1);
        for _ in 0..n {
            // Remove the top line of the scroll region
            let removed_line = self.grid.remove(self.scroll_top);
            
            // Save to scrollback only if scrolling from the very top of the screen
            if self.scroll_top == 0 {
                self.scrollback.push(removed_line);
                // Trim scrollback if it exceeds the limit
                if self.scrollback.len() > self.scrollback_limit {
                    self.scrollback.remove(0);
                }
            }
            
            // Insert a new blank line at the bottom of the scroll region
            self.grid
                .insert(self.scroll_bottom, vec![Cell::default(); self.cols]);
        }
    }

    /// Scrolls the scroll region down by n lines.
    fn scroll_down(&mut self, n: usize) {
        let n = n.min(self.scroll_bottom - self.scroll_top + 1);
        for _ in 0..n {
            // Remove the bottom line of the scroll region
            self.grid.remove(self.scroll_bottom);
            // Insert a new blank line at the top of the scroll region
            self.grid
                .insert(self.scroll_top, vec![Cell::default(); self.cols]);
        }
    }

    /// Scrolls the viewport up (into scrollback history) by n lines.
    /// Returns the new scroll offset.
    pub fn scroll_viewport_up(&mut self, n: usize) -> usize {
        let max_offset = self.scrollback.len();
        self.scroll_offset = (self.scroll_offset + n).min(max_offset);
        self.dirty = true;
        self.scroll_offset
    }

    /// Scrolls the viewport down (toward live terminal) by n lines.
    /// Returns the new scroll offset.
    pub fn scroll_viewport_down(&mut self, n: usize) -> usize {
        self.scroll_offset = self.scroll_offset.saturating_sub(n);
        self.dirty = true;
        self.scroll_offset
    }

    /// Resets viewport to show live terminal (scroll_offset = 0).
    pub fn reset_scroll(&mut self) {
        if self.scroll_offset != 0 {
            self.scroll_offset = 0;
            self.dirty = true;
        }
    }

    /// Returns the visible rows accounting for scroll offset.
    /// This combines scrollback lines with the current grid.
    pub fn visible_rows(&self) -> Vec<&Vec<Cell>> {
        let mut rows = Vec::with_capacity(self.rows);
        
        if self.scroll_offset == 0 {
            // No scrollback viewing, just return the grid
            for row in &self.grid {
                rows.push(row);
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
                    let scrollback_idx = scrollback_len - self.scroll_offset + i;
                    if scrollback_idx < scrollback_len {
                        rows.push(&self.scrollback[scrollback_idx]);
                    } else {
                        // Shouldn't happen, but fall back to grid
                        rows.push(&self.grid[i]);
                    }
                } else {
                    // This row comes from the grid
                    let grid_idx = i - lines_from_scrollback;
                    if grid_idx < self.grid.len() {
                        rows.push(&self.grid[grid_idx]);
                    }
                }
            }
        }
        
        rows
    }

    /// Inserts n blank lines at the cursor position, scrolling lines below down.
    fn insert_lines(&mut self, n: usize) {
        if self.cursor_row < self.scroll_top || self.cursor_row > self.scroll_bottom {
            return;
        }
        let n = n.min(self.scroll_bottom - self.cursor_row + 1);
        for _ in 0..n {
            // Remove the bottom line of the scroll region
            self.grid.remove(self.scroll_bottom);
            // Insert a new blank line at the cursor row
            self.grid
                .insert(self.cursor_row, vec![Cell::default(); self.cols]);
        }
    }

    /// Deletes n lines at the cursor position, scrolling lines below up.
    fn delete_lines(&mut self, n: usize) {
        if self.cursor_row < self.scroll_top || self.cursor_row > self.scroll_bottom {
            return;
        }
        let n = n.min(self.scroll_bottom - self.cursor_row + 1);
        for _ in 0..n {
            // Remove the line at cursor
            self.grid.remove(self.cursor_row);
            // Insert a new blank line at the bottom of the scroll region
            self.grid
                .insert(self.scroll_bottom, vec![Cell::default(); self.cols]);
        }
    }

    /// Inserts n blank characters at the cursor, shifting existing chars right.
    fn insert_characters(&mut self, n: usize) {
        let row = &mut self.grid[self.cursor_row];
        let n = n.min(self.cols - self.cursor_col);
        // Remove n characters from the end
        for _ in 0..n {
            row.pop();
        }
        // Insert n blank characters at cursor position
        for _ in 0..n {
            row.insert(self.cursor_col, Cell::default());
        }
    }

    /// Deletes n characters at the cursor, shifting remaining chars left.
    fn delete_characters(&mut self, n: usize) {
        let row = &mut self.grid[self.cursor_row];
        let n = n.min(self.cols - self.cursor_col);
        // Remove n characters at cursor position
        for _ in 0..n {
            if self.cursor_col < row.len() {
                row.remove(self.cursor_col);
            }
        }
        // Pad with blank characters at the end
        while row.len() < self.cols {
            row.push(Cell::default());
        }
    }

    /// Erases n characters at the cursor (replaces with spaces, doesn't shift).
    fn erase_characters(&mut self, n: usize) {
        let n = n.min(self.cols - self.cursor_col);
        for i in 0..n {
            if self.cursor_col + i < self.cols {
                self.grid[self.cursor_row][self.cursor_col + i] = Cell::default();
            }
        }
    }

    /// Clears the current line from cursor to end.
    fn clear_line_from_cursor(&mut self) {
        for col in self.cursor_col..self.cols {
            self.grid[self.cursor_row][col] = Cell::default();
        }
    }

    /// Clears the entire screen.
    fn clear_screen(&mut self) {
        for row in &mut self.grid {
            for cell in row {
                *cell = Cell::default();
            }
        }
        self.cursor_col = 0;
        self.cursor_row = 0;
    }

    /// Handles Kitty keyboard protocol escape sequences.
    fn handle_keyboard_protocol(&mut self, params: &[u16], intermediates: &[u8]) {
        match intermediates {
            // CSI ? u - Query current keyboard flags
            [b'?'] => {
                let response = query_response(self.keyboard.flags());
                self.response_queue.extend(response);
            }
            // CSI = flags ; mode u - Set keyboard flags
            [b'='] => {
                let flags = params.first().copied().unwrap_or(0) as u8;
                let mode = params.get(1).copied().unwrap_or(1) as u8;
                self.keyboard.set_flags(flags, mode);
                log::debug!(
                    "Keyboard flags set to {:?} (mode {})",
                    self.keyboard.flags(),
                    mode
                );
            }
            // CSI > flags u - Push keyboard flags onto stack
            [b'>'] => {
                let flags = if params.is_empty() {
                    None
                } else {
                    Some(params[0] as u8)
                };
                self.keyboard.push(flags);
                log::debug!("Keyboard flags pushed: {:?}", self.keyboard.flags());
            }
            // CSI < number u - Pop keyboard flags from stack
            [b'<'] => {
                let count = params.first().copied().unwrap_or(1) as usize;
                self.keyboard.pop(count);
                log::debug!("Keyboard flags popped: {:?}", self.keyboard.flags());
            }
            _ => {
                // Unknown intermediate, ignore
            }
        }
    }
}

impl Perform for Terminal {
    fn print(&mut self, c: char) {
        if self.cursor_col >= self.cols {
            self.cursor_col = 0;
            self.cursor_row += 1;
            if self.cursor_row > self.scroll_bottom {
                self.scroll_up(1);
                self.cursor_row = self.scroll_bottom;
            }
        }

        self.grid[self.cursor_row][self.cursor_col] = Cell {
            character: c,
            fg_color: self.current_fg,
            bg_color: self.current_bg,
            bold: self.current_bold,
            italic: self.current_italic,
            underline: self.current_underline,
        };

        self.cursor_col += 1;
    }

    fn execute(&mut self, byte: u8) {
        match byte {
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
            // Line feed
            0x0A => {
                self.cursor_row += 1;
                if self.cursor_row > self.scroll_bottom {
                    self.scroll_up(1);
                    self.cursor_row = self.scroll_bottom;
                }
            }
            // Carriage return
            0x0D => {
                self.cursor_col = 0;
            }
            _ => {}
        }
    }

    fn hook(&mut self, _params: &Params, _intermediates: &[u8], _ignore: bool, _action: char) {}

    fn put(&mut self, _byte: u8) {}

    fn unhook(&mut self) {}

    fn osc_dispatch(&mut self, params: &[&[u8]], _bell_terminated: bool) {
        // Handle OSC sequences
        if params.is_empty() {
            return;
        }
        
        // First param is the OSC number
        let osc_num = match std::str::from_utf8(params[0]) {
            Ok(s) => s.parse::<u32>().unwrap_or(u32::MAX),
            Err(_) => return,
        };
        
        match osc_num {
            // OSC 4 - Set/query indexed color
            4 => {
                // Format: OSC 4 ; index ; color ST
                // params[0] = "4", params[1] = "index", params[2] = "color"
                if params.len() >= 3 {
                    if let Ok(index_str) = std::str::from_utf8(params[1]) {
                        if let Ok(index) = index_str.parse::<u8>() {
                            if let Ok(color_spec) = std::str::from_utf8(params[2]) {
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
                if params.len() >= 2 {
                    if let Ok(color_spec) = std::str::from_utf8(params[1]) {
                        if let Some(rgb) = ColorPalette::parse_color_spec(color_spec) {
                            self.palette.default_fg = rgb;
                            log::debug!("OSC 10: Set default foreground to {:?}", rgb);
                        }
                    }
                }
            }
            // OSC 11 - Set/query default background color
            11 => {
                if params.len() >= 2 {
                    if let Ok(color_spec) = std::str::from_utf8(params[1]) {
                        if let Some(rgb) = ColorPalette::parse_color_spec(color_spec) {
                            self.palette.default_bg = rgb;
                            log::debug!("OSC 11: Set default background to {:?}", rgb);
                        }
                    }
                }
            }
            // OSC 0, 1, 2 - Set window title (ignore for now)
            0 | 1 | 2 => {}
            _ => {
                log::debug!("Unhandled OSC {}", osc_num);
            }
        }
    }

    fn csi_dispatch(&mut self, params: &Params, intermediates: &[u8], _ignore: bool, action: char) {
        // For most commands, we just need the first value of each parameter group
        let flat_params: Vec<u16> = params.iter().map(|p| p[0]).collect();

        match action {
            // Cursor Up
            'A' => {
                let n = flat_params.first().copied().unwrap_or(1).max(1) as usize;
                self.cursor_row = self.cursor_row.saturating_sub(n);
            }
            // Cursor Down
            'B' => {
                let n = flat_params.first().copied().unwrap_or(1).max(1) as usize;
                self.cursor_row = (self.cursor_row + n).min(self.rows - 1);
            }
            // Cursor Forward
            'C' => {
                let n = flat_params.first().copied().unwrap_or(1).max(1) as usize;
                self.cursor_col = (self.cursor_col + n).min(self.cols - 1);
            }
            // Cursor Back
            'D' => {
                let n = flat_params.first().copied().unwrap_or(1).max(1) as usize;
                self.cursor_col = self.cursor_col.saturating_sub(n);
            }
            // Cursor Position
            'H' | 'f' => {
                let row = flat_params.first().copied().unwrap_or(1).max(1) as usize;
                let col = flat_params.get(1).copied().unwrap_or(1).max(1) as usize;
                self.cursor_row = (row - 1).min(self.rows - 1);
                self.cursor_col = (col - 1).min(self.cols - 1);
            }
            // Erase in Display
            'J' => {
                let mode = flat_params.first().copied().unwrap_or(0);
                match mode {
                    0 => {
                        // Clear from cursor to end of screen
                        self.clear_line_from_cursor();
                        for row in (self.cursor_row + 1)..self.rows {
                            for cell in &mut self.grid[row] {
                                *cell = Cell::default();
                            }
                        }
                    }
                    1 => {
                        // Clear from start to cursor
                        for row in 0..self.cursor_row {
                            for cell in &mut self.grid[row] {
                                *cell = Cell::default();
                            }
                        }
                        for col in 0..=self.cursor_col {
                            self.grid[self.cursor_row][col] = Cell::default();
                        }
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
                let mode = flat_params.first().copied().unwrap_or(0);
                match mode {
                    0 => self.clear_line_from_cursor(),
                    1 => {
                        for col in 0..=self.cursor_col {
                            self.grid[self.cursor_row][col] = Cell::default();
                        }
                    }
                    2 => {
                        for cell in &mut self.grid[self.cursor_row] {
                            *cell = Cell::default();
                        }
                    }
                    _ => {}
                }
            }
            // SGR (Select Graphic Rendition)
            'm' => {
                // Handle SGR with proper sub-parameter support
                // VTE gives us parameter groups - each group can have sub-params (colon-separated)
                // We need to handle both:
                //   - Legacy: ESC[38;5;196m  -> groups: [38], [5], [196]
                //   - Modern: ESC[38:5:196m  -> groups: [38, 5, 196]
                //   - Modern: ESC[38:2:r:g:bm -> groups: [38, 2, r, g, b]
                
                let param_groups: Vec<Vec<u16>> = params.iter()
                    .map(|subparams| subparams.iter().copied().collect())
                    .collect();
                
                log::debug!("SGR param_groups: {:?}", param_groups);
                
                if param_groups.is_empty() {
                    self.current_fg = Color::Default;
                    self.current_bg = Color::Default;
                    self.current_bold = false;
                    self.current_italic = false;
                    self.current_underline = false;
                    return;
                }

                let mut i = 0;
                while i < param_groups.len() {
                    let group = &param_groups[i];
                    let code = group.first().copied().unwrap_or(0);
                    
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
                        7 => {
                            // Reverse video - swap fg and bg
                            std::mem::swap(&mut self.current_fg, &mut self.current_bg);
                        }
                        22 => self.current_bold = false,
                        23 => self.current_italic = false,
                        24 => self.current_underline = false,
                        27 => {
                            // Reverse video off - swap back (simplified)
                            std::mem::swap(&mut self.current_fg, &mut self.current_bg);
                        }
                        30..=37 => self.current_fg = Color::Indexed((code - 30) as u8),
                        38 => {
                            // Foreground color - check for sub-parameters first (colon format)
                            if group.len() >= 3 && group[1] == 5 {
                                // Colon format: 38:5:index
                                self.current_fg = Color::Indexed(group[2] as u8);
                            } else if group.len() >= 5 && group[1] == 2 {
                                // Colon format: 38:2:r:g:b or 38:2:colorspace:r:g:b
                                // Check if we have colorspace indicator
                                if group.len() >= 6 {
                                    // 38:2:colorspace:r:g:b
                                    self.current_fg = Color::Rgb(
                                        group[3] as u8,
                                        group[4] as u8,
                                        group[5] as u8,
                                    );
                                } else {
                                    // 38:2:r:g:b
                                    self.current_fg = Color::Rgb(
                                        group[2] as u8,
                                        group[3] as u8,
                                        group[4] as u8,
                                    );
                                }
                            } else if i + 2 < param_groups.len() {
                                // Semicolon format: check next groups
                                let mode = param_groups[i + 1].first().copied().unwrap_or(0);
                                if mode == 5 {
                                    // 38;5;index
                                    let idx = param_groups[i + 2].first().copied().unwrap_or(0);
                                    self.current_fg = Color::Indexed(idx as u8);
                                    i += 2;
                                } else if mode == 2 && i + 4 < param_groups.len() {
                                    // 38;2;r;g;b
                                    let r = param_groups[i + 2].first().copied().unwrap_or(0);
                                    let g = param_groups[i + 3].first().copied().unwrap_or(0);
                                    let b = param_groups[i + 4].first().copied().unwrap_or(0);
                                    self.current_fg = Color::Rgb(r as u8, g as u8, b as u8);
                                    i += 4;
                                }
                            }
                        }
                        39 => self.current_fg = Color::Default,
                        40..=47 => self.current_bg = Color::Indexed((code - 40) as u8),
                        48 => {
                            // Background color - check for sub-parameters first (colon format)
                            if group.len() >= 3 && group[1] == 5 {
                                // Colon format: 48:5:index
                                self.current_bg = Color::Indexed(group[2] as u8);
                            } else if group.len() >= 5 && group[1] == 2 {
                                // Colon format: 48:2:r:g:b or 48:2:colorspace:r:g:b
                                if group.len() >= 6 {
                                    // 48:2:colorspace:r:g:b
                                    self.current_bg = Color::Rgb(
                                        group[3] as u8,
                                        group[4] as u8,
                                        group[5] as u8,
                                    );
                                } else {
                                    // 48:2:r:g:b
                                    self.current_bg = Color::Rgb(
                                        group[2] as u8,
                                        group[3] as u8,
                                        group[4] as u8,
                                    );
                                }
                            } else if i + 2 < param_groups.len() {
                                // Semicolon format: check next groups
                                let mode = param_groups[i + 1].first().copied().unwrap_or(0);
                                if mode == 5 {
                                    // 48;5;index
                                    let idx = param_groups[i + 2].first().copied().unwrap_or(0);
                                    self.current_bg = Color::Indexed(idx as u8);
                                    i += 2;
                                } else if mode == 2 && i + 4 < param_groups.len() {
                                    // 48;2;r;g;b
                                    let r = param_groups[i + 2].first().copied().unwrap_or(0);
                                    let g = param_groups[i + 3].first().copied().unwrap_or(0);
                                    let b = param_groups[i + 4].first().copied().unwrap_or(0);
                                    self.current_bg = Color::Rgb(r as u8, g as u8, b as u8);
                                    i += 4;
                                }
                            }
                        }
                        49 => self.current_bg = Color::Default,
                        90..=97 => {
                            self.current_fg = Color::Indexed((code - 90 + 8) as u8)
                        }
                        100..=107 => {
                            self.current_bg = Color::Indexed((code - 100 + 8) as u8)
                        }
                        _ => {}
                    }
                    i += 1;
                }
            }
            // Set Scrolling Region (DECSTBM)
            'r' => {
                let top = flat_params.first().copied().unwrap_or(1).max(1) as usize;
                let bottom = flat_params.get(1).copied().unwrap_or(self.rows as u16).max(1) as usize;
                self.scroll_top = (top - 1).min(self.rows - 1);
                self.scroll_bottom = (bottom - 1).min(self.rows - 1);
                if self.scroll_top > self.scroll_bottom {
                    std::mem::swap(&mut self.scroll_top, &mut self.scroll_bottom);
                }
                // Move cursor to home position
                self.cursor_row = 0;
                self.cursor_col = 0;
            }
            // Scroll Up (SU)
            'S' => {
                let n = flat_params.first().copied().unwrap_or(1).max(1) as usize;
                self.scroll_up(n);
            }
            // Scroll Down (SD)
            'T' => {
                let n = flat_params.first().copied().unwrap_or(1).max(1) as usize;
                self.scroll_down(n);
            }
            // Insert Lines (IL)
            'L' => {
                let n = flat_params.first().copied().unwrap_or(1).max(1) as usize;
                self.insert_lines(n);
            }
            // Delete Lines (DL)
            'M' => {
                let n = flat_params.first().copied().unwrap_or(1).max(1) as usize;
                self.delete_lines(n);
            }
            // Insert Characters (ICH)
            '@' => {
                let n = flat_params.first().copied().unwrap_or(1).max(1) as usize;
                self.insert_characters(n);
            }
            // Delete Characters (DCH)
            'P' => {
                let n = flat_params.first().copied().unwrap_or(1).max(1) as usize;
                self.delete_characters(n);
            }
            // Erase Characters (ECH)
            'X' => {
                let n = flat_params.first().copied().unwrap_or(1).max(1) as usize;
                self.erase_characters(n);
            }
            // Kitty keyboard protocol
            'u' => {
                self.handle_keyboard_protocol(&flat_params, intermediates);
            }
            // DECSCUSR - Set Cursor Style (CSI Ps SP q)
            'q' if intermediates == [b' '] => {
                let style = flat_params.first().copied().unwrap_or(0);
                self.cursor_shape = match style {
                    0 | 1 => CursorShape::BlinkingBlock,  // 0 = default (blinking block), 1 = blinking block
                    2 => CursorShape::SteadyBlock,
                    3 => CursorShape::BlinkingUnderline,
                    4 => CursorShape::SteadyUnderline,
                    5 => CursorShape::BlinkingBar,
                    6 => CursorShape::SteadyBar,
                    _ => CursorShape::BlinkingBlock,
                };
                log::debug!("DECSCUSR: cursor shape set to {:?}", self.cursor_shape);
            }
            // DEC Private Mode Set (CSI ? Ps h)
            'h' if intermediates == [b'?'] => {
                for &param in &flat_params {
                    match param {
                        25 => {
                            // DECTCEM - Show cursor
                            self.cursor_visible = true;
                            log::debug!("DECTCEM: cursor visible");
                        }
                        _ => {
                            log::debug!("Unhandled DEC private mode set: {}", param);
                        }
                    }
                }
            }
            // DEC Private Mode Reset (CSI ? Ps l)
            'l' if intermediates == [b'?'] => {
                for &param in &flat_params {
                    match param {
                        25 => {
                            // DECTCEM - Hide cursor
                            self.cursor_visible = false;
                            log::debug!("DECTCEM: cursor hidden");
                        }
                        _ => {
                            log::debug!("Unhandled DEC private mode reset: {}", param);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    fn esc_dispatch(&mut self, _intermediates: &[u8], _ignore: bool, _byte: u8) {}
}
