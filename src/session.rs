//! Terminal session management.
//!
//! A Session owns a PTY and its associated terminal state.

use crate::protocol::{CellColor, CursorInfo, CursorStyle, PaneSnapshot, RenderCell, SessionId};
use crate::pty::Pty;
use crate::terminal::{Cell, Color, ColorPalette, CursorShape, Terminal};
use vte::Parser;

/// A terminal session with its PTY and state.
pub struct Session {
    /// Unique session identifier.
    pub id: SessionId,
    /// The PTY connected to the shell.
    pub pty: Pty,
    /// Terminal state (grid, cursor, colors, etc.).
    pub terminal: Terminal,
    /// VTE parser for this session.
    parser: Parser,
    /// Whether the session has new output to send.
    pub dirty: bool,
}

impl Session {
    /// Creates a new session with the given dimensions.
    pub fn new(id: SessionId, cols: usize, rows: usize) -> Result<Self, crate::pty::PtyError> {
        let pty = Pty::spawn(None)?;
        pty.resize(cols as u16, rows as u16)?;
        
        let terminal = Terminal::new(cols, rows);
        
        Ok(Self {
            id,
            pty,
            terminal,
            parser: Parser::new(),
            dirty: true,
        })
    }
    
    /// Reads available data from the PTY and processes it.
    /// Returns the number of bytes read.
    pub fn poll(&mut self, buffer: &mut [u8]) -> Result<usize, crate::pty::PtyError> {
        let mut total = 0;
        
        loop {
            match self.pty.read(buffer) {
                Ok(0) => break, // WOULDBLOCK or no data
                Ok(n) => {
                    self.terminal.process(&buffer[..n], &mut self.parser);
                    total += n;
                    self.dirty = true;
                }
                Err(e) => return Err(e),
            }
        }
        
        Ok(total)
    }
    
    /// Writes data to the PTY (keyboard input).
    pub fn write(&self, data: &[u8]) -> Result<usize, crate::pty::PtyError> {
        self.pty.write(data)
    }
    
    /// Resizes the session.
    pub fn resize(&mut self, cols: usize, rows: usize) {
        self.terminal.resize(cols, rows);
        let _ = self.pty.resize(cols as u16, rows as u16);
        self.dirty = true;
    }
    
    /// Gets any pending response from the terminal (e.g., query responses).
    pub fn take_response(&mut self) -> Option<Vec<u8>> {
        self.terminal.take_response()
    }
    
    /// Converts internal Color to protocol CellColor, resolving indexed colors using the palette.
    fn convert_color(color: &Color, palette: &ColorPalette) -> CellColor {
        match color {
            Color::Default => CellColor::Default,
            Color::Rgb(r, g, b) => CellColor::Rgb(*r, *g, *b),
            Color::Indexed(i) => {
                // Resolve indexed colors to RGB using the palette
                let [r, g, b] = palette.colors[*i as usize];
                CellColor::Rgb(r, g, b)
            }
        }
    }
    
    /// Converts internal Cell to protocol RenderCell.
    fn convert_cell(cell: &Cell, palette: &ColorPalette) -> RenderCell {
        RenderCell {
            character: cell.character,
            fg_color: Self::convert_color(&cell.fg_color, palette),
            bg_color: Self::convert_color(&cell.bg_color, palette),
            bold: cell.bold,
            italic: cell.italic,
            underline: cell.underline,
        }
    }
    
    /// Creates a snapshot of the terminal state for sending to client.
    pub fn snapshot(&self, pane_id: u32) -> PaneSnapshot {
        let palette = &self.terminal.palette;
        
        // Use visible_rows() which accounts for scroll offset
        let cells: Vec<Vec<RenderCell>> = self.terminal.visible_rows()
            .iter()
            .map(|row| row.iter().map(|cell| Self::convert_cell(cell, palette)).collect())
            .collect();
        
        // Convert terminal cursor shape to protocol cursor style
        let cursor_style = match self.terminal.cursor_shape {
            CursorShape::BlinkingBlock | CursorShape::SteadyBlock => CursorStyle::Block,
            CursorShape::BlinkingUnderline | CursorShape::SteadyUnderline => CursorStyle::Underline,
            CursorShape::BlinkingBar | CursorShape::SteadyBar => CursorStyle::Bar,
        };
        
        // When scrolled back, adjust cursor row to account for offset
        // and hide cursor if it's not visible in the viewport
        let (cursor_row, cursor_visible) = if self.terminal.scroll_offset > 0 {
            // Cursor is at the "live" position, but we're viewing history
            // The cursor should appear scroll_offset rows lower, or be hidden
            let adjusted_row = self.terminal.cursor_row + self.terminal.scroll_offset;
            if adjusted_row >= self.terminal.rows {
                // Cursor is not in visible area
                (0, false)
            } else {
                (adjusted_row, self.terminal.cursor_visible)
            }
        } else {
            (self.terminal.cursor_row, self.terminal.cursor_visible)
        };
        
        PaneSnapshot {
            pane_id,
            cells,
            cursor: CursorInfo {
                col: self.terminal.cursor_col,
                row: cursor_row,
                visible: cursor_visible,
                style: cursor_style,
            },
            scroll_offset: self.terminal.scroll_offset,
            scrollback_len: self.terminal.scrollback.len(),
        }
    }
    
    /// Returns the raw file descriptor for polling.
    pub fn fd(&self) -> std::os::fd::BorrowedFd<'_> {
        self.pty.master_fd()
    }
    
    /// Marks the session as clean (updates sent).
    pub fn mark_clean(&mut self) {
        self.dirty = false;
        self.terminal.dirty = false;
    }
}
