//! Protocol messages for daemon/client communication.
//!
//! The daemon owns all terminal state (sessions, tabs, panes).
//! The client is a thin rendering layer that receives cell data and sends input.

use serde::{Deserialize, Serialize};

/// Unique identifier for a session (owns a PTY + terminal state).
pub type SessionId = u32;

/// Unique identifier for a pane within a tab.
pub type PaneId = u32;

/// Unique identifier for a tab.
pub type TabId = u32;

/// Direction for splitting a pane.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SplitDirection {
    /// Split horizontally (new pane below).
    Horizontal,
    /// Split vertically (new pane to the right).
    Vertical,
}

/// Direction for pane navigation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Direction {
    /// Navigate up.
    Up,
    /// Navigate down.
    Down,
    /// Navigate left.
    Left,
    /// Navigate right.
    Right,
}

/// Cursor shape styles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CursorStyle {
    /// Block cursor (like normal mode in vim).
    #[default]
    Block,
    /// Underline cursor.
    Underline,
    /// Bar/beam cursor (like insert mode in vim).
    Bar,
}

/// A single cell to be rendered by the client.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RenderCell {
    pub character: char,
    pub fg_color: CellColor,
    pub bg_color: CellColor,
    pub bold: bool,
    pub italic: bool,
    pub underline: bool,
}

/// Color representation for protocol messages.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq)]
pub enum CellColor {
    /// Default foreground or background.
    Default,
    /// RGB color.
    Rgb(u8, u8, u8),
    /// Indexed color (0-255).
    Indexed(u8),
}

/// A pane's layout within a tab.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PaneInfo {
    pub id: PaneId,
    pub session_id: SessionId,
    /// Position and size in cells (for future splits).
    /// For now, always (0, 0, cols, rows).
    pub x: usize,
    pub y: usize,
    pub cols: usize,
    pub rows: usize,
}

/// A tab containing one or more panes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TabInfo {
    pub id: TabId,
    /// Index of the active/focused pane within this tab.
    pub active_pane: usize,
    pub panes: Vec<PaneInfo>,
}

/// Cursor information for a pane.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CursorInfo {
    pub col: usize,
    pub row: usize,
    pub visible: bool,
    pub style: CursorStyle,
}

/// Full window state sent to client on connect.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WindowState {
    /// All tabs.
    pub tabs: Vec<TabInfo>,
    /// Index of the active tab.
    pub active_tab: usize,
    /// Terminal dimensions in cells.
    pub cols: usize,
    pub rows: usize,
}

/// Messages sent from client to daemon.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ClientMessage {
    /// Client is connecting and requests full state.
    /// Includes the client's window size.
    Hello { cols: usize, rows: usize },

    /// Keyboard input to send to the focused session.
    Input { data: Vec<u8> },

    /// Window was resized.
    Resize { cols: usize, rows: usize },

    /// Request to create a new tab.
    CreateTab,

    /// Request to close the current tab.
    CloseTab { tab_id: TabId },

    /// Switch to a different tab by ID.
    SwitchTab { tab_id: TabId },

    /// Switch to next tab.
    NextTab,

    /// Switch to previous tab.
    PrevTab,

    /// Switch to tab by index (0-based).
    SwitchTabIndex { index: usize },

    /// Split the current pane.
    SplitPane { direction: SplitDirection },

    /// Close the current pane (closes tab if last pane).
    ClosePane,

    /// Focus a pane in the given direction.
    FocusPane { direction: Direction },

    /// Scroll the viewport (for scrollback viewing).
    /// Positive delta scrolls up (into history), negative scrolls down (toward live).
    Scroll { pane_id: PaneId, delta: i32 },

    /// Client is disconnecting gracefully.
    Goodbye,
}

/// Messages sent from daemon to client.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum DaemonMessage {
    /// Full state snapshot (sent on connect and major changes).
    FullState {
        window: WindowState,
        /// Cell data for all visible panes, keyed by pane ID.
        /// Each pane has rows x cols cells.
        panes: Vec<PaneSnapshot>,
    },

    /// Incremental update for a single pane.
    PaneUpdate {
        pane_id: PaneId,
        cells: Vec<Vec<RenderCell>>,
        cursor: CursorInfo,
    },

    /// Active tab changed.
    TabChanged { active_tab: usize },

    /// A tab was created.
    TabCreated { tab: TabInfo },

    /// A tab was closed.
    TabClosed { tab_id: TabId },

    /// A pane was created (split).
    PaneCreated { tab_id: TabId, pane: PaneInfo },

    /// A pane was closed.
    PaneClosed { tab_id: TabId, pane_id: PaneId },

    /// Active pane changed within a tab.
    PaneFocused { tab_id: TabId, active_pane: usize },

    /// Daemon is shutting down.
    Shutdown,
}

/// Snapshot of a pane's content.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PaneSnapshot {
    pub pane_id: PaneId,
    pub cells: Vec<Vec<RenderCell>>,
    pub cursor: CursorInfo,
    /// Current scroll offset (0 = live terminal, >0 = viewing scrollback).
    pub scroll_offset: usize,
    /// Total lines in scrollback buffer.
    pub scrollback_len: usize,
}

/// Wire format for messages: length-prefixed JSON.
/// Format: [4 bytes little-endian length][JSON payload]
pub mod wire {
    use super::*;
    use std::io::{self, Read, Write};

    /// Write a message to a writer with length prefix.
    pub fn write_message<W: Write, M: Serialize>(writer: &mut W, msg: &M) -> io::Result<()> {
        let json = serde_json::to_vec(msg).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let len = json.len() as u32;
        writer.write_all(&len.to_le_bytes())?;
        writer.write_all(&json)?;
        writer.flush()?;
        Ok(())
    }

    /// Read a message from a reader with length prefix.
    pub fn read_message<R: Read, M: for<'de> Deserialize<'de>>(reader: &mut R) -> io::Result<M> {
        let mut len_buf = [0u8; 4];
        reader.read_exact(&mut len_buf)?;
        let len = u32::from_le_bytes(len_buf) as usize;

        // Sanity check to prevent huge allocations
        if len > 64 * 1024 * 1024 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "message too large",
            ));
        }

        let mut buf = vec![0u8; len];
        reader.read_exact(&mut buf)?;
        
        serde_json::from_slice(&buf).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}
