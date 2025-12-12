//! ZTerm client - connects to daemon and handles rendering.

use crate::daemon::{is_running, socket_path, start_daemon};
use crate::protocol::{ClientMessage, DaemonMessage, Direction, PaneSnapshot, SplitDirection, WindowState};
use std::io::{self, Read, Write};
use std::os::unix::net::UnixStream;
use std::time::Duration;

/// Client connection to the daemon.
pub struct DaemonClient {
    stream: UnixStream,
    /// Current window state from daemon.
    pub window: Option<WindowState>,
    /// Current pane snapshots.
    pub panes: Vec<PaneSnapshot>,
}

impl DaemonClient {
    /// Connects to the daemon, starting it if necessary.
    pub fn connect() -> io::Result<Self> {
        // Start daemon if not running
        if !is_running() {
            log::info!("Starting daemon...");
            start_daemon()?;
            
            // Wait for daemon to start
            for _ in 0..50 {
                if is_running() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            
            if !is_running() {
                return Err(io::Error::new(
                    io::ErrorKind::ConnectionRefused,
                    "failed to start daemon",
                ));
            }
        }
        
        let socket = socket_path();
        log::info!("Connecting to daemon at {:?}", socket);
        
        let stream = UnixStream::connect(&socket)?;
        // Keep blocking mode for initial handshake
        stream.set_nonblocking(false)?;
        
        Ok(Self {
            stream,
            window: None,
            panes: Vec::new(),
        })
    }
    
    /// Sends hello message with initial window size.
    pub fn hello(&mut self, cols: usize, rows: usize) -> io::Result<()> {
        self.send(&ClientMessage::Hello { cols, rows })
    }
    
    /// Sets the socket to non-blocking mode for use after initial handshake.
    pub fn set_nonblocking(&mut self) -> io::Result<()> {
        self.stream.set_nonblocking(true)
    }
    
    /// Sends keyboard input.
    pub fn send_input(&mut self, data: Vec<u8>) -> io::Result<()> {
        self.send(&ClientMessage::Input { data })
    }
    
    /// Sends resize notification.
    pub fn send_resize(&mut self, cols: usize, rows: usize) -> io::Result<()> {
        self.send(&ClientMessage::Resize { cols, rows })
    }
    
    /// Requests creation of a new tab.
    pub fn create_tab(&mut self) -> io::Result<()> {
        self.send(&ClientMessage::CreateTab)
    }
    
    /// Requests closing the current tab.
    pub fn close_tab(&mut self, tab_id: u32) -> io::Result<()> {
        self.send(&ClientMessage::CloseTab { tab_id })
    }
    
    /// Requests switching to the next tab.
    pub fn next_tab(&mut self) -> io::Result<()> {
        self.send(&ClientMessage::NextTab)
    }
    
    /// Requests switching to the previous tab.
    pub fn prev_tab(&mut self) -> io::Result<()> {
        self.send(&ClientMessage::PrevTab)
    }
    
    /// Requests switching to a tab by index (0-based).
    pub fn switch_tab_index(&mut self, index: usize) -> io::Result<()> {
        self.send(&ClientMessage::SwitchTabIndex { index })
    }
    
    /// Requests splitting the current pane horizontally (new pane below).
    pub fn split_horizontal(&mut self) -> io::Result<()> {
        self.send(&ClientMessage::SplitPane { direction: SplitDirection::Horizontal })
    }
    
    /// Requests splitting the current pane vertically (new pane to the right).
    pub fn split_vertical(&mut self) -> io::Result<()> {
        self.send(&ClientMessage::SplitPane { direction: SplitDirection::Vertical })
    }
    
    /// Requests closing the current pane (closes tab if last pane).
    pub fn close_pane(&mut self) -> io::Result<()> {
        self.send(&ClientMessage::ClosePane)
    }
    
    /// Requests focusing a pane in the given direction.
    pub fn focus_pane(&mut self, direction: Direction) -> io::Result<()> {
        self.send(&ClientMessage::FocusPane { direction })
    }
    
    /// Sends a message to the daemon.
    pub fn send(&mut self, msg: &ClientMessage) -> io::Result<()> {
        let json = serde_json::to_vec(msg)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let len = json.len() as u32;
        self.stream.write_all(&len.to_le_bytes())?;
        self.stream.write_all(&json)?;
        self.stream.flush()?;
        Ok(())
    }
    
    /// Tries to receive a message (non-blocking).
    /// Socket must be set to non-blocking mode first.
    pub fn try_recv(&mut self) -> io::Result<Option<DaemonMessage>> {
        // Try to read the length prefix (non-blocking)
        let mut len_buf = [0u8; 4];
        match self.stream.read(&mut len_buf) {
            Ok(0) => {
                // EOF - daemon disconnected
                return Err(io::Error::new(io::ErrorKind::ConnectionReset, "daemon disconnected"));
            }
            Ok(n) if n < 4 => {
                // Partial read - shouldn't happen often with 4-byte prefix
                // For now, treat as no data
                return Ok(None);
            }
            Ok(_) => {} // Got all 4 bytes
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {
                return Ok(None);
            }
            Err(e) => return Err(e),
        }
        
        let len = u32::from_le_bytes(len_buf) as usize;
        
        // Sanity check
        if len > 64 * 1024 * 1024 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "message too large"));
        }
        
        // For the message body, temporarily use blocking mode with timeout
        // since we know the data should be available
        self.stream.set_nonblocking(false)?;
        self.stream.set_read_timeout(Some(Duration::from_secs(5)))?;
        
        let mut buf = vec![0u8; len];
        let result = self.stream.read_exact(&mut buf);
        
        // Restore non-blocking mode
        self.stream.set_nonblocking(true)?;
        
        result?;
        
        let msg: DaemonMessage = serde_json::from_slice(&buf)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        
        self.handle_message(&msg);
        Ok(Some(msg))
    }
    
    /// Receives a message (blocking).
    pub fn recv(&mut self) -> io::Result<DaemonMessage> {
        // Use a long timeout for blocking reads
        self.stream.set_read_timeout(Some(Duration::from_secs(30)))?;
        
        // Read length prefix
        let mut len_buf = [0u8; 4];
        self.stream.read_exact(&mut len_buf)?;
        let len = u32::from_le_bytes(len_buf) as usize;
        
        // Sanity check
        if len > 64 * 1024 * 1024 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "message too large"));
        }
        
        // Read message body
        let mut buf = vec![0u8; len];
        self.stream.read_exact(&mut buf)?;
        
        let msg: DaemonMessage = serde_json::from_slice(&buf)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        
        self.handle_message(&msg);
        Ok(msg)
    }
    
    /// Handles a received message by updating local state.
    fn handle_message(&mut self, msg: &DaemonMessage) {
        match msg {
            DaemonMessage::FullState { window, panes } => {
                self.window = Some(window.clone());
                self.panes = panes.clone();
            }
            DaemonMessage::PaneUpdate { pane_id, cells, cursor } => {
                // Update existing pane or add new
                if let Some(pane) = self.panes.iter_mut().find(|p| p.pane_id == *pane_id) {
                    pane.cells = cells.clone();
                    pane.cursor = cursor.clone();
                } else {
                    self.panes.push(PaneSnapshot {
                        pane_id: *pane_id,
                        cells: cells.clone(),
                        cursor: cursor.clone(),
                        scroll_offset: 0,
                        scrollback_len: 0,
                    });
                }
            }
            DaemonMessage::TabChanged { active_tab } => {
                if let Some(ref mut window) = self.window {
                    window.active_tab = *active_tab;
                }
            }
            DaemonMessage::TabCreated { tab } => {
                if let Some(ref mut window) = self.window {
                    window.tabs.push(tab.clone());
                }
            }
            DaemonMessage::TabClosed { tab_id } => {
                if let Some(ref mut window) = self.window {
                    window.tabs.retain(|t| t.id != *tab_id);
                }
            }
            DaemonMessage::PaneCreated { tab_id, pane } => {
                if let Some(ref mut window) = self.window {
                    if let Some(tab) = window.tabs.iter_mut().find(|t| t.id == *tab_id) {
                        tab.panes.push(pane.clone());
                    }
                }
            }
            DaemonMessage::PaneClosed { tab_id, pane_id } => {
                if let Some(ref mut window) = self.window {
                    if let Some(tab) = window.tabs.iter_mut().find(|t| t.id == *tab_id) {
                        tab.panes.retain(|p| p.id != *pane_id);
                    }
                }
                // Also remove from pane snapshots
                self.panes.retain(|p| p.pane_id != *pane_id);
            }
            DaemonMessage::PaneFocused { tab_id, active_pane } => {
                if let Some(ref mut window) = self.window {
                    if let Some(tab) = window.tabs.iter_mut().find(|t| t.id == *tab_id) {
                        tab.active_pane = *active_pane;
                    }
                }
            }
            DaemonMessage::Shutdown => {
                log::info!("Daemon shutting down");
            }
        }
    }
    
    /// Gets the active pane snapshot.
    pub fn active_pane(&self) -> Option<&PaneSnapshot> {
        let window = self.window.as_ref()?;
        let tab = window.tabs.get(window.active_tab)?;
        let pane_info = tab.panes.get(tab.active_pane)?;
        self.panes.iter().find(|p| p.pane_id == pane_info.id)
    }
    
    /// Returns the file descriptor for polling.
    pub fn as_fd(&self) -> std::os::fd::BorrowedFd<'_> {
        use std::os::fd::AsFd;
        self.stream.as_fd()
    }
    
    /// Returns the raw file descriptor for initial poll registration.
    pub fn as_raw_fd(&self) -> std::os::fd::RawFd {
        use std::os::fd::AsRawFd;
        self.stream.as_raw_fd()
    }
    
    /// Sends goodbye message before disconnecting.
    pub fn goodbye(&mut self) -> io::Result<()> {
        self.send(&ClientMessage::Goodbye)
    }
}

impl Drop for DaemonClient {
    fn drop(&mut self) {
        let _ = self.goodbye();
    }
}
