//! ZTerm daemon - manages terminal sessions and communicates with clients.

use crate::protocol::{ClientMessage, DaemonMessage, PaneSnapshot};
use crate::window_state::WindowStateManager;
use polling::{Event, Events, Poller};
use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::os::fd::AsRawFd;
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::PathBuf;
use std::time::Duration;

/// Get the socket path for the daemon.
pub fn socket_path() -> PathBuf {
    let runtime_dir = std::env::var("XDG_RUNTIME_DIR")
        .unwrap_or_else(|_| format!("/tmp/zterm-{}", unsafe { libc::getuid() }));
    PathBuf::from(runtime_dir).join("zterm.sock")
}

/// Event keys for the poller.
const LISTENER_KEY: usize = 0;
const CLIENT_KEY_BASE: usize = 1000;
const SESSION_KEY_BASE: usize = 2000;

/// A connected client.
struct Client {
    stream: UnixStream,
}

impl Client {
    fn new(stream: UnixStream) -> io::Result<Self> {
        // Set to non-blocking mode for use with poller
        stream.set_nonblocking(true)?;
        Ok(Self { stream })
    }
    
    fn send(&mut self, msg: &DaemonMessage) -> io::Result<()> {
        let json = serde_json::to_vec(msg)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        let len = json.len() as u32;
        
        // Temporarily set blocking mode for sends to ensure complete writes
        self.stream.set_nonblocking(false)?;
        self.stream.set_write_timeout(Some(Duration::from_secs(5)))?;
        
        let result = (|| {
            self.stream.write_all(&len.to_le_bytes())?;
            self.stream.write_all(&json)?;
            self.stream.flush()
        })();
        
        // Restore non-blocking mode
        self.stream.set_nonblocking(true)?;
        
        result
    }
    
    /// Tries to receive a message. Returns:
    /// - Ok(Some(msg)) if a message was received
    /// - Ok(None) if no data available (would block)
    /// - Err with ConnectionReset if client disconnected
    /// - Err for other errors
    fn try_recv(&mut self) -> io::Result<Option<ClientMessage>> {
        // Try to read length prefix (non-blocking)
        let mut len_buf = [0u8; 4];
        match self.stream.read(&mut len_buf) {
            Ok(0) => {
                // EOF - client disconnected
                return Err(io::Error::new(io::ErrorKind::ConnectionReset, "client disconnected"));
            }
            Ok(n) if n < 4 => {
                // Partial read - need to read more (shouldn't happen often with small prefix)
                // For now, treat as would-block and let next poll handle it
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
        
        // Read the message body - since we're non-blocking, we need to handle partial reads
        // For simplicity, temporarily set blocking with timeout for the body
        self.stream.set_nonblocking(false)?;
        self.stream.set_read_timeout(Some(Duration::from_secs(5)))?;
        
        let mut buf = vec![0u8; len];
        let result = self.stream.read_exact(&mut buf);
        
        // Restore non-blocking mode
        self.stream.set_nonblocking(true)?;
        
        result?;
        
        let msg: ClientMessage = serde_json::from_slice(&buf)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
        
        Ok(Some(msg))
    }
}

/// The daemon server.
pub struct Daemon {
    listener: UnixListener,
    poller: Poller,
    state: WindowStateManager,
    clients: HashMap<usize, Client>,
    next_client_id: usize,
    read_buffer: Vec<u8>,
}

impl Daemon {
    /// Creates and starts a new daemon.
    pub fn new() -> io::Result<Self> {
        let socket = socket_path();
        
        // Remove old socket if it exists
        let _ = std::fs::remove_file(&socket);
        
        // Create parent directory if needed
        if let Some(parent) = socket.parent() {
            std::fs::create_dir_all(parent)?;
        }
        
        let listener = UnixListener::bind(&socket)?;
        listener.set_nonblocking(true)?;
        
        log::info!("Daemon listening on {:?}", socket);
        
        let poller = Poller::new()?;
        
        // Register listener for new connections
        unsafe {
            poller.add(listener.as_raw_fd(), Event::readable(LISTENER_KEY))?;
        }
        
        // Create initial window state (will create session when client connects with size)
        let state = WindowStateManager::new(80, 24); // Default size, will be updated
        
        Ok(Self {
            listener,
            poller,
            state,
            clients: HashMap::new(),
            next_client_id: 0,
            read_buffer: vec![0u8; 65536],
        })
    }
    
    /// Runs the daemon main loop.
    pub fn run(&mut self) -> io::Result<()> {
        let mut events = Events::new();
        
        loop {
            events.clear();
            
            // Poll with a longer timeout - we'll wake up when there's actual I/O
            // Use None for infinite wait, or a reasonable timeout for periodic checks
            self.poller.wait(&mut events, Some(Duration::from_millis(100)))?;
            
            // Track which sessions had output so we only read from those
            let mut sessions_with_output: Vec<u32> = Vec::new();
            
            for event in events.iter() {
                match event.key {
                    LISTENER_KEY => {
                        self.accept_client()?;
                    }
                    key if key >= CLIENT_KEY_BASE && key < SESSION_KEY_BASE => {
                        let client_id = key - CLIENT_KEY_BASE;
                        if let Err(e) = self.handle_client(client_id) {
                            log::warn!("Client {} error: {}", client_id, e);
                            self.remove_client(client_id);
                        }
                    }
                    key if key >= SESSION_KEY_BASE => {
                        let session_id = (key - SESSION_KEY_BASE) as u32;
                        sessions_with_output.push(session_id);
                        // Don't re-register here - we'll do it AFTER reading from the session
                    }
                    _ => {}
                }
            }
            
            // Read from sessions that have data, THEN re-register for polling
            // This order is critical: we must fully drain the buffer before re-registering
            // to avoid busy-looping with level-triggered polling
            for session_id in sessions_with_output {
                if let Some(session) = self.state.sessions.get_mut(&session_id) {
                    // First, drain all available data from the PTY
                    let _ = session.poll(&mut self.read_buffer);
                    
                    // Now re-register for polling (after buffer is drained)
                    let _ = self.poller.modify(
                        session.fd(),
                        Event::readable(SESSION_KEY_BASE + session_id as usize),
                    );
                }
            }
            
            // Send updates to clients if any session is dirty
            if self.state.any_dirty() {
                self.broadcast_updates()?;
                self.state.mark_all_clean();
            }
            
            // Re-register listener
            self.poller.modify(&self.listener, Event::readable(LISTENER_KEY))?;
            
            // If no tabs exist, create one automatically
            if self.state.tabs.is_empty() {
                log::info!("No tabs open, creating new tab");
                if let Err(e) = self.state.create_initial_tab() {
                    log::error!("Failed to create tab: {}", e);
                } else {
                    // Register session FD for polling
                    for session in self.state.sessions.values() {
                        unsafe {
                            let _ = self.poller.add(
                                session.fd().as_raw_fd(),
                                Event::readable(SESSION_KEY_BASE + session.id as usize),
                            );
                        }
                    }
                }
            }
        }
        
        // Notify clients of shutdown
        for client in self.clients.values_mut() {
            let _ = client.send(&DaemonMessage::Shutdown);
        }
        
        // Clean up socket
        let _ = std::fs::remove_file(socket_path());
        
        Ok(())
    }
    
    fn accept_client(&mut self) -> io::Result<()> {
        match self.listener.accept() {
            Ok((stream, _addr)) => {
                let client_id = self.next_client_id;
                self.next_client_id += 1;
                
                log::info!("Client {} connected", client_id);
                
                let client = Client::new(stream)?;
                
                // Register client socket for reading
                unsafe {
                    self.poller.add(
                        client.stream.as_raw_fd(),
                        Event::readable(CLIENT_KEY_BASE + client_id),
                    )?;
                }
                
                self.clients.insert(client_id, client);
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => {}
            Err(e) => return Err(e),
        }
        Ok(())
    }
    
    fn handle_client(&mut self, client_id: usize) -> io::Result<()> {
        // First, collect all messages from this client
        // We read all available messages BEFORE re-registering to avoid busy-looping
        let messages: Vec<ClientMessage> = {
            let client = self.clients.get_mut(&client_id)
                .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "client not found"))?;
            
            let mut msgs = Vec::new();
            loop {
                match client.try_recv() {
                    Ok(Some(msg)) => msgs.push(msg),
                    Ok(None) => break,
                    Err(e) => return Err(e),
                }
            }
            
            // Re-register for more events AFTER draining the socket
            self.poller.modify(&client.stream, Event::readable(CLIENT_KEY_BASE + client_id))?;
            
            msgs
        };
        
        // Now process messages without holding client borrow
        for msg in messages {
            match msg {
                ClientMessage::Hello { cols, rows } => {
                    log::info!("Client {} says hello with size {}x{}", client_id, cols, rows);
                    
                    // Update dimensions
                    self.state.resize(cols, rows);
                    
                    // Create initial tab if none exists
                    if self.state.tabs.is_empty() {
                        self.state.create_initial_tab()
                            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
                        
                        // Register session FD for polling
                        for session in self.state.sessions.values() {
                            unsafe {
                                self.poller.add(
                                    session.fd().as_raw_fd(),
                                    Event::readable(SESSION_KEY_BASE + session.id as usize),
                                )?;
                            }
                        }
                    }
                    
                    // Send full state to client
                    self.send_full_state(client_id)?;
                }
                
                ClientMessage::Input { data } => {
                    if let Some(session) = self.state.focused_session_mut() {
                        let _ = session.write(&data);
                        
                        // Send any terminal responses back to PTY
                        if let Some(response) = session.take_response() {
                            let _ = session.write(&response);
                        }
                    }
                }
                
                ClientMessage::Resize { cols, rows } => {
                    log::debug!("Client {} resize to {}x{}", client_id, cols, rows);
                    self.state.resize(cols, rows);
                    
                    // Send updated state
                    self.broadcast_updates()?;
                }
                
                ClientMessage::CreateTab => {
                    match self.state.create_tab() {
                        Ok(tab_id) => {
                            log::info!("Created tab {}", tab_id);
                            
                            // Register new session for polling
                            if let Some(tab) = self.state.tabs.iter().find(|t| t.id == tab_id) {
                                if let Some(pane) = tab.panes.first() {
                                    if let Some(session) = self.state.sessions.get(&pane.session_id) {
                                        unsafe {
                                            let _ = self.poller.add(
                                                session.fd().as_raw_fd(),
                                                Event::readable(SESSION_KEY_BASE + session.id as usize),
                                            );
                                        }
                                    }
                                }
                            }
                            
                            // Broadcast full state update
                            self.broadcast_full_state()?;
                        }
                        Err(e) => {
                            log::error!("Failed to create tab: {}", e);
                        }
                    }
                }
                
                ClientMessage::CloseTab { tab_id } => {
                    if self.state.close_tab(tab_id) {
                        log::info!("Closed tab {}", tab_id);
                        self.broadcast_full_state()?;
                    }
                }
                
                ClientMessage::SwitchTab { tab_id } => {
                    if self.state.switch_tab(tab_id) {
                        log::debug!("Switched to tab {}", tab_id);
                        
                        // Send tab changed message
                        let active_tab = self.state.active_tab;
                        for client in self.clients.values_mut() {
                            let _ = client.send(&DaemonMessage::TabChanged { active_tab });
                        }
                        
                        // Send current pane content
                        self.broadcast_updates()?;
                    }
                }
                
                ClientMessage::NextTab => {
                    if self.state.next_tab() {
                        log::debug!("Switched to next tab");
                        let active_tab = self.state.active_tab;
                        for client in self.clients.values_mut() {
                            let _ = client.send(&DaemonMessage::TabChanged { active_tab });
                        }
                        self.broadcast_updates()?;
                    }
                }
                
                ClientMessage::PrevTab => {
                    if self.state.prev_tab() {
                        log::debug!("Switched to previous tab");
                        let active_tab = self.state.active_tab;
                        for client in self.clients.values_mut() {
                            let _ = client.send(&DaemonMessage::TabChanged { active_tab });
                        }
                        self.broadcast_updates()?;
                    }
                }
                
                ClientMessage::SwitchTabIndex { index } => {
                    if self.state.switch_tab_index(index) {
                        log::debug!("Switched to tab index {}", index);
                        let active_tab = self.state.active_tab;
                        for client in self.clients.values_mut() {
                            let _ = client.send(&DaemonMessage::TabChanged { active_tab });
                        }
                        self.broadcast_updates()?;
                    }
                }
                
                ClientMessage::SplitPane { direction } => {
                    match self.state.split_pane(direction) {
                        Ok((tab_id, pane_info)) => {
                            log::info!("Split pane in tab {}, new pane {}", tab_id, pane_info.id);
                            
                            // Register new session for polling
                            if let Some(session) = self.state.sessions.get(&pane_info.session_id) {
                                unsafe {
                                    let _ = self.poller.add(
                                        session.fd().as_raw_fd(),
                                        Event::readable(SESSION_KEY_BASE + session.id as usize),
                                    );
                                }
                            }
                            
                            // Broadcast full state update
                            self.broadcast_full_state()?;
                        }
                        Err(e) => {
                            log::error!("Failed to split pane: {}", e);
                        }
                    }
                }
                
                ClientMessage::ClosePane => {
                    if let Some((tab_id, pane_id, tab_closed)) = self.state.close_pane() {
                        if tab_closed {
                            log::info!("Closed pane {} (last pane, closed tab {})", pane_id, tab_id);
                        } else {
                            log::info!("Closed pane {} in tab {}", pane_id, tab_id);
                        }
                        self.broadcast_full_state()?;
                    }
                }
                
                ClientMessage::FocusPane { direction } => {
                    if let Some((tab_id, active_pane)) = self.state.focus_pane_direction(direction) {
                        log::debug!("Focused pane in direction {:?}", direction);
                        for client in self.clients.values_mut() {
                            let _ = client.send(&DaemonMessage::PaneFocused { tab_id, active_pane });
                        }
                        self.broadcast_updates()?;
                    }
                }
                
                ClientMessage::Scroll { pane_id, delta } => {
                    // Find the session ID for this pane
                    let session_id = self.state.active_tab()
                        .and_then(|tab| tab.panes.iter().find(|p| p.id == pane_id))
                        .map(|pane| pane.session_id);
                    
                    // Now adjust scroll on the session
                    if let Some(session_id) = session_id {
                        if let Some(session) = self.state.sessions.get_mut(&session_id) {
                            if delta > 0 {
                                session.terminal.scroll_viewport_up(delta as usize);
                            } else if delta < 0 {
                                session.terminal.scroll_viewport_down((-delta) as usize);
                            }
                            // Mark session dirty to trigger update
                            session.dirty = true;
                        }
                    }
                    self.broadcast_updates()?;
                }
                
                ClientMessage::Goodbye => {
                    log::info!("Client {} disconnecting", client_id);
                    return Err(io::Error::new(io::ErrorKind::ConnectionReset, "goodbye"));
                }
            }
        }
        
        Ok(())
    }
    
    fn remove_client(&mut self, client_id: usize) {
        if let Some(client) = self.clients.remove(&client_id) {
            let _ = self.poller.delete(&client.stream);
            log::info!("Client {} removed", client_id);
        }
    }
    
    
    fn send_full_state(&mut self, client_id: usize) -> io::Result<()> {
        let window = self.state.to_protocol();
        
        // Collect snapshots for all visible panes in the active tab
        let panes: Vec<PaneSnapshot> = if let Some(tab) = self.state.active_tab() {
            tab.panes.iter().filter_map(|pane| {
                self.state.sessions.get(&pane.session_id)
                    .map(|session| session.snapshot(pane.id))
            }).collect()
        } else {
            Vec::new()
        };
        
        let msg = DaemonMessage::FullState { window, panes };
        
        if let Some(client) = self.clients.get_mut(&client_id) {
            client.send(&msg)?;
        }
        
        Ok(())
    }
    
    fn broadcast_full_state(&mut self) -> io::Result<()> {
        let window = self.state.to_protocol();
        
        let panes: Vec<PaneSnapshot> = if let Some(tab) = self.state.active_tab() {
            tab.panes.iter().filter_map(|pane| {
                self.state.sessions.get(&pane.session_id)
                    .map(|session| session.snapshot(pane.id))
            }).collect()
        } else {
            Vec::new()
        };
        
        let msg = DaemonMessage::FullState { window, panes };
        
        for client in self.clients.values_mut() {
            let _ = client.send(&msg);
        }
        
        Ok(())
    }
    
    fn broadcast_updates(&mut self) -> io::Result<()> {
        // For now, send full state on updates
        // TODO: Send incremental PaneUpdate messages instead
        self.broadcast_full_state()
    }
}

/// Check if the daemon is running.
pub fn is_running() -> bool {
    let socket = socket_path();
    if !socket.exists() {
        return false;
    }
    
    // Try to connect
    match UnixStream::connect(&socket) {
        Ok(_) => true,
        Err(_) => {
            // Socket exists but can't connect - stale socket
            let _ = std::fs::remove_file(&socket);
            false
        }
    }
}

/// Start the daemon in the background.
pub fn start_daemon() -> io::Result<()> {
    use std::process::Command;
    
    // Get path to current executable
    let exe = std::env::current_exe()?;
    let daemon_exe = exe.with_file_name("ztermd");
    
    if !daemon_exe.exists() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("daemon executable not found: {:?}", daemon_exe),
        ));
    }
    
    // Spawn daemon in background
    Command::new(&daemon_exe)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()?;
    
    // Wait a bit for it to start
    std::thread::sleep(Duration::from_millis(100));
    
    Ok(())
}
