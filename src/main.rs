//! ZTerm - GPU-accelerated terminal emulator.
//! 
//! Single-process architecture: owns PTY, terminal state, and rendering.
//! Supports window close/reopen without losing terminal state.

use zterm::config::{Action, Config};
use zterm::keyboard::{FunctionalKey, KeyEncoder, KeyEventType, KeyboardState, Modifiers};
use zterm::pty::Pty;
use zterm::renderer::{EdgeGlow, PaneRenderInfo, Renderer};
use zterm::terminal::{Direction, Terminal, TerminalCommand, MouseTrackingMode};

use std::collections::HashMap;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use polling::{Event, Events, Poller};
use winit::application::ApplicationHandler;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, KeyEvent, MouseButton, Modifiers as WinitModifiers, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{Key, NamedKey};
use winit::platform::wayland::EventLoopBuilderExtWayland;
use winit::window::{Window, WindowId};

/// Kitty-style shared buffer for PTY I/O using double-buffering.
/// 
/// Uses two buffers that swap roles:
/// - I/O thread writes to the "write" buffer
/// - Main thread parses from the "read" buffer  
/// - On `swap()`, the buffers exchange roles
/// 
/// This gives us:
/// - Zero-copy parsing (main thread reads directly from buffer)
/// - No lock contention during parsing (each thread has its own buffer)
/// - No memmove needed
const PTY_BUF_SIZE: usize = 4 * 1024 * 1024; // 4MB like Kitty

struct SharedPtyBuffer {
    inner: Mutex<DoubleBuffer>,
}

struct DoubleBuffer {
    /// Two buffers that swap roles
    bufs: [Vec<u8>; 2],
    /// Which buffer the I/O thread writes to (0 or 1)
    write_idx: usize,
    /// How many bytes are pending in the write buffer
    write_len: usize,
}

impl SharedPtyBuffer {
    fn new() -> Self {
        Self {
            inner: Mutex::new(DoubleBuffer {
                bufs: [vec![0u8; PTY_BUF_SIZE], vec![0u8; PTY_BUF_SIZE]],
                write_idx: 0,
                write_len: 0,
            }),
        }
    }
    
    /// Read from PTY fd into the write buffer. Called by I/O thread.
    /// Returns number of bytes read, 0 if no space/would block, -1 on error.
    fn read_from_fd(&self, fd: i32) -> isize {
        let mut inner = self.inner.lock().unwrap();
        
        let available = PTY_BUF_SIZE.saturating_sub(inner.write_len);
        if available == 0 {
            return 0; // Buffer full, need swap
        }
        
        let write_idx = inner.write_idx;
        let write_len = inner.write_len;
        let buf_ptr = unsafe { inner.bufs[write_idx].as_mut_ptr().add(write_len) };
        
        let result = unsafe { 
            libc::read(fd, buf_ptr as *mut libc::c_void, available) 
        };
        
        if result > 0 {
            inner.write_len += result as usize;
        }
        result
    }
    
    /// Check if there's space in the write buffer.
    fn has_space(&self) -> bool {
        let inner = self.inner.lock().unwrap();
        inner.write_len < PTY_BUF_SIZE
    }
    
    /// Swap buffers and return data to parse. Called by main thread.
    /// The I/O thread will start writing to the other buffer.
    fn take_pending(&self) -> Vec<u8> {
        let mut inner = self.inner.lock().unwrap();
        
        if inner.write_len == 0 {
            return Vec::new(); // Nothing new to parse
        }
        
        // Swap: the write buffer becomes the read buffer
        let read_idx = inner.write_idx;
        let read_len = inner.write_len;
        
        // Switch I/O thread to the other buffer
        inner.write_idx = 1 - inner.write_idx;
        inner.write_len = 0;
        
        // Return a copy of the data to parse
        // (We have to copy because we can't return a reference with the mutex)
        inner.bufs[read_idx][..read_len].to_vec()
    }
}

/// Unique identifier for a pane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct PaneId(u64);

impl PaneId {
    fn new() -> Self {
        use std::sync::atomic::AtomicU64;
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        Self(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// A single pane containing a terminal and its PTY.
struct Pane {
    /// Unique identifier for this pane.
    id: PaneId,
    /// Terminal state (grid, cursor, scrollback, etc.).
    terminal: Terminal,
    /// PTY connection to the shell.
    pty: Pty,
    /// Raw file descriptor for the PTY (for polling).
    pty_fd: i32,
    /// Shared buffer for this pane's PTY I/O.
    pty_buffer: Arc<SharedPtyBuffer>,
    /// Selection state for this pane.
    selection: Option<Selection>,
    /// Whether we're currently selecting in this pane.
    is_selecting: bool,
    /// Last scrollback length for tracking changes.
    last_scrollback_len: u32,
    /// When the focus animation started (for smooth fade).
    focus_animation_start: std::time::Instant,
    /// Whether this pane was focused before the current animation.
    was_focused: bool,
}

impl Pane {
    /// Create a new pane with its own terminal and PTY.
    fn new(cols: usize, rows: usize, scrollback_lines: usize) -> Result<Self, String> {
        let terminal = Terminal::new(cols, rows, scrollback_lines);
        let pty = Pty::spawn(None).map_err(|e| format!("Failed to spawn PTY: {}", e))?;
        
        // Set terminal size
        if let Err(e) = pty.resize(cols as u16, rows as u16) {
            log::warn!("Failed to set initial PTY size: {}", e);
        }
        
        let pty_fd = pty.as_raw_fd();
        
        Ok(Self {
            id: PaneId::new(),
            terminal,
            pty,
            pty_fd,
            pty_buffer: Arc::new(SharedPtyBuffer::new()),
            selection: None,
            is_selecting: false,
            last_scrollback_len: 0,
            focus_animation_start: std::time::Instant::now(),
            was_focused: true, // New panes start as focused
        })
    }
    
    /// Resize the terminal and PTY.
    fn resize(&mut self, cols: usize, rows: usize) {
        self.terminal.resize(cols, rows);
        if let Err(e) = self.pty.resize(cols as u16, rows as u16) {
            log::warn!("Failed to resize PTY: {}", e);
        }
    }
    
    /// Write data to the PTY.
    fn write_to_pty(&mut self, data: &[u8]) {
        if let Err(e) = self.pty.write(data) {
            log::warn!("Failed to write to PTY: {}", e);
        }
    }
    
    /// Check if the shell has exited.
    fn child_exited(&self) -> bool {
        self.pty.child_exited()
    }
    
    /// Check if the foreground process matches any of the given program names.
    /// Used for pass-through keybindings (e.g., passing Alt+Arrow to Neovim).
    fn foreground_matches(&self, programs: &[String]) -> bool {
        if programs.is_empty() {
            return false;
        }
        if let Some(fg_name) = self.pty.foreground_process_name() {
            programs.iter().any(|p| p == &fg_name)
        } else {
            false
        }
    }
    
    /// Calculate the current dim factor based on animation progress.
    /// Returns a value between `inactive_dim` (for unfocused) and 1.0 (for focused).
    fn calculate_dim_factor(&mut self, is_focused: bool, fade_duration_ms: u64, inactive_dim: f32) -> f32 {
        // Detect focus change
        if is_focused != self.was_focused {
            self.focus_animation_start = std::time::Instant::now();
            self.was_focused = is_focused;
        }
        
        // If no animation (instant), return target value immediately
        if fade_duration_ms == 0 {
            return if is_focused { 1.0 } else { inactive_dim };
        }
        
        let elapsed = self.focus_animation_start.elapsed().as_millis() as f32;
        let duration = fade_duration_ms as f32;
        let progress = (elapsed / duration).min(1.0);
        
        // Smooth easing (ease-out cubic)
        let eased = 1.0 - (1.0 - progress).powi(3);
        
        if is_focused {
            // Fading in: from inactive_dim to 1.0
            inactive_dim + (1.0 - inactive_dim) * eased
        } else {
            // Fading out: from 1.0 to inactive_dim
            1.0 - (1.0 - inactive_dim) * eased
        }
    }
}

/// Geometry of a pane in pixels.
#[derive(Debug, Clone, Copy)]
struct PaneGeometry {
    /// Left edge in pixels.
    x: f32,
    /// Top edge in pixels.
    y: f32,
    /// Width in pixels.
    width: f32,
    /// Height in pixels.
    height: f32,
    /// Number of columns.
    cols: usize,
    /// Number of rows.
    rows: usize,
}

/// A node in the split tree - either a split or a leaf (pane).
enum SplitNode {
    /// A leaf node containing a pane.
    Leaf {
        pane_id: PaneId,
        /// Cached geometry, updated during layout.
        geometry: PaneGeometry,
    },
    /// A split node with two children.
    Split {
        /// True for horizontal split (panes side-by-side), false for vertical (panes stacked).
        horizontal: bool,
        /// Size ratio of the first child (0.0 to 1.0).
        ratio: f32,
        /// First child (left or top).
        first: Box<SplitNode>,
        /// Second child (right or bottom).
        second: Box<SplitNode>,
    },
}

impl SplitNode {
    /// Create a new leaf node.
    fn leaf(pane_id: PaneId) -> Self {
        SplitNode::Leaf {
            pane_id,
            geometry: PaneGeometry {
                x: 0.0,
                y: 0.0,
                width: 0.0,
                height: 0.0,
                cols: 0,
                rows: 0,
            },
        }
    }
    
    /// Split this node, replacing it with a split containing the original and a new pane.
    /// Returns the new node that should replace this one.
    fn split(self, new_pane_id: PaneId, horizontal: bool) -> Self {
        SplitNode::Split {
            horizontal,
            ratio: 0.5,
            first: Box::new(self),
            second: Box::new(SplitNode::leaf(new_pane_id)),
        }
    }
    
    /// Calculate layout for all nodes given the available space.
    fn layout(&mut self, x: f32, y: f32, width: f32, height: f32, cell_width: f32, cell_height: f32, border_width: f32) {
        match self {
            SplitNode::Leaf { geometry, .. } => {
                let cols = ((width - border_width) / cell_width).floor() as usize;
                let rows = ((height - border_width) / cell_height).floor() as usize;
                *geometry = PaneGeometry {
                    x,
                    y,
                    width,
                    height,
                    cols: cols.max(1),
                    rows: rows.max(1),
                };
            }
            SplitNode::Split { horizontal, ratio, first, second } => {
                if *horizontal {
                    // Side-by-side split
                    let first_width = (width * *ratio) - border_width / 2.0;
                    let second_width = width - first_width - border_width;
                    first.layout(x, y, first_width, height, cell_width, cell_height, border_width);
                    second.layout(x + first_width + border_width, y, second_width, height, cell_width, cell_height, border_width);
                } else {
                    // Stacked split
                    let first_height = (height * *ratio) - border_width / 2.0;
                    let second_height = height - first_height - border_width;
                    first.layout(x, y, width, first_height, cell_width, cell_height, border_width);
                    second.layout(x, y + first_height + border_width, width, second_height, cell_width, cell_height, border_width);
                }
            }
        }
    }
    
    /// Find the geometry for a specific pane.
    fn find_geometry(&self, target_id: PaneId) -> Option<PaneGeometry> {
        match self {
            SplitNode::Leaf { pane_id, geometry } => {
                if *pane_id == target_id {
                    Some(*geometry)
                } else {
                    None
                }
            }
            SplitNode::Split { first, second, .. } => {
                first.find_geometry(target_id).or_else(|| second.find_geometry(target_id))
            }
        }
    }
    
    /// Collect all pane geometries.
    fn collect_geometries(&self, geometries: &mut Vec<(PaneId, PaneGeometry)>) {
        match self {
            SplitNode::Leaf { pane_id, geometry } => {
                geometries.push((*pane_id, *geometry));
            }
            SplitNode::Split { first, second, .. } => {
                first.collect_geometries(geometries);
                second.collect_geometries(geometries);
            }
        }
    }
    
    /// Find a neighbor pane in the given direction.
    /// Returns the pane ID of the neighbor, if any.
    fn find_neighbor(&self, target_id: PaneId, direction: Direction) -> Option<PaneId> {
        // First, find the geometry of the target pane
        let target_geom = self.find_geometry(target_id)?;
        
        // Collect all geometries
        let mut all_geoms = Vec::new();
        self.collect_geometries(&mut all_geoms);
        
        // Find the best candidate in the given direction
        let mut best: Option<(PaneId, f32)> = None;
        
        for (pane_id, geom) in all_geoms {
            if pane_id == target_id {
                continue;
            }
            
            let is_neighbor = match direction {
                Direction::Up => {
                    // Neighbor is above: its bottom edge is near our top edge
                    geom.y + geom.height <= target_geom.y + 5.0 &&
                    Self::overlaps_horizontally(&geom, &target_geom)
                }
                Direction::Down => {
                    // Neighbor is below: its top edge is near our bottom edge
                    geom.y >= target_geom.y + target_geom.height - 5.0 &&
                    Self::overlaps_horizontally(&geom, &target_geom)
                }
                Direction::Left => {
                    // Neighbor is to the left: its right edge is near our left edge
                    geom.x + geom.width <= target_geom.x + 5.0 &&
                    Self::overlaps_vertically(&geom, &target_geom)
                }
                Direction::Right => {
                    // Neighbor is to the right: its left edge is near our right edge
                    geom.x >= target_geom.x + target_geom.width - 5.0 &&
                    Self::overlaps_vertically(&geom, &target_geom)
                }
            };
            
            if is_neighbor {
                // Calculate distance (for choosing closest)
                let distance = match direction {
                    Direction::Up => target_geom.y - (geom.y + geom.height),
                    Direction::Down => geom.y - (target_geom.y + target_geom.height),
                    Direction::Left => target_geom.x - (geom.x + geom.width),
                    Direction::Right => geom.x - (target_geom.x + target_geom.width),
                };
                
                if distance >= 0.0 {
                    if best.is_none() || distance < best.unwrap().1 {
                        best = Some((pane_id, distance));
                    }
                }
            }
        }
        
        best.map(|(id, _)| id)
    }
    
    fn overlaps_horizontally(a: &PaneGeometry, b: &PaneGeometry) -> bool {
        let a_left = a.x;
        let a_right = a.x + a.width;
        let b_left = b.x;
        let b_right = b.x + b.width;
        a_left < b_right && a_right > b_left
    }
    
    fn overlaps_vertically(a: &PaneGeometry, b: &PaneGeometry) -> bool {
        let a_top = a.y;
        let a_bottom = a.y + a.height;
        let b_top = b.y;
        let b_bottom = b.y + b.height;
        a_top < b_bottom && a_bottom > b_top
    }
    
    /// Remove a pane from the tree. Returns the new tree root (or None if tree is empty).
    fn remove_pane(self, target_id: PaneId) -> Option<SplitNode> {
        match self {
            SplitNode::Leaf { pane_id, .. } => {
                if pane_id == target_id {
                    None // Remove this leaf
                } else {
                    Some(self) // Keep this leaf
                }
            }
            SplitNode::Split { horizontal, ratio, first, second } => {
                // Check if target is in first or second subtree
                let first_has_target = first.contains_pane(target_id);
                let second_has_target = second.contains_pane(target_id);
                
                if first_has_target {
                    match first.remove_pane(target_id) {
                        Some(new_first) => Some(SplitNode::Split {
                            horizontal,
                            ratio,
                            first: Box::new(new_first),
                            second,
                        }),
                        None => Some(*second), // First child removed, promote second
                    }
                } else if second_has_target {
                    match second.remove_pane(target_id) {
                        Some(new_second) => Some(SplitNode::Split {
                            horizontal,
                            ratio,
                            first,
                            second: Box::new(new_second),
                        }),
                        None => Some(*first), // Second child removed, promote first
                    }
                } else {
                    Some(SplitNode::Split { horizontal, ratio, first, second })
                }
            }
        }
    }
    
    /// Check if this tree contains the given pane.
    fn contains_pane(&self, target_id: PaneId) -> bool {
        match self {
            SplitNode::Leaf { pane_id, .. } => *pane_id == target_id,
            SplitNode::Split { first, second, .. } => {
                first.contains_pane(target_id) || second.contains_pane(target_id)
            }
        }
    }
    
    /// Split the pane with the given ID.
    fn split_pane(self, target_id: PaneId, new_pane_id: PaneId, horizontal: bool) -> Self {
        match self {
            SplitNode::Leaf { pane_id, geometry } => {
                if pane_id == target_id {
                    SplitNode::Leaf { pane_id, geometry }.split(new_pane_id, horizontal)
                } else {
                    SplitNode::Leaf { pane_id, geometry }
                }
            }
            SplitNode::Split { horizontal: h, ratio, first, second } => {
                SplitNode::Split {
                    horizontal: h,
                    ratio,
                    first: Box::new(first.split_pane(target_id, new_pane_id, horizontal)),
                    second: Box::new(second.split_pane(target_id, new_pane_id, horizontal)),
                }
            }
        }
    }
}

/// Unique identifier for a tab.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct TabId(u64);

impl TabId {
    fn new() -> Self {
        use std::sync::atomic::AtomicU64;
        static NEXT_ID: AtomicU64 = AtomicU64::new(1);
        Self(NEXT_ID.fetch_add(1, Ordering::Relaxed))
    }
}

/// A single tab containing one or more panes arranged in a split tree.
struct Tab {
    /// Unique identifier for this tab.
    #[allow(dead_code)]
    id: TabId,
    /// All panes in this tab, keyed by PaneId.
    panes: HashMap<PaneId, Pane>,
    /// The split tree structure.
    split_root: SplitNode,
    /// Currently active pane ID.
    active_pane: PaneId,
    /// Tab title (from OSC or shell).
    #[allow(dead_code)]
    title: String,
}

impl Tab {
    /// Create a new tab with a single pane.
    fn new(cols: usize, rows: usize, scrollback_lines: usize) -> Result<Self, String> {
        let pane = Pane::new(cols, rows, scrollback_lines)?;
        let pane_id = pane.id;
        
        let mut panes = HashMap::new();
        panes.insert(pane_id, pane);
        
        Ok(Self {
            id: TabId::new(),
            panes,
            split_root: SplitNode::leaf(pane_id),
            active_pane: pane_id,
            title: String::from("zsh"),
        })
    }
    
    /// Get the active pane.
    fn active_pane(&self) -> Option<&Pane> {
        self.panes.get(&self.active_pane)
    }
    
    /// Get the active pane mutably.
    fn active_pane_mut(&mut self) -> Option<&mut Pane> {
        self.panes.get_mut(&self.active_pane)
    }
    
    /// Resize all panes based on new window dimensions.
    fn resize(&mut self, width: f32, height: f32, cell_width: f32, cell_height: f32, border_width: f32) {
        // Recalculate layout
        self.split_root.layout(0.0, 0.0, width, height, cell_width, cell_height, border_width);
        
        // Resize each pane's terminal based on its geometry
        let mut geometries = Vec::new();
        self.split_root.collect_geometries(&mut geometries);
        
        for (pane_id, geom) in geometries {
            if let Some(pane) = self.panes.get_mut(&pane_id) {
                pane.resize(geom.cols, geom.rows);
            }
        }
    }
    
    /// Write data to the active pane's PTY.
    fn write_to_pty(&mut self, data: &[u8]) {
        if let Some(pane) = self.active_pane_mut() {
            pane.write_to_pty(data);
        }
    }
    
    /// Check if any pane's shell has exited and clean up.
    /// Returns true if all panes have exited (tab should close).
    fn check_exited_panes(&mut self) -> bool {
        // Collect exited pane IDs
        let exited: Vec<PaneId> = self.panes
            .iter()
            .filter(|(_, pane)| pane.child_exited())
            .map(|(id, _)| *id)
            .collect();
        
        // Remove exited panes
        for pane_id in exited {
            self.remove_pane(pane_id);
        }
        
        self.panes.is_empty()
    }
    
    /// Split the active pane.
    fn split(&mut self, horizontal: bool, cols: usize, rows: usize, scrollback_lines: usize) -> Result<PaneId, String> {
        let new_pane = Pane::new(cols, rows, scrollback_lines)?;
        let new_pane_id = new_pane.id;
        
        // Add to panes map
        self.panes.insert(new_pane_id, new_pane);
        
        // Update split tree
        let old_root = std::mem::replace(&mut self.split_root, SplitNode::leaf(PaneId(0)));
        self.split_root = old_root.split_pane(self.active_pane, new_pane_id, horizontal);
        
        // Focus the new pane
        self.active_pane = new_pane_id;
        
        Ok(new_pane_id)
    }
    
    /// Remove a pane from the tab.
    fn remove_pane(&mut self, pane_id: PaneId) {
        // Remove from map
        self.panes.remove(&pane_id);
        
        // Update split tree
        let old_root = std::mem::replace(&mut self.split_root, SplitNode::leaf(PaneId(0)));
        if let Some(new_root) = old_root.remove_pane(pane_id) {
            self.split_root = new_root;
        }
        
        // If we removed the active pane, select a new one
        if self.active_pane == pane_id {
            if let Some(first_pane_id) = self.panes.keys().next() {
                self.active_pane = *first_pane_id;
            }
        }
    }
    
    /// Close the active pane.
    fn close_active_pane(&mut self) {
        let pane_id = self.active_pane;
        self.remove_pane(pane_id);
    }
    
    /// Navigate to a neighbor pane in the given direction.
    fn focus_neighbor(&mut self, direction: Direction) {
        if let Some(neighbor_id) = self.split_root.find_neighbor(self.active_pane, direction) {
            self.active_pane = neighbor_id;
        }
    }
    
    /// Get pane by ID.
    fn get_pane(&self, pane_id: PaneId) -> Option<&Pane> {
        self.panes.get(&pane_id)
    }
    
    /// Get pane by ID mutably.
    fn get_pane_mut(&mut self, pane_id: PaneId) -> Option<&mut Pane> {
        self.panes.get_mut(&pane_id)
    }
    
    /// Collect all pane geometries for rendering.
    fn collect_pane_geometries(&self) -> Vec<(PaneId, PaneGeometry)> {
        let mut geometries = Vec::new();
        self.split_root.collect_geometries(&mut geometries);
        geometries
    }
    
    /// Check if all panes have exited (tab should be closed).
    fn child_exited(&mut self) -> bool {
        self.check_exited_panes()
    }
}

/// PID file location for single-instance support.
fn pid_file_path() -> std::path::PathBuf {
    let runtime_dir = std::env::var("XDG_RUNTIME_DIR")
        .unwrap_or_else(|_| "/tmp".to_string());
    std::path::PathBuf::from(runtime_dir).join("zterm.pid")
}

/// Check if another instance is running and signal it to show window.
/// Returns true if we signaled an existing instance (and should exit).
fn signal_existing_instance() -> bool {
    let pid_path = pid_file_path();
    
    if let Ok(contents) = std::fs::read_to_string(&pid_path) {
        if let Ok(pid) = contents.trim().parse::<i32>() {
            // Check if process is alive
            let alive = unsafe { libc::kill(pid, 0) == 0 };
            
            if alive {
                // Send SIGUSR1 to show window
                log::info!("Signaling existing instance (PID {})", pid);
                unsafe { libc::kill(pid, libc::SIGUSR1) };
                return true;
            } else {
                // Stale PID file, remove it
                let _ = std::fs::remove_file(&pid_path);
            }
        }
    }
    
    false
}

/// Write our PID to the PID file.
fn write_pid_file() -> std::io::Result<()> {
    let pid = std::process::id();
    std::fs::write(pid_file_path(), pid.to_string())
}

/// Remove the PID file on exit.
fn remove_pid_file() {
    let _ = std::fs::remove_file(pid_file_path());
}

/// A cell position in the terminal grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CellPosition {
    col: usize,
    row: isize,
}

/// Selection state for mouse text selection.
#[derive(Clone, Debug)]
struct Selection {
    start: CellPosition,
    end: CellPosition,
}

impl Selection {
    fn normalized(&self) -> (CellPosition, CellPosition) {
        if self.start.row < self.end.row 
            || (self.start.row == self.end.row && self.start.col <= self.end.col) {
            (self.start, self.end)
        } else {
            (self.end, self.start)
        }
    }
    
    fn to_screen_coords(&self, current_scroll_offset: usize, visible_rows: usize) -> Option<(usize, usize, usize, usize)> {
        let (start, end) = self.normalized();
        let scroll_offset = current_scroll_offset as isize;
        let screen_start_row = start.row + scroll_offset;
        let screen_end_row = end.row + scroll_offset;
        
        if screen_end_row < 0 || screen_start_row >= visible_rows as isize {
            return None;
        }
        
        let screen_start_row = screen_start_row.max(0) as usize;
        let screen_end_row = (screen_end_row as usize).min(visible_rows.saturating_sub(1));
        let start_col = if start.row + scroll_offset < 0 { 0 } else { start.col };
        let end_col = if end.row + scroll_offset >= visible_rows as isize { usize::MAX } else { end.col };
        
        Some((start_col, screen_start_row, end_col, screen_end_row))
    }
}

/// User event for the event loop.
#[derive(Debug, Clone)]
enum UserEvent {
    /// Signal received to show the window.
    ShowWindow,
    /// PTY has data available for a specific pane.
    PtyReadable(PaneId),
}

/// Main application state.
struct App {
    /// Window (None when headless/closed).
    window: Option<Arc<Window>>,
    /// GPU renderer (None when headless).
    renderer: Option<Renderer>,
    /// All open tabs.
    tabs: Vec<Tab>,
    /// Index of the currently active tab.
    active_tab: usize,
    /// Application configuration.
    config: Config,
    /// Keybinding action map.
    action_map: HashMap<(bool, bool, bool, bool, String), Action>,
    /// Current modifier state.
    modifiers: WinitModifiers,
    /// Keyboard state for encoding.
    keyboard_state: KeyboardState,
    /// Event loop proxy for signaling from other threads.
    event_loop_proxy: Option<EventLoopProxy<UserEvent>>,
    /// Shutdown signal.
    shutdown: Arc<AtomicBool>,
    /// Current mouse cursor position.
    cursor_position: PhysicalPosition<f64>,
    /// Frame counter for FPS logging.
    frame_count: u64,
    /// Last time we logged FPS.
    last_frame_log: std::time::Instant,
    /// Whether window should be created on next opportunity.
    should_create_window: bool,
    /// Edge glow animation state (for when navigation fails).
    edge_glow: Option<EdgeGlow>,
}

const PTY_KEY: usize = 1;

impl App {
    fn new() -> Self {
        let config = Config::load();
        log::info!("Config: font_size={}", config.font_size);
        
        let action_map = config.keybindings.build_action_map();
        log::info!("Action map built with {} bindings:", action_map.len());
        for (key, action) in &action_map {
            log::info!("  {:?} => {:?}", key, action);
        }
        
        Self {
            window: None,
            renderer: None,
            tabs: Vec::new(),
            active_tab: 0,
            config,
            action_map,
            modifiers: WinitModifiers::default(),
            keyboard_state: KeyboardState::new(),
            event_loop_proxy: None,
            shutdown: Arc::new(AtomicBool::new(false)),
            cursor_position: PhysicalPosition::new(0.0, 0.0),
            frame_count: 0,
            last_frame_log: std::time::Instant::now(),
            should_create_window: false,
            edge_glow: None,
        }
    }
    
    fn set_event_loop_proxy(&mut self, proxy: EventLoopProxy<UserEvent>) {
        self.event_loop_proxy = Some(proxy);
    }
    
    /// Create a new tab and start its I/O thread.
    /// Returns the index of the new tab.
    fn create_tab(&mut self, cols: usize, rows: usize) -> Option<usize> {
        log::info!("Creating new tab with {}x{} terminal", cols, rows);
        
        match Tab::new(cols, rows, self.config.scrollback_lines) {
            Ok(tab) => {
                let tab_idx = self.tabs.len();
                
                // Start I/O threads for all panes in this tab
                for pane in tab.panes.values() {
                    self.start_pane_io_thread(pane);
                }
                
                self.tabs.push(tab);
                self.active_tab = tab_idx;
                
                log::info!("Tab {} created (total: {})", tab_idx, self.tabs.len());
                Some(tab_idx)
            }
            Err(e) => {
                log::error!("Failed to create tab: {}", e);
                None
            }
        }
    }
    
    /// Start background I/O thread for a pane's PTY.
    fn start_pane_io_thread(&self, pane: &Pane) {
        self.start_pane_io_thread_with_info(pane.id, pane.pty_fd, pane.pty_buffer.clone());
    }
    
    /// Start background I/O thread for a pane's PTY with explicit info.
    fn start_pane_io_thread_with_info(&self, pane_id: PaneId, pty_fd: i32, pty_buffer: Arc<SharedPtyBuffer>) {
        let Some(proxy) = self.event_loop_proxy.clone() else { return };
        let shutdown = self.shutdown.clone();
        
        std::thread::Builder::new()
            .name(format!("pty-io-{}", pane_id.0))
            .spawn(move || {
                const INPUT_DELAY: Duration = Duration::from_millis(3);
                
                let poller = match Poller::new() {
                    Ok(p) => p,
                    Err(e) => {
                        log::error!("Failed to create PTY poller: {}", e);
                        return;
                    }
                };
                
                unsafe {
                    if let Err(e) = poller.add(pty_fd, Event::readable(PTY_KEY)) {
                        log::error!("Failed to add PTY to poller: {}", e);
                        return;
                    }
                }
                
                let mut events = Events::new();
                let mut last_wakeup_at = std::time::Instant::now();
                let mut has_pending_wakeup = false;
                
                while !shutdown.load(Ordering::Relaxed) {
                    events.clear();
                    
                    let has_space = pty_buffer.has_space();
                    
                    let timeout = if has_pending_wakeup {
                        let elapsed = last_wakeup_at.elapsed();
                        Some(INPUT_DELAY.saturating_sub(elapsed))
                    } else {
                        Some(Duration::from_millis(100))
                    };
                    
                    match poller.wait(&mut events, timeout) {
                        Ok(_) if !events.is_empty() && has_space => {
                            loop {
                                let result = pty_buffer.read_from_fd(pty_fd);
                                if result < 0 {
                                    let err = std::io::Error::last_os_error();
                                    if err.kind() == std::io::ErrorKind::Interrupted {
                                        continue;
                                    }
                                    if err.kind() == std::io::ErrorKind::WouldBlock {
                                        break;
                                    }
                                    log::debug!("PTY read error: {}", err);
                                    break;
                                } else if result == 0 {
                                    break;
                                } else {
                                    has_pending_wakeup = true;
                                    continue;
                                }
                            }
                            
                            let now = std::time::Instant::now();
                            if now.duration_since(last_wakeup_at) >= INPUT_DELAY {
                                let _ = proxy.send_event(UserEvent::PtyReadable(pane_id));
                                last_wakeup_at = now;
                                has_pending_wakeup = false;
                            }
                            
                            unsafe {
                                let _ = poller.modify(
                                    std::os::fd::BorrowedFd::borrow_raw(pty_fd),
                                    Event::readable(PTY_KEY),
                                );
                            }
                        }
                        Ok(_) => {
                            if has_pending_wakeup {
                                let now = std::time::Instant::now();
                                if now.duration_since(last_wakeup_at) >= INPUT_DELAY {
                                    let _ = proxy.send_event(UserEvent::PtyReadable(pane_id));
                                    last_wakeup_at = now;
                                    has_pending_wakeup = false;
                                }
                            }
                        }
                        Err(e) => {
                            log::error!("PTY poll error: {}", e);
                            break;
                        }
                    }
                }
                
                log::debug!("PTY I/O thread for pane {} exiting", pane_id.0);
            })
            .expect("Failed to spawn PTY I/O thread");
    }
    
    /// Create the window and renderer.
    fn create_window(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return; // Window already exists
        }
        
        log::info!("Creating window");
        
        let mut window_attributes = Window::default_attributes()
            .with_title("ZTerm")
            .with_inner_size(PhysicalSize::new(800, 600));

        if self.config.background_opacity < 1.0 {
            window_attributes = window_attributes.with_transparent(true);
        }

        let window = Arc::new(
            event_loop
                .create_window(window_attributes)
                .expect("Failed to create window"),
        );

        let renderer = pollster::block_on(Renderer::new(window.clone(), &self.config));
        let (cols, rows) = renderer.terminal_size();
        
        // Create first tab if no tabs exist
        if self.tabs.is_empty() {
            self.create_tab(cols, rows);
        } else {
            // Resize existing tabs to match window
            self.resize_all_panes();
        }

        self.window = Some(window);
        self.renderer = Some(renderer);
        self.should_create_window = false;
        
        log::info!("Window created: {}x{} cells", cols, rows);
    }
    
    /// Destroy the window but keep terminal state.
    fn destroy_window(&mut self) {
        log::info!("Destroying window (keeping terminal alive)");
        self.renderer = None;
        self.window = None;
    }
    
    /// Resize all panes in all tabs based on renderer dimensions.
    fn resize_all_panes(&mut self) {
        let Some(renderer) = &self.renderer else { return };
        
        let cell_width = renderer.cell_width;
        let cell_height = renderer.cell_height;
        let width = renderer.width as f32;
        let height = renderer.height as f32 - renderer.tab_bar_height();
        let border_width = 2.0; // Border width in pixels
        
        for tab in &mut self.tabs {
            tab.resize(width, height, cell_width, cell_height, border_width);
        }
    }
    
    /// Process PTY data for a specific pane.
    /// Returns true if any data was processed.
    fn poll_pane(&mut self, pane_id: PaneId) -> bool {
        // Find the pane across all tabs and process data
        let mut processed = false;
        let mut commands = Vec::new();
        
        for tab in &mut self.tabs {
            if let Some(pane) = tab.get_pane_mut(pane_id) {
                // Take all pending data atomically
                let data = pane.pty_buffer.take_pending();
                let len = data.len();
                
                if len == 0 {
                    return false;
                }
                
                let process_start = std::time::Instant::now();
                pane.terminal.process(&data);
                let process_time_ns = process_start.elapsed().as_nanos() as u64;
                
                if process_time_ns > 5_000_000 {
                    log::info!("PTY: process={:.2}ms bytes={}",
                        process_time_ns as f64 / 1_000_000.0,
                        len);
                }
                
                // Collect any commands from the terminal
                commands = pane.terminal.take_commands();
                processed = true;
                break;
            }
        }
        
        // Handle commands outside the borrow
        for cmd in commands {
            self.handle_terminal_command(cmd);
        }
        
        processed
    }
    
    /// Handle a command from the terminal (triggered by OSC sequences).
    fn handle_terminal_command(&mut self, cmd: TerminalCommand) {
        match cmd {
            TerminalCommand::NavigatePane(direction) => {
                log::debug!("Terminal requested pane navigation: {:?}", direction);
                self.focus_pane(direction);
            }
        }
    }
    
    /// Send bytes to the active tab's PTY.
    fn write_to_pty(&mut self, data: &[u8]) {
        if let Some(tab) = self.tabs.get_mut(self.active_tab) {
            tab.write_to_pty(data);
        }
    }
    
    /// Get the active tab, if any.
    fn active_tab(&self) -> Option<&Tab> {
        self.tabs.get(self.active_tab)
    }
    
    /// Get the active tab mutably, if any.
    fn active_tab_mut(&mut self) -> Option<&mut Tab> {
        self.tabs.get_mut(self.active_tab)
    }
    
    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width == 0 || new_size.height == 0 {
            return;
        }

        if let Some(renderer) = &mut self.renderer {
            renderer.resize(new_size.width, new_size.height);
        }
        
        // Resize all panes
        self.resize_all_panes();
        
        if let Some(renderer) = &self.renderer {
            let (cols, rows) = renderer.terminal_size();
            log::debug!("Resized to {}x{} cells", cols, rows);
        }
    }
    
    fn get_scroll_offset(&self) -> usize {
        self.active_tab()
            .and_then(|t| t.active_pane())
            .map(|p| p.terminal.scroll_offset)
            .unwrap_or(0)
    }
    
    fn has_mouse_tracking(&self) -> bool {
        self.active_tab()
            .and_then(|t| t.active_pane())
            .map(|p| p.terminal.mouse_tracking != MouseTrackingMode::None)
            .unwrap_or(false)
    }
    
    fn get_mouse_modifiers(&self) -> u8 {
        let mod_state = self.modifiers.state();
        let mut mods = 0u8;
        if mod_state.shift_key() { mods |= 1; }
        if mod_state.alt_key() { mods |= 2; }
        if mod_state.control_key() { mods |= 4; }
        mods
    }
    
    fn send_mouse_event(&mut self, button: u8, col: u16, row: u16, pressed: bool, is_motion: bool) {
        let seq = {
            let Some(tab) = self.active_tab() else { return };
            let Some(pane) = tab.active_pane() else { return };
            pane.terminal.encode_mouse(button, col, row, pressed, is_motion, self.get_mouse_modifiers())
        };
        if !seq.is_empty() {
            self.write_to_pty(&seq);
        }
    }

    fn check_keybinding(&mut self, event: &KeyEvent) -> bool {
        if event.state != ElementState::Pressed || event.repeat {
            return false;
        }

        let mod_state = self.modifiers.state();
        let ctrl = mod_state.control_key();
        let alt = mod_state.alt_key();
        let shift = mod_state.shift_key();
        let super_key = mod_state.super_key();

        let key_name = match &event.logical_key {
            Key::Named(named) => {
                match named {
                    NamedKey::Tab => "tab".to_string(),
                    NamedKey::Enter => "enter".to_string(),
                    NamedKey::Escape => "escape".to_string(),
                    NamedKey::Backspace => "backspace".to_string(),
                    NamedKey::Delete => "delete".to_string(),
                    NamedKey::Insert => "insert".to_string(),
                    NamedKey::Home => "home".to_string(),
                    NamedKey::End => "end".to_string(),
                    NamedKey::PageUp => "pageup".to_string(),
                    NamedKey::PageDown => "pagedown".to_string(),
                    NamedKey::ArrowUp => "up".to_string(),
                    NamedKey::ArrowDown => "down".to_string(),
                    NamedKey::ArrowLeft => "left".to_string(),
                    NamedKey::ArrowRight => "right".to_string(),
                    NamedKey::Space => " ".to_string(),
                    NamedKey::F1 => "f1".to_string(),
                    NamedKey::F2 => "f2".to_string(),
                    NamedKey::F3 => "f3".to_string(),
                    NamedKey::F4 => "f4".to_string(),
                    NamedKey::F5 => "f5".to_string(),
                    NamedKey::F6 => "f6".to_string(),
                    NamedKey::F7 => "f7".to_string(),
                    NamedKey::F8 => "f8".to_string(),
                    NamedKey::F9 => "f9".to_string(),
                    NamedKey::F10 => "f10".to_string(),
                    NamedKey::F11 => "f11".to_string(),
                    NamedKey::F12 => "f12".to_string(),
                    _ => return false,
                }
            }
            Key::Character(c) => c.to_lowercase(),
            _ => return false,
        };

        let lookup = (ctrl, alt, shift, super_key, key_name.clone());
        log::debug!("Keybind lookup: {:?}", lookup);
        let Some(action) = self.action_map.get(&lookup).copied() else {
            return false;
        };

        log::info!("Executing action: {:?}", action);

        self.execute_action(action);
        true
    }

    fn execute_action(&mut self, action: Action) {
        match action {
            Action::Copy => {
                self.copy_selection_to_clipboard();
            }
            Action::Paste => {
                self.paste_from_clipboard();
            }
            Action::NewTab => {
                if let Some(renderer) = &self.renderer {
                    let (cols, rows) = renderer.terminal_size();
                    self.create_tab(cols, rows);
                    // Resize the new tab to calculate pane geometries
                    self.resize_all_panes();
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
            }
            Action::ClosePane => {
                self.close_active_pane();
            }
            Action::NextTab => {
                if !self.tabs.is_empty() {
                    self.active_tab = (self.active_tab + 1) % self.tabs.len();
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
            }
            Action::PrevTab => {
                if !self.tabs.is_empty() {
                    self.active_tab = if self.active_tab == 0 {
                        self.tabs.len() - 1
                    } else {
                        self.active_tab - 1
                    };
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
            }
            Action::Tab1 => self.switch_to_tab(0),
            Action::Tab2 => self.switch_to_tab(1),
            Action::Tab3 => self.switch_to_tab(2),
            Action::Tab4 => self.switch_to_tab(3),
            Action::Tab5 => self.switch_to_tab(4),
            Action::Tab6 => self.switch_to_tab(5),
            Action::Tab7 => self.switch_to_tab(6),
            Action::Tab8 => self.switch_to_tab(7),
            Action::Tab9 => self.switch_to_tab(8),
            Action::SplitHorizontal => {
                self.split_pane(true);
            }
            Action::SplitVertical => {
                self.split_pane(false);
            }
            Action::FocusPaneUp => {
                self.focus_pane_or_pass_key(Direction::Up, b'A');
            }
            Action::FocusPaneDown => {
                self.focus_pane_or_pass_key(Direction::Down, b'B');
            }
            Action::FocusPaneLeft => {
                self.focus_pane_or_pass_key(Direction::Left, b'D');
            }
            Action::FocusPaneRight => {
                self.focus_pane_or_pass_key(Direction::Right, b'C');
            }
        }
    }
    
    fn split_pane(&mut self, horizontal: bool) {
        // Get terminal dimensions
        let (cols, rows) = if let Some(renderer) = &self.renderer {
            renderer.terminal_size()
        } else {
            return;
        };
        
        let scrollback_lines = self.config.scrollback_lines;
        let active_tab = self.active_tab;
        
        // Create the new pane and get its info for the I/O thread
        let new_pane_info = if let Some(tab) = self.tabs.get_mut(active_tab) {
            match tab.split(horizontal, cols, rows, scrollback_lines) {
                Ok(new_pane_id) => {
                    // Get the info we need to start the I/O thread
                    tab.get_pane(new_pane_id).map(|pane| {
                        (pane.id, pane.pty_fd, pane.pty_buffer.clone())
                    })
                }
                Err(e) => {
                    log::error!("Failed to split pane: {}", e);
                    None
                }
            }
        } else {
            None
        };
        
        // Start I/O thread for the new pane (outside the tab borrow)
        if let Some((pane_id, pty_fd, pty_buffer)) = new_pane_info {
            self.start_pane_io_thread_with_info(pane_id, pty_fd, pty_buffer);
            // Recalculate layout
            self.resize_all_panes();
            if let Some(window) = &self.window {
                window.request_redraw();
            }
            log::info!("Split pane (horizontal={}), new pane {}", horizontal, pane_id.0);
        }
    }
    
    /// Focus neighbor pane or pass keys through to applications like Neovim.
    /// If the foreground process matches `pass_keys_to_programs`, send the Alt+Arrow
    /// escape sequence to the PTY. Otherwise, focus the neighboring pane.
    fn focus_pane_or_pass_key(&mut self, direction: Direction, arrow_letter: u8) {
        // Check if we should pass keys to the foreground process
        let should_pass = if let Some(tab) = self.tabs.get(self.active_tab) {
            if let Some(pane) = tab.active_pane() {
                pane.foreground_matches(&self.config.pass_keys_to_programs)
            } else {
                false
            }
        } else {
            false
        };
        
        if should_pass {
            // Send Alt+Arrow escape sequence: \x1b[1;3X where X is A/B/C/D
            let escape_seq = [0x1b, b'[', b'1', b';', b'3', arrow_letter];
            self.write_to_pty(&escape_seq);
        } else {
            self.focus_pane(direction);
        }
    }
    
    fn focus_pane(&mut self, direction: Direction) {
        let navigated = if let Some(tab) = self.tabs.get_mut(self.active_tab) {
            let old_pane = tab.active_pane;
            tab.focus_neighbor(direction);
            tab.active_pane != old_pane
        } else {
            false
        };
        
        if !navigated {
            // No neighbor in that direction - trigger edge glow animation
            self.edge_glow = Some(EdgeGlow::new(direction));
        }
        
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
    
    fn close_active_pane(&mut self) {
        let should_close_tab = if let Some(tab) = self.tabs.get_mut(self.active_tab) {
            tab.close_active_pane();
            tab.panes.is_empty()
        } else {
            false
        };
        
        if should_close_tab {
            self.tabs.remove(self.active_tab);
            if !self.tabs.is_empty() && self.active_tab >= self.tabs.len() {
                self.active_tab = self.tabs.len() - 1;
            }
        } else {
            // Recalculate layout after removing pane
            self.resize_all_panes();
        }
        
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
    
    fn switch_to_tab(&mut self, idx: usize) {
        if idx < self.tabs.len() {
            self.active_tab = idx;
            if let Some(window) = &self.window {
                window.request_redraw();
            }
        }
    }
    
    fn paste_from_clipboard(&mut self) {
        let output = match Command::new("wl-paste")
            .arg("--no-newline")
            .output()
        {
            Ok(output) => output,
            Err(e) => {
                log::warn!("Failed to run wl-paste: {}", e);
                return;
            }
        };
        
        if output.status.success() && !output.stdout.is_empty() {
            self.write_to_pty(&output.stdout);
        }
    }
    
    fn copy_selection_to_clipboard(&mut self) {
        let Some(tab) = self.active_tab() else { return };
        let Some(pane) = tab.active_pane() else { return };
        let Some(selection) = &pane.selection else { return };
        let terminal = &pane.terminal;
        
        let (start, end) = selection.normalized();
        let mut text = String::new();
        
        let scroll_offset = terminal.scroll_offset as isize;
        let rows = terminal.rows;
        
        let screen_start_row = (start.row + scroll_offset).max(0) as usize;
        let screen_end_row = ((end.row + scroll_offset).max(0) as usize).min(rows.saturating_sub(1));
        
        let visible_rows = terminal.visible_rows();
        
        for screen_row in screen_start_row..=screen_end_row {
            if screen_row >= visible_rows.len() {
                break;
            }
            
            let content_row = screen_row as isize - scroll_offset;
            if content_row < start.row || content_row > end.row {
                continue;
            }
            
            let row_cells = visible_rows[screen_row];
            let cols = row_cells.len();
            let col_start = if content_row == start.row { start.col } else { 0 };
            let col_end = if content_row == end.row { end.col } else { cols.saturating_sub(1) };
            
            let mut line = String::new();
            for col in col_start..=col_end.min(cols.saturating_sub(1)) {
                let c = row_cells[col].character;
                if c != '\0' {
                    line.push(c);
                }
            }
            
            text.push_str(line.trim_end());
            if content_row < end.row {
                text.push('\n');
            }
        }
        
        if text.is_empty() {
            return;
        }
        
        match Command::new("wl-copy")
            .stdin(Stdio::piped())
            .spawn()
        {
            Ok(mut child) => {
                if let Some(mut stdin) = child.stdin.take() {
                    let _ = stdin.write_all(text.as_bytes());
                }
                let _ = child.wait();
            }
            Err(e) => {
                log::warn!("Failed to run wl-copy: {}", e);
            }
        }
    }
    
    fn handle_keyboard_input(&mut self, event: KeyEvent) {
        if self.check_keybinding(&event) {
            return;
        }

        let event_type = match event.state {
            ElementState::Pressed => {
                if event.repeat { KeyEventType::Repeat } else { KeyEventType::Press }
            }
            ElementState::Released => KeyEventType::Release,
        };

        if event_type == KeyEventType::Release && !self.keyboard_state.report_events() {
            return;
        }

        let mod_state = self.modifiers.state();
        let modifiers = Modifiers {
            shift: mod_state.shift_key(),
            alt: mod_state.alt_key(),
            ctrl: mod_state.control_key(),
            super_key: mod_state.super_key(),
            hyper: false,
            meta: false,
            caps_lock: false,
            num_lock: false,
        };

        let encoder = KeyEncoder::new(&self.keyboard_state);

        let bytes: Option<Vec<u8>> = match &event.logical_key {
            Key::Named(named) => {
                let func_key = match named {
                    NamedKey::Enter => Some(FunctionalKey::Enter),
                    NamedKey::Backspace => Some(FunctionalKey::Backspace),
                    NamedKey::Tab => Some(FunctionalKey::Tab),
                    NamedKey::Escape => Some(FunctionalKey::Escape),
                    NamedKey::Space => None,
                    NamedKey::ArrowUp => Some(FunctionalKey::Up),
                    NamedKey::ArrowDown => Some(FunctionalKey::Down),
                    NamedKey::ArrowRight => Some(FunctionalKey::Right),
                    NamedKey::ArrowLeft => Some(FunctionalKey::Left),
                    NamedKey::Home => Some(FunctionalKey::Home),
                    NamedKey::End => Some(FunctionalKey::End),
                    NamedKey::PageUp => Some(FunctionalKey::PageUp),
                    NamedKey::PageDown => Some(FunctionalKey::PageDown),
                    NamedKey::Insert => Some(FunctionalKey::Insert),
                    NamedKey::Delete => Some(FunctionalKey::Delete),
                    NamedKey::F1 => Some(FunctionalKey::F1),
                    NamedKey::F2 => Some(FunctionalKey::F2),
                    NamedKey::F3 => Some(FunctionalKey::F3),
                    NamedKey::F4 => Some(FunctionalKey::F4),
                    NamedKey::F5 => Some(FunctionalKey::F5),
                    NamedKey::F6 => Some(FunctionalKey::F6),
                    NamedKey::F7 => Some(FunctionalKey::F7),
                    NamedKey::F8 => Some(FunctionalKey::F8),
                    NamedKey::F9 => Some(FunctionalKey::F9),
                    NamedKey::F10 => Some(FunctionalKey::F10),
                    NamedKey::F11 => Some(FunctionalKey::F11),
                    NamedKey::F12 => Some(FunctionalKey::F12),
                    NamedKey::CapsLock => Some(FunctionalKey::CapsLock),
                    NamedKey::ScrollLock => Some(FunctionalKey::ScrollLock),
                    NamedKey::NumLock => Some(FunctionalKey::NumLock),
                    NamedKey::PrintScreen => Some(FunctionalKey::PrintScreen),
                    NamedKey::Pause => Some(FunctionalKey::Pause),
                    NamedKey::ContextMenu => Some(FunctionalKey::Menu),
                    _ => None,
                };

                if let Some(key) = func_key {
                    Some(encoder.encode_functional(key, modifiers, event_type))
                } else if *named == NamedKey::Space {
                    Some(encoder.encode_char(' ', modifiers, event_type))
                } else {
                    None
                }
            }
            Key::Character(c) => {
                if let Some(ch) = c.chars().next() {
                    let key_char = ch.to_lowercase().next().unwrap_or(ch);
                    Some(encoder.encode_char(key_char, modifiers, event_type))
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(bytes) = bytes {
            // Reset scroll when typing
            if let Some(tab) = self.active_tab_mut() {
                if let Some(pane) = tab.active_pane_mut() {
                    if pane.terminal.scroll_offset > 0 {
                        pane.terminal.scroll_offset = 0;
                    }
                }
            }
            self.write_to_pty(&bytes);
        }
    }
}

impl ApplicationHandler<UserEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_none() {
            self.create_window(event_loop);
        }
    }
    
    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::ShowWindow => {
                log::info!("Received signal to show window");
                if self.window.is_none() {
                    self.create_window(event_loop);
                }
            }
            UserEvent::PtyReadable(pane_id) => {
                // I/O thread has batched wakeups - read all available data now
                let start = std::time::Instant::now();
                self.poll_pane(pane_id);
                let process_time = start.elapsed();
                
                // Request redraw to display the new content
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
                
                if process_time.as_millis() > 5 {
                    log::info!("PTY process took {:?}", process_time);
                }
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                log::info!("Window close requested - hiding window");
                self.destroy_window();
                // Don't exit - keep running headless
            }

            WindowEvent::Resized(new_size) => {
                self.resize(new_size);
            }

            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                log::info!("Scale factor changed to {}", scale_factor);
                let should_resize = if let Some(renderer) = &mut self.renderer {
                    renderer.set_scale_factor(scale_factor)
                } else {
                    false
                };
                if should_resize {
                    self.resize_all_panes();
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            WindowEvent::ModifiersChanged(new_modifiers) => {
                self.modifiers = new_modifiers;
            }

            WindowEvent::MouseWheel { delta, .. } => {
                let lines = match delta {
                    MouseScrollDelta::LineDelta(_, y) => (y * 3.0) as i32,
                    MouseScrollDelta::PixelDelta(pos) => (pos.y / 20.0) as i32,
                };
                
                if lines != 0 {
                    if self.has_mouse_tracking() {
                        if let Some(renderer) = &self.renderer {
                            if let Some((col, row)) = renderer.pixel_to_cell(
                                self.cursor_position.x,
                                self.cursor_position.y
                            ) {
                                let button = if lines > 0 { 64 } else { 65 };
                                let count = lines.abs().min(3);
                                for _ in 0..count {
                                    self.send_mouse_event(button, col as u16, row as u16, true, false);
                                }
                            }
                        }
                    } else if let Some(tab) = self.active_tab_mut() {
                        // Positive lines = scroll wheel up = go into history (increase offset)
                        if let Some(pane) = tab.active_pane_mut() {
                            pane.terminal.scroll(lines);
                        }
                    }
                    if let Some(window) = &self.window {
                        window.request_redraw();
                    }
                }
            }
            
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_position = position;
                
                let is_selecting = self.active_tab()
                    .and_then(|t| t.active_pane())
                    .map(|p| p.is_selecting)
                    .unwrap_or(false);
                if is_selecting && !self.has_mouse_tracking() {
                    if let Some(renderer) = &self.renderer {
                        if let Some((col, screen_row)) = renderer.pixel_to_cell(position.x, position.y) {
                            let scroll_offset = self.get_scroll_offset();
                            let content_row = screen_row as isize - scroll_offset as isize;
                            
                            if let Some(tab) = self.active_tab_mut() {
                                if let Some(pane) = tab.active_pane_mut() {
                                    if let Some(ref mut selection) = pane.selection {
                                        selection.end = CellPosition { col, row: content_row };
                                        if let Some(window) = &self.window {
                                            window.request_redraw();
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            
            WindowEvent::MouseInput { state, button, .. } => {
                let button_code = match button {
                    MouseButton::Left => 0,
                    MouseButton::Middle => 1,
                    MouseButton::Right => 2,
                    _ => return,
                };
                
                if self.has_mouse_tracking() {
                    if let Some(renderer) = &self.renderer {
                        if let Some((col, row)) = renderer.pixel_to_cell(
                            self.cursor_position.x,
                            self.cursor_position.y
                        ) {
                            let pressed = state == ElementState::Pressed;
                            self.send_mouse_event(button_code, col as u16, row as u16, pressed, false);
                            if button == MouseButton::Left {
                                if let Some(tab) = self.active_tab_mut() {
                                    if let Some(pane) = tab.active_pane_mut() {
                                        pane.is_selecting = pressed;
                                    }
                                }
                            }
                        }
                    }
                    if let Some(tab) = self.active_tab_mut() {
                        if let Some(pane) = tab.active_pane_mut() {
                            pane.selection = None;
                        }
                    }
                } else if button == MouseButton::Left {
                    match state {
                        ElementState::Pressed => {
                            if let Some(renderer) = &self.renderer {
                                if let Some((col, screen_row)) = renderer.pixel_to_cell(
                                    self.cursor_position.x, 
                                    self.cursor_position.y
                                ) {
                                    let scroll_offset = self.get_scroll_offset();
                                    let content_row = screen_row as isize - scroll_offset as isize;
                                    let pos = CellPosition { col, row: content_row };
                                    if let Some(tab) = self.active_tab_mut() {
                                        if let Some(pane) = tab.active_pane_mut() {
                                            pane.selection = Some(Selection { start: pos, end: pos });
                                            pane.is_selecting = true;
                                        }
                                    }
                                }
                            }
                        }
                        ElementState::Released => {
                            let was_selecting = self.active_tab()
                                .and_then(|t| t.active_pane())
                                .map(|p| p.is_selecting)
                                .unwrap_or(false);
                            if was_selecting {
                                if let Some(tab) = self.active_tab_mut() {
                                    if let Some(pane) = tab.active_pane_mut() {
                                        pane.is_selecting = false;
                                    }
                                }
                                self.copy_selection_to_clipboard();
                            }
                        }
                    }
                }
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            WindowEvent::KeyboardInput { event, .. } => {
                self.handle_keyboard_input(event);
                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }

            WindowEvent::RedrawRequested => {
                let frame_start = std::time::Instant::now();
                self.frame_count += 1;
                
                if self.last_frame_log.elapsed() >= Duration::from_secs(1) {
                    log::debug!("FPS: {}", self.frame_count);
                    self.frame_count = 0;
                    self.last_frame_log = std::time::Instant::now();
                }
                
                // Note: poll_pane() is called from UserEvent::PtyReadable, not here.
                // This avoids double-processing and keeps rendering fast.
                
                // Send any terminal responses back to PTY (for active pane)
                if let Some(tab) = self.active_tab_mut() {
                    if let Some(pane) = tab.active_pane_mut() {
                        if let Some(response) = pane.terminal.take_response() {
                            pane.write_to_pty(&response);
                        }
                        
                        // Track scrollback changes for selection adjustment
                        let scrollback_len = pane.terminal.scrollback.len() as u32;
                        if scrollback_len != pane.last_scrollback_len {
                            let lines_added = scrollback_len.saturating_sub(pane.last_scrollback_len) as isize;
                            if let Some(ref mut selection) = pane.selection {
                                selection.start.row -= lines_added;
                                selection.end.row -= lines_added;
                            }
                            pane.last_scrollback_len = scrollback_len;
                        }
                    }
                }
                
                // Render all panes
                let render_start = std::time::Instant::now();
                let num_tabs = self.tabs.len();
                let active_tab_idx = self.active_tab;
                let fade_duration_ms = self.config.inactive_pane_fade_ms;
                let inactive_dim = self.config.inactive_pane_dim;
                
                if let Some(renderer) = &mut self.renderer {
                    if let Some(tab) = self.tabs.get_mut(active_tab_idx) {
                        // Collect all pane geometries
                        let geometries = tab.collect_pane_geometries();
                        let active_pane_id = tab.active_pane;
                        
                        // First pass: calculate dim factors (needs mutable access)
                        let mut dim_factors: Vec<(PaneId, f32)> = Vec::new();
                        for (pane_id, _) in &geometries {
                            if let Some(pane) = tab.panes.get_mut(pane_id) {
                                let is_active = *pane_id == active_pane_id;
                                let dim_factor = pane.calculate_dim_factor(is_active, fade_duration_ms, inactive_dim);
                                dim_factors.push((*pane_id, dim_factor));
                            }
                        }
                        
                        // Build render info for all panes
                        let mut pane_render_data: Vec<(&Terminal, PaneRenderInfo, Option<(usize, usize, usize, usize)>)> = Vec::new();
                        
                        for (pane_id, geom) in &geometries {
                            if let Some(pane) = tab.panes.get(pane_id) {
                                let is_active = *pane_id == active_pane_id;
                                let scroll_offset = pane.terminal.scroll_offset;
                                
                                // Get pre-calculated dim factor
                                let dim_factor = dim_factors.iter()
                                    .find(|(id, _)| id == pane_id)
                                    .map(|(_, f)| *f)
                                    .unwrap_or(if is_active { 1.0 } else { inactive_dim });
                                
                                // Convert selection to screen coords for this pane
                                let selection = if is_active {
                                    pane.selection.as_ref()
                                        .and_then(|sel| sel.to_screen_coords(scroll_offset, geom.rows))
                                } else {
                                    None
                                };
                                
                                let render_info = PaneRenderInfo {
                                    x: geom.x,
                                    y: geom.y,
                                    width: geom.width,
                                    height: geom.height,
                                    cols: geom.cols,
                                    rows: geom.rows,
                                    is_active,
                                    dim_factor,
                                };
                                
                                pane_render_data.push((&pane.terminal, render_info, selection));
                            }
                        }
                        
                        // Request redraw if any animation is in progress
                        let animation_in_progress = dim_factors.iter().any(|(id, factor)| {
                            let is_active = *id == active_pane_id;
                            if is_active {
                                *factor < 1.0
                            } else {
                                *factor > inactive_dim
                            }
                        });
                        
                        if animation_in_progress {
                            if let Some(window) = &self.window {
                                window.request_redraw();
                            }
                        }
                        
                        // Handle edge glow animation
                        let edge_glow_ref = self.edge_glow.as_ref();
                        let glow_in_progress = edge_glow_ref.map(|g| !g.is_finished()).unwrap_or(false);
                        
                        match renderer.render_panes(&pane_render_data, num_tabs, active_tab_idx, edge_glow_ref) {
                            Ok(_) => {}
                            Err(wgpu::SurfaceError::Lost) => {
                                renderer.resize(renderer.width, renderer.height);
                            }
                            Err(wgpu::SurfaceError::OutOfMemory) => {
                                log::error!("Out of GPU memory!");
                                event_loop.exit();
                            }
                            Err(e) => {
                                log::error!("Render error: {:?}", e);
                            }
                        }
                        
                        // Request redraw if edge glow is animating
                        if glow_in_progress {
                            if let Some(window) = &self.window {
                                window.request_redraw();
                            }
                        }
                    }
                }
                
                // Clean up finished edge glow animation
                if self.edge_glow.as_ref().map(|g| g.is_finished()).unwrap_or(false) {
                    self.edge_glow = None;
                }
                
                let render_time = render_start.elapsed();
                let frame_time = frame_start.elapsed();
                
                if frame_time.as_millis() > 10 {
                    log::info!("Slow frame: total={:?} render={:?}", 
                        frame_time, render_time);
                }
            }

            _ => {}
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // Check if all tabs have exited
        if self.tabs.is_empty() {
            log::info!("All tabs closed, exiting");
            event_loop.exit();
            return;
        }
        
        // Check for exited tabs and remove them
        let mut i = 0;
        while i < self.tabs.len() {
            if self.tabs[i].child_exited() {
                log::info!("Tab {} shell exited", i);
                self.tabs.remove(i);
                if self.active_tab >= self.tabs.len() && !self.tabs.is_empty() {
                    self.active_tab = self.tabs.len() - 1;
                }
            } else {
                i += 1;
            }
        }
        
        if self.tabs.is_empty() {
            log::info!("All tabs closed, exiting");
            event_loop.exit();
            return;
        }
        
        // Batching is done in the I/O thread (Kitty-style).
        // We just wait for events here.
        event_loop.set_control_flow(ControlFlow::Wait);
    }
}

impl Drop for App {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
        remove_pid_file();
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    log::info!("Starting ZTerm");

    // Check for existing instance
    if signal_existing_instance() {
        log::info!("Signaled existing instance, exiting");
        return;
    }

    // Write PID file
    if let Err(e) = write_pid_file() {
        log::warn!("Failed to write PID file: {}", e);
    }

    // Set up SIGUSR1 handler
    unsafe {
        libc::signal(libc::SIGUSR1, handle_sigusr1 as usize);
    }

    // Create event loop
    let event_loop = EventLoop::<UserEvent>::with_user_event()
        .with_any_thread(true)
        .build()
        .expect("Failed to create event loop");

    event_loop.set_control_flow(ControlFlow::Wait);

    let mut app = App::new();
    let proxy = event_loop.create_proxy();
    app.set_event_loop_proxy(proxy.clone());
    
    // Store proxy for signal handler (uses the global static defined below)
    unsafe {
        EVENT_PROXY = Some(proxy);
    }

    event_loop.run_app(&mut app).expect("Event loop error");
}

// Global static for signal handler access
static mut EVENT_PROXY: Option<EventLoopProxy<UserEvent>> = None;

extern "C" fn handle_sigusr1(_: i32) {
    // Signal handler - must be async-signal-safe
    // We can only set a flag here, the actual window creation happens in the event loop
    unsafe {
        if let Some(ref proxy) = EVENT_PROXY {
            let _ = proxy.send_event(UserEvent::ShowWindow);
        }
    }
}
