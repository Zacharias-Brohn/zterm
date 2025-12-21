//! ZTerm - GPU-accelerated terminal emulator.
//! 
//! Single-process architecture: owns PTY, terminal state, and rendering.
//! Supports window close/reopen without losing terminal state.

use zterm::config::{Action, Config};
use zterm::keyboard::{FunctionalKey, KeyEncoder, KeyEventType, KeyboardState, Modifiers};
use zterm::pty::Pty;
use zterm::renderer::{EdgeGlow, PaneRenderInfo, Renderer, StatuslineComponent, StatuslineContent, StatuslineSection};
use zterm::terminal::{Direction, Terminal, TerminalCommand, MouseTrackingMode};
use zterm::vt_parser::SharedParser;

use std::collections::HashMap;
use std::io::Write;
use std::os::fd::AsRawFd;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use notify::{RecommendedWatcher, RecursiveMode, Watcher};
use polling::{Event, Events, Poller};
use winit::application::ApplicationHandler;
use winit::dpi::{PhysicalPosition, PhysicalSize};
use winit::event::{ElementState, KeyEvent, MouseButton, Modifiers as WinitModifiers, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{Key, NamedKey};
use winit::platform::wayland::EventLoopBuilderExtWayland;
use winit::window::{Window, WindowId};

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
    /// Shared parser with integrated buffer (Kitty-style).
    /// I/O thread writes directly to this, main thread parses in-place.
    shared_parser: Arc<SharedParser>,
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
    /// Custom statusline content set by applications (e.g., neovim).
    /// Contains raw ANSI escape sequences for colors.
    /// When Some, this overrides the default CWD/git statusline.
    custom_statusline: Option<String>,
}

impl Pane {
    /// Create a new pane with its own terminal and PTY.
    fn new(cols: usize, rows: usize, scrollback_lines: usize) -> Result<Self, String> {
        let terminal = Terminal::new(cols, rows, scrollback_lines);
        
        // Calculate pixel dimensions (use default cell size estimate)
        let default_cell_width = 10u16;
        let default_cell_height = 20u16;
        let width_px = cols as u16 * default_cell_width;
        let height_px = rows as u16 * default_cell_height;
        
        // Spawn PTY with initial size - this sets the size BEFORE forking,
        // so the shell inherits the correct terminal dimensions immediately.
        // This prevents race conditions where .zshrc runs before resize().
        let pty = Pty::spawn(
            None, 
            cols as u16, 
            rows as u16, 
            width_px, 
            height_px
        ).map_err(|e| format!("Failed to spawn PTY: {}", e))?;
        
        let pty_fd = pty.as_raw_fd();
        
        Ok(Self {
            id: PaneId::new(),
            terminal,
            pty,
            pty_fd,
            shared_parser: Arc::new(SharedParser::new()),
            selection: None,
            is_selecting: false,
            last_scrollback_len: 0,
            focus_animation_start: std::time::Instant::now(),
            was_focused: true, // New panes start as focused
            custom_statusline: None,
        })
    }
    
    /// Resize the terminal and PTY.
    /// Only sends SIGWINCH to the PTY if the size actually changed.
    fn resize(&mut self, cols: usize, rows: usize, width_px: u16, height_px: u16) {
        // Check if size actually changed before sending SIGWINCH
        // This prevents spurious signals that can interrupt programs like fastfetch
        let size_changed = cols != self.terminal.cols || rows != self.terminal.rows;
        
        self.terminal.resize(cols, rows);
        
        if size_changed {
            if let Err(e) = self.pty.resize(cols as u16, rows as u16, width_px, height_px) {
                log::warn!("Failed to resize PTY: {}", e);
            }
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
    /// Returns the actual used (width, height) after cell alignment.
    /// Note: border_width is kept for API compatibility but borders are now overlaid on panes.
    fn layout(&mut self, x: f32, y: f32, width: f32, height: f32, cell_width: f32, cell_height: f32, _border_width: f32) -> (f32, f32) {
        match self {
            SplitNode::Leaf { geometry, .. } => {
                // Calculate how many cells fit
                let cols = (width / cell_width).floor() as usize;
                let rows = (height / cell_height).floor() as usize;
                // Store the full allocated dimensions (not just cell-aligned)
                // This ensures edge glow and pane dimming cover the full pane area
                *geometry = PaneGeometry {
                    x,
                    y,
                    width,   // Full allocated width
                    height,  // Full allocated height
                    cols: cols.max(1),
                    rows: rows.max(1),
                };
                // Return cell-aligned dimensions for layout calculations
                let actual_width = cols.max(1) as f32 * cell_width;
                let actual_height = rows.max(1) as f32 * cell_height;
                (actual_width, actual_height)
            }
            SplitNode::Split { horizontal, ratio, first, second } => {
                if *horizontal {
                    // Side-by-side split (horizontal means panes are side-by-side)
                    // No border space reserved - border will be overlaid on pane edges
                    let total_cols = (width / cell_width).floor() as usize;
                    
                    // Distribute columns by ratio
                    let first_cols = ((total_cols as f32) * *ratio).round() as usize;
                    let second_cols = total_cols.saturating_sub(first_cols);
                    
                    // Convert back to pixel widths
                    let first_alloc_width = first_cols.max(1) as f32 * cell_width;
                    let second_alloc_width = second_cols.max(1) as f32 * cell_width;
                    
                    // Layout panes flush against each other (border overlays the edge)
                    let (first_actual_w, first_actual_h) = first.layout(x, y, first_alloc_width, height, cell_width, cell_height, _border_width);
                    let (second_actual_w, second_actual_h) = second.layout(x + first_actual_w, y, second_alloc_width, height, cell_width, cell_height, _border_width);
                    
                    // Total used size: both panes (no border gap)
                    (first_actual_w + second_actual_w, first_actual_h.max(second_actual_h))
                } else {
                    // Stacked split (vertical means panes are stacked)
                    // No border space reserved - border will be overlaid on pane edges
                    let total_rows = (height / cell_height).floor() as usize;
                    
                    // Distribute rows by ratio
                    let first_rows = ((total_rows as f32) * *ratio).round() as usize;
                    let second_rows = total_rows.saturating_sub(first_rows);
                    
                    // Convert back to pixel heights
                    let first_alloc_height = first_rows.max(1) as f32 * cell_height;
                    let second_alloc_height = second_rows.max(1) as f32 * cell_height;
                    
                    // Layout panes flush against each other (border overlays the edge)
                    let (first_actual_w, first_actual_h) = first.layout(x, y, width, first_alloc_height, cell_width, cell_height, _border_width);
                    let (second_actual_w, second_actual_h) = second.layout(x, y + first_actual_h, width, second_alloc_height, cell_width, cell_height, _border_width);
                    
                    // Total used size: both panes (no border gap)
                    (first_actual_w.max(second_actual_w), first_actual_h + second_actual_h)
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
    /// Actual used grid dimensions (width, height) after cell alignment.
    /// Used for centering the grid in the window.
    grid_used_dimensions: (f32, f32),
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
            grid_used_dimensions: (0.0, 0.0), // Will be set on first resize
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
        // Recalculate layout - returns actual used dimensions for centering
        let used_dims = self.split_root.layout(0.0, 0.0, width, height, cell_width, cell_height, border_width);
        
        // Store the used dimensions for this tab
        self.grid_used_dimensions = used_dims;
        
        // Resize each pane's terminal based on its geometry
        let mut geometries = Vec::new();
        self.split_root.collect_geometries(&mut geometries);
        
        for (pane_id, geom) in geometries {
            if let Some(pane) = self.panes.get_mut(&pane_id) {
                // Report pixel dimensions as exact cell grid size (cols * cell_width, rows * cell_height)
                // This ensures applications like kitten icat calculate image placement correctly
                let pixel_width = (geom.cols as f32 * cell_width) as u16;
                let pixel_height = (geom.rows as f32 * cell_height) as u16;
                pane.resize(geom.cols, geom.rows, pixel_width, pixel_height);
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

/// Build a statusline section for the current working directory.
/// 
/// Transforms the path into styled segments within a section:
/// - Replaces $HOME prefix with "~"
/// - Each directory segment gets " " prefix and its own color
/// - Arrow separator "" between segments inherits previous segment's color
/// - Colors cycle through palette indices 2-7 (skipping 0-1 which are often close to white)
/// - Last segment is bold
/// - Section has a dark gray background color (#282828)
/// - Section ends with powerline arrow transition
fn build_cwd_section(cwd: &str) -> StatuslineSection {
    // Colors to cycle through (skip 0 and 1 which are often near-white in custom schemes)
    const COLORS: [u8; 6] = [2, 3, 4, 5, 6, 7];
    
    let mut components = Vec::new();
    
    // Get home directory and replace prefix with ~
    let display_path = if let Ok(home) = std::env::var("HOME") {
        if cwd.starts_with(&home) {
            format!("~{}", &cwd[home.len()..])
        } else {
            cwd.to_string()
        }
    } else {
        cwd.to_string()
    };
    
    // Split path into segments
    let segments: Vec<&str> = display_path
        .split('/')
        .filter(|s| !s.is_empty())
        .collect();
    
    if segments.is_empty() {
        // Root directory
        components.push(StatuslineComponent::new(" \u{F07C} / ").fg(COLORS[0]));
        return StatuslineSection::with_rgb_bg(0x28, 0x28, 0x28).with_components(components);
    }
    
    // Add leading space for padding
    components.push(StatuslineComponent::new(" "));
    
    let last_idx = segments.len() - 1;
    
    for (i, segment) in segments.iter().enumerate() {
        // Cycle through colors for each segment
        let color = COLORS[i % COLORS.len()];
        
        if i > 0 {
            // Add arrow separator with previous segment's color
            // U+E0B1 is the powerline thin chevron right
            let prev_color = COLORS[(i - 1) % COLORS.len()];
            components.push(StatuslineComponent::new(" \u{E0B1} ").fg(prev_color));
        }
        
        // Directory segment with folder icon prefix
        // U+F07C is the folder-open icon from Nerd Fonts
        let text = format!("\u{F07C} {}", segment);
        let component = if i == last_idx {
            // Last segment is bold
            StatuslineComponent::new(text).fg(color).bold()
        } else {
            StatuslineComponent::new(text).fg(color)
        };
        
        components.push(component);
    }
    
    // Add trailing space for padding before the powerline arrow
    components.push(StatuslineComponent::new(" "));
    
    // Use dark gray (#282828) as section background
    StatuslineSection::with_rgb_bg(0x28, 0x28, 0x28).with_components(components)
}

/// Git repository status information.
#[derive(Debug, Default)]
struct GitStatus {
    /// Current branch or HEAD reference.
    head: String,
    /// Number of commits ahead of upstream.
    ahead: usize,
    /// Number of commits behind upstream.
    behind: usize,
    /// Working directory changes (modified, deleted, untracked, etc.).
    working_changed: usize,
    /// Working directory status string (e.g., "~1 +2 -1").
    working_string: String,
    /// Staged changes count.
    staging_changed: usize,
    /// Staging status string.
    staging_string: String,
    /// Number of stashed changes.
    stash_count: usize,
}

/// Get git status for a directory.
/// Returns None if not in a git repository.
fn get_git_status(cwd: &str) -> Option<GitStatus> {
    use std::process::Command;
    
    // Check if we're in a git repo and get the branch name
    let head_output = Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .current_dir(cwd)
        .output()
        .ok()?;
    
    if !head_output.status.success() {
        return None;
    }
    
    let head = String::from_utf8_lossy(&head_output.stdout).trim().to_string();
    
    // Get ahead/behind status
    let mut ahead = 0;
    let mut behind = 0;
    if let Ok(output) = Command::new("git")
        .args(["rev-list", "--left-right", "--count", "HEAD...@{upstream}"])
        .current_dir(cwd)
        .output()
    {
        if output.status.success() {
            let counts = String::from_utf8_lossy(&output.stdout);
            let parts: Vec<&str> = counts.trim().split_whitespace().collect();
            if parts.len() == 2 {
                ahead = parts[0].parse().unwrap_or(0);
                behind = parts[1].parse().unwrap_or(0);
            }
        }
    }
    
    // Get working directory and staging status using git status --porcelain
    let mut working_modified = 0;
    let mut working_added = 0;
    let mut working_deleted = 0;
    let mut staging_modified = 0;
    let mut staging_added = 0;
    let mut staging_deleted = 0;
    
    if let Ok(output) = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(cwd)
        .output()
    {
        if output.status.success() {
            let status = String::from_utf8_lossy(&output.stdout);
            for line in status.lines() {
                if line.len() < 2 {
                    continue;
                }
                let chars: Vec<char> = line.chars().collect();
                let staging_char = chars[0];
                let working_char = chars[1];
                
                // Staging status (first column)
                match staging_char {
                    'M' => staging_modified += 1,
                    'A' => staging_added += 1,
                    'D' => staging_deleted += 1,
                    'R' => staging_modified += 1, // renamed
                    'C' => staging_added += 1,    // copied
                    _ => {}
                }
                
                // Working directory status (second column)
                match working_char {
                    'M' => working_modified += 1,
                    'D' => working_deleted += 1,
                    '?' => working_added += 1, // untracked
                    _ => {}
                }
            }
        }
    }
    
    // Build status strings like oh-my-posh format
    let working_changed = working_modified + working_added + working_deleted;
    let mut working_parts = Vec::new();
    if working_modified > 0 {
        working_parts.push(format!("~{}", working_modified));
    }
    if working_added > 0 {
        working_parts.push(format!("+{}", working_added));
    }
    if working_deleted > 0 {
        working_parts.push(format!("-{}", working_deleted));
    }
    let working_string = working_parts.join(" ");
    
    let staging_changed = staging_modified + staging_added + staging_deleted;
    let mut staging_parts = Vec::new();
    if staging_modified > 0 {
        staging_parts.push(format!("~{}", staging_modified));
    }
    if staging_added > 0 {
        staging_parts.push(format!("+{}", staging_added));
    }
    if staging_deleted > 0 {
        staging_parts.push(format!("-{}", staging_deleted));
    }
    let staging_string = staging_parts.join(" ");
    
    // Get stash count
    let mut stash_count = 0;
    if let Ok(output) = Command::new("git")
        .args(["stash", "list"])
        .current_dir(cwd)
        .output()
    {
        if output.status.success() {
            let stash = String::from_utf8_lossy(&output.stdout);
            stash_count = stash.lines().count();
        }
    }
    
    Some(GitStatus {
        head,
        ahead,
        behind,
        working_changed,
        working_string,
        staging_changed,
        staging_string,
        stash_count,
    })
}

/// Build a statusline section for git status.
/// Returns None if not in a git repository.
fn build_git_section(cwd: &str) -> Option<StatuslineSection> {
    let status = get_git_status(cwd)?;
    
    // Determine foreground color based on state (matching oh-my-posh template)
    // Priority order (last match wins in oh-my-posh):
    // 1. Default: #0da300 (green)
    // 2. If working or staging changed: #FF9248 (orange)
    // 3. If both ahead and behind: #ff4500 (red-orange)
    // 4. If ahead or behind: #B388FF (purple)
    let fg_color: (u8, u8, u8) = if status.ahead > 0 && status.behind > 0 {
        (0xff, 0x45, 0x00) // #ff4500 - red-orange
    } else if status.ahead > 0 || status.behind > 0 {
        (0xB3, 0x88, 0xFF) // #B388FF - purple
    } else if status.working_changed > 0 || status.staging_changed > 0 {
        (0xFF, 0x92, 0x48) // #FF9248 - orange
    } else {
        (0x0d, 0xa3, 0x00) // #0da300 - green
    };
    
    let mut components = Vec::new();
    
    // Leading space
    components.push(StatuslineComponent::new(" ").rgb_fg(fg_color.0, fg_color.1, fg_color.2));
    
    // Branch name (HEAD)
    // Use git branch icon U+E0A0
    let head_text = format!("\u{E0A0} {}", status.head);
    components.push(StatuslineComponent::new(head_text).rgb_fg(fg_color.0, fg_color.1, fg_color.2));
    
    // Branch status (ahead/behind)
    if status.ahead > 0 || status.behind > 0 {
        let mut branch_status = String::new();
        if status.ahead > 0 {
            branch_status.push_str(&format!(" ↑{}", status.ahead));
        }
        if status.behind > 0 {
            branch_status.push_str(&format!(" ↓{}", status.behind));
        }
        components.push(StatuslineComponent::new(branch_status).rgb_fg(fg_color.0, fg_color.1, fg_color.2));
    }
    
    // Working directory changes - U+F044 is the edit/pencil icon
    if status.working_changed > 0 {
        let working_text = format!(" \u{F044} {}", status.working_string);
        components.push(StatuslineComponent::new(working_text).rgb_fg(fg_color.0, fg_color.1, fg_color.2));
    }
    
    // Separator between working and staging (if both have changes)
    if status.working_changed > 0 && status.staging_changed > 0 {
        components.push(StatuslineComponent::new(" |").rgb_fg(fg_color.0, fg_color.1, fg_color.2));
    }
    
    // Staged changes - U+F046 is the check/staged icon
    if status.staging_changed > 0 {
        let staging_text = format!(" \u{F046} {}", status.staging_string);
        components.push(StatuslineComponent::new(staging_text).rgb_fg(fg_color.0, fg_color.1, fg_color.2));
    }
    
    // Stash count - U+EB4B is the stash icon
    if status.stash_count > 0 {
        let stash_text = format!(" \u{EB4B} {}", status.stash_count);
        components.push(StatuslineComponent::new(stash_text).rgb_fg(fg_color.0, fg_color.1, fg_color.2));
    }
    
    // Trailing space
    components.push(StatuslineComponent::new(" ").rgb_fg(fg_color.0, fg_color.1, fg_color.2));
    
    // Background: #232323
    Some(StatuslineSection::with_rgb_bg(0x23, 0x23, 0x23).with_components(components))
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
    /// Tick event - signals main loop to process all pending PTY data and render.
    /// This is like Kitty's glfwPostEmptyEvent() + process_global_state pattern.
    Tick,
    /// Config file was modified and should be reloaded.
    ConfigReloaded,
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
    /// Edge glow animations (for when navigation fails). Multiple can be active simultaneously.
    edge_glows: Vec<EdgeGlow>,
    #[cfg(feature = "render_timing")]
    /// Cumulative parse time for benchmarking (nanoseconds).
    total_parse_ns: u64,
    #[cfg(feature = "render_timing")]
    /// Cumulative render time for benchmarking (nanoseconds).
    total_render_ns: u64,
    #[cfg(feature = "render_timing")]
    /// Number of parse calls.
    parse_count: u64,
    #[cfg(feature = "render_timing")]
    /// Number of render calls.
    render_count: u64,
    #[cfg(feature = "render_timing")]
    /// Last time we logged cumulative stats.
    last_stats_log: std::time::Instant,
    /// Last time we rendered a frame (for repaint_delay throttling).
    last_render_at: std::time::Instant,
    /// Whether a fatal render error occurred (e.g., OutOfMemory).
    render_fatal_error: bool,
}

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
            edge_glows: Vec::new(),
            #[cfg(feature = "render_timing")]
            total_parse_ns: 0,
            #[cfg(feature = "render_timing")]
            total_render_ns: 0,
            #[cfg(feature = "render_timing")]
            parse_count: 0,
            #[cfg(feature = "render_timing")]
            render_count: 0,
            #[cfg(feature = "render_timing")]
            last_stats_log: std::time::Instant::now(),
            last_render_at: std::time::Instant::now(),
            render_fatal_error: false,
        }
    }
    
    fn set_event_loop_proxy(&mut self, proxy: EventLoopProxy<UserEvent>) {
        self.event_loop_proxy = Some(proxy);
    }
    
    /// Reload configuration from disk and apply changes.
    fn reload_config(&mut self) {
        log::info!("Reloading configuration...");
        let new_config = Config::load();
        
        // Check what changed and apply updates
        let font_size_changed = (new_config.font_size - self.config.font_size).abs() > 0.01;
        let opacity_changed = (new_config.background_opacity - self.config.background_opacity).abs() > 0.01;
        let tab_bar_changed = new_config.tab_bar_position != self.config.tab_bar_position;
        
        // Update the config
        self.config = new_config;
        
        // Rebuild action map for keybindings
        self.action_map = self.config.keybindings.build_action_map();
        
        // Apply renderer changes if we have a renderer
        if let Some(renderer) = &mut self.renderer {
            if opacity_changed {
                renderer.set_background_opacity(self.config.background_opacity);
                log::info!("Updated background opacity to {}", self.config.background_opacity);
            }
            
            if tab_bar_changed {
                renderer.set_tab_bar_position(self.config.tab_bar_position);
                log::info!("Updated tab bar position to {:?}", self.config.tab_bar_position);
            }
            
            if font_size_changed {
                renderer.set_font_size(self.config.font_size);
                log::info!("Updated font size to {}", self.config.font_size);
                // Font size change requires resize to recalculate cell dimensions
                self.resize_all_panes();
            }
        }
        
        // Request redraw to apply visual changes
        self.request_redraw();
        
        log::info!("Configuration reloaded successfully");
    }
    
    /// Request a window redraw if window is available.
    #[inline]
    fn request_redraw(&self) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
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
        self.start_pane_io_thread_with_info(pane.id, pane.pty_fd, pane.shared_parser.clone());
    }
    
    /// Start background I/O thread for a pane's PTY with explicit info.
    /// 
    /// Kitty-style design:
    /// - I/O thread reads PTY data directly into SharedParser's buffer
    /// - Commits bytes atomically (pending counter)
    /// - Sends Tick to main thread after INPUT_DELAY
    /// - When buffer is full, disables PTY polling and waits for wakeup
    /// - Main thread wakes us after parsing (which frees space)
    fn start_pane_io_thread_with_info(&self, pane_id: PaneId, pty_fd: i32, shared_parser: Arc<SharedParser>) {
        let Some(proxy) = self.event_loop_proxy.clone() else { return };
        let shutdown = self.shutdown.clone();
        let wakeup_fd = shared_parser.wakeup_fd();
        
        std::thread::Builder::new()
            .name(format!("pty-io-{}", pane_id.0))
            .spawn(move || {
                // Input delay: batch rapid input bursts to reduce overhead.
                // Kitty uses 3ms, but we use 0 for minimal latency.
                // The main thread's REPAINT_DELAY (10ms) already provides batching.
                const INPUT_DELAY: Duration = Duration::from_millis(0);
                const PTY_KEY: usize = 0;
                const WAKEUP_KEY: usize = 1;
                
                let poller = match Poller::new() {
                    Ok(p) => p,
                    Err(e) => {
                        log::error!("Failed to create PTY poller: {}", e);
                        return;
                    }
                };
                
                // Add PTY fd
                unsafe {
                    if let Err(e) = poller.add(pty_fd, Event::readable(PTY_KEY)) {
                        log::error!("Failed to add PTY to poller: {}", e);
                        return;
                    }
                    // Add wakeup fd - used to wake us when buffer space becomes available
                    if let Err(e) = poller.add(wakeup_fd, Event::readable(WAKEUP_KEY)) {
                        log::error!("Failed to add wakeup fd to poller: {}", e);
                        return;
                    }
                }
                
                let mut events = Events::new();
                let mut last_tick_at = std::time::Instant::now();
                let mut has_pending_data = false;
                
                // Debug tracking
                let mut total_bytes_read: u64 = 0;
                let mut loop_count: u64 = 0;
                let io_start = std::time::Instant::now();
                
                while !shutdown.load(Ordering::Relaxed) {
                    events.clear();
                    loop_count += 1;
                    
                    // Check if we have space - if not, disable PTY polling until woken
                    let has_space = shared_parser.has_space();
                    
                    // Set up poll events: always listen on wakeup_fd, only listen on pty_fd if we have space
                    unsafe {
                        let pty_event = if has_space { Event::readable(PTY_KEY) } else { Event::none(PTY_KEY) };
                        let _ = poller.modify(std::os::fd::BorrowedFd::borrow_raw(pty_fd), pty_event);
                    }
                    
                    // Kitty-style timeout: if we have pending data OR buffer is full, use a timeout.
                    // When buffer is full, we need to periodically re-check if space became available
                    // (don't rely solely on wakeup - that can lead to deadlock).
                    // When we have space and no pending data, we can block indefinitely.
                    let timeout = if has_pending_data || !has_space {
                        let elapsed = last_tick_at.elapsed();
                        Some(INPUT_DELAY.saturating_sub(elapsed))
                    } else {
                        None // Block indefinitely until data arrives
                    };
                    
                    let wait_start = std::time::Instant::now();
                    match poller.wait(&mut events, timeout) {
                        Ok(_) => {
                            let wait_time = wait_start.elapsed();
                            let mut got_wakeup = false;
                            let mut got_pty_data = false;
                            
                            for ev in events.iter() {
                                if ev.key == WAKEUP_KEY {
                                    got_wakeup = true;
                                }
                                if ev.key == PTY_KEY && ev.readable {
                                    got_pty_data = true;
                                }
                            }
                            
                            // Log long waits (only with render_timing feature)
                            #[cfg(feature = "render_timing")]
                            if wait_time.as_millis() > 50 {
                                log::warn!("[IO-{}] Long wait: {:?} has_space={} has_pending={} got_wakeup={} got_pty={} timeout={:?}",
                                    pane_id.0, wait_time, has_space, has_pending_data, got_wakeup, got_pty_data, timeout);
                            }
                            
                            #[cfg(not(feature = "render_timing"))]
                            let _ = wait_time; // silence unused warning
                            
                            // Drain wakeup fd if signaled
                            if got_wakeup {
                                log::trace!("[IO-{}] Got wakeup from main thread", pane_id.0);
                                shared_parser.drain_wakeup();
                                // Re-arm wakeup fd
                                unsafe {
                                    let _ = poller.modify(
                                        std::os::fd::BorrowedFd::borrow_raw(wakeup_fd),
                                        Event::readable(WAKEUP_KEY),
                                    );
                                }
                            }
                            
                            // Read PTY data if:
                            // 1. Poll said PTY is readable, OR
                            // 2. We just got woken up (space became available) - PTY might have data
                            //    we couldn't read before because our buffer was full
                            // The PTY fd is non-blocking, so reading when empty just returns EAGAIN
                            let fresh_has_space = shared_parser.has_space();
                            let should_try_read = (got_pty_data || got_wakeup) && fresh_has_space;
                            
                            if should_try_read {
                                let mut bytes_this_loop: i64 = 0;
                                loop {
                                    let result = shared_parser.read_from_fd(pty_fd);
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
                                        log::debug!("[IO-{}] PTY EOF", pane_id.0);
                                        break;
                                    } else {
                                        bytes_this_loop += result as i64;
                                        total_bytes_read += result as u64;
                                        has_pending_data = true;
                                        // Check if buffer became full
                                        if !shared_parser.has_space() {
                                            log::trace!("[IO-{}] Buffer full after reading {} bytes", pane_id.0, bytes_this_loop);
                                            break;
                                        }
                                    }
                                }
                            } else if got_wakeup && !fresh_has_space {
                                // Buffer is full but we got a wakeup - main is parsing and will
                                // free space soon. Mark pending so we send a tick after delay.
                                has_pending_data = true;
                                log::trace!("[IO-{}] Buffer full after wakeup, will tick", pane_id.0);
                            }
                            
                            // Send Tick to main thread if we have pending data and enough time passed
                            // Like Kitty: just send the wakeup, don't try to deduplicate
                            if has_pending_data {
                                let now = std::time::Instant::now();
                                if now.duration_since(last_tick_at) >= INPUT_DELAY {
                                    log::trace!("[IO-{}] Sending Tick", pane_id.0);
                                    let _ = proxy.send_event(UserEvent::Tick);
                                    last_tick_at = now;
                                    has_pending_data = false;
                                }
                            }
                        }
                        Err(e) => {
                            if e.kind() != std::io::ErrorKind::Interrupted {
                                log::error!("PTY poll error: {}", e);
                                break;
                            }
                        }
                    }
                }
                
                #[cfg(feature = "render_timing")]
                {
                    let elapsed = io_start.elapsed();
                    log::info!("[IO-{}] Thread exiting: loops={} total_bytes={} elapsed={:?} throughput={:.1} MB/s",
                        pane_id.0, loop_count, total_bytes_read, elapsed,
                        total_bytes_read as f64 / elapsed.as_secs_f64() / 1_000_000.0);
                }
                
                #[cfg(not(feature = "render_timing"))]
                let _ = (io_start, loop_count, total_bytes_read); // silence unused warnings
                
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
        // Extract values we need from renderer first
        // Use raw available pixel space so layout can handle cell alignment properly
        let (cell_width, cell_height, available_width, available_height) = {
            let Some(renderer) = &self.renderer else { return };
            let cell_width = renderer.cell_metrics.cell_width as f32;
            let cell_height = renderer.cell_metrics.cell_height as f32;
            let (available_width, available_height) = renderer.available_grid_space();
            (cell_width, cell_height, available_width, available_height)
        };
        
        let border_width = 2.0; // Border width in pixels
        
        for tab in self.tabs.iter_mut() {
            tab.resize(available_width, available_height, cell_width, cell_height, border_width);
            
            // Update cell size on all terminals (needed for Kitty graphics protocol)
            for pane in tab.panes.values_mut() {
                pane.terminal.set_cell_size(cell_width, cell_height);
            }
        }
        
        // Update the renderer with the active tab's used dimensions for proper centering
        if let Some(tab) = self.tabs.get(self.active_tab) {
            let used_dims = tab.grid_used_dimensions;
            if let Some(renderer) = &mut self.renderer {
                renderer.set_grid_used_dimensions(used_dims.0, used_dims.1);
            }
        }
    }
    
    /// Process PTY data for a specific pane using Kitty-style in-place parsing.
    /// 
    /// Returns (processed_any, has_more_data, bytes_parsed) - processed_any is true if data was parsed,
    /// has_more_data is true if there's still pending data after the time budget expired.
    /// 
    /// Key differences from previous design:
    /// 1. begin_parse_pass() compacts buffer and makes pending data visible
    /// 2. Parsing happens in-place - no copying data out of buffer
    /// 3. end_parse_pass() reports consumed bytes, wakes I/O only if buffer was full
    /// 4. Loop continues parsing as long as time budget allows
    fn poll_pane(&mut self, pane_id: PaneId) -> (bool, bool) {
        let mut ever_processed = false;
        let mut all_commands = Vec::new();
        
        for tab in &mut self.tabs {
            if let Some(pane) = tab.get_pane_mut(pane_id) {
                // Use SharedParser's run_parse_pass - it releases lock during parsing
                // so I/O thread can continue writing while we parse
                ever_processed = pane.shared_parser.run_parse_pass(&mut pane.terminal);
                
                if ever_processed {
                    pane.terminal.mark_dirty();
                    // Collect any commands from the terminal
                    all_commands.extend(pane.terminal.take_commands());
                }
                
                // Check for pending data - I/O thread may have written more
                let has_more_data = pane.shared_parser.has_pending_data();
                
                // Handle commands outside the borrow
                for cmd in all_commands {
                    self.handle_terminal_command(pane_id, cmd);
                }
                
                return (ever_processed, has_more_data);
            }
        }
        
        (false, false)
    }
    
    /// Handle a command from the terminal (triggered by OSC sequences).
    fn handle_terminal_command(&mut self, pane_id: PaneId, cmd: TerminalCommand) {
        match cmd {
            TerminalCommand::NavigatePane(direction) => {
                log::debug!("Terminal requested pane navigation: {:?}", direction);
                self.focus_pane(direction);
            }
            TerminalCommand::SetStatusline(statusline) => {
                log::debug!("Pane {:?} set statusline: {:?}", pane_id, statusline.as_ref().map(|s| s.len()));
                // Find the pane and set its custom statusline
                for tab in &mut self.tabs {
                    if let Some(pane) = tab.get_pane_mut(pane_id) {
                        pane.custom_statusline = statusline;
                        break;
                    }
                }
            }
        }
    }
    
    /// Render a frame directly. Called from Tick handler for async rendering.
    /// Returns true if animations are in progress and another render should be scheduled.
    /// 
    /// This is the Kitty-style approach: parse all input, then render once, all in the
    /// same event handler. This avoids the overhead of bouncing between UserEvent::Tick
    /// and WindowEvent::RedrawRequested.
    fn do_render(&mut self) -> bool {
        #[cfg(feature = "render_timing")]
        let render_start = std::time::Instant::now();
        let mut needs_another_frame = false;
        
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
        let num_tabs = self.tabs.len();
        let active_tab_idx = self.active_tab;
        let fade_duration_ms = self.config.inactive_pane_fade_ms;
        let inactive_dim = self.config.inactive_pane_dim;
        
        if let Some(renderer) = &mut self.renderer {
            if let Some(tab) = self.tabs.get_mut(active_tab_idx) {
                // Collect all pane geometries
                let geometries = tab.collect_pane_geometries();
                let active_pane_id = tab.active_pane;
                
                // First pass: sync images and calculate dim factors (needs mutable access)
                let mut dim_factors: Vec<(PaneId, f32)> = Vec::new();
                for (pane_id, _) in &geometries {
                    if let Some(pane) = tab.panes.get_mut(pane_id) {
                        let is_active = *pane_id == active_pane_id;
                        let dim_factor = pane.calculate_dim_factor(is_active, fade_duration_ms, inactive_dim);
                        dim_factors.push((*pane_id, dim_factor));
                        
                        // Sync terminal images to GPU (Kitty graphics protocol)
                        renderer.sync_images(&mut pane.terminal.image_storage);
                    }
                }
                
                // Clear custom statusline if the foreground process is no longer neovim/vim
                if let Some(pane) = tab.panes.get_mut(&active_pane_id) {
                    if pane.custom_statusline.is_some() {
                        if let Some(proc_name) = pane.pty.foreground_process_name() {
                            let is_vim = proc_name == "nvim" || proc_name == "vim" || proc_name == "vi";
                            if !is_vim {
                                pane.custom_statusline = None;
                            }
                        }
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
                            pane_id: pane_id.0,
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
                
                // Check if any animation is in progress
                let animation_in_progress = dim_factors.iter().any(|(id, factor)| {
                    let is_active = *id == active_pane_id;
                    if is_active {
                        *factor < 1.0
                    } else {
                        *factor > inactive_dim
                    }
                });
                
                let glow_in_progress = !self.edge_glows.is_empty();
                
                let image_animation_in_progress = tab.panes.values().any(|pane| {
                    pane.terminal.image_storage.has_animations()
                });
                
                needs_another_frame = animation_in_progress || glow_in_progress || image_animation_in_progress;
                
                // Get the statusline content for the active pane
                let statusline_content: StatuslineContent = tab.panes.get(&active_pane_id)
                    .map(|pane| {
                        if let Some(ref custom) = pane.custom_statusline {
                            StatuslineContent::Raw(custom.clone())
                        } else if let Some(cwd) = pane.pty.foreground_cwd() {
                            let mut sections = vec![build_cwd_section(&cwd)];
                            if let Some(git_section) = build_git_section(&cwd) {
                                sections.push(git_section);
                            }
                            StatuslineContent::Sections(sections)
                        } else {
                            StatuslineContent::Sections(Vec::new())
                        }
                    })
                    .unwrap_or_default();
                
                match renderer.render_panes(&pane_render_data, num_tabs, active_tab_idx, &self.edge_glows, self.config.edge_glow_intensity, &statusline_content) {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost) => {
                        renderer.resize(renderer.width, renderer.height);
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => {
                        log::error!("Out of GPU memory!");
                        self.render_fatal_error = true;
                    }
                    Err(e) => {
                        log::error!("Render error: {:?}", e);
                    }
                }
            }
        }
        
        // Clean up finished edge glow animations
        self.edge_glows.retain(|g| !g.is_finished());
        
        // Update stats
        #[cfg(feature = "render_timing")]
        {
            let render_time = render_start.elapsed();
            self.total_render_ns += render_time.as_nanos() as u64;
            self.render_count += 1;
        }
        self.last_render_at = std::time::Instant::now();
        
        needs_another_frame
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
    
    /// Get the active pane of the active tab, if any.
    fn active_pane(&self) -> Option<&Pane> {
        self.active_tab().and_then(|t| t.active_pane())
    }
    
    /// Get the active pane of the active tab mutably, if any.
    fn active_pane_mut(&mut self) -> Option<&mut Pane> {
        self.active_tab_mut().and_then(|t| t.active_pane_mut())
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
        self.active_pane()
            .map(|p| p.terminal.scroll_offset)
            .unwrap_or(0)
    }
    
    fn has_mouse_tracking(&self) -> bool {
        self.active_pane()
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
            let Some(pane) = self.active_pane() else { return };
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

        let key_name: String = match &event.logical_key {
            Key::Named(named) => {
                let name: &'static str = match named {
                    NamedKey::Tab => "tab",
                    NamedKey::Enter => "enter",
                    NamedKey::Escape => "escape",
                    NamedKey::Backspace => "backspace",
                    NamedKey::Delete => "delete",
                    NamedKey::Insert => "insert",
                    NamedKey::Home => "home",
                    NamedKey::End => "end",
                    NamedKey::PageUp => "pageup",
                    NamedKey::PageDown => "pagedown",
                    NamedKey::ArrowUp => "up",
                    NamedKey::ArrowDown => "down",
                    NamedKey::ArrowLeft => "left",
                    NamedKey::ArrowRight => "right",
                    NamedKey::Space => " ",
                    NamedKey::F1 => "f1",
                    NamedKey::F2 => "f2",
                    NamedKey::F3 => "f3",
                    NamedKey::F4 => "f4",
                    NamedKey::F5 => "f5",
                    NamedKey::F6 => "f6",
                    NamedKey::F7 => "f7",
                    NamedKey::F8 => "f8",
                    NamedKey::F9 => "f9",
                    NamedKey::F10 => "f10",
                    NamedKey::F11 => "f11",
                    NamedKey::F12 => "f12",
                    _ => return false,
                };
                name.to_string()
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
                    self.request_redraw();
                }
            }
            Action::ClosePane => {
                self.close_active_pane();
            }
            Action::NextTab => {
                if !self.tabs.is_empty() {
                    let next_tab = (self.active_tab + 1) % self.tabs.len();
                    self.switch_to_tab(next_tab);
                }
            }
            Action::PrevTab => {
                if !self.tabs.is_empty() {
                    let prev_tab = if self.active_tab == 0 {
                        self.tabs.len() - 1
                    } else {
                        self.active_tab - 1
                    };
                    self.switch_to_tab(prev_tab);
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
                        (pane.id, pane.pty_fd, pane.shared_parser.clone())
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
        if let Some((pane_id, pty_fd, shared_parser)) = new_pane_info {
            self.start_pane_io_thread_with_info(pane_id, pty_fd, shared_parser);
            // Recalculate layout
            self.resize_all_panes();
            self.request_redraw();
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
        // Get current active pane geometry before attempting navigation
        let active_pane_geom = if let Some(tab) = self.tabs.get(self.active_tab) {
            tab.split_root.find_geometry(tab.active_pane)
        } else {
            None
        };
        
        let navigated = if let Some(tab) = self.tabs.get_mut(self.active_tab) {
            let old_pane = tab.active_pane;
            tab.focus_neighbor(direction);
            tab.active_pane != old_pane
        } else {
            false
        };
        
        if !navigated {
            // No neighbor in that direction - trigger edge glow animation
            // Use renderer's helper to calculate proper screen-space glow bounds
            if let (Some(geom), Some(renderer)) = (active_pane_geom, &self.renderer) {
                let (glow_x, glow_y, glow_width, glow_height) = 
                    renderer.calculate_edge_glow_bounds(geom.x, geom.y, geom.width, geom.height);
                
                self.edge_glows.push(EdgeGlow::new(
                    direction,
                    glow_x,
                    glow_y,
                    glow_width,
                    glow_height,
                ));
            }
        }
        
        self.request_redraw();
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
        
        self.request_redraw();
    }
    
    fn switch_to_tab(&mut self, idx: usize) {
        if idx < self.tabs.len() {
            self.active_tab = idx;
            // Update grid dimensions for proper centering of the new active tab
            self.update_active_tab_grid_dimensions();
            self.request_redraw();
        }
    }
    
    /// Update the renderer's grid dimensions based on the active tab's stored dimensions.
    fn update_active_tab_grid_dimensions(&mut self) {
        if let Some(tab) = self.tabs.get(self.active_tab) {
            let used_dims = tab.grid_used_dimensions;
            if let Some(renderer) = &mut self.renderer {
                renderer.set_grid_used_dimensions(used_dims.0, used_dims.1);
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
        log::debug!("KeyEvent: {:?} state={:?} repeat={}", event.logical_key, event.state, event.repeat);
        
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
            log::debug!("Ignoring release event (not in enhanced mode)");
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
            log::debug!("Sending {} bytes to PTY: {:?}", bytes.len(), bytes);
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
        #[cfg(feature = "render_timing")]
        let start = std::time::Instant::now();
        if self.window.is_none() {
            self.create_window(event_loop);
        }
        #[cfg(feature = "render_timing")]
        log::info!("App resumed (window creation): {:?}", start.elapsed());
    }
    
    fn user_event(&mut self, event_loop: &ActiveEventLoop, event: UserEvent) {
        match event {
            UserEvent::ShowWindow => {
                log::info!("Received signal to show window");
                if self.window.is_none() {
                    self.create_window(event_loop);
                }
            }
            UserEvent::Tick => {
                log::info!("[MAIN] Tick received");
                // Check for fatal render errors from previous frames
                if self.render_fatal_error {
                    log::error!("Fatal render error occurred, exiting");
                    event_loop.exit();
                    return;
                }
                
                #[cfg(feature = "render_timing")]
                let tick_received_at = std::time::Instant::now();
                #[cfg(feature = "render_timing")]
                let _ = tick_received_at; // silence unused warning for now
                
                // Like Kitty's process_global_state: parse panes (with time budget), then render
                #[cfg(feature = "render_timing")]
                let tick_start = std::time::Instant::now();
                let mut any_input = false;
                let mut any_has_more = false;
                let mut any_not_synchronized = false;
                
                // Collect all pane IDs first to avoid borrow issues
                let pane_ids: Vec<PaneId> = self.tabs.iter()
                    .flat_map(|tab| tab.panes.keys().copied())
                    .collect();
                
                // Poll each pane - parses data up to time budget, returns if more pending
                for pane_id in &pane_ids {
                    let (processed, has_more) = self.poll_pane(*pane_id);
                    if processed {
                        any_input = true;
                    }
                    if has_more {
                        any_has_more = true;
                    }
                }
                
                // Log detailed parse stats if any tick was slow (only with render_timing feature)
                #[cfg(feature = "render_timing")]
                {
                    let parse_time = tick_start.elapsed();
                    if parse_time.as_millis() > 5 {
                        for tab in &mut self.tabs {
                            for pane in tab.panes.values_mut() {
                                pane.terminal.stats.log_if_slow(0); // Log all stats when tick is slow
                                pane.terminal.stats.reset(); // Reset for next tick
                            }
                        }
                    }
                }
                
                // CRITICAL: Send any terminal responses back to PTY immediately after parsing.
                // This must happen regardless of rendering, or the application will hang
                // waiting for responses to queries like DSR (Device Status Report).
                for tab in &mut self.tabs {
                    for pane in tab.panes.values_mut() {
                        if let Some(response) = pane.terminal.take_response() {
                            log::debug!("[RESPONSE] Sending {} bytes to PTY", response.len());
                            pane.write_to_pty(&response);
                        }
                    }
                }
                
                // Check if any terminal is NOT in synchronized mode (needs render)
                for tab in &self.tabs {
                    for pane in tab.panes.values() {
                        if !pane.terminal.is_synchronized() {
                            any_not_synchronized = true;
                            break;
                        }
                    }
                    if any_not_synchronized { break; }
                }
                
                // Render directly here (Kitty-style), throttled by repaint_delay
                // 6ms ≈ 166 FPS, ensures we hit 144 FPS on high refresh displays
                const REPAINT_DELAY: Duration = Duration::from_millis(6);
                let time_since_last_render = self.last_render_at.elapsed();
                
                let should_render = any_input && any_not_synchronized && 
                    time_since_last_render >= REPAINT_DELAY &&
                    self.renderer.is_some();
                
                #[cfg(feature = "render_timing")]
                let render_start = std::time::Instant::now();
                
                if should_render {
                    let needs_another_frame = self.do_render();
                    
                    // If animations are in progress, schedule another render via redraw
                    if needs_another_frame {
                        self.request_redraw();
                    }
                } else if any_input && any_not_synchronized && self.renderer.is_some() {
                    // We had input but skipped render due to throttling.
                    // Request a redraw so we don't lose this frame.
                    self.request_redraw();
                }
                
                #[cfg(feature = "render_timing")]
                let render_time = if should_render { render_start.elapsed() } else { Duration::ZERO };
                
                // If there's more data pending, send another Tick immediately
                // This ensures we keep processing without waiting for I/O thread wakeup
                if any_has_more {
                    if let Some(proxy) = &self.event_loop_proxy {
                        let _ = proxy.send_event(UserEvent::Tick);
                    }
                }
                
                // Log every tick during benchmark for analysis (only with render_timing feature)
                #[cfg(feature = "render_timing")]
                {
                    let tick_time = tick_start.elapsed();
                    if tick_time.as_millis() > 5 {
                        log::info!("[TICK] render={:?} total={:?} has_more={} rendered={}",
                            render_time, tick_time, any_has_more, should_render);
                    }
                }
            }
            UserEvent::ConfigReloaded => {
                self.reload_config();
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
                self.request_redraw();
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
                    self.request_redraw();
                }
            }
            
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_position = position;
                
                let is_selecting = self.active_pane()
                    .map(|p| p.is_selecting)
                    .unwrap_or(false);
                    
                if is_selecting && self.has_mouse_tracking() {
                    // Send mouse drag/motion events to PTY for apps like Neovim
                    if let Some(renderer) = &self.renderer {
                        if let Some((col, row)) = renderer.pixel_to_cell(position.x, position.y) {
                            // Button 0 (left) with motion flag
                            self.send_mouse_event(0, col as u16, row as u16, true, true);
                        }
                    }
                } else if is_selecting && !self.has_mouse_tracking() {
                    // Terminal-native selection
                    if let Some(renderer) = &self.renderer {
                        if let Some((col, screen_row)) = renderer.pixel_to_cell(position.x, position.y) {
                            let scroll_offset = self.get_scroll_offset();
                            let content_row = screen_row as isize - scroll_offset as isize;
                            
                            if let Some(tab) = self.active_tab_mut() {
                                if let Some(pane) = tab.active_pane_mut() {
                                    if let Some(ref mut selection) = pane.selection {
                                        selection.end = CellPosition { col, row: content_row };
                                        self.request_redraw();
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
                                    log::debug!("Selection started at col={}, content_row={}, screen_row={}, scroll_offset={}", col, content_row, screen_row, scroll_offset);
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
                            let was_selecting = self.active_pane()
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
                self.request_redraw();
            }

            WindowEvent::KeyboardInput { event, .. } => {
                self.handle_keyboard_input(event);
                // Don't request_redraw here - let the Tick handler do parsing and rendering
                // after PTY responds. Calling request_redraw here causes a render of stale
                // state before the PTY response arrives.
            }

            WindowEvent::RedrawRequested => {
                // Check for fatal render errors
                if self.render_fatal_error {
                    log::error!("Fatal render error occurred, exiting");
                    event_loop.exit();
                    return;
                }
                
                #[cfg(feature = "render_timing")]
                let frame_start = std::time::Instant::now();
                self.frame_count += 1;
                
                if self.last_frame_log.elapsed() >= Duration::from_secs(1) {
                    log::debug!("FPS: {}", self.frame_count);
                    self.frame_count = 0;
                    self.last_frame_log = std::time::Instant::now();
                }
                
                // Use shared render logic
                let needs_another_frame = self.do_render();
                
                // If animations are in progress, schedule another render
                if needs_another_frame {
                    self.request_redraw();
                }
                
                // Log cumulative stats every second (only with render_timing feature)
                #[cfg(feature = "render_timing")]
                if self.last_stats_log.elapsed() >= Duration::from_secs(1) {
                    let parse_ms = self.total_parse_ns as f64 / 1_000_000.0;
                    let render_ms = self.total_render_ns as f64 / 1_000_000.0;
                    log::info!("STATS: parse={:.1}ms/{} render={:.1}ms/{} ratio={:.2}",
                        parse_ms, self.parse_count,
                        render_ms, self.render_count,
                        if parse_ms > 0.0 { render_ms / parse_ms } else { 0.0 });
                    self.total_parse_ns = 0;
                    self.total_render_ns = 0;
                    self.parse_count = 0;
                    self.render_count = 0;
                    self.last_stats_log = std::time::Instant::now();
                }
                
                #[cfg(feature = "render_timing")]
                {
                    let frame_time = frame_start.elapsed();
                    if frame_time.as_millis() > 16 {
                        log::info!("Slow frame: {:?}", frame_time);
                    }
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
        let mut tabs_removed = false;
        while i < self.tabs.len() {
            if self.tabs[i].child_exited() {
                log::info!("Tab {} shell exited", i);
                self.tabs.remove(i);
                tabs_removed = true;
                if self.active_tab >= self.tabs.len() && !self.tabs.is_empty() {
                    self.active_tab = self.tabs.len() - 1;
                }
            } else {
                i += 1;
            }
        }
        
        // Update grid dimensions if tabs were removed
        if tabs_removed && !self.tabs.is_empty() {
            self.update_active_tab_grid_dimensions();
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

/// Set up a file watcher to monitor the config file for changes.
/// Returns the watcher (must be kept alive for watching to continue).
fn setup_config_watcher(proxy: EventLoopProxy<UserEvent>) -> Option<RecommendedWatcher> {
    let config_path = match Config::config_path() {
        Some(path) => path,
        None => {
            log::warn!("Could not determine config path, config hot-reload disabled");
            return None;
        }
    };
    
    // Watch the parent directory since the file might be replaced atomically
    let watch_path = match config_path.parent() {
        Some(parent) => parent.to_path_buf(),
        None => {
            log::warn!("Could not determine config directory, config hot-reload disabled");
            return None;
        }
    };
    
    let config_filename = config_path.file_name().map(|s| s.to_os_string());
    
    let mut watcher = match notify::recommended_watcher(move |res: Result<notify::Event, notify::Error>| {
        match res {
            Ok(event) => {
                // Only trigger on modify/create events for the config file
                use notify::EventKind;
                match event.kind {
                    EventKind::Modify(_) | EventKind::Create(_) => {
                        // Check if the event is for our config file
                        let is_config_file = event.paths.iter().any(|p| {
                            p.file_name().map(|s| s.to_os_string()) == config_filename
                        });
                        
                        if is_config_file {
                            log::debug!("Config file changed, triggering reload");
                            let _ = proxy.send_event(UserEvent::ConfigReloaded);
                        }
                    }
                    _ => {}
                }
            }
            Err(e) => {
                log::warn!("Config watcher error: {:?}", e);
            }
        }
    }) {
        Ok(w) => w,
        Err(e) => {
            log::warn!("Failed to create config watcher: {:?}", e);
            return None;
        }
    };
    
    if let Err(e) = watcher.watch(&watch_path, RecursiveMode::NonRecursive) {
        log::warn!("Failed to watch config directory {:?}: {:?}", watch_path, e);
        return None;
    }
    
    log::info!("Config hot-reload enabled, watching {:?}", watch_path);
    Some(watcher)
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
        EVENT_PROXY = Some(proxy.clone());
    }

    // Set up config file watcher for hot-reloading
    let _config_watcher = setup_config_watcher(proxy);

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
