//! Window state management for tabs and panes.
//!
//! The daemon maintains the full UI state including tabs, panes, and which is active.

use crate::protocol::{Direction, PaneId, PaneInfo, SessionId, SplitDirection, TabId, TabInfo, WindowState as ProtocolWindowState};
use crate::session::Session;
use std::collections::HashMap;

/// Check if two ranges overlap.
fn ranges_overlap(a_start: usize, a_end: usize, b_start: usize, b_end: usize) -> bool {
    a_start < b_end && b_start < a_end
}

/// A pane within a tab.
pub struct Pane {
    pub id: PaneId,
    pub session_id: SessionId,
    /// Position in cells (for future splits).
    pub x: usize,
    pub y: usize,
    /// Size in cells.
    pub cols: usize,
    pub rows: usize,
}

impl Pane {
    pub fn new(id: PaneId, session_id: SessionId, cols: usize, rows: usize) -> Self {
        Self {
            id,
            session_id,
            x: 0,
            y: 0,
            cols,
            rows,
        }
    }
    
    pub fn new_at(id: PaneId, session_id: SessionId, x: usize, y: usize, cols: usize, rows: usize) -> Self {
        Self {
            id,
            session_id,
            x,
            y,
            cols,
            rows,
        }
    }
    
    pub fn to_info(&self) -> PaneInfo {
        PaneInfo {
            id: self.id,
            session_id: self.session_id,
            x: self.x,
            y: self.y,
            cols: self.cols,
            rows: self.rows,
        }
    }
}

/// A tab containing one or more panes.
pub struct Tab {
    pub id: TabId,
    pub panes: Vec<Pane>,
    pub active_pane: usize,
}

impl Tab {
    pub fn new(id: TabId, pane: Pane) -> Self {
        Self {
            id,
            panes: vec![pane],
            active_pane: 0,
        }
    }
    
    pub fn active_pane(&self) -> Option<&Pane> {
        self.panes.get(self.active_pane)
    }
    
    pub fn to_info(&self) -> TabInfo {
        TabInfo {
            id: self.id,
            active_pane: self.active_pane,
            panes: self.panes.iter().map(|p| p.to_info()).collect(),
        }
    }
}

/// Manages all window state: tabs, panes, sessions.
pub struct WindowStateManager {
    /// All sessions, keyed by session ID.
    pub sessions: HashMap<SessionId, Session>,
    /// All tabs in order.
    pub tabs: Vec<Tab>,
    /// Index of the active tab.
    pub active_tab: usize,
    /// Terminal dimensions in cells.
    pub cols: usize,
    pub rows: usize,
    /// Next session ID to assign.
    next_session_id: SessionId,
    /// Next pane ID to assign.
    next_pane_id: PaneId,
    /// Next tab ID to assign.
    next_tab_id: TabId,
}

impl WindowStateManager {
    /// Creates a new window state manager with initial dimensions.
    pub fn new(cols: usize, rows: usize) -> Self {
        Self {
            sessions: HashMap::new(),
            tabs: Vec::new(),
            active_tab: 0,
            cols,
            rows,
            next_session_id: 0,
            next_pane_id: 0,
            next_tab_id: 0,
        }
    }
    
    /// Creates the initial tab with a single session.
    pub fn create_initial_tab(&mut self) -> Result<(), crate::pty::PtyError> {
        let session_id = self.next_session_id;
        self.next_session_id += 1;
        
        let session = Session::new(session_id, self.cols, self.rows)?;
        self.sessions.insert(session_id, session);
        
        let pane_id = self.next_pane_id;
        self.next_pane_id += 1;
        
        let pane = Pane::new(pane_id, session_id, self.cols, self.rows);
        
        let tab_id = self.next_tab_id;
        self.next_tab_id += 1;
        
        let tab = Tab::new(tab_id, pane);
        self.tabs.push(tab);
        
        Ok(())
    }
    
    /// Creates a new tab with a new session.
    pub fn create_tab(&mut self) -> Result<TabId, crate::pty::PtyError> {
        let session_id = self.next_session_id;
        self.next_session_id += 1;
        
        let session = Session::new(session_id, self.cols, self.rows)?;
        self.sessions.insert(session_id, session);
        
        let pane_id = self.next_pane_id;
        self.next_pane_id += 1;
        
        let pane = Pane::new(pane_id, session_id, self.cols, self.rows);
        
        let tab_id = self.next_tab_id;
        self.next_tab_id += 1;
        
        let tab = Tab::new(tab_id, pane);
        self.tabs.push(tab);
        
        // Switch to the new tab
        self.active_tab = self.tabs.len() - 1;
        
        Ok(tab_id)
    }
    
    /// Closes a tab and its sessions.
    pub fn close_tab(&mut self, tab_id: TabId) -> bool {
        if let Some(idx) = self.tabs.iter().position(|t| t.id == tab_id) {
            let tab = self.tabs.remove(idx);
            
            // Remove all sessions owned by this tab's panes
            for pane in &tab.panes {
                self.sessions.remove(&pane.session_id);
            }
            
            // Adjust active tab index
            if self.tabs.is_empty() {
                self.active_tab = 0;
            } else if self.active_tab >= self.tabs.len() {
                self.active_tab = self.tabs.len() - 1;
            }
            
            true
        } else {
            false
        }
    }
    
    /// Switches to a tab by ID.
    pub fn switch_tab(&mut self, tab_id: TabId) -> bool {
        if let Some(idx) = self.tabs.iter().position(|t| t.id == tab_id) {
            self.active_tab = idx;
            true
        } else {
            false
        }
    }
    
    /// Switches to the next tab (wrapping around).
    pub fn next_tab(&mut self) -> bool {
        if self.tabs.is_empty() {
            return false;
        }
        self.active_tab = (self.active_tab + 1) % self.tabs.len();
        true
    }
    
    /// Switches to the previous tab (wrapping around).
    pub fn prev_tab(&mut self) -> bool {
        if self.tabs.is_empty() {
            return false;
        }
        if self.active_tab == 0 {
            self.active_tab = self.tabs.len() - 1;
        } else {
            self.active_tab -= 1;
        }
        true
    }
    
    /// Switches to a tab by index (0-based).
    pub fn switch_tab_index(&mut self, index: usize) -> bool {
        if index < self.tabs.len() {
            self.active_tab = index;
            true
        } else {
            false
        }
    }
    
    /// Splits the active pane in the active tab.
    /// Returns (tab_id, new_pane_info) on success.
    pub fn split_pane(&mut self, direction: SplitDirection) -> Result<(TabId, PaneInfo), crate::pty::PtyError> {
        let tab = self.tabs.get_mut(self.active_tab)
            .ok_or_else(|| crate::pty::PtyError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "No active tab")))?;
        
        let tab_id = tab.id;
        let active_pane = tab.panes.get_mut(tab.active_pane)
            .ok_or_else(|| crate::pty::PtyError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "No active pane")))?;
        
        // Calculate new dimensions
        let (new_x, new_y, new_cols, new_rows, orig_cols, orig_rows) = match direction {
            SplitDirection::Horizontal => {
                // Split top/bottom: new pane goes below
                let half_rows = active_pane.rows / 2;
                let new_rows = active_pane.rows - half_rows;
                let new_y = active_pane.y + half_rows;
                (
                    active_pane.x,
                    new_y,
                    active_pane.cols,
                    new_rows,
                    active_pane.cols,
                    half_rows,
                )
            }
            SplitDirection::Vertical => {
                // Split left/right: new pane goes to the right
                let half_cols = active_pane.cols / 2;
                let new_cols = active_pane.cols - half_cols;
                let new_x = active_pane.x + half_cols;
                (
                    new_x,
                    active_pane.y,
                    new_cols,
                    active_pane.rows,
                    half_cols,
                    active_pane.rows,
                )
            }
        };
        
        // Update original pane dimensions
        let orig_session_id = active_pane.session_id;
        active_pane.cols = orig_cols;
        active_pane.rows = orig_rows;
        
        // Resize the original session
        if let Some(session) = self.sessions.get_mut(&orig_session_id) {
            session.resize(orig_cols, orig_rows);
        }
        
        // Create new session for the new pane
        let session_id = self.next_session_id;
        self.next_session_id += 1;
        
        let session = Session::new(session_id, new_cols, new_rows)?;
        self.sessions.insert(session_id, session);
        
        // Create new pane
        let pane_id = self.next_pane_id;
        self.next_pane_id += 1;
        
        let new_pane = Pane::new_at(pane_id, session_id, new_x, new_y, new_cols, new_rows);
        let pane_info = new_pane.to_info();
        
        // Add pane to tab and focus it
        let tab = self.tabs.get_mut(self.active_tab).unwrap();
        tab.panes.push(new_pane);
        tab.active_pane = tab.panes.len() - 1;
        
        Ok((tab_id, pane_info))
    }
    
    /// Closes the active pane in the active tab.
    /// Returns Some((tab_id, pane_id, tab_closed)) on success.
    /// If tab_closed is true, the tab was removed because it was the last pane.
    pub fn close_pane(&mut self) -> Option<(TabId, PaneId, bool)> {
        let tab = self.tabs.get_mut(self.active_tab)?;
        let tab_id = tab.id;
        
        if tab.panes.is_empty() {
            return None;
        }
        
        // Capture closed pane's geometry before removing
        let closed_pane = tab.panes.remove(tab.active_pane);
        let pane_id = closed_pane.id;
        let closed_x = closed_pane.x;
        let closed_y = closed_pane.y;
        let closed_cols = closed_pane.cols;
        let closed_rows = closed_pane.rows;
        
        // Remove the session
        self.sessions.remove(&closed_pane.session_id);
        
        // If this was the last pane, close the tab
        if tab.panes.is_empty() {
            self.tabs.remove(self.active_tab);
            
            // Adjust active tab index
            if !self.tabs.is_empty() && self.active_tab >= self.tabs.len() {
                self.active_tab = self.tabs.len() - 1;
            }
            
            return Some((tab_id, pane_id, true));
        }
        
        // Adjust active pane index
        if tab.active_pane >= tab.panes.len() {
            tab.active_pane = tab.panes.len() - 1;
        }
        
        // Recalculate pane layouts after closing - pass closed pane geometry
        self.recalculate_pane_layout_after_close(
            self.active_tab,
            closed_x,
            closed_y,
            closed_cols,
            closed_rows,
        );
        
        Some((tab_id, pane_id, false))
    }
    
    /// Focuses a pane in the given direction from the current pane.
    /// Returns (tab_id, new_active_pane_index) on success.
    pub fn focus_pane_direction(&mut self, direction: Direction) -> Option<(TabId, usize)> {
        let tab = self.tabs.get_mut(self.active_tab)?;
        let tab_id = tab.id;
        
        if tab.panes.len() <= 1 {
            return None;
        }
        
        let current_pane = tab.panes.get(tab.active_pane)?;
        
        // Current pane's bounding box and center
        let curr_x = current_pane.x;
        let curr_y = current_pane.y;
        let curr_right = curr_x + current_pane.cols;
        let curr_bottom = curr_y + current_pane.rows;
        let curr_center_x = curr_x + current_pane.cols / 2;
        let curr_center_y = curr_y + current_pane.rows / 2;
        
        let mut best_idx: Option<usize> = None;
        let mut best_score: Option<(i64, i64)> = None; // (negative_overlap, distance) - lower is better
        
        for (idx, pane) in tab.panes.iter().enumerate() {
            if idx == tab.active_pane {
                continue;
            }
            
            let pane_x = pane.x;
            let pane_y = pane.y;
            let pane_right = pane_x + pane.cols;
            let pane_bottom = pane_y + pane.rows;
            
            // Check if pane is in the correct direction
            let in_direction = match direction {
                Direction::Up => pane_bottom <= curr_y,
                Direction::Down => pane_y >= curr_bottom,
                Direction::Left => pane_right <= curr_x,
                Direction::Right => pane_x >= curr_right,
            };
            
            if !in_direction {
                continue;
            }
            
            // Calculate overlap on perpendicular axis and distance
            let (overlap, distance) = match direction {
                Direction::Up | Direction::Down => {
                    // Horizontal overlap
                    let overlap_start = curr_x.max(pane_x);
                    let overlap_end = curr_right.min(pane_right);
                    let overlap = if overlap_end > overlap_start {
                        (overlap_end - overlap_start) as i64
                    } else {
                        0
                    };
                    
                    // Vertical distance (edge to edge)
                    let dist = match direction {
                        Direction::Up => (curr_y as i64) - (pane_bottom as i64),
                        Direction::Down => (pane_y as i64) - (curr_bottom as i64),
                        _ => unreachable!(),
                    };
                    
                    (overlap, dist)
                }
                Direction::Left | Direction::Right => {
                    // Vertical overlap
                    let overlap_start = curr_y.max(pane_y);
                    let overlap_end = curr_bottom.min(pane_bottom);
                    let overlap = if overlap_end > overlap_start {
                        (overlap_end - overlap_start) as i64
                    } else {
                        0
                    };
                    
                    // Horizontal distance (edge to edge)
                    let dist = match direction {
                        Direction::Left => (curr_x as i64) - (pane_right as i64),
                        Direction::Right => (pane_x as i64) - (curr_right as i64),
                        _ => unreachable!(),
                    };
                    
                    (overlap, dist)
                }
            };
            
            // Score: prefer more overlap (so negate it), then prefer closer distance
            let score = (-overlap, distance);
            
            if best_score.is_none() || score < best_score.unwrap() {
                best_score = Some(score);
                best_idx = Some(idx);
            }
        }
        
        // If no exact directional match, try a fallback: find the nearest pane
        // in the general direction (allowing for some tolerance)
        if best_idx.is_none() {
            for (idx, pane) in tab.panes.iter().enumerate() {
                if idx == tab.active_pane {
                    continue;
                }
                
                let pane_center_x = pane.x + pane.cols / 2;
                let pane_center_y = pane.y + pane.rows / 2;
                
                // Check if pane center is in the general direction
                let in_general_direction = match direction {
                    Direction::Up => pane_center_y < curr_center_y,
                    Direction::Down => pane_center_y > curr_center_y,
                    Direction::Left => pane_center_x < curr_center_x,
                    Direction::Right => pane_center_x > curr_center_x,
                };
                
                if !in_general_direction {
                    continue;
                }
                
                // Calculate distance from center to center
                let dx = (pane_center_x as i64) - (curr_center_x as i64);
                let dy = (pane_center_y as i64) - (curr_center_y as i64);
                let distance = dx * dx + dy * dy; // squared distance is fine for comparison
                
                let score = (0i64, distance); // no overlap bonus for fallback
                
                if best_score.is_none() || score < best_score.unwrap() {
                    best_score = Some(score);
                    best_idx = Some(idx);
                }
            }
        }
        
        if let Some(idx) = best_idx {
            tab.active_pane = idx;
            Some((tab_id, idx))
        } else {
            None
        }
    }
    
    /// Recalculates pane layout after a pane is closed.
    /// Expands neighboring panes to fill the closed pane's space.
    /// Prefers expanding smaller panes to balance the layout.
    fn recalculate_pane_layout_after_close(
        &mut self,
        tab_idx: usize,
        closed_x: usize,
        closed_y: usize,
        closed_cols: usize,
        closed_rows: usize,
    ) {
        let Some(tab) = self.tabs.get_mut(tab_idx) else {
            return;
        };
        
        // If only one pane remains, give it full size
        if tab.panes.len() == 1 {
            let pane = &mut tab.panes[0];
            pane.x = 0;
            pane.y = 0;
            pane.cols = self.cols;
            pane.rows = self.rows;
            
            if let Some(session) = self.sessions.get_mut(&pane.session_id) {
                session.resize(self.cols, self.rows);
            }
            return;
        }
        
        let closed_right = closed_x + closed_cols;
        let closed_bottom = closed_y + closed_rows;
        
        // Find all panes that perfectly match an edge (same width/height as closed pane)
        // These are candidates for absorbing the space.
        // We'll pick the smallest one to balance the layout.
        
        #[derive(Debug, Clone, Copy)]
        enum ExpandDirection {
            Left,   // pane is to the left, expand right
            Right,  // pane is to the right, expand left  
            Top,    // pane is above, expand down
            Bottom, // pane is below, expand up
        }
        
        let mut perfect_matches: Vec<(usize, usize, ExpandDirection)> = Vec::new(); // (idx, area, direction)
        
        for (idx, pane) in tab.panes.iter().enumerate() {
            let pane_right = pane.x + pane.cols;
            let pane_bottom = pane.y + pane.rows;
            let area = pane.cols * pane.rows;
            
            // Left neighbor with exact same height
            if pane_right == closed_x && pane.y == closed_y && pane.rows == closed_rows {
                perfect_matches.push((idx, area, ExpandDirection::Left));
            }
            
            // Right neighbor with exact same height
            if pane.x == closed_right && pane.y == closed_y && pane.rows == closed_rows {
                perfect_matches.push((idx, area, ExpandDirection::Right));
            }
            
            // Top neighbor with exact same width
            if pane_bottom == closed_y && pane.x == closed_x && pane.cols == closed_cols {
                perfect_matches.push((idx, area, ExpandDirection::Top));
            }
            
            // Bottom neighbor with exact same width
            if pane.y == closed_bottom && pane.x == closed_x && pane.cols == closed_cols {
                perfect_matches.push((idx, area, ExpandDirection::Bottom));
            }
        }
        
        // If we have perfect matches, pick the smallest pane
        if !perfect_matches.is_empty() {
            // Sort by area (smallest first)
            perfect_matches.sort_by_key(|(_, area, _)| *area);
            let (idx, _, direction) = perfect_matches[0];
            
            let pane = &mut tab.panes[idx];
            match direction {
                ExpandDirection::Left => {
                    // Pane is to the left, expand right
                    pane.cols += closed_cols;
                }
                ExpandDirection::Right => {
                    // Pane is to the right, expand left
                    pane.x = closed_x;
                    pane.cols += closed_cols;
                }
                ExpandDirection::Top => {
                    // Pane is above, expand down
                    pane.rows += closed_rows;
                }
                ExpandDirection::Bottom => {
                    // Pane is below, expand up
                    pane.y = closed_y;
                    pane.rows += closed_rows;
                }
            }
            
            if let Some(session) = self.sessions.get_mut(&pane.session_id) {
                session.resize(pane.cols, pane.rows);
            }
            return;
        }
        
        // No perfect match - need to expand multiple panes.
        // Determine which direction has the most coverage and expand all panes on that edge.
        
        // Calculate coverage for each edge direction
        let mut bottom_neighbors: Vec<usize> = Vec::new(); // panes below closed pane
        let mut top_neighbors: Vec<usize> = Vec::new();    // panes above closed pane
        let mut right_neighbors: Vec<usize> = Vec::new();  // panes to the right
        let mut left_neighbors: Vec<usize> = Vec::new();   // panes to the left
        
        let mut bottom_coverage = 0usize;
        let mut top_coverage = 0usize;
        let mut right_coverage = 0usize;
        let mut left_coverage = 0usize;
        
        for (idx, pane) in tab.panes.iter().enumerate() {
            let pane_right = pane.x + pane.cols;
            let pane_bottom = pane.y + pane.rows;
            
            // Bottom neighbors: their top edge touches closed pane's bottom edge
            if pane.y == closed_bottom {
                let overlap_start = pane.x.max(closed_x);
                let overlap_end = pane_right.min(closed_right);
                if overlap_end > overlap_start {
                    bottom_neighbors.push(idx);
                    bottom_coverage += overlap_end - overlap_start;
                }
            }
            
            // Top neighbors: their bottom edge touches closed pane's top edge
            if pane_bottom == closed_y {
                let overlap_start = pane.x.max(closed_x);
                let overlap_end = pane_right.min(closed_right);
                if overlap_end > overlap_start {
                    top_neighbors.push(idx);
                    top_coverage += overlap_end - overlap_start;
                }
            }
            
            // Right neighbors: their left edge touches closed pane's right edge
            if pane.x == closed_right {
                let overlap_start = pane.y.max(closed_y);
                let overlap_end = pane_bottom.min(closed_bottom);
                if overlap_end > overlap_start {
                    right_neighbors.push(idx);
                    right_coverage += overlap_end - overlap_start;
                }
            }
            
            // Left neighbors: their right edge touches closed pane's left edge
            if pane_right == closed_x {
                let overlap_start = pane.y.max(closed_y);
                let overlap_end = pane_bottom.min(closed_bottom);
                if overlap_end > overlap_start {
                    left_neighbors.push(idx);
                    left_coverage += overlap_end - overlap_start;
                }
            }
        }
        
        // For partial matches, prefer the side with smaller total area (to balance layout)
        // Calculate total area for each side
        let bottom_area: usize = bottom_neighbors.iter()
            .map(|&idx| tab.panes[idx].cols * tab.panes[idx].rows)
            .sum();
        let top_area: usize = top_neighbors.iter()
            .map(|&idx| tab.panes[idx].cols * tab.panes[idx].rows)
            .sum();
        let right_area: usize = right_neighbors.iter()
            .map(|&idx| tab.panes[idx].cols * tab.panes[idx].rows)
            .sum();
        let left_area: usize = left_neighbors.iter()
            .map(|&idx| tab.panes[idx].cols * tab.panes[idx].rows)
            .sum();
        
        // Build candidates: (neighbors, coverage, total_area)
        let mut candidates: Vec<(&Vec<usize>, usize, usize, &str)> = Vec::new();
        if !bottom_neighbors.is_empty() {
            candidates.push((&bottom_neighbors, bottom_coverage, bottom_area, "bottom"));
        }
        if !top_neighbors.is_empty() {
            candidates.push((&top_neighbors, top_coverage, top_area, "top"));
        }
        if !right_neighbors.is_empty() {
            candidates.push((&right_neighbors, right_coverage, right_area, "right"));
        }
        if !left_neighbors.is_empty() {
            candidates.push((&left_neighbors, left_coverage, left_area, "left"));
        }
        
        if candidates.is_empty() {
            return;
        }
        
        // Sort by: coverage (descending), then area (ascending - prefer smaller)
        candidates.sort_by(|a, b| {
            b.1.cmp(&a.1) // coverage descending
                .then_with(|| a.2.cmp(&b.2)) // area ascending
        });
        
        let (neighbors, _, _, direction) = candidates[0];
        
        // Collect session IDs to resize after modifying panes
        let mut sessions_to_resize: Vec<(SessionId, usize, usize)> = Vec::new();
        
        match direction {
            "bottom" => {
                for &idx in neighbors {
                    let pane = &mut tab.panes[idx];
                    pane.y = closed_y;
                    pane.rows += closed_rows;
                    sessions_to_resize.push((pane.session_id, pane.cols, pane.rows));
                }
            }
            "top" => {
                for &idx in neighbors {
                    let pane = &mut tab.panes[idx];
                    pane.rows += closed_rows;
                    sessions_to_resize.push((pane.session_id, pane.cols, pane.rows));
                }
            }
            "right" => {
                for &idx in neighbors {
                    let pane = &mut tab.panes[idx];
                    pane.x = closed_x;
                    pane.cols += closed_cols;
                    sessions_to_resize.push((pane.session_id, pane.cols, pane.rows));
                }
            }
            "left" => {
                for &idx in neighbors {
                    let pane = &mut tab.panes[idx];
                    pane.cols += closed_cols;
                    sessions_to_resize.push((pane.session_id, pane.cols, pane.rows));
                }
            }
            _ => {}
        }
        
        // Resize all affected sessions
        for (session_id, cols, rows) in sessions_to_resize {
            if let Some(session) = self.sessions.get_mut(&session_id) {
                session.resize(cols, rows);
            }
        }
    }
    
    /// Gets the currently active tab.
    pub fn active_tab(&self) -> Option<&Tab> {
        self.tabs.get(self.active_tab)
    }
    
    /// Gets the currently active tab mutably.
    pub fn active_tab_mut(&mut self) -> Option<&mut Tab> {
        self.tabs.get_mut(self.active_tab)
    }
    
    /// Gets the currently focused session (active pane of active tab).
    pub fn focused_session(&self) -> Option<&Session> {
        self.active_tab()
            .and_then(|tab| tab.active_pane())
            .and_then(|pane| self.sessions.get(&pane.session_id))
    }
    
    /// Gets the currently focused session mutably.
    pub fn focused_session_mut(&mut self) -> Option<&mut Session> {
        let session_id = self.active_tab()
            .and_then(|tab| tab.active_pane())
            .map(|pane| pane.session_id)?;
        self.sessions.get_mut(&session_id)
    }
    
    /// Resizes all sessions to new dimensions.
    /// Recalculates pane layouts to maintain proper split ratios.
    pub fn resize(&mut self, cols: usize, rows: usize) {
        let old_cols = self.cols;
        let old_rows = self.rows;
        self.cols = cols;
        self.rows = rows;
        
        if old_cols == 0 || old_rows == 0 || cols == 0 || rows == 0 {
            return;
        }
        
        for tab in &mut self.tabs {
            if tab.panes.is_empty() {
                continue;
            }
            
            // Single pane: just give it full size
            if tab.panes.len() == 1 {
                let pane = &mut tab.panes[0];
                pane.x = 0;
                pane.y = 0;
                pane.cols = cols;
                pane.rows = rows;
                continue;
            }
            
            // Multiple panes: convert to ratios, then back to cells
            // This preserves the relative split positions
            
            // First, convert each pane's geometry to ratios (0.0 - 1.0)
            let ratios: Vec<(f64, f64, f64, f64)> = tab.panes.iter().map(|pane| {
                let x_ratio = pane.x as f64 / old_cols as f64;
                let y_ratio = pane.y as f64 / old_rows as f64;
                let w_ratio = pane.cols as f64 / old_cols as f64;
                let h_ratio = pane.rows as f64 / old_rows as f64;
                (x_ratio, y_ratio, w_ratio, h_ratio)
            }).collect();
            
            // Convert back to cell positions with new dimensions
            for (pane, (x_ratio, y_ratio, w_ratio, h_ratio)) in tab.panes.iter_mut().zip(ratios.iter()) {
                pane.x = (x_ratio * cols as f64).round() as usize;
                pane.y = (y_ratio * rows as f64).round() as usize;
                pane.cols = (w_ratio * cols as f64).round() as usize;
                pane.rows = (h_ratio * rows as f64).round() as usize;
                
                // Ensure minimum size
                pane.cols = pane.cols.max(1);
                pane.rows = pane.rows.max(1);
            }
            
            // Fix gaps and overlaps by adjusting panes that share edges
            // For each pair of adjacent panes, ensure they meet exactly
            Self::fix_pane_edges(&mut tab.panes, cols, rows);
        }
        
        // Resize all sessions to match their pane sizes
        for tab in &self.tabs {
            for pane in &tab.panes {
                if let Some(session) = self.sessions.get_mut(&pane.session_id) {
                    session.resize(pane.cols, pane.rows);
                }
            }
        }
    }
    
    /// Fixes gaps and overlaps between panes after resize.
    /// Ensures adjacent panes meet exactly and edge panes extend to window boundaries.
    fn fix_pane_edges(panes: &mut [Pane], cols: usize, rows: usize) {
        let n = panes.len();
        if n == 0 {
            return;
        }
        
        // For each pane, check if it should extend to the window edge
        for pane in panes.iter_mut() {
            // If pane is at x=0, ensure it starts at 0
            if pane.x <= 1 {
                let old_right = pane.x + pane.cols;
                pane.x = 0;
                pane.cols = old_right; // maintain right edge position
            }
            
            // If pane is at y=0, ensure it starts at 0
            if pane.y <= 1 {
                let old_bottom = pane.y + pane.rows;
                pane.y = 0;
                pane.rows = old_bottom; // maintain bottom edge position
            }
        }
        
        // For each pair of panes, if they're adjacent, make them meet exactly
        for i in 0..n {
            for j in (i + 1)..n {
                // Get the two panes' boundaries
                let (i_right, i_bottom) = {
                    let p = &panes[i];
                    (p.x + p.cols, p.y + p.rows)
                };
                let (j_right, j_bottom) = {
                    let p = &panes[j];
                    (p.x + p.cols, p.y + p.rows)
                };
                let (i_x, i_y) = (panes[i].x, panes[i].y);
                let (j_x, j_y) = (panes[j].x, panes[j].y);
                
                // Check if j is to the right of i (vertical split)
                // i's right edge should meet j's left edge
                if i_right.abs_diff(j_x) <= 2 && 
                   ranges_overlap(i_y, i_bottom, j_y, j_bottom) {
                    // They should meet - adjust j's x to match i's right edge
                    let meet_point = i_right;
                    let j_old_right = j_right;
                    panes[j].x = meet_point;
                    panes[j].cols = j_old_right.saturating_sub(meet_point).max(1);
                }
                
                // Check if i is to the right of j
                if j_right.abs_diff(i_x) <= 2 &&
                   ranges_overlap(i_y, i_bottom, j_y, j_bottom) {
                    let meet_point = j_right;
                    let i_old_right = i_right;
                    panes[i].x = meet_point;
                    panes[i].cols = i_old_right.saturating_sub(meet_point).max(1);
                }
                
                // Check if j is below i (horizontal split)
                if i_bottom.abs_diff(j_y) <= 2 &&
                   ranges_overlap(i_x, i_right, j_x, j_right) {
                    let meet_point = i_bottom;
                    let j_old_bottom = j_bottom;
                    panes[j].y = meet_point;
                    panes[j].rows = j_old_bottom.saturating_sub(meet_point).max(1);
                }
                
                // Check if i is below j
                if j_bottom.abs_diff(i_y) <= 2 &&
                   ranges_overlap(i_x, i_right, j_x, j_right) {
                    let meet_point = j_bottom;
                    let i_old_bottom = i_bottom;
                    panes[i].y = meet_point;
                    panes[i].rows = i_old_bottom.saturating_sub(meet_point).max(1);
                }
            }
        }
        
        // Finally, extend edge panes to window boundaries
        for pane in panes.iter_mut() {
            let pane_right = pane.x + pane.cols;
            let pane_bottom = pane.y + pane.rows;
            
            // Extend to right edge if close
            if pane_right >= cols.saturating_sub(2) {
                pane.cols = cols.saturating_sub(pane.x).max(1);
            }
            
            // Extend to bottom edge if close
            if pane_bottom >= rows.saturating_sub(2) {
                pane.rows = rows.saturating_sub(pane.y).max(1);
            }
        }
    }
    
    /// Creates a protocol WindowState message.
    pub fn to_protocol(&self) -> ProtocolWindowState {
        ProtocolWindowState {
            tabs: self.tabs.iter().map(|t| t.to_info()).collect(),
            active_tab: self.active_tab,
            cols: self.cols,
            rows: self.rows,
        }
    }
    
    /// Returns whether any session has new output.
    pub fn any_dirty(&self) -> bool {
        self.sessions.values().any(|s| s.dirty)
    }
    
    /// Marks all sessions as clean.
    pub fn mark_all_clean(&mut self) {
        for session in self.sessions.values_mut() {
            session.mark_clean();
        }
    }
}
