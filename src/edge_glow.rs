//! Edge glow animation for visual feedback.
//!
//! Creates an organic glow effect when navigation fails: a single light node appears at center,
//! then splits into two that travel outward to the corners while fading.

use crate::terminal::Direction;

/// Edge glow animation state for visual feedback when navigation fails.
/// Creates an organic glow effect: a single light node appears at center,
/// then splits into two that travel outward to the corners while fading.
/// Animation logic is handled in the shader (shader.wgsl).
#[derive(Debug, Clone, Copy)]
pub struct EdgeGlow {
    /// Which edge to glow (based on the direction the user tried to navigate).
    pub direction: Direction,
    /// When the animation started.
    pub start_time: std::time::Instant,
    /// Pane bounds - left edge in pixels.
    pub pane_x: f32,
    /// Pane bounds - top edge in pixels.
    pub pane_y: f32,
    /// Pane bounds - width in pixels.
    pub pane_width: f32,
    /// Pane bounds - height in pixels.
    pub pane_height: f32,
}

impl EdgeGlow {
    /// Duration of the glow animation in milliseconds.
    pub const DURATION_MS: u64 = 500;

    /// Create a new edge glow animation constrained to a pane's bounds.
    pub fn new(direction: Direction, pane_x: f32, pane_y: f32, pane_width: f32, pane_height: f32) -> Self {
        Self {
            direction,
            start_time: std::time::Instant::now(),
            pane_x,
            pane_y,
            pane_width,
            pane_height,
        }
    }

    /// Get the current animation progress (0.0 to 1.0).
    pub fn progress(&self) -> f32 {
        let elapsed = self.start_time.elapsed().as_millis() as f32;
        let duration = Self::DURATION_MS as f32;
        (elapsed / duration).min(1.0)
    }

    /// Check if the animation has completed.
    pub fn is_finished(&self) -> bool {
        self.progress() >= 1.0
    }
}
