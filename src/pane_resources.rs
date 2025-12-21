//! Per-pane GPU resources for multi-pane terminal rendering.
//!
//! This module provides GPU resource management for individual terminal panes,
//! following Kitty's VAO-per-window approach where each pane gets its own
//! buffers and bind group for independent rendering.

// ═══════════════════════════════════════════════════════════════════════════════
// PER-PANE GPU RESOURCES (Like Kitty's VAO per window)
// ═══════════════════════════════════════════════════════════════════════════════

/// GPU resources for a single pane.
/// Like Kitty's VAO, each pane gets its own buffers and bind group.
/// This allows uploading each pane's cell data independently before rendering.
pub struct PaneGpuResources {
    /// Cell storage buffer - contains GPUCell array for this pane's visible cells.
    pub cell_buffer: wgpu::Buffer,
    /// Grid parameters uniform buffer for this pane.
    pub grid_params_buffer: wgpu::Buffer,
    /// Bind group for instanced rendering (@group(1)) - references this pane's buffers.
    pub bind_group: wgpu::BindGroup,
    /// Buffer capacity (max cells) - used to detect when buffer needs resizing.
    pub capacity: usize,
}
