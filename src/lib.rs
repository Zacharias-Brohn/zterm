//! ZTerm - A GPU-accelerated terminal emulator for Wayland.
//!
//! Single-process architecture: one process owns PTY, terminal state, and rendering.

pub mod box_drawing;
pub mod color;
pub mod color_font;
pub mod config;
pub mod font_loader;
pub mod edge_glow;
pub mod gpu_types;
pub mod graphics;
pub mod image_renderer;
pub mod keyboard;
pub mod pane_resources;
pub mod pipeline;
pub mod pty;
pub mod renderer;
pub mod statusline;
pub mod terminal;
pub mod simd_utf8;
pub mod vt_parser;
