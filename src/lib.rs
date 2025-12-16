//! ZTerm - A GPU-accelerated terminal emulator for Wayland.
//!
//! Single-process architecture: one process owns PTY, terminal state, and rendering.

pub mod config;
pub mod graphics;
pub mod keyboard;
pub mod pty;
pub mod renderer;
pub mod terminal;
pub mod vt_parser;
