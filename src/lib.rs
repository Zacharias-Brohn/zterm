//! ZTerm - A GPU-accelerated terminal emulator for Wayland.
//!
//! This library provides shared functionality between the daemon and client.

pub mod client;
pub mod config;
pub mod daemon;
pub mod keyboard;
pub mod protocol;
pub mod pty;
pub mod renderer;
pub mod session;
pub mod terminal;
pub mod window_state;
