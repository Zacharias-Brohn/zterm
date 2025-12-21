# Terminal.rs Reorganization Plan

This document identifies sections of `src/terminal.rs` (2336 lines) that could be extracted into separate files to improve code organization, maintainability, and testability.

---

## Summary of Proposed Extractions

| New File | Lines | Primary Contents |
|----------|-------|------------------|
| `src/cell.rs` | ~90 | `Cell`, `Color`, `CursorShape` |
| `src/color.rs` | ~115 | `ColorPalette` and color parsing |
| `src/mouse.rs` | ~110 | `MouseTrackingMode`, `MouseEncoding`, `encode_mouse()` |
| `src/scrollback.rs` | ~100 | `ScrollbackBuffer` ring buffer |
| `src/terminal_commands.rs` | ~30 | `TerminalCommand`, `Direction` |
| `src/stats.rs` | ~45 | `ProcessingStats` |

**Total extractable: ~490 lines** (reducing terminal.rs by ~21%)

---

## 1. Cell Types and Color Enum

**Lines:** 30-87 (Cell, Color, CursorShape)

**Contents:**
- `Cell` struct (lines 31-45) - terminal grid cell with character and attributes
- `impl Default for Cell` (lines 47-60)
- `Color` enum (lines 63-69) - `Default`, `Rgb(u8, u8, u8)`, `Indexed(u8)`
- `CursorShape` enum (lines 72-87) - cursor style variants

**Proposed file:** `src/cell.rs`

**Dependencies:**
```rust
// No external dependencies - these are self-contained types
```

**Exports needed by terminal.rs:**
```rust
pub use cell::{Cell, Color, CursorShape};
```

**Challenges:**
- None. These are simple data types with no logic dependencies.

**Benefits:**
- `Cell` and `Color` are referenced by renderer and could be imported directly
- Makes the core data structures discoverable
- Easy to test color conversion independently

---

## 2. Color Palette

**Lines:** 119-240

**Contents:**
- `ColorPalette` struct (lines 120-128) - 256-color palette storage
- `impl Default for ColorPalette` (lines 130-177) - ANSI + 216 color cube + grayscale
- `ColorPalette::parse_color_spec()` (lines 181-209) - parse `#RRGGBB` and `rgb:RR/GG/BB`
- `ColorPalette::to_rgba()` (lines 212-224) - foreground color conversion
- `ColorPalette::to_rgba_bg()` (lines 227-239) - background color conversion

**Proposed file:** `src/color.rs`

**Dependencies:**
```rust
use crate::cell::Color;  // For Color enum
```

**Challenges:**
- Depends on `Color` enum (extract cell.rs first)
- `to_rgba()` and `to_rgba_bg()` are called by the renderer

**Benefits:**
- Color parsing logic is self-contained and testable
- Palette initialization is complex (color cube math) and benefits from isolation
- Could add color scheme loading from config files in the future

---

## 3. Mouse Tracking and Encoding

**Lines:** 89-117, 962-1059

**Contents:**
- `MouseTrackingMode` enum (lines 90-103)
- `MouseEncoding` enum (lines 106-117)
- `Terminal::encode_mouse()` method (lines 964-1059) - encode mouse events for PTY

**Proposed file:** `src/mouse.rs`

**Dependencies:**
```rust
// MouseTrackingMode and MouseEncoding are self-contained enums
// encode_mouse() would need to be a standalone function or trait
```

**Refactoring approach:**
```rust
// In mouse.rs
pub fn encode_mouse_event(
    tracking: MouseTrackingMode,
    encoding: MouseEncoding,
    button: u8,
    col: u16,
    row: u16,
    pressed: bool,
    is_motion: bool,
    modifiers: u8,
) -> Vec<u8>
```

**Challenges:**
- `encode_mouse()` is currently a method on `Terminal` but only reads `mouse_tracking` and `mouse_encoding`
- Need to change call sites to pass mode/encoding explicitly, OR keep as Terminal method but move enums

**Benefits:**
- Mouse protocol logic is self-contained and well-documented
- Could add unit tests for X10, SGR, URXVT encoding without instantiating Terminal

---

## 4. Scrollback Buffer

**Lines:** 314-425

**Contents:**
- `ScrollbackBuffer` struct (lines 326-335) - Kitty-style ring buffer
- `ScrollbackBuffer::new()` (lines 340-351) - lazy allocation
- `ScrollbackBuffer::len()`, `is_empty()`, `is_full()` (lines 354-369)
- `ScrollbackBuffer::push()` (lines 377-403) - O(1) ring buffer insertion
- `ScrollbackBuffer::get()` (lines 407-415) - logical index access
- `ScrollbackBuffer::clear()` (lines 419-424)

**Proposed file:** `src/scrollback.rs`

**Dependencies:**
```rust
use crate::cell::Cell;  // For Vec<Cell> line storage
```

**Challenges:**
- Depends on `Cell` type (extract cell.rs first)
- Otherwise completely self-contained with great documentation

**Benefits:**
- Ring buffer is a reusable data structure
- Excellent candidate for unit testing (push, wrap-around, get by index)
- Performance-critical code that benefits from isolation for profiling

---

## 5. Terminal Commands

**Lines:** 8-28

**Contents:**
- `TerminalCommand` enum (lines 10-19) - commands sent from terminal to application
- `Direction` enum (lines 22-28) - navigation direction

**Proposed file:** `src/terminal_commands.rs`

**Dependencies:**
```rust
// No dependencies - self-contained enums
```

**Challenges:**
- Very small extraction, but cleanly separates protocol from implementation

**Benefits:**
- Defines the terminal-to-application interface
- Could grow as more custom OSC commands are added
- Clear documentation of what commands exist

---

## 6. Processing Stats

**Lines:** 268-312

**Contents:**
- `ProcessingStats` struct (lines 269-288) - timing/debugging statistics
- `ProcessingStats::reset()` (lines 291-293)
- `ProcessingStats::log_if_slow()` (lines 295-311) - conditional performance logging

**Proposed file:** `src/stats.rs`

**Dependencies:**
```rust
// No dependencies - uses only log crate
```

**Challenges:**
- Only used for debugging/profiling
- Could be feature-gated behind a `profiling` feature flag

**Benefits:**
- Separates debug instrumentation from core logic
- Could be conditionally compiled out for release builds

---

## 7. Saved Cursor and Alternate Screen (Keep in terminal.rs)

**Lines:** 243-265

**Contents:**
- `SavedCursor` struct (lines 243-253)
- `AlternateScreen` struct (lines 256-265)

**Recommendation:** Keep in terminal.rs

**Rationale:**
- These are private implementation details of cursor save/restore and alternate screen
- Tightly coupled to Terminal's internal state
- No benefit from extraction

---

## 8. Handler Implementation (Keep in terminal.rs)

**Lines:** 1236-1904, 1906-2335

**Contents:**
- `impl Handler for Terminal` - VT parser callback implementations
- CSI handling, SGR parsing, DEC private modes
- Keyboard protocol, OSC handling, graphics protocol

**Recommendation:** Keep in terminal.rs (or consider splitting if file grows further)

**Rationale:**
- These are the core terminal emulation callbacks
- Heavily intertwined with Terminal's internal state
- Could potentially split into `terminal_csi.rs`, `terminal_sgr.rs`, etc. but adds complexity

---

## Recommended Extraction Order

1. **`src/cell.rs`** - No dependencies, foundational types
2. **`src/terminal_commands.rs`** - No dependencies, simple enums
3. **`src/stats.rs`** - No dependencies, debugging utility
4. **`src/color.rs`** - Depends on `cell.rs`, self-contained logic
5. **`src/scrollback.rs`** - Depends on `cell.rs`, self-contained data structure
6. **`src/mouse.rs`** - Self-contained enums, may need refactoring for encode function

---

## Example: cell.rs Implementation

```rust
//! Terminal cell and color types.

/// A single cell in the terminal grid.
#[derive(Clone, Copy, Debug)]
pub struct Cell {
    pub character: char,
    pub fg_color: Color,
    pub bg_color: Color,
    pub bold: bool,
    pub italic: bool,
    /// Underline style: 0=none, 1=single, 2=double, 3=curly, 4=dotted, 5=dashed
    pub underline_style: u8,
    /// Strikethrough decoration
    pub strikethrough: bool,
    /// If true, this cell is the continuation of a wide (double-width) character.
    pub wide_continuation: bool,
}

impl Default for Cell {
    fn default() -> Self {
        Self {
            character: ' ',
            fg_color: Color::Default,
            bg_color: Color::Default,
            bold: false,
            italic: false,
            underline_style: 0,
            strikethrough: false,
            wide_continuation: false,
        }
    }
}

/// Terminal colors.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Color {
    #[default]
    Default,
    Rgb(u8, u8, u8),
    Indexed(u8),
}

/// Cursor shape styles (DECSCUSR).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum CursorShape {
    #[default]
    BlinkingBlock,
    SteadyBlock,
    BlinkingUnderline,
    SteadyUnderline,
    BlinkingBar,
    SteadyBar,
}
```

---

## Impact on lib.rs

After extraction, `lib.rs` would need:

```rust
pub mod cell;
pub mod color;
pub mod mouse;
pub mod scrollback;
pub mod stats;
pub mod terminal;
pub mod terminal_commands;
// ... existing modules ...
```

And `terminal.rs` would add:

```rust
use crate::cell::{Cell, Color, CursorShape};
use crate::color::ColorPalette;
use crate::mouse::{MouseEncoding, MouseTrackingMode};
use crate::scrollback::ScrollbackBuffer;
use crate::stats::ProcessingStats;
use crate::terminal_commands::{Direction, TerminalCommand};
```

---

## Testing Opportunities

Extracting these modules enables focused unit tests:

- **cell.rs**: Default cell values, wide_continuation handling
- **color.rs**: `parse_color_spec()` for various formats, palette indexing, RGBA conversion
- **mouse.rs**: Encoding tests for X10, SGR, URXVT formats, tracking mode filtering
- **scrollback.rs**: Ring buffer push/get/wrap, capacity limits, clear behavior

---

## Notes

- The `Terminal` struct itself (lines 428-512) should remain in `terminal.rs` as the central state container
- Private helper structs like `SavedCursor` and `AlternateScreen` should stay in `terminal.rs`
- The `Handler` trait implementation spans ~600 lines but is core terminal logic
- Consider feature-gating `ProcessingStats` behind `#[cfg(feature = "profiling")]`
