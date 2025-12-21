# Code Reorganization Analysis

This document identifies sections of code that could be moved into separate files to improve code organization.

## Summary

| File | Lines | Assessment | Recommended Extractions |
|------|-------|------------|------------------------|
| `pty.rs` | 260 | Well-organized | None needed |
| `keyboard.rs` | 558 | Could benefit from split | 1 extraction recommended |
| `graphics.rs` | 1846 | Needs refactoring | 3-4 extractions recommended |

---

## pty.rs (260 lines)

### Assessment: Well-Organized - No Changes Needed

The file is focused and cohesive:
- `PtyError` enum (lines 10-28) - Error types for PTY operations
- `Pty` struct and impl (lines 31-250) - Core PTY functionality
- All code directly relates to PTY management

**Rationale for keeping as-is:**
- Single responsibility: pseudo-terminal handling
- Manageable size (260 lines)
- All components are tightly coupled
- No reusable utilities that other modules would need

---

## keyboard.rs (558 lines)

### Assessment: Could Benefit from Minor Refactoring

The file contains the Kitty keyboard protocol implementation with several logical sections.

### Recommended Extraction: None (Optional Minor Refactoring)

While the file has distinct sections, they are all tightly coupled around the keyboard protocol:

1. **Flags/Types** (lines 8-95): `KeyboardFlags`, `KeyEventType`, `Modifiers`
2. **FunctionalKey enum** (lines 96-195): Key code definitions
3. **KeyboardState** (lines 197-292): Protocol state management
4. **KeyEncoder** (lines 294-548): Key encoding logic

**Rationale for keeping as-is:**
- All components implement a single protocol specification
- `KeyEncoder` depends on `KeyboardState`, `Modifiers`, `FunctionalKey`
- The file is under 600 lines, which is manageable
- Splitting would require importing everything back together in practice

**Optional Consideration:**
If the codebase grows to support multiple keyboard protocols, consider:
- `keyboard/mod.rs` - Public API
- `keyboard/kitty.rs` - Kitty protocol implementation
- `keyboard/legacy.rs` - Legacy encoding (currently in `encode_legacy_*` methods)

---

## graphics.rs (1846 lines)

### Assessment: Needs Refactoring - Multiple Extractions Recommended

This file is too large and contains several distinct logical modules that could be separated.

### Extraction 1: Animation Module

**Location:** Lines 391-746 (animation-related code)

**Components to extract:**
- `AnimationState` enum (lines 707-717)
- `AnimationData` struct (lines 719-736)
- `AnimationFrame` struct (lines 738-745)
- `decode_gif()` function (lines 393-459)
- `decode_webm()` function (lines 461-646, when feature enabled)

**Suggested file:** `src/graphics/animation.rs`

**Dependencies:**
```rust
use std::io::{Cursor, Read};
use std::time::Instant;
use image::{codecs::gif::GifDecoder, AnimationDecoder};
use super::GraphicsError;
```

**Challenges:**
- `decode_webm` is feature-gated (`#[cfg(feature = "webm")]`)
- Need to re-export types from `graphics/mod.rs`

---

### Extraction 2: Graphics Protocol Types

**Location:** Lines 16-162, 648-789

**Components to extract:**
- `Action` enum (lines 17-34)
- `Format` enum (lines 36-48)
- `Transmission` enum (lines 50-62)
- `Compression` enum (lines 64-69)
- `DeleteTarget` enum (lines 71-92)
- `GraphicsCommand` struct (lines 94-162)
- `GraphicsError` enum (lines 648-673)
- `ImageData` struct (lines 675-705)
- `PlacementResult` struct (lines 747-758)
- `ImagePlacement` struct (lines 760-789)

**Suggested file:** `src/graphics/types.rs`

**Dependencies:**
```rust
use std::time::Instant;
use super::animation::{AnimationData, AnimationState};
```

**Challenges:**
- `GraphicsCommand` has methods that depend on decoding logic
- Consider keeping `GraphicsCommand::parse()` and `decode_*` methods in types.rs or a separate `parsing.rs`

---

### Extraction 3: Image Storage

**Location:** Lines 791-1807

**Components to extract:**
- `ImageStorage` struct and impl (lines 791-1807)
- `ChunkBuffer` struct (lines 808-813)

**Suggested file:** `src/graphics/storage.rs`

**Dependencies:**
```rust
use std::collections::HashMap;
use std::time::Instant;
use super::types::*;
use super::animation::*;
```

**Challenges:**
- This is the largest section (~1000 lines)
- Contains many handler methods (`handle_transmit`, `handle_put`, etc.)
- Could be further split into:
  - `storage.rs` - Core storage and simple operations
  - `handlers.rs` - Command handlers
  - `animation_handlers.rs` - Animation-specific handlers (lines 1026-1399)

---

### Extraction 4: Base64 Utility

**Location:** Lines 1809-1817

**Components to extract:**
- `base64_decode()` function

**Suggested file:** Could go in a general `src/utils.rs` or stay in graphics

**Dependencies:**
```rust
use base64::Engine;
use super::GraphicsError;
```

**Challenges:** Minimal - this is a simple utility function

---

### Recommended Graphics Module Structure

```
src/
  graphics/
    mod.rs          # Re-exports, module declarations
    types.rs        # Enums, structs, GraphicsCommand parsing
    animation.rs    # AnimationData, AnimationFrame, GIF/WebM decoding
    storage.rs      # ImageStorage, placement logic
    handlers.rs     # Command handlers (optional further split)
```

**mod.rs example:**
```rust
mod animation;
mod handlers;
mod storage;
mod types;

pub use animation::{AnimationData, AnimationFrame, AnimationState, decode_gif};
pub use storage::ImageStorage;
pub use types::*;

#[cfg(feature = "webm")]
pub use animation::decode_webm;
```

---

## Implementation Priority

1. **High Priority:** Split `graphics.rs` - it's nearly 1900 lines and hard to navigate
2. **Low Priority:** `keyboard.rs` is fine as-is but could be modularized if protocol support expands
3. **No Action:** `pty.rs` is well-organized

---

## Migration Notes

When splitting `graphics.rs`:

1. Start by creating `src/graphics/` directory
2. Move `graphics.rs` to `src/graphics/mod.rs` temporarily
3. Extract types first (fewest dependencies)
4. Extract animation module
5. Extract storage module
6. Update imports in `renderer.rs`, `terminal.rs`, and any other consumers
7. Run tests after each extraction to catch breakages

The tests at the bottom of `graphics.rs` (lines 1819-1845) should remain in `mod.rs` or be split into module-specific test files.
