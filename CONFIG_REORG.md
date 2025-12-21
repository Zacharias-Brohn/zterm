# Config.rs Reorganization Analysis

This document analyzes `/src/config.rs` (438 lines) and identifies sections that could be extracted into separate files to improve code organization.

## Current Structure Overview

The file contains:
- `TabBarPosition` enum (lines 10-21)
- `Keybind` struct + parsing logic (lines 23-166)
- `Action` enum (lines 168-206)
- `Keybindings` struct + default impl (lines 208-321)
- `Config` struct + load/save logic (lines 323-437)

---

## Recommended Extractions

### 1. Keybind Parsing Module

**Proposed file:** `src/keybind.rs`

**Lines:** 23-166 (144 lines)

**Types involved:**
- `struct Keybind`
- `impl Keybind` (including `parse()` and `normalize_key_name()`)

**Current code:**
```rust
pub struct Keybind(pub String);

impl Keybind {
    pub fn parse(&self) -> Option<(bool, bool, bool, bool, String)> { ... }
    fn normalize_key_name(name: &str) -> Option<&'static str> { ... }
}
```

**Dependencies needed:**
- `serde::{Deserialize, Serialize}` (for derive macros)

**Why extract:**
- Self-contained parsing logic with no dependencies on other config types
- The `normalize_key_name` function is a substantial lookup table (70+ lines)
- Could be tested independently with unit tests for key parsing edge cases
- Reusable if keybinding logic is needed elsewhere (e.g., a config editor UI)

**Challenges:**
- None significant. This is a clean extraction.

**After extraction, config.rs would:**
```rust
pub use keybind::Keybind;
// or
mod keybind;
pub use keybind::Keybind;
```

---

### 2. Actions Module

**Proposed file:** `src/action.rs`

**Lines:** 168-206 (39 lines)

**Types involved:**
- `enum Action`

**Current code:**
```rust
pub enum Action {
    NewTab,
    NextTab,
    PrevTab,
    Tab1, Tab2, ... Tab9,
    SplitHorizontal,
    SplitVertical,
    ClosePane,
    FocusPaneUp, FocusPaneDown, FocusPaneLeft, FocusPaneRight,
    Copy,
    Paste,
}
```

**Dependencies needed:**
- `serde::{Deserialize, Serialize}`

**Why extract:**
- `Action` is a distinct concept used throughout the codebase (keyboard.rs likely references it)
- Decouples the "what can be done" from "how it's configured"
- Makes it easy to add new actions without touching config logic
- Could be extended with action metadata (description, default keybind, etc.)

**Challenges:**
- Small file on its own (39 lines). Could be bundled with `keybind.rs` as a combined `src/keybindings.rs` module.

**Alternative:** Combine with Keybindings into a single module (see option 3).

---

### 3. Combined Keybindings Module (Recommended)

**Proposed file:** `src/keybindings.rs`

**Lines:** 23-321 (299 lines)

**Types involved:**
- `struct Keybind` + impl
- `enum Action`
- `struct Keybindings` + impl (including `Default` and `build_action_map`)

**Dependencies needed:**
- `serde::{Deserialize, Serialize}`
- `std::collections::HashMap`

**Why extract:**
- These three types form a cohesive "keybindings subsystem"
- `Keybindings::build_action_map()` ties together `Keybind`, `Action`, and `Keybindings`
- Reduces config.rs from 438 lines to ~140 lines
- Clear separation: config.rs handles general settings, keybindings.rs handles input mapping

**What stays in config.rs:**
- `TabBarPosition` enum
- `Config` struct with `keybindings: Keybindings` field
- `Config::load()`, `Config::save()`, `Config::config_path()`

**After extraction, config.rs would:**
```rust
mod keybindings;
pub use keybindings::{Action, Keybind, Keybindings};
```

**Challenges:**
- Need to ensure `Keybindings` is re-exported for external use
- The `Keybindings` struct is embedded in `Config`, so it must remain `pub`

---

### 4. TabBarPosition Enum

**Proposed file:** Could stay in `config.rs` or move to `src/ui.rs` / `src/types.rs`

**Lines:** 10-21 (12 lines)

**Types involved:**
- `enum TabBarPosition`

**Why extract:**
- Very small (12 lines) - extraction may be overkill
- Could be grouped with other UI-related enums if more are added in the future

**Recommendation:** Keep in `config.rs` for now. Only extract if you add more UI-related configuration enums (e.g., `CursorStyle`, `ScrollbarPosition`, etc.).

---

## Recommended Module Structure

```
src/
  config.rs          # Config struct, load/save, TabBarPosition (~140 lines)
  keybindings.rs     # Keybind, Action, Keybindings (~300 lines)
  # or alternatively:
  keybindings/
    mod.rs           # Re-exports
    keybind.rs       # Keybind struct + parsing
    action.rs        # Action enum
    bindings.rs      # Keybindings struct
```

For a codebase of this size, the single `keybindings.rs` file is recommended over a subdirectory.

---

## Implementation Steps

### Step 1: Create `src/keybindings.rs`

1. Create new file `src/keybindings.rs`
2. Move lines 23-321 from config.rs
3. Add module header:
   ```rust
   //! Keybinding types and parsing for ZTerm.
   
   use serde::{Deserialize, Serialize};
   use std::collections::HashMap;
   ```
4. Ensure all types are `pub`

### Step 2: Update `src/config.rs`

1. Remove lines 23-321
2. Add at the top (after the module doc comment):
   ```rust
   mod keybindings;
   pub use keybindings::{Action, Keybind, Keybindings};
   ```
3. Keep existing imports that `Config` needs

### Step 3: Update `src/lib.rs` or `src/main.rs`

If `keybindings` types are used directly elsewhere, update the module declarations:
```rust
// In lib.rs, if keybindings needs to be public:
pub mod keybindings;
// Or re-export from config:
pub use config::{Action, Keybind, Keybindings};
```

### Step 4: Verify compilation

```bash
cargo check
cargo test
```

---

## Summary Table

| Section | Lines | New File | Priority | Effort |
|---------|-------|----------|----------|--------|
| Keybind + Action + Keybindings | 23-321 (299 lines) | `keybindings.rs` | **High** | Low |
| TabBarPosition | 10-21 (12 lines) | Keep in config.rs | Low | N/A |
| Config struct + impls | 323-437 (115 lines) | Keep in config.rs | N/A | N/A |

**Recommended action:** Extract the keybindings module as a single file. This provides the best balance of organization improvement vs. complexity.

---

## Benefits After Reorganization

1. **config.rs** becomes focused on:
   - Core configuration values (font, opacity, scrollback, etc.)
   - File I/O (load/save)
   - ~140 lines instead of 438

2. **keybindings.rs** becomes a focused module for:
   - Input parsing and normalization
   - Action definitions
   - Keybinding-to-action mapping
   - ~300 lines, highly cohesive

3. **Testing:** Keybinding parsing can be unit tested in isolation

4. **Future extensibility:**
   - Adding new actions: edit `keybindings.rs`
   - Adding new config options: edit `config.rs`
   - Clear separation of concerns
