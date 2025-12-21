# Main.rs Reorganization Plan

This document identifies sections of `src/main.rs` (2793 lines) that could be extracted into separate modules to improve code organization, maintainability, and testability.

## Summary of Proposed Extractions

| Module Name | Lines | Primary Contents |
|-------------|-------|------------------|
| `pty_buffer.rs` | ~180 | `SharedPtyBuffer`, `BufferState` |
| `pane.rs` | ~400 | `PaneId`, `Pane`, `PaneGeometry`, `SplitNode` |
| `tab.rs` | ~150 | `TabId`, `Tab` |
| `selection.rs` | ~45 | `CellPosition`, `Selection` |
| `statusline.rs` | ~220 | `GitStatus`, `build_cwd_section()`, `build_git_section()`, `get_git_status()` |
| `instance.rs` | ~45 | PID file management, `signal_existing_instance()` |

---

## 1. PTY Buffer Module (`pty_buffer.rs`)

**Lines: 30-227 (~197 lines)**

### Contents
- `PTY_BUF_SIZE` constant
- `SharedPtyBuffer` struct and impl
- `BufferState` struct
- `unsafe impl Sync/Send for SharedPtyBuffer`

### Description
This is a self-contained, zero-copy PTY I/O buffer implementation inspired by Kitty. It has no dependencies on other application-specific types and is purely concerned with efficient I/O buffering between an I/O thread and the main thread.

### Types/Functions to Extract
```rust
const PTY_BUF_SIZE: usize
struct BufferState
struct SharedPtyBuffer
impl SharedPtyBuffer
impl Drop for SharedPtyBuffer
unsafe impl Sync for SharedPtyBuffer
unsafe impl Send for SharedPtyBuffer
```

### Dependencies
- `std::cell::UnsafeCell`
- `std::sync::Mutex`
- `libc` (for `eventfd`, `read`, `write`, `close`)

### Challenges
- **Unsafe code**: Contains `UnsafeCell` and raw pointer manipulation. Must carefully preserve safety invariants.
- **libc dependency**: Uses Linux-specific `eventfd` syscalls.

### Recommendation
**High priority extraction.** This is a well-documented, self-contained component with clear boundaries. Would benefit from its own unit tests for buffer operations.

---

## 2. Pane Module (`pane.rs`)

**Lines: 229-691 (~462 lines)**

### Contents
- `PaneId` - unique identifier with atomic generation
- `Pane` - terminal + PTY + selection state
- `PaneGeometry` - pixel layout information
- `SplitNode` - tree structure for split pane layouts

### Types/Functions to Extract
```rust
struct PaneId
impl PaneId

struct Pane
impl Pane
  - new()
  - resize()
  - write_to_pty()
  - child_exited()
  - foreground_matches()
  - calculate_dim_factor()

struct PaneGeometry

enum SplitNode
impl SplitNode
  - leaf()
  - split()
  - layout()
  - find_geometry()
  - collect_geometries()
  - find_neighbor()
  - overlaps_horizontally()
  - overlaps_vertically()
  - remove_pane()
  - contains_pane()
  - split_pane()
```

### Dependencies
- `zterm::terminal::{Direction, Terminal, TerminalCommand, MouseTrackingMode}`
- `zterm::pty::Pty`
- `std::sync::Arc`
- `std::os::fd::AsRawFd`
- `SharedPtyBuffer` (from pty_buffer module)
- `Selection` (from selection module)

### Challenges
- **Cross-module dependencies**: `Pane` references `SharedPtyBuffer` and `Selection`, so those would need to be extracted first or extracted together.
- **`SplitNode` complexity**: The recursive tree structure has complex layout logic. Consider keeping `SplitNode` with `Pane` since they're tightly coupled.

### Recommendation
**High priority extraction.** The pane management is a distinct concern from the main application loop. This would make the split tree logic easier to test in isolation.

---

## 3. Tab Module (`tab.rs`)

**Lines: 693-872 (~180 lines)**

### Contents
- `TabId` - unique identifier with atomic generation
- `Tab` - collection of panes with a split tree

### Types/Functions to Extract
```rust
struct TabId
impl TabId

struct Tab
impl Tab
  - new()
  - active_pane() / active_pane_mut()
  - resize()
  - write_to_pty()
  - check_exited_panes()
  - split()
  - remove_pane()
  - close_active_pane()
  - focus_neighbor()
  - get_pane() / get_pane_mut()
  - collect_pane_geometries()
  - child_exited()
```

### Dependencies
- `std::collections::HashMap`
- `PaneId`, `Pane`, `PaneGeometry`, `SplitNode` (from pane module)
- `zterm::terminal::Direction`

### Challenges
- **Tight coupling with Pane module**: Tab is essentially a container for panes. Consider combining into a single `pane.rs` module or using `pane/mod.rs` with submodules.

### Recommendation
**Medium priority.** Could be combined with the pane module under `pane/mod.rs` with `pane/tab.rs` as a submodule, or kept as a separate `tab.rs`.

---

## 4. Selection Module (`selection.rs`)

**Lines: 1218-1259 (~42 lines)**

### Contents
- `CellPosition` - column/row position
- `Selection` - start/end positions for text selection

### Types/Functions to Extract
```rust
struct CellPosition
struct Selection
impl Selection
  - normalized()
  - to_screen_coords()
```

### Dependencies
- None (completely self-contained)

### Challenges
- **Very small**: Only 42 lines. May be too small to justify its own file.

### Recommendation
**Low priority as standalone.** Consider bundling with the pane module since selection is per-pane state, or creating a `types.rs` for small shared types.

---

## 5. Statusline Module (`statusline.rs`)

**Lines: 917-1216 (~300 lines)**

### Contents
- `build_cwd_section()` - creates CWD statusline section
- `GitStatus` struct - git repository state
- `get_git_status()` - queries git for repo status
- `build_git_section()` - creates git statusline section

### Types/Functions to Extract
```rust
fn build_cwd_section(cwd: &str) -> StatuslineSection
struct GitStatus
fn get_git_status(cwd: &str) -> Option<GitStatus>
fn build_git_section(cwd: &str) -> Option<StatuslineSection>
```

### Dependencies
- `zterm::renderer::{StatuslineComponent, StatuslineSection}`
- `std::process::Command` (for git commands)
- `std::env` (for HOME variable)

### Challenges
- **External process calls**: Uses `git` commands. Consider whether this should be async or cached.
- **Powerline icons**: Uses hardcoded Unicode codepoints (Nerd Font icons).

### Recommendation
**High priority extraction.** This is completely independent of the main application state. Would benefit from:
- Caching git status (it's currently queried every frame)
- Unit tests for path transformation logic
- Potential async git queries

---

## 6. Instance Management Module (`instance.rs`)

**Lines: 874-915, 2780-2792 (~55 lines total)**

### Contents
- PID file path management
- Single-instance detection and signaling
- Signal handler for SIGUSR1

### Types/Functions to Extract
```rust
fn pid_file_path() -> PathBuf
fn signal_existing_instance() -> bool
fn write_pid_file() -> std::io::Result<()>
fn remove_pid_file()

// From end of file:
static mut EVENT_PROXY: Option<EventLoopProxy<UserEvent>>
extern "C" fn handle_sigusr1(_: i32)
```

### Dependencies
- `std::fs`
- `std::path::PathBuf`
- `libc` (for `kill` syscall)
- `winit::event_loop::EventLoopProxy`
- `UserEvent` enum

### Challenges
- **Global static**: The `EVENT_PROXY` static is `unsafe` and tightly coupled to the signal handler.
- **Split location**: The signal handler is at the end of the file, separate from PID functions.

### Recommendation
**Medium priority.** Small but distinct concern. The global static handling could be cleaner in a dedicated module.

---

## 7. Additional Observations

### UserEvent Enum (Line 1262-1270)
```rust
enum UserEvent {
    ShowWindow,
    PtyReadable(PaneId),
    ConfigReloaded,
}
```
This is a small enum but is referenced throughout. Consider placing in a `types.rs` or `events.rs` module.

### Config Watcher (Lines 2674-2735)
The `setup_config_watcher()` function is self-contained and could go in the instance module or a dedicated `config_watcher.rs`.

### App Struct (Lines 1272-1334)
The `App` struct and its impl are the core of the application and should remain in `main.rs`. However, some of its methods could potentially be split:
- I/O thread management (lines 1418-1553)
- Keyboard/keybinding handling (lines 1773-2221)
- Mouse handling (scattered through `window_event`)

### Keyboard Handling
Lines 1773-2221 contain significant keyboard handling logic. This could potentially be extracted, but it's tightly integrated with the `App` state.

---

## Suggested Module Structure

```
src/
  main.rs              (~1200 lines - App, ApplicationHandler, main())
  lib.rs               (existing)
  pty_buffer.rs        (new - ~200 lines)
  pane.rs              (new - ~500 lines, includes SplitNode)
  tab.rs               (new - ~180 lines)
  selection.rs         (new - ~45 lines, or merge with pane.rs)
  statusline.rs        (new - ~300 lines)
  instance.rs          (new - ~60 lines)
  config.rs            (existing)
  ...
```

Alternative with submodules:
```
src/
  main.rs
  lib.rs
  pane/
    mod.rs             (re-exports)
    pane.rs            (Pane, PaneId)
    split.rs           (SplitNode, PaneGeometry)
    selection.rs       (Selection, CellPosition)
  tab.rs
  pty_buffer.rs
  statusline.rs
  instance.rs
  ...
```

---

## Implementation Order

1. **`pty_buffer.rs`** - No internal dependencies, completely self-contained
2. **`selection.rs`** - No dependencies, simple extraction
3. **`statusline.rs`** - No internal dependencies, high value for testability
4. **`instance.rs`** - Small, isolated concern
5. **`pane.rs`** - Depends on pty_buffer and selection
6. **`tab.rs`** - Depends on pane

---

## Testing Opportunities

After extraction, these modules would benefit from unit tests:

| Module | Testable Functionality |
|--------|----------------------|
| `pty_buffer` | Buffer overflow handling, space checking, wakeup signaling |
| `selection` | `normalized()` ordering, `to_screen_coords()` boundary conditions |
| `statusline` | Path normalization (~/ replacement), git status parsing |
| `pane` / `SplitNode` | Layout calculations, neighbor finding, tree operations |
| `instance` | PID file creation/cleanup (integration test) |

---

## Notes on Maintaining Backward Compatibility

All extracted types should be re-exported from `lib.rs` or a prelude if they're used externally. The current architecture appears to be internal to the binary, so this is likely not a concern.

The `Pane` and `Tab` types are not part of the public API (defined in `main.rs`), so extraction won't affect external consumers.
