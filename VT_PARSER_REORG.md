# VT Parser Reorganization Recommendations

This document analyzes `src/vt_parser.rs` (1033 lines) and identifies sections that could be extracted into separate files to improve code organization, testability, and maintainability.

## Current File Structure Overview

| Lines | Section | Description |
|-------|---------|-------------|
| 1-49 | Constants & UTF-8 Tables | Parser limits, UTF-8 DFA decode table |
| 51-133 | UTF-8 Decoder | `Utf8Decoder` struct and implementation |
| 135-265 | State & CSI Types | `State` enum, `CsiState` enum, `CsiParams` struct |
| 267-832 | Parser Core | Main `Parser` struct with all parsing logic |
| 835-906 | Handler Trait | `Handler` trait definition |
| 908-1032 | Tests | Unit tests |

---

## Recommended Extractions

### 1. UTF-8 Decoder Module

**File:** `src/utf8_decoder.rs`

**Lines:** 27-133

**Components:**
- `UTF8_ACCEPT`, `UTF8_REJECT` constants (lines 28-29)
- `UTF8_DECODE_TABLE` static (lines 33-49)
- `decode_utf8()` function (lines 52-62)
- `Utf8Decoder` struct and impl (lines 66-133)
- `REPLACEMENT_CHAR` constant (line 25)

**Dependencies:**
- None (completely self-contained)

**Rationale:**
- This is a completely standalone UTF-8 DFA decoder based on Bjoern Hoehrmann's design
- Zero dependencies on the rest of the parser
- Could be reused in other parts of the codebase (keyboard input, file parsing)
- Independently testable
- ~100 lines, a good size for a focused module

**Extraction Difficulty:** Easy

**Example structure:**
```rust
// src/utf8_decoder.rs
pub const REPLACEMENT_CHAR: char = '\u{FFFD}';

const UTF8_ACCEPT: u8 = 0;
const UTF8_REJECT: u8 = 12;

static UTF8_DECODE_TABLE: [u8; 364] = [ /* ... */ ];

#[inline]
fn decode_utf8(state: &mut u8, codep: &mut u32, byte: u8) -> u8 { /* ... */ }

#[derive(Debug, Default)]
pub struct Utf8Decoder { /* ... */ }

impl Utf8Decoder {
    pub fn new() -> Self { /* ... */ }
    pub fn reset(&mut self) { /* ... */ }
    pub fn decode_to_esc(&mut self, src: &[u8], output: &mut Vec<char>) -> (usize, bool) { /* ... */ }
}
```

---

### 2. CSI Parameters Module

**File:** `src/csi_params.rs`

**Lines:** 14-265 (constants and CSI-related types)

**Components:**
- `MAX_CSI_PARAMS` constant (line 15)
- `CsiState` enum (lines 165-171)
- `CsiParams` struct and impl (lines 174-265)

**Dependencies:**
- None (self-contained data structure)

**Rationale:**
- `CsiParams` is a self-contained data structure for CSI parameter parsing
- Has its own sub-state machine (`CsiState`)
- The struct is 2KB+ in size due to the arrays - isolating it makes the size impact clearer
- Could be tested independently for parameter parsing edge cases
- The `get()`, `add_digit()`, `commit_param()` methods form a cohesive unit

**Extraction Difficulty:** Easy

**Note:** `CsiState` is currently private and only used within CSI parsing. It should remain private to the module.

---

### 3. Handler Trait Module

**File:** `src/vt_handler.rs`

**Lines:** 835-906

**Components:**
- `Handler` trait (lines 840-906)
- `CsiParams` would need to be re-exported or the trait would depend on `csi_params` module

**Dependencies:**
- `CsiParams` type (for `csi()` method signature)

**Rationale:**
- Clear separation between the parser implementation and the callback interface
- Makes it easier for consumers to implement handlers without pulling in parser internals
- Trait documentation is substantial and benefits from its own file
- Allows different modules to implement handlers without circular dependencies

**Extraction Difficulty:** Easy (after `CsiParams` is extracted)

---

### 4. Parser Constants Module

**File:** `src/vt_constants.rs` (or inline in a `mod.rs` approach)

**Lines:** 14-25

**Components:**
- `MAX_CSI_PARAMS` (already mentioned above)
- `MAX_OSC_LEN` (line 19)
- `MAX_ESCAPE_LEN` (line 22)
- `REPLACEMENT_CHAR` (line 25, if not moved to utf8_decoder)

**Dependencies:**
- None

**Rationale:**
- Centralizes magic numbers
- Easy to find and adjust limits
- However, these are only 4 constants, so this extraction is optional

**Extraction Difficulty:** Trivial

**Recommendation:** Keep these in the main parser file or move to a `mod.rs` if using a directory structure.

---

### 5. Parser State Enum

**File:** Could remain in `vt_parser.rs` or move to `vt_handler.rs`

**Lines:** 136-162

**Components:**
- `State` enum (lines 136-156)
- `Default` impl (lines 158-162)

**Dependencies:**
- None

**Rationale:**
- The `State` enum is public and part of the `Parser` struct
- It's tightly coupled with the parser's operation
- Small enough (~25 lines) to not warrant its own file

**Recommendation:** Keep in main parser file or combine with handler trait.

---

## Proposed Directory Structure

### Option A: Flat Module Structure (Recommended)

```
src/
  vt_parser.rs        # Main Parser struct, State enum, parsing logic (~700 lines)
  utf8_decoder.rs     # UTF-8 DFA decoder (~110 lines)
  csi_params.rs       # CsiParams struct and CsiState (~100 lines)
  vt_handler.rs       # Handler trait (~75 lines)
```

**lib.rs changes:**
```rust
mod utf8_decoder;
mod csi_params;
mod vt_handler;
mod vt_parser;

pub use vt_parser::{Parser, State};
pub use csi_params::{CsiParams, MAX_CSI_PARAMS};
pub use vt_handler::Handler;
```

### Option B: Directory Module Structure

```
src/
  vt_parser/
    mod.rs            # Re-exports and constants
    parser.rs         # Main Parser struct
    utf8.rs           # UTF-8 decoder
    csi.rs            # CSI params
    handler.rs        # Handler trait
    tests.rs          # Tests (optional, can stay inline)
```

---

## Extraction Priority

| Priority | Module | Lines Saved | Benefit |
|----------|--------|-------------|---------|
| 1 | `utf8_decoder.rs` | ~110 | Completely independent, reusable |
| 2 | `csi_params.rs` | ~100 | Clear data structure boundary |
| 3 | `vt_handler.rs` | ~75 | Cleaner API surface |
| 4 | Constants | ~10 | Optional, low impact |

---

## Challenges and Considerations

### 1. Test Organization
- Lines 908-1032 contain tests that use private test helpers (`TestHandler`)
- If the `Handler` trait is extracted, `TestHandler` could move to a test module
- Consider using `#[cfg(test)]` modules in each file

### 2. Circular Dependencies
- `Handler` trait references `CsiParams` - extract `CsiParams` first
- `Parser` uses both `Utf8Decoder` and `CsiParams` - both should be extracted before any handler extraction

### 3. Public API Surface
- Currently public: `MAX_CSI_PARAMS`, `State`, `CsiParams`, `Parser`, `Handler`, `Utf8Decoder`
- After extraction, ensure re-exports maintain the same public API

### 4. Performance Considerations
- The UTF-8 decoder uses `#[inline]` extensively - ensure this is preserved
- `CsiParams::reset()` is hot and optimized to avoid memset - document this

---

## Migration Steps

1. **Extract `utf8_decoder.rs`**
   - Move lines 25-133 to new file
   - Add `mod utf8_decoder;` to lib.rs
   - Update `vt_parser.rs` to `use crate::utf8_decoder::Utf8Decoder;`

2. **Extract `csi_params.rs`**
   - Move lines 14-15 (MAX_CSI_PARAMS) and 164-265 to new file
   - Make `CsiState` private to the module (`pub(crate)` at most)
   - Add `mod csi_params;` to lib.rs

3. **Extract `vt_handler.rs`**
   - Move lines 835-906 to new file
   - Add `use crate::csi_params::CsiParams;`
   - Add `mod vt_handler;` to lib.rs

4. **Update imports in `vt_parser.rs`**
   ```rust
   use crate::utf8_decoder::Utf8Decoder;
   use crate::csi_params::{CsiParams, CsiState, MAX_CSI_PARAMS};
   use crate::vt_handler::Handler;
   ```

5. **Verify public API unchanged**
   - Ensure lib.rs re-exports all previously public items
   - Run tests to verify nothing broke

---

## Code That Should Stay in `vt_parser.rs`

The following should remain in the main parser file:

- `State` enum (lines 136-162) - tightly coupled to parser
- `Parser` struct (lines 268-299) - core type
- All `Parser` methods (lines 301-832) - core parsing logic
- Constants `MAX_OSC_LEN`, `MAX_ESCAPE_LEN` (lines 19, 22) - parser-specific limits

After extraction, `vt_parser.rs` would be ~700 lines focused purely on the state machine and escape sequence parsing logic.

---

## Summary

The `vt_parser.rs` file has clear natural boundaries:

1. **UTF-8 decoding** - completely standalone, based on external algorithm
2. **CSI parameter handling** - self-contained data structure with its own state
3. **Handler trait** - defines the callback interface
4. **Core parser** - the state machine and escape sequence processing

Extracting the first three would reduce `vt_parser.rs` from 1033 lines to ~700 lines while improving:
- Code navigation
- Testability of individual components
- Reusability of the UTF-8 decoder
- API clarity (handler trait in its own file)
