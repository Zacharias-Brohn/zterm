# ZTerm Redundancy Audit

Generated: 2025-12-20

## Summary

| File | Current Lines | Est. Savings | % Reduction |
|------|---------------|--------------|-------------|
| renderer.rs | 8426 | ~1000-1400 | 12-17% |
| terminal.rs | 2366 | ~150-180 | ~7% |
| main.rs | 2787 | ~150-180 | ~6% |
| vt_parser.rs | 1044 | ~60-90 | 6-9% |
| Other files | ~3000 | ~100 | ~3% |
| **Total** | **~17600** | **~1500-2000** | **~10%** |

---

## High Priority

### 1. renderer.rs: Pipeline Creation Boilerplate
**Lines:** 2200-2960 (~760 lines)  
**Estimated Savings:** 400-500 lines

**Problem:** Nearly identical pipeline creation code with slight variations in entry points, blend modes, and bind group layouts.

**Suggested Fix:** Create a pipeline builder:
```rust
struct PipelineBuilder<'a> {
    device: &'a wgpu::Device,
    shader: &'a wgpu::ShaderModule,
    layout: &'a wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
}

impl PipelineBuilder<'_> {
    fn build(self, vs_entry: &str, fs_entry: &str, blend: Option<wgpu::BlendState>) -> wgpu::RenderPipeline
}
```

---

### 2. renderer.rs: Box Drawing Character Rendering
**Lines:** 4800-5943 (~1150 lines)  
**Estimated Savings:** 400-600 lines

**Problem:** Massive match statement with repeated `hline()`/`vline()`/`fill_rect()` patterns. Many similar light/heavy/double line variants.

**Suggested Fix:**
- Use lookup tables for line positions/thicknesses
- Create a `BoxDrawingSpec` struct with line positions
- Consolidate repeated drawing patterns into parameterized helpers

---

### 3. main.rs: Duplicate NamedKey-to-String Matching
**Lines:** 1778-1812, 2146-2181  
**Estimated Savings:** ~70 lines

**Problem:** `check_keybinding()` and `handle_keyboard_input()` both have nearly identical `NamedKey` to string/FunctionalKey matching logic for F1-F12, arrow keys, Home/End/PageUp/PageDown, etc.

**Suggested Fix:**
```rust
fn named_key_to_str(named: &NamedKey) -> Option<&'static str> { ... }
fn named_key_to_functional(named: &NamedKey) -> Option<FunctionalKey> { ... }
```

---

### 4. terminal.rs: Duplicated SGR Extended Color Parsing
**Lines:** 2046-2106  
**Estimated Savings:** ~30 lines

**Problem:** SGR 38 (foreground) and SGR 48 (background) extended color parsing logic is nearly identical - duplicated twice each (once for sub-params, once for regular params).

**Suggested Fix:**
```rust
fn parse_extended_color(&self, params: &CsiParams, i: &mut usize) -> Option<Color> {
    let mode = params.get(*i + 1, 0);
    if mode == 5 && *i + 2 < params.num_params {
        *i += 2;
        Some(Color::Indexed(params.params[*i] as u8))
    } else if mode == 2 && *i + 4 < params.num_params {
        let color = Color::Rgb(
            params.params[*i + 2] as u8,
            params.params[*i + 3] as u8,
            params.params[*i + 4] as u8,
        );
        *i += 4;
        Some(color)
    } else {
        None
    }
}
```

---

### 5. terminal.rs: Cursor-Row-With-Scroll Pattern
**Lines:** 1239-1246, 1262-1270, 1319-1324, 1836-1842, 1844-1851, 1916-1922, 1934-1940  
**Estimated Savings:** ~25 lines

**Problem:** The pattern "increment cursor_row, check against scroll_bottom, scroll_up(1) if needed" is repeated 7 times.

**Suggested Fix:**
```rust
#[inline]
fn advance_row(&mut self) {
    self.cursor_row += 1;
    if self.cursor_row > self.scroll_bottom {
        self.scroll_up(1);
        self.cursor_row = self.scroll_bottom;
    }
}
```

---

### 6. terminal.rs: Cell Construction with Current Attributes
**Lines:** 1278-1287, 1963-1972, 1985-1994  
**Estimated Savings:** ~20 lines

**Problem:** `Cell` construction using current attributes is repeated 3 times with nearly identical code.

**Suggested Fix:**
```rust
#[inline]
fn current_cell(&self, character: char, wide_continuation: bool) -> Cell {
    Cell {
        character,
        fg_color: self.current_fg,
        bg_color: self.current_bg,
        bold: self.current_bold,
        italic: self.current_italic,
        underline_style: self.current_underline_style,
        strikethrough: self.current_strikethrough,
        wide_continuation,
    }
}
```

---

### 7. config.rs: normalize_key_name Allocates String for Static Values
**Lines:** 89-160  
**Estimated Savings:** Eliminates 50+ string allocations

**Problem:** Every match arm allocates a new String even though most are static:
```rust
"left" | "arrowleft" | "arrow_left" => "left".to_string(),
```

**Suggested Fix:**
```rust
fn normalize_key_name(name: &str) -> Cow<'static, str> {
    match name {
        "left" | "arrowleft" | "arrow_left" => Cow::Borrowed("left"),
        // ...
        _ => Cow::Owned(name.to_string()),
    }
}
```

---

### 8. config.rs: Repeated Parse-and-Insert Blocks in build_action_map
**Lines:** 281-349  
**Estimated Savings:** ~40 lines

**Problem:** 20+ repeated blocks:
```rust
if let Some(parsed) = self.new_tab.parse() {
    map.insert(parsed, Action::NewTab);
}
```

**Suggested Fix:**
```rust
let bindings: &[(&Keybind, Action)] = &[
    (&self.new_tab, Action::NewTab),
    (&self.next_tab, Action::NextTab),
    // ...
];
for (keybind, action) in bindings {
    if let Some(parsed) = keybind.parse() {
        map.insert(parsed, *action);
    }
}
```

---

### 9. vt_parser.rs: Duplicate OSC/String Command Terminator Handling
**Lines:** 683-755, 773-843  
**Estimated Savings:** ~30 lines

**Problem:** `consume_osc` and `consume_string_command` have nearly identical structure for finding and handling terminators (ESC, BEL, C1 ST).

**Suggested Fix:** Extract a common helper:
```rust
fn consume_st_terminated<F>(
    &mut self,
    bytes: &[u8],
    pos: usize,
    buffer: &mut Vec<u8>,
    include_bel: bool,
    on_complete: F,
) -> usize
where
    F: FnOnce(&mut Self, &[u8])
```

---

### 10. vt_parser.rs: Duplicate Control Char Handling in CSI States
**Lines:** 547-551, 593-596, 650-653  
**Estimated Savings:** ~15 lines

**Problem:** Identical control character handling appears in all three `CsiState` match arms:
```rust
0x00..=0x1F => {
    if ch != 0x1B {
        handler.control(ch);
    }
}
```

**Suggested Fix:** Move control char handling before the `match self.csi.state` block:
```rust
if ch <= 0x1F && ch != 0x1B {
    handler.control(ch);
    consumed += 1;
    continue;
}
```

---

### 11. main.rs: Repeated active_tab().and_then(active_pane()) Pattern
**Lines:** Various (10+ occurrences)  
**Estimated Savings:** ~30 lines

**Problem:** This nested Option chain appears throughout the code.

**Suggested Fix:**
```rust
fn active_pane(&self) -> Option<&Pane> {
    self.active_tab().and_then(|t| t.active_pane())
}

fn active_pane_mut(&mut self) -> Option<&mut Pane> {
    self.active_tab_mut().and_then(|t| t.active_pane_mut())
}
```

---

## Medium Priority

### 12. renderer.rs: set_scale_factor and set_font_size Duplicate Cell Metric Recalc
**Lines:** 3306-3413  
**Estimated Savings:** ~50 lines

**Problem:** ~100 lines of nearly identical cell metric recalculation logic.

**Suggested Fix:** Extract to a shared `recalculate_cell_metrics(&mut self)` method.

---

### 13. renderer.rs: find_font_for_char and find_color_font_for_char Similar
**Lines:** 939-1081  
**Estimated Savings:** ~60-80 lines

**Problem:** ~140 lines of similar fontconfig query patterns.

**Suggested Fix:** Extract common fontconfig query helper, parameterize the charset/color requirements.

---

### 14. renderer.rs: place_glyph_in_cell_canvas vs Color Variant
**Lines:** 6468-6554  
**Estimated Savings:** ~40-50 lines

**Problem:** ~90 lines of nearly identical logic, differing only in bytes-per-pixel (1 vs 4).

**Suggested Fix:**
```rust
fn place_glyph_in_cell_canvas_impl(&self, bitmap: &[u8], ..., bytes_per_pixel: usize) -> Vec<u8>
```

---

### 15. renderer.rs: render_rect vs render_overlay_rect Near-Identical
**Lines:** 7192-7215  
**Estimated Savings:** ~10 lines

**Problem:** Near-identical functions pushing to different Vec.

**Suggested Fix:**
```rust
fn render_quad(&mut self, x: f32, y: f32, w: f32, h: f32, color: [f32; 4], overlay: bool)
```

---

### 16. renderer.rs: Pane Border Adjacency Checks Repeated
**Lines:** 7471-7587  
**Estimated Savings:** ~60-80 lines

**Problem:** ~120 lines of repetitive adjacency detection for 4 directions.

**Suggested Fix:**
```rust
fn check_pane_adjacency(&self, a: &PaneInfo, b: &PaneInfo) -> Vec<Border>
```

---

### 17. terminal.rs: to_rgba and to_rgba_bg Nearly Identical
**Lines:** 212-239  
**Estimated Savings:** ~15 lines

**Problem:** These two methods differ only in the `Color::Default` case.

**Suggested Fix:**
```rust
pub fn to_rgba(&self, color: &Color, is_bg: bool) -> [f32; 4] {
    let [r, g, b] = match color {
        Color::Default => if is_bg { self.default_bg } else { self.default_fg },
        Color::Rgb(r, g, b) => [*r, *g, *b],
        Color::Indexed(idx) => self.colors[*idx as usize],
    };
    [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0]
}
```

---

### 18. terminal.rs: insert_lines/delete_lines Share Dirty Marking
**Lines:** 1080-1134  
**Estimated Savings:** ~6 lines

**Problem:** Both use identical dirty marking loop that could use existing `mark_region_dirty`.

**Suggested Fix:** Replace:
```rust
for line in self.cursor_row..=self.scroll_bottom {
    self.mark_line_dirty(line);
}
```
with:
```rust
self.mark_region_dirty(self.cursor_row, self.scroll_bottom);
```

---

### 19. terminal.rs: handle_dec_private_mode_set/reset Are Mirror Images
**Lines:** 2148-2295  
**Estimated Savings:** ~50 lines

**Problem:** ~70 lines each, almost mirror images with `true` vs `false`.

**Suggested Fix:**
```rust
fn handle_dec_private_mode(&mut self, params: &CsiParams, set: bool) {
    for i in 0..params.num_params {
        match params.params[i] {
            1 => self.application_cursor_keys = set,
            7 => self.auto_wrap = set,
            25 => self.cursor_visible = set,
            // ...
        }
    }
}
```

---

### 20. main.rs: Duplicate Git Status String Building
**Lines:** 1095-1118  
**Estimated Savings:** ~15 lines

**Problem:** Identical logic for building `working_string` and `staging_string` with `~N`, `+N`, `-N` format.

**Suggested Fix:**
```rust
fn format_git_changes(modified: usize, added: usize, deleted: usize) -> String {
    let mut parts = Vec::new();
    if modified > 0 { parts.push(format!("~{}", modified)); }
    if added > 0 { parts.push(format!("+{}", added)); }
    if deleted > 0 { parts.push(format!("-{}", deleted)); }
    parts.join(" ")
}
```

---

### 21. main.rs: Repeated StatuslineComponent RGB Color Application
**Lines:** 1169-1212  
**Estimated Savings:** ~10 lines

**Problem:** Multiple `StatuslineComponent::new(...).rgb_fg(fg_color.0, fg_color.1, fg_color.2)` calls with exact same color.

**Suggested Fix:**
```rust
let with_fg = |text: &str| StatuslineComponent::new(text).rgb_fg(fg_color.0, fg_color.1, fg_color.2);
components.push(with_fg(" "));
components.push(with_fg(&head_text));
```

---

### 22. main.rs: Tab1-Tab9 as Separate Match Arms
**Lines:** 1862-1870  
**Estimated Savings:** ~8 lines

**Problem:** Nine separate match arms each calling `self.switch_to_tab(N)`.

**Suggested Fix:**
```rust
impl Action {
    fn tab_index(&self) -> Option<usize> {
        match self {
            Action::Tab1 => Some(0),
            Action::Tab2 => Some(1),
            // ...
        }
    }
}
// Then in match:
action if action.tab_index().is_some() => {
    self.switch_to_tab(action.tab_index().unwrap());
}
```

---

### 23. keyboard.rs: encode_arrow and encode_f1_f4 Are Identical
**Lines:** 462-485  
**Estimated Savings:** ~15 lines

**Problem:** These two methods have identical implementations.

**Suggested Fix:** Remove `encode_f1_f4` and use `encode_arrow` for both, or rename to `encode_ss3_key`.

---

### 24. keyboard.rs: Repeated to_string().as_bytes() Allocations
**Lines:** 356, 365, 372, 378, 466, 479, 490, 493, 554  
**Estimated Savings:** Reduced allocations

**Problem:** Multiple places call `.to_string().as_bytes()` on integers.

**Suggested Fix:**
```rust
fn write_u32_to_vec(n: u32, buf: &mut Vec<u8>) {
    use std::io::Write;
    write!(buf, "{}", n).unwrap();
}
```
Or use `itoa` crate for zero-allocation integer formatting.

---

### 25. graphics.rs: Duplicate AnimationData Construction
**Lines:** 444-456, 623-635  
**Estimated Savings:** ~10 lines

**Problem:** Both `decode_gif` and `decode_webm` create identical `AnimationData` structs.

**Suggested Fix:**
```rust
impl AnimationData {
    pub fn new(frames: Vec<AnimationFrame>, total_duration_ms: u64) -> Self {
        Self {
            frames,
            current_frame: 0,
            frame_start: None,
            looping: true,
            total_duration_ms,
            state: AnimationState::Running,
            loops_remaining: None,
        }
    }
}
```

---

### 26. graphics.rs: Duplicate RGBA Stride Handling
**Lines:** 554-566, 596-607  
**Estimated Savings:** ~15 lines

**Problem:** Code for handling RGBA stride appears twice (nearly identical).

**Suggested Fix:**
```rust
fn copy_with_stride(data: &[u8], width: u32, height: u32, stride: usize) -> Vec<u8> {
    let row_bytes = (width * 4) as usize;
    if stride == row_bytes {
        data[..(width * height * 4) as usize].to_vec()
    } else {
        let mut result = Vec::with_capacity((width * height * 4) as usize);
        for row in 0..height as usize {
            let start = row * stride;
            result.extend_from_slice(&data[start..start + row_bytes]);
        }
        result
    }
}
```

---

### 27. graphics.rs: Duplicate File/TempFile/SharedMemory Reading Logic
**Lines:** 1051-1098, 1410-1489  
**Estimated Savings:** ~30 lines

**Problem:** Both `handle_animation_frame` and `store_image` have similar file reading logic.

**Suggested Fix:**
```rust
fn load_transmission_data(&mut self, cmd: &mut GraphicsCommand) -> Result<Vec<u8>, GraphicsError>
```

---

### 28. pty.rs: Duplicate Winsize/ioctl Pattern
**Lines:** 57-67, 170-185  
**Estimated Savings:** ~8 lines

**Problem:** Same `libc::winsize` struct creation and `TIOCSWINSZ` ioctl pattern duplicated.

**Suggested Fix:**
```rust
fn set_winsize(fd: RawFd, cols: u16, rows: u16, xpixel: u16, ypixel: u16) -> Result<(), PtyError> {
    let winsize = libc::winsize {
        ws_row: rows,
        ws_col: cols,
        ws_xpixel: xpixel,
        ws_ypixel: ypixel,
    };
    let result = unsafe { libc::ioctl(fd, libc::TIOCSWINSZ, &winsize) };
    if result == -1 {
        Err(PtyError::Io(std::io::Error::last_os_error()))
    } else {
        Ok(())
    }
}
```

---

### 29. vt_parser.rs: Repeated Max Length Check Pattern
**Lines:** 537-541, 693-697, 745-749, 785-789, 831-835  
**Estimated Savings:** ~15 lines

**Problem:** This pattern appears 5 times:
```rust
if self.escape_len + X > MAX_ESCAPE_LEN {
    log::debug!("... sequence too long, aborting");
    self.state = State::Normal;
    return consumed;
}
```

**Suggested Fix:**
```rust
#[inline]
fn check_max_len(&mut self, additional: usize) -> bool {
    if self.escape_len + additional > MAX_ESCAPE_LEN {
        log::debug!("Escape sequence too long, aborting");
        self.state = State::Normal;
        true
    } else {
        false
    }
}
```

---

## Low Priority

### 30. main.rs: PaneId/TabId Have Identical Implementations
**Lines:** 230-239, 694-703  
**Estimated Savings:** ~15 lines

**Problem:** Both use identical `new()` implementations with static atomics.

**Suggested Fix:**
```rust
macro_rules! define_id {
    ($name:ident) => {
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name(u64);
        
        impl $name {
            pub fn new() -> Self {
                use std::sync::atomic::{AtomicU64, Ordering};
                static NEXT_ID: AtomicU64 = AtomicU64::new(1);
                Self(NEXT_ID.fetch_add(1, Ordering::Relaxed))
            }
        }
    };
}
define_id!(PaneId);
define_id!(TabId);
```

---

### 31. main.rs: Duplicate "All Tabs Closed" Exit Checks
**Lines:** 2622-2653  
**Estimated Savings:** ~5 lines

**Problem:** Two separate checks with same log message in `about_to_wait()`.

**Suggested Fix:** Restructure to have a single exit point after cleanup logic.

---

### 32. terminal.rs: scroll_viewport_up/down Similarity
**Lines:** 885-915  
**Estimated Savings:** ~8 lines

**Problem:** Similar structure - both check for alternate screen, calculate offset, set dirty.

**Suggested Fix:** Merge into single method with signed delta parameter.

---

### 33. terminal.rs: screen_alignment Cell Construction Verbose
**Lines:** 1881-1890  
**Estimated Savings:** ~6 lines

**Problem:** Constructs Cell with all default values explicitly.

**Suggested Fix:**
```rust
*cell = Cell { character: 'E', ..Cell::default() };
```

---

### 34. keyboard.rs: Repeated UTF-8 Char Encoding Pattern
**Lines:** 503-505, 529-532, 539-541  
**Estimated Savings:** ~8 lines

**Problem:** Pattern appears 3 times:
```rust
let mut buf = [0u8; 4];
let s = c.encode_utf8(&mut buf);
return s.as_bytes().to_vec();
```

**Suggested Fix:**
```rust
fn char_to_vec(c: char) -> Vec<u8> {
    let mut buf = [0u8; 4];
    c.encode_utf8(&mut buf).as_bytes().to_vec()
}
```

---

### 35. config.rs: Tab1-Tab9 as Separate Enum Variants/Fields
**Lines:** 174-182, 214-230  
**Estimated Savings:** Structural improvement

**Problem:** Having separate `Tab1`, `Tab2`, ... `Tab9` variants and corresponding struct fields is verbose.

**Suggested Fix:** Use `Tab(u8)` variant and `tab_keys: [Keybind; 9]` array. Note: This changes the JSON config format.

---

### 36. pty.rs: Inconsistent AsRawFd Usage
**Lines:** 63, 177  
**Estimated Savings:** Cleanup only

**Problem:** Uses fully-qualified `std::os::fd::AsRawFd::as_raw_fd` despite importing the trait.

**Suggested Fix:**
```rust
// Change from:
let fd = std::os::fd::AsRawFd::as_raw_fd(&master);
// To:
let fd = master.as_raw_fd();
```

---

### 37. pty.rs: Repeated /proc Path Pattern
**Lines:** 222-243  
**Estimated Savings:** ~4 lines

**Problem:** Both `foreground_process_name` and `foreground_cwd` build `/proc/{pgid}/...` paths similarly.

**Suggested Fix:**
```rust
fn proc_path(&self, file: &str) -> Option<std::path::PathBuf> {
    let pgid = self.foreground_pgid()?;
    Some(std::path::PathBuf::from(format!("/proc/{}/{}", pgid, file)))
}
```
