// Glyph rendering shader for terminal emulator
// Supports both legacy quad-based rendering and new instanced cell rendering
// Uses Kitty-style "legacy" gamma-incorrect text blending for crisp rendering

// ═══════════════════════════════════════════════════════════════════════════════
// GAMMA CONVERSION FUNCTIONS (for legacy text rendering)
// ═══════════════════════════════════════════════════════════════════════════════

// Luminance weights for perceived brightness (ITU-R BT.709)
const Y: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

// Convert linear RGB to sRGB
fn linear2srgb(x: f32) -> f32 {
    if x <= 0.0031308 {
        return 12.92 * x;
    } else {
        return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
    }
}

// Convert sRGB to linear RGB
fn srgb2linear(x: f32) -> f32 {
    if x <= 0.04045 {
        return x / 12.92;
    } else {
        return pow((x + 0.055) / 1.055, 2.4);
    }
}

// Kitty's legacy gamma-incorrect text blending
// This simulates how text was blended before gamma-correct rendering became standard.
// It makes dark text on light backgrounds appear thicker and light text on dark
// backgrounds appear thinner, which many users prefer for readability.
//
// The input colors are in sRGB space. We convert to linear for the luminance
// calculation, then simulate gamma-incorrect blending.
fn foreground_contrast_legacy(over_srgb: vec3<f32>, over_alpha: f32, under_srgb: vec3<f32>) -> f32 {
    // Convert sRGB colors to linear for luminance calculation
    let over_linear = vec3<f32>(srgb2linear(over_srgb.r), srgb2linear(over_srgb.g), srgb2linear(over_srgb.b));
    let under_linear = vec3<f32>(srgb2linear(under_srgb.r), srgb2linear(under_srgb.g), srgb2linear(under_srgb.b));
    
    let under_luminance = dot(under_linear, Y);
    let over_luminance = dot(over_linear, Y);
    
    // Avoid division by zero when luminances are equal
    let luminance_diff = over_luminance - under_luminance;
    if abs(luminance_diff) < 0.001 {
        return over_alpha;
    }
    
    // Kitty's formula: simulate gamma-incorrect blending
    // This is the solution to:
    // linear2srgb(over * alpha2 + under * (1 - alpha2)) = linear2srgb(over) * alpha + linear2srgb(under) * (1 - alpha)
    // ^ gamma correct blending with new alpha              ^ gamma incorrect blending with old alpha
    let blended_srgb = linear2srgb(over_luminance) * over_alpha + linear2srgb(under_luminance) * (1.0 - over_alpha);
    let blended_linear = srgb2linear(blended_srgb);
    let new_alpha = (blended_linear - under_luminance) / luminance_diff;
    
    return clamp(new_alpha, 0.0, 1.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// LEGACY QUAD-BASED RENDERING (for backwards compatibility)
// ═══════════════════════════════════════════════════════════════════════════════

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) bg_color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) bg_color: vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(in.position, 0.0, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    out.bg_color = in.bg_color;
    return out;
}

@group(0) @binding(0)
var atlas_texture: texture_2d<f32>;
@group(0) @binding(1)
var atlas_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // If UV is at origin (0,0), this is a background-only quad
    let is_background_only = in.uv.x == 0.0 && in.uv.y == 0.0;
    
    if is_background_only {
        // Just render the background color (fully opaque)
        return in.bg_color;
    }
    
    // Sample from RGBA atlas
    let glyph_sample = textureSample(atlas_texture, atlas_sampler, in.uv);
    
    // Detect color glyphs: regular glyphs are stored as white (1,1,1) with alpha
    // Color glyphs have actual RGB colors. Check if any RGB channel is not white.
    let is_color_glyph = glyph_sample.r < 0.99 || glyph_sample.g < 0.99 || glyph_sample.b < 0.99;
    
    if is_color_glyph {
        // Color glyph (emoji) - use atlas color directly
        return glyph_sample;
    }
    
    // Regular glyph - use alpha with foreground color
    let glyph_alpha = glyph_sample.a;
    
    // Apply legacy gamma-incorrect blending for crisp text
    let adjusted_alpha = foreground_contrast_legacy(in.color.rgb, glyph_alpha, in.bg_color.rgb);
    
    // Output foreground color with adjusted alpha for blending
    return vec4<f32>(in.color.rgb, in.color.a * adjusted_alpha);
}

// ═══════════════════════════════════════════════════════════════════════════════
// KITTY-STYLE INSTANCED CELL RENDERING
// ═══════════════════════════════════════════════════════════════════════════════

// Color table uniform containing 256 indexed colors + default fg/bg
struct ColorTable {
    // 256 indexed colors + default_fg (256) + default_bg (257)
    colors: array<vec4<f32>, 258>,
}

// Grid parameters uniform
struct GridParams {
    // Grid dimensions in cells
    cols: u32,
    rows: u32,
    // Cell dimensions in pixels
    cell_width: f32,
    cell_height: f32,
    // Screen dimensions in pixels
    screen_width: f32,
    screen_height: f32,
    // X offset for pane position
    x_offset: f32,
    // Y offset for tab bar + pane position
    y_offset: f32,
    // Cursor position (-1 if hidden)
    cursor_col: i32,
    cursor_row: i32,
    // Cursor style: 0=block, 1=underline, 2=bar
    cursor_style: u32,
    // Background opacity for transparency (0.0 = transparent, 1.0 = opaque)
    background_opacity: f32,
    // Selection range (-1 values mean no selection)
    selection_start_col: i32,
    selection_start_row: i32,
    selection_end_col: i32,
    selection_end_row: i32,
}

// GPUCell instance data (matches Rust GPUCell struct)
struct GPUCell {
    fg: u32,
    bg: u32,
    decoration_fg: u32,
    sprite_idx: u32,
    attrs: u32,
}

// Sprite info for glyph positioning
// In Kitty's model, sprites are always cell-sized and glyphs are pre-positioned
// within the sprite at the correct baseline. No offset math needed.
struct SpriteInfo {
    // UV coordinates in atlas (x, y, width, height) - normalized 0-1
    uv: vec4<f32>,
    // Padding (previously offset, now unused)
    _padding: vec2<f32>,
    // Size in pixels (width, height) - always matches cell dimensions
    size: vec2<f32>,
}

// Uniforms and storage buffers for instanced rendering
@group(1) @binding(0)
var<uniform> color_table: ColorTable;

@group(1) @binding(1)
var<uniform> grid_params: GridParams;

@group(1) @binding(2)
var<storage, read> cells: array<GPUCell>;

@group(1) @binding(3)
var<storage, read> sprites: array<SpriteInfo>;

// Constants for packed color decoding
const COLOR_TYPE_DEFAULT: u32 = 0u;
const COLOR_TYPE_INDEXED: u32 = 1u;
const COLOR_TYPE_RGB: u32 = 2u;

// Constants for cell attributes
const ATTR_DECORATION_MASK: u32 = 0x7u;
const ATTR_BOLD_BIT: u32 = 0x8u;
const ATTR_ITALIC_BIT: u32 = 0x10u;
const ATTR_REVERSE_BIT: u32 = 0x20u;
const ATTR_STRIKE_BIT: u32 = 0x40u;
const ATTR_DIM_BIT: u32 = 0x80u;
const ATTR_SELECTED_BIT: u32 = 0x100u;

// Colored glyph flag
const COLORED_GLYPH_FLAG: u32 = 0x80000000u;

// Cursor shape constants
const CURSOR_BLOCK: u32 = 0u;
const CURSOR_UNDERLINE: u32 = 1u;
const CURSOR_BAR: u32 = 2u;

// Check if a cell is within the selection range
// Selection is specified as (start_col, start_row) to (end_col, end_row), normalized
// so start <= end in reading order
fn is_cell_selected(col: u32, row: u32) -> bool {
    // Check if selection is active (-1 values mean no selection)
    if grid_params.selection_start_col < 0 || grid_params.selection_start_row < 0 {
        return false;
    }
    
    let sel_start_col = u32(grid_params.selection_start_col);
    let sel_start_row = u32(grid_params.selection_start_row);
    let sel_end_col = u32(grid_params.selection_end_col);
    let sel_end_row = u32(grid_params.selection_end_row);
    
    // Check if cell is within row range
    if row < sel_start_row || row > sel_end_row {
        return false;
    }
    
    // Single row selection
    if sel_start_row == sel_end_row {
        return col >= sel_start_col && col <= sel_end_col;
    }
    
    // Multi-row selection
    if row == sel_start_row {
        // First row: from start_col to end of line
        return col >= sel_start_col;
    } else if row == sel_end_row {
        // Last row: from start of line to end_col
        return col <= sel_end_col;
    } else {
        // Middle rows: entire row is selected
        return true;
    }
}

// Vertex output for instanced cell rendering
struct CellVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) fg_color: vec4<f32>,
    @location(2) bg_color: vec4<f32>,
    @location(3) @interpolate(flat) is_background: u32,
    @location(4) @interpolate(flat) is_colored_glyph: u32,
    @location(5) @interpolate(flat) is_cursor: u32,
    @location(6) @interpolate(flat) cursor_shape: u32,
    @location(7) cursor_color: vec4<f32>,
    @location(8) cell_pos: vec2<f32>,      // Cell top-left position in pixels
    @location(9) @interpolate(flat) cell_size: vec2<f32>,  // Cell width/height in pixels
}

// Resolve a packed color to RGBA (in linear space for GPU rendering)
fn resolve_color(packed: u32, is_foreground: bool) -> vec4<f32> {
    let color_type = packed & 0xFFu;
    
    if color_type == COLOR_TYPE_DEFAULT {
        // Default color - use color table entry 256 (fg) or 257 (bg)
        // Color table is already in linear space
        if is_foreground {
            return color_table.colors[256];
        } else {
            return color_table.colors[257];
        }
    } else if color_type == COLOR_TYPE_INDEXED {
        // Indexed color - look up in color table
        // Color table is already in linear space
        let index = (packed >> 8u) & 0xFFu;
        return color_table.colors[index];
    } else {
        // RGB color - extract components and convert sRGB to linear
        let r = f32((packed >> 8u) & 0xFFu) / 255.0;
        let g = f32((packed >> 16u) & 0xFFu) / 255.0;
        let b = f32((packed >> 24u) & 0xFFu) / 255.0;
        return vec4<f32>(srgb_to_linear(r), srgb_to_linear(g), srgb_to_linear(b), 1.0);
    }
}

// Convert sRGB to linear (for GPU rendering to sRGB surface)
fn srgb_to_linear(c: f32) -> f32 {
    if c <= 0.04045 {
        return c / 12.92;
    } else {
        return pow((c + 0.055) / 1.055, 2.4);
    }
}

// Convert pixel coordinate to NDC
fn pixel_to_ndc(pixel: vec2<f32>, screen: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        (pixel.x / screen.x) * 2.0 - 1.0,
        1.0 - (pixel.y / screen.y) * 2.0
    );
}

// Background vertex shader (renders cell backgrounds)
// vertex_index: 0-3 for quad corners
// instance_index: cell index in row-major order
@vertex
fn vs_cell_bg(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32
) -> CellVertexOutput {
    let col = instance_index % grid_params.cols;
    let row = instance_index / grid_params.cols;
    
    // Skip if out of bounds
    if row >= grid_params.rows {
        var out: CellVertexOutput;
        out.clip_position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return out;
    }
    
    // Get cell data
    let cell = cells[instance_index];
    
    // Calculate cell pixel position
    let cell_x = grid_params.x_offset + f32(col) * grid_params.cell_width;
    let cell_y = grid_params.y_offset + f32(row) * grid_params.cell_height;
    
    // Quad vertex positions for TriangleStrip (0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right)
    // TriangleStrip produces triangles: (0,1,2) and (1,2,3)
    var positions: array<vec2<f32>, 4>;
    positions[0] = vec2<f32>(cell_x, cell_y);                                              // top-left
    positions[1] = vec2<f32>(cell_x + grid_params.cell_width, cell_y);                     // top-right
    positions[2] = vec2<f32>(cell_x, cell_y + grid_params.cell_height);                    // bottom-left
    positions[3] = vec2<f32>(cell_x + grid_params.cell_width, cell_y + grid_params.cell_height); // bottom-right
    
    let screen_size = vec2<f32>(grid_params.screen_width, grid_params.screen_height);
    let ndc_pos = pixel_to_ndc(positions[vertex_index], screen_size);
    
    // Resolve colors
    let attrs = cell.attrs;
    let is_reverse = (attrs & ATTR_REVERSE_BIT) != 0u;
    
    var fg = resolve_color(cell.fg, true);
    var bg = resolve_color(cell.bg, false);
    
    // Handle reverse video
    if is_reverse {
        let tmp = fg;
        fg = bg;
        bg = tmp;
    }
    
    // Check if this cell is selected using GridParams selection range
    let is_selected = is_cell_selected(col, row);
    if is_selected {
        fg = vec4<f32>(0.0, 0.0, 0.0, 1.0);  // Black foreground
        bg = vec4<f32>(1.0, 1.0, 1.0, 1.0);  // White background
    }
    
    // Check if this cell is the cursor
    let is_cursor_cell = (i32(col) == grid_params.cursor_col) && (i32(row) == grid_params.cursor_row);
    
    // For default background (type 0), use fully transparent so the window's
    // clear color (which has background_opacity applied) shows through.
    // Only non-default backgrounds should be opaque.
    // But NOT if the cell is selected (selection always has white bg)
    let bg_type = cell.bg & 0xFFu;
    if bg_type == COLOR_TYPE_DEFAULT && !is_reverse && !is_selected {
        bg.a = 0.0;
    }
    
    // Calculate cursor color
    // If the cell is empty (no glyph), use default foreground color for cursor
    // Otherwise use the cell's foreground color
    var cursor_color: vec4<f32>;
    let sprite_idx = cell.sprite_idx & ~COLORED_GLYPH_FLAG;
    if sprite_idx == 0u {
        // Empty cell - use default foreground color for cursor
        cursor_color = color_table.colors[256]; // default_fg
    } else {
        // Cell has a glyph - use its foreground color
        cursor_color = fg;
    }
    cursor_color.a = 1.0;
    
    var out: CellVertexOutput;
    out.clip_position = vec4<f32>(ndc_pos, 0.0, 1.0);
    out.uv = vec2<f32>(0.0, 0.0); // Not used for background
    out.fg_color = fg;
    out.bg_color = bg;
    out.is_background = 1u;
    out.is_colored_glyph = 0u;
    out.is_cursor = select(0u, 1u, is_cursor_cell);
    out.cursor_shape = grid_params.cursor_style;
    out.cursor_color = cursor_color;
    out.cell_pos = vec2<f32>(cell_x, cell_y);
    out.cell_size = vec2<f32>(grid_params.cell_width, grid_params.cell_height);
    
    return out;
}

// Glyph vertex shader (renders cell glyphs)
@vertex
fn vs_cell_glyph(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32
) -> CellVertexOutput {
    let col = instance_index % grid_params.cols;
    let row = instance_index / grid_params.cols;
    
    // Skip if out of bounds
    if row >= grid_params.rows {
        var out: CellVertexOutput;
        out.clip_position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return out;
    }
    
    // Get cell data
    let cell = cells[instance_index];
    let sprite_idx = cell.sprite_idx & ~COLORED_GLYPH_FLAG;
    let is_colored = (cell.sprite_idx & COLORED_GLYPH_FLAG) != 0u;
    
    // Skip if no glyph
    if sprite_idx == 0u {
        var out: CellVertexOutput;
        out.clip_position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return out;
    }
    
    // Get sprite info
    let sprite = sprites[sprite_idx];
    
    // Skip if sprite has no size
    if sprite.size.x <= 0.0 || sprite.size.y <= 0.0 {
        var out: CellVertexOutput;
        out.clip_position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return out;
    }
    
    // Calculate cell pixel position
    let cell_x = grid_params.x_offset + f32(col) * grid_params.cell_width;
    let cell_y = grid_params.y_offset + f32(row) * grid_params.cell_height;
    
    // Kitty model: sprites are cell-sized with glyphs pre-positioned at baseline.
    // Just map the sprite directly to the cell.
    let glyph_x = cell_x;
    let glyph_y = cell_y;
    
    // Quad vertex positions for TriangleStrip (0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right)
    var positions: array<vec2<f32>, 4>;
    positions[0] = vec2<f32>(glyph_x, glyph_y);                                  // top-left
    positions[1] = vec2<f32>(glyph_x + sprite.size.x, glyph_y);                  // top-right
    positions[2] = vec2<f32>(glyph_x, glyph_y + sprite.size.y);                  // bottom-left
    positions[3] = vec2<f32>(glyph_x + sprite.size.x, glyph_y + sprite.size.y);  // bottom-right
    
    // UV coordinates (matching vertex positions)
    var uvs: array<vec2<f32>, 4>;
    uvs[0] = vec2<f32>(sprite.uv.x, sprite.uv.y);                                // top-left
    uvs[1] = vec2<f32>(sprite.uv.x + sprite.uv.z, sprite.uv.y);                  // top-right
    uvs[2] = vec2<f32>(sprite.uv.x, sprite.uv.y + sprite.uv.w);                  // bottom-left
    uvs[3] = vec2<f32>(sprite.uv.x + sprite.uv.z, sprite.uv.y + sprite.uv.w);    // bottom-right
    
    let screen_size = vec2<f32>(grid_params.screen_width, grid_params.screen_height);
    let ndc_pos = pixel_to_ndc(positions[vertex_index], screen_size);
    
    // Resolve colors
    let attrs = cell.attrs;
    let is_reverse = (attrs & ATTR_REVERSE_BIT) != 0u;
    
    var fg = resolve_color(cell.fg, true);
    var bg = resolve_color(cell.bg, false);
    
    if is_reverse {
        let tmp = fg;
        fg = bg;
        bg = tmp;
    }
    
    // Check if this cell is selected using GridParams selection range
    let is_selected = is_cell_selected(col, row);
    if is_selected {
        fg = vec4<f32>(0.0, 0.0, 0.0, 1.0);  // Black foreground
        bg = vec4<f32>(1.0, 1.0, 1.0, 1.0);  // White background
    }
    
    // Check if this cell is the cursor
    let is_cursor_cell = (i32(col) == grid_params.cursor_col) && (i32(row) == grid_params.cursor_row);
    
    // For block cursor, invert text color (use bg as fg)
    var cursor_text_color = bg;
    cursor_text_color.a = 1.0;
    if is_cursor_cell && grid_params.cursor_style == CURSOR_BLOCK {
        fg = cursor_text_color;
    }
    
    var out: CellVertexOutput;
    out.clip_position = vec4<f32>(ndc_pos, 0.0, 1.0);
    out.uv = uvs[vertex_index];
    out.fg_color = fg;
    out.bg_color = bg;  // Pass background for legacy gamma blending
    out.is_background = 0u;
    out.is_colored_glyph = select(0u, 1u, is_colored);
    out.is_cursor = select(0u, 1u, is_cursor_cell);
    out.cursor_shape = grid_params.cursor_style;
    out.cursor_color = cursor_text_color;
    out.cell_pos = vec2<f32>(cell_x, cell_y);
    out.cell_size = vec2<f32>(grid_params.cell_width, grid_params.cell_height);
    
    return out;
}

// Fragment shader for cell rendering (both background and glyph)
@fragment
fn fs_cell(in: CellVertexOutput) -> @location(0) vec4<f32> {
    if in.is_background == 1u {
        // Check if this is a cursor cell
        if in.is_cursor == 1u {
            // Calculate fragment position relative to cell
            let frag_pos = in.clip_position.xy;
            let cell_local = frag_pos - in.cell_pos;
            
            if in.cursor_shape == CURSOR_BLOCK {
                // Block cursor - fill entire cell with cursor color
                return in.cursor_color;
            } else if in.cursor_shape == CURSOR_UNDERLINE {
                // Underline cursor - bottom 10% or at least 2 pixels
                let underline_height = max(2.0, in.cell_size.y * 0.1);
                if cell_local.y >= in.cell_size.y - underline_height {
                    return in.cursor_color;
                }
            } else if in.cursor_shape == CURSOR_BAR {
                // Bar cursor - left 10% or at least 2 pixels
                let bar_width = max(2.0, in.cell_size.x * 0.1);
                if cell_local.x < bar_width {
                    return in.cursor_color;
                }
            }
        }
        
        // Normal background - just output the bg color
        return in.bg_color;
    }
    
    // Glyph - sample from RGBA atlas
    let glyph_sample = textureSample(atlas_texture, atlas_sampler, in.uv);
    
    if in.is_colored_glyph == 1u {
        // Colored glyph (emoji) - use atlas color directly with premultiplied alpha blending
        // The atlas stores RGBA color from the emoji font
        return glyph_sample;
    }
    
    // Regular glyph - atlas stores white (1,1,1) with alpha in A channel
    // Use the alpha channel for text rendering
    let glyph_alpha = glyph_sample.a;
    
    // Apply legacy gamma-incorrect blending for crisp text
    let adjusted_alpha = foreground_contrast_legacy(in.fg_color.rgb, glyph_alpha, in.bg_color.rgb);
    
    // Normal glyph - tint with foreground color using adjusted alpha
    return vec4<f32>(in.fg_color.rgb, in.fg_color.a * adjusted_alpha);
}
