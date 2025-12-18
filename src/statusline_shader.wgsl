// Statusline shader - optimized for single-row text rendering
// Simpler than the full terminal cell shader, focused on text with colors

// ═══════════════════════════════════════════════════════════════════════════════
// GAMMA CONVERSION FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

// Luminance weights for perceived brightness (ITU-R BT.709)
const Y: vec3<f32> = vec3<f32>(0.2126, 0.7152, 0.0722);

// Convert sRGB to linear RGB
fn srgb2linear(x: f32) -> f32 {
    if x <= 0.04045 {
        return x / 12.92;
    } else {
        return pow((x + 0.055) / 1.055, 2.4);
    }
}

// Convert linear RGB to sRGB
fn linear2srgb(x: f32) -> f32 {
    if x <= 0.0031308 {
        return 12.92 * x;
    } else {
        return 1.055 * pow(x, 1.0 / 2.4) - 0.055;
    }
}

// Kitty's legacy gamma-incorrect text blending for crisp rendering
fn foreground_contrast_legacy(over_srgb: vec3<f32>, over_alpha: f32, under_srgb: vec3<f32>) -> f32 {
    let over_linear = vec3<f32>(srgb2linear(over_srgb.r), srgb2linear(over_srgb.g), srgb2linear(over_srgb.b));
    let under_linear = vec3<f32>(srgb2linear(under_srgb.r), srgb2linear(under_srgb.g), srgb2linear(under_srgb.b));
    
    let under_luminance = dot(under_linear, Y);
    let over_luminance = dot(over_linear, Y);
    
    let luminance_diff = over_luminance - under_luminance;
    if abs(luminance_diff) < 0.001 {
        return over_alpha;
    }
    
    let blended_srgb = linear2srgb(over_luminance) * over_alpha + linear2srgb(under_luminance) * (1.0 - over_alpha);
    let blended_linear = srgb2linear(blended_srgb);
    let new_alpha = (blended_linear - under_luminance) / luminance_diff;
    
    return clamp(new_alpha, 0.0, 1.0);
}

// ═══════════════════════════════════════════════════════════════════════════════
// STATUSLINE DATA STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

// Per-cell data for statusline rendering
// Matches GPUCell struct in renderer.rs exactly for buffer compatibility
struct StatuslineCell {
    // Foreground color (packed: type in low byte, color data in upper bytes)
    fg: u32,
    // Background color (packed same way)
    bg: u32,
    // Decoration foreground color (unused in statusline, but needed for struct alignment)
    decoration_fg: u32,
    // Sprite index in atlas (0 = no glyph/space). High bit = colored glyph.
    sprite_idx: u32,
    // Cell attributes (unused in statusline, but needed for struct alignment)
    attrs: u32,
}

// Sprite info for glyph positioning
struct SpriteInfo {
    // UV coordinates in atlas (x, y, width, height) - normalized 0-1
    uv: vec4<f32>,
    // Padding
    _padding: vec2<f32>,
    // Size in pixels (width, height)
    size: vec2<f32>,
}

// Statusline parameters uniform
struct StatuslineParams {
    // Number of characters in statusline
    char_count: u32,
    // Cell dimensions in pixels
    cell_width: f32,
    cell_height: f32,
    // Screen dimensions in pixels
    screen_width: f32,
    screen_height: f32,
    // Y position of statusline (in pixels from top)
    y_offset: f32,
    // Padding for alignment
    _padding: vec2<f32>,
}

// Color table for indexed colors
struct ColorTable {
    colors: array<vec4<f32>, 258>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// BINDINGS
// ═══════════════════════════════════════════════════════════════════════════════

@group(0) @binding(0)
var atlas_texture: texture_2d<f32>;
@group(0) @binding(1)
var atlas_sampler: sampler;

@group(1) @binding(0)
var<uniform> color_table: ColorTable;

@group(1) @binding(1)
var<uniform> params: StatuslineParams;

@group(1) @binding(2)
var<storage, read> cells: array<StatuslineCell>;

@group(1) @binding(3)
var<storage, read> sprites: array<SpriteInfo>;

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

const COLOR_TYPE_DEFAULT: u32 = 0u;
const COLOR_TYPE_INDEXED: u32 = 1u;
const COLOR_TYPE_RGB: u32 = 2u;

const COLORED_GLYPH_FLAG: u32 = 0x80000000u;

// ═══════════════════════════════════════════════════════════════════════════════
// VERTEX OUTPUT
// ═══════════════════════════════════════════════════════════════════════════════

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) fg_color: vec4<f32>,
    @location(2) bg_color: vec4<f32>,
    @location(3) @interpolate(flat) is_background: u32,
    @location(4) @interpolate(flat) is_colored_glyph: u32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

// Resolve a packed color to RGBA
fn resolve_color(packed: u32, is_foreground: bool) -> vec4<f32> {
    let color_type = packed & 0xFFu;
    
    if color_type == COLOR_TYPE_DEFAULT {
        if is_foreground {
            return color_table.colors[256];
        } else {
            return color_table.colors[257];
        }
    } else if color_type == COLOR_TYPE_INDEXED {
        let index = (packed >> 8u) & 0xFFu;
        return color_table.colors[index];
    } else {
        // RGB color
        let r = f32((packed >> 8u) & 0xFFu) / 255.0;
        let g = f32((packed >> 16u) & 0xFFu) / 255.0;
        let b = f32((packed >> 24u) & 0xFFu) / 255.0;
        return vec4<f32>(srgb2linear(r), srgb2linear(g), srgb2linear(b), 1.0);
    }
}

// Convert pixel coordinate to NDC
fn pixel_to_ndc(pixel: vec2<f32>, screen: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        (pixel.x / screen.x) * 2.0 - 1.0,
        1.0 - (pixel.y / screen.y) * 2.0
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// BACKGROUND VERTEX SHADER
// ═══════════════════════════════════════════════════════════════════════════════

@vertex
fn vs_statusline_bg(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    // Skip if out of bounds
    if instance_index >= params.char_count {
        var out: VertexOutput;
        out.clip_position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return out;
    }
    
    let cell = cells[instance_index];
    
    // Calculate cell position (single row, left to right)
    let cell_x = f32(instance_index) * params.cell_width;
    let cell_y = params.y_offset;
    
    // Quad vertex positions for TriangleStrip
    var positions: array<vec2<f32>, 4>;
    positions[0] = vec2<f32>(cell_x, cell_y);
    positions[1] = vec2<f32>(cell_x + params.cell_width, cell_y);
    positions[2] = vec2<f32>(cell_x, cell_y + params.cell_height);
    positions[3] = vec2<f32>(cell_x + params.cell_width, cell_y + params.cell_height);
    
    let screen_size = vec2<f32>(params.screen_width, params.screen_height);
    let ndc_pos = pixel_to_ndc(positions[vertex_index], screen_size);
    
    let fg = resolve_color(cell.fg, true);
    var bg = resolve_color(cell.bg, false);
    
    // For default background, use transparent
    let bg_type = cell.bg & 0xFFu;
    if bg_type == COLOR_TYPE_DEFAULT {
        bg.a = 0.0;
    }
    
    var out: VertexOutput;
    out.clip_position = vec4<f32>(ndc_pos, 0.0, 1.0);
    out.uv = vec2<f32>(0.0, 0.0);
    out.fg_color = fg;
    out.bg_color = bg;
    out.is_background = 1u;
    out.is_colored_glyph = 0u;
    
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════════
// GLYPH VERTEX SHADER
// ═══════════════════════════════════════════════════════════════════════════════

@vertex
fn vs_statusline_glyph(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32
) -> VertexOutput {
    // Skip if out of bounds
    if instance_index >= params.char_count {
        var out: VertexOutput;
        out.clip_position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return out;
    }
    
    let cell = cells[instance_index];
    let sprite_idx = cell.sprite_idx & ~COLORED_GLYPH_FLAG;
    let is_colored = (cell.sprite_idx & COLORED_GLYPH_FLAG) != 0u;
    
    // Skip if no glyph
    if sprite_idx == 0u {
        var out: VertexOutput;
        out.clip_position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return out;
    }
    
    let sprite = sprites[sprite_idx];
    
    // Skip if sprite has no size
    if sprite.size.x <= 0.0 || sprite.size.y <= 0.0 {
        var out: VertexOutput;
        out.clip_position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return out;
    }
    
    // Calculate glyph position
    let glyph_x = f32(instance_index) * params.cell_width;
    let glyph_y = params.y_offset;
    
    // Quad vertex positions
    var positions: array<vec2<f32>, 4>;
    positions[0] = vec2<f32>(glyph_x, glyph_y);
    positions[1] = vec2<f32>(glyph_x + sprite.size.x, glyph_y);
    positions[2] = vec2<f32>(glyph_x, glyph_y + sprite.size.y);
    positions[3] = vec2<f32>(glyph_x + sprite.size.x, glyph_y + sprite.size.y);
    
    // UV coordinates
    var uvs: array<vec2<f32>, 4>;
    uvs[0] = vec2<f32>(sprite.uv.x, sprite.uv.y);
    uvs[1] = vec2<f32>(sprite.uv.x + sprite.uv.z, sprite.uv.y);
    uvs[2] = vec2<f32>(sprite.uv.x, sprite.uv.y + sprite.uv.w);
    uvs[3] = vec2<f32>(sprite.uv.x + sprite.uv.z, sprite.uv.y + sprite.uv.w);
    
    let screen_size = vec2<f32>(params.screen_width, params.screen_height);
    let ndc_pos = pixel_to_ndc(positions[vertex_index], screen_size);
    
    let fg = resolve_color(cell.fg, true);
    let bg = resolve_color(cell.bg, false);
    
    var out: VertexOutput;
    out.clip_position = vec4<f32>(ndc_pos, 0.0, 1.0);
    out.uv = uvs[vertex_index];
    out.fg_color = fg;
    out.bg_color = bg;
    out.is_background = 0u;
    out.is_colored_glyph = select(0u, 1u, is_colored);
    
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════════
// FRAGMENT SHADER
// ═══════════════════════════════════════════════════════════════════════════════

@fragment
fn fs_statusline(in: VertexOutput) -> @location(0) vec4<f32> {
    if in.is_background == 1u {
        return in.bg_color;
    }
    
    // Sample glyph from atlas
    let glyph_sample = textureSample(atlas_texture, atlas_sampler, in.uv);
    
    if in.is_colored_glyph == 1u {
        // Colored glyph (emoji) - use atlas color directly
        return glyph_sample;
    }
    
    // Regular glyph - apply foreground color with legacy gamma blending
    let glyph_alpha = glyph_sample.a;
    let adjusted_alpha = foreground_contrast_legacy(in.fg_color.rgb, glyph_alpha, in.bg_color.rgb);
    
    return vec4<f32>(in.fg_color.rgb, in.fg_color.a * adjusted_alpha);
}
