//! GPU data structures for terminal rendering.
//!
//! Contains vertex formats, uniform structures, and constants for wgpu rendering.
//! All structures use `#[repr(C)]` and implement `bytemuck::Pod` for GPU compatibility.

use bytemuck::{Pod, Zeroable};

// ═══════════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════════

/// Size of the glyph atlas texture (like Kitty's max_texture_size).
/// 8192x8192 provides massive capacity before needing additional layers.
pub const ATLAS_SIZE: u32 = 8192;

/// Maximum number of atlas layers (like Kitty's max_array_len).
/// With 8192x8192 per layer, this provides virtually unlimited glyph storage.
pub const MAX_ATLAS_LAYERS: u32 = 64;

/// Bytes per pixel in the RGBA atlas (4 for RGBA8).
pub const ATLAS_BPP: u32 = 4;

/// Maximum number of simultaneous edge glows.
pub const MAX_EDGE_GLOWS: usize = 16;

/// Color type constants for packed color encoding.
pub const COLOR_TYPE_DEFAULT: u32 = 0;
pub const COLOR_TYPE_INDEXED: u32 = 1;
pub const COLOR_TYPE_RGB: u32 = 2;

/// Attribute bit flags.
pub const ATTR_BOLD: u32 = 0x8;
pub const ATTR_ITALIC: u32 = 0x10;
pub const ATTR_REVERSE: u32 = 0x20;
pub const ATTR_STRIKE: u32 = 0x40;
pub const ATTR_DIM: u32 = 0x80;
pub const ATTR_UNDERLINE: u32 = 0x1; // Part of decoration mask
pub const ATTR_SELECTED: u32 = 0x100; // Cell is selected (for selection highlighting)

/// Flag for colored glyphs (emoji).
pub const COLORED_GLYPH_FLAG: u32 = 0x80000000;

/// Pre-rendered cursor sprite indices (like Kitty's cursor_shape_map).
/// These sprites are created at fixed indices in the sprite array after initialization.
/// Index 0 is reserved for "no glyph" (empty cell).
pub const CURSOR_SPRITE_BEAM: u32 = 1;      // Bar/beam cursor (vertical line on left)
pub const CURSOR_SPRITE_UNDERLINE: u32 = 2; // Underline cursor (horizontal line at bottom)
pub const CURSOR_SPRITE_HOLLOW: u32 = 3;    // Hollow/unfocused cursor (outline rectangle)

/// Pre-rendered decoration sprite indices (like Kitty's decoration sprites).
/// These are created after cursor sprites and used for text decorations.
/// The shader uses these to render underlines, strikethrough, etc.
pub const DECORATION_SPRITE_STRIKETHROUGH: u32 = 4;    // Strikethrough line
pub const DECORATION_SPRITE_UNDERLINE: u32 = 5;        // Single underline
pub const DECORATION_SPRITE_DOUBLE_UNDERLINE: u32 = 6; // Double underline
pub const DECORATION_SPRITE_UNDERCURL: u32 = 7;        // Wavy/curly underline
pub const DECORATION_SPRITE_DOTTED: u32 = 8;           // Dotted underline
pub const DECORATION_SPRITE_DASHED: u32 = 9;           // Dashed underline

/// First available sprite index for regular glyphs (after reserved cursor and decoration sprites)
pub const FIRST_GLYPH_SPRITE: u32 = 10;

// ═══════════════════════════════════════════════════════════════════════════════
// VERTEX STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

/// Vertex for rendering textured quads.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GlyphVertex {
    pub position: [f32; 2],
    pub uv: [f32; 2],
    pub color: [f32; 4],
    pub bg_color: [f32; 4],
}

impl GlyphVertex {
    pub const ATTRIBS: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
        0 => Float32x2,  // position
        1 => Float32x2,  // uv
        2 => Float32x4,  // color (fg)
        3 => Float32x4,  // bg_color
    ];

    pub fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GlyphVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// EDGE GLOW STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

/// Per-glow instance data (48 bytes, aligned to 16 bytes).
/// Must match GlowInstance in shader.wgsl exactly.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct GlowInstance {
    pub direction: u32,
    pub progress: f32,
    pub color_r: f32,
    pub color_g: f32,
    pub color_b: f32,
    // Pane bounds in pixels
    pub pane_x: f32,
    pub pane_y: f32,
    pub pane_width: f32,
    pub pane_height: f32,
    pub _padding1: f32,
    pub _padding2: f32,
    pub _padding3: f32,
}

/// GPU-compatible edge glow uniform data.
/// Must match the layout in shader.wgsl exactly.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct EdgeGlowUniforms {
    pub screen_width: f32,
    pub screen_height: f32,
    pub terminal_y_offset: f32,
    pub glow_intensity: f32,
    pub glow_count: u32,
    pub _padding: [u32; 3], // Pad to 16-byte alignment before array
    pub glows: [GlowInstance; MAX_EDGE_GLOWS],
}

// ═══════════════════════════════════════════════════════════════════════════════
// IMAGE STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

/// GPU-compatible image uniform data.
/// Must match the layout in image_shader.wgsl exactly.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct ImageUniforms {
    pub screen_width: f32,
    pub screen_height: f32,
    pub pos_x: f32,
    pub pos_y: f32,
    pub display_width: f32,
    pub display_height: f32,
    pub src_x: f32,
    pub src_y: f32,
    pub src_width: f32,
    pub src_height: f32,
    pub _padding1: f32,
    pub _padding2: f32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// INSTANCED CELL RENDERING STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

/// GPU cell data for instanced rendering.
/// Matches GPUCell in glyph_shader.wgsl exactly.
///
/// Like Kitty, we store a sprite_idx that references pre-rendered glyphs in the atlas.
/// This allows us to update GPU buffers with a simple memcpy when content changes,
/// rather than rebuilding vertex buffers every frame.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct GPUCell {
    /// Foreground color (packed: type in low byte, then RGB or index)
    pub fg: u32,
    /// Background color (packed: type in low byte, then RGB or index)
    pub bg: u32,
    /// Decoration foreground color (for underlines, etc.)
    pub decoration_fg: u32,
    /// Sprite index in the sprite info array. High bit set = colored glyph.
    /// 0 = no glyph (space or empty)
    pub sprite_idx: u32,
    /// Cell attributes (bold, italic, reverse, etc.)
    pub attrs: u32,
}

/// Sprite info for glyph positioning.
/// Matches SpriteInfo in glyph_shader.wgsl exactly.
///
/// In Kitty's model, sprites are always cell-sized and glyphs are pre-positioned
/// within the sprite at the correct baseline. The shader just maps the sprite
/// to the cell 1:1, with no offset math needed.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct SpriteInfo {
    /// UV coordinates in atlas (x, y, width, height) - normalized 0-1
    pub uv: [f32; 4],
    /// Atlas layer index (z-coordinate for texture array) and padding
    /// layer is the first f32, second f32 is unused padding
    pub layer: f32,
    pub _padding: f32,
    /// Size in pixels (width, height) - always matches cell dimensions
    pub size: [f32; 2],
}

/// Font cell metrics with integer dimensions (like Kitty's FontCellMetrics).
/// Using integers ensures pixel-perfect alignment and avoids floating-point precision issues.
#[derive(Copy, Clone, Debug)]
pub struct FontCellMetrics {
    /// Cell width in pixels (computed using ceil from font advance).
    pub cell_width: u32,
    /// Cell height in pixels (computed using ceil from font height).
    pub cell_height: u32,
    /// Baseline offset from top of cell in pixels.
    pub baseline: u32,
    /// Y position for underline (from top of cell, in pixels).
    /// Computed from font metrics: ascender - underline_position.
    pub underline_position: u32,
    /// Thickness of underline in pixels.
    pub underline_thickness: u32,
    /// Y position for strikethrough (from top of cell, in pixels).
    /// Typically around 65% of baseline from top.
    pub strikethrough_position: u32,
    /// Thickness of strikethrough in pixels.
    pub strikethrough_thickness: u32,
}

/// Grid parameters uniform for instanced rendering.
/// Matches GridParams in glyph_shader.wgsl exactly.
/// Uses Kitty-style NDC positioning: viewport is set per-pane, so shader
/// works in pure NDC space without needing pixel offsets.
/// Cell dimensions are integers like Kitty for pixel-perfect rendering.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct GridParams {
    pub cols: u32,
    pub rows: u32,
    pub cell_width: u32,
    pub cell_height: u32,
    pub cursor_col: i32,
    pub cursor_row: i32,
    pub cursor_style: u32,
    pub background_opacity: f32,
    // Selection range (-1 values mean no selection)
    pub selection_start_col: i32,
    pub selection_start_row: i32,
    pub selection_end_col: i32,
    pub selection_end_row: i32,
}

/// GPU quad instance for instanced rectangle rendering.
/// Matches Quad in glyph_shader.wgsl exactly.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct Quad {
    /// X position in pixels
    pub x: f32,
    /// Y position in pixels
    pub y: f32,
    /// Width in pixels
    pub width: f32,
    /// Height in pixels
    pub height: f32,
    /// Color (linear RGBA)
    pub color: [f32; 4],
}

/// Parameters for quad rendering.
/// Matches QuadParams in glyph_shader.wgsl exactly.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct QuadParams {
    pub screen_width: f32,
    pub screen_height: f32,
    pub _padding: [f32; 2],
}

/// Parameters for statusline rendering.
/// Matches StatuslineParams in statusline_shader.wgsl exactly.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, Pod, Zeroable)]
pub struct StatuslineParams {
    /// Number of characters in statusline
    pub char_count: u32,
    /// Cell width in pixels
    pub cell_width: f32,
    /// Cell height in pixels
    pub cell_height: f32,
    /// Screen width in pixels
    pub screen_width: f32,
    /// Screen height in pixels
    pub screen_height: f32,
    /// Y offset from top of screen in pixels
    pub y_offset: f32,
    /// Padding for alignment (to match shader struct layout)
    pub _padding: [f32; 2],
}
