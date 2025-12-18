//! GPU-accelerated terminal rendering using wgpu with a glyph atlas.
//! Uses rustybuzz (HarfBuzz port) for text shaping to support font features.

use crate::config::TabBarPosition;
use crate::graphics::{ImageData, ImagePlacement, ImageStorage};
use crate::terminal::{Color, ColorPalette, CursorShape, Direction, Terminal};
use ab_glyph::{Font, FontRef, GlyphId, ScaleFont};
use rustybuzz::UnicodeBuffer;
use ttf_parser::Tag;
use std::cell::{OnceCell, RefCell};
use std::collections::{HashMap, HashSet};
use std::ffi::CStr;
use std::path::PathBuf;
use std::sync::Arc;

// Fontconfig for dynamic font fallback
use fontconfig::Fontconfig;

// FreeType + Cairo for color emoji rendering
use freetype::Library as FtLibrary;
use cairo::{Format, ImageSurface};

/// Pane geometry for multi-pane rendering.
/// Describes where to render a pane within the window.
#[derive(Debug, Clone, Copy)]
pub struct PaneRenderInfo {
    /// Unique identifier for this pane (used to track GPU resources).
    /// Like Kitty's vao_idx, this maps to per-pane GPU buffers and bind groups.
    pub pane_id: u64,
    /// Left edge in pixels.
    pub x: f32,
    /// Top edge in pixels.
    pub y: f32,
    /// Width in pixels.
    pub width: f32,
    /// Height in pixels.
    pub height: f32,
    /// Number of columns.
    pub cols: usize,
    /// Number of rows.
    pub rows: usize,
    /// Whether this is the active pane.
    pub is_active: bool,
    /// Dim factor for this pane (0.0 = fully dimmed, 1.0 = fully bright).
    /// Used for smooth fade animations when switching pane focus.
    pub dim_factor: f32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// PER-PANE GPU RESOURCES (Like Kitty's VAO per window)
// ═══════════════════════════════════════════════════════════════════════════════

/// GPU resources for a single pane.
/// Like Kitty's VAO, each pane gets its own buffers and bind group.
/// This allows uploading each pane's cell data independently before rendering.
pub struct PaneGpuResources {
    /// Cell storage buffer - contains GPUCell array for this pane's visible cells.
    pub cell_buffer: wgpu::Buffer,
    /// Grid parameters uniform buffer for this pane.
    pub grid_params_buffer: wgpu::Buffer,
    /// Bind group for instanced rendering (@group(1)) - references this pane's buffers.
    pub bind_group: wgpu::BindGroup,
    /// Buffer capacity (max cells) - used to detect when buffer needs resizing.
    pub capacity: usize,
}

// ═══════════════════════════════════════════════════════════════════════════════
// STATUSLINE COMPONENTS
// ═══════════════════════════════════════════════════════════════════════════════

/// Color specification for statusline components.
/// Uses the terminal's indexed color palette (0-255), RGB, or default fg.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatuslineColor {
    /// Use the default foreground color.
    Default,
    /// Use an indexed color from the 256-color palette (0-15 for ANSI colors).
    Indexed(u8),
    /// Use an RGB color.
    Rgb(u8, u8, u8),
}

impl Default for StatuslineColor {
    fn default() -> Self {
        StatuslineColor::Default
    }
}

/// A single component/segment of the statusline.
/// Components are rendered left-to-right with optional separators.
#[derive(Debug, Clone)]
pub struct StatuslineComponent {
    /// The text content of this component.
    pub text: String,
    /// Foreground color for this component.
    pub fg: StatuslineColor,
    /// Whether this text should be bold.
    pub bold: bool,
}

impl StatuslineComponent {
    /// Create a new statusline component with default styling.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            fg: StatuslineColor::Default,
            bold: false,
        }
    }
    
    /// Set the foreground color using an indexed palette color.
    pub fn fg(mut self, color_index: u8) -> Self {
        self.fg = StatuslineColor::Indexed(color_index);
        self
    }
    
    /// Set the foreground color using RGB values.
    pub fn rgb_fg(mut self, r: u8, g: u8, b: u8) -> Self {
        self.fg = StatuslineColor::Rgb(r, g, b);
        self
    }
    
    /// Set bold styling.
    pub fn bold(mut self) -> Self {
        self.bold = true;
        self
    }
    
    /// Create a separator component (e.g., "/", " > ", etc.).
    pub fn separator(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            fg: StatuslineColor::Indexed(8), // Dim gray by default
            bold: false,
        }
    }
}

/// A section of the statusline with its own background color.
/// Sections are rendered left-to-right and end with a powerline transition arrow.
#[derive(Debug, Clone)]
pub struct StatuslineSection {
    /// The components within this section.
    pub components: Vec<StatuslineComponent>,
    /// Background color for this section.
    pub bg: StatuslineColor,
}

impl StatuslineSection {
    /// Create a new section with the given indexed background color.
    pub fn new(bg_color: u8) -> Self {
        Self {
            components: Vec::new(),
            bg: StatuslineColor::Indexed(bg_color),
        }
    }
    
    /// Create a new section with an RGB background color.
    pub fn with_rgb_bg(r: u8, g: u8, b: u8) -> Self {
        Self {
            components: Vec::new(),
            bg: StatuslineColor::Rgb(r, g, b),
        }
    }
    
    /// Create a new section with the default (transparent) background.
    pub fn transparent() -> Self {
        Self {
            components: Vec::new(),
            bg: StatuslineColor::Default,
        }
    }
    
    /// Add a component to this section.
    pub fn push(mut self, component: StatuslineComponent) -> Self {
        self.components.push(component);
        self
    }
    
    /// Add multiple components to this section.
    pub fn with_components(mut self, components: Vec<StatuslineComponent>) -> Self {
        self.components = components;
        self
    }
}

/// Content to display in the statusline.
/// Either structured sections (for ZTerm's default CWD/git display) or raw ANSI
/// content (from neovim or other programs that provide their own statusline).
#[derive(Debug, Clone)]
pub enum StatuslineContent {
    /// Structured sections with powerline-style transitions.
    Sections(Vec<StatuslineSection>),
    /// Raw ANSI-formatted string (rendered as-is without section styling).
    Raw(String),
}

impl Default for StatuslineContent {
    fn default() -> Self {
        StatuslineContent::Sections(Vec::new())
    }
}

/// Edge glow animation state for visual feedback when navigation fails.
/// Creates an organic glow effect: a single light node appears at center,
/// then splits into two that travel outward to the corners while fading.
/// Animation logic is handled in the shader (shader.wgsl).
#[derive(Debug, Clone, Copy)]
pub struct EdgeGlow {
    /// Which edge to glow (based on the direction the user tried to navigate).
    pub direction: Direction,
    /// When the animation started.
    pub start_time: std::time::Instant,
    /// Pane bounds - left edge in pixels.
    pub pane_x: f32,
    /// Pane bounds - top edge in pixels.
    pub pane_y: f32,
    /// Pane bounds - width in pixels.
    pub pane_width: f32,
    /// Pane bounds - height in pixels.
    pub pane_height: f32,
}

impl EdgeGlow {
    /// Duration of the glow animation in milliseconds.
    pub const DURATION_MS: u64 = 500;

    /// Create a new edge glow animation constrained to a pane's bounds.
    pub fn new(direction: Direction, pane_x: f32, pane_y: f32, pane_width: f32, pane_height: f32) -> Self {
        Self {
            direction,
            start_time: std::time::Instant::now(),
            pane_x,
            pane_y,
            pane_width,
            pane_height,
        }
    }

    /// Get the current animation progress (0.0 to 1.0).
    pub fn progress(&self) -> f32 {
        let elapsed = self.start_time.elapsed().as_millis() as f32;
        let duration = Self::DURATION_MS as f32;
        (elapsed / duration).min(1.0)
    }

    /// Check if the animation has completed.
    pub fn is_finished(&self) -> bool {
        self.progress() >= 1.0
    }
}

/// Size of the glyph atlas texture.
const ATLAS_SIZE: u32 = 1024;

/// Bytes per pixel in the RGBA atlas (4 for RGBA8).
const ATLAS_BPP: u32 = 4;

/// Cached glyph information.
/// In Kitty's model, all glyphs are stored as cell-sized sprites with the glyph
/// pre-positioned at the correct baseline within the sprite.
#[derive(Clone, Copy, Debug)]
struct GlyphInfo {
    /// UV coordinates in the atlas (left, top, width, height) normalized 0-1.
    uv: [f32; 4],
    /// Size of the sprite in pixels (always cell_width x cell_height).
    size: [f32; 2],
    /// Whether this is a colored glyph (emoji).
    is_colored: bool,
}

/// Wrapper to hold the rustybuzz Face with a 'static lifetime.
/// This is safe because we keep font_data alive for the lifetime of the Renderer.
struct ShapingContext {
    face: rustybuzz::Face<'static>,
    /// OpenType features to enable during shaping (liga, calt, etc.)
    /// Note: This field is kept for potential future use when we need to modify
    /// features per-context. Currently shaping_features on Renderer is used instead.
    #[allow(dead_code)]
    features: Vec<rustybuzz::Feature>,
}

/// Font style variant indices.
/// These map to the indices in font_variants array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub enum FontStyle {
    Regular = 0,
    Bold = 1,
    Italic = 2,
    BoldItalic = 3,
}

impl FontStyle {
    /// Get the font style from bold and italic flags.
    pub fn from_flags(bold: bool, italic: bool) -> Self {
        match (bold, italic) {
            (false, false) => FontStyle::Regular,
            (true, false) => FontStyle::Bold,
            (false, true) => FontStyle::Italic,
            (true, true) => FontStyle::BoldItalic,
        }
    }
}

/// A font variant with its data and parsed references.
struct FontVariant {
    /// Owned font data (kept alive for the lifetime of the font references).
    #[allow(dead_code)]
    data: Box<[u8]>,
    /// ab_glyph font reference for rasterization.
    font: FontRef<'static>,
    /// rustybuzz face for text shaping.
    face: rustybuzz::Face<'static>,
}

/// Result of shaping a text sequence.
#[derive(Clone, Debug)]
struct ShapedGlyphs {
    /// Glyph IDs, advances, offsets, and cluster indices.
    /// Each tuple is (glyph_id, x_advance, x_offset, y_offset, cluster).
    /// x_offset/y_offset are for texture healing - they shift the glyph without affecting advance.
    glyphs: Vec<(u16, f32, f32, f32, u32)>,
}

/// Vertex for rendering textured quads.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GlyphVertex {
    position: [f32; 2],
    uv: [f32; 2],
    color: [f32; 4],
    bg_color: [f32; 4],
}

impl GlyphVertex {
    const ATTRIBS: [wgpu::VertexAttribute; 4] = wgpu::vertex_attr_array![
        0 => Float32x2,  // position
        1 => Float32x2,  // uv
        2 => Float32x4,  // color (fg)
        3 => Float32x4,  // bg_color
    ];

    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<GlyphVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBS,
        }
    }
}

/// Maximum number of simultaneous edge glows.
const MAX_EDGE_GLOWS: usize = 16;

/// Per-glow instance data (48 bytes, aligned to 16 bytes).
/// Must match GlowInstance in shader.wgsl exactly.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GlowInstance {
    direction: u32,
    progress: f32,
    color_r: f32,
    color_g: f32,
    color_b: f32,
    // Pane bounds in pixels
    pane_x: f32,
    pane_y: f32,
    pane_width: f32,
    pane_height: f32,
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
}

/// GPU-compatible edge glow uniform data.
/// Must match the layout in shader.wgsl exactly.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct EdgeGlowUniforms {
    screen_width: f32,
    screen_height: f32,
    terminal_y_offset: f32,
    glow_intensity: f32,
    glow_count: u32,
    _padding: [u32; 3], // Pad to 16-byte alignment before array
    glows: [GlowInstance; MAX_EDGE_GLOWS],
}

/// GPU-compatible image uniform data.
/// Must match the layout in image_shader.wgsl exactly.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ImageUniforms {
    screen_width: f32,
    screen_height: f32,
    pos_x: f32,
    pos_y: f32,
    display_width: f32,
    display_height: f32,
    src_x: f32,
    src_y: f32,
    src_width: f32,
    src_height: f32,
    _padding1: f32,
    _padding2: f32,
}

/// Cached GPU texture for an image.
#[allow(dead_code)]
struct GpuImage {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    uniform_buffer: wgpu::Buffer,
    bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// KITTY-STYLE INSTANCED CELL RENDERING STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

/// GPU cell data for instanced rendering.
/// Matches GPUCell in glyph_shader.wgsl exactly.
/// 
/// Like Kitty, we store a sprite_idx that references pre-rendered glyphs in the atlas.
/// This allows us to update GPU buffers with a simple memcpy when content changes,
/// rather than rebuilding vertex buffers every frame.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
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

/// Sprite info for glyph positioning.
/// Matches SpriteInfo in glyph_shader.wgsl exactly.
/// 
/// In Kitty's model, sprites are always cell-sized and glyphs are pre-positioned
/// within the sprite at the correct baseline. The shader just maps the sprite
/// to the cell 1:1, with no offset math needed.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpriteInfo {
    /// UV coordinates in atlas (x, y, width, height) - normalized 0-1
    pub uv: [f32; 4],
    /// Padding to maintain alignment (previously offset, now unused)
    pub _padding: [f32; 2],
    /// Size in pixels (width, height) - always matches cell dimensions
    pub size: [f32; 2],
}

/// Grid parameters uniform for instanced rendering.
/// Matches GridParams in glyph_shader.wgsl exactly.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GridParams {
    cols: u32,
    rows: u32,
    cell_width: f32,
    cell_height: f32,
    screen_width: f32,
    screen_height: f32,
    x_offset: f32,
    y_offset: f32,
    cursor_col: i32,
    cursor_row: i32,
    cursor_style: u32,
    background_opacity: f32,
    // Selection range (-1 values mean no selection)
    selection_start_col: i32,
    selection_start_row: i32,
    selection_end_col: i32,
    selection_end_row: i32,
}

/// GPU quad instance for instanced rectangle rendering.
/// Matches Quad in glyph_shader.wgsl exactly.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
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
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct QuadParams {
    screen_width: f32,
    screen_height: f32,
    _padding: [f32; 2],
}

/// Parameters for statusline rendering.
/// Matches StatuslineParams in statusline_shader.wgsl exactly.
#[repr(C)]
#[derive(Copy, Clone, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct StatuslineParams {
    /// Number of characters in statusline
    char_count: u32,
    /// Cell width in pixels
    cell_width: f32,
    /// Cell height in pixels
    cell_height: f32,
    /// Screen width in pixels
    screen_width: f32,
    /// Screen height in pixels
    screen_height: f32,
    /// Y offset from top of screen in pixels
    y_offset: f32,
    /// Padding for alignment (to match shader struct layout)
    _padding: [f32; 2],
}

/// Color table uniform containing 256 indexed colors + default fg/bg.
/// Matches ColorTable in glyph_shader.wgsl.
/// Note: We don't use this directly - colors are resolved per-cell on CPU side.
/// This struct is kept for documentation/future use.
#[allow(dead_code)]
struct ColorTable {
    /// 256 indexed colors + default_fg (256) + default_bg (257)
    colors: [[f32; 4]; 258],
}

/// Key for looking up sprites in the sprite map.
/// A sprite is uniquely identified by the glyph content and style.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
struct SpriteKey {
    /// The character or ligature string
    text: String,
    /// Font style (regular, bold, italic, bold-italic)
    style: FontStyle,
    /// Whether this is a colored glyph (emoji)
    colored: bool,
}

/// Target sprite buffer for glyph allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SpriteTarget {
    /// Terminal pane sprites (main sprite buffer)
    Terminal,
    /// Statusline sprites (separate buffer)
    Statusline,
}

/// The terminal renderer.
pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_config: wgpu::SurfaceConfiguration,

    // Glyph rendering pipeline
    glyph_pipeline: wgpu::RenderPipeline,
    glyph_bind_group: wgpu::BindGroup,

    // Edge glow rendering pipeline
    edge_glow_pipeline: wgpu::RenderPipeline,
    edge_glow_bind_group: wgpu::BindGroup,
    edge_glow_uniform_buffer: wgpu::Buffer,

    // Image rendering pipeline (Kitty graphics protocol)
    image_pipeline: wgpu::RenderPipeline,
    image_bind_group_layout: wgpu::BindGroupLayout,
    image_sampler: wgpu::Sampler,
    /// Cached GPU textures for images, keyed by image ID.
    image_textures: HashMap<u32, GpuImage>,

    // Atlas texture
    atlas_texture: wgpu::Texture,
    atlas_data: Vec<u8>,
    atlas_dirty: bool,

    // Font and shaping
    #[allow(dead_code)] // Kept alive for rustybuzz::Face and FontRef which borrow it
    font_data: Box<[u8]>,
    /// Primary font for rasterization (borrows font_data)
    primary_font: FontRef<'static>,
    /// Font style variants: [Regular, Bold, Italic, BoldItalic]
    /// Each entry is Option because some variants may not be available.
    /// Index 0 (Regular) is always Some (same as primary_font's data).
    font_variants: [Option<FontVariant>; 4],
    /// Fallback fonts with their owned data
    fallback_fonts: Vec<(Box<[u8]>, FontRef<'static>)>,
    /// Fontconfig handle for dynamic font discovery (lazy initialized)
    fontconfig: OnceCell<Option<Fontconfig>>,
    /// Set of font paths we've already tried (to avoid reloading)
    tried_font_paths: HashSet<PathBuf>,
    /// Color font renderer (FreeType + Cairo) for emoji - lazy initialized
    /// Using RefCell because ColorFontRenderer needs mutable access to cache font faces
    color_font_renderer: RefCell<Option<ColorFontRenderer>>,
    /// Cache mapping characters to their color font path (if any)
    color_font_cache: HashMap<char, Option<PathBuf>>,
    shaping_ctx: ShapingContext,
    /// OpenType features for shaping (shared across all font variants)
    shaping_features: Vec<rustybuzz::Feature>,
    char_cache: HashMap<char, GlyphInfo>,    // cache char -> rendered glyph
    ligature_cache: HashMap<String, ShapedGlyphs>, // cache multi-char -> shaped glyphs
    /// Glyph cache keyed by (font_style, font_index, glyph_id)
    /// font_style is FontStyle as usize, font_index is 0 for primary, 1+ for fallbacks
    glyph_cache: HashMap<(usize, usize, u16), GlyphInfo>,
    atlas_cursor_x: u32,
    atlas_cursor_y: u32,
    atlas_row_height: u32,

    // Dynamic vertex/index buffers
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    vertex_capacity: usize,
    index_capacity: usize,

    /// Base font size in points (from config).
    base_font_size: f32,
    /// Current scale factor.
    pub scale_factor: f64,
    /// Screen DPI (dots per inch), used for scaling box drawing characters.
    /// Default is 96.0 if not available from the system.
    dpi: f64,
    /// Effective font size in pixels (base_font_size * scale_factor).
    pub font_size: f32,
    /// Scale factor to convert font units to pixels.
    /// This is font_size / height_unscaled, matching ab_glyph's calculation.
    font_units_to_px: f32,
    /// Cell dimensions in pixels.
    pub cell_width: f32,
    pub cell_height: f32,
    /// Baseline offset from top of cell in pixels.
    /// Glyphs are positioned so their baseline sits at this Y position within the cell.
    baseline: f32,
    /// Window dimensions.
    pub width: u32,
    pub height: u32,
    /// Color palette for rendering.
    palette: ColorPalette,
    /// Tab bar position.
    tab_bar_position: TabBarPosition,
    /// Background opacity (0.0 = transparent, 1.0 = opaque).
    background_opacity: f32,
    /// Actual used grid dimensions (set by pane layout, used for centering).
    /// When there are splits, this is the total size of all panes + borders.
    grid_used_width: f32,
    grid_used_height: f32,

    // Reusable vertex/index buffers to avoid per-frame allocations
    bg_vertices: Vec<GlyphVertex>,
    bg_indices: Vec<u32>,
    glyph_vertices: Vec<GlyphVertex>,
    glyph_indices: Vec<u32>,

    // ═══════════════════════════════════════════════════════════════════════════════
    // KITTY-STYLE INSTANCED RENDERING STATE
    // ═══════════════════════════════════════════════════════════════════════════════
    
    /// Sprite map: maps glyph content + style to sprite index.
    /// The sprite index is used in GPUCell.sprite_idx to reference the glyph in the atlas.
    sprite_map: HashMap<SpriteKey, u32>,
    /// Sprite info array: UV coordinates and offsets for each sprite.
    /// Index 0 is reserved for "no glyph" (space).
    sprite_info: Vec<SpriteInfo>,
    /// Next sprite index to allocate.
    next_sprite_idx: u32,
    /// GPU cell buffer for all visible cells (flattened row-major).
    /// Updated only when terminal content changes.
    gpu_cells: Vec<GPUCell>,
    /// Whether the GPU cell buffer needs to be re-uploaded.
    cells_dirty: bool,
    /// Last rendered grid dimensions (cols, rows) to detect resizes.
    last_grid_size: (usize, usize),

    // GPU buffers for instanced rendering
    /// Cell storage buffer - contains GPUCell array for all visible cells.
    cell_buffer: wgpu::Buffer,
    /// Sprite storage buffer - contains SpriteInfo array for all sprites.
    sprite_buffer: wgpu::Buffer,
    /// Current capacity of sprite buffer (number of sprites it can hold).
    sprite_buffer_capacity: usize,
    /// Grid parameters uniform buffer.
    grid_params_buffer: wgpu::Buffer,
    /// Color table uniform buffer (258 colors: 256 indexed + default fg/bg).
    color_table_buffer: wgpu::Buffer,
    /// Bind group for instanced rendering (@group(1)).
    instanced_bind_group: wgpu::BindGroup,
    /// Background pipeline for instanced cell rendering.
    cell_bg_pipeline: wgpu::RenderPipeline,
    /// Glyph pipeline for instanced cell rendering.
    cell_glyph_pipeline: wgpu::RenderPipeline,

    /// Current selection range for rendering (start_col, start_row, end_col, end_row).
    /// If set, cells within this range will be rendered with inverted colors.
    selection: Option<(usize, usize, usize, usize)>,

    // ═══════════════════════════════════════════════════════════════════════════════
    // PER-PANE GPU RESOURCES (Like Kitty's VAO per window)
    // ═══════════════════════════════════════════════════════════════════════════════
    
    /// Bind group layout for instanced rendering - needed to create per-pane bind groups.
    instanced_bind_group_layout: wgpu::BindGroupLayout,
    /// Per-pane GPU resources, keyed by pane_id.
    /// Like Kitty's VAO array, each pane gets its own cell buffer, grid params buffer, and bind group.
    pane_resources: HashMap<u64, PaneGpuResources>,

    // ═══════════════════════════════════════════════════════════════════════════════
    // STATUSLINE RENDERING (dedicated shader and pipeline)
    // ═══════════════════════════════════════════════════════════════════════════════
    
    /// GPU cells for the statusline (single row).
    statusline_gpu_cells: Vec<GPUCell>,
    /// GPU buffer for statusline cells.
    statusline_cell_buffer: wgpu::Buffer,
    /// Maximum columns for statusline (to size buffer appropriately).
    statusline_max_cols: usize,
    /// Statusline params uniform buffer.
    statusline_params_buffer: wgpu::Buffer,
    /// Bind group layout for statusline rendering.
    statusline_bind_group_layout: wgpu::BindGroupLayout,
    /// Bind group for statusline rendering.
    statusline_bind_group: wgpu::BindGroup,
    /// Pipeline for statusline background rendering.
    statusline_bg_pipeline: wgpu::RenderPipeline,
    /// Pipeline for statusline glyph rendering.
    statusline_glyph_pipeline: wgpu::RenderPipeline,
    /// Separate sprite map for statusline (isolated from terminal sprites).
    statusline_sprite_map: HashMap<SpriteKey, u32>,
    /// Sprite info array for statusline.
    statusline_sprite_info: Vec<SpriteInfo>,
    /// Next sprite index for statusline.
    statusline_next_sprite_idx: u32,
    /// GPU buffer for statusline sprites.
    statusline_sprite_buffer: wgpu::Buffer,
    /// Capacity of the statusline sprite buffer.
    statusline_sprite_buffer_capacity: usize,

    // ═══════════════════════════════════════════════════════════════════════════════
    // INSTANCED QUAD RENDERING (for rectangles, borders, overlays, tab bar)
    // ═══════════════════════════════════════════════════════════════════════════════
    
    /// GPU quads for rectangle rendering.
    quads: Vec<Quad>,
    /// GPU buffer for quad instances.
    quad_buffer: wgpu::Buffer,
    /// Maximum number of quads (to size buffer appropriately).
    max_quads: usize,
    /// Quad params uniform buffer.
    quad_params_buffer: wgpu::Buffer,
    /// Pipeline for instanced quad rendering.
    quad_pipeline: wgpu::RenderPipeline,
    /// Bind group for quad rendering.
    quad_bind_group: wgpu::BindGroup,
    
    /// GPU quads for overlay rendering (rendered on top of everything).
    overlay_quads: Vec<Quad>,
    /// GPU buffer for overlay quad instances (separate from main quads).
    overlay_quad_buffer: wgpu::Buffer,
    /// Bind group for overlay quad rendering.
    overlay_quad_bind_group: wgpu::BindGroup,
}

// ═══════════════════════════════════════════════════════════════════════════════
// FONTCONFIG HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Find a font that contains the given character using fontconfig.
/// Returns the path to the font file and whether it's a color font.
/// 
/// For emoji characters (detected via the `emojis` crate), this function
/// explicitly requests a color font from fontconfig, similar to how Kitty
/// handles emoji presentation: FC_FAMILY = "emoji" and FC_COLOR = true.
fn find_font_for_char(_fc: &Fontconfig, c: char) -> Option<(PathBuf, bool)> {
    use fontconfig_sys as fcsys;
    use fcsys::*;
    use fcsys::constants::FC_COLOR;

    // Check if this character is an emoji using the emojis crate (O(1) lookup)
    let char_str = c.to_string();
    let is_emoji = emojis::get(&char_str).is_some();

    unsafe {
        // Create a pattern
        let pat = FcPatternCreate();
        if pat.is_null() {
            return None;
        }

        // Create a charset with the target character
        let charset = FcCharSetCreate();
        if charset.is_null() {
            FcPatternDestroy(pat);
            return None;
        }

        // Add the character to the charset
        FcCharSetAddChar(charset, c as u32);

        // Add the charset to the pattern
        let fc_charset_cstr = CStr::from_bytes_with_nul(b"charset\0").unwrap();
        FcPatternAddCharSet(pat, fc_charset_cstr.as_ptr(), charset);

        // For emoji characters, explicitly request a color font from the "emoji" family
        // This matches Kitty's approach in fontconfig.c:create_fallback_face()
        if is_emoji {
            let fc_family_cstr = CStr::from_bytes_with_nul(b"family\0").unwrap();
            let emoji_family = CStr::from_bytes_with_nul(b"emoji\0").unwrap();
            FcPatternAddString(pat, fc_family_cstr.as_ptr(), emoji_family.as_ptr() as *const u8);
            FcPatternAddBool(pat, FC_COLOR.as_ptr() as *const i8, 1); // Request color font
        }

        // Run substitutions
        FcConfigSubstitute(std::ptr::null_mut(), pat, FcMatchPattern);
        FcDefaultSubstitute(pat);

        // Find matching font
        let mut result = FcResultNoMatch;
        let matched = FcFontMatch(std::ptr::null_mut(), pat, &mut result);

        let font_result = if !matched.is_null() && result == FcResultMatch {
            // Get the file path from the matched pattern
            let mut file_ptr: *mut FcChar8 = std::ptr::null_mut();
            let fc_file_cstr = CStr::from_bytes_with_nul(b"file\0").unwrap();
            if FcPatternGetString(matched, fc_file_cstr.as_ptr(), 0, &mut file_ptr) == FcResultMatch
            {
                let path_cstr = CStr::from_ptr(file_ptr as *const i8);
                let path = PathBuf::from(path_cstr.to_string_lossy().into_owned());
                
                // Check if the font is a color font (FC_COLOR property)
                let mut is_color: i32 = 0;
                let has_color = FcPatternGetBool(matched, FC_COLOR.as_ptr() as *const i8, 0, &mut is_color) == FcResultMatch && is_color != 0;
                
                log::debug!("find_font_for_char: found font for U+{:04X} '{}': {:?} (color={}, requested_emoji={})", 
                           c as u32, c, path, has_color, is_emoji);
                Some((path, has_color))
            } else {
                None
            }
        } else {
            None
        };

        // Cleanup
        if !matched.is_null() {
            FcPatternDestroy(matched);
        }
        FcCharSetDestroy(charset);
        FcPatternDestroy(pat);

        font_result
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// COLOR EMOJI RENDERING (FreeType + Cairo)
// ═══════════════════════════════════════════════════════════════════════════════

/// Find a color font (emoji font) that contains the given character using fontconfig.
/// Returns the path to the font file if found.
fn find_color_font_for_char(c: char) -> Option<PathBuf> {
    use fontconfig_sys as fcsys;
    use fcsys::*;
    use fcsys::constants::{FC_CHARSET, FC_COLOR, FC_FILE};

    log::debug!("find_color_font_for_char: looking for color font for U+{:04X} '{}'", c as u32, c);

    unsafe {
        // Create a pattern
        let pat = FcPatternCreate();
        if pat.is_null() {
            log::debug!("find_color_font_for_char: FcPatternCreate failed");
            return None;
        }

        // Create a charset with the target character
        let charset = FcCharSetCreate();
        if charset.is_null() {
            FcPatternDestroy(pat);
            log::debug!("find_color_font_for_char: FcCharSetCreate failed");
            return None;
        }

        // Add the character to the charset
        FcCharSetAddChar(charset, c as u32);

        // Add the charset to the pattern
        FcPatternAddCharSet(pat, FC_CHARSET.as_ptr() as *const i8, charset);
        
        // Request a color font
        FcPatternAddBool(pat, FC_COLOR.as_ptr() as *const i8, 1); // FcTrue = 1

        // Run substitutions
        FcConfigSubstitute(std::ptr::null_mut(), pat, FcMatchPattern);
        FcDefaultSubstitute(pat);

        // Find matching font
        let mut result = FcResultNoMatch;
        let matched = FcFontMatch(std::ptr::null_mut(), pat, &mut result);

        let font_path = if !matched.is_null() && result == FcResultMatch {
            // Check if the matched font is actually a color font
            let mut is_color: i32 = 0;
            let has_color = FcPatternGetBool(matched, FC_COLOR.as_ptr() as *const i8, 0, &mut is_color) == FcResultMatch && is_color != 0;
            
            log::debug!("find_color_font_for_char: matched font, is_color={}", has_color);
            
            if has_color {
                // Get the file path from the matched pattern
                let mut file_ptr: *mut u8 = std::ptr::null_mut();
                if FcPatternGetString(matched, FC_FILE.as_ptr() as *const i8, 0, &mut file_ptr) == FcResultMatch {
                    let path_cstr = CStr::from_ptr(file_ptr as *const i8);
                    let path = PathBuf::from(path_cstr.to_string_lossy().into_owned());
                    log::debug!("find_color_font_for_char: found color font {:?}", path);
                    Some(path)
                } else {
                    log::debug!("find_color_font_for_char: couldn't get file path");
                    None
                }
            } else {
                log::debug!("find_color_font_for_char: matched font is not a color font");
                None
            }
        } else {
            log::debug!("find_color_font_for_char: no match found (result={:?})", result);
            None
        };

        // Cleanup
        if !matched.is_null() {
            FcPatternDestroy(matched);
        }
        FcCharSetDestroy(charset);
        FcPatternDestroy(pat);

        font_path
    }
}

/// Lazy-initialized color font renderer using FreeType + Cairo.
/// Only created when a color emoji is first encountered.
/// Cairo is required for proper color font rendering (COLR, CBDT, sbix formats).
struct ColorFontRenderer {
    /// FreeType library instance
    ft_library: FtLibrary,
    /// Loaded FreeType faces and their Cairo font faces, keyed by font path
    faces: HashMap<PathBuf, (freetype::Face, cairo::FontFace)>,
    /// Reusable Cairo surface for rendering
    surface: Option<ImageSurface>,
    /// Current surface dimensions
    surface_size: (i32, i32),
}

impl ColorFontRenderer {
    fn new() -> Result<Self, freetype::Error> {
        let ft_library = FtLibrary::init()?;
        Ok(Self {
            ft_library,
            faces: HashMap::new(),
            surface: None,
            surface_size: (0, 0),
        })
    }

    /// Ensure faces are loaded and return font size to set
    fn ensure_faces_loaded(&mut self, path: &PathBuf) -> bool {
        if !self.faces.contains_key(path) {
            match self.ft_library.new_face(path, 0) {
                Ok(ft_face) => {
                    // Create Cairo font face from FreeType face
                    match cairo::FontFace::create_from_ft(&ft_face) {
                        Ok(cairo_face) => {
                            self.faces.insert(path.clone(), (ft_face, cairo_face));
                            true
                        }
                        Err(e) => {
                            log::warn!("Failed to create Cairo font face for {:?}: {:?}", path, e);
                            false
                        }
                    }
                }
                Err(e) => {
                    log::warn!("Failed to load color font {:?}: {:?}", path, e);
                    false
                }
            }
        } else {
            true
        }
    }

    /// Render a color glyph using FreeType + Cairo.
    /// Returns (width, height, RGBA bitmap, offset_x, offset_y) or None if rendering fails.
    fn render_color_glyph(
        &mut self,
        font_path: &PathBuf,
        c: char,
        font_size_px: f32,
        cell_width: u32,
        cell_height: u32,
    ) -> Option<(u32, u32, Vec<u8>, f32, f32)> {
        log::debug!("render_color_glyph: U+{:04X} '{}' font={:?}", c as u32, c, font_path);
        
        // Ensure faces are loaded
        if !self.ensure_faces_loaded(font_path) {
            log::debug!("render_color_glyph: failed to load faces");
            return None;
        }
        log::debug!("render_color_glyph: faces loaded successfully, faces count={}", self.faces.len());
        
        // Get glyph index from FreeType face
        // Note: We do NOT call set_pixel_sizes here because CBDT (bitmap) fonts have fixed sizes
        // and will fail. Cairo handles font sizing internally.
        let glyph_index = {
            let face_entry = self.faces.get(font_path);
            if face_entry.is_none() {
                log::debug!("render_color_glyph: face not found in hashmap after ensure_faces_loaded!");
                return None;
            }
            let (ft_face, _) = face_entry?;
            log::debug!("render_color_glyph: got ft_face, getting char index for U+{:04X}", c as u32);
            let idx = ft_face.get_char_index(c as usize);
            log::debug!("render_color_glyph: FreeType glyph index for U+{:04X} = {:?}", c as u32, idx);
            if idx.is_none() {
                log::debug!("render_color_glyph: glyph index is None - char not in font!");
                return None;
            }
            idx?
        };
        
        // Clone the Cairo font face (it's reference-counted)
        let cairo_face = {
            let (_, cairo_face) = self.faces.get(font_path)?;
            cairo_face.clone()
        };

        // For emoji, we typically render at 2x cell width (double-width character)
        let render_width = (cell_width * 2).max(cell_height) as i32;
        let render_height = cell_height as i32;
        
        log::debug!("render_color_glyph: render size {}x{}", render_width, render_height);
        
        // Ensure we have a large enough surface
        let surface_width = render_width.max(256);
        let surface_height = render_height.max(256);
        
        if self.surface.is_none() || self.surface_size.0 < surface_width || self.surface_size.1 < surface_height {
            let new_width = surface_width.max(self.surface_size.0);
            let new_height = surface_height.max(self.surface_size.1);
            match ImageSurface::create(Format::ARgb32, new_width, new_height) {
                Ok(surface) => {
                    log::debug!("render_color_glyph: created Cairo surface {}x{}", new_width, new_height);
                    self.surface = Some(surface);
                    self.surface_size = (new_width, new_height);
                }
                Err(e) => {
                    log::warn!("Failed to create Cairo surface: {:?}", e);
                    return None;
                }
            }
        }
        
        let surface = self.surface.as_mut()?;
        
        // Create Cairo context
        let cr = match cairo::Context::new(surface) {
            Ok(cr) => cr,
            Err(e) => {
                log::warn!("Failed to create Cairo context: {:?}", e);
                return None;
            }
        };
        
        // Clear the surface
        cr.set_operator(cairo::Operator::Clear);
        cr.paint().ok()?;
        cr.set_operator(cairo::Operator::Over);
        
        // Set the font face and initial size
        cr.set_font_face(&cairo_face);
        
        // Target dimensions for the glyph (2 cells wide, 1 cell tall for emoji)
        let target_width = render_width as f64;
        let target_height = render_height as f64;
        
        // Start with the requested font size and reduce until glyph fits
        // This matches Kitty's fit_cairo_glyph() approach
        let mut current_size = font_size_px as f64;
        let min_size = 2.0;
        
        cr.set_font_size(current_size);
        let mut glyph = cairo::Glyph::new(glyph_index as u64, 0.0, 0.0);
        let mut text_extents = cr.glyph_extents(&[glyph]).ok()?;
        
        while current_size > min_size && (text_extents.width() > target_width || text_extents.height() > target_height) {
            let ratio = (target_width / text_extents.width()).min(target_height / text_extents.height());
            let new_size = (ratio * current_size).max(min_size);
            if new_size >= current_size {
                current_size -= 2.0;
            } else {
                current_size = new_size;
            }
            cr.set_font_size(current_size);
            text_extents = cr.glyph_extents(&[glyph]).ok()?;
        }
        
        log::debug!("render_color_glyph: fitted font size {:.1} (from {:.1}), glyph extents {:.1}x{:.1}", 
                   current_size, font_size_px, text_extents.width(), text_extents.height());
        
        // Get font metrics for positioning with the final size
        let font_extents = cr.font_extents().ok()?;
        log::debug!("render_color_glyph: font extents - ascent={:.1}, descent={:.1}, height={:.1}", 
                   font_extents.ascent(), font_extents.descent(), font_extents.height());
        
        // Create glyph with positioning at baseline
        // y position should be at baseline (ascent from top)
        glyph = cairo::Glyph::new(glyph_index as u64, 0.0, font_extents.ascent());
        
        // Get final glyph extents for sizing
        text_extents = cr.glyph_extents(&[glyph]).ok()?;
        log::debug!("render_color_glyph: text extents - width={:.1}, height={:.1}, x_bearing={:.1}, y_bearing={:.1}, x_advance={:.1}", 
                   text_extents.width(), text_extents.height(), 
                   text_extents.x_bearing(), text_extents.y_bearing(),
                   text_extents.x_advance());
        
        // Set source color to white - the atlas stores colors directly for emoji
        cr.set_source_rgba(1.0, 1.0, 1.0, 1.0);
        
        // Render the glyph
        if let Err(e) = cr.show_glyphs(&[glyph]) {
            log::warn!("render_color_glyph: show_glyphs failed: {:?}", e);
            return None;
        }
        log::debug!("render_color_glyph: cairo show_glyphs succeeded");
        
        // Flush and get surface reference again
        drop(cr); // Drop the context before accessing surface data
        let surface = self.surface.as_mut()?;
        surface.flush();
        
        // Calculate actual glyph bounds
        let glyph_width = text_extents.width().ceil() as u32;
        let glyph_height = text_extents.height().ceil() as u32;
        
        log::debug!("render_color_glyph: glyph size {}x{}", glyph_width, glyph_height);
        
        if glyph_width == 0 || glyph_height == 0 {
            log::debug!("render_color_glyph: zero size glyph, returning None");
            return None;
        }
        
        // The actual rendered area - use the text extents to determine position
        let x_offset = text_extents.x_bearing();
        let y_offset = text_extents.y_bearing();
        
        // Calculate source rectangle in the surface
        let src_x = x_offset.max(0.0) as i32;
        let src_y = (font_extents.ascent() + y_offset).max(0.0) as i32;
        
        log::debug!("render_color_glyph: source rect starts at ({}, {})", src_x, src_y);
        
        // Get surface data
        let stride = surface.stride() as usize;
        let surface_data = surface.data().ok()?;
        
        // Extract the glyph region and convert ARGB -> RGBA
        let out_width = glyph_width.min(render_width as u32);
        let out_height = glyph_height.min(render_height as u32);
        
        let mut rgba = vec![0u8; (out_width * out_height * 4) as usize];
        let mut non_zero_pixels = 0u32;
        let mut has_color = false;
        
        for y in 0..out_height as i32 {
            for x in 0..out_width as i32 {
                let src_pixel_x = src_x + x;
                let src_pixel_y = src_y + y;
                
                if src_pixel_x >= 0 && src_pixel_x < self.surface_size.0 
                   && src_pixel_y >= 0 && src_pixel_y < self.surface_size.1 {
                    let src_idx = (src_pixel_y as usize) * stride + (src_pixel_x as usize) * 4;
                    let dst_idx = (y as usize * out_width as usize + x as usize) * 4;
                    
                    if src_idx + 3 < surface_data.len() {
                        // Cairo uses ARGB in native byte order (on little-endian: BGRA in memory)
                        // We need to convert to RGBA
                        let b = surface_data[src_idx];
                        let g = surface_data[src_idx + 1];
                        let r = surface_data[src_idx + 2];
                        let a = surface_data[src_idx + 3];
                        
                        if a > 0 {
                            non_zero_pixels += 1;
                            // Check if this is actual color (not just white/gray)
                            if r != g || g != b {
                                has_color = true;
                            }
                        }
                        
                        // Un-premultiply alpha if needed (Cairo uses premultiplied alpha)
                        if a > 0 && a < 255 {
                            let inv_alpha = 255.0 / a as f32;
                            rgba[dst_idx] = (r as f32 * inv_alpha).min(255.0) as u8;
                            rgba[dst_idx + 1] = (g as f32 * inv_alpha).min(255.0) as u8;
                            rgba[dst_idx + 2] = (b as f32 * inv_alpha).min(255.0) as u8;
                            rgba[dst_idx + 3] = a;
                        } else {
                            rgba[dst_idx] = r;
                            rgba[dst_idx + 1] = g;
                            rgba[dst_idx + 2] = b;
                            rgba[dst_idx + 3] = a;
                        }
                    }
                }
            }
        }
        
        log::debug!("render_color_glyph: extracted {}x{} pixels, {} non-zero, has_color={}", 
                   out_width, out_height, non_zero_pixels, has_color);
        
        // Check if we actually got any non-transparent pixels
        let has_content = rgba.chunks(4).any(|p| p[3] > 0);
        if !has_content {
            log::debug!("render_color_glyph: no visible content, returning None");
            return None;
        }
        
        // Kitty convention: bitmap_top = -y_bearing (distance from baseline to glyph top)
        let offset_x = text_extents.x_bearing() as f32;
        let offset_y = -text_extents.y_bearing() as f32;
        
        log::debug!("render_color_glyph: SUCCESS - returning {}x{} glyph, offset=({:.1}, {:.1})", 
                   out_width, out_height, offset_x, offset_y);
        
        Some((out_width, out_height, rgba, offset_x, offset_y))
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FONT LOADING HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Try to load a font file and create both ab_glyph and rustybuzz handles.
/// Returns None if the file doesn't exist or can't be parsed.
fn load_font_variant(path: &std::path::Path) -> Option<FontVariant> {
    let data = std::fs::read(path).ok()?.into_boxed_slice();
    
    // Parse with ab_glyph
    let font: FontRef<'static> = {
        let font = FontRef::try_from_slice(&data).ok()?;
        // SAFETY: We keep data alive in the FontVariant struct
        unsafe { std::mem::transmute(font) }
    };
    
    // Parse with rustybuzz
    let face: rustybuzz::Face<'static> = {
        let face = rustybuzz::Face::from_slice(&data, 0)?;
        // SAFETY: We keep data alive in the FontVariant struct
        unsafe { std::mem::transmute(face) }
    };
    
    Some(FontVariant { data, font, face })
}

/// Find font files for a font family using fontconfig.
/// Returns paths for (regular, bold, italic, bold_italic).
/// Any variant that can't be found will be None.
fn find_font_family_variants(family: &str) -> [Option<PathBuf>; 4] {
    use fontconfig_sys as fcsys;
    use fcsys::*;
    use fcsys::constants::{FC_FAMILY, FC_WEIGHT, FC_SLANT, FC_FILE};
    use std::ffi::CString;
    
    let mut results: [Option<PathBuf>; 4] = [None, None, None, None];
    
    // Style queries: (weight, slant) pairs for each variant
    // FC_WEIGHT_REGULAR = 80, FC_WEIGHT_BOLD = 200
    // FC_SLANT_ROMAN = 0, FC_SLANT_ITALIC = 100
    let styles: [(i32, i32); 4] = [
        (80, 0),    // Regular
        (200, 0),   // Bold
        (80, 100),  // Italic
        (200, 100), // BoldItalic
    ];
    
    unsafe {
        let family_cstr = match CString::new(family) {
            Ok(s) => s,
            Err(_) => return results,
        };
        
        for (idx, (weight, slant)) in styles.iter().enumerate() {
            let pat = FcPatternCreate();
            if pat.is_null() {
                continue;
            }
            
            // Set family name
            FcPatternAddString(pat, FC_FAMILY.as_ptr() as *const i8, family_cstr.as_ptr() as *const u8);
            // Set weight
            FcPatternAddInteger(pat, FC_WEIGHT.as_ptr() as *const i8, *weight);
            // Set slant
            FcPatternAddInteger(pat, FC_SLANT.as_ptr() as *const i8, *slant);
            
            FcConfigSubstitute(std::ptr::null_mut(), pat, FcMatchPattern);
            FcDefaultSubstitute(pat);
            
            let mut result: FcResult = FcResultMatch;
            let matched = FcFontMatch(std::ptr::null_mut(), pat, &mut result);
            
            if result == FcResultMatch && !matched.is_null() {
                let mut file_ptr: *mut u8 = std::ptr::null_mut();
                if FcPatternGetString(matched, FC_FILE.as_ptr() as *const i8, 0, &mut file_ptr) == FcResultMatch {
                    if !file_ptr.is_null() {
                        let path_cstr = std::ffi::CStr::from_ptr(file_ptr as *const i8);
                        if let Ok(path_str) = path_cstr.to_str() {
                            results[idx] = Some(PathBuf::from(path_str));
                        }
                    }
                }
                FcPatternDestroy(matched);
            }
            
            FcPatternDestroy(pat);
        }
    }
    
    results
}

/// Load font variants for a font family.
/// Returns array of font variants, with index 0 being the regular font.
/// Falls back to hardcoded paths if fontconfig fails.
fn load_font_family(font_family: Option<&str>) -> (Box<[u8]>, FontRef<'static>, [Option<FontVariant>; 4]) {
    // Try to use fontconfig to find the font family
    if let Some(family) = font_family {
        let paths = find_font_family_variants(family);
        log::info!("Font family '{}' resolved to:", family);
        for (i, path) in paths.iter().enumerate() {
            let style = match i {
                0 => "Regular",
                1 => "Bold",
                2 => "Italic",
                3 => "BoldItalic",
                _ => "Unknown",
            };
            if let Some(p) = path {
                log::info!("  {}: {:?}", style, p);
            }
        }
        
        // Load the regular font (required)
        if let Some(regular_path) = &paths[0] {
            if let Some(regular) = load_font_variant(regular_path) {
                let primary_font = regular.font.clone();
                let font_data = regular.data.clone();
                
                // Load other variants
                let variants: [Option<FontVariant>; 4] = [
                    Some(regular),
                    paths[1].as_ref().and_then(|p| load_font_variant(p)),
                    paths[2].as_ref().and_then(|p| load_font_variant(p)),
                    paths[3].as_ref().and_then(|p| load_font_variant(p)),
                ];
                
                return (font_data, primary_font, variants);
            }
        }
        log::warn!("Failed to load font family '{}', falling back to defaults", family);
    }
    
    // Fallback: try hardcoded paths
    let fallback_fonts = [
        ("/usr/share/fonts/TTF/0xProtoNerdFont-Regular.ttf", 
         "/usr/share/fonts/TTF/0xProtoNerdFont-Bold.ttf",
         "/usr/share/fonts/TTF/0xProtoNerdFont-Italic.ttf",
         "/usr/share/fonts/TTF/0xProtoNerdFont-BoldItalic.ttf"),
        ("/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Regular.ttf",
         "/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Bold.ttf",
         "/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Italic.ttf",
         "/usr/share/fonts/TTF/JetBrainsMonoNerdFont-BoldItalic.ttf"),
        ("/usr/share/fonts/TTF/JetBrainsMono-Regular.ttf",
         "/usr/share/fonts/TTF/JetBrainsMono-Bold.ttf",
         "/usr/share/fonts/TTF/JetBrainsMono-Italic.ttf",
         "/usr/share/fonts/TTF/JetBrainsMono-BoldItalic.ttf"),
    ];
    
    for (regular, bold, italic, bold_italic) in fallback_fonts {
        let regular_path = std::path::Path::new(regular);
        if let Some(regular_variant) = load_font_variant(regular_path) {
            let primary_font = regular_variant.font.clone();
            let font_data = regular_variant.data.clone();
            
            let variants: [Option<FontVariant>; 4] = [
                Some(regular_variant),
                load_font_variant(std::path::Path::new(bold)),
                load_font_variant(std::path::Path::new(italic)),
                load_font_variant(std::path::Path::new(bold_italic)),
            ];
            
            log::info!("Loaded font from fallback paths:");
            log::info!("  Regular: {}", regular);
            if variants[1].is_some() { log::info!("  Bold: {}", bold); }
            if variants[2].is_some() { log::info!("  Italic: {}", italic); }
            if variants[3].is_some() { log::info!("  BoldItalic: {}", bold_italic); }
            
            return (font_data, primary_font, variants);
        }
    }
    
    // Last resort: try NotoSansMono
    let noto_regular = std::path::Path::new("/usr/share/fonts/noto/NotoSansMono-Regular.ttf");
    if let Some(regular_variant) = load_font_variant(noto_regular) {
        let primary_font = regular_variant.font.clone();
        let font_data = regular_variant.data.clone();
        let variants: [Option<FontVariant>; 4] = [Some(regular_variant), None, None, None];
        log::info!("Loaded NotoSansMono as fallback");
        return (font_data, primary_font, variants);
    }
    
    panic!("Failed to load any monospace font");
}

// ═══════════════════════════════════════════════════════════════════════════════
// BOX DRAWING HELPER TYPES
// ═══════════════════════════════════════════════════════════════════════════════

/// Which corner of a cell for corner triangle rendering
#[derive(Clone, Copy)]
enum Corner {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

/// Supersampled canvas for anti-aliased rendering of box drawing characters.
/// Renders at 4x resolution then downsamples for smooth edges.
struct SupersampledCanvas {
    bitmap: Vec<u8>,
    width: usize,
    height: usize,
    ss_width: usize,
    ss_height: usize,
}

impl SupersampledCanvas {
    const FACTOR: usize = 4;

    fn new(width: usize, height: usize) -> Self {
        let ss_width = width * Self::FACTOR;
        let ss_height = height * Self::FACTOR;
        Self {
            bitmap: vec![0u8; ss_width * ss_height],
            width,
            height,
            ss_width,
            ss_height,
        }
    }

    /// Blend a pixel with alpha compositing
    #[inline]
    fn blend_pixel(&mut self, x: usize, y: usize, alpha: f64) {
        if x < self.ss_width && y < self.ss_height && alpha > 0.0 {
            let old_alpha = self.bitmap[y * self.ss_width + x] as f64 / 255.0;
            let new_alpha = alpha + (1.0 - alpha) * old_alpha;
            self.bitmap[y * self.ss_width + x] = (new_alpha * 255.0) as u8;
        }
    }

    /// Draw a thick line along x-axis with y computed by a function
    fn thick_line_h(&mut self, x1: usize, x2: usize, y_at_x: impl Fn(usize) -> f64, thickness: usize) {
        let delta = thickness / 2;
        let extra = thickness % 2;
        for x in x1..x2.min(self.ss_width) {
            let y_center = y_at_x(x) as i32;
            let y_start = (y_center - delta as i32).max(0) as usize;
            let y_end = ((y_center + delta as i32 + extra as i32) as usize).min(self.ss_height);
            for y in y_start..y_end {
                self.bitmap[y * self.ss_width + x] = 255;
            }
        }
    }

    /// Draw a thick point (for curve rendering)
    fn thick_point(&mut self, x: f64, y: f64, thickness: f64) {
        let half = thickness / 2.0;
        let x_start = (x - half).max(0.0) as usize;
        let x_end = ((x + half).ceil() as usize).min(self.ss_width);
        let y_start = (y - half).max(0.0) as usize;
        let y_end = ((y + half).ceil() as usize).min(self.ss_height);
        for py in y_start..y_end {
            for px in x_start..x_end {
                self.bitmap[py * self.ss_width + px] = 255;
            }
        }
    }

    /// Fill a corner triangle. Corner specifies which corner of the cell the right angle is in.
    /// inverted=false fills the triangle itself, inverted=true fills everything except the triangle.
    fn fill_corner_triangle(&mut self, corner: Corner, inverted: bool) {
        let w = self.ss_width;
        let h = self.ss_height;
        // Use (ss_size - 1) as max coordinate, matching Kitty's approach
        let max_x = (w - 1) as f64;
        let max_y = (h - 1) as f64;

        for py in 0..h {
            let y = py as f64;
            for px in 0..w {
                let x = px as f64;

                // Calculate edge y for this x based on corner
                // The diagonal goes from one corner to the opposite corner
                let (edge_y, fill_below) = match corner {
                    // BottomLeft: diagonal from (0, max_y) to (max_x, 0), fill below the line
                    Corner::BottomLeft => (max_y - (max_y / max_x) * x, true),
                    // TopLeft: diagonal from (0, 0) to (max_x, max_y), fill above the line
                    Corner::TopLeft => ((max_y / max_x) * x, false),
                    // BottomRight: diagonal from (0, 0) to (max_x, max_y), fill below the line
                    Corner::BottomRight => ((max_y / max_x) * x, true),
                    // TopRight: diagonal from (0, max_y) to (max_x, 0), fill above the line
                    Corner::TopRight => (max_y - (max_y / max_x) * x, false),
                };

                let in_triangle = if fill_below { y >= edge_y } else { y <= edge_y };
                let should_fill = if inverted { !in_triangle } else { in_triangle };

                if should_fill {
                    self.bitmap[py * w + px] = 255;
                }
            }
        }
    }

    /// Fill a powerline arrow triangle pointing left or right.
    /// Uses Kitty's approach: define line equations and fill based on y_limits.
    fn fill_powerline_arrow(&mut self, left: bool, inverted: bool) {
        let w = self.ss_width;
        let h = self.ss_height;
        // Use (ss_size - 1) as max coordinate, matching Kitty's approach
        let max_x = (w - 1) as f64;
        let max_y = (h - 1) as f64;
        let mid_y = max_y / 2.0;

        for py in 0..h {
            let y = py as f64;
            for px in 0..w {
                let x = px as f64;

                let (upper_y, lower_y) = if left {
                    // Left-pointing: tip at (0, mid), base from (max_x, 0) to (max_x, max_y)
                    // Upper line: from (max_x, 0) to (0, mid_y) -> y = mid_y/max_x * (max_x - x)
                    // Lower line: from (max_x, max_y) to (0, mid_y) -> y = max_y - mid_y/max_x * (max_x - x)
                    let upper = (mid_y / max_x) * (max_x - x);
                    let lower = max_y - (mid_y / max_x) * (max_x - x);
                    (upper, lower)
                } else {
                    // Right-pointing: tip at (max_x, mid), base from (0, 0) to (0, max_y)
                    // Upper line: from (0, 0) to (max_x, mid_y) -> y = mid_y/max_x * x
                    // Lower line: from (0, max_y) to (max_x, mid_y) -> y = max_y - mid_y/max_x * x
                    let upper = (mid_y / max_x) * x;
                    let lower = max_y - (mid_y / max_x) * x;
                    (upper, lower)
                };

                let in_shape = y >= upper_y && y <= lower_y;
                let should_fill = if inverted { !in_shape } else { in_shape };

                if should_fill {
                    self.bitmap[py * w + px] = 255;
                }
            }
        }
    }

    /// Draw powerline arrow outline (chevron shape - two diagonal lines meeting at a point)
    fn stroke_powerline_arrow(&mut self, left: bool, thickness: usize) {
        let w = self.ss_width;
        let h = self.ss_height;
        // Use (ss_size - 1) as max coordinate, matching Kitty's approach
        let max_x = (w - 1) as f64;
        let max_y = (h - 1) as f64;
        let mid_y = max_y / 2.0;

        if left {
            // Left-pointing chevron <: lines meeting at (0, mid_y)
            self.thick_line_h(0, w, |x| (mid_y / max_x) * (max_x - x as f64), thickness);
            self.thick_line_h(0, w, |x| max_y - (mid_y / max_x) * (max_x - x as f64), thickness);
        } else {
            // Right-pointing chevron >: lines meeting at (max_x, mid_y)
            self.thick_line_h(0, w, |x| (mid_y / max_x) * x as f64, thickness);
            self.thick_line_h(0, w, |x| max_y - (mid_y / max_x) * x as f64, thickness);
        }
    }

    /// Fill region using a Bezier curve (for "D" shaped powerline semicircles).
    /// The curve goes from top-left to bottom-left, bulging to the right.
    /// Bezier: P0=(0,0), P1=(cx,0), P2=(cx,h), P3=(0,h)
    /// This creates a "D" shape that bulges to the right.
    fn fill_bezier_d(&mut self, left: bool) {
        let w = self.ss_width;
        let h = self.ss_height;
        // Use (ss_size - 1) as max coordinate, matching Kitty's approach
        let max_x = (w - 1) as f64;
        let max_y = (h - 1) as f64;

        // Control point X: determines how far the curve bulges
        // At t=0.5, bezier_x = 0.75 * cx, so cx = max_x / 0.75 to reach max_x
        let cx = max_x / 0.75;

        for py in 0..h {
            let target_y = py as f64;

            // Find t where y(t) = target_y
            // y(t) = max_y * t^2 * (3 - 2t)
            let t = Self::find_t_for_bezier_y(max_y, target_y);

            // Calculate x at this t
            let u = 1.0 - t;
            let bx = 3.0 * cx * t * u;

            // Clamp to cell width
            let x_extent = (bx.round() as usize).min(w - 1);

            if left {
                // Left semicircle: fill from (w - 1 - x_extent) to (w - 1)
                let start_x = (w - 1).saturating_sub(x_extent);
                for px in start_x..w {
                    self.bitmap[py * w + px] = 255;
                }
            } else {
                // Right semicircle: fill from 0 to x_extent
                for px in 0..=x_extent {
                    self.bitmap[py * w + px] = 255;
                }
            }
        }
    }

    /// Binary search for t where bezier_y(t) ≈ target_y
    /// y(t) = h * t^2 * (3 - 2t), monotonically increasing from 0 to h
    fn find_t_for_bezier_y(h: f64, target_y: f64) -> f64 {
        let mut t_low = 0.0;
        let mut t_high = 1.0;

        for _ in 0..20 {
            let t_mid = (t_low + t_high) / 2.0;
            let y = h * t_mid * t_mid * (3.0 - 2.0 * t_mid);

            if y < target_y {
                t_low = t_mid;
            } else {
                t_high = t_mid;
            }
        }

        (t_low + t_high) / 2.0
    }

    /// Draw Bezier curve outline (for outline powerline semicircles)
    fn stroke_bezier_d(&mut self, left: bool, thickness: f64) {
        let w = self.ss_width;
        let h = self.ss_height;
        // Use (ss_size - 1) as max coordinate, matching Kitty's approach
        let max_x = (w - 1) as f64;
        let max_y = (h - 1) as f64;
        let cx = max_x / 0.75;

        let steps = (h * 2) as usize;
        for i in 0..=steps {
            let t = i as f64 / steps as f64;
            let u = 1.0 - t;
            let bx = 3.0 * cx * t * u;
            let by = max_y * t * t * (3.0 - 2.0 * t);

            // Clamp bx to cell width
            let bx_clamped = bx.min(max_x);
            let x = if left { max_x - bx_clamped } else { bx_clamped };
            self.thick_point(x, by, thickness);
        }
    }

    /// Fill a circle centered in the cell
    fn fill_circle(&mut self, radius_factor: f64) {
        let cx = self.ss_width as f64 / 2.0;
        let cy = self.ss_height as f64 / 2.0;
        let radius = (cx.min(cy) - 0.5) * radius_factor;
        let limit = radius * radius;

        for py in 0..self.ss_height {
            for px in 0..self.ss_width {
                let dx = px as f64 - cx;
                let dy = py as f64 - cy;
                if dx * dx + dy * dy <= limit {
                    self.bitmap[py * self.ss_width + px] = 255;
                }
            }
        }
    }

    /// Fill a circle with a specific radius
    fn fill_circle_radius(&mut self, radius: f64) {
        let cx = self.ss_width as f64 / 2.0;
        let cy = self.ss_height as f64 / 2.0;
        let limit = radius * radius;

        for py in 0..self.ss_height {
            for px in 0..self.ss_width {
                let dx = px as f64 - cx;
                let dy = py as f64 - cy;
                if dx * dx + dy * dy <= limit {
                    self.bitmap[py * self.ss_width + px] = 255;
                }
            }
        }
    }

    /// Stroke a circle outline with anti-aliasing
    fn stroke_circle(&mut self, radius: f64, line_width: f64) {
        let cx = self.ss_width as f64 / 2.0;
        let cy = self.ss_height as f64 / 2.0;
        let half_thickness = line_width / 2.0;

        for py in 0..self.ss_height {
            for px in 0..self.ss_width {
                let pixel_x = px as f64 + 0.5;
                let pixel_y = py as f64 + 0.5;

                let dx = pixel_x - cx;
                let dy = pixel_y - cy;
                let dist_to_center = (dx * dx + dy * dy).sqrt();
                let distance = (dist_to_center - radius).abs();

                let alpha = (half_thickness - distance + 0.5).clamp(0.0, 1.0);
                self.blend_pixel(px, py, alpha);
            }
        }
    }

    /// Stroke an arc (partial circle) with anti-aliasing
    fn stroke_arc(&mut self, radius: f64, line_width: f64, start_angle: f64, end_angle: f64) {
        let cx = self.ss_width as f64 / 2.0;
        let cy = self.ss_height as f64 / 2.0;
        let half_thickness = line_width / 2.0;

        // Sample points along the arc
        let num_samples = (self.ss_width.max(self.ss_height) * 2) as usize;
        let angle_range = end_angle - start_angle;

        for i in 0..=num_samples {
            let t = i as f64 / num_samples as f64;
            let angle = start_angle + angle_range * t;
            let arc_x = cx + radius * angle.cos();
            let arc_y = cy + radius * angle.sin();

            // Draw anti-aliased point at this position
            self.stroke_point_aa(arc_x, arc_y, half_thickness);
        }
    }

    /// Draw an anti-aliased point
    fn stroke_point_aa(&mut self, x: f64, y: f64, half_thickness: f64) {
        let x_start = ((x - half_thickness - 1.0).max(0.0)) as usize;
        let x_end = ((x + half_thickness + 2.0) as usize).min(self.ss_width);
        let y_start = ((y - half_thickness - 1.0).max(0.0)) as usize;
        let y_end = ((y + half_thickness + 2.0) as usize).min(self.ss_height);

        for py in y_start..y_end {
            for px in x_start..x_end {
                let pixel_x = px as f64 + 0.5;
                let pixel_y = py as f64 + 0.5;
                let dx = pixel_x - x;
                let dy = pixel_y - y;
                let distance = (dx * dx + dy * dy).sqrt();

                let alpha = (half_thickness - distance + 0.5).clamp(0.0, 1.0);
                self.blend_pixel(px, py, alpha);
            }
        }
    }

    /// Downsample to final resolution
    fn downsample(&self, output: &mut [u8]) {
        for y in 0..self.height {
            for x in 0..self.width {
                let src_x = x * Self::FACTOR;
                let src_y = y * Self::FACTOR;
                let mut total: u32 = 0;
                for sy in src_y..src_y + Self::FACTOR {
                    for sx in src_x..src_x + Self::FACTOR {
                        total += self.bitmap[sy * self.ss_width + sx] as u32;
                    }
                }
                output[y * self.width + x] = (total / (Self::FACTOR * Self::FACTOR) as u32) as u8;
            }
        }
    }
}

use crate::config::Config;

impl Renderer {
    /// Creates a new renderer for the given window.
    pub async fn new(window: Arc<winit::window::Window>, config: &Config) -> Self {
        let size = window.inner_size();
        let scale_factor = window.scale_factor();

        // Calculate DPI from scale factor
        // Standard assumption: scale_factor 1.0 = 96 DPI (Windows/Linux default)
        // macOS uses 72 as base DPI, but winit normalizes this
        let dpi = 96.0 * scale_factor;

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });

        let surface = instance.create_surface(window).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find a suitable GPU adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Terminal Device"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::default(),
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .expect("Failed to create device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);

        // Select alpha mode for transparency support
        // Prefer PreMultiplied for proper transparency blending, fall back to others
        let alpha_mode = if config.background_opacity < 1.0 {
            if surface_caps.alpha_modes.contains(&wgpu::CompositeAlphaMode::PreMultiplied) {
                wgpu::CompositeAlphaMode::PreMultiplied
            } else if surface_caps.alpha_modes.contains(&wgpu::CompositeAlphaMode::PostMultiplied) {
                wgpu::CompositeAlphaMode::PostMultiplied
            } else {
                log::warn!("Transparency requested but compositor doesn't support alpha blending");
                surface_caps.alpha_modes[0]
            }
        } else {
            surface_caps.alpha_modes[0]
        };

        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            // Use Immediate for lowest latency (no vsync wait)
            // Fall back to Mailbox if Immediate not supported
            present_mode: if surface_caps.present_modes.contains(&wgpu::PresentMode::Immediate) {
                wgpu::PresentMode::Immediate
            } else {
                wgpu::PresentMode::Mailbox
            },
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &surface_config);

        // Load primary font and font variants (regular, bold, italic, bold-italic)
        let (font_data, primary_font, font_variants) = load_font_family(config.font_family.as_deref());

        // Fontconfig will be initialized lazily on first fallback font lookup
        // Start with empty fallback fonts - will be loaded on-demand via fontconfig
        let fallback_fonts: Vec<(Box<[u8]>, FontRef<'static>)> = Vec::new();
        let tried_font_paths: HashSet<PathBuf> = HashSet::new();

        // Enable OpenType features for ligatures and contextual alternates
        // These are the standard features used by coding fonts like Fira Code, JetBrains Mono, etc.
        let shaping_features = vec![
            // Standard ligatures (fi, fl, etc.)
            rustybuzz::Feature::new(Tag::from_bytes(b"liga"), 1, ..),
            // Contextual alternates (programming ligatures like ->, =>, etc.)
            rustybuzz::Feature::new(Tag::from_bytes(b"calt"), 1, ..),
            // Discretionary ligatures (optional ligatures)
            rustybuzz::Feature::new(Tag::from_bytes(b"dlig"), 1, ..),
        ];

        // Create shaping context using the regular font variant's face
        // The face is borrowed from font_variants[0], which is always Some
        let shaping_ctx = {
            let regular_variant = font_variants[0].as_ref()
                .expect("Regular font variant should always be present");
            ShapingContext { 
                face: regular_variant.face.clone(), 
                features: shaping_features.clone(),
            }
        };

        // Calculate cell dimensions from font metrics using ab_glyph
        // 
        // The config font_size is in pixels. Scale by display scale factor for HiDPI.
        // Round to integer for pixel-perfect glyph rendering.
        let base_font_size = config.font_size;
        let font_size = (base_font_size * scale_factor as f32).round();
        
        let scaled_font = primary_font.as_scaled(font_size);
        
        // Get advance width for 'M' (em width)
        let m_glyph_id = primary_font.glyph_id('M');
        let cell_width = scaled_font.h_advance(m_glyph_id).round();

        // Use font line metrics for cell height
        // ab_glyph's height() = ascent - descent (where descent is negative)
        let cell_height = scaled_font.height().round();
        
        // Calculate baseline offset from top of cell.
        // The baseline is where the bottom of uppercase letters sit.
        // ascent is the distance from baseline to top of tallest glyph.
        let baseline = scaled_font.ascent().round();
        
        // Calculate the correct scale factor for converting font units to pixels.
        // This matches ab_glyph's calculation: scale / height_unscaled
        // where height_unscaled = ascent - descent (the font's natural line height).
        let font_units_to_px = font_size / primary_font.height_unscaled();

        // Create atlas texture
        let atlas_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Glyph Atlas"),
            size: wgpu::Extent3d {
                width: ATLAS_SIZE,
                height: ATLAS_SIZE,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        let atlas_view = atlas_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let atlas_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create bind group layout
        let glyph_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Glyph Bind Group Layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
            });

        let glyph_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Glyph Bind Group"),
            layout: &glyph_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&atlas_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&atlas_sampler),
                },
            ],
        });

        // Create shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Glyph Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("glyph_shader.wgsl").into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Glyph Pipeline Layout"),
            bind_group_layouts: &[&glyph_bind_group_layout],
            push_constant_ranges: &[],
        });

        let glyph_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Glyph Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[GlyphVertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // ═══════════════════════════════════════════════════════════════════════════════
        // EDGE GLOW PIPELINE SETUP
        // ═══════════════════════════════════════════════════════════════════════════════

        // Create edge glow shader
        let edge_glow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Edge Glow Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        // Create uniform buffer for edge glow parameters
        let edge_glow_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Edge Glow Uniform Buffer"),
            size: std::mem::size_of::<EdgeGlowUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout for edge glow
        let edge_glow_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Edge Glow Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create bind group for edge glow
        let edge_glow_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Edge Glow Bind Group"),
            layout: &edge_glow_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: edge_glow_uniform_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline layout for edge glow
        let edge_glow_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Edge Glow Pipeline Layout"),
            bind_group_layouts: &[&edge_glow_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create edge glow render pipeline
        let edge_glow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Edge Glow Pipeline"),
            layout: Some(&edge_glow_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &edge_glow_shader,
                entry_point: Some("vs_main"),
                buffers: &[], // Fullscreen triangle, no vertex buffer needed
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &edge_glow_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    // Premultiplied alpha blending for proper glow compositing
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // ═══════════════════════════════════════════════════════════════════════════════
        // IMAGE PIPELINE SETUP (Kitty Graphics Protocol)
        // ═══════════════════════════════════════════════════════════════════════════════

        // Create image shader
        let image_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Image Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("image_shader.wgsl").into()),
        });

        // Create sampler for images (linear filtering for smooth scaling)
        let image_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Image Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create bind group layout for images
        let image_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Image Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        // Create pipeline layout for images
        let image_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Image Pipeline Layout"),
            bind_group_layouts: &[&image_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create image render pipeline
        let image_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Image Pipeline"),
            layout: Some(&image_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &image_shader,
                entry_point: Some("vs_main"),
                buffers: &[], // Quad generated in shader
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &image_shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    // Premultiplied alpha blending (shader outputs premultiplied)
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                        alpha: wgpu::BlendComponent {
                            src_factor: wgpu::BlendFactor::One,
                            dst_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                            operation: wgpu::BlendOperation::Add,
                        },
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Create initial buffers with some capacity
        let initial_vertex_capacity = 4096;
        let initial_index_capacity = 6144;

        let vertex_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Glyph Vertex Buffer"),
            size: (initial_vertex_capacity * std::mem::size_of::<GlyphVertex>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let index_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Glyph Index Buffer"),
            size: (initial_index_capacity * std::mem::size_of::<u32>()) as u64,
            usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // ═══════════════════════════════════════════════════════════════════════════════
        // KITTY-STYLE INSTANCED RENDERING SETUP
        // ═══════════════════════════════════════════════════════════════════════════════

        // Initial capacity: 200x50 grid = 10000 cells, 4096 sprites
        let initial_cells = 10000;
        let initial_sprites = 4096;

        // Cell storage buffer - holds GPUCell array
        let cell_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Storage Buffer"),
            size: (initial_cells * std::mem::size_of::<GPUCell>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Statusline cell buffer - single row, max 500 columns
        let statusline_max_cols = 500;
        let statusline_cell_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Statusline Cell Buffer"),
            size: (statusline_max_cols * std::mem::size_of::<GPUCell>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Sprite storage buffer - holds SpriteInfo array
        let sprite_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sprite Storage Buffer"),
            size: (initial_sprites * std::mem::size_of::<SpriteInfo>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Grid parameters uniform buffer
        let grid_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grid Params Buffer"),
            size: std::mem::size_of::<GridParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Color table uniform buffer - 258 colors * 16 bytes (vec4<f32>)
        let color_table_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Color Table Buffer"),
            size: (258 * 16) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group layout for instanced rendering (@group(1))
        let instanced_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Instanced Bind Group Layout"),
            entries: &[
                // @binding(0): color_table (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(1): grid_params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(2): cells (storage, read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(3): sprites (storage, read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Create bind group for instanced rendering
        let instanced_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Instanced Bind Group"),
            layout: &instanced_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: color_table_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grid_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cell_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: sprite_buffer.as_entire_binding(),
                },
            ],
        });

        // ═══════════════════════════════════════════════════════════════════════════════
        // STATUSLINE RENDERING SETUP (dedicated shader and pipeline)
        // ═══════════════════════════════════════════════════════════════════════════════
        
        let statusline_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Statusline Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("statusline_shader.wgsl").into()),
        });
        
        // Statusline params uniform buffer
        let statusline_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Statusline Params Buffer"),
            size: std::mem::size_of::<StatuslineParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Statusline sprite buffer (separate from terminal sprites)
        let statusline_sprite_buffer_capacity = 256; // Smaller than terminal - statusline has fewer glyphs
        let statusline_sprite_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Statusline Sprite Buffer"),
            size: (statusline_sprite_buffer_capacity * std::mem::size_of::<SpriteInfo>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create bind group layout for statusline rendering (@group(1))
        // Same bindings as instanced_bind_group_layout but with StatuslineParams instead of GridParams
        let statusline_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Statusline Bind Group Layout"),
            entries: &[
                // @binding(0): color_table (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(1): statusline_params (uniform)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(2): cells (storage, read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // @binding(3): sprites (storage, read-only)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        // Create bind group for statusline rendering
        let statusline_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Statusline Bind Group"),
            layout: &statusline_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: color_table_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: statusline_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: statusline_cell_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: statusline_sprite_buffer.as_entire_binding(),
                },
            ],
        });
        
        // Create pipeline layout for statusline rendering
        let statusline_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Statusline Pipeline Layout"),
            bind_group_layouts: &[&glyph_bind_group_layout, &statusline_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Statusline background pipeline
        let statusline_bg_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Statusline Background Pipeline"),
            layout: Some(&statusline_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &statusline_shader,
                entry_point: Some("vs_statusline_bg"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &statusline_shader,
                entry_point: Some("fs_statusline"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });
        
        // Statusline glyph pipeline
        let statusline_glyph_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Statusline Glyph Pipeline"),
            layout: Some(&statusline_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &statusline_shader,
                entry_point: Some("vs_statusline_glyph"),
                buffers: &[],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &statusline_shader,
                entry_point: Some("fs_statusline"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Create pipeline layout for instanced cell rendering
        // Uses @group(0) for atlas texture/sampler and @group(1) for cell data
        let instanced_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Instanced Pipeline Layout"),
            bind_group_layouts: &[&glyph_bind_group_layout, &instanced_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Background pipeline - uses vs_cell_bg and fs_cell
        let cell_bg_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Cell Background Pipeline"),
            layout: Some(&instanced_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_cell_bg"),
                buffers: &[], // No vertex buffers - uses instancing
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_cell"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // Glyph pipeline - uses vs_cell_glyph and fs_cell
        let cell_glyph_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Cell Glyph Pipeline"),
            layout: Some(&instanced_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_cell_glyph"),
                buffers: &[], // No vertex buffers - uses instancing
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_cell"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        // ═══════════════════════════════════════════════════════════════════════════════
        // INSTANCED QUAD RENDERING SETUP
        // For rectangles, borders, overlays, and tab bar backgrounds
        // ═══════════════════════════════════════════════════════════════════════════════
        
        let quad_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Quad Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("quad_shader.wgsl").into()),
        });
        
        // Maximum number of quads we can render in one batch
        let max_quads: usize = 256;
        
        // Quad buffer for instance data
        let quad_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Quad Buffer"),
            size: (max_quads * std::mem::size_of::<Quad>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Quad params uniform buffer
        let quad_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Quad Params Buffer"),
            size: std::mem::size_of::<QuadParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Bind group layout for quad rendering
        let quad_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Quad Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        // Bind group for quad rendering
        let quad_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Quad Bind Group"),
            layout: &quad_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: quad_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: quad_buffer.as_entire_binding(),
                },
            ],
        });
        
        // Overlay quad buffer for instance data (separate from main quads)
        let overlay_quad_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Overlay Quad Buffer"),
            size: (max_quads * std::mem::size_of::<Quad>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Bind group for overlay quad rendering (uses same params buffer but different quad buffer)
        let overlay_quad_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Overlay Quad Bind Group"),
            layout: &quad_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: quad_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: overlay_quad_buffer.as_entire_binding(),
                },
            ],
        });
        
        // Pipeline layout for quad rendering
        let quad_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Quad Pipeline Layout"),
            bind_group_layouts: &[&quad_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Quad pipeline
        let quad_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Quad Pipeline"),
            layout: Some(&quad_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &quad_shader,
                entry_point: Some("vs_quad"),
                buffers: &[], // No vertex buffers - uses instancing
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &quad_shader,
                entry_point: Some("fs_quad"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        Self {
            surface,
            device,
            queue,
            surface_config,
            glyph_pipeline,
            glyph_bind_group,
            edge_glow_pipeline,
            edge_glow_bind_group,
            edge_glow_uniform_buffer,
            image_pipeline,
            image_bind_group_layout,
            image_sampler,
            image_textures: HashMap::new(),
            atlas_texture,
            atlas_data: vec![0u8; (ATLAS_SIZE * ATLAS_SIZE * ATLAS_BPP) as usize],
            atlas_dirty: false,
            font_data,
            primary_font,
            font_variants,
            fallback_fonts,
            fontconfig: OnceCell::new(),
            tried_font_paths,
            color_font_renderer: RefCell::new(None),
            color_font_cache: HashMap::new(),
            shaping_ctx,
            shaping_features,
            char_cache: HashMap::new(),
            ligature_cache: HashMap::new(),
            glyph_cache: HashMap::new(),
            atlas_cursor_x: 0,
            atlas_cursor_y: 0,
            atlas_row_height: 0,
            vertex_buffer,
            index_buffer,
            vertex_capacity: initial_vertex_capacity,
            index_capacity: initial_index_capacity,
            base_font_size,
            scale_factor,
            dpi,
            font_size,
            font_units_to_px,
            cell_width,
            cell_height,
            baseline,
            width: size.width,
            height: size.height,
            palette: ColorPalette::default(),
            tab_bar_position: config.tab_bar_position,
            background_opacity: config.background_opacity.clamp(0.0, 1.0),
            // Initialize with single-pane dimensions (will be updated by layout)
            grid_used_width: 0.0,
            grid_used_height: 0.0,
            // Pre-allocate reusable buffers for rendering
            bg_vertices: Vec::with_capacity(4096),
            bg_indices: Vec::with_capacity(6144),
            glyph_vertices: Vec::with_capacity(4096),
            glyph_indices: Vec::with_capacity(6144),
            // Kitty-style instanced rendering state
            sprite_map: HashMap::new(),
            // Index 0 is reserved for "no glyph" (space/empty)
            sprite_info: vec![SpriteInfo::default()],
            next_sprite_idx: 1,
            gpu_cells: Vec::new(),
            cells_dirty: true,
            last_grid_size: (0, 0),
            // GPU buffers for instanced rendering
            cell_buffer,
            sprite_buffer,
            sprite_buffer_capacity: initial_sprites,
            grid_params_buffer,
            color_table_buffer,
            instanced_bind_group,
            cell_bg_pipeline,
            cell_glyph_pipeline,
            selection: None,
            // Per-pane GPU resources (like Kitty's VAO per window)
            instanced_bind_group_layout,
            pane_resources: HashMap::new(),
            // Statusline rendering (dedicated shader and pipeline)
            statusline_gpu_cells: Vec::with_capacity(statusline_max_cols),
            statusline_cell_buffer,
            statusline_max_cols,
            statusline_params_buffer,
            statusline_bind_group_layout,
            statusline_bind_group,
            statusline_bg_pipeline,
            statusline_glyph_pipeline,
            statusline_sprite_map: HashMap::new(),
            statusline_sprite_info: vec![SpriteInfo::default()], // Index 0 reserved for "no glyph"
            statusline_next_sprite_idx: 1,
            statusline_sprite_buffer,
            statusline_sprite_buffer_capacity,
            // Instanced quad rendering
            quads: Vec::with_capacity(max_quads),
            quad_buffer,
            max_quads,
            quad_params_buffer,
            quad_pipeline,
            quad_bind_group,
            overlay_quads: Vec::with_capacity(32),
            overlay_quad_buffer,
            overlay_quad_bind_group,
        }
    }

    /// Returns the height of the tab bar in pixels (one cell height, or 0 if hidden).
    pub fn tab_bar_height(&self) -> f32 {
        match self.tab_bar_position {
            TabBarPosition::Hidden => 0.0,
            _ => self.cell_height,
        }
    }

    /// Returns the height of the statusline in pixels (one cell height).
    pub fn statusline_height(&self) -> f32 {
        self.cell_height
    }

    /// Returns the Y position where the statusline starts.
    /// The statusline is rendered below the tab bar (if top) or above it (if bottom).
    pub fn statusline_y(&self) -> f32 {
        match self.tab_bar_position {
            TabBarPosition::Top => self.tab_bar_height(),
            TabBarPosition::Bottom => self.height as f32 - self.tab_bar_height() - self.statusline_height(),
            TabBarPosition::Hidden => 0.0,
        }
    }

    /// Returns the Y offset where the terminal content starts.
    /// Accounts for both the tab bar and the statusline.
    pub fn terminal_y_offset(&self) -> f32 {
        match self.tab_bar_position {
            TabBarPosition::Top => self.tab_bar_height() + self.statusline_height(),
            TabBarPosition::Hidden => self.statusline_height(),
            _ => 0.0,
        }
    }

    /// Sets the current selection range for highlighting.
    /// Pass None to clear the selection.
    /// The selection is specified as (start_col, start_row, end_col, end_row) in normalized order.
    pub fn set_selection(&mut self, selection: Option<(usize, usize, usize, usize)>) {
        self.selection = selection;
    }

    /// Resizes the rendering surface.
    pub fn resize(&mut self, new_width: u32, new_height: u32) {
        if new_width > 0 && new_height > 0 {
            self.width = new_width;
            self.height = new_height;
            self.surface_config.width = new_width;
            self.surface_config.height = new_height;
            self.surface.configure(&self.device, &self.surface_config);
        }
    }

    /// Calculates terminal dimensions in cells, accounting for tab bar and statusline.
    pub fn terminal_size(&self) -> (usize, usize) {
        let available_height = self.height as f32 - self.tab_bar_height() - self.statusline_height();
        let cols = (self.width as f32 / self.cell_width).floor() as usize;
        let rows = (available_height / self.cell_height).floor() as usize;
        (cols.max(1), rows.max(1))
    }

    /// Returns the raw available pixel dimensions for the terminal grid area.
    /// This is the space available for panes before any cell alignment.
    pub fn available_grid_space(&self) -> (f32, f32) {
        let available_width = self.width as f32;
        let available_height = self.height as f32 - self.tab_bar_height() - self.statusline_height();
        (available_width, available_height)
    }

    /// Sets the actual used grid dimensions (from pane layout).
    /// This is called after layout to ensure centering accounts for splits and borders.
    pub fn set_grid_used_dimensions(&mut self, width: f32, height: f32) {
        self.grid_used_width = width;
        self.grid_used_height = height;
    }

    /// Returns the horizontal offset needed to center the cell grid in the window.
    /// Uses the actual used width from pane layout if set, otherwise calculates from terminal_size.
    pub fn grid_x_offset(&self) -> f32 {
        let used_width = if self.grid_used_width > 0.0 {
            self.grid_used_width
        } else {
            let (cols, _) = self.terminal_size();
            cols as f32 * self.cell_width
        };
        (self.width as f32 - used_width) / 2.0
    }

    /// Returns the vertical offset needed to center the cell grid in the terminal area.
    /// Uses the actual used height from pane layout if set, otherwise calculates from terminal_size.
    pub fn grid_y_offset(&self) -> f32 {
        let used_height = if self.grid_used_height > 0.0 {
            self.grid_used_height
        } else {
            let (_, rows) = self.terminal_size();
            rows as f32 * self.cell_height
        };
        let available_height = self.height as f32 - self.tab_bar_height() - self.statusline_height();
        (available_height - used_height) / 2.0
    }

    /// Calculates screen-space bounds for edge glow given pane geometry.
    /// Takes pane coordinates in grid-relative space and transforms them to screen coordinates,
    /// extending to fill the terminal grid area (but not into tab bar or statusline).
    /// Returns (screen_x, screen_y, width, height) for the glow mask area.
    pub fn calculate_edge_glow_bounds(&self, pane_x: f32, pane_y: f32, pane_width: f32, pane_height: f32) -> (f32, f32, f32, f32) {
        let grid_x_offset = self.grid_x_offset();
        let grid_y_offset = self.grid_y_offset();
        let terminal_y_offset = self.terminal_y_offset();
        let (available_width, available_height) = self.available_grid_space();
        
        // Calculate the terminal grid area boundaries in screen coordinates
        // This is the area where content is rendered, excluding tab bar and statusline
        let grid_top = terminal_y_offset;
        let grid_bottom = terminal_y_offset + available_height;
        let grid_left = 0.0_f32;
        let grid_right = self.width as f32;
        
        log::debug!("calculate_edge_glow_bounds: pane=({}, {}, {}, {})", pane_x, pane_y, pane_width, pane_height);
        log::debug!("  grid area: top={}, bottom={}, left={}, right={}", grid_top, grid_bottom, grid_left, grid_right);
        log::debug!("  offsets: grid_x={}, grid_y={}, terminal_y={}", grid_x_offset, grid_y_offset, terminal_y_offset);
        
        // Transform pane coordinates to screen space (same as border rendering)
        let mut screen_x = grid_x_offset + pane_x;
        let mut screen_y = terminal_y_offset + grid_y_offset + pane_y;
        let mut width = pane_width;
        let mut height = pane_height;
        
        log::debug!("  initial screen: ({}, {}, {}, {})", screen_x, screen_y, width, height);
        
        // Use a larger epsilon to account for cell-alignment gaps in split panes
        // With cell-aligned splits, gaps can be up to one cell height
        let epsilon = self.cell_height.max(self.cell_width);
        
        // Left edge at screen boundary - extend to screen left edge
        if pane_x < epsilon {
            width += screen_x - grid_left;
            screen_x = grid_left;
        }
        
        // Right edge at screen boundary - extend to screen right edge
        if (pane_x + pane_width) >= available_width - epsilon {
            width = grid_right - screen_x;
        }
        
        // Top edge at grid boundary - extend to grid top (respects tab bar/statusline at top)
        if pane_y < epsilon {
            height += screen_y - grid_top;
            screen_y = grid_top;
        }
        
        // Bottom edge at grid boundary - extend to grid bottom (respects tab bar/statusline at bottom)
        if (pane_y + pane_height) >= available_height - epsilon {
            height = grid_bottom - screen_y;
        }
        
        log::debug!("  final screen: ({}, {}, {}, {})", screen_x, screen_y, width, height);
        
        (screen_x, screen_y, width, height)
    }

    /// Calculates screen-space bounds for dim overlay given pane geometry.
    /// Takes pane coordinates in grid-relative space and transforms them to screen coordinates,
    /// extending to fill the terminal grid area (but not into tab bar or statusline).
    /// This is identical to calculate_edge_glow_bounds - ensures dim overlays cover the full
    /// pane area including centering margins, matching edge glow behavior.
    /// Returns (screen_x, screen_y, width, height) for the overlay area.
    pub fn calculate_dim_overlay_bounds(&self, pane_x: f32, pane_y: f32, pane_width: f32, pane_height: f32) -> (f32, f32, f32, f32) {
        let grid_x_offset = self.grid_x_offset();
        let grid_y_offset = self.grid_y_offset();
        let terminal_y_offset = self.terminal_y_offset();
        let (available_width, available_height) = self.available_grid_space();
        
        // Calculate the terminal grid area boundaries in screen coordinates
        // This is the area where content is rendered, excluding tab bar and statusline
        let grid_top = terminal_y_offset;
        let grid_bottom = terminal_y_offset + available_height;
        let grid_left = 0.0_f32;
        let grid_right = self.width as f32;
        
        // Transform pane coordinates to screen space (same as border rendering)
        let mut screen_x = grid_x_offset + pane_x;
        let mut screen_y = terminal_y_offset + grid_y_offset + pane_y;
        let mut width = pane_width;
        let mut height = pane_height;
        
        // Use a larger epsilon to account for cell-alignment gaps in split panes
        // With cell-aligned splits, gaps can be up to one cell height
        let epsilon = self.cell_height.max(self.cell_width);
        
        // Left edge at screen boundary - extend to screen left edge
        if pane_x < epsilon {
            width += screen_x - grid_left;
            screen_x = grid_left;
        }
        
        // Right edge at screen boundary - extend to screen right edge
        if (pane_x + pane_width) >= available_width - epsilon {
            width = grid_right - screen_x;
        }
        
        // Top edge at grid boundary - extend to grid top (respects tab bar/statusline at top)
        if pane_y < epsilon {
            height += screen_y - grid_top;
            screen_y = grid_top;
        }
        
        // Bottom edge at grid boundary - extend to grid bottom (respects tab bar/statusline at bottom)
        if (pane_y + pane_height) >= available_height - epsilon {
            height = grid_bottom - screen_y;
        }
        
        (screen_x, screen_y, width, height)
    }

    /// Converts a pixel position to a terminal cell position.
    /// Returns None if the position is outside the terminal area (e.g., in the tab bar or statusline).
    pub fn pixel_to_cell(&self, x: f64, y: f64) -> Option<(usize, usize)> {
        let terminal_y_offset = self.terminal_y_offset();
        let tab_bar_height = self.tab_bar_height();
        let statusline_height = self.statusline_height();
        let grid_x_offset = self.grid_x_offset();
        let grid_y_offset = self.grid_y_offset();
        let height = self.height as f32;

        // Check if position is in the tab bar or statusline area
        match self.tab_bar_position {
            TabBarPosition::Top => {
                // Tab bar at top, statusline below it
                if (y as f32) < tab_bar_height + statusline_height {
                    return None;
                }
            }
            TabBarPosition::Bottom => {
                // Statusline above tab bar, both at bottom
                let statusline_y = height - tab_bar_height - statusline_height;
                if (y as f32) >= statusline_y {
                    return None;
                }
            }
            TabBarPosition::Hidden => {
                // Just statusline at top
                if (y as f32) < statusline_height {
                    return None;
                }
            }
        }

        // Adjust position to be relative to the centered grid
        let grid_x = x as f32 - grid_x_offset;
        let grid_y = y as f32 - terminal_y_offset - grid_y_offset;

        // Check if position is in the padding area (outside the centered grid)
        if grid_x < 0.0 || grid_y < 0.0 {
            return None;
        }

        // Calculate cell position
        let col = (grid_x / self.cell_width).floor() as usize;
        let row = (grid_y / self.cell_height).floor() as usize;

        // Get terminal dimensions to check bounds
        let (max_cols, max_rows) = self.terminal_size();

        // Return None if outside the grid bounds
        if col >= max_cols || row >= max_rows {
            return None;
        }

        Some((col, row))
    }

    /// Updates the scale factor and recalculates font/cell dimensions.
    /// Returns true if the cell dimensions changed (terminal needs resize).
    pub fn set_scale_factor(&mut self, new_scale: f64) -> bool {
        if (self.scale_factor - new_scale).abs() < 0.001 {
            return false;
        }

        let old_cell_width = self.cell_width;
        let old_cell_height = self.cell_height;

        self.scale_factor = new_scale;
        self.dpi = 96.0 * new_scale;
        
        // Font size in pixels, rounded for pixel-perfect rendering
        self.font_size = (self.base_font_size * new_scale as f32).round();

        // Recalculate cell dimensions using ab_glyph
        let scaled_font = self.primary_font.as_scaled(self.font_size);
        let m_glyph_id = self.primary_font.glyph_id('M');
        self.cell_width = scaled_font.h_advance(m_glyph_id).round();
        self.cell_height = scaled_font.height().round();
        
        // Update the font units to pixels scale factor
        self.font_units_to_px = self.font_size / self.primary_font.height_unscaled();

        log::info!(
            "Scale factor changed to {}: font {}px -> {}px, cell: {}x{}",
            new_scale, self.base_font_size, self.font_size, self.cell_width, self.cell_height
        );

        // Clear all glyph caches - they were rendered at the old size
        self.char_cache.clear();
        self.ligature_cache.clear();
        self.glyph_cache.clear();

        // Reset atlas and sprite tracking
        self.atlas_cursor_x = 0;
        self.atlas_cursor_y = 0;
        self.atlas_row_height = 0;
        self.atlas_data.fill(0);
        self.atlas_dirty = true;
        
        // Clear sprite maps since sprite indices are now invalid
        self.sprite_map.clear();
        self.sprite_info.clear();
        self.sprite_info.push(SpriteInfo::default());
        self.next_sprite_idx = 1;
        self.cells_dirty = true;
        
        self.statusline_sprite_map.clear();
        self.statusline_sprite_info.clear();
        self.statusline_sprite_info.push(SpriteInfo::default());
        self.statusline_next_sprite_idx = 1;

        // Return true if cell dimensions changed
        (self.cell_width - old_cell_width).abs() > 0.01
            || (self.cell_height - old_cell_height).abs() > 0.01
    }

    /// Set the background opacity for transparent terminal rendering.
    pub fn set_background_opacity(&mut self, opacity: f32) {
        self.background_opacity = opacity.clamp(0.0, 1.0);
    }

    /// Set the tab bar position.
    pub fn set_tab_bar_position(&mut self, position: TabBarPosition) {
        self.tab_bar_position = position;
    }

    /// Set the base font size and recalculate cell dimensions.
    /// Returns true if the cell dimensions changed (terminal needs resize).
    pub fn set_font_size(&mut self, size: f32) -> bool {
        if (self.base_font_size - size).abs() < 0.01 {
            return false;
        }

        let old_cell_width = self.cell_width;
        let old_cell_height = self.cell_height;

        self.base_font_size = size;
        
        // Font size in pixels, rounded for pixel-perfect rendering
        self.font_size = (size * self.scale_factor as f32).round();

        // Recalculate cell dimensions using ab_glyph
        let scaled_font = self.primary_font.as_scaled(self.font_size);
        let m_glyph_id = self.primary_font.glyph_id('M');
        self.cell_width = scaled_font.h_advance(m_glyph_id).round();
        self.cell_height = scaled_font.height().round();
        
        // Update the font units to pixels scale factor
        self.font_units_to_px = self.font_size / self.primary_font.height_unscaled();

        log::info!(
            "Font size changed to {}px -> {}px, cell: {}x{}",
            size, self.font_size, self.cell_width, self.cell_height
        );

        // Clear all glyph caches - they were rendered at the old size
        self.char_cache.clear();
        self.ligature_cache.clear();
        self.glyph_cache.clear();

        // Reset atlas and sprite tracking
        self.atlas_cursor_x = 0;
        self.atlas_cursor_y = 0;
        self.atlas_row_height = 0;
        self.atlas_data.fill(0);
        self.atlas_dirty = true;
        
        // Clear sprite maps since sprite indices are now invalid
        self.sprite_map.clear();
        self.sprite_info.clear();
        self.sprite_info.push(SpriteInfo::default());
        self.next_sprite_idx = 1;
        self.cells_dirty = true;
        
        self.statusline_sprite_map.clear();
        self.statusline_sprite_info.clear();
        self.statusline_sprite_info.push(SpriteInfo::default());
        self.statusline_next_sprite_idx = 1;

        // Return true if cell dimensions changed
        (self.cell_width - old_cell_width).abs() > 0.01
            || (self.cell_height - old_cell_height).abs() > 0.01
    }

    /// Reset the glyph atlas when it becomes full.
    /// This clears all cached glyphs and resets the atlas cursor.
    fn reset_atlas(&mut self) {
        log::info!("Resetting glyph atlas (was full)");
        
        // Clear all glyph caches - they need to be re-rasterized
        self.char_cache.clear();
        self.ligature_cache.clear();
        self.glyph_cache.clear();
        
        // Also clear sprite map since sprite indices are now invalid
        self.sprite_map.clear();
        self.sprite_info.clear();
        self.sprite_info.push(SpriteInfo::default()); // Index 0 = no glyph
        self.next_sprite_idx = 1;
        self.cells_dirty = true; // Force re-upload of cell data
        
        // Also clear statusline sprite tracking - they share the same atlas
        self.statusline_sprite_map.clear();
        self.statusline_sprite_info.clear();
        self.statusline_sprite_info.push(SpriteInfo::default()); // Index 0 = no glyph
        self.statusline_next_sprite_idx = 1;
        
        // Reset atlas cursor and data
        self.atlas_cursor_x = 0;
        self.atlas_cursor_y = 0;
        self.atlas_row_height = 0;
        self.atlas_data.fill(0);
        self.atlas_dirty = true;
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // KITTY-STYLE SPRITE AND CELL HELPERS
    // ═══════════════════════════════════════════════════════════════════════════════

    /// Pack a terminal Color into u32 format for GPU.
    /// Format: type in low byte, then color data in higher bytes.
    #[inline]
    fn pack_color(color: &Color) -> u32 {
        match color {
            Color::Default => COLOR_TYPE_DEFAULT,
            Color::Indexed(idx) => COLOR_TYPE_INDEXED | ((*idx as u32) << 8),
            Color::Rgb(r, g, b) => {
                COLOR_TYPE_RGB | ((*r as u32) << 8) | ((*g as u32) << 16) | ((*b as u32) << 24)
            }
        }
    }

    /// Pack cell attributes into u32 format for GPU.
    #[inline]
    fn pack_attrs(bold: bool, italic: bool, underline: bool) -> u32 {
        let mut attrs = 0u32;
        if bold { attrs |= ATTR_BOLD; }
        if italic { attrs |= ATTR_ITALIC; }
        if underline { attrs |= ATTR_UNDERLINE; }
        attrs
    }

    /// Pack a StatuslineColor into u32 format for GPU.
    #[inline]
    fn pack_statusline_color(color: StatuslineColor) -> u32 {
        match color {
            StatuslineColor::Default => COLOR_TYPE_DEFAULT,
            StatuslineColor::Indexed(idx) => COLOR_TYPE_INDEXED | ((idx as u32) << 8),
            StatuslineColor::Rgb(r, g, b) => {
                COLOR_TYPE_RGB | ((r as u32) << 8) | ((g as u32) << 16) | ((b as u32) << 24)
            }
        }
    }

    /// Get or create a sprite index for a character.
    /// Returns (sprite_idx, is_colored).
    /// 
    /// This uses the same approach as Kitty: shape the text with HarfBuzz using
    /// the appropriate font variant (regular, bold, italic, bold-italic), then
    /// rasterize the resulting glyph ID with the styled font.
    /// 
    /// The `target` parameter specifies which sprite buffer to use:
    /// - `SpriteTarget::Terminal` uses the main terminal sprite buffer
    /// - `SpriteTarget::Statusline` uses the separate statusline sprite buffer
    fn get_or_create_sprite_for(&mut self, c: char, style: FontStyle, target: SpriteTarget) -> (u32, bool) {
        // Skip spaces and null characters - they use sprite index 0
        if c == ' ' || c == '\0' {
            return (0, false);
        }
        
        // Select the appropriate sprite tracking based on target
        let (sprite_map, _sprite_info, _next_sprite_idx) = match target {
            SpriteTarget::Terminal => (
                &mut self.sprite_map,
                &mut self.sprite_info,
                &mut self.next_sprite_idx,
            ),
            SpriteTarget::Statusline => (
                &mut self.statusline_sprite_map,
                &mut self.statusline_sprite_info,
                &mut self.statusline_next_sprite_idx,
            ),
        };
        
        // Check if we already have this sprite
        let key = SpriteKey {
            text: c.to_string(),
            style,
            colored: false, // Will check for emoji below
        };
        
        if let Some(&idx) = sprite_map.get(&key) {
            // Check if it's a colored glyph
            let is_colored = (idx & COLORED_GLYPH_FLAG) != 0;
            return (idx, is_colored);
        }
        
        // Check for emoji with color key
        let color_key = SpriteKey {
            text: c.to_string(),
            style,
            colored: true,
        };
        if let Some(&idx) = sprite_map.get(&color_key) {
            return (idx, true);
        }
        
        // Need to rasterize this glyph
        // First, check if it's an emoji
        let char_str = c.to_string();
        let is_emoji = emojis::get(&char_str).is_some();
        
        // For emoji, box-drawing, and multi-cell symbols (PUA/dingbats), use rasterize_char
        // which has scaling logic for oversized glyphs. Regular text uses HarfBuzz shaping.
        let glyph = if is_emoji || Self::is_box_drawing(c) || Self::is_multicell_symbol(c) {
            // These don't need style variants or use rasterize_char for scaling
            self.rasterize_char(c)
        } else {
            // Shape the single character with HarfBuzz using the styled font
            // This gets us the correct glyph ID for the styled font variant
            let shaped = self.shape_text_with_style(&char_str, style);
            
            if shaped.glyphs.is_empty() {
                // Fallback to regular rasterization if shaping fails
                self.rasterize_char(c)
            } else {
                // Get the glyph ID from shaping
                let (glyph_id, _x_advance, _x_offset, _y_offset, _cluster) = shaped.glyphs[0];
                
                // If glyph_id is 0, the font doesn't have this character (.notdef)
                // Fall back to rasterize_char which has full font fallback support
                if glyph_id == 0 {
                    self.rasterize_char(c)
                } else {
                    // Rasterize with the styled font
                    self.get_glyph_by_id_with_style(glyph_id, style)
                }
            }
        };
        
        // If glyph has no size, return 0
        if glyph.size[0] <= 0.0 || glyph.size[1] <= 0.0 {
            return (0, false);
        }
        
        // Create sprite info from glyph info
        // In Kitty's model, glyphs are pre-positioned in cell-sized sprites,
        // so no offset is needed - the shader just maps sprite to cell 1:1
        let sprite = SpriteInfo {
            uv: glyph.uv,
            _padding: [0.0, 0.0],
            size: glyph.size,
        };
        
        // Re-borrow the sprite tracking for the target (needed after self borrows above)
        let (sprite_map, sprite_info, next_sprite_idx) = match target {
            SpriteTarget::Terminal => (
                &mut self.sprite_map,
                &mut self.sprite_info,
                &mut self.next_sprite_idx,
            ),
            SpriteTarget::Statusline => (
                &mut self.statusline_sprite_map,
                &mut self.statusline_sprite_info,
                &mut self.statusline_next_sprite_idx,
            ),
        };
        
        // Allocate new sprite index
        let sprite_idx = *next_sprite_idx;
        *next_sprite_idx += 1;
        
        // Add to sprite info array (ensure we have enough capacity)
        while sprite_info.len() <= sprite_idx as usize {
            sprite_info.push(SpriteInfo::default());
        }
        sprite_info[sprite_idx as usize] = sprite;
        
        // Mark as colored if it's an emoji
        let final_idx = if is_emoji || glyph.is_colored {
            sprite_idx | COLORED_GLYPH_FLAG
        } else {
            sprite_idx
        };
        
        // Cache the mapping
        let cache_key = SpriteKey {
            text: c.to_string(),
            style,
            colored: is_emoji || glyph.is_colored,
        };
        sprite_map.insert(cache_key, final_idx);
        
        (final_idx, is_emoji || glyph.is_colored)
    }
    
    /// Get or create a sprite index for a character in the terminal sprite buffer.
    /// Returns (sprite_idx, is_colored).
    /// 
    /// This is a convenience wrapper around `get_or_create_sprite_for` that uses
    /// the terminal sprite buffer.
    fn get_or_create_sprite(&mut self, c: char, style: FontStyle) -> (u32, bool) {
        self.get_or_create_sprite_for(c, style, SpriteTarget::Terminal)
    }

    /// Convert terminal cells to GPU cells for a visible row.
    /// This is called when terminal content changes to update the GPU buffer.
    /// 
    /// Note: This method cannot take &mut self because it's called from update_gpu_cells
    /// which needs to borrow both self (for sprite lookups) and self.gpu_cells (for output).
    /// Instead, we pass in the necessary state explicitly.
    fn cells_to_gpu_row_static(
        row: &[crate::terminal::Cell],
        gpu_row: &mut [GPUCell],
        cols: usize,
        sprite_map: &HashMap<SpriteKey, u32>,
    ) {
        let mut col = 0;
        while col < cols.min(row.len()) {
            let cell = &row[col];
            
            // Skip wide character continuations - they share the sprite of the previous cell
            if cell.wide_continuation {
                gpu_row[col] = GPUCell {
                    fg: Self::pack_color(&cell.fg_color),
                    bg: Self::pack_color(&cell.bg_color),
                    decoration_fg: 0,
                    sprite_idx: 0, // No glyph for continuation
                    attrs: Self::pack_attrs(cell.bold, cell.italic, cell.underline),
                };
                col += 1;
                continue;
            }
            
            // Get font style
            let style = FontStyle::from_flags(cell.bold, cell.italic);
            let c = cell.character;
            
            // Check for symbol+empty multi-cell pattern
            // Like Kitty, look for symbol character followed by empty cells
            if c != ' ' && c != '\0' && Self::is_multicell_symbol(c) && !Self::is_box_drawing(c) {
                // Count trailing empty cells to determine if this is a multi-cell group
                let mut num_empty = 0;
                const MAX_EXTRA_CELLS: usize = 4;
                
                while col + num_empty + 1 < row.len() && num_empty < MAX_EXTRA_CELLS {
                    let next_char = row[col + num_empty + 1].character;
                    // Check for space, en-space, or empty/null cell
                    if next_char == ' ' || next_char == '\u{2002}' || next_char == '\0' {
                        num_empty += 1;
                    } else {
                        break;
                    }
                }
                
                if num_empty > 0 {
                    let total_cells = 1 + num_empty;
                    
                    // Try to find multi-cell sprites - check colored first, then non-colored
                    // This avoids expensive emoji detection in the hot path
                    let first_key_colored = SpriteKey {
                        text: format!("{}_0", c),
                        style,
                        colored: true,
                    };
                    let first_key_normal = SpriteKey {
                        text: format!("{}_0", c),
                        style,
                        colored: false,
                    };
                    
                    let (first_sprite, is_colored) = if let Some(&sprite) = sprite_map.get(&first_key_colored) {
                        (Some(sprite), true)
                    } else if let Some(&sprite) = sprite_map.get(&first_key_normal) {
                        (Some(sprite), false)
                    } else {
                        (None, false)
                    };
                    
                    if let Some(first_sprite) = first_sprite {
                        // Use multi-cell sprites for each cell in the group
                        for cell_idx in 0..total_cells {
                            if col + cell_idx >= cols {
                                break;
                            }
                            
                            let sprite_idx = if cell_idx == 0 {
                                first_sprite
                            } else {
                                let key = SpriteKey {
                                    text: format!("{}_{}", c, cell_idx),
                                    style,
                                    colored: is_colored,
                                };
                                sprite_map.get(&key).copied().unwrap_or(0)
                            };
                            
                            // For colored glyphs (emoji), set the COLORED_GLYPH_FLAG so the shader
                            // knows to use the atlas color directly instead of applying fg color
                            let final_sprite_idx = if is_colored {
                                sprite_idx | COLORED_GLYPH_FLAG
                            } else {
                                sprite_idx
                            };
                            
                            // Use the symbol cell's foreground color for all cells in the group
                            let current_cell = &row[col + cell_idx];
                            gpu_row[col + cell_idx] = GPUCell {
                                fg: Self::pack_color(&cell.fg_color),
                                bg: Self::pack_color(&current_cell.bg_color),
                                decoration_fg: 0,
                                sprite_idx: final_sprite_idx,
                                attrs: Self::pack_attrs(cell.bold, cell.italic, cell.underline),
                            };
                        }
                        
                        col += total_cells;
                        continue;
                    }
                }
            }
            
            // Check for emoji multi-cell pattern (colored glyphs followed by empty cells)
            // This is separate from PUA because emoji detection happens via sprite lookup
            if c != ' ' && c != '\0' {
                let mut num_empty = 0;
                const MAX_EXTRA_CELLS: usize = 1; // Emoji are 2 cells wide
                
                while col + num_empty + 1 < row.len() && num_empty < MAX_EXTRA_CELLS {
                    let next_char = row[col + num_empty + 1].character;
                    if next_char == ' ' || next_char == '\u{2002}' || next_char == '\0' {
                        num_empty += 1;
                    } else {
                        break;
                    }
                }
                
                if num_empty > 0 {
                    // Check if we have colored multi-cell sprites for this character
                    let first_key = SpriteKey {
                        text: format!("{}_0", c),
                        style,
                        colored: true,
                    };
                    
                    if let Some(&first_sprite) = sprite_map.get(&first_key) {
                        let total_cells = 1 + num_empty;
                        
                        for cell_idx in 0..total_cells {
                            if col + cell_idx >= cols {
                                break;
                            }
                            
                            let sprite_idx = if cell_idx == 0 {
                                first_sprite
                            } else {
                                let key = SpriteKey {
                                    text: format!("{}_{}", c, cell_idx),
                                    style,
                                    colored: true,
                                };
                                sprite_map.get(&key).copied().unwrap_or(0)
                            };
                            
                            let current_cell = &row[col + cell_idx];
                            gpu_row[col + cell_idx] = GPUCell {
                                fg: Self::pack_color(&cell.fg_color),
                                bg: Self::pack_color(&current_cell.bg_color),
                                decoration_fg: 0,
                                sprite_idx: sprite_idx | COLORED_GLYPH_FLAG,
                                attrs: Self::pack_attrs(cell.bold, cell.italic, cell.underline),
                            };
                        }
                        
                        col += total_cells;
                        continue;
                    }
                }
            }
            
            // Regular character lookup
            let sprite_idx = if c == ' ' || c == '\0' {
                0
            } else {
                // Check cache - first try non-colored, then colored
                let key = SpriteKey {
                    text: c.to_string(),
                    style,
                    colored: false,
                };
                if let Some(&idx) = sprite_map.get(&key) {
                    idx
                } else {
                    let color_key = SpriteKey {
                        text: c.to_string(),
                        style,
                        colored: true,
                    };
                    sprite_map.get(&color_key).copied().unwrap_or(0)
                }
            };
            
            gpu_row[col] = GPUCell {
                fg: Self::pack_color(&cell.fg_color),
                bg: Self::pack_color(&cell.bg_color),
                decoration_fg: 0,
                sprite_idx,
                attrs: Self::pack_attrs(cell.bold, cell.italic, cell.underline),
            };
            col += 1;
        }
        
        // Fill remaining columns with empty cells
        for col_idx in row.len()..cols {
            gpu_row[col_idx] = GPUCell::default();
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // PER-PANE GPU RESOURCE MANAGEMENT (Like Kitty's VAO per window)
    // ═══════════════════════════════════════════════════════════════════════════════

    /// Get or create GPU resources for a pane.
    /// Like Kitty's create_cell_vao(), this allocates per-pane buffers and bind group.
    /// 
    /// Following Kitty's approach: we check if size matches exactly and reallocate if needed.
    /// This is simpler than tracking capacity with headroom.
    fn get_or_create_pane_resources(&mut self, pane_id: u64, required_cells: usize) -> &PaneGpuResources {
        // Check if we need to create or resize (like Kitty's alloc_buffer size check)
        let needs_create = match self.pane_resources.get(&pane_id) {
            None => true,
            Some(res) => res.capacity != required_cells,  // Reallocate if size changed (Kitty's approach)
        };
        
        if needs_create {
            // Create new buffers with exact size needed (like Kitty - no headroom)
            let capacity = required_cells;
            
            let cell_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Pane {} Cell Buffer", pane_id)),
                size: (capacity * std::mem::size_of::<GPUCell>()) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            
            let grid_params_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("Pane {} Grid Params Buffer", pane_id)),
                size: std::mem::size_of::<GridParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            
            // Create bind group referencing this pane's buffers + shared resources
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(&format!("Pane {} Bind Group", pane_id)),
                layout: &self.instanced_bind_group_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.color_table_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: grid_params_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: cell_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: self.sprite_buffer.as_entire_binding(),
                    },
                ],
            });
            
            self.pane_resources.insert(pane_id, PaneGpuResources {
                cell_buffer,
                grid_params_buffer,
                bind_group,
                capacity,
            });
        }
        
        self.pane_resources.get(&pane_id).unwrap()
    }
    
    /// Remove GPU resources for panes that no longer exist.
    /// Like Kitty's remove_vao(), this frees GPU resources when panes are destroyed.
    /// 
    /// Call this after rendering with a set of active pane IDs.
    pub fn cleanup_unused_pane_resources(&mut self, active_pane_ids: &std::collections::HashSet<u64>) {
        self.pane_resources.retain(|id, _| active_pane_ids.contains(id));
    }

    /// Update GPU cell buffer from terminal content.
    /// Like Kitty, this only processes dirty lines to minimize work.
    /// 
    /// Returns true if any cells were updated (buffer needs upload to GPU).
    pub fn update_gpu_cells(&mut self, terminal: &Terminal) -> bool {
        let cols = terminal.cols;
        let rows = terminal.rows;
        let total_cells = cols * rows;
        
        // Check if grid size changed - need full rebuild
        let size_changed = self.last_grid_size != (cols, rows);
        if size_changed {
            self.gpu_cells.resize(total_cells, GPUCell::default());
            self.last_grid_size = (cols, rows);
            self.cells_dirty = true;
        }
        
        // Get visible rows (accounts for scroll offset)
        let visible_rows = terminal.visible_rows();
        
        // First pass: ensure all characters have sprites
        // This needs mutable access to self for sprite creation
        // Like Kitty's render_line(), detect PUA+space patterns for multi-cell rendering
        for row in visible_rows.iter() {
            let mut col = 0;
            while col < row.len() {
                let cell = &row[col];
                
                if cell.character == ' ' || cell.character == '\0' || cell.wide_continuation {
                    col += 1;
                    continue;
                }
                
                let c = cell.character;
                let style = FontStyle::from_flags(cell.bold, cell.italic);
                
                // Check if this is a symbol that might need multi-cell rendering
                // Like Kitty's render_line() at fonts.c:1873-1912
                // This includes PUA characters and dingbats
                if Self::is_multicell_symbol(c) && !Self::is_box_drawing(c) {
                    // Get the glyph's natural width to determine desired cells
                    let glyph_width = self.get_glyph_width(c);
                    let desired_cells = (glyph_width / self.cell_width).ceil() as usize;
                    
                    log::debug!("Symbol check U+{:04X}: glyph_width={:.1}, cell_width={:.1}, desired_cells={}", 
                               c as u32, glyph_width, self.cell_width, desired_cells);
                    
                    if desired_cells > 1 {
                        // Count trailing empty cells (spaces or null characters)
                        // Like Kitty's loop at fonts.c:1888-1903, but also including empty cells
                        let mut num_empty = 0;
                        const MAX_EXTRA_CELLS: usize = 4; // Like Kitty's MAX_NUM_EXTRA_GLYPHS_PUA
                        
                        while col + num_empty + 1 < row.len()
                            && num_empty + 1 < desired_cells
                            && num_empty < MAX_EXTRA_CELLS
                        {
                            let next_char = row[col + num_empty + 1].character;
                            log::debug!("  next char at col {}: U+{:04X} '{}'", 
                                       col + num_empty + 1, next_char as u32, next_char);
                            // Check for space, en-space, or empty/null cell
                            if next_char == ' ' || next_char == '\u{2002}' || next_char == '\0' {
                                num_empty += 1;
                            } else {
                                break;
                            }
                        }
                        
                        log::debug!("  found {} trailing empty cells", num_empty);
                        
                        if num_empty > 0 {
                            // We have symbol + empty cells - render as multi-cell
                            let total_cells = 1 + num_empty;
                            
                            // Check if we already have sprites for this multi-cell group
                            // PUA symbols are not colored
                            let first_key = SpriteKey {
                                text: format!("{}_0_{}", c, total_cells),
                                style,
                                colored: false,
                            };
                            
                            if self.sprite_map.get(&first_key).is_none() {
                                // Need to rasterize
                                let cell_sprites = self.rasterize_pua_multicell(c, total_cells);
                                
                                // Store each cell's sprite with a unique key
                                for (cell_idx, glyph) in cell_sprites.into_iter().enumerate() {
                                    if glyph.size[0] > 0.0 && glyph.size[1] > 0.0 {
                                        let key = SpriteKey {
                                            text: format!("{}_{}", c, cell_idx),
                                            style,
                                            colored: false,
                                        };
                                        
                                        // Create sprite info from glyph info
                                        let sprite = SpriteInfo {
                                            uv: glyph.uv,
                                            _padding: [0.0, 0.0],
                                            size: glyph.size,
                                        };
                                        
                                        // Use next_sprite_idx like get_or_create_sprite does
                                        let sprite_idx = self.next_sprite_idx;
                                        self.next_sprite_idx += 1;
                                        
                                        // Ensure sprite_info array is large enough
                                        while self.sprite_info.len() <= sprite_idx as usize {
                                            self.sprite_info.push(SpriteInfo::default());
                                        }
                                        self.sprite_info[sprite_idx as usize] = sprite;
                                        self.sprite_map.insert(key, sprite_idx);
                                    }
                                }
                            }
                            
                            // Skip the spaces we consumed
                            col += total_cells;
                            continue;
                        }
                    }
                }
                
                // Regular character - create sprite as normal
                // This also handles emoji detection (via emojis::get, but cached per character)
                let (sprite_idx, is_colored) = self.get_or_create_sprite(c, style);
                
                // If this is a colored glyph (emoji) followed by empty cells, create multi-cell sprites
                if is_colored && sprite_idx != 0 {
                    // Count trailing empty cells for potential multi-cell emoji
                    let mut num_empty = 0;
                    const MAX_EXTRA_CELLS: usize = 1; // Emoji are typically 2 cells wide
                    
                    while col + num_empty + 1 < row.len() && num_empty < MAX_EXTRA_CELLS {
                        let next_char = row[col + num_empty + 1].character;
                        if next_char == ' ' || next_char == '\u{2002}' || next_char == '\0' {
                            num_empty += 1;
                        } else {
                            break;
                        }
                    }
                    
                    if num_empty > 0 {
                        let total_cells = 1 + num_empty;
                        
                        // Check if we already have multi-cell sprites for this emoji
                        let first_key = SpriteKey {
                            text: format!("{}_0", c),
                            style,
                            colored: true,
                        };
                        
                        if self.sprite_map.get(&first_key).is_none() {
                            // Need to create multi-cell emoji sprites
                            let cell_sprites = self.rasterize_emoji_multicell(c, total_cells);
                            
                            for (cell_idx, glyph) in cell_sprites.into_iter().enumerate() {
                                if glyph.size[0] > 0.0 && glyph.size[1] > 0.0 {
                                    let key = SpriteKey {
                                        text: format!("{}_{}", c, cell_idx),
                                        style,
                                        colored: true,
                                    };
                                    
                                    let sprite = SpriteInfo {
                                        uv: glyph.uv,
                                        _padding: [0.0, 0.0],
                                        size: glyph.size,
                                    };
                                    
                                    // Use next_sprite_idx like get_or_create_sprite does
                                    let idx = self.next_sprite_idx;
                                    self.next_sprite_idx += 1;
                                    
                                    // Ensure sprite_info array is large enough
                                    while self.sprite_info.len() <= idx as usize {
                                        self.sprite_info.push(SpriteInfo::default());
                                    }
                                    self.sprite_info[idx as usize] = sprite;
                                    self.sprite_map.insert(key, idx);
                                }
                            }
                        }
                        
                        col += total_cells;
                        continue;
                    }
                }
                
                col += 1;
            }
        }
        
        // Re-get visible rows (the reference was invalidated by get_or_create_sprite)
        let visible_rows = terminal.visible_rows();
        
        // Check dirty lines and update only those
        let dirty_bitmap = terminal.get_dirty_lines();
        let mut any_updated = false;
        
        // If we did a full reset or size changed, update all lines
        if self.cells_dirty {
            for (row_idx, row) in visible_rows.iter().enumerate() {
                if row_idx >= rows {
                    break;
                }
                let start = row_idx * cols;
                let end = start + cols;
                Self::cells_to_gpu_row_static(row, &mut self.gpu_cells[start..end], cols, &self.sprite_map);
            }
            self.cells_dirty = false;
            any_updated = true;
        } else {
            // Only update dirty lines
            for row_idx in 0..rows.min(64) {
                let bit = 1u64 << row_idx;
                if (dirty_bitmap & bit) != 0 {
                    if row_idx < visible_rows.len() {
                        let start = row_idx * cols;
                        let end = start + cols;
                        Self::cells_to_gpu_row_static(visible_rows[row_idx], &mut self.gpu_cells[start..end], cols, &self.sprite_map);
                        any_updated = true;
                    }
                }
            }
            
            // For terminals with more than 64 rows, check additional dirty_lines words
            if rows > 64 && dirty_bitmap != 0 {
                for row_idx in 64..rows.min(visible_rows.len()) {
                    let start = row_idx * cols;
                    let end = start + cols;
                    Self::cells_to_gpu_row_static(visible_rows[row_idx], &mut self.gpu_cells[start..end], cols, &self.sprite_map);
                    any_updated = true;
                }
            }
        }
        
        any_updated
    }

    /// Parse ANSI escape sequences from raw statusline content.
    /// Returns a vector of (char, fg_color, bg_color, bold) tuples.
    fn parse_ansi_statusline(content: &str) -> Vec<(char, StatuslineColor, StatuslineColor, bool)> {
        let mut result = Vec::new();
        let chars: Vec<char> = content.chars().collect();
        let mut i = 0;
        
        // Current styling state
        let mut fg = StatuslineColor::Default;
        let mut bg = StatuslineColor::Rgb(0x1a, 0x1a, 0x1a); // Default statusline background
        let mut bold = false;
        
        while i < chars.len() {
            let c = chars[i];
            
            // Check for escape sequence (ESC = 0x1B)
            if c == '\x1b' && i + 1 < chars.len() && chars[i + 1] == '[' {
                // Parse CSI sequence: ESC [ params m
                i += 2; // Skip ESC [
                
                // Collect parameters
                let mut params: Vec<u16> = Vec::new();
                let mut current_param: u16 = 0;
                let mut has_digit = false;
                
                while i < chars.len() {
                    let pc = chars[i];
                    if pc.is_ascii_digit() {
                        current_param = current_param * 10 + (pc as u16 - '0' as u16);
                        has_digit = true;
                        i += 1;
                    } else if pc == ';' || pc == ':' {
                        params.push(if has_digit { current_param } else { 0 });
                        current_param = 0;
                        has_digit = false;
                        i += 1;
                    } else if pc == 'm' {
                        // SGR sequence complete
                        params.push(if has_digit { current_param } else { 0 });
                        i += 1;
                        
                        // Process SGR parameters
                        let mut pi = 0;
                        while pi < params.len() {
                            let code = params[pi];
                            match code {
                                0 => {
                                    fg = StatuslineColor::Default;
                                    bg = StatuslineColor::Rgb(0x1a, 0x1a, 0x1a);
                                    bold = false;
                                }
                                1 => bold = true,
                                22 => bold = false,
                                30..=37 => fg = StatuslineColor::Indexed((code - 30) as u8),
                                38 => {
                                    // Extended foreground color
                                    if pi + 1 < params.len() {
                                        let mode = params[pi + 1];
                                        if mode == 5 && pi + 2 < params.len() {
                                            fg = StatuslineColor::Indexed(params[pi + 2] as u8);
                                            pi += 2;
                                        } else if mode == 2 && pi + 4 < params.len() {
                                            fg = StatuslineColor::Rgb(
                                                params[pi + 2] as u8,
                                                params[pi + 3] as u8,
                                                params[pi + 4] as u8,
                                            );
                                            pi += 4;
                                        }
                                    }
                                }
                                39 => fg = StatuslineColor::Default,
                                40..=47 => bg = StatuslineColor::Indexed((code - 40) as u8),
                                48 => {
                                    // Extended background color
                                    if pi + 1 < params.len() {
                                        let mode = params[pi + 1];
                                        if mode == 5 && pi + 2 < params.len() {
                                            bg = StatuslineColor::Indexed(params[pi + 2] as u8);
                                            pi += 2;
                                        } else if mode == 2 && pi + 4 < params.len() {
                                            bg = StatuslineColor::Rgb(
                                                params[pi + 2] as u8,
                                                params[pi + 3] as u8,
                                                params[pi + 4] as u8,
                                            );
                                            pi += 4;
                                        }
                                    }
                                }
                                49 => bg = StatuslineColor::Rgb(0x1a, 0x1a, 0x1a), // Reset to default statusline bg
                                90..=97 => fg = StatuslineColor::Indexed((code - 90 + 8) as u8),
                                100..=107 => bg = StatuslineColor::Indexed((code - 100 + 8) as u8),
                                _ => {}
                            }
                            pi += 1;
                        }
                        break;
                    } else {
                        // Unknown sequence terminator, skip it
                        i += 1;
                        break;
                    }
                }
            } else if c >= ' ' {
                // Printable character - add to result with current styling
                result.push((c, fg, bg, bold));
                i += 1;
            } else {
                // Skip other control characters
                i += 1;
            }
        }
        
        result
    }

    /// Update statusline GPU cells from StatuslineContent.
    /// This converts the statusline sections/components into GPUCell format for instanced rendering.
    /// 
    /// `target_width` is the desired width in pixels - for Raw content (like neovim statuslines),
    /// this is used to expand the middle gap to fill the full window width.
    /// 
    /// Returns the number of columns used.
    fn update_statusline_cells(&mut self, content: &StatuslineContent, target_width: f32) -> usize {
        self.statusline_gpu_cells.clear();
        
        // Calculate target columns based on window width
        // Use ceil() to ensure we cover the entire window edge-to-edge
        // (the rightmost cell may extend slightly past the window, which is fine)
        let target_cols = if self.cell_width > 0.0 {
            (target_width / self.cell_width).ceil() as usize
        } else {
            self.statusline_max_cols
        };
        
        // Default background color for statusline (dark gray)
        let default_bg = Self::pack_statusline_color(StatuslineColor::Rgb(0x1a, 0x1a, 0x1a));
        let _ = default_bg; // Silence unused warning - used by Sections path
        
        match content {
            StatuslineContent::Raw(ansi_content) => {
                // Parse ANSI escape sequences to extract colors and text
                let parsed = Self::parse_ansi_statusline(ansi_content);
                
                // Find the middle gap (largest consecutive run of spaces)
                // and expand it to fill the target width
                let current_len = parsed.len();
                
                if current_len < target_cols && current_len > 0 {
                    // Find the largest gap of consecutive spaces
                    let mut best_gap_start = 0;
                    let mut best_gap_len = 0;
                    let mut current_gap_start = 0;
                    let mut current_gap_len = 0;
                    let mut in_gap = false;
                    
                    for (i, (c, _, _, _)) in parsed.iter().enumerate() {
                        if *c == ' ' {
                            if !in_gap {
                                current_gap_start = i;
                                current_gap_len = 0;
                                in_gap = true;
                            }
                            current_gap_len += 1;
                        } else {
                            if in_gap && current_gap_len > best_gap_len {
                                // Prefer gaps in the middle (not at start or end)
                                let is_middle = current_gap_start > 0 && (current_gap_start + current_gap_len) < current_len;
                                if is_middle || best_gap_len == 0 {
                                    best_gap_start = current_gap_start;
                                    best_gap_len = current_gap_len;
                                }
                            }
                            in_gap = false;
                        }
                    }
                    // Check final gap
                    if in_gap && current_gap_len > best_gap_len {
                        let is_middle = current_gap_start > 0;
                        if is_middle || best_gap_len == 0 {
                            best_gap_start = current_gap_start;
                            best_gap_len = current_gap_len;
                        }
                    }
                    
                    // Calculate how many extra spaces we need
                    let extra_spaces = target_cols.saturating_sub(current_len);
                    
                    // Get the background color for padding (from the gap area)
                    let gap_bg = if best_gap_len > 0 && best_gap_start < parsed.len() {
                        parsed[best_gap_start].2
                    } else {
                        StatuslineColor::Rgb(0x1a, 0x1a, 0x1a)
                    };
                    
                    // The position right before right-hand content starts (end of gap)
                    let gap_end = best_gap_start + best_gap_len;
                    
                    // Render with expanded gap - insert extra padding at the END of the gap
                    for (i, (c, fg_color, bg_color, bold)) in parsed.iter().enumerate() {
                        // Insert extra padding right before the right-hand content
                        if i == gap_end && extra_spaces > 0 && best_gap_len > 0 {
                            let padding_bg = Self::pack_statusline_color(gap_bg);
                            for _ in 0..extra_spaces {
                                if self.statusline_gpu_cells.len() >= self.statusline_max_cols {
                                    break;
                                }
                                self.statusline_gpu_cells.push(GPUCell {
                                    fg: 0,
                                    bg: padding_bg,
                                    decoration_fg: 0,
                                    sprite_idx: 0,
                                    attrs: 0,
                                });
                            }
                        }
                        
                        if self.statusline_gpu_cells.len() >= self.statusline_max_cols {
                            break;
                        }
                        
                        let fg = Self::pack_statusline_color(*fg_color);
                        let bg = Self::pack_statusline_color(*bg_color);
                        let style = if *bold { FontStyle::Bold } else { FontStyle::Regular };
                        let attrs = Self::pack_attrs(*bold, false, false);
                        
                        let (sprite_idx, is_colored) = if *c == ' ' || *c == '\0' {
                            (0, false)
                        } else {
                            self.get_or_create_sprite_for(*c, style, SpriteTarget::Statusline)
                        };
                        
                        let final_sprite_idx = if is_colored {
                            sprite_idx | COLORED_GLYPH_FLAG
                        } else {
                            sprite_idx
                        };
                        
                        self.statusline_gpu_cells.push(GPUCell {
                            fg,
                            bg,
                            decoration_fg: 0,
                            sprite_idx: final_sprite_idx,
                            attrs,
                        });
                    }
                    
                    // If gap is at the very end (right content is empty), add padding after everything
                    if gap_end == parsed.len() && extra_spaces > 0 && best_gap_len > 0 {
                        let padding_bg = Self::pack_statusline_color(gap_bg);
                        for _ in 0..extra_spaces {
                            if self.statusline_gpu_cells.len() >= self.statusline_max_cols {
                                break;
                            }
                            self.statusline_gpu_cells.push(GPUCell {
                                fg: 0,
                                bg: padding_bg,
                                decoration_fg: 0,
                                sprite_idx: 0,
                                attrs: 0,
                            });
                        }
                    }
                } else {
                    // No expansion needed, render as-is
                    for (c, fg_color, bg_color, bold) in parsed {
                        if self.statusline_gpu_cells.len() >= self.statusline_max_cols {
                            break;
                        }
                        
                        let fg = Self::pack_statusline_color(fg_color);
                        let bg = Self::pack_statusline_color(bg_color);
                        let style = if bold { FontStyle::Bold } else { FontStyle::Regular };
                        let attrs = Self::pack_attrs(bold, false, false);
                        
                        let (sprite_idx, is_colored) = if c == ' ' || c == '\0' {
                            (0, false)
                        } else {
                            self.get_or_create_sprite_for(c, style, SpriteTarget::Statusline)
                        };
                        
                        let final_sprite_idx = if is_colored {
                            sprite_idx | COLORED_GLYPH_FLAG
                        } else {
                            sprite_idx
                        };
                        
                        self.statusline_gpu_cells.push(GPUCell {
                            fg,
                            bg,
                            decoration_fg: 0,
                            sprite_idx: final_sprite_idx,
                            attrs,
                        });
                    }
                }
            }
            StatuslineContent::Sections(sections) => {
                for (section_idx, section) in sections.iter().enumerate() {
                    let section_bg = Self::pack_statusline_color(section.bg);
                    
                    // Get next section's background for powerline arrow transition
                    let next_section_bg = if section_idx + 1 < sections.len() {
                        Self::pack_statusline_color(sections[section_idx + 1].bg)
                    } else {
                        default_bg
                    };
                    
                    for component in section.components.iter() {
                        let component_fg = Self::pack_statusline_color(component.fg);
                        let style = if component.bold { FontStyle::Bold } else { FontStyle::Regular };
                        let attrs = Self::pack_attrs(component.bold, false, false);
                        
                        // Process characters with lookahead for multi-cell symbols
                        let chars: Vec<char> = component.text.chars().collect();
                        let mut char_idx = 0;
                        
                        while char_idx < chars.len() {
                            if self.statusline_gpu_cells.len() >= self.statusline_max_cols {
                                break;
                            }
                            
                            let c = chars[char_idx];
                            
                            // Check for multi-cell symbol pattern
                            let is_powerline_char = ('\u{E0B0}'..='\u{E0BF}').contains(&c);
                            let is_multicell_with_space = !is_powerline_char 
                                && Self::is_multicell_symbol(c) 
                                && !Self::is_box_drawing(c)
                                && char_idx + 1 < chars.len() 
                                && chars[char_idx + 1] == ' ';
                            
                            if is_multicell_with_space {
                                // Render as 2-cell symbol
                                let multi_style = FontStyle::Regular;
                                
                                // Check if we already have multi-cell sprites
                                let first_key = SpriteKey {
                                    text: format!("{}_0", c),
                                    style: multi_style,
                                    colored: false,
                                };
                                
                                if self.statusline_sprite_map.get(&first_key).is_none() {
                                    // Need to rasterize multi-cell sprites
                                    let cell_sprites = self.rasterize_pua_multicell(c, 2);
                                    
                                    for (cell_i, glyph) in cell_sprites.into_iter().enumerate() {
                                        if glyph.size[0] > 0.0 && glyph.size[1] > 0.0 {
                                            let key = SpriteKey {
                                                text: format!("{}_{}", c, cell_i),
                                                style: multi_style,
                                                colored: false,
                                            };
                                            
                                            let sprite = SpriteInfo {
                                                uv: glyph.uv,
                                                _padding: [0.0, 0.0],
                                                size: glyph.size,
                                            };
                                            
                                            // Use statusline sprite tracking
                                            let sprite_idx = self.statusline_next_sprite_idx;
                                            self.statusline_next_sprite_idx += 1;
                                            
                                            // Ensure sprite_info array is large enough
                                            while self.statusline_sprite_info.len() <= sprite_idx as usize {
                                                self.statusline_sprite_info.push(SpriteInfo::default());
                                            }
                                            self.statusline_sprite_info[sprite_idx as usize] = sprite;
                                            
                                            self.statusline_sprite_map.insert(key, sprite_idx);
                                        }
                                    }
                                }
                                
                                // Add GPUCells for both parts
                                for cell_i in 0..2 {
                                    if self.statusline_gpu_cells.len() >= self.statusline_max_cols {
                                        break;
                                    }
                                    
                                    let key = SpriteKey {
                                        text: format!("{}_{}", c, cell_i),
                                        style: multi_style,
                                        colored: false,
                                    };
                                    
                                    let sprite_idx = self.statusline_sprite_map.get(&key).copied().unwrap_or(0);
                                    
                                    self.statusline_gpu_cells.push(GPUCell {
                                        fg: component_fg,
                                        bg: section_bg,
                                        decoration_fg: 0,
                                        sprite_idx,
                                        attrs,
                                    });
                                }
                                
                                // Skip symbol and space
                                char_idx += 2;
                                continue;
                            }
                            
                            // Regular character
                            let (sprite_idx, is_colored) = if c == ' ' || c == '\0' {
                                (0, false)
                            } else {
                                self.get_or_create_sprite_for(c, style, SpriteTarget::Statusline)
                            };
                            
                            log::debug!("        char '{}' (U+{:04X}) -> sprite_idx={}, is_colored={}", 
                                c, c as u32, sprite_idx, is_colored);
                            
                            let final_sprite_idx = if is_colored {
                                sprite_idx | COLORED_GLYPH_FLAG
                            } else {
                                sprite_idx
                            };
                            
                            self.statusline_gpu_cells.push(GPUCell {
                                fg: component_fg,
                                bg: section_bg,
                                decoration_fg: 0,
                                sprite_idx: final_sprite_idx,
                                attrs,
                            });
                            
                            char_idx += 1;
                        }
                    }
                    
                    // Add powerline arrow at end of section if it has a background
                    let has_bg = matches!(section.bg, StatuslineColor::Indexed(_) | StatuslineColor::Rgb(_, _, _));
                    if has_bg && self.statusline_gpu_cells.len() < self.statusline_max_cols {
                        // The powerline arrow character
                        let arrow_char = '\u{E0B0}';
                        let (sprite_idx, _) = self.get_or_create_sprite_for(arrow_char, FontStyle::Regular, SpriteTarget::Statusline);
                        
                        // Arrow foreground is current section's bg, arrow background is next section's bg
                        self.statusline_gpu_cells.push(GPUCell {
                            fg: section_bg,      // Arrow takes section bg color as its foreground
                            bg: next_section_bg, // Background is the next section's background
                            decoration_fg: 0,
                            sprite_idx,
                            attrs: 0,
                        });
                    }
                }
            }
        }
        
        // Fill remaining width with default background cells
        // This ensures the statusline covers the entire window width
        let default_bg_packed = Self::pack_statusline_color(StatuslineColor::Default);
        while self.statusline_gpu_cells.len() < target_cols && self.statusline_gpu_cells.len() < self.statusline_max_cols {
            self.statusline_gpu_cells.push(GPUCell {
                fg: 0,
                bg: default_bg_packed,
                decoration_fg: 0,
                sprite_idx: 0,
                attrs: 0,
            });
        }
        
        self.statusline_gpu_cells.len()
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // BOX DRAWING HELPER FUNCTIONS
    // ═══════════════════════════════════════════════════════════════════════════

    /// Calculate line thickness based on DPI and scale, similar to Kitty's thickness_as_float.
    /// Level 0 = hairline, 1 = light, 2 = medium, 3 = heavy
    fn box_thickness(&self, level: usize) -> f64 {
        // Kitty's box_drawing_scale defaults: [0.001, 1.0, 1.5, 2.0] in points
        const BOX_DRAWING_SCALE: [f64; 4] = [0.001, 1.0, 1.5, 2.0];
        let pts = BOX_DRAWING_SCALE[level.min(3)];
        // thickness = scale * pts * dpi / 72.0
        (pts * self.dpi / 72.0).max(1.0)
    }

    /// Check if a character is a box-drawing character that should be rendered procedurally.
    fn is_box_drawing(c: char) -> bool {
        let cp = c as u32;
        // Box Drawing: U+2500-U+257F
        // Block Elements: U+2580-U+259F
        // Geometric Shapes (subset): U+25A0-U+25FF (circles, arcs, triangles)
        // Braille Patterns: U+2800-U+28FF
        // Powerline Symbols: U+E0B0-U+E0BF
        (0x2500..=0x257F).contains(&cp)
            || (0x2580..=0x259F).contains(&cp)
            || (0x25A0..=0x25FF).contains(&cp)
            || (0x2800..=0x28FF).contains(&cp)
            || (0xE0B0..=0xE0BF).contains(&cp)
    }

    /// Check if a character is in the Unicode Private Use Area (PUA).
    /// Nerd Fonts and other symbol fonts use PUA codepoints.
    /// Returns true for:
    /// - BMP Private Use Area: U+E000-U+F8FF
    /// - Supplementary Private Use Area-A: U+F0000-U+FFFFD
    /// - Supplementary Private Use Area-B: U+100000-U+10FFFD
    fn is_private_use(c: char) -> bool {
        let cp = c as u32;
        (0xE000..=0xF8FF).contains(&cp)
            || (0xF0000..=0xFFFFD).contains(&cp)
            || (0x100000..=0x10FFFD).contains(&cp)
    }

    /// Check if a character is a symbol that may need multi-cell rendering.
    /// This includes PUA characters and dingbats.
    /// Emoji are handled separately via the colored sprite path.
    /// Used to detect symbols that might be wider than a single cell.
    fn is_multicell_symbol(c: char) -> bool {
        let cp = c as u32;
        // Private Use Areas
        if Self::is_private_use(c) {
            return true;
        }
        // Dingbats: U+2700-U+27BF (like Kitty's is_non_emoji_dingbat)
        // This includes arrows like ➜ (U+279C)
        if (0x2700..=0x27BF).contains(&cp) {
            return true;
        }
        // Miscellaneous Symbols: U+2600-U+26FF
        if (0x2600..=0x26FF).contains(&cp) {
            return true;
        }
        false
    }

    /// Get the rendered width of a glyph in pixels.
    /// Used to determine if a PUA glyph needs multiple cells.
    /// Like Kitty's get_glyph_width() in freetype.c, this returns the actual
    /// bitmap/bounding box width, not the advance width.
    fn get_glyph_width(&self, c: char) -> f32 {
        use ab_glyph::Font;
        
        // Try primary font first
        let glyph_id = self.primary_font.glyph_id(c);
        if glyph_id.0 != 0 {
            let scaled = self.primary_font.as_scaled(self.font_size);
            // Create a Glyph from the GlyphId
            let glyph = glyph_id.with_scale(self.font_size);
            // Use pixel bounds width (like Kitty's B.width)
            // This is the actual rendered glyph width, not the advance width
            if let Some(outlined) = scaled.outline_glyph(glyph) {
                let bounds = outlined.px_bounds();
                let width = bounds.max.x - bounds.min.x;
                if width > 0.0 {
                    return width;
                }
            }
            // Fall back to h_advance if no outline
            return scaled.h_advance(glyph_id);
        }
        
        // Try fallback fonts
        for (_, fallback_font) in &self.fallback_fonts {
            let fb_glyph_id = fallback_font.glyph_id(c);
            if fb_glyph_id.0 != 0 {
                let scaled = fallback_font.as_scaled(self.font_size);
                // Create a Glyph from the GlyphId
                let glyph = fb_glyph_id.with_scale(self.font_size);
                // Use pixel bounds width (like Kitty's B.width)
                if let Some(outlined) = scaled.outline_glyph(glyph) {
                    let bounds = outlined.px_bounds();
                    let width = bounds.max.x - bounds.min.x;
                    if width > 0.0 {
                        return width;
                    }
                }
                // Fall back to h_advance if no outline
                return scaled.h_advance(fb_glyph_id);
            }
        }
        
        // Default to one cell width if glyph not found
        self.cell_width
    }

    /// Render a box-drawing character procedurally to a bitmap.
    /// Returns (bitmap, supersampled) where supersampled indicates if anti-aliasing was used.
    fn render_box_char(&self, c: char) -> Option<(Vec<u8>, bool)> {
        let w = self.cell_width.ceil() as usize;
        let h = self.cell_height.ceil() as usize;
        let mut bitmap = vec![0u8; w * h];
        let mut supersampled = false;

        let mid_x = w / 2;
        let mid_y = h / 2;
        let light = 2.max((self.font_size / 8.0).round() as usize);  // 2px minimum, scales with font
        let heavy = light * 2;  // 4px minimum

        // For double lines
        let double_gap = light + 2;
        let double_off = double_gap / 2;

        // Helper: draw horizontal line
        let hline = |buf: &mut [u8], x1: usize, x2: usize, y: usize, t: usize| {
            let y_start = y.saturating_sub(t / 2);
            let y_end = (y_start + t).min(h);
            for py in y_start..y_end {
                for px in x1..x2.min(w) {
                    buf[py * w + px] = 255;
                }
            }
        };

        // Helper: draw vertical line
        let vline = |buf: &mut [u8], y1: usize, y2: usize, x: usize, t: usize| {
            let x_start = x.saturating_sub(t / 2);
            let x_end = (x_start + t).min(w);
            for py in y1..y2.min(h) {
                for px in x_start..x_end {
                    buf[py * w + px] = 255;
                }
            }
        };

        // Helper: fill rectangle
        let fill_rect = |buf: &mut [u8], x1: usize, y1: usize, x2: usize, y2: usize| {
            for py in y1..y2.min(h) {
                for px in x1..x2.min(w) {
                    buf[py * w + px] = 255;
                }
            }
        };

        match c {
            // ═══════════════════════════════════════════════════════════════
            // LIGHT BOX DRAWING (single lines)
            // ═══════════════════════════════════════════════════════════════

            // Horizontal and vertical lines
            '─' => hline(&mut bitmap, 0, w, mid_y, light),
            '│' => vline(&mut bitmap, 0, h, mid_x, light),

            // Light corners
            '┌' => {
                hline(&mut bitmap, mid_x, w, mid_y, light);
                vline(&mut bitmap, mid_y, h, mid_x, light);
            }
            '┐' => {
                hline(&mut bitmap, 0, mid_x + 1, mid_y, light);
                vline(&mut bitmap, mid_y, h, mid_x, light);
            }
            '└' => {
                hline(&mut bitmap, mid_x, w, mid_y, light);
                vline(&mut bitmap, 0, mid_y + 1, mid_x, light);
            }
            '┘' => {
                hline(&mut bitmap, 0, mid_x + 1, mid_y, light);
                vline(&mut bitmap, 0, mid_y + 1, mid_x, light);
            }

            // Light T-junctions
            '├' => {
                vline(&mut bitmap, 0, h, mid_x, light);
                hline(&mut bitmap, mid_x, w, mid_y, light);
            }
            '┤' => {
                vline(&mut bitmap, 0, h, mid_x, light);
                hline(&mut bitmap, 0, mid_x + 1, mid_y, light);
            }
            '┬' => {
                hline(&mut bitmap, 0, w, mid_y, light);
                vline(&mut bitmap, mid_y, h, mid_x, light);
            }
            '┴' => {
                hline(&mut bitmap, 0, w, mid_y, light);
                vline(&mut bitmap, 0, mid_y + 1, mid_x, light);
            }

            // Light cross
            '┼' => {
                hline(&mut bitmap, 0, w, mid_y, light);
                vline(&mut bitmap, 0, h, mid_x, light);
            }

            // ═══════════════════════════════════════════════════════════════
            // HEAVY BOX DRAWING (bold lines)
            // ═══════════════════════════════════════════════════════════════

            '━' => hline(&mut bitmap, 0, w, mid_y, heavy),
            '┃' => vline(&mut bitmap, 0, h, mid_x, heavy),

            // Heavy corners
            '┏' => {
                hline(&mut bitmap, mid_x, w, mid_y, heavy);
                vline(&mut bitmap, mid_y, h, mid_x, heavy);
            }
            '┓' => {
                hline(&mut bitmap, 0, mid_x + 1, mid_y, heavy);
                vline(&mut bitmap, mid_y, h, mid_x, heavy);
            }
            '┗' => {
                hline(&mut bitmap, mid_x, w, mid_y, heavy);
                vline(&mut bitmap, 0, mid_y + 1, mid_x, heavy);
            }
            '┛' => {
                hline(&mut bitmap, 0, mid_x + 1, mid_y, heavy);
                vline(&mut bitmap, 0, mid_y + 1, mid_x, heavy);
            }

            // Heavy T-junctions
            '┣' => {
                vline(&mut bitmap, 0, h, mid_x, heavy);
                hline(&mut bitmap, mid_x, w, mid_y, heavy);
            }
            '┫' => {
                vline(&mut bitmap, 0, h, mid_x, heavy);
                hline(&mut bitmap, 0, mid_x + 1, mid_y, heavy);
            }
            '┳' => {
                hline(&mut bitmap, 0, w, mid_y, heavy);
                vline(&mut bitmap, mid_y, h, mid_x, heavy);
            }
            '┻' => {
                hline(&mut bitmap, 0, w, mid_y, heavy);
                vline(&mut bitmap, 0, mid_y + 1, mid_x, heavy);
            }

            // Heavy cross
            '╋' => {
                hline(&mut bitmap, 0, w, mid_y, heavy);
                vline(&mut bitmap, 0, h, mid_x, heavy);
            }

            // ═══════════════════════════════════════════════════════════════
            // MIXED LIGHT/HEAVY
            // ═══════════════════════════════════════════════════════════════

            // Light horizontal, heavy vertical corners
            '┎' => {
                hline(&mut bitmap, mid_x, w, mid_y, light);
                vline(&mut bitmap, mid_y, h, mid_x, heavy);
            }
            '┒' => {
                hline(&mut bitmap, 0, mid_x + 1, mid_y, light);
                vline(&mut bitmap, mid_y, h, mid_x, heavy);
            }
            '┖' => {
                hline(&mut bitmap, mid_x, w, mid_y, light);
                vline(&mut bitmap, 0, mid_y + 1, mid_x, heavy);
            }
            '┚' => {
                hline(&mut bitmap, 0, mid_x + 1, mid_y, light);
                vline(&mut bitmap, 0, mid_y + 1, mid_x, heavy);
            }

            // Heavy horizontal, light vertical corners
            '┍' => {
                hline(&mut bitmap, mid_x, w, mid_y, heavy);
                vline(&mut bitmap, mid_y, h, mid_x, light);
            }
            '┑' => {
                hline(&mut bitmap, 0, mid_x + 1, mid_y, heavy);
                vline(&mut bitmap, mid_y, h, mid_x, light);
            }
            '┕' => {
                hline(&mut bitmap, mid_x, w, mid_y, heavy);
                vline(&mut bitmap, 0, mid_y + 1, mid_x, light);
            }
            '┙' => {
                hline(&mut bitmap, 0, mid_x + 1, mid_y, heavy);
                vline(&mut bitmap, 0, mid_y + 1, mid_x, light);
            }

            // Mixed T-junctions (vertical heavy, horizontal light)
            '┠' => {
                vline(&mut bitmap, 0, h, mid_x, heavy);
                hline(&mut bitmap, mid_x, w, mid_y, light);
            }
            '┨' => {
                vline(&mut bitmap, 0, h, mid_x, heavy);
                hline(&mut bitmap, 0, mid_x + 1, mid_y, light);
            }
            '┰' => {
                hline(&mut bitmap, 0, w, mid_y, light);
                vline(&mut bitmap, mid_y, h, mid_x, heavy);
            }
            '┸' => {
                hline(&mut bitmap, 0, w, mid_y, light);
                vline(&mut bitmap, 0, mid_y + 1, mid_x, heavy);
            }

            // Mixed T-junctions (vertical light, horizontal heavy)
            '┝' => {
                vline(&mut bitmap, 0, h, mid_x, light);
                hline(&mut bitmap, mid_x, w, mid_y, heavy);
            }
            '┥' => {
                vline(&mut bitmap, 0, h, mid_x, light);
                hline(&mut bitmap, 0, mid_x + 1, mid_y, heavy);
            }
            '┯' => {
                hline(&mut bitmap, 0, w, mid_y, heavy);
                vline(&mut bitmap, mid_y, h, mid_x, light);
            }
            '┷' => {
                hline(&mut bitmap, 0, w, mid_y, heavy);
                vline(&mut bitmap, 0, mid_y + 1, mid_x, light);
            }

            // More mixed T-junctions
            '┞' => {
                vline(&mut bitmap, 0, mid_y + 1, mid_x, light);
                vline(&mut bitmap, mid_y, h, mid_x, heavy);
                hline(&mut bitmap, mid_x, w, mid_y, light);
            }
            '┟' => {
                vline(&mut bitmap, 0, mid_y + 1, mid_x, heavy);
                vline(&mut bitmap, mid_y, h, mid_x, light);
                hline(&mut bitmap, mid_x, w, mid_y, light);
            }
            '┡' => {
                vline(&mut bitmap, 0, mid_y + 1, mid_x, light);
                vline(&mut bitmap, mid_y, h, mid_x, heavy);
                hline(&mut bitmap, mid_x, w, mid_y, heavy);
            }
            '┢' => {
                vline(&mut bitmap, 0, mid_y + 1, mid_x, heavy);
                vline(&mut bitmap, mid_y, h, mid_x, light);
                hline(&mut bitmap, mid_x, w, mid_y, heavy);
            }
            '┦' => {
                vline(&mut bitmap, 0, mid_y + 1, mid_x, light);
                vline(&mut bitmap, mid_y, h, mid_x, heavy);
                hline(&mut bitmap, 0, mid_x + 1, mid_y, light);
            }
            '┧' => {
                vline(&mut bitmap, 0, mid_y + 1, mid_x, heavy);
                vline(&mut bitmap, mid_y, h, mid_x, light);
                hline(&mut bitmap, 0, mid_x + 1, mid_y, light);
            }
            '┩' => {
                vline(&mut bitmap, 0, mid_y + 1, mid_x, light);
                vline(&mut bitmap, mid_y, h, mid_x, heavy);
                hline(&mut bitmap, 0, mid_x + 1, mid_y, heavy);
            }
            '┪' => {
                vline(&mut bitmap, 0, mid_y + 1, mid_x, heavy);
                vline(&mut bitmap, mid_y, h, mid_x, light);
                hline(&mut bitmap, 0, mid_x + 1, mid_y, heavy);
            }
            '┭' => {
                hline(&mut bitmap, 0, mid_x + 1, mid_y, light);
                hline(&mut bitmap, mid_x, w, mid_y, heavy);
                vline(&mut bitmap, mid_y, h, mid_x, light);
            }
            '┮' => {
                hline(&mut bitmap, 0, mid_x + 1, mid_y, heavy);
                hline(&mut bitmap, mid_x, w, mid_y, light);
                vline(&mut bitmap, mid_y, h, mid_x, light);
            }
            '┱' => {
                hline(&mut bitmap, 0, mid_x + 1, mid_y, light);
                hline(&mut bitmap, mid_x, w, mid_y, heavy);
                vline(&mut bitmap, mid_y, h, mid_x, heavy);
            }
            '┲' => {
                hline(&mut bitmap, 0, mid_x + 1, mid_y, heavy);
                hline(&mut bitmap, mid_x, w, mid_y, light);
                vline(&mut bitmap, mid_y, h, mid_x, heavy);
            }
            '┵' => {
                hline(&mut bitmap, 0, mid_x + 1, mid_y, light);
                hline(&mut bitmap, mid_x, w, mid_y, heavy);
                vline(&mut bitmap, 0, mid_y + 1, mid_x, light);
            }
            '┶' => {
                hline(&mut bitmap, 0, mid_x + 1, mid_y, heavy);
                hline(&mut bitmap, mid_x, w, mid_y, light);
                vline(&mut bitmap, 0, mid_y + 1, mid_x, light);
            }
            '┹' => {
                hline(&mut bitmap, 0, mid_x + 1, mid_y, light);
                hline(&mut bitmap, mid_x, w, mid_y, heavy);
                vline(&mut bitmap, 0, mid_y + 1, mid_x, heavy);
            }
            '┺' => {
                hline(&mut bitmap, 0, mid_x + 1, mid_y, heavy);
                hline(&mut bitmap, mid_x, w, mid_y, light);
                vline(&mut bitmap, 0, mid_y + 1, mid_x, heavy);
            }

            // Mixed crosses
            '╀' => {
                hline(&mut bitmap, 0, w, mid_y, light);
                vline(&mut bitmap, 0, mid_y + 1, mid_x, heavy);
                vline(&mut bitmap, mid_y, h, mid_x, light);
            }
            '╁' => {
                hline(&mut bitmap, 0, w, mid_y, light);
                vline(&mut bitmap, 0, mid_y + 1, mid_x, light);
                vline(&mut bitmap, mid_y, h, mid_x, heavy);
            }
            '╂' => {
                hline(&mut bitmap, 0, w, mid_y, light);
                vline(&mut bitmap, 0, h, mid_x, heavy);
            }
            '╃' => {
                hline(&mut bitmap, 0, mid_x + 1, mid_y, heavy);
                hline(&mut bitmap, mid_x, w, mid_y, light);
                vline(&mut bitmap, 0, mid_y + 1, mid_x, heavy);
                vline(&mut bitmap, mid_y, h, mid_x, light);
            }
            '╄' => {
                hline(&mut bitmap, 0, mid_x + 1, mid_y, light);
                hline(&mut bitmap, mid_x, w, mid_y, heavy);
                vline(&mut bitmap, 0, mid_y + 1, mid_x, heavy);
                vline(&mut bitmap, mid_y, h, mid_x, light);
            }
            '╅' => {
                hline(&mut bitmap, 0, mid_x + 1, mid_y, heavy);
                hline(&mut bitmap, mid_x, w, mid_y, light);
                vline(&mut bitmap, 0, mid_y + 1, mid_x, light);
                vline(&mut bitmap, mid_y, h, mid_x, heavy);
            }
            '╆' => {
                hline(&mut bitmap, 0, mid_x + 1, mid_y, light);
                hline(&mut bitmap, mid_x, w, mid_y, heavy);
                vline(&mut bitmap, 0, mid_y + 1, mid_x, light);
                vline(&mut bitmap, mid_y, h, mid_x, heavy);
            }
            '╇' => {
                hline(&mut bitmap, 0, w, mid_y, heavy);
                vline(&mut bitmap, 0, mid_y + 1, mid_x, light);
                vline(&mut bitmap, mid_y, h, mid_x, heavy);
            }
            '╈' => {
                hline(&mut bitmap, 0, w, mid_y, heavy);
                vline(&mut bitmap, 0, mid_y + 1, mid_x, heavy);
                vline(&mut bitmap, mid_y, h, mid_x, light);
            }
            '╉' => {
                hline(&mut bitmap, 0, mid_x + 1, mid_y, light);
                hline(&mut bitmap, mid_x, w, mid_y, heavy);
                vline(&mut bitmap, 0, h, mid_x, heavy);
            }
            '╊' => {
                hline(&mut bitmap, 0, mid_x + 1, mid_y, heavy);
                hline(&mut bitmap, mid_x, w, mid_y, light);
                vline(&mut bitmap, 0, h, mid_x, heavy);
            }

            // ═══════════════════════════════════════════════════════════════
            // DOUBLE LINES
            // ═══════════════════════════════════════════════════════════════

            '═' => {
                hline(&mut bitmap, 0, w, mid_y.saturating_sub(double_off), light);
                hline(&mut bitmap, 0, w, mid_y + double_off, light);
            }
            '║' => {
                vline(&mut bitmap, 0, h, mid_x.saturating_sub(double_off), light);
                vline(&mut bitmap, 0, h, mid_x + double_off, light);
            }

            // Double corners
            '╔' => {
                hline(&mut bitmap, mid_x, w, mid_y.saturating_sub(double_off), light);
                hline(&mut bitmap, mid_x + double_off, w, mid_y + double_off, light);
                vline(&mut bitmap, mid_y, h, mid_x.saturating_sub(double_off), light);
                vline(&mut bitmap, mid_y.saturating_sub(double_off), h, mid_x + double_off, light);
            }
            '╗' => {
                hline(&mut bitmap, 0, mid_x + 1, mid_y.saturating_sub(double_off), light);
                hline(&mut bitmap, 0, mid_x.saturating_sub(double_off) + 1, mid_y + double_off, light);
                vline(&mut bitmap, mid_y, h, mid_x + double_off, light);
                vline(&mut bitmap, mid_y.saturating_sub(double_off), h, mid_x.saturating_sub(double_off), light);
            }
            '╚' => {
                hline(&mut bitmap, mid_x, w, mid_y + double_off, light);
                hline(&mut bitmap, mid_x + double_off, w, mid_y.saturating_sub(double_off), light);
                vline(&mut bitmap, 0, mid_y + 1, mid_x.saturating_sub(double_off), light);
                vline(&mut bitmap, 0, mid_y + double_off + 1, mid_x + double_off, light);
            }
            '╝' => {
                hline(&mut bitmap, 0, mid_x + 1, mid_y + double_off, light);
                hline(&mut bitmap, 0, mid_x.saturating_sub(double_off) + 1, mid_y.saturating_sub(double_off), light);
                vline(&mut bitmap, 0, mid_y + 1, mid_x + double_off, light);
                vline(&mut bitmap, 0, mid_y + double_off + 1, mid_x.saturating_sub(double_off), light);
            }

            // Double T-junctions
            '╠' => {
                vline(&mut bitmap, 0, h, mid_x.saturating_sub(double_off), light);
                vline(&mut bitmap, 0, mid_y.saturating_sub(double_off) + 1, mid_x + double_off, light);
                vline(&mut bitmap, mid_y + double_off, h, mid_x + double_off, light);
                hline(&mut bitmap, mid_x + double_off, w, mid_y.saturating_sub(double_off), light);
                hline(&mut bitmap, mid_x + double_off, w, mid_y + double_off, light);
            }
            '╣' => {
                vline(&mut bitmap, 0, h, mid_x + double_off, light);
                vline(&mut bitmap, 0, mid_y.saturating_sub(double_off) + 1, mid_x.saturating_sub(double_off), light);
                vline(&mut bitmap, mid_y + double_off, h, mid_x.saturating_sub(double_off), light);
                hline(&mut bitmap, 0, mid_x.saturating_sub(double_off) + 1, mid_y.saturating_sub(double_off), light);
                hline(&mut bitmap, 0, mid_x.saturating_sub(double_off) + 1, mid_y + double_off, light);
            }
            '╦' => {
                hline(&mut bitmap, 0, w, mid_y.saturating_sub(double_off), light);
                hline(&mut bitmap, 0, mid_x.saturating_sub(double_off) + 1, mid_y + double_off, light);
                hline(&mut bitmap, mid_x + double_off, w, mid_y + double_off, light);
                vline(&mut bitmap, mid_y + double_off, h, mid_x.saturating_sub(double_off), light);
                vline(&mut bitmap, mid_y + double_off, h, mid_x + double_off, light);
            }
            '╩' => {
                hline(&mut bitmap, 0, w, mid_y + double_off, light);
                hline(&mut bitmap, 0, mid_x.saturating_sub(double_off) + 1, mid_y.saturating_sub(double_off), light);
                hline(&mut bitmap, mid_x + double_off, w, mid_y.saturating_sub(double_off), light);
                vline(&mut bitmap, 0, mid_y.saturating_sub(double_off) + 1, mid_x.saturating_sub(double_off), light);
                vline(&mut bitmap, 0, mid_y.saturating_sub(double_off) + 1, mid_x + double_off, light);
            }

            // Double cross
            '╬' => {
                vline(&mut bitmap, 0, mid_y.saturating_sub(double_off) + 1, mid_x.saturating_sub(double_off), light);
                vline(&mut bitmap, 0, mid_y.saturating_sub(double_off) + 1, mid_x + double_off, light);
                vline(&mut bitmap, mid_y + double_off, h, mid_x.saturating_sub(double_off), light);
                vline(&mut bitmap, mid_y + double_off, h, mid_x + double_off, light);
                hline(&mut bitmap, 0, mid_x.saturating_sub(double_off) + 1, mid_y.saturating_sub(double_off), light);
                hline(&mut bitmap, 0, mid_x.saturating_sub(double_off) + 1, mid_y + double_off, light);
                hline(&mut bitmap, mid_x + double_off, w, mid_y.saturating_sub(double_off), light);
                hline(&mut bitmap, mid_x + double_off, w, mid_y + double_off, light);
            }

            // ═══════════════════════════════════════════════════════════════
            // SINGLE/DOUBLE MIXED
            // ═══════════════════════════════════════════════════════════════

            // Single horizontal, double vertical corners
            '╒' => {
                hline(&mut bitmap, mid_x + double_off, w, mid_y, light);
                vline(&mut bitmap, mid_y, h, mid_x.saturating_sub(double_off), light);
                vline(&mut bitmap, mid_y, h, mid_x + double_off, light);
            }
            '╓' => {
                hline(&mut bitmap, mid_x, w, mid_y, light);
                vline(&mut bitmap, mid_y, h, mid_x, light);
            }
            '╕' => {
                hline(&mut bitmap, 0, mid_x.saturating_sub(double_off) + 1, mid_y, light);
                vline(&mut bitmap, mid_y, h, mid_x.saturating_sub(double_off), light);
                vline(&mut bitmap, mid_y, h, mid_x + double_off, light);
            }
            '╖' => {
                hline(&mut bitmap, 0, mid_x + 1, mid_y, light);
                vline(&mut bitmap, mid_y, h, mid_x, light);
            }
            '╘' => {
                hline(&mut bitmap, mid_x + double_off, w, mid_y, light);
                vline(&mut bitmap, 0, mid_y + 1, mid_x.saturating_sub(double_off), light);
                vline(&mut bitmap, 0, mid_y + 1, mid_x + double_off, light);
            }
            '╙' => {
                hline(&mut bitmap, mid_x, w, mid_y, light);
                vline(&mut bitmap, 0, mid_y + 1, mid_x, light);
            }
            '╛' => {
                hline(&mut bitmap, 0, mid_x.saturating_sub(double_off) + 1, mid_y, light);
                vline(&mut bitmap, 0, mid_y + 1, mid_x.saturating_sub(double_off), light);
                vline(&mut bitmap, 0, mid_y + 1, mid_x + double_off, light);
            }
            '╜' => {
                hline(&mut bitmap, 0, mid_x + 1, mid_y, light);
                vline(&mut bitmap, 0, mid_y + 1, mid_x, light);
            }

            // Mixed T-junctions
            '╞' => {
                vline(&mut bitmap, 0, h, mid_x.saturating_sub(double_off), light);
                vline(&mut bitmap, 0, h, mid_x + double_off, light);
                hline(&mut bitmap, mid_x + double_off, w, mid_y, light);
            }
            '╟' => {
                vline(&mut bitmap, 0, h, mid_x, light);
                hline(&mut bitmap, mid_x, w, mid_y.saturating_sub(double_off), light);
                hline(&mut bitmap, mid_x, w, mid_y + double_off, light);
            }
            '╡' => {
                vline(&mut bitmap, 0, h, mid_x.saturating_sub(double_off), light);
                vline(&mut bitmap, 0, h, mid_x + double_off, light);
                hline(&mut bitmap, 0, mid_x.saturating_sub(double_off) + 1, mid_y, light);
            }
            '╢' => {
                vline(&mut bitmap, 0, h, mid_x, light);
                hline(&mut bitmap, 0, mid_x + 1, mid_y.saturating_sub(double_off), light);
                hline(&mut bitmap, 0, mid_x + 1, mid_y + double_off, light);
            }
            '╤' => {
                hline(&mut bitmap, 0, w, mid_y.saturating_sub(double_off), light);
                hline(&mut bitmap, 0, w, mid_y + double_off, light);
                vline(&mut bitmap, mid_y + double_off, h, mid_x, light);
            }
            '╥' => {
                hline(&mut bitmap, 0, w, mid_y, light);
                vline(&mut bitmap, mid_y, h, mid_x.saturating_sub(double_off), light);
                vline(&mut bitmap, mid_y, h, mid_x + double_off, light);
            }
            '╧' => {
                hline(&mut bitmap, 0, w, mid_y.saturating_sub(double_off), light);
                hline(&mut bitmap, 0, w, mid_y + double_off, light);
                vline(&mut bitmap, 0, mid_y.saturating_sub(double_off) + 1, mid_x, light);
            }
            '╨' => {
                hline(&mut bitmap, 0, w, mid_y, light);
                vline(&mut bitmap, 0, mid_y + 1, mid_x.saturating_sub(double_off), light);
                vline(&mut bitmap, 0, mid_y + 1, mid_x + double_off, light);
            }

            // Mixed crosses
            '╪' => {
                hline(&mut bitmap, 0, w, mid_y.saturating_sub(double_off), light);
                hline(&mut bitmap, 0, w, mid_y + double_off, light);
                vline(&mut bitmap, 0, mid_y.saturating_sub(double_off) + 1, mid_x, light);
                vline(&mut bitmap, mid_y + double_off, h, mid_x, light);
            }
            '╫' => {
                hline(&mut bitmap, 0, mid_x.saturating_sub(double_off) + 1, mid_y, light);
                hline(&mut bitmap, mid_x + double_off, w, mid_y, light);
                vline(&mut bitmap, 0, h, mid_x.saturating_sub(double_off), light);
                vline(&mut bitmap, 0, h, mid_x + double_off, light);
            }

            // ═══════════════════════════════════════════════════════════════
            // ROUNDED CORNERS (using SDF like Kitty, with anti-aliasing)
            // ═══════════════════════════════════════════════════════════════

            '╭' | '╮' | '╯' | '╰' => {
                // Kitty-style rounded corner using signed distance field
                // Translated directly from kitty/decorations.c rounded_corner()

                // hline_limits: for a horizontal line at y with thickness t,
                // returns range [y - t/2, y - t/2 + t]
                let hori_line_start = mid_y.saturating_sub(light / 2);
                let hori_line_end = (hori_line_start + light).min(h);
                let hori_line_height = hori_line_end - hori_line_start;

                // vline_limits: for a vertical line at x with thickness t,
                // returns range [x - t/2, x - t/2 + t]
                let vert_line_start = mid_x.saturating_sub(light / 2);
                let vert_line_end = (vert_line_start + light).min(w);
                let vert_line_width = vert_line_end - vert_line_start;

                // adjusted_Hx/Hy: center of the line in each direction
                let adjusted_hx = vert_line_start as f64 + vert_line_width as f64 / 2.0;
                let adjusted_hy = hori_line_start as f64 + hori_line_height as f64 / 2.0;

                let stroke = (hori_line_height.max(vert_line_width)) as f64;
                let corner_radius = adjusted_hx.min(adjusted_hy);
                let bx = adjusted_hx - corner_radius;
                let by = adjusted_hy - corner_radius;

                let aa_corner = 0.5;  // anti-aliasing amount (kitty uses supersample_factor * 0.5)
                let half_stroke = 0.5 * stroke;

                // Determine shifts based on corner type (matching Kitty's Edge flags)
                // RIGHT_EDGE = 4, TOP_EDGE = 2
                // ╭ = TOP_LEFT (top-left corner, line goes right and down)
                // ╮ = TOP_RIGHT (top-right corner, line goes left and down)
                // ╰ = BOTTOM_LEFT (bottom-left corner, line goes right and up)
                // ╯ = BOTTOM_RIGHT (bottom-right corner, line goes left and up)
                let (is_right, is_top) = match c {
                    '╭' => (false, true),   // TOP_LEFT
                    '╮' => (true, true),    // TOP_RIGHT
                    '╰' => (false, false),  // BOTTOM_LEFT
                    '╯' => (true, false),   // BOTTOM_RIGHT
                    _ => unreachable!(),
                };

                let x_shift = if is_right { adjusted_hx } else { -adjusted_hx };
                let y_shift = if is_top { -adjusted_hy } else { adjusted_hy };

                // Smoothstep for anti-aliasing
                let smoothstep = |edge0: f64, edge1: f64, x: f64| -> f64 {
                    if edge0 == edge1 {
                        return if x < edge0 { 0.0 } else { 1.0 };
                    }
                    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
                    t * t * (3.0 - 2.0 * t)
                };

                for py in 0..h {
                    let sample_y = py as f64 + y_shift + 0.5;
                    let pos_y = sample_y - adjusted_hy;

                    for px in 0..w {
                        let sample_x = px as f64 + x_shift + 0.5;
                        let pos_x = sample_x - adjusted_hx;

                        let qx = pos_x.abs() - bx;
                        let qy = pos_y.abs() - by;
                        let dx = if qx > 0.0 { qx } else { 0.0 };
                        let dy = if qy > 0.0 { qy } else { 0.0 };
                        let dist = (dx * dx + dy * dy).sqrt() + qx.max(qy).min(0.0) - corner_radius;

                        let aa = if qx > 1e-7 && qy > 1e-7 { aa_corner } else { 0.0 };
                        let outer = half_stroke - dist;
                        let inner = -half_stroke - dist;
                        let alpha = smoothstep(-aa, aa, outer) - smoothstep(-aa, aa, inner);

                        if alpha <= 0.0 {
                            continue;
                        }
                        let value = (alpha.clamp(0.0, 1.0) * 255.0).round() as u8;
                        let idx = py * w + px;
                        if value > bitmap[idx] {
                            bitmap[idx] = value;
                        }
                    }
                }
            }

            // ═══════════════════════════════════════════════════════════════
            // DASHED/DOTTED LINES
            // ═══════════════════════════════════════════════════════════════

            '┄' => {
                let seg = w / 8;
                for i in 0..4 {
                    let x1 = i * 2 * seg;
                    let x2 = (x1 + seg).min(w);
                    hline(&mut bitmap, x1, x2, mid_y, light);
                }
            }
            '┅' => {
                let seg = w / 8;
                for i in 0..4 {
                    let x1 = i * 2 * seg;
                    let x2 = (x1 + seg).min(w);
                    hline(&mut bitmap, x1, x2, mid_y, heavy);
                }
            }
            '┆' => {
                let seg = h / 8;
                for i in 0..4 {
                    let y1 = i * 2 * seg;
                    let y2 = (y1 + seg).min(h);
                    vline(&mut bitmap, y1, y2, mid_x, light);
                }
            }
            '┇' => {
                let seg = h / 8;
                for i in 0..4 {
                    let y1 = i * 2 * seg;
                    let y2 = (y1 + seg).min(h);
                    vline(&mut bitmap, y1, y2, mid_x, heavy);
                }
            }
            '┈' => {
                let seg = w / 12;
                for i in 0..6 {
                    let x1 = i * 2 * seg;
                    let x2 = (x1 + seg).min(w);
                    hline(&mut bitmap, x1, x2, mid_y, light);
                }
            }
            '┉' => {
                let seg = w / 12;
                for i in 0..6 {
                    let x1 = i * 2 * seg;
                    let x2 = (x1 + seg).min(w);
                    hline(&mut bitmap, x1, x2, mid_y, heavy);
                }
            }
            '┊' => {
                let seg = h / 12;
                for i in 0..6 {
                    let y1 = i * 2 * seg;
                    let y2 = (y1 + seg).min(h);
                    vline(&mut bitmap, y1, y2, mid_x, light);
                }
            }
            '┋' => {
                let seg = h / 12;
                for i in 0..6 {
                    let y1 = i * 2 * seg;
                    let y2 = (y1 + seg).min(h);
                    vline(&mut bitmap, y1, y2, mid_x, heavy);
                }
            }

            // Double dashed
            '╌' => {
                let seg = w / 4;
                hline(&mut bitmap, 0, seg, mid_y, light);
                hline(&mut bitmap, seg * 2, seg * 3, mid_y, light);
            }
            '╍' => {
                let seg = w / 4;
                hline(&mut bitmap, 0, seg, mid_y, heavy);
                hline(&mut bitmap, seg * 2, seg * 3, mid_y, heavy);
            }
            '╎' => {
                let seg = h / 4;
                vline(&mut bitmap, 0, seg, mid_x, light);
                vline(&mut bitmap, seg * 2, seg * 3, mid_x, light);
            }
            '╏' => {
                let seg = h / 4;
                vline(&mut bitmap, 0, seg, mid_x, heavy);
                vline(&mut bitmap, seg * 2, seg * 3, mid_x, heavy);
            }

            // ═══════════════════════════════════════════════════════════════
            // HALF LINES (line to edge)
            // ═══════════════════════════════════════════════════════════════

            '╴' => hline(&mut bitmap, 0, mid_x + 1, mid_y, light),
            '╵' => vline(&mut bitmap, 0, mid_y + 1, mid_x, light),
            '╶' => hline(&mut bitmap, mid_x, w, mid_y, light),
            '╷' => vline(&mut bitmap, mid_y, h, mid_x, light),
            '╸' => hline(&mut bitmap, 0, mid_x + 1, mid_y, heavy),
            '╹' => vline(&mut bitmap, 0, mid_y + 1, mid_x, heavy),
            '╺' => hline(&mut bitmap, mid_x, w, mid_y, heavy),
            '╻' => vline(&mut bitmap, mid_y, h, mid_x, heavy),

            // Mixed half lines
            '╼' => {
                hline(&mut bitmap, 0, mid_x + 1, mid_y, light);
                hline(&mut bitmap, mid_x, w, mid_y, heavy);
            }
            '╽' => {
                vline(&mut bitmap, 0, mid_y + 1, mid_x, light);
                vline(&mut bitmap, mid_y, h, mid_x, heavy);
            }
            '╾' => {
                hline(&mut bitmap, 0, mid_x + 1, mid_y, heavy);
                hline(&mut bitmap, mid_x, w, mid_y, light);
            }
            '╿' => {
                vline(&mut bitmap, 0, mid_y + 1, mid_x, heavy);
                vline(&mut bitmap, mid_y, h, mid_x, light);
            }

            // ═══════════════════════════════════════════════════════════════
            // DIAGONAL LINES
            // ═══════════════════════════════════════════════════════════════

            '╱' => {
                for i in 0..w.max(h) {
                    let x = w.saturating_sub(1).saturating_sub(i * w / h.max(1));
                    let y = i * h / w.max(1);
                    if x < w && y < h {
                        for t in 0..light {
                            if x + t < w { bitmap[y * w + x + t] = 255; }
                        }
                    }
                }
            }
            '╲' => {
                for i in 0..w.max(h) {
                    let x = i * w / h.max(1);
                    let y = i * h / w.max(1);
                    if x < w && y < h {
                        for t in 0..light {
                            if x + t < w { bitmap[y * w + x + t] = 255; }
                        }
                    }
                }
            }
            '╳' => {
                // Draw both diagonals
                for i in 0..w.max(h) {
                    let x1 = w.saturating_sub(1).saturating_sub(i * w / h.max(1));
                    let x2 = i * w / h.max(1);
                    let y = i * h / w.max(1);
                    if y < h {
                        for t in 0..light {
                            if x1 + t < w { bitmap[y * w + x1 + t] = 255; }
                            if x2 + t < w { bitmap[y * w + x2 + t] = 255; }
                        }
                    }
                }
            }

            // ═══════════════════════════════════════════════════════════════
            // BLOCK ELEMENTS (U+2580-U+259F)
            // ═══════════════════════════════════════════════════════════════

            '▀' => fill_rect(&mut bitmap, 0, 0, w, h / 2),
            '▁' => fill_rect(&mut bitmap, 0, h * 7 / 8, w, h),
            '▂' => fill_rect(&mut bitmap, 0, h * 3 / 4, w, h),
            '▃' => fill_rect(&mut bitmap, 0, h * 5 / 8, w, h),
            '▄' => fill_rect(&mut bitmap, 0, h / 2, w, h),
            '▅' => fill_rect(&mut bitmap, 0, h * 3 / 8, w, h),
            '▆' => fill_rect(&mut bitmap, 0, h / 4, w, h),
            '▇' => fill_rect(&mut bitmap, 0, h / 8, w, h),
            '█' => fill_rect(&mut bitmap, 0, 0, w, h),
            '▉' => fill_rect(&mut bitmap, 0, 0, w * 7 / 8, h),
            '▊' => fill_rect(&mut bitmap, 0, 0, w * 3 / 4, h),
            '▋' => fill_rect(&mut bitmap, 0, 0, w * 5 / 8, h),
            '▌' => fill_rect(&mut bitmap, 0, 0, w / 2, h),
            '▍' => fill_rect(&mut bitmap, 0, 0, w * 3 / 8, h),
            '▎' => fill_rect(&mut bitmap, 0, 0, w / 4, h),
            '▏' => fill_rect(&mut bitmap, 0, 0, w / 8, h),
            '▐' => fill_rect(&mut bitmap, w / 2, 0, w, h),

            // Shades
            '░' => {
                for y in 0..h {
                    for x in 0..w {
                        if (x + y) % 4 == 0 { bitmap[y * w + x] = 255; }
                    }
                }
            }
            '▒' => {
                for y in 0..h {
                    for x in 0..w {
                        if (x + y) % 2 == 0 { bitmap[y * w + x] = 255; }
                    }
                }
            }
            '▓' => {
                for y in 0..h {
                    for x in 0..w {
                        if (x + y) % 4 != 0 { bitmap[y * w + x] = 255; }
                    }
                }
            }

            // Right half blocks and upper eighth
            '▕' => fill_rect(&mut bitmap, w * 7 / 8, 0, w, h),
            '▔' => fill_rect(&mut bitmap, 0, 0, w, h / 8), // Upper one eighth block

            // Quadrants
            '▖' => fill_rect(&mut bitmap, 0, h / 2, w / 2, h),
            '▗' => fill_rect(&mut bitmap, w / 2, h / 2, w, h),
            '▘' => fill_rect(&mut bitmap, 0, 0, w / 2, h / 2),
            '▙' => {
                fill_rect(&mut bitmap, 0, 0, w / 2, h);
                fill_rect(&mut bitmap, w / 2, h / 2, w, h);
            }
            '▚' => {
                fill_rect(&mut bitmap, 0, 0, w / 2, h / 2);
                fill_rect(&mut bitmap, w / 2, h / 2, w, h);
            }
            '▛' => {
                fill_rect(&mut bitmap, 0, 0, w, h / 2);
                fill_rect(&mut bitmap, 0, h / 2, w / 2, h);
            }
            '▜' => {
                fill_rect(&mut bitmap, 0, 0, w, h / 2);
                fill_rect(&mut bitmap, w / 2, h / 2, w, h);
            }
            '▝' => fill_rect(&mut bitmap, w / 2, 0, w, h / 2),
            '▞' => {
                fill_rect(&mut bitmap, w / 2, 0, w, h / 2);
                fill_rect(&mut bitmap, 0, h / 2, w / 2, h);
            }
            '▟' => {
                fill_rect(&mut bitmap, w / 2, 0, w, h);
                fill_rect(&mut bitmap, 0, h / 2, w / 2, h);
            }

            // ═══════════════════════════════════════════════════════════════
            // BRAILLE PATTERNS (U+2800-U+28FF)
            // Uses Kitty's distribute_dots algorithm for proper spacing
            // ═══════════════════════════════════════════════════════════════

            c if (0x2800..=0x28FF).contains(&(c as u32)) => {
                let which = (c as u32 - 0x2800) as u8;
                if which != 0 {
                    // Kitty's distribute_dots algorithm
                    // For horizontal: 2 dots across width
                    // For vertical: 4 dots down height
                    let num_x_dots = 2usize;
                    let num_y_dots = 4usize;

                    // distribute_dots for x (2 dots)
                    let dot_width = 1.max(w / (2 * num_x_dots));
                    let mut x_gaps = [dot_width; 2];
                    let mut extra = w.saturating_sub(2 * num_x_dots * dot_width);
                    let mut idx = 0;
                    while extra > 0 {
                        x_gaps[idx] += 1;
                        idx = (idx + 1) % num_x_dots;
                        extra -= 1;
                    }
                    x_gaps[0] /= 2;
                    let x_summed: [usize; 2] = [x_gaps[0], x_gaps[0] + x_gaps[1]];

                    // distribute_dots for y (4 dots)
                    let dot_height = 1.max(h / (2 * num_y_dots));
                    let mut y_gaps = [dot_height; 4];
                    let mut extra = h.saturating_sub(2 * num_y_dots * dot_height);
                    let mut idx = 0;
                    while extra > 0 {
                        y_gaps[idx] += 1;
                        idx = (idx + 1) % num_y_dots;
                        extra -= 1;
                    }
                    y_gaps[0] /= 2;
                    let y_summed: [usize; 4] = [
                        y_gaps[0],
                        y_gaps[0] + y_gaps[1],
                        y_gaps[0] + y_gaps[1] + y_gaps[2],
                        y_gaps[0] + y_gaps[1] + y_gaps[2] + y_gaps[3],
                    ];

                    // Draw braille dots as rectangles (matching Kitty)
                    // Bit mapping: 0=dot1, 1=dot2, 2=dot3, 3=dot4, 4=dot5, 5=dot6, 6=dot7, 7=dot8
                    // Layout:  col 0  col 1
                    // row 0:   dot1   dot4
                    // row 1:   dot2   dot5
                    // row 2:   dot3   dot6
                    // row 3:   dot7   dot8
                    for bit in 0u8..8 {
                        if which & (1 << bit) != 0 {
                            let q = bit + 1;
                            let col = match q {
                                1 | 2 | 3 | 7 => 0,
                                _ => 1,
                            };
                            let row = match q {
                                1 | 4 => 0,
                                2 | 5 => 1,
                                3 | 6 => 2,
                                _ => 3,
                            };

                            let x_start = x_summed[col] + col * dot_width;
                            let y_start = y_summed[row] + row * dot_height;

                            if y_start < h && x_start < w {
                                let x_end = (x_start + dot_width).min(w);
                                let y_end = (y_start + dot_height).min(h);
                                for py in y_start..y_end {
                                    for px in x_start..x_end {
                                        bitmap[py * w + px] = 255;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            // ═══════════════════════════════════════════════════════════════
            // POWERLINE SYMBOLS (U+E0B0-U+E0BF)
            // Ported from Kitty's decorations.c with proper DPI scaling
            // ═══════════════════════════════════════════════════════════════

            // E0B0: Right-pointing solid triangle
            '\u{E0B0}' => {
                let mut canvas = SupersampledCanvas::new(w, h);
                canvas.fill_powerline_arrow(false, false);
                canvas.downsample(&mut bitmap); supersampled = true;
            }

            // E0B1: Right-pointing chevron (outline)
            '\u{E0B1}' => {
                let mut canvas = SupersampledCanvas::new(w, h);
                let thickness = (self.box_thickness(1) * SupersampledCanvas::FACTOR as f64).round() as usize;
                canvas.stroke_powerline_arrow(false, thickness);
                canvas.downsample(&mut bitmap); supersampled = true;
            }

            // E0B2: Left-pointing solid triangle
            '\u{E0B2}' => {
                let mut canvas = SupersampledCanvas::new(w, h);
                canvas.fill_powerline_arrow(true, false);
                canvas.downsample(&mut bitmap); supersampled = true;
            }

            // E0B3: Left-pointing chevron (outline)
            '\u{E0B3}' => {
                let mut canvas = SupersampledCanvas::new(w, h);
                let thickness = (self.box_thickness(1) * SupersampledCanvas::FACTOR as f64).round() as usize;
                canvas.stroke_powerline_arrow(true, thickness);
                canvas.downsample(&mut bitmap); supersampled = true;
            }

            // E0B4: Right semicircle (filled)
            '\u{E0B4}' => {
                let mut canvas = SupersampledCanvas::new(w, h);
                canvas.fill_bezier_d(false);
                canvas.downsample(&mut bitmap); supersampled = true;
            }

            // E0B5: Right semicircle (outline)
            '\u{E0B5}' => {
                let mut canvas = SupersampledCanvas::new(w, h);
                let thickness = self.box_thickness(1) * SupersampledCanvas::FACTOR as f64;
                canvas.stroke_bezier_d(false, thickness);
                canvas.downsample(&mut bitmap); supersampled = true;
            }

            // E0B6: Left semicircle (filled)
            '\u{E0B6}' => {
                let mut canvas = SupersampledCanvas::new(w, h);
                canvas.fill_bezier_d(true);
                canvas.downsample(&mut bitmap); supersampled = true;
            }

            // E0B7: Left semicircle (outline)
            '\u{E0B7}' => {
                let mut canvas = SupersampledCanvas::new(w, h);
                let thickness = self.box_thickness(1) * SupersampledCanvas::FACTOR as f64;
                canvas.stroke_bezier_d(true, thickness);
                canvas.downsample(&mut bitmap); supersampled = true;
            }

            // E0B8-E0BF: Corner triangles
            '\u{E0B8}' => {
                let mut canvas = SupersampledCanvas::new(w, h);
                canvas.fill_corner_triangle(Corner::BottomLeft, false);
                canvas.downsample(&mut bitmap); supersampled = true;
            }
            '\u{E0B9}' => {
                let mut canvas = SupersampledCanvas::new(w, h);
                canvas.fill_corner_triangle(Corner::BottomLeft, true);
                canvas.downsample(&mut bitmap); supersampled = true;
            }
            '\u{E0BA}' => {
                let mut canvas = SupersampledCanvas::new(w, h);
                canvas.fill_corner_triangle(Corner::TopLeft, false);
                canvas.downsample(&mut bitmap); supersampled = true;
            }
            '\u{E0BB}' => {
                let mut canvas = SupersampledCanvas::new(w, h);
                canvas.fill_corner_triangle(Corner::TopLeft, true);
                canvas.downsample(&mut bitmap); supersampled = true;
            }
            '\u{E0BC}' => {
                let mut canvas = SupersampledCanvas::new(w, h);
                canvas.fill_corner_triangle(Corner::BottomRight, false);
                canvas.downsample(&mut bitmap); supersampled = true;
            }
            '\u{E0BD}' => {
                let mut canvas = SupersampledCanvas::new(w, h);
                canvas.fill_corner_triangle(Corner::BottomRight, true);
                canvas.downsample(&mut bitmap); supersampled = true;
            }
            '\u{E0BE}' => {
                let mut canvas = SupersampledCanvas::new(w, h);
                canvas.fill_corner_triangle(Corner::TopRight, false);
                canvas.downsample(&mut bitmap); supersampled = true;
            }
            '\u{E0BF}' => {
                let mut canvas = SupersampledCanvas::new(w, h);
                canvas.fill_corner_triangle(Corner::TopRight, true);
                canvas.downsample(&mut bitmap); supersampled = true;
            }

            // ═══════════════════════════════════════════════════════════════
            // GEOMETRIC SHAPES - Circles, Arcs, and Triangles (U+25A0-U+25FF)
            // ═══════════════════════════════════════════════════════════════

            // ● U+25CF: Black circle (filled)
            '●' => {
                let mut canvas = SupersampledCanvas::new(w, h);
                canvas.fill_circle(1.0);
                canvas.downsample(&mut bitmap); supersampled = true;
            }

            // ○ U+25CB: White circle (outline)
            '○' => {
                let mut canvas = SupersampledCanvas::new(w, h);
                let line_width = self.box_thickness(1) * SupersampledCanvas::FACTOR as f64;
                let half_line = line_width / 2.0;
                let cx = canvas.ss_width as f64 / 2.0;
                let cy = canvas.ss_height as f64 / 2.0;
                let radius = 0.0_f64.max(cx.min(cy) - half_line);
                canvas.stroke_circle(radius, line_width);
                canvas.downsample(&mut bitmap); supersampled = true;
            }

            // ◉ U+25C9: Fisheye (filled center + circle outline)
            '◉' => {
                let mut canvas = SupersampledCanvas::new(w, h);
                let cx = canvas.ss_width as f64 / 2.0;
                let cy = canvas.ss_height as f64 / 2.0;
                let radius = cx.min(cy);
                let central_radius = (2.0 / 3.0) * radius;

                // Fill central circle
                canvas.fill_circle_radius(central_radius);

                // Draw outer ring
                let line_width = (SupersampledCanvas::FACTOR as f64).max((radius - central_radius) / 2.5);
                let outer_radius = 0.0_f64.max(cx.min(cy) - line_width / 2.0);
                canvas.stroke_circle(outer_radius, line_width);

                canvas.downsample(&mut bitmap); supersampled = true;
            }

            // ◜ U+25DC: Upper left quadrant circular arc (180° to 270°)
            '◜' => {
                let mut canvas = SupersampledCanvas::new(w, h);
                let line_width = self.box_thickness(1) * SupersampledCanvas::FACTOR as f64;
                let half_line = 0.5_f64.max(line_width / 2.0);
                let cx = canvas.ss_width as f64 / 2.0;
                let cy = canvas.ss_height as f64 / 2.0;
                let radius = 0.0_f64.max(cx.min(cy) - half_line);
                canvas.stroke_arc(radius, line_width, std::f64::consts::PI, 3.0 * std::f64::consts::PI / 2.0);
                canvas.downsample(&mut bitmap); supersampled = true;
            }

            // ◝ U+25DD: Upper right quadrant circular arc (270° to 360°)
            '◝' => {
                let mut canvas = SupersampledCanvas::new(w, h);
                let line_width = self.box_thickness(1) * SupersampledCanvas::FACTOR as f64;
                let half_line = 0.5_f64.max(line_width / 2.0);
                let cx = canvas.ss_width as f64 / 2.0;
                let cy = canvas.ss_height as f64 / 2.0;
                let radius = 0.0_f64.max(cx.min(cy) - half_line);
                canvas.stroke_arc(radius, line_width, 3.0 * std::f64::consts::PI / 2.0, 2.0 * std::f64::consts::PI);
                canvas.downsample(&mut bitmap); supersampled = true;
            }

            // ◞ U+25DE: Lower right quadrant circular arc (0° to 90°)
            '◞' => {
                let mut canvas = SupersampledCanvas::new(w, h);
                let line_width = self.box_thickness(1) * SupersampledCanvas::FACTOR as f64;
                let half_line = 0.5_f64.max(line_width / 2.0);
                let cx = canvas.ss_width as f64 / 2.0;
                let cy = canvas.ss_height as f64 / 2.0;
                let radius = 0.0_f64.max(cx.min(cy) - half_line);
                canvas.stroke_arc(radius, line_width, 0.0, std::f64::consts::PI / 2.0);
                canvas.downsample(&mut bitmap); supersampled = true;
            }

            // ◟ U+25DF: Lower left quadrant circular arc (90° to 180°)
            '◟' => {
                let mut canvas = SupersampledCanvas::new(w, h);
                let line_width = self.box_thickness(1) * SupersampledCanvas::FACTOR as f64;
                let half_line = 0.5_f64.max(line_width / 2.0);
                let cx = canvas.ss_width as f64 / 2.0;
                let cy = canvas.ss_height as f64 / 2.0;
                let radius = 0.0_f64.max(cx.min(cy) - half_line);
                canvas.stroke_arc(radius, line_width, std::f64::consts::PI / 2.0, std::f64::consts::PI);
                canvas.downsample(&mut bitmap); supersampled = true;
            }

            // ◠ U+25E0: Upper half arc (180° to 360°)
            '◠' => {
                let mut canvas = SupersampledCanvas::new(w, h);
                let line_width = self.box_thickness(1) * SupersampledCanvas::FACTOR as f64;
                let half_line = 0.5_f64.max(line_width / 2.0);
                let cx = canvas.ss_width as f64 / 2.0;
                let cy = canvas.ss_height as f64 / 2.0;
                let radius = 0.0_f64.max(cx.min(cy) - half_line);
                canvas.stroke_arc(radius, line_width, std::f64::consts::PI, 2.0 * std::f64::consts::PI);
                canvas.downsample(&mut bitmap); supersampled = true;
            }

            // ◡ U+25E1: Lower half arc (0° to 180°)
            '◡' => {
                let mut canvas = SupersampledCanvas::new(w, h);
                let line_width = self.box_thickness(1) * SupersampledCanvas::FACTOR as f64;
                let half_line = 0.5_f64.max(line_width / 2.0);
                let cx = canvas.ss_width as f64 / 2.0;
                let cy = canvas.ss_height as f64 / 2.0;
                let radius = 0.0_f64.max(cx.min(cy) - half_line);
                canvas.stroke_arc(radius, line_width, 0.0, std::f64::consts::PI);
                canvas.downsample(&mut bitmap); supersampled = true;
            }

            // Fall through for unimplemented characters
            _ => return None,
        }

        Some((bitmap, supersampled))
    }

    /// Get or rasterize a glyph by character, with font fallback.
    /// Returns the GlyphInfo for the character.
    fn rasterize_char(&mut self, c: char) -> GlyphInfo {
        // Check cache first
        if let Some(info) = self.char_cache.get(&c) {
            // Log cache hits for emoji to debug first-emoji issue
            if info.is_colored {
                log::debug!("CACHE HIT for color glyph U+{:04X} '{}'", c as u32, c);
            }
            return *info;
        }
        
        log::debug!("CACHE MISS for U+{:04X} '{}' - will rasterize", c as u32, c);

        // Check if this is a box-drawing character - render procedurally
        // Box-drawing characters are already cell-sized, positioned at (0,0)
        if Self::is_box_drawing(c) {
            if let Some((bitmap, _supersampled)) = self.render_box_char(c) {
                // Box-drawing bitmaps are already cell-sized and fill from top-left.
                // Use upload_cell_canvas_to_atlas directly since no repositioning needed.
                let info = self.upload_cell_canvas_to_atlas(&bitmap, false);
                self.char_cache.insert(c, info);
                return info;
            }
        }

        // Check if this is an emoji BEFORE checking primary font.
        // Like Kitty, we skip the primary font for emoji since it may report a glyph
        // (tofu/fallback) that isn't a proper color emoji. Go straight to fontconfig.
        let char_str = c.to_string();
        let is_emoji = emojis::get(&char_str).is_some();
        
        // Track whether we found the glyph in a regular font
        let mut found_in_regular_font = false;
        
        // Rasterize glyph data: (width, height, bitmap, offset_x, offset_y)
        let raster_result: Option<(u32, u32, Vec<u8>, f32, f32)> = if is_emoji {
            // Emoji: skip primary font, will be handled by fontconfig color font path below
            log::debug!("Character U+{:04X} is emoji, skipping primary font check", c as u32);
            None
        } else if { let glyph_id = self.primary_font.glyph_id(c); glyph_id.0 != 0 } {
            // Primary font has this glyph (non-emoji)
            let glyph_id = self.primary_font.glyph_id(c);
            found_in_regular_font = true;
            self.rasterize_glyph_ab(&self.primary_font.clone(), glyph_id)
        } else {
            // Try already-loaded fallback fonts first (but NOT for emoji)
            let mut result = None;
            if !is_emoji {
                for (_, fallback_font) in &self.fallback_fonts {
                    let fb_glyph_id = fallback_font.glyph_id(c);
                    if fb_glyph_id.0 != 0 {
                        result = self.rasterize_glyph_ab(&fallback_font.clone(), fb_glyph_id);
                        found_in_regular_font = true;
                        break;
                    }
                }
            }

            // If no cached fallback has the glyph (or it's emoji), use fontconfig to find one
            if result.is_none() {
                // Lazy-initialize fontconfig on first use
                let fc = self.fontconfig.get_or_init(|| {
                    log::debug!("Initializing fontconfig for fallback font lookup");
                    Fontconfig::new()
                });
                if let Some(fc) = fc {
                    // Query fontconfig for a font that has this character
                    // This now also tells us if the font is a color font
                    if let Some((path, is_color_font)) = find_font_for_char(fc, c) {
                        // If fontconfig returns a COLOR font, use Cairo to render it
                        // (ab_glyph can't render color glyphs from COLR/CBDT/sbix fonts)
                        if is_color_font {
                            log::debug!("Fontconfig found color font for U+{:04X}, using Cairo renderer", c as u32);
                            
                            // Render color glyph in a separate scope to release borrow before atlas ops
                            let color_glyph_data: Option<(u32, u32, Vec<u8>, f32, f32)> = {
                                let mut renderer_cell = self.color_font_renderer.borrow_mut();
                                if renderer_cell.is_none() {
                                    *renderer_cell = ColorFontRenderer::new().ok();
                                    if renderer_cell.is_some() {
                                        log::debug!("Initialized color font renderer for emoji support");
                                    } else {
                                        log::warn!("Failed to initialize color font renderer");
                                    }
                                }
                                
                                if let Some(ref mut renderer) = *renderer_cell {
                                    log::debug!("Attempting to render color glyph for U+{:04X} with font_size={}, cell={}x{}", 
                                               c as u32, self.font_size, self.cell_width as u32, self.cell_height as u32);
                                    
                                    renderer.render_color_glyph(
                                        &path, c, self.font_size, self.cell_width as u32, self.cell_height as u32
                                    )
                                } else {
                                    None
                                }
                            }; // renderer_cell borrow ends here
                            
                            if let Some((w, h, rgba, ox, oy)) = color_glyph_data {
                                log::debug!("Successfully rendered color glyph U+{:04X}: {}x{} pixels, offset=({}, {})", 
                                           c as u32, w, h, ox, oy);
                                
                                // Place the color glyph in a cell-sized canvas at baseline
                                let canvas = self.place_color_glyph_in_cell_canvas(
                                    &rgba, w, h, ox, oy
                                );
                                let info = self.upload_cell_canvas_to_atlas(&canvas, true);
                                
                                self.char_cache.insert(c, info);
                                return info;
                            }
                            // If color rendering failed, fall through to try ab_glyph
                            log::debug!("Color rendering failed for U+{:04X}, trying ab_glyph fallback", c as u32);
                        }
                        
                        // Non-color font or color rendering failed: use ab_glyph
                        // Only load if we haven't tried this path before
                        if !self.tried_font_paths.contains(&path) {
                            self.tried_font_paths.insert(path.clone());

                            if let Ok(data) = std::fs::read(&path) {
                                let data: Box<[u8]> = data.into_boxed_slice();
                                if let Ok(font) = FontRef::try_from_slice(&data) {
                                    log::debug!("Loaded fallback font via fontconfig: {}", path.display());

                                    // Check if this font actually has the glyph
                                    let fb_glyph_id = font.glyph_id(c);
                                    if fb_glyph_id.0 != 0 {
                                        result = self.rasterize_glyph_ab(&font, fb_glyph_id);
                                        found_in_regular_font = true;
                                    }

                                    // Cache the font for future use
                                    // SAFETY: We're storing data alongside the FontRef that borrows it
                                    let font_static: FontRef<'static> = unsafe { std::mem::transmute(font) };
                                    self.fallback_fonts.push((data, font_static));
                                }
                            }
                        }
                    }
                }
            }

            // Don't fall back to .notdef yet - we may still try color fonts below
            result
        };
        
        // If no regular font has this glyph, try color fonts (emoji) as last resort
        // This handles cases where no font at all was found via normal fontconfig
        if !found_in_regular_font {
            log::debug!("Character U+{:04X} '{}' not found in regular fonts, trying dedicated color font query", c as u32, c);
            
            // Check color font cache or query fontconfig for color font explicitly
            let color_path = self.color_font_cache.entry(c).or_insert_with(|| {
                let path = find_color_font_for_char(c);
                log::debug!("Fontconfig color font query for U+{:04X}: {:?}", c as u32, path);
                path
            }).clone();
            
            if let Some(ref path) = color_path {
                log::debug!("Found color font for U+{:04X}: {:?}", c as u32, path);
                
                // Render color glyph in a separate scope to release borrow before atlas ops
                let color_glyph_data: Option<(u32, u32, Vec<u8>, f32, f32)> = {
                    let mut renderer_cell = self.color_font_renderer.borrow_mut();
                    if renderer_cell.is_none() {
                        *renderer_cell = ColorFontRenderer::new().ok();
                        if renderer_cell.is_some() {
                            log::debug!("Initialized color font renderer for emoji support");
                        } else {
                            log::warn!("Failed to initialize color font renderer");
                        }
                    }
                    
                    if let Some(ref mut renderer) = *renderer_cell {
                        log::debug!("Attempting to render color glyph for U+{:04X} with font_size={}, cell={}x{}", 
                                   c as u32, self.font_size, self.cell_width as u32, self.cell_height as u32);
                        
                        renderer.render_color_glyph(
                            path, c, self.font_size, self.cell_width as u32, self.cell_height as u32
                        )
                    } else {
                        None
                    }
                }; // renderer_cell borrow ends here
                
                if let Some((w, h, rgba, ox, oy)) = color_glyph_data {
                    log::debug!("Successfully rendered color glyph U+{:04X}: {}x{} pixels, offset=({}, {})", 
                               c as u32, w, h, ox, oy);
                    
                    // Place the color glyph in a cell-sized canvas at baseline
                    let canvas = self.place_color_glyph_in_cell_canvas(
                        &rgba, w, h, ox, oy
                    );
                    let info = self.upload_cell_canvas_to_atlas(&canvas, true);
                    
                    self.char_cache.insert(c, info);
                    return info;
                }
            }
        }
        
        // Fall back to .notdef from primary font if we still have no glyph
        let raster_result = raster_result.or_else(|| {
            let notdef_glyph_id = self.primary_font.glyph_id(c);
            self.rasterize_glyph_ab(&self.primary_font.clone(), notdef_glyph_id)
        });

        // Handle rasterization result
        let Some((glyph_width, glyph_height, bitmap, offset_x, offset_y)) = raster_result else {
            // Empty glyph (e.g., space)
            let info = GlyphInfo {
                uv: [0.0, 0.0, 0.0, 0.0],
                size: [0.0, 0.0],
                is_colored: false,
            };
            self.char_cache.insert(c, info);
            return info;
        };

        if bitmap.is_empty() || glyph_width == 0 || glyph_height == 0 {
            // Empty glyph (e.g., space)
            let info = GlyphInfo {
                uv: [0.0, 0.0, 0.0, 0.0],
                size: [0.0, 0.0],
                is_colored: false,
            };
            self.char_cache.insert(c, info);
            return info;
        }

        // Check if this is an oversized symbol glyph that needs rescaling.
        // PUA glyphs (Nerd Fonts), dingbats, and other symbols that are wider than
        // one cell should be rescaled to fit when rendered standalone (not part of
        // a multi-cell group).
        let (final_bitmap, final_width, final_height, final_offset_x, final_offset_y) = 
            if Self::is_multicell_symbol(c) {
                let cell_w = self.cell_width;
                // Use just the glyph bitmap width for comparison, not offset_x + width
                // offset_x is the left bearing which can be negative
                let glyph_w = glyph_width as f32;
                
                log::debug!("Scaling check for U+{:04X}: glyph_width={}, cell_width={:.1}, offset_x={:.1}", 
                           c as u32, glyph_width, cell_w, offset_x);
                
                if glyph_w > cell_w {
                    // Glyph is wider than cell - rescale to fit
                    // Calculate scale factor to fit within cell width with small margin
                    let target_width = cell_w * 0.95; // Leave 5% margin
                    let scale_factor = target_width / glyph_w;
                    
                    log::debug!("Scaling U+{:04X} by factor {:.2} (glyph_w={:.1} > cell_w={:.1})",
                               c as u32, scale_factor, glyph_w, cell_w);
                    
                    // Rescale bitmap using simple nearest-neighbor (good enough for icons)
                    let new_width = (glyph_width as f32 * scale_factor).ceil() as u32;
                    let new_height = (glyph_height as f32 * scale_factor).ceil() as u32;
                    
                    if new_width > 0 && new_height > 0 {
                        let mut scaled_bitmap = vec![0u8; (new_width * new_height) as usize];
                        
                        for y in 0..new_height {
                            for x in 0..new_width {
                                // Map to source coordinates
                                let src_x = ((x as f32 / scale_factor) as u32).min(glyph_width - 1);
                                let src_y = ((y as f32 / scale_factor) as u32).min(glyph_height - 1);
                                let src_idx = (src_y * glyph_width + src_x) as usize;
                                let dst_idx = (y * new_width + x) as usize;
                                scaled_bitmap[dst_idx] = bitmap[src_idx];
                            }
                        }
                        
                        // Adjust offset to center the scaled glyph
                        let new_offset_x = (cell_w - new_width as f32) / 2.0;
                        let new_offset_y = offset_y * scale_factor;
                        
                        (scaled_bitmap, new_width, new_height, new_offset_x, new_offset_y)
                    } else {
                        (bitmap, glyph_width, glyph_height, offset_x, offset_y)
                    }
                } else {
                    (bitmap, glyph_width, glyph_height, offset_x, offset_y)
                }
            } else {
                (bitmap, glyph_width, glyph_height, offset_x, offset_y)
            };

        // Place the glyph in a cell-sized canvas at the correct baseline position
        let canvas = self.place_glyph_in_cell_canvas(
            &final_bitmap, final_width, final_height, final_offset_x, final_offset_y
        );
        let info = self.upload_cell_canvas_to_atlas(&canvas, false);

        self.char_cache.insert(c, info);
        info
    }
    
    /// Rasterize a PUA character into a multi-cell canvas and return GlyphInfo for each cell.
    /// This is used when a PUA glyph is followed by space(s) - the glyph spans multiple cells.
    /// 
    /// Like Kitty's approach:
    /// 1. Render the glyph to a canvas sized for `num_cells` cells
    /// 2. Center the glyph horizontally within the canvas
    /// 3. Extract each cell's portion as a separate sprite
    /// 
    /// Returns a Vec of GlyphInfo, one for each cell.
    fn rasterize_pua_multicell(&mut self, c: char, num_cells: usize) -> Vec<GlyphInfo> {
        let cell_w = self.cell_width.ceil() as usize;
        let cell_h = self.cell_height.ceil() as usize;
        let canvas_width = cell_w * num_cells;
        
        // First, rasterize the glyph at full size
        let raster_result: Option<(u32, u32, Vec<u8>, f32, f32)> = {
            let glyph_id = self.primary_font.glyph_id(c);
            if glyph_id.0 != 0 {
                self.rasterize_glyph_ab(&self.primary_font.clone(), glyph_id)
            } else {
                // Try fallback fonts
                let mut result = None;
                for (_, fallback_font) in &self.fallback_fonts {
                    let fb_glyph_id = fallback_font.glyph_id(c);
                    if fb_glyph_id.0 != 0 {
                        result = self.rasterize_glyph_ab(&fallback_font.clone(), fb_glyph_id);
                        break;
                    }
                }
                result
            }
        };
        
        let Some((glyph_width, glyph_height, bitmap, _offset_x, offset_y)) = raster_result else {
            // Empty glyph - return empty sprites for each cell
            return vec![GlyphInfo {
                uv: [0.0, 0.0, 0.0, 0.0],
                size: [0.0, 0.0],
                is_colored: false,
            }; num_cells];
        };
        
        if bitmap.is_empty() || glyph_width == 0 || glyph_height == 0 {
            return vec![GlyphInfo {
                uv: [0.0, 0.0, 0.0, 0.0],
                size: [0.0, 0.0],
                is_colored: false,
            }; num_cells];
        }
        
        // Create a multi-cell canvas
        let mut canvas = vec![0u8; canvas_width * cell_h];
        
        // Position glyph at x=0 (left-aligned), like Kitty's model where
        // glyphs are positioned at origin without offset adjustments
        let dest_x = 0i32;
        
        // Calculate vertical position using baseline, same as single-cell rendering
        // dest_y = baseline - glyph_height - offset_y
        let dest_y = (self.baseline - glyph_height as f32 - offset_y).round() as i32;
        
        // Copy glyph bitmap to the multi-cell canvas
        for gy in 0..glyph_height as i32 {
            let cy = dest_y + gy;
            if cy < 0 || cy >= cell_h as i32 {
                continue;
            }
            for gx in 0..glyph_width as i32 {
                let cx = dest_x + gx;
                if cx < 0 || cx >= canvas_width as i32 {
                    continue;
                }
                let src_idx = (gy as u32 * glyph_width + gx as u32) as usize;
                let dst_idx = cy as usize * canvas_width + cx as usize;
                canvas[dst_idx] = canvas[dst_idx].max(bitmap[src_idx]);
            }
        }
        
        // Extract each cell's portion as a separate sprite
        let mut sprites = Vec::with_capacity(num_cells);
        
        for cell_idx in 0..num_cells {
            // Extract this cell's portion from the canvas
            let mut cell_canvas = vec![0u8; cell_w * cell_h];
            let cell_start_x = cell_idx * cell_w;
            
            for y in 0..cell_h {
                for x in 0..cell_w {
                    let src_idx = y * canvas_width + cell_start_x + x;
                    let dst_idx = y * cell_w + x;
                    cell_canvas[dst_idx] = canvas[src_idx];
                }
            }
            
            // Upload this cell's sprite to the atlas
            let info = self.upload_cell_canvas_to_atlas(&cell_canvas, false);
            sprites.push(info);
        }
        
        sprites
    }
    
    /// Rasterize an emoji into a multi-cell canvas and return GlyphInfo for each cell.
    /// This uses the Cairo color font renderer since emoji are color glyphs.
    /// 
    /// Returns a Vec of GlyphInfo, one for each cell.
    fn rasterize_emoji_multicell(&mut self, c: char, num_cells: usize) -> Vec<GlyphInfo> {
        let cell_w = self.cell_width.ceil() as usize;
        let cell_h = self.cell_height.ceil() as usize;
        let canvas_width = cell_w * num_cells;
        
        // Find a color font for this emoji (find_color_font_for_char handles fontconfig internally)
        let Some(font_path) = find_color_font_for_char(c) else {
            log::debug!("No color font found for emoji U+{:04X}", c as u32);
            return vec![GlyphInfo {
                uv: [0.0, 0.0, 0.0, 0.0],
                size: [0.0, 0.0],
                is_colored: true,
            }; num_cells];
        };
        
        // Render the emoji using Cairo at full multi-cell size
        let color_glyph_data: Option<(u32, u32, Vec<u8>, f32, f32)> = {
            let mut renderer_cell = self.color_font_renderer.borrow_mut();
            if renderer_cell.is_none() {
                *renderer_cell = ColorFontRenderer::new().ok();
            }
            
            if let Some(ref mut renderer) = *renderer_cell {
                // Render at multi-cell width
                renderer.render_color_glyph(
                    &font_path, c, self.font_size, 
                    (cell_w * num_cells) as u32, cell_h as u32
                )
            } else {
                None
            }
        };
        
        let Some((glyph_width, glyph_height, rgba, offset_x, offset_y)) = color_glyph_data else {
            log::debug!("Failed to render emoji U+{:04X}", c as u32);
            return vec![GlyphInfo {
                uv: [0.0, 0.0, 0.0, 0.0],
                size: [0.0, 0.0],
                is_colored: true,
            }; num_cells];
        };
        
        if rgba.is_empty() || glyph_width == 0 || glyph_height == 0 {
            return vec![GlyphInfo {
                uv: [0.0, 0.0, 0.0, 0.0],
                size: [0.0, 0.0],
                is_colored: true,
            }; num_cells];
        }
        
        // Create a multi-cell RGBA canvas
        let mut canvas = vec![0u8; canvas_width * cell_h * 4];
        
        // Position the glyph - for color glyphs, offset_y is ascent (distance from baseline to TOP)
        let dest_x = offset_x.round() as i32;
        let dest_y = (self.baseline - offset_y).round() as i32;
        
        // Copy the RGBA bitmap to the multi-cell canvas
        for gy in 0..glyph_height as i32 {
            let cy = dest_y + gy;
            if cy < 0 || cy >= cell_h as i32 {
                continue;
            }
            for gx in 0..glyph_width as i32 {
                let cx = dest_x + gx;
                if cx < 0 || cx >= canvas_width as i32 {
                    continue;
                }
                let src_idx = (gy as u32 * glyph_width + gx as u32) as usize * 4;
                let dst_idx = (cy as usize * canvas_width + cx as usize) * 4;
                if src_idx + 3 < rgba.len() && dst_idx + 3 < canvas.len() {
                    canvas[dst_idx] = rgba[src_idx];
                    canvas[dst_idx + 1] = rgba[src_idx + 1];
                    canvas[dst_idx + 2] = rgba[src_idx + 2];
                    canvas[dst_idx + 3] = rgba[src_idx + 3];
                }
            }
        }
        
        // Extract each cell's portion as a separate sprite
        let mut sprites = Vec::with_capacity(num_cells);
        
        for cell_idx in 0..num_cells {
            // Extract this cell's RGBA portion from the canvas
            let mut cell_canvas = vec![0u8; cell_w * cell_h * 4];
            let cell_start_x = cell_idx * cell_w;
            
            for y in 0..cell_h {
                for x in 0..cell_w {
                    let src_idx = (y * canvas_width + cell_start_x + x) * 4;
                    let dst_idx = (y * cell_w + x) * 4;
                    if src_idx + 3 < canvas.len() && dst_idx + 3 < cell_canvas.len() {
                        cell_canvas[dst_idx] = canvas[src_idx];
                        cell_canvas[dst_idx + 1] = canvas[src_idx + 1];
                        cell_canvas[dst_idx + 2] = canvas[src_idx + 2];
                        cell_canvas[dst_idx + 3] = canvas[src_idx + 3];
                    }
                }
            }
            
            // Upload this cell's sprite to the atlas (colored = true for RGBA)
            let info = self.upload_cell_canvas_to_atlas(&cell_canvas, true);
            sprites.push(info);
        }
        
        sprites
    }
    
    /// Rasterize a glyph using ab_glyph with pixel-perfect alignment.
    /// Returns (width, height, bitmap, offset_x, offset_y) or None if glyph has no outline.
    /// offset_x is the left bearing (horizontal offset from cursor), snapped to integer pixels
    /// offset_y is compatible with fontdue's ymin (distance from baseline to glyph bottom, negative for descenders)
    fn rasterize_glyph_ab(&self, font: &FontRef<'_>, glyph_id: GlyphId) -> Option<(u32, u32, Vec<u8>, f32, f32)> {
        // First, get the unpositioned glyph bounds to determine pixel-aligned position
        let unpositioned = glyph_id.with_scale_and_position(self.font_size, ab_glyph::point(0.0, 0.0));
        let outlined_check = font.outline_glyph(unpositioned)?;
        let raw_bounds = outlined_check.px_bounds();
        
        // Snap to integer pixel boundaries for crisp rendering.
        // Floor the min coordinates to ensure the glyph bitmap starts at an integer pixel.
        // This prevents antialiasing artifacts where horizontal/vertical lines appear
        // to have uneven thickness due to fractional pixel positioning.
        let snapped_min_x = raw_bounds.min.x.floor();
        let snapped_min_y = raw_bounds.min.y.floor();
        
        // Position the glyph so its bounds start at integer pixels.
        // We offset by the fractional part to align to pixel grid.
        let offset_to_snap_x = snapped_min_x - raw_bounds.min.x;
        let offset_to_snap_y = snapped_min_y - raw_bounds.min.y;
        let snapped_glyph = glyph_id.with_scale_and_position(
            self.font_size,
            ab_glyph::point(offset_to_snap_x, offset_to_snap_y),
        );
        
        let outlined = font.outline_glyph(snapped_glyph)?;
        let bounds = outlined.px_bounds();
        
        // Now bounds.min.x and bounds.min.y should be very close to integers
        let width = bounds.width().ceil() as u32;
        let height = bounds.height().ceil() as u32;
        
        if width == 0 || height == 0 {
            return None;
        }
        
        let mut bitmap = vec![0u8; (width * height) as usize];
        
        outlined.draw(|x, y, coverage| {
            let x = x as u32;
            let y = y as u32;
            if x < width && y < height {
                let idx = (y * width + x) as usize;
                bitmap[idx] = (coverage * 255.0) as u8;
            }
        });
        
        // Use the snapped (integer) offsets for positioning.
        // offset_x = left bearing, snapped to integer pixels
        // offset_y = distance from baseline to glyph BOTTOM (fontdue's ymin convention)
        //
        // ab_glyph's bounds.min.y is the TOP of the glyph (negative = above baseline)
        // ab_glyph's bounds.max.y is the BOTTOM of the glyph (positive = below baseline)
        // 
        // We use the snapped bounds which are now at integer pixel positions.
        let offset_x = snapped_min_x;
        let offset_y = -(raw_bounds.max.y + offset_to_snap_y);  // Snap the bottom too
        
        Some((width, height, bitmap, offset_x, offset_y))
    }

    /// Place a glyph bitmap into a cell-sized canvas at the correct baseline position.
    /// This follows Kitty's model where sprites are always cell-sized.
    /// 
    /// Parameters:
    /// - bitmap: The rasterized glyph bitmap (grayscale)
    /// - glyph_width, glyph_height: Dimensions of the bitmap
    /// - offset_x: Left bearing (horizontal offset from cell origin)
    /// - offset_y: Distance from baseline to glyph bottom (negative = below baseline)
    /// 
    /// Returns: Cell-sized canvas with the glyph positioned at baseline
    fn place_glyph_in_cell_canvas(
        &self,
        bitmap: &[u8],
        glyph_width: u32,
        glyph_height: u32,
        offset_x: f32,
        offset_y: f32,
    ) -> Vec<u8> {
        let cell_w = self.cell_width.ceil() as usize;
        let cell_h = self.cell_height.ceil() as usize;
        let mut canvas = vec![0u8; cell_w * cell_h];
        
        // Calculate destination position in the cell canvas.
        // baseline is the Y position where the baseline sits (from top of cell).
        // offset_y is the distance from baseline to glyph bottom.
        // glyph_top = baseline - (glyph_height + offset_y)
        //           = baseline - glyph_height - offset_y
        // Since offset_y can be negative (for descenders), this works correctly.
        let dest_x = offset_x.round() as i32;
        let dest_y = (self.baseline - glyph_height as f32 - offset_y).round() as i32;
        
        // Copy the glyph bitmap to the canvas, clipping to cell bounds
        for gy in 0..glyph_height as i32 {
            let cy = dest_y + gy;
            if cy < 0 || cy >= cell_h as i32 {
                continue;
            }
            for gx in 0..glyph_width as i32 {
                let cx = dest_x + gx;
                if cx < 0 || cx >= cell_w as i32 {
                    continue;
                }
                let src_idx = (gy as u32 * glyph_width + gx as u32) as usize;
                let dst_idx = cy as usize * cell_w + cx as usize;
                // Use max to handle overlapping glyphs (shouldn't happen for single chars)
                canvas[dst_idx] = canvas[dst_idx].max(bitmap[src_idx]);
            }
        }
        
        canvas
    }

    /// Place a colored (RGBA) glyph bitmap into a cell-sized RGBA canvas.
    /// Used for emoji and other color glyphs.
    fn place_color_glyph_in_cell_canvas(
        &self,
        bitmap: &[u8],
        glyph_width: u32,
        glyph_height: u32,
        offset_x: f32,
        offset_y: f32,
    ) -> Vec<u8> {
        let cell_w = self.cell_width.ceil() as usize;
        let cell_h = self.cell_height.ceil() as usize;
        let mut canvas = vec![0u8; cell_w * cell_h * 4]; // RGBA
        
        // For color glyphs, offset_y is the ascent (distance from baseline to TOP of glyph)
        // So dest_y = baseline - offset_y positions the top of the glyph correctly
        let dest_x = offset_x.round() as i32;
        let dest_y = (self.baseline - offset_y).round() as i32;
        
        // Copy the RGBA bitmap to the canvas
        for gy in 0..glyph_height as i32 {
            let cy = dest_y + gy;
            if cy < 0 || cy >= cell_h as i32 {
                continue;
            }
            for gx in 0..glyph_width as i32 {
                let cx = dest_x + gx;
                if cx < 0 || cx >= cell_w as i32 {
                    continue;
                }
                let src_idx = (gy as u32 * glyph_width + gx as u32) as usize * 4;
                let dst_idx = (cy as usize * cell_w + cx as usize) * 4;
                // For color glyphs, just copy the RGBA values
                // (could do alpha blending if needed, but single glyph per cell)
                if src_idx + 3 < bitmap.len() && dst_idx + 3 < canvas.len() {
                    canvas[dst_idx] = bitmap[src_idx];
                    canvas[dst_idx + 1] = bitmap[src_idx + 1];
                    canvas[dst_idx + 2] = bitmap[src_idx + 2];
                    canvas[dst_idx + 3] = bitmap[src_idx + 3];
                }
            }
        }
        
        canvas
    }

    /// Upload a cell-sized grayscale canvas to the atlas.
    /// Returns GlyphInfo with UV coordinates pointing to the uploaded sprite.
    fn upload_cell_canvas_to_atlas(&mut self, canvas: &[u8], is_colored: bool) -> GlyphInfo {
        let cell_w = self.cell_width.ceil() as u32;
        let cell_h = self.cell_height.ceil() as u32;
        
        // Check if we need to move to next row
        if self.atlas_cursor_x + cell_w > ATLAS_SIZE {
            self.atlas_cursor_x = 0;
            self.atlas_cursor_y += self.atlas_row_height + 1;
            self.atlas_row_height = 0;
        }
        
        // Check if atlas is full - reset and retry
        if self.atlas_cursor_y + cell_h > ATLAS_SIZE {
            self.reset_atlas();
            if self.atlas_cursor_x + cell_w > ATLAS_SIZE {
                self.atlas_cursor_x = 0;
                self.atlas_cursor_y += self.atlas_row_height + 1;
                self.atlas_row_height = 0;
            }
        }
        
        // Copy canvas to atlas
        if is_colored {
            // RGBA canvas - copy directly
            for y in 0..cell_h as usize {
                for x in 0..cell_w as usize {
                    let src_idx = (y * cell_w as usize + x) * 4;
                    let dst_x = self.atlas_cursor_x + x as u32;
                    let dst_y = self.atlas_cursor_y + y as u32;
                    let dst_idx = ((dst_y * ATLAS_SIZE + dst_x) * ATLAS_BPP) as usize;
                    if src_idx + 3 < canvas.len() && dst_idx + 3 < self.atlas_data.len() {
                        self.atlas_data[dst_idx] = canvas[src_idx];
                        self.atlas_data[dst_idx + 1] = canvas[src_idx + 1];
                        self.atlas_data[dst_idx + 2] = canvas[src_idx + 2];
                        self.atlas_data[dst_idx + 3] = canvas[src_idx + 3];
                    }
                }
            }
        } else {
            // Grayscale canvas - convert to RGBA (white with alpha)
            for y in 0..cell_h as usize {
                for x in 0..cell_w as usize {
                    let src_idx = y * cell_w as usize + x;
                    let dst_x = self.atlas_cursor_x + x as u32;
                    let dst_y = self.atlas_cursor_y + y as u32;
                    let dst_idx = ((dst_y * ATLAS_SIZE + dst_x) * ATLAS_BPP) as usize;
                    if src_idx < canvas.len() && dst_idx + 3 < self.atlas_data.len() {
                        self.atlas_data[dst_idx] = 255;     // R
                        self.atlas_data[dst_idx + 1] = 255; // G
                        self.atlas_data[dst_idx + 2] = 255; // B
                        self.atlas_data[dst_idx + 3] = canvas[src_idx]; // A
                    }
                }
            }
        }
        self.atlas_dirty = true;
        
        // Calculate UV coordinates
        let uv_x = self.atlas_cursor_x as f32 / ATLAS_SIZE as f32;
        let uv_y = self.atlas_cursor_y as f32 / ATLAS_SIZE as f32;
        let uv_w = cell_w as f32 / ATLAS_SIZE as f32;
        let uv_h = cell_h as f32 / ATLAS_SIZE as f32;
        
        // Update atlas cursor
        self.atlas_cursor_x += cell_w + 1;
        self.atlas_row_height = self.atlas_row_height.max(cell_h);
        
        GlyphInfo {
            uv: [uv_x, uv_y, uv_w, uv_h],
            size: [cell_w as f32, cell_h as f32],
            is_colored,
        }
    }

    /// Get or rasterize a glyph by its glyph ID from the primary font.
    /// Used for ligatures where we have the glyph ID from rustybuzz.
    /// Note: Kept for potential fallback use. Use get_glyph_by_id_with_style for styled text.
    #[allow(dead_code)]
    fn get_glyph_by_id(&mut self, glyph_id: u16) -> GlyphInfo {
        // Cache key: (font_style, font_index, glyph_id)
        // For now, we use Regular style (0) and primary font index (0)
        let cache_key = (FontStyle::Regular as usize, 0usize, glyph_id);
        if let Some(info) = self.glyph_cache.get(&cache_key) {
            return *info;
        }

        // Rasterize the glyph by ID from primary font using ab_glyph
        let ab_glyph_id = GlyphId(glyph_id);
        let raster_result = self.rasterize_glyph_ab(&self.primary_font.clone(), ab_glyph_id);

        let Some((glyph_width, glyph_height, bitmap, offset_x, offset_y)) = raster_result else {
            // Empty glyph (e.g., space)
            let info = GlyphInfo {
                uv: [0.0, 0.0, 0.0, 0.0],
                size: [0.0, 0.0],
                is_colored: false,
            };
            self.glyph_cache.insert(cache_key, info);
            return info;
        };

        if bitmap.is_empty() || glyph_width == 0 || glyph_height == 0 {
            // Empty glyph (e.g., space)
            let info = GlyphInfo {
                uv: [0.0, 0.0, 0.0, 0.0],
                size: [0.0, 0.0],
                is_colored: false,
            };
            self.glyph_cache.insert(cache_key, info);
            return info;
        }

        // Place the glyph in a cell-sized canvas at the correct baseline position
        let canvas = self.place_glyph_in_cell_canvas(
            &bitmap, glyph_width, glyph_height, offset_x, offset_y
        );
        let info = self.upload_cell_canvas_to_atlas(&canvas, false);

        self.glyph_cache.insert(cache_key, info);
        info
    }

    /// Get or rasterize a glyph by its glyph ID from a specific font variant.
    /// Uses bold/italic font if available, otherwise falls back to regular.
    fn get_glyph_by_id_with_style(&mut self, glyph_id: u16, style: FontStyle) -> GlyphInfo {
        // Cache key: (font_style, font_index, glyph_id)
        // font_index 0 = primary/regular font
        let cache_key = (style as usize, 0usize, glyph_id);
        if let Some(info) = self.glyph_cache.get(&cache_key) {
            return *info;
        }

        // Get the font for the requested style
        let font = if style == FontStyle::Regular {
            self.primary_font.clone()
        } else if let Some(ref variant) = self.font_variants[style as usize] {
            variant.font.clone()
        } else {
            // Fall back to regular font if variant not available
            self.primary_font.clone()
        };

        // Rasterize the glyph by ID using ab_glyph
        let ab_glyph_id = GlyphId(glyph_id);
        let raster_result = self.rasterize_glyph_ab(&font, ab_glyph_id);

        let Some((glyph_width, glyph_height, bitmap, offset_x, offset_y)) = raster_result else {
            // Empty glyph (e.g., space)
            let info = GlyphInfo {
                uv: [0.0, 0.0, 0.0, 0.0],
                size: [0.0, 0.0],
                is_colored: false,
            };
            self.glyph_cache.insert(cache_key, info);
            return info;
        };

        if bitmap.is_empty() || glyph_width == 0 || glyph_height == 0 {
            // Empty glyph (e.g., space)
            let info = GlyphInfo {
                uv: [0.0, 0.0, 0.0, 0.0],
                size: [0.0, 0.0],
                is_colored: false,
            };
            self.glyph_cache.insert(cache_key, info);
            return info;
        }

        // Place the glyph in a cell-sized canvas at the correct baseline position
        let canvas = self.place_glyph_in_cell_canvas(
            &bitmap, glyph_width, glyph_height, offset_x, offset_y
        );
        let info = self.upload_cell_canvas_to_atlas(&canvas, false);

        self.glyph_cache.insert(cache_key, info);
        info
    }

    /// Shape a text string using HarfBuzz/rustybuzz.
    /// Returns glyph IDs with advances and offsets for texture healing.
    /// Note: Kept for potential fallback use. Use shape_text_with_style for styled text.
    #[allow(dead_code)]
    fn shape_text(&mut self, text: &str) -> ShapedGlyphs {
        // Check cache first
        if let Some(cached) = self.ligature_cache.get(text) {
            return cached.clone();
        }

        let _chars: Vec<char> = text.chars().collect();

        let mut buffer = UnicodeBuffer::new();
        buffer.push_str(text);

        // Shape with OpenType features enabled (liga, calt, dlig)
        let glyph_buffer = rustybuzz::shape(&self.shaping_ctx.face, &self.shaping_ctx.features, buffer);
        let glyph_infos = glyph_buffer.glyph_infos();
        let glyph_positions = glyph_buffer.glyph_positions();

        let glyphs: Vec<(u16, f32, f32, f32, u32)> = glyph_infos
            .iter()
            .zip(glyph_positions.iter())
            .map(|(info, pos)| {
                let glyph_id = info.glyph_id as u16;
                // Ensure glyph is rasterized
                self.get_glyph_by_id(glyph_id);
                // Convert from font units to pixels using the correct scale factor.
                // This matches ab_glyph's calculation: font_size / height_unscaled
                let x_advance = pos.x_advance as f32 * self.font_units_to_px;
                let x_offset = pos.x_offset as f32 * self.font_units_to_px;
                let y_offset = pos.y_offset as f32 * self.font_units_to_px;
                (glyph_id, x_advance, x_offset, y_offset, info.cluster)
            })
            .collect();

        let shaped = ShapedGlyphs {
            glyphs,
        };
        self.ligature_cache.insert(text.to_string(), shaped.clone());
        shaped
    }

    /// Shape a text string using HarfBuzz/rustybuzz with a specific font style.
    /// Uses the bold/italic font variant if available, otherwise falls back to regular.
    fn shape_text_with_style(&mut self, text: &str, style: FontStyle) -> ShapedGlyphs {
        // For now, we'll create a cache key that includes style
        // TODO: Could optimize by having separate caches per style
        let cache_key = format!("{}\x00{}", style as usize, text);
        if let Some(cached) = self.ligature_cache.get(&cache_key) {
            return cached.clone();
        }

        let mut buffer = UnicodeBuffer::new();
        buffer.push_str(text);

        // Get the face for the requested style, falling back to regular if not available
        let face = if style == FontStyle::Regular {
            &self.shaping_ctx.face
        } else if let Some(ref variant) = self.font_variants[style as usize] {
            &variant.face
        } else {
            // Fall back to regular font
            &self.shaping_ctx.face
        };

        // Shape with OpenType features enabled (liga, calt, dlig)
        let glyph_buffer = rustybuzz::shape(face, &self.shaping_features, buffer);
        let glyph_infos = glyph_buffer.glyph_infos();
        let glyph_positions = glyph_buffer.glyph_positions();

        let glyphs: Vec<(u16, f32, f32, f32, u32)> = glyph_infos
            .iter()
            .zip(glyph_positions.iter())
            .map(|(info, pos)| {
                let glyph_id = info.glyph_id as u16;
                // Note: We don't pre-rasterize here; that happens in render_glyphs_to_canvas_with_style
                // Convert from font units to pixels using the correct scale factor.
                let x_advance = pos.x_advance as f32 * self.font_units_to_px;
                let x_offset = pos.x_offset as f32 * self.font_units_to_px;
                let y_offset = pos.y_offset as f32 * self.font_units_to_px;
                (glyph_id, x_advance, x_offset, y_offset, info.cluster)
            })
            .collect();

        let shaped = ShapedGlyphs { glyphs };
        self.ligature_cache.insert(cache_key, shaped.clone());
        shaped
    }


    /// Convert sRGB component (0.0-1.0) to linear RGB.
    /// This is needed because we're rendering to an sRGB surface.
    #[inline]
    fn srgb_to_linear(c: f32) -> f32 {
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    }

    /// Convert pixel X coordinate to NDC, snapped to pixel boundaries.
    #[inline]
    fn pixel_to_ndc_x(pixel: f32, screen_width: f32) -> f32 {
        let snapped = pixel.round();
        (snapped / screen_width) * 2.0 - 1.0
    }

    /// Convert pixel Y coordinate to NDC (inverted), snapped to pixel boundaries.
    #[inline]
    fn pixel_to_ndc_y(pixel: f32, screen_height: f32) -> f32 {
        let snapped = pixel.round();
        1.0 - (snapped / screen_height) * 2.0
    }


    /// Draw a filled rectangle.
    fn render_rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: [f32; 4]) {
        // Add quad to the batch for instanced rendering
        if self.quads.len() < self.max_quads {
            self.quads.push(Quad {
                x,
                y,
                width: w,
                height: h,
                color,
            });
        }
    }

    /// Draw a filled rectangle to the overlay layer (rendered on top of everything).
    fn render_overlay_rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: [f32; 4]) {
        // Add quad to the overlay batch for instanced rendering (rendered last)
        self.overlay_quads.push(Quad {
            x,
            y,
            width: w,
            height: h,
            color,
        });
    }

    /// Prepare edge glow uniform data for shader-based rendering.
    /// Returns the uniform data to be uploaded to the GPU.
    /// Prepare combined edge glow uniform data for all active glows.
    fn prepare_edge_glow_uniforms(&self, glows: &[EdgeGlow], terminal_y_offset: f32, intensity: f32) -> EdgeGlowUniforms {
        // Use the same color as the active pane border (palette color 4 - typically blue)
        let [r, g, b] = self.palette.colors[4];
        let color_r = Self::srgb_to_linear(r as f32 / 255.0);
        let color_g = Self::srgb_to_linear(g as f32 / 255.0);
        let color_b = Self::srgb_to_linear(b as f32 / 255.0);

        let mut glow_instances = [GlowInstance {
            direction: 0,
            progress: 0.0,
            color_r: 0.0,
            color_g: 0.0,
            color_b: 0.0,
            pane_x: 0.0,
            pane_y: 0.0,
            pane_width: 0.0,
            pane_height: 0.0,
            _padding1: 0.0,
            _padding2: 0.0,
            _padding3: 0.0,
        }; MAX_EDGE_GLOWS];

        let glow_count = glows.len().min(MAX_EDGE_GLOWS);

        for (i, glow) in glows.iter().take(MAX_EDGE_GLOWS).enumerate() {
            let direction = match glow.direction {
                Direction::Up => 0,
                Direction::Down => 1,
                Direction::Left => 2,
                Direction::Right => 3,
            };

            // Glow coordinates are already in screen space (transformed by calculate_edge_glow_bounds)
            glow_instances[i] = GlowInstance {
                direction,
                progress: glow.progress(),
                color_r,
                color_g,
                color_b,
                pane_x: glow.pane_x,
                pane_y: glow.pane_y,
                pane_width: glow.pane_width,
                pane_height: glow.pane_height,
                _padding1: 0.0,
                _padding2: 0.0,
                _padding3: 0.0,
            };
        }

        EdgeGlowUniforms {
            screen_width: self.width as f32,
            screen_height: self.height as f32,
            terminal_y_offset,
            glow_intensity: intensity,
            glow_count: glow_count as u32,
            _padding: [0; 3],
            glows: glow_instances,
        }
    }

    /// Render multiple panes with borders.
    ///
    /// Arguments:
    /// - `panes`: List of (terminal, pane_info, selection) tuples
    /// - `num_tabs`: Number of tabs for the tab bar
    /// - `active_tab`: Index of the active tab
    /// - `edge_glows`: Active edge glow animations for visual feedback
    /// - `edge_glow_intensity`: Intensity of edge glow effect (0.0 = disabled, 1.0 = full)
    /// - `statusline_content`: Content to render in the statusline
    pub fn render_panes(
        &mut self,
        panes: &[(&Terminal, PaneRenderInfo, Option<(usize, usize, usize, usize)>)],
        num_tabs: usize,
        active_tab: usize,
        edge_glows: &[EdgeGlow],
        edge_glow_intensity: f32,
        statusline_content: &StatuslineContent,
    ) -> Result<(), wgpu::SurfaceError> {
        // Sync palette from first terminal
        if let Some((terminal, _, _)) = panes.first() {
            self.palette = terminal.palette.clone();
            log::debug!("render_panes: synced palette from first terminal, default_bg={:?}, default_fg={:?}", 
                self.palette.default_bg, self.palette.default_fg);
        } else {
            log::debug!("render_panes: no panes, using existing palette");
        }

        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Clear buffers
        self.bg_vertices.clear();
        self.bg_indices.clear();
        self.glyph_vertices.clear();
        self.glyph_indices.clear();
        self.quads.clear();
        self.overlay_quads.clear();

        // Check if atlas is getting full and reset proactively
        // This prevents mid-render failures and ensures all glyphs can be rendered
        let atlas_usage = self.atlas_cursor_y as f32 / ATLAS_SIZE as f32;
        if atlas_usage > 0.9 {
            self.reset_atlas();
        }

        let width = self.width as f32;
        let height = self.height as f32;
        let tab_bar_height = self.tab_bar_height();
        let terminal_y_offset = self.terminal_y_offset();
        
        // Grid centering offsets - center the cell grid in the window
        let grid_x_offset = self.grid_x_offset();
        let grid_y_offset = self.grid_y_offset();

        // ═══════════════════════════════════════════════════════════════════
        // RENDER TAB BAR (same as render_from_terminal)
        // ═══════════════════════════════════════════════════════════════════
        if self.tab_bar_position != TabBarPosition::Hidden && num_tabs > 0 {
            let tab_bar_y = match self.tab_bar_position {
                TabBarPosition::Top => 0.0,
                TabBarPosition::Bottom => height - tab_bar_height,
                TabBarPosition::Hidden => unreachable!(),
            };

            // Use same color as statusline: 0x1a1a1a (26, 26, 26) in sRGB
            // Linear RGB value: ~0.00972
            let tab_bar_bg = [
                Self::srgb_to_linear(26.0 / 255.0),
                Self::srgb_to_linear(26.0 / 255.0),
                Self::srgb_to_linear(26.0 / 255.0),
                1.0,
            ];

            // Draw tab bar background
            log::debug!("render_panes: drawing tab bar at y={}, height={}, num_tabs={}, quads_before={}", 
                tab_bar_y, tab_bar_height, num_tabs, self.quads.len());
            self.render_rect(0.0, tab_bar_y, width, tab_bar_height, tab_bar_bg);
            log::debug!("render_panes: after tab bar rect, quads_count={}", self.quads.len());

            // Render each tab
            let mut tab_x = 4.0_f32;
            let tab_padding = 8.0_f32;
            let min_tab_width = self.cell_width * 8.0;

            for idx in 0..num_tabs {
                let is_active = idx == active_tab;
                let title = format!(" {} ", idx + 1);
                let title_width = title.chars().count() as f32 * self.cell_width;
                let tab_width = title_width.max(min_tab_width);

                let tab_bg = if is_active {
                    // Active tab: brightest - significantly brighter than tab bar
                    let [r, g, b] = self.palette.default_bg;
                    let boost = 50.0_f32; // More visible for active tab
                    [
                        Self::srgb_to_linear((r as f32 + boost).min(255.0) / 255.0),
                        Self::srgb_to_linear((g as f32 + boost).min(255.0) / 255.0),
                        Self::srgb_to_linear((b as f32 + boost).min(255.0) / 255.0),
                        1.0,
                    ]
                } else {
                    // Inactive tab: slightly brighter than tab bar background
                    let [r, g, b] = self.palette.default_bg;
                    let boost = 30.0_f32;
                    [
                        Self::srgb_to_linear((r as f32 + boost).min(255.0) / 255.0),
                        Self::srgb_to_linear((g as f32 + boost).min(255.0) / 255.0),
                        Self::srgb_to_linear((b as f32 + boost).min(255.0) / 255.0),
                        1.0,
                    ]
                };

                let tab_fg = {
                    let [r, g, b] = self.palette.default_fg;
                    let alpha = if is_active { 1.0 } else { 0.6 };
                    [
                        Self::srgb_to_linear(r as f32 / 255.0),
                        Self::srgb_to_linear(g as f32 / 255.0),
                        Self::srgb_to_linear(b as f32 / 255.0),
                        alpha,
                    ]
                };

                // Draw tab background
                self.render_rect(tab_x, tab_bar_y + 2.0, tab_width, tab_bar_height - 4.0, tab_bg);

                // Render tab title text
                let text_y = tab_bar_y + (tab_bar_height - self.cell_height) / 2.0;
                let text_x = tab_x + (tab_width - title_width) / 2.0;

                for (char_idx, c) in title.chars().enumerate() {
                    if c == ' ' {
                        continue;
                    }
                    let glyph = self.rasterize_char(c);
                    if glyph.size[0] > 0.0 && glyph.size[1] > 0.0 {
                        // In Kitty's model, glyphs are cell-sized and positioned at (0,0)
                        let char_x = text_x + char_idx as f32 * self.cell_width;
                        let glyph_x = char_x.round();
                        let glyph_y = text_y.round();

                        let left = Self::pixel_to_ndc_x(glyph_x, width);
                        let right = Self::pixel_to_ndc_x(glyph_x + glyph.size[0], width);
                        let top = Self::pixel_to_ndc_y(glyph_y, height);
                        let bottom = Self::pixel_to_ndc_y(glyph_y + glyph.size[1], height);

                        let base_idx = self.glyph_vertices.len() as u32;
                        self.glyph_vertices.push(GlyphVertex {
                            position: [left, top],
                            uv: [glyph.uv[0], glyph.uv[1]],
                            color: tab_fg,
                            bg_color: [0.0, 0.0, 0.0, 0.0],
                        });
                        self.glyph_vertices.push(GlyphVertex {
                            position: [right, top],
                            uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1]],
                            color: tab_fg,
                            bg_color: [0.0, 0.0, 0.0, 0.0],
                        });
                        self.glyph_vertices.push(GlyphVertex {
                            position: [right, bottom],
                            uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1] + glyph.uv[3]],
                            color: tab_fg,
                            bg_color: [0.0, 0.0, 0.0, 0.0],
                        });
                        self.glyph_vertices.push(GlyphVertex {
                            position: [left, bottom],
                            uv: [glyph.uv[0], glyph.uv[1] + glyph.uv[3]],
                            color: tab_fg,
                            bg_color: [0.0, 0.0, 0.0, 0.0],
                        });
                        self.glyph_indices.extend_from_slice(&[
                            base_idx, base_idx + 1, base_idx + 2,
                            base_idx, base_idx + 2, base_idx + 3,
                        ]);
                    }
                }

                tab_x += tab_width + tab_padding;
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // RENDER PANE BORDERS (only between adjacent panes)
        // ═══════════════════════════════════════════════════════════════════
        let border_thickness = 2.0;
        let active_border_color = {
            // Use a bright accent color for active pane
            let [r, g, b] = self.palette.colors[4]; // Blue from palette
            [
                Self::srgb_to_linear(r as f32 / 255.0),
                Self::srgb_to_linear(g as f32 / 255.0),
                Self::srgb_to_linear(b as f32 / 255.0),
                1.0,
            ]
        };
        let inactive_border_color = {
            // Use a dimmer color for inactive panes
            let [r, g, b] = self.palette.default_bg;
            let factor = 1.5_f32.min(2.0);
            [
                Self::srgb_to_linear((r as f32 / 255.0) * factor),
                Self::srgb_to_linear((g as f32 / 255.0) * factor),
                Self::srgb_to_linear((b as f32 / 255.0) * factor),
                1.0,
            ]
        };

        // Only draw borders if there's more than one pane
        // Panes are now flush against each other, so we draw borders at shared edges
        // Borders are rendered as overlays so they appear on top of pane content
        if panes.len() > 1 {
            // Tolerance for detecting adjacent panes (should be touching or very close)
            let adjacency_tolerance = 1.0;
            
            // Calculate grid boundaries for extending borders to screen edges
            // Same technique as edge glow and dim overlay
            let (available_width, available_height) = self.available_grid_space();
            let grid_top = terminal_y_offset;
            let grid_bottom = terminal_y_offset + available_height;
            let grid_left = 0.0_f32;
            let grid_right = width;
            let epsilon = self.cell_height.max(self.cell_width);

            // Check each pair of panes to find adjacent ones
            for i in 0..panes.len() {
                for j in (i + 1)..panes.len() {
                    let (_, info_a, _) = &panes[i];
                    let (_, info_b, _) = &panes[j];

                    // Use active border color if either pane is active
                    let border_color = if info_a.is_active || info_b.is_active {
                        active_border_color
                    } else {
                        inactive_border_color
                    };

                    // Calculate absolute positions (with terminal_y_offset and grid centering)
                    let a_x = grid_x_offset + info_a.x;
                    let a_y = terminal_y_offset + grid_y_offset + info_a.y;
                    let a_right = a_x + info_a.width;
                    let a_bottom = a_y + info_a.height;

                    let b_x = grid_x_offset + info_b.x;
                    let b_y = terminal_y_offset + grid_y_offset + info_b.y;
                    let b_right = b_x + info_b.width;
                    let b_bottom = b_y + info_b.height;

                    // Check for vertical adjacency (panes side by side)
                    // Pane A is to the left of pane B (A's right edge touches B's left edge)
                    if (a_right - b_x).abs() < adjacency_tolerance {
                        // Check if they overlap vertically
                        let mut top = a_y.max(b_y);
                        let mut bottom = a_bottom.min(b_bottom);
                        if bottom > top {
                            // Extend to grid edges if both panes reach the edge
                            // Top edge: extend if both panes are at grid top
                            if info_a.y < epsilon && info_b.y < epsilon {
                                top = grid_top;
                            }
                            // Bottom edge: extend if both panes reach grid bottom
                            if (info_a.y + info_a.height) >= available_height - epsilon 
                               && (info_b.y + info_b.height) >= available_height - epsilon {
                                bottom = grid_bottom;
                            }
                            // Draw vertical border centered on their shared edge
                            let border_x = a_right - border_thickness / 2.0;
                            self.render_overlay_rect(border_x, top, border_thickness, bottom - top, border_color);
                        }
                    }
                    // Pane B is to the left of pane A
                    if (b_right - a_x).abs() < adjacency_tolerance {
                        let mut top = a_y.max(b_y);
                        let mut bottom = a_bottom.min(b_bottom);
                        if bottom > top {
                            // Extend to grid edges if both panes reach the edge
                            if info_a.y < epsilon && info_b.y < epsilon {
                                top = grid_top;
                            }
                            if (info_a.y + info_a.height) >= available_height - epsilon 
                               && (info_b.y + info_b.height) >= available_height - epsilon {
                                bottom = grid_bottom;
                            }
                            let border_x = b_right - border_thickness / 2.0;
                            self.render_overlay_rect(border_x, top, border_thickness, bottom - top, border_color);
                        }
                    }

                    // Check for horizontal adjacency (panes stacked)
                    // Pane A is above pane B (A's bottom edge touches B's top edge)
                    if (a_bottom - b_y).abs() < adjacency_tolerance {
                        // Check if they overlap horizontally
                        let mut left = a_x.max(b_x);
                        let mut right = a_right.min(b_right);
                        if right > left {
                            // Extend to screen edges if both panes reach the edge
                            // Left edge: extend if both panes are at grid left
                            if info_a.x < epsilon && info_b.x < epsilon {
                                left = grid_left;
                            }
                            // Right edge: extend if both panes reach grid right
                            if (info_a.x + info_a.width) >= available_width - epsilon 
                               && (info_b.x + info_b.width) >= available_width - epsilon {
                                right = grid_right;
                            }
                            // Draw horizontal border centered on their shared edge
                            let border_y = a_bottom - border_thickness / 2.0;
                            self.render_overlay_rect(left, border_y, right - left, border_thickness, border_color);
                        }
                    }
                    // Pane B is above pane A
                    if (b_bottom - a_y).abs() < adjacency_tolerance {
                        let mut left = a_x.max(b_x);
                        let mut right = a_right.min(b_right);
                        if right > left {
                            // Extend to screen edges if both panes reach the edge
                            if info_a.x < epsilon && info_b.x < epsilon {
                                left = grid_left;
                            }
                            if (info_a.x + info_a.width) >= available_width - epsilon 
                               && (info_b.x + info_b.width) >= available_width - epsilon {
                                right = grid_right;
                            }
                            let border_y = b_bottom - border_thickness / 2.0;
                            self.render_overlay_rect(left, border_y, right - left, border_thickness, border_color);
                        }
                    }
                }
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // RENDER EACH PANE'S CONTENT (Like Kitty's per-window VAO approach)
        // ═══════════════════════════════════════════════════════════════════
        // Each pane gets its own GPU buffers and bind group.
        // We upload all pane data BEFORE starting the render pass,
        // then use each pane's bind group during rendering.
        struct PaneRenderData {
            pane_id: u64,
            cols: u32,
            rows: u32,
            dim_overlay: Option<(f32, f32, f32, f32, [f32; 4])>, // (x, y, w, h, color)
        }
        let mut pane_render_list: Vec<PaneRenderData> = Vec::new();
        
        // First pass: collect pane data, ensure GPU resources exist, and upload data
        for (terminal, info, selection) in panes {
            // Apply grid centering offsets to pane position
            let pane_x = grid_x_offset + info.x;
            let pane_y = terminal_y_offset + grid_y_offset + info.y;
            let pane_width = info.width;
            let pane_height = info.height;
            
            log::debug!("render_panes: pane {} at ({}, {}), size {}x{}, bottom_edge={}", 
                info.pane_id, pane_x, pane_y, pane_width, pane_height, pane_y + pane_height);

            // Update GPU cells for this terminal (populates self.gpu_cells)
            self.update_gpu_cells(terminal);
            
            // Calculate pane dimensions in cells
            let cols = (pane_width / self.cell_width).floor() as u32;
            let rows = (pane_height / self.cell_height).floor() as u32;
            
            // Use the actual gpu_cells size for buffer allocation (terminal.cols * terminal.rows)
            // This may differ from pane pixel dimensions due to rounding
            let actual_cells = self.gpu_cells.len();
            
            // Ensure this pane has GPU resources (like Kitty's create_cell_vao)
            // This creates or resizes buffers as needed
            let _pane_res = self.get_or_create_pane_resources(info.pane_id, actual_cells);
            
            // Build grid params for this pane
            let (sel_start_col, sel_start_row, sel_end_col, sel_end_row) = match selection {
                Some((sc, sr, ec, er)) => (*sc as i32, *sr as i32, *ec as i32, *er as i32),
                None => (-1, -1, -1, -1),
            };
            let grid_params = GridParams {
                cols,
                rows,
                cell_width: self.cell_width,
                cell_height: self.cell_height,
                screen_width: self.width as f32,
                screen_height: self.height as f32,
                x_offset: pane_x,
                y_offset: pane_y,
                // Hide cursor when scrolled into scrollback buffer or when cursor is explicitly hidden
                cursor_col: if terminal.cursor_visible && terminal.scroll_offset == 0 { terminal.cursor_col as i32 } else { -1 },
                cursor_row: if terminal.cursor_visible && terminal.scroll_offset == 0 { terminal.cursor_row as i32 } else { -1 },
                cursor_style: match terminal.cursor_shape {
                    CursorShape::BlinkingBlock | CursorShape::SteadyBlock => 0,
                    CursorShape::BlinkingUnderline | CursorShape::SteadyUnderline => 1,
                    CursorShape::BlinkingBar | CursorShape::SteadyBar => 2,
                },
                background_opacity: self.background_opacity,
                selection_start_col: sel_start_col,
                selection_start_row: sel_start_row,
                selection_end_col: sel_end_col,
                selection_end_row: sel_end_row,
            };
            
            // Upload this pane's cell data to its own buffer (like Kitty's send_cell_data_to_gpu)
            // This happens BEFORE the render pass, so each pane has its own data
            if let Some(pane_res) = self.pane_resources.get(&info.pane_id) {
                // Safety check: verify buffer can hold the data
                let data_size = self.gpu_cells.len() * std::mem::size_of::<GPUCell>();
                let buffer_size = pane_res.capacity * std::mem::size_of::<GPUCell>();
                if data_size > buffer_size {
                    // This shouldn't happen if get_or_create_pane_resources worked correctly
                    eprintln!(
                        "BUG: Buffer size mismatch for pane {}: data={} bytes, buffer={} bytes, gpu_cells.len()={}, capacity={}",
                        info.pane_id, data_size, buffer_size, self.gpu_cells.len(), pane_res.capacity
                    );
                    // Skip this pane to avoid crash - will be fixed next frame
                    continue;
                }
                
                self.queue.write_buffer(
                    &pane_res.cell_buffer,
                    0,
                    bytemuck::cast_slice(&self.gpu_cells),
                );
                self.queue.write_buffer(
                    &pane_res.grid_params_buffer,
                    0,
                    bytemuck::bytes_of(&grid_params),
                );
            }
            
            // Build dim overlay if needed - use calculate_dim_overlay_bounds to extend
            // edge panes to fill the terminal grid area (matching edge glow behavior)
            let dim_overlay = if info.dim_factor < 1.0 {
                let overlay_alpha = 1.0 - info.dim_factor;
                let overlay_color = [0.0, 0.0, 0.0, overlay_alpha];
                // Pass raw grid-relative coordinates, the helper transforms to screen space
                let (ox, oy, ow, oh) = self.calculate_dim_overlay_bounds(info.x, info.y, info.width, info.height);
                Some((ox, oy, ow, oh, overlay_color))
            } else {
                None
            };
            
            pane_render_list.push(PaneRenderData {
                pane_id: info.pane_id,
                cols,
                rows,
                dim_overlay,
            });
        }
        
        // Clean up resources for panes that no longer exist (like Kitty's remove_vao)
        let active_pane_ids: std::collections::HashSet<u64> = pane_render_list.iter().map(|p| p.pane_id).collect();
        self.cleanup_unused_pane_resources(&active_pane_ids);

        // ═══════════════════════════════════════════════════════════════════
        // UPLOAD SHARED DATA (color table)
        // ═══════════════════════════════════════════════════════════════════
        {
            let mut color_table_data = [[0.0f32; 4]; 258];
            for i in 0..256 {
                let [r, g, b] = self.palette.colors[i];
                color_table_data[i] = [
                    Self::srgb_to_linear(r as f32 / 255.0),
                    Self::srgb_to_linear(g as f32 / 255.0),
                    Self::srgb_to_linear(b as f32 / 255.0),
                    1.0,
                ];
            }
            let [fg_r, fg_g, fg_b] = self.palette.default_fg;
            color_table_data[256] = [
                Self::srgb_to_linear(fg_r as f32 / 255.0),
                Self::srgb_to_linear(fg_g as f32 / 255.0),
                Self::srgb_to_linear(fg_b as f32 / 255.0),
                1.0,
            ];
            let [bg_r, bg_g, bg_b] = self.palette.default_bg;
            color_table_data[257] = [
                Self::srgb_to_linear(bg_r as f32 / 255.0),
                Self::srgb_to_linear(bg_g as f32 / 255.0),
                Self::srgb_to_linear(bg_b as f32 / 255.0),
                1.0,
            ];
            self.queue.write_buffer(&self.color_table_buffer, 0, bytemuck::cast_slice(&color_table_data));
        }

        // ═══════════════════════════════════════════════════════════════════
        // PREPARE STATUSLINE FOR RENDERING (dedicated shader)
        // Must happen AFTER pane content rendering so sprite indices are correct
        // ═══════════════════════════════════════════════════════════════════
        let statusline_cols = {
            let statusline_y = self.statusline_y();
            
            // Update statusline GPU cells from content, passing window width for gap expansion
            let cols = self.update_statusline_cells(statusline_content, width);
            
            if cols > 0 {
                // Upload statusline cells to GPU
                self.queue.write_buffer(
                    &self.statusline_cell_buffer,
                    0,
                    bytemuck::cast_slice(&self.statusline_gpu_cells),
                );
                
                // Create params for statusline shader
                let statusline_params = StatuslineParams {
                    char_count: cols as u32,
                    cell_width: self.cell_width,
                    cell_height: self.cell_height,
                    screen_width: width,
                    screen_height: height,
                    y_offset: statusline_y,
                    _padding: [0.0, 0.0],
                };
                
                // Upload statusline params
                self.queue.write_buffer(
                    &self.statusline_params_buffer,
                    0,
                    bytemuck::cast_slice(&[statusline_params]),
                );
            }
            
            cols
        };
        
        // Upload terminal sprites (shared between all panes)
        // Must happen after all sprites have been created
        // Resize sprite buffer if needed
        if !self.sprite_info.is_empty() {
            let required_sprites = self.sprite_info.len();
            if required_sprites > self.sprite_buffer_capacity {
                // Need to resize - create a new larger buffer
                let new_capacity = (required_sprites * 3 / 2).max(self.sprite_buffer_capacity * 2);
                self.sprite_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Sprite Storage Buffer"),
                    size: (new_capacity * std::mem::size_of::<SpriteInfo>()) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.sprite_buffer_capacity = new_capacity;
                
                // Recreate all per-pane bind groups since they reference the sprite buffer
                let pane_ids: Vec<u64> = self.pane_resources.keys().cloned().collect();
                for pane_id in pane_ids {
                    if let Some(pane_res) = self.pane_resources.get(&pane_id) {
                        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some(&format!("Pane {} Bind Group", pane_id)),
                            layout: &self.instanced_bind_group_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: self.color_table_buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: pane_res.grid_params_buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 2,
                                    resource: pane_res.cell_buffer.as_entire_binding(),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 3,
                                    resource: self.sprite_buffer.as_entire_binding(),
                                },
                            ],
                        });
                        // Update the bind group in pane_resources
                        if let Some(pane_res_mut) = self.pane_resources.get_mut(&pane_id) {
                            pane_res_mut.bind_group = bind_group;
                        }
                    }
                }
            }
            
            self.queue.write_buffer(&self.sprite_buffer, 0, bytemuck::cast_slice(&self.sprite_info));
        }
        
        // Upload statusline sprites (separate buffer from terminal)
        if !self.statusline_sprite_info.is_empty() {
            let required_sprites = self.statusline_sprite_info.len();
            if required_sprites > self.statusline_sprite_buffer_capacity {
                // Need to resize - create a new larger buffer
                let new_capacity = (required_sprites * 3 / 2).max(self.statusline_sprite_buffer_capacity * 2);
                self.statusline_sprite_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Statusline Sprite Buffer"),
                    size: (new_capacity * std::mem::size_of::<SpriteInfo>()) as u64,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                self.statusline_sprite_buffer_capacity = new_capacity;
                
                // Recreate statusline bind group since it references the sprite buffer
                self.statusline_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Statusline Bind Group"),
                    layout: &self.statusline_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.color_table_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: self.statusline_params_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: self.statusline_cell_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: self.statusline_sprite_buffer.as_entire_binding(),
                        },
                    ],
                });
            }
            
            self.queue.write_buffer(&self.statusline_sprite_buffer, 0, bytemuck::cast_slice(&self.statusline_sprite_info));
        }

        // ═══════════════════════════════════════════════════════════════════
        // PREPARE IMAGE RENDERS (Kitty Graphics Protocol)
        // ═══════════════════════════════════════════════════════════════════
        let mut image_renders: Vec<(u32, ImageUniforms)> = Vec::new();
        for (terminal, info, _) in panes {
            // Apply grid centering offsets to pane position
            let pane_x = grid_x_offset + info.x;
            let pane_y = terminal_y_offset + grid_y_offset + info.y;

            let renders = self.prepare_image_renders(
                terminal.image_storage.placements(),
                pane_x,
                pane_y,
                self.cell_width,
                self.cell_height,
                width,
                height,
                terminal.scrollback.len(),
                terminal.scroll_offset,
                info.rows,
            );
            image_renders.extend(renders);
        }

        // ═══════════════════════════════════════════════════════════════════
        // PREPARE EDGE GLOW UNIFORMS (combined for all active glows)
        // ═══════════════════════════════════════════════════════════════════
        let edge_glow_uniforms = if !edge_glows.is_empty() && edge_glow_intensity > 0.0 {
            Some(self.prepare_edge_glow_uniforms(edge_glows, terminal_y_offset, edge_glow_intensity))
        } else {
            None
        };

        // ═══════════════════════════════════════════════════════════════════
        // SUBMIT TO GPU
        // ═══════════════════════════════════════════════════════════════════
        let bg_vertex_count = self.bg_vertices.len();
        let glyph_vertex_count = self.glyph_vertices.len();
        let total_vertex_count = bg_vertex_count + glyph_vertex_count;
        let total_index_count = self.bg_indices.len() + self.glyph_indices.len();

        // Resize buffers if needed
        if total_vertex_count > self.vertex_capacity {
            self.vertex_capacity = total_vertex_count * 2;
            self.vertex_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Glyph Vertex Buffer"),
                size: (self.vertex_capacity * std::mem::size_of::<GlyphVertex>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        if total_index_count > self.index_capacity {
            self.index_capacity = total_index_count * 2;
            self.index_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Glyph Index Buffer"),
                size: (self.index_capacity * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        // Upload vertices: bg, then glyph
        self.queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&self.bg_vertices));
        self.queue.write_buffer(
            &self.vertex_buffer,
            (bg_vertex_count * std::mem::size_of::<GlyphVertex>()) as u64,
            bytemuck::cast_slice(&self.glyph_vertices),
        );

        // Upload indices: bg, then glyph (adjusted)
        self.queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(&self.bg_indices));

        let glyph_vertex_offset = bg_vertex_count as u32;
        let bg_index_bytes = self.bg_indices.len() * std::mem::size_of::<u32>();

        if !self.glyph_indices.is_empty() {
            let adjusted_indices: Vec<u32> = self.glyph_indices.iter()
                .map(|i| i + glyph_vertex_offset)
                .collect();
            self.queue.write_buffer(
                &self.index_buffer,
                bg_index_bytes as u64,
                bytemuck::cast_slice(&adjusted_indices),
            );
        }

        // Upload quad params and instances for instanced quad rendering
        let quad_params = QuadParams {
            screen_width: width,
            screen_height: height,
            _padding: [0.0, 0.0],
        };
        self.queue.write_buffer(&self.quad_params_buffer, 0, bytemuck::cast_slice(&[quad_params]));
        
        // Upload quads if we have any
        if !self.quads.is_empty() {
            self.queue.write_buffer(&self.quad_buffer, 0, bytemuck::cast_slice(&self.quads));
        }
        
        // Upload overlay quads if we have any (will be rendered after main quads)
        // We reuse the same buffer, uploading overlay quads when needed during rendering

        if self.atlas_dirty {
            self.queue.write_texture(
                wgpu::ImageCopyTexture {
                    texture: &self.atlas_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &self.atlas_data,
                wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(ATLAS_SIZE * ATLAS_BPP),
                    rows_per_image: Some(ATLAS_SIZE),
                },
                wgpu::Extent3d {
                    width: ATLAS_SIZE,
                    height: ATLAS_SIZE,
                    depth_or_array_layers: 1,
                },
            );
            self.atlas_dirty = false;
        }

        // Create command encoder and render
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let [bg_r, bg_g, bg_b] = self.palette.default_bg;
            let bg_r_linear = Self::srgb_to_linear(bg_r as f32 / 255.0) as f64;
            let bg_g_linear = Self::srgb_to_linear(bg_g as f32 / 255.0) as f64;
            let bg_b_linear = Self::srgb_to_linear(bg_b as f32 / 255.0) as f64;
            let bg_alpha = self.background_opacity as f64;
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: bg_r_linear,
                            g: bg_g_linear,
                            b: bg_b_linear,
                            a: bg_alpha,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.glyph_pipeline);
            render_pass.set_bind_group(0, &self.glyph_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            
            // ═══════════════════════════════════════════════════════════════════
            // INSTANCED QUAD RENDERING (tab bar backgrounds, borders, etc.)
            // Rendered FIRST so backgrounds appear behind text
            // ═══════════════════════════════════════════════════════════════════
            if !self.quads.is_empty() {
                render_pass.set_pipeline(&self.quad_pipeline);
                render_pass.set_bind_group(0, &self.quad_bind_group, &[]);
                render_pass.draw(0..4, 0..self.quads.len() as u32);
            }
            
            // Draw bg + glyph indices (tab bar text uses legacy vertex rendering)
            // Rendered AFTER quads so text appears on top of backgrounds
            render_pass.set_pipeline(&self.glyph_pipeline);
            render_pass.set_bind_group(0, &self.glyph_bind_group, &[]);
            render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            render_pass.draw_indexed(0..total_index_count as u32, 0, 0..1);

            // ═══════════════════════════════════════════════════════════════════
            // INSTANCED CELL RENDERING (Like Kitty's per-window VAO approach)
            // Each pane has its own bind group with its own buffers.
            // Data was already uploaded before the render pass started.
            // ═══════════════════════════════════════════════════════════════════
            for pane_data in &pane_render_list {
                let instance_count = pane_data.cols * pane_data.rows;
                
                // Get this pane's bind group (data already uploaded)
                if let Some(pane_res) = self.pane_resources.get(&pane_data.pane_id) {
                    // Draw cell backgrounds
                    render_pass.set_pipeline(&self.cell_bg_pipeline);
                    render_pass.set_bind_group(0, &self.glyph_bind_group, &[]); // Atlas (shared)
                    render_pass.set_bind_group(1, &pane_res.bind_group, &[]); // This pane's data
                    render_pass.draw(0..4, 0..instance_count); // 4 vertices per quad, N instances
                    
                    // Draw cell glyphs
                    render_pass.set_pipeline(&self.cell_glyph_pipeline);
                    render_pass.set_bind_group(0, &self.glyph_bind_group, &[]); // Atlas (shared)
                    render_pass.set_bind_group(1, &pane_res.bind_group, &[]); // This pane's data
                    render_pass.draw(0..4, 0..instance_count); // 4 vertices per quad, N instances
                }
            }
            
            // ═══════════════════════════════════════════════════════════════════
            // STATUSLINE RENDERING (dedicated shader)
            // Render the statusline using its own pipelines
            // ═══════════════════════════════════════════════════════════════════
            if statusline_cols > 0 {
                let instance_count = statusline_cols as u32;
                
                // Draw statusline backgrounds
                render_pass.set_pipeline(&self.statusline_bg_pipeline);
                render_pass.set_bind_group(0, &self.glyph_bind_group, &[]); // Atlas
                render_pass.set_bind_group(1, &self.statusline_bind_group, &[]); // Statusline data
                render_pass.draw(0..4, 0..instance_count);
                
                // Draw statusline glyphs
                render_pass.set_pipeline(&self.statusline_glyph_pipeline);
                render_pass.set_bind_group(0, &self.glyph_bind_group, &[]); // Atlas
                render_pass.set_bind_group(1, &self.statusline_bind_group, &[]); // Statusline data
                render_pass.draw(0..4, 0..instance_count);
            }
            
            // ═══════════════════════════════════════════════════════════════════
            // ADD DIM OVERLAYS FOR INACTIVE PANES
            // ═══════════════════════════════════════════════════════════════════
            for pane_data in &pane_render_list {
                if let Some((x, y, w, h, color)) = pane_data.dim_overlay {
                    self.overlay_quads.push(Quad { x, y, width: w, height: h, color });
                }
            }
            
            // ═══════════════════════════════════════════════════════════════════
            // INSTANCED OVERLAY QUAD RENDERING (dimming overlays, borders)
            // Rendered last so overlays appear on top of everything
            // ═══════════════════════════════════════════════════════════════════
            if !self.overlay_quads.is_empty() {
                // Upload overlay quads to the SEPARATE overlay buffer to avoid overwriting tab bar quads
                self.queue.write_buffer(&self.overlay_quad_buffer, 0, bytemuck::cast_slice(&self.overlay_quads));
                render_pass.set_pipeline(&self.quad_pipeline);
                render_pass.set_bind_group(0, &self.overlay_quad_bind_group, &[]);
                render_pass.draw(0..4, 0..self.overlay_quads.len() as u32);
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // IMAGE PASS (Kitty Graphics Protocol images, after glyph rendering)
        // Each image is rendered with its own draw call using separate bind groups
        // ═══════════════════════════════════════════════════════════════════
        for (image_id, uniforms) in &image_renders {
            // Check if we have the GPU texture for this image
            if let Some(gpu_image) = self.image_textures.get(image_id) {
                // Upload uniforms to this image's dedicated uniform buffer
                self.queue.write_buffer(
                    &gpu_image.uniform_buffer,
                    0,
                    bytemuck::cast_slice(&[*uniforms]),
                );

                // Create a render pass for this image (load existing content)
                let mut image_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("Image Pass"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view: &view,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Load, // Preserve existing content
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                });

                image_pass.set_pipeline(&self.image_pipeline);
                image_pass.set_bind_group(0, &gpu_image.bind_group, &[]);
                image_pass.draw(0..4, 0..1); // Triangle strip quad
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // EDGE GLOW PASS (shader-based, after main rendering)
        // All active glows are rendered in a single pass via uniform array
        // ═══════════════════════════════════════════════════════════════════
        if let Some(uniforms) = &edge_glow_uniforms {
            // Upload uniforms
            self.queue.write_buffer(
                &self.edge_glow_uniform_buffer,
                0,
                bytemuck::cast_slice(&[*uniforms]),
            );

            // Render pass for this edge glow (load existing content)
            let mut glow_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Edge Glow Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Preserve existing content
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            glow_pass.set_pipeline(&self.edge_glow_pipeline);
            glow_pass.set_bind_group(0, &self.edge_glow_bind_group, &[]);
            glow_pass.draw(0..3, 0..1); // Fullscreen triangle
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    // ═══════════════════════════════════════════════════════════════════════════════
    // IMAGE RENDERING (Kitty Graphics Protocol)
    // ═══════════════════════════════════════════════════════════════════════════════

    /// Upload an image to the GPU, creating or updating its texture.
    pub fn upload_image(&mut self, image: &ImageData) {
        // Get current frame data (handles animation frames automatically)
        let data = image.current_frame_data();
        
        // Check if we already have this image
        if let Some(existing) = self.image_textures.get(&image.id) {
            if existing.width == image.width && existing.height == image.height {
                // Same dimensions, just update the data
                self.queue.write_texture(
                    wgpu::ImageCopyTexture {
                        texture: &existing.texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    data,
                    wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(image.width * 4),
                        rows_per_image: Some(image.height),
                    },
                    wgpu::Extent3d {
                        width: image.width,
                        height: image.height,
                        depth_or_array_layers: 1,
                    },
                );
                return;
            }
            // Different dimensions, need to recreate
        }

        // Create new texture
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("Image {}", image.id)),
            size: wgpu::Extent3d {
                width: image.width,
                height: image.height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });

        // Upload the data
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::ImageDataLayout {
                offset: 0,
                bytes_per_row: Some(image.width * 4),
                rows_per_image: Some(image.height),
            },
            wgpu::Extent3d {
                width: image.width,
                height: image.height,
                depth_or_array_layers: 1,
            },
        );

        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Create per-image uniform buffer
        let uniform_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Image {} Uniform Buffer", image.id)),
            size: std::mem::size_of::<ImageUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group for this image with its own uniform buffer
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Image {} Bind Group", image.id)),
            layout: &self.image_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.image_sampler),
                },
            ],
        });

        self.image_textures.insert(image.id, GpuImage {
            texture,
            view,
            uniform_buffer,
            bind_group,
            width: image.width,
            height: image.height,
        });

        log::debug!(
            "Uploaded image {} ({}x{}) to GPU",
            image.id,
            image.width,
            image.height
        );
    }

    /// Remove an image from the GPU.
    pub fn remove_image(&mut self, image_id: u32) {
        if self.image_textures.remove(&image_id).is_some() {
            log::debug!("Removed image {} from GPU", image_id);
        }
    }

    /// Sync images from terminal's image storage to GPU.
    /// Uploads new/changed images and removes deleted ones.
    /// Also updates animation frames.
    pub fn sync_images(&mut self, storage: &mut ImageStorage) {
        // Update animations and get list of changed image IDs
        let changed_ids = storage.update_animations();

        // Re-upload frames that changed due to animation
        for id in &changed_ids {
            if let Some(image) = storage.get_image(*id) {
                self.upload_image(image);
            }
        }

        if !storage.dirty && changed_ids.is_empty() {
            return;
        }

        // Upload all images (upload_image handles deduplication)
        for image in storage.images().values() {
            self.upload_image(image);
        }

        // Remove textures for deleted images
        let current_ids: std::collections::HashSet<u32> = storage.images().keys().copied().collect();
        let gpu_ids: Vec<u32> = self.image_textures.keys().copied().collect();
        for id in gpu_ids {
            if !current_ids.contains(&id) {
                self.remove_image(id);
            }
        }

        storage.clear_dirty();
    }

    /// Render images for a pane. Called from render_pane_content.
    /// Returns a Vec of (image_id, uniforms) for deferred rendering.
    fn prepare_image_renders(
        &self,
        placements: &[ImagePlacement],
        pane_x: f32,
        pane_y: f32,
        cell_width: f32,
        cell_height: f32,
        screen_width: f32,
        screen_height: f32,
        scrollback_len: usize,
        scroll_offset: usize,
        visible_rows: usize,
    ) -> Vec<(u32, ImageUniforms)> {
        let mut renders = Vec::new();

        for placement in placements {
            // Check if we have the GPU texture for this image
            let gpu_image = match self.image_textures.get(&placement.image_id) {
                Some(img) => img,
                None => continue, // Skip if not uploaded yet
            };

            // Convert absolute row to visible screen row
            // placement.row is absolute (scrollback_len_at_placement + cursor_row)
            // visible_row = absolute_row - scrollback_len + scroll_offset
            let absolute_row = placement.row as isize;
            let visible_row = absolute_row - scrollback_len as isize + scroll_offset as isize;

            // Check if image is visible on screen
            // Image spans from visible_row to visible_row + placement.rows
            let image_bottom = visible_row + placement.rows as isize;
            if image_bottom < 0 || visible_row >= visible_rows as isize {
                continue; // Image is completely off-screen
            }

            // Calculate display position in pixels
            let pos_x = pane_x + (placement.col as f32 * cell_width) + placement.x_offset as f32;
            let pos_y = pane_y + (visible_row as f32 * cell_height) + placement.y_offset as f32;

            log::debug!(
                "Image render: pane_x={} col={} cell_width={} x_offset={} => pos_x={}",
                pane_x, placement.col, cell_width, placement.x_offset, pos_x
            );

            // Calculate display size in pixels
            let display_width = placement.cols as f32 * cell_width;
            let display_height = placement.rows as f32 * cell_height;

            // Calculate source rectangle in normalized coordinates
            let src_x = placement.src_x as f32 / gpu_image.width as f32;
            let src_y = placement.src_y as f32 / gpu_image.height as f32;
            let src_width = if placement.src_width == 0 {
                1.0 - src_x
            } else {
                placement.src_width as f32 / gpu_image.width as f32
            };
            let src_height = if placement.src_height == 0 {
                1.0 - src_y
            } else {
                placement.src_height as f32 / gpu_image.height as f32
            };

            let uniforms = ImageUniforms {
                screen_width,
                screen_height,
                pos_x,
                pos_y,
                display_width,
                display_height,
                src_x,
                src_y,
                src_width,
                src_height,
                _padding1: 0.0,
                _padding2: 0.0,
            };

            renders.push((placement.image_id, uniforms));
        }

        renders
    }

}
