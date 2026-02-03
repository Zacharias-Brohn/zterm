//! GPU-accelerated terminal rendering using wgpu with a glyph atlas.
//! Uses rustybuzz (HarfBuzz port) for text shaping to support font features.

use crate::box_drawing::{is_box_drawing, render_box_char};
use crate::color::LinearPalette;
use crate::color_font::{find_color_font_for_char, ColorFontRenderer};
use crate::config::{Config, TabBarPosition};
use crate::font_loader::{find_font_for_char, load_font_family, FontVariant};
use crate::pane_resources::PaneGpuResources;
use crate::pipeline::PipelineBuilder;
use crate::gpu_types::{
    GlowInstance, GlyphVertex, GridParams, ImageUniforms,
    EdgeGlowUniforms, QuadParams, StatuslineParams,
    ATLAS_SIZE, MAX_ATLAS_LAYERS, ATLAS_BPP, MAX_EDGE_GLOWS,
    COLOR_TYPE_DEFAULT, COLOR_TYPE_INDEXED, COLOR_TYPE_RGB,
    ATTR_BOLD, ATTR_ITALIC, ATTR_STRIKE,
    COLORED_GLYPH_FLAG,
    CURSOR_SPRITE_BEAM, CURSOR_SPRITE_UNDERLINE, CURSOR_SPRITE_HOLLOW,
    DECORATION_SPRITE_STRIKETHROUGH, DECORATION_SPRITE_UNDERLINE, DECORATION_SPRITE_DOUBLE_UNDERLINE,
    DECORATION_SPRITE_UNDERCURL, DECORATION_SPRITE_DOTTED, DECORATION_SPRITE_DASHED,
    FIRST_GLYPH_SPRITE,
};
use crate::graphics::ImageStorage;
use crate::image_renderer::ImageRenderer;
use crate::terminal::{Color, ColorPalette, CursorShape, Direction, Terminal};
use ab_glyph::{Font, FontRef, GlyphId, ScaleFont};
use rustybuzz::UnicodeBuffer;
use ttf_parser::Tag;
use std::cell::{OnceCell, RefCell};
use std::collections::HashSet;
use std::num::NonZeroU32;
use rustc_hash::FxHashMap;
use std::path::PathBuf;
use std::sync::Arc;

// Fontconfig for dynamic font fallback
use fontconfig::Fontconfig;

// Re-export types for backwards compatibility
pub use crate::edge_glow::EdgeGlow;
pub use crate::statusline::{StatuslineColor, StatuslineComponent, StatuslineSection, StatuslineContent};
pub use crate::gpu_types::{FontCellMetrics, GPUCell, Quad, SpriteInfo};

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
    /// Atlas layer index (z-coordinate for texture array).
    layer: f32,
}

impl GlyphInfo {
    /// Empty glyph info (e.g., for space characters or failed rasterization).
    const EMPTY: Self = Self {
        uv: [0.0, 0.0, 0.0, 0.0],
        layer: 0.0,
        size: [0.0, 0.0],
        is_colored: false,
    };
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

/// Result of shaping a text sequence.
#[derive(Clone, Debug)]
struct ShapedGlyphs {
    /// Glyph IDs, advances, offsets, and cluster indices.
    /// Each tuple is (glyph_id, x_advance, x_offset, y_offset, cluster).
    /// x_offset/y_offset are for texture healing - they shift the glyph without affecting advance.
    glyphs: Vec<(u16, f32, f32, f32, u32)>,
}

impl From<GlyphInfo> for SpriteInfo {
    #[inline]
    fn from(info: GlyphInfo) -> Self {
        Self {
            uv: info.uv,
            layer: info.layer,
            _padding: 0.0,
            size: info.size,
        }
    }
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
/// Key for sprite lookup - optimized to avoid heap allocation.
/// Most sprites are single characters, multi-cell symbols use (char, cell_index).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct SpriteKey {
    /// The character (single char for most glyphs)
    ch: char,
    /// Cell index for multi-cell symbols (0 for single-cell)
    cell_index: u8,
    /// Font style (regular, bold, italic, bold-italic)
    style: FontStyle,
    /// Whether this is a colored glyph (emoji)
    colored: bool,
}

impl SpriteKey {
    /// Create a key for a single-cell sprite (the common case)
    /// Uses cell_index=255 as sentinel to distinguish from multi-cell cell 0
    #[inline]
    fn single(ch: char, style: FontStyle, colored: bool) -> Self {
        Self { ch, cell_index: 255, style, colored }
    }
    
    /// Create a key for a multi-cell sprite
    #[inline]
    fn multi(ch: char, cell_index: u8, style: FontStyle, colored: bool) -> Self {
        Self { ch, cell_index, style, colored }
    }
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
    /// Image renderer for Kitty graphics protocol.
    image_renderer: ImageRenderer,

    // Atlas textures - vector of separate 2D textures for O(1) layer addition
    // Unlike a texture_2d_array, adding a new layer just means creating a new texture
    // without copying existing data.
    atlas_textures: Vec<wgpu::Texture>,
    atlas_views: Vec<wgpu::TextureView>,
    /// Atlas sampler - stored for use when recreating bind group after layer addition
    atlas_sampler: wgpu::Sampler,
    /// Bind group layout for glyph rendering - needed to recreate bind group after texture changes
    glyph_bind_group_layout: wgpu::BindGroupLayout,
    /// Current layer being written to (index into atlas_textures)
    atlas_current_layer: u32,

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
    color_font_cache: FxHashMap<char, Option<PathBuf>>,
    shaping_ctx: ShapingContext,
    /// OpenType features for shaping (shared across all font variants)
    shaping_features: Vec<rustybuzz::Feature>,
    char_cache: FxHashMap<char, GlyphInfo>,    // cache char -> rendered glyph
    ligature_cache: FxHashMap<String, ShapedGlyphs>, // cache multi-char -> shaped glyphs
    /// Glyph cache keyed by (font_style, font_index, glyph_id)
    /// font_style is FontStyle as usize, font_index is 0 for primary, 1+ for fallbacks
    glyph_cache: FxHashMap<(usize, usize, u16), GlyphInfo>,
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
    /// Font cell metrics with integer dimensions (like Kitty).
    /// Using integers ensures pixel-perfect alignment and avoids floating-point precision issues.
    pub cell_metrics: FontCellMetrics,
    /// Window dimensions.
    pub width: u32,
    pub height: u32,
    /// Color palette for rendering (sRGB).
    palette: ColorPalette,
    /// Pre-computed linear palette for GPU (avoids repeated sRGB→linear conversions).
    linear_palette: LinearPalette,
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
    sprite_map: FxHashMap<SpriteKey, u32>,
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
    pane_resources: FxHashMap<u64, PaneGpuResources>,

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
    statusline_sprite_map: FxHashMap<SpriteKey, u32>,
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

impl Renderer {
    /// Creates a new renderer for the given window.
    pub async fn new(window: Arc<winit::window::Window>, config: &Config) -> Self {
        let size = window.inner_size();
        let scale_factor = window.scale_factor();

        // Calculate DPI from scale factor
        // Standard assumption: scale_factor 1.0 = 96 DPI (Windows/Linux default)
        // macOS uses 72 as base DPI, but winit normalizes this
        let dpi = 96.0 * scale_factor;

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
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
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Terminal Device"),
                // TEXTURE_BINDING_ARRAY is required for our Vec<Texture> atlas approach
                // which uses binding_array<texture_2d<f32>> in shaders.
                // SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING is required
                // because we index the texture array with a non-uniform value (layer from vertex data).
                required_features: wgpu::Features::TEXTURE_BINDING_ARRAY
                    | wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING,
                required_limits: wgpu::Limits {
                    // We need at least MAX_ATLAS_LAYERS (64) textures in our binding array
                    max_binding_array_elements_per_shader_stage: MAX_ATLAS_LAYERS,
                    ..wgpu::Limits::default()
                },
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                ..Default::default()
            })
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
                face: regular_variant.face().clone(), 
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
        // Like Kitty, use ceil() to ensure glyphs always fit in cells
        let m_glyph_id = primary_font.glyph_id('M');
        let cell_width = scaled_font.h_advance(m_glyph_id).ceil() as u32;

        // Use font line metrics for cell height
        // ab_glyph's height() = ascent - descent (where descent is negative)
        // Like Kitty, use ceil() to ensure glyphs always fit
        let cell_height = scaled_font.height().ceil() as u32;
        
        // Calculate baseline offset from top of cell.
        // The baseline is where the bottom of uppercase letters sit.
        // ascent is the distance from baseline to top of tallest glyph.
        let baseline = scaled_font.ascent().ceil() as u32;
        
        // Calculate underline position and thickness (like Kitty's freetype.c)
        // Use DPI-aware thickness calculation: thickness_pts * dpi / 72.0
        let underline_thickness = ((1.0 * dpi / 72.0).round() as u32).max(1).min(cell_height);
        // Underline position is typically just below the baseline
        // Kitty computes: ascender - underline_position from font metrics
        // Since we don't have direct access to OS/2 table, use baseline + small offset
        let underline_position = (baseline + underline_thickness).min(cell_height - 1);
        
        // Calculate strikethrough position and thickness (like Kitty)
        // Kitty: strikethrough_position = floor(baseline * 0.65) if not in font metrics
        let strikethrough_position = ((baseline as f32 * 0.65).floor() as u32).min(cell_height - 1);
        let strikethrough_thickness = underline_thickness; // Same as underline by default
        
        // Create FontCellMetrics struct (like Kitty)
        let cell_metrics = FontCellMetrics {
            cell_width,
            cell_height,
            baseline,
            underline_position,
            underline_thickness,
            strikethrough_position,
            strikethrough_thickness,
        };
        
        // Calculate the correct scale factor for converting font units to pixels.
        // This matches ab_glyph's calculation: scale / height_unscaled
        // where height_unscaled = ascent - descent (the font's natural line height).
        let font_units_to_px = font_size / primary_font.height_unscaled();

        // Create atlas as a Vec of separate 2D textures for O(1) layer addition.
        // Unlike a texture_2d_array, adding a new layer just means creating a new texture
        // without copying existing data. wgpu requires bind group arrays to have exactly
        // `count` textures, so we fill unused slots with 1x1 dummy textures.
        let mut atlas_textures: Vec<wgpu::Texture> = Vec::with_capacity(MAX_ATLAS_LAYERS as usize);
        let mut atlas_views: Vec<wgpu::TextureView> = Vec::with_capacity(MAX_ATLAS_LAYERS as usize);
        
        // Helper to create a real atlas layer (8192x8192)
        let create_atlas_layer = |device: &wgpu::Device| -> (wgpu::Texture, wgpu::TextureView) {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Glyph Atlas Layer"),
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
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        };
        
        // Helper to create a dummy texture (1x1) for unused slots
        let create_dummy_texture = |device: &wgpu::Device| -> (wgpu::Texture, wgpu::TextureView) {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("Dummy Atlas Texture"),
                size: wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
            (texture, view)
        };
        
        // First texture is real (layer 0)
        let (tex, view) = create_atlas_layer(&device);
        atlas_textures.push(tex);
        atlas_views.push(view);
        
        // Fill remaining slots with dummy textures
        for _ in 1..MAX_ATLAS_LAYERS {
            let (tex, view) = create_dummy_texture(&device);
            atlas_textures.push(tex);
            atlas_views.push(view);
        }
        
        let atlas_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        // Create bind group layout - use D2 with count for binding_array
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
                        count: Some(NonZeroU32::new(MAX_ATLAS_LAYERS).unwrap()),
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                        count: None,
                    },
                ],
            });

        // Create bind group with TextureViewArray
        let atlas_view_refs: Vec<&wgpu::TextureView> = atlas_views.iter().collect();
        let glyph_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Glyph Bind Group"),
            layout: &glyph_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureViewArray(&atlas_view_refs),
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
            immediate_size: 0,
        });

        let glyph_pipeline = PipelineBuilder::new(&device, &shader, &pipeline_layout, surface_config.format)
            .build_full(
                "Glyph Pipeline",
                "vs_main",
                "fs_main",
                wgpu::BlendState::ALPHA_BLENDING,
                wgpu::PrimitiveTopology::TriangleList,
                &[GlyphVertex::desc()],
            );

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
            immediate_size: 0,
        });

        // Create edge glow render pipeline
        let edge_glow_pipeline = PipelineBuilder::new(&device, &edge_glow_shader, &edge_glow_pipeline_layout, surface_config.format)
            .build_full(
                "Edge Glow Pipeline",
                "vs_main",
                "fs_main",
                wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING,
                wgpu::PrimitiveTopology::TriangleList,
                &[],
            );

        // ═══════════════════════════════════════════════════════════════════════════════
        // IMAGE PIPELINE SETUP (Kitty Graphics Protocol)
        // ═══════════════════════════════════════════════════════════════════════════════

        // Create image shader
        let image_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Image Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("image_shader.wgsl").into()),
        });

        // Create ImageRenderer (handles sampler and bind group layout)
        let image_renderer = ImageRenderer::new(&device);

        // Create pipeline layout for images
        let image_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Image Pipeline Layout"),
            bind_group_layouts: &[image_renderer.bind_group_layout()],
            immediate_size: 0,
        });

        // Create image render pipeline
        // Premultiplied alpha blending (shader outputs premultiplied)
        let image_blend = wgpu::BlendState {
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
        };
        let image_pipeline = PipelineBuilder::new(&device, &image_shader, &image_pipeline_layout, surface_config.format)
            .build("Image Pipeline", "vs_main", "fs_main", image_blend);

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
            immediate_size: 0,
        });
        
        // Statusline pipelines share shader and layout
        let statusline_builder = PipelineBuilder::new(&device, &statusline_shader, &statusline_pipeline_layout, surface_config.format);
        let statusline_bg_pipeline = statusline_builder.build(
            "Statusline Background Pipeline",
            "vs_statusline_bg",
            "fs_statusline",
            wgpu::BlendState::ALPHA_BLENDING,
        );
        let statusline_glyph_pipeline = statusline_builder.build(
            "Statusline Glyph Pipeline",
            "vs_statusline_glyph",
            "fs_statusline",
            wgpu::BlendState::ALPHA_BLENDING,
        );

        // Create pipeline layout for instanced cell rendering
        // Uses @group(0) for atlas texture/sampler and @group(1) for cell data
        let instanced_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Instanced Pipeline Layout"),
            bind_group_layouts: &[&glyph_bind_group_layout, &instanced_bind_group_layout],
            immediate_size: 0,
        });

        // Cell pipelines share shader and layout
        let cell_builder = PipelineBuilder::new(&device, &shader, &instanced_pipeline_layout, surface_config.format);
        let cell_bg_pipeline = cell_builder.build(
            "Cell Background Pipeline",
            "vs_cell_bg",
            "fs_cell",
            wgpu::BlendState::ALPHA_BLENDING,
        );
        let cell_glyph_pipeline = cell_builder.build(
            "Cell Glyph Pipeline",
            "vs_cell_glyph",
            "fs_cell",
            wgpu::BlendState::ALPHA_BLENDING,
        );

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
            immediate_size: 0,
        });
        
        // Quad pipeline
        let quad_pipeline = PipelineBuilder::new(&device, &quad_shader, &quad_pipeline_layout, surface_config.format)
            .build("Quad Pipeline", "vs_quad", "fs_quad", wgpu::BlendState::ALPHA_BLENDING);

        let mut renderer = Self {
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
            image_renderer,
            atlas_textures,
            atlas_views,
            atlas_sampler,
            glyph_bind_group_layout,
            atlas_current_layer: 0,
            font_data,
            primary_font,
            font_variants,
            fallback_fonts,
            fontconfig: OnceCell::new(),
            tried_font_paths,
            color_font_renderer: RefCell::new(None),
            color_font_cache: FxHashMap::default(),
            shaping_ctx,
            shaping_features,
            char_cache: FxHashMap::default(),
            ligature_cache: FxHashMap::default(),
            glyph_cache: FxHashMap::default(),
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
            cell_metrics,
            width: size.width,
            height: size.height,
            palette: ColorPalette::default(),
            linear_palette: LinearPalette::default(),
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
            sprite_map: FxHashMap::default(),
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
            pane_resources: FxHashMap::default(),
            // Statusline rendering (dedicated shader and pipeline)
            statusline_gpu_cells: Vec::with_capacity(statusline_max_cols),
            statusline_cell_buffer,
            statusline_max_cols,
            statusline_params_buffer,
            statusline_bind_group_layout,
            statusline_bind_group,
            statusline_bg_pipeline,
            statusline_glyph_pipeline,
            statusline_sprite_map: FxHashMap::default(),
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
        };
        
        // Create pre-rendered cursor sprites at fixed indices (like Kitty's send_prerendered_sprites)
        renderer.create_cursor_sprites();
        // Create pre-rendered decoration sprites (underline, undercurl, strikethrough, etc.)
        renderer.create_decoration_sprites();
        
        renderer
    }

    /// Returns the height of the tab bar in pixels (one cell height, or 0 if hidden).
    pub fn tab_bar_height(&self) -> f32 {
        match self.tab_bar_position {
            TabBarPosition::Hidden => 0.0,
            _ => self.cell_metrics.cell_height as f32,
        }
    }

    /// Returns the height of the statusline in pixels (one cell height).
    pub fn statusline_height(&self) -> f32 {
        self.cell_metrics.cell_height as f32
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
        let cols = (self.width as f32 / self.cell_metrics.cell_width as f32).floor() as usize;
        let rows = (available_height / self.cell_metrics.cell_height as f32).floor() as usize;
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
            cols as f32 * self.cell_metrics.cell_width as f32
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
            rows as f32 * self.cell_metrics.cell_height as f32
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
        let epsilon = (self.cell_metrics.cell_height.max(self.cell_metrics.cell_width)) as f32;
        
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
    /// This delegates to calculate_edge_glow_bounds as the logic is identical.
    /// Returns (screen_x, screen_y, width, height) for the overlay area.
    #[inline]
    pub fn calculate_dim_overlay_bounds(&self, pane_x: f32, pane_y: f32, pane_width: f32, pane_height: f32) -> (f32, f32, f32, f32) {
        self.calculate_edge_glow_bounds(pane_x, pane_y, pane_width, pane_height)
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
        let col = (grid_x / self.cell_metrics.cell_width as f32).floor() as usize;
        let row = (grid_y / self.cell_metrics.cell_height as f32).floor() as usize;

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

        let old_cell_width = self.cell_metrics.cell_width;
        let old_cell_height = self.cell_metrics.cell_height;

        self.scale_factor = new_scale;
        self.dpi = 96.0 * new_scale;
        
        // Font size in pixels, rounded for pixel-perfect rendering
        self.font_size = (self.base_font_size * new_scale as f32).round();

        // Recalculate cell dimensions using ab_glyph
        // Like Kitty, use ceil() to ensure glyphs always fit
        let scaled_font = self.primary_font.as_scaled(self.font_size);
        let m_glyph_id = self.primary_font.glyph_id('M');
        self.cell_metrics.cell_width = scaled_font.h_advance(m_glyph_id).ceil() as u32;
        self.cell_metrics.cell_height = scaled_font.height().ceil() as u32;
        
        // Update baseline - critical for correct glyph positioning!
        // Like Kitty, baseline is the font's ascent (distance from baseline to top of glyphs).
        self.cell_metrics.baseline = scaled_font.ascent().ceil() as u32;
        
        // Update underline/strikethrough metrics
        let underline_thickness = ((1.0 * self.dpi / 72.0).round() as u32).max(1).min(self.cell_metrics.cell_height);
        self.cell_metrics.underline_thickness = underline_thickness;
        self.cell_metrics.underline_position = (self.cell_metrics.baseline + underline_thickness).min(self.cell_metrics.cell_height - 1);
        self.cell_metrics.strikethrough_position = ((self.cell_metrics.baseline as f32 * 0.65).floor() as u32).min(self.cell_metrics.cell_height - 1);
        self.cell_metrics.strikethrough_thickness = underline_thickness;
        
        // Update the font units to pixels scale factor
        self.font_units_to_px = self.font_size / self.primary_font.height_unscaled();

        log::info!(
            "Scale factor changed to {}: font {}px -> {}px, cell: {}x{}, baseline: {}",
            new_scale, self.base_font_size, self.font_size, self.cell_metrics.cell_width, self.cell_metrics.cell_height, self.cell_metrics.baseline
        );

        // Reset atlas and all sprite/glyph caches (includes cursor sprite creation)
        self.reset_atlas();

        // Return true if cell dimensions changed
        self.cell_metrics.cell_width != old_cell_width
            || self.cell_metrics.cell_height != old_cell_height
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

        let old_cell_width = self.cell_metrics.cell_width;
        let old_cell_height = self.cell_metrics.cell_height;

        self.base_font_size = size;
        
        // Font size in pixels, rounded for pixel-perfect rendering
        self.font_size = (size * self.scale_factor as f32).round();

        // Recalculate cell dimensions using ab_glyph
        // Like Kitty, use ceil() to ensure glyphs always fit
        let scaled_font = self.primary_font.as_scaled(self.font_size);
        let m_glyph_id = self.primary_font.glyph_id('M');
        self.cell_metrics.cell_width = scaled_font.h_advance(m_glyph_id).ceil() as u32;
        self.cell_metrics.cell_height = scaled_font.height().ceil() as u32;
        
        // Update baseline - critical for correct glyph positioning!
        // Like Kitty, baseline is the font's ascent (distance from baseline to top of glyphs).
        self.cell_metrics.baseline = scaled_font.ascent().ceil() as u32;
        
        // Update underline/strikethrough metrics
        let underline_thickness = ((1.0 * self.dpi / 72.0).round() as u32).max(1).min(self.cell_metrics.cell_height);
        self.cell_metrics.underline_thickness = underline_thickness;
        self.cell_metrics.underline_position = (self.cell_metrics.baseline + underline_thickness).min(self.cell_metrics.cell_height - 1);
        self.cell_metrics.strikethrough_position = ((self.cell_metrics.baseline as f32 * 0.65).floor() as u32).min(self.cell_metrics.cell_height - 1);
        self.cell_metrics.strikethrough_thickness = underline_thickness;
        
        // Update the font units to pixels scale factor
        self.font_units_to_px = self.font_size / self.primary_font.height_unscaled();

        log::info!(
            "Font size changed to {}px -> {}px, cell: {}x{}, baseline: {}",
            size, self.font_size, self.cell_metrics.cell_width, self.cell_metrics.cell_height, self.cell_metrics.baseline
        );

        // Reset atlas and all sprite/glyph caches (includes cursor sprite creation)
        self.reset_atlas();

        // Return true if cell dimensions changed
        self.cell_metrics.cell_width != old_cell_width
            || self.cell_metrics.cell_height != old_cell_height
    }

    /// Reset the glyph atlas when font size or scale factor changes.
    /// This clears all cached glyphs (which are now invalid) and resets the atlas.
    /// NOTE: This should ONLY be called for font/scale changes, NOT when atlas is full
    /// (for that case, we add a new layer via add_atlas_layer()).
    fn reset_atlas(&mut self) {
        log::info!("Resetting glyph atlas (font/scale changed)");
        
        // Clear all glyph caches - they need to be re-rasterized at new size
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
        
        // Reset atlas cursor and go back to layer 0
        self.atlas_cursor_x = 0;
        self.atlas_cursor_y = 0;
        self.atlas_row_height = 0;
        self.atlas_current_layer = 0;
        
        // Create pre-rendered cursor sprites at fixed indices (like Kitty)
        self.create_cursor_sprites();
        // Create pre-rendered decoration sprites (underline, undercurl, strikethrough, etc.)
        self.create_decoration_sprites();
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
    /// underline_style: 0=none, 1=single, 2=double, 3=curly, 4=dotted, 5=dashed
    #[inline]
    fn pack_attrs(bold: bool, italic: bool, underline_style: u8, strikethrough: bool) -> u32 {
        let mut attrs = (underline_style as u32) & 0x7; // 3 bits for decoration type
        if bold { attrs |= ATTR_BOLD; }
        if italic { attrs |= ATTR_ITALIC; }
        if strikethrough { attrs |= ATTR_STRIKE; }
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
        let key = SpriteKey::single(c, style, false);
        
        if let Some(&idx) = sprite_map.get(&key) {
            // Check if it's a colored glyph
            let is_colored = (idx & COLORED_GLYPH_FLAG) != 0;
            return (idx, is_colored);
        }
        
        // Check for emoji with color key
        let color_key = SpriteKey::single(c, style, true);
        if let Some(&idx) = sprite_map.get(&color_key) {
            return (idx, true);
        }
        
        // Need to rasterize this glyph
        // For box-drawing and multi-cell symbols (PUA/dingbats), use rasterize_char
        // which has full font fallback and color font support.
        // Regular text uses HarfBuzz shaping for proper glyph selection.
        let glyph = if is_box_drawing(c) || Self::is_multicell_symbol(c) {
            // These don't need style variants or use rasterize_char for scaling/color
            self.rasterize_char(c)
        } else {
            // Shape the single character with HarfBuzz using the styled font
            // This gets us the correct glyph ID for the styled font variant
            let char_str = c.to_string();
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
            layer: glyph.layer,
            _padding: 0.0,
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
        
        // Mark as colored if glyph is colored (emoji rendered via color font)
        let final_idx = if glyph.is_colored {
            sprite_idx | COLORED_GLYPH_FLAG
        } else {
            sprite_idx
        };
        
        // Cache the mapping
        let cache_key = SpriteKey::single(c, style, glyph.is_colored);
        sprite_map.insert(cache_key, final_idx);
        
        (final_idx, glyph.is_colored)
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
        sprite_map: &FxHashMap<SpriteKey, u32>,
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
                    attrs: Self::pack_attrs(cell.bold, cell.italic, cell.underline_style, cell.strikethrough),
                };
                col += 1;
                continue;
            }
            
            // Get font style
            let style = FontStyle::from_flags(cell.bold, cell.italic);
            let c = cell.character;
            
            // Check for symbol+empty multi-cell pattern
            // Like Kitty, look for symbol character followed by empty cells
            if c != ' ' && c != '\0' && Self::is_multicell_symbol(c) && !is_box_drawing(c) {
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
                    
                    // Try to find multi-cell sprites - check non-colored first (more common), then colored
                    let first_key_normal = SpriteKey::multi(c, 0, style, false);
                    
                    let (first_sprite, is_colored) = if let Some(&sprite) = sprite_map.get(&first_key_normal) {
                        (Some(sprite), false)
                    } else {
                        let first_key_colored = SpriteKey::multi(c, 0, style, true);
                        if let Some(&sprite) = sprite_map.get(&first_key_colored) {
                            (Some(sprite), true)
                        } else {
                            (None, false)
                        }
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
                                let key = SpriteKey::multi(c, cell_idx as u8, style, is_colored);
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
                                attrs: Self::pack_attrs(cell.bold, cell.italic, cell.underline_style, cell.strikethrough),
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
                    let next_cell = &row[col + num_empty + 1];
                    let next_char = next_cell.character;
                    if next_char == ' ' || next_char == '\u{2002}' || next_char == '\0' {
                        num_empty += 1;
                    } else {
                        break;
                    }
                }
                
                if num_empty > 0 {
                    // Check if we have colored multi-cell sprites for this character
                    let first_key = SpriteKey::multi(c, 0, style, true);
                    
                    if let Some(&first_sprite) = sprite_map.get(&first_key) {
                        let total_cells = 1 + num_empty;
                        
                        for cell_idx in 0..total_cells {
                            if col + cell_idx >= cols {
                                break;
                            }
                            
                            let sprite_idx = if cell_idx == 0 {
                                first_sprite
                            } else {
                                let key = SpriteKey::multi(c, cell_idx as u8, style, true);
                                sprite_map.get(&key).copied().unwrap_or(0)
                            };
                            
                            let current_cell = &row[col + cell_idx];
                            gpu_row[col + cell_idx] = GPUCell {
                                fg: Self::pack_color(&cell.fg_color),
                                bg: Self::pack_color(&current_cell.bg_color),
                                decoration_fg: 0,
                                sprite_idx: sprite_idx | COLORED_GLYPH_FLAG,
                                attrs: Self::pack_attrs(cell.bold, cell.italic, cell.underline_style, cell.strikethrough),
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
                let key = SpriteKey::single(c, style, false);
                if let Some(&idx) = sprite_map.get(&key) {
                    idx
                } else {
                    let color_key = SpriteKey::single(c, style, true);
                    sprite_map.get(&color_key).copied().unwrap_or(0)
                }
            };
            
            gpu_row[col] = GPUCell {
                fg: Self::pack_color(&cell.fg_color),
                bg: Self::pack_color(&cell.bg_color),
                decoration_fg: 0,
                sprite_idx,
                attrs: Self::pack_attrs(cell.bold, cell.italic, cell.underline_style, cell.strikethrough),
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
        
        // TEMPORARY DEBUG: Force full rebuild every frame to test if dirty-line tracking is the issue
        // TODO: Remove this once the rendering bug is fixed
        self.cells_dirty = true;
        
        // Check if grid size changed - need full rebuild
        let size_changed = self.last_grid_size != (cols, rows);
        if size_changed {
            self.gpu_cells.resize(total_cells, GPUCell::default());
            self.last_grid_size = (cols, rows);
            self.cells_dirty = true;
        }
        
        // First pass: ensure all characters have sprites
        // This needs mutable access to self for sprite creation
        // Like Kitty's render_line(), detect PUA+space patterns for multi-cell rendering
        // OPTIMIZATION: Only process dirty lines or when full rebuild is needed
        // OPTIMIZATION: Use get_visible_row() to avoid Vec allocation
        for row_idx in 0..rows {
            // Skip clean lines (unless size changed, which sets cells_dirty)
            if !self.cells_dirty && !terminal.is_line_dirty(row_idx) {
                continue;
            }
            
            let Some(row) = terminal.get_visible_row(row_idx) else {
                continue;
            };
            
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
                if Self::is_multicell_symbol(c) && !is_box_drawing(c) {
                    // Get the glyph's natural width to determine desired cells
                    let glyph_width = self.get_glyph_width(c);
                    let desired_cells = (glyph_width / self.cell_metrics.cell_width as f32).ceil() as usize;
                    
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
                            let first_key = SpriteKey::multi(c, 0, style, false);
                            
                            if self.sprite_map.get(&first_key).is_none() {
                                // Need to rasterize
                                let cell_sprites = self.rasterize_pua_multicell(c, total_cells);
                                
                                // Store each cell's sprite with a unique key
                                for (cell_idx, glyph) in cell_sprites.into_iter().enumerate() {
                                    if glyph.size[0] > 0.0 && glyph.size[1] > 0.0 {
                                        let key = SpriteKey::multi(c, cell_idx as u8, style, false);
                                        
                                        // Create sprite info from glyph info
                                        let sprite = SpriteInfo {
                                            uv: glyph.uv,
                                            layer: glyph.layer,
                                            _padding: 0.0,
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
                let (sprite_idx, is_colored) = self.get_or_create_sprite(c, style);
                
                // DEBUG: Log colored glyph detection
                if is_colored {
                    log::debug!("EMOJI MULTICELL CHECK: col={} char=U+{:04X} '{}' sprite_idx={} is_colored={}",
                               col, c as u32, c, sprite_idx, is_colored);
                }
                
                // If this is a colored glyph (emoji) followed by empty cells, create multi-cell sprites
                if is_colored && sprite_idx != 0 {
                    // Count trailing empty cells for potential multi-cell emoji
                    let mut num_empty = 0;
                    const MAX_EXTRA_CELLS: usize = 1; // Emoji are typically 2 cells wide
                    
                    while col + num_empty + 1 < row.len() && num_empty < MAX_EXTRA_CELLS {
                        let next_cell = &row[col + num_empty + 1];
                        let next_char = next_cell.character;
                        log::debug!("  checking next cell at col={}: char=U+{:04X} '{}' wide_cont={}",
                                   col + num_empty + 1, next_char as u32, next_char, next_cell.wide_continuation);
                        if next_char == ' ' || next_char == '\u{2002}' || next_char == '\0' {
                            num_empty += 1;
                        } else {
                            break;
                        }
                    }
                    
                    log::debug!("  found {} trailing empty cells", num_empty);
                    
                    if num_empty > 0 {
                        let total_cells = 1 + num_empty;
                        log::debug!("  creating multi-cell sprites for {} cells", total_cells);
                        
                        // Check if we already have multi-cell sprites for this emoji
                        let first_key = SpriteKey::multi(c, 0, style, true);
                        
                        if self.sprite_map.get(&first_key).is_none() {
                            log::debug!("  rasterizing multi-cell emoji U+{:04X}", c as u32);
                            let cell_sprites = self.rasterize_emoji_multicell(c, total_cells);
                            log::debug!("  got {} cell sprites", cell_sprites.len());
                            
                            for (cell_idx, glyph) in cell_sprites.into_iter().enumerate() {
                                log::debug!("    cell {} sprite size: {:?}", cell_idx, glyph.size);
                                if glyph.size[0] > 0.0 && glyph.size[1] > 0.0 {
                                    let key = SpriteKey::multi(c, cell_idx as u8, style, true);
                                    
                                    let sprite = SpriteInfo {
                                        uv: glyph.uv,
                                        layer: glyph.layer,
                                        _padding: 0.0,
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
        
        // Second pass: convert cells to GPU format
        // OPTIMIZATION: Use get_visible_row() to avoid Vec allocation
        let mut any_updated = false;
        
        // DEBUG: Log grid dimensions and buffer state
        static DEBUG_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let frame_num = DEBUG_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if frame_num % 60 == 0 {  // Log every 60 frames (~1 second at 60fps)
            log::info!("DEBUG update_gpu_cells: cols={} rows={} total={} gpu_cells.len={} cells_dirty={}", 
                cols, rows, total_cells, self.gpu_cells.len(), self.cells_dirty);
        }
        
        // If we did a full reset or size changed, update all lines
        if self.cells_dirty {
            static ROW_DEBUG_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let row_frame = ROW_DEBUG_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            
            if row_frame % 60 == 0 {
                let first_col: String = (0..rows).filter_map(|r| {
                    terminal.get_visible_row(r).and_then(|row| {
                        row.first().map(|cell| {
                            let c = cell.character;
                            if c == '\0' { ' ' } else { c }
                        })
                    })
                }).collect();
                log::info!("DEBUG col0: \"{}\"", first_col);
            }
            
            for row_idx in 0..rows {
                if let Some(row) = terminal.get_visible_row(row_idx) {
                    let start = row_idx * cols;
                    let end = start + cols;
                    
                    if end > self.gpu_cells.len() {
                        log::error!("DEBUG BUG: row_idx={} start={} end={} but gpu_cells.len={}", 
                            row_idx, start, end, self.gpu_cells.len());
                        continue;
                    }
                    
                    Self::cells_to_gpu_row_static(row, &mut self.gpu_cells[start..end], cols, &self.sprite_map);
                }
            }
            self.cells_dirty = false;
            any_updated = true;
        } else {
            // Only update dirty lines - use is_line_dirty() which handles all 256 lines
            for row_idx in 0..rows {
                if terminal.is_line_dirty(row_idx) {
                    if let Some(row) = terminal.get_visible_row(row_idx) {
                        let start = row_idx * cols;
                        let end = start + cols;
                        Self::cells_to_gpu_row_static(row, &mut self.gpu_cells[start..end], cols, &self.sprite_map);
                        any_updated = true;
                    }
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
        let target_cols = if self.cell_metrics.cell_width > 0 {
            (target_width / self.cell_metrics.cell_width as f32).ceil() as usize
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
                        let attrs = Self::pack_attrs(*bold, false, 0, false);
                        
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
                        let attrs = Self::pack_attrs(bold, false, 0, false);
                        
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
                        let attrs = Self::pack_attrs(component.bold, false, 0, false);
                        
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
                                && !is_box_drawing(c)
                                && char_idx + 1 < chars.len() 
                                && chars[char_idx + 1] == ' ';
                            
                            if is_multicell_with_space {
                                // Render as 2-cell symbol
                                let multi_style = FontStyle::Regular;
                                
                                // Check if we already have multi-cell sprites
                                let first_key = SpriteKey::multi(c, 0, multi_style, false);
                                
                                if self.statusline_sprite_map.get(&first_key).is_none() {
                                    // Need to rasterize multi-cell sprites
                                    let cell_sprites = self.rasterize_pua_multicell(c, 2);
                                    
                                    for (cell_i, glyph) in cell_sprites.into_iter().enumerate() {
                                        if glyph.size[0] > 0.0 && glyph.size[1] > 0.0 {
                                            let key = SpriteKey::multi(c, cell_i as u8, multi_style, false);
                                            
                                            let sprite = SpriteInfo {
                                                uv: glyph.uv,
                                                layer: glyph.layer,
                                                _padding: 0.0,
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
                                    
                                    let key = SpriteKey::multi(c, cell_i as u8, multi_style, false);
                                    
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
            let glyph = glyph_id.with_scale(self.font_size);
            if let Some(outlined) = scaled.outline_glyph(glyph) {
                let bounds = outlined.px_bounds();
                let width = bounds.max.x - bounds.min.x;
                if width > 0.0 {
                    return width;
                }
            }
            return scaled.h_advance(glyph_id);
        }
        
        // Try fallback fonts
        for (_, fallback_font) in &self.fallback_fonts {
            let fb_glyph_id = fallback_font.glyph_id(c);
            if fb_glyph_id.0 != 0 {
                let scaled = fallback_font.as_scaled(self.font_size);
                let glyph = fb_glyph_id.with_scale(self.font_size);
                if let Some(outlined) = scaled.outline_glyph(glyph) {
                    let bounds = outlined.px_bounds();
                    let width = bounds.max.x - bounds.min.x;
                    if width > 0.0 {
                        return width;
                    }
                }
                return scaled.h_advance(fb_glyph_id);
            }
        }
        
        // Default to one cell width if glyph not found
        self.cell_metrics.cell_width as f32
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
        if is_box_drawing(c) {
            if let Some((bitmap, _supersampled)) = render_box_char(
                c,
                self.cell_metrics.cell_width as usize,
                self.cell_metrics.cell_height as usize,
                self.font_size,
                self.dpi,
            ) {
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
                    if let Some(path) = find_font_for_char(fc, c) {
                        // Load the font and rasterize with ab_glyph
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
                                   c as u32, self.font_size, self.cell_metrics.cell_width, self.cell_metrics.cell_height);
                        
                        renderer.render_color_glyph(
                            path, c, self.font_size, self.cell_metrics.cell_width, self.cell_metrics.cell_height
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
            self.char_cache.insert(c, GlyphInfo::EMPTY);
            return GlyphInfo::EMPTY;
        };

        if bitmap.is_empty() || glyph_width == 0 || glyph_height == 0 {
            // Empty glyph (e.g., space)
            self.char_cache.insert(c, GlyphInfo::EMPTY);
            return GlyphInfo::EMPTY;
        }

        // Check if this is an oversized symbol glyph that needs rescaling.
        // PUA glyphs (Nerd Fonts), dingbats, and other symbols that are wider than
        // one cell should be rescaled to fit when rendered standalone (not part of
        // a multi-cell group).
        let (final_bitmap, final_width, final_height, final_offset_x, final_offset_y) = 
            if Self::is_multicell_symbol(c) {
                let cell_w = self.cell_metrics.cell_width as f32;
                // Use just the glyph bitmap width for comparison, not offset_x + width
                // offset_x is the left bearing which can be negative
                let glyph_w = glyph_width as f32;
                
                log::debug!("Scaling check for U+{:04X}: glyph_width={}, cell_width={}, offset_x={:.1}", 
                           c as u32, glyph_width, self.cell_metrics.cell_width, offset_x);
                
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
        let cell_w = self.cell_metrics.cell_width as usize;
        let cell_h = self.cell_metrics.cell_height as usize;
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
            return vec![GlyphInfo::EMPTY; num_cells];
        };
        
        if bitmap.is_empty() || glyph_width == 0 || glyph_height == 0 {
            return vec![GlyphInfo::EMPTY; num_cells];
        }
        
        // Create a multi-cell canvas
        let mut canvas = vec![0u8; canvas_width * cell_h];
        
        // Position glyph at x=0 (left-aligned), like Kitty's model where
        // glyphs are positioned at origin without offset adjustments
        let dest_x = 0i32;
        
        // Calculate vertical position using baseline, same as single-cell rendering
        // dest_y = baseline - glyph_height - offset_y
        let dest_y = (self.cell_metrics.baseline as f32 - glyph_height as f32 - offset_y).round() as i32;
        
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
        let cell_w = self.cell_metrics.cell_width as usize;
        let cell_h = self.cell_metrics.cell_height as usize;
        let canvas_width = cell_w * num_cells;
        
        // Find a color font for this emoji (find_color_font_for_char handles fontconfig internally)
        let Some(font_path) = find_color_font_for_char(c) else {
            log::debug!("No color font found for emoji U+{:04X}", c as u32);
            return vec![GlyphInfo {
                uv: [0.0, 0.0, 0.0, 0.0],
                layer: 0.0,
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
                layer: 0.0,
                size: [0.0, 0.0],
                is_colored: true,
            }; num_cells];
        };
        
        if rgba.is_empty() || glyph_width == 0 || glyph_height == 0 {
            return vec![GlyphInfo {
                uv: [0.0, 0.0, 0.0, 0.0],
                layer: 0.0,
                size: [0.0, 0.0],
                is_colored: true,
            }; num_cells];
        }
        
        // Create a multi-cell RGBA canvas
        let mut canvas = vec![0u8; canvas_width * cell_h * 4];
        
        // Position the glyph - for color glyphs, offset_y is ascent (distance from baseline to TOP)
        let dest_x = offset_x.round() as i32;
        let dest_y = (self.cell_metrics.baseline as f32 - offset_y).round() as i32;
        
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
        let cell_w = self.cell_metrics.cell_width as usize;
        let cell_h = self.cell_metrics.cell_height as usize;
        let mut canvas = vec![0u8; cell_w * cell_h];
        
        // Calculate destination position in the cell canvas.
        // baseline is the Y position where the baseline sits (from top of cell).
        // offset_y is the distance from baseline to glyph bottom.
        // glyph_top = baseline - (glyph_height + offset_y)
        //           = baseline - glyph_height - offset_y
        // Since offset_y can be negative (for descenders), this works correctly.
        let dest_x = offset_x.round() as i32;
        let dest_y = (self.cell_metrics.baseline as f32 - glyph_height as f32 - offset_y).round() as i32;
        
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
        let cell_w = self.cell_metrics.cell_width as usize;
        let cell_h = self.cell_metrics.cell_height as usize;
        let mut canvas = vec![0u8; cell_w * cell_h * 4]; // RGBA
        
        // For color glyphs, offset_y is the ascent (distance from baseline to TOP of glyph)
        // So dest_y = baseline - offset_y positions the top of the glyph correctly
        let dest_x = offset_x.round() as i32;
        let dest_y = (self.cell_metrics.baseline as f32 - offset_y).round() as i32;
        
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
    /// Like Kitty's send_sprite_to_gpu(), uploads immediately to the GPU texture
    /// using write_texture with only the cell-sized region (not the full layer).
    fn upload_cell_canvas_to_atlas(&mut self, canvas: &[u8], is_colored: bool) -> GlyphInfo {
        let cell_w = self.cell_metrics.cell_width;
        let cell_h = self.cell_metrics.cell_height;
        
        // Check if we need to move to next row
        if self.atlas_cursor_x + cell_w > ATLAS_SIZE {
            self.atlas_cursor_x = 0;
            self.atlas_cursor_y += self.atlas_row_height + 1;
            self.atlas_row_height = 0;
        }
        
        // Check if current layer is full - add a new layer (like Kitty)
        if self.atlas_cursor_y + cell_h > ATLAS_SIZE {
            self.add_atlas_layer();
            self.atlas_cursor_x = 0;
            self.atlas_cursor_y = 0;
            self.atlas_row_height = 0;
        }
        
        let layer = self.atlas_current_layer;
        
        // Prepare the sprite data in RGBA format (cell_w * cell_h * 4 bytes)
        // This is a small buffer that will be uploaded directly to the GPU
        let sprite_size = (cell_w * cell_h * ATLAS_BPP) as usize;
        let mut sprite_data = vec![0u8; sprite_size];
        
        if is_colored {
            // RGBA canvas - copy directly
            for y in 0..cell_h as usize {
                for x in 0..cell_w as usize {
                    let src_idx = (y * cell_w as usize + x) * 4;
                    let dst_idx = (y * cell_w as usize + x) * 4;
                    if src_idx + 3 < canvas.len() && dst_idx + 3 < sprite_data.len() {
                        sprite_data[dst_idx] = canvas[src_idx];
                        sprite_data[dst_idx + 1] = canvas[src_idx + 1];
                        sprite_data[dst_idx + 2] = canvas[src_idx + 2];
                        sprite_data[dst_idx + 3] = canvas[src_idx + 3];
                    }
                }
            }
        } else {
            // Grayscale canvas - convert to RGBA (white with alpha)
            for y in 0..cell_h as usize {
                for x in 0..cell_w as usize {
                    let src_idx = y * cell_w as usize + x;
                    let dst_idx = (y * cell_w as usize + x) * 4;
                    if src_idx < canvas.len() && dst_idx + 3 < sprite_data.len() {
                        sprite_data[dst_idx] = 255;     // R
                        sprite_data[dst_idx + 1] = 255; // G
                        sprite_data[dst_idx + 2] = 255; // B
                        sprite_data[dst_idx + 3] = canvas[src_idx]; // A
                    }
                }
            }
        }
        
        // Upload immediately to GPU - like Kitty's glTexSubImage3D call
        // This uploads only the cell-sized region, not the full 8192x8192 layer
        // With Vec<Texture>, we select the texture by layer index and always use z=0
        self.queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.atlas_textures[layer as usize],
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: self.atlas_cursor_x,
                    y: self.atlas_cursor_y,
                    z: 0, // Always 0 - layer is selected by texture index
                },
                aspect: wgpu::TextureAspect::All,
            },
            &sprite_data,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(cell_w * ATLAS_BPP),
                rows_per_image: Some(cell_h),
            },
            wgpu::Extent3d {
                width: cell_w,
                height: cell_h,
                depth_or_array_layers: 1,
            },
        );
        
        // Calculate UV coordinates
        let uv_x = self.atlas_cursor_x as f32 / ATLAS_SIZE as f32;
        let uv_y = self.atlas_cursor_y as f32 / ATLAS_SIZE as f32;
        let uv_w = cell_w as f32 / ATLAS_SIZE as f32;
        let uv_h = cell_h as f32 / ATLAS_SIZE as f32;
        let layer_f = layer as f32;
        
        // Update atlas cursor
        self.atlas_cursor_x += cell_w + 1;
        self.atlas_row_height = self.atlas_row_height.max(cell_h);
        
        GlyphInfo {
            uv: [uv_x, uv_y, uv_w, uv_h],
            size: [cell_w as f32, cell_h as f32],
            is_colored,
            layer: layer_f,
        }
    }
    
    /// Add a new layer to the atlas (like Kitty's realloc_sprite_texture).
    /// This switches to the next layer, creating the real texture if needed.
    fn add_atlas_layer(&mut self) {
        let new_layer = self.atlas_current_layer + 1;
        
        if new_layer >= MAX_ATLAS_LAYERS {
            log::error!("Atlas layer limit reached ({} layers), cannot add more", MAX_ATLAS_LAYERS);
            return;
        }
        
        log::info!("Adding atlas layer {} (was on layer {})", new_layer, self.atlas_current_layer);
        
        // Create real texture for the new layer (replacing the dummy)
        self.ensure_atlas_layer_capacity(new_layer);
        
        // Now switch to the new layer
        self.atlas_current_layer = new_layer;
    }
    
    /// Ensure the atlas has a real texture at the given layer index.
    /// With our Vec<Texture> approach, this just replaces the dummy texture at that index
    /// with a real one. No copying of existing data is needed - O(1) operation.
    /// 
    /// We track which layers are "real" vs "dummy" by checking atlas_current_layer.
    /// Layers 0..=atlas_current_layer are real, layers above are dummies.
    fn ensure_atlas_layer_capacity(&mut self, target_layer: u32) {
        // Layer 0 is always real (created at init), and all layers up to
        // atlas_current_layer are real. Only create if target is beyond current.
        if target_layer <= self.atlas_current_layer {
            return;
        }
        
        if target_layer >= MAX_ATLAS_LAYERS {
            log::error!("Atlas layer limit reached: {} >= {}", target_layer, MAX_ATLAS_LAYERS);
            return;
        }
        
        log::info!("Adding atlas layer {} (replacing dummy texture)", target_layer);
        
        // Create new real texture (8192x8192)
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Glyph Atlas Layer"),
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
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        // Replace dummy texture at this index with real texture
        self.atlas_textures[target_layer as usize] = texture;
        self.atlas_views[target_layer as usize] = view;
        
        // Recreate bind group with updated views (cheap - just metadata)
        self.glyph_bind_group = self.create_atlas_bind_group();
    }
    
    /// Create the glyph bind group with all atlas texture views.
    /// Called during initialization and when adding new atlas layers.
    fn create_atlas_bind_group(&self) -> wgpu::BindGroup {
        let view_refs: Vec<&wgpu::TextureView> = self.atlas_views.iter().collect();
        
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Glyph Bind Group"),
            layout: &self.glyph_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureViewArray(&view_refs),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.atlas_sampler),
                },
            ],
        })
    }

    /// Create pre-rendered cursor sprites in the atlas (like Kitty's send_prerendered_sprites).
    /// This creates sprites at fixed indices for beam, underline, and hollow cursors.
    /// Must be called after sprite_info is initialized with index 0 reserved.
    fn create_cursor_sprites(&mut self) {
        let cell_w = self.cell_metrics.cell_width as usize;
        let cell_h = self.cell_metrics.cell_height as usize;
        let cell_area = cell_w * cell_h;
        
        // Calculate DPI-aware cursor thicknesses (Kitty-style: thickness_pts * dpi / 72.0)
        let beam_thickness = (1.5 * self.dpi / 72.0)
            .round()
            .max(1.0)
            .min(cell_w as f64) as usize;
        let underline_thickness = (2.0 * self.dpi / 72.0)
            .round()
            .max(1.0)
            .min(cell_h as f64) as usize;
        let hollow_thickness = (1.0 * self.dpi / 72.0)
            .round()
            .max(1.0)
            .min(cell_w.min(cell_h) as f64) as usize;
        
        // Create grayscale canvas for each cursor type
        let mut canvas = vec![0u8; cell_area];
        
        // === Beam cursor (vertical bar on left edge) ===
        // Like Kitty's add_beam_cursor / vert() function
        canvas.fill(0);
        for y in 0..cell_h {
            for x in 0..beam_thickness {
                canvas[y * cell_w + x] = 255;
            }
        }
        let beam_info = self.upload_cell_canvas_to_atlas(&canvas, false);
        let beam_sprite = SpriteInfo::from(beam_info);
        
        // === Underline cursor (horizontal bar at bottom) ===
        // Like Kitty's add_underline_cursor / horz() function
        canvas.fill(0);
        let underline_top = cell_h.saturating_sub(underline_thickness);
        for y in underline_top..cell_h {
            for x in 0..cell_w {
                canvas[y * cell_w + x] = 255;
            }
        }
        let underline_info = self.upload_cell_canvas_to_atlas(&canvas, false);
        let underline_sprite = SpriteInfo::from(underline_info);
        
        // === Hollow cursor (rectangle outline) ===
        // Like Kitty's add_hollow_cursor function
        canvas.fill(0);
        // Top edge
        for y in 0..hollow_thickness {
            for x in 0..cell_w {
                canvas[y * cell_w + x] = 255;
            }
        }
        // Bottom edge
        for y in cell_h.saturating_sub(hollow_thickness)..cell_h {
            for x in 0..cell_w {
                canvas[y * cell_w + x] = 255;
            }
        }
        // Left edge
        for y in 0..cell_h {
            for x in 0..hollow_thickness {
                canvas[y * cell_w + x] = 255;
            }
        }
        // Right edge
        for y in 0..cell_h {
            for x in cell_w.saturating_sub(hollow_thickness)..cell_w {
                canvas[y * cell_w + x] = 255;
            }
        }
        let hollow_info = self.upload_cell_canvas_to_atlas(&canvas, false);
        let hollow_sprite = SpriteInfo::from(hollow_info);
        
        // Store sprites at their fixed indices
        // sprite_info[0] = no glyph (already set)
        // sprite_info[1] = beam cursor (CURSOR_SPRITE_BEAM)
        // sprite_info[2] = underline cursor (CURSOR_SPRITE_UNDERLINE)
        // sprite_info[3] = hollow cursor (CURSOR_SPRITE_HOLLOW)
        while self.sprite_info.len() < FIRST_GLYPH_SPRITE as usize {
            self.sprite_info.push(SpriteInfo::default());
        }
        self.sprite_info[CURSOR_SPRITE_BEAM as usize] = beam_sprite;
        self.sprite_info[CURSOR_SPRITE_UNDERLINE as usize] = underline_sprite;
        self.sprite_info[CURSOR_SPRITE_HOLLOW as usize] = hollow_sprite;
        self.next_sprite_idx = FIRST_GLYPH_SPRITE;
        
        log::debug!(
            "Created cursor sprites: beam={}px wide, underline={}px tall, hollow={}px border",
            beam_thickness, underline_thickness, hollow_thickness
        );
    }

    /// Create pre-rendered decoration sprites in the atlas (like Kitty's decorations.c).
    /// This creates sprites for strikethrough, underline, undercurl, dotted, dashed, and double underline.
    /// Must be called after create_cursor_sprites().
    fn create_decoration_sprites(&mut self) {
        let cell_w = self.cell_metrics.cell_width as usize;
        let cell_h = self.cell_metrics.cell_height as usize;
        let cell_area = cell_w * cell_h;
        
        let underline_pos = self.cell_metrics.underline_position as usize;
        let underline_thick = self.cell_metrics.underline_thickness as usize;
        let strike_pos = self.cell_metrics.strikethrough_position as usize;
        let strike_thick = self.cell_metrics.strikethrough_thickness as usize;
        
        // Helper: draw horizontal line at y_start for 'thickness' rows
        let draw_hline = |canvas: &mut [u8], y_start: usize, thickness: usize| {
            for y in y_start..(y_start + thickness).min(cell_h) {
                for x in 0..cell_w {
                    canvas[y * cell_w + x] = 255;
                }
            }
        };
        
        // Create canvas for decorations
        let mut canvas = vec![0u8; cell_area];
        
        // === Strikethrough (like Kitty's add_strikethrough) ===
        canvas.fill(0);
        let strike_half = strike_thick / 2;
        let strike_top = if strike_half > strike_pos { 0 } else { strike_pos - strike_half };
        draw_hline(&mut canvas, strike_top, strike_thick);
        let strike_info = self.upload_cell_canvas_to_atlas(&canvas, false);
        let strike_sprite = SpriteInfo::from(strike_info);
        
        // === Single Underline (like Kitty's add_straight_underline) ===
        canvas.fill(0);
        let under_half = underline_thick / 2;
        let under_top = if under_half > underline_pos { 0 } else { underline_pos - under_half };
        draw_hline(&mut canvas, under_top, underline_thick);
        let underline_info = self.upload_cell_canvas_to_atlas(&canvas, false);
        let underline_sprite = SpriteInfo::from(underline_info);
        
        // === Double Underline (like Kitty's add_double_underline) ===
        canvas.fill(0);
        // Two lines: one at underline_pos - thickness, one at underline_pos
        let a = underline_pos.saturating_sub(underline_thick);
        let b = underline_pos.min(cell_h - 1);
        let (top, bottom) = if a <= b { (a, b) } else { (b, a) };
        // Ensure at least 2 pixels gap between lines
        let (top, bottom) = if bottom.saturating_sub(top) < 2 {
            let bottom = (bottom + 1).min(cell_h - 1);
            let top = if bottom >= 2 { top } else { top.saturating_sub(1) };
            (top, bottom)
        } else {
            (top, bottom)
        };
        // Draw single-pixel lines at top and bottom
        if top < cell_h {
            for x in 0..cell_w { canvas[top * cell_w + x] = 255; }
        }
        if bottom < cell_h && bottom != top {
            for x in 0..cell_w { canvas[bottom * cell_w + x] = 255; }
        }
        let double_info = self.upload_cell_canvas_to_atlas(&canvas, false);
        let double_sprite = SpriteInfo::from(double_info);
        
        // === Undercurl (like Kitty's add_curl_underline with Wu antialiasing) ===
        // This follows Kitty's decorations.c add_curl_underline() exactly
        canvas.fill(0);
        
        let max_x = cell_w.saturating_sub(1);
        let max_y = cell_h.saturating_sub(1);
        
        // Wave factor: 2*PI for one full wave per cell (like Kitty's default undercurl_style)
        let xfactor = 2.0 * std::f64::consts::PI / max_x as f64;
        
        // Calculate position and thickness like Kitty does
        let d_quot = underline_thick / 2;
        let d_rem = underline_thick % 2;
        let position = underline_pos.min(cell_h.saturating_sub(d_quot + d_rem));
        let thickness = underline_thick.max(1).min(cell_h.saturating_sub(position + 1));
        
        // max_height is the descender space from the font
        let max_height = cell_h.saturating_sub(position.saturating_sub(thickness / 2));
        // half_height is the wave amplitude (1/4 of available space so it's not too large)
        let half_height = (max_height / 4).max(1);
        
        // Adjust thickness like Kitty: reduce slightly for thinner appearance
        // Note: thickness CAN become 0, which means only antialiased edges are drawn (1px line)
        let thickness = if thickness < 3 {
            thickness.saturating_sub(1)  // Can become 0 for thin 1px line
        } else {
            thickness.saturating_sub(2)
        };
        
        // Center the wave vertically in the underline area
        let position = position + half_height * 2;
        let position = if position + half_height > max_y {
            max_y.saturating_sub(half_height)
        } else {
            position
        };
        
        // Helper to add intensity at a position (like Kitty's add_intensity)
        let add_intensity = |canvas: &mut [u8], x: usize, y: i32, val: u8, position: usize| {
            let y = (y + position as i32).clamp(0, max_y as i32) as usize;
            if y < cell_h && x < cell_w {
                let idx = y * cell_w + x;
                canvas[idx] = canvas[idx].saturating_add(val);
            }
        };
        
        // Draw antialiased cosine wave using Wu algorithm (like Kitty)
        // Cosine waves always have slope <= 1 so are never steep
        for x in 0..cell_w {
            let y = (half_height as f64) * (x as f64 * xfactor).cos();
            let y1 = (y - thickness as f64).floor() as i32;  // upper bound
            let y2 = y.ceil() as i32;  // lower bound
            
            // Wu antialiasing intensity based on fractional part
            let frac = (y - y.floor()).abs();
            let intensity = (255.0 * frac) as u8;
            let i1 = 255u8.saturating_sub(intensity);  // upper edge intensity
            let i2 = intensity;  // lower edge intensity
            
            // Draw antialiased upper bound
            add_intensity(&mut canvas, x, y1, i1, position);
            
            // Draw antialiased lower bound  
            add_intensity(&mut canvas, x, y2, i2, position);
            
            // Fill between upper and lower bound with full intensity
            for t in 1..=thickness {
                add_intensity(&mut canvas, x, y1 + t as i32, 255, position);
            }
        }
        let curl_info = self.upload_cell_canvas_to_atlas(&canvas, false);
        let curl_sprite = SpriteInfo::from(curl_info);
        
        // === Dotted Underline (like Kitty's add_dotted_underline) ===
        canvas.fill(0);
        let num_dots = (cell_w / (2 * underline_thick.max(1))).max(1);
        let dot_size = (cell_w / (2 * num_dots)).max(1);
        
        // Distribute dots evenly
        for y in under_top..(under_top + underline_thick).min(cell_h) {
            let mut x = dot_size / 2; // Start with half gap
            for _ in 0..num_dots {
                for dx in 0..dot_size {
                    if x + dx < cell_w {
                        canvas[y * cell_w + x + dx] = 255;
                    }
                }
                x += dot_size * 2; // Dot + gap
            }
        }
        let dotted_info = self.upload_cell_canvas_to_atlas(&canvas, false);
        let dotted_sprite = SpriteInfo::from(dotted_info);
        
        // === Dashed Underline (like Kitty's add_dashed_underline) ===
        canvas.fill(0);
        let quarter_width = cell_w / 4;
        let dash_width = cell_w.saturating_sub(3 * quarter_width);
        let second_dash_start = 3 * quarter_width;
        
        for y in under_top..(under_top + underline_thick).min(cell_h) {
            // First dash at start
            for x in 0..dash_width {
                if x < cell_w {
                    canvas[y * cell_w + x] = 255;
                }
            }
            // Second dash
            for x in second_dash_start..(second_dash_start + dash_width).min(cell_w) {
                canvas[y * cell_w + x] = 255;
            }
        }
        let dashed_info = self.upload_cell_canvas_to_atlas(&canvas, false);
        let dashed_sprite = SpriteInfo::from(dashed_info);
        
        // Store sprites at their fixed indices
        // Ensure sprite_info has enough capacity
        while self.sprite_info.len() < FIRST_GLYPH_SPRITE as usize {
            self.sprite_info.push(SpriteInfo::default());
        }
        self.sprite_info[DECORATION_SPRITE_STRIKETHROUGH as usize] = strike_sprite;
        self.sprite_info[DECORATION_SPRITE_UNDERLINE as usize] = underline_sprite;
        self.sprite_info[DECORATION_SPRITE_DOUBLE_UNDERLINE as usize] = double_sprite;
        self.sprite_info[DECORATION_SPRITE_UNDERCURL as usize] = curl_sprite;
        self.sprite_info[DECORATION_SPRITE_DOTTED as usize] = dotted_sprite;
        self.sprite_info[DECORATION_SPRITE_DASHED as usize] = dashed_sprite;
        self.next_sprite_idx = FIRST_GLYPH_SPRITE;
        
        log::debug!(
            "Created decoration sprites: underline at y={}, strikethrough at y={}, thickness={}px",
            underline_pos, strike_pos, underline_thick
        );
    }

    /// Get or rasterize a glyph by its glyph ID from the primary font.
    /// Used for ligatures where we have the glyph ID from rustybuzz.
    /// Delegates to get_glyph_by_id_with_style with Regular style.
    #[allow(dead_code)]
    #[inline]
    fn get_glyph_by_id(&mut self, glyph_id: u16) -> GlyphInfo {
        self.get_glyph_by_id_with_style(glyph_id, FontStyle::Regular)
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
            variant.clone_font()
        } else {
            // Fall back to regular font if variant not available
            self.primary_font.clone()
        };

        // Rasterize the glyph by ID using ab_glyph
        let ab_glyph_id = GlyphId(glyph_id);
        let raster_result = self.rasterize_glyph_ab(&font, ab_glyph_id);

        let Some((glyph_width, glyph_height, bitmap, offset_x, offset_y)) = raster_result else {
            // Empty glyph (e.g., space)
            self.glyph_cache.insert(cache_key, GlyphInfo::EMPTY);
            return GlyphInfo::EMPTY;
        };

        if bitmap.is_empty() || glyph_width == 0 || glyph_height == 0 {
            // Empty glyph (e.g., space)
            self.glyph_cache.insert(cache_key, GlyphInfo::EMPTY);
            return GlyphInfo::EMPTY;
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
    /// Delegates to shape_text_with_style with Regular style.
    #[allow(dead_code)]
    #[inline]
    fn shape_text(&mut self, text: &str) -> ShapedGlyphs {
        self.shape_text_with_style(text, FontStyle::Regular)
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
            variant.face()
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
        // Use pre-computed linear palette
        let [color_r, color_g, color_b, _] = self.linear_palette.color_table[4];

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
        #[cfg(feature = "render_timing")]
        let frame_start = std::time::Instant::now();
        
        // Sync palette from first terminal (update both sRGB and linear versions)
        if let Some((terminal, _, _)) = panes.first() {
            self.palette = terminal.palette.clone();
            self.linear_palette = LinearPalette::from_palette(&self.palette);
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

        // NOTE: With Kitty-style multi-layer atlas, we no longer reset when full.
        // Instead, add_atlas_layer() is called when the current layer fills up.

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
            // Pre-computed linear RGB value for srgb_to_linear(26/255) ≈ 0.00972
            const TAB_BAR_BG_LINEAR: f32 = 0.00972;
            let tab_bar_bg = [TAB_BAR_BG_LINEAR, TAB_BAR_BG_LINEAR, TAB_BAR_BG_LINEAR, 1.0];

            // Draw tab bar background
            log::debug!("render_panes: drawing tab bar at y={}, height={}, num_tabs={}, quads_before={}", 
                tab_bar_y, tab_bar_height, num_tabs, self.quads.len());
            self.render_rect(0.0, tab_bar_y, width, tab_bar_height, tab_bar_bg);
            log::debug!("render_panes: after tab bar rect, quads_count={}", self.quads.len());

            // Render each tab
            let mut tab_x = 4.0_f32;
            let tab_padding = 8.0_f32;
            let min_tab_width = self.cell_metrics.cell_width as f32 * 8.0;

            for idx in 0..num_tabs {
                let is_active = idx == active_tab;
                let title = format!(" {} ", idx + 1);
                let title_width = title.chars().count() as f32 * self.cell_metrics.cell_width as f32;
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
                    let [r, g, b, _] = self.linear_palette.color_table[256]; // default_fg
                    let alpha = if is_active { 1.0 } else { 0.6 };
                    [r, g, b, alpha]
                };

                // Draw tab background
                self.render_rect(tab_x, tab_bar_y + 2.0, tab_width, tab_bar_height - 4.0, tab_bg);

                // Render tab title text
                let text_y = tab_bar_y + (tab_bar_height - self.cell_metrics.cell_height as f32) / 2.0;
                let text_x = tab_x + (tab_width - title_width) / 2.0;

                for (char_idx, c) in title.chars().enumerate() {
                    if c == ' ' {
                        continue;
                    }
                    let glyph = self.rasterize_char(c);
                    if glyph.size[0] > 0.0 && glyph.size[1] > 0.0 {
                        // In Kitty's model, glyphs are cell-sized and positioned at (0,0)
                        let char_x = text_x + char_idx as f32 * self.cell_metrics.cell_width as f32;
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
        // Use pre-computed linear palette for active border (palette color 4 - typically blue)
        let active_border_color = self.linear_palette.color_table[4];
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
            let epsilon = (self.cell_metrics.cell_height.max(self.cell_metrics.cell_width)) as f32;

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
            // Viewport for Kitty-style NDC rendering (x, y, width, height in pixels)
            viewport: (f32, f32, f32, f32),
            dim_overlay: Option<(f32, f32, f32, f32, [f32; 4])>, // (x, y, w, h, color)
        }
        let mut pane_render_list: Vec<PaneRenderData> = Vec::new();
        
        #[cfg(feature = "render_timing")]
        let pane_loop_start = std::time::Instant::now();
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
            #[cfg(feature = "render_timing")]
            let t0 = std::time::Instant::now();
            self.update_gpu_cells(terminal);
            #[cfg(feature = "render_timing")]
            {
                let update_time = t0.elapsed();
                if update_time.as_micros() > 500 {
                    log::info!("update_gpu_cells took {:?}", update_time);
                }
            }
            
            let cols = terminal.cols as u32;
            let rows = terminal.rows as u32;
            
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
                cell_width: self.cell_metrics.cell_width,
                cell_height: self.cell_metrics.cell_height,
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
            
            // DEBUG: Log grid params every 60 frames
            static PANE_DEBUG_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let pane_frame = PANE_DEBUG_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if pane_frame % 60 == 0 {
                log::info!("DEBUG pane {}: grid_params cols={} rows={} gpu_cells.len={} expected={}", 
                    info.pane_id, grid_params.cols, grid_params.rows, 
                    self.gpu_cells.len(), (grid_params.cols * grid_params.rows) as usize);
                
                // Sample a few cells to see if sprite indices look reasonable
                if !self.gpu_cells.is_empty() {
                    let sample_indices = [0, 1, 2, cols as usize, cols as usize + 1];
                    for &idx in &sample_indices {
                        if idx < self.gpu_cells.len() {
                            let cell = &self.gpu_cells[idx];
                            let sprite_idx = cell.sprite_idx & !0x80000000;
                            log::info!("DEBUG   cell[{}]: sprite_idx={} fg={:#x} bg={:#x}", 
                                idx, sprite_idx, cell.fg, cell.bg);
                            
                            if sprite_idx > 0 && (sprite_idx as usize) < self.sprite_info.len() {
                                let sprite = &self.sprite_info[sprite_idx as usize];
                                log::info!("DEBUG     sprite[{}]: uv=({:.3},{:.3},{:.3},{:.3}) layer={} size=({:.1},{:.1})",
                                    sprite_idx, sprite.uv[0], sprite.uv[1], sprite.uv[2], sprite.uv[3],
                                    sprite.layer, sprite.size[0], sprite.size[1]);
                            }
                        }
                    }
                }
            }
            
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
            
            // Viewport dimensions for Kitty-style NDC rendering
            // The viewport is set to the pane's pixel area, so the shader works in pure NDC space
            // Cell dimensions are already integers like Kitty - no floating-point accumulation errors
            let viewport_width = (cols * self.cell_metrics.cell_width) as f32;
            let viewport_height = (rows * self.cell_metrics.cell_height) as f32;
            // Also round the viewport position to pixel boundaries
            let viewport_x = pane_x.round();
            let viewport_y = pane_y.round();
            
            pane_render_list.push(PaneRenderData {
                pane_id: info.pane_id,
                cols,
                rows,
                viewport: (viewport_x, viewport_y, viewport_width, viewport_height),
                dim_overlay,
            });
        }
        #[cfg(feature = "render_timing")]
        {
            let pane_loop_time = pane_loop_start.elapsed();
            if pane_loop_time.as_micros() > 500 {
                log::info!("pane_loop took {:?}", pane_loop_time);
            }
        }
        
        // Clean up resources for panes that no longer exist (like Kitty's remove_vao)
        let active_pane_ids: std::collections::HashSet<u64> = pane_render_list.iter().map(|p| p.pane_id).collect();
        self.cleanup_unused_pane_resources(&active_pane_ids);

        // ═══════════════════════════════════════════════════════════════════
        // UPLOAD SHARED DATA (color table - uses pre-computed linear palette)
        // ═══════════════════════════════════════════════════════════════════
        self.queue.write_buffer(&self.color_table_buffer, 0, bytemuck::cast_slice(&self.linear_palette.color_table));

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
                    cell_width: self.cell_metrics.cell_width as f32,
                    cell_height: self.cell_metrics.cell_height as f32,
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

            let renders = self.image_renderer.prepare_image_renders(
                terminal.image_storage.placements(),
                pane_x,
                pane_y,
                self.cell_metrics.cell_width as f32,
                self.cell_metrics.cell_height as f32,
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

        // Atlas uploads now happen immediately in upload_cell_canvas_to_atlas()
        // like Kitty's send_sprite_to_gpu() - no batched layer uploads needed

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
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
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
            // 
            // Kitty-style viewport approach: set viewport to pane area so shader
            // can work in pure NDC space (-1 to +1), avoiding floating-point
            // precision issues that cause wobbly/misaligned text.
            // ═══════════════════════════════════════════════════════════════════
            for pane_data in &pane_render_list {
                let instance_count = pane_data.cols * pane_data.rows;
                
                // Get this pane's bind group (data already uploaded)
                if let Some(pane_res) = self.pane_resources.get(&pane_data.pane_id) {
                    // Set viewport to this pane's area (Kitty-style)
                    let (vp_x, vp_y, vp_w, vp_h) = pane_data.viewport;
                    render_pass.set_viewport(vp_x, vp_y, vp_w, vp_h, 0.0, 1.0);
                    
                    // Set scissor rect to clip rendering to pane bounds
                    let scissor_x = (vp_x.round().max(0.0) as u32).min(self.width);
                    let scissor_y = (vp_y.round().max(0.0) as u32).min(self.height);
                    let scissor_w = (vp_w.round() as u32).min(self.width.saturating_sub(scissor_x));
                    let scissor_h = (vp_h.round() as u32).min(self.height.saturating_sub(scissor_y));
                    
                    if scissor_w == 0 || scissor_h == 0 {
                        continue;
                    }
                    render_pass.set_scissor_rect(scissor_x, scissor_y, scissor_w, scissor_h);
                    
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
            
            // Restore full-screen viewport and scissor for remaining rendering (statusline, overlays)
            render_pass.set_viewport(0.0, 0.0, self.width as f32, self.height as f32, 0.0, 1.0);
            render_pass.set_scissor_rect(0, 0, self.width, self.height);
            
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
            if let Some(gpu_image) = self.image_renderer.get(image_id) {
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
                        depth_slice: None,
                    })],
                    depth_stencil_attachment: None,
                    occlusion_query_set: None,
                    timestamp_writes: None,
                    multiview_mask: None,
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
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });

            glow_pass.set_pipeline(&self.edge_glow_pipeline);
            glow_pass.set_bind_group(0, &self.edge_glow_bind_group, &[]);
            glow_pass.draw(0..3, 0..1); // Fullscreen triangle
        }

        #[cfg(feature = "render_timing")]
        let before_submit = frame_start.elapsed();
        self.queue.submit(std::iter::once(encoder.finish()));
        #[cfg(feature = "render_timing")]
        let after_submit = frame_start.elapsed();
        output.present();
        
        // Log timing if frame took more than 1ms (only with render_timing feature)
        #[cfg(feature = "render_timing")]
        {
            let after_present = frame_start.elapsed();
            if after_present.as_micros() > 1000 {
                log::info!("render_panes: before_submit={:?} submit={:?} present={:?} total={:?}",
                    before_submit,
                    after_submit - before_submit,
                    after_present - after_submit,
                    after_present);
            }
        }

        Ok(())
    }

    /// Sync images from terminal's image storage to GPU.
    /// Uploads new/changed images and removes deleted ones.
    /// Also updates animation frames.
    pub fn sync_images(&mut self, storage: &mut ImageStorage) {
        self.image_renderer.sync_images(&self.device, &self.queue, storage);
    }

}
