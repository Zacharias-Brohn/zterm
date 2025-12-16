//! GPU-accelerated terminal rendering using wgpu with a glyph atlas.
//! Uses rustybuzz (HarfBuzz port) for text shaping to support font features.

use crate::config::TabBarPosition;
use crate::graphics::{ImageData, ImagePlacement, ImageStorage};
use crate::terminal::{Color, ColorPalette, CursorShape, Direction, Terminal};
use ab_glyph::{Font, FontRef, GlyphId, ScaleFont};
use rustybuzz::UnicodeBuffer;
use ttf_parser::Tag;
use std::cell::OnceCell;
use std::collections::{HashMap, HashSet};
use std::ffi::CStr;
use std::path::PathBuf;
use std::sync::Arc;

// Fontconfig for dynamic font fallback
use fontconfig::Fontconfig;

/// Pane geometry for multi-pane rendering.
/// Describes where to render a pane within the window.
#[derive(Debug, Clone, Copy)]
pub struct PaneRenderInfo {
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

/// Cached glyph information.
#[derive(Clone, Copy, Debug)]
struct GlyphInfo {
    /// UV coordinates in the atlas (left, top, width, height) normalized 0-1.
    uv: [f32; 4],
    /// Offset from cell origin to glyph origin.
    offset: [f32; 2],
    /// Size of the glyph in pixels.
    size: [f32; 2],
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

/// Cached cell sprites for a text run.
/// When we render a text run using Kitty's canvas approach, we get one sprite per cell.
/// This caches those sprites so we don't re-render the same text runs.
#[derive(Clone, Debug)]
struct TextRunSprites {
    /// UV coordinates and sizes for each cell in the run.
    /// Each entry is (uv_x, uv_y, uv_w, uv_h) in the atlas, plus cell_width and cell_height.
    cells: Vec<[f32; 4]>,
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
    shaping_ctx: ShapingContext,
    /// OpenType features for shaping (shared across all font variants)
    shaping_features: Vec<rustybuzz::Feature>,
    char_cache: HashMap<char, GlyphInfo>,    // cache char -> rendered glyph
    ligature_cache: HashMap<String, ShapedGlyphs>, // cache multi-char -> shaped glyphs
    /// Glyph cache keyed by (font_style, font_index, glyph_id)
    /// font_style is FontStyle as usize, font_index is 0 for primary, 1+ for fallbacks
    glyph_cache: HashMap<(usize, usize, u16), GlyphInfo>,
    /// Cache for text run sprites (Kitty-style texture healing).
    /// Keyed by the text string of the run. Value contains UV coords for each cell.
    text_run_cache: HashMap<String, TextRunSprites>,
    /// Reusable canvas buffer for rendering text runs (Kitty-style texture healing).
    /// This is a temporary buffer we render multiple glyphs into, then slice into cells.
    canvas_buffer: Vec<u8>,
    /// Current canvas dimensions (width, height) in pixels.
    canvas_size: (u32, u32),
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
    /// Window dimensions.
    pub width: u32,
    pub height: u32,
    /// Color palette for rendering.
    palette: ColorPalette,
    /// Tab bar position.
    tab_bar_position: TabBarPosition,
    /// Background opacity (0.0 = transparent, 1.0 = opaque).
    background_opacity: f32,

    // Reusable vertex/index buffers to avoid per-frame allocations
    bg_vertices: Vec<GlyphVertex>,
    bg_indices: Vec<u32>,
    glyph_vertices: Vec<GlyphVertex>,
    glyph_indices: Vec<u32>,
    overlay_vertices: Vec<GlyphVertex>,
    overlay_indices: Vec<u32>,

    /// Current selection range for rendering (start_col, start_row, end_col, end_row).
    /// If set, cells within this range will be rendered with inverted colors.
    selection: Option<(usize, usize, usize, usize)>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// FONTCONFIG HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Find a font that contains the given character using fontconfig.
/// Returns the path to the font file if found.
fn find_font_for_char(_fc: &Fontconfig, c: char) -> Option<PathBuf> {
    use fontconfig_sys as fcsys;
    use fcsys::*;

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

        // Run substitutions
        FcConfigSubstitute(std::ptr::null_mut(), pat, FcMatchPattern);
        FcDefaultSubstitute(pat);

        // Find matching font
        let mut result = FcResultNoMatch;
        let matched = FcFontMatch(std::ptr::null_mut(), pat, &mut result);

        let font_path = if !matched.is_null() && result == FcResultMatch {
            // Get the file path from the matched pattern
            let mut file_ptr: *mut FcChar8 = std::ptr::null_mut();
            let fc_file_cstr = CStr::from_bytes_with_nul(b"file\0").unwrap();
            if FcPatternGetString(matched, fc_file_cstr.as_ptr(), 0, &mut file_ptr) == FcResultMatch
            {
                let path_cstr = CStr::from_ptr(file_ptr as *const i8);
                Some(PathBuf::from(path_cstr.to_string_lossy().into_owned()))
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

        font_path
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
            format: wgpu::TextureFormat::R8Unorm,
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
            atlas_data: vec![0u8; (ATLAS_SIZE * ATLAS_SIZE) as usize],
            atlas_dirty: false,
            font_data,
            primary_font,
            font_variants,
            fallback_fonts,
            fontconfig: OnceCell::new(),
            tried_font_paths,
            shaping_ctx,
            shaping_features,
            char_cache: HashMap::new(),
            ligature_cache: HashMap::new(),
            glyph_cache: HashMap::new(),
            text_run_cache: HashMap::new(),
            // Initial canvas size - will be resized as needed
            canvas_buffer: vec![0u8; (cell_width as usize * 16) * cell_height as usize],
            canvas_size: (cell_width as u32 * 16, cell_height as u32),
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
            width: size.width,
            height: size.height,
            palette: ColorPalette::default(),
            tab_bar_position: config.tab_bar_position,
            background_opacity: config.background_opacity.clamp(0.0, 1.0),
            // Pre-allocate reusable buffers for rendering
            bg_vertices: Vec::with_capacity(4096),
            bg_indices: Vec::with_capacity(6144),
            glyph_vertices: Vec::with_capacity(4096),
            glyph_indices: Vec::with_capacity(6144),
            overlay_vertices: Vec::with_capacity(64),
            overlay_indices: Vec::with_capacity(96),
            selection: None,
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

    /// Converts a pixel position to a terminal cell position.
    /// Returns None if the position is outside the terminal area (e.g., in the tab bar or statusline).
    pub fn pixel_to_cell(&self, x: f64, y: f64) -> Option<(usize, usize)> {
        let terminal_y_offset = self.terminal_y_offset();
        let tab_bar_height = self.tab_bar_height();
        let statusline_height = self.statusline_height();
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

        // Adjust y to be relative to terminal area
        let terminal_y = y as f32 - terminal_y_offset;

        // Calculate cell position
        let col = (x as f32 / self.cell_width).floor() as usize;
        let row = (terminal_y / self.cell_height).floor() as usize;

        // Get terminal dimensions to clamp
        let (max_cols, max_rows) = self.terminal_size();

        // Clamp to valid range
        let col = col.min(max_cols.saturating_sub(1));
        let row = row.min(max_rows.saturating_sub(1));

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
        self.text_run_cache.clear();

        // Reset atlas
        self.atlas_cursor_x = 0;
        self.atlas_cursor_y = 0;
        self.atlas_row_height = 0;
        self.atlas_data.fill(0);
        self.atlas_dirty = true;

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
        self.text_run_cache.clear();

        // Reset atlas
        self.atlas_cursor_x = 0;
        self.atlas_cursor_y = 0;
        self.atlas_row_height = 0;
        self.atlas_data.fill(0);
        self.atlas_dirty = true;

        // Return true if cell dimensions changed
        (self.cell_width - old_cell_width).abs() > 0.01
            || (self.cell_height - old_cell_height).abs() > 0.01
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
            return *info;
        }

        // Check if this is a box-drawing character - render procedurally
        if Self::is_box_drawing(c) {
            if let Some((bitmap, supersampled)) = self.render_box_char(c) {
                let glyph_width = self.cell_width.ceil() as u32;
                let glyph_height = self.cell_height.ceil() as u32;

                // Check if we need to move to next row
                if self.atlas_cursor_x + glyph_width > ATLAS_SIZE {
                    self.atlas_cursor_x = 0;
                    self.atlas_cursor_y += self.atlas_row_height + 1;
                    self.atlas_row_height = 0;
                }

                // Check if atlas is full
                if self.atlas_cursor_y + glyph_height > ATLAS_SIZE {
                    log::warn!("Glyph atlas is full!");
                    let info = GlyphInfo {
                        uv: [0.0, 0.0, 0.0, 0.0],
                        offset: [0.0, 0.0],
                        size: [0.0, 0.0],
                    };
                    self.char_cache.insert(c, info);
                    return info;
                }

                // Copy bitmap to atlas
                for y in 0..glyph_height as usize {
                    for x in 0..glyph_width as usize {
                        let src_idx = y * glyph_width as usize + x;
                        let dst_x = self.atlas_cursor_x + x as u32;
                        let dst_y = self.atlas_cursor_y + y as u32;
                        let dst_idx = (dst_y * ATLAS_SIZE + dst_x) as usize;
                        self.atlas_data[dst_idx] = bitmap[src_idx];
                    }
                }
                self.atlas_dirty = true;

                // Calculate UV coordinates
                let uv_x = self.atlas_cursor_x as f32 / ATLAS_SIZE as f32;
                let uv_y = self.atlas_cursor_y as f32 / ATLAS_SIZE as f32;
                let uv_w = glyph_width as f32 / ATLAS_SIZE as f32;
                let uv_h = glyph_height as f32 / ATLAS_SIZE as f32;

                // For supersampled glyphs, use actual cell dimensions to avoid bleeding
                // For pixel-perfect glyphs, use ceiled bitmap dimensions
                let (size_w, size_h) = if supersampled {
                    (self.cell_width as f32, self.cell_height as f32)
                } else {
                    (glyph_width as f32, glyph_height as f32)
                };

                let info = GlyphInfo {
                    uv: [uv_x, uv_y, uv_w, uv_h],
                    offset: [0.0, 0.0],
                    size: [size_w, size_h],
                };

                // Update atlas cursor
                self.atlas_cursor_x += glyph_width + 1;
                self.atlas_row_height = self.atlas_row_height.max(glyph_height);

                self.char_cache.insert(c, info);
                return info;
            }
        }

        // Try primary font first, then fallbacks using ab_glyph
        let glyph_id = self.primary_font.glyph_id(c);
        
        // Rasterize glyph data: (width, height, bitmap, offset_x, offset_y)
        let raster_result: Option<(u32, u32, Vec<u8>, f32, f32)> = if glyph_id.0 != 0 {
            // Primary font has this glyph
            self.rasterize_glyph_ab(&self.primary_font.clone(), glyph_id)
        } else {
            // Try already-loaded fallback fonts first
            let mut result = None;
            for (_, fallback_font) in &self.fallback_fonts {
                let fb_glyph_id = fallback_font.glyph_id(c);
                if fb_glyph_id.0 != 0 {
                    result = self.rasterize_glyph_ab(&fallback_font.clone(), fb_glyph_id);
                    break;
                }
            }

            // If no cached fallback has the glyph, use fontconfig to find one
            if result.is_none() {
                // Lazy-initialize fontconfig on first use
                let fc = self.fontconfig.get_or_init(|| {
                    log::debug!("Initializing fontconfig for fallback font lookup");
                    Fontconfig::new()
                });
                if let Some(fc) = fc {
                    // Query fontconfig for a font that has this character
                    if let Some(path) = find_font_for_char(fc, c) {
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

            // Use primary font's .notdef if no fallback has the glyph
            result.or_else(|| self.rasterize_glyph_ab(&self.primary_font.clone(), glyph_id))
        };

        // Handle rasterization result
        let Some((glyph_width, glyph_height, bitmap, offset_x, offset_y)) = raster_result else {
            // Empty glyph (e.g., space)
            let info = GlyphInfo {
                uv: [0.0, 0.0, 0.0, 0.0],
                offset: [0.0, 0.0],
                size: [0.0, 0.0],
            };
            self.char_cache.insert(c, info);
            return info;
        };

        if bitmap.is_empty() || glyph_width == 0 || glyph_height == 0 {
            // Empty glyph (e.g., space)
            let info = GlyphInfo {
                uv: [0.0, 0.0, 0.0, 0.0],
                offset: [0.0, 0.0],
                size: [0.0, 0.0],
            };
            self.char_cache.insert(c, info);
            return info;
        }

        // Check if we need to move to next row
        if self.atlas_cursor_x + glyph_width > ATLAS_SIZE {
            self.atlas_cursor_x = 0;
            self.atlas_cursor_y += self.atlas_row_height + 1;
            self.atlas_row_height = 0;
        }

        // Check if atlas is full
        if self.atlas_cursor_y + glyph_height > ATLAS_SIZE {
            log::warn!("Glyph atlas is full!");
            let info = GlyphInfo {
                uv: [0.0, 0.0, 0.0, 0.0],
                offset: [0.0, 0.0],
                size: [0.0, 0.0],
            };
            self.char_cache.insert(c, info);
            return info;
        }

        // Copy bitmap to atlas
        for y in 0..glyph_height as usize {
            for x in 0..glyph_width as usize {
                let src_idx = y * glyph_width as usize + x;
                let dst_x = self.atlas_cursor_x + x as u32;
                let dst_y = self.atlas_cursor_y + y as u32;
                let dst_idx = (dst_y * ATLAS_SIZE + dst_x) as usize;
                self.atlas_data[dst_idx] = bitmap[src_idx];
            }
        }
        self.atlas_dirty = true;

        // Calculate UV coordinates
        let uv_x = self.atlas_cursor_x as f32 / ATLAS_SIZE as f32;
        let uv_y = self.atlas_cursor_y as f32 / ATLAS_SIZE as f32;
        let uv_w = glyph_width as f32 / ATLAS_SIZE as f32;
        let uv_h = glyph_height as f32 / ATLAS_SIZE as f32;

        let info = GlyphInfo {
            uv: [uv_x, uv_y, uv_w, uv_h],
            offset: [offset_x, offset_y],
            size: [glyph_width as f32, glyph_height as f32],
        };

        // Update atlas cursor
        self.atlas_cursor_x += glyph_width + 1;
        self.atlas_row_height = self.atlas_row_height.max(glyph_height);

        self.char_cache.insert(c, info);
        info
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
                offset: [0.0, 0.0],
                size: [0.0, 0.0],
            };
            self.glyph_cache.insert(cache_key, info);
            return info;
        };

        if bitmap.is_empty() || glyph_width == 0 || glyph_height == 0 {
            // Empty glyph (e.g., space)
            let info = GlyphInfo {
                uv: [0.0, 0.0, 0.0, 0.0],
                offset: [0.0, 0.0],
                size: [0.0, 0.0],
            };
            self.glyph_cache.insert(cache_key, info);
            return info;
        }

        // Check if we need to move to next row
        if self.atlas_cursor_x + glyph_width > ATLAS_SIZE {
            self.atlas_cursor_x = 0;
            self.atlas_cursor_y += self.atlas_row_height + 1;
            self.atlas_row_height = 0;
        }

        // Check if atlas is full
        if self.atlas_cursor_y + glyph_height > ATLAS_SIZE {
            log::warn!("Glyph atlas is full!");
            let info = GlyphInfo {
                uv: [0.0, 0.0, 0.0, 0.0],
                offset: [0.0, 0.0],
                size: [0.0, 0.0],
            };
            self.glyph_cache.insert(cache_key, info);
            return info;
        }

        // Copy bitmap to atlas
        for y in 0..glyph_height as usize {
            for x in 0..glyph_width as usize {
                let src_idx = y * glyph_width as usize + x;
                let dst_x = self.atlas_cursor_x + x as u32;
                let dst_y = self.atlas_cursor_y + y as u32;
                let dst_idx = (dst_y * ATLAS_SIZE + dst_x) as usize;
                self.atlas_data[dst_idx] = bitmap[src_idx];
            }
        }
        self.atlas_dirty = true;

        // Calculate UV coordinates
        let uv_x = self.atlas_cursor_x as f32 / ATLAS_SIZE as f32;
        let uv_y = self.atlas_cursor_y as f32 / ATLAS_SIZE as f32;
        let uv_w = glyph_width as f32 / ATLAS_SIZE as f32;
        let uv_h = glyph_height as f32 / ATLAS_SIZE as f32;

        let info = GlyphInfo {
            uv: [uv_x, uv_y, uv_w, uv_h],
            offset: [offset_x, offset_y],
            size: [glyph_width as f32, glyph_height as f32],
        };

        // Update atlas cursor
        self.atlas_cursor_x += glyph_width + 1;
        self.atlas_row_height = self.atlas_row_height.max(glyph_height);

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
                offset: [0.0, 0.0],
                size: [0.0, 0.0],
            };
            self.glyph_cache.insert(cache_key, info);
            return info;
        };

        if bitmap.is_empty() || glyph_width == 0 || glyph_height == 0 {
            // Empty glyph (e.g., space)
            let info = GlyphInfo {
                uv: [0.0, 0.0, 0.0, 0.0],
                offset: [0.0, 0.0],
                size: [0.0, 0.0],
            };
            self.glyph_cache.insert(cache_key, info);
            return info;
        }

        // Check if we need to move to next row
        if self.atlas_cursor_x + glyph_width > ATLAS_SIZE {
            self.atlas_cursor_x = 0;
            self.atlas_cursor_y += self.atlas_row_height + 1;
            self.atlas_row_height = 0;
        }

        // Check if atlas is full
        if self.atlas_cursor_y + glyph_height > ATLAS_SIZE {
            log::warn!("Glyph atlas is full!");
            let info = GlyphInfo {
                uv: [0.0, 0.0, 0.0, 0.0],
                offset: [0.0, 0.0],
                size: [0.0, 0.0],
            };
            self.glyph_cache.insert(cache_key, info);
            return info;
        }

        // Copy bitmap to atlas
        for y in 0..glyph_height as usize {
            for x in 0..glyph_width as usize {
                let src_idx = y * glyph_width as usize + x;
                let dst_x = self.atlas_cursor_x + x as u32;
                let dst_y = self.atlas_cursor_y + y as u32;
                let dst_idx = (dst_y * ATLAS_SIZE + dst_x) as usize;
                self.atlas_data[dst_idx] = bitmap[src_idx];
            }
        }
        self.atlas_dirty = true;

        // Calculate UV coordinates
        let uv_x = self.atlas_cursor_x as f32 / ATLAS_SIZE as f32;
        let uv_y = self.atlas_cursor_y as f32 / ATLAS_SIZE as f32;
        let uv_w = glyph_width as f32 / ATLAS_SIZE as f32;
        let uv_h = glyph_height as f32 / ATLAS_SIZE as f32;

        let info = GlyphInfo {
            uv: [uv_x, uv_y, uv_w, uv_h],
            offset: [offset_x, offset_y],
            size: [glyph_width as f32, glyph_height as f32],
        };

        // Update atlas cursor
        self.atlas_cursor_x += glyph_width + 1;
        self.atlas_row_height = self.atlas_row_height.max(glyph_height);

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

    // ═══════════════════════════════════════════════════════════════════════════
    // KITTY-STYLE TEXTURE HEALING: CANVAS-BASED TEXT RUN RENDERING
    // ═══════════════════════════════════════════════════════════════════════════
    //
    // This implements Kitty's approach to texture healing:
    // 1. Render all glyphs in a text run into a single multi-cell canvas
    // 2. Use HarfBuzz x_offset and x_advance for glyph positioning
    // 3. Extract individual cell-sized sprites from the canvas
    // 4. Upload each cell sprite to the atlas
    //
    // This allows narrow characters (like 'i') to leave space for adjacent
    // wider characters (like 'm'), creating more balanced text appearance.

    /// Ensure the canvas buffer is large enough for the given number of cells.
    fn ensure_canvas_size(&mut self, num_cells: usize) {
        let required_width = (self.cell_width as usize) * num_cells;
        let required_height = self.cell_height as usize;
        let required_size = required_width * required_height;
        
        if self.canvas_buffer.len() < required_size {
            self.canvas_buffer.resize(required_size, 0);
        }
        self.canvas_size = (required_width as u32, required_height as u32);
    }

    /// Clear the canvas buffer to transparent.
    fn clear_canvas(&mut self, num_cells: usize) {
        let canvas_width = (self.cell_width as usize) * num_cells;
        let canvas_height = self.cell_height as usize;
        let size = canvas_width * canvas_height;
        // Only clear the portion we're using
        self.canvas_buffer[..size].fill(0);
    }

    /// Render shaped glyphs into the canvas buffer using HarfBuzz positions.
    /// This is the core of Kitty's texture healing approach.
    /// Note: Kept for potential fallback use. Use render_glyphs_to_canvas_with_style for styled text.
    /// 
    /// Arguments:
    /// - `shaped`: The shaped glyph data from HarfBuzz
    /// - `num_cells`: Number of cells this text run spans
    /// - `baseline_offset`: Offset from top of cell to baseline (typically cell_height * 0.8)
    #[allow(dead_code)]
    fn render_glyphs_to_canvas(&mut self, shaped: &ShapedGlyphs, num_cells: usize, baseline_offset: f32) {
        let canvas_width = (self.cell_width as usize) * num_cells;
        let canvas_height = self.cell_height as usize;
        
        // Track cursor position in canvas coordinates
        let mut cursor_x: f32 = 0.0;
        
        for &(glyph_id, x_advance, x_offset, y_offset, _cluster) in &shaped.glyphs {
            // Get the rasterized glyph bitmap
            let glyph = self.get_glyph_by_id(glyph_id);
            
            if glyph.size[0] <= 0.0 || glyph.size[1] <= 0.0 {
                // Empty glyph (e.g., space) - just advance cursor
                cursor_x += x_advance;
                continue;
            }
            
            // Calculate glyph position in canvas:
            // - cursor_x: accumulated from x_advance
            // - x_offset: HarfBuzz adjustment for texture healing
            // - glyph.offset[0]: left bearing from the font
            let glyph_x = cursor_x + x_offset + glyph.offset[0];
            
            // Y position: baseline_offset is from top to baseline
            // glyph.offset[1] is distance from baseline to glyph bottom (negative = below baseline)
            // For canvas coords (y=0 at top), we need: baseline_offset - glyph_top
            // glyph_top = glyph.offset[1] + glyph.size[1] (since offset is to bottom)
            let glyph_y = baseline_offset + y_offset - glyph.offset[1] - glyph.size[1];
            
            // Round to integers for pixel placement
            let dest_x = glyph_x.round() as i32;
            let dest_y = glyph_y.round() as i32;
            
            // Get glyph bitmap from atlas
            // The glyph.uv gives us the location in the atlas
            let atlas_x = (glyph.uv[0] * ATLAS_SIZE as f32) as u32;
            let atlas_y = (glyph.uv[1] * ATLAS_SIZE as f32) as u32;
            let glyph_w = glyph.size[0] as u32;
            let glyph_h = glyph.size[1] as u32;
            
            // Copy glyph bitmap from atlas to canvas, with bounds checking
            for gy in 0..glyph_h {
                let canvas_y = dest_y + gy as i32;
                if canvas_y < 0 || canvas_y >= canvas_height as i32 {
                    continue;
                }
                
                for gx in 0..glyph_w {
                    let canvas_x = dest_x + gx as i32;
                    if canvas_x < 0 || canvas_x >= canvas_width as i32 {
                        continue;
                    }
                    
                    // Source in atlas
                    let src_idx = ((atlas_y + gy) * ATLAS_SIZE + (atlas_x + gx)) as usize;
                    // Destination in canvas
                    let dst_idx = (canvas_y as usize) * canvas_width + (canvas_x as usize);
                    
                    // Blend: use max for overlapping glyphs (simple compositing)
                    let src_alpha = self.atlas_data[src_idx];
                    let dst_alpha = self.canvas_buffer[dst_idx];
                    self.canvas_buffer[dst_idx] = src_alpha.max(dst_alpha);
                }
            }
            
            // Advance cursor by glyph's advance width.
            // Round the advance to ensure it aligns with cell boundaries for monospace fonts.
            // This matches Kitty's approach: x += roundf(x_advance)
            cursor_x += x_advance.round();
        }
    }

    /// Render shaped glyphs into the canvas buffer using a specific font style.
    /// This version uses the appropriate font variant for bold/italic text.
    fn render_glyphs_to_canvas_with_style(&mut self, shaped: &ShapedGlyphs, num_cells: usize, baseline_offset: f32, style: FontStyle) {
        let canvas_width = (self.cell_width as usize) * num_cells;
        let canvas_height = self.cell_height as usize;
        
        // Track cursor position in canvas coordinates
        let mut cursor_x: f32 = 0.0;
        
        for &(glyph_id, x_advance, x_offset, y_offset, _cluster) in &shaped.glyphs {
            // Get the rasterized glyph bitmap using the correct font variant
            let glyph = self.get_glyph_by_id_with_style(glyph_id, style);
            
            if glyph.size[0] <= 0.0 || glyph.size[1] <= 0.0 {
                // Empty glyph (e.g., space) - just advance cursor
                cursor_x += x_advance;
                continue;
            }
            
            // Calculate glyph position in canvas:
            // - cursor_x: accumulated from x_advance
            // - x_offset: HarfBuzz adjustment for texture healing
            // - glyph.offset[0]: left bearing from the font
            let glyph_x = cursor_x + x_offset + glyph.offset[0];
            
            // Y position: baseline_offset is from top to baseline
            // glyph.offset[1] is distance from baseline to glyph bottom (negative = below baseline)
            // For canvas coords (y=0 at top), we need: baseline_offset - glyph_top
            // glyph_top = glyph.offset[1] + glyph.size[1] (since offset is to bottom)
            let glyph_y = baseline_offset + y_offset - glyph.offset[1] - glyph.size[1];
            
            // Round to integers for pixel placement
            let dest_x = glyph_x.round() as i32;
            let dest_y = glyph_y.round() as i32;
            
            // Get glyph bitmap from atlas
            // The glyph.uv gives us the location in the atlas
            let atlas_x = (glyph.uv[0] * ATLAS_SIZE as f32) as u32;
            let atlas_y = (glyph.uv[1] * ATLAS_SIZE as f32) as u32;
            let glyph_w = glyph.size[0] as u32;
            let glyph_h = glyph.size[1] as u32;
            
            // Copy glyph bitmap from atlas to canvas, with bounds checking
            for gy in 0..glyph_h {
                let canvas_y = dest_y + gy as i32;
                if canvas_y < 0 || canvas_y >= canvas_height as i32 {
                    continue;
                }
                
                for gx in 0..glyph_w {
                    let canvas_x = dest_x + gx as i32;
                    if canvas_x < 0 || canvas_x >= canvas_width as i32 {
                        continue;
                    }
                    
                    // Source in atlas
                    let src_idx = ((atlas_y + gy) * ATLAS_SIZE + (atlas_x + gx)) as usize;
                    // Destination in canvas
                    let dst_idx = (canvas_y as usize) * canvas_width + (canvas_x as usize);
                    
                    // Blend: use max for overlapping glyphs (simple compositing)
                    let src_alpha = self.atlas_data[src_idx];
                    let dst_alpha = self.canvas_buffer[dst_idx];
                    self.canvas_buffer[dst_idx] = src_alpha.max(dst_alpha);
                }
            }
            
            // Advance cursor by glyph's advance width.
            // Round the advance to ensure it aligns with cell boundaries for monospace fonts.
            cursor_x += x_advance.round();
        }
    }

    /// Extract a single cell from the canvas and upload it to the atlas.
    /// Returns the GlyphInfo for this cell's sprite.
    fn extract_cell_from_canvas(&mut self, cell_index: usize, num_cells: usize) -> [f32; 4] {
        let cell_width = self.cell_width as u32;
        let cell_height = self.cell_height as u32;
        let canvas_width = cell_width * num_cells as u32;
        
        // Check if we need to move to next row in atlas
        if self.atlas_cursor_x + cell_width > ATLAS_SIZE {
            self.atlas_cursor_x = 0;
            self.atlas_cursor_y += self.atlas_row_height + 1;
            self.atlas_row_height = 0;
        }
        
        // Check if atlas is full
        if self.atlas_cursor_y + cell_height > ATLAS_SIZE {
            log::warn!("Glyph atlas is full during text run rendering!");
            return [0.0, 0.0, 0.0, 0.0];
        }
        
        // Copy cell region from canvas to atlas
        let src_x_start = cell_index as u32 * cell_width;
        for y in 0..cell_height {
            for x in 0..cell_width {
                let src_idx = (y * canvas_width + src_x_start + x) as usize;
                let dst_x = self.atlas_cursor_x + x;
                let dst_y = self.atlas_cursor_y + y;
                let dst_idx = (dst_y * ATLAS_SIZE + dst_x) as usize;
                self.atlas_data[dst_idx] = self.canvas_buffer[src_idx];
            }
        }
        self.atlas_dirty = true;
        
        // Calculate UV coordinates
        let uv_x = self.atlas_cursor_x as f32 / ATLAS_SIZE as f32;
        let uv_y = self.atlas_cursor_y as f32 / ATLAS_SIZE as f32;
        let uv_w = cell_width as f32 / ATLAS_SIZE as f32;
        let uv_h = cell_height as f32 / ATLAS_SIZE as f32;
        
        // Update atlas cursor
        self.atlas_cursor_x += cell_width + 1;
        self.atlas_row_height = self.atlas_row_height.max(cell_height);
        
        [uv_x, uv_y, uv_w, uv_h]
    }

    /// Render a text run using Kitty's canvas-based approach for texture healing.
    /// Returns TextRunSprites containing UV coordinates for each cell.
    fn render_text_run(&mut self, text: &str, num_cells: usize, style: FontStyle) -> TextRunSprites {
        // Check cache first - include font style in cache key
        let cache_key = format!("{}\x00{}", style as usize, text);
        if let Some(cached) = self.text_run_cache.get(&cache_key) {
            return cached.clone();
        }
        
        // Shape the text using the appropriate font variant
        let shaped = self.shape_text_with_style(text, style);
        
        // Ensure canvas is big enough
        self.ensure_canvas_size(num_cells);
        self.clear_canvas(num_cells);
        
        // Calculate baseline offset (typically ~80% down from top of cell)
        let baseline_offset = self.cell_height * 0.8;
        
        // Render all glyphs into the canvas using the appropriate font variant
        self.render_glyphs_to_canvas_with_style(&shaped, num_cells, baseline_offset, style);
        
        // Extract each cell from the canvas
        let mut cells = Vec::with_capacity(num_cells);
        for i in 0..num_cells {
            let uv = self.extract_cell_from_canvas(i, num_cells);
            cells.push(uv);
        }
        
        let sprites = TextRunSprites { cells };
        self.text_run_cache.insert(cache_key, sprites.clone());
        sprites
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

    /// Render a single pane's terminal content at a given position.
    /// This is a helper method for multi-pane rendering.
    ///
    /// Arguments:
    /// - `terminal`: The terminal state for this pane
    /// - `pane_x`: Left edge of pane in pixels
    /// - `pane_y`: Top edge of pane in pixels
    /// - `pane_width`: Width of pane in pixels
    /// - `pane_height`: Height of pane in pixels
    /// - `is_active`: Whether this is the active pane (for cursor rendering)
    /// - `selection`: Optional selection range (start_col, start_row, end_col, end_row)
    /// - `dim_factor`: Dimming factor (0.0 = fully dimmed, 1.0 = fully bright) - used for overlay
    fn render_pane_content(
        &mut self,
        terminal: &Terminal,
        pane_x: f32,
        pane_y: f32,
        pane_width: f32,
        pane_height: f32,
        is_active: bool,
        selection: Option<(usize, usize, usize, usize)>,
        _dim_factor: f32, // Dimming is now done via overlay in render_panes
    ) {
        let width = self.width as f32;
        let height = self.height as f32;

        // Calculate pane's terminal dimensions
        let cols = (pane_width / self.cell_width).floor() as usize;
        let rows = (pane_height / self.cell_height).floor() as usize;

        // Cache palette values
        let palette_default_fg = self.palette.default_fg;
        let palette_colors = self.palette.colors;

        // Helper to convert Color to linear RGBA
        let color_to_rgba = |color: &Color, is_foreground: bool| -> [f32; 4] {
            match color {
                Color::Default => {
                    if is_foreground {
                        let [r, g, b] = palette_default_fg;
                        [
                            Self::srgb_to_linear(r as f32 / 255.0),
                            Self::srgb_to_linear(g as f32 / 255.0),
                            Self::srgb_to_linear(b as f32 / 255.0),
                            1.0,
                        ]
                    } else {
                        [0.0, 0.0, 0.0, 0.0]
                    }
                }
                Color::Rgb(r, g, b) => [
                    Self::srgb_to_linear(*r as f32 / 255.0),
                    Self::srgb_to_linear(*g as f32 / 255.0),
                    Self::srgb_to_linear(*b as f32 / 255.0),
                    1.0,
                ],
                Color::Indexed(idx) => {
                    let [r, g, b] = palette_colors[*idx as usize];
                    [
                        Self::srgb_to_linear(r as f32 / 255.0),
                        Self::srgb_to_linear(g as f32 / 255.0),
                        Self::srgb_to_linear(b as f32 / 255.0),
                        1.0,
                    ]
                }
            }
        };

        // Helper to check if a cell is selected
        let is_cell_selected = |col: usize, row: usize| -> bool {
            let Some((start_col, start_row, end_col, end_row)) = selection else {
                return false;
            };
            if row < start_row || row > end_row {
                return false;
            }
            if start_row == end_row {
                return col >= start_col && col <= end_col;
            }
            if row == start_row {
                return col >= start_col;
            } else if row == end_row {
                return col <= end_col;
            } else {
                return true;
            }
        };

        // Get visible rows (accounts for scroll offset)
        let visible_rows = terminal.visible_rows();

        // Render each row
        for (row_idx, row) in visible_rows.iter().enumerate() {
            if row_idx >= rows {
                break;
            }

            // Find the last non-empty cell for selection clipping
            // Note: U+10EEEE is Kitty graphics placeholder, treat as empty
            const KITTY_PLACEHOLDER_CHAR: char = '\u{10EEEE}';
            let last_content_col = row.iter()
                .enumerate()
                .rev()
                .find(|(_, cell)| cell.character != ' ' && cell.character != '\0' && cell.character != KITTY_PLACEHOLDER_CHAR)
                .map(|(idx, _)| idx)
                .unwrap_or(0);

            // ═══════════════════════════════════════════════════════════════════════════
            // TEXT RUN SHAPING FOR TEXTURE HEALING
            // ═══════════════════════════════════════════════════════════════════════════
            // Instead of rendering character-by-character, we group consecutive cells
            // with the same styling into "text runs" and shape them together. This allows
            // the font's calt (contextual alternates) feature to apply texture healing,
            // which adjusts character positions based on their neighbors.
            
            let cell_y = pane_y + row_idx as f32 * self.cell_height;
            
            let mut col_idx = 0;
            while col_idx < row.len() && col_idx < cols {
                let cell = &row[col_idx];
                
                // Determine colors for this cell (with selection override)
                let (base_fg, base_bg) = if is_cell_selected(col_idx, row_idx) && col_idx <= last_content_col {
                    ([0.0f32, 0.0, 0.0, 1.0], [1.0f32, 1.0, 1.0, 1.0])
                } else {
                    (color_to_rgba(&cell.fg_color, true), color_to_rgba(&cell.bg_color, false))
                };
                
                // Track font style for bold/italic rendering
                let base_style = FontStyle::from_flags(cell.bold, cell.italic);
                
                // Collect a run of cells with the same fg/bg colors AND font style
                let mut run_text = String::new();
                let mut run_cells: Vec<(usize, char, bool)> = Vec::new(); // (col, char, is_box_drawing)
                
                while col_idx < row.len() && col_idx < cols {
                    let run_cell = &row[col_idx];
                    
                    // Check if this cell has the same colors
                    let (cell_fg, cell_bg) = if is_cell_selected(col_idx, row_idx) && col_idx <= last_content_col {
                        ([0.0f32, 0.0, 0.0, 1.0], [1.0f32, 1.0, 1.0, 1.0])
                    } else {
                        (color_to_rgba(&run_cell.fg_color, true), color_to_rgba(&run_cell.bg_color, false))
                    };
                    
                    // Check if font style (bold/italic) matches
                    let cell_style = FontStyle::from_flags(run_cell.bold, run_cell.italic);
                    
                    if cell_fg != base_fg || cell_bg != base_bg || cell_style != base_style {
                        break; // Different colors or font style, end this run
                    }
                    
                    let c = run_cell.character;
                    const KITTY_PLACEHOLDER: char = '\u{10EEEE}';
                    let is_renderable = c != ' ' && c != '\0' && c != KITTY_PLACEHOLDER;
                    let is_box = Self::is_box_drawing(c);
                    
                    run_cells.push((col_idx, c, is_box));
                    if is_renderable && !is_box {
                        run_text.push(c);
                    } else {
                        // Use a placeholder that won't affect shaping
                        run_text.push(' ');
                    }
                    
                    col_idx += 1;
                }
                
                // Render backgrounds for all cells in the run
                for &(cell_col, _, _) in &run_cells {
                    let cell_x = pane_x + cell_col as f32 * self.cell_width;
                    let cell_left = Self::pixel_to_ndc_x(cell_x, width);
                    let cell_right = Self::pixel_to_ndc_x(cell_x + self.cell_width, width);
                    let cell_top = Self::pixel_to_ndc_y(cell_y, height);
                    let cell_bottom = Self::pixel_to_ndc_y(cell_y + self.cell_height, height);
                    
                    let base_idx = self.bg_vertices.len() as u32;
                    self.bg_vertices.push(GlyphVertex {
                        position: [cell_left, cell_top],
                        uv: [0.0, 0.0],
                        color: base_fg,
                        bg_color: base_bg,
                    });
                    self.bg_vertices.push(GlyphVertex {
                        position: [cell_right, cell_top],
                        uv: [0.0, 0.0],
                        color: base_fg,
                        bg_color: base_bg,
                    });
                    self.bg_vertices.push(GlyphVertex {
                        position: [cell_right, cell_bottom],
                        uv: [0.0, 0.0],
                        color: base_fg,
                        bg_color: base_bg,
                    });
                    self.bg_vertices.push(GlyphVertex {
                        position: [cell_left, cell_bottom],
                        uv: [0.0, 0.0],
                        color: base_fg,
                        bg_color: base_bg,
                    });
                    self.bg_indices.extend_from_slice(&[
                        base_idx, base_idx + 1, base_idx + 2,
                        base_idx, base_idx + 2, base_idx + 3,
                    ]);
                }
                
                // ═══════════════════════════════════════════════════════════════
                // RENDER TEXT RUN USING KITTY'S CANVAS-BASED APPROACH
                // ═══════════════════════════════════════════════════════════════
                // Use canvas-based rendering for texture healing:
                // 1. Render all glyphs into a multi-cell canvas using HarfBuzz positions
                // 2. Extract each cell's portion as a sprite
                // 3. Render sprites at cell positions
                
                // First, collect renderable (non-box-drawing) characters for shaping
                let mut shape_text = String::new();
                let mut shape_indices: Vec<usize> = Vec::new(); // Maps shape_text position to run_cells index
                
                for (i, &(_, c, is_box)) in run_cells.iter().enumerate() {
                    const KITTY_PLACEHOLDER: char = '\u{10EEEE}';
                    if c != ' ' && c != '\0' && c != KITTY_PLACEHOLDER && !is_box {
                        shape_text.push(c);
                        shape_indices.push(i);
                    }
                }
                
                // Render the text run using canvas approach if there's text to shape
                let sprites = if !shape_text.is_empty() && shape_indices.len() == run_cells.len() {
                    // All cells are shapeable text - use canvas rendering
                    Some(self.render_text_run(&run_text, run_cells.len(), base_style))
                } else {
                    None
                };
                
                // Render each cell
                for (run_idx, &(cell_col, c, is_box)) in run_cells.iter().enumerate() {
                    let cell_x = pane_x + cell_col as f32 * self.cell_width;
                    
                    const KITTY_PLACEHOLDER: char = '\u{10EEEE}';
                    if c == ' ' || c == '\0' || c == KITTY_PLACEHOLDER {
                        // Empty cell - nothing to render
                        continue;
                    }
                    
                    if is_box {
                        // Box drawing - render without shaping/canvas
                        let glyph = self.rasterize_char(c);
                        if glyph.size[0] > 0.0 && glyph.size[1] > 0.0 {
                            let glyph_x = cell_x;
                            let glyph_y = cell_y;
                            
                            let left = Self::pixel_to_ndc_x(glyph_x, width);
                            let right = Self::pixel_to_ndc_x(glyph_x + glyph.size[0], width);
                            let top = Self::pixel_to_ndc_y(glyph_y, height);
                            let bottom = Self::pixel_to_ndc_y(glyph_y + glyph.size[1], height);
                            
                            let base_idx = self.glyph_vertices.len() as u32;
                            self.glyph_vertices.push(GlyphVertex {
                                position: [left, top],
                                uv: [glyph.uv[0], glyph.uv[1]],
                                color: base_fg,
                                bg_color: [0.0, 0.0, 0.0, 0.0],
                            });
                            self.glyph_vertices.push(GlyphVertex {
                                position: [right, top],
                                uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1]],
                                color: base_fg,
                                bg_color: [0.0, 0.0, 0.0, 0.0],
                            });
                            self.glyph_vertices.push(GlyphVertex {
                                position: [right, bottom],
                                uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1] + glyph.uv[3]],
                                color: base_fg,
                                bg_color: [0.0, 0.0, 0.0, 0.0],
                            });
                            self.glyph_vertices.push(GlyphVertex {
                                position: [left, bottom],
                                uv: [glyph.uv[0], glyph.uv[1] + glyph.uv[3]],
                                color: base_fg,
                                bg_color: [0.0, 0.0, 0.0, 0.0],
                            });
                            self.glyph_indices.extend_from_slice(&[
                                base_idx, base_idx + 1, base_idx + 2,
                                base_idx, base_idx + 2, base_idx + 3,
                            ]);
                        }
                    } else if let Some(ref run_sprites) = sprites {
                        // Use pre-rendered sprite from canvas
                        if run_idx < run_sprites.cells.len() {
                            let uv = run_sprites.cells[run_idx];
                            if uv[2] > 0.0 && uv[3] > 0.0 {
                                let left = Self::pixel_to_ndc_x(cell_x, width);
                                let right = Self::pixel_to_ndc_x(cell_x + self.cell_width, width);
                                let top = Self::pixel_to_ndc_y(cell_y, height);
                                let bottom = Self::pixel_to_ndc_y(cell_y + self.cell_height, height);
                                
                                let base_idx = self.glyph_vertices.len() as u32;
                                self.glyph_vertices.push(GlyphVertex {
                                    position: [left, top],
                                    uv: [uv[0], uv[1]],
                                    color: base_fg,
                                    bg_color: [0.0, 0.0, 0.0, 0.0],
                                });
                                self.glyph_vertices.push(GlyphVertex {
                                    position: [right, top],
                                    uv: [uv[0] + uv[2], uv[1]],
                                    color: base_fg,
                                    bg_color: [0.0, 0.0, 0.0, 0.0],
                                });
                                self.glyph_vertices.push(GlyphVertex {
                                    position: [right, bottom],
                                    uv: [uv[0] + uv[2], uv[1] + uv[3]],
                                    color: base_fg,
                                    bg_color: [0.0, 0.0, 0.0, 0.0],
                                });
                                self.glyph_vertices.push(GlyphVertex {
                                    position: [left, bottom],
                                    uv: [uv[0], uv[1] + uv[3]],
                                    color: base_fg,
                                    bg_color: [0.0, 0.0, 0.0, 0.0],
                                });
                                self.glyph_indices.extend_from_slice(&[
                                    base_idx, base_idx + 1, base_idx + 2,
                                    base_idx, base_idx + 2, base_idx + 3,
                                ]);
                            }
                        }
                    } else {
                        // Fallback: render character individually
                        let glyph = self.rasterize_char(c);
                        if glyph.size[0] > 0.0 && glyph.size[1] > 0.0 {
                            let glyph_x = (cell_x + glyph.offset[0]).round();
                            let glyph_y = (cell_y + self.cell_height * 0.8 - glyph.offset[1] - glyph.size[1]).round();
                            
                            let left = Self::pixel_to_ndc_x(glyph_x, width);
                            let right = Self::pixel_to_ndc_x(glyph_x + glyph.size[0], width);
                            let top = Self::pixel_to_ndc_y(glyph_y, height);
                            let bottom = Self::pixel_to_ndc_y(glyph_y + glyph.size[1], height);
                            
                            let base_idx = self.glyph_vertices.len() as u32;
                            self.glyph_vertices.push(GlyphVertex {
                                position: [left, top],
                                uv: [glyph.uv[0], glyph.uv[1]],
                                color: base_fg,
                                bg_color: [0.0, 0.0, 0.0, 0.0],
                            });
                            self.glyph_vertices.push(GlyphVertex {
                                position: [right, top],
                                uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1]],
                                color: base_fg,
                                bg_color: [0.0, 0.0, 0.0, 0.0],
                            });
                            self.glyph_vertices.push(GlyphVertex {
                                position: [right, bottom],
                                uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1] + glyph.uv[3]],
                                color: base_fg,
                                bg_color: [0.0, 0.0, 0.0, 0.0],
                            });
                            self.glyph_vertices.push(GlyphVertex {
                                position: [left, bottom],
                                uv: [glyph.uv[0], glyph.uv[1] + glyph.uv[3]],
                                color: base_fg,
                                bg_color: [0.0, 0.0, 0.0, 0.0],
                            });
                            self.glyph_indices.extend_from_slice(&[
                                base_idx, base_idx + 1, base_idx + 2,
                                base_idx, base_idx + 2, base_idx + 3,
                            ]);
                        }
                    }
                }
            }
        }

        // Render cursor only for active pane
        if is_active && terminal.cursor_visible && terminal.scroll_offset == 0
           && terminal.cursor_row < rows && terminal.cursor_col < cols {
            let cursor_col = terminal.cursor_col;
            let cursor_row = terminal.cursor_row;
            let cursor_x = pane_x + cursor_col as f32 * self.cell_width;
            let cursor_y = pane_y + cursor_row as f32 * self.cell_height;

            // Get cell under cursor
            let cursor_cell = visible_rows.get(cursor_row).and_then(|row| row.get(cursor_col));

            let (cell_fg, cell_bg, cell_char) = if let Some(cell) = cursor_cell {
                let fg = color_to_rgba(&cell.fg_color, true);
                let bg = color_to_rgba(&cell.bg_color, false);
                (fg, bg, cell.character)
            } else {
                let fg = {
                    let [r, g, b] = self.palette.default_fg;
                    [
                        Self::srgb_to_linear(r as f32 / 255.0),
                        Self::srgb_to_linear(g as f32 / 255.0),
                        Self::srgb_to_linear(b as f32 / 255.0),
                        1.0,
                    ]
                };
                (fg, [0.0, 0.0, 0.0, 0.0], ' ')
            };

            // Kitty graphics Unicode placeholder should be treated as empty
            const KITTY_PLACEHOLDER: char = '\u{10EEEE}';
            let has_character = cell_char != ' ' && cell_char != '\0' && cell_char != KITTY_PLACEHOLDER;

            let cursor_bg_color = if has_character {
                [cell_fg[0], cell_fg[1], cell_fg[2], 1.0]
            } else {
                if cell_bg[3] < 0.01 {
                    let white = Self::srgb_to_linear(0.9);
                    [white, white, white, 1.0]
                } else {
                    [1.0 - cell_bg[0], 1.0 - cell_bg[1], 1.0 - cell_bg[2], 1.0]
                }
            };

            let cursor_style = match terminal.cursor_shape {
                CursorShape::BlinkingBlock | CursorShape::SteadyBlock => 0,
                CursorShape::BlinkingUnderline | CursorShape::SteadyUnderline => 1,
                CursorShape::BlinkingBar | CursorShape::SteadyBar => 2,
            };

            let (left, right, top, bottom) = match cursor_style {
                0 => (
                    cursor_x,
                    cursor_x + self.cell_width,
                    cursor_y,
                    cursor_y + self.cell_height,
                ),
                1 => {
                    let underline_height = 2.0_f32.max(self.cell_height * 0.1);
                    (
                        cursor_x,
                        cursor_x + self.cell_width,
                        cursor_y + self.cell_height - underline_height,
                        cursor_y + self.cell_height,
                    )
                }
                _ => {
                    let bar_width = 2.0_f32.max(self.cell_width * 0.1);
                    (
                        cursor_x,
                        cursor_x + bar_width,
                        cursor_y,
                        cursor_y + self.cell_height,
                    )
                }
            };

            let cursor_left = Self::pixel_to_ndc_x(left, width);
            let cursor_right = Self::pixel_to_ndc_x(right, width);
            let cursor_top = Self::pixel_to_ndc_y(top, height);
            let cursor_bottom = Self::pixel_to_ndc_y(bottom, height);

            let base_idx = self.glyph_vertices.len() as u32;
            self.glyph_vertices.push(GlyphVertex {
                position: [cursor_left, cursor_top],
                uv: [0.0, 0.0],
                color: cursor_bg_color,
                bg_color: cursor_bg_color,
            });
            self.glyph_vertices.push(GlyphVertex {
                position: [cursor_right, cursor_top],
                uv: [0.0, 0.0],
                color: cursor_bg_color,
                bg_color: cursor_bg_color,
            });
            self.glyph_vertices.push(GlyphVertex {
                position: [cursor_right, cursor_bottom],
                uv: [0.0, 0.0],
                color: cursor_bg_color,
                bg_color: cursor_bg_color,
            });
            self.glyph_vertices.push(GlyphVertex {
                position: [cursor_left, cursor_bottom],
                uv: [0.0, 0.0],
                color: cursor_bg_color,
                bg_color: cursor_bg_color,
            });
            self.glyph_indices.extend_from_slice(&[
                base_idx, base_idx + 1, base_idx + 2,
                base_idx, base_idx + 2, base_idx + 3,
            ]);

            // If block cursor with character, render it inverted
            if cursor_style == 0 && has_character {
                let char_color = if cell_bg[3] < 0.01 {
                    [0.0, 0.0, 0.0, 1.0]
                } else {
                    [cell_bg[0], cell_bg[1], cell_bg[2], 1.0]
                };

                let glyph = self.rasterize_char(cell_char);
                if glyph.size[0] > 0.0 && glyph.size[1] > 0.0 {
                    let (glyph_x, glyph_y) = if Self::is_box_drawing(cell_char) {
                        (cursor_x, cursor_y)
                    } else {
                        let baseline_y = (cursor_y + self.cell_height * 0.8).round();
                        let gx = (cursor_x + glyph.offset[0]).round();
                        let gy = (baseline_y - glyph.offset[1] - glyph.size[1]).round();
                        (gx, gy)
                    };

                    let g_left = Self::pixel_to_ndc_x(glyph_x, width);
                    let g_right = Self::pixel_to_ndc_x(glyph_x + glyph.size[0], width);
                    let g_top = Self::pixel_to_ndc_y(glyph_y, height);
                    let g_bottom = Self::pixel_to_ndc_y(glyph_y + glyph.size[1], height);

                    let base_idx = self.glyph_vertices.len() as u32;
                    self.glyph_vertices.push(GlyphVertex {
                        position: [g_left, g_top],
                        uv: [glyph.uv[0], glyph.uv[1]],
                        color: char_color,
                        bg_color: [0.0, 0.0, 0.0, 0.0],
                    });
                    self.glyph_vertices.push(GlyphVertex {
                        position: [g_right, g_top],
                        uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1]],
                        color: char_color,
                        bg_color: [0.0, 0.0, 0.0, 0.0],
                    });
                    self.glyph_vertices.push(GlyphVertex {
                        position: [g_right, g_bottom],
                        uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1] + glyph.uv[3]],
                        color: char_color,
                        bg_color: [0.0, 0.0, 0.0, 0.0],
                    });
                    self.glyph_vertices.push(GlyphVertex {
                        position: [g_left, g_bottom],
                        uv: [glyph.uv[0], glyph.uv[1] + glyph.uv[3]],
                        color: char_color,
                        bg_color: [0.0, 0.0, 0.0, 0.0],
                    });
                    self.glyph_indices.extend_from_slice(&[
                        base_idx, base_idx + 1, base_idx + 2,
                        base_idx, base_idx + 2, base_idx + 3,
                    ]);
                }
            }
        }
    }

    /// Draw a filled rectangle.
    fn render_rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: [f32; 4]) {
        let width = self.width as f32;
        let height = self.height as f32;

        let left = Self::pixel_to_ndc_x(x, width);
        let right = Self::pixel_to_ndc_x(x + w, width);
        let top = Self::pixel_to_ndc_y(y, height);
        let bottom = Self::pixel_to_ndc_y(y + h, height);

        let base_idx = self.bg_vertices.len() as u32;
        self.bg_vertices.push(GlyphVertex {
            position: [left, top],
            uv: [0.0, 0.0],
            color,
            bg_color: color,
        });
        self.bg_vertices.push(GlyphVertex {
            position: [right, top],
            uv: [0.0, 0.0],
            color,
            bg_color: color,
        });
        self.bg_vertices.push(GlyphVertex {
            position: [right, bottom],
            uv: [0.0, 0.0],
            color,
            bg_color: color,
        });
        self.bg_vertices.push(GlyphVertex {
            position: [left, bottom],
            uv: [0.0, 0.0],
            color,
            bg_color: color,
        });
        self.bg_indices.extend_from_slice(&[
            base_idx, base_idx + 1, base_idx + 2,
            base_idx, base_idx + 2, base_idx + 3,
        ]);
    }

    /// Draw a filled rectangle to the overlay layer (rendered on top of everything).
    fn render_overlay_rect(&mut self, x: f32, y: f32, w: f32, h: f32, color: [f32; 4]) {
        let width = self.width as f32;
        let height = self.height as f32;

        let left = Self::pixel_to_ndc_x(x, width);
        let right = Self::pixel_to_ndc_x(x + w, width);
        let top = Self::pixel_to_ndc_y(y, height);
        let bottom = Self::pixel_to_ndc_y(y + h, height);

        let base_idx = self.overlay_vertices.len() as u32;
        self.overlay_vertices.push(GlyphVertex {
            position: [left, top],
            uv: [0.0, 0.0],
            color,
            bg_color: color,
        });
        self.overlay_vertices.push(GlyphVertex {
            position: [right, top],
            uv: [0.0, 0.0],
            color,
            bg_color: color,
        });
        self.overlay_vertices.push(GlyphVertex {
            position: [right, bottom],
            uv: [0.0, 0.0],
            color,
            bg_color: color,
        });
        self.overlay_vertices.push(GlyphVertex {
            position: [left, bottom],
            uv: [0.0, 0.0],
            color,
            bg_color: color,
        });
        self.overlay_indices.extend_from_slice(&[
            base_idx, base_idx + 1, base_idx + 2,
            base_idx, base_idx + 2, base_idx + 3,
        ]);
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
    /// - `statusline_sections`: Sections to render in the statusline
    pub fn render_panes(
        &mut self,
        panes: &[(&Terminal, PaneRenderInfo, Option<(usize, usize, usize, usize)>)],
        num_tabs: usize,
        active_tab: usize,
        edge_glows: &[EdgeGlow],
        edge_glow_intensity: f32,
        statusline_sections: &[StatuslineSection],
    ) -> Result<(), wgpu::SurfaceError> {
        // Sync palette from first terminal
        if let Some((terminal, _, _)) = panes.first() {
            self.palette = terminal.palette.clone();
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
        self.overlay_vertices.clear();
        self.overlay_indices.clear();

        let width = self.width as f32;
        let height = self.height as f32;
        let tab_bar_height = self.tab_bar_height();
        let terminal_y_offset = self.terminal_y_offset();

        // ═══════════════════════════════════════════════════════════════════
        // RENDER TAB BAR (same as render_from_terminal)
        // ═══════════════════════════════════════════════════════════════════
        if self.tab_bar_position != TabBarPosition::Hidden && num_tabs > 0 {
            let tab_bar_y = match self.tab_bar_position {
                TabBarPosition::Top => 0.0,
                TabBarPosition::Bottom => height - tab_bar_height,
                TabBarPosition::Hidden => unreachable!(),
            };

            let tab_bar_bg = {
                let [r, g, b] = self.palette.default_bg;
                let factor = 0.85_f32;
                [
                    Self::srgb_to_linear((r as f32 / 255.0) * factor),
                    Self::srgb_to_linear((g as f32 / 255.0) * factor),
                    Self::srgb_to_linear((b as f32 / 255.0) * factor),
                    1.0,
                ]
            };

            // Draw tab bar background
            self.render_rect(0.0, tab_bar_y, width, tab_bar_height, tab_bar_bg);

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
                    let [r, g, b] = self.palette.default_bg;
                    [
                        Self::srgb_to_linear(r as f32 / 255.0),
                        Self::srgb_to_linear(g as f32 / 255.0),
                        Self::srgb_to_linear(b as f32 / 255.0),
                        1.0,
                    ]
                } else {
                    tab_bar_bg
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
                        let char_x = text_x + char_idx as f32 * self.cell_width;
                        let baseline_y = (text_y + self.cell_height * 0.8).round();
                        let glyph_x = (char_x + glyph.offset[0]).round();
                        let glyph_y = (baseline_y - glyph.offset[1] - glyph.size[1]).round();

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
        // RENDER STATUSLINE
        // ═══════════════════════════════════════════════════════════════════
        {
            let statusline_y = self.statusline_y();
            let statusline_height = self.statusline_height();
            
            // Statusline background (slightly different shade than tab bar)
            let statusline_bg = {
                let [r, g, b] = self.palette.default_bg;
                let factor = 0.9_f32;
                [
                    Self::srgb_to_linear((r as f32 / 255.0) * factor),
                    Self::srgb_to_linear((g as f32 / 255.0) * factor),
                    Self::srgb_to_linear((b as f32 / 255.0) * factor),
                    1.0,
                ]
            };
            
            // Draw statusline background
            self.render_rect(0.0, statusline_y, width, statusline_height, statusline_bg);
            
            // Render statusline sections
            let text_y = statusline_y + (statusline_height - self.cell_height) / 2.0;
            let mut cursor_x = 0.0_f32;
            
            // Helper to convert StatuslineColor to linear RGBA
            let color_to_rgba = |color: StatuslineColor, palette: &crate::terminal::ColorPalette| -> [f32; 4] {
                match color {
                    StatuslineColor::Default => {
                        let [r, g, b] = palette.default_fg;
                        [
                            Self::srgb_to_linear(r as f32 / 255.0),
                            Self::srgb_to_linear(g as f32 / 255.0),
                            Self::srgb_to_linear(b as f32 / 255.0),
                            1.0,
                        ]
                    }
                    StatuslineColor::Indexed(idx) => {
                        let [r, g, b] = palette.colors[idx as usize];
                        [
                            Self::srgb_to_linear(r as f32 / 255.0),
                            Self::srgb_to_linear(g as f32 / 255.0),
                            Self::srgb_to_linear(b as f32 / 255.0),
                            1.0,
                        ]
                    }
                    StatuslineColor::Rgb(r, g, b) => {
                        [
                            Self::srgb_to_linear(r as f32 / 255.0),
                            Self::srgb_to_linear(g as f32 / 255.0),
                            Self::srgb_to_linear(b as f32 / 255.0),
                            1.0,
                        ]
                    }
                }
            };
            
            for (section_idx, section) in statusline_sections.iter().enumerate() {
                let section_start_x = cursor_x;
                
                // Calculate section width by counting characters
                let mut section_char_count = 0usize;
                for component in &section.components {
                    section_char_count += component.text.chars().count();
                }
                let section_width = section_char_count as f32 * self.cell_width;
                
                // Draw section background if it has a color (Indexed or Rgb)
                let has_bg = matches!(section.bg, StatuslineColor::Indexed(_) | StatuslineColor::Rgb(_, _, _));
                if has_bg {
                    let section_bg_color = color_to_rgba(section.bg, &self.palette);
                    self.render_rect(section_start_x, statusline_y, section_width, statusline_height, section_bg_color);
                }
                
                // Render components within this section
                for component in &section.components {
                    let component_fg = color_to_rgba(component.fg, &self.palette);
                    
                    for c in component.text.chars() {
                        if cursor_x + self.cell_width > width {
                            break;
                        }
                        
                        if c == ' ' {
                            cursor_x += self.cell_width;
                            continue;
                        }
                        
                        let glyph = self.rasterize_char(c);
                        if glyph.size[0] > 0.0 && glyph.size[1] > 0.0 {
                            let is_powerline_char = ('\u{E0B0}'..='\u{E0BF}').contains(&c);
                            
                            let glyph_x = (cursor_x + glyph.offset[0]).round();
                            let glyph_y = if is_powerline_char {
                                statusline_y
                            } else {
                                let baseline_y = (text_y + self.cell_height * 0.8).round();
                                (baseline_y - glyph.offset[1] - glyph.size[1]).round()
                            };

                            let left = Self::pixel_to_ndc_x(glyph_x, width);
                            let right = Self::pixel_to_ndc_x(glyph_x + glyph.size[0], width);
                            let top = Self::pixel_to_ndc_y(glyph_y, height);
                            let bottom = Self::pixel_to_ndc_y(glyph_y + glyph.size[1], height);

                            let base_idx = self.glyph_vertices.len() as u32;
                            self.glyph_vertices.push(GlyphVertex {
                                position: [left, top],
                                uv: [glyph.uv[0], glyph.uv[1]],
                                color: component_fg,
                                bg_color: [0.0, 0.0, 0.0, 0.0],
                            });
                            self.glyph_vertices.push(GlyphVertex {
                                position: [right, top],
                                uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1]],
                                color: component_fg,
                                bg_color: [0.0, 0.0, 0.0, 0.0],
                            });
                            self.glyph_vertices.push(GlyphVertex {
                                position: [right, bottom],
                                uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1] + glyph.uv[3]],
                                color: component_fg,
                                bg_color: [0.0, 0.0, 0.0, 0.0],
                            });
                            self.glyph_vertices.push(GlyphVertex {
                                position: [left, bottom],
                                uv: [glyph.uv[0], glyph.uv[1] + glyph.uv[3]],
                                color: component_fg,
                                bg_color: [0.0, 0.0, 0.0, 0.0],
                            });
                            self.glyph_indices.extend_from_slice(&[
                                base_idx, base_idx + 1, base_idx + 2,
                                base_idx, base_idx + 2, base_idx + 3,
                            ]);
                        }
                        
                        cursor_x += self.cell_width;
                    }
                }
                
                // Draw powerline transition arrow at end of section (if section has a background)
                if has_bg {
                    // Determine the next section's background color (or statusline bg if last section)
                    let next_bg = if section_idx + 1 < statusline_sections.len() {
                        color_to_rgba(statusline_sections[section_idx + 1].bg, &self.palette)
                    } else {
                        statusline_bg
                    };
                    
                    // The arrow's foreground is this section's background
                    let arrow_fg = color_to_rgba(section.bg, &self.palette);
                    
                    // Render the powerline arrow (U+E0B0)
                    let arrow_char = '\u{E0B0}';
                    let glyph = self.rasterize_char(arrow_char);
                    if glyph.size[0] > 0.0 && glyph.size[1] > 0.0 {
                        let glyph_x = (cursor_x + glyph.offset[0]).round();
                        let glyph_y = statusline_y; // Powerline chars at top
                        
                        // Draw background rectangle for the arrow cell
                        self.render_rect(cursor_x, statusline_y, self.cell_width, statusline_height, next_bg);

                        let left = Self::pixel_to_ndc_x(glyph_x, width);
                        let right = Self::pixel_to_ndc_x(glyph_x + glyph.size[0], width);
                        let top = Self::pixel_to_ndc_y(glyph_y, height);
                        let bottom = Self::pixel_to_ndc_y(glyph_y + glyph.size[1], height);

                        let base_idx = self.glyph_vertices.len() as u32;
                        self.glyph_vertices.push(GlyphVertex {
                            position: [left, top],
                            uv: [glyph.uv[0], glyph.uv[1]],
                            color: arrow_fg,
                            bg_color: [0.0, 0.0, 0.0, 0.0],
                        });
                        self.glyph_vertices.push(GlyphVertex {
                            position: [right, top],
                            uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1]],
                            color: arrow_fg,
                            bg_color: [0.0, 0.0, 0.0, 0.0],
                        });
                        self.glyph_vertices.push(GlyphVertex {
                            position: [right, bottom],
                            uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1] + glyph.uv[3]],
                            color: arrow_fg,
                            bg_color: [0.0, 0.0, 0.0, 0.0],
                        });
                        self.glyph_vertices.push(GlyphVertex {
                            position: [left, bottom],
                            uv: [glyph.uv[0], glyph.uv[1] + glyph.uv[3]],
                            color: arrow_fg,
                            bg_color: [0.0, 0.0, 0.0, 0.0],
                        });
                        self.glyph_indices.extend_from_slice(&[
                            base_idx, base_idx + 1, base_idx + 2,
                            base_idx, base_idx + 2, base_idx + 3,
                        ]);
                    }
                    
                    cursor_x += self.cell_width;
                }
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
        // The layout leaves a gap between panes, so we look for gaps and draw borders there
        if panes.len() > 1 {
            // Maximum gap size to consider as "adjacent" (layout uses border_width gap)
            let max_gap = 20.0;

            // Check each pair of panes to find adjacent ones with gaps
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

                    // Calculate absolute positions (with terminal_y_offset)
                    let a_x = info_a.x;
                    let a_y = terminal_y_offset + info_a.y;
                    let a_right = a_x + info_a.width;
                    let a_bottom = a_y + info_a.height;

                    let b_x = info_b.x;
                    let b_y = terminal_y_offset + info_b.y;
                    let b_right = b_x + info_b.width;
                    let b_bottom = b_y + info_b.height;

                    // Check for vertical adjacency (horizontal gap between panes)
                    // Pane A is to the left of pane B
                    let h_gap_ab = b_x - a_right;
                    if h_gap_ab > 0.0 && h_gap_ab < max_gap {
                        // Check if they overlap vertically
                        let top = a_y.max(b_y);
                        let bottom = a_bottom.min(b_bottom);
                        if bottom > top {
                            // Draw vertical border in the gap
                            let border_x = a_right + (h_gap_ab - border_thickness) / 2.0;
                            self.render_rect(border_x, top, border_thickness, bottom - top, border_color);
                        }
                    }
                    // Pane B is to the left of pane A
                    let h_gap_ba = a_x - b_right;
                    if h_gap_ba > 0.0 && h_gap_ba < max_gap {
                        let top = a_y.max(b_y);
                        let bottom = a_bottom.min(b_bottom);
                        if bottom > top {
                            let border_x = b_right + (h_gap_ba - border_thickness) / 2.0;
                            self.render_rect(border_x, top, border_thickness, bottom - top, border_color);
                        }
                    }

                    // Check for horizontal adjacency (vertical gap between panes)
                    // Pane A is above pane B
                    let v_gap_ab = b_y - a_bottom;
                    if v_gap_ab > 0.0 && v_gap_ab < max_gap {
                        // Check if they overlap horizontally
                        let left = a_x.max(b_x);
                        let right = a_right.min(b_right);
                        if right > left {
                            // Draw horizontal border in the gap
                            let border_y = a_bottom + (v_gap_ab - border_thickness) / 2.0;
                            self.render_rect(left, border_y, right - left, border_thickness, border_color);
                        }
                    }
                    // Pane B is above pane A
                    let v_gap_ba = a_y - b_bottom;
                    if v_gap_ba > 0.0 && v_gap_ba < max_gap {
                        let left = a_x.max(b_x);
                        let right = a_right.min(b_right);
                        if right > left {
                            let border_y = b_bottom + (v_gap_ba - border_thickness) / 2.0;
                            self.render_rect(left, border_y, right - left, border_thickness, border_color);
                        }
                    }
                }
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // RENDER EACH PANE'S CONTENT
        // ═══════════════════════════════════════════════════════════════════
        for (terminal, info, selection) in panes {
            // No content offset needed - borders are drawn at shared edges only
            let pane_x = info.x;
            let pane_y = terminal_y_offset + info.y;
            let pane_width = info.width;
            let pane_height = info.height;

            self.render_pane_content(
                terminal,
                pane_x,
                pane_y,
                pane_width,
                pane_height,
                info.is_active,
                *selection,
                info.dim_factor,
            );

            // Draw dimming overlay for inactive panes
            // dim_factor of 1.0 = no dimming, dim_factor of 0.6 = 40% dark overlay
            if info.dim_factor < 1.0 {
                let overlay_alpha = 1.0 - info.dim_factor;
                let overlay_color = [0.0, 0.0, 0.0, overlay_alpha];
                self.render_overlay_rect(pane_x, pane_y, pane_width, pane_height, overlay_color);
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // PREPARE IMAGE RENDERS (Kitty Graphics Protocol)
        // ═══════════════════════════════════════════════════════════════════
        let mut image_renders: Vec<(u32, ImageUniforms)> = Vec::new();
        for (terminal, info, _) in panes {
            let pane_x = info.x;
            let pane_y = terminal_y_offset + info.y;

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
        let overlay_vertex_count = self.overlay_vertices.len();
        let total_vertex_count = bg_vertex_count + glyph_vertex_count + overlay_vertex_count;
        let total_index_count = self.bg_indices.len() + self.glyph_indices.len() + self.overlay_indices.len();

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

        // Upload vertices: bg, then glyph, then overlay
        self.queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&self.bg_vertices));
        self.queue.write_buffer(
            &self.vertex_buffer,
            (bg_vertex_count * std::mem::size_of::<GlyphVertex>()) as u64,
            bytemuck::cast_slice(&self.glyph_vertices),
        );

        if !self.overlay_vertices.is_empty() {
            self.queue.write_buffer(
                &self.vertex_buffer,
                ((bg_vertex_count + glyph_vertex_count) * std::mem::size_of::<GlyphVertex>()) as u64,
                bytemuck::cast_slice(&self.overlay_vertices),
            );
        }

        // Upload indices: bg, then glyph (adjusted), then overlay (adjusted)
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

        let overlay_vertex_offset = (bg_vertex_count + glyph_vertex_count) as u32;
        let glyph_index_bytes = self.glyph_indices.len() * std::mem::size_of::<u32>();

        if !self.overlay_indices.is_empty() {
            let adjusted_indices: Vec<u32> = self.overlay_indices.iter()
                .map(|i| i + overlay_vertex_offset)
                .collect();
            self.queue.write_buffer(
                &self.index_buffer,
                (bg_index_bytes + glyph_index_bytes) as u64,
                bytemuck::cast_slice(&adjusted_indices),
            );
        }

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
                    bytes_per_row: Some(ATLAS_SIZE),
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
            render_pass.draw_indexed(0..total_index_count as u32, 0, 0..1);
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
                    &image.data,
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
            &image.data,
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
