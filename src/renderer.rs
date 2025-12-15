//! GPU-accelerated terminal rendering using wgpu with a glyph atlas.
//! Uses rustybuzz (HarfBuzz port) for text shaping to support font features.

use crate::config::TabBarPosition;
use crate::terminal::{Color, ColorPalette, CursorShape, Terminal};
use fontdue::Font as FontdueFont;
use rustybuzz::UnicodeBuffer;
use ttf_parser::Tag;
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
    features: Vec<rustybuzz::Feature>,
}

/// Result of shaping a text sequence.
#[derive(Clone, Debug)]
struct ShapedGlyphs {
    /// Glyph IDs, advances, and cluster indices.
    /// Each tuple is (glyph_id, advance, cluster).
    glyphs: Vec<(u16, f32, u32)>,
    /// Whether this represents a ligature (one visual glyph for multiple characters).
    is_ligature: bool,
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

/// The terminal renderer.
pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    surface_config: wgpu::SurfaceConfiguration,

    // Glyph rendering pipeline
    glyph_pipeline: wgpu::RenderPipeline,
    glyph_bind_group: wgpu::BindGroup,

    // Atlas texture
    atlas_texture: wgpu::Texture,
    atlas_data: Vec<u8>,
    atlas_dirty: bool,

    // Font and shaping
    #[allow(dead_code)] // Kept alive for rustybuzz::Face which borrows it
    font_data: Box<[u8]>,
    fontdue_font: FontdueFont,
    fallback_fonts: Vec<FontdueFont>,
    /// Fontconfig handle for dynamic font discovery
    fontconfig: Option<Fontconfig>,
    /// Set of font paths we've already tried (to avoid reloading)
    tried_font_paths: HashSet<PathBuf>,
    shaping_ctx: ShapingContext,
    char_cache: HashMap<char, GlyphInfo>,    // cache char -> rendered glyph
    ligature_cache: HashMap<String, ShapedGlyphs>, // cache multi-char -> shaped glyphs
    glyph_cache: HashMap<(usize, u16), GlyphInfo>,   // keyed by (font_index, glyph ID)
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

        // Load primary font
        let font_data: Box<[u8]> = std::fs::read("/usr/share/fonts/TTF/0xProtoNerdFont-Regular.ttf")
            .or_else(|_| std::fs::read("/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Regular.ttf"))
            .or_else(|_| std::fs::read("/usr/share/fonts/TTF/JetBrainsMono-Regular.ttf"))
            .or_else(|_| std::fs::read("/usr/share/fonts/noto/NotoSansMono-Regular.ttf"))
            .expect("Failed to load any monospace font")
            .into_boxed_slice();

        let fontdue_font = FontdueFont::from_bytes(
            &font_data[..],
            fontdue::FontSettings::default(),
        )
        .expect("Failed to parse font with fontdue");

        // Initialize fontconfig for dynamic font fallback
        let fontconfig = Fontconfig::new();
        if fontconfig.is_none() {
            log::warn!("Failed to initialize fontconfig - Unicode fallback may not work");
        }
        
        // Start with empty fallback fonts - will be loaded on-demand via fontconfig
        let fallback_fonts: Vec<FontdueFont> = Vec::new();
        let tried_font_paths: HashSet<PathBuf> = HashSet::new();

        // Create rustybuzz Face for text shaping (ligatures).
        // SAFETY: We transmute to 'static because font_data lives as long as Renderer.
        // The Face only borrows the data, so this is safe as long as we don't drop font_data
        // before dropping the Face, which is guaranteed by struct drop order.
        let face: rustybuzz::Face<'static> = {
            let face = rustybuzz::Face::from_slice(&font_data, 0)
                .expect("Failed to parse font for shaping");
            unsafe { std::mem::transmute(face) }
        };
        
        // Enable OpenType features for ligatures and contextual alternates
        // These are the standard features used by coding fonts like Fira Code, JetBrains Mono, etc.
        let features = vec![
            // Standard ligatures (fi, fl, etc.)
            rustybuzz::Feature::new(Tag::from_bytes(b"liga"), 1, ..),
            // Contextual alternates (programming ligatures like ->, =>, etc.)
            rustybuzz::Feature::new(Tag::from_bytes(b"calt"), 1, ..),
            // Discretionary ligatures (optional ligatures)
            rustybuzz::Feature::new(Tag::from_bytes(b"dlig"), 1, ..),
        ];
        let shaping_ctx = ShapingContext { face, features };

        // Calculate cell dimensions from font metrics
        // Scale font size by the display scale factor for crisp rendering
        let base_font_size = config.font_size;
        let font_size = base_font_size * scale_factor as f32;
        let metrics = fontdue_font.metrics('M', font_size);
        let cell_width = metrics.advance_width.ceil();
        
        // Use actual font line metrics for cell height (matching Kitty's approach)
        // Kitty uses the font's "height" metric which is: ascent - descent + line_gap
        // In fontdue, this is provided as "new_line_size"
        let cell_height = if let Some(line_metrics) = fontdue_font.horizontal_line_metrics(font_size) {
            line_metrics.new_line_size
        } else {
            // Fallback if no line metrics available
            font_size * 1.2
        };

        log::info!("Scale factor: {}, font size: {}pt -> {}px, cell: {}x{}", 
                   scale_factor, base_font_size, font_size, cell_width, cell_height);

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
            atlas_texture,
            atlas_data: vec![0u8; (ATLAS_SIZE * ATLAS_SIZE) as usize],
            atlas_dirty: false,
            font_data,
            fontdue_font,
            fallback_fonts,
            fontconfig,
            tried_font_paths,
            shaping_ctx,
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

    /// Returns the Y offset where the terminal content starts.
    pub fn terminal_y_offset(&self) -> f32 {
        match self.tab_bar_position {
            TabBarPosition::Top => self.tab_bar_height(),
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

    /// Calculates terminal dimensions in cells, accounting for tab bar.
    pub fn terminal_size(&self) -> (usize, usize) {
        let available_height = self.height as f32 - self.tab_bar_height();
        let cols = (self.width as f32 / self.cell_width).floor() as usize;
        let rows = (available_height / self.cell_height).floor() as usize;
        (cols.max(1), rows.max(1))
    }
    
    /// Converts a pixel position to a terminal cell position.
    /// Returns None if the position is outside the terminal area (e.g., in the tab bar).
    pub fn pixel_to_cell(&self, x: f64, y: f64) -> Option<(usize, usize)> {
        let terminal_y_offset = self.terminal_y_offset();
        let tab_bar_height = self.tab_bar_height();
        let height = self.height as f32;
        
        // Check if position is in the tab bar area (which could be at top or bottom)
        match self.tab_bar_position {
            TabBarPosition::Top => {
                if (y as f32) < tab_bar_height {
                    return None;
                }
            }
            TabBarPosition::Bottom => {
                if (y as f32) >= height - tab_bar_height {
                    return None;
                }
            }
            TabBarPosition::Hidden => {}
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
        self.font_size = self.base_font_size * new_scale as f32;

        // Recalculate cell dimensions
        let metrics = self.fontdue_font.metrics('M', self.font_size);
        self.cell_width = metrics.advance_width.ceil();
        self.cell_height = if let Some(line_metrics) = self.fontdue_font.horizontal_line_metrics(self.font_size) {
            line_metrics.new_line_size
        } else {
            self.font_size * 1.2
        };

        log::info!(
            "Scale factor changed to {}: font {}pt -> {}px, cell: {}x{}",
            new_scale, self.base_font_size, self.font_size, self.cell_width, self.cell_height
        );

        // Clear all glyph caches - they were rendered at the old size
        self.char_cache.clear();
        self.ligature_cache.clear();
        self.glyph_cache.clear();

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

        // Try primary font first, then fallbacks
        let (metrics, bitmap) = {
            // Check if primary font has this glyph
            let glyph_idx = self.fontdue_font.lookup_glyph_index(c);
            if glyph_idx != 0 {
                self.fontdue_font.rasterize(c, self.font_size)
            } else {
                // Try already-loaded fallback fonts first
                let mut result = None;
                for fallback in &self.fallback_fonts {
                    let fb_glyph_idx = fallback.lookup_glyph_index(c);
                    if fb_glyph_idx != 0 {
                        result = Some(fallback.rasterize(c, self.font_size));
                        break;
                    }
                }
                
                // If no cached fallback has the glyph, use fontconfig to find one
                if result.is_none() {
                    if let Some(ref fc) = self.fontconfig {
                        // Query fontconfig for a font that has this character
                        if let Some(path) = find_font_for_char(fc, c) {
                            // Only load if we haven't tried this path before
                            if !self.tried_font_paths.contains(&path) {
                                self.tried_font_paths.insert(path.clone());
                                
                                if let Ok(data) = std::fs::read(&path) {
                                    if let Ok(font) = FontdueFont::from_bytes(
                                        data.as_slice(),
                                        fontdue::FontSettings::default(),
                                    ) {
                                        log::debug!("Loaded fallback font via fontconfig: {}", path.display());
                                        
                                        // Check if this font actually has the glyph
                                        let fb_glyph_idx = font.lookup_glyph_index(c);
                                        if fb_glyph_idx != 0 {
                                            result = Some(font.rasterize(c, self.font_size));
                                        }
                                        
                                        // Cache the font for future use
                                        self.fallback_fonts.push(font);
                                    }
                                }
                            }
                        }
                    }
                }
                
                // Use primary font's .notdef if no fallback has the glyph
                result.unwrap_or_else(|| self.fontdue_font.rasterize(c, self.font_size))
            }
        };

        if bitmap.is_empty() || metrics.width == 0 || metrics.height == 0 {
            // Empty glyph (e.g., space)
            let info = GlyphInfo {
                uv: [0.0, 0.0, 0.0, 0.0],
                offset: [0.0, 0.0],
                size: [0.0, 0.0],
            };
            self.char_cache.insert(c, info);
            return info;
        }

        let glyph_width = metrics.width as u32;
        let glyph_height = metrics.height as u32;

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
        for y in 0..metrics.height {
            for x in 0..metrics.width {
                let src_idx = y * metrics.width + x;
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
            offset: [metrics.xmin as f32, metrics.ymin as f32],
            size: [glyph_width as f32, glyph_height as f32],
        };

        // Update atlas cursor
        self.atlas_cursor_x += glyph_width + 1;
        self.atlas_row_height = self.atlas_row_height.max(glyph_height);

        self.char_cache.insert(c, info);
        info
    }

    /// Get or rasterize a glyph by its glyph ID from the primary font.
    /// Used for ligatures where we have the glyph ID from rustybuzz.
    fn get_glyph_by_id(&mut self, glyph_id: u16) -> GlyphInfo {
        let cache_key = (0usize, glyph_id); // font index 0 = primary font
        if let Some(info) = self.glyph_cache.get(&cache_key) {
            return *info;
        }

        // Rasterize the glyph by ID from primary font
        let (metrics, bitmap) = self.fontdue_font.rasterize_indexed(glyph_id, self.font_size);

        if bitmap.is_empty() || metrics.width == 0 || metrics.height == 0 {
            // Empty glyph (e.g., space)
            let info = GlyphInfo {
                uv: [0.0, 0.0, 0.0, 0.0],
                offset: [0.0, 0.0],
                size: [0.0, 0.0],
            };
            self.glyph_cache.insert(cache_key, info);
            return info;
        }

        let glyph_width = metrics.width as u32;
        let glyph_height = metrics.height as u32;

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
        for y in 0..metrics.height {
            for x in 0..metrics.width {
                let src_idx = y * metrics.width + x;
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
            offset: [metrics.xmin as f32, metrics.ymin as f32],
            size: [glyph_width as f32, glyph_height as f32],
        };

        // Update atlas cursor
        self.atlas_cursor_x += glyph_width + 1;
        self.atlas_row_height = self.atlas_row_height.max(glyph_height);

        self.glyph_cache.insert(cache_key, info);
        info
    }

    /// Shape a single character to get its default glyph ID.
    /// Used for ligature detection by comparing combined vs individual shaping.
    fn shape_single_char(&self, c: char) -> u16 {
        let mut buffer = UnicodeBuffer::new();
        buffer.push_str(&c.to_string());
        let glyph_buffer = rustybuzz::shape(&self.shaping_ctx.face, &[], buffer);
        let infos = glyph_buffer.glyph_infos();
        if infos.is_empty() {
            0
        } else {
            infos[0].glyph_id as u16
        }
    }

    /// Shape a multi-character text string (for ligatures).
    /// Detects if the font produces a ligature by comparing glyph IDs
    /// when shaped together vs shaped individually.
    fn shape_text(&mut self, text: &str) -> ShapedGlyphs {
        // Check cache first
        if let Some(cached) = self.ligature_cache.get(text) {
            return cached.clone();
        }

        let chars: Vec<char> = text.chars().collect();
        let char_count = chars.len();
        
        let mut buffer = UnicodeBuffer::new();
        buffer.push_str(text);

        // Shape with OpenType features enabled (liga, calt, dlig)
        let glyph_buffer = rustybuzz::shape(&self.shaping_ctx.face, &self.shaping_ctx.features, buffer);
        let glyph_infos = glyph_buffer.glyph_infos();
        let glyph_positions = glyph_buffer.glyph_positions();

        let glyphs: Vec<(u16, f32, u32)> = glyph_infos
            .iter()
            .zip(glyph_positions.iter())
            .map(|(info, pos)| {
                let glyph_id = info.glyph_id as u16;
                // Ensure glyph is rasterized
                self.get_glyph_by_id(glyph_id);
                // Convert advance from font units to pixels
                let advance = pos.x_advance as f32 * self.font_size / self.shaping_ctx.face.units_per_em() as f32;
                (glyph_id, advance, info.cluster)
            })
            .collect();

        // Get individual glyph IDs for comparison
        let individual_glyphs: Vec<u16> = chars.iter().map(|&c| self.shape_single_char(c)).collect();

        // Detect ligature by comparing combined vs individual shaping
        // A ligature occurred if:
        // 1. Fewer glyphs than input characters, OR
        // 2. Any glyph ID differs from the individual character's glyph ID
        let fewer_glyphs = glyphs.len() < char_count;
        
        let has_substitution = if glyphs.len() == char_count {
            glyphs.iter().zip(individual_glyphs.iter())
                .any(|((combined_id, _, _), &individual_id)| *combined_id != individual_id)
        } else {
            true // If glyph count differs, it's definitely a substitution
        };
        
        let is_ligature = fewer_glyphs || has_substitution;

        // Debug: log shaping results for ligature patterns
        log::debug!(
            "shape_text: '{}' ({} chars) -> {} glyphs, is_ligature={}, combined={:?}, individual={:?}",
            text,
            char_count,
            glyphs.len(),
            is_ligature,
            glyphs.iter().map(|(id, _, _)| *id).collect::<Vec<_>>(),
            individual_glyphs
        );

        let shaped = ShapedGlyphs { 
            glyphs, 
            is_ligature,
        };
        self.ligature_cache.insert(text.to_string(), shaped.clone());
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
        
        // Common programming ligatures to check (longest first for greedy matching)
        const LIGATURE_PATTERNS: &[&str] = &[
            // 3-char
            "===", "!==", ">>>", "<<<", "||=", "&&=", "??=", "...", "-->", "<--", "<->",
            "www",
            // 2-char  
            "=>", "->", "<-", ">=", "<=", "==", "!=", "::", "&&", "||", "??", "..", "++",
            "--", "<<", ">>", "|>", "<|", "/*", "*/", "//", "##", ":=", "~=", "<>",
        ];
        
        // Render each row
        for (row_idx, row) in visible_rows.iter().enumerate() {
            if row_idx >= rows {
                break;
            }
            
            // Find the last non-empty cell for selection clipping
            let last_content_col = row.iter()
                .enumerate()
                .rev()
                .find(|(_, cell)| cell.character != ' ' && cell.character != '\0')
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            
            let mut col_idx = 0;
            while col_idx < row.len() && col_idx < cols {
                let cell = &row[col_idx];
                let cell_x = pane_x + col_idx as f32 * self.cell_width;
                let cell_y = pane_y + row_idx as f32 * self.cell_height;
                
                let mut fg_color = color_to_rgba(&cell.fg_color, true);
                let mut bg_color = color_to_rgba(&cell.bg_color, false);
                
                // Handle selection
                if is_cell_selected(col_idx, row_idx) && col_idx <= last_content_col {
                    fg_color = [0.0, 0.0, 0.0, 1.0];
                    bg_color = [1.0, 1.0, 1.0, 1.0];
                }
                
                // Check for ligatures by looking ahead
                let mut ligature_len = 0;
                let mut ligature_shaped: Option<ShapedGlyphs> = None;
                
                for pattern in LIGATURE_PATTERNS {
                    let pat_len = pattern.len();
                    if col_idx + pat_len <= row.len() {
                        // Build the candidate string from consecutive cells
                        let candidate: String = row[col_idx..col_idx + pat_len]
                            .iter()
                            .map(|c| c.character)
                            .collect();
                        
                        if candidate == *pattern {
                            // Only form ligatures from cells with matching foreground colors.
                            // This prevents ghost/completion text (which typically has a
                            // different color) from being combined with typed text.
                            let first_fg = &row[col_idx].fg_color;
                            let all_same_color = row[col_idx..col_idx + pat_len]
                                .iter()
                                .all(|c| &c.fg_color == first_fg);
                            
                            if !all_same_color {
                                continue;
                            }
                            
                            // Check if font actually produces a ligature
                            let shaped = self.shape_text(&candidate);
                            // Use our improved ligature detection
                            if shaped.is_ligature {
                                ligature_shaped = Some(shaped);
                                ligature_len = pat_len;
                                break;
                            }
                        }
                    }
                }
                
                if let Some(shaped) = ligature_shaped {
                    // Render ligature spanning multiple cells
                    // Add background for all cells in the ligature
                    for i in 0..ligature_len {
                        let bg_cell_x = pane_x + (col_idx + i) as f32 * self.cell_width;
                        let cell_left = Self::pixel_to_ndc_x(bg_cell_x, width);
                        let cell_right = Self::pixel_to_ndc_x(bg_cell_x + self.cell_width, width);
                        let cell_top = Self::pixel_to_ndc_y(cell_y, height);
                        let cell_bottom = Self::pixel_to_ndc_y(cell_y + self.cell_height, height);

                        let base_idx = self.bg_vertices.len() as u32;
                        self.bg_vertices.push(GlyphVertex {
                            position: [cell_left, cell_top],
                            uv: [0.0, 0.0],
                            color: fg_color,
                            bg_color,
                        });
                        self.bg_vertices.push(GlyphVertex {
                            position: [cell_right, cell_top],
                            uv: [0.0, 0.0],
                            color: fg_color,
                            bg_color,
                        });
                        self.bg_vertices.push(GlyphVertex {
                            position: [cell_right, cell_bottom],
                            uv: [0.0, 0.0],
                            color: fg_color,
                            bg_color,
                        });
                        self.bg_vertices.push(GlyphVertex {
                            position: [cell_left, cell_bottom],
                            uv: [0.0, 0.0],
                            color: fg_color,
                            bg_color,
                        });
                        self.bg_indices.extend_from_slice(&[
                            base_idx, base_idx + 1, base_idx + 2,
                            base_idx, base_idx + 2, base_idx + 3,
                        ]);
                    }

                    // Render all glyphs from the shaped output with their proper advances.
                    // For ligatures like "->", the font produces:
                    // - Glyph 0: spacer (0x0 invisible), advance = cell_width
                    // - Glyph 1: ligature glyph with negative xmin to extend back into cell 0
                    let baseline_y = (cell_y + self.cell_height * 0.8).round();
                    let mut cursor_x = cell_x;
                    
                    for &(glyph_id, advance, _cluster) in &shaped.glyphs {
                        let glyph = self.get_glyph_by_id(glyph_id);
                        
                        // Only render if glyph has content (spacers are 0x0)
                        if glyph.size[0] > 0.0 && glyph.size[1] > 0.0 {
                            // Use the glyph's horizontal offset (xmin) - this allows
                            // ligature glyphs with negative xmin to extend backwards
                            let glyph_x = (cursor_x + glyph.offset[0]).round();
                            let glyph_y = (baseline_y - glyph.offset[1] - glyph.size[1]).round();

                            let left = Self::pixel_to_ndc_x(glyph_x, width);
                            let right = Self::pixel_to_ndc_x(glyph_x + glyph.size[0], width);
                            let top = Self::pixel_to_ndc_y(glyph_y, height);
                            let bottom = Self::pixel_to_ndc_y(glyph_y + glyph.size[1], height);

                            let base_idx = self.glyph_vertices.len() as u32;
                            self.glyph_vertices.push(GlyphVertex {
                                position: [left, top],
                                uv: [glyph.uv[0], glyph.uv[1]],
                                color: fg_color,
                                bg_color: [0.0, 0.0, 0.0, 0.0],
                            });
                            self.glyph_vertices.push(GlyphVertex {
                                position: [right, top],
                                uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1]],
                                color: fg_color,
                                bg_color: [0.0, 0.0, 0.0, 0.0],
                            });
                            self.glyph_vertices.push(GlyphVertex {
                                position: [right, bottom],
                                uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1] + glyph.uv[3]],
                                color: fg_color,
                                bg_color: [0.0, 0.0, 0.0, 0.0],
                            });
                            self.glyph_vertices.push(GlyphVertex {
                                position: [left, bottom],
                                uv: [glyph.uv[0], glyph.uv[1] + glyph.uv[3]],
                                color: fg_color,
                                bg_color: [0.0, 0.0, 0.0, 0.0],
                            });
                            self.glyph_indices.extend_from_slice(&[
                                base_idx, base_idx + 1, base_idx + 2,
                                base_idx, base_idx + 2, base_idx + 3,
                            ]);
                        }
                        
                        // Advance cursor for next glyph
                        cursor_x += advance;
                    }
                    
                    // Skip the cells consumed by the ligature
                    col_idx += ligature_len;
                    continue;
                }
                
                // No ligature - render single cell
                
                // Cell bounds
                let cell_left = Self::pixel_to_ndc_x(cell_x, width);
                let cell_right = Self::pixel_to_ndc_x(cell_x + self.cell_width, width);
                let cell_top = Self::pixel_to_ndc_y(cell_y, height);
                let cell_bottom = Self::pixel_to_ndc_y(cell_y + self.cell_height, height);
                
                // Add background quad
                let base_idx = self.bg_vertices.len() as u32;
                self.bg_vertices.push(GlyphVertex {
                    position: [cell_left, cell_top],
                    uv: [0.0, 0.0],
                    color: fg_color,
                    bg_color,
                });
                self.bg_vertices.push(GlyphVertex {
                    position: [cell_right, cell_top],
                    uv: [0.0, 0.0],
                    color: fg_color,
                    bg_color,
                });
                self.bg_vertices.push(GlyphVertex {
                    position: [cell_right, cell_bottom],
                    uv: [0.0, 0.0],
                    color: fg_color,
                    bg_color,
                });
                self.bg_vertices.push(GlyphVertex {
                    position: [cell_left, cell_bottom],
                    uv: [0.0, 0.0],
                    color: fg_color,
                    bg_color,
                });
                self.bg_indices.extend_from_slice(&[
                    base_idx, base_idx + 1, base_idx + 2,
                    base_idx, base_idx + 2, base_idx + 3,
                ]);
                
                // Add glyph if it has content
                let c = cell.character;
                if c != ' ' && c != '\0' {
                    let glyph = self.rasterize_char(c);
                    if glyph.size[0] > 0.0 && glyph.size[1] > 0.0 {
                        let (glyph_x, glyph_y) = if Self::is_box_drawing(c) {
                            (cell_x, cell_y)
                        } else {
                            let baseline_y = (cell_y + self.cell_height * 0.8).round();
                            let gx = (cell_x + glyph.offset[0]).round();
                            let gy = (baseline_y - glyph.offset[1] - glyph.size[1]).round();
                            (gx, gy)
                        };
                        
                        let left = Self::pixel_to_ndc_x(glyph_x, width);
                        let right = Self::pixel_to_ndc_x(glyph_x + glyph.size[0], width);
                        let top = Self::pixel_to_ndc_y(glyph_y, height);
                        let bottom = Self::pixel_to_ndc_y(glyph_y + glyph.size[1], height);
                        
                        let base_idx = self.glyph_vertices.len() as u32;
                        self.glyph_vertices.push(GlyphVertex {
                            position: [left, top],
                            uv: [glyph.uv[0], glyph.uv[1]],
                            color: fg_color,
                            bg_color: [0.0, 0.0, 0.0, 0.0],
                        });
                        self.glyph_vertices.push(GlyphVertex {
                            position: [right, top],
                            uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1]],
                            color: fg_color,
                            bg_color: [0.0, 0.0, 0.0, 0.0],
                        });
                        self.glyph_vertices.push(GlyphVertex {
                            position: [right, bottom],
                            uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1] + glyph.uv[3]],
                            color: fg_color,
                            bg_color: [0.0, 0.0, 0.0, 0.0],
                        });
                        self.glyph_vertices.push(GlyphVertex {
                            position: [left, bottom],
                            uv: [glyph.uv[0], glyph.uv[1] + glyph.uv[3]],
                            color: fg_color,
                            bg_color: [0.0, 0.0, 0.0, 0.0],
                        });
                        self.glyph_indices.extend_from_slice(&[
                            base_idx, base_idx + 1, base_idx + 2,
                            base_idx, base_idx + 2, base_idx + 3,
                        ]);
                    }
                }
                
                col_idx += 1;
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
            
            let has_character = cell_char != ' ' && cell_char != '\0';
            
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
    
    /// Render multiple panes with borders.
    /// 
    /// Arguments:
    /// - `panes`: List of (terminal, pane_info, selection) tuples
    /// - `num_tabs`: Number of tabs for the tab bar
    /// - `active_tab`: Index of the active tab
    pub fn render_panes(
        &mut self,
        panes: &[(&Terminal, PaneRenderInfo, Option<(usize, usize, usize, usize)>)],
        num_tabs: usize,
        active_tab: usize,
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
        
        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();
        
        Ok(())
    }

}
