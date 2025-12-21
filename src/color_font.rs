//! Color font (emoji) rendering using FreeType + Cairo.
//!
//! This module provides color emoji rendering support by using FreeType to load
//! color fonts (COLR, CBDT, sbix formats) and Cairo to render them.

use cairo::{Format, ImageSurface};
use freetype::Library as FtLibrary;
use std::collections::HashMap;
use std::ffi::CStr;
use std::path::PathBuf;

// ═══════════════════════════════════════════════════════════════════════════════
// COLOR FONT LOOKUP
// ═══════════════════════════════════════════════════════════════════════════════

/// Find a color font (emoji font) that contains the given character using fontconfig.
/// Returns the path to the font file if found.
pub fn find_color_font_for_char(c: char) -> Option<PathBuf> {
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

// ═══════════════════════════════════════════════════════════════════════════════
// COLOR FONT RENDERER
// ═══════════════════════════════════════════════════════════════════════════════

/// Lazy-initialized color font renderer using FreeType + Cairo.
/// Only created when a color emoji is first encountered.
/// Cairo is required for proper color font rendering (COLR, CBDT, sbix formats).
pub struct ColorFontRenderer {
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
    pub fn new() -> Result<Self, freetype::Error> {
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
    pub fn render_color_glyph(
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
