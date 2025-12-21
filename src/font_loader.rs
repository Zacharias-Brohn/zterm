//! Font loading and discovery using fontconfig.
//!
//! This module provides font loading utilities including:
//! - Finding fonts by family name with style variants (regular, bold, italic, bold-italic)
//! - Finding fonts that contain specific characters (for fallback)
//! - Loading font data for use with ab_glyph and rustybuzz

use ab_glyph::FontRef;
use fontconfig::Fontconfig;
use std::ffi::CStr;
use std::path::PathBuf;

// ═══════════════════════════════════════════════════════════════════════════════
// FONT VARIANT
// ═══════════════════════════════════════════════════════════════════════════════

/// A font variant with its data and parsed references.
pub struct FontVariant {
    /// Owned font data (kept alive for the lifetime of the font references).
    #[allow(dead_code)]
    data: Box<[u8]>,
    /// ab_glyph font reference for rasterization.
    font: FontRef<'static>,
    /// rustybuzz face for text shaping.
    face: rustybuzz::Face<'static>,
}

impl FontVariant {
    /// Get a reference to the ab_glyph font.
    pub fn font(&self) -> &FontRef<'static> {
        &self.font
    }

    /// Get a reference to the rustybuzz face.
    pub fn face(&self) -> &rustybuzz::Face<'static> {
        &self.face
    }

    /// Clone the font reference (ab_glyph FontRef is Clone).
    pub fn clone_font(&self) -> FontRef<'static> {
        self.font.clone()
    }

    /// Clone the font data.
    pub fn clone_data(&self) -> Box<[u8]> {
        self.data.clone()
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// FONT DISCOVERY
// ═══════════════════════════════════════════════════════════════════════════════

/// Find a font that contains the given character using fontconfig.
/// Returns the path to the font file if found.
/// 
/// Note: For emoji, use `find_color_font_for_char` from the color_font module instead,
/// which explicitly requests color fonts.
pub fn find_font_for_char(_fc: &Fontconfig, c: char) -> Option<PathBuf> {
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

        let font_result = if !matched.is_null() && result == FcResultMatch {
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

        font_result
    }
}

/// Find font files for a font family using fontconfig.
/// Returns paths for (regular, bold, italic, bold_italic).
/// Any variant that can't be found will be None.
pub fn find_font_family_variants(family: &str) -> [Option<PathBuf>; 4] {
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

// ═══════════════════════════════════════════════════════════════════════════════
// FONT LOADING
// ═══════════════════════════════════════════════════════════════════════════════

/// Try to load a font file and create both ab_glyph and rustybuzz handles.
/// Returns None if the file doesn't exist or can't be parsed.
pub fn load_font_variant(path: &std::path::Path) -> Option<FontVariant> {
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

/// Load font variants for a font family.
/// Returns array of font variants, with index 0 being the regular font.
/// Falls back to hardcoded paths if fontconfig fails.
pub fn load_font_family(font_family: Option<&str>) -> (Box<[u8]>, FontRef<'static>, [Option<FontVariant>; 4]) {
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
                let primary_font = regular.clone_font();
                let font_data = regular.clone_data();
                
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
            let primary_font = regular_variant.clone_font();
            let font_data = regular_variant.clone_data();
            
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
        let primary_font = regular_variant.clone_font();
        let font_data = regular_variant.clone_data();
        let variants: [Option<FontVariant>; 4] = [Some(regular_variant), None, None, None];
        log::info!("Loaded NotoSansMono as fallback");
        return (font_data, primary_font, variants);
    }
    
    panic!("Failed to load any monospace font");
}
