//! Linear color palette for GPU rendering.
//!
//! Provides pre-computed sRGB to linear RGB conversion for efficient GPU color handling.

use crate::terminal::ColorPalette;

/// Pre-computed linear RGB color palette.
/// Avoids repeated sRGB→linear conversions during rendering.
/// The color_table contains [258][4] floats: 256 indexed colors + default fg (256) + default bg (257).
#[derive(Clone)]
pub struct LinearPalette {
    /// Pre-computed linear RGBA colors ready for GPU upload.
    /// Index 0-255: palette colors, 256: default_fg, 257: default_bg
    pub color_table: [[f32; 4]; 258],
}

impl LinearPalette {
    /// Convert sRGB component (0.0-1.0) to linear RGB.
    #[inline]
    pub fn srgb_to_linear(c: f32) -> f32 {
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    }

    /// Convert an sRGB [u8; 3] color to linear [f32; 4] with alpha=1.0.
    #[inline]
    pub fn rgb_to_linear(rgb: [u8; 3]) -> [f32; 4] {
        [
            Self::srgb_to_linear(rgb[0] as f32 / 255.0),
            Self::srgb_to_linear(rgb[1] as f32 / 255.0),
            Self::srgb_to_linear(rgb[2] as f32 / 255.0),
            1.0,
        ]
    }

    /// Create a LinearPalette from a ColorPalette.
    pub fn from_palette(palette: &ColorPalette) -> Self {
        let mut color_table = [[0.0f32; 4]; 258];

        for i in 0..256 {
            color_table[i] = Self::rgb_to_linear(palette.colors[i]);
        }
        color_table[256] = Self::rgb_to_linear(palette.default_fg);
        color_table[257] = Self::rgb_to_linear(palette.default_bg);

        Self { color_table }
    }
}

impl Default for LinearPalette {
    fn default() -> Self {
        Self::from_palette(&ColorPalette::default())
    }
}
