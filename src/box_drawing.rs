//! Box drawing character rendering.
//!
//! This module provides procedural rendering of Unicode box drawing characters,
//! block elements, Braille patterns, and Powerline symbols.

// ═══════════════════════════════════════════════════════════════════════════════
// CORNER ENUM
// ═══════════════════════════════════════════════════════════════════════════════

/// Which corner of a cell for corner triangle rendering
#[derive(Clone, Copy)]
pub enum Corner {
    TopLeft,
    TopRight,
    BottomLeft,
    BottomRight,
}

// ═══════════════════════════════════════════════════════════════════════════════
// SUPERSAMPLED CANVAS
// ═══════════════════════════════════════════════════════════════════════════════

/// Supersampled canvas for anti-aliased rendering of box drawing characters.
/// Renders at 4x resolution then downsamples for smooth edges.
pub struct SupersampledCanvas {
    bitmap: Vec<u8>,
    width: usize,
    height: usize,
    pub ss_width: usize,
    pub ss_height: usize,
}

impl SupersampledCanvas {
    pub const FACTOR: usize = 4;

    pub fn new(width: usize, height: usize) -> Self {
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
    pub fn blend_pixel(&mut self, x: usize, y: usize, alpha: f64) {
        if x < self.ss_width && y < self.ss_height && alpha > 0.0 {
            let old_alpha = self.bitmap[y * self.ss_width + x] as f64 / 255.0;
            let new_alpha = alpha + (1.0 - alpha) * old_alpha;
            self.bitmap[y * self.ss_width + x] = (new_alpha * 255.0) as u8;
        }
    }

    /// Draw a thick line along x-axis with y computed by a function
    pub fn thick_line_h(&mut self, x1: usize, x2: usize, y_at_x: impl Fn(usize) -> f64, thickness: usize) {
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
    pub fn thick_point(&mut self, x: f64, y: f64, thickness: f64) {
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
    pub fn fill_corner_triangle(&mut self, corner: Corner, inverted: bool) {
        let w = self.ss_width;
        let h = self.ss_height;
        let max_x = (w - 1) as f64;
        let max_y = (h - 1) as f64;

        for py in 0..h {
            let y = py as f64;
            for px in 0..w {
                let x = px as f64;

                let (edge_y, fill_below) = match corner {
                    Corner::BottomLeft => (max_y - (max_y / max_x) * x, true),
                    Corner::TopLeft => ((max_y / max_x) * x, false),
                    Corner::BottomRight => ((max_y / max_x) * x, true),
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
    pub fn fill_powerline_arrow(&mut self, left: bool, inverted: bool) {
        let w = self.ss_width;
        let h = self.ss_height;
        let max_x = (w - 1) as f64;
        let max_y = (h - 1) as f64;
        let mid_y = max_y / 2.0;

        for py in 0..h {
            let y = py as f64;
            for px in 0..w {
                let x = px as f64;

                let (upper_y, lower_y) = if left {
                    let upper = (mid_y / max_x) * (max_x - x);
                    let lower = max_y - (mid_y / max_x) * (max_x - x);
                    (upper, lower)
                } else {
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

    /// Draw powerline arrow outline (chevron shape)
    pub fn stroke_powerline_arrow(&mut self, left: bool, thickness: usize) {
        let w = self.ss_width;
        let h = self.ss_height;
        let max_x = (w - 1) as f64;
        let max_y = (h - 1) as f64;
        let mid_y = max_y / 2.0;

        if left {
            self.thick_line_h(0, w, |x| (mid_y / max_x) * (max_x - x as f64), thickness);
            self.thick_line_h(0, w, |x| max_y - (mid_y / max_x) * (max_x - x as f64), thickness);
        } else {
            self.thick_line_h(0, w, |x| (mid_y / max_x) * x as f64, thickness);
            self.thick_line_h(0, w, |x| max_y - (mid_y / max_x) * x as f64, thickness);
        }
    }

    /// Fill region using a Bezier curve (for "D" shaped powerline semicircles).
    pub fn fill_bezier_d(&mut self, left: bool) {
        let w = self.ss_width;
        let h = self.ss_height;
        let max_x = (w - 1) as f64;
        let max_y = (h - 1) as f64;
        let cx = max_x / 0.75;

        for py in 0..h {
            let target_y = py as f64;
            let t = Self::find_t_for_bezier_y(max_y, target_y);
            let u = 1.0 - t;
            let bx = 3.0 * cx * t * u;
            let x_extent = (bx.round() as usize).min(w - 1);

            if left {
                let start_x = (w - 1).saturating_sub(x_extent);
                for px in start_x..w {
                    self.bitmap[py * w + px] = 255;
                }
            } else {
                for px in 0..=x_extent {
                    self.bitmap[py * w + px] = 255;
                }
            }
        }
    }

    /// Binary search for t where bezier_y(t) ≈ target_y
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
    pub fn stroke_bezier_d(&mut self, left: bool, thickness: f64) {
        let w = self.ss_width;
        let h = self.ss_height;
        let max_x = (w - 1) as f64;
        let max_y = (h - 1) as f64;
        let cx = max_x / 0.75;

        let steps = (h * 2) as usize;
        for i in 0..=steps {
            let t = i as f64 / steps as f64;
            let u = 1.0 - t;
            let bx = 3.0 * cx * t * u;
            let by = max_y * t * t * (3.0 - 2.0 * t);

            let bx_clamped = bx.min(max_x);
            let x = if left { max_x - bx_clamped } else { bx_clamped };
            self.thick_point(x, by, thickness);
        }
    }

    /// Fill a circle centered in the cell
    pub fn fill_circle(&mut self, radius_factor: f64) {
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
    pub fn fill_circle_radius(&mut self, radius: f64) {
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
    pub fn stroke_circle(&mut self, radius: f64, line_width: f64) {
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
    pub fn stroke_arc(&mut self, radius: f64, line_width: f64, start_angle: f64, end_angle: f64) {
        let cx = self.ss_width as f64 / 2.0;
        let cy = self.ss_height as f64 / 2.0;
        let half_thickness = line_width / 2.0;

        let num_samples = (self.ss_width.max(self.ss_height) * 2) as usize;
        let angle_range = end_angle - start_angle;

        for i in 0..=num_samples {
            let t = i as f64 / num_samples as f64;
            let angle = start_angle + angle_range * t;
            let arc_x = cx + radius * angle.cos();
            let arc_y = cy + radius * angle.sin();

            self.stroke_point_aa(arc_x, arc_y, half_thickness);
        }
    }

    /// Draw an anti-aliased point
    pub fn stroke_point_aa(&mut self, x: f64, y: f64, half_thickness: f64) {
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
    pub fn downsample(&self, output: &mut [u8]) {
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

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

/// Calculate line thickness based on DPI and scale, similar to Kitty's thickness_as_float.
/// Level 0 = hairline, 1 = light, 2 = medium, 3 = heavy
pub fn box_thickness(level: usize, dpi: f64) -> f64 {
    const BOX_DRAWING_SCALE: [f64; 4] = [0.001, 1.0, 1.5, 2.0];
    let pts = BOX_DRAWING_SCALE[level.min(3)];
    (pts * dpi / 72.0).max(1.0)
}

/// Check if a character is a box-drawing character that should be rendered procedurally.
pub fn is_box_drawing(c: char) -> bool {
    let cp = c as u32;
    (0x2500..=0x257F).contains(&cp)
        || (0x2580..=0x259F).contains(&cp)
        || (0x25A0..=0x25FF).contains(&cp)
        || (0x2800..=0x28FF).contains(&cp)
        || (0xE0B0..=0xE0BF).contains(&cp)
}

// ═══════════════════════════════════════════════════════════════════════════════
// RENDER BOX CHAR
// ═══════════════════════════════════════════════════════════════════════════════

/// Render a box-drawing character procedurally to a bitmap.
/// Returns (bitmap, supersampled) where supersampled indicates if anti-aliasing was used.
pub fn render_box_char(
    c: char,
    cell_width: usize,
    cell_height: usize,
    font_size: f32,
    dpi: f64,
) -> Option<(Vec<u8>, bool)> {
    let w = cell_width;
    let h = cell_height;
    let mut bitmap = vec![0u8; w * h];
    let mut supersampled = false;

    let mid_x = w / 2;
    let mid_y = h / 2;
    let light = 2.max((font_size / 8.0).round() as usize);
    let heavy = light * 2;

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

    // Box drawing arm thickness encoding: 0=none, 1=light, 2=heavy
    let box_arms: Option<(u8, u8, u8, u8)> = match c {
        // Light lines
        '\u{2500}' => Some((1, 1, 0, 0)), // ─
        '\u{2502}' => Some((0, 0, 1, 1)), // │
        // Light corners
        '\u{250C}' => Some((0, 1, 0, 1)), // ┌
        '\u{2510}' => Some((1, 0, 0, 1)), // ┐
        '\u{2514}' => Some((0, 1, 1, 0)), // └
        '\u{2518}' => Some((1, 0, 1, 0)), // ┘
        // Light T-junctions
        '\u{251C}' => Some((0, 1, 1, 1)), // ├
        '\u{2524}' => Some((1, 0, 1, 1)), // ┤
        '\u{252C}' => Some((1, 1, 0, 1)), // ┬
        '\u{2534}' => Some((1, 1, 1, 0)), // ┴
        // Light cross
        '\u{253C}' => Some((1, 1, 1, 1)), // ┼
        // Heavy lines
        '\u{2501}' => Some((2, 2, 0, 0)), // ━
        '\u{2503}' => Some((0, 0, 2, 2)), // ┃
        // Heavy corners
        '\u{250F}' => Some((0, 2, 0, 2)), // ┏
        '\u{2513}' => Some((2, 0, 0, 2)), // ┓
        '\u{2517}' => Some((0, 2, 2, 0)), // ┗
        '\u{251B}' => Some((2, 0, 2, 0)), // ┛
        // Heavy T-junctions
        '\u{2523}' => Some((0, 2, 2, 2)), // ┣
        '\u{252B}' => Some((2, 0, 2, 2)), // ┫
        '\u{2533}' => Some((2, 2, 0, 2)), // ┳
        '\u{253B}' => Some((2, 2, 2, 0)), // ┻
        // Heavy cross
        '\u{254B}' => Some((2, 2, 2, 2)), // ╋
        // Mixed light/heavy corners
        '\u{250E}' => Some((0, 1, 0, 2)), // ┎
        '\u{2512}' => Some((1, 0, 0, 2)), // ┒
        '\u{2516}' => Some((0, 1, 2, 0)), // ┖
        '\u{251A}' => Some((1, 0, 2, 0)), // ┚
        '\u{250D}' => Some((0, 2, 0, 1)), // ┍
        '\u{2511}' => Some((2, 0, 0, 1)), // ┑
        '\u{2515}' => Some((0, 2, 1, 0)), // ┕
        '\u{2519}' => Some((2, 0, 1, 0)), // ┙
        // Mixed T-junctions (vertical heavy, horizontal light)
        '\u{2520}' => Some((0, 1, 2, 2)), // ┠
        '\u{2528}' => Some((1, 0, 2, 2)), // ┨
        '\u{2530}' => Some((1, 1, 0, 2)), // ┰
        '\u{2538}' => Some((1, 1, 2, 0)), // ┸
        // Mixed T-junctions (vertical light, horizontal heavy)
        '\u{251D}' => Some((0, 2, 1, 1)), // ┝
        '\u{2525}' => Some((2, 0, 1, 1)), // ┥
        '\u{252F}' => Some((2, 2, 0, 1)), // ┯
        '\u{2537}' => Some((2, 2, 1, 0)), // ┷
        // Mixed cross (heavy horizontal, light vertical)
        '\u{2542}' => Some((1, 1, 2, 2)), // ╂
        _ => None,
    };

    // Handle simple box drawing via lookup table
    if let Some((left, right, up, down)) = box_arms {
        let thickness = |arm: u8| -> usize {
            match arm {
                1 => light,
                2 => heavy,
                _ => 0,
            }
        };
        if left > 0 {
            hline(&mut bitmap, 0, mid_x + 1, mid_y, thickness(left));
        }
        if right > 0 {
            hline(&mut bitmap, mid_x, w, mid_y, thickness(right));
        }
        if up > 0 {
            vline(&mut bitmap, 0, mid_y + 1, mid_x, thickness(up));
        }
        if down > 0 {
            vline(&mut bitmap, mid_y, h, mid_x, thickness(down));
        }
        return Some((bitmap, supersampled));
    }

    // Continue with match for remaining characters
    render_box_char_extended(
        c, &mut bitmap, &mut supersampled,
        w, h, mid_x, mid_y, light, heavy, double_off,
        dpi, hline, vline, fill_rect,
    )?;

    Some((bitmap, supersampled))
}

// Part 2 of render_box_char - handles extended characters
fn render_box_char_extended<H, V, F>(
    c: char,
    bitmap: &mut [u8],
    supersampled: &mut bool,
    w: usize,
    h: usize,
    mid_x: usize,
    mid_y: usize,
    light: usize,
    heavy: usize,
    double_off: usize,
    dpi: f64,
    hline: H,
    vline: V,
    fill_rect: F,
) -> Option<()>
where
    H: Fn(&mut [u8], usize, usize, usize, usize),
    V: Fn(&mut [u8], usize, usize, usize, usize),
    F: Fn(&mut [u8], usize, usize, usize, usize),
{
    match c {
        // Mixed T-junctions continued
        '\u{251E}' => {
            vline(bitmap, 0, mid_y + 1, mid_x, light);
            vline(bitmap, mid_y, h, mid_x, heavy);
            hline(bitmap, mid_x, w, mid_y, light);
        }
        '\u{251F}' => {
            vline(bitmap, 0, mid_y + 1, mid_x, heavy);
            vline(bitmap, mid_y, h, mid_x, light);
            hline(bitmap, mid_x, w, mid_y, light);
        }
        '\u{2521}' => {
            vline(bitmap, 0, mid_y + 1, mid_x, light);
            vline(bitmap, mid_y, h, mid_x, heavy);
            hline(bitmap, mid_x, w, mid_y, heavy);
        }
        '\u{2522}' => {
            vline(bitmap, 0, mid_y + 1, mid_x, heavy);
            vline(bitmap, mid_y, h, mid_x, light);
            hline(bitmap, mid_x, w, mid_y, heavy);
        }
        '\u{2526}' => {
            vline(bitmap, 0, mid_y + 1, mid_x, light);
            vline(bitmap, mid_y, h, mid_x, heavy);
            hline(bitmap, 0, mid_x + 1, mid_y, light);
        }
        '\u{2527}' => {
            vline(bitmap, 0, mid_y + 1, mid_x, heavy);
            vline(bitmap, mid_y, h, mid_x, light);
            hline(bitmap, 0, mid_x + 1, mid_y, light);
        }
        '\u{2529}' => {
            vline(bitmap, 0, mid_y + 1, mid_x, light);
            vline(bitmap, mid_y, h, mid_x, heavy);
            hline(bitmap, 0, mid_x + 1, mid_y, heavy);
        }
        '\u{252A}' => {
            vline(bitmap, 0, mid_y + 1, mid_x, heavy);
            vline(bitmap, mid_y, h, mid_x, light);
            hline(bitmap, 0, mid_x + 1, mid_y, heavy);
        }
        '\u{252D}' => {
            hline(bitmap, 0, mid_x + 1, mid_y, light);
            hline(bitmap, mid_x, w, mid_y, heavy);
            vline(bitmap, mid_y, h, mid_x, light);
        }
        '\u{252E}' => {
            hline(bitmap, 0, mid_x + 1, mid_y, heavy);
            hline(bitmap, mid_x, w, mid_y, light);
            vline(bitmap, mid_y, h, mid_x, light);
        }
        '\u{2531}' => {
            hline(bitmap, 0, mid_x + 1, mid_y, light);
            hline(bitmap, mid_x, w, mid_y, heavy);
            vline(bitmap, mid_y, h, mid_x, heavy);
        }
        '\u{2532}' => {
            hline(bitmap, 0, mid_x + 1, mid_y, heavy);
            hline(bitmap, mid_x, w, mid_y, light);
            vline(bitmap, mid_y, h, mid_x, heavy);
        }
        '\u{2535}' => {
            hline(bitmap, 0, mid_x + 1, mid_y, light);
            hline(bitmap, mid_x, w, mid_y, heavy);
            vline(bitmap, 0, mid_y + 1, mid_x, light);
        }
        '\u{2536}' => {
            hline(bitmap, 0, mid_x + 1, mid_y, heavy);
            hline(bitmap, mid_x, w, mid_y, light);
            vline(bitmap, 0, mid_y + 1, mid_x, light);
        }
        '\u{2539}' => {
            hline(bitmap, 0, mid_x + 1, mid_y, light);
            hline(bitmap, mid_x, w, mid_y, heavy);
            vline(bitmap, 0, mid_y + 1, mid_x, heavy);
        }
        '\u{253A}' => {
            hline(bitmap, 0, mid_x + 1, mid_y, heavy);
            hline(bitmap, mid_x, w, mid_y, light);
            vline(bitmap, 0, mid_y + 1, mid_x, heavy);
        }
        // Mixed crosses
        '\u{2540}' => {
            hline(bitmap, 0, w, mid_y, light);
            vline(bitmap, 0, mid_y + 1, mid_x, heavy);
            vline(bitmap, mid_y, h, mid_x, light);
        }
        '\u{2541}' => {
            hline(bitmap, 0, w, mid_y, light);
            vline(bitmap, 0, mid_y + 1, mid_x, light);
            vline(bitmap, mid_y, h, mid_x, heavy);
        }
        '\u{2543}' => {
            hline(bitmap, 0, mid_x + 1, mid_y, heavy);
            hline(bitmap, mid_x, w, mid_y, light);
            vline(bitmap, 0, mid_y + 1, mid_x, heavy);
            vline(bitmap, mid_y, h, mid_x, light);
        }
        '\u{2544}' => {
            hline(bitmap, 0, mid_x + 1, mid_y, light);
            hline(bitmap, mid_x, w, mid_y, heavy);
            vline(bitmap, 0, mid_y + 1, mid_x, heavy);
            vline(bitmap, mid_y, h, mid_x, light);
        }
        '\u{2545}' => {
            hline(bitmap, 0, mid_x + 1, mid_y, heavy);
            hline(bitmap, mid_x, w, mid_y, light);
            vline(bitmap, 0, mid_y + 1, mid_x, light);
            vline(bitmap, mid_y, h, mid_x, heavy);
        }
        '\u{2546}' => {
            hline(bitmap, 0, mid_x + 1, mid_y, light);
            hline(bitmap, mid_x, w, mid_y, heavy);
            vline(bitmap, 0, mid_y + 1, mid_x, light);
            vline(bitmap, mid_y, h, mid_x, heavy);
        }
        '\u{2547}' => {
            hline(bitmap, 0, w, mid_y, heavy);
            vline(bitmap, 0, mid_y + 1, mid_x, light);
            vline(bitmap, mid_y, h, mid_x, heavy);
        }
        '\u{2548}' => {
            hline(bitmap, 0, w, mid_y, heavy);
            vline(bitmap, 0, mid_y + 1, mid_x, heavy);
            vline(bitmap, mid_y, h, mid_x, light);
        }
        '\u{2549}' => {
            hline(bitmap, 0, mid_x + 1, mid_y, light);
            hline(bitmap, mid_x, w, mid_y, heavy);
            vline(bitmap, 0, h, mid_x, heavy);
        }
        '\u{254A}' => {
            hline(bitmap, 0, mid_x + 1, mid_y, heavy);
            hline(bitmap, mid_x, w, mid_y, light);
            vline(bitmap, 0, h, mid_x, heavy);
        }
        // Delegate to part 3 for double lines, blocks, etc.
        _ => {
            return render_box_char_part3(
                c, bitmap, supersampled,
                w, h, mid_x, mid_y, light, heavy, double_off,
                dpi, hline, vline, fill_rect,
            );
        }
    }
    Some(())
}

// Part 3 - double lines and block elements
fn render_box_char_part3<H, V, F>(
    c: char,
    bitmap: &mut [u8],
    supersampled: &mut bool,
    w: usize,
    h: usize,
    mid_x: usize,
    mid_y: usize,
    light: usize,
    _heavy: usize,
    double_off: usize,
    dpi: f64,
    hline: H,
    vline: V,
    fill_rect: F,
) -> Option<()>
where
    H: Fn(&mut [u8], usize, usize, usize, usize),
    V: Fn(&mut [u8], usize, usize, usize, usize),
    F: Fn(&mut [u8], usize, usize, usize, usize),
{
    match c {
        // Double lines
        '\u{2550}' => {
            hline(bitmap, 0, w, mid_y.saturating_sub(double_off), light);
            hline(bitmap, 0, w, mid_y + double_off, light);
        }
        '\u{2551}' => {
            vline(bitmap, 0, h, mid_x.saturating_sub(double_off), light);
            vline(bitmap, 0, h, mid_x + double_off, light);
        }
        // Double corners
        '\u{2554}' => {
            hline(bitmap, mid_x, w, mid_y.saturating_sub(double_off), light);
            hline(bitmap, mid_x + double_off, w, mid_y + double_off, light);
            vline(bitmap, mid_y, h, mid_x.saturating_sub(double_off), light);
            vline(bitmap, mid_y.saturating_sub(double_off), h, mid_x + double_off, light);
        }
        '\u{2557}' => {
            hline(bitmap, 0, mid_x + 1, mid_y.saturating_sub(double_off), light);
            hline(bitmap, 0, mid_x.saturating_sub(double_off) + 1, mid_y + double_off, light);
            vline(bitmap, mid_y, h, mid_x + double_off, light);
            vline(bitmap, mid_y.saturating_sub(double_off), h, mid_x.saturating_sub(double_off), light);
        }
        '\u{255A}' => {
            hline(bitmap, mid_x, w, mid_y + double_off, light);
            hline(bitmap, mid_x + double_off, w, mid_y.saturating_sub(double_off), light);
            vline(bitmap, 0, mid_y + 1, mid_x.saturating_sub(double_off), light);
            vline(bitmap, 0, mid_y + double_off + 1, mid_x + double_off, light);
        }
        '\u{255D}' => {
            hline(bitmap, 0, mid_x + 1, mid_y + double_off, light);
            hline(bitmap, 0, mid_x.saturating_sub(double_off) + 1, mid_y.saturating_sub(double_off), light);
            vline(bitmap, 0, mid_y + 1, mid_x + double_off, light);
            vline(bitmap, 0, mid_y + double_off + 1, mid_x.saturating_sub(double_off), light);
        }
        // Double T-junctions
        '\u{2560}' => {
            vline(bitmap, 0, h, mid_x.saturating_sub(double_off), light);
            vline(bitmap, 0, mid_y.saturating_sub(double_off) + 1, mid_x + double_off, light);
            vline(bitmap, mid_y + double_off, h, mid_x + double_off, light);
            hline(bitmap, mid_x + double_off, w, mid_y.saturating_sub(double_off), light);
            hline(bitmap, mid_x + double_off, w, mid_y + double_off, light);
        }
        '\u{2563}' => {
            vline(bitmap, 0, h, mid_x + double_off, light);
            vline(bitmap, 0, mid_y.saturating_sub(double_off) + 1, mid_x.saturating_sub(double_off), light);
            vline(bitmap, mid_y + double_off, h, mid_x.saturating_sub(double_off), light);
            hline(bitmap, 0, mid_x.saturating_sub(double_off) + 1, mid_y.saturating_sub(double_off), light);
            hline(bitmap, 0, mid_x.saturating_sub(double_off) + 1, mid_y + double_off, light);
        }
        '\u{2566}' => {
            hline(bitmap, 0, w, mid_y.saturating_sub(double_off), light);
            hline(bitmap, 0, mid_x.saturating_sub(double_off) + 1, mid_y + double_off, light);
            hline(bitmap, mid_x + double_off, w, mid_y + double_off, light);
            vline(bitmap, mid_y + double_off, h, mid_x.saturating_sub(double_off), light);
            vline(bitmap, mid_y + double_off, h, mid_x + double_off, light);
        }
        '\u{2569}' => {
            hline(bitmap, 0, w, mid_y + double_off, light);
            hline(bitmap, 0, mid_x.saturating_sub(double_off) + 1, mid_y.saturating_sub(double_off), light);
            hline(bitmap, mid_x + double_off, w, mid_y.saturating_sub(double_off), light);
            vline(bitmap, 0, mid_y.saturating_sub(double_off) + 1, mid_x.saturating_sub(double_off), light);
            vline(bitmap, 0, mid_y.saturating_sub(double_off) + 1, mid_x + double_off, light);
        }
        // Double cross
        '\u{256C}' => {
            vline(bitmap, 0, mid_y.saturating_sub(double_off) + 1, mid_x.saturating_sub(double_off), light);
            vline(bitmap, 0, mid_y.saturating_sub(double_off) + 1, mid_x + double_off, light);
            vline(bitmap, mid_y + double_off, h, mid_x.saturating_sub(double_off), light);
            vline(bitmap, mid_y + double_off, h, mid_x + double_off, light);
            hline(bitmap, 0, mid_x.saturating_sub(double_off) + 1, mid_y.saturating_sub(double_off), light);
            hline(bitmap, 0, mid_x.saturating_sub(double_off) + 1, mid_y + double_off, light);
            hline(bitmap, mid_x + double_off, w, mid_y.saturating_sub(double_off), light);
            hline(bitmap, mid_x + double_off, w, mid_y + double_off, light);
        }
        // Delegate remaining to part 4
        _ => {
            return render_box_char_part4(
                c, bitmap, supersampled,
                w, h, mid_x, mid_y, light, double_off,
                dpi, hline, vline, fill_rect,
            );
        }
    }
    Some(())
}

// Part 4 - single/double mixed corners and T-junctions
fn render_box_char_part4<H, V, F>(
    c: char,
    bitmap: &mut [u8],
    supersampled: &mut bool,
    w: usize,
    h: usize,
    mid_x: usize,
    mid_y: usize,
    light: usize,
    double_off: usize,
    dpi: f64,
    hline: H,
    vline: V,
    fill_rect: F,
) -> Option<()>
where
    H: Fn(&mut [u8], usize, usize, usize, usize),
    V: Fn(&mut [u8], usize, usize, usize, usize),
    F: Fn(&mut [u8], usize, usize, usize, usize),
{
    match c {
        // Single/double mixed corners
        '\u{2552}' => {
            hline(bitmap, mid_x + double_off, w, mid_y, light);
            vline(bitmap, mid_y, h, mid_x.saturating_sub(double_off), light);
            vline(bitmap, mid_y, h, mid_x + double_off, light);
        }
        '\u{2553}' => {
            hline(bitmap, mid_x, w, mid_y, light);
            vline(bitmap, mid_y, h, mid_x, light);
        }
        '\u{2555}' => {
            hline(bitmap, 0, mid_x.saturating_sub(double_off) + 1, mid_y, light);
            vline(bitmap, mid_y, h, mid_x.saturating_sub(double_off), light);
            vline(bitmap, mid_y, h, mid_x + double_off, light);
        }
        '\u{2556}' => {
            hline(bitmap, 0, mid_x + 1, mid_y, light);
            vline(bitmap, mid_y, h, mid_x, light);
        }
        '\u{2558}' => {
            hline(bitmap, mid_x + double_off, w, mid_y, light);
            vline(bitmap, 0, mid_y + 1, mid_x.saturating_sub(double_off), light);
            vline(bitmap, 0, mid_y + 1, mid_x + double_off, light);
        }
        '\u{2559}' => {
            hline(bitmap, mid_x, w, mid_y, light);
            vline(bitmap, 0, mid_y + 1, mid_x, light);
        }
        '\u{255B}' => {
            hline(bitmap, 0, mid_x.saturating_sub(double_off) + 1, mid_y, light);
            vline(bitmap, 0, mid_y + 1, mid_x.saturating_sub(double_off), light);
            vline(bitmap, 0, mid_y + 1, mid_x + double_off, light);
        }
        '\u{255C}' => {
            hline(bitmap, 0, mid_x + 1, mid_y, light);
            vline(bitmap, 0, mid_y + 1, mid_x, light);
        }
        // Mixed T-junctions
        '\u{255E}' => {
            vline(bitmap, 0, h, mid_x.saturating_sub(double_off), light);
            vline(bitmap, 0, h, mid_x + double_off, light);
            hline(bitmap, mid_x + double_off, w, mid_y, light);
        }
        '\u{255F}' => {
            vline(bitmap, 0, h, mid_x, light);
            hline(bitmap, mid_x, w, mid_y.saturating_sub(double_off), light);
            hline(bitmap, mid_x, w, mid_y + double_off, light);
        }
        '\u{2561}' => {
            vline(bitmap, 0, h, mid_x.saturating_sub(double_off), light);
            vline(bitmap, 0, h, mid_x + double_off, light);
            hline(bitmap, 0, mid_x.saturating_sub(double_off) + 1, mid_y, light);
        }
        '\u{2562}' => {
            vline(bitmap, 0, h, mid_x, light);
            hline(bitmap, 0, mid_x + 1, mid_y.saturating_sub(double_off), light);
            hline(bitmap, 0, mid_x + 1, mid_y + double_off, light);
        }
        '\u{2564}' => {
            hline(bitmap, 0, w, mid_y.saturating_sub(double_off), light);
            hline(bitmap, 0, w, mid_y + double_off, light);
            vline(bitmap, mid_y + double_off, h, mid_x, light);
        }
        '\u{2565}' => {
            hline(bitmap, 0, w, mid_y, light);
            vline(bitmap, mid_y, h, mid_x.saturating_sub(double_off), light);
            vline(bitmap, mid_y, h, mid_x + double_off, light);
        }
        '\u{2567}' => {
            hline(bitmap, 0, w, mid_y.saturating_sub(double_off), light);
            hline(bitmap, 0, w, mid_y + double_off, light);
            vline(bitmap, 0, mid_y.saturating_sub(double_off) + 1, mid_x, light);
        }
        '\u{2568}' => {
            hline(bitmap, 0, w, mid_y, light);
            vline(bitmap, 0, mid_y + 1, mid_x.saturating_sub(double_off), light);
            vline(bitmap, 0, mid_y + 1, mid_x + double_off, light);
        }
        // Mixed crosses
        '\u{256A}' => {
            hline(bitmap, 0, w, mid_y.saturating_sub(double_off), light);
            hline(bitmap, 0, w, mid_y + double_off, light);
            vline(bitmap, 0, mid_y.saturating_sub(double_off) + 1, mid_x, light);
            vline(bitmap, mid_y + double_off, h, mid_x, light);
        }
        '\u{256B}' => {
            hline(bitmap, 0, mid_x.saturating_sub(double_off) + 1, mid_y, light);
            hline(bitmap, mid_x + double_off, w, mid_y, light);
            vline(bitmap, 0, h, mid_x.saturating_sub(double_off), light);
            vline(bitmap, 0, h, mid_x + double_off, light);
        }
        // Delegate to part 5
        _ => {
            return render_box_char_part5(
                c, bitmap, supersampled,
                w, h, mid_x, mid_y, light,
                dpi, hline, vline, fill_rect,
            );
        }
    }
    Some(())
}

// Part 5 - rounded corners and dashed lines
fn render_box_char_part5<H, V, F>(
    c: char,
    bitmap: &mut [u8],
    supersampled: &mut bool,
    w: usize,
    h: usize,
    mid_x: usize,
    mid_y: usize,
    light: usize,
    dpi: f64,
    hline: H,
    vline: V,
    fill_rect: F,
) -> Option<()>
where
    H: Fn(&mut [u8], usize, usize, usize, usize),
    V: Fn(&mut [u8], usize, usize, usize, usize),
    F: Fn(&mut [u8], usize, usize, usize, usize),
{
    let heavy = light * 2;
    
    match c {
        // Rounded corners
        '\u{256D}' | '\u{256E}' | '\u{256F}' | '\u{2570}' => {
            let hori_line_start = mid_y.saturating_sub(light / 2);
            let hori_line_end = (hori_line_start + light).min(h);
            let hori_line_height = hori_line_end - hori_line_start;

            let vert_line_start = mid_x.saturating_sub(light / 2);
            let vert_line_end = (vert_line_start + light).min(w);
            let vert_line_width = vert_line_end - vert_line_start;

            let adjusted_hx = vert_line_start as f64 + vert_line_width as f64 / 2.0;
            let adjusted_hy = hori_line_start as f64 + hori_line_height as f64 / 2.0;

            let stroke = (hori_line_height.max(vert_line_width)) as f64;
            let corner_radius = adjusted_hx.min(adjusted_hy);
            let bx = adjusted_hx - corner_radius;
            let by = adjusted_hy - corner_radius;

            let aa_corner = 0.5;
            let half_stroke = 0.5 * stroke;

            let (is_right, is_top) = match c {
                '\u{256D}' => (false, true),
                '\u{256E}' => (true, true),
                '\u{2570}' => (false, false),
                '\u{256F}' => (true, false),
                _ => unreachable!(),
            };

            let x_shift = if is_right { adjusted_hx } else { -adjusted_hx };
            let y_shift = if is_top { -adjusted_hy } else { adjusted_hy };

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
        // Dashed lines (triple dash)
        '\u{2504}' => {
            let seg = w / 8;
            for i in 0..4 {
                let x1 = i * 2 * seg;
                let x2 = (x1 + seg).min(w);
                hline(bitmap, x1, x2, mid_y, light);
            }
        }
        '\u{2505}' => {
            let seg = w / 8;
            for i in 0..4 {
                let x1 = i * 2 * seg;
                let x2 = (x1 + seg).min(w);
                hline(bitmap, x1, x2, mid_y, heavy);
            }
        }
        '\u{2506}' => {
            let seg = h / 8;
            for i in 0..4 {
                let y1 = i * 2 * seg;
                let y2 = (y1 + seg).min(h);
                vline(bitmap, y1, y2, mid_x, light);
            }
        }
        '\u{2507}' => {
            let seg = h / 8;
            for i in 0..4 {
                let y1 = i * 2 * seg;
                let y2 = (y1 + seg).min(h);
                vline(bitmap, y1, y2, mid_x, heavy);
            }
        }
        '\u{2508}' => {
            let seg = w / 12;
            for i in 0..6 {
                let x1 = i * 2 * seg;
                let x2 = (x1 + seg).min(w);
                hline(bitmap, x1, x2, mid_y, light);
            }
        }
        '\u{2509}' => {
            let seg = w / 12;
            for i in 0..6 {
                let x1 = i * 2 * seg;
                let x2 = (x1 + seg).min(w);
                hline(bitmap, x1, x2, mid_y, heavy);
            }
        }
        '\u{250A}' => {
            let seg = h / 12;
            for i in 0..6 {
                let y1 = i * 2 * seg;
                let y2 = (y1 + seg).min(h);
                vline(bitmap, y1, y2, mid_x, light);
            }
        }
        '\u{250B}' => {
            let seg = h / 12;
            for i in 0..6 {
                let y1 = i * 2 * seg;
                let y2 = (y1 + seg).min(h);
                vline(bitmap, y1, y2, mid_x, heavy);
            }
        }
        // Double dashed
        '\u{254C}' => {
            let seg = w / 4;
            hline(bitmap, 0, seg, mid_y, light);
            hline(bitmap, seg * 2, seg * 3, mid_y, light);
        }
        '\u{254D}' => {
            let seg = w / 4;
            hline(bitmap, 0, seg, mid_y, heavy);
            hline(bitmap, seg * 2, seg * 3, mid_y, heavy);
        }
        '\u{254E}' => {
            let seg = h / 4;
            vline(bitmap, 0, seg, mid_x, light);
            vline(bitmap, seg * 2, seg * 3, mid_x, light);
        }
        '\u{254F}' => {
            let seg = h / 4;
            vline(bitmap, 0, seg, mid_x, heavy);
            vline(bitmap, seg * 2, seg * 3, mid_x, heavy);
        }
        // Delegate to part 6
        _ => {
            return render_box_char_part6(
                c, bitmap, supersampled,
                w, h, mid_x, mid_y, light, heavy,
                dpi, hline, vline, fill_rect,
            );
        }
    }
    Some(())
}

// Part 6 - half lines, diagonals, and block elements
fn render_box_char_part6<H, V, F>(
    c: char,
    bitmap: &mut [u8],
    supersampled: &mut bool,
    w: usize,
    h: usize,
    mid_x: usize,
    mid_y: usize,
    light: usize,
    heavy: usize,
    dpi: f64,
    hline: H,
    vline: V,
    fill_rect: F,
) -> Option<()>
where
    H: Fn(&mut [u8], usize, usize, usize, usize),
    V: Fn(&mut [u8], usize, usize, usize, usize),
    F: Fn(&mut [u8], usize, usize, usize, usize),
{
    match c {
        // Half lines
        '\u{2574}' => hline(bitmap, 0, mid_x + 1, mid_y, light),
        '\u{2575}' => vline(bitmap, 0, mid_y + 1, mid_x, light),
        '\u{2576}' => hline(bitmap, mid_x, w, mid_y, light),
        '\u{2577}' => vline(bitmap, mid_y, h, mid_x, light),
        '\u{2578}' => hline(bitmap, 0, mid_x + 1, mid_y, heavy),
        '\u{2579}' => vline(bitmap, 0, mid_y + 1, mid_x, heavy),
        '\u{257A}' => hline(bitmap, mid_x, w, mid_y, heavy),
        '\u{257B}' => vline(bitmap, mid_y, h, mid_x, heavy),
        // Mixed half lines
        '\u{257C}' => {
            hline(bitmap, 0, mid_x + 1, mid_y, light);
            hline(bitmap, mid_x, w, mid_y, heavy);
        }
        '\u{257D}' => {
            vline(bitmap, 0, mid_y + 1, mid_x, light);
            vline(bitmap, mid_y, h, mid_x, heavy);
        }
        '\u{257E}' => {
            hline(bitmap, 0, mid_x + 1, mid_y, heavy);
            hline(bitmap, mid_x, w, mid_y, light);
        }
        '\u{257F}' => {
            vline(bitmap, 0, mid_y + 1, mid_x, heavy);
            vline(bitmap, mid_y, h, mid_x, light);
        }
        // Diagonal lines
        '\u{2571}' => {
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
        '\u{2572}' => {
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
        '\u{2573}' => {
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
        // Block elements
        '\u{2580}' => fill_rect(bitmap, 0, 0, w, h / 2),
        '\u{2581}' => fill_rect(bitmap, 0, h * 7 / 8, w, h),
        '\u{2582}' => fill_rect(bitmap, 0, h * 3 / 4, w, h),
        '\u{2583}' => fill_rect(bitmap, 0, h * 5 / 8, w, h),
        '\u{2584}' => fill_rect(bitmap, 0, h / 2, w, h),
        '\u{2585}' => fill_rect(bitmap, 0, h * 3 / 8, w, h),
        '\u{2586}' => fill_rect(bitmap, 0, h / 4, w, h),
        '\u{2587}' => fill_rect(bitmap, 0, h / 8, w, h),
        '\u{2588}' => fill_rect(bitmap, 0, 0, w, h),
        '\u{2589}' => fill_rect(bitmap, 0, 0, w * 7 / 8, h),
        '\u{258A}' => fill_rect(bitmap, 0, 0, w * 3 / 4, h),
        '\u{258B}' => fill_rect(bitmap, 0, 0, w * 5 / 8, h),
        '\u{258C}' => fill_rect(bitmap, 0, 0, w / 2, h),
        '\u{258D}' => fill_rect(bitmap, 0, 0, w * 3 / 8, h),
        '\u{258E}' => fill_rect(bitmap, 0, 0, w / 4, h),
        '\u{258F}' => fill_rect(bitmap, 0, 0, w / 8, h),
        '\u{2590}' => fill_rect(bitmap, w / 2, 0, w, h),
        // Shades
        '\u{2591}' => {
            for y in 0..h {
                for x in 0..w {
                    if (x + y) % 4 == 0 { bitmap[y * w + x] = 255; }
                }
            }
        }
        '\u{2592}' => {
            for y in 0..h {
                for x in 0..w {
                    if (x + y) % 2 == 0 { bitmap[y * w + x] = 255; }
                }
            }
        }
        '\u{2593}' => {
            for y in 0..h {
                for x in 0..w {
                    if (x + y) % 4 != 0 { bitmap[y * w + x] = 255; }
                }
            }
        }
        // Right half and upper eighth
        '\u{2595}' => fill_rect(bitmap, w * 7 / 8, 0, w, h),
        '\u{2594}' => fill_rect(bitmap, 0, 0, w, h / 8),
        // Quadrants
        '\u{2596}' => fill_rect(bitmap, 0, h / 2, w / 2, h),
        '\u{2597}' => fill_rect(bitmap, w / 2, h / 2, w, h),
        '\u{2598}' => fill_rect(bitmap, 0, 0, w / 2, h / 2),
        '\u{2599}' => {
            fill_rect(bitmap, 0, 0, w / 2, h);
            fill_rect(bitmap, w / 2, h / 2, w, h);
        }
        '\u{259A}' => {
            fill_rect(bitmap, 0, 0, w / 2, h / 2);
            fill_rect(bitmap, w / 2, h / 2, w, h);
        }
        '\u{259B}' => {
            fill_rect(bitmap, 0, 0, w, h / 2);
            fill_rect(bitmap, 0, h / 2, w / 2, h);
        }
        '\u{259C}' => {
            fill_rect(bitmap, 0, 0, w, h / 2);
            fill_rect(bitmap, w / 2, h / 2, w, h);
        }
        '\u{259D}' => fill_rect(bitmap, w / 2, 0, w, h / 2),
        '\u{259E}' => {
            fill_rect(bitmap, w / 2, 0, w, h / 2);
            fill_rect(bitmap, 0, h / 2, w / 2, h);
        }
        '\u{259F}' => {
            fill_rect(bitmap, w / 2, 0, w, h);
            fill_rect(bitmap, 0, h / 2, w / 2, h);
        }
        // Delegate to part 7
        _ => {
            return render_box_char_part7(
                c, bitmap, supersampled,
                w, h, dpi,
            );
        }
    }
    Some(())
}

// Part 7 - Braille patterns
fn render_box_char_part7(
    c: char,
    bitmap: &mut [u8],
    supersampled: &mut bool,
    w: usize,
    h: usize,
    dpi: f64,
) -> Option<()> {
    match c {
        // Braille patterns (U+2800-U+28FF)
        c if (0x2800..=0x28FF).contains(&(c as u32)) => {
            let which = (c as u32 - 0x2800) as u8;
            if which != 0 {
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
        // Delegate to part 8 for Powerline
        _ => {
            return render_box_char_part8(c, bitmap, supersampled, w, h, dpi);
        }
    }
    Some(())
}

// Part 8 - Powerline symbols (U+E0B0-U+E0BF)
fn render_box_char_part8(
    c: char,
    bitmap: &mut [u8],
    supersampled: &mut bool,
    w: usize,
    h: usize,
    dpi: f64,
) -> Option<()> {
    match c {
        // E0B0: Right-pointing solid triangle
        '\u{E0B0}' => {
            let mut canvas = SupersampledCanvas::new(w, h);
            canvas.fill_powerline_arrow(false, false);
            canvas.downsample(bitmap);
            *supersampled = true;
        }
        // E0B1: Right-pointing chevron (outline)
        '\u{E0B1}' => {
            let mut canvas = SupersampledCanvas::new(w, h);
            let thickness = (box_thickness(1, dpi) * SupersampledCanvas::FACTOR as f64).round() as usize;
            canvas.stroke_powerline_arrow(false, thickness);
            canvas.downsample(bitmap);
            *supersampled = true;
        }
        // E0B2: Left-pointing solid triangle
        '\u{E0B2}' => {
            let mut canvas = SupersampledCanvas::new(w, h);
            canvas.fill_powerline_arrow(true, false);
            canvas.downsample(bitmap);
            *supersampled = true;
        }
        // E0B3: Left-pointing chevron (outline)
        '\u{E0B3}' => {
            let mut canvas = SupersampledCanvas::new(w, h);
            let thickness = (box_thickness(1, dpi) * SupersampledCanvas::FACTOR as f64).round() as usize;
            canvas.stroke_powerline_arrow(true, thickness);
            canvas.downsample(bitmap);
            *supersampled = true;
        }
        // E0B4: Right semicircle (filled)
        '\u{E0B4}' => {
            let mut canvas = SupersampledCanvas::new(w, h);
            canvas.fill_bezier_d(false);
            canvas.downsample(bitmap);
            *supersampled = true;
        }
        // E0B5: Right semicircle (outline)
        '\u{E0B5}' => {
            let mut canvas = SupersampledCanvas::new(w, h);
            let thickness = box_thickness(1, dpi) * SupersampledCanvas::FACTOR as f64;
            canvas.stroke_bezier_d(false, thickness);
            canvas.downsample(bitmap);
            *supersampled = true;
        }
        // E0B6: Left semicircle (filled)
        '\u{E0B6}' => {
            let mut canvas = SupersampledCanvas::new(w, h);
            canvas.fill_bezier_d(true);
            canvas.downsample(bitmap);
            *supersampled = true;
        }
        // E0B7: Left semicircle (outline)
        '\u{E0B7}' => {
            let mut canvas = SupersampledCanvas::new(w, h);
            let thickness = box_thickness(1, dpi) * SupersampledCanvas::FACTOR as f64;
            canvas.stroke_bezier_d(true, thickness);
            canvas.downsample(bitmap);
            *supersampled = true;
        }
        // E0B8-E0BF: Corner triangles
        '\u{E0B8}' => {
            let mut canvas = SupersampledCanvas::new(w, h);
            canvas.fill_corner_triangle(Corner::BottomLeft, false);
            canvas.downsample(bitmap);
            *supersampled = true;
        }
        '\u{E0B9}' => {
            let mut canvas = SupersampledCanvas::new(w, h);
            canvas.fill_corner_triangle(Corner::BottomLeft, true);
            canvas.downsample(bitmap);
            *supersampled = true;
        }
        '\u{E0BA}' => {
            let mut canvas = SupersampledCanvas::new(w, h);
            canvas.fill_corner_triangle(Corner::TopLeft, false);
            canvas.downsample(bitmap);
            *supersampled = true;
        }
        '\u{E0BB}' => {
            let mut canvas = SupersampledCanvas::new(w, h);
            canvas.fill_corner_triangle(Corner::TopLeft, true);
            canvas.downsample(bitmap);
            *supersampled = true;
        }
        '\u{E0BC}' => {
            let mut canvas = SupersampledCanvas::new(w, h);
            canvas.fill_corner_triangle(Corner::BottomRight, false);
            canvas.downsample(bitmap);
            *supersampled = true;
        }
        '\u{E0BD}' => {
            let mut canvas = SupersampledCanvas::new(w, h);
            canvas.fill_corner_triangle(Corner::BottomRight, true);
            canvas.downsample(bitmap);
            *supersampled = true;
        }
        '\u{E0BE}' => {
            let mut canvas = SupersampledCanvas::new(w, h);
            canvas.fill_corner_triangle(Corner::TopRight, false);
            canvas.downsample(bitmap);
            *supersampled = true;
        }
        '\u{E0BF}' => {
            let mut canvas = SupersampledCanvas::new(w, h);
            canvas.fill_corner_triangle(Corner::TopRight, true);
            canvas.downsample(bitmap);
            *supersampled = true;
        }
        // Delegate to part 9 for geometric shapes
        _ => {
            return render_box_char_part9(c, bitmap, supersampled, w, h, dpi);
        }
    }
    Some(())
}

// Part 9 - Geometric shapes (circles, arcs)
fn render_box_char_part9(
    c: char,
    bitmap: &mut [u8],
    supersampled: &mut bool,
    w: usize,
    h: usize,
    dpi: f64,
) -> Option<()> {
    match c {
        // Black circle (filled)
        '\u{25CF}' => {
            let mut canvas = SupersampledCanvas::new(w, h);
            canvas.fill_circle(1.0);
            canvas.downsample(bitmap);
            *supersampled = true;
        }
        // White circle (outline)
        '\u{25CB}' => {
            let mut canvas = SupersampledCanvas::new(w, h);
            let line_width = box_thickness(1, dpi) * SupersampledCanvas::FACTOR as f64;
            let half_line = line_width / 2.0;
            let cx = canvas.ss_width as f64 / 2.0;
            let cy = canvas.ss_height as f64 / 2.0;
            let radius = 0.0_f64.max(cx.min(cy) - half_line);
            canvas.stroke_circle(radius, line_width);
            canvas.downsample(bitmap);
            *supersampled = true;
        }
        // Fisheye
        '\u{25C9}' => {
            let mut canvas = SupersampledCanvas::new(w, h);
            let cx = canvas.ss_width as f64 / 2.0;
            let cy = canvas.ss_height as f64 / 2.0;
            let radius = cx.min(cy);
            let central_radius = (2.0 / 3.0) * radius;
            canvas.fill_circle_radius(central_radius);
            let line_width = (SupersampledCanvas::FACTOR as f64).max((radius - central_radius) / 2.5);
            let outer_radius = 0.0_f64.max(cx.min(cy) - line_width / 2.0);
            canvas.stroke_circle(outer_radius, line_width);
            canvas.downsample(bitmap);
            *supersampled = true;
        }
        // Quadrant arcs
        '\u{25DC}' => {
            let mut canvas = SupersampledCanvas::new(w, h);
            let line_width = box_thickness(1, dpi) * SupersampledCanvas::FACTOR as f64;
            let half_line = 0.5_f64.max(line_width / 2.0);
            let cx = canvas.ss_width as f64 / 2.0;
            let cy = canvas.ss_height as f64 / 2.0;
            let radius = 0.0_f64.max(cx.min(cy) - half_line);
            canvas.stroke_arc(radius, line_width, std::f64::consts::PI, 3.0 * std::f64::consts::PI / 2.0);
            canvas.downsample(bitmap);
            *supersampled = true;
        }
        '\u{25DD}' => {
            let mut canvas = SupersampledCanvas::new(w, h);
            let line_width = box_thickness(1, dpi) * SupersampledCanvas::FACTOR as f64;
            let half_line = 0.5_f64.max(line_width / 2.0);
            let cx = canvas.ss_width as f64 / 2.0;
            let cy = canvas.ss_height as f64 / 2.0;
            let radius = 0.0_f64.max(cx.min(cy) - half_line);
            canvas.stroke_arc(radius, line_width, 3.0 * std::f64::consts::PI / 2.0, 2.0 * std::f64::consts::PI);
            canvas.downsample(bitmap);
            *supersampled = true;
        }
        '\u{25DE}' => {
            let mut canvas = SupersampledCanvas::new(w, h);
            let line_width = box_thickness(1, dpi) * SupersampledCanvas::FACTOR as f64;
            let half_line = 0.5_f64.max(line_width / 2.0);
            let cx = canvas.ss_width as f64 / 2.0;
            let cy = canvas.ss_height as f64 / 2.0;
            let radius = 0.0_f64.max(cx.min(cy) - half_line);
            canvas.stroke_arc(radius, line_width, 0.0, std::f64::consts::PI / 2.0);
            canvas.downsample(bitmap);
            *supersampled = true;
        }
        '\u{25DF}' => {
            let mut canvas = SupersampledCanvas::new(w, h);
            let line_width = box_thickness(1, dpi) * SupersampledCanvas::FACTOR as f64;
            let half_line = 0.5_f64.max(line_width / 2.0);
            let cx = canvas.ss_width as f64 / 2.0;
            let cy = canvas.ss_height as f64 / 2.0;
            let radius = 0.0_f64.max(cx.min(cy) - half_line);
            canvas.stroke_arc(radius, line_width, std::f64::consts::PI / 2.0, std::f64::consts::PI);
            canvas.downsample(bitmap);
            *supersampled = true;
        }
        // Half arcs
        '\u{25E0}' => {
            let mut canvas = SupersampledCanvas::new(w, h);
            let line_width = box_thickness(1, dpi) * SupersampledCanvas::FACTOR as f64;
            let half_line = 0.5_f64.max(line_width / 2.0);
            let cx = canvas.ss_width as f64 / 2.0;
            let cy = canvas.ss_height as f64 / 2.0;
            let radius = 0.0_f64.max(cx.min(cy) - half_line);
            canvas.stroke_arc(radius, line_width, std::f64::consts::PI, 2.0 * std::f64::consts::PI);
            canvas.downsample(bitmap);
            *supersampled = true;
        }
        '\u{25E1}' => {
            let mut canvas = SupersampledCanvas::new(w, h);
            let line_width = box_thickness(1, dpi) * SupersampledCanvas::FACTOR as f64;
            let half_line = 0.5_f64.max(line_width / 2.0);
            let cx = canvas.ss_width as f64 / 2.0;
            let cy = canvas.ss_height as f64 / 2.0;
            let radius = 0.0_f64.max(cx.min(cy) - half_line);
            canvas.stroke_arc(radius, line_width, 0.0, std::f64::consts::PI);
            canvas.downsample(bitmap);
            *supersampled = true;
        }
        // Unimplemented character
        _ => return None,
    }
    Some(())
}
