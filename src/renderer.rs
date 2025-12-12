//! GPU-accelerated terminal rendering using wgpu with a glyph atlas.
//! Uses rustybuzz (HarfBuzz port) for text shaping to support font features.

use crate::config::TabBarPosition;
use crate::protocol::{CellColor, CursorStyle, PaneId, PaneInfo, PaneSnapshot, TabInfo};
use crate::terminal::{Color, ColorPalette, CursorShape, Terminal};
use fontdue::Font as FontdueFont;
use rustybuzz::UnicodeBuffer;
use std::collections::HashMap;
use std::sync::Arc;

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
    /// Advance width (how much to move cursor after this glyph).
    advance: f32,
}

/// Wrapper to hold the rustybuzz Face with a 'static lifetime.
/// This is safe because we keep font_data alive for the lifetime of the Renderer.
struct ShapingContext {
    face: rustybuzz::Face<'static>,
}

/// Result of shaping a text sequence.
#[derive(Clone, Debug)]
struct ShapedGlyphs {
    /// Glyph IDs and their advances.
    glyphs: Vec<(u16, f32)>,
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
    font_data: Box<[u8]>,
    fontdue_font: FontdueFont,
    fallback_fonts: Vec<FontdueFont>,
    fallback_font_paths: Vec<&'static str>,  // Paths for lazy loading
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
}

use crate::config::Config;

impl Renderer {
    /// Creates a new renderer for the given window.
    pub async fn new(window: Arc<winit::window::Window>, config: &Config) -> Self {
        let size = window.inner_size();
        let scale_factor = window.scale_factor();

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
            present_mode: wgpu::PresentMode::Mailbox,
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

        // Store fallback font paths for lazy loading (loaded on first cache miss)
        let fallback_font_paths: Vec<&'static str> = vec![
            // Nerd Font symbols
            "/usr/share/fonts/TTF/SymbolsNerdFont-Regular.ttf",
            "/usr/share/fonts/TTF/SymbolsNerdFontMono-Regular.ttf",
            // Noto fonts for broad Unicode coverage
            "/usr/share/fonts/noto/NotoSansMono-Regular.ttf",
            "/usr/share/fonts/noto/NotoSansSymbols-Regular.ttf",
            "/usr/share/fonts/noto/NotoSansSymbols2-Regular.ttf",
            "/usr/share/fonts/noto/NotoEmoji-Regular.ttf",
            // DejaVu has good symbol coverage
            "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
            "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
        ];
        // Start with empty fallback fonts - will be loaded lazily
        let fallback_fonts: Vec<FontdueFont> = Vec::new();

        // Create rustybuzz Face for text shaping (ligatures).
        // SAFETY: We transmute to 'static because font_data lives as long as Renderer.
        // The Face only borrows the data, so this is safe as long as we don't drop font_data
        // before dropping the Face, which is guaranteed by struct drop order.
        let face: rustybuzz::Face<'static> = {
            let face = rustybuzz::Face::from_slice(&font_data, 0)
                .expect("Failed to parse font for shaping");
            unsafe { std::mem::transmute(face) }
        };
        let shaping_ctx = ShapingContext { face };

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
            fallback_font_paths,
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
            font_size,
            cell_width,
            cell_height,
            width: size.width,
            height: size.height,
            palette: ColorPalette::default(),
            tab_bar_position: config.tab_bar_position,
            background_opacity: config.background_opacity.clamp(0.0, 1.0),
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

    /// Check if a character is a box-drawing character that should be rendered procedurally.
    fn is_box_drawing(c: char) -> bool {
        let cp = c as u32;
        // Box Drawing: U+2500-U+257F
        // Block Elements: U+2580-U+259F
        (0x2500..=0x257F).contains(&cp) || (0x2580..=0x259F).contains(&cp)
    }

    /// Render a box-drawing character procedurally to a bitmap.
    /// Returns (bitmap, width, height) where the bitmap fills the entire cell.
    fn render_box_char(&self, c: char) -> Option<Vec<u8>> {
        let w = self.cell_width.ceil() as usize;
        let h = self.cell_height.ceil() as usize;
        let mut bitmap = vec![0u8; w * h];
        
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
            
            // Right half blocks
            '▕' => fill_rect(&mut bitmap, w * 7 / 8, 0, w, h),
            
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

            _ => return None,
        }

        Some(bitmap)
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
            if let Some(bitmap) = self.render_box_char(c) {
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
                        advance: self.cell_width,
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

                // Box-drawing chars fill the entire cell, positioned at origin
                let info = GlyphInfo {
                    uv: [uv_x, uv_y, uv_w, uv_h],
                    offset: [0.0, 0.0],
                    size: [glyph_width as f32, glyph_height as f32],
                    advance: self.cell_width,
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
                // Lazy load fallback fonts on first cache miss if not yet loaded
                if self.fallback_fonts.is_empty() && !self.fallback_font_paths.is_empty() {
                    log::debug!("Loading fallback fonts lazily...");
                    let paths = std::mem::take(&mut self.fallback_font_paths);
                    for path in paths {
                        if let Ok(data) = std::fs::read(path) {
                            if let Ok(font) = FontdueFont::from_bytes(data.as_slice(), fontdue::FontSettings::default()) {
                                log::debug!("Loaded fallback font: {}", path);
                                self.fallback_fonts.push(font);
                            }
                        }
                    }
                    log::debug!("Loaded {} fallback fonts", self.fallback_fonts.len());
                }
                
                // Try fallback fonts
                let mut result = None;
                for fallback in &self.fallback_fonts {
                    let fb_glyph_idx = fallback.lookup_glyph_index(c);
                    if fb_glyph_idx != 0 {
                        result = Some(fallback.rasterize(c, self.font_size));
                        break;
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
                advance: metrics.advance_width,
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
                advance: metrics.advance_width,
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
            advance: metrics.advance_width,
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
                advance: metrics.advance_width,
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
                advance: metrics.advance_width,
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
            advance: metrics.advance_width,
        };

        // Update atlas cursor
        self.atlas_cursor_x += glyph_width + 1;
        self.atlas_row_height = self.atlas_row_height.max(glyph_height);

        self.glyph_cache.insert(cache_key, info);
        info
    }

    /// Shape a multi-character text string (for ligatures).
    /// Returns the shaped glyphs. If the font produces a ligature,
    /// there will be fewer glyphs than input characters.
    fn shape_text(&mut self, text: &str) -> ShapedGlyphs {
        // Check cache first
        if let Some(cached) = self.ligature_cache.get(text) {
            return cached.clone();
        }

        let mut buffer = UnicodeBuffer::new();
        buffer.push_str(text);

        let glyph_buffer = rustybuzz::shape(&self.shaping_ctx.face, &[], buffer);
        let glyph_infos = glyph_buffer.glyph_infos();
        let glyph_positions = glyph_buffer.glyph_positions();

        let glyphs: Vec<(u16, f32)> = glyph_infos
            .iter()
            .zip(glyph_positions.iter())
            .map(|(info, pos)| {
                let glyph_id = info.glyph_id as u16;
                // Ensure glyph is rasterized
                self.get_glyph_by_id(glyph_id);
                // Convert advance from font units to pixels
                // rustybuzz uses 26.6 fixed point, so divide by 64
                let advance = pos.x_advance as f32 / 64.0;
                (glyph_id, advance)
            })
            .collect();

        let shaped = ShapedGlyphs { glyphs };
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

    /// Converts a terminal Color to RGBA in linear color space using the palette.
    /// Default backgrounds are fully transparent to let the window clear color show through.
    /// Explicit background colors remain fully opaque.
    fn color_to_rgba(color: &Color, is_foreground: bool, palette: &crate::terminal::ColorPalette) -> [f32; 4] {
        // For default background: fully transparent so clear color shows through
        if !is_foreground && *color == Color::Default {
            return [0.0, 0.0, 0.0, 0.0];
        }
        
        let srgb = if is_foreground {
            palette.to_rgba(color)
        } else {
            palette.to_rgba_bg(color)
        };
        // Convert sRGB to linear for the GPU (which will convert back to sRGB for display)
        [
            Self::srgb_to_linear(srgb[0]),
            Self::srgb_to_linear(srgb[1]),
            Self::srgb_to_linear(srgb[2]),
            srgb[3], // Alpha stays linear (1.0 for explicit colors)
        ]
    }

    /// Converts a protocol CellColor to RGBA in linear color space.
    /// Default backgrounds are fully transparent to let the window clear color show through.
    /// Explicit background colors remain fully opaque.
    fn cell_color_to_rgba(&self, color: &CellColor, is_foreground: bool) -> [f32; 4] {
        // For default background: fully transparent so clear color shows through
        if !is_foreground && *color == CellColor::Default {
            return [0.0, 0.0, 0.0, 0.0];
        }

        let srgb = match color {
            CellColor::Default => {
                // Only foreground gets here (background returns early above)
                let [r, g, b] = self.palette.default_fg;
                [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0]
            }
            CellColor::Rgb(r, g, b) => [*r as f32 / 255.0, *g as f32 / 255.0, *b as f32 / 255.0, 1.0],
            CellColor::Indexed(idx) => {
                let [r, g, b] = self.palette.colors[*idx as usize];
                [r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0]
            }
        };
        [
            Self::srgb_to_linear(srgb[0]),
            Self::srgb_to_linear(srgb[1]),
            Self::srgb_to_linear(srgb[2]),
            srgb[3],
        ]
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

    /// Renders the terminal.
    pub fn render(&mut self, terminal: &mut Terminal) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Use two separate lists: backgrounds first, then glyphs
        // This ensures wide glyphs (like Nerd Font icons) can extend beyond their cell
        // without being covered by adjacent cell backgrounds
        let mut bg_vertices: Vec<GlyphVertex> = Vec::with_capacity(terminal.cols * terminal.rows * 4);
        let mut bg_indices: Vec<u32> = Vec::with_capacity(terminal.cols * terminal.rows * 6);
        let mut glyph_vertices: Vec<GlyphVertex> = Vec::with_capacity(terminal.cols * terminal.rows * 4);
        let mut glyph_indices: Vec<u32> = Vec::with_capacity(terminal.cols * terminal.rows * 6);

        let width = self.width as f32;
        let height = self.height as f32;

        // Common programming ligatures to check (longest first for greedy matching)
        const LIGATURE_PATTERNS: &[&str] = &[
            // 3-char
            "===", "!==", ">>>", "<<<", "||=", "&&=", "??=", "...", "-->", "<--", "<->",
            // 2-char  
            "=>", "->", "<-", ">=", "<=", "==", "!=", "::", "&&", "||", "??", "..", "++",
            "--", "<<", ">>", "|>", "<|", "/*", "*/", "//", "##", ":=", "~=", "<>",
        ];

        for (row_idx, row) in terminal.grid.iter().enumerate() {
            let mut col_idx = 0;
            while col_idx < row.len() {
                let cell = &row[col_idx];
                let cell_x = col_idx as f32 * self.cell_width;
                let cell_y = row_idx as f32 * self.cell_height;

                let fg_color = Self::color_to_rgba(&cell.fg_color, true, &terminal.palette);
                let bg_color = Self::color_to_rgba(&cell.bg_color, false, &terminal.palette);

                // Check for ligatures by looking ahead
                let mut ligature_len = 0;
                let mut ligature_glyph: Option<GlyphInfo> = None;

                for pattern in LIGATURE_PATTERNS {
                    let pat_len = pattern.len();
                    if col_idx + pat_len <= row.len() {
                        // Build the candidate string from consecutive cells
                        let candidate: String = row[col_idx..col_idx + pat_len]
                            .iter()
                            .map(|c| c.character)
                            .collect();
                        
                        if candidate == *pattern {
                            // Check if font actually produces a ligature
                            let shaped = self.shape_text(&candidate);
                            if shaped.glyphs.len() == 1 {
                                // It's a ligature!
                                let glyph_id = shaped.glyphs[0].0;
                                ligature_glyph = Some(self.get_glyph_by_id(glyph_id));
                                ligature_len = pat_len;
                                break;
                            }
                        }
                    }
                }

                if let Some(glyph) = ligature_glyph {
                    // Render ligature spanning multiple cells
                    let span_width = ligature_len as f32 * self.cell_width;
                    
                    // Add background for all cells in the ligature
                    for i in 0..ligature_len {
                        let bg_cell_x = (col_idx + i) as f32 * self.cell_width;
                        let cell_left = Self::pixel_to_ndc_x(bg_cell_x, width);
                        let cell_right = Self::pixel_to_ndc_x(bg_cell_x + self.cell_width, width);
                        let cell_top = Self::pixel_to_ndc_y(cell_y, height);
                        let cell_bottom = Self::pixel_to_ndc_y(cell_y + self.cell_height, height);

                        let base_idx = bg_vertices.len() as u32;
                        bg_vertices.push(GlyphVertex {
                            position: [cell_left, cell_top],
                            uv: [0.0, 0.0],
                            color: fg_color,
                            bg_color,
                        });
                        bg_vertices.push(GlyphVertex {
                            position: [cell_right, cell_top],
                            uv: [0.0, 0.0],
                            color: fg_color,
                            bg_color,
                        });
                        bg_vertices.push(GlyphVertex {
                            position: [cell_right, cell_bottom],
                            uv: [0.0, 0.0],
                            color: fg_color,
                            bg_color,
                        });
                        bg_vertices.push(GlyphVertex {
                            position: [cell_left, cell_bottom],
                            uv: [0.0, 0.0],
                            color: fg_color,
                            bg_color,
                        });
                        bg_indices.extend_from_slice(&[
                            base_idx, base_idx + 1, base_idx + 2,
                            base_idx, base_idx + 2, base_idx + 3,
                        ]);
                    }

                    // Add the ligature glyph centered over the span
                    if glyph.size[0] > 0.0 && glyph.size[1] > 0.0 {
                        let baseline_y = (cell_y + self.cell_height * 0.8).round();
                        // Center the ligature horizontally over the span
                        let glyph_x = (cell_x + (span_width - glyph.size[0]) / 2.0 + glyph.offset[0]).round();
                        let glyph_y = (baseline_y - glyph.offset[1] - glyph.size[1]).round();

                        let left = Self::pixel_to_ndc_x(glyph_x, width);
                        let right = Self::pixel_to_ndc_x(glyph_x + glyph.size[0], width);
                        let top = Self::pixel_to_ndc_y(glyph_y, height);
                        let bottom = Self::pixel_to_ndc_y(glyph_y + glyph.size[1], height);

                        let base_idx = glyph_vertices.len() as u32;
                        glyph_vertices.push(GlyphVertex {
                            position: [left, top],
                            uv: [glyph.uv[0], glyph.uv[1]],
                            color: fg_color,
                            bg_color: [0.0, 0.0, 0.0, 0.0],
                        });
                        glyph_vertices.push(GlyphVertex {
                            position: [right, top],
                            uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1]],
                            color: fg_color,
                            bg_color: [0.0, 0.0, 0.0, 0.0],
                        });
                        glyph_vertices.push(GlyphVertex {
                            position: [right, bottom],
                            uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1] + glyph.uv[3]],
                            color: fg_color,
                            bg_color: [0.0, 0.0, 0.0, 0.0],
                        });
                        glyph_vertices.push(GlyphVertex {
                            position: [left, bottom],
                            uv: [glyph.uv[0], glyph.uv[1] + glyph.uv[3]],
                            color: fg_color,
                            bg_color: [0.0, 0.0, 0.0, 0.0],
                        });
                        glyph_indices.extend_from_slice(&[
                            base_idx, base_idx + 1, base_idx + 2,
                            base_idx, base_idx + 2, base_idx + 3,
                        ]);
                    }

                    col_idx += ligature_len;
                } else {
                    // Regular single-character rendering with font fallback
                    let glyph = self.rasterize_char(cell.character);

                    // Cell bounds (pixel-aligned)
                    let cell_left = Self::pixel_to_ndc_x(cell_x, width);
                    let cell_right = Self::pixel_to_ndc_x(cell_x + self.cell_width, width);
                    let cell_top = Self::pixel_to_ndc_y(cell_y, height);
                    let cell_bottom = Self::pixel_to_ndc_y(cell_y + self.cell_height, height);

                    // Add background quad
                    let base_idx = bg_vertices.len() as u32;
                    bg_vertices.push(GlyphVertex {
                        position: [cell_left, cell_top],
                        uv: [0.0, 0.0],
                        color: fg_color,
                        bg_color,
                    });
                    bg_vertices.push(GlyphVertex {
                        position: [cell_right, cell_top],
                        uv: [0.0, 0.0],
                        color: fg_color,
                        bg_color,
                    });
                    bg_vertices.push(GlyphVertex {
                        position: [cell_right, cell_bottom],
                        uv: [0.0, 0.0],
                        color: fg_color,
                        bg_color,
                    });
                    bg_vertices.push(GlyphVertex {
                        position: [cell_left, cell_bottom],
                        uv: [0.0, 0.0],
                        color: fg_color,
                        bg_color,
                    });
                    bg_indices.extend_from_slice(&[
                        base_idx, base_idx + 1, base_idx + 2,
                        base_idx, base_idx + 2, base_idx + 3,
                    ]);

                    // Add glyph quad if it has content
                    if glyph.size[0] > 0.0 && glyph.size[1] > 0.0 {
                        // Box-drawing characters fill the entire cell
                        let (glyph_x, glyph_y) = if Self::is_box_drawing(cell.character) {
                            (cell_x, cell_y)
                        } else {
                            // Calculate glyph position with pixel alignment
                            let baseline_y = (cell_y + self.cell_height * 0.8).round();
                            let gx = (cell_x + glyph.offset[0]).round();
                            let gy = (baseline_y - glyph.offset[1] - glyph.size[1]).round();
                            (gx, gy)
                        };

                        // Glyph quad (pixel-aligned) - no clipping, allow overflow
                        let left = Self::pixel_to_ndc_x(glyph_x, width);
                        let right = Self::pixel_to_ndc_x(glyph_x + glyph.size[0], width);
                        let top = Self::pixel_to_ndc_y(glyph_y, height);
                        let bottom = Self::pixel_to_ndc_y(glyph_y + glyph.size[1], height);

                        let base_idx = glyph_vertices.len() as u32;
                        glyph_vertices.push(GlyphVertex {
                            position: [left, top],
                            uv: [glyph.uv[0], glyph.uv[1]],
                            color: fg_color,
                            bg_color: [0.0, 0.0, 0.0, 0.0],
                        });
                        glyph_vertices.push(GlyphVertex {
                            position: [right, top],
                            uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1]],
                            color: fg_color,
                            bg_color: [0.0, 0.0, 0.0, 0.0],
                        });
                        glyph_vertices.push(GlyphVertex {
                            position: [right, bottom],
                            uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1] + glyph.uv[3]],
                            color: fg_color,
                            bg_color: [0.0, 0.0, 0.0, 0.0],
                        });
                        glyph_vertices.push(GlyphVertex {
                            position: [left, bottom],
                            uv: [glyph.uv[0], glyph.uv[1] + glyph.uv[3]],
                            color: fg_color,
                            bg_color: [0.0, 0.0, 0.0, 0.0],
                        });
                        glyph_indices.extend_from_slice(&[
                            base_idx, base_idx + 1, base_idx + 2,
                            base_idx, base_idx + 2, base_idx + 3,
                        ]);
                    }

                    col_idx += 1;
                }
            }
        }

        // Add cursor (rendered on top of everything)
        let cursor_x = terminal.cursor_col as f32 * self.cell_width;
        let cursor_y = terminal.cursor_row as f32 * self.cell_height;

        // Get the cell under the cursor to determine colors
        let cursor_cell = terminal.grid
            .get(terminal.cursor_row)
            .and_then(|row| row.get(terminal.cursor_col));

        // Get fg and bg colors from the cell under cursor
        let (cell_fg, cell_bg, cell_char) = if let Some(cell) = cursor_cell {
            let fg = Self::color_to_rgba(&cell.fg_color, true, &terminal.palette);
            let bg = Self::color_to_rgba(&cell.bg_color, false, &terminal.palette);
            (fg, bg, cell.character)
        } else {
            // Default colors if cell doesn't exist
            let fg = Self::color_to_rgba(&Color::Default, true, &terminal.palette);
            let bg = [0.0, 0.0, 0.0, 0.0];
            (fg, bg, ' ')
        };

        let has_character = cell_char != ' ' && cell_char != '\0';

        // Cursor color: invert the background, or use fg if there's a character
        let cursor_bg_color = if has_character {
            // Character present: cursor takes fg color as background
            [cell_fg[0], cell_fg[1], cell_fg[2], 1.0]
        } else {
            // Empty cell: invert the background color
            if cell_bg[3] < 0.01 {
                // Transparent background -> white cursor
                let white = Self::srgb_to_linear(0.9);
                [white, white, white, 1.0]
            } else {
                // Invert the background color
                [1.0 - cell_bg[0], 1.0 - cell_bg[1], 1.0 - cell_bg[2], 1.0]
            }
        };

        // Determine cursor bounds based on shape
        let (left, right, top, bottom) = match terminal.cursor_shape {
            CursorShape::BlinkingBlock | CursorShape::SteadyBlock => (
                cursor_x,
                cursor_x + self.cell_width,
                cursor_y,
                cursor_y + self.cell_height,
            ),
            CursorShape::BlinkingUnderline | CursorShape::SteadyUnderline => {
                let underline_height = 2.0_f32.max(self.cell_height * 0.1);
                (
                    cursor_x,
                    cursor_x + self.cell_width,
                    cursor_y + self.cell_height - underline_height,
                    cursor_y + self.cell_height,
                )
            }
            CursorShape::BlinkingBar | CursorShape::SteadyBar => {
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

        let base_idx = glyph_vertices.len() as u32;

        glyph_vertices.push(GlyphVertex {
            position: [cursor_left, cursor_top],
            uv: [0.0, 0.0],
            color: cursor_bg_color,
            bg_color: cursor_bg_color,
        });
        glyph_vertices.push(GlyphVertex {
            position: [cursor_right, cursor_top],
            uv: [0.0, 0.0],
            color: cursor_bg_color,
            bg_color: cursor_bg_color,
        });
        glyph_vertices.push(GlyphVertex {
            position: [cursor_right, cursor_bottom],
            uv: [0.0, 0.0],
            color: cursor_bg_color,
            bg_color: cursor_bg_color,
        });
        glyph_vertices.push(GlyphVertex {
            position: [cursor_left, cursor_bottom],
            uv: [0.0, 0.0],
            color: cursor_bg_color,
            bg_color: cursor_bg_color,
        });

        glyph_indices.extend_from_slice(&[
            base_idx,
            base_idx + 1,
            base_idx + 2,
            base_idx,
            base_idx + 2,
            base_idx + 3,
        ]);

        // If block cursor and there's a character, re-render it with inverted color
        let is_block_cursor = matches!(
            terminal.cursor_shape,
            CursorShape::BlinkingBlock | CursorShape::SteadyBlock
        );
        if is_block_cursor && has_character {
            // Character color: use bg color (inverted from normal)
            let char_color = if cell_bg[3] < 0.01 {
                // If bg was transparent, use black for the character
                [0.0, 0.0, 0.0, 1.0]
            } else {
                [cell_bg[0], cell_bg[1], cell_bg[2], 1.0]
            };

            let glyph = self.rasterize_char(cell_char);
            if glyph.size[0] > 0.0 && glyph.size[1] > 0.0 {
                let cell_x = cursor_x;
                let cell_y = cursor_y;
                let (glyph_x, glyph_y) = if Self::is_box_drawing(cell_char) {
                    (cell_x, cell_y)
                } else {
                    let baseline_y = (cell_y + self.cell_height * 0.8).round();
                    let gx = (cell_x + glyph.offset[0]).round();
                    let gy = (baseline_y - glyph.offset[1] - glyph.size[1]).round();
                    (gx, gy)
                };

                let g_left = Self::pixel_to_ndc_x(glyph_x, width);
                let g_right = Self::pixel_to_ndc_x(glyph_x + glyph.size[0], width);
                let g_top = Self::pixel_to_ndc_y(glyph_y, height);
                let g_bottom = Self::pixel_to_ndc_y(glyph_y + glyph.size[1], height);

                let base_idx = glyph_vertices.len() as u32;
                glyph_vertices.push(GlyphVertex {
                    position: [g_left, g_top],
                    uv: [glyph.uv[0], glyph.uv[1]],
                    color: char_color,
                    bg_color: [0.0, 0.0, 0.0, 0.0],
                });
                glyph_vertices.push(GlyphVertex {
                    position: [g_right, g_top],
                    uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1]],
                    color: char_color,
                    bg_color: [0.0, 0.0, 0.0, 0.0],
                });
                glyph_vertices.push(GlyphVertex {
                    position: [g_right, g_bottom],
                    uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1] + glyph.uv[3]],
                    color: char_color,
                    bg_color: [0.0, 0.0, 0.0, 0.0],
                });
                glyph_vertices.push(GlyphVertex {
                    position: [g_left, g_bottom],
                    uv: [glyph.uv[0], glyph.uv[1] + glyph.uv[3]],
                    color: char_color,
                    bg_color: [0.0, 0.0, 0.0, 0.0],
                });
                glyph_indices.extend_from_slice(&[
                    base_idx, base_idx + 1, base_idx + 2,
                    base_idx, base_idx + 2, base_idx + 3,
                ]);
            }
        }

        // Combine: backgrounds first, then glyphs (with adjusted indices)
        let mut vertices = bg_vertices;
        let mut indices = bg_indices;
        
        let glyph_vertex_offset = vertices.len() as u32;
        vertices.extend(glyph_vertices);
        indices.extend(glyph_indices.iter().map(|i| i + glyph_vertex_offset));

        // Resize buffers if needed
        if vertices.len() > self.vertex_capacity {
            self.vertex_capacity = vertices.len() * 2;
            self.vertex_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Glyph Vertex Buffer"),
                size: (self.vertex_capacity * std::mem::size_of::<GlyphVertex>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        if indices.len() > self.index_capacity {
            self.index_capacity = indices.len() * 2;
            self.index_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Glyph Index Buffer"),
                size: (self.index_capacity * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        // Upload vertex and index data
        self.queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        self.queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(&indices));

        // Upload atlas if dirty
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
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Render Encoder"),
            });

        {
            // Clear color from palette, converted to linear space
            let [bg_r, bg_g, bg_b] = terminal.palette.default_bg;
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
            render_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        terminal.dirty = false;

        Ok(())
    }

    /// Renders a pane from protocol data (used by client).
    pub fn render_pane(&mut self, pane: &PaneSnapshot) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let rows = pane.cells.len();
        let cols = if rows > 0 { pane.cells[0].len() } else { 0 };

        let mut bg_vertices: Vec<GlyphVertex> = Vec::with_capacity(cols * rows * 4);
        let mut bg_indices: Vec<u32> = Vec::with_capacity(cols * rows * 6);
        let mut glyph_vertices: Vec<GlyphVertex> = Vec::with_capacity(cols * rows * 4);
        let mut glyph_indices: Vec<u32> = Vec::with_capacity(cols * rows * 6);

        let width = self.width as f32;
        let height = self.height as f32;

        // Common programming ligatures to check (longest first for greedy matching)
        const LIGATURE_PATTERNS: &[&str] = &[
            // 3-char
            "===", "!==", ">>>", "<<<", "||=", "&&=", "??=", "...", "-->", "<--", "<->",
            // 2-char  
            "=>", "->", "<-", ">=", "<=", "==", "!=", "::", "&&", "||", "??", "..", "++",
            "--", "<<", ">>", "|>", "<|", "/*", "*/", "//", "##", ":=", "~=", "<>",
        ];

        for (row_idx, row) in pane.cells.iter().enumerate() {
            let mut col_idx = 0;
            while col_idx < row.len() {
                let cell = &row[col_idx];
                let cell_x = col_idx as f32 * self.cell_width;
                let cell_y = row_idx as f32 * self.cell_height;

                let fg_color = self.cell_color_to_rgba(&cell.fg_color, true);
                let bg_color = self.cell_color_to_rgba(&cell.bg_color, false);

                // Check for ligatures by looking ahead
                let mut ligature_len = 0;
                let mut ligature_glyph: Option<GlyphInfo> = None;

                for pattern in LIGATURE_PATTERNS {
                    let pat_len = pattern.len();
                    if col_idx + pat_len <= row.len() {
                        let candidate: String = row[col_idx..col_idx + pat_len]
                            .iter()
                            .map(|c| c.character)
                            .collect();
                        
                        if candidate == *pattern {
                            let shaped = self.shape_text(&candidate);
                            if shaped.glyphs.len() == 1 {
                                let glyph_id = shaped.glyphs[0].0;
                                ligature_glyph = Some(self.get_glyph_by_id(glyph_id));
                                ligature_len = pat_len;
                                break;
                            }
                        }
                    }
                }

                if let Some(glyph) = ligature_glyph {
                    let span_width = ligature_len as f32 * self.cell_width;
                    
                    for i in 0..ligature_len {
                        let bg_cell_x = (col_idx + i) as f32 * self.cell_width;
                        let cell_left = Self::pixel_to_ndc_x(bg_cell_x, width);
                        let cell_right = Self::pixel_to_ndc_x(bg_cell_x + self.cell_width, width);
                        let cell_top = Self::pixel_to_ndc_y(cell_y, height);
                        let cell_bottom = Self::pixel_to_ndc_y(cell_y + self.cell_height, height);

                        let base_idx = bg_vertices.len() as u32;
                        bg_vertices.push(GlyphVertex {
                            position: [cell_left, cell_top],
                            uv: [0.0, 0.0],
                            color: fg_color,
                            bg_color,
                        });
                        bg_vertices.push(GlyphVertex {
                            position: [cell_right, cell_top],
                            uv: [0.0, 0.0],
                            color: fg_color,
                            bg_color,
                        });
                        bg_vertices.push(GlyphVertex {
                            position: [cell_right, cell_bottom],
                            uv: [0.0, 0.0],
                            color: fg_color,
                            bg_color,
                        });
                        bg_vertices.push(GlyphVertex {
                            position: [cell_left, cell_bottom],
                            uv: [0.0, 0.0],
                            color: fg_color,
                            bg_color,
                        });
                        bg_indices.extend_from_slice(&[
                            base_idx, base_idx + 1, base_idx + 2,
                            base_idx, base_idx + 2, base_idx + 3,
                        ]);
                    }

                    if glyph.size[0] > 0.0 && glyph.size[1] > 0.0 {
                        let baseline_y = (cell_y + self.cell_height * 0.8).round();
                        let glyph_x = (cell_x + (span_width - glyph.size[0]) / 2.0 + glyph.offset[0]).round();
                        let glyph_y = (baseline_y - glyph.offset[1] - glyph.size[1]).round();

                        let left = Self::pixel_to_ndc_x(glyph_x, width);
                        let right = Self::pixel_to_ndc_x(glyph_x + glyph.size[0], width);
                        let top = Self::pixel_to_ndc_y(glyph_y, height);
                        let bottom = Self::pixel_to_ndc_y(glyph_y + glyph.size[1], height);

                        let base_idx = glyph_vertices.len() as u32;
                        glyph_vertices.push(GlyphVertex {
                            position: [left, top],
                            uv: [glyph.uv[0], glyph.uv[1]],
                            color: fg_color,
                            bg_color: [0.0, 0.0, 0.0, 0.0],
                        });
                        glyph_vertices.push(GlyphVertex {
                            position: [right, top],
                            uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1]],
                            color: fg_color,
                            bg_color: [0.0, 0.0, 0.0, 0.0],
                        });
                        glyph_vertices.push(GlyphVertex {
                            position: [right, bottom],
                            uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1] + glyph.uv[3]],
                            color: fg_color,
                            bg_color: [0.0, 0.0, 0.0, 0.0],
                        });
                        glyph_vertices.push(GlyphVertex {
                            position: [left, bottom],
                            uv: [glyph.uv[0], glyph.uv[1] + glyph.uv[3]],
                            color: fg_color,
                            bg_color: [0.0, 0.0, 0.0, 0.0],
                        });
                        glyph_indices.extend_from_slice(&[
                            base_idx, base_idx + 1, base_idx + 2,
                            base_idx, base_idx + 2, base_idx + 3,
                        ]);
                    }

                    col_idx += ligature_len;
                } else {
                    // Single character rendering
                    let cell_left = Self::pixel_to_ndc_x(cell_x, width);
                    let cell_right = Self::pixel_to_ndc_x(cell_x + self.cell_width, width);
                    let cell_top = Self::pixel_to_ndc_y(cell_y, height);
                    let cell_bottom = Self::pixel_to_ndc_y(cell_y + self.cell_height, height);

                    let base_idx = bg_vertices.len() as u32;
                    bg_vertices.push(GlyphVertex {
                        position: [cell_left, cell_top],
                        uv: [0.0, 0.0],
                        color: fg_color,
                        bg_color,
                    });
                    bg_vertices.push(GlyphVertex {
                        position: [cell_right, cell_top],
                        uv: [0.0, 0.0],
                        color: fg_color,
                        bg_color,
                    });
                    bg_vertices.push(GlyphVertex {
                        position: [cell_right, cell_bottom],
                        uv: [0.0, 0.0],
                        color: fg_color,
                        bg_color,
                    });
                    bg_vertices.push(GlyphVertex {
                        position: [cell_left, cell_bottom],
                        uv: [0.0, 0.0],
                        color: fg_color,
                        bg_color,
                    });
                    bg_indices.extend_from_slice(&[
                        base_idx, base_idx + 1, base_idx + 2,
                        base_idx, base_idx + 2, base_idx + 3,
                    ]);

                    let c = cell.character;
                    if c != ' ' && c != '\0' {
                        let glyph = self.rasterize_char(c);
                        if glyph.size[0] > 0.0 && glyph.size[1] > 0.0 {
                            // Box-drawing characters fill the entire cell
                            let (glyph_x, glyph_y) = if Self::is_box_drawing(c) {
                                (cell_x, cell_y)
                            } else {
                                // Calculate glyph position with baseline alignment
                                let baseline_y = (cell_y + self.cell_height * 0.8).round();
                                let gx = (cell_x + glyph.offset[0]).round();
                                let gy = (baseline_y - glyph.offset[1] - glyph.size[1]).round();
                                (gx, gy)
                            };

                            let left = Self::pixel_to_ndc_x(glyph_x, width);
                            let right = Self::pixel_to_ndc_x(glyph_x + glyph.size[0], width);
                            let top = Self::pixel_to_ndc_y(glyph_y, height);
                            let bottom = Self::pixel_to_ndc_y(glyph_y + glyph.size[1], height);

                            let base_idx = glyph_vertices.len() as u32;
                            glyph_vertices.push(GlyphVertex {
                                position: [left, top],
                                uv: [glyph.uv[0], glyph.uv[1]],
                                color: fg_color,
                                bg_color: [0.0, 0.0, 0.0, 0.0],
                            });
                            glyph_vertices.push(GlyphVertex {
                                position: [right, top],
                                uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1]],
                                color: fg_color,
                                bg_color: [0.0, 0.0, 0.0, 0.0],
                            });
                            glyph_vertices.push(GlyphVertex {
                                position: [right, bottom],
                                uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1] + glyph.uv[3]],
                                color: fg_color,
                                bg_color: [0.0, 0.0, 0.0, 0.0],
                            });
                            glyph_vertices.push(GlyphVertex {
                                position: [left, bottom],
                                uv: [glyph.uv[0], glyph.uv[1] + glyph.uv[3]],
                                color: fg_color,
                                bg_color: [0.0, 0.0, 0.0, 0.0],
                            });
                            glyph_indices.extend_from_slice(&[
                                base_idx, base_idx + 1, base_idx + 2,
                                base_idx, base_idx + 2, base_idx + 3,
                            ]);
                        }
                    }

                    col_idx += 1;
                }
            }
        }

        // Add cursor
        if pane.cursor.visible {
            let cursor_x = pane.cursor.col as f32 * self.cell_width;
            let cursor_y = pane.cursor.row as f32 * self.cell_height;

            // Get the cell under the cursor to determine colors
            let cursor_cell = pane.cells
                .get(pane.cursor.row)
                .and_then(|row| row.get(pane.cursor.col));

            // Get fg and bg colors from the cell under cursor
            let (cell_fg, cell_bg, cell_char) = if let Some(cell) = cursor_cell {
                let fg = self.cell_color_to_rgba(&cell.fg_color, true);
                let bg = self.cell_color_to_rgba(&cell.bg_color, false);
                (fg, bg, cell.character)
            } else {
                // Default colors if cell doesn't exist
                let fg = self.cell_color_to_rgba(&CellColor::Default, true);
                let bg = [0.0, 0.0, 0.0, 0.0];
                (fg, bg, ' ')
            };

            let has_character = cell_char != ' ' && cell_char != '\0';

            // Cursor color: invert the background, or use fg if there's a character
            let cursor_bg_color = if has_character {
                // Character present: cursor takes fg color as background
                [cell_fg[0], cell_fg[1], cell_fg[2], 1.0]
            } else {
                // Empty cell: invert the background color
                if cell_bg[3] < 0.01 {
                    // Transparent background -> white cursor
                    let white = Self::srgb_to_linear(0.9);
                    [white, white, white, 1.0]
                } else {
                    // Invert the background color
                    [1.0 - cell_bg[0], 1.0 - cell_bg[1], 1.0 - cell_bg[2], 1.0]
                }
            };

            // Determine cursor bounds based on style
            let (left, right, top, bottom) = match pane.cursor.style {
                CursorStyle::Block => (
                    cursor_x,
                    cursor_x + self.cell_width,
                    cursor_y,
                    cursor_y + self.cell_height,
                ),
                CursorStyle::Underline => {
                    let underline_height = 2.0_f32.max(self.cell_height * 0.1);
                    (
                        cursor_x,
                        cursor_x + self.cell_width,
                        cursor_y + self.cell_height - underline_height,
                        cursor_y + self.cell_height,
                    )
                }
                CursorStyle::Bar => {
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

            let base_idx = glyph_vertices.len() as u32;

            glyph_vertices.push(GlyphVertex {
                position: [cursor_left, cursor_top],
                uv: [0.0, 0.0],
                color: cursor_bg_color,
                bg_color: cursor_bg_color,
            });
            glyph_vertices.push(GlyphVertex {
                position: [cursor_right, cursor_top],
                uv: [0.0, 0.0],
                color: cursor_bg_color,
                bg_color: cursor_bg_color,
            });
            glyph_vertices.push(GlyphVertex {
                position: [cursor_right, cursor_bottom],
                uv: [0.0, 0.0],
                color: cursor_bg_color,
                bg_color: cursor_bg_color,
            });
            glyph_vertices.push(GlyphVertex {
                position: [cursor_left, cursor_bottom],
                uv: [0.0, 0.0],
                color: cursor_bg_color,
                bg_color: cursor_bg_color,
            });

            glyph_indices.extend_from_slice(&[
                base_idx, base_idx + 1, base_idx + 2,
                base_idx, base_idx + 2, base_idx + 3,
            ]);

            // If block cursor and there's a character, re-render it with inverted color
            if matches!(pane.cursor.style, CursorStyle::Block) && has_character {
                // Character color: use bg color (inverted from normal)
                let char_color = if cell_bg[3] < 0.01 {
                    // If bg was transparent, use black for the character
                    [0.0, 0.0, 0.0, 1.0]
                } else {
                    [cell_bg[0], cell_bg[1], cell_bg[2], 1.0]
                };

                let glyph = self.rasterize_char(cell_char);
                if glyph.size[0] > 0.0 && glyph.size[1] > 0.0 {
                    let cell_x = cursor_x;
                    let cell_y = cursor_y;
                    let (glyph_x, glyph_y) = if Self::is_box_drawing(cell_char) {
                        (cell_x, cell_y)
                    } else {
                        let baseline_y = (cell_y + self.cell_height * 0.8).round();
                        let gx = (cell_x + glyph.offset[0]).round();
                        let gy = (baseline_y - glyph.offset[1] - glyph.size[1]).round();
                        (gx, gy)
                    };

                    let g_left = Self::pixel_to_ndc_x(glyph_x, width);
                    let g_right = Self::pixel_to_ndc_x(glyph_x + glyph.size[0], width);
                    let g_top = Self::pixel_to_ndc_y(glyph_y, height);
                    let g_bottom = Self::pixel_to_ndc_y(glyph_y + glyph.size[1], height);

                    let base_idx = glyph_vertices.len() as u32;
                    glyph_vertices.push(GlyphVertex {
                        position: [g_left, g_top],
                        uv: [glyph.uv[0], glyph.uv[1]],
                        color: char_color,
                        bg_color: [0.0, 0.0, 0.0, 0.0],
                    });
                    glyph_vertices.push(GlyphVertex {
                        position: [g_right, g_top],
                        uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1]],
                        color: char_color,
                        bg_color: [0.0, 0.0, 0.0, 0.0],
                    });
                    glyph_vertices.push(GlyphVertex {
                        position: [g_right, g_bottom],
                        uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1] + glyph.uv[3]],
                        color: char_color,
                        bg_color: [0.0, 0.0, 0.0, 0.0],
                    });
                    glyph_vertices.push(GlyphVertex {
                        position: [g_left, g_bottom],
                        uv: [glyph.uv[0], glyph.uv[1] + glyph.uv[3]],
                        color: char_color,
                        bg_color: [0.0, 0.0, 0.0, 0.0],
                    });
                    glyph_indices.extend_from_slice(&[
                        base_idx, base_idx + 1, base_idx + 2,
                        base_idx, base_idx + 2, base_idx + 3,
                    ]);
                }
            }
        }

        // Combine vertices
        let mut vertices = bg_vertices;
        let mut indices = bg_indices;
        
        let glyph_vertex_offset = vertices.len() as u32;
        vertices.extend(glyph_vertices);
        indices.extend(glyph_indices.iter().map(|i| i + glyph_vertex_offset));

        // Resize buffers if needed
        if vertices.len() > self.vertex_capacity {
            self.vertex_capacity = vertices.len() * 2;
            self.vertex_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Glyph Vertex Buffer"),
                size: (self.vertex_capacity * std::mem::size_of::<GlyphVertex>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        if indices.len() > self.index_capacity {
            self.index_capacity = indices.len() * 2;
            self.index_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Glyph Index Buffer"),
                size: (self.index_capacity * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        // Upload data
        self.queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        self.queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(&indices));

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
            render_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }

    /// Renders multiple panes with tab bar from protocol data (used by client).
    /// The tab bar is rendered outside the terminal grid at the configured position.
    /// 
    /// Arguments:
    /// - `panes`: All pane snapshots with their layout info for the active tab
    /// - `active_pane_id`: The ID of the focused pane (for cursor rendering)
    /// - `tabs`: Tab information for the tab bar
    /// - `active_tab`: Index of the active tab
    pub fn render_with_tabs(
        &mut self,
        panes: &[(&PaneSnapshot, &PaneInfo)],
        active_pane_id: PaneId,
        tabs: &[TabInfo],
        active_tab: usize,
    ) -> Result<(), wgpu::SurfaceError> {
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Estimate total cells across all panes for buffer capacity
        let total_cells: usize = panes.iter()
            .map(|(snap, _)| snap.cells.len() * snap.cells.first().map_or(0, |r| r.len()))
            .sum();
        
        let mut bg_vertices: Vec<GlyphVertex> = Vec::with_capacity(total_cells * 4 + 64);
        let mut bg_indices: Vec<u32> = Vec::with_capacity(total_cells * 6 + 96);
        let mut glyph_vertices: Vec<GlyphVertex> = Vec::with_capacity(total_cells * 4 + 256);
        let mut glyph_indices: Vec<u32> = Vec::with_capacity(total_cells * 6 + 384);

        let width = self.width as f32;
        let height = self.height as f32;
        let tab_bar_height = self.tab_bar_height();
        let terminal_y_offset = self.terminal_y_offset();

        // ═══════════════════════════════════════════════════════════════════
        // RENDER TAB BAR (outside terminal grid)
        // ═══════════════════════════════════════════════════════════════════
        if self.tab_bar_position != TabBarPosition::Hidden && !tabs.is_empty() {
            let tab_bar_y = match self.tab_bar_position {
                TabBarPosition::Top => 0.0,
                TabBarPosition::Bottom => height - tab_bar_height,
                TabBarPosition::Hidden => unreachable!(),
            };

            // Tab bar background - slightly different from terminal background
            let tab_bar_bg = {
                let [r, g, b] = self.palette.default_bg;
                // Darken/lighten slightly for visual separation
                let factor = 0.85_f32;
                [
                    Self::srgb_to_linear((r as f32 / 255.0) * factor),
                    Self::srgb_to_linear((g as f32 / 255.0) * factor),
                    Self::srgb_to_linear((b as f32 / 255.0) * factor),
                    1.0,
                ]
            };

            // Draw tab bar background
            let bar_left = Self::pixel_to_ndc_x(0.0, width);
            let bar_right = Self::pixel_to_ndc_x(width, width);
            let bar_top = Self::pixel_to_ndc_y(tab_bar_y, height);
            let bar_bottom = Self::pixel_to_ndc_y(tab_bar_y + tab_bar_height, height);

            let base_idx = bg_vertices.len() as u32;
            bg_vertices.push(GlyphVertex {
                position: [bar_left, bar_top],
                uv: [0.0, 0.0],
                color: tab_bar_bg,
                bg_color: tab_bar_bg,
            });
            bg_vertices.push(GlyphVertex {
                position: [bar_right, bar_top],
                uv: [0.0, 0.0],
                color: tab_bar_bg,
                bg_color: tab_bar_bg,
            });
            bg_vertices.push(GlyphVertex {
                position: [bar_right, bar_bottom],
                uv: [0.0, 0.0],
                color: tab_bar_bg,
                bg_color: tab_bar_bg,
            });
            bg_vertices.push(GlyphVertex {
                position: [bar_left, bar_bottom],
                uv: [0.0, 0.0],
                color: tab_bar_bg,
                bg_color: tab_bar_bg,
            });
            bg_indices.extend_from_slice(&[
                base_idx, base_idx + 1, base_idx + 2,
                base_idx, base_idx + 2, base_idx + 3,
            ]);

            // Render each tab
            let mut tab_x = 4.0_f32; // Start with small padding
            let tab_padding = 8.0_f32;
            let min_tab_width = self.cell_width * 8.0; // Minimum width for tab

            for (idx, _tab) in tabs.iter().enumerate() {
                let is_active = idx == active_tab;
                
                // Generate tab title (for now, just "Tab N" or use first pane info)
                let title = format!(" {} ", idx + 1);
                let title_width = title.chars().count() as f32 * self.cell_width;
                let tab_width = title_width.max(min_tab_width);

                // Tab background
                let tab_bg = if is_active {
                    // Active tab uses terminal background
                    let [r, g, b] = self.palette.default_bg;
                    [
                        Self::srgb_to_linear(r as f32 / 255.0),
                        Self::srgb_to_linear(g as f32 / 255.0),
                        Self::srgb_to_linear(b as f32 / 255.0),
                        1.0,
                    ]
                } else {
                    // Inactive tabs are slightly darker
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

                // Tab background rect
                let tab_left = Self::pixel_to_ndc_x(tab_x, width);
                let tab_right = Self::pixel_to_ndc_x(tab_x + tab_width, width);
                
                // For top tab bar: active tab extends to touch terminal content
                // For bottom tab bar: active tab extends upward
                let (tab_top, tab_bottom) = if is_active {
                    match self.tab_bar_position {
                        TabBarPosition::Top => (
                            Self::pixel_to_ndc_y(tab_bar_y + 2.0, height),
                            Self::pixel_to_ndc_y(tab_bar_y + tab_bar_height, height),
                        ),
                        TabBarPosition::Bottom => (
                            Self::pixel_to_ndc_y(tab_bar_y, height),
                            Self::pixel_to_ndc_y(tab_bar_y + tab_bar_height - 2.0, height),
                        ),
                        TabBarPosition::Hidden => unreachable!(),
                    }
                } else {
                    (
                        Self::pixel_to_ndc_y(tab_bar_y + 4.0, height),
                        Self::pixel_to_ndc_y(tab_bar_y + tab_bar_height - 2.0, height),
                    )
                };

                let base_idx = bg_vertices.len() as u32;
                bg_vertices.push(GlyphVertex {
                    position: [tab_left, tab_top],
                    uv: [0.0, 0.0],
                    color: tab_bg,
                    bg_color: tab_bg,
                });
                bg_vertices.push(GlyphVertex {
                    position: [tab_right, tab_top],
                    uv: [0.0, 0.0],
                    color: tab_bg,
                    bg_color: tab_bg,
                });
                bg_vertices.push(GlyphVertex {
                    position: [tab_right, tab_bottom],
                    uv: [0.0, 0.0],
                    color: tab_bg,
                    bg_color: tab_bg,
                });
                bg_vertices.push(GlyphVertex {
                    position: [tab_left, tab_bottom],
                    uv: [0.0, 0.0],
                    color: tab_bg,
                    bg_color: tab_bg,
                });
                bg_indices.extend_from_slice(&[
                    base_idx, base_idx + 1, base_idx + 2,
                    base_idx, base_idx + 2, base_idx + 3,
                ]);

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

                        let base_idx = glyph_vertices.len() as u32;
                        glyph_vertices.push(GlyphVertex {
                            position: [left, top],
                            uv: [glyph.uv[0], glyph.uv[1]],
                            color: tab_fg,
                            bg_color: [0.0, 0.0, 0.0, 0.0],
                        });
                        glyph_vertices.push(GlyphVertex {
                            position: [right, top],
                            uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1]],
                            color: tab_fg,
                            bg_color: [0.0, 0.0, 0.0, 0.0],
                        });
                        glyph_vertices.push(GlyphVertex {
                            position: [right, bottom],
                            uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1] + glyph.uv[3]],
                            color: tab_fg,
                            bg_color: [0.0, 0.0, 0.0, 0.0],
                        });
                        glyph_vertices.push(GlyphVertex {
                            position: [left, bottom],
                            uv: [glyph.uv[0], glyph.uv[1] + glyph.uv[3]],
                            color: tab_fg,
                            bg_color: [0.0, 0.0, 0.0, 0.0],
                        });
                        glyph_indices.extend_from_slice(&[
                            base_idx, base_idx + 1, base_idx + 2,
                            base_idx, base_idx + 2, base_idx + 3,
                        ]);
                    }
                }

                tab_x += tab_width + tab_padding;
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // RENDER TERMINAL CONTENT (all panes, offset by tab bar)
        // ═══════════════════════════════════════════════════════════════════
        const LIGATURE_PATTERNS: &[&str] = &[
            "===", "!==", ">>>", "<<<", "||=", "&&=", "??=", "...", "-->", "<--", "<->",
            "=>", "->", "<-", ">=", "<=", "==", "!=", "::", "&&", "||", "??", "..", "++",
            "--", "<<", ">>", "|>", "<|", "/*", "*/", "//", "##", ":=", "~=", "<>",
        ];

        // Separator line color (slightly brighter than background)
        let separator_color = {
            let [r, g, b] = self.palette.default_bg;
            let factor = 1.5_f32;
            [
                Self::srgb_to_linear(((r as f32 / 255.0) * factor).min(1.0)),
                Self::srgb_to_linear(((g as f32 / 255.0) * factor).min(1.0)),
                Self::srgb_to_linear(((b as f32 / 255.0) * factor).min(1.0)),
                1.0,
            ]
        };

        // Render each pane
        for (pane, pane_info) in panes.iter() {
            let is_active_pane = pane.pane_id == active_pane_id;
            
            // Calculate pane position in pixels
            let pane_x_offset = pane_info.x as f32 * self.cell_width;
            let pane_y_offset = terminal_y_offset + pane_info.y as f32 * self.cell_height;

            for (row_idx, row) in pane.cells.iter().enumerate() {
                // Skip rows outside the pane bounds
                if row_idx >= pane_info.rows {
                    break;
                }
                
                let mut col_idx = 0;
                while col_idx < row.len() && col_idx < pane_info.cols {
                    let cell = &row[col_idx];
                    let cell_x = pane_x_offset + col_idx as f32 * self.cell_width;
                    let cell_y = pane_y_offset + row_idx as f32 * self.cell_height;

                    let fg_color = self.cell_color_to_rgba(&cell.fg_color, true);
                    let bg_color = self.cell_color_to_rgba(&cell.bg_color, false);

                    // Check for ligatures
                    let mut ligature_len = 0;
                    let mut ligature_glyph: Option<GlyphInfo> = None;

                    for pattern in LIGATURE_PATTERNS {
                        let pat_len = pattern.len();
                        if col_idx + pat_len <= row.len() && col_idx + pat_len <= pane_info.cols {
                            let candidate: String = row[col_idx..col_idx + pat_len]
                                .iter()
                                .map(|c| c.character)
                                .collect();
                            
                            if candidate == *pattern {
                                let shaped = self.shape_text(&candidate);
                                if shaped.glyphs.len() == 1 {
                                    let glyph_id = shaped.glyphs[0].0;
                                    ligature_glyph = Some(self.get_glyph_by_id(glyph_id));
                                    ligature_len = pat_len;
                                    break;
                                }
                            }
                        }
                    }

                    if let Some(glyph) = ligature_glyph {
                        let span_width = ligature_len as f32 * self.cell_width;
                        
                        for i in 0..ligature_len {
                            let bg_cell_x = pane_x_offset + (col_idx + i) as f32 * self.cell_width;
                            let cell_left = Self::pixel_to_ndc_x(bg_cell_x, width);
                            let cell_right = Self::pixel_to_ndc_x(bg_cell_x + self.cell_width, width);
                            let cell_top = Self::pixel_to_ndc_y(cell_y, height);
                            let cell_bottom = Self::pixel_to_ndc_y(cell_y + self.cell_height, height);

                            let base_idx = bg_vertices.len() as u32;
                            bg_vertices.push(GlyphVertex {
                                position: [cell_left, cell_top],
                                uv: [0.0, 0.0],
                                color: fg_color,
                                bg_color,
                            });
                            bg_vertices.push(GlyphVertex {
                                position: [cell_right, cell_top],
                                uv: [0.0, 0.0],
                                color: fg_color,
                                bg_color,
                            });
                            bg_vertices.push(GlyphVertex {
                                position: [cell_right, cell_bottom],
                                uv: [0.0, 0.0],
                                color: fg_color,
                                bg_color,
                            });
                            bg_vertices.push(GlyphVertex {
                                position: [cell_left, cell_bottom],
                                uv: [0.0, 0.0],
                                color: fg_color,
                                bg_color,
                            });
                            bg_indices.extend_from_slice(&[
                                base_idx, base_idx + 1, base_idx + 2,
                                base_idx, base_idx + 2, base_idx + 3,
                            ]);
                        }

                        if glyph.size[0] > 0.0 && glyph.size[1] > 0.0 {
                            let baseline_y = (cell_y + self.cell_height * 0.8).round();
                            let glyph_x = (cell_x + (span_width - glyph.size[0]) / 2.0 + glyph.offset[0]).round();
                            let glyph_y = (baseline_y - glyph.offset[1] - glyph.size[1]).round();

                            let left = Self::pixel_to_ndc_x(glyph_x, width);
                            let right = Self::pixel_to_ndc_x(glyph_x + glyph.size[0], width);
                            let top = Self::pixel_to_ndc_y(glyph_y, height);
                            let bottom = Self::pixel_to_ndc_y(glyph_y + glyph.size[1], height);

                            let base_idx = glyph_vertices.len() as u32;
                            glyph_vertices.push(GlyphVertex {
                                position: [left, top],
                                uv: [glyph.uv[0], glyph.uv[1]],
                                color: fg_color,
                                bg_color: [0.0, 0.0, 0.0, 0.0],
                            });
                            glyph_vertices.push(GlyphVertex {
                                position: [right, top],
                                uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1]],
                                color: fg_color,
                                bg_color: [0.0, 0.0, 0.0, 0.0],
                            });
                            glyph_vertices.push(GlyphVertex {
                                position: [right, bottom],
                                uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1] + glyph.uv[3]],
                                color: fg_color,
                                bg_color: [0.0, 0.0, 0.0, 0.0],
                            });
                            glyph_vertices.push(GlyphVertex {
                                position: [left, bottom],
                                uv: [glyph.uv[0], glyph.uv[1] + glyph.uv[3]],
                                color: fg_color,
                                bg_color: [0.0, 0.0, 0.0, 0.0],
                            });
                            glyph_indices.extend_from_slice(&[
                                base_idx, base_idx + 1, base_idx + 2,
                                base_idx, base_idx + 2, base_idx + 3,
                            ]);
                        }

                        col_idx += ligature_len;
                    } else {
                        // Single character rendering
                        let cell_left = Self::pixel_to_ndc_x(cell_x, width);
                        let cell_right = Self::pixel_to_ndc_x(cell_x + self.cell_width, width);
                        let cell_top = Self::pixel_to_ndc_y(cell_y, height);
                        let cell_bottom = Self::pixel_to_ndc_y(cell_y + self.cell_height, height);

                        let base_idx = bg_vertices.len() as u32;
                        bg_vertices.push(GlyphVertex {
                            position: [cell_left, cell_top],
                            uv: [0.0, 0.0],
                            color: fg_color,
                            bg_color,
                        });
                        bg_vertices.push(GlyphVertex {
                            position: [cell_right, cell_top],
                            uv: [0.0, 0.0],
                            color: fg_color,
                            bg_color,
                        });
                        bg_vertices.push(GlyphVertex {
                            position: [cell_right, cell_bottom],
                            uv: [0.0, 0.0],
                            color: fg_color,
                            bg_color,
                        });
                        bg_vertices.push(GlyphVertex {
                            position: [cell_left, cell_bottom],
                            uv: [0.0, 0.0],
                            color: fg_color,
                            bg_color,
                        });
                        bg_indices.extend_from_slice(&[
                            base_idx, base_idx + 1, base_idx + 2,
                            base_idx, base_idx + 2, base_idx + 3,
                        ]);

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

                                let base_idx = glyph_vertices.len() as u32;
                                glyph_vertices.push(GlyphVertex {
                                    position: [left, top],
                                    uv: [glyph.uv[0], glyph.uv[1]],
                                    color: fg_color,
                                    bg_color: [0.0, 0.0, 0.0, 0.0],
                                });
                                glyph_vertices.push(GlyphVertex {
                                    position: [right, top],
                                    uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1]],
                                    color: fg_color,
                                    bg_color: [0.0, 0.0, 0.0, 0.0],
                                });
                                glyph_vertices.push(GlyphVertex {
                                    position: [right, bottom],
                                    uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1] + glyph.uv[3]],
                                    color: fg_color,
                                    bg_color: [0.0, 0.0, 0.0, 0.0],
                                });
                                glyph_vertices.push(GlyphVertex {
                                    position: [left, bottom],
                                    uv: [glyph.uv[0], glyph.uv[1] + glyph.uv[3]],
                                    color: fg_color,
                                    bg_color: [0.0, 0.0, 0.0, 0.0],
                                });
                                glyph_indices.extend_from_slice(&[
                                    base_idx, base_idx + 1, base_idx + 2,
                                    base_idx, base_idx + 2, base_idx + 3,
                                ]);
                            }
                        }

                        col_idx += 1;
                    }
                }
            }

            // Draw cursor only in the active pane
            if is_active_pane && pane.cursor.visible {
                let cursor_x = pane_x_offset + pane.cursor.col as f32 * self.cell_width;
                let cursor_y = pane_y_offset + pane.cursor.row as f32 * self.cell_height;

                // Get the cell under the cursor to determine colors
                let cursor_cell = pane.cells
                    .get(pane.cursor.row)
                    .and_then(|row| row.get(pane.cursor.col));

                // Get fg and bg colors from the cell under cursor
                let (cell_fg, cell_bg, cell_char) = if let Some(cell) = cursor_cell {
                    let fg = self.cell_color_to_rgba(&cell.fg_color, true);
                    let bg = self.cell_color_to_rgba(&cell.bg_color, false);
                    (fg, bg, cell.character)
                } else {
                    // Default colors if cell doesn't exist
                    let fg = self.cell_color_to_rgba(&CellColor::Default, true);
                    let bg = [0.0, 0.0, 0.0, 0.0];
                    (fg, bg, ' ')
                };

                let has_character = cell_char != ' ' && cell_char != '\0';

                // Cursor color: invert the background, or use fg if there's a character
                let cursor_bg_color = if has_character {
                    // Character present: cursor takes fg color as background
                    [cell_fg[0], cell_fg[1], cell_fg[2], 1.0]
                } else {
                    // Empty cell: invert the background color
                    // If bg is transparent/default, invert to white; otherwise invert RGB
                    if cell_bg[3] < 0.01 {
                        // Transparent background -> white cursor
                        let white = Self::srgb_to_linear(0.9);
                        [white, white, white, 1.0]
                    } else {
                        // Invert the background color
                        [1.0 - cell_bg[0], 1.0 - cell_bg[1], 1.0 - cell_bg[2], 1.0]
                    }
                };

                // Determine cursor bounds based on style
                let (left, right, top, bottom) = match pane.cursor.style {
                    CursorStyle::Block => (
                        cursor_x,
                        cursor_x + self.cell_width,
                        cursor_y,
                        cursor_y + self.cell_height,
                    ),
                    CursorStyle::Underline => {
                        let underline_height = 2.0_f32.max(self.cell_height * 0.1);
                        (
                            cursor_x,
                            cursor_x + self.cell_width,
                            cursor_y + self.cell_height - underline_height,
                            cursor_y + self.cell_height,
                        )
                    }
                    CursorStyle::Bar => {
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

                let base_idx = glyph_vertices.len() as u32;

                glyph_vertices.push(GlyphVertex {
                    position: [cursor_left, cursor_top],
                    uv: [0.0, 0.0],
                    color: cursor_bg_color,
                    bg_color: cursor_bg_color,
                });
                glyph_vertices.push(GlyphVertex {
                    position: [cursor_right, cursor_top],
                    uv: [0.0, 0.0],
                    color: cursor_bg_color,
                    bg_color: cursor_bg_color,
                });
                glyph_vertices.push(GlyphVertex {
                    position: [cursor_right, cursor_bottom],
                    uv: [0.0, 0.0],
                    color: cursor_bg_color,
                    bg_color: cursor_bg_color,
                });
                glyph_vertices.push(GlyphVertex {
                    position: [cursor_left, cursor_bottom],
                    uv: [0.0, 0.0],
                    color: cursor_bg_color,
                    bg_color: cursor_bg_color,
                });

                glyph_indices.extend_from_slice(&[
                    base_idx, base_idx + 1, base_idx + 2,
                    base_idx, base_idx + 2, base_idx + 3,
                ]);

                // If block cursor and there's a character, re-render it with inverted color
                if matches!(pane.cursor.style, CursorStyle::Block) && has_character {
                    // Character color: use bg color (inverted from normal)
                    let char_color = if cell_bg[3] < 0.01 {
                        // If bg was transparent, use black for the character
                        [0.0, 0.0, 0.0, 1.0]
                    } else {
                        [cell_bg[0], cell_bg[1], cell_bg[2], 1.0]
                    };

                    let glyph = self.rasterize_char(cell_char);
                    if glyph.size[0] > 0.0 && glyph.size[1] > 0.0 {
                        let cell_x = cursor_x;
                        let cell_y = cursor_y;
                        let (glyph_x, glyph_y) = if Self::is_box_drawing(cell_char) {
                            (cell_x, cell_y)
                        } else {
                            let baseline_y = (cell_y + self.cell_height * 0.8).round();
                            let gx = (cell_x + glyph.offset[0]).round();
                            let gy = (baseline_y - glyph.offset[1] - glyph.size[1]).round();
                            (gx, gy)
                        };

                        let g_left = Self::pixel_to_ndc_x(glyph_x, width);
                        let g_right = Self::pixel_to_ndc_x(glyph_x + glyph.size[0], width);
                        let g_top = Self::pixel_to_ndc_y(glyph_y, height);
                        let g_bottom = Self::pixel_to_ndc_y(glyph_y + glyph.size[1], height);

                        let base_idx = glyph_vertices.len() as u32;
                        glyph_vertices.push(GlyphVertex {
                            position: [g_left, g_top],
                            uv: [glyph.uv[0], glyph.uv[1]],
                            color: char_color,
                            bg_color: [0.0, 0.0, 0.0, 0.0],
                        });
                        glyph_vertices.push(GlyphVertex {
                            position: [g_right, g_top],
                            uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1]],
                            color: char_color,
                            bg_color: [0.0, 0.0, 0.0, 0.0],
                        });
                        glyph_vertices.push(GlyphVertex {
                            position: [g_right, g_bottom],
                            uv: [glyph.uv[0] + glyph.uv[2], glyph.uv[1] + glyph.uv[3]],
                            color: char_color,
                            bg_color: [0.0, 0.0, 0.0, 0.0],
                        });
                        glyph_vertices.push(GlyphVertex {
                            position: [g_left, g_bottom],
                            uv: [glyph.uv[0], glyph.uv[1] + glyph.uv[3]],
                            color: char_color,
                            bg_color: [0.0, 0.0, 0.0, 0.0],
                        });
                        glyph_indices.extend_from_slice(&[
                            base_idx, base_idx + 1, base_idx + 2,
                            base_idx, base_idx + 2, base_idx + 3,
                        ]);
                    }
                }
            }

        }

        // ═══════════════════════════════════════════════════════════════════
        // DRAW PANE SEPARATORS (only between adjacent panes, sized to the pane bounds)
        // ═══════════════════════════════════════════════════════════════════
        if panes.len() > 1 {
            let separator_thickness = 1.0_f32;
            
            for (_, pane_info) in panes.iter() {
                let pane_left_x = pane_info.x as f32 * self.cell_width;
                let pane_right_x = pane_left_x + pane_info.cols as f32 * self.cell_width;
                let pane_top_y = terminal_y_offset + pane_info.y as f32 * self.cell_height;
                let pane_bottom_y = pane_top_y + pane_info.rows as f32 * self.cell_height;
                
                // Check if there's a pane directly to the right (draw vertical separator on right edge)
                let has_pane_right = panes.iter().any(|(_, other)| {
                    let other_left_x = other.x as f32 * self.cell_width;
                    let other_top_y = terminal_y_offset + other.y as f32 * self.cell_height;
                    let other_bottom_y = other_top_y + other.rows as f32 * self.cell_height;
                    // Other pane starts at our right edge and overlaps vertically
                    (other_left_x - pane_right_x).abs() < 2.0
                        && other_top_y < pane_bottom_y
                        && other_bottom_y > pane_top_y
                });
                
                if has_pane_right {
                    // Draw vertical separator at right edge, spanning this pane's height
                    // Extend to window edges if this pane is at top/bottom
                    let sep_top = if pane_info.y == 0 { terminal_y_offset } else { pane_top_y };
                    let terminal_bottom = height - if matches!(self.tab_bar_position, TabBarPosition::Bottom) { tab_bar_height } else { 0.0 };
                    let max_pane_bottom: f32 = panes.iter()
                        .map(|(_, p)| terminal_y_offset + p.y as f32 * self.cell_height + p.rows as f32 * self.cell_height)
                        .fold(0.0_f32, |a, b| a.max(b));
                    let sep_bottom = if (pane_bottom_y - max_pane_bottom).abs() < 2.0 { terminal_bottom } else { pane_bottom_y };
                    
                    let sep_left_ndc = Self::pixel_to_ndc_x(pane_right_x, width);
                    let sep_right_ndc = Self::pixel_to_ndc_x(pane_right_x + separator_thickness, width);
                    let sep_top_ndc = Self::pixel_to_ndc_y(sep_top, height);
                    let sep_bottom_ndc = Self::pixel_to_ndc_y(sep_bottom, height);

                    let base_idx = glyph_vertices.len() as u32;
                    glyph_vertices.push(GlyphVertex {
                        position: [sep_left_ndc, sep_top_ndc],
                        uv: [0.0, 0.0],
                        color: separator_color,
                        bg_color: separator_color,
                    });
                    glyph_vertices.push(GlyphVertex {
                        position: [sep_right_ndc, sep_top_ndc],
                        uv: [0.0, 0.0],
                        color: separator_color,
                        bg_color: separator_color,
                    });
                    glyph_vertices.push(GlyphVertex {
                        position: [sep_right_ndc, sep_bottom_ndc],
                        uv: [0.0, 0.0],
                        color: separator_color,
                        bg_color: separator_color,
                    });
                    glyph_vertices.push(GlyphVertex {
                        position: [sep_left_ndc, sep_bottom_ndc],
                        uv: [0.0, 0.0],
                        color: separator_color,
                        bg_color: separator_color,
                    });
                    glyph_indices.extend_from_slice(&[
                        base_idx, base_idx + 1, base_idx + 2,
                        base_idx, base_idx + 2, base_idx + 3,
                    ]);
                }
                
                // Check if there's a pane directly below (draw horizontal separator on bottom edge)
                let has_pane_below = panes.iter().any(|(_, other)| {
                    let other_left_x = other.x as f32 * self.cell_width;
                    let other_right_x = other_left_x + other.cols as f32 * self.cell_width;
                    let other_top_y = terminal_y_offset + other.y as f32 * self.cell_height;
                    // Other pane starts at our bottom edge and overlaps horizontally
                    (other_top_y - pane_bottom_y).abs() < 2.0
                        && other_left_x < pane_right_x
                        && other_right_x > pane_left_x
                });
                
                if has_pane_below {
                    // Draw horizontal separator at bottom edge, spanning this pane's width
                    // Extend to window edges if this pane is at left/right
                    let sep_left = if pane_info.x == 0 { 0.0 } else { pane_left_x };
                    let max_pane_right: f32 = panes.iter()
                        .map(|(_, p)| p.x as f32 * self.cell_width + p.cols as f32 * self.cell_width)
                        .fold(0.0_f32, |a, b| a.max(b));
                    let sep_right = if (pane_right_x - max_pane_right).abs() < 2.0 { width } else { pane_right_x };
                    
                    let sep_left_ndc = Self::pixel_to_ndc_x(sep_left, width);
                    let sep_right_ndc = Self::pixel_to_ndc_x(sep_right, width);
                    let sep_top_ndc = Self::pixel_to_ndc_y(pane_bottom_y, height);
                    let sep_bottom_ndc = Self::pixel_to_ndc_y(pane_bottom_y + separator_thickness, height);

                    let base_idx = glyph_vertices.len() as u32;
                    glyph_vertices.push(GlyphVertex {
                        position: [sep_left_ndc, sep_top_ndc],
                        uv: [0.0, 0.0],
                        color: separator_color,
                        bg_color: separator_color,
                    });
                    glyph_vertices.push(GlyphVertex {
                        position: [sep_right_ndc, sep_top_ndc],
                        uv: [0.0, 0.0],
                        color: separator_color,
                        bg_color: separator_color,
                    });
                    glyph_vertices.push(GlyphVertex {
                        position: [sep_right_ndc, sep_bottom_ndc],
                        uv: [0.0, 0.0],
                        color: separator_color,
                        bg_color: separator_color,
                    });
                    glyph_vertices.push(GlyphVertex {
                        position: [sep_left_ndc, sep_bottom_ndc],
                        uv: [0.0, 0.0],
                        color: separator_color,
                        bg_color: separator_color,
                    });
                    glyph_indices.extend_from_slice(&[
                        base_idx, base_idx + 1, base_idx + 2,
                        base_idx, base_idx + 2, base_idx + 3,
                    ]);
                }
            }
        }

        // Combine vertices
        let mut vertices = bg_vertices;
        let mut indices = bg_indices;
        
        let glyph_vertex_offset = vertices.len() as u32;
        vertices.extend(glyph_vertices);
        indices.extend(glyph_indices.iter().map(|i| i + glyph_vertex_offset));

        // Resize buffers if needed
        if vertices.len() > self.vertex_capacity {
            self.vertex_capacity = vertices.len() * 2;
            self.vertex_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Glyph Vertex Buffer"),
                size: (self.vertex_capacity * std::mem::size_of::<GlyphVertex>()) as u64,
                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        if indices.len() > self.index_capacity {
            self.index_capacity = indices.len() * 2;
            self.index_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Glyph Index Buffer"),
                size: (self.index_capacity * std::mem::size_of::<u32>()) as u64,
                usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }

        // Upload data
        self.queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&vertices));
        self.queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(&indices));

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
            render_pass.draw_indexed(0..indices.len() as u32, 0, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}
