//! GPU-accelerated terminal rendering using wgpu with a glyph atlas.
//! Uses rustybuzz (HarfBuzz port) for text shaping to support font features.

use crate::config::TabBarPosition;
use crate::terminal::{Color, ColorPalette, CursorShape, GPUCell, Terminal};
use fontdue::Font as FontdueFont;
use rustybuzz::UnicodeBuffer;
use std::collections::HashMap;
use std::sync::Arc;
use wgpu::util::DeviceExt;

// ═══════════════════════════════════════════════════════════════════════════════
// KITTY-STYLE INSTANCED RENDERING STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

/// Color table for shader uniform (258 colors: 256 indexed + default fg/bg).
/// Each color is stored as [R, G, B, A] in linear color space.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct ColorTableUniform {
    colors: [[f32; 4]; 258],
}

// Manual bytemuck implementations since Pod/Zeroable aren't derived for [T; 258]
unsafe impl bytemuck::Zeroable for ColorTableUniform {}
unsafe impl bytemuck::Pod for ColorTableUniform {}

impl Default for ColorTableUniform {
    fn default() -> Self {
        Self {
            colors: [[0.0; 4]; 258],
        }
    }
}

/// Grid parameters for instanced rendering.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct GridParamsUniform {
    /// Number of columns
    cols: u32,
    /// Number of rows
    rows: u32,
    /// Cell width in pixels
    cell_width: f32,
    /// Cell height in pixels
    cell_height: f32,
    /// Screen width in pixels
    screen_width: f32,
    /// Screen height in pixels
    screen_height: f32,
    /// Y offset for tab bar
    y_offset: f32,
    /// Cursor column (-1 if hidden)
    cursor_col: i32,
    /// Cursor row (-1 if hidden)
    cursor_row: i32,
    /// Cursor style: 0=block, 1=underline, 2=bar
    cursor_style: u32,
    /// Padding for 16-byte alignment
    _padding: [u32; 2],
}

/// Sprite info for glyph atlas lookup.
/// Matches the SpriteInfo struct in the shader.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SpriteInfo {
    /// UV coordinates in atlas (x, y, width, height) - normalized 0-1
    pub uv: [f32; 4],
    /// Offset from cell origin (x, y) in pixels
    pub offset: [f32; 2],
    /// Size in pixels (width, height)
    pub size: [f32; 2],
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
    #[allow(dead_code)] // Kept alive for rustybuzz::Face which borrows it
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
    
    // Reusable vertex/index buffers to avoid per-frame allocations
    bg_vertices: Vec<GlyphVertex>,
    bg_indices: Vec<u32>,
    glyph_vertices: Vec<GlyphVertex>,
    glyph_indices: Vec<u32>,
    
    /// Current selection range for rendering (start_col, start_row, end_col, end_row).
    /// If set, cells within this range will be rendered with inverted colors.
    selection: Option<(usize, usize, usize, usize)>,
    
    // ═══════════════════════════════════════════════════════════════════════════════
    // KITTY-STYLE INSTANCED RENDERING INFRASTRUCTURE
    // ═══════════════════════════════════════════════════════════════════════════════
    
    /// Instanced rendering pipeline for backgrounds
    cell_bg_pipeline: Option<wgpu::RenderPipeline>,
    /// Instanced rendering pipeline for glyphs
    cell_glyph_pipeline: Option<wgpu::RenderPipeline>,
    /// Bind group for instanced rendering (color table, grid params, cells, sprites)
    cell_bind_group: Option<wgpu::BindGroup>,
    /// Bind group layout for instanced rendering
    cell_bind_group_layout: Option<wgpu::BindGroupLayout>,
    
    /// Color table uniform buffer (258 colors)
    color_table_buffer: Option<wgpu::Buffer>,
    /// Grid parameters uniform buffer
    grid_params_buffer: Option<wgpu::Buffer>,
    /// GPU cell storage buffer
    cell_buffer: Option<wgpu::Buffer>,
    /// Cell buffer capacity (number of cells)
    cell_buffer_capacity: usize,
    /// Sprite info storage buffer
    sprite_buffer: Option<wgpu::Buffer>,
    /// Sprite buffer capacity
    sprite_buffer_capacity: usize,
    
    /// Index buffer for instanced quads (shared between bg and glyph)
    quad_index_buffer: Option<wgpu::Buffer>,
    
    /// CPU-side sprite info array (maps sprite_idx -> SpriteInfo)
    sprite_info: Vec<SpriteInfo>,
    /// Map from character to sprite index for fast lookup
    char_to_sprite: HashMap<char, u32>,
    /// Next available sprite index
    next_sprite_idx: u32,
    
    /// Whether to use instanced rendering (can be disabled for debugging)
    use_instanced_rendering: bool,
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

        // ═══════════════════════════════════════════════════════════════════════════════
        // KITTY-STYLE INSTANCED RENDERING INITIALIZATION
        // ═══════════════════════════════════════════════════════════════════════════════
        
        // Initial capacity for cell buffer (e.g., 80x24 terminal = 1920 cells)
        let initial_cell_capacity: usize = 80 * 40;
        let initial_sprite_capacity: usize = 512;
        
        // Create color table uniform buffer (258 colors * 4 floats * 4 bytes = 4128 bytes)
        let color_table_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Color Table Buffer"),
            size: std::mem::size_of::<ColorTableUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create grid params uniform buffer
        let grid_params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Grid Params Buffer"),
            size: std::mem::size_of::<GridParamsUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create cell storage buffer (GPUCell is 20 bytes)
        let cell_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Storage Buffer"),
            size: (initial_cell_capacity * std::mem::size_of::<GPUCell>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create sprite info storage buffer (SpriteInfo is 32 bytes)
        let sprite_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sprite Storage Buffer"),
            size: (initial_sprite_capacity * std::mem::size_of::<SpriteInfo>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        
        // Create index buffer for instanced quads (6 indices per quad: 0,1,2, 0,2,3)
        let quad_indices: [u16; 6] = [0, 1, 2, 0, 2, 3];
        let quad_index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Quad Index Buffer"),
            contents: bytemuck::cast_slice(&quad_indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        
        // Create bind group layout for instanced rendering (group 1)
        let cell_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Cell Bind Group Layout"),
            entries: &[
                // binding 0: ColorTable uniform
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
                // binding 1: GridParams uniform
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
                // binding 2: cells storage buffer (read-only)
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
                // binding 3: sprites storage buffer (read-only)
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
        let cell_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cell Bind Group"),
            layout: &cell_bind_group_layout,
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
        
        // Create pipeline layout for instanced rendering (uses both group 0 and group 1)
        let cell_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Cell Pipeline Layout"),
            bind_group_layouts: &[&glyph_bind_group_layout, &cell_bind_group_layout],
            push_constant_ranges: &[],
        });
        
        // Create background pipeline
        let cell_bg_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Cell Background Pipeline"),
            layout: Some(&cell_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_cell_bg"),
                buffers: &[], // No vertex buffers - using instancing
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
        
        // Create glyph pipeline
        let cell_glyph_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Cell Glyph Pipeline"),
            layout: Some(&cell_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_cell_glyph"),
                buffers: &[], // No vertex buffers - using instancing
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
        
        // Initialize sprite info array with entry 0 = no glyph/empty
        let sprite_info = vec![SpriteInfo::default()];
        
        // ═══════════════════════════════════════════════════════════════════════════════
        // END KITTY-STYLE INSTANCED RENDERING INITIALIZATION
        // ═══════════════════════════════════════════════════════════════════════════════

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
            // Pre-allocate reusable buffers for rendering
            bg_vertices: Vec::with_capacity(4096),
            bg_indices: Vec::with_capacity(6144),
            glyph_vertices: Vec::with_capacity(4096),
            glyph_indices: Vec::with_capacity(6144),
            selection: None,
            
            // Kitty-style instanced rendering infrastructure
            cell_bg_pipeline: Some(cell_bg_pipeline),
            cell_glyph_pipeline: Some(cell_glyph_pipeline),
            cell_bind_group: Some(cell_bind_group),
            cell_bind_group_layout: Some(cell_bind_group_layout),
            color_table_buffer: Some(color_table_buffer),
            grid_params_buffer: Some(grid_params_buffer),
            cell_buffer: Some(cell_buffer),
            cell_buffer_capacity: initial_cell_capacity,
            sprite_buffer: Some(sprite_buffer),
            sprite_buffer_capacity: initial_sprite_capacity,
            quad_index_buffer: Some(quad_index_buffer),
            sprite_info,
            char_to_sprite: HashMap::new(),
            next_sprite_idx: 1, // 0 is reserved for empty/no glyph
            use_instanced_rendering: true, // Use Kitty-style instanced rendering
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
    
    /// Checks if a cell at (col, row) is within the current selection.
    fn is_cell_selected(&self, col: usize, row: usize) -> bool {
        let Some((start_col, start_row, end_col, end_row)) = self.selection else {
            return false;
        };
        
        // Check if the row is within the selection range
        if row < start_row || row > end_row {
            return false;
        }
        
        // For single-row selection
        if start_row == end_row {
            return col >= start_col && col <= end_col;
        }
        
        // For multi-row selection
        if row == start_row {
            // First row: from start_col to end of line
            return col >= start_col;
        } else if row == end_row {
            // Last row: from start of line to end_col
            return col <= end_col;
        } else {
            // Middle rows: entire row is selected
            return true;
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

        // Reuse pre-allocated buffers - clear instead of reallocating
        // This ensures wide glyphs (like Nerd Font icons) can extend beyond their cell
        // without being covered by adjacent cell backgrounds
        self.bg_vertices.clear();
        self.bg_indices.clear();
        self.glyph_vertices.clear();
        self.glyph_indices.clear();

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

        // Combine: backgrounds first, then glyphs (with adjusted indices)
        // We need to calculate total counts and adjust glyph indices
        let bg_vertex_count = self.bg_vertices.len();
        let total_vertex_count = bg_vertex_count + self.glyph_vertices.len();
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

        // Upload background vertices first, then glyph vertices
        self.queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&self.bg_vertices));
        self.queue.write_buffer(
            &self.vertex_buffer,
            (bg_vertex_count * std::mem::size_of::<GlyphVertex>()) as u64,
            bytemuck::cast_slice(&self.glyph_vertices),
        );
        
        // Upload background indices first
        self.queue.write_buffer(&self.index_buffer, 0, bytemuck::cast_slice(&self.bg_indices));
        
        // Upload glyph indices with offset adjustment (need to adjust indices by bg_vertex_count)
        // Create adjusted indices on the stack if small enough, otherwise use temporary allocation
        let glyph_vertex_offset = bg_vertex_count as u32;
        let bg_index_bytes = self.bg_indices.len() * std::mem::size_of::<u32>();
        
        // Write adjusted glyph indices
        if !self.glyph_indices.is_empty() {
            // For large batches, we need a temporary buffer - this is unavoidable
            // but happens only once per frame instead of incrementally
            let adjusted_indices: Vec<u32> = self.glyph_indices.iter()
                .map(|i| i + glyph_vertex_offset)
                .collect();
            self.queue.write_buffer(
                &self.index_buffer,
                bg_index_bytes as u64,
                bytemuck::cast_slice(&adjusted_indices),
            );
        }

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
            render_pass.draw_indexed(0..total_index_count as u32, 0, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        terminal.dirty = false;

        Ok(())
    }
    /// Renders terminal content directly from a Terminal reference.
    /// This is the new preferred method that avoids cross-process synchronization.
    /// 
    /// Arguments:
    /// - `terminal`: Reference to the terminal state (same process)
    /// - `num_tabs`: Number of tabs for the tab bar (0 to hide)
    /// - `active_tab`: Index of the active tab
    pub fn render_from_terminal(
        &mut self,
        terminal: &Terminal,
        num_tabs: usize,
        active_tab: usize,
    ) -> Result<(), wgpu::SurfaceError> {
        // Sync palette from terminal (OSC sequences update terminal.palette)
        self.palette = terminal.palette.clone();
        
        let output = self.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let cols = terminal.cols;
        let rows = terminal.rows;
        
        // Reuse pre-allocated buffers
        self.bg_vertices.clear();
        self.bg_indices.clear();
        self.glyph_vertices.clear();
        self.glyph_indices.clear();

        let width = self.width as f32;
        let height = self.height as f32;
        let tab_bar_height = self.tab_bar_height();
        let terminal_y_offset = self.terminal_y_offset();

        // ═══════════════════════════════════════════════════════════════════
        // RENDER TAB BAR
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
            let bar_left = Self::pixel_to_ndc_x(0.0, width);
            let bar_right = Self::pixel_to_ndc_x(width, width);
            let bar_top = Self::pixel_to_ndc_y(tab_bar_y, height);
            let bar_bottom = Self::pixel_to_ndc_y(tab_bar_y + tab_bar_height, height);

            let base_idx = self.bg_vertices.len() as u32;
            self.bg_vertices.push(GlyphVertex {
                position: [bar_left, bar_top],
                uv: [0.0, 0.0],
                color: tab_bar_bg,
                bg_color: tab_bar_bg,
            });
            self.bg_vertices.push(GlyphVertex {
                position: [bar_right, bar_top],
                uv: [0.0, 0.0],
                color: tab_bar_bg,
                bg_color: tab_bar_bg,
            });
            self.bg_vertices.push(GlyphVertex {
                position: [bar_right, bar_bottom],
                uv: [0.0, 0.0],
                color: tab_bar_bg,
                bg_color: tab_bar_bg,
            });
            self.bg_vertices.push(GlyphVertex {
                position: [bar_left, bar_bottom],
                uv: [0.0, 0.0],
                color: tab_bar_bg,
                bg_color: tab_bar_bg,
            });
            self.bg_indices.extend_from_slice(&[
                base_idx, base_idx + 1, base_idx + 2,
                base_idx, base_idx + 2, base_idx + 3,
            ]);

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

                let tab_top = Self::pixel_to_ndc_y(tab_bar_y + 2.0, height);
                let tab_bottom = Self::pixel_to_ndc_y(tab_bar_y + tab_bar_height - 2.0, height);
                let tab_left = Self::pixel_to_ndc_x(tab_x, width);
                let tab_right = Self::pixel_to_ndc_x(tab_x + tab_width, width);

                let base_idx = self.bg_vertices.len() as u32;
                self.bg_vertices.push(GlyphVertex {
                    position: [tab_left, tab_top],
                    uv: [0.0, 0.0],
                    color: tab_bg,
                    bg_color: tab_bg,
                });
                self.bg_vertices.push(GlyphVertex {
                    position: [tab_right, tab_top],
                    uv: [0.0, 0.0],
                    color: tab_bg,
                    bg_color: tab_bg,
                });
                self.bg_vertices.push(GlyphVertex {
                    position: [tab_right, tab_bottom],
                    uv: [0.0, 0.0],
                    color: tab_bg,
                    bg_color: tab_bg,
                });
                self.bg_vertices.push(GlyphVertex {
                    position: [tab_left, tab_bottom],
                    uv: [0.0, 0.0],
                    color: tab_bg,
                    bg_color: tab_bg,
                });
                self.bg_indices.extend_from_slice(&[
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
        // RENDER TERMINAL CONTENT FROM TERMINAL STATE
        // ═══════════════════════════════════════════════════════════════════
        
        // Get visible rows (accounts for scroll offset)
        let visible_rows = terminal.visible_rows();
        
        // Cache palette values to avoid borrow conflicts with rasterize_char
        let palette_default_fg = self.palette.default_fg;
        let palette_colors = self.palette.colors;
        
        // Helper to convert Color to linear RGBA (uses cached palette)
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
                        // Default background: transparent
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

        // Render each row
        for (row_idx, row) in visible_rows.iter().enumerate() {
            if row_idx >= rows {
                break;
            }
            
            // Find the last non-empty cell in this row for selection clipping
            let last_content_col = row.iter()
                .enumerate()
                .rev()
                .find(|(_, cell)| cell.character != ' ' && cell.character != '\0')
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            
            for (col_idx, cell) in row.iter().enumerate() {
                if col_idx >= cols {
                    break;
                }
                
                let cell_x = col_idx as f32 * self.cell_width;
                let cell_y = terminal_y_offset + row_idx as f32 * self.cell_height;

                let mut fg_color = color_to_rgba(&cell.fg_color, true);
                let mut bg_color = color_to_rgba(&cell.bg_color, false);
                
                // Handle selection
                if self.is_cell_selected(col_idx, row_idx) && col_idx <= last_content_col {
                    fg_color = [0.0, 0.0, 0.0, 1.0];  // Black foreground
                    bg_color = [1.0, 1.0, 1.0, 1.0];  // White background
                }

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
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // RENDER CURSOR
        // ═══════════════════════════════════════════════════════════════════
        // Only show cursor when viewing live terminal (not scrolled into history)
        if terminal.cursor_visible && terminal.scroll_offset == 0 
           && terminal.cursor_row < rows && terminal.cursor_col < cols {
            let cursor_col = terminal.cursor_col;
            let cursor_row = terminal.cursor_row;
            let cursor_x = cursor_col as f32 * self.cell_width;
            let cursor_y = terminal_y_offset + cursor_row as f32 * self.cell_height;

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

            // Convert cursor shape to style
            let cursor_style = match terminal.cursor_shape {
                CursorShape::BlinkingBlock | CursorShape::SteadyBlock => 0,
                CursorShape::BlinkingUnderline | CursorShape::SteadyUnderline => 1,
                CursorShape::BlinkingBar | CursorShape::SteadyBar => 2,
            };

            let (left, right, top, bottom) = match cursor_style {
                0 => ( // Block
                    cursor_x,
                    cursor_x + self.cell_width,
                    cursor_y,
                    cursor_y + self.cell_height,
                ),
                1 => { // Underline
                    let underline_height = 2.0_f32.max(self.cell_height * 0.1);
                    (
                        cursor_x,
                        cursor_x + self.cell_width,
                        cursor_y + self.cell_height - underline_height,
                        cursor_y + self.cell_height,
                    )
                }
                _ => { // Bar
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

            // If block cursor and there's a character, render it inverted
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

        // ═══════════════════════════════════════════════════════════════════
        // SUBMIT TO GPU
        // ═══════════════════════════════════════════════════════════════════
        let bg_vertex_count = self.bg_vertices.len();
        let total_vertex_count = bg_vertex_count + self.glyph_vertices.len();
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

        // Upload vertices
        self.queue.write_buffer(&self.vertex_buffer, 0, bytemuck::cast_slice(&self.bg_vertices));
        self.queue.write_buffer(
            &self.vertex_buffer,
            (bg_vertex_count * std::mem::size_of::<GlyphVertex>()) as u64,
            bytemuck::cast_slice(&self.glyph_vertices),
        );

        // Upload indices
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

    // ═══════════════════════════════════════════════════════════════════════════════
    // KITTY-STYLE INSTANCED RENDERING HELPER METHODS
    // ═══════════════════════════════════════════════════════════════════════════════

    /// Update the color table uniform buffer with the current palette.
    /// The color table contains 258 colors: 256 indexed colors + default fg (256) + default bg (257).
    fn update_color_table(&mut self, palette: &ColorPalette) {
        let Some(ref buffer) = self.color_table_buffer else {
            return;
        };

        let mut color_table = ColorTableUniform::default();

        // Fill 256 indexed colors
        for i in 0..256 {
            let [r, g, b] = palette.colors[i];
            color_table.colors[i] = [
                r as f32 / 255.0,
                g as f32 / 255.0,
                b as f32 / 255.0,
                1.0,
            ];
        }

        // Default foreground at index 256
        let [fg_r, fg_g, fg_b] = palette.default_fg;
        color_table.colors[256] = [
            fg_r as f32 / 255.0,
            fg_g as f32 / 255.0,
            fg_b as f32 / 255.0,
            1.0,
        ];

        // Default background at index 257
        let [bg_r, bg_g, bg_b] = palette.default_bg;
        color_table.colors[257] = [
            bg_r as f32 / 255.0,
            bg_g as f32 / 255.0,
            bg_b as f32 / 255.0,
            1.0,
        ];

        self.queue.write_buffer(buffer, 0, bytemuck::bytes_of(&color_table));
    }

    /// Get or create a sprite index for a character.
    /// Returns the sprite index, or 0 if the character has no visible glyph.
    fn get_or_create_sprite(&mut self, c: char) -> u32 {
        // Check cache first
        if let Some(&idx) = self.char_to_sprite.get(&c) {
            return idx;
        }

        // Space and control characters have no visible glyph
        if c == ' ' || c == '\0' || c.is_control() {
            self.char_to_sprite.insert(c, 0);
            return 0;
        }

        // Rasterize the character to get its glyph info
        let glyph_info = self.rasterize_char(c);

        // If the glyph has no visible pixels, return 0
        if glyph_info.size[0] <= 0.0 || glyph_info.size[1] <= 0.0 {
            self.char_to_sprite.insert(c, 0);
            return 0;
        }

        // Assign a new sprite index
        let sprite_idx = self.next_sprite_idx;
        self.next_sprite_idx += 1;

        // Ensure we have capacity in the sprite_info vector
        while self.sprite_info.len() <= sprite_idx as usize {
            self.sprite_info.push(SpriteInfo::default());
        }

        // Store sprite info
        self.sprite_info[sprite_idx as usize] = SpriteInfo {
            uv: glyph_info.uv,
            offset: glyph_info.offset,
            size: glyph_info.size,
        };

        // Cache the mapping
        self.char_to_sprite.insert(c, sprite_idx);

        sprite_idx
    }

    /// Ensure the cell buffer has enough capacity for the given number of cells.
    fn ensure_cell_buffer_capacity(&mut self, num_cells: usize) {
        if num_cells <= self.cell_buffer_capacity {
            return;
        }

        // Grow by 2x or to the required size, whichever is larger
        let new_capacity = (self.cell_buffer_capacity * 2).max(num_cells);

        let new_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Storage Buffer"),
            size: (new_capacity * std::mem::size_of::<GPUCell>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.cell_buffer = Some(new_buffer);
        self.cell_buffer_capacity = new_capacity;

        // Recreate bind group with new buffer
        self.recreate_cell_bind_group();
    }

    /// Ensure the sprite buffer has enough capacity.
    fn ensure_sprite_buffer_capacity(&mut self, num_sprites: usize) {
        if num_sprites <= self.sprite_buffer_capacity {
            return;
        }

        let new_capacity = (self.sprite_buffer_capacity * 2).max(num_sprites);

        let new_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Sprite Storage Buffer"),
            size: (new_capacity * std::mem::size_of::<SpriteInfo>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        self.sprite_buffer = Some(new_buffer);
        self.sprite_buffer_capacity = new_capacity;

        // Recreate bind group with new buffer
        self.recreate_cell_bind_group();
    }

    /// Recreate the cell bind group after buffer reallocation.
    fn recreate_cell_bind_group(&mut self) {
        let Some(ref layout) = self.cell_bind_group_layout else {
            return;
        };
        let Some(ref color_table_buffer) = self.color_table_buffer else {
            return;
        };
        let Some(ref grid_params_buffer) = self.grid_params_buffer else {
            return;
        };
        let Some(ref cell_buffer) = self.cell_buffer else {
            return;
        };
        let Some(ref sprite_buffer) = self.sprite_buffer else {
            return;
        };

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cell Bind Group"),
            layout,
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

        self.cell_bind_group = Some(bind_group);
    }

    /// Render using Kitty-style instanced rendering.
    /// This is the new high-performance rendering path.
    pub fn render_instanced(&mut self, terminal: &mut Terminal) -> Result<(), wgpu::SurfaceError> {
        // Early return if instanced rendering is not set up
        if !self.use_instanced_rendering {
            return self.render(terminal);
        }

        let output = self.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let (cols, rows) = self.terminal_size();
        let num_cells = cols * rows;

        // Ensure buffers are large enough
        self.ensure_cell_buffer_capacity(num_cells);
        self.ensure_sprite_buffer_capacity(self.next_sprite_idx as usize + 256);

        // Update color table from palette
        self.update_color_table(&terminal.palette);

        // Build GPU cells array
        let mut gpu_cells = Vec::with_capacity(num_cells);
        for row_idx in 0..rows.min(terminal.grid.len()) {
            let row = &terminal.grid[row_idx];
            for col_idx in 0..cols {
                if col_idx < row.len() {
                    let cell = &row[col_idx];
                    let sprite_idx = self.get_or_create_sprite(cell.character);
                    gpu_cells.push(GPUCell::from_cell(cell, sprite_idx));
                } else {
                    gpu_cells.push(GPUCell::empty());
                }
            }
        }
        // Fill remaining rows with empty cells
        while gpu_cells.len() < num_cells {
            gpu_cells.push(GPUCell::empty());
        }

        // Upload cell data
        if let Some(ref buffer) = self.cell_buffer {
            self.queue.write_buffer(buffer, 0, bytemuck::cast_slice(&gpu_cells));
        }

        // Upload sprite info
        if let Some(ref buffer) = self.sprite_buffer {
            self.queue.write_buffer(buffer, 0, bytemuck::cast_slice(&self.sprite_info));
        }

        // Update grid params
        let grid_params = GridParamsUniform {
            cols: cols as u32,
            rows: rows as u32,
            cell_width: self.cell_width,
            cell_height: self.cell_height,
            screen_width: self.width as f32,
            screen_height: self.height as f32,
            y_offset: self.terminal_y_offset(),
            cursor_col: if terminal.cursor_visible { terminal.cursor_col as i32 } else { -1 },
            cursor_row: if terminal.cursor_visible { terminal.cursor_row as i32 } else { -1 },
            cursor_style: match terminal.cursor_shape {
                CursorShape::BlinkingBlock | CursorShape::SteadyBlock => 0,
                CursorShape::BlinkingUnderline | CursorShape::SteadyUnderline => 1,
                CursorShape::BlinkingBar | CursorShape::SteadyBar => 2,
            },
            _padding: [0, 0],
        };

        if let Some(ref buffer) = self.grid_params_buffer {
            self.queue.write_buffer(buffer, 0, bytemuck::bytes_of(&grid_params));
        }

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

        // Render
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Instanced Render Encoder"),
        });

        {
            // Clear with background color
            let [bg_r, bg_g, bg_b] = terminal.palette.default_bg;
            let bg_r_linear = Self::srgb_to_linear(bg_r as f32 / 255.0) as f64;
            let bg_g_linear = Self::srgb_to_linear(bg_g as f32 / 255.0) as f64;
            let bg_b_linear = Self::srgb_to_linear(bg_b as f32 / 255.0) as f64;
            let bg_alpha = self.background_opacity as f64;

            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Instanced Render Pass"),
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

            // Get references to avoid borrow issues
            let cell_bg_pipeline = self.cell_bg_pipeline.as_ref();
            let cell_glyph_pipeline = self.cell_glyph_pipeline.as_ref();
            let glyph_bind_group = &self.glyph_bind_group;
            let cell_bind_group = self.cell_bind_group.as_ref();
            let quad_index_buffer = self.quad_index_buffer.as_ref();

            if let (Some(bg_pipeline), Some(glyph_pipeline), Some(cell_bg), Some(idx_buf)) = 
                (cell_bg_pipeline, cell_glyph_pipeline, cell_bind_group, quad_index_buffer) 
            {
                // Pass 1: Render backgrounds
                render_pass.set_pipeline(bg_pipeline);
                render_pass.set_bind_group(0, glyph_bind_group, &[]);
                render_pass.set_bind_group(1, cell_bg, &[]);
                render_pass.set_index_buffer(idx_buf.slice(..), wgpu::IndexFormat::Uint16);
                render_pass.draw_indexed(0..6, 0, 0..num_cells as u32);

                // Pass 2: Render glyphs
                render_pass.set_pipeline(glyph_pipeline);
                // Bind groups already set
                render_pass.draw_indexed(0..6, 0, 0..num_cells as u32);
            }
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}
