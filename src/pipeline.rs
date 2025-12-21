//! Render pipeline builder for wgpu.
//!
//! This module provides a builder pattern for creating wgpu render pipelines
//! with common settings, reducing boilerplate when creating multiple pipelines.

// ═══════════════════════════════════════════════════════════════════════════════
// PIPELINE BUILDER
// ═══════════════════════════════════════════════════════════════════════════════

/// Builder for creating render pipelines with common settings.
/// Captures the device, shader, layout, and format that are shared across many pipelines.
pub struct PipelineBuilder<'a> {
    device: &'a wgpu::Device,
    shader: &'a wgpu::ShaderModule,
    layout: &'a wgpu::PipelineLayout,
    format: wgpu::TextureFormat,
}

impl<'a> PipelineBuilder<'a> {
    /// Create a new pipeline builder with shared settings.
    pub fn new(
        device: &'a wgpu::Device,
        shader: &'a wgpu::ShaderModule,
        layout: &'a wgpu::PipelineLayout,
        format: wgpu::TextureFormat,
    ) -> Self {
        Self { device, shader, layout, format }
    }

    /// Build a pipeline with TriangleStrip topology and no vertex buffers (most common case).
    pub fn build(&self, label: &str, vs_entry: &str, fs_entry: &str, blend: wgpu::BlendState) -> wgpu::RenderPipeline {
        self.build_full(label, vs_entry, fs_entry, blend, wgpu::PrimitiveTopology::TriangleStrip, &[])
    }

    /// Build a pipeline with custom topology and vertex buffers.
    pub fn build_full(
        &self,
        label: &str,
        vs_entry: &str,
        fs_entry: &str,
        blend: wgpu::BlendState,
        topology: wgpu::PrimitiveTopology,
        vertex_buffers: &[wgpu::VertexBufferLayout<'_>],
    ) -> wgpu::RenderPipeline {
        self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(self.layout),
            vertex: wgpu::VertexState {
                module: self.shader,
                entry_point: Some(vs_entry),
                buffers: vertex_buffers,
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: self.shader,
                entry_point: Some(fs_entry),
                targets: &[Some(wgpu::ColorTargetState {
                    format: self.format,
                    blend: Some(blend),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
    }
}
