//! Image rendering for the Kitty Graphics Protocol.
//!
//! This module handles GPU-accelerated rendering of images in the terminal,
//! supporting the Kitty Graphics Protocol for inline image display.

use std::collections::HashMap;
use crate::gpu_types::ImageUniforms;
use crate::graphics::{ImageData, ImagePlacement, ImageStorage};

// ═══════════════════════════════════════════════════════════════════════════════
// GPU IMAGE
// ═══════════════════════════════════════════════════════════════════════════════

/// Cached GPU texture for an image.
pub struct GpuImage {
    pub texture: wgpu::Texture,
    pub view: wgpu::TextureView,
    pub uniform_buffer: wgpu::Buffer,
    pub bind_group: wgpu::BindGroup,
    pub width: u32,
    pub height: u32,
}

// ═══════════════════════════════════════════════════════════════════════════════
// IMAGE RENDERER
// ═══════════════════════════════════════════════════════════════════════════════

/// Manages GPU resources for image rendering.
/// Handles uploading, caching, and preparing images for rendering.
pub struct ImageRenderer {
    /// Bind group layout for image rendering.
    bind_group_layout: wgpu::BindGroupLayout,
    /// Sampler for image textures.
    sampler: wgpu::Sampler,
    /// Cached GPU textures for images, keyed by image ID.
    textures: HashMap<u32, GpuImage>,
}

impl ImageRenderer {
    /// Create a new ImageRenderer with the necessary GPU resources.
    pub fn new(device: &wgpu::Device) -> Self {
        // Create sampler for images (linear filtering for smooth scaling)
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Image Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Nearest,
            ..Default::default()
        });

        // Create bind group layout for images
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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

        Self {
            bind_group_layout,
            sampler,
            textures: HashMap::new(),
        }
    }

    /// Get the bind group layout for creating the image pipeline.
    pub fn bind_group_layout(&self) -> &wgpu::BindGroupLayout {
        &self.bind_group_layout
    }

    /// Get a GPU image by ID.
    pub fn get(&self, image_id: &u32) -> Option<&GpuImage> {
        self.textures.get(image_id)
    }

    /// Upload an image to the GPU, creating or updating its texture.
    pub fn upload_image(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, image: &ImageData) {
        // Get current frame data (handles animation frames automatically)
        let data = image.current_frame_data();
        
        // Check if we already have this image
        if let Some(existing) = self.textures.get(&image.id) {
            if existing.width == image.width && existing.height == image.height {
                // Same dimensions, just update the data
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &existing.texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    data,
                    wgpu::TexelCopyBufferLayout {
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
        let texture = device.create_texture(&wgpu::TextureDescriptor {
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
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            data,
            wgpu::TexelCopyBufferLayout {
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
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("Image {} Uniform Buffer", image.id)),
            size: std::mem::size_of::<ImageUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create bind group for this image with its own uniform buffer
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("Image {} Bind Group", image.id)),
            layout: &self.bind_group_layout,
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
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });

        self.textures.insert(image.id, GpuImage {
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
        if self.textures.remove(&image_id).is_some() {
            log::debug!("Removed image {} from GPU", image_id);
        }
    }

    /// Sync images from terminal's image storage to GPU.
    /// Uploads new/changed images and removes deleted ones.
    /// Also updates animation frames.
    pub fn sync_images(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, storage: &mut ImageStorage) {
        // Update animations and get list of changed image IDs
        let changed_ids = storage.update_animations();

        // Re-upload frames that changed due to animation
        for id in &changed_ids {
            if let Some(image) = storage.get_image(*id) {
                self.upload_image(device, queue, image);
            }
        }

        if !storage.dirty && changed_ids.is_empty() {
            return;
        }

        // Upload all images (upload_image handles deduplication)
        for image in storage.images().values() {
            self.upload_image(device, queue, image);
        }

        // Remove textures for deleted images
        let current_ids: std::collections::HashSet<u32> = storage.images().keys().copied().collect();
        let gpu_ids: Vec<u32> = self.textures.keys().copied().collect();
        for id in gpu_ids {
            if !current_ids.contains(&id) {
                self.remove_image(id);
            }
        }

        storage.clear_dirty();
    }

    /// Prepare image renders for a pane.
    /// Returns a Vec of (image_id, uniforms) for deferred rendering.
    pub fn prepare_image_renders(
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
            let gpu_image = match self.textures.get(&placement.image_id) {
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
