# Atlas Texture Refactor: Array of Textures

## Problem

When adding a new layer to the glyph atlas, the current implementation creates a new texture array with N+1 layers and copies all N existing layers to it. This causes performance issues that scale with the number of layers:

- Layer 1→2: Copy 256MB (8192×8192×4 bytes)
- Layer 2→3: Copy 512MB
- Layer 3→4: Copy 768MB
- etc.

Observed frame times when adding layers:
- Layer 1 added: 14.4ms
- Layer 2 added: 21.9ms
- Layer 3 added: 34.2ms

## Solution

Instead of using a single `texture_2d_array` that must be reallocated and copied when growing, use a **`Vec` of separate 2D textures**. When a new layer is needed, simply create a new texture and add it to the vector. No copying of existing texture data is required.

The bind group must be recreated to include the new texture, but this is a cheap CPU-side operation (just creating metadata/pointers).

## Current Implementation

### Rust (renderer.rs)

**Struct fields:**
```rust
atlas_texture: wgpu::Texture,           // Single texture array
atlas_view: wgpu::TextureView,          // Single view
atlas_num_layers: u32,                  // Number of layers in the array
atlas_current_layer: u32,               // Current layer being written to
```

**Bind group layout (binding 0):**
```rust
ty: wgpu::BindingType::Texture {
    view_dimension: wgpu::TextureViewDimension::D2Array,
    // ...
},
count: None,  // Single texture
```

**Bind group entry:**
```rust
wgpu::BindGroupEntry {
    binding: 0,
    resource: wgpu::BindingResource::TextureView(&atlas_view),
}
```

### WGSL (glyph_shader.wgsl)

**Texture declaration:**
```wgsl
@group(0) @binding(0)
var atlas_texture: texture_2d_array<f32>;
```

**Sampling:**
```wgsl
let sample = textureSample(atlas_texture, atlas_sampler, uv, layer_index);
```

## New Implementation

### Rust (renderer.rs)

**Struct fields:**
```rust
atlas_textures: Vec<wgpu::Texture>,     // Vector of separate textures
atlas_views: Vec<wgpu::TextureView>,    // Vector of views (one per texture)
atlas_current_layer: u32,               // Current layer being written to
// atlas_num_layers removed - use atlas_textures.len() instead
```

**Bind group layout (binding 0):**
```rust
ty: wgpu::BindingType::Texture {
    view_dimension: wgpu::TextureViewDimension::D2,  // Changed from D2Array
    // ...
},
count: Some(NonZeroU32::new(MAX_ATLAS_LAYERS).unwrap()),  // Array of textures
```

**Bind group entry:**
```rust
wgpu::BindGroupEntry {
    binding: 0,
    resource: wgpu::BindingResource::TextureViewArray(&atlas_view_refs),
}
```

Where `atlas_view_refs` is a `Vec<&wgpu::TextureView>` containing references to all views.

**Note:** wgpu requires the bind group to have exactly `count` textures. We'll need to either:
1. Pre-create dummy textures to fill unused slots, OR
2. Recreate the bind group layout when adding textures (more complex)

Option 1 is simpler: create small 1x1 dummy textures for unused slots up to MAX_ATLAS_LAYERS.

### WGSL (glyph_shader.wgsl)

**Texture declaration:**
```wgsl
@group(0) @binding(0)
var atlas_textures: binding_array<texture_2d<f32>>;
```

**Sampling:**
```wgsl
let sample = textureSample(atlas_textures[layer_index], atlas_sampler, uv);
```

**Note:** `binding_array` requires the `binding_array` feature in wgpu, which should be enabled by default on most backends.

## Implementation Steps

### Step 1: Update Struct Fields

In `renderer.rs`, change:
```rust
// Old
atlas_texture: wgpu::Texture,
atlas_view: wgpu::TextureView,
atlas_num_layers: u32,

// New
atlas_textures: Vec<wgpu::Texture>,
atlas_views: Vec<wgpu::TextureView>,
```

### Step 2: Create Helper for New Atlas Layer

```rust
fn create_atlas_layer(device: &wgpu::Device) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Glyph Atlas Layer"),
        size: wgpu::Extent3d {
            width: ATLAS_SIZE,
            height: ATLAS_SIZE,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}
```

### Step 3: Update Bind Group Layout

```rust
use std::num::NonZeroU32;

let glyph_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
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
            count: Some(NonZeroU32::new(MAX_ATLAS_LAYERS).unwrap()),
        },
        // ... sampler entry unchanged
    ],
});
```

### Step 4: Initialize with Dummy Textures

At initialization, create one real texture and fill the rest with 1x1 dummy textures:

```rust
fn create_dummy_texture(device: &wgpu::Device) -> (wgpu::Texture, wgpu::TextureView) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("Dummy Atlas Texture"),
        size: wgpu::Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    (texture, view)
}

// In new():
let mut atlas_textures = Vec::with_capacity(MAX_ATLAS_LAYERS as usize);
let mut atlas_views = Vec::with_capacity(MAX_ATLAS_LAYERS as usize);

// First texture is real
let (tex, view) = create_atlas_layer(&device);
atlas_textures.push(tex);
atlas_views.push(view);

// Fill rest with dummies
for _ in 1..MAX_ATLAS_LAYERS {
    let (tex, view) = create_dummy_texture(&device);
    atlas_textures.push(tex);
    atlas_views.push(view);
}
```

### Step 5: Update Bind Group Creation

```rust
fn create_atlas_bind_group(&self) -> wgpu::BindGroup {
    let view_refs: Vec<&wgpu::TextureView> = self.atlas_views.iter().collect();
    
    self.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Glyph Bind Group"),
        layout: &self.glyph_bind_group_layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureViewArray(&view_refs),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::Sampler(&self.atlas_sampler),
            },
        ],
    })
}
```

### Step 6: Update add_atlas_layer / ensure_atlas_layer_capacity

```rust
fn ensure_atlas_layer_capacity(&mut self, target_layer: u32) {
    // Count real layers (non-dummy)
    let current_real_layers = self.atlas_current_layer + 1;
    
    if target_layer < current_real_layers {
        return; // Already have this layer
    }
    
    if target_layer >= MAX_ATLAS_LAYERS {
        log::error!("Atlas layer limit reached");
        return;
    }
    
    log::info!("Adding atlas layer {}", target_layer);
    
    // Create new real texture
    let (texture, view) = Self::create_atlas_layer(&self.device);
    
    // Replace dummy at this index with real texture
    self.atlas_textures[target_layer as usize] = texture;
    self.atlas_views[target_layer as usize] = view;
    
    // Recreate bind group with updated view
    self.glyph_bind_group = self.create_atlas_bind_group();
}
```

### Step 7: Update upload_cell_canvas_to_atlas

Change texture reference from `&self.atlas_texture` to `&self.atlas_textures[layer as usize]`:

```rust
self.queue.write_texture(
    wgpu::TexelCopyTextureInfo {
        texture: &self.atlas_textures[layer as usize],  // Changed
        mip_level: 0,
        origin: wgpu::Origin3d {
            x: self.atlas_cursor_x,
            y: self.atlas_cursor_y,
            z: 0,  // Always 0 now, layer is selected by texture index
        },
        aspect: wgpu::TextureAspect::All,
    },
    // ... rest unchanged
);
```

### Step 8: Update Shader

In `glyph_shader.wgsl`:

```wgsl
// Old
@group(0) @binding(0)
var atlas_texture: texture_2d_array<f32>;

// New
@group(0) @binding(0)
var atlas_textures: binding_array<texture_2d<f32>>;
```

Update all `textureSample` calls:

```wgsl
// Old
let sample = textureSample(atlas_texture, atlas_sampler, uv, layer_index);

// New
let sample = textureSample(atlas_textures[layer_index], atlas_sampler, uv);
```

**Locations to update in glyph_shader.wgsl:**
- Line 91: Declaration
- Line 106: `fs_main` sampling
- Line 700: Cursor sprite sampling in `fs_cell`
- Line 723: Glyph sampling in `fs_cell`
- Line 747: Underline sampling in `fs_cell`
- Line 761: Strikethrough sampling in `fs_cell`

### Step 9: Update statusline_shader.wgsl

The statusline shader also uses the atlas. Check and update similarly.

### Step 10: Update Other References

Search for all uses of:
- `atlas_texture` 
- `atlas_view`
- `atlas_num_layers`
- `D2Array`

And update accordingly.

## Testing

1. Build and run: `cargo build && cargo run`
2. Verify glyphs render correctly
3. Use terminal heavily to trigger layer additions
4. Check logs for "Adding atlas layer" messages
5. Verify no slow frame warnings during layer addition
6. Test with emoji (color glyphs) to ensure they still work

## Rollback Plan

If issues arise, the changes can be reverted by:
1. Restoring `texture_2d_array` in shaders
2. Restoring single `atlas_texture`/`atlas_view` in Rust
3. Restoring the copy-based layer addition

## References

- wgpu binding arrays: https://docs.rs/wgpu/latest/wgpu/enum.BindingResource.html#variant.TextureViewArray
- WGSL binding_array: https://www.w3.org/TR/WGSL/#binding-array
- Kitty's approach: `/tmp/kitty/kitty/shaders.c` (uses `GL_TEXTURE_2D_ARRAY` with `glCopyImageSubData`)
