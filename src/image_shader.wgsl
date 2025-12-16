// Image rendering shader for Kitty graphics protocol
// Renders RGBA images with proper alpha blending

struct ImageUniforms {
    // Screen dimensions in pixels
    screen_width: f32,
    screen_height: f32,
    // Image position in pixels (top-left corner)
    pos_x: f32,
    pos_y: f32,
    // Image display size in pixels
    display_width: f32,
    display_height: f32,
    // Source rectangle in normalized coordinates (0-1)
    src_x: f32,
    src_y: f32,
    src_width: f32,
    src_height: f32,
    // Padding for alignment
    _padding1: f32,
    _padding2: f32,
}

@group(0) @binding(0)
var<uniform> uniforms: ImageUniforms;

@group(0) @binding(1)
var image_texture: texture_2d<f32>;

@group(0) @binding(2)
var image_sampler: sampler;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// Convert pixel coordinate to NDC
fn pixel_to_ndc(pixel: vec2<f32>, screen: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        (pixel.x / screen.x) * 2.0 - 1.0,
        1.0 - (pixel.y / screen.y) * 2.0
    );
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // Calculate quad corners in pixel space
    let x0 = uniforms.pos_x;
    let y0 = uniforms.pos_y;
    let x1 = uniforms.pos_x + uniforms.display_width;
    let y1 = uniforms.pos_y + uniforms.display_height;
    
    // Quad vertex positions (0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right)
    // Using triangle strip order
    var positions: array<vec2<f32>, 4>;
    positions[0] = vec2<f32>(x0, y0);
    positions[1] = vec2<f32>(x1, y0);
    positions[2] = vec2<f32>(x0, y1);
    positions[3] = vec2<f32>(x1, y1);
    
    // UV coordinates mapping to source rectangle
    var uvs: array<vec2<f32>, 4>;
    let u0 = uniforms.src_x;
    let v0 = uniforms.src_y;
    let u1 = uniforms.src_x + uniforms.src_width;
    let v1 = uniforms.src_y + uniforms.src_height;
    
    uvs[0] = vec2<f32>(u0, v0);
    uvs[1] = vec2<f32>(u1, v0);
    uvs[2] = vec2<f32>(u0, v1);
    uvs[3] = vec2<f32>(u1, v1);
    
    let screen_size = vec2<f32>(uniforms.screen_width, uniforms.screen_height);
    let ndc_pos = pixel_to_ndc(positions[vertex_index], screen_size);
    
    out.clip_position = vec4<f32>(ndc_pos, 0.0, 1.0);
    out.uv = uvs[vertex_index];
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Sample the image texture
    let color = textureSample(image_texture, image_sampler, in.uv);
    
    // Return with premultiplied alpha for proper blending
    return vec4<f32>(color.rgb * color.a, color.a);
}
