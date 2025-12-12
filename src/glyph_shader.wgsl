// Glyph rendering shader for terminal emulator

struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) bg_color: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) bg_color: vec4<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.clip_position = vec4<f32>(in.position, 0.0, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    out.bg_color = in.bg_color;
    return out;
}

@group(0) @binding(0)
var atlas_texture: texture_2d<f32>;
@group(0) @binding(1)
var atlas_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // If UV is at origin (0,0), this is a background-only quad
    let is_background_only = in.uv.x == 0.0 && in.uv.y == 0.0;
    
    if is_background_only {
        // Just render the background color (fully opaque)
        return in.bg_color;
    }
    
    // Sample the glyph alpha from the atlas
    let glyph_alpha = textureSample(atlas_texture, atlas_sampler, in.uv).r;
    
    // Output foreground color with glyph alpha for blending
    // The background was already rendered, so we just blend the glyph on top
    return vec4<f32>(in.color.rgb, in.color.a * glyph_alpha);
}
