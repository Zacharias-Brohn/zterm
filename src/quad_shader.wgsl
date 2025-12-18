// Instanced quad rendering shader for rectangles, borders, overlays, and tab bar
// Simple shader that renders colored rectangles using instancing

// ═══════════════════════════════════════════════════════════════════════════════
// DATA STRUCTURES
// ═══════════════════════════════════════════════════════════════════════════════

// Quad instance data - stored in a storage buffer
// Each quad has position, size, and color
struct Quad {
    // Position in pixels (x, y)
    x: f32,
    y: f32,
    // Size in pixels (width, height)
    width: f32,
    height: f32,
    // Color (linear RGBA)
    color: vec4<f32>,
}

// Quad rendering uniforms (screen dimensions)
struct QuadParams {
    screen_width: f32,
    screen_height: f32,
    _padding: vec2<f32>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// BINDINGS
// ═══════════════════════════════════════════════════════════════════════════════

// Uniform for screen dimensions
@group(0) @binding(0)
var<uniform> quad_params: QuadParams;

// Storage buffer for quad instances
@group(0) @binding(1)
var<storage, read> quads: array<Quad>;

// ═══════════════════════════════════════════════════════════════════════════════
// VERTEX OUTPUT
// ═══════════════════════════════════════════════════════════════════════════════

struct QuadVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

// ═══════════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════════

// Convert pixel coordinate to NDC
fn pixel_to_ndc(pixel: vec2<f32>, screen: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        (pixel.x / screen.x) * 2.0 - 1.0,
        1.0 - (pixel.y / screen.y) * 2.0
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// VERTEX SHADER
// ═══════════════════════════════════════════════════════════════════════════════

@vertex
fn vs_quad(
    @builtin(vertex_index) vertex_index: u32,
    @builtin(instance_index) instance_index: u32
) -> QuadVertexOutput {
    let quad = quads[instance_index];
    
    // Quad vertex positions for TriangleStrip (0=top-left, 1=top-right, 2=bottom-left, 3=bottom-right)
    var positions: array<vec2<f32>, 4>;
    positions[0] = vec2<f32>(quad.x, quad.y);                           // top-left
    positions[1] = vec2<f32>(quad.x + quad.width, quad.y);              // top-right
    positions[2] = vec2<f32>(quad.x, quad.y + quad.height);             // bottom-left
    positions[3] = vec2<f32>(quad.x + quad.width, quad.y + quad.height); // bottom-right
    
    let screen_size = vec2<f32>(quad_params.screen_width, quad_params.screen_height);
    let ndc_pos = pixel_to_ndc(positions[vertex_index], screen_size);
    
    var out: QuadVertexOutput;
    out.clip_position = vec4<f32>(ndc_pos, 0.0, 1.0);
    out.color = quad.color;
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════════
// FRAGMENT SHADER
// ═══════════════════════════════════════════════════════════════════════════════

@fragment
fn fs_quad(in: QuadVertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}
