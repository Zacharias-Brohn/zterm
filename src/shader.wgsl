// Edge Glow Shader
// Renders a soft glow effect at terminal edges for failed pane navigation feedback.
// The glow appears as a light node at center that splits into two and travels to corners.

// Uniform buffer with glow parameters
struct EdgeGlowParams {
    // Screen dimensions in pixels
    screen_width: f32,
    screen_height: f32,
    // Terminal area offset (for tab bar)
    terminal_y_offset: f32,
    // Direction: 0=Up, 1=Down, 2=Left, 3=Right
    direction: u32,
    // Animation progress (0.0 to 1.0)
    progress: f32,
    // Glow color (linear RGB) - stored as separate floats to avoid vec3 alignment issues
    color_r: f32,
    color_g: f32,
    color_b: f32,
    // Whether glow is enabled (1 = yes, 0 = no)
    enabled: u32,
    // Padding to align to 16 bytes
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
}

@group(0) @binding(0)
var<uniform> params: EdgeGlowParams;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,  // 0-1 normalized screen coordinates
}

// Fullscreen triangle vertex shader
// Uses vertex_index 0,1,2 to create a triangle that covers the screen
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // Generate fullscreen triangle vertices
    // This creates a triangle that covers [-1,1] in clip space
    let x = f32(i32(vertex_index) - 1);
    let y = f32(i32(vertex_index & 1u) * 2 - 1);
    
    // Positions for a fullscreen triangle
    var pos: vec2<f32>;
    switch vertex_index {
        case 0u: { pos = vec2<f32>(-1.0, -1.0); }
        case 1u: { pos = vec2<f32>(3.0, -1.0); }
        case 2u: { pos = vec2<f32>(-1.0, 3.0); }
        default: { pos = vec2<f32>(0.0, 0.0); }
    }
    
    out.clip_position = vec4<f32>(pos, 0.0, 1.0);
    // Convert to 0-1 UV (flip Y since clip space Y is up, pixel Y is down)
    out.uv = vec2<f32>((pos.x + 1.0) * 0.5, (1.0 - pos.y) * 0.5);
    
    return out;
}

// Constants
const PI: f32 = 3.14159265359;
const PHASE1_END: f32 = 0.15;      // Phase 1 ends at 15% progress
const GLOW_RADIUS: f32 = 90.0;     // Base radius of glow
const GLOW_ASPECT: f32 = 2.0;      // Stretch factor along edge (ellipse)

// Smooth gaussian-like falloff
fn glow_falloff(dist: f32, radius: f32) -> f32 {
    let normalized = dist / radius;
    if normalized > 1.0 {
        return 0.0;
    }
    // Smooth falloff: (1 - x^2)^3 gives nice soft edges
    let t = 1.0 - normalized * normalized;
    return t * t * t;
}

// Ease-out cubic
fn ease_out_cubic(t: f32) -> f32 {
    let t1 = 1.0 - t;
    return 1.0 - t1 * t1 * t1;
}

// Calculate distance from point to glow center, accounting for ellipse shape
fn ellipse_distance(point: vec2<f32>, center: vec2<f32>, radius_along: f32, radius_perp: f32, is_horizontal: bool) -> f32 {
    let delta = point - center;
    var normalized: vec2<f32>;
    if is_horizontal {
        normalized = vec2<f32>(delta.x / radius_along, delta.y / radius_perp);
    } else {
        normalized = vec2<f32>(delta.x / radius_perp, delta.y / radius_along);
    }
    return length(normalized) * min(radius_along, radius_perp);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Early out if not enabled
    if params.enabled == 0u {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    let progress = params.progress;
    
    // Convert UV to pixel coordinates
    let pixel = vec2<f32>(
        in.uv.x * params.screen_width,
        in.uv.y * params.screen_height
    );
    
    let terminal_height = params.screen_height - params.terminal_y_offset;
    let is_horizontal = params.direction == 0u || params.direction == 1u;
    
    // Calculate glow parameters based on animation phase
    var alpha: f32;
    var size_factor: f32;
    var split: f32;
    
    if progress < PHASE1_END {
        // Phase 1: Fade in, grow
        let t = progress / PHASE1_END;
        let ease = ease_out_cubic(t);
        alpha = ease * 0.8;
        size_factor = 0.3 + 0.7 * ease;
        split = 0.0;
    } else {
        // Phase 2: Split and fade out
        let t = (progress - PHASE1_END) / (1.0 - PHASE1_END);
        let fade = 1.0 - t;
        alpha = fade * fade * 0.8;
        size_factor = 1.0 - 0.3 * t;
        split = ease_out_cubic(t);
    }
    
    let base_radius = GLOW_RADIUS * size_factor;
    let radius_along = base_radius * GLOW_ASPECT;
    let radius_perp = base_radius;
    
    // Calculate edge center and travel distance based on direction
    var edge_center: vec2<f32>;
    var travel: vec2<f32>;
    
    switch params.direction {
        // Up - top edge
        case 0u: {
            edge_center = vec2<f32>(params.screen_width / 2.0, params.terminal_y_offset);
            travel = vec2<f32>(params.screen_width / 2.0, 0.0);
        }
        // Down - bottom edge
        case 1u: {
            edge_center = vec2<f32>(params.screen_width / 2.0, params.screen_height);
            travel = vec2<f32>(params.screen_width / 2.0, 0.0);
        }
        // Left - left edge
        case 2u: {
            edge_center = vec2<f32>(0.0, params.terminal_y_offset + terminal_height / 2.0);
            travel = vec2<f32>(0.0, terminal_height / 2.0);
        }
        // Right - right edge
        case 3u: {
            edge_center = vec2<f32>(params.screen_width, params.terminal_y_offset + terminal_height / 2.0);
            travel = vec2<f32>(0.0, terminal_height / 2.0);
        }
        default: {
            edge_center = vec2<f32>(0.0, 0.0);
            travel = vec2<f32>(0.0, 0.0);
        }
    }
    
    var glow_intensity: f32 = 0.0;
    
    if split < 0.01 {
        // Single glow at center
        let dist = ellipse_distance(pixel, edge_center, radius_along, radius_perp, is_horizontal);
        glow_intensity = glow_falloff(dist, base_radius);
    } else {
        // Two glows splitting apart
        let split_radius = base_radius * (1.0 - 0.2 * split);
        let split_radius_along = radius_along * (1.0 - 0.2 * split);
        let split_radius_perp = radius_perp * (1.0 - 0.2 * split);
        
        let center1 = edge_center - travel * split;
        let center2 = edge_center + travel * split;
        
        let dist1 = ellipse_distance(pixel, center1, split_radius_along, split_radius_perp, is_horizontal);
        let dist2 = ellipse_distance(pixel, center2, split_radius_along, split_radius_perp, is_horizontal);
        
        // Combine both glows (additive but capped)
        let glow1 = glow_falloff(dist1, split_radius);
        let glow2 = glow_falloff(dist2, split_radius);
        glow_intensity = min(glow1 + glow2, 1.0);
    }
    
    // Apply alpha
    let final_alpha = glow_intensity * alpha;
    
    // Output with premultiplied alpha for proper blending
    let color = vec3<f32>(params.color_r, params.color_g, params.color_b);
    return vec4<f32>(color * final_alpha, final_alpha);
}
