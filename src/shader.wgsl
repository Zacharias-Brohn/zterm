// Edge Glow Shader
// Renders natural-looking light effects at terminal edges for failed pane navigation feedback.
// Supports multiple simultaneous lights that blend together.
// Features: bright hot center, colored mid-range, soft outer halo with bloom.

// Maximum number of simultaneous glows
const MAX_GLOWS: u32 = 16u;

// Per-glow parameters (48 bytes each, aligned to 16 bytes)
struct GlowInstance {
    // Direction: 0=Up, 1=Down, 2=Left, 3=Right
    direction: u32,
    // Animation progress (0.0 to 1.0)
    progress: f32,
    // Glow color (linear RGB)
    color_r: f32,
    color_g: f32,
    color_b: f32,
    // Pane bounds in pixels
    pane_x: f32,
    pane_y: f32,
    pane_width: f32,
    pane_height: f32,
    // Padding to align to 16 bytes
    _padding1: f32,
    _padding2: f32,
    _padding3: f32,
}

// Global parameters + array of glow instances
struct EdgeGlowParams {
    // Screen dimensions in pixels
    screen_width: f32,
    screen_height: f32,
    // Terminal area offset (for tab bar)
    terminal_y_offset: f32,
    // Glow intensity multiplier (0.0 = disabled, 1.0 = full)
    glow_intensity: f32,
    // Number of active glows
    glow_count: u32,
    // Padding to align to 16 bytes before array
    _padding1: u32,
    _padding2: u32,
    _padding3: u32,
    // Array of glow instances
    glows: array<GlowInstance, 16>,
}

@group(0) @binding(0)
var<uniform> params: EdgeGlowParams;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,  // 0-1 normalized screen coordinates
}

// Fullscreen triangle vertex shader
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    var out: VertexOutput;
    
    // Positions for a fullscreen triangle
    var pos: vec2<f32>;
    switch vertex_index {
        case 0u: { pos = vec2<f32>(-1.0, -1.0); }
        case 1u: { pos = vec2<f32>(3.0, -1.0); }
        case 2u: { pos = vec2<f32>(-1.0, 3.0); }
        default: { pos = vec2<f32>(0.0, 0.0); }
    }
    
    out.clip_position = vec4<f32>(pos, 0.0, 1.0);
    out.uv = vec2<f32>((pos.x + 1.0) * 0.5, (1.0 - pos.y) * 0.5);
    
    return out;
}

// Constants
const PI: f32 = 3.14159265359;
const PHASE1_END: f32 = 0.15;      // Phase 1 ends at 15% progress
const GLOW_RADIUS: f32 = 80.0;     // Core radius of the light
const GLOW_ASPECT: f32 = 2.5;      // Stretch factor along edge (ellipse)

// Ease-out cubic
fn ease_out_cubic(t: f32) -> f32 {
    let t1 = 1.0 - t;
    return 1.0 - t1 * t1 * t1;
}

// Calculate normalized distance from point to glow center (elliptical)
// Returns 0 at center, 1 at edge of core, >1 outside
fn ellipse_dist_normalized(point: vec2<f32>, center: vec2<f32>, radius_along: f32, radius_perp: f32, is_horizontal: bool) -> f32 {
    let delta = point - center;
    var normalized: vec2<f32>;
    if is_horizontal {
        normalized = vec2<f32>(delta.x / radius_along, delta.y / radius_perp);
    } else {
        normalized = vec2<f32>(delta.x / radius_perp, delta.y / radius_along);
    }
    return length(normalized);
}

// Natural light intensity falloff
// Creates a bright core with soft extended halo
fn light_intensity(dist: f32) -> f32 {
    // Multi-layer falloff for natural light appearance:
    // 1. Bright core (inverse square-ish, clamped)
    // 2. Soft halo that extends further
    
    if dist < 0.001 {
        return 1.0;
    }
    
    // Core intensity - bright center that falls off quickly
    // Using smoothed inverse for the hot center
    let core = 1.0 / (1.0 + dist * dist * 4.0);
    
    // Soft halo - gaussian-like falloff that extends further
    let halo = exp(-dist * dist * 1.5);
    
    // Combine: core dominates near center, halo extends the glow
    return core * 0.7 + halo * 0.5;
}

// Calculate the "hotness" - how white/bright the center should be
// Returns 0-1 where 1 = pure white (hottest), 0 = base color
fn light_hotness(dist: f32) -> f32 {
    // Very bright white core that quickly transitions to color
    let hot = 1.0 / (1.0 + dist * dist * 12.0);
    return hot * hot; // Square it for sharper transition
}

// Calculate contribution from a single glow at the given pixel
// Returns (intensity, hotness, 1.0) packed in vec3
fn calculate_glow(pixel: vec2<f32>, glow: GlowInstance) -> vec3<f32> {
    // Get pane bounds from the glow instance
    let pane_x = glow.pane_x;
    let pane_y = glow.pane_y;
    let pane_width = glow.pane_width;
    let pane_height = glow.pane_height;
    
    // Mask: if pixel is outside pane bounds, return zero contribution
    if pixel.x < pane_x || pixel.x > pane_x + pane_width ||
       pixel.y < pane_y || pixel.y > pane_y + pane_height {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    
    let progress = glow.progress;
    let is_horizontal = glow.direction == 0u || glow.direction == 1u;
    
    // Calculate glow parameters based on animation phase
    var intensity_mult: f32;
    var size_factor: f32;
    var split: f32;
    
    if progress < PHASE1_END {
        // Phase 1: Appear and grow
        let t = progress / PHASE1_END;
        let ease = ease_out_cubic(t);
        intensity_mult = ease;
        size_factor = 0.4 + 0.6 * ease;
        split = 0.0;
    } else {
        // Phase 2: Split and fade out
        let t = (progress - PHASE1_END) / (1.0 - PHASE1_END);
        let fade = 1.0 - t;
        // Slower fade for more visible effect
        intensity_mult = fade * fade * fade;
        size_factor = 1.0 - 0.2 * t;
        split = ease_out_cubic(t);
    }
    
    let base_radius = GLOW_RADIUS * size_factor;
    let radius_along = base_radius * GLOW_ASPECT;
    let radius_perp = base_radius;
    
    // Calculate edge center and travel distance based on direction
    // Now using pane bounds instead of screen bounds
    var edge_center: vec2<f32>;
    var travel: vec2<f32>;
    
    switch glow.direction {
        // Up - top edge of pane
        case 0u: {
            edge_center = vec2<f32>(pane_x + pane_width / 2.0, pane_y);
            travel = vec2<f32>(pane_width / 2.0, 0.0);
        }
        // Down - bottom edge of pane
        case 1u: {
            edge_center = vec2<f32>(pane_x + pane_width / 2.0, pane_y + pane_height);
            travel = vec2<f32>(pane_width / 2.0, 0.0);
        }
        // Left - left edge of pane
        case 2u: {
            edge_center = vec2<f32>(pane_x, pane_y + pane_height / 2.0);
            travel = vec2<f32>(0.0, pane_height / 2.0);
        }
        // Right - right edge of pane
        case 3u: {
            edge_center = vec2<f32>(pane_x + pane_width, pane_y + pane_height / 2.0);
            travel = vec2<f32>(0.0, pane_height / 2.0);
        }
        default: {
            edge_center = vec2<f32>(0.0, 0.0);
            travel = vec2<f32>(0.0, 0.0);
        }
    }
    
    // Accumulate light from one or two sources
    var total_intensity: f32 = 0.0;
    var total_hotness: f32 = 0.0;
    
    if split < 0.01 {
        // Single light at center
        let dist = ellipse_dist_normalized(pixel, edge_center, radius_along, radius_perp, is_horizontal);
        total_intensity = light_intensity(dist);
        total_hotness = light_hotness(dist);
    } else {
        // Two lights splitting apart
        let split_factor = 1.0 - 0.15 * split;
        let r_along = radius_along * split_factor;
        let r_perp = radius_perp * split_factor;
        
        let center1 = edge_center - travel * split;
        let center2 = edge_center + travel * split;
        
        let dist1 = ellipse_dist_normalized(pixel, center1, r_along, r_perp, is_horizontal);
        let dist2 = ellipse_dist_normalized(pixel, center2, r_along, r_perp, is_horizontal);
        
        let intensity1 = light_intensity(dist1);
        let intensity2 = light_intensity(dist2);
        let hotness1 = light_hotness(dist1);
        let hotness2 = light_hotness(dist2);
        
        // Additive blending for overlapping lights (capped)
        total_intensity = min(intensity1 + intensity2, 1.5);
        total_hotness = max(hotness1, hotness2);
    }
    
    // Apply animation intensity multiplier
    total_intensity *= intensity_mult;
    total_hotness *= intensity_mult;
    
    // Return intensity, hotness, and a flag that this glow contributed
    return vec3<f32>(total_intensity, total_hotness, 1.0);
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Early out if no glows
    if params.glow_count == 0u {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    
    // Convert UV to pixel coordinates
    let pixel = vec2<f32>(
        in.uv.x * params.screen_width,
        in.uv.y * params.screen_height
    );
    
    // Accumulate contributions from all active glows
    var total_intensity: f32 = 0.0;
    var total_hotness: f32 = 0.0;
    var accum_color = vec3<f32>(0.0, 0.0, 0.0);
    var color_weight: f32 = 0.0;
    
    for (var i: u32 = 0u; i < params.glow_count && i < MAX_GLOWS; i++) {
        let glow = params.glows[i];
        let result = calculate_glow(pixel, glow);
        let intensity = result.x;
        let hotness = result.y;
        
        // Accumulate intensity and hotness additively
        total_intensity += intensity;
        total_hotness = max(total_hotness, hotness);
        
        // Weight color contribution by intensity
        let base_color = vec3<f32>(glow.color_r, glow.color_g, glow.color_b);
        accum_color += base_color * intensity;
        color_weight += intensity;
    }
    
    // Cap intensity for overlapping glows
    total_intensity = min(total_intensity, 1.5);
    
    // Calculate final base color (weighted average)
    var base_color = vec3<f32>(0.0, 0.0, 0.0);
    if color_weight > 0.001 {
        base_color = accum_color / color_weight;
    }
    
    // Mix between base color and white based on hotness
    // Hot center = white, outer regions = base color
    let white = vec3<f32>(1.0, 1.0, 1.0);
    let final_color = mix(base_color, white, total_hotness * 0.8);
    
    // Final alpha based on intensity, scaled by global glow_intensity setting
    let final_alpha = clamp(total_intensity * 0.9 * params.glow_intensity, 0.0, 1.0);
    
    // Output with premultiplied alpha for proper blending
    return vec4<f32>(final_color * final_alpha, final_alpha);
}
