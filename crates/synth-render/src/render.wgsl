struct SceneUniform {
    view_projection: mat4x4<f32>,
    inverse_view_projection: mat4x4<f32>,
    light_view_projection: mat4x4<f32>,
    roof_color: vec4<f32>,
    wall_color: vec4<f32>,
    ground_color: vec4<f32>,
    occluder_color: vec4<f32>,
    background_color: vec4<f32>,
    light_direction: vec4<f32>,
    camera_position: vec4<f32>,
    lighting: vec4<f32>,
    environment: vec4<f32>,
    color_grading: vec4<f32>,
    atmosphere: vec4<f32>,
    material_roughness: vec4<f32>,
    local_light_positions: array<vec4<f32>, 4>,
    local_light_colors: array<vec4<f32>, 4>,
    material_uv_tiling: array<vec4<f32>, 32>,
    material_layers: array<vec4<u32>, 4>,
    material_patterns: array<vec4<u32>, 4>,
    texture_flags: vec4<u32>,
};

@group(0) @binding(0)
var<uniform> scene: SceneUniform;
@group(0) @binding(1)
var shadow_map: texture_depth_2d;
@group(0) @binding(2)
var shadow_sampler: sampler_comparison;
@group(0) @binding(3)
var material_albedo: texture_2d_array<f32>;
@group(0) @binding(4)
var material_normal: texture_2d_array<f32>;
@group(0) @binding(5)
var material_arm: texture_2d_array<f32>;
@group(0) @binding(6)
var material_sampler: sampler;
@group(0) @binding(7)
var environment_map: texture_2d<f32>;
@group(0) @binding(8)
var environment_sampler: sampler;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) face_coord: vec2<f32>,
    @location(3) semantic_id: u32,
    @location(4) material: u32,
    @location(5) appearance: vec4<f32>,
    @location(6) pattern: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) face_coord: vec2<f32>,
    @location(3) @interpolate(flat) semantic_id: u32,
    @location(4) @interpolate(flat) material: u32,
    @location(5) @interpolate(flat) appearance: vec4<f32>,
    @location(6) @interpolate(flat) pattern: u32,
};

@vertex
fn vertex_main(vertex: VertexInput) -> VertexOutput {
    var output: VertexOutput;
    output.clip_position = scene.view_projection * vec4<f32>(vertex.position, 1.0);
    output.world_position = vertex.position;
    output.normal = vertex.normal;
    output.face_coord = vertex.face_coord;
    output.semantic_id = vertex.semantic_id;
    output.material = vertex.material;
    output.appearance = vertex.appearance;
    output.pattern = vertex.pattern;
    return output;
}

@vertex
fn shadow_vertex_main(vertex: VertexInput) -> @builtin(position) vec4<f32> {
    return scene.light_view_projection * vec4<f32>(vertex.position, 1.0);
}

struct SkyVertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) ndc: vec2<f32>,
};

@vertex
fn sky_vertex_main(@builtin(vertex_index) vertex_index: u32) -> SkyVertexOutput {
    var positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var output: SkyVertexOutput;
    output.ndc = positions[vertex_index];
    output.clip_position = vec4<f32>(output.ndc, 0.999999, 1.0);
    return output;
}

fn hash21(value: vec2<f32>) -> f32 {
    let seeded = value + vec2<f32>(scene.environment.y * 0.013, scene.environment.y * 0.071);
    return fract(sin(dot(seeded, vec2<f32>(127.1, 311.7))) * 43758.5453);
}

fn noise2(value: vec2<f32>) -> f32 {
    let cell = floor(value);
    let local = fract(value);
    let smoothed = local * local * (3.0 - 2.0 * local);
    return mix(
        mix(hash21(cell), hash21(cell + vec2<f32>(1.0, 0.0)), smoothed.x),
        mix(
            hash21(cell + vec2<f32>(0.0, 1.0)),
            hash21(cell + vec2<f32>(1.0, 1.0)),
            smoothed.x,
        ),
        smoothed.y,
    );
}

fn procedural_sky(direction: vec3<f32>) -> vec3<f32> {
    let horizon_amount = pow(1.0 - max(direction.y, 0.0), 2.0);
    let day_zenith = scene.background_color.rgb * vec3<f32>(0.55, 0.72, 1.05);
    let day_horizon = mix(scene.background_color.rgb * 1.25, vec3<f32>(0.78, 0.79, 0.77), scene.lighting.w);
    var day = mix(day_zenith, day_horizon, horizon_amount);
    let light_direction = normalize(scene.light_direction.xyz);
    let sun = pow(max(dot(direction, light_direction), 0.0), 700.0);
    day += vec3<f32>(1.0, 0.72, 0.38) * sun * 5.0;

    let longitude = atan2(direction.z, direction.x) * 0.15915494 + scene.atmosphere.y * 0.15915494;
    let cloud_uv = vec2<f32>(longitude * 18.0, direction.y * 8.0);
    let clouds = smoothstep(0.62, 0.82, noise2(cloud_uv) * 0.65 + noise2(cloud_uv * 2.17) * 0.35);
    day = mix(day, vec3<f32>(0.72, 0.75, 0.77), clouds * scene.lighting.z * 0.8);

    let night_zenith = vec3<f32>(0.004, 0.008, 0.025);
    let developed_horizon = select(1.0, 0.0, scene.texture_flags.w == 2u);
    let night_horizon = mix(vec3<f32>(0.025, 0.035, 0.065), vec3<f32>(0.12, 0.075, 0.045), developed_horizon);
    var night = mix(night_zenith, night_horizon, horizon_amount);
    let star_cell = floor(vec2<f32>(longitude * 1200.0, direction.y * 600.0));
    let cloud_transmission = pow(1.0 - scene.lighting.z, 2.0);
    let stars = select(0.0, pow(hash21(star_cell), 24.0), direction.y > 0.02);
    night += vec3<f32>(0.72, 0.82, 1.0) * stars * 1.5 * cloud_transmission;
    let sampled_light = normalize(scene.light_direction.xyz);
    let moon_direction = normalize(vec3<f32>(
        sampled_light.x,
        max(abs(sampled_light.y), 0.28),
        sampled_light.z,
    ));
    let moon = smoothstep(0.9992, 0.9997, dot(direction, moon_direction));
    night += vec3<f32>(0.78, 0.84, 1.0) * moon * 2.0 * cloud_transmission;
    return mix(day, night, scene.lighting.y);
}

fn apply_color_grade(color: vec3<f32>) -> vec3<f32> {
    let temperature_offset = clamp((6500.0 - scene.color_grading.y) / 5000.0, -1.0, 1.0);
    let warm = max(temperature_offset, 0.0);
    let cool = max(-temperature_offset, 0.0);
    let tint = vec3<f32>(1.0)
        + warm * vec3<f32>(0.12, -0.015, -0.18)
        + cool * vec3<f32>(-0.12, 0.01, 0.17);
    return max(color * tint * exp2(scene.color_grading.x), vec3<f32>(0.0));
}

fn aces_filmic(color: vec3<f32>) -> vec3<f32> {
    // Narkowicz ACES approximation. The render target is sRGB, so WGPU applies
    // the final linear-to-sRGB transfer after this display-linear curve.
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp(
        (color * (a * color + vec3<f32>(b)))
            / (color * (c * color + vec3<f32>(d)) + vec3<f32>(e)),
        vec3<f32>(0.0),
        vec3<f32>(1.0),
    );
}

fn final_display_transform(linear_radiance: vec3<f32>) -> vec3<f32> {
    return aces_filmic(apply_color_grade(linear_radiance));
}

fn environment_radiance(direction: vec3<f32>, roughness: f32) -> vec3<f32> {
    let uv = vec2<f32>(
        fract(atan2(direction.z, direction.x) * 0.15915494 + 0.5 + scene.atmosphere.y * 0.15915494),
        acos(clamp(direction.y, -1.0, 1.0)) * 0.31830989,
    );
    let lod = clamp(roughness, 0.0, 1.0) * scene.material_roughness.w;
    let hdr = textureSampleLevel(environment_map, environment_sampler, uv, lod).rgb;
    return hdr * scene.atmosphere.w * scene.environment.x;
}

@fragment
fn sky_fragment_main(input: SkyVertexOutput) -> @location(0) vec4<f32> {
    let far_clip = vec4<f32>(input.ndc, 1.0, 1.0);
    let world_h = scene.inverse_view_projection * far_clip;
    let world = world_h.xyz / world_h.w;
    let direction = normalize(world - scene.camera_position.xyz);
    var color = procedural_sky(direction);
    if scene.texture_flags.y != 0u {
        let uv = vec2<f32>(
            fract(atan2(direction.z, direction.x) * 0.15915494 + 0.5 + scene.atmosphere.y * 0.15915494),
            acos(clamp(direction.y, -1.0, 1.0)) * 0.31830989,
        );
        let hdr_radiance = textureSampleLevel(environment_map, environment_sampler, uv, 0.0).rgb
            * scene.atmosphere.w
            * scene.environment.x;
        let visibility_haze = 1.0 - clamp(scene.atmosphere.x / 60.0, 0.0, 1.0);
        let categorical_weather = select(
            0.0,
            0.24,
            scene.atmosphere.z == 2.0 || scene.atmosphere.z == 3.0,
        );
        let atmospheric_mix = clamp(
            scene.lighting.z * 0.22
                + scene.lighting.w * 0.32
                + visibility_haze * 0.46
                + categorical_weather,
            0.0,
            0.82,
        );
        color = mix(hdr_radiance, procedural_sky(direction), atmospheric_mix);
    }
    return vec4<f32>(final_display_transform(color), 1.0);
}

struct MaterialSurface {
    albedo: vec3<f32>,
    normal: vec3<f32>,
    roughness: f32,
    metallic: f32,
    ambient_occlusion: f32,
    emissive: vec3<f32>,
};

fn logical_base_color(material: u32) -> vec3<f32> {
    if material == 0u { return scene.roof_color.rgb; }
    if material == 1u { return scene.wall_color.rgb; }
    if material == 2u { return scene.ground_color.rgb; }
    if material == 3u { return scene.occluder_color.rgb; }
    if material == 4u { return vec3<f32>(0.035, 0.075, 0.105); }
    if material == 5u { return vec3<f32>(0.72, 0.69, 0.62); }
    if material == 6u { return vec3<f32>(0.78, 0.025, 0.018); }
    if material == 7u { return vec3<f32>(0.055, 0.06, 0.065); }
    if material == 8u { return vec3<f32>(0.045, 0.19, 0.035); }
    if material == 9u { return vec3<f32>(0.15, 0.075, 0.035); }
    if material == 10u { return vec3<f32>(0.28, 0.3, 0.31); }
    if material == 11u { return vec3<f32>(0.22, 0.24, 0.25); }
    if material == 12u { return vec3<f32>(0.82, 0.78, 0.58); }
    if material == 13u { return vec3<f32>(0.42, 0.43, 0.41); }
    if material == 14u { return scene.wall_color.rgb * 0.72; }
    return vec3<f32>(0.035, 0.22, 0.42);
}

fn physical_layer(material: u32) -> u32 {
    let row = material / 4u;
    let column = material % 4u;
    return scene.material_layers[row][column];
}

fn selected_pattern(material: u32, vertex_pattern: u32) -> u32 {
    if vertex_pattern != 0u { return vertex_pattern; }
    let row = material / 4u;
    let column = material % 4u;
    return scene.material_patterns[row][column];
}

fn procedural_surface(
    material: u32,
    uv: vec2<f32>,
    world_position: vec3<f32>,
    appearance: vec4<f32>,
    pattern: u32,
) -> MaterialSurface {
    var surface: MaterialSurface;
    surface.albedo = logical_base_color(material);
    surface.normal = vec3<f32>(0.0);
    surface.roughness = 0.72;
    surface.metallic = 0.0;
    surface.ambient_occlusion = 1.0;
    surface.emissive = vec3<f32>(0.0);
    if max(max(appearance.r, appearance.g), appearance.b) > 0.0001 {
        surface.albedo = appearance.rgb;
    }
    let variation = noise2(world_position.xz * 0.37 + uv * 3.1);

    if material == 0u {
        let seam = smoothstep(0.035, 0.0, abs(fract(uv.x * 7.0) - 0.5) - 0.47);
        let weather = mix(0.82, 1.08, variation);
        surface.albedo *= weather * (1.0 - seam * 0.18);
        surface.roughness = scene.material_roughness.x;
    } else if material == 1u {
        surface.albedo *= mix(0.82, 1.08, variation);
        if pattern == 2u {
            let brick_uv = vec2<f32>(uv.x * 12.0 + floor(uv.y * 9.0) * 0.5, uv.y * 9.0);
            let mortar = select(0.0, 1.0, fract(brick_uv.x) < 0.055 || fract(brick_uv.y) < 0.07);
            surface.albedo = mix(surface.albedo, vec3<f32>(0.34), mortar * 0.38);
        } else if pattern == 3u {
            let panel_seam = select(0.0, 1.0, fract(uv.x * 18.0) < 0.035);
            surface.albedo *= mix(1.0, 0.66, panel_seam);
        }
        let vertical_stain = smoothstep(0.15, 0.95, noise2(vec2<f32>(world_position.x * 0.18, world_position.y * 0.42)));
        surface.albedo *= mix(1.0, mix(0.9, 0.58, vertical_stain), appearance.a);
        surface.roughness = scene.material_roughness.y;
    } else if material == 2u {
        if pattern == 5u {
            surface.albedo = scene.ground_color.rgb * mix(0.78, 1.16, variation);
            surface.roughness = scene.material_roughness.z;
        } else {
            let grass = mix(vec3<f32>(0.055, 0.11, 0.035), scene.ground_color.rgb, 0.55);
            surface.albedo = grass * mix(0.72, 1.2, variation);
            surface.roughness = scene.material_roughness.z;
        }
    } else if material == 4u {
        let panel = floor(uv.x * 12.0);
        let lit = select(0.0, 1.0, hash21(vec2<f32>(panel, floor(world_position.y * 0.5))) > 0.38);
        surface.albedo *= mix(0.75, 1.35, variation);
        surface.roughness = select(0.16, appearance.a, appearance.a > 0.0);
        surface.metallic = 0.18;
        if pattern == 6u {
            surface.emissive = vec3<f32>(1.0, 0.62, 0.25) * lit * scene.lighting.y * 1.15;
        }
    } else if material == 6u {
        let border = select(0.0, 1.0, uv.x < 0.055 || uv.x > 0.945 || uv.y < 0.08 || uv.y > 0.92);
        let hat_brim = select(0.0, 1.0, uv.y > 0.57 && uv.y < 0.66 && uv.x > 0.17 && uv.x < 0.83);
        let hat_crown = select(
            0.0,
            1.0,
            uv.y >= 0.66 && uv.y < 0.85
                && abs(uv.x - 0.5) < mix(0.13, 0.31, (0.85 - uv.y) / 0.19),
        );
        let letter_band = select(0.0, 1.0, uv.y > 0.24 && uv.y < 0.46 && fract(uv.x * 8.0) > 0.28);
        if pattern == 9u {
            let graphic = max(max(hat_brim, hat_crown), letter_band * 0.82);
            surface.albedo = mix(surface.albedo * 0.72, vec3<f32>(0.96, 0.78, 0.45), max(border, graphic));
            surface.emissive = surface.albedo * appearance.a * scene.lighting.y;
        } else {
            surface.albedo = mix(vec3<f32>(0.09, 0.11, 0.13), vec3<f32>(0.72), border * 0.35);
        }
        surface.roughness = 0.34;
    } else if material == 7u {
        let aggregate = mix(0.72, 1.22, noise2(world_position.xz * 3.7));
        surface.albedo *= aggregate;
        surface.roughness = 0.94;
    } else if material == 8u {
        surface.albedo *= mix(0.62, 1.38, noise2(world_position.xz * 2.1 + world_position.yy));
        surface.roughness = 0.9;
    } else if material == 10u {
        let streak = smoothstep(0.42, 0.9, noise2(vec2<f32>(world_position.x * 0.12, world_position.y * 0.31)));
        surface.albedo *= mix(0.96, 0.82, streak);
        surface.roughness = select(0.72, appearance.a, appearance.a > 0.0);
        if pattern == 2u || pattern == 11u {
            let brick_uv = vec2<f32>(uv.x * 12.0 + floor(uv.y * 9.0) * 0.5, uv.y * 9.0);
            let mortar = select(0.0, 1.0, fract(brick_uv.x) < 0.055 || fract(brick_uv.y) < 0.07);
            surface.albedo = mix(surface.albedo, vec3<f32>(0.34), mortar * 0.34);
        } else if pattern == 3u || pattern == 12u {
            let seam = select(0.0, 1.0, fract(uv.x * 18.0) < 0.035);
            surface.albedo *= mix(1.0, 0.66, seam);
        }
        let grid = floor(vec2<f32>(uv.x * 8.0, uv.y * 12.0));
        let window_x = fract(uv.x * 8.0);
        let window_y = fract(uv.y * 12.0);
        let is_window = window_x > 0.16 && window_x < 0.84 && window_y > 0.14 && window_y < 0.78;
        let lit = hash21(grid) > 0.42;
        if is_window && (pattern == 8u || pattern == 11u || pattern == 12u) {
            surface.albedo = vec3<f32>(0.025, 0.045, 0.07);
            surface.roughness = 0.2;
            surface.emissive = vec3<f32>(1.0, 0.67, 0.3) * select(0.0, 1.0, lit) * scene.lighting.y * 1.4;
        }
    } else if material == 11u {
        surface.albedo *= mix(0.78, 1.17, variation);
        surface.roughness = 0.48;
        surface.metallic = 0.62;
    } else if material == 12u {
        surface.emissive = surface.albedo * scene.lighting.y * 0.15;
        surface.roughness = 0.8;
    } else if material == 14u {
        let outline = select(0.0, 1.0, uv.x < 0.075 || uv.x > 0.925 || uv.y < 0.11 || uv.y > 0.89);
        let fixing = select(
            0.0,
            1.0,
            distance(uv, vec2<f32>(0.22, 0.5)) < 0.035
                || distance(uv, vec2<f32>(0.5, 0.5)) < 0.035
                || distance(uv, vec2<f32>(0.78, 0.5)) < 0.035,
        );
        surface.albedo = mix(
            scene.wall_color.rgb * mix(0.92, 0.62, appearance.a),
            vec3<f32>(0.4),
            max(outline * 0.38, fixing * 0.75),
        );
        surface.roughness = 0.92;
    } else if material == 15u {
        let stripe = select(0.0, 1.0, fract((uv.x + uv.y * 0.35) * 6.0) > 0.52);
        let letter_bar = select(0.0, 1.0, uv.y > 0.3 && uv.y < 0.68 && fract(uv.x * 9.0) > 0.3);
        let base = select(vec3<f32>(0.025, 0.18, 0.38), appearance.rgb, max(max(appearance.r, appearance.g), appearance.b) > 0.0001);
        surface.albedo = mix(base, vec3<f32>(0.05, 0.48, 0.3), stripe * 0.35);
        surface.albedo = mix(surface.albedo, vec3<f32>(0.82, 0.9, 0.75), letter_bar * 0.55);
        surface.emissive = surface.albedo * appearance.a * scene.lighting.y;
        surface.roughness = 0.38;
    }
    return surface;
}

fn textured_surface(material: u32, uv: vec2<f32>, normal: vec3<f32>, procedural: MaterialSurface) -> MaterialSurface {
    let layer = physical_layer(material);
    if layer >= 32u || (scene.texture_flags.x & (1u << layer)) == 0u {
        return procedural;
    }
    let tiled_uv = uv * scene.material_uv_tiling[layer].xy;
    let albedo_sample = textureSample(material_albedo, material_sampler, tiled_uv, i32(layer));
    let normal_sample = textureSample(material_normal, material_sampler, tiled_uv, i32(layer)).xyz * 2.0 - 1.0;
    let arm_sample = textureSample(material_arm, material_sampler, tiled_uv, i32(layer));
    let reference = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(normal.y) > 0.95);
    let tangent = normalize(cross(reference, normal));
    let bitangent = normalize(cross(normal, tangent));

    var surface = procedural;
    let texture_luminance = max(dot(albedo_sample.rgb, vec3<f32>(0.2126, 0.7152, 0.0722)), 0.12);
    let texture_detail = clamp(albedo_sample.rgb / texture_luminance, vec3<f32>(0.3), vec3<f32>(2.2));
    let detail_strength = select(0.72, 0.38, material == 0u);
    surface.albedo = procedural.albedo * mix(vec3<f32>(1.0), texture_detail, detail_strength);
    surface.ambient_occlusion = arm_sample.r;
    surface.roughness = clamp(surface.roughness + (arm_sample.g - 0.5) * 0.45, 0.04, 1.0);
    surface.metallic = arm_sample.b;
    surface.normal = normalize(tangent * normal_sample.x + bitangent * normal_sample.y + normal * normal_sample.z);
    return surface;
}

fn apply_ground_wetness(material: u32, surface: MaterialSurface) -> MaterialSurface {
    var wet = surface;
    if material == 2u || material == 7u || material == 13u {
        wet.albedo *= mix(1.0, 0.58, scene.environment.w);
        wet.roughness = mix(wet.roughness, 0.16, scene.environment.w);
    }
    return wet;
}

fn shadow_visibility(world_position: vec3<f32>, normal: vec3<f32>) -> f32 {
    let clip = scene.light_view_projection * vec4<f32>(world_position, 1.0);
    let ndc = clip.xyz / clip.w;
    let uv = vec2<f32>(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5);
    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) || ndc.z < 0.0 || ndc.z > 1.0 {
        return 1.0;
    }
    let bias = max(0.00035 * (1.0 - dot(normal, normalize(scene.light_direction.xyz))), 0.00008);
    let dimensions = vec2<f32>(textureDimensions(shadow_map));
    let radius = mix(0.25, 3.5, scene.environment.z);
    var visibility = 0.0;
    for (var offset_y = -1; offset_y <= 1; offset_y += 1) {
        for (var offset_x = -1; offset_x <= 1; offset_x += 1) {
            let offset = vec2<f32>(f32(offset_x), f32(offset_y)) * radius / dimensions;
            visibility += textureSampleCompare(shadow_map, shadow_sampler, uv + offset, ndc.z - bias);
        }
    }
    return visibility / 9.0;
}

fn local_illumination(world_position: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
    var illumination = vec3<f32>(0.0);
    for (var index = 0u; index < 4u; index += 1u) {
        if index >= scene.texture_flags.z { break; }
        let light = scene.local_light_positions[index];
        let delta = light.xyz - world_position;
        let distance = length(delta);
        let attenuation = pow(max(1.0 - distance / light.w, 0.0), 2.0);
        let diffuse = max(dot(normal, normalize(delta)), 0.0);
        illumination += scene.local_light_colors[index].rgb
            * scene.local_light_colors[index].w
            * attenuation
            * (0.2 + diffuse * 0.8);
    }
    return illumination * scene.lighting.y;
}

struct FragmentOutput {
    @location(0) color: vec4<f32>,
    @location(1) semantic_id: u32,
    @location(2) face_coordinates: vec4<f32>,
};

@fragment
fn fragment_main(input: VertexOutput) -> FragmentOutput {
    let geometric_normal = normalize(input.normal);
    var surface = procedural_surface(
        input.material,
        input.face_coord,
        input.world_position,
        input.appearance,
        selected_pattern(input.material, input.pattern),
    );
    surface = textured_surface(input.material, input.face_coord, geometric_normal, surface);
    surface = apply_ground_wetness(input.material, surface);
    let shading_normal = select(
        geometric_normal,
        normalize(surface.normal),
        dot(surface.normal, surface.normal) > 0.25,
    );
    let light_direction = normalize(scene.light_direction.xyz);
    let view_direction = normalize(scene.camera_position.xyz - input.world_position);
    let half_direction = normalize(light_direction + view_direction);
    let diffuse = max(dot(shading_normal, light_direction), 0.0);
    let shadow = shadow_visibility(input.world_position, shading_normal);
    let direct_scale = scene.light_direction.w;
    let ambient = scene.lighting.x;
    let specular_power = mix(6.0, 96.0, 1.0 - surface.roughness);
    let specular = pow(max(dot(shading_normal, half_direction), 0.0), specular_power)
        * mix(0.035, 0.32, surface.metallic)
        * shadow
        * direct_scale;
    let directional = diffuse * shadow * direct_scale;
    let local = local_illumination(input.world_position, shading_normal);
    var color = surface.albedo
        * surface.ambient_occlusion
        * (ambient + directional);
    color += surface.albedo * local + vec3<f32>(specular) + surface.emissive;
    if scene.texture_flags.y != 0u {
        let reflected = reflect(-view_direction, shading_normal);
        let reflection = environment_radiance(reflected, surface.roughness);
        let diffuse_environment = environment_radiance(shading_normal, 1.0);
        let fresnel = pow(1.0 - max(dot(shading_normal, view_direction), 0.0), 5.0);
        let glass_weight = select(0.0, 0.52, input.material == 4u);
        let wet_weight = select(
            0.0,
            scene.environment.w * 0.38,
            input.material == 2u || input.material == 7u || input.material == 13u,
        );
        let reflective_weight = clamp(
            surface.metallic * 0.42
                + (1.0 - surface.roughness) * (0.08 + fresnel * 0.18)
                + glass_weight
                + wet_weight,
            0.0,
            0.72,
        );
        color = mix(color, reflection, reflective_weight);
        color += diffuse_environment * surface.albedo * surface.ambient_occlusion * 0.055;
    }

    let view_distance = distance(scene.camera_position.xyz, input.world_position);
    let visibility_m = max(scene.atmosphere.x * 1000.0, 1.0);
    let extinction = 3.912 / visibility_m + scene.lighting.w * 0.004;
    let transmittance = exp(-view_distance * extinction);
    let horizontal_view = normalize(vec3<f32>(-view_direction.x, 0.08, -view_direction.z));
    var aerial_light = procedural_sky(horizontal_view);
    if scene.texture_flags.y != 0u {
        aerial_light = environment_radiance(horizontal_view, 1.0) * 0.12;
    }
    color = mix(aerial_light, color, transmittance);

    var output: FragmentOutput;
    output.color = vec4<f32>(final_display_transform(color), 1.0);
    output.semantic_id = input.semantic_id;
    output.face_coordinates = select(
        vec4<f32>(0.0),
        vec4<f32>(input.face_coord, f32(input.semantic_id), 1.0),
        input.semantic_id != 0u,
    );
    return output;
}

@fragment
fn amodal_fragment_main(input: VertexOutput) -> @location(0) u32 {
    if input.semantic_id == 0u {
        discard;
    }
    return input.semantic_id;
}
