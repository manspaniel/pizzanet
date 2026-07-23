use serde::{Deserialize, Serialize};

use crate::{CompositionSamplingConfig, FloatRange, U32Range};

/// Complete, serializable input to deterministic scene sampling.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GeneratorConfig {
    /// Output image dimensions.
    pub image: ImageConfig,
    /// Sequence length and cadence.
    pub sequence: SequenceConfig,
    /// Building shell ranges.
    pub scene: SceneSamplingConfig,
    /// Parametric roof ranges.
    pub roof: RoofSamplingConfig,
    /// Camera-path ranges.
    pub camera: CameraSamplingConfig,
    /// Roof and wall material distributions.
    pub materials: MaterialSamplingConfig,
    /// Sun, sky, and cloud ranges.
    pub lighting: LightingSamplingConfig,
    /// Foreground and rooftop clutter distribution.
    pub occluders: OccluderSamplingConfig,
    /// Site, weather, façade, signage, background, and vegetation composition.
    pub composition: CompositionSamplingConfig,
}

/// Render-target dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImageConfig {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl Default for ImageConfig {
    fn default() -> Self {
        Self {
            width: 640,
            height: 480,
        }
    }
}

/// Coherent sequence settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequenceConfig {
    /// Number of camera frames generated around one building.
    pub frame_count: u32,
    /// Nominal elapsed time between frames.
    pub frame_interval_ms: u32,
}

impl Default for SequenceConfig {
    fn default() -> Self {
        Self {
            frame_count: 24,
            frame_interval_ms: 100,
        }
    }
}

/// Building shell ranges, in metres.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneSamplingConfig {
    /// Exterior wall-to-wall width.
    pub footprint_width_m: FloatRange,
    /// Exterior wall-to-wall depth.
    pub footprint_depth_m: FloatRange,
    /// Wall height up to the eave.
    pub wall_height_m: FloatRange,
    /// Minimum half-extent of generated ground around the building. Sampling
    /// grows it when necessary to contain the complete scene and camera path.
    pub ground_half_extent_m: FloatRange,
}

impl Default for SceneSamplingConfig {
    fn default() -> Self {
        Self {
            footprint_width_m: FloatRange::new(18.0, 31.0),
            footprint_depth_m: FloatRange::new(14.0, 25.0),
            wall_height_m: FloatRange::new(3.3, 5.2),
            ground_half_extent_m: FloatRange::new(35.0, 65.0),
        }
    }
}

/// Recognizable proportion families found across former and surviving huts.
///
/// The category is deliberately architectural rather than chronological: it
/// describes where the two roof stages break and how strongly the crown rises.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum RoofMorphology {
    /// A compact footprint with an early break into a conspicuously tall crown.
    TallEarlyCrown,
    /// A near-square footprint with a steep skirt and especially tall crown.
    NearSquareTall,
    /// The common, evenly proportioned two-stage silhouette.
    #[default]
    BalancedClassic,
    /// A broad footprint whose lower roof carries farther inward to a low crown.
    LowWideLate,
    /// A shallow, broad-crowned silhouette associated with later remodelling.
    ShallowRemodelled,
}

/// Correlated dimensional envelope for one roof morphology.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoofMorphologyProfile {
    /// Stable morphology category persisted into each sampled scene.
    pub morphology: RoofMorphology,
    /// Relative integer selection weight.
    pub weight: u32,
    /// Building width divided by building depth.
    pub footprint_aspect_ratio: FloatRange,
    /// Eave extension beyond each wall.
    pub overhang_m: FloatRange,
    /// Upper shoulder width divided by eave width.
    pub shoulder_width_fraction: FloatRange,
    /// Upper shoulder depth divided by eave depth.
    pub shoulder_depth_fraction: FloatRange,
    /// Rise from the eave to the shoulder.
    pub lower_rise_m: FloatRange,
    /// Rise from the crown base/shoulder to its flat top.
    pub upper_rise_m: FloatRange,
    /// Crown-top width divided by crown-base/shoulder width.
    pub crown_top_width_fraction: FloatRange,
    /// Crown-top depth divided by crown-base/shoulder depth.
    pub crown_top_depth_fraction: FloatRange,
}

/// Ranges for the recognizable two-stage roof silhouette.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct RoofSamplingConfig {
    /// Eave extension beyond each wall.
    pub overhang_m: FloatRange,
    /// Upper shoulder width divided by eave width.
    pub shoulder_width_fraction: FloatRange,
    /// Upper shoulder depth divided by eave depth.
    pub shoulder_depth_fraction: FloatRange,
    /// Rise from the eave to the shoulder.
    pub lower_rise_m: FloatRange,
    /// Rise from the crown base/shoulder to its flat top.
    pub upper_rise_m: FloatRange,
    /// Crown-top width divided by crown-base/shoulder width.
    pub crown_top_width_fraction: FloatRange,
    /// Crown-top depth divided by crown-base/shoulder depth.
    pub crown_top_depth_fraction: FloatRange,
    /// Small, bounded left/right dimensional perturbation.
    pub asymmetry_fraction: FloatRange,
    /// Weighted, correlated silhouette families. Each profile is intersected
    /// with the global roof and scene bounds before sampling.
    pub profiles: Vec<RoofMorphologyProfile>,
}

impl Default for RoofSamplingConfig {
    fn default() -> Self {
        Self {
            overhang_m: FloatRange::new(0.75, 2.5),
            shoulder_width_fraction: FloatRange::new(0.40, 0.68),
            shoulder_depth_fraction: FloatRange::new(0.36, 0.62),
            lower_rise_m: FloatRange::new(1.35, 4.1),
            upper_rise_m: FloatRange::new(1.15, 4.6),
            crown_top_width_fraction: FloatRange::new(0.72, 0.97),
            crown_top_depth_fraction: FloatRange::new(0.62, 0.94),
            asymmetry_fraction: FloatRange::new(-0.025, 0.025),
            profiles: vec![
                RoofMorphologyProfile {
                    morphology: RoofMorphology::TallEarlyCrown,
                    weight: 24,
                    footprint_aspect_ratio: FloatRange::new(1.05, 1.34),
                    overhang_m: FloatRange::new(1.35, 2.4),
                    shoulder_width_fraction: FloatRange::new(0.44, 0.53),
                    shoulder_depth_fraction: FloatRange::new(0.36, 0.47),
                    lower_rise_m: FloatRange::new(2.6, 3.8),
                    upper_rise_m: FloatRange::new(3.25, 4.2),
                    crown_top_width_fraction: FloatRange::new(0.78, 0.89),
                    crown_top_depth_fraction: FloatRange::new(0.68, 0.82),
                },
                RoofMorphologyProfile {
                    morphology: RoofMorphology::NearSquareTall,
                    weight: 11,
                    footprint_aspect_ratio: FloatRange::new(0.92, 1.18),
                    overhang_m: FloatRange::new(1.2, 2.5),
                    shoulder_width_fraction: FloatRange::new(0.40, 0.52),
                    shoulder_depth_fraction: FloatRange::new(0.40, 0.54),
                    lower_rise_m: FloatRange::new(2.7, 4.1),
                    upper_rise_m: FloatRange::new(3.3, 4.6),
                    crown_top_width_fraction: FloatRange::new(0.72, 0.88),
                    crown_top_depth_fraction: FloatRange::new(0.65, 0.84),
                },
                RoofMorphologyProfile {
                    morphology: RoofMorphology::BalancedClassic,
                    weight: 38,
                    footprint_aspect_ratio: FloatRange::new(1.18, 1.58),
                    overhang_m: FloatRange::new(1.2, 2.15),
                    shoulder_width_fraction: FloatRange::new(0.48, 0.58),
                    shoulder_depth_fraction: FloatRange::new(0.4, 0.51),
                    lower_rise_m: FloatRange::new(2.4, 3.5),
                    upper_rise_m: FloatRange::new(2.55, 3.45),
                    crown_top_width_fraction: FloatRange::new(0.81, 0.92),
                    crown_top_depth_fraction: FloatRange::new(0.68, 0.84),
                },
                RoofMorphologyProfile {
                    morphology: RoofMorphology::LowWideLate,
                    weight: 18,
                    footprint_aspect_ratio: FloatRange::new(1.42, 1.9),
                    overhang_m: FloatRange::new(1.0, 1.8),
                    shoulder_width_fraction: FloatRange::new(0.54, 0.62),
                    shoulder_depth_fraction: FloatRange::new(0.46, 0.54),
                    lower_rise_m: FloatRange::new(2.2, 3.0),
                    upper_rise_m: FloatRange::new(2.2, 2.85),
                    crown_top_width_fraction: FloatRange::new(0.85, 0.94),
                    crown_top_depth_fraction: FloatRange::new(0.74, 0.86),
                },
                RoofMorphologyProfile {
                    morphology: RoofMorphology::ShallowRemodelled,
                    weight: 9,
                    footprint_aspect_ratio: FloatRange::new(1.15, 1.75),
                    overhang_m: FloatRange::new(0.75, 1.7),
                    shoulder_width_fraction: FloatRange::new(0.52, 0.68),
                    shoulder_depth_fraction: FloatRange::new(0.44, 0.62),
                    lower_rise_m: FloatRange::new(1.35, 2.55),
                    upper_rise_m: FloatRange::new(1.15, 2.35),
                    crown_top_width_fraction: FloatRange::new(0.82, 0.97),
                    crown_top_depth_fraction: FloatRange::new(0.74, 0.94),
                },
            ],
        }
    }
}

/// Camera placement and lens distributions.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CameraSamplingConfig {
    /// Horizontal distance from the building origin, in metres.
    pub distance_m: FloatRange,
    /// Camera height above the ground, in metres.
    pub height_m: FloatRange,
    /// Signed orbit arc covered by a sequence, in degrees.
    pub sweep_degrees: FloatRange,
    /// Relative radial movement over the orbit.
    pub radial_motion_fraction: FloatRange,
    /// Horizontal field of view, in degrees.
    pub horizontal_fov_degrees: FloatRange,
    /// Vertical point-of-interest offset above the wall eave, in metres.
    pub target_above_eave_m: FloatRange,
    /// Relative orbit-path selection weight.
    pub orbit_weight: u32,
    /// Relative lateral-walk selection weight.
    pub lateral_walk_weight: u32,
    /// Relative approach-arc selection weight.
    pub approach_arc_weight: u32,
    /// Relative corner-reveal selection weight.
    pub corner_reveal_weight: u32,
    /// Probability of a smooth focal-length change over the sequence.
    pub zoom_probability: f32,
    /// End/start horizontal-FOV ratio for zoomed sequences.
    pub zoom_ratio: FloatRange,
    /// Probability of deliberately cropping part of the roof.
    pub partial_crop_probability: f32,
    /// Desired target-width fraction for ordinary views.
    pub target_width_fraction: FloatRange,
    /// Desired target-width fraction for distant views.
    pub distant_target_width_fraction: FloatRange,
    /// Desired target-width fraction for close, but non-truncated, views.
    pub close_target_width_fraction: FloatRange,
    /// Relative selection weight for distant views.
    pub distant_view_weight: u32,
    /// Relative selection weight for ordinary views.
    pub normal_view_weight: u32,
    /// Relative selection weight for close views.
    pub close_view_weight: u32,
    /// Desired target-width fraction for deliberate partial crops; values over one crop.
    pub partial_target_width_fraction: FloatRange,
    /// Desired crop depth as a fraction of the roof's projected image span.
    pub framing_offset_fraction: FloatRange,
    /// Smooth handheld positional sway amplitude in metres.
    pub handheld_sway_m: FloatRange,
}

impl Default for CameraSamplingConfig {
    fn default() -> Self {
        Self {
            distance_m: FloatRange::new(8.0, 180.0),
            height_m: FloatRange::new(1.35, 2.0),
            sweep_degrees: FloatRange::new(24.0, 95.0),
            radial_motion_fraction: FloatRange::new(-0.18, 0.18),
            horizontal_fov_degrees: FloatRange::new(52.0, 76.0),
            target_above_eave_m: FloatRange::new(1.2, 3.5),
            orbit_weight: 38,
            lateral_walk_weight: 26,
            approach_arc_weight: 18,
            corner_reveal_weight: 18,
            zoom_probability: 0.34,
            zoom_ratio: FloatRange::new(0.72, 1.28),
            partial_crop_probability: 0.08,
            target_width_fraction: FloatRange::new(0.30, 0.70),
            distant_target_width_fraction: FloatRange::new(0.15, 0.30),
            close_target_width_fraction: FloatRange::new(0.68, 0.84),
            distant_view_weight: 20,
            normal_view_weight: 60,
            close_view_weight: 20,
            partial_target_width_fraction: FloatRange::new(0.90, 1.08),
            framing_offset_fraction: FloatRange::new(0.10, 0.22),
            handheld_sway_m: FloatRange::new(0.0, 0.075),
        }
    }
}

/// Weighted material entry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialChoice {
    /// Stable palette identifier recorded in generated samples.
    pub id: String,
    /// Relative integer selection weight.
    pub weight: u32,
    /// Base color in nonlinear sRGB, each channel in `[0, 1]`.
    pub base_color_srgb: [f32; 3],
    /// Maximum independent fractional sRGB-channel variation per instance.
    ///
    /// Missing values decode as zero so legacy configurations retain their
    /// exact fixed-colour behaviour.
    #[serde(default)]
    pub base_color_variation: f32,
    /// Physically based roughness range.
    pub roughness: FloatRange,
    /// Amount of fading, staining, and patch variation.
    pub weathering: FloatRange,
}

/// Material distributions for a generated building.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialSamplingConfig {
    /// Roof surface palette.
    pub roof: Vec<MaterialChoice>,
    /// Wall surface palette.
    pub walls: Vec<MaterialChoice>,
}

impl Default for MaterialSamplingConfig {
    fn default() -> Self {
        let finish = |id: &str, weight, color, base_color_variation| MaterialChoice {
            id: id.to_owned(),
            weight,
            base_color_srgb: color,
            base_color_variation,
            roughness: FloatRange::new(0.48, 0.92),
            weathering: FloatRange::new(0.0, 0.8),
        };
        Self {
            roof: vec![
                finish("original_red", 24, [0.58, 0.055, 0.035], 0.14),
                finish("faded_red", 18, [0.46, 0.10, 0.075], 0.14),
                finish("terracotta_orange", 9, [0.68, 0.24, 0.08], 0.12),
                finish("weathered_tan_brown", 7, [0.40, 0.27, 0.15], 0.12),
                finish("light_metal", 8, [0.72, 0.74, 0.72], 0.08),
                finish("repainted_neutral", 34, [0.37, 0.38, 0.37], 0.12),
                finish("repainted_dark", 24, [0.12, 0.15, 0.17], 0.12),
                finish("repainted_blue", 13, [0.08, 0.22, 0.34], 0.14),
                finish("repainted_green", 11, [0.10, 0.25, 0.16], 0.14),
            ],
            walls: vec![
                finish("warm_brick", 28, [0.39, 0.18, 0.11], 0.10),
                finish("painted_render", 30, [0.68, 0.64, 0.54], 0.08),
                finish("neutral_cladding", 18, [0.43, 0.44, 0.42], 0.08),
                finish("dark_repaint", 10, [0.12, 0.14, 0.15], 0.10),
                finish("blue_repaint", 8, [0.12, 0.27, 0.38], 0.10),
                finish("green_repaint", 6, [0.14, 0.31, 0.20], 0.10),
            ],
        }
    }
}

/// Outdoor illumination ranges.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LightingSamplingConfig {
    /// Sun elevation above the horizon, in degrees.
    pub sun_elevation_degrees: FloatRange,
    /// Direct-light intensity in arbitrary renderer units.
    pub sun_intensity: FloatRange,
    /// Indirect sky-light intensity in arbitrary renderer units.
    pub sky_intensity: FloatRange,
    /// Cloud coverage in `[0, 1]`.
    pub cloud_coverage: FloatRange,
    /// Atmospheric haze in `[0, 1]`.
    pub haze: FloatRange,
}

impl Default for LightingSamplingConfig {
    fn default() -> Self {
        Self {
            sun_elevation_degrees: FloatRange::new(-20.0, 78.0),
            sun_intensity: FloatRange::new(0.35, 6.0),
            sky_intensity: FloatRange::new(0.15, 1.4),
            cloud_coverage: FloatRange::new(0.0, 1.0),
            haze: FloatRange::new(0.0, 0.75),
        }
    }
}

/// Coarse occluder category. Asset selection belongs to the renderer/scene crate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OccluderKind {
    /// Tree or large bush.
    Vegetation,
    /// Parked or moving vehicle.
    Vehicle,
    /// Pole, mast, or street light.
    Pole,
    /// Road or tenant sign.
    Sign,
    /// Adjacent built structure.
    Building,
    /// Person in the foreground.
    Pedestrian,
    /// Roof-mounted plant or duct.
    ///
    /// Retained for backwards-compatible record decoding. New datasets do not
    /// sample this kind because credible placement differs between roof
    /// topologies and would otherwise leak the target class.
    RooftopEquipment,
}

/// Weighted occluder-category entry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OccluderChoice {
    /// Coarse renderer-independent category.
    pub kind: OccluderKind,
    /// Relative integer selection weight.
    pub weight: u32,
}

/// Distribution of independent scene occluders.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct OccluderSamplingConfig {
    /// Number of instances in a scene.
    pub count: U32Range,
    /// Weighted categories.
    pub choices: Vec<OccluderChoice>,
    /// Radial placement from the building centre, in metres.
    pub distance_m: FloatRange,
    /// Uniform instance scale multiplier.
    pub scale: FloatRange,
    /// Probability that a non-rooftop instance is staged between camera and target.
    pub foreground_probability: f32,
    /// Fraction along the camera-to-target segment for foreground placement.
    pub foreground_depth_fraction: FloatRange,
    /// Absolute lateral offset from the sightline in metres.
    pub foreground_lateral_offset_m: FloatRange,
}

impl Default for OccluderSamplingConfig {
    fn default() -> Self {
        Self {
            count: U32Range::new(0, 5),
            choices: vec![
                OccluderChoice {
                    kind: OccluderKind::Vegetation,
                    weight: 18,
                },
                OccluderChoice {
                    kind: OccluderKind::Vehicle,
                    weight: 30,
                },
                OccluderChoice {
                    kind: OccluderKind::Pole,
                    weight: 12,
                },
                OccluderChoice {
                    kind: OccluderKind::Sign,
                    weight: 10,
                },
                OccluderChoice {
                    kind: OccluderKind::Building,
                    weight: 5,
                },
                OccluderChoice {
                    kind: OccluderKind::Pedestrian,
                    weight: 12,
                },
            ],
            distance_m: FloatRange::new(4.0, 34.0),
            scale: FloatRange::new(0.65, 1.25),
            foreground_probability: 0.20,
            foreground_depth_fraction: FloatRange::new(0.18, 0.72),
            foreground_lateral_offset_m: FloatRange::new(1.25, 7.5),
        }
    }
}
