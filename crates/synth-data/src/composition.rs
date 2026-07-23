//! Renderer-independent plans for realistic sites surrounding the target roof.

use serde::{Deserialize, Serialize};

use crate::{FloatRange, MaterialChoice, SampledMaterial, U32Range, Vec2, Vec3};

/// Categorical capture-time regime with correlated sky and camera exposure.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DayPhase {
    /// Sun clearly above the horizon.
    Day,
    /// Dawn or dusk transition with mixed natural and artificial light.
    Twilight,
    /// Sun below astronomical usefulness; site lighting dominates.
    Night,
}

/// Correlated sampling ranges for one capture-time regime.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DayPhaseProfile {
    /// Capture-time category.
    pub phase: DayPhase,
    /// Relative integer selection weight.
    pub weight: u32,
    /// Solar elevation in degrees.
    pub sun_elevation_degrees: FloatRange,
    /// Camera exposure in EV100.
    pub camera_exposure_ev100: FloatRange,
    /// Artificial site-light strength in renderer units.
    pub artificial_light_strength: FloatRange,
}

/// Weighted day, twilight, and night regimes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DayPhaseSamplingConfig {
    /// Available correlated time-of-day profiles.
    pub profiles: Vec<DayPhaseProfile>,
}

impl Default for DayPhaseSamplingConfig {
    fn default() -> Self {
        Self {
            profiles: vec![
                DayPhaseProfile {
                    phase: DayPhase::Day,
                    weight: 55,
                    sun_elevation_degrees: FloatRange::new(10.0, 78.0),
                    camera_exposure_ev100: FloatRange::new(10.5, 15.2),
                    artificial_light_strength: FloatRange::new(0.0, 0.12),
                },
                DayPhaseProfile {
                    phase: DayPhase::Twilight,
                    weight: 15,
                    sun_elevation_degrees: FloatRange::new(-6.0, 10.0),
                    camera_exposure_ev100: FloatRange::new(5.0, 10.5),
                    artificial_light_strength: FloatRange::new(0.18, 1.25),
                },
                DayPhaseProfile {
                    phase: DayPhase::Night,
                    weight: 30,
                    sun_elevation_degrees: FloatRange::new(-20.0, -6.0),
                    camera_exposure_ev100: FloatRange::new(-1.0, 5.5),
                    artificial_light_strength: FloatRange::new(0.7, 2.8),
                },
            ],
        }
    }
}

/// Broad site regime used to correlate built density, vegetation, and roads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SceneDomain {
    /// Dense central-city fabric with taller neighbours and little planting.
    City,
    /// Urban commercial corridor dominated by low shops and arterial roads.
    Urban,
    /// Car-oriented suburban retail strip with landscaping and parking.
    Suburban,
    /// Highway or major-road frontage with sparse neighbouring development.
    Roadside,
    /// Remote or small-town setting with little surrounding construction.
    Remote,
}

/// Road form associated with a sampled domain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoadKind {
    /// Narrow access or local street.
    LocalStreet,
    /// Multi-lane city or suburban arterial.
    UrbanArterial,
    /// Frontage road beside a highway.
    HighwayFrontage,
    /// Low-volume rural road.
    RuralRoad,
}

/// Correlated density and infrastructure ranges for one site regime.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneDomainProfile {
    /// Site category.
    pub domain: SceneDomain,
    /// Relative integer selection weight.
    pub weight: u32,
    /// Neighbouring building count.
    pub background_buildings: U32Range,
    /// Tree and shrub count.
    pub vegetation: U32Range,
    /// Marked parking-bay count.
    pub parking_bays: U32Range,
    /// Road form.
    pub road_kind: RoadKind,
    /// Number of road lanes.
    pub road_lanes: U32Range,
    /// Probability of a raised curb.
    pub curb_probability: f32,
    /// Utility or street-light pole count.
    pub utility_poles: U32Range,
    /// Probability that adjacent poles carry overhead lines.
    pub overhead_line_probability: f32,
}

/// Weighted site-domain regimes.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SceneDomainSamplingConfig {
    /// Available correlated site profiles.
    pub profiles: Vec<SceneDomainProfile>,
}

impl Default for SceneDomainSamplingConfig {
    fn default() -> Self {
        Self {
            profiles: vec![
                SceneDomainProfile {
                    domain: SceneDomain::City,
                    weight: 12,
                    background_buildings: U32Range::new(6, 12),
                    vegetation: U32Range::new(0, 5),
                    parking_bays: U32Range::new(0, 10),
                    road_kind: RoadKind::UrbanArterial,
                    road_lanes: U32Range::new(2, 5),
                    curb_probability: 0.96,
                    utility_poles: U32Range::new(2, 7),
                    overhead_line_probability: 0.12,
                },
                SceneDomainProfile {
                    domain: SceneDomain::Urban,
                    weight: 24,
                    background_buildings: U32Range::new(3, 9),
                    vegetation: U32Range::new(1, 8),
                    parking_bays: U32Range::new(6, 28),
                    road_kind: RoadKind::UrbanArterial,
                    road_lanes: U32Range::new(2, 4),
                    curb_probability: 0.88,
                    utility_poles: U32Range::new(2, 8),
                    overhead_line_probability: 0.52,
                },
                SceneDomainProfile {
                    domain: SceneDomain::Suburban,
                    weight: 34,
                    background_buildings: U32Range::new(1, 6),
                    vegetation: U32Range::new(5, 18),
                    parking_bays: U32Range::new(14, 46),
                    road_kind: RoadKind::UrbanArterial,
                    road_lanes: U32Range::new(2, 4),
                    curb_probability: 0.82,
                    utility_poles: U32Range::new(1, 7),
                    overhead_line_probability: 0.58,
                },
                SceneDomainProfile {
                    domain: SceneDomain::Roadside,
                    weight: 20,
                    background_buildings: U32Range::new(0, 4),
                    vegetation: U32Range::new(3, 15),
                    parking_bays: U32Range::new(8, 34),
                    road_kind: RoadKind::HighwayFrontage,
                    road_lanes: U32Range::new(2, 5),
                    curb_probability: 0.45,
                    utility_poles: U32Range::new(1, 6),
                    overhead_line_probability: 0.52,
                },
                SceneDomainProfile {
                    domain: SceneDomain::Remote,
                    weight: 10,
                    background_buildings: U32Range::new(0, 2),
                    vegetation: U32Range::new(8, 24),
                    parking_bays: U32Range::new(3, 18),
                    road_kind: RoadKind::RuralRoad,
                    road_lanes: U32Range::new(1, 2),
                    curb_probability: 0.16,
                    utility_poles: U32Range::new(0, 5),
                    overhead_line_probability: 0.68,
                },
            ],
        }
    }
}

/// Correlated outdoor condition used to avoid implausible independent lighting draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeatherPreset {
    /// Harder shadows, low cloud, and usually dry ground.
    Clear,
    /// Broken cloud with useful direct and indirect illumination.
    PartlyCloudy,
    /// Diffuse sky and soft shadows.
    Overcast,
    /// Reduced visibility and contrast despite limited cloud.
    Hazy,
    /// Broken or heavy cloud with visibly wet surfaces.
    AfterRain,
}

/// Physically correlated ranges associated with one weather category.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeatherProfile {
    /// Stable weather category.
    pub preset: WeatherPreset,
    /// Relative integer selection weight.
    pub weight: u32,
    /// Fractional cloud coverage.
    pub cloud_coverage: FloatRange,
    /// Atmospheric haze amount.
    pub haze: FloatRange,
    /// Direct-light intensity in renderer units.
    pub sun_intensity: FloatRange,
    /// Indirect-light intensity in renderer units.
    pub sky_intensity: FloatRange,
    /// Penumbra softness in `[0, 1]`.
    pub shadow_softness: FloatRange,
    /// Ground wetness in `[0, 1]`.
    pub ground_wetness: FloatRange,
    /// Approximate atmospheric visibility in kilometres.
    pub visibility_km: FloatRange,
    /// Correlated daylight white point in kelvin.
    pub color_temperature_k: FloatRange,
}

/// Weighted weather profiles used by one generation run.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WeatherSamplingConfig {
    /// Available correlated outdoor conditions.
    pub profiles: Vec<WeatherProfile>,
}

impl Default for WeatherSamplingConfig {
    fn default() -> Self {
        let profile = |preset,
                       weight,
                       cloud_coverage,
                       haze,
                       sun_intensity,
                       sky_intensity,
                       shadow_softness,
                       ground_wetness,
                       visibility_km,
                       color_temperature_k| WeatherProfile {
            preset,
            weight,
            cloud_coverage,
            haze,
            sun_intensity,
            sky_intensity,
            shadow_softness,
            ground_wetness,
            visibility_km,
            color_temperature_k,
        };
        Self {
            profiles: vec![
                profile(
                    WeatherPreset::Clear,
                    22,
                    FloatRange::new(0.0, 0.18),
                    FloatRange::new(0.0, 0.12),
                    FloatRange::new(2.5, 6.0),
                    FloatRange::new(0.15, 0.58),
                    FloatRange::new(0.05, 0.28),
                    FloatRange::new(0.0, 0.03),
                    FloatRange::new(35.0, 90.0),
                    FloatRange::new(5_200.0, 6_800.0),
                ),
                profile(
                    WeatherPreset::PartlyCloudy,
                    32,
                    FloatRange::new(0.20, 0.62),
                    FloatRange::new(0.02, 0.20),
                    FloatRange::new(1.2, 4.5),
                    FloatRange::new(0.30, 0.95),
                    FloatRange::new(0.18, 0.58),
                    FloatRange::new(0.0, 0.12),
                    FloatRange::new(24.0, 75.0),
                    FloatRange::new(5_000.0, 7_200.0),
                ),
                profile(
                    WeatherPreset::Overcast,
                    24,
                    FloatRange::new(0.72, 1.0),
                    FloatRange::new(0.08, 0.32),
                    FloatRange::new(0.35, 0.90),
                    FloatRange::new(0.62, 1.40),
                    FloatRange::new(0.78, 1.0),
                    FloatRange::new(0.0, 0.30),
                    FloatRange::new(12.0, 45.0),
                    FloatRange::new(6_000.0, 7_500.0),
                ),
                profile(
                    WeatherPreset::Hazy,
                    12,
                    FloatRange::new(0.05, 0.46),
                    FloatRange::new(0.38, 0.75),
                    FloatRange::new(0.45, 2.2),
                    FloatRange::new(0.40, 1.10),
                    FloatRange::new(0.35, 0.72),
                    FloatRange::new(0.0, 0.06),
                    FloatRange::new(4.0, 18.0),
                    FloatRange::new(4_500.0, 6_100.0),
                ),
                profile(
                    WeatherPreset::AfterRain,
                    10,
                    FloatRange::new(0.45, 0.92),
                    FloatRange::new(0.05, 0.30),
                    FloatRange::new(0.40, 2.8),
                    FloatRange::new(0.48, 1.30),
                    FloatRange::new(0.45, 0.90),
                    FloatRange::new(0.45, 1.0),
                    FloatRange::new(14.0, 55.0),
                    FloatRange::new(5_600.0, 7_400.0),
                ),
            ],
        }
    }
}

/// Architectural façade details layered over the existing wall material.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FacadeSamplingConfig {
    /// Fraction of visible wall area occupied by glazing.
    pub glazing_fraction: FloatRange,
    /// Glazing and dark window-band palette.
    pub glazing_materials: Vec<MaterialChoice>,
    /// Width of the principal entrance in metres.
    pub entrance_width_m: FloatRange,
    /// Additional façade dirt and staining amount.
    pub weathering: FloatRange,
}

/// Coarse use of an addition attached to the primary building shell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BuildingExtensionKind {
    /// A broad, low dining-room or seating wing.
    DiningWing,
    /// A compact glazed or enclosed entrance vestibule.
    EntranceVestibule,
    /// A plainer back-of-house, kitchen, or service annex.
    ServiceAnnex,
}

/// Roof form used by a low addition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BuildingExtensionRoof {
    /// A shallow flat roof with a visible perimeter cap.
    Flat,
    /// A single-slope roof rising toward the original building.
    Shed,
}

/// Distribution for optional additions attached to the primary building.
///
/// This configuration is deliberately independent of target class. A target
/// roof and an ordinary negative sampled with the same building seed therefore
/// receive exactly the same additions.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BuildingExtensionSamplingConfig {
    /// Relative weight for leaving the original building shell unchanged.
    pub none_weight: u32,
    /// Relative weight for one attached addition.
    pub one_weight: u32,
    /// Relative weight for two additions on opposite façades.
    pub two_weight: u32,
    /// Addition width divided by the host façade width.
    pub facade_width_fraction: FloatRange,
    /// Distance projected outward from the original wall.
    pub projection_m: FloatRange,
    /// Addition wall height divided by the original wall height.
    pub wall_height_fraction: FloatRange,
    /// Centre offset divided by the remaining usable façade span.
    pub facade_offset_fraction: FloatRange,
    /// Extra rise of a shed roof toward the original building.
    pub shed_rise_m: FloatRange,
    /// Relative weight for a dining-room wing.
    pub dining_wing_weight: u32,
    /// Relative weight for an entrance vestibule.
    pub entrance_vestibule_weight: u32,
    /// Relative weight for a service annex.
    pub service_annex_weight: u32,
    /// Relative weight for a flat-roofed addition.
    pub flat_roof_weight: u32,
    /// Relative weight for a shed-roofed addition.
    pub shed_roof_weight: u32,
}

impl Default for BuildingExtensionSamplingConfig {
    fn default() -> Self {
        Self {
            none_weight: 52,
            one_weight: 42,
            two_weight: 6,
            facade_width_fraction: FloatRange::new(0.2, 0.58),
            projection_m: FloatRange::new(1.8, 6.5),
            wall_height_fraction: FloatRange::new(0.5, 0.84),
            facade_offset_fraction: FloatRange::new(-0.85, 0.85),
            shed_rise_m: FloatRange::new(0.18, 0.72),
            dining_wing_weight: 44,
            entrance_vestibule_weight: 24,
            service_annex_weight: 32,
            flat_roof_weight: 58,
            shed_roof_weight: 42,
        }
    }
}

impl Default for FacadeSamplingConfig {
    fn default() -> Self {
        Self {
            glazing_fraction: FloatRange::new(0.10, 0.42),
            glazing_materials: vec![
                material("blue_grey_glass", 45, [0.12, 0.18, 0.22], 0.06, 0.20),
                material("neutral_glass", 35, [0.16, 0.17, 0.17], 0.05, 0.18),
                material("dark_tint", 20, [0.045, 0.055, 0.06], 0.08, 0.28),
            ],
            entrance_width_m: FloatRange::new(1.6, 3.8),
            weathering: FloatRange::new(0.0, 0.72),
        }
    }
}

/// Sign appearance independent of Pizza Hut recognition labels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignageKind {
    /// A retained Pizza Hut sign.
    PizzaHut,
    /// A removed sign leaving fixings, fade, or a painted ghost.
    RemovedGhost,
    /// Signage for a later unrelated tenant.
    RebrandedTenant,
}

/// Signage distribution and physically bounded façade placement.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SignageSamplingConfig {
    /// Weight for generating no sign.
    pub none_weight: u32,
    /// Weight for retained Pizza Hut branding.
    pub pizza_hut_weight: u32,
    /// Weight for a removed-sign trace.
    pub removed_ghost_weight: u32,
    /// Weight for unrelated tenant branding.
    pub rebranded_tenant_weight: u32,
    /// Sign width as a fraction of its façade width.
    pub width_fraction: FloatRange,
    /// Physical sign height in metres.
    pub height_m: FloatRange,
    /// Sign centre as a fraction of wall height.
    pub vertical_fraction: FloatRange,
    /// Light emission for signs that support it.
    pub emissive_strength: FloatRange,
}

impl Default for SignageSamplingConfig {
    fn default() -> Self {
        Self {
            none_weight: 20,
            pizza_hut_weight: 16,
            removed_ghost_weight: 30,
            rebranded_tenant_weight: 34,
            width_fraction: FloatRange::new(0.16, 0.46),
            height_m: FloatRange::new(0.65, 1.75),
            vertical_fraction: FloatRange::new(0.46, 0.78),
            emissive_strength: FloatRange::new(0.0, 2.2),
        }
    }
}

/// Procedural neighbouring-building distribution.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BackgroundBuildingSamplingConfig {
    /// Number of neighbouring masses.
    pub count: U32Range,
    /// Clear setback from the target building's bounding radius.
    pub setback_m: FloatRange,
    /// Building width.
    pub width_m: FloatRange,
    /// Building depth.
    pub depth_m: FloatRange,
    /// Building height.
    pub height_m: FloatRange,
    /// Maximum yaw perturbation around a site-facing orientation.
    pub yaw_jitter_degrees: FloatRange,
}

impl Default for BackgroundBuildingSamplingConfig {
    fn default() -> Self {
        Self {
            count: U32Range::new(0, 12),
            setback_m: FloatRange::new(18.0, 52.0),
            width_m: FloatRange::new(8.0, 26.0),
            depth_m: FloatRange::new(7.0, 22.0),
            height_m: FloatRange::new(3.2, 28.0),
            yaw_jitter_degrees: FloatRange::new(-18.0, 18.0),
        }
    }
}

/// Coarse procedural vegetation family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum VegetationKind {
    /// Broad-canopy deciduous tree.
    DeciduousTree,
    /// Narrow evergreen tree.
    EvergreenTree,
    /// Palm-like tree.
    Palm,
    /// Low shrub or hedge mass.
    Shrub,
}

/// Site vegetation distribution.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VegetationSamplingConfig {
    /// Number of independent plants.
    pub count: U32Range,
    /// Relative deciduous-tree weight.
    pub deciduous_weight: u32,
    /// Relative evergreen-tree weight.
    pub evergreen_weight: u32,
    /// Relative palm weight.
    pub palm_weight: u32,
    /// Relative shrub weight.
    pub shrub_weight: u32,
    /// Clear distance from the target footprint.
    pub building_setback_m: FloatRange,
    /// Additional radial site distance.
    pub site_distance_m: FloatRange,
    /// Tree height range.
    pub tree_height_m: FloatRange,
    /// Shrub height range.
    pub shrub_height_m: FloatRange,
    /// Canopy radius divided by plant height.
    pub canopy_radius_fraction: FloatRange,
}

impl Default for VegetationSamplingConfig {
    fn default() -> Self {
        Self {
            count: U32Range::new(0, 24),
            deciduous_weight: 42,
            evergreen_weight: 22,
            palm_weight: 12,
            shrub_weight: 24,
            building_setback_m: FloatRange::new(1.5, 5.0),
            site_distance_m: FloatRange::new(0.0, 28.0),
            tree_height_m: FloatRange::new(4.5, 13.5),
            shrub_height_m: FloatRange::new(0.55, 2.2),
            canopy_radius_fraction: FloatRange::new(0.18, 0.38),
        }
    }
}

/// Complete site-composition configuration.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompositionSamplingConfig {
    /// Correlated daylight and night-time exposure regimes.
    pub day_phase: DayPhaseSamplingConfig,
    /// Correlated city, urban, suburban, roadside, and remote regimes.
    pub domains: SceneDomainSamplingConfig,
    /// Correlated weather and sky conditions.
    pub weather: WeatherSamplingConfig,
    /// Ground and car-park surface palette.
    pub ground_materials: Vec<MaterialChoice>,
    /// Glazing and entrance variation.
    pub facade: FacadeSamplingConfig,
    /// Optional wings, vestibules, and service annexes attached to the shell.
    #[serde(default)]
    pub building_extensions: BuildingExtensionSamplingConfig,
    /// Sign presence, type, and placement.
    pub signage: SignageSamplingConfig,
    /// Neighbouring building masses.
    pub background_buildings: BackgroundBuildingSamplingConfig,
    /// Trees and shrubs around the site.
    pub vegetation: VegetationSamplingConfig,
}

impl Default for CompositionSamplingConfig {
    fn default() -> Self {
        Self {
            day_phase: DayPhaseSamplingConfig::default(),
            domains: SceneDomainSamplingConfig::default(),
            weather: WeatherSamplingConfig::default(),
            ground_materials: vec![
                material("aged_asphalt", 42, [0.19, 0.20, 0.19], 0.66, 0.96),
                material("light_concrete", 26, [0.52, 0.50, 0.45], 0.58, 0.90),
                material("dark_asphalt", 22, [0.095, 0.105, 0.10], 0.60, 0.92),
                material("paving_blocks", 10, [0.40, 0.34, 0.28], 0.62, 0.94),
            ],
            facade: FacadeSamplingConfig::default(),
            building_extensions: BuildingExtensionSamplingConfig::default(),
            signage: SignageSamplingConfig::default(),
            background_buildings: BackgroundBuildingSamplingConfig::default(),
            vegetation: VegetationSamplingConfig::default(),
        }
    }
}

/// Cardinal façade in the right-handed, Y-up site coordinate system.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FacadeSide {
    /// Wall at negative Z.
    Front,
    /// Wall at positive X.
    Right,
    /// Wall at positive Z.
    Back,
    /// Wall at negative X.
    Left,
}

/// Correlated sampled environment values used by sky, lighting, and materials.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledEnvironment {
    /// Day, twilight, or night regime.
    pub day_phase: DayPhase,
    /// Sampled site-density regime.
    pub domain: SceneDomain,
    /// Selected weather family.
    pub weather: WeatherPreset,
    /// Penumbra softness in `[0, 1]`.
    pub shadow_softness: f32,
    /// Ground wetness in `[0, 1]`.
    pub ground_wetness: f32,
    /// Approximate atmospheric visibility in kilometres.
    pub visibility_km: f32,
    /// Correlated daylight white point in kelvin.
    pub color_temperature_k: f32,
    /// Camera exposure in EV100, correlated with day phase.
    pub camera_exposure_ev100: f32,
    /// Artificial site-light strength, correlated with day phase.
    pub artificial_light_strength: f32,
}

/// Glazing, entrance, and weathering values for the target façade.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledFacade {
    /// Fraction of wall area occupied by glazing.
    pub glazing_fraction: f32,
    /// Resolved glazing material.
    pub glazing_material: SampledMaterial,
    /// Principal entrance side.
    pub entrance_side: FacadeSide,
    /// Principal entrance width in metres.
    pub entrance_width_m: f32,
    /// Additional dirt and staining amount.
    pub weathering: f32,
}

/// One low addition structurally attached to a primary-building façade.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledBuildingExtension {
    /// Coarse real-world use controlling proportions and façade treatment.
    pub kind: BuildingExtensionKind,
    /// Original-building façade to which the addition is attached.
    pub facade: FacadeSide,
    /// Ground-level world-space centre of the addition footprint.
    pub position: Vec3,
    /// Axis-aligned width, wall height, and depth in metres.
    pub size_m: Vec3,
    /// Low roof form above the addition walls.
    pub roof: BuildingExtensionRoof,
    /// Shed rise toward the original building; zero for a flat roof.
    pub roof_rise_m: f32,
}

/// One resolved sign plane on the target building.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledSignage {
    /// Retained, removed, or replacement sign class.
    pub kind: SignageKind,
    /// Host wall.
    pub facade: FacadeSide,
    /// Sign centre in target-building local coordinates.
    pub center: Vec3,
    /// Physical width and height in metres.
    pub size_m: Vec2,
    /// Light emission in renderer units; removed ghosts are always zero.
    pub emissive_strength: f32,
    /// Surface fading and damage in `[0, 1]`.
    pub weathering: f32,
}

/// One neighbouring building mass with a resolved façade material.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledBackgroundBuilding {
    /// Coarse building use/form selected from the site regime.
    #[serde(default)]
    pub kind: BackgroundBuildingKind,
    /// World-space centre at ground level.
    pub position: Vec3,
    /// Width, height, and depth in metres.
    pub size_m: Vec3,
    /// Rotation around world Y.
    pub yaw_degrees: f32,
    /// Resolved façade finish.
    pub material: SampledMaterial,
}

/// Coarse neighbouring-building form.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackgroundBuildingKind {
    /// One- or two-storey shop, restaurant, or service building.
    #[default]
    LowCommercial,
    /// Denser mixed-use or office mass used mainly in city scenes.
    MidriseMixedUse,
    /// Detached or small attached residential mass.
    Residential,
    /// Warehouse or workshop shed.
    IndustrialShed,
}

/// Parking, road, curb, and utility layout correlated with a site domain.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledSiteInfrastructure {
    /// Site regime from which all fields were sampled.
    pub domain: SceneDomain,
    /// Number of marked parking bays.
    pub parking_bays: u32,
    /// Centre of the principal parking field in world coordinates.
    pub parking_center: Vec3,
    /// Parking bay width and length in metres.
    pub parking_bay_size_m: Vec2,
    /// Orientation of parking rows around world Y.
    pub parking_yaw_degrees: f32,
    /// Adjacent road form.
    pub road_kind: RoadKind,
    /// Number of road lanes.
    pub road_lanes: u32,
    /// Nominal lane width in metres.
    pub lane_width_m: f32,
    /// Road centre-line offset along world Z.
    pub road_center_z_m: f32,
    /// Whether the site edge has a raised curb.
    pub raised_curb: bool,
    /// Utility and street-light poles along the road.
    pub utility_poles: Vec<SampledUtilityPole>,
    /// Overhead connections between pole indexes.
    pub utility_lines: Vec<SampledUtilityLine>,
}

/// One utility or street-light pole.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledUtilityPole {
    /// Pole base in world coordinates.
    pub position: Vec3,
    /// Pole height in metres.
    pub height_m: f32,
    /// Whether the pole carries an artificial luminaire.
    pub has_light: bool,
}

/// One overhead cable span referencing sampled pole indexes.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledUtilityLine {
    /// First pole index.
    pub start_pole: u16,
    /// Second pole index.
    pub end_pole: u16,
    /// Mid-span sag in metres.
    pub sag_m: f32,
}

/// One procedural plant with enough dimensions for a renderer to resolve an asset.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledVegetation {
    /// Coarse plant family.
    pub kind: VegetationKind,
    /// World-space base position.
    pub position: Vec3,
    /// Overall plant height in metres.
    pub height_m: f32,
    /// Approximate canopy or hedge radius in metres.
    pub canopy_radius_m: f32,
    /// Rotation around world Y.
    pub yaw_degrees: f32,
    /// Seasonal hue/value variation in `[-1, 1]`.
    pub seasonal_variation: f32,
}

/// Fully sampled realism plan surrounding the target roof.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledComposition {
    /// Source-asset IDs resolved by the producer after procedural sampling.
    /// IDs refer to hashed/licensed entries in [`crate::DatasetManifest`].
    #[serde(default)]
    pub source_asset_ids: Vec<String>,
    /// Correlated environment, absent only in legacy deserialized plans.
    pub environment: Option<SampledEnvironment>,
    /// Resolved ground material, absent only in legacy plans.
    pub ground_material: Option<SampledMaterial>,
    /// Target-building façade detail, absent only in legacy plans.
    pub facade: Option<SampledFacade>,
    /// Optional additions attached to the primary building shell.
    #[serde(default)]
    pub building_extensions: Vec<SampledBuildingExtension>,
    /// Zero or more façade signs.
    pub signage: Vec<SampledSignage>,
    /// Neighbouring building masses.
    pub background_buildings: Vec<SampledBackgroundBuilding>,
    /// Site vegetation.
    pub vegetation: Vec<SampledVegetation>,
    /// Parking, road, curb, and utility layout.
    pub infrastructure: Option<SampledSiteInfrastructure>,
}

fn material(
    id: &str,
    weight: u32,
    base_color_srgb: [f32; 3],
    roughness_min: f32,
    roughness_max: f32,
) -> MaterialChoice {
    MaterialChoice {
        id: id.to_owned(),
        weight,
        base_color_srgb,
        base_color_variation: 0.0,
        roughness: FloatRange::new(roughness_min, roughness_max),
        weathering: FloatRange::new(0.0, 0.82),
    }
}
