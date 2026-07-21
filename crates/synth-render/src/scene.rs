//! Renderer-owned procedural scene descriptions and geometry assembly.

use glam::{Quat, Vec3};
use serde::{Deserialize, Serialize};

use crate::{
    MaterialSlot, RenderError, RenderMesh, RenderVertex, SurfacePattern, append_quad,
    append_quad_tinted, append_quad_tinted_pattern, assets::MAX_TEXTURE_LAYERS,
};

/// Per-scene mapping from logical surface slots to resident PBR texture layers.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MaterialSelection {
    layers: [u32; MaterialSlot::COUNT],
    #[serde(default = "default_patterns")]
    patterns: [SurfacePattern; MaterialSlot::COUNT],
}

impl Default for MaterialSelection {
    fn default() -> Self {
        Self {
            layers: std::array::from_fn(|index| index as u32),
            patterns: std::array::from_fn(default_surface_pattern),
        }
    }
}

impl MaterialSelection {
    /// Selects a prepared physical texture layer for one logical material.
    pub fn set(&mut self, slot: MaterialSlot, texture_layer: u32) -> Result<(), RenderError> {
        if texture_layer as usize >= MAX_TEXTURE_LAYERS {
            return Err(RenderError::InvalidTextureData);
        }
        self.layers[slot.as_u32() as usize] = texture_layer;
        Ok(())
    }

    /// Returns the selected physical texture layer.
    #[must_use]
    pub const fn get(self, slot: MaterialSlot) -> u32 {
        self.layers[slot.as_u32() as usize]
    }

    /// Selects procedural surface semantics independently from texture layers.
    pub fn set_pattern(&mut self, slot: MaterialSlot, pattern: SurfacePattern) {
        self.patterns[slot.as_u32() as usize] = pattern;
    }

    /// Returns the selected procedural treatment for a logical material slot.
    #[must_use]
    pub const fn pattern(self, slot: MaterialSlot) -> SurfacePattern {
        self.patterns[slot.as_u32() as usize]
    }

    pub(crate) const fn layers(self) -> [u32; MaterialSlot::COUNT] {
        self.layers
    }

    pub(crate) const fn patterns(self) -> [SurfacePattern; MaterialSlot::COUNT] {
        self.patterns
    }

    fn is_valid(self) -> bool {
        self.layers
            .iter()
            .all(|layer| (*layer as usize) < MAX_TEXTURE_LAYERS)
            && self.patterns.iter().all(|pattern| pattern.is_known())
    }
}

fn default_surface_pattern(index: usize) -> SurfacePattern {
    match index as u32 {
        0 => SurfacePattern::ROOF_SEAMS,
        4 => SurfacePattern::BUILDING_GLASS,
        6 => SurfacePattern::PIZZA_HUT_SIGN,
        7 => SurfacePattern::ASPHALT,
        10 => SurfacePattern::BACKGROUND_WINDOWS,
        _ => SurfacePattern::SMOOTH,
    }
}

fn default_patterns() -> [SurfacePattern; MaterialSlot::COUNT] {
    std::array::from_fn(default_surface_pattern)
}

/// Broad setting used to vary background density and street furniture.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnvironmentDomain {
    /// Dense multi-storey surroundings and street lighting.
    City,
    /// Low commercial neighbours, parking, signs, and mixed vegetation.
    #[default]
    Urban,
    /// Open ground with sparse buildings and mature vegetation.
    Remote,
}

/// Lighting period used by procedural sky, emissive materials, and local lights.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TimeOfDay {
    /// Sun and sky illumination dominate.
    #[default]
    Day,
    /// Low-angle sky with partially active artificial lighting.
    Twilight,
    /// Low sky illumination with illuminated windows and signs.
    Night,
}

impl TimeOfDay {
    pub(crate) const fn emission_scale(self) -> f32 {
        match self {
            Self::Day => 0.0,
            Self::Twilight => 0.45,
            Self::Night => 1.0,
        }
    }
}

/// Weather family used for atmospheric and ground response.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WeatherAppearance {
    /// Harder shadows and a mostly unobscured sky.
    #[default]
    Clear,
    /// Broken cloud with direct and indirect illumination.
    PartlyCloudy,
    /// Diffuse, cloud-dominated sky.
    Overcast,
    /// Reduced atmospheric visibility.
    Hazy,
    /// Wet ground beneath broken or heavy cloud.
    AfterRain,
}

/// Reproducible environment controls carried with a render mesh.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RenderEnvironment {
    /// Surrounding land-use density.
    pub domain: EnvironmentDomain,
    /// Day or night appearance family.
    pub time_of_day: TimeOfDay,
    /// Procedural cloud amount in `[0, 1]` when no environment map is supplied.
    pub cloud_cover: f32,
    /// Atmospheric desaturation near the horizon in `[0, 1]`.
    pub haze: f32,
    /// Approximate atmospheric visibility in kilometres.
    pub visibility_km: f32,
    /// Correlated categorical weather family.
    pub weather: WeatherAppearance,
    /// Linear exposure multiplier applied to an environment map.
    pub environment_exposure: f32,
    /// Overall RGB exposure compensation in stops.
    pub exposure_ev: f32,
    /// White-balance target in Kelvin.
    pub color_temperature_kelvin: f32,
    /// Directional shadow-filter radius in `[0, 1]`.
    pub shadow_softness: f32,
    /// Darkening and gloss applied to ground surfaces in `[0, 1]`.
    pub ground_wetness: f32,
    /// Stable seed used by shader-side surface and sky variation.
    pub seed: u32,
    /// Rotation applied to an equirectangular environment around world up.
    pub environment_yaw_radians: f32,
}

impl Default for RenderEnvironment {
    fn default() -> Self {
        Self {
            domain: EnvironmentDomain::Urban,
            time_of_day: TimeOfDay::Day,
            cloud_cover: 0.25,
            haze: 0.15,
            visibility_km: 45.0,
            weather: WeatherAppearance::PartlyCloudy,
            environment_exposure: 1.0,
            exposure_ev: 0.0,
            color_temperature_kelvin: 6_500.0,
            shadow_softness: 0.25,
            ground_wetness: 0.0,
            seed: 1,
            environment_yaw_radians: 0.0,
        }
    }
}

impl RenderEnvironment {
    pub(crate) fn is_valid(self) -> bool {
        (0.0..=1.0).contains(&self.cloud_cover)
            && (0.0..=1.0).contains(&self.haze)
            && self.visibility_km.is_finite()
            && self.visibility_km > 0.0
            && self.environment_exposure.is_finite()
            && self.environment_exposure >= 0.0
            && self.exposure_ev.is_finite()
            && self.color_temperature_kelvin.is_finite()
            && (1_500.0..=12_000.0).contains(&self.color_temperature_kelvin)
            && (0.0..=1.0).contains(&self.shadow_softness)
            && (0.0..=1.0).contains(&self.ground_wetness)
            && self.environment_yaw_radians.is_finite()
    }
}

/// Cardinal side of the target building in its right-handed, Y-up frame.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BuildingSide {
    /// Wall at negative Z.
    #[default]
    Front,
    /// Wall at positive X.
    Right,
    /// Wall at positive Z.
    Back,
    /// Wall at negative X.
    Left,
}

/// Coarse use of an addition attached to the primary building.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BuildingExtensionKind {
    /// A broad, windowed dining-room wing.
    DiningWing,
    /// A compact glazed entrance vestibule.
    EntranceVestibule,
    /// A plainer kitchen or service annex.
    ServiceAnnex,
}

/// Low roof form above a primary-building addition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BuildingExtensionRoof {
    /// Flat roof with a shallow perimeter cap.
    Flat,
    /// Single-slope roof rising toward the original building.
    Shed,
}

/// One low structural addition joined to a primary-building façade.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BuildingExtensionDescription {
    /// Coarse real-world use controlling wall treatment.
    pub kind: BuildingExtensionKind,
    /// Original-building wall receiving the addition.
    pub facade: BuildingSide,
    /// Ground-level world-space centre of the addition footprint.
    pub position: [f32; 3],
    /// Axis-aligned width, wall height, and depth.
    pub size: [f32; 3],
    /// Addition roof form.
    pub roof: BuildingExtensionRoof,
    /// Rise toward the host wall for a shed roof.
    pub roof_rise_m: f32,
}

/// Dimensions and repetition controls for the restaurant facade.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FacadeDescription {
    /// Height of the bottom of the long window band.
    pub window_sill_height_m: f32,
    /// Height of facade glazing.
    pub window_height_m: f32,
    /// Number of glazed panels on the front and rear elevations.
    pub long_side_window_count: u32,
    /// Number of glazed panels on each short elevation.
    pub short_side_window_count: u32,
    /// Width of the main front entrance.
    pub entrance_width_m: f32,
    /// Height of the main front entrance.
    pub entrance_height_m: f32,
    /// Whether to add a shallow entrance canopy.
    pub entrance_canopy: bool,
    /// Wall carrying the principal entrance and its canopy.
    pub entrance_side: BuildingSide,
    /// Dirt, fading, and staining applied to exterior walls in `[0, 1]`.
    pub weathering: f32,
    /// Sampled nonlinear glazing base colour.
    pub glazing_color_srgb: [f32; 3],
    /// Sampled glazing roughness in `[0, 1]`.
    pub glazing_roughness: f32,
    /// Dirt and fading folded into glazing appearance in `[0, 1]`.
    pub glazing_weathering: f32,
}

impl Default for FacadeDescription {
    fn default() -> Self {
        Self {
            window_sill_height_m: 0.75,
            window_height_m: 1.55,
            long_side_window_count: 8,
            short_side_window_count: 5,
            entrance_width_m: 1.8,
            entrance_height_m: 2.25,
            entrance_canopy: true,
            entrance_side: BuildingSide::Front,
            weathering: 0.15,
            glazing_color_srgb: [0.12, 0.18, 0.22],
            glazing_roughness: 0.16,
            glazing_weathering: 0.05,
        }
    }
}

/// Visual identity of an attached or freestanding sign.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignStyle {
    /// Active Pizza Hut branding.
    PizzaHut,
    /// Faded outline or mounting scar after sign removal.
    RemovedGhost,
    /// Replacement branding on a converted former restaurant.
    Tenant,
    /// Unbranded backing panel.
    Blank,
}

/// Physical mounting family for sign geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignMount {
    /// Flat panel mounted to an exterior wall.
    Facade,
    /// Panel near the roof crown or eave.
    Roof,
    /// Freestanding panel with a generated support pole.
    Pole,
}

/// Explicit sign placement and appearance for one scene.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SignDescription {
    /// Branding state.
    pub style: SignStyle,
    /// Attachment family.
    pub mount: SignMount,
    /// World-space panel centre.
    pub center: [f32; 3],
    /// Panel width and height in metres.
    pub size: [f32; 2],
    /// Rotation around world up in radians.
    pub yaw_radians: f32,
    /// Night-time emissive multiplier; zero disables emission.
    pub emissive_strength: f32,
    /// Surface fading and damage in `[0, 1]`.
    pub weathering: f32,
}

/// Coarse geometry family for a neighbouring building.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackgroundBuildingKind {
    /// One- or two-storey commercial box.
    #[default]
    LowCommercial,
    /// Taller mixed-use or office mass with repeated windows.
    MidriseMixedUse,
    /// Smaller building with a pitched roof.
    Residential,
    /// Wide metal-clad shed or workshop.
    IndustrialShed,
}

/// One non-target building contributing depth and appearance context.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct BackgroundBuilding {
    /// World-space centre of the footprint at ground level.
    pub center: [f32; 3],
    /// Width, height, and depth in metres.
    pub size: [f32; 3],
    /// Rotation around world up in radians.
    pub yaw_radians: f32,
    /// Coarse massing family.
    pub kind: BackgroundBuildingKind,
    /// Sampled nonlinear base-colour tint.
    pub base_color_srgb: [f32; 3],
    /// Surface weathering in `[0, 1]`.
    pub weathering: f32,
    /// Sampled surface roughness in `[0, 1]`.
    pub roughness: f32,
    /// Per-building wall treatment; roof caps use a non-windowed derivative.
    pub surface_pattern: SurfacePattern,
}

/// Coarse geometry family for one procedural plant.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TreeKind {
    /// Broad-canopy deciduous tree.
    #[default]
    Deciduous,
    /// Narrow layered evergreen.
    Evergreen,
    /// Tall trunk with a compact frond crown.
    Palm,
    /// Low multi-lobed shrub or hedge mass.
    Shrub,
}

/// One low-poly vegetation instance.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TreeInstance {
    /// World-space position at the centre of the trunk base.
    pub position: [f32; 3],
    /// Total tree height in metres.
    pub height_m: f32,
    /// Approximate crown radius in metres.
    pub crown_radius_m: f32,
    /// Coarse plant geometry family.
    pub kind: TreeKind,
    /// Rotation around world up in radians.
    pub yaw_radians: f32,
    /// Seasonal foliage variation in `[-1, 1]`.
    pub seasonal_variation: f32,
}

/// Supported renderer-side coarse occluder geometry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SceneOccluderKind {
    /// Parked passenger vehicle.
    Vehicle,
    /// Freestanding commercial sign.
    PoleSign,
    /// Utility pole, cross-arm, and short cable runs.
    UtilityPole,
    /// Low planted shrub cluster.
    Shrub,
    /// Rooftop or ground-mounted HVAC cabinet.
    Hvac,
    /// Street or parking-lot lamp.
    StreetLight,
    /// Standing low-poly person for near-camera occlusion.
    Pedestrian,
    /// Generic neighbouring commercial building.
    Building,
    /// Generic vegetation resolved as a small tree cluster.
    Vegetation,
}

/// Placement of one generated occluder.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneOccluder {
    /// Geometry family.
    pub kind: SceneOccluderKind,
    /// World-space base position.
    pub position: [f32; 3],
    /// Rotation around world up in radians.
    pub yaw_radians: f32,
    /// Uniform size multiplier used only with family-default geometry.
    ///
    /// When `nominal_size_m` is populated it is already the fully resolved size
    /// and this value is retained solely as sampling provenance.
    pub scale: f32,
    /// Fully resolved width, height, and depth in metres, or all zeros to request
    /// that the renderer multiply its family default by `scale`.
    pub nominal_size_m: [f32; 3],
}

/// Parking and adjacent-road geometry surrounding the target building.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SiteInfrastructure {
    /// Number of marked parking bays.
    pub parking_bays: u32,
    /// World-space centre of the principal parking row.
    pub parking_center: [f32; 3],
    /// Width and length of one parking bay in metres.
    pub parking_bay_size_m: [f32; 2],
    /// Parking-row orientation around world up.
    pub parking_yaw_radians: f32,
    /// Number of lanes in the adjacent road.
    pub road_lanes: u32,
    /// Coarse road family controlling markings and shoulders.
    pub road_kind: SiteRoadKind,
    /// Width of one road lane in metres.
    pub lane_width_m: f32,
    /// Road centre-line position along world Z.
    pub road_center_z_m: f32,
    /// Whether a raised curb separates the site and road.
    pub raised_curb: bool,
    /// Exact utility or street-light pole placements.
    pub utility_poles: Vec<UtilityPoleDescription>,
    /// Exact overhead cable spans between poles.
    pub utility_lines: Vec<UtilityLineDescription>,
}

impl SiteInfrastructure {
    fn for_scene(
        footprint_width_m: f32,
        footprint_depth_m: f32,
        ground_half_extent_m: f32,
    ) -> Self {
        Self {
            parking_bays: (footprint_width_m / 2.55).round().clamp(2.0, 16.0) as u32,
            parking_center: [0.0, 0.0, -footprint_depth_m * 0.82],
            parking_bay_size_m: [2.55, 4.7],
            parking_yaw_radians: 0.0,
            road_lanes: 2,
            road_kind: SiteRoadKind::UrbanArterial,
            lane_width_m: (ground_half_extent_m * 0.14).max(3.0),
            road_center_z_m: -ground_half_extent_m * 0.78,
            raised_curb: true,
            utility_poles: Vec::new(),
            utility_lines: Vec::new(),
        }
    }

    fn is_valid(&self) -> bool {
        self.parking_center
            .iter()
            .chain(self.parking_bay_size_m.iter())
            .chain(
                [
                    self.parking_yaw_radians,
                    self.lane_width_m,
                    self.road_center_z_m,
                ]
                .iter(),
            )
            .all(|value| value.is_finite())
            && self.parking_bay_size_m.iter().all(|value| *value > 0.0)
            && self.road_lanes > 0
            && self.lane_width_m > 0.0
            && self
                .utility_poles
                .iter()
                .all(UtilityPoleDescription::is_valid)
            && self
                .utility_lines
                .iter()
                .all(UtilityLineDescription::is_valid)
    }
}

/// Coarse adjacent-road appearance family.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SiteRoadKind {
    /// Narrow access or local street.
    LocalStreet,
    /// Multi-lane city or suburban arterial.
    #[default]
    UrbanArterial,
    /// Marked frontage road with shoulders.
    HighwayFrontage,
    /// Low-volume rural road with a centre line.
    RuralRoad,
}

/// One pole generated from sampled site infrastructure.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct UtilityPoleDescription {
    /// World-space base position.
    pub position: [f32; 3],
    /// Pole height in metres.
    pub height_m: f32,
    /// Whether a luminaire is attached near the top.
    pub has_light: bool,
}

impl UtilityPoleDescription {
    fn is_valid(&self) -> bool {
        self.position.iter().all(|value| value.is_finite())
            && self.height_m.is_finite()
            && self.height_m > 0.0
    }
}

/// One sagging overhead cable represented by world-space endpoints.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct UtilityLineDescription {
    /// Cable endpoint on the first pole.
    pub start: [f32; 3],
    /// Cable endpoint on the second pole.
    pub end: [f32; 3],
    /// Mid-span vertical sag in metres.
    pub sag_m: f32,
}

impl UtilityLineDescription {
    fn is_valid(&self) -> bool {
        self.start
            .iter()
            .chain(self.end.iter())
            .chain([self.sag_m].iter())
            .all(|value| value.is_finite())
            && self.sag_m >= 0.0
            && self.start != self.end
    }
}

/// Local point light embedded in the scene description.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneLight {
    /// World-space light position.
    pub position: [f32; 3],
    /// Linear RGB light colour.
    pub color: [f32; 3],
    /// Radiometric scale used by the compact shader.
    pub intensity: f32,
    /// Distance at which contribution reaches zero.
    pub range_m: f32,
}

impl SceneLight {
    pub(crate) fn is_valid(self) -> bool {
        self.position
            .iter()
            .chain(self.color.iter())
            .chain([self.intensity, self.range_m].iter())
            .all(|value| value.is_finite())
            && self.color.iter().all(|value| *value >= 0.0)
            && self.intensity >= 0.0
            && self.range_m > 0.0
    }
}

/// Complete renderer-side scene assembled around one semantic roof.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SceneDescription {
    /// Building exterior width.
    pub footprint_width_m: f32,
    /// Building exterior depth.
    pub footprint_depth_m: f32,
    /// Eave height above the ground plane.
    pub wall_height_m: f32,
    /// Half extent of the generated ground square.
    pub ground_half_extent_m: f32,
    /// Facade layout controls.
    pub facade: FacadeDescription,
    /// Optional low wings, vestibules, and service annexes.
    pub building_extensions: Vec<BuildingExtensionDescription>,
    /// Sky, density, and time-of-day controls.
    pub environment: RenderEnvironment,
    /// Logical-to-physical PBR texture selection for this building and context.
    pub materials: MaterialSelection,
    /// Parking, road, curb, pole, and cable geometry.
    pub site: SiteInfrastructure,
    /// Attached, roof, ghost, tenant, and freestanding signs.
    pub signs: Vec<SignDescription>,
    /// Explicit contextual building boxes.
    pub background_buildings: Vec<BackgroundBuilding>,
    /// Explicit vegetation placements.
    pub trees: Vec<TreeInstance>,
    /// Explicit foreground, utility, rooftop, and sign objects.
    pub occluders: Vec<SceneOccluder>,
    /// Additional local lights, capped by the renderer at four.
    pub lights: Vec<SceneLight>,
}

impl SceneDescription {
    /// Creates an empty-context scene description ready for explicit population.
    #[must_use]
    pub fn new(
        footprint_width_m: f32,
        footprint_depth_m: f32,
        wall_height_m: f32,
        ground_half_extent_m: f32,
        environment: RenderEnvironment,
    ) -> Self {
        Self {
            footprint_width_m,
            footprint_depth_m,
            wall_height_m,
            ground_half_extent_m,
            facade: FacadeDescription::default(),
            building_extensions: Vec::new(),
            environment,
            materials: MaterialSelection::default(),
            site: SiteInfrastructure::for_scene(
                footprint_width_m,
                footprint_depth_m,
                ground_half_extent_m,
            ),
            signs: Vec::new(),
            background_buildings: Vec::new(),
            trees: Vec::new(),
            occluders: Vec::new(),
            lights: Vec::new(),
        }
    }

    /// Creates a visibly distinct city, urban, or remote context without external assets.
    pub fn contextual(
        footprint_width_m: f32,
        footprint_depth_m: f32,
        wall_height_m: f32,
        ground_half_extent_m: f32,
        environment: RenderEnvironment,
    ) -> Result<Self, RenderError> {
        let mut scene = Self::new(
            footprint_width_m,
            footprint_depth_m,
            wall_height_m,
            ground_half_extent_m,
            environment,
        );
        scene.populate_context();
        scene.validate()?;
        Ok(scene)
    }

    /// Converts the renderer-independent sampled composition into visible geometry.
    ///
    /// This is the canonical bridge for dataset generation: every sampled
    /// occluder category, façade placement, sign, neighbouring mass, plant, road,
    /// pole, and cable span is resolved here instead of falling back to a generic
    /// contextual scene.
    pub fn from_sampled(sampled: &synth_data::SampledScene) -> Result<Self, RenderError> {
        let seed = sampled_scene_seed(sampled);
        let sampled_environment = sampled.composition.environment;
        let domain = sampled_environment
            .map(|environment| map_domain(environment.domain))
            .unwrap_or(EnvironmentDomain::Urban);
        let time_of_day = sampled_environment
            .map(|environment| map_day_phase(environment.day_phase))
            .unwrap_or_else(|| time_from_sun_elevation(sampled.lighting.sun_elevation_degrees));
        let visibility_km = sampled_environment
            .map(|environment| environment.visibility_km)
            .unwrap_or_else(|| (60.0 * (1.0 - sampled.lighting.haze)).max(2.0));
        let emission_scale = time_of_day.emission_scale();
        let mut description = Self::new(
            sampled.building.footprint_width_m,
            sampled.building.footprint_depth_m,
            sampled.building.wall_height_m,
            sampled.building.ground_half_extent_m,
            RenderEnvironment {
                domain,
                time_of_day,
                cloud_cover: sampled.lighting.cloud_coverage,
                haze: sampled.lighting.haze,
                visibility_km,
                weather: sampled_environment
                    .map(|environment| map_weather(environment.weather))
                    .unwrap_or(WeatherAppearance::PartlyCloudy),
                environment_exposure: match (time_of_day, domain) {
                    (TimeOfDay::Day, _) => 0.9,
                    (TimeOfDay::Twilight, _) => 0.6,
                    (TimeOfDay::Night, EnvironmentDomain::City) => 0.24,
                    (TimeOfDay::Night, EnvironmentDomain::Urban) => 0.32,
                    (TimeOfDay::Night, EnvironmentDomain::Remote) => 0.52,
                },
                exposure_ev: sampled_environment
                    .map(|environment| {
                        ((10.0 - environment.camera_exposure_ev100) * 0.08).clamp(-0.65, 0.65)
                    })
                    .unwrap_or(0.0),
                color_temperature_kelvin: sampled_environment
                    .map(|environment| environment.color_temperature_k)
                    .unwrap_or(6_500.0),
                shadow_softness: sampled_environment
                    .map(|environment| environment.shadow_softness)
                    .unwrap_or(0.25),
                ground_wetness: sampled_environment
                    .map(|environment| environment.ground_wetness)
                    .unwrap_or(0.0),
                seed,
                environment_yaw_radians: seed_to_yaw(seed),
            },
        );

        if let Some(facade) = &sampled.composition.facade {
            let glazing_scale = (facade.glazing_fraction / 0.25).clamp(0.45, 1.8);
            description.facade = FacadeDescription {
                window_sill_height_m: (sampled.building.wall_height_m * 0.18).clamp(0.55, 0.9),
                window_height_m: (sampled.building.wall_height_m * facade.glazing_fraction * 1.35)
                    .clamp(1.0, sampled.building.wall_height_m * 0.58),
                long_side_window_count: ((sampled.building.footprint_width_m / 2.5) * glazing_scale)
                    .round()
                    .clamp(1.0, 18.0) as u32,
                short_side_window_count: ((sampled.building.footprint_depth_m / 2.5)
                    * glazing_scale)
                    .round()
                    .clamp(1.0, 14.0) as u32,
                entrance_width_m: facade.entrance_width_m,
                entrance_height_m: 2.25_f32.min(sampled.building.wall_height_m - 0.35),
                entrance_canopy: true,
                entrance_side: map_facade_side(facade.entrance_side),
                weathering: facade.weathering,
                glazing_color_srgb: facade.glazing_material.base_color_srgb,
                glazing_roughness: facade.glazing_material.roughness,
                glazing_weathering: facade.glazing_material.weathering,
            };
        }

        description.building_extensions = sampled
            .composition
            .building_extensions
            .iter()
            .map(|extension| BuildingExtensionDescription {
                kind: match extension.kind {
                    synth_data::BuildingExtensionKind::DiningWing => {
                        BuildingExtensionKind::DiningWing
                    }
                    synth_data::BuildingExtensionKind::EntranceVestibule => {
                        BuildingExtensionKind::EntranceVestibule
                    }
                    synth_data::BuildingExtensionKind::ServiceAnnex => {
                        BuildingExtensionKind::ServiceAnnex
                    }
                },
                facade: map_facade_side(extension.facade),
                position: vec3_array(extension.position),
                size: vec3_array(extension.size_m),
                roof: match extension.roof {
                    synth_data::BuildingExtensionRoof::Flat => BuildingExtensionRoof::Flat,
                    synth_data::BuildingExtensionRoof::Shed => BuildingExtensionRoof::Shed,
                },
                roof_rise_m: extension.roof_rise_m,
            })
            .collect();

        description.signs = sampled
            .composition
            .signage
            .iter()
            .map(|sign| SignDescription {
                style: match sign.kind {
                    synth_data::SignageKind::PizzaHut => SignStyle::PizzaHut,
                    synth_data::SignageKind::RemovedGhost => SignStyle::RemovedGhost,
                    synth_data::SignageKind::RebrandedTenant => SignStyle::Tenant,
                },
                mount: SignMount::Facade,
                center: vec3_array(sign.center),
                size: [sign.size_m.x, sign.size_m.y],
                yaw_radians: match sign.facade {
                    synth_data::FacadeSide::Front | synth_data::FacadeSide::Back => 0.0,
                    synth_data::FacadeSide::Right | synth_data::FacadeSide::Left => {
                        std::f32::consts::FRAC_PI_2
                    }
                },
                emissive_strength: sign.emissive_strength,
                weathering: sign.weathering,
            })
            .collect();

        description.background_buildings = sampled
            .composition
            .background_buildings
            .iter()
            .map(|building| BackgroundBuilding {
                center: vec3_array(building.position),
                size: vec3_array(building.size_m),
                yaw_radians: building.yaw_degrees.to_radians(),
                kind: match building.kind {
                    synth_data::BackgroundBuildingKind::LowCommercial => {
                        BackgroundBuildingKind::LowCommercial
                    }
                    synth_data::BackgroundBuildingKind::MidriseMixedUse => {
                        BackgroundBuildingKind::MidriseMixedUse
                    }
                    synth_data::BackgroundBuildingKind::Residential => {
                        BackgroundBuildingKind::Residential
                    }
                    synth_data::BackgroundBuildingKind::IndustrialShed => {
                        BackgroundBuildingKind::IndustrialShed
                    }
                },
                base_color_srgb: building.material.base_color_srgb,
                weathering: building.material.weathering,
                roughness: building.material.roughness,
                surface_pattern: background_surface_pattern(&building.material.id, building.kind),
            })
            .collect();

        for vegetation in &sampled.composition.vegetation {
            description.trees.push(TreeInstance {
                position: vec3_array(vegetation.position),
                height_m: vegetation.height_m,
                crown_radius_m: vegetation.canopy_radius_m,
                kind: match vegetation.kind {
                    synth_data::VegetationKind::DeciduousTree => TreeKind::Deciduous,
                    synth_data::VegetationKind::EvergreenTree => TreeKind::Evergreen,
                    synth_data::VegetationKind::Palm => TreeKind::Palm,
                    synth_data::VegetationKind::Shrub => TreeKind::Shrub,
                },
                yaw_radians: vegetation.yaw_degrees.to_radians(),
                seasonal_variation: vegetation.seasonal_variation,
            });
        }

        description
            .occluders
            .extend(sampled.occluders.iter().map(|occluder| SceneOccluder {
                kind: match occluder.kind {
                    synth_data::OccluderKind::Vegetation => SceneOccluderKind::Vegetation,
                    synth_data::OccluderKind::Vehicle => SceneOccluderKind::Vehicle,
                    synth_data::OccluderKind::Pole => SceneOccluderKind::UtilityPole,
                    synth_data::OccluderKind::Sign => SceneOccluderKind::PoleSign,
                    synth_data::OccluderKind::Building => SceneOccluderKind::Building,
                    synth_data::OccluderKind::Pedestrian => SceneOccluderKind::Pedestrian,
                    synth_data::OccluderKind::RooftopEquipment => SceneOccluderKind::Hvac,
                },
                position: vec3_array(occluder.position),
                yaw_radians: occluder.yaw_degrees.to_radians(),
                scale: occluder.scale,
                nominal_size_m: vec3_array(occluder.nominal_size_m),
            }));

        if let Some(infrastructure) = &sampled.composition.infrastructure {
            let utility_poles = infrastructure
                .utility_poles
                .iter()
                .map(|pole| UtilityPoleDescription {
                    position: vec3_array(pole.position),
                    height_m: pole.height_m,
                    has_light: pole.has_light,
                })
                .collect::<Vec<_>>();
            let utility_lines = infrastructure
                .utility_lines
                .iter()
                .map(|line| {
                    let start = utility_poles
                        .get(usize::from(line.start_pole))
                        .ok_or(RenderError::InvalidSceneDescription)?;
                    let end = utility_poles
                        .get(usize::from(line.end_pole))
                        .ok_or(RenderError::InvalidSceneDescription)?;
                    Ok(UtilityLineDescription {
                        start: [
                            start.position[0],
                            start.position[1] + start.height_m * 0.91,
                            start.position[2],
                        ],
                        end: [
                            end.position[0],
                            end.position[1] + end.height_m * 0.91,
                            end.position[2],
                        ],
                        sag_m: line.sag_m,
                    })
                })
                .collect::<Result<Vec<_>, RenderError>>()?;
            description.site = SiteInfrastructure {
                parking_bays: infrastructure.parking_bays,
                parking_center: vec3_array(infrastructure.parking_center),
                parking_bay_size_m: [
                    infrastructure.parking_bay_size_m.x,
                    infrastructure.parking_bay_size_m.y,
                ],
                parking_yaw_radians: infrastructure.parking_yaw_degrees.to_radians(),
                road_lanes: infrastructure.road_lanes,
                road_kind: match infrastructure.road_kind {
                    synth_data::RoadKind::LocalStreet => SiteRoadKind::LocalStreet,
                    synth_data::RoadKind::UrbanArterial => SiteRoadKind::UrbanArterial,
                    synth_data::RoadKind::HighwayFrontage => SiteRoadKind::HighwayFrontage,
                    synth_data::RoadKind::RuralRoad => SiteRoadKind::RuralRoad,
                },
                lane_width_m: infrastructure.lane_width_m,
                road_center_z_m: infrastructure.road_center_z_m,
                raised_curb: infrastructure.raised_curb,
                utility_poles,
                utility_lines,
            };
        }

        let artificial_strength = sampled_environment
            .map(|environment| environment.artificial_light_strength)
            .unwrap_or(0.0);
        for sign in &description.signs {
            if emission_scale > 0.0 && sign.emissive_strength > 0.0 && description.lights.len() < 4
            {
                description.lights.push(SceneLight {
                    position: sign.center,
                    color: [1.0, 0.62, 0.32],
                    intensity: sign.emissive_strength,
                    range_m: (sign.size[0] * 2.5).max(5.0),
                });
            }
        }
        for pole in &description.site.utility_poles {
            if emission_scale > 0.0
                && pole.has_light
                && artificial_strength > 0.0
                && description.lights.len() < 4
            {
                description.lights.push(SceneLight {
                    position: [
                        pole.position[0] + 1.0,
                        pole.position[1] + pole.height_m - 0.25,
                        pole.position[2],
                    ],
                    color: [1.0, 0.78, 0.5],
                    intensity: artificial_strength * 3.0,
                    range_m: (pole.height_m * 2.4).max(10.0),
                });
            }
        }
        description.validate()?;
        Ok(description)
    }

    /// Checks dimensions, object transforms, and light limits before mesh assembly.
    pub fn validate(&self) -> Result<(), RenderError> {
        let dimensions = [
            self.footprint_width_m,
            self.footprint_depth_m,
            self.wall_height_m,
            self.ground_half_extent_m,
            self.facade.window_sill_height_m,
            self.facade.window_height_m,
            self.facade.entrance_width_m,
            self.facade.entrance_height_m,
        ];
        if dimensions
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
            || self.ground_half_extent_m < self.footprint_width_m.max(self.footprint_depth_m) * 0.5
            || self.facade.window_sill_height_m + self.facade.window_height_m >= self.wall_height_m
            || self.facade.entrance_height_m >= self.wall_height_m
            || self.facade.long_side_window_count == 0
            || self.facade.short_side_window_count == 0
            || !(0.0..=1.0).contains(&self.facade.weathering)
            || self
                .facade
                .glazing_color_srgb
                .iter()
                .any(|value| !value.is_finite() || *value < 0.0)
            || !(0.0..=1.0).contains(&self.facade.glazing_roughness)
            || !(0.0..=1.0).contains(&self.facade.glazing_weathering)
            || !self.environment.is_valid()
            || !self.materials.is_valid()
            || !self.site.is_valid()
            || self.building_extensions.iter().any(|extension| {
                extension
                    .position
                    .iter()
                    .chain(extension.size.iter())
                    .chain([extension.roof_rise_m].iter())
                    .any(|value| !value.is_finite())
                    || extension.position[1] != 0.0
                    || extension.size.iter().any(|value| *value <= 0.0)
                    || extension.size[1] + extension.roof_rise_m >= self.wall_height_m
                    || extension.roof_rise_m < 0.0
                    || match extension.roof {
                        BuildingExtensionRoof::Flat => extension.roof_rise_m != 0.0,
                        BuildingExtensionRoof::Shed => extension.roof_rise_m <= 0.0,
                    }
            })
            || self.signs.iter().any(|sign| {
                sign.center
                    .iter()
                    .chain(sign.size.iter())
                    .chain([sign.yaw_radians, sign.emissive_strength, sign.weathering].iter())
                    .any(|value| !value.is_finite())
                    || sign.size.iter().any(|value| *value <= 0.0)
                    || sign.emissive_strength < 0.0
                    || !(0.0..=1.0).contains(&sign.weathering)
            })
            || self.background_buildings.iter().any(|building| {
                building
                    .center
                    .iter()
                    .chain(building.size.iter())
                    .chain(building.base_color_srgb.iter())
                    .chain(
                        [
                            building.yaw_radians,
                            building.weathering,
                            building.roughness,
                        ]
                        .iter(),
                    )
                    .any(|value| !value.is_finite())
                    || building.size.iter().any(|value| *value <= 0.0)
                    || building.base_color_srgb.iter().any(|value| *value < 0.0)
                    || !(0.0..=1.0).contains(&building.weathering)
                    || !(0.0..=1.0).contains(&building.roughness)
                    || !building.surface_pattern.is_known()
            })
            || self.trees.iter().any(|tree| {
                tree.position
                    .iter()
                    .chain(
                        [
                            tree.height_m,
                            tree.crown_radius_m,
                            tree.yaw_radians,
                            tree.seasonal_variation,
                        ]
                        .iter(),
                    )
                    .any(|value| !value.is_finite())
                    || tree.height_m <= 0.0
                    || tree.crown_radius_m <= 0.0
                    || !(-1.0..=1.0).contains(&tree.seasonal_variation)
            })
            || self.occluders.iter().any(|occluder| {
                occluder
                    .position
                    .iter()
                    .chain(occluder.nominal_size_m.iter())
                    .chain([occluder.yaw_radians, occluder.scale].iter())
                    .any(|value| !value.is_finite())
                    || occluder.scale <= 0.0
                    || occluder.nominal_size_m.iter().any(|value| *value < 0.0)
                    || !(occluder.nominal_size_m.iter().all(|value| *value == 0.0)
                        || occluder.nominal_size_m.iter().all(|value| *value > 0.0))
            })
            || self.lights.len() > 4
            || self.lights.iter().any(|light| !light.is_valid())
        {
            return Err(RenderError::InvalidSceneDescription);
        }
        Ok(())
    }

    fn populate_context(&mut self) {
        let g = self.ground_half_extent_m;
        let h = self.wall_height_m;
        self.site = SiteInfrastructure::for_scene(
            self.footprint_width_m,
            self.footprint_depth_m,
            self.ground_half_extent_m,
        );
        match self.environment.domain {
            EnvironmentDomain::City => {
                self.site.parking_bays = 4;
                self.site.road_lanes = 4;
                self.site.road_kind = SiteRoadKind::UrbanArterial;
                self.site.utility_poles = vec![
                    utility_pole([-0.48 * g, 0.0, -0.62 * g], 6.2, true),
                    utility_pole([0.42 * g, 0.0, -0.62 * g], 6.2, true),
                ];
            }
            EnvironmentDomain::Urban => {
                self.site.parking_bays = 10;
                self.site.road_lanes = 4;
                self.site.road_kind = SiteRoadKind::UrbanArterial;
                self.site.utility_poles = vec![
                    utility_pole([-0.7 * g, 0.0, -0.62 * g], 7.5, false),
                    utility_pole([0.0, 0.0, -0.62 * g], 7.5, true),
                    utility_pole([0.7 * g, 0.0, -0.62 * g], 7.5, false),
                ];
                self.site.utility_lines = adjacent_utility_lines(&self.site.utility_poles, 0.7);
            }
            EnvironmentDomain::Remote => {
                self.site.parking_bays = 5;
                self.site.road_lanes = 2;
                self.site.road_kind = SiteRoadKind::RuralRoad;
                self.site.raised_curb = false;
                self.site.utility_poles = vec![
                    utility_pole([-0.62 * g, 0.0, -0.68 * g], 7.8, false),
                    utility_pole([0.62 * g, 0.0, -0.68 * g], 7.8, false),
                ];
                self.site.utility_lines = adjacent_utility_lines(&self.site.utility_poles, 1.1);
            }
        }
        self.background_buildings = match self.environment.domain {
            EnvironmentDomain::City => vec![
                background([-0.68 * g, 0.0, 0.62 * g], [12.0, 16.0, 9.0], 0.08),
                background([0.62 * g, 0.0, 0.66 * g], [15.0, 22.0, 11.0], -0.12),
                background([-0.72 * g, 0.0, -0.28 * g], [10.0, 11.0, 8.0], 0.04),
                background([0.72 * g, 0.0, -0.18 * g], [13.0, 14.0, 10.0], -0.06),
                background([-0.26 * g, 0.0, 0.78 * g], [14.0, 9.0, 7.0], 0.0),
                background([0.27 * g, 0.0, 0.82 * g], [11.0, 12.0, 8.0], 0.03),
            ],
            EnvironmentDomain::Urban => vec![
                background([-0.62 * g, 0.0, 0.65 * g], [13.0, 5.5, 8.0], 0.08),
                background([0.57 * g, 0.0, 0.7 * g], [17.0, 6.5, 10.0], -0.08),
                background([-0.72 * g, 0.0, -0.18 * g], [10.0, 4.5, 8.0], 0.03),
            ],
            EnvironmentDomain::Remote => Vec::new(),
        };

        self.trees = match self.environment.domain {
            EnvironmentDomain::City => vec![
                tree([-0.45 * g, 0.0, -0.5 * g], 7.0, 2.1),
                tree([0.5 * g, 0.0, 0.36 * g], 6.0, 1.8),
            ],
            EnvironmentDomain::Urban => vec![
                tree([-0.48 * g, 0.0, -0.46 * g], 8.0, 2.5),
                tree([0.48 * g, 0.0, 0.35 * g], 6.5, 2.0),
                tree([-0.36 * g, 0.0, 0.52 * g], 7.2, 2.2),
                tree([0.62 * g, 0.0, -0.2 * g], 5.8, 1.8),
            ],
            EnvironmentDomain::Remote => vec![
                tree([-0.36 * g, 0.0, -0.42 * g], 10.0, 3.2),
                tree([0.42 * g, 0.0, -0.35 * g], 8.5, 2.8),
                tree([-0.55 * g, 0.0, 0.14 * g], 11.0, 3.5),
                tree([0.58 * g, 0.0, 0.24 * g], 9.2, 3.0),
                tree([-0.22 * g, 0.0, 0.62 * g], 8.0, 2.5),
                tree([0.28 * g, 0.0, 0.7 * g], 9.5, 3.1),
            ],
        };

        self.occluders = vec![
            occluder(
                SceneOccluderKind::Vehicle,
                [-0.28 * g, 0.0, -0.32 * g],
                0.08,
                1.0,
            ),
            occluder(
                SceneOccluderKind::PoleSign,
                [0.38 * g, 0.0, -0.46 * g],
                0.0,
                1.0,
            ),
            occluder(
                SceneOccluderKind::Shrub,
                [
                    -0.28 * self.footprint_width_m,
                    0.0,
                    -0.55 * self.footprint_depth_m,
                ],
                0.0,
                1.0,
            ),
            occluder(
                SceneOccluderKind::Shrub,
                [
                    0.28 * self.footprint_width_m,
                    0.0,
                    -0.55 * self.footprint_depth_m,
                ],
                0.0,
                1.15,
            ),
            occluder(
                SceneOccluderKind::Hvac,
                [0.18 * self.footprint_width_m, h + 0.6, 0.05],
                0.1,
                0.8,
            ),
        ];
        let sign_width = (self.footprint_width_m * 0.24).clamp(2.8, 5.5);
        let sign_height = (self.wall_height_m * 0.18).clamp(0.55, 1.0);
        self.signs = vec![SignDescription {
            style: SignStyle::PizzaHut,
            mount: SignMount::Facade,
            center: [
                0.0,
                self.wall_height_m - sign_height * 0.75,
                -self.footprint_depth_m * 0.5 - 0.055,
            ],
            size: [sign_width, sign_height],
            yaw_radians: 0.0,
            emissive_strength: match self.environment.time_of_day {
                TimeOfDay::Day => 0.0,
                TimeOfDay::Twilight => 0.6,
                TimeOfDay::Night => 1.4,
            },
            weathering: 0.15,
        }];
        if self.environment.time_of_day != TimeOfDay::Day {
            self.lights = vec![
                SceneLight {
                    position: [0.0, 2.7, -self.footprint_depth_m * 0.58],
                    color: [1.0, 0.68, 0.34],
                    intensity: 3.0,
                    range_m: 8.0,
                },
                SceneLight {
                    position: [0.22 * g, 5.2, -0.55 * g],
                    color: [1.0, 0.82, 0.58],
                    intensity: 5.0,
                    range_m: 13.0,
                },
            ];
        }
    }
}

impl RenderMesh {
    /// Bakes a semantic roof and complete procedural context into one aligned draw mesh.
    pub fn from_scene(
        roof: &roof_geometry::RoofGeometry,
        description: &SceneDescription,
    ) -> Result<Self, RenderError> {
        description.validate()?;
        let mut mesh = Self {
            environment: description.environment,
            materials: description.materials,
            lights: description.lights.clone(),
            ..Self::default()
        };
        append_roof(&mut mesh, roof, description.wall_height_m);
        append_building(&mut mesh, description);
        for extension in &description.building_extensions {
            append_building_extension(&mut mesh, *extension);
        }
        append_ground_context(&mut mesh, description);
        for building in &description.background_buildings {
            append_background_building(&mut mesh, *building);
        }
        for tree in &description.trees {
            append_tree(&mut mesh, *tree);
        }
        for occluder in &description.occluders {
            append_occluder(&mut mesh, *occluder);
        }
        for sign in &description.signs {
            append_sign(&mut mesh, *sign);
        }
        mesh.validate()?;
        Ok(mesh)
    }

    /// Bakes a complete negative scene whose primary building has a genuine
    /// ordinary roof. All of its vertices use semantic ID zero, so an ordinary
    /// roof can never leak target masks or target geometry labels.
    pub fn from_ordinary_scene(
        roof: synth_data::SampledOrdinaryRoof,
        description: &SceneDescription,
    ) -> Result<Self, RenderError> {
        description.validate()?;
        let mut mesh = Self {
            environment: description.environment,
            materials: description.materials,
            lights: description.lights.clone(),
            ..Self::default()
        };
        append_ordinary_roof(&mut mesh, roof, description.wall_height_m);
        append_building(&mut mesh, description);
        for extension in &description.building_extensions {
            append_building_extension(&mut mesh, *extension);
        }
        append_ground_context(&mut mesh, description);
        for building in &description.background_buildings {
            append_background_building(&mut mesh, *building);
        }
        for tree in &description.trees {
            append_tree(&mut mesh, *tree);
        }
        for occluder in &description.occluders {
            append_occluder(&mut mesh, *occluder);
        }
        for sign in &description.signs {
            append_sign(&mut mesh, *sign);
        }
        mesh.validate()?;
        Ok(mesh)
    }
}

fn append_ordinary_roof(
    mesh: &mut RenderMesh,
    roof: synth_data::SampledOrdinaryRoof,
    wall_height: f32,
) {
    use synth_data::OrdinaryRoofFamily;

    let x = roof.eave_width_m * 0.5;
    let z = roof.eave_depth_m * 0.5;
    let y = wall_height;
    let top = y + roof.rise_m;
    match roof.family {
        OrdinaryRoofFamily::Flat => append_box(
            mesh,
            [0.0, y + roof.rise_m * 0.5, 0.0],
            [roof.eave_width_m, roof.rise_m, roof.eave_depth_m],
            0.0,
            MaterialSlot::ROOF,
        ),
        OrdinaryRoofFamily::Gable => {
            append_quad(
                mesh,
                [[-x, y, -z], [x, y, -z], [x, top, 0.0], [-x, top, 0.0]],
                MaterialSlot::ROOF,
            );
            append_quad(
                mesh,
                [[x, y, z], [-x, y, z], [-x, top, 0.0], [x, top, 0.0]],
                MaterialSlot::ROOF,
            );
            append_unlabelled_triangle(
                mesh,
                [[-x, y, z], [-x, y, -z], [-x, top, 0.0]],
                MaterialSlot::ROOF,
            );
            append_unlabelled_triangle(
                mesh,
                [[x, y, -z], [x, y, z], [x, top, 0.0]],
                MaterialSlot::ROOF,
            );
        }
        OrdinaryRoofFamily::Hip => {
            let ridge_x = x * roof.ridge_length_fraction;
            append_quad(
                mesh,
                [
                    [-x, y, -z],
                    [x, y, -z],
                    [ridge_x, top, 0.0],
                    [-ridge_x, top, 0.0],
                ],
                MaterialSlot::ROOF,
            );
            append_quad(
                mesh,
                [
                    [x, y, z],
                    [-x, y, z],
                    [-ridge_x, top, 0.0],
                    [ridge_x, top, 0.0],
                ],
                MaterialSlot::ROOF,
            );
            append_unlabelled_triangle(
                mesh,
                [[-x, y, z], [-x, y, -z], [-ridge_x, top, 0.0]],
                MaterialSlot::ROOF,
            );
            append_unlabelled_triangle(
                mesh,
                [[x, y, -z], [x, y, z], [ridge_x, top, 0.0]],
                MaterialSlot::ROOF,
            );
        }
        OrdinaryRoofFamily::Shed => {
            append_quad(
                mesh,
                [[-x, y, -z], [x, y, -z], [x, top, z], [-x, top, z]],
                MaterialSlot::ROOF,
            );
            append_quad(
                mesh,
                [[x, y, z], [-x, y, z], [-x, top, z], [x, top, z]],
                MaterialSlot::ROOF,
            );
            append_unlabelled_triangle(
                mesh,
                [[-x, y, z], [-x, y, -z], [-x, top, z]],
                MaterialSlot::ROOF,
            );
            append_unlabelled_triangle(
                mesh,
                [[x, y, -z], [x, y, z], [x, top, z]],
                MaterialSlot::ROOF,
            );
        }
        OrdinaryRoofFamily::Mansard => {
            let inset_x = x * roof.inset_fraction;
            let inset_z = z * roof.inset_fraction;
            append_frustum_sides(mesh, x, z, y, inset_x, inset_z, top);
            append_quad(
                mesh,
                [
                    [-inset_x, top, -inset_z],
                    [-inset_x, top, inset_z],
                    [inset_x, top, inset_z],
                    [inset_x, top, -inset_z],
                ],
                MaterialSlot::ROOF,
            );
        }
        OrdinaryRoofFamily::Pyramid => append_pyramid(mesh, x, z, y, top),
        OrdinaryRoofFamily::Cupola => {
            append_pyramid(mesh, x, z, y, top);
            let cupola_x = x * roof.inset_fraction.clamp(0.12, 0.32);
            let cupola_z = z * roof.inset_fraction.clamp(0.12, 0.32);
            let body_height = roof.cap_height_m * 0.62;
            append_box(
                mesh,
                [0.0, top + body_height * 0.5, 0.0],
                [cupola_x * 2.0, body_height, cupola_z * 2.0],
                0.0,
                MaterialSlot::ROOF,
            );
            append_pyramid(
                mesh,
                cupola_x * 1.18,
                cupola_z * 1.18,
                top + body_height,
                top + roof.cap_height_m,
            );
        }
    }
}

fn append_frustum_sides(
    mesh: &mut RenderMesh,
    lower_x: f32,
    lower_z: f32,
    lower_y: f32,
    upper_x: f32,
    upper_z: f32,
    upper_y: f32,
) {
    for face in [
        [
            [-lower_x, lower_y, -lower_z],
            [lower_x, lower_y, -lower_z],
            [upper_x, upper_y, -upper_z],
            [-upper_x, upper_y, -upper_z],
        ],
        [
            [lower_x, lower_y, lower_z],
            [-lower_x, lower_y, lower_z],
            [-upper_x, upper_y, upper_z],
            [upper_x, upper_y, upper_z],
        ],
        [
            [-lower_x, lower_y, lower_z],
            [-lower_x, lower_y, -lower_z],
            [-upper_x, upper_y, -upper_z],
            [-upper_x, upper_y, upper_z],
        ],
        [
            [lower_x, lower_y, -lower_z],
            [lower_x, lower_y, lower_z],
            [upper_x, upper_y, upper_z],
            [upper_x, upper_y, -upper_z],
        ],
    ] {
        append_quad(mesh, face, MaterialSlot::ROOF);
    }
}

fn append_pyramid(mesh: &mut RenderMesh, x: f32, z: f32, y: f32, top: f32) {
    for triangle in [
        [[-x, y, -z], [x, y, -z], [0.0, top, 0.0]],
        [[x, y, z], [-x, y, z], [0.0, top, 0.0]],
        [[-x, y, z], [-x, y, -z], [0.0, top, 0.0]],
        [[x, y, -z], [x, y, z], [0.0, top, 0.0]],
    ] {
        append_unlabelled_triangle(mesh, triangle, MaterialSlot::ROOF);
    }
}

fn append_unlabelled_triangle(
    mesh: &mut RenderMesh,
    positions: [[f32; 3]; 3],
    material: MaterialSlot,
) {
    let a = Vec3::from_array(positions[0]);
    let b = Vec3::from_array(positions[1]);
    let c = Vec3::from_array(positions[2]);
    let normal = (b - a).cross(c - a).normalize_or_zero().to_array();
    let base = mesh.vertices.len() as u32;
    for (position, face_coord) in positions
        .into_iter()
        .zip([[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]])
    {
        mesh.vertices.push(RenderVertex {
            position,
            normal,
            face_coord,
            semantic_id: 0,
            material,
            appearance: [0.0; 4],
            pattern: SurfacePattern::INHERIT,
        });
    }
    mesh.indices.extend_from_slice(&[base, base + 1, base + 2]);
}

fn append_background_building(mesh: &mut RenderMesh, building: BackgroundBuilding) {
    let weather = building.weathering;
    let weathered = building
        .base_color_srgb
        .map(|channel| channel + (0.48 - channel) * weather * 0.28);
    let tint = [
        srgb_to_linear(weathered[0]),
        srgb_to_linear(weathered[1]),
        srgb_to_linear(weathered[2]),
        building.roughness,
    ];
    let base_center = [
        building.center[0],
        building.center[1] + building.size[1] * 0.5,
        building.center[2],
    ];
    match building.kind {
        BackgroundBuildingKind::LowCommercial => append_background_box(
            mesh,
            base_center,
            building.size,
            building.yaw_radians,
            tint,
            building.surface_pattern,
        ),
        BackgroundBuildingKind::MidriseMixedUse => {
            append_background_box(
                mesh,
                base_center,
                building.size,
                building.yaw_radians,
                tint,
                building.surface_pattern,
            );
            append_box(
                mesh,
                [
                    building.center[0],
                    building.center[1] + building.size[1] + 0.35,
                    building.center[2],
                ],
                [building.size[0] * 0.32, 0.7, building.size[2] * 0.28],
                building.yaw_radians,
                MaterialSlot::METAL,
            );
        }
        BackgroundBuildingKind::Residential => {
            let wall_height = building.size[1] * 0.72;
            append_background_box(
                mesh,
                [
                    building.center[0],
                    building.center[1] + wall_height * 0.5,
                    building.center[2],
                ],
                [building.size[0], wall_height, building.size[2]],
                building.yaw_radians,
                tint,
                building.surface_pattern,
            );
            append_pitched_roof(mesh, building, wall_height, tint);
        }
        BackgroundBuildingKind::IndustrialShed => {
            append_background_box(
                mesh,
                base_center,
                building.size,
                building.yaw_radians,
                tint,
                building.surface_pattern,
            );
            append_box_tinted(
                mesh,
                [
                    building.center[0],
                    building.center[1] + building.size[1] + 0.08,
                    building.center[2],
                ],
                [building.size[0] + 0.5, 0.16, building.size[2] + 0.5],
                building.yaw_radians,
                MaterialSlot::METAL,
                tint,
            );
        }
    }
}

fn append_background_box(
    mesh: &mut RenderMesh,
    center: [f32; 3],
    size: [f32; 3],
    yaw: f32,
    appearance: [f32; 4],
    wall_pattern: SurfacePattern,
) {
    let half = Vec3::from_array(size) * 0.5;
    let rotation = Quat::from_rotation_y(yaw);
    let center = Vec3::from_array(center);
    let transform = |point: [f32; 3]| (rotation * Vec3::from_array(point) + center).to_array();
    let (x, y, z) = (half.x, half.y, half.z);
    for face in [
        [[x, -y, -z], [-x, -y, -z], [-x, y, -z], [x, y, -z]],
        [[x, -y, z], [x, -y, -z], [x, y, -z], [x, y, z]],
        [[-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]],
        [[-x, -y, -z], [-x, -y, z], [-x, y, z], [-x, y, -z]],
    ] {
        append_quad_tinted_pattern(
            mesh,
            face.map(transform),
            MaterialSlot::BACKGROUND_WALL,
            appearance,
            wall_pattern,
        );
    }
    let cap_pattern = background_cap_pattern(wall_pattern);
    for face in [
        [[-x, y, -z], [-x, y, z], [x, y, z], [x, y, -z]],
        [[-x, -y, z], [-x, -y, -z], [x, -y, -z], [x, -y, z]],
    ] {
        append_quad_tinted_pattern(
            mesh,
            face.map(transform),
            MaterialSlot::BACKGROUND_WALL,
            appearance,
            cap_pattern,
        );
    }
}

fn background_cap_pattern(pattern: SurfacePattern) -> SurfacePattern {
    if pattern == SurfacePattern::BACKGROUND_WINDOWS_BRICK {
        SurfacePattern::BRICK
    } else if pattern == SurfacePattern::BACKGROUND_WINDOWS_CLADDING {
        SurfacePattern::VERTICAL_CLADDING
    } else {
        SurfacePattern::SMOOTH
    }
}

fn append_pitched_roof(
    mesh: &mut RenderMesh,
    building: BackgroundBuilding,
    wall_height: f32,
    tint: [f32; 4],
) {
    let half_x = building.size[0] * 0.5;
    let half_z = building.size[2] * 0.5;
    let eave_y = wall_height;
    let ridge_y = building.size[1];
    let rotation = Quat::from_rotation_y(building.yaw_radians);
    let origin = Vec3::from_array(building.center);
    let transform = |point: [f32; 3]| (rotation * Vec3::from_array(point) + origin).to_array();
    append_quad_tinted(
        mesh,
        [
            [-half_x, eave_y, -half_z],
            [half_x, eave_y, -half_z],
            [half_x, ridge_y, 0.0],
            [-half_x, ridge_y, 0.0],
        ]
        .map(transform),
        MaterialSlot::METAL,
        tint,
    );
    append_quad_tinted(
        mesh,
        [
            [half_x, eave_y, half_z],
            [-half_x, eave_y, half_z],
            [-half_x, ridge_y, 0.0],
            [half_x, ridge_y, 0.0],
        ]
        .map(transform),
        MaterialSlot::METAL,
        tint,
    );
}

fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn append_roof(mesh: &mut RenderMesh, roof: &roof_geometry::RoofGeometry, wall_height: f32) {
    let base = mesh.vertices.len() as u32;
    mesh.vertices
        .extend(roof.mesh.vertices.iter().map(|vertex| RenderVertex {
            position: [
                vertex.position[0],
                vertex.position[1] + wall_height,
                vertex.position[2],
            ],
            normal: vertex.normal,
            face_coord: vertex.face_coord,
            semantic_id: vertex.face_id.as_u32(),
            material: MaterialSlot::ROOF,
            appearance: [0.0; 4],
            pattern: SurfacePattern::INHERIT,
        }));
    mesh.indices
        .extend(roof.mesh.indices.iter().map(|index| base + index));
}

fn append_building_extension(mesh: &mut RenderMesh, extension: BuildingExtensionDescription) {
    let center = [
        extension.position[0],
        extension.size[1] * 0.5,
        extension.position[2],
    ];
    append_box(mesh, center, extension.size, 0.0, MaterialSlot::WALL);

    let half_x = extension.size[0] * 0.5;
    let half_z = extension.size[2] * 0.5;
    let min_x = extension.position[0] - half_x;
    let max_x = extension.position[0] + half_x;
    let min_z = extension.position[2] - half_z;
    let max_z = extension.position[2] + half_z;
    let roof_base = extension.size[1] + 0.06;
    match extension.roof {
        BuildingExtensionRoof::Flat => append_box(
            mesh,
            [extension.position[0], roof_base, extension.position[2]],
            [extension.size[0] + 0.28, 0.14, extension.size[2] + 0.28],
            0.0,
            MaterialSlot::ROOF,
        ),
        BuildingExtensionRoof::Shed => {
            let outer = roof_base;
            let inner = roof_base + extension.roof_rise_m;
            let (y_min_x, y_max_x, y_min_z, y_max_z) = match extension.facade {
                BuildingSide::Front => (outer, outer, outer, inner),
                BuildingSide::Back => (outer, outer, inner, outer),
                BuildingSide::Right => (inner, outer, outer, outer),
                BuildingSide::Left => (outer, inner, outer, outer),
            };
            let corner_y = |x_is_max: bool, z_is_max: bool| {
                if matches!(extension.facade, BuildingSide::Right | BuildingSide::Left) {
                    if x_is_max { y_max_x } else { y_min_x }
                } else if z_is_max {
                    y_max_z
                } else {
                    y_min_z
                }
            };
            append_quad(
                mesh,
                [
                    [min_x - 0.12, corner_y(false, false), min_z - 0.12],
                    [min_x - 0.12, corner_y(false, true), max_z + 0.12],
                    [max_x + 0.12, corner_y(true, true), max_z + 0.12],
                    [max_x + 0.12, corner_y(true, false), min_z - 0.12],
                ],
                MaterialSlot::ROOF,
            );
        }
    }

    if extension.kind == BuildingExtensionKind::ServiceAnnex {
        return;
    }
    let sill = if extension.kind == BuildingExtensionKind::EntranceVestibule {
        0.18
    } else {
        0.62
    };
    let top = (extension.size[1] - 0.35).max(sill + 0.8);
    let inset = if extension.kind == BuildingExtensionKind::EntranceVestibule {
        0.28
    } else {
        0.5
    };
    let epsilon = 0.035;
    let panel = match extension.facade {
        BuildingSide::Front => [
            [max_x - inset, sill, min_z - epsilon],
            [min_x + inset, sill, min_z - epsilon],
            [min_x + inset, top, min_z - epsilon],
            [max_x - inset, top, min_z - epsilon],
        ],
        BuildingSide::Back => [
            [min_x + inset, sill, max_z + epsilon],
            [max_x - inset, sill, max_z + epsilon],
            [max_x - inset, top, max_z + epsilon],
            [min_x + inset, top, max_z + epsilon],
        ],
        BuildingSide::Right => [
            [max_x + epsilon, sill, max_z - inset],
            [max_x + epsilon, sill, min_z + inset],
            [max_x + epsilon, top, min_z + inset],
            [max_x + epsilon, top, max_z - inset],
        ],
        BuildingSide::Left => [
            [min_x - epsilon, sill, min_z + inset],
            [min_x - epsilon, sill, max_z - inset],
            [min_x - epsilon, top, max_z - inset],
            [min_x - epsilon, top, min_z + inset],
        ],
    };
    append_quad(mesh, panel, MaterialSlot::GLASS);
}

fn append_building(mesh: &mut RenderMesh, scene: &SceneDescription) {
    let x = scene.footprint_width_m * 0.5;
    let z = scene.footprint_depth_m * 0.5;
    let y = scene.wall_height_m;
    let wall_appearance = [0.0, 0.0, 0.0, scene.facade.weathering];
    append_quad_tinted(
        mesh,
        [[x, 0.0, -z], [-x, 0.0, -z], [-x, y, -z], [x, y, -z]],
        MaterialSlot::WALL,
        wall_appearance,
    );
    append_quad_tinted(
        mesh,
        [[x, 0.0, z], [x, 0.0, -z], [x, y, -z], [x, y, z]],
        MaterialSlot::WALL,
        wall_appearance,
    );
    append_quad_tinted(
        mesh,
        [[-x, 0.0, z], [x, 0.0, z], [x, y, z], [-x, y, z]],
        MaterialSlot::WALL,
        wall_appearance,
    );
    append_quad_tinted(
        mesh,
        [[-x, 0.0, -z], [-x, 0.0, z], [-x, y, z], [-x, y, -z]],
        MaterialSlot::WALL,
        wall_appearance,
    );

    append_window_band(mesh, scene, Side::Front);
    append_window_band(mesh, scene, Side::Back);
    append_window_band(mesh, scene, Side::Left);
    append_window_band(mesh, scene, Side::Right);

    append_entrance(mesh, scene);
    for corner in [[x, -z], [-x, -z], [x, z], [-x, z]] {
        append_box(
            mesh,
            [corner[0], y * 0.5, corner[1]],
            [0.16, y, 0.16],
            0.0,
            MaterialSlot::TRIM,
        );
    }
    append_box(
        mesh,
        [0.0, y - 0.2, -z - 0.03],
        [scene.footprint_width_m, 0.22, 0.12],
        0.0,
        MaterialSlot::TRIM,
    );
}

fn append_entrance(mesh: &mut RenderMesh, scene: &SceneDescription) {
    let x = scene.footprint_width_m * 0.5;
    let z = scene.footprint_depth_m * 0.5;
    let half_width = scene.facade.entrance_width_m * 0.5;
    let top = scene.facade.entrance_height_m;
    let epsilon = 0.035;
    let door = match scene.facade.entrance_side {
        BuildingSide::Front => [
            [half_width, 0.02, -z - epsilon],
            [-half_width, 0.02, -z - epsilon],
            [-half_width, top, -z - epsilon],
            [half_width, top, -z - epsilon],
        ],
        BuildingSide::Back => [
            [-half_width, 0.02, z + epsilon],
            [half_width, 0.02, z + epsilon],
            [half_width, top, z + epsilon],
            [-half_width, top, z + epsilon],
        ],
        BuildingSide::Right => [
            [x + epsilon, 0.02, half_width],
            [x + epsilon, 0.02, -half_width],
            [x + epsilon, top, -half_width],
            [x + epsilon, top, half_width],
        ],
        BuildingSide::Left => [
            [-x - epsilon, 0.02, -half_width],
            [-x - epsilon, 0.02, half_width],
            [-x - epsilon, top, half_width],
            [-x - epsilon, top, -half_width],
        ],
    };
    append_quad_tinted(
        mesh,
        door,
        MaterialSlot::GLASS,
        facade_glass_appearance(&scene.facade),
    );
    if !scene.facade.entrance_canopy {
        return;
    }
    let outward = 0.65;
    let canopy_width = scene.facade.entrance_width_m + 1.0;
    let (center, size) = match scene.facade.entrance_side {
        BuildingSide::Front => ([0.0, top + 0.18, -z - outward], [canopy_width, 0.18, 1.25]),
        BuildingSide::Back => ([0.0, top + 0.18, z + outward], [canopy_width, 0.18, 1.25]),
        BuildingSide::Right => ([x + outward, top + 0.18, 0.0], [1.25, 0.18, canopy_width]),
        BuildingSide::Left => ([-x - outward, top + 0.18, 0.0], [1.25, 0.18, canopy_width]),
    };
    append_box(mesh, center, size, 0.0, MaterialSlot::TRIM);
}

fn append_sign(mesh: &mut RenderMesh, sign: SignDescription) {
    let material = match sign.style {
        SignStyle::PizzaHut => MaterialSlot::SIGN,
        SignStyle::RemovedGhost | SignStyle::Blank => MaterialSlot::GHOST_SIGN,
        SignStyle::Tenant => MaterialSlot::TENANT_SIGN,
    };
    let panel_depth = if sign.mount == SignMount::Facade {
        0.08
    } else {
        0.18
    };
    let appearance = match sign.style {
        SignStyle::PizzaHut => {
            let fade = 1.0 - sign.weathering * 0.55;
            [
                0.78 * fade,
                0.025 * fade,
                0.018 * fade,
                sign.emissive_strength,
            ]
        }
        SignStyle::Tenant => {
            let fade = 1.0 - sign.weathering * 0.45;
            [
                0.035 * fade,
                0.22 * fade,
                0.42 * fade,
                sign.emissive_strength,
            ]
        }
        SignStyle::RemovedGhost | SignStyle::Blank => [0.0, 0.0, 0.0, sign.weathering],
    };
    let pattern = if sign.style == SignStyle::PizzaHut {
        SurfacePattern::PIZZA_HUT_SIGN
    } else {
        SurfacePattern::GENERIC_SIGN
    };
    append_panel_tinted_pattern(
        mesh,
        sign.center,
        [sign.size[0], sign.size[1], panel_depth],
        sign.yaw_radians,
        material,
        appearance,
        pattern,
    );
    if sign.mount == SignMount::Pole {
        let base = [sign.center[0], 0.0, sign.center[2]];
        append_cylinder(
            mesh,
            base,
            (sign.size[0] * 0.04).clamp(0.08, 0.2),
            (sign.center[1] - sign.size[1] * 0.5).max(0.1),
            8,
            MaterialSlot::METAL,
        );
    }
}

fn append_panel_tinted_pattern(
    mesh: &mut RenderMesh,
    center: [f32; 3],
    size: [f32; 3],
    yaw: f32,
    face_material: MaterialSlot,
    appearance: [f32; 4],
    pattern: SurfacePattern,
) {
    let half = Vec3::from_array(size) * 0.5;
    let rotation = Quat::from_rotation_y(yaw);
    let center = Vec3::from_array(center);
    let transform = |point: [f32; 3]| (rotation * Vec3::from_array(point) + center).to_array();
    let (x, y, z) = (half.x, half.y, half.z);
    for face in [
        [[x, -y, -z], [-x, -y, -z], [-x, y, -z], [x, y, -z]],
        [[-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]],
    ] {
        append_quad_tinted_pattern(
            mesh,
            face.map(transform),
            face_material,
            appearance,
            pattern,
        );
    }
    for face in [
        [[x, -y, z], [x, -y, -z], [x, y, -z], [x, y, z]],
        [[-x, -y, -z], [-x, -y, z], [-x, y, z], [-x, y, -z]],
        [[-x, y, -z], [-x, y, z], [x, y, z], [x, y, -z]],
        [[-x, -y, z], [-x, -y, -z], [x, -y, -z], [x, -y, z]],
    ] {
        append_quad_tinted_pattern(
            mesh,
            face.map(transform),
            MaterialSlot::TRIM,
            [0.0; 4],
            SurfacePattern::SMOOTH,
        );
    }
}

#[derive(Clone, Copy)]
enum Side {
    Front,
    Back,
    Left,
    Right,
}

fn append_window_band(mesh: &mut RenderMesh, scene: &SceneDescription, side: Side) {
    let x = scene.footprint_width_m * 0.5;
    let z = scene.footprint_depth_m * 0.5;
    let sill = scene.facade.window_sill_height_m;
    let top = sill + scene.facade.window_height_m;
    let inset = 0.065;
    let margin = 0.08;
    let (positions, count, span, vertical) = match side {
        Side::Front => (
            [
                [x - margin, sill, -z - inset],
                [-x + margin, sill, -z - inset],
                [-x + margin, top, -z - inset],
                [x - margin, top, -z - inset],
            ],
            scene.facade.long_side_window_count,
            scene.footprint_width_m - margin * 2.0,
            false,
        ),
        Side::Back => (
            [
                [-x + margin, sill, z + inset],
                [x - margin, sill, z + inset],
                [x - margin, top, z + inset],
                [-x + margin, top, z + inset],
            ],
            scene.facade.long_side_window_count,
            scene.footprint_width_m - margin * 2.0,
            false,
        ),
        Side::Left => (
            [
                [-x - inset, sill, -z + margin],
                [-x - inset, sill, z - margin],
                [-x - inset, top, z - margin],
                [-x - inset, top, -z + margin],
            ],
            scene.facade.short_side_window_count,
            scene.footprint_depth_m - margin * 2.0,
            true,
        ),
        Side::Right => (
            [
                [x + inset, sill, z - margin],
                [x + inset, sill, -z + margin],
                [x + inset, top, -z + margin],
                [x + inset, top, z - margin],
            ],
            scene.facade.short_side_window_count,
            scene.footprint_depth_m - margin * 2.0,
            true,
        ),
    };
    append_quad_tinted(
        mesh,
        positions,
        MaterialSlot::GLASS,
        facade_glass_appearance(&scene.facade),
    );

    let spacing = span / count as f32;
    for index in 1..count {
        let offset = -span * 0.5 + spacing * index as f32;
        let centre = match side {
            Side::Front => [offset, (sill + top) * 0.5, -z - inset * 1.2],
            Side::Back => [-offset, (sill + top) * 0.5, z + inset * 1.2],
            Side::Left => [-x - inset * 1.2, (sill + top) * 0.5, offset],
            Side::Right => [x + inset * 1.2, (sill + top) * 0.5, -offset],
        };
        let size = if vertical {
            [0.07, scene.facade.window_height_m + 0.05, 0.11]
        } else {
            [0.11, scene.facade.window_height_m + 0.05, 0.07]
        };
        append_box(mesh, centre, size, 0.0, MaterialSlot::TRIM);
    }
}

fn facade_glass_appearance(facade: &FacadeDescription) -> [f32; 4] {
    let weathering = facade.glazing_weathering;
    [
        srgb_to_linear(facade.glazing_color_srgb[0]) * (1.0 - weathering * 0.32),
        srgb_to_linear(facade.glazing_color_srgb[1]) * (1.0 - weathering * 0.3),
        srgb_to_linear(facade.glazing_color_srgb[2]) * (1.0 - weathering * 0.25),
        (facade.glazing_roughness + weathering * 0.28).clamp(0.04, 1.0),
    ]
}

fn append_ground_context(mesh: &mut RenderMesh, scene: &SceneDescription) {
    let g = scene.ground_half_extent_m;
    append_quad(
        mesh,
        [
            [-g, -0.03, -g],
            [-g, -0.03, g],
            [g, -0.03, g],
            [g, -0.03, -g],
        ],
        MaterialSlot::GROUND,
    );

    if scene.site.parking_bays > 0 {
        let bay_width = scene.site.parking_bay_size_m[0];
        let bay_length = scene.site.parking_bay_size_m[1];
        let row_width = bay_width * scene.site.parking_bays as f32;
        append_box(
            mesh,
            [
                scene.site.parking_center[0],
                -0.012,
                scene.site.parking_center[2],
            ],
            [row_width + 1.0, 0.025, bay_length + 2.0],
            scene.site.parking_yaw_radians,
            MaterialSlot::ASPHALT,
        );
        for index in 0..=scene.site.parking_bays {
            let local_x = -row_width * 0.5 + bay_width * index as f32;
            let offset = rotate_ground_offset(local_x, 0.0, scene.site.parking_yaw_radians);
            append_box(
                mesh,
                [
                    scene.site.parking_center[0] + offset[0],
                    0.012,
                    scene.site.parking_center[2] + offset[1],
                ],
                [0.075, 0.018, bay_length],
                scene.site.parking_yaw_radians,
                MaterialSlot::MARKING,
            );
        }
    }

    let road_depth = scene.site.road_lanes as f32 * scene.site.lane_width_m;
    append_box(
        mesh,
        [0.0, -0.012, scene.site.road_center_z_m],
        [g * 2.0, 0.025, road_depth],
        0.0,
        MaterialSlot::ASPHALT,
    );
    for lane in 1..scene.site.road_lanes {
        let draw_boundary = match scene.site.road_kind {
            SiteRoadKind::UrbanArterial | SiteRoadKind::HighwayFrontage => true,
            SiteRoadKind::LocalStreet | SiteRoadKind::RuralRoad => {
                lane * 2 == scene.site.road_lanes
            }
        };
        if !draw_boundary {
            continue;
        }
        let z =
            scene.site.road_center_z_m - road_depth * 0.5 + scene.site.lane_width_m * lane as f32;
        append_box(
            mesh,
            [0.0, 0.012, z],
            [g * 1.85, 0.018, 0.075],
            0.0,
            MaterialSlot::MARKING,
        );
    }
    if scene.site.road_kind == SiteRoadKind::HighwayFrontage {
        for edge in [-1.0_f32, 1.0] {
            append_box(
                mesh,
                [
                    0.0,
                    0.012,
                    scene.site.road_center_z_m + edge * (road_depth * 0.5 - 0.22),
                ],
                [g * 1.95, 0.018, 0.11],
                0.0,
                MaterialSlot::MARKING,
            );
        }
    }
    if scene.site.raised_curb {
        append_box(
            mesh,
            [
                0.0,
                0.06,
                scene.site.road_center_z_m + road_depth * 0.5 + 0.11,
            ],
            [g * 2.0, 0.12, 0.22],
            0.0,
            MaterialSlot::CONCRETE,
        );
    }

    for pole in &scene.site.utility_poles {
        append_utility_pole(mesh, *pole);
    }
    for line in &scene.site.utility_lines {
        let midpoint = [
            (line.start[0] + line.end[0]) * 0.5,
            (line.start[1] + line.end[1]) * 0.5 - line.sag_m,
            (line.start[2] + line.end[2]) * 0.5,
        ];
        append_beam(mesh, line.start, midpoint, 0.025, MaterialSlot::METAL);
        append_beam(mesh, midpoint, line.end, 0.025, MaterialSlot::METAL);
    }
}

fn rotate_ground_offset(x: f32, z: f32, yaw: f32) -> [f32; 2] {
    let (sin, cos) = yaw.sin_cos();
    [cos * x + sin * z, -sin * x + cos * z]
}

fn append_utility_pole(mesh: &mut RenderMesh, pole: UtilityPoleDescription) {
    append_cylinder(
        mesh,
        pole.position,
        (pole.height_m * 0.02).clamp(0.1, 0.2),
        pole.height_m,
        8,
        if pole.has_light {
            MaterialSlot::METAL
        } else {
            MaterialSlot::TRUNK
        },
    );
    if pole.has_light {
        append_box(
            mesh,
            [
                pole.position[0] + 0.55,
                pole.position[1] + pole.height_m - 0.08,
                pole.position[2],
            ],
            [1.2, 0.13, 0.13],
            0.0,
            MaterialSlot::METAL,
        );
        append_box(
            mesh,
            [
                pole.position[0] + 1.05,
                pole.position[1] + pole.height_m - 0.22,
                pole.position[2],
            ],
            [0.38, 0.18, 0.32],
            0.0,
            MaterialSlot::TRIM,
        );
    } else {
        append_box(
            mesh,
            [
                pole.position[0],
                pole.position[1] + pole.height_m * 0.91,
                pole.position[2],
            ],
            [2.4, 0.14, 0.14],
            0.0,
            MaterialSlot::TRUNK,
        );
    }
}

fn append_tree(mesh: &mut RenderMesh, tree: TreeInstance) {
    if tree.kind != TreeKind::Shrub {
        let trunk_fraction = if tree.kind == TreeKind::Palm {
            0.82
        } else {
            0.48
        };
        let trunk_height = tree.height_m * trunk_fraction;
        append_cylinder(
            mesh,
            tree.position,
            tree.crown_radius_m * 0.16,
            trunk_height,
            8,
            MaterialSlot::TRUNK,
        );
    }
    let seasonal = tree.seasonal_variation;
    let foliage = [
        (0.045 + (-seasonal).max(0.0) * 0.16).clamp(0.0, 1.0),
        (0.19 + seasonal * 0.055 - (-seasonal).max(0.0) * 0.04).clamp(0.0, 1.0),
        (0.035 - (-seasonal).max(0.0) * 0.015).clamp(0.0, 1.0),
        0.0,
    ];
    match tree.kind {
        TreeKind::Deciduous => {
            let crown_y = tree.position[1] + tree.height_m * 0.68;
            for offset in [
                [0.0, 0.0, 0.0],
                [-0.42, -0.08, 0.15],
                [0.38, 0.03, -0.18],
                [0.08, 0.2, 0.25],
                [-0.08, 0.12, -0.36],
            ] {
                let rotated = rotate_ground_offset(offset[0], offset[2], tree.yaw_radians);
                append_ellipsoid_tinted(
                    mesh,
                    [
                        tree.position[0] + rotated[0] * tree.crown_radius_m,
                        crown_y + offset[1] * tree.crown_radius_m,
                        tree.position[2] + rotated[1] * tree.crown_radius_m,
                    ],
                    [
                        tree.crown_radius_m * 0.78,
                        tree.crown_radius_m * 0.62,
                        tree.crown_radius_m * 0.74,
                    ],
                    MaterialSlot::FOLIAGE,
                    foliage,
                );
            }
        }
        TreeKind::Evergreen => {
            for (index, (height_fraction, radius_fraction)) in
                [(0.52, 1.0), (0.68, 0.78), (0.82, 0.52)]
                    .into_iter()
                    .enumerate()
            {
                let offset = rotate_ground_offset(
                    tree.crown_radius_m * 0.08 * index as f32,
                    0.0,
                    tree.yaw_radians,
                );
                append_ellipsoid_tinted(
                    mesh,
                    [
                        tree.position[0] + offset[0],
                        tree.position[1] + tree.height_m * height_fraction,
                        tree.position[2] + offset[1],
                    ],
                    [
                        tree.crown_radius_m * radius_fraction,
                        tree.crown_radius_m * radius_fraction * 0.72,
                        tree.crown_radius_m * radius_fraction,
                    ],
                    MaterialSlot::FOLIAGE,
                    foliage,
                );
            }
        }
        TreeKind::Palm => {
            let crown_y = tree.position[1] + tree.height_m * 0.86;
            append_ellipsoid_tinted(
                mesh,
                [tree.position[0], crown_y, tree.position[2]],
                [
                    tree.crown_radius_m * 0.48,
                    tree.crown_radius_m * 0.34,
                    tree.crown_radius_m * 0.48,
                ],
                MaterialSlot::FOLIAGE,
                foliage,
            );
            for index in 0..6 {
                let yaw = tree.yaw_radians + std::f32::consts::TAU * index as f32 / 6.0;
                let offset = rotate_ground_offset(tree.crown_radius_m * 0.62, 0.0, yaw);
                append_box_tinted(
                    mesh,
                    [
                        tree.position[0] + offset[0],
                        crown_y - tree.crown_radius_m * 0.12,
                        tree.position[2] + offset[1],
                    ],
                    [tree.crown_radius_m * 1.25, 0.1, tree.crown_radius_m * 0.22],
                    -yaw,
                    MaterialSlot::FOLIAGE,
                    foliage,
                );
            }
        }
        TreeKind::Shrub => {
            for offset in [-0.65_f32, 0.0, 0.65] {
                let rotated =
                    rotate_ground_offset(offset * tree.crown_radius_m, 0.0, tree.yaw_radians);
                append_ellipsoid_tinted(
                    mesh,
                    [
                        tree.position[0] + rotated[0],
                        tree.position[1] + tree.height_m * 0.48,
                        tree.position[2] + rotated[1],
                    ],
                    [
                        tree.crown_radius_m * 0.72,
                        tree.height_m * 0.48,
                        tree.crown_radius_m * 0.62,
                    ],
                    MaterialSlot::FOLIAGE,
                    foliage,
                );
            }
        }
    }
}

fn append_occluder(mesh: &mut RenderMesh, occluder: SceneOccluder) {
    let p = occluder.position;
    match occluder.kind {
        SceneOccluderKind::Vehicle => {
            let size = occluder_size(occluder, [4.2, 1.44, 1.75]);
            append_box(
                mesh,
                [p[0], p[1] + size[1] * 0.29, p[2]],
                [size[0], size[1] * 0.58, size[2]],
                occluder.yaw_radians,
                MaterialSlot::METAL,
            );
            append_box_tinted_pattern(
                mesh,
                [p[0], p[1] + size[1] * 0.76, p[2]],
                [size[0] * 0.54, size[1] * 0.42, size[2] * 0.88],
                occluder.yaw_radians,
                MaterialSlot::GLASS,
                [0.0; 4],
                SurfacePattern::VEHICLE_GLASS,
            );
        }
        SceneOccluderKind::PoleSign => {
            let size = occluder_size(occluder, [2.5, 5.1, 0.18]);
            let panel_height = (size[1] * 0.28).clamp(0.45, size[1] * 0.55);
            append_cylinder(
                mesh,
                p,
                (size[0] * 0.045).clamp(0.06, 0.22),
                size[1] - panel_height * 0.5,
                8,
                MaterialSlot::METAL,
            );
            append_panel_tinted_pattern(
                mesh,
                [p[0], p[1] + size[1] - panel_height * 0.5, p[2]],
                [size[0], panel_height, size[2]],
                occluder.yaw_radians,
                MaterialSlot::SIGN,
                [0.0; 4],
                SurfacePattern::GENERIC_SIGN,
            );
        }
        SceneOccluderKind::UtilityPole => {
            let size = occluder_size(occluder, [0.32, 7.5, 0.32]);
            append_cylinder(mesh, p, size[0] * 0.5, size[1], 8, MaterialSlot::TRUNK);
            append_box(
                mesh,
                [p[0], p[1] + size[1] * 0.91, p[2]],
                [size[0] * 9.0, size[0] * 0.5, size[2] * 0.5],
                occluder.yaw_radians,
                MaterialSlot::TRUNK,
            );
            for x_offset in [-1.1_f32, 0.0, 1.1] {
                append_box(
                    mesh,
                    [
                        p[0] + x_offset * size[0] * 3.0,
                        p[1] + size[1] * 0.9,
                        p[2] + size[1] * 0.46,
                    ],
                    [size[0] * 0.1, size[0] * 0.1, size[1] * 0.92],
                    occluder.yaw_radians,
                    MaterialSlot::METAL,
                );
            }
        }
        SceneOccluderKind::Shrub => {
            let size = occluder_size(occluder, [2.9, 1.3, 1.5]);
            for offset in [-0.7_f32, 0.0, 0.7] {
                append_ellipsoid_tinted(
                    mesh,
                    [p[0] + offset * size[0] * 0.32, p[1] + size[1] * 0.45, p[2]],
                    [size[0] * 0.34, size[1] * 0.52, size[2] * 0.46],
                    MaterialSlot::FOLIAGE,
                    [0.0; 4],
                );
            }
        }
        SceneOccluderKind::Hvac => {
            let size = occluder_size(occluder, [2.0, 1.2, 1.4]);
            append_box(
                mesh,
                [p[0], p[1] + size[1] * 0.5, p[2]],
                size,
                occluder.yaw_radians,
                MaterialSlot::METAL,
            );
        }
        SceneOccluderKind::StreetLight => {
            let size = occluder_size(occluder, [0.22, 5.2, 0.22]);
            append_cylinder(mesh, p, size[0] * 0.5, size[1], 8, MaterialSlot::METAL);
            append_box(
                mesh,
                [p[0] + size[1] * 0.11, p[1] + size[1] * 0.99, p[2]],
                [size[1] * 0.23, size[0] * 0.6, size[2] * 0.6],
                0.0,
                MaterialSlot::METAL,
            );
            append_box(
                mesh,
                [p[0] + size[1] * 0.2, p[1] + size[1] * 0.96, p[2]],
                [size[0] * 1.7, size[0] * 0.8, size[2] * 1.45],
                0.0,
                MaterialSlot::TRIM,
            );
        }
        SceneOccluderKind::Pedestrian => {
            let size = occluder_size(occluder, [0.62, 1.72, 0.42]);
            append_cylinder(
                mesh,
                p,
                size[0] * 0.29,
                size[1] * 0.73,
                8,
                MaterialSlot::OCCLUDER,
            );
            append_ellipsoid_tinted(
                mesh,
                [p[0], p[1] + size[1] * 0.9, p[2]],
                [size[0] * 0.36, size[0] * 0.44, size[2] * 0.48],
                MaterialSlot::OCCLUDER,
                [0.0; 4],
            );
        }
        SceneOccluderKind::Building => {
            let size = occluder_size(occluder, [8.0, 5.0, 6.0]);
            append_background_box(
                mesh,
                [p[0], p[1] + size[1] * 0.5, p[2]],
                size,
                occluder.yaw_radians,
                [0.28, 0.3, 0.31, 0.78],
                SurfacePattern::BACKGROUND_WINDOWS,
            );
        }
        SceneOccluderKind::Vegetation => {
            let size = occluder_size(occluder, [3.6, 5.5, 3.6]);
            append_tree(
                mesh,
                TreeInstance {
                    position: p,
                    height_m: size[1],
                    crown_radius_m: size[0].max(size[2]) * 0.5,
                    kind: TreeKind::Deciduous,
                    yaw_radians: occluder.yaw_radians,
                    seasonal_variation: 0.0,
                },
            );
        }
    }
}

fn occluder_size(occluder: SceneOccluder, base: [f32; 3]) -> [f32; 3] {
    if occluder.nominal_size_m.iter().all(|value| *value > 0.0) {
        occluder.nominal_size_m
    } else {
        base.map(|value| value * occluder.scale)
    }
}

fn append_beam(
    mesh: &mut RenderMesh,
    start: [f32; 3],
    end: [f32; 3],
    radius: f32,
    material: MaterialSlot,
) {
    let start = Vec3::from_array(start);
    let end = Vec3::from_array(end);
    let direction = (end - start).normalize_or_zero();
    if direction == Vec3::ZERO {
        return;
    }
    let reference = if direction.dot(Vec3::Y).abs() > 0.95 {
        Vec3::X
    } else {
        Vec3::Y
    };
    let side = direction.cross(reference).normalize() * radius;
    let up = side.cross(direction).normalize() * radius;
    let corners = [
        start - side - up,
        start + side - up,
        start + side + up,
        start - side + up,
        end - side - up,
        end + side - up,
        end + side + up,
        end - side + up,
    ];
    for face in [[0, 4, 5, 1], [1, 5, 6, 2], [2, 6, 7, 3], [3, 7, 4, 0]] {
        append_quad(mesh, face.map(|index| corners[index].to_array()), material);
    }
}

fn append_box(
    mesh: &mut RenderMesh,
    center: [f32; 3],
    size: [f32; 3],
    yaw: f32,
    material: MaterialSlot,
) {
    append_box_tinted(mesh, center, size, yaw, material, [0.0; 4]);
}

fn append_box_tinted(
    mesh: &mut RenderMesh,
    center: [f32; 3],
    size: [f32; 3],
    yaw: f32,
    material: MaterialSlot,
    appearance: [f32; 4],
) {
    append_box_tinted_pattern(
        mesh,
        center,
        size,
        yaw,
        material,
        appearance,
        SurfacePattern::INHERIT,
    );
}

fn append_box_tinted_pattern(
    mesh: &mut RenderMesh,
    center: [f32; 3],
    size: [f32; 3],
    yaw: f32,
    material: MaterialSlot,
    appearance: [f32; 4],
    pattern: SurfacePattern,
) {
    let half = Vec3::from_array(size) * 0.5;
    let rotation = Quat::from_rotation_y(yaw);
    let center = Vec3::from_array(center);
    let transform = |point: [f32; 3]| (rotation * Vec3::from_array(point) + center).to_array();
    let x = half.x;
    let y = half.y;
    let z = half.z;
    for face in [
        [[x, -y, -z], [-x, -y, -z], [-x, y, -z], [x, y, -z]],
        [[x, -y, z], [x, -y, -z], [x, y, -z], [x, y, z]],
        [[-x, -y, z], [x, -y, z], [x, y, z], [-x, y, z]],
        [[-x, -y, -z], [-x, -y, z], [-x, y, z], [-x, y, -z]],
        [[-x, y, -z], [-x, y, z], [x, y, z], [x, y, -z]],
        [[-x, -y, z], [-x, -y, -z], [x, -y, -z], [x, -y, z]],
    ] {
        append_quad_tinted_pattern(mesh, face.map(transform), material, appearance, pattern);
    }
}

fn append_cylinder(
    mesh: &mut RenderMesh,
    base: [f32; 3],
    radius: f32,
    height: f32,
    segments: u32,
    material: MaterialSlot,
) {
    let base_y = base[1];
    for segment in 0..segments {
        let a0 = std::f32::consts::TAU * segment as f32 / segments as f32;
        let a1 = std::f32::consts::TAU * (segment + 1) as f32 / segments as f32;
        append_quad(
            mesh,
            [
                [
                    base[0] + radius * a1.cos(),
                    base_y,
                    base[2] + radius * a1.sin(),
                ],
                [
                    base[0] + radius * a0.cos(),
                    base_y,
                    base[2] + radius * a0.sin(),
                ],
                [
                    base[0] + radius * a0.cos(),
                    base_y + height,
                    base[2] + radius * a0.sin(),
                ],
                [
                    base[0] + radius * a1.cos(),
                    base_y + height,
                    base[2] + radius * a1.sin(),
                ],
            ],
            material,
        );
    }
}

fn append_ellipsoid_tinted(
    mesh: &mut RenderMesh,
    center: [f32; 3],
    radii: [f32; 3],
    material: MaterialSlot,
    appearance: [f32; 4],
) {
    const RINGS: u32 = 7;
    const SEGMENTS: u32 = 12;
    let first_vertex = mesh.vertices.len() as u32;
    for ring in 0..=RINGS {
        let v = ring as f32 / RINGS as f32;
        let theta = std::f32::consts::PI * v;
        let radial = theta.sin();
        let y = theta.cos();
        for segment in 0..=SEGMENTS {
            let u = segment as f32 / SEGMENTS as f32;
            let phi = std::f32::consts::TAU * u;
            let unit = [radial * phi.cos(), y, radial * phi.sin()];
            let position = [
                center[0] + unit[0] * radii[0],
                center[1] + unit[1] * radii[1],
                center[2] + unit[2] * radii[2],
            ];
            let normal = Vec3::new(unit[0] / radii[0], unit[1] / radii[1], unit[2] / radii[2])
                .normalize_or_zero()
                .to_array();
            mesh.vertices.push(RenderVertex {
                position,
                normal,
                face_coord: [u, v],
                semantic_id: 0,
                material,
                appearance,
                pattern: SurfacePattern::INHERIT,
            });
        }
    }
    let stride = SEGMENTS + 1;
    for ring in 0..RINGS {
        for segment in 0..SEGMENTS {
            let top_left = first_vertex + ring * stride + segment;
            let top_right = top_left + 1;
            let bottom_left = top_left + stride;
            let bottom_right = bottom_left + 1;
            mesh.indices.extend_from_slice(&[
                top_left,
                top_right,
                bottom_left,
                top_right,
                bottom_right,
                bottom_left,
            ]);
        }
    }
}

fn background(center: [f32; 3], size: [f32; 3], yaw_radians: f32) -> BackgroundBuilding {
    BackgroundBuilding {
        center,
        size,
        yaw_radians,
        kind: if size[1] > 9.0 {
            BackgroundBuildingKind::MidriseMixedUse
        } else {
            BackgroundBuildingKind::LowCommercial
        },
        base_color_srgb: [0.72, 0.72, 0.7],
        weathering: 0.25,
        roughness: 0.78,
        surface_pattern: SurfacePattern::BACKGROUND_WINDOWS,
    }
}

fn background_surface_pattern(
    material_id: &str,
    kind: synth_data::BackgroundBuildingKind,
) -> SurfacePattern {
    let windowed = !matches!(kind, synth_data::BackgroundBuildingKind::IndustrialShed);
    if material_id.contains("brick") {
        if windowed {
            SurfacePattern::BACKGROUND_WINDOWS_BRICK
        } else {
            SurfacePattern::BRICK
        }
    } else if material_id.contains("cladding") || material_id.contains("repaint") {
        if windowed {
            SurfacePattern::BACKGROUND_WINDOWS_CLADDING
        } else {
            SurfacePattern::VERTICAL_CLADDING
        }
    } else if windowed {
        SurfacePattern::BACKGROUND_WINDOWS
    } else {
        SurfacePattern::SMOOTH
    }
}

fn vec3_array(value: synth_data::Vec3) -> [f32; 3] {
    [value.x, value.y, value.z]
}

fn map_domain(domain: synth_data::SceneDomain) -> EnvironmentDomain {
    match domain {
        synth_data::SceneDomain::City => EnvironmentDomain::City,
        synth_data::SceneDomain::Urban
        | synth_data::SceneDomain::Suburban
        | synth_data::SceneDomain::Roadside => EnvironmentDomain::Urban,
        synth_data::SceneDomain::Remote => EnvironmentDomain::Remote,
    }
}

fn map_day_phase(phase: synth_data::DayPhase) -> TimeOfDay {
    match phase {
        synth_data::DayPhase::Day => TimeOfDay::Day,
        synth_data::DayPhase::Twilight => TimeOfDay::Twilight,
        synth_data::DayPhase::Night => TimeOfDay::Night,
    }
}

fn map_weather(weather: synth_data::WeatherPreset) -> WeatherAppearance {
    match weather {
        synth_data::WeatherPreset::Clear => WeatherAppearance::Clear,
        synth_data::WeatherPreset::PartlyCloudy => WeatherAppearance::PartlyCloudy,
        synth_data::WeatherPreset::Overcast => WeatherAppearance::Overcast,
        synth_data::WeatherPreset::Hazy => WeatherAppearance::Hazy,
        synth_data::WeatherPreset::AfterRain => WeatherAppearance::AfterRain,
    }
}

fn map_facade_side(side: synth_data::FacadeSide) -> BuildingSide {
    match side {
        synth_data::FacadeSide::Front => BuildingSide::Front,
        synth_data::FacadeSide::Right => BuildingSide::Right,
        synth_data::FacadeSide::Back => BuildingSide::Back,
        synth_data::FacadeSide::Left => BuildingSide::Left,
    }
}

fn time_from_sun_elevation(elevation_degrees: f32) -> TimeOfDay {
    if elevation_degrees > 10.0 {
        TimeOfDay::Day
    } else if elevation_degrees > -8.0 {
        TimeOfDay::Twilight
    } else {
        TimeOfDay::Night
    }
}

fn sampled_scene_seed(sampled: &synth_data::SampledScene) -> u32 {
    let mut hash = 2_166_136_261_u32;
    for byte in sampled
        .roof_material
        .id
        .bytes()
        .chain(sampled.wall_material.id.bytes())
        .chain(
            [
                sampled.building.footprint_width_m.to_bits(),
                sampled.building.footprint_depth_m.to_bits(),
                sampled.building.wall_height_m.to_bits(),
                sampled.roof.crown_top_width_m.to_bits(),
                sampled.roof.crown_top_depth_m.to_bits(),
            ]
            .into_iter()
            .flat_map(u32::to_le_bytes),
        )
    {
        hash ^= u32::from(byte);
        hash = hash.wrapping_mul(16_777_619);
    }
    hash.max(1)
}

fn seed_to_yaw(seed: u32) -> f32 {
    seed as f32 / u32::MAX as f32 * std::f32::consts::TAU
}

fn tree(position: [f32; 3], height_m: f32, crown_radius_m: f32) -> TreeInstance {
    TreeInstance {
        position,
        height_m,
        crown_radius_m,
        kind: TreeKind::Deciduous,
        yaw_radians: 0.0,
        seasonal_variation: 0.0,
    }
}

fn occluder(
    kind: SceneOccluderKind,
    position: [f32; 3],
    yaw_radians: f32,
    scale: f32,
) -> SceneOccluder {
    SceneOccluder {
        kind,
        position,
        yaw_radians,
        scale,
        nominal_size_m: [0.0; 3],
    }
}

fn utility_pole(position: [f32; 3], height_m: f32, has_light: bool) -> UtilityPoleDescription {
    UtilityPoleDescription {
        position,
        height_m,
        has_light,
    }
}

fn adjacent_utility_lines(
    poles: &[UtilityPoleDescription],
    sag_m: f32,
) -> Vec<UtilityLineDescription> {
    poles
        .windows(2)
        .map(|pair| UtilityLineDescription {
            start: [
                pair[0].position[0],
                pair[0].position[1] + pair[0].height_m * 0.91,
                pair[0].position[2],
            ],
            end: [
                pair[1].position[0],
                pair[1].position[1] + pair[1].height_m * 0.91,
                pair[1].position[2],
            ],
            sag_m,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn building_extensions_add_unlabelled_wall_roof_and_glazing_geometry() {
        for roof in [BuildingExtensionRoof::Flat, BuildingExtensionRoof::Shed] {
            let mut mesh = RenderMesh::default();
            append_building_extension(
                &mut mesh,
                BuildingExtensionDescription {
                    kind: BuildingExtensionKind::DiningWing,
                    facade: BuildingSide::Front,
                    position: [1.0, 0.0, -9.0],
                    size: [7.0, 3.0, 4.0],
                    roof,
                    roof_rise_m: if roof == BuildingExtensionRoof::Shed {
                        0.5
                    } else {
                        0.0
                    },
                },
            );

            assert!(!mesh.vertices.is_empty());
            assert!(mesh.vertices.iter().all(|vertex| vertex.semantic_id == 0));
            assert!(
                mesh.vertices
                    .iter()
                    .any(|vertex| vertex.material == MaterialSlot::WALL)
            );
            assert!(
                mesh.vertices
                    .iter()
                    .any(|vertex| vertex.material == MaterialSlot::ROOF)
            );
            assert!(
                mesh.vertices
                    .iter()
                    .any(|vertex| vertex.material == MaterialSlot::GLASS)
            );
        }
    }

    #[test]
    fn every_ordinary_negative_roof_is_unlabelled_render_geometry() {
        let sampler = synth_data::SequenceSampler::new(synth_data::GeneratorConfig::default())
            .expect("default sampler");
        for (index, family) in synth_data::OrdinaryRoofFamily::ALL.into_iter().enumerate() {
            let sampled = sampler
                .sample(synth_data::SequenceRequest::procedural(
                    format!("ordinary_{}", family.as_str()),
                    40_000 + index as u64,
                    synth_data::TargetKind::Negative,
                ))
                .expect("ordinary scene")
                .scene;
            let ordinary = sampled.ordinary_roof.expect("ordinary roof dimensions");
            let description = SceneDescription::from_sampled(&sampled).expect("scene description");
            let mesh = RenderMesh::from_ordinary_scene(ordinary, &description)
                .expect("ordinary scene mesh");

            assert_eq!(ordinary.family, family);
            assert!(mesh.vertices.iter().all(|vertex| vertex.semantic_id == 0));
            assert!(
                mesh.vertices
                    .iter()
                    .any(|vertex| vertex.material == MaterialSlot::ROOF)
            );
            assert!(mesh.vertices.iter().any(|vertex| {
                vertex.material == MaterialSlot::ROOF
                    && vertex.position[1] > sampled.building.wall_height_m
            }));
        }
    }

    #[test]
    fn context_density_changes_by_domain() {
        let city = SceneDescription::contextual(
            20.0,
            14.0,
            4.0,
            40.0,
            RenderEnvironment {
                domain: EnvironmentDomain::City,
                ..RenderEnvironment::default()
            },
        )
        .unwrap();
        let remote = SceneDescription::contextual(
            20.0,
            14.0,
            4.0,
            40.0,
            RenderEnvironment {
                domain: EnvironmentDomain::Remote,
                ..RenderEnvironment::default()
            },
        )
        .unwrap();

        assert!(city.background_buildings.len() > remote.background_buildings.len());
        assert!(remote.trees.len() > city.trees.len());
    }

    #[test]
    fn night_context_includes_local_lighting() {
        let scene = SceneDescription::contextual(
            20.0,
            14.0,
            4.0,
            40.0,
            RenderEnvironment {
                time_of_day: TimeOfDay::Night,
                ..RenderEnvironment::default()
            },
        )
        .unwrap();
        assert!(!scene.lights.is_empty());
        assert!(scene.lights.iter().all(|light| light.is_valid()));
    }

    #[test]
    fn artificial_emission_is_suppressed_by_day_and_scaled_at_twilight() {
        assert_eq!(TimeOfDay::Day.emission_scale(), 0.0);
        assert_eq!(TimeOfDay::Twilight.emission_scale(), 0.45);
        assert_eq!(TimeOfDay::Night.emission_scale(), 1.0);

        let sampler = synth_data::SequenceSampler::new(synth_data::GeneratorConfig::default())
            .expect("default sampler");
        let mut sampled = sampler
            .sample(synth_data::SequenceRequest::procedural(
                "classic_two_stage",
                17,
                synth_data::TargetKind::Target,
            ))
            .expect("sampled sequence")
            .scene;
        sampled.composition.infrastructure = None;
        sampled.composition.signage = vec![synth_data::SampledSignage {
            kind: synth_data::SignageKind::PizzaHut,
            facade: synth_data::FacadeSide::Front,
            center: synth_data::Vec3::new(0.0, 3.0, -7.0),
            size_m: synth_data::Vec2::new(4.0, 1.2),
            emissive_strength: 2.0,
            weathering: 0.1,
        }];
        sampled
            .composition
            .environment
            .as_mut()
            .expect("sampled environment")
            .artificial_light_strength = 0.0;
        sampled
            .composition
            .environment
            .as_mut()
            .expect("sampled environment")
            .day_phase = synth_data::DayPhase::Day;
        let day = SceneDescription::from_sampled(&sampled).unwrap();
        assert!(day.lights.is_empty());
        sampled
            .composition
            .environment
            .as_mut()
            .expect("sampled environment")
            .day_phase = synth_data::DayPhase::Twilight;
        let twilight = SceneDescription::from_sampled(&sampled).unwrap();
        assert_eq!(twilight.lights.len(), 1);
        assert_eq!(
            twilight.lights[0].intensity * TimeOfDay::Twilight.emission_scale(),
            0.9
        );
        sampled
            .composition
            .environment
            .as_mut()
            .expect("sampled environment")
            .day_phase = synth_data::DayPhase::Night;
        let night = SceneDescription::from_sampled(&sampled).unwrap();
        assert_eq!(night.lights.len(), 1);
        assert_eq!(night.lights[0].intensity, 2.0);
    }

    #[test]
    fn background_caps_do_not_receive_window_patterns() {
        let mut mesh = RenderMesh::default();
        append_background_box(
            &mut mesh,
            [0.0, 2.0, 0.0],
            [8.0, 4.0, 6.0],
            0.0,
            [0.3, 0.2, 0.1, 0.8],
            SurfacePattern::BACKGROUND_WINDOWS_BRICK,
        );
        assert!(mesh.vertices.iter().any(|vertex| {
            vertex.normal[1].abs() < 0.1
                && vertex.pattern == SurfacePattern::BACKGROUND_WINDOWS_BRICK
        }));
        assert!(
            mesh.vertices
                .iter()
                .filter(|vertex| vertex.normal[1].abs() > 0.9)
                .all(|vertex| vertex.pattern == SurfacePattern::BRICK)
        );
    }

    #[test]
    fn generic_signs_and_vehicle_glass_do_not_inherit_branded_or_window_emission() {
        let roof = roof_geometry::generate_roof(&roof_geometry::RoofParameters::default()).unwrap();
        let mut scene = SceneDescription::new(20.0, 14.0, 4.0, 40.0, RenderEnvironment::default());
        scene.occluders = vec![
            SceneOccluder {
                kind: SceneOccluderKind::PoleSign,
                position: [8.0, 0.0, -8.0],
                yaw_radians: 0.2,
                scale: 1.0,
                nominal_size_m: [2.5, 5.0, 0.2],
            },
            SceneOccluder {
                kind: SceneOccluderKind::Vehicle,
                position: [-7.0, 0.0, -8.0],
                yaw_radians: -0.2,
                scale: 1.0,
                nominal_size_m: [4.2, 1.5, 1.8],
            },
        ];
        let mesh = RenderMesh::from_scene(&roof, &scene).unwrap();
        assert!(mesh.vertices.iter().any(|vertex| {
            vertex.material == MaterialSlot::SIGN && vertex.pattern == SurfacePattern::GENERIC_SIGN
        }));
        assert!(!mesh.vertices.iter().any(|vertex| {
            vertex.material == MaterialSlot::SIGN
                && vertex.pattern == SurfacePattern::PIZZA_HUT_SIGN
        }));
        assert!(mesh.vertices.iter().any(|vertex| {
            vertex.material == MaterialSlot::GLASS
                && vertex.pattern == SurfacePattern::VEHICLE_GLASS
        }));
    }

    #[test]
    fn branded_sign_art_is_limited_to_two_panel_faces() {
        let mut mesh = RenderMesh::default();
        append_sign(
            &mut mesh,
            SignDescription {
                style: SignStyle::PizzaHut,
                mount: SignMount::Facade,
                center: [0.0, 3.0, -7.0],
                size: [4.0, 1.2],
                yaw_radians: 0.0,
                emissive_strength: 1.0,
                weathering: 0.1,
            },
        );
        let art_vertices = mesh
            .vertices
            .iter()
            .filter(|vertex| vertex.pattern == SurfacePattern::PIZZA_HUT_SIGN)
            .collect::<Vec<_>>();
        assert_eq!(art_vertices.len(), 8);
        assert!(
            art_vertices
                .iter()
                .all(|vertex| vertex.material == MaterialSlot::SIGN)
        );
        assert!(mesh.vertices.iter().any(|vertex| {
            vertex.material == MaterialSlot::TRIM && vertex.pattern == SurfacePattern::SMOOTH
        }));
    }

    #[test]
    fn resolved_nominal_occluder_size_is_not_scaled_twice() {
        let resolved = SceneOccluder {
            kind: SceneOccluderKind::Vehicle,
            position: [0.0; 3],
            yaw_radians: 0.0,
            scale: 1.8,
            nominal_size_m: [4.0, 1.5, 2.0],
        };
        assert_eq!(occluder_size(resolved, [2.0, 1.0, 1.0]), [4.0, 1.5, 2.0]);
        let unresolved = SceneOccluder {
            nominal_size_m: [0.0; 3],
            ..resolved
        };
        assert_eq!(occluder_size(unresolved, [2.0, 1.0, 1.0]), [3.6, 1.8, 1.8]);
    }

    #[test]
    fn assembled_context_is_nonsemantic_except_for_roof() {
        let roof = roof_geometry::generate_roof(&roof_geometry::RoofParameters::default()).unwrap();
        let scene =
            SceneDescription::contextual(20.0, 14.0, 4.0, 40.0, RenderEnvironment::default())
                .unwrap();
        let mesh = RenderMesh::from_scene(&roof, &scene).unwrap();
        let semantic_vertices = mesh
            .vertices
            .iter()
            .filter(|vertex| vertex.semantic_id != 0)
            .count();

        assert_eq!(semantic_vertices, roof.mesh.vertices.len());
        assert!(mesh.vertices.len() > roof.mesh.vertices.len() * 10);
    }

    #[test]
    fn sampled_scene_maps_full_composition_into_visible_description() {
        let sampler = synth_data::SequenceSampler::new(synth_data::GeneratorConfig::default())
            .expect("default sampler");
        let mut sampled = sampler
            .sample(synth_data::SequenceRequest::procedural(
                "classic_two_stage",
                73,
                synth_data::TargetKind::Target,
            ))
            .expect("sampled sequence")
            .scene;
        let facade = sampled.composition.facade.as_mut().expect("sampled facade");
        facade.entrance_side = synth_data::FacadeSide::Left;
        facade.weathering = 0.72;
        let environment = sampled
            .composition
            .environment
            .as_mut()
            .expect("sampled environment");
        environment.day_phase = synth_data::DayPhase::Night;
        environment.domain = synth_data::SceneDomain::Remote;
        environment.visibility_km = 6.5;
        environment.shadow_softness = 0.81;
        environment.ground_wetness = 0.64;
        let kinds = [
            synth_data::OccluderKind::Vegetation,
            synth_data::OccluderKind::Vehicle,
            synth_data::OccluderKind::Pole,
            synth_data::OccluderKind::Sign,
            synth_data::OccluderKind::Building,
            synth_data::OccluderKind::Pedestrian,
            synth_data::OccluderKind::RooftopEquipment,
        ];
        sampled.occluders = kinds
            .into_iter()
            .enumerate()
            .map(|(index, kind)| synth_data::SampledOccluder {
                kind,
                position: synth_data::Vec3::new(index as f32 * 2.0 - 6.0, 0.0, 10.0),
                yaw_degrees: index as f32 * 17.0,
                scale: 1.0,
                placement: synth_data::OccluderPlacement::Site,
                nominal_size_m: synth_data::Vec3::new(2.0, 3.0, 1.5),
            })
            .collect();

        let description = SceneDescription::from_sampled(&sampled).unwrap();
        assert_eq!(description.facade.entrance_side, BuildingSide::Left);
        assert_eq!(description.facade.weathering, 0.72);
        assert_eq!(description.environment.time_of_day, TimeOfDay::Night);
        assert_eq!(description.environment.domain, EnvironmentDomain::Remote);
        assert_eq!(description.environment.visibility_km, 6.5);
        assert_eq!(description.environment.shadow_softness, 0.81);
        assert_eq!(description.environment.ground_wetness, 0.64);
        assert_ne!(description.environment.environment_yaw_radians, 0.0);
        assert_eq!(
            description.site.parking_bays,
            sampled
                .composition
                .infrastructure
                .as_ref()
                .expect("sampled infrastructure")
                .parking_bays
        );
        for expected in [
            SceneOccluderKind::Vegetation,
            SceneOccluderKind::Vehicle,
            SceneOccluderKind::UtilityPole,
            SceneOccluderKind::PoleSign,
            SceneOccluderKind::Building,
            SceneOccluderKind::Pedestrian,
            SceneOccluderKind::Hvac,
        ] {
            assert!(
                description
                    .occluders
                    .iter()
                    .any(|occluder| occluder.kind == expected)
            );
        }
    }

    #[test]
    fn entrance_side_moves_canopy_to_the_selected_wall() {
        let roof = roof_geometry::generate_roof(&roof_geometry::RoofParameters::default()).unwrap();
        let mut front = SceneDescription::new(20.0, 14.0, 4.0, 40.0, RenderEnvironment::default());
        front.facade.entrance_side = BuildingSide::Front;
        let front_mesh = RenderMesh::from_scene(&roof, &front).unwrap();
        let mut left = front;
        left.facade.entrance_side = BuildingSide::Left;
        let left_mesh = RenderMesh::from_scene(&roof, &left).unwrap();

        let front_min_z = front_mesh
            .vertices
            .iter()
            .filter(|vertex| vertex.material == MaterialSlot::TRIM)
            .map(|vertex| vertex.position[2])
            .fold(f32::INFINITY, f32::min);
        let left_min_x = left_mesh
            .vertices
            .iter()
            .filter(|vertex| vertex.material == MaterialSlot::TRIM)
            .map(|vertex| vertex.position[0])
            .fold(f32::INFINITY, f32::min);
        assert!(front_min_z < -7.5);
        assert!(left_min_x < -10.5);
    }
}
