use std::{collections::BTreeMap, f32::consts::TAU};

use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha20Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    BackgroundBuildingKind, BuildingExtensionKind, BuildingExtensionRoof, CameraIntrinsics,
    CameraModel, DatasetSplit, DayPhase, DayPhaseProfile, DistortionModel, FacadeSide, FloatRange,
    GeneratorConfig, ImageTransform, MaterialChoice, OccluderChoice, OccluderKind, RigidTransform,
    RoofInstanceRecord, RoofMorphology, RoofMorphologyProfile, SampledBackgroundBuilding,
    SampledBuildingExtension, SampledComposition, SampledEnvironment, SampledFacade,
    SampledSignage, SampledSiteInfrastructure, SampledUtilityLine, SampledUtilityPole,
    SampledVegetation, SceneDomain, SceneDomainProfile, SequenceFrameRef, SequenceRecord,
    SignageKind, SplitKey, SplitPolicy, SplitPolicyError, TargetKind, Validate, ValidationReport,
    Vec2, Vec3, VegetationKind, WeatherProfile,
    deterministic::{derive_seed, stable_hash64},
};

const CAMERA_COLLISION_CLEARANCE_M: f32 = 1.0;
const CAMERA_VIEW_CLEARANCE_M: f32 = 0.5;
pub(crate) const OCCLUDER_TARGET_CLEARANCE_M: f32 = 0.75;
pub(crate) const ROOFTOP_EDGE_CLEARANCE_M: f32 = 0.15;
pub(crate) const GROUND_EDGE_MARGIN_M: f32 = 4.0;

/// One deterministic generation request. Every camera view shares this identity.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequenceRequest {
    /// Procedural building/roof family.
    pub building_family: String,
    /// Root seed for the building, scene, and camera sequence.
    pub building_seed: u64,
    /// Accepted target, deliberate near miss, or empty negative.
    pub target_kind: TargetKind,
    /// Optional source asset group that must remain in one split.
    pub source_asset_group: Option<String>,
}

impl SequenceRequest {
    /// Creates a wholly procedural request.
    #[must_use]
    pub fn procedural(
        building_family: impl Into<String>,
        building_seed: u64,
        target_kind: TargetKind,
    ) -> Self {
        Self {
            building_family: building_family.into(),
            building_seed,
            target_kind,
            source_asset_group: None,
        }
    }

    fn split_key(&self) -> SplitKey {
        SplitKey {
            building_family: self.building_family.clone(),
            building_seed: self.building_seed,
            source_asset_group: self.source_asset_group.clone(),
        }
    }
}

#[derive(Clone, Copy)]
struct CameraPathParameters {
    kind: CameraPathKind,
    start: f32,
    sweep: f32,
    distance: f32,
    height: f32,
    radial_motion: f32,
    lateral_span: f32,
    approach_amount: f32,
    handheld_sway: f32,
    sway_phase: f32,
}

#[derive(Clone, Copy)]
struct CameraSceneObstacles<'a> {
    extensions: &'a [SampledBuildingExtension],
    backgrounds: &'a [SampledBackgroundBuilding],
    vegetation: &'a [SampledVegetation],
}

#[derive(Clone, Copy)]
struct CameraObstacleSize {
    half_x: f32,
    half_z: f32,
    height_m: f32,
}

/// Collision and framing envelope of the roof that is actually rendered.
///
/// Negative scenes must never use the otherwise-unlabelled two-tier reference
/// roof for camera placement. Keeping this small contract separate makes that
/// dependency explicit at every camera/collision call site.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RoofEnvelope {
    eave_width_m: f32,
    eave_depth_m: f32,
    maximum_height_m: f32,
}

impl RoofEnvelope {
    fn target(roof: SampledRoof) -> Self {
        Self {
            eave_width_m: roof.eave_width_m,
            eave_depth_m: roof.eave_depth_m,
            maximum_height_m: roof.lower_rise_m + roof.upper_rise_m,
        }
    }

    fn ordinary(roof: SampledOrdinaryRoof) -> Self {
        Self {
            eave_width_m: roof.eave_width_m,
            eave_depth_m: roof.eave_depth_m,
            maximum_height_m: roof.maximum_height_m(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ImageBounds {
    min_x: f32,
    max_x: f32,
    min_y: f32,
    max_y: f32,
}

fn sample_foreground_occluder_position(
    rng: &mut ChaCha20Rng,
    config: &crate::OccluderSamplingConfig,
    building: SampledBuilding,
    camera: Vec3,
    target: Vec3,
    size: CameraObstacleSize,
    frames: &[CameraFramePlan],
) -> Option<Vec3> {
    let sightline = sub(target, camera);
    let lateral = normalize(Vec3::new(-sightline.z, 0.0, sightline.x));
    for _ in 0..32 {
        let depth = sample_range(rng, config.foreground_depth_fraction);
        let offset = sample_range(rng, config.foreground_lateral_offset_m) * sample_signed(rng);
        let point = add(camera, scale(sightline, depth));
        let candidate = add(Vec3::new(point.x, 0.0, point.z), scale(lateral, offset));
        if outside_target_footprint(candidate, building, size.half_x, size.half_z)
            && occluder_position_clears_camera_path(candidate, size, frames)
        {
            return Some(candidate);
        }
    }
    None
}

fn sample_site_occluder_position(
    rng: &mut ChaCha20Rng,
    config: &crate::OccluderSamplingConfig,
    building: SampledBuilding,
    size: CameraObstacleSize,
    frames: &[CameraFramePlan],
) -> Option<Vec3> {
    for _ in 0..64 {
        let angle = rng.random::<f32>() * TAU;
        let distance = sample_range(rng, config.distance_m);
        let candidate = Vec3::new(
            libm::cosf(angle) * distance,
            0.0,
            libm::sinf(angle) * distance,
        );
        if outside_target_footprint(candidate, building, size.half_x, size.half_z)
            && occluder_position_clears_camera_path(candidate, size, frames)
        {
            return Some(candidate);
        }
    }
    None
}

fn occluder_position_clears_camera_path(
    position: Vec3,
    size: CameraObstacleSize,
    frames: &[CameraFramePlan],
) -> bool {
    let (minimum, maximum) = camera_clearance_bounds(position, size);
    let mut previous = None;
    for frame in frames {
        let camera = frame.camera.world_from_camera.translation;
        if point_inside_aabb(camera, minimum, maximum)
            || previous
                .is_some_and(|start| segment_intersects_aabb(start, camera, minimum, maximum))
        {
            return false;
        }
        previous = Some(camera);
    }
    true
}

fn camera_clearance_bounds(position: Vec3, size: CameraObstacleSize) -> (Vec3, Vec3) {
    (
        Vec3::new(
            position.x - size.half_x - CAMERA_COLLISION_CLEARANCE_M,
            position.y - CAMERA_COLLISION_CLEARANCE_M,
            position.z - size.half_z - CAMERA_COLLISION_CLEARANCE_M,
        ),
        Vec3::new(
            position.x + size.half_x + CAMERA_COLLISION_CLEARANCE_M,
            position.y + size.height_m + CAMERA_COLLISION_CLEARANCE_M,
            position.z + size.half_z + CAMERA_COLLISION_CLEARANCE_M,
        ),
    )
}

fn outside_target_footprint(
    position: Vec3,
    building: SampledBuilding,
    half_x: f32,
    half_z: f32,
) -> bool {
    position.x.abs() >= building.footprint_width_m * 0.5 + half_x + OCCLUDER_TARGET_CLEARANCE_M
        || position.z.abs()
            >= building.footprint_depth_m * 0.5 + half_z + OCCLUDER_TARGET_CLEARANCE_M
}

fn collision_free_camera_path(
    initial: CameraPathParameters,
    building: SampledBuilding,
    roof: RoofEnvelope,
    obstacles: CameraSceneObstacles<'_>,
    frame_count: u32,
    configured_min_distance: f32,
) -> CameraPathParameters {
    let mut target_inner_radius = libm::sqrtf(
        (roof.eave_width_m * 0.5 + CAMERA_COLLISION_CLEARANCE_M).powi(2)
            + (roof.eave_depth_m * 0.5 + CAMERA_COLLISION_CLEARANCE_M).powi(2),
    ) + initial.handheld_sway
        + 0.05;
    for extension in obstacles.extensions {
        let half_x = extension.size_m.x * 0.5 + CAMERA_COLLISION_CLEARANCE_M;
        let half_z = extension.size_m.z * 0.5 + CAMERA_COLLISION_CLEARANCE_M;
        let outer_x = extension.position.x.abs() + half_x;
        let outer_z = extension.position.z.abs() + half_z;
        target_inner_radius = target_inner_radius
            .max(libm::sqrtf(outer_x * outer_x + outer_z * outer_z) + initial.handheld_sway + 0.05);
    }
    let background_inner_radius =
        obstacles
            .backgrounds
            .iter()
            .fold(f32::INFINITY, |minimum, item| {
                let (half_x, half_z) = rotated_half_extents(item.size_m, item.yaw_degrees);
                let expanded_radius = libm::sqrtf(
                    (half_x + CAMERA_COLLISION_CLEARANCE_M).powi(2)
                        + (half_z + CAMERA_COLLISION_CLEARANCE_M).powi(2),
                );
                let centre_radius = libm::sqrtf(
                    item.position.x * item.position.x + item.position.z * item.position.z,
                );
                minimum.min(centre_radius - expanded_radius - initial.handheld_sway - 0.05)
            });
    let corridor_inner = target_inner_radius.max(configured_min_distance);
    let corridor_midpoint =
        if background_inner_radius.is_finite() && background_inner_radius > corridor_inner + 0.5 {
            (corridor_inner + background_inner_radius) * 0.5
        } else {
            initial.distance.max(corridor_inner)
        };

    for distance_step in 0..8 {
        let blend = distance_step as f32 / 7.0;
        let distance = initial.distance + (corridor_midpoint - initial.distance) * blend;
        let mut candidate = with_camera_distance(initial, distance);
        candidate = constrain_camera_path_to_annulus(candidate, target_inner_radius, f32::INFINITY);
        for angle_step in 0..96 {
            candidate.start = initial.start + angle_step as f32 * 2.399_963_1;
            if camera_path_is_clear(candidate, building, roof, obstacles, frame_count) {
                return candidate;
            }
        }
    }

    if background_inner_radius > target_inner_radius + 0.5 {
        let distance = (target_inner_radius + background_inner_radius) * 0.5;
        let mut fallback = with_camera_distance(initial, distance);
        fallback = constrain_camera_path_to_annulus(
            fallback,
            target_inner_radius,
            background_inner_radius,
        );
        if camera_path_is_clear(fallback, building, roof, obstacles, frame_count) {
            return fallback;
        }
    }

    let background_outer_radius =
        obstacles
            .backgrounds
            .iter()
            .fold(target_inner_radius, |maximum, item| {
                let (half_x, half_z) = rotated_half_extents(item.size_m, item.yaw_degrees);
                let expanded_radius = libm::sqrtf(
                    (half_x + CAMERA_COLLISION_CLEARANCE_M).powi(2)
                        + (half_z + CAMERA_COLLISION_CLEARANCE_M).powi(2),
                );
                let centre_radius = libm::sqrtf(
                    item.position.x * item.position.x + item.position.z * item.position.z,
                );
                maximum.max(centre_radius + expanded_radius)
            })
            + initial.handheld_sway
            + 1.0;
    let mut fallback = with_camera_distance(initial, background_outer_radius + 2.0);
    fallback = constrain_camera_path_to_annulus(fallback, background_outer_radius, f32::INFINITY);
    for angle_step in 0..192 {
        fallback.start = initial.start + angle_step as f32 * 2.399_963_1;
        if camera_path_is_clear(fallback, building, roof, obstacles, frame_count) {
            return fallback;
        }
    }
    fallback
}

fn with_camera_distance(
    mut parameters: CameraPathParameters,
    distance: f32,
) -> CameraPathParameters {
    let lateral_ratio = parameters.lateral_span / parameters.distance;
    parameters.distance = distance;
    parameters.lateral_span = distance * lateral_ratio;
    parameters
}

fn constrain_camera_path_to_annulus(
    mut parameters: CameraPathParameters,
    inner_radius: f32,
    outer_radius: f32,
) -> CameraPathParameters {
    parameters.distance = parameters.distance.max(inner_radius + 0.05);
    if outer_radius.is_finite() {
        parameters.distance = parameters.distance.min(outer_radius - 0.05);
    }
    let inward_room = (parameters.distance - inner_radius).max(0.0);
    let outward_room = if outer_radius.is_finite() {
        (outer_radius - parameters.distance).max(0.0)
    } else {
        f32::INFINITY
    };
    let radial_fraction = inward_room.min(outward_room) / parameters.distance;
    match parameters.kind {
        CameraPathKind::Orbit => {
            parameters.radial_motion = parameters
                .radial_motion
                .clamp(-radial_fraction, radial_fraction);
        }
        CameraPathKind::ApproachArc => {
            parameters.approach_amount = parameters.approach_amount.min(radial_fraction);
        }
        CameraPathKind::LateralWalk if outer_radius.is_finite() => {
            let maximum_span = libm::sqrtf(
                (outer_radius * outer_radius - parameters.distance * parameters.distance).max(0.0),
            );
            parameters.lateral_span = parameters.lateral_span.min(maximum_span);
        }
        CameraPathKind::LateralWalk | CameraPathKind::CornerReveal => {}
    }
    parameters
}

fn camera_path_position(parameters: CameraPathParameters, progress: f32) -> Vec3 {
    let eased = smoothstep(progress);
    let (angle, mut position) = match parameters.kind {
        CameraPathKind::Orbit => {
            let angle = parameters.start + parameters.sweep * progress;
            let radius =
                parameters.distance * (1.0 + parameters.radial_motion * (2.0 * progress - 1.0));
            (
                angle,
                Vec3::new(
                    libm::cosf(angle) * radius,
                    parameters.height,
                    libm::sinf(angle) * radius,
                ),
            )
        }
        CameraPathKind::LateralWalk => {
            let radial = Vec3::new(
                libm::cosf(parameters.start),
                0.0,
                libm::sinf(parameters.start),
            );
            let tangent = Vec3::new(-radial.z, 0.0, radial.x);
            (
                parameters.start,
                add(
                    Vec3::new(
                        radial.x * parameters.distance,
                        parameters.height,
                        radial.z * parameters.distance,
                    ),
                    scale(tangent, parameters.lateral_span * (2.0 * progress - 1.0)),
                ),
            )
        }
        CameraPathKind::ApproachArc => {
            let angle = parameters.start + parameters.sweep * 0.38 * progress;
            let radius =
                parameters.distance * (1.0 + parameters.approach_amount * (1.0 - 2.0 * progress));
            (
                angle,
                Vec3::new(
                    libm::cosf(angle) * radius,
                    parameters.height,
                    libm::sinf(angle) * radius,
                ),
            )
        }
        CameraPathKind::CornerReveal => {
            let angle = parameters.start + parameters.sweep * eased;
            (
                angle,
                Vec3::new(
                    libm::cosf(angle) * parameters.distance,
                    parameters.height,
                    libm::sinf(angle) * parameters.distance,
                ),
            )
        }
    };
    let tangent = Vec3::new(-libm::sinf(angle), 0.0, libm::cosf(angle));
    position = add(
        position,
        scale(
            tangent,
            parameters.handheld_sway * libm::sinf(TAU * progress + parameters.sway_phase),
        ),
    );
    position.y +=
        parameters.handheld_sway * 0.32 * libm::sinf(2.0 * TAU * progress + parameters.sway_phase);
    position
}

fn camera_path_is_clear(
    parameters: CameraPathParameters,
    building: SampledBuilding,
    roof: RoofEnvelope,
    obstacles: CameraSceneObstacles<'_>,
    frame_count: u32,
) -> bool {
    let mut previous = None;
    for frame_index in 0..frame_count {
        let position =
            camera_path_position(parameters, sequence_progress(frame_index, frame_count));
        if camera_point_intersects_scene(
            position,
            building,
            roof,
            obstacles.extensions,
            obstacles.backgrounds,
        ) {
            return false;
        }
        if camera_point_intersects_vegetation(position, obstacles.vegetation) {
            return false;
        }
        if camera_view_intersects_background(position, building, roof, obstacles.backgrounds) {
            return false;
        }
        if let Some(start) = previous
            && camera_segment_intersects_scene(
                start,
                position,
                building,
                roof,
                obstacles.extensions,
                obstacles.backgrounds,
            )
        {
            return false;
        }
        if let Some(start) = previous
            && camera_segment_intersects_vegetation(start, position, obstacles.vegetation)
        {
            return false;
        }
        previous = Some(position);
    }
    true
}

fn vegetation_camera_radius(vegetation: &SampledVegetation) -> f32 {
    // Match the renderer's widest low-poly foliage extent conservatively. The
    // clearance volume deliberately covers empty space beneath a crown too: a
    // camera immediately beside a trunk can still let that trunk fill the FOV.
    let extent_factor = match vegetation.kind {
        VegetationKind::DeciduousTree | VegetationKind::Palm => 1.25,
        VegetationKind::EvergreenTree => 1.0,
        VegetationKind::Shrub => 1.4,
    };
    vegetation.canopy_radius_m * extent_factor + CAMERA_COLLISION_CLEARANCE_M
}

fn vegetation_camera_bounds(vegetation: &SampledVegetation) -> (Vec3, Vec3) {
    let radius = vegetation_camera_radius(vegetation);
    (
        Vec3::new(
            vegetation.position.x - radius,
            vegetation.position.y - CAMERA_COLLISION_CLEARANCE_M,
            vegetation.position.z - radius,
        ),
        Vec3::new(
            vegetation.position.x + radius,
            vegetation.position.y + vegetation.height_m + CAMERA_COLLISION_CLEARANCE_M,
            vegetation.position.z + radius,
        ),
    )
}

pub(crate) fn camera_point_intersects_vegetation(
    point: Vec3,
    vegetation: &[SampledVegetation],
) -> bool {
    vegetation.iter().any(|item| {
        let (minimum, maximum) = vegetation_camera_bounds(item);
        point_inside_aabb(point, minimum, maximum)
    })
}

pub(crate) fn camera_segment_intersects_vegetation(
    start: Vec3,
    end: Vec3,
    vegetation: &[SampledVegetation],
) -> bool {
    vegetation.iter().any(|item| {
        let (minimum, maximum) = vegetation_camera_bounds(item);
        segment_intersects_aabb(start, end, minimum, maximum)
    })
}

fn occluder_camera_bounds(occluder: &SampledOccluder) -> (Vec3, Vec3) {
    let (half_x, half_z) = rotated_half_extents(occluder.nominal_size_m, occluder.yaw_degrees);
    camera_clearance_bounds(
        occluder.position,
        CameraObstacleSize {
            half_x,
            half_z,
            height_m: occluder.nominal_size_m.y,
        },
    )
}

pub(crate) fn camera_point_intersects_occluders(
    point: Vec3,
    occluders: &[SampledOccluder],
) -> bool {
    occluders.iter().any(|occluder| {
        let (minimum, maximum) = occluder_camera_bounds(occluder);
        point_inside_aabb(point, minimum, maximum)
    })
}

pub(crate) fn camera_segment_intersects_occluders(
    start: Vec3,
    end: Vec3,
    occluders: &[SampledOccluder],
) -> bool {
    occluders.iter().any(|occluder| {
        let (minimum, maximum) = occluder_camera_bounds(occluder);
        segment_intersects_aabb(start, end, minimum, maximum)
    })
}

pub(crate) fn camera_view_intersects_background(
    camera: Vec3,
    building: SampledBuilding,
    roof: RoofEnvelope,
    backgrounds: &[SampledBackgroundBuilding],
) -> bool {
    let roof_center = Vec3::new(
        0.0,
        building.wall_height_m + roof.maximum_height_m * 0.5,
        0.0,
    );
    backgrounds.iter().any(|item| {
        let (half_x, half_z) = rotated_half_extents(item.size_m, item.yaw_degrees);
        segment_intersects_aabb(
            camera,
            roof_center,
            Vec3::new(
                item.position.x - half_x - CAMERA_VIEW_CLEARANCE_M,
                item.position.y - CAMERA_VIEW_CLEARANCE_M,
                item.position.z - half_z - CAMERA_VIEW_CLEARANCE_M,
            ),
            Vec3::new(
                item.position.x + half_x + CAMERA_VIEW_CLEARANCE_M,
                item.position.y + item.size_m.y + CAMERA_VIEW_CLEARANCE_M,
                item.position.z + half_z + CAMERA_VIEW_CLEARANCE_M,
            ),
        )
    })
}

pub(crate) fn camera_point_intersects_scene(
    point: Vec3,
    building: SampledBuilding,
    roof: RoofEnvelope,
    extensions: &[SampledBuildingExtension],
    backgrounds: &[SampledBackgroundBuilding],
) -> bool {
    if point_inside_aabb(
        point,
        Vec3::new(
            -roof.eave_width_m * 0.5 - CAMERA_COLLISION_CLEARANCE_M,
            -CAMERA_COLLISION_CLEARANCE_M,
            -roof.eave_depth_m * 0.5 - CAMERA_COLLISION_CLEARANCE_M,
        ),
        Vec3::new(
            roof.eave_width_m * 0.5 + CAMERA_COLLISION_CLEARANCE_M,
            building.wall_height_m + roof.maximum_height_m + CAMERA_COLLISION_CLEARANCE_M,
            roof.eave_depth_m * 0.5 + CAMERA_COLLISION_CLEARANCE_M,
        ),
    ) {
        return true;
    }
    if extensions.iter().any(|extension| {
        point_inside_aabb(
            point,
            Vec3::new(
                extension.position.x - extension.size_m.x * 0.5 - CAMERA_COLLISION_CLEARANCE_M,
                -CAMERA_COLLISION_CLEARANCE_M,
                extension.position.z - extension.size_m.z * 0.5 - CAMERA_COLLISION_CLEARANCE_M,
            ),
            Vec3::new(
                extension.position.x + extension.size_m.x * 0.5 + CAMERA_COLLISION_CLEARANCE_M,
                extension.size_m.y + extension.roof_rise_m + CAMERA_COLLISION_CLEARANCE_M,
                extension.position.z + extension.size_m.z * 0.5 + CAMERA_COLLISION_CLEARANCE_M,
            ),
        )
    }) {
        return true;
    }
    backgrounds.iter().any(|item| {
        let (half_x, half_z) = rotated_half_extents(item.size_m, item.yaw_degrees);
        point_inside_aabb(
            point,
            Vec3::new(
                item.position.x - half_x - CAMERA_COLLISION_CLEARANCE_M,
                item.position.y - CAMERA_COLLISION_CLEARANCE_M,
                item.position.z - half_z - CAMERA_COLLISION_CLEARANCE_M,
            ),
            Vec3::new(
                item.position.x + half_x + CAMERA_COLLISION_CLEARANCE_M,
                item.position.y + item.size_m.y + CAMERA_COLLISION_CLEARANCE_M,
                item.position.z + half_z + CAMERA_COLLISION_CLEARANCE_M,
            ),
        )
    })
}

pub(crate) fn camera_segment_intersects_scene(
    start: Vec3,
    end: Vec3,
    building: SampledBuilding,
    roof: RoofEnvelope,
    extensions: &[SampledBuildingExtension],
    backgrounds: &[SampledBackgroundBuilding],
) -> bool {
    if segment_intersects_aabb(
        start,
        end,
        Vec3::new(
            -roof.eave_width_m * 0.5 - CAMERA_COLLISION_CLEARANCE_M,
            -CAMERA_COLLISION_CLEARANCE_M,
            -roof.eave_depth_m * 0.5 - CAMERA_COLLISION_CLEARANCE_M,
        ),
        Vec3::new(
            roof.eave_width_m * 0.5 + CAMERA_COLLISION_CLEARANCE_M,
            building.wall_height_m + roof.maximum_height_m + CAMERA_COLLISION_CLEARANCE_M,
            roof.eave_depth_m * 0.5 + CAMERA_COLLISION_CLEARANCE_M,
        ),
    ) {
        return true;
    }
    if extensions.iter().any(|extension| {
        segment_intersects_aabb(
            start,
            end,
            Vec3::new(
                extension.position.x - extension.size_m.x * 0.5 - CAMERA_COLLISION_CLEARANCE_M,
                -CAMERA_COLLISION_CLEARANCE_M,
                extension.position.z - extension.size_m.z * 0.5 - CAMERA_COLLISION_CLEARANCE_M,
            ),
            Vec3::new(
                extension.position.x + extension.size_m.x * 0.5 + CAMERA_COLLISION_CLEARANCE_M,
                extension.size_m.y + extension.roof_rise_m + CAMERA_COLLISION_CLEARANCE_M,
                extension.position.z + extension.size_m.z * 0.5 + CAMERA_COLLISION_CLEARANCE_M,
            ),
        )
    }) {
        return true;
    }
    backgrounds.iter().any(|item| {
        let (half_x, half_z) = rotated_half_extents(item.size_m, item.yaw_degrees);
        segment_intersects_aabb(
            start,
            end,
            Vec3::new(
                item.position.x - half_x - CAMERA_COLLISION_CLEARANCE_M,
                item.position.y - CAMERA_COLLISION_CLEARANCE_M,
                item.position.z - half_z - CAMERA_COLLISION_CLEARANCE_M,
            ),
            Vec3::new(
                item.position.x + half_x + CAMERA_COLLISION_CLEARANCE_M,
                item.position.y + item.size_m.y + CAMERA_COLLISION_CLEARANCE_M,
                item.position.z + half_z + CAMERA_COLLISION_CLEARANCE_M,
            ),
        )
    })
}

fn point_inside_aabb(point: Vec3, minimum: Vec3, maximum: Vec3) -> bool {
    point.x >= minimum.x
        && point.x <= maximum.x
        && point.y >= minimum.y
        && point.y <= maximum.y
        && point.z >= minimum.z
        && point.z <= maximum.z
}

fn segment_intersects_aabb(start: Vec3, end: Vec3, minimum: Vec3, maximum: Vec3) -> bool {
    let delta = sub(end, start);
    let mut near = 0.0_f32;
    let mut far = 1.0_f32;
    for (origin, direction, lower, upper) in [
        (start.x, delta.x, minimum.x, maximum.x),
        (start.y, delta.y, minimum.y, maximum.y),
        (start.z, delta.z, minimum.z, maximum.z),
    ] {
        if direction.abs() < 1.0e-6 {
            if origin < lower || origin > upper {
                return false;
            }
        } else {
            let first = (lower - origin) / direction;
            let second = (upper - origin) / direction;
            near = near.max(first.min(second));
            far = far.min(first.max(second));
            if near > far {
                return false;
            }
        }
    }
    true
}

fn solve_framing_aim(
    position: Vec3,
    target_center: Vec3,
    intent: FramingIntent,
    crop_depth_fraction: f32,
    roof_points: &[Vec3],
    intrinsics: CameraIntrinsics,
) -> Vec3 {
    if intent == FramingIntent::Centered {
        return target_center;
    }
    let forward = normalize(sub(target_center, position));
    let right = normalize(cross(forward, Vec3::new(0.0, 1.0, 0.0)));
    let up = cross(right, forward);
    let direction = match intent {
        FramingIntent::PartialLeft => right,
        FramingIntent::PartialRight => scale(right, -1.0),
        FramingIntent::PartialTop => scale(up, -1.0),
        FramingIntent::PartialBottom => up,
        FramingIntent::Centered => return target_center,
    };
    let Some(base_bounds) = projected_bounds(position, target_center, roof_points, intrinsics)
    else {
        return target_center;
    };
    let span = match intent {
        FramingIntent::PartialLeft | FramingIntent::PartialRight => {
            base_bounds.max_x - base_bounds.min_x
        }
        FramingIntent::PartialTop | FramingIntent::PartialBottom => {
            base_bounds.max_y - base_bounds.min_y
        }
        FramingIntent::Centered => 0.0,
    };
    let crop_margin = (span * crop_depth_fraction).max(2.0);
    let desired_edge = match intent {
        FramingIntent::PartialLeft | FramingIntent::PartialTop => -crop_margin,
        FramingIntent::PartialRight => intrinsics.width as f32 + crop_margin,
        FramingIntent::PartialBottom => intrinsics.height as f32 + crop_margin,
        FramingIntent::Centered => 0.0,
    };
    if framing_edge_reached(base_bounds, intent, desired_edge) {
        return target_center;
    }

    let distance = length(sub(target_center, position));
    let mut high = 1.0_f32;
    let mut high_bounds = base_bounds;
    for _ in 0..20 {
        let aim = add(target_center, scale(direction, high));
        if let Some(bounds) = projected_bounds(position, aim, roof_points, intrinsics) {
            high_bounds = bounds;
            if framing_edge_reached(bounds, intent, desired_edge) {
                break;
            }
        }
        high *= 1.6;
        if high > distance * 4.0 {
            break;
        }
    }
    if !framing_edge_reached(high_bounds, intent, desired_edge) {
        return add(target_center, scale(direction, high));
    }
    let mut low = 0.0_f32;
    for _ in 0..28 {
        let middle = (low + high) * 0.5;
        let aim = add(target_center, scale(direction, middle));
        let reached = projected_bounds(position, aim, roof_points, intrinsics)
            .is_some_and(|bounds| framing_edge_reached(bounds, intent, desired_edge));
        if reached {
            high = middle;
        } else {
            low = middle;
        }
    }
    add(target_center, scale(direction, high))
}

fn framing_edge_reached(bounds: ImageBounds, intent: FramingIntent, desired: f32) -> bool {
    match intent {
        FramingIntent::PartialLeft => bounds.min_x <= desired,
        FramingIntent::PartialRight => bounds.max_x >= desired,
        FramingIntent::PartialTop => bounds.min_y <= desired,
        FramingIntent::PartialBottom => bounds.max_y >= desired,
        FramingIntent::Centered => true,
    }
}

fn projected_bounds(
    position: Vec3,
    aim: Vec3,
    points: &[Vec3],
    intrinsics: CameraIntrinsics,
) -> Option<ImageBounds> {
    let forward = normalize(sub(aim, position));
    let right = normalize(cross(forward, Vec3::new(0.0, 1.0, 0.0)));
    let up = cross(right, forward);
    let mut bounds = ImageBounds {
        min_x: f32::INFINITY,
        max_x: f32::NEG_INFINITY,
        min_y: f32::INFINITY,
        max_y: f32::NEG_INFINITY,
    };
    for point in points {
        let relative = sub(*point, position);
        let depth = dot(relative, forward);
        if depth <= 1.0e-4 {
            return None;
        }
        let image_x = intrinsics.fx * dot(relative, right) / depth + intrinsics.cx;
        let image_y = intrinsics.cy - intrinsics.fy * dot(relative, up) / depth;
        bounds.min_x = bounds.min_x.min(image_x);
        bounds.max_x = bounds.max_x.max(image_x);
        bounds.min_y = bounds.min_y.min(image_y);
        bounds.max_y = bounds.max_y.max(image_y);
    }
    Some(bounds)
}

fn target_roof_control_points(building: SampledBuilding, roof: SampledRoof) -> Vec<Vec3> {
    let eave_y = building.wall_height_m;
    let shoulder_y = eave_y + roof.lower_rise_m;
    let crown_y = shoulder_y + roof.upper_rise_m;
    let rectangle = |width: f32, depth: f32, y: f32| {
        [
            Vec3::new(-width * 0.5, y, -depth * 0.5),
            Vec3::new(width * 0.5, y, -depth * 0.5),
            Vec3::new(width * 0.5, y, depth * 0.5),
            Vec3::new(-width * 0.5, y, depth * 0.5),
        ]
    };
    let eave = rectangle(roof.eave_width_m, roof.eave_depth_m, eave_y);
    let shoulder = rectangle(roof.shoulder_width_m, roof.shoulder_depth_m, shoulder_y);
    let crown = rectangle(roof.crown_top_width_m, roof.crown_top_depth_m, crown_y);
    vec![
        eave[0],
        eave[1],
        eave[2],
        eave[3],
        shoulder[0],
        shoulder[1],
        shoulder[2],
        shoulder[3],
        crown[0],
        crown[1],
        crown[2],
        crown[3],
    ]
}

fn ordinary_roof_control_points(building: SampledBuilding, roof: SampledOrdinaryRoof) -> Vec<Vec3> {
    let x = roof.eave_width_m * 0.5;
    let z = roof.eave_depth_m * 0.5;
    let y = building.wall_height_m;
    let top = y + roof.rise_m;
    let mut points = vec![
        Vec3::new(-x, y, -z),
        Vec3::new(x, y, -z),
        Vec3::new(x, y, z),
        Vec3::new(-x, y, z),
    ];
    match roof.family {
        OrdinaryRoofFamily::Flat => points.extend([
            Vec3::new(-x, top, -z),
            Vec3::new(x, top, -z),
            Vec3::new(x, top, z),
            Vec3::new(-x, top, z),
        ]),
        OrdinaryRoofFamily::Gable => {
            points.extend([Vec3::new(-x, top, 0.0), Vec3::new(x, top, 0.0)]);
        }
        OrdinaryRoofFamily::Hip => {
            let ridge_x = x * roof.ridge_length_fraction;
            points.extend([Vec3::new(-ridge_x, top, 0.0), Vec3::new(ridge_x, top, 0.0)]);
        }
        OrdinaryRoofFamily::Shed => {
            points.extend([Vec3::new(-x, top, z), Vec3::new(x, top, z)]);
        }
        OrdinaryRoofFamily::Mansard => {
            let inset_x = x * roof.inset_fraction;
            let inset_z = z * roof.inset_fraction;
            points.extend([
                Vec3::new(-inset_x, top, -inset_z),
                Vec3::new(inset_x, top, -inset_z),
                Vec3::new(inset_x, top, inset_z),
                Vec3::new(-inset_x, top, inset_z),
            ]);
        }
        OrdinaryRoofFamily::Pyramid => points.push(Vec3::new(0.0, top, 0.0)),
        OrdinaryRoofFamily::Cupola => {
            points.push(Vec3::new(0.0, top, 0.0));
            let cupola_x = x * roof.inset_fraction.clamp(0.12, 0.32);
            let cupola_z = z * roof.inset_fraction.clamp(0.12, 0.32);
            let body_top = top + roof.cap_height_m * 0.62;
            points.extend([
                Vec3::new(-cupola_x, top, -cupola_z),
                Vec3::new(cupola_x, top, -cupola_z),
                Vec3::new(cupola_x, top, cupola_z),
                Vec3::new(-cupola_x, top, cupola_z),
                Vec3::new(-cupola_x, body_top, -cupola_z),
                Vec3::new(cupola_x, body_top, -cupola_z),
                Vec3::new(cupola_x, body_top, cupola_z),
                Vec3::new(-cupola_x, body_top, cupola_z),
                Vec3::new(0.0, top + roof.cap_height_m, 0.0),
            ]);
        }
    }
    points
}

pub(crate) fn actual_roof_envelope(scene: &SampledScene) -> RoofEnvelope {
    scene
        .ordinary_roof
        .map_or_else(|| RoofEnvelope::target(scene.roof), RoofEnvelope::ordinary)
}

fn sequence_progress(frame_index: u32, frame_count: u32) -> f32 {
    if frame_count <= 1 {
        0.5
    } else {
        frame_index as f32 / (frame_count - 1) as f32
    }
}

pub(crate) fn rotated_half_extents(size: Vec3, yaw_degrees: f32) -> (f32, f32) {
    let yaw = yaw_degrees.to_radians();
    let cosine = libm::cosf(yaw).abs();
    let sine = libm::sinf(yaw).abs();
    (
        cosine * size.x * 0.5 + sine * size.z * 0.5,
        sine * size.x * 0.5 + cosine * size.z * 0.5,
    )
}

pub(crate) fn sampled_content_half_extent(
    building: SampledBuilding,
    roof: RoofEnvelope,
    composition: &SampledComposition,
    occluders: &[SampledOccluder],
    frames: &[CameraFramePlan],
) -> f32 {
    let mut extent = (roof.eave_width_m * 0.5).max(roof.eave_depth_m * 0.5);
    for extension in &composition.building_extensions {
        extent = extent
            .max(extension.position.x.abs() + extension.size_m.x * 0.5)
            .max(extension.position.z.abs() + extension.size_m.z * 0.5);
    }
    for background in &composition.background_buildings {
        let (half_x, half_z) = rotated_half_extents(background.size_m, background.yaw_degrees);
        extent = extent
            .max(background.position.x.abs() + half_x)
            .max(background.position.z.abs() + half_z);
    }
    for vegetation in &composition.vegetation {
        extent = extent
            .max(vegetation.position.x.abs() + vegetation.canopy_radius_m)
            .max(vegetation.position.z.abs() + vegetation.canopy_radius_m);
    }
    if let Some(infrastructure) = &composition.infrastructure {
        let parking_span = infrastructure.parking_bay_size_m.x.max(
            infrastructure.parking_bay_size_m.y
                * libm::sqrtf(infrastructure.parking_bays.max(1) as f32),
        );
        extent = extent
            .max(infrastructure.parking_center.x.abs() + parking_span)
            .max(infrastructure.parking_center.z.abs() + parking_span)
            .max(
                infrastructure.road_center_z_m.abs()
                    + infrastructure.road_lanes as f32 * infrastructure.lane_width_m * 0.5,
            );
        for pole in &infrastructure.utility_poles {
            extent = extent
                .max(pole.position.x.abs() + 0.25)
                .max(pole.position.z.abs() + 0.25);
        }
    }
    for occluder in occluders {
        let (half_x, half_z) = rotated_half_extents(occluder.nominal_size_m, occluder.yaw_degrees);
        extent = extent
            .max(occluder.position.x.abs() + half_x)
            .max(occluder.position.z.abs() + half_z);
    }
    for frame in frames {
        let position = frame.camera.world_from_camera.translation;
        extent = extent.max(position.x.abs()).max(position.z.abs());
    }
    extent.max((building.footprint_width_m * 0.5).max(building.footprint_depth_m * 0.5))
}

/// Fully sampled building shell.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledBuilding {
    /// Exterior width in metres.
    pub footprint_width_m: f32,
    /// Exterior depth in metres.
    pub footprint_depth_m: f32,
    /// Wall height to the roof eave in metres.
    pub wall_height_m: f32,
    /// Ground-plane half extent in metres, expanded after composition so every
    /// finite scene element and camera position retains an edge margin.
    pub ground_half_extent_m: f32,
}

/// Fully sampled two-stage roof geometry.
///
/// These names intentionally map directly to the shared roof-geometry parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledRoof {
    /// Correlated architectural proportion family used to generate the dimensions.
    #[serde(default)]
    pub morphology: RoofMorphology,
    /// Overall eave width in metres.
    pub eave_width_m: f32,
    /// Overall eave depth in metres.
    pub eave_depth_m: f32,
    /// Upper shoulder width in metres.
    pub shoulder_width_m: f32,
    /// Upper shoulder depth in metres.
    pub shoulder_depth_m: f32,
    /// Eave-to-shoulder rise in metres.
    pub lower_rise_m: f32,
    /// Crown-base to crown-top rise in metres.
    pub upper_rise_m: f32,
    /// Full width of the inset crown-top rectangle in metres.
    pub crown_top_width_m: f32,
    /// Full depth of the inset crown-top rectangle in metres.
    pub crown_top_depth_m: f32,
    /// Signed bounded perturbation available to family-specific geometry.
    pub asymmetry_fraction: f32,
}

/// Ordinary non-target roof families used for same-renderer negative examples.
///
/// None of these families contains the target's two stacked rectangular
/// frusta. They deliberately share the target generator's building shell,
/// materials, lighting, signage, and surroundings so roof form remains the
/// discriminating signal.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OrdinaryRoofFamily {
    /// Nearly level commercial roof with a shallow fascia.
    Flat,
    /// Two planes meeting along a ridge.
    Gable,
    /// Four planes meeting along a shortened ridge.
    Hip,
    /// One continuous mono-pitch plane.
    Shed,
    /// Steep perimeter slopes terminating at one flat top.
    Mansard,
    /// Four planes meeting at a single apex.
    Pyramid,
    /// Conventional hipped roof with a small ventilating cupola.
    Cupola,
}

impl OrdinaryRoofFamily {
    /// Stable order used to balance generated negative corpora.
    pub const ALL: [Self; 7] = [
        Self::Flat,
        Self::Gable,
        Self::Hip,
        Self::Shed,
        Self::Mansard,
        Self::Pyramid,
        Self::Cupola,
    ];

    /// Stable request/manifest spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Flat => "flat",
            Self::Gable => "gable",
            Self::Hip => "hip",
            Self::Shed => "shed",
            Self::Mansard => "mansard",
            Self::Pyramid => "pyramid",
            Self::Cupola => "cupola",
        }
    }

    fn from_building_family(value: &str) -> Option<Self> {
        let suffix = value.strip_prefix("ordinary_").unwrap_or(value);
        Self::ALL
            .into_iter()
            .find(|family| family.as_str() == suffix)
    }
}

/// Fully sampled dimensions for an ordinary non-target roof.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledOrdinaryRoof {
    /// Procedural roof topology.
    pub family: OrdinaryRoofFamily,
    /// Overall roof width including overhang.
    pub eave_width_m: f32,
    /// Overall roof depth including overhang.
    pub eave_depth_m: f32,
    /// Main vertical rise above the wall plate.
    pub rise_m: f32,
    /// Ridge length as a fraction of eave width for ridge-bearing families.
    pub ridge_length_fraction: f32,
    /// Width/depth fraction of a flat mansard top or cupola body.
    pub inset_fraction: f32,
    /// Additional cupola body/cap height; zero for other families.
    pub cap_height_m: f32,
}

impl SampledOrdinaryRoof {
    /// Highest point above the wall plate, used by camera and clutter planning.
    #[must_use]
    pub fn maximum_height_m(self) -> f32 {
        self.rise_m + self.cap_height_m
    }
}

impl SampledRoof {
    /// Returns the exact parameter names consumed by `roof-geometry`.
    #[must_use]
    pub fn geometry_parameters(self) -> BTreeMap<String, f32> {
        [
            ("eave_width", self.eave_width_m),
            ("eave_depth", self.eave_depth_m),
            ("shoulder_width", self.shoulder_width_m),
            ("shoulder_depth", self.shoulder_depth_m),
            ("lower_rise", self.lower_rise_m),
            ("upper_rise", self.upper_rise_m),
            ("crown_top_width", self.crown_top_width_m),
            ("crown_top_depth", self.crown_top_depth_m),
        ]
        .into_iter()
        .map(|(name, value)| (name.to_owned(), value))
        .collect()
    }
}

/// Fully resolved surface material.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledMaterial {
    /// Stable palette identifier.
    pub id: String,
    /// Nonlinear sRGB base colour.
    pub base_color_srgb: [f32; 3],
    /// PBR roughness.
    pub roughness: f32,
    /// Weathering amount in `[0, 1]`.
    pub weathering: f32,
}

/// Fully sampled outdoor illumination.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledLighting {
    /// Sun bearing around world +Y, in degrees.
    pub sun_azimuth_degrees: f32,
    /// Sun elevation in degrees.
    pub sun_elevation_degrees: f32,
    /// Direct-light renderer intensity.
    pub sun_intensity: f32,
    /// Indirect sky renderer intensity.
    pub sky_intensity: f32,
    /// Cloud coverage in `[0, 1]`.
    pub cloud_coverage: f32,
    /// Atmospheric haze in `[0, 1]`.
    pub haze: f32,
}

/// Renderer-independent placement of a coarse occluder category.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledOccluder {
    /// Category later resolved to an asset by the scene renderer.
    pub kind: OccluderKind,
    /// World-space base position.
    pub position: Vec3,
    /// World-Y rotation in degrees.
    pub yaw_degrees: f32,
    /// Uniform asset scale multiplier.
    pub scale: f32,
    /// Camera/site relationship used by composition and coverage checks.
    #[serde(default)]
    pub placement: OccluderPlacement,
    /// Approximate width, height, and depth after scaling, in metres.
    #[serde(default)]
    pub nominal_size_m: Vec3,
}

/// Placement regime for a coarse occluding object.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum OccluderPlacement {
    /// General site clutter independent of the camera path.
    #[default]
    Site,
    /// Intentionally between the coherent camera path and the target.
    Foreground,
    /// Equipment placed within the roof footprint.
    Rooftop,
}

/// Complete scene state shared by every frame in a sequence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SampledScene {
    /// Building envelope.
    pub building: SampledBuilding,
    /// Two-tier target dimensions. In negative scenes these are retained only
    /// as a matched distribution reference; camera planning uses
    /// `ordinary_roof`, which is the geometry actually rendered.
    pub roof: SampledRoof,
    /// Non-target roof rendered for negative scenes. When present it replaces,
    /// rather than relabels, the two-stage target geometry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ordinary_roof: Option<SampledOrdinaryRoof>,
    /// Roof finish.
    pub roof_material: SampledMaterial,
    /// Wall finish.
    pub wall_material: SampledMaterial,
    /// Lighting environment.
    pub lighting: SampledLighting,
    /// Independent foreground, background, and rooftop clutter.
    pub occluders: Vec<SampledOccluder>,
    /// Correlated site-realism and environmental plan.
    #[serde(default)]
    pub composition: SampledComposition,
}

/// Smooth camera-path family shared by a coherent sequence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CameraPathKind {
    /// Curved walk around the building.
    #[default]
    Orbit,
    /// Predominantly sideways translation from one façade view.
    LateralWalk,
    /// Simultaneous approach and modest orbit.
    ApproachArc,
    /// Eased orbit that reveals another corner late in the sequence.
    CornerReveal,
}

/// Smooth focal-length behavior over one sequence.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ZoomBehavior {
    /// Constant horizontal field of view.
    #[default]
    Fixed,
    /// Horizontal field of view decreases smoothly.
    SmoothIn,
    /// Horizontal field of view increases smoothly.
    SmoothOut,
}

/// Deliberate composition intent. Actual coverage remains render-derived ground truth.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FramingIntent {
    /// Keep the target near the image centre.
    #[default]
    Centered,
    /// Deliberately crop the target's left side.
    PartialLeft,
    /// Deliberately crop the target's right side.
    PartialRight,
    /// Deliberately crop the target's upper side.
    PartialTop,
    /// Deliberately crop the target's lower side.
    PartialBottom,
}

/// Apparent primary-building scale selected before exact camera placement.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ApparentScale {
    /// Building roof spans roughly 15--30% of image width.
    Distant,
    /// Building roof spans roughly 30--70% of image width.
    #[default]
    Normal,
    /// Building roof spans roughly 70--95% of image width.
    Close,
    /// Intentional edge truncation.
    Partial,
}

/// Sequence-level camera plan explaining the exact per-frame poses and intrinsics.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CameraMotionPlan {
    /// Coherent translation family.
    pub path_kind: CameraPathKind,
    /// Constant or smooth focal-length behavior.
    pub zoom_behavior: ZoomBehavior,
    /// Centered or deliberately partial composition.
    pub framing_intent: FramingIntent,
    /// Distant, normal, close, or deliberately partial framing stratum.
    #[serde(default)]
    pub apparent_scale: ApparentScale,
    /// Desired roof-width/image-width ratio used to choose distance. It is an
    /// intent, not a semantic label; values over one deliberately request a crop.
    pub target_width_fraction_goal: f32,
    /// Initial horizontal field of view in degrees.
    pub start_horizontal_fov_degrees: f32,
    /// Final horizontal field of view in degrees.
    pub end_horizontal_fov_degrees: f32,
    /// World-space point of interest before the sampled framing offset.
    pub target_center: Vec3,
    /// World-space aim offset at the sequence midpoint. Curved paths rotate the
    /// corresponding offset in camera right/up space per frame.
    pub framing_offset: Vec3,
    /// Smooth positional sway amplitude in metres.
    pub handheld_sway_m: f32,
}

impl Default for CameraMotionPlan {
    fn default() -> Self {
        Self {
            path_kind: CameraPathKind::Orbit,
            zoom_behavior: ZoomBehavior::Fixed,
            framing_intent: FramingIntent::Centered,
            apparent_scale: ApparentScale::Normal,
            target_width_fraction_goal: 0.5,
            start_horizontal_fov_degrees: 60.0,
            end_horizontal_fov_degrees: 60.0,
            target_center: Vec3::new(0.0, 0.0, 0.0),
            framing_offset: Vec3::new(0.0, 0.0, 0.0),
            handheld_sway_m: 0.0,
        }
    }
}

/// Exact camera associated with a generation frame.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CameraFramePlan {
    /// Zero-based frame index.
    pub frame_index: u32,
    /// Nominal sequence time in nanoseconds.
    pub timestamp_ns: u64,
    /// Pose and intrinsics used by every render pass.
    pub camera: CameraModel,
}

/// Deterministically sampled scene and coherent camera path, ready to render.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequencePlan {
    /// Stable sequence ID derived from request identity.
    pub sequence_id: String,
    /// Original building-level request.
    pub request: SequenceRequest,
    /// Hash of the complete serialized generator configuration.
    pub config_fingerprint: String,
    /// Sampled scene state.
    pub scene: SampledScene,
    /// Ordered camera views.
    pub frames: Vec<CameraFramePlan>,
    /// Coherent path, zoom, and composition intent behind the exact frame cameras.
    #[serde(default)]
    pub camera_motion: CameraMotionPlan,
}

impl SequencePlan {
    /// Returns the exact sample key for an in-range frame index.
    #[must_use]
    pub fn frame_key(&self, frame_index: u32) -> Option<String> {
        (frame_index < self.frames.len() as u32).then(|| frame_key(&self.sequence_id, frame_index))
    }

    /// Creates the compact roof instance used by final frame records.
    ///
    /// Negative scenes render their independent ordinary roof and never expose
    /// the target parameterization as supervision.
    #[must_use]
    pub fn roof_instance(&self) -> Option<RoofInstanceRecord> {
        (self.request.target_kind != TargetKind::Negative).then(|| RoofInstanceRecord {
            family: self.request.building_family.clone(),
            world_from_roof: RigidTransform {
                translation: Vec3::new(0.0, self.scene.building.wall_height_m, 0.0),
                rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
            },
            parameters: self.scene.roof.geometry_parameters(),
        })
    }

    /// Applies the building-level split policy and creates a sequence record.
    pub fn into_record(self, policy: &SplitPolicy) -> Result<SequenceRecord, SplitPolicyError> {
        let split = policy.assign(&self.request.split_key())?;
        let frames = self
            .frames
            .iter()
            .map(|frame| SequenceFrameRef {
                sample_key: frame_key(&self.sequence_id, frame.frame_index),
                frame_index: frame.frame_index,
                timestamp_ns: frame.timestamp_ns,
            })
            .collect();

        Ok(SequenceRecord {
            schema_version: crate::DATASET_SCHEMA_VERSION.to_owned(),
            sequence_id: self.sequence_id,
            building_family: self.request.building_family,
            building_seed: self.request.building_seed,
            source_asset_group: self.request.source_asset_group,
            split,
            target_kind: self.request.target_kind,
            config_fingerprint: self.config_fingerprint,
            scene: self.scene,
            camera_motion: self.camera_motion,
            frames,
        })
    }

    /// Computes the split without consuming the plan.
    pub fn split(&self, policy: &SplitPolicy) -> Result<DatasetSplit, SplitPolicyError> {
        policy.assign(&self.request.split_key())
    }
}

/// Validated deterministic sampler. It is cheap to clone and safe to use per job.
#[derive(Clone, Debug)]
pub struct SequenceSampler {
    config: GeneratorConfig,
    config_fingerprint: String,
}

impl SequenceSampler {
    /// Validates and fingerprints a generator configuration.
    pub fn new(config: GeneratorConfig) -> Result<Self, SamplingError> {
        let report = config.validate();
        if !report.is_valid() {
            return Err(SamplingError::InvalidConfig(report));
        }
        let encoded = serde_json::to_vec(&config)?;
        let hash = stable_hash64(&[b"generator-config-v2", &encoded]);
        Ok(Self {
            config,
            config_fingerprint: format!("stable64:{hash:016x}"),
        })
    }

    /// Returns the validated configuration.
    #[must_use]
    pub const fn config(&self) -> &GeneratorConfig {
        &self.config
    }

    /// Returns the reproducibility fingerprint stored in plans and manifests.
    #[must_use]
    pub fn config_fingerprint(&self) -> &str {
        &self.config_fingerprint
    }

    /// Samples a scene and camera path from independent ChaCha20 substreams.
    pub fn sample(&self, request: SequenceRequest) -> Result<SequencePlan, SamplingError> {
        if request.building_family.trim().is_empty() {
            return Err(SamplingError::EmptyBuildingFamily);
        }

        let roof_profile = self.sample_roof_profile(request.building_seed);
        let mut building = self.sample_building(request.building_seed, roof_profile);
        let roof = self.sample_roof(request.building_seed, building, roof_profile);
        let ordinary_roof = (request.target_kind == TargetKind::Negative)
            .then(|| self.sample_ordinary_roof(request.building_seed, building, roof, &request));
        let roof_envelope =
            ordinary_roof.map_or_else(|| RoofEnvelope::target(roof), RoofEnvelope::ordinary);
        let roof_material = self.sample_material(request.building_seed, true);
        let wall_material = self.sample_material(request.building_seed, false);
        let domain = self.sample_domain(request.building_seed);
        let (lighting, environment) = self.sample_environment(request.building_seed, domain);
        let composition =
            self.sample_composition(request.building_seed, building, environment, domain);
        let (camera_motion, frames) = self.sample_cameras(
            request.building_seed,
            building,
            roof,
            ordinary_roof,
            CameraSceneObstacles {
                extensions: &composition.building_extensions,
                backgrounds: &composition.background_buildings,
                vegetation: &composition.vegetation,
            },
        );
        let occluders = self.sample_occluders(request.building_seed, building, &frames);
        building.ground_half_extent_m = building.ground_half_extent_m.max(
            sampled_content_half_extent(building, roof_envelope, &composition, &occluders, &frames)
                + GROUND_EDGE_MARGIN_M,
        );
        let target_kind_id = match request.target_kind {
            TargetKind::Target => b"target".as_slice(),
            TargetKind::NearMiss => b"near_miss".as_slice(),
            TargetKind::Negative => b"negative".as_slice(),
        };
        let id_hash = stable_hash64(&[
            b"sequence-id-v2",
            request.building_family.as_bytes(),
            &request.building_seed.to_le_bytes(),
            request
                .source_asset_group
                .as_deref()
                .unwrap_or("")
                .as_bytes(),
            target_kind_id,
            self.config_fingerprint.as_bytes(),
        ]);

        Ok(SequencePlan {
            sequence_id: format!("seq-{id_hash:016x}"),
            request,
            config_fingerprint: self.config_fingerprint.clone(),
            scene: SampledScene {
                building,
                roof,
                ordinary_roof,
                roof_material,
                wall_material,
                lighting,
                occluders,
                composition,
            },
            frames,
            camera_motion,
        })
    }

    fn sample_roof_profile(&self, seed: u64) -> RoofMorphologyProfile {
        let mut rng = rng(seed, "roof-morphology");
        *choose_roof_profile(&mut rng, &self.config.roof.profiles)
    }

    fn sample_building(&self, seed: u64, profile: RoofMorphologyProfile) -> SampledBuilding {
        let mut rng = rng(seed, "building");
        let scene = self.config.scene;
        let globally_feasible_aspect = FloatRange::new(
            scene.footprint_width_m.min / scene.footprint_depth_m.max,
            scene.footprint_width_m.max / scene.footprint_depth_m.min,
        );
        let aspect_range = profile
            .footprint_aspect_ratio
            .intersection(globally_feasible_aspect)
            .expect("validated roof profile must overlap the supported footprint aspect ratios");
        let aspect_ratio = sample_range(&mut rng, aspect_range);
        let width_range = scene
            .footprint_width_m
            .intersection(FloatRange::new(
                scene.footprint_depth_m.min * aspect_ratio,
                scene.footprint_depth_m.max * aspect_ratio,
            ))
            .expect("sampled supported aspect ratio must admit a building footprint");
        let footprint_width_m = sample_range(&mut rng, width_range);
        let footprint_depth_m = (footprint_width_m / aspect_ratio)
            .clamp(scene.footprint_depth_m.min, scene.footprint_depth_m.max);
        SampledBuilding {
            footprint_width_m,
            footprint_depth_m,
            wall_height_m: sample_range(&mut rng, scene.wall_height_m),
            ground_half_extent_m: sample_range(&mut rng, scene.ground_half_extent_m),
        }
    }

    fn sample_roof(
        &self,
        seed: u64,
        building: SampledBuilding,
        profile: RoofMorphologyProfile,
    ) -> SampledRoof {
        let mut rng = rng(seed, "roof");
        let roof = &self.config.roof;
        let overhang = sample_range(
            &mut rng,
            intersect_supported(roof.overhang_m, profile.overhang_m),
        );
        let eave_width = building.footprint_width_m + 2.0 * overhang;
        let eave_depth = building.footprint_depth_m + 2.0 * overhang;
        let shoulder_width = eave_width
            * sample_range(
                &mut rng,
                intersect_supported(
                    roof.shoulder_width_fraction,
                    profile.shoulder_width_fraction,
                ),
            );
        let shoulder_depth = eave_depth
            * sample_range(
                &mut rng,
                intersect_supported(
                    roof.shoulder_depth_fraction,
                    profile.shoulder_depth_fraction,
                ),
            );
        let crown_top_width = shoulder_width
            * sample_range(
                &mut rng,
                intersect_supported(
                    roof.crown_top_width_fraction,
                    profile.crown_top_width_fraction,
                ),
            );
        let crown_top_depth = shoulder_depth
            * sample_range(
                &mut rng,
                intersect_supported(
                    roof.crown_top_depth_fraction,
                    profile.crown_top_depth_fraction,
                ),
            );
        SampledRoof {
            morphology: profile.morphology,
            eave_width_m: eave_width,
            eave_depth_m: eave_depth,
            shoulder_width_m: shoulder_width,
            shoulder_depth_m: shoulder_depth,
            lower_rise_m: sample_range(
                &mut rng,
                intersect_supported(roof.lower_rise_m, profile.lower_rise_m),
            ),
            upper_rise_m: sample_range(
                &mut rng,
                intersect_supported(roof.upper_rise_m, profile.upper_rise_m),
            ),
            crown_top_width_m: crown_top_width,
            crown_top_depth_m: crown_top_depth,
            asymmetry_fraction: sample_range(&mut rng, roof.asymmetry_fraction),
        }
    }

    fn sample_ordinary_roof(
        &self,
        seed: u64,
        _building: SampledBuilding,
        matched_target_roof: SampledRoof,
        request: &SequenceRequest,
    ) -> SampledOrdinaryRoof {
        let mut rng = rng(seed, "ordinary-roof");
        let family = OrdinaryRoofFamily::from_building_family(&request.building_family)
            .unwrap_or_else(|| {
                OrdinaryRoofFamily::ALL[rng.random_range(0..OrdinaryRoofFamily::ALL.len())]
            });
        let rise_m = match family {
            OrdinaryRoofFamily::Flat => sample_range(&mut rng, FloatRange::new(0.16, 0.48)),
            OrdinaryRoofFamily::Gable | OrdinaryRoofFamily::Hip => {
                sample_range(&mut rng, FloatRange::new(1.4, 4.6))
            }
            OrdinaryRoofFamily::Shed => sample_range(&mut rng, FloatRange::new(1.0, 3.8)),
            OrdinaryRoofFamily::Mansard => sample_range(&mut rng, FloatRange::new(1.8, 4.2)),
            OrdinaryRoofFamily::Pyramid => sample_range(&mut rng, FloatRange::new(2.0, 5.2)),
            OrdinaryRoofFamily::Cupola => sample_range(&mut rng, FloatRange::new(1.2, 3.4)),
        };
        SampledOrdinaryRoof {
            family,
            // Use the exact same sampled eave envelope as the paired target.
            // This prevents overhang or building size from becoming a class cue.
            eave_width_m: matched_target_roof.eave_width_m,
            eave_depth_m: matched_target_roof.eave_depth_m,
            rise_m,
            ridge_length_fraction: sample_range(&mut rng, FloatRange::new(0.28, 0.72)),
            inset_fraction: sample_range(&mut rng, FloatRange::new(0.18, 0.68)),
            cap_height_m: if family == OrdinaryRoofFamily::Cupola {
                sample_range(&mut rng, FloatRange::new(0.8, 2.1))
            } else {
                0.0
            },
        }
    }

    fn sample_material(&self, seed: u64, roof: bool) -> SampledMaterial {
        let domain = if roof {
            "roof-material"
        } else {
            "wall-material"
        };
        let mut rng = rng(seed, domain);
        let choices = if roof {
            &self.config.materials.roof
        } else {
            &self.config.materials.walls
        };
        let choice = choose_material(&mut rng, choices);
        resolve_material_choice(&mut rng, choice)
    }

    fn sample_domain(&self, seed: u64) -> SceneDomainProfile {
        let mut rng = rng(seed, "site-domain");
        *choose_domain(&mut rng, &self.config.composition.domains.profiles)
    }

    fn sample_environment(
        &self,
        seed: u64,
        domain: SceneDomainProfile,
    ) -> (SampledLighting, SampledEnvironment) {
        let mut weather_rng = rng(seed, "weather");
        let weather = *choose_weather(&mut weather_rng, &self.config.composition.weather.profiles);
        let mut phase_rng = rng(seed, "day-phase");
        let phase = *choose_day_phase(&mut phase_rng, &self.config.composition.day_phase.profiles);
        let mut light_rng = rng(seed, "lighting");
        let light = self.config.lighting;
        let sun_elevation = sample_range(
            &mut light_rng,
            intersect_ranges(light.sun_elevation_degrees, phase.sun_elevation_degrees),
        );
        let mut sun_intensity = sample_range(
            &mut light_rng,
            intersect_ranges(light.sun_intensity, weather.sun_intensity),
        );
        let mut sky_intensity = sample_range(
            &mut light_rng,
            intersect_ranges(light.sky_intensity, weather.sky_intensity),
        );
        let domain_light = match domain.domain {
            SceneDomain::City => 1.30,
            SceneDomain::Urban => 1.15,
            SceneDomain::Suburban => 1.0,
            SceneDomain::Roadside => 0.78,
            SceneDomain::Remote => 0.42,
        };
        match phase.phase {
            DayPhase::Day => {}
            DayPhase::Twilight => {
                sun_intensity *= 0.28;
                sky_intensity *= 0.62;
            }
            DayPhase::Night => {
                sun_intensity = 0.0;
                sky_intensity *= 0.12 * domain_light;
            }
        }
        let artificial_light_strength =
            sample_range(&mut phase_rng, phase.artificial_light_strength) * domain_light;
        let color_temperature_k = match phase.phase {
            DayPhase::Day => sample_range(&mut light_rng, weather.color_temperature_k),
            DayPhase::Twilight => sample_range(
                &mut light_rng,
                preferred_intersection(
                    weather.color_temperature_k,
                    FloatRange::new(3_400.0, 6_800.0),
                ),
            ),
            DayPhase::Night => sample_range(
                &mut light_rng,
                preferred_intersection(
                    weather.color_temperature_k,
                    FloatRange::new(2_600.0, 4_800.0),
                ),
            ),
        };
        (
            SampledLighting {
                sun_azimuth_degrees: sample_range(&mut light_rng, FloatRange::new(0.0, 360.0)),
                sun_elevation_degrees: sun_elevation,
                sun_intensity,
                sky_intensity,
                cloud_coverage: sample_range(
                    &mut light_rng,
                    intersect_ranges(light.cloud_coverage, weather.cloud_coverage),
                ),
                haze: sample_range(&mut light_rng, intersect_ranges(light.haze, weather.haze)),
            },
            SampledEnvironment {
                day_phase: phase.phase,
                domain: domain.domain,
                weather: weather.preset,
                shadow_softness: sample_range(&mut light_rng, weather.shadow_softness),
                ground_wetness: sample_range(&mut light_rng, weather.ground_wetness),
                visibility_km: sample_range(&mut light_rng, weather.visibility_km),
                color_temperature_k,
                camera_exposure_ev100: sample_range(&mut phase_rng, phase.camera_exposure_ev100),
                artificial_light_strength,
            },
        )
    }

    fn sample_composition(
        &self,
        seed: u64,
        building: SampledBuilding,
        environment: SampledEnvironment,
        domain: SceneDomainProfile,
    ) -> SampledComposition {
        let config = &self.config.composition;

        let mut ground_rng = rng(seed, "ground-material");
        let ground_material = resolved_material(&mut ground_rng, &config.ground_materials);

        let mut facade_rng = rng(seed, "facade");
        let facade_config = &config.facade;
        let facade = SampledFacade {
            glazing_fraction: sample_range(&mut facade_rng, facade_config.glazing_fraction),
            glazing_material: resolved_material(&mut facade_rng, &facade_config.glazing_materials),
            entrance_side: sample_facade_side(&mut facade_rng),
            entrance_width_m: sample_range(&mut facade_rng, facade_config.entrance_width_m)
                .min(building.footprint_width_m.min(building.footprint_depth_m) * 0.32),
            weathering: sample_range(&mut facade_rng, facade_config.weathering),
        };

        let signage = self.sample_signage(seed, building, environment.day_phase);
        let building_extensions = self.sample_building_extensions(seed, building);
        let background_buildings = self.sample_background_buildings(seed, building, domain);
        let vegetation = self.sample_vegetation(seed, building, domain);
        let infrastructure = self.sample_infrastructure(seed, building, domain);

        SampledComposition {
            source_asset_ids: Vec::new(),
            environment: Some(environment),
            ground_material: Some(ground_material),
            facade: Some(facade),
            building_extensions,
            signage,
            background_buildings,
            vegetation,
            infrastructure: Some(infrastructure),
        }
    }

    fn sample_building_extensions(
        &self,
        seed: u64,
        building: SampledBuilding,
    ) -> Vec<SampledBuildingExtension> {
        let mut rng = rng(seed, "building-extensions");
        let config = self.config.composition.building_extensions;
        let total = u64::from(config.none_weight)
            + u64::from(config.one_weight)
            + u64::from(config.two_weight);
        let draw = rng.random_range(0..total);
        let count = if draw < u64::from(config.none_weight) {
            0
        } else if draw < u64::from(config.none_weight) + u64::from(config.one_weight) {
            1
        } else {
            2
        };
        if count == 0 {
            return Vec::new();
        }

        let first_facade = sample_facade_side(&mut rng);
        let facades = if count == 1 {
            [first_facade, first_facade]
        } else {
            [first_facade, opposite_facade(first_facade)]
        };
        facades[..count]
            .iter()
            .copied()
            .map(|facade| {
                let kind = sample_extension_kind(&mut rng, config);
                let roof = sample_extension_roof(&mut rng, config);
                let facade_width = match facade {
                    FacadeSide::Front | FacadeSide::Back => building.footprint_width_m,
                    FacadeSide::Right | FacadeSide::Left => building.footprint_depth_m,
                };
                let kind_width_scale = match kind {
                    BuildingExtensionKind::DiningWing => 1.0,
                    BuildingExtensionKind::EntranceVestibule => 0.56,
                    BuildingExtensionKind::ServiceAnnex => 0.78,
                };
                let kind_depth_scale = match kind {
                    BuildingExtensionKind::DiningWing => 1.0,
                    BuildingExtensionKind::EntranceVestibule => 0.58,
                    BuildingExtensionKind::ServiceAnnex => 0.82,
                };
                let width = (facade_width
                    * sample_range(&mut rng, config.facade_width_fraction)
                    * kind_width_scale)
                    .clamp(2.4, facade_width * 0.72);
                let projection = (sample_range(&mut rng, config.projection_m) * kind_depth_scale)
                    .clamp(1.2, 6.5);
                let remaining_span = (facade_width - width - 0.8).max(0.0);
                let offset =
                    sample_range(&mut rng, config.facade_offset_fraction) * remaining_span * 0.5;
                let height = (building.wall_height_m
                    * sample_range(&mut rng, config.wall_height_fraction))
                .clamp(2.15, building.wall_height_m - 0.45);
                let maximum_rise = (building.wall_height_m - height - 0.18).max(0.08);
                let roof_rise_m = if roof == BuildingExtensionRoof::Shed {
                    sample_range(&mut rng, config.shed_rise_m).min(maximum_rise)
                } else {
                    0.0
                };
                let overlap = 0.12;
                let (position, size_m) = match facade {
                    FacadeSide::Front => (
                        Vec3::new(
                            offset,
                            0.0,
                            -building.footprint_depth_m * 0.5 - projection * 0.5 + overlap,
                        ),
                        Vec3::new(width, height, projection),
                    ),
                    FacadeSide::Back => (
                        Vec3::new(
                            offset,
                            0.0,
                            building.footprint_depth_m * 0.5 + projection * 0.5 - overlap,
                        ),
                        Vec3::new(width, height, projection),
                    ),
                    FacadeSide::Right => (
                        Vec3::new(
                            building.footprint_width_m * 0.5 + projection * 0.5 - overlap,
                            0.0,
                            offset,
                        ),
                        Vec3::new(projection, height, width),
                    ),
                    FacadeSide::Left => (
                        Vec3::new(
                            -building.footprint_width_m * 0.5 - projection * 0.5 + overlap,
                            0.0,
                            offset,
                        ),
                        Vec3::new(projection, height, width),
                    ),
                };
                SampledBuildingExtension {
                    kind,
                    facade,
                    position,
                    size_m,
                    roof,
                    roof_rise_m,
                }
            })
            .collect()
    }

    fn sample_signage(
        &self,
        seed: u64,
        building: SampledBuilding,
        phase: DayPhase,
    ) -> Vec<SampledSignage> {
        let mut rng = rng(seed, "signage");
        let config = self.config.composition.signage;
        let total = u64::from(config.none_weight)
            + u64::from(config.pizza_hut_weight)
            + u64::from(config.removed_ghost_weight)
            + u64::from(config.rebranded_tenant_weight);
        let draw = rng.random_range(0..total);
        let pizza_hut_threshold =
            u64::from(config.none_weight) + u64::from(config.pizza_hut_weight);
        let removed_ghost_threshold = pizza_hut_threshold + u64::from(config.removed_ghost_weight);
        let kind = if draw < u64::from(config.none_weight) {
            return Vec::new();
        } else if draw < pizza_hut_threshold {
            SignageKind::PizzaHut
        } else if draw < removed_ghost_threshold {
            SignageKind::RemovedGhost
        } else {
            SignageKind::RebrandedTenant
        };
        let facade = sample_facade_side(&mut rng);
        let facade_width = match facade {
            FacadeSide::Front | FacadeSide::Back => building.footprint_width_m,
            FacadeSide::Right | FacadeSide::Left => building.footprint_depth_m,
        };
        let width =
            (facade_width * sample_range(&mut rng, config.width_fraction)).min(facade_width - 0.8);
        let height = sample_range(&mut rng, config.height_m).min(building.wall_height_m * 0.36);
        let vertical_fraction = sample_range(&mut rng, config.vertical_fraction);
        let center_y = (building.wall_height_m * vertical_fraction).clamp(
            height * 0.5 + 0.15,
            building.wall_height_m - height * 0.5 - 0.15,
        );
        let available = (facade_width - width) * 0.42;
        let horizontal = sample_signed(&mut rng) * available * rng.random::<f32>();
        let outside = 0.035;
        let center = match facade {
            FacadeSide::Front => Vec3::new(
                horizontal,
                center_y,
                -building.footprint_depth_m * 0.5 - outside,
            ),
            FacadeSide::Right => Vec3::new(
                building.footprint_width_m * 0.5 + outside,
                center_y,
                horizontal,
            ),
            FacadeSide::Back => Vec3::new(
                -horizontal,
                center_y,
                building.footprint_depth_m * 0.5 + outside,
            ),
            FacadeSide::Left => Vec3::new(
                -building.footprint_width_m * 0.5 - outside,
                center_y,
                -horizontal,
            ),
        };
        let phase_light = match phase {
            DayPhase::Day => 0.0,
            DayPhase::Twilight => 0.48,
            DayPhase::Night => 1.0,
        };
        let emissive_strength = if kind == SignageKind::RemovedGhost {
            0.0
        } else {
            sample_range(&mut rng, config.emissive_strength) * phase_light
        };
        let weathering = match kind {
            SignageKind::RemovedGhost => sample_range(&mut rng, FloatRange::new(0.52, 1.0)),
            SignageKind::PizzaHut => sample_range(&mut rng, FloatRange::new(0.0, 0.66)),
            SignageKind::RebrandedTenant => sample_range(&mut rng, FloatRange::new(0.0, 0.52)),
        };
        vec![SampledSignage {
            kind,
            facade,
            center,
            size_m: Vec2::new(width, height),
            emissive_strength,
            weathering,
        }]
    }

    fn sample_background_buildings(
        &self,
        seed: u64,
        building: SampledBuilding,
        domain: SceneDomainProfile,
    ) -> Vec<SampledBackgroundBuilding> {
        let mut rng = rng(seed, "background-buildings");
        let config = self.config.composition.background_buildings;
        let count_range = intersect_u32_ranges(config.count, domain.background_buildings);
        let count = sample_u32(&mut rng, count_range.min, count_range.max);
        if count == 0 {
            return Vec::new();
        }
        let target_radius = libm::sqrtf(
            (building.footprint_width_m * 0.5).powi(2) + (building.footprint_depth_m * 0.5).powi(2),
        );
        let base_angle = rng.random::<f32>() * TAU;
        let step = TAU / count as f32;
        (0..count)
            .map(|index| {
                let kind = sample_background_kind(&mut rng, domain.domain);
                let width = sample_range(&mut rng, config.width_m);
                let depth = sample_range(&mut rng, config.depth_m);
                let raw_height = sample_range(&mut rng, config.height_m);
                let height = match kind {
                    BackgroundBuildingKind::MidriseMixedUse => raw_height.clamp(9.0, 28.0),
                    BackgroundBuildingKind::LowCommercial => raw_height.clamp(3.2, 9.5),
                    BackgroundBuildingKind::Residential => raw_height.clamp(3.2, 8.5),
                    BackgroundBuildingKind::IndustrialShed => raw_height.clamp(4.0, 11.0),
                };
                let angular_jitter = (rng.random::<f32>() - 0.5) * step * 0.42;
                let angle = base_angle + index as f32 * step + angular_jitter;
                let own_radius = libm::sqrtf((width * 0.5).powi(2) + (depth * 0.5).powi(2));
                let distance =
                    target_radius + own_radius + sample_range(&mut rng, config.setback_m);
                SampledBackgroundBuilding {
                    kind,
                    position: Vec3::new(
                        libm::cosf(angle) * distance,
                        0.0,
                        libm::sinf(angle) * distance,
                    ),
                    size_m: Vec3::new(width, height, depth),
                    yaw_degrees: angle.to_degrees()
                        + 90.0
                        + sample_range(&mut rng, config.yaw_jitter_degrees),
                    material: resolved_material(&mut rng, &self.config.materials.walls),
                }
            })
            .collect()
    }

    fn sample_vegetation(
        &self,
        seed: u64,
        building: SampledBuilding,
        domain: SceneDomainProfile,
    ) -> Vec<SampledVegetation> {
        let mut rng = rng(seed, "vegetation");
        let config = self.config.composition.vegetation;
        let count_range = intersect_u32_ranges(config.count, domain.vegetation);
        let count = sample_u32(&mut rng, count_range.min, count_range.max);
        if count == 0 {
            return Vec::new();
        }
        let building_radius = libm::sqrtf(
            (building.footprint_width_m * 0.5).powi(2) + (building.footprint_depth_m * 0.5).powi(2),
        );
        let base_angle = rng.random::<f32>() * TAU;
        let step = TAU / count as f32;
        (0..count)
            .map(|index| {
                let kind = sample_vegetation_kind(&mut rng, config);
                let height = if kind == VegetationKind::Shrub {
                    sample_range(&mut rng, config.shrub_height_m)
                } else {
                    sample_range(&mut rng, config.tree_height_m)
                };
                let angle =
                    base_angle + index as f32 * step + (rng.random::<f32>() - 0.5) * step * 0.72;
                let distance = building_radius
                    + sample_range(&mut rng, config.building_setback_m)
                    + sample_range(&mut rng, config.site_distance_m);
                SampledVegetation {
                    kind,
                    position: Vec3::new(
                        libm::cosf(angle) * distance,
                        0.0,
                        libm::sinf(angle) * distance,
                    ),
                    height_m: height,
                    canopy_radius_m: height * sample_range(&mut rng, config.canopy_radius_fraction),
                    yaw_degrees: rng.random::<f32>() * 360.0,
                    seasonal_variation: sample_signed(&mut rng) * rng.random::<f32>(),
                }
            })
            .collect()
    }

    fn sample_infrastructure(
        &self,
        seed: u64,
        building: SampledBuilding,
        domain: SceneDomainProfile,
    ) -> SampledSiteInfrastructure {
        let mut rng = rng(seed, "site-infrastructure");
        let parking_bays = sample_u32(&mut rng, domain.parking_bays.min, domain.parking_bays.max);
        let road_lanes = sample_u32(&mut rng, domain.road_lanes.min, domain.road_lanes.max);
        let lane_width_m = sample_range(&mut rng, FloatRange::new(3.0, 3.7));
        let parking_center_z = -(building.footprint_depth_m * 0.5 + 9.0);
        let road_center_z_m = parking_center_z - 11.0 - road_lanes as f32 * lane_width_m * 0.5;
        let pole_count = sample_u32(&mut rng, domain.utility_poles.min, domain.utility_poles.max);
        let pole_span = (building.ground_half_extent_m * 1.45).min(72.0);
        let utility_poles = (0..pole_count)
            .map(|index| {
                let progress = if pole_count <= 1 {
                    0.5
                } else {
                    index as f32 / (pole_count - 1) as f32
                };
                SampledUtilityPole {
                    position: Vec3::new(
                        (progress - 0.5) * pole_span,
                        0.0,
                        road_center_z_m - road_lanes as f32 * lane_width_m * 0.5 - 1.2,
                    ),
                    height_m: sample_range(&mut rng, FloatRange::new(5.2, 10.8)),
                    has_light: matches!(domain.domain, SceneDomain::City | SceneDomain::Urban)
                        || rng.random::<f32>() < 0.42,
                }
            })
            .collect::<Vec<_>>();
        let mut utility_lines = Vec::new();
        for index in 0..utility_poles.len().saturating_sub(1) {
            if rng.random::<f32>() < domain.overhead_line_probability {
                utility_lines.push(SampledUtilityLine {
                    start_pole: index as u16,
                    end_pole: (index + 1) as u16,
                    sag_m: sample_range(&mut rng, FloatRange::new(0.22, 1.15)),
                });
            }
        }
        SampledSiteInfrastructure {
            domain: domain.domain,
            parking_bays,
            parking_center: Vec3::new(0.0, 0.0, parking_center_z),
            parking_bay_size_m: Vec2::new(
                sample_range(&mut rng, FloatRange::new(2.35, 2.75)),
                sample_range(&mut rng, FloatRange::new(4.7, 5.6)),
            ),
            parking_yaw_degrees: if rng.random::<bool>() { 0.0 } else { 90.0 },
            road_kind: domain.road_kind,
            road_lanes,
            lane_width_m,
            road_center_z_m,
            raised_curb: rng.random::<f32>() < domain.curb_probability,
            utility_poles,
            utility_lines,
        }
    }

    fn sample_occluders(
        &self,
        seed: u64,
        building: SampledBuilding,
        frames: &[CameraFramePlan],
    ) -> Vec<SampledOccluder> {
        let config = &self.config.occluders;
        let mut kind_rng = rng(seed, "occluder-kinds");
        let count = sample_u32(&mut kind_rng, config.count.min, config.count.max);
        let kinds = (0..count)
            .map(|_| choose_non_rooftop_occluder(&mut kind_rng, &config.choices))
            .collect::<Vec<_>>();
        let mut rng = rng(seed, "occluder-placement");
        let middle_camera = frames
            .get(frames.len() / 2)
            .map(|frame| frame.camera.world_from_camera.translation)
            .unwrap_or(Vec3::new(0.0, 1.6, 30.0));
        let target = Vec3::new(0.0, building.wall_height_m + 1.5, 0.0);
        kinds
            .into_iter()
            .filter_map(|choice| {
                let scale_value = sample_range(&mut rng, config.scale);
                let yaw_degrees = rng.random::<f32>() * 360.0;
                let nominal_size_m = occluder_size(choice.kind, scale_value);
                let (half_x, half_z) = rotated_half_extents(nominal_size_m, yaw_degrees);
                let obstacle_size = CameraObstacleSize {
                    half_x,
                    half_z,
                    height_m: nominal_size_m.y,
                };
                let placement_and_position = if rng.random::<f32>()
                    < config.foreground_probability * foreground_probability_scale(choice.kind)
                {
                    sample_foreground_occluder_position(
                        &mut rng,
                        config,
                        building,
                        middle_camera,
                        target,
                        obstacle_size,
                        frames,
                    )
                    .map(|position| (OccluderPlacement::Foreground, position))
                    .or_else(|| {
                        sample_site_occluder_position(
                            &mut rng,
                            config,
                            building,
                            obstacle_size,
                            frames,
                        )
                        .map(|position| (OccluderPlacement::Site, position))
                    })
                } else {
                    sample_site_occluder_position(&mut rng, config, building, obstacle_size, frames)
                        .map(|position| (OccluderPlacement::Site, position))
                };
                let (placement, position) = placement_and_position?;
                Some(SampledOccluder {
                    kind: choice.kind,
                    position,
                    yaw_degrees,
                    scale: scale_value,
                    placement,
                    nominal_size_m,
                })
            })
            .collect()
    }

    fn sample_cameras(
        &self,
        seed: u64,
        building: SampledBuilding,
        roof: SampledRoof,
        ordinary_roof: Option<SampledOrdinaryRoof>,
        obstacles: CameraSceneObstacles<'_>,
    ) -> (CameraMotionPlan, Vec<CameraFramePlan>) {
        let mut rng = rng(seed, "camera");
        let camera = self.config.camera;
        let image = self.config.image;
        let sequence = self.config.sequence;
        let path_kind = choose_camera_path(&mut rng, camera);
        let partial_crop = rng.random::<f32>() < camera.partial_crop_probability;
        let framing_intent = if partial_crop {
            match rng.random_range(0..4_u32) {
                0 => FramingIntent::PartialLeft,
                1 => FramingIntent::PartialRight,
                2 => FramingIntent::PartialTop,
                _ => FramingIntent::PartialBottom,
            }
        } else {
            FramingIntent::Centered
        };
        let apparent_scale = if partial_crop {
            ApparentScale::Partial
        } else {
            let normal_threshold = camera
                .distant_view_weight
                .saturating_add(camera.normal_view_weight);
            let total = normal_threshold.saturating_add(camera.close_view_weight);
            let draw = rng.random_range(0..total);
            if draw < camera.distant_view_weight {
                ApparentScale::Distant
            } else if draw < normal_threshold {
                ApparentScale::Normal
            } else {
                ApparentScale::Close
            }
        };
        let framing_range = match apparent_scale {
            ApparentScale::Distant => camera.distant_target_width_fraction,
            ApparentScale::Normal => camera.target_width_fraction,
            ApparentScale::Close => camera.close_target_width_fraction,
            ApparentScale::Partial => camera.partial_target_width_fraction,
        };
        let target_width_fraction_goal = sample_range(&mut rng, framing_range);
        let height = sample_range(&mut rng, camera.height_m);
        let start = rng.random::<f32>() * TAU;
        let sweep_sign = if rng.random::<bool>() { 1.0 } else { -1.0 };
        let sweep = sample_range(&mut rng, camera.sweep_degrees).to_radians() * sweep_sign;
        let radial_motion = sample_range(&mut rng, camera.radial_motion_fraction);
        let start_fov_degrees = sample_range(&mut rng, camera.horizontal_fov_degrees);
        let start_fov = start_fov_degrees.to_radians();
        let roof_envelope =
            ordinary_roof.map_or_else(|| RoofEnvelope::target(roof), RoofEnvelope::ordinary);
        let desired_distance = roof_envelope.eave_width_m
            / (2.0 * target_width_fraction_goal * libm::tanf(start_fov * 0.5));
        let distance = desired_distance.clamp(camera.distance_m.min, camera.distance_m.max);
        let target_center = Vec3::new(
            0.0,
            building.wall_height_m + sample_range(&mut rng, camera.target_above_eave_m),
            0.0,
        );
        let crop_depth_fraction = sample_range(&mut rng, camera.framing_offset_fraction);
        let zoomed = rng.random::<f32>() < camera.zoom_probability;
        let fov_ratio = if zoomed {
            sample_range(&mut rng, camera.zoom_ratio)
        } else {
            1.0
        };
        let end_fov_degrees = (start_fov_degrees * fov_ratio).clamp(
            camera.horizontal_fov_degrees.min,
            camera.horizontal_fov_degrees.max,
        );
        let zoom_behavior = if (end_fov_degrees - start_fov_degrees).abs() < 0.1 {
            ZoomBehavior::Fixed
        } else if end_fov_degrees < start_fov_degrees {
            ZoomBehavior::SmoothIn
        } else {
            ZoomBehavior::SmoothOut
        };
        let handheld_sway = sample_range(&mut rng, camera.handheld_sway_m);
        let sway_phase = rng.random::<f32>() * TAU;
        let lateral_span = (distance * libm::tanf(sweep.abs() * 0.5) * 0.55)
            .clamp(distance * 0.12, distance * 0.65);
        let approach_amount = radial_motion.abs().clamp(0.16, 0.42);
        let path = collision_free_camera_path(
            CameraPathParameters {
                kind: path_kind,
                start,
                sweep,
                distance,
                height,
                radial_motion,
                lateral_span,
                approach_amount,
                handheld_sway,
                sway_phase,
            },
            building,
            roof_envelope,
            obstacles,
            sequence.frame_count,
            camera.distance_m.min,
        );
        let roof_points = ordinary_roof.map_or_else(
            || target_roof_control_points(building, roof),
            |ordinary| ordinary_roof_control_points(building, ordinary),
        );
        let midpoint_index = sequence.frame_count / 2;
        let mut framing_offset = Vec3::new(0.0, 0.0, 0.0);
        let mut frames = Vec::with_capacity(sequence.frame_count as usize);
        for frame_index in 0..sequence.frame_count {
            let progress = sequence_progress(frame_index, sequence.frame_count);
            let eased = smoothstep(progress);
            let position = camera_path_position(path, progress);
            let horizontal_fov_degrees =
                start_fov_degrees + (end_fov_degrees - start_fov_degrees) * eased;
            let horizontal_fov = horizontal_fov_degrees.to_radians();
            let fx = image.width as f32 / (2.0 * libm::tanf(horizontal_fov * 0.5));
            let intrinsics = CameraIntrinsics {
                width: image.width,
                height: image.height,
                fx,
                fy: fx,
                cx: image.width as f32 * 0.5,
                cy: image.height as f32 * 0.5,
                skew: 0.0,
            };
            let aim = solve_framing_aim(
                position,
                target_center,
                framing_intent,
                crop_depth_fraction,
                &roof_points,
                intrinsics,
            );
            if frame_index == midpoint_index {
                framing_offset = sub(aim, target_center);
            }
            frames.push(CameraFramePlan {
                frame_index,
                timestamp_ns: u64::from(frame_index)
                    * u64::from(sequence.frame_interval_ms)
                    * 1_000_000,
                camera: CameraModel {
                    intrinsics,
                    distortion: DistortionModel::None,
                    world_from_camera: look_at(position, aim),
                    output_from_sensor: ImageTransform::IDENTITY,
                },
            });
        }
        (
            CameraMotionPlan {
                path_kind,
                zoom_behavior,
                framing_intent,
                apparent_scale,
                target_width_fraction_goal,
                start_horizontal_fov_degrees: start_fov_degrees,
                end_horizontal_fov_degrees: end_fov_degrees,
                target_center,
                framing_offset,
                handheld_sway_m: handheld_sway,
            },
            frames,
        )
    }
}

/// Sampling cannot proceed with an invalid or unserializable configuration.
#[derive(Debug, Error)]
pub enum SamplingError {
    /// Configuration validation found one or more errors.
    #[error("invalid generator configuration: {0}")]
    InvalidConfig(ValidationReport),
    /// Stable grouping requires a non-empty family.
    #[error("building family must not be empty")]
    EmptyBuildingFamily,
    /// Configuration could not be serialized for fingerprinting.
    #[error("failed to fingerprint generator configuration: {0}")]
    Serialization(#[from] serde_json::Error),
}

fn foreground_probability_scale(kind: OccluderKind) -> f32 {
    match kind {
        OccluderKind::Pedestrian => 1.0,
        OccluderKind::Vehicle => 0.8,
        OccluderKind::Vegetation => 0.55,
        OccluderKind::Sign => 0.45,
        OccluderKind::Pole => 0.35,
        OccluderKind::Building | OccluderKind::RooftopEquipment => 0.0,
    }
}

fn rng(seed: u64, domain: &str) -> ChaCha20Rng {
    ChaCha20Rng::seed_from_u64(derive_seed(seed, domain))
}

fn sample_range(rng: &mut ChaCha20Rng, range: FloatRange) -> f32 {
    range.min + (range.max - range.min) * rng.random::<f32>()
}

fn intersect_supported(global: FloatRange, profile: FloatRange) -> FloatRange {
    global
        .intersection(profile)
        .expect("validated roof profile must overlap the supported global range")
}

fn sample_u32(rng: &mut ChaCha20Rng, min: u32, max: u32) -> u32 {
    if min == max {
        min
    } else {
        rng.random_range(min..=max)
    }
}

fn choose_material<'a>(rng: &mut ChaCha20Rng, choices: &'a [MaterialChoice]) -> &'a MaterialChoice {
    let total = choices
        .iter()
        .map(|entry| u64::from(entry.weight))
        .sum::<u64>();
    let mut draw = rng.random_range(0..total);
    for entry in choices {
        let weight = u64::from(entry.weight);
        if draw < weight {
            return entry;
        }
        draw -= weight;
    }
    &choices[choices.len() - 1]
}

fn choose_non_rooftop_occluder<'a>(
    rng: &mut ChaCha20Rng,
    choices: &'a [OccluderChoice],
) -> &'a OccluderChoice {
    let total = choices
        .iter()
        .filter(|entry| entry.kind != OccluderKind::RooftopEquipment)
        .map(|entry| u64::from(entry.weight))
        .sum::<u64>();
    let mut draw = rng.random_range(0..total);
    choices
        .iter()
        .filter(|entry| entry.kind != OccluderKind::RooftopEquipment)
        .find(|entry| {
            let weight = u64::from(entry.weight);
            if draw < weight {
                true
            } else {
                draw -= weight;
                false
            }
        })
        .expect("validated occluder distribution has a non-rooftop choice")
}

fn choose_roof_profile<'a>(
    rng: &mut ChaCha20Rng,
    choices: &'a [RoofMorphologyProfile],
) -> &'a RoofMorphologyProfile {
    let total = choices.iter().map(|entry| u64::from(entry.weight)).sum();
    let mut draw = rng.random_range(0..total);
    for entry in choices {
        if draw < u64::from(entry.weight) {
            return entry;
        }
        draw -= u64::from(entry.weight);
    }
    &choices[choices.len() - 1]
}

fn resolved_material(rng: &mut ChaCha20Rng, choices: &[MaterialChoice]) -> SampledMaterial {
    let choice = choose_material(rng, choices);
    resolve_material_choice(rng, choice)
}

fn resolve_material_choice(rng: &mut ChaCha20Rng, choice: &MaterialChoice) -> SampledMaterial {
    SampledMaterial {
        id: choice.id.clone(),
        base_color_srgb: varied_base_color(rng, choice),
        roughness: sample_range(rng, choice.roughness),
        weathering: sample_range(rng, choice.weathering),
    }
}

fn varied_base_color(rng: &mut ChaCha20Rng, choice: &MaterialChoice) -> [f32; 3] {
    if choice.base_color_variation == 0.0 {
        return choice.base_color_srgb;
    }
    choice.base_color_srgb.map(|channel| {
        let fractional_offset = sample_range(
            rng,
            FloatRange::new(-choice.base_color_variation, choice.base_color_variation),
        );
        (channel * (1.0 + fractional_offset)).clamp(0.0, 1.0)
    })
}

fn choose_weather<'a>(rng: &mut ChaCha20Rng, choices: &'a [WeatherProfile]) -> &'a WeatherProfile {
    let total = choices.iter().map(|entry| u64::from(entry.weight)).sum();
    let mut draw = rng.random_range(0..total);
    for entry in choices {
        if draw < u64::from(entry.weight) {
            return entry;
        }
        draw -= u64::from(entry.weight);
    }
    &choices[choices.len() - 1]
}

fn choose_day_phase<'a>(
    rng: &mut ChaCha20Rng,
    choices: &'a [DayPhaseProfile],
) -> &'a DayPhaseProfile {
    let total = choices.iter().map(|entry| u64::from(entry.weight)).sum();
    let mut draw = rng.random_range(0..total);
    for entry in choices {
        if draw < u64::from(entry.weight) {
            return entry;
        }
        draw -= u64::from(entry.weight);
    }
    &choices[choices.len() - 1]
}

fn choose_domain<'a>(
    rng: &mut ChaCha20Rng,
    choices: &'a [SceneDomainProfile],
) -> &'a SceneDomainProfile {
    let total = choices.iter().map(|entry| u64::from(entry.weight)).sum();
    let mut draw = rng.random_range(0..total);
    for entry in choices {
        if draw < u64::from(entry.weight) {
            return entry;
        }
        draw -= u64::from(entry.weight);
    }
    &choices[choices.len() - 1]
}

fn opposite_facade(facade: FacadeSide) -> FacadeSide {
    match facade {
        FacadeSide::Front => FacadeSide::Back,
        FacadeSide::Right => FacadeSide::Left,
        FacadeSide::Back => FacadeSide::Front,
        FacadeSide::Left => FacadeSide::Right,
    }
}

fn sample_extension_kind(
    rng: &mut ChaCha20Rng,
    config: crate::BuildingExtensionSamplingConfig,
) -> BuildingExtensionKind {
    let total = u64::from(config.dining_wing_weight)
        + u64::from(config.entrance_vestibule_weight)
        + u64::from(config.service_annex_weight);
    let draw = rng.random_range(0..total);
    if draw < u64::from(config.dining_wing_weight) {
        BuildingExtensionKind::DiningWing
    } else if draw
        < u64::from(config.dining_wing_weight) + u64::from(config.entrance_vestibule_weight)
    {
        BuildingExtensionKind::EntranceVestibule
    } else {
        BuildingExtensionKind::ServiceAnnex
    }
}

fn sample_extension_roof(
    rng: &mut ChaCha20Rng,
    config: crate::BuildingExtensionSamplingConfig,
) -> BuildingExtensionRoof {
    let total = u64::from(config.flat_roof_weight) + u64::from(config.shed_roof_weight);
    if rng.random_range(0..total) < u64::from(config.flat_roof_weight) {
        BuildingExtensionRoof::Flat
    } else {
        BuildingExtensionRoof::Shed
    }
}

fn choose_camera_path(
    rng: &mut ChaCha20Rng,
    config: crate::CameraSamplingConfig,
) -> CameraPathKind {
    let weights = [
        config.orbit_weight,
        config.lateral_walk_weight,
        config.approach_arc_weight,
        config.corner_reveal_weight,
    ];
    let total = weights.iter().map(|weight| u64::from(*weight)).sum::<u64>();
    let draw = rng.random_range(0..total);
    let mut cumulative = 0_u64;
    for (index, weight) in weights.into_iter().enumerate() {
        cumulative += u64::from(weight);
        if draw < cumulative {
            return match index {
                0 => CameraPathKind::Orbit,
                1 => CameraPathKind::LateralWalk,
                2 => CameraPathKind::ApproachArc,
                _ => CameraPathKind::CornerReveal,
            };
        }
    }
    CameraPathKind::CornerReveal
}

fn sample_facade_side(rng: &mut ChaCha20Rng) -> FacadeSide {
    match rng.random_range(0..4_u32) {
        0 => FacadeSide::Front,
        1 => FacadeSide::Right,
        2 => FacadeSide::Back,
        _ => FacadeSide::Left,
    }
}

fn sample_background_kind(rng: &mut ChaCha20Rng, domain: SceneDomain) -> BackgroundBuildingKind {
    let draw = rng.random_range(0..100_u32);
    match domain {
        SceneDomain::City if draw < 54 => BackgroundBuildingKind::MidriseMixedUse,
        SceneDomain::City if draw < 86 => BackgroundBuildingKind::LowCommercial,
        SceneDomain::City => BackgroundBuildingKind::Residential,
        SceneDomain::Urban if draw < 64 => BackgroundBuildingKind::LowCommercial,
        SceneDomain::Urban if draw < 82 => BackgroundBuildingKind::MidriseMixedUse,
        SceneDomain::Urban => BackgroundBuildingKind::IndustrialShed,
        SceneDomain::Suburban if draw < 62 => BackgroundBuildingKind::LowCommercial,
        SceneDomain::Suburban if draw < 84 => BackgroundBuildingKind::Residential,
        SceneDomain::Suburban => BackgroundBuildingKind::IndustrialShed,
        SceneDomain::Roadside if draw < 55 => BackgroundBuildingKind::LowCommercial,
        SceneDomain::Roadside => BackgroundBuildingKind::IndustrialShed,
        SceneDomain::Remote if draw < 44 => BackgroundBuildingKind::Residential,
        SceneDomain::Remote if draw < 72 => BackgroundBuildingKind::LowCommercial,
        SceneDomain::Remote => BackgroundBuildingKind::IndustrialShed,
    }
}

fn sample_vegetation_kind(
    rng: &mut ChaCha20Rng,
    config: crate::VegetationSamplingConfig,
) -> VegetationKind {
    let weights = [
        config.deciduous_weight,
        config.evergreen_weight,
        config.palm_weight,
        config.shrub_weight,
    ];
    let total = weights.iter().map(|weight| u64::from(*weight)).sum::<u64>();
    let draw = rng.random_range(0..total);
    let mut cumulative = 0_u64;
    for (index, weight) in weights.into_iter().enumerate() {
        cumulative += u64::from(weight);
        if draw < cumulative {
            return match index {
                0 => VegetationKind::DeciduousTree,
                1 => VegetationKind::EvergreenTree,
                2 => VegetationKind::Palm,
                _ => VegetationKind::Shrub,
            };
        }
    }
    VegetationKind::Shrub
}

fn occluder_size(kind: OccluderKind, scale_value: f32) -> Vec3 {
    let base = match kind {
        OccluderKind::Vegetation => Vec3::new(4.0, 7.0, 4.0),
        OccluderKind::Vehicle => Vec3::new(4.6, 1.7, 1.9),
        OccluderKind::Pole => Vec3::new(0.22, 7.5, 0.22),
        OccluderKind::Sign => Vec3::new(2.8, 3.6, 0.28),
        OccluderKind::Building => Vec3::new(9.0, 5.5, 8.0),
        OccluderKind::Pedestrian => Vec3::new(0.62, 1.72, 0.42),
        OccluderKind::RooftopEquipment => Vec3::new(1.8, 1.35, 1.5),
    };
    scale(base, scale_value)
}

fn intersect_ranges(left: FloatRange, right: FloatRange) -> FloatRange {
    FloatRange::new(left.min.max(right.min), left.max.min(right.max))
}

fn preferred_intersection(source: FloatRange, preferred: FloatRange) -> FloatRange {
    let intersection = intersect_ranges(source, preferred);
    if intersection.is_valid() {
        intersection
    } else {
        source
    }
}

fn intersect_u32_ranges(left: crate::U32Range, right: crate::U32Range) -> crate::U32Range {
    crate::U32Range::new(left.min.max(right.min), left.max.min(right.max))
}

fn sample_signed(rng: &mut ChaCha20Rng) -> f32 {
    if rng.random::<bool>() { 1.0 } else { -1.0 }
}

fn smoothstep(value: f32) -> f32 {
    value * value * (3.0 - 2.0 * value)
}

fn frame_key(sequence_id: &str, frame_index: u32) -> String {
    format!("{sequence_id}-{frame_index:06}")
}

fn look_at(position: Vec3, target: Vec3) -> RigidTransform {
    let forward = normalize(sub(target, position));
    let right = normalize(cross(forward, Vec3::new(0.0, 1.0, 0.0)));
    let up = cross(right, forward);
    let back = scale(forward, -1.0);
    let rotation = quaternion_from_columns(right, up, back);
    RigidTransform {
        translation: position,
        rotation_xyzw: rotation,
    }
}

fn sub(left: Vec3, right: Vec3) -> Vec3 {
    Vec3::new(left.x - right.x, left.y - right.y, left.z - right.z)
}

fn add(left: Vec3, right: Vec3) -> Vec3 {
    Vec3::new(left.x + right.x, left.y + right.y, left.z + right.z)
}

fn scale(value: Vec3, factor: f32) -> Vec3 {
    Vec3::new(value.x * factor, value.y * factor, value.z * factor)
}

fn dot(left: Vec3, right: Vec3) -> f32 {
    left.x * right.x + left.y * right.y + left.z * right.z
}

fn length(value: Vec3) -> f32 {
    libm::sqrtf(dot(value, value))
}

fn cross(left: Vec3, right: Vec3) -> Vec3 {
    Vec3::new(
        left.y * right.z - left.z * right.y,
        left.z * right.x - left.x * right.z,
        left.x * right.y - left.y * right.x,
    )
}

fn normalize(value: Vec3) -> Vec3 {
    let length = libm::sqrtf(value.x * value.x + value.y * value.y + value.z * value.z);
    scale(value, 1.0 / length)
}

fn quaternion_from_columns(x: Vec3, y: Vec3, z: Vec3) -> [f32; 4] {
    // Rotation matrix elements, indexed by row and column.
    let m00 = x.x;
    let m01 = y.x;
    let m02 = z.x;
    let m10 = x.y;
    let m11 = y.y;
    let m12 = z.y;
    let m20 = x.z;
    let m21 = y.z;
    let m22 = z.z;
    let trace = m00 + m11 + m22;
    let (qx, qy, qz, qw) = if trace > 0.0 {
        let s = 2.0 * libm::sqrtf(trace + 1.0);
        ((m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25 * s)
    } else if m00 > m11 && m00 > m22 {
        let s = 2.0 * libm::sqrtf(1.0 + m00 - m11 - m22);
        (0.25 * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s)
    } else if m11 > m22 {
        let s = 2.0 * libm::sqrtf(1.0 + m11 - m00 - m22);
        ((m01 + m10) / s, 0.25 * s, (m12 + m21) / s, (m02 - m20) / s)
    } else {
        let s = 2.0 * libm::sqrtf(1.0 + m22 - m00 - m11);
        ((m02 + m20) / s, (m12 + m21) / s, 0.25 * s, (m10 - m01) / s)
    };
    let length = libm::sqrtf(qx * qx + qy * qy + qz * qz + qw * qw);
    [qx / length, qy / length, qz / length, qw / length]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sampler(frame_count: u32) -> SequenceSampler {
        let mut config = GeneratorConfig::default();
        config.sequence.frame_count = frame_count;
        SequenceSampler::new(config).unwrap()
    }

    fn sampled_plan(sampler: &SequenceSampler, seed: u64, target_kind: TargetKind) -> SequencePlan {
        sampler
            .sample(SequenceRequest::procedural(
                "classic_two_stage",
                seed,
                target_kind,
            ))
            .unwrap()
    }

    fn assert_camera_path_clears_sampled_clutter(plan: &SequencePlan) {
        let mut previous = None;
        for frame in &plan.frames {
            let position = frame.camera.world_from_camera.translation;
            assert!(
                !camera_point_intersects_vegetation(position, &plan.scene.composition.vegetation,),
                "seed {} frame {} intersects composition vegetation",
                plan.request.building_seed,
                frame.frame_index,
            );
            assert!(
                !camera_point_intersects_occluders(position, &plan.scene.occluders),
                "seed {} frame {} intersects a sampled occluder",
                plan.request.building_seed,
                frame.frame_index,
            );
            if let Some(start) = previous {
                assert!(
                    !camera_segment_intersects_vegetation(
                        start,
                        position,
                        &plan.scene.composition.vegetation,
                    ),
                    "seed {} segment ending at frame {} intersects composition vegetation",
                    plan.request.building_seed,
                    frame.frame_index,
                );
                assert!(
                    !camera_segment_intersects_occluders(start, position, &plan.scene.occluders),
                    "seed {} segment ending at frame {} intersects a sampled occluder",
                    plan.request.building_seed,
                    frame.frame_index,
                );
            }
            previous = Some(position);
        }
    }

    #[test]
    fn look_at_produces_valid_transform() {
        let transform = look_at(Vec3::new(4.0, 2.0, 7.0), Vec3::new(0.0, 5.0, 0.0));
        assert!(transform.is_valid());
    }

    #[test]
    fn frame_key_is_lexically_ordered() {
        assert!(frame_key("seq-test", 9) < frame_key("seq-test", 10));
    }

    #[test]
    fn seed_2713_camera_clears_the_sampled_evergreen() {
        let plan = sampled_plan(&sampler(1), 2713, TargetKind::Target);

        assert!(
            plan.scene
                .composition
                .vegetation
                .iter()
                .any(|item| item.kind == VegetationKind::EvergreenTree),
            "the regression seed must retain its composition evergreens",
        );
        assert_camera_path_clears_sampled_clutter(&plan);
        assert!(plan.validate().is_valid());
    }

    #[test]
    fn bounded_seed_sweep_keeps_camera_paths_clear_of_sampled_clutter() {
        const SEED_COUNT: u64 = 512;
        let sampler = sampler(5);
        for seed in 0..SEED_COUNT {
            let target_kind = if seed % 2 == 0 {
                TargetKind::Target
            } else {
                TargetKind::Negative
            };
            let plan = sampled_plan(&sampler, seed, target_kind);
            assert_camera_path_clears_sampled_clutter(&plan);
            assert!(
                plan.validate().is_valid(),
                "seed {seed} generated an invalid plan"
            );
        }
    }
}
