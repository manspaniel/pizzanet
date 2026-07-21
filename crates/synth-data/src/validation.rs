use std::{collections::HashSet, fmt, path::Component, path::Path};

use serde::{Deserialize, Serialize};

use crate::{
    AssetRef, CameraMotionPlan, DATASET_SCHEMA_VERSION, DatasetManifest, DayPhase, EdgeVisibility,
    FloatRange, FrameRecord, GeneratorConfig, LabelClass, LocatorLabel, MaterialChoice,
    NormalizedBoundingBox, OccluderPlacement, SampledMaterial, SampledScene, SequencePlan,
    SequenceRecord, SplitKey, TargetKind, U32Range, Vec3, Visibility, ZoomBehavior,
    sampling::{
        GROUND_EDGE_MARGIN_M, OCCLUDER_TARGET_CLEARANCE_M, ROOFTOP_EDGE_CLEARANCE_M,
        actual_roof_envelope, camera_point_intersects_scene, camera_segment_intersects_scene,
        camera_view_intersects_background, rotated_half_extents, sampled_content_half_extent,
    },
};

/// Importance of one validation finding.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    /// Record or configuration must not be accepted.
    Error,
    /// Record is structurally usable but merits inspection.
    Warning,
}

/// One stable, machine-readable validation finding.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidationIssue {
    /// Error or warning.
    pub severity: Severity,
    /// Stable snake-case identifier suitable for metrics.
    pub code: String,
    /// JSON-like path to the failing value.
    pub path: String,
    /// Concise human-readable explanation.
    pub message: String,
}

/// Complete validation result. Warnings do not make a report invalid.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ValidationReport {
    /// Findings in deterministic traversal order.
    pub issues: Vec<ValidationIssue>,
}

impl ValidationReport {
    /// Returns `true` when no error-severity findings exist.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self
            .issues
            .iter()
            .any(|issue| issue.severity == Severity::Error)
    }

    /// Number of error findings.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|issue| issue.severity == Severity::Error)
            .count()
    }

    /// Number of warning findings.
    #[must_use]
    pub fn warning_count(&self) -> usize {
        self.issues
            .iter()
            .filter(|issue| issue.severity == Severity::Warning)
            .count()
    }

    fn error(&mut self, code: &str, path: impl Into<String>, message: impl Into<String>) {
        self.issues.push(ValidationIssue {
            severity: Severity::Error,
            code: code.to_owned(),
            path: path.into(),
            message: message.into(),
        });
    }

    fn warning(&mut self, code: &str, path: impl Into<String>, message: impl Into<String>) {
        self.issues.push(ValidationIssue {
            severity: Severity::Warning,
            code: code.to_owned(),
            path: path.into(),
            message: message.into(),
        });
    }

    fn append(&mut self, mut other: Self) {
        self.issues.append(&mut other.issues);
    }
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{} error(s), {} warning(s)",
            self.error_count(),
            self.warning_count()
        )?;
        for issue in &self.issues {
            write!(
                formatter,
                "; {:?} {} at {}: {}",
                issue.severity, issue.code, issue.path, issue.message
            )?;
        }
        Ok(())
    }
}

impl std::error::Error for ValidationReport {}

/// Type with context-free structural validation.
pub trait Validate {
    /// Checks every field and returns all findings instead of failing fast.
    fn validate(&self) -> ValidationReport;
}

impl Validate for GeneratorConfig {
    fn validate(&self) -> ValidationReport {
        let mut report = ValidationReport::default();
        if self.image.width == 0 || self.image.height == 0 {
            report.error(
                "empty_image",
                "image",
                "image width and height must both be positive",
            );
        }
        if self.sequence.frame_count == 0 {
            report.error(
                "empty_sequence",
                "sequence.frame_count",
                "at least one frame is required",
            );
        }
        if self.sequence.frame_interval_ms == 0 {
            report.error(
                "zero_frame_interval",
                "sequence.frame_interval_ms",
                "frame interval must be positive",
            );
        }

        let scene_ranges = [
            (
                "scene.footprint_width_m",
                self.scene.footprint_width_m,
                true,
            ),
            (
                "scene.footprint_depth_m",
                self.scene.footprint_depth_m,
                true,
            ),
            ("scene.wall_height_m", self.scene.wall_height_m, true),
            (
                "scene.ground_half_extent_m",
                self.scene.ground_half_extent_m,
                true,
            ),
        ];
        for (path, range, positive) in scene_ranges {
            validate_range(&mut report, path, range, positive);
        }

        for (path, range, positive) in [
            ("roof.overhang_m", self.roof.overhang_m, true),
            (
                "roof.shoulder_width_fraction",
                self.roof.shoulder_width_fraction,
                true,
            ),
            (
                "roof.shoulder_depth_fraction",
                self.roof.shoulder_depth_fraction,
                true,
            ),
            ("roof.lower_rise_m", self.roof.lower_rise_m, true),
            ("roof.upper_rise_m", self.roof.upper_rise_m, true),
            (
                "roof.crown_top_width_fraction",
                self.roof.crown_top_width_fraction,
                true,
            ),
            (
                "roof.crown_top_depth_fraction",
                self.roof.crown_top_depth_fraction,
                true,
            ),
            (
                "roof.asymmetry_fraction",
                self.roof.asymmetry_fraction,
                false,
            ),
        ] {
            validate_range(&mut report, path, range, positive);
        }
        for (path, range) in [
            (
                "roof.shoulder_width_fraction",
                self.roof.shoulder_width_fraction,
            ),
            (
                "roof.shoulder_depth_fraction",
                self.roof.shoulder_depth_fraction,
            ),
            (
                "roof.crown_top_width_fraction",
                self.roof.crown_top_width_fraction,
            ),
            (
                "roof.crown_top_depth_fraction",
                self.roof.crown_top_depth_fraction,
            ),
        ] {
            validate_unit_range(&mut report, path, range);
        }
        if self.roof.asymmetry_fraction.min < -0.25 || self.roof.asymmetry_fraction.max > 0.25 {
            report.error(
                "excessive_asymmetry",
                "roof.asymmetry_fraction",
                "asymmetry must remain within [-0.25, 0.25]",
            );
        }
        validate_roof_profiles(&mut report, self);

        for (path, range, positive) in [
            ("camera.distance_m", self.camera.distance_m, true),
            ("camera.height_m", self.camera.height_m, true),
            ("camera.sweep_degrees", self.camera.sweep_degrees, true),
            (
                "camera.radial_motion_fraction",
                self.camera.radial_motion_fraction,
                false,
            ),
            (
                "camera.horizontal_fov_degrees",
                self.camera.horizontal_fov_degrees,
                true,
            ),
            (
                "camera.target_above_eave_m",
                self.camera.target_above_eave_m,
                false,
            ),
            ("camera.zoom_ratio", self.camera.zoom_ratio, true),
            (
                "camera.target_width_fraction",
                self.camera.target_width_fraction,
                true,
            ),
            (
                "camera.distant_target_width_fraction",
                self.camera.distant_target_width_fraction,
                true,
            ),
            (
                "camera.close_target_width_fraction",
                self.camera.close_target_width_fraction,
                true,
            ),
            (
                "camera.partial_target_width_fraction",
                self.camera.partial_target_width_fraction,
                true,
            ),
            (
                "camera.framing_offset_fraction",
                self.camera.framing_offset_fraction,
                true,
            ),
            ("camera.handheld_sway_m", self.camera.handheld_sway_m, false),
        ] {
            validate_range(&mut report, path, range, positive);
        }
        if self.camera.horizontal_fov_degrees.min <= 1.0
            || self.camera.horizontal_fov_degrees.max >= 179.0
        {
            report.error(
                "invalid_camera_fov",
                "camera.horizontal_fov_degrees",
                "horizontal field of view must remain between 1 and 179 degrees",
            );
        }
        if self.camera.radial_motion_fraction.min <= -1.0 {
            report.error(
                "invalid_radial_motion",
                "camera.radial_motion_fraction",
                "radial motion cannot place the camera at or through the origin",
            );
        }
        if [
            self.camera.orbit_weight,
            self.camera.lateral_walk_weight,
            self.camera.approach_arc_weight,
            self.camera.corner_reveal_weight,
        ]
        .iter()
        .all(|weight| *weight == 0)
        {
            report.error(
                "empty_camera_path_distribution",
                "camera",
                "at least one camera-path weight must be positive",
            );
        }
        if [
            self.camera.distant_view_weight,
            self.camera.normal_view_weight,
            self.camera.close_view_weight,
        ]
        .iter()
        .all(|weight| *weight == 0)
        {
            report.error(
                "empty_camera_scale_distribution",
                "camera",
                "at least one apparent-scale weight must be positive",
            );
        }
        for (path, probability) in [
            ("camera.zoom_probability", self.camera.zoom_probability),
            (
                "camera.partial_crop_probability",
                self.camera.partial_crop_probability,
            ),
        ] {
            validate_probability(&mut report, path, probability);
        }
        if self.camera.zoom_ratio.min < 0.5 || self.camera.zoom_ratio.max > 2.0 {
            report.error(
                "implausible_zoom_ratio",
                "camera.zoom_ratio",
                "end/start FOV ratio must remain within [0.5, 2.0]",
            );
        }
        if self.camera.partial_target_width_fraction.max > 1.5
            || self.camera.distant_target_width_fraction.min < 0.1
            || self.camera.close_target_width_fraction.max > 1.0
            || self.camera.framing_offset_fraction.max > 0.6
            || self.camera.handheld_sway_m.min < 0.0
            || self.camera.handheld_sway_m.max > 0.5
        {
            report.error(
                "implausible_camera_composition",
                "camera",
                "crop, offset, and handheld-sway ranges exceed supported physical bounds",
            );
        }

        validate_materials(&mut report, "materials.roof", &self.materials.roof);
        validate_materials(&mut report, "materials.walls", &self.materials.walls);

        for (path, range, positive) in [
            (
                "lighting.sun_elevation_degrees",
                self.lighting.sun_elevation_degrees,
                false,
            ),
            ("lighting.sun_intensity", self.lighting.sun_intensity, false),
            ("lighting.sky_intensity", self.lighting.sky_intensity, false),
            (
                "lighting.cloud_coverage",
                self.lighting.cloud_coverage,
                false,
            ),
            ("lighting.haze", self.lighting.haze, false),
        ] {
            validate_range(&mut report, path, range, positive);
        }
        validate_unit_range(
            &mut report,
            "lighting.cloud_coverage",
            self.lighting.cloud_coverage,
        );
        validate_unit_range(&mut report, "lighting.haze", self.lighting.haze);
        if self.lighting.sun_intensity.min < 0.0 || self.lighting.sky_intensity.min < 0.0 {
            report.error(
                "negative_light",
                "lighting",
                "light intensity cannot be negative",
            );
        }

        validate_u32_range(&mut report, "occluders.count", self.occluders.count);
        validate_range(
            &mut report,
            "occluders.distance_m",
            self.occluders.distance_m,
            false,
        );
        validate_range(&mut report, "occluders.scale", self.occluders.scale, true);
        validate_range(
            &mut report,
            "occluders.foreground_depth_fraction",
            self.occluders.foreground_depth_fraction,
            true,
        );
        validate_unit_range(
            &mut report,
            "occluders.foreground_depth_fraction",
            self.occluders.foreground_depth_fraction,
        );
        validate_range(
            &mut report,
            "occluders.foreground_lateral_offset_m",
            self.occluders.foreground_lateral_offset_m,
            false,
        );
        validate_probability(
            &mut report,
            "occluders.foreground_probability",
            self.occluders.foreground_probability,
        );
        if self.occluders.choices.is_empty()
            || self
                .occluders
                .choices
                .iter()
                .filter(|choice| choice.kind != crate::OccluderKind::RooftopEquipment)
                .all(|choice| choice.weight == 0)
        {
            report.error(
                "empty_weighted_distribution",
                "occluders.choices",
                "at least one positive-weight non-rooftop occluder choice is required",
            );
        }
        validate_composition_config(&mut report, self);
        report
    }
}

impl Validate for SequencePlan {
    fn validate(&self) -> ValidationReport {
        let mut report = ValidationReport::default();
        if self.sequence_id.trim().is_empty()
            || self.request.building_family.trim().is_empty()
            || self.config_fingerprint.trim().is_empty()
        {
            report.error(
                "empty_plan_identity",
                "sequence_id",
                "sequence, building-family, and configuration identities are required",
            );
        }
        validate_sampled_scene(&mut report, &self.scene);
        match (self.request.target_kind, self.scene.ordinary_roof) {
            (TargetKind::Negative, None) => report.error(
                "missing_ordinary_negative_roof",
                "scene.ordinary_roof",
                "negative scenes must render an explicit ordinary roof family",
            ),
            (TargetKind::Target | TargetKind::NearMiss, Some(_)) => report.error(
                "target_has_ordinary_roof",
                "scene.ordinary_roof",
                "target and near-miss scenes cannot replace the target roof with an ordinary roof",
            ),
            _ => {}
        }
        validate_camera_motion_plan(&mut report, "camera_motion", self.camera_motion);
        if self.frames.is_empty() {
            report.error(
                "empty_camera_plan",
                "frames",
                "sequence plan requires at least one exact camera frame",
            );
        }
        let mut previous_timestamp = None;
        let mut previous_position = None;
        for (index, frame) in self.frames.iter().enumerate() {
            if frame.frame_index as usize != index {
                report.error(
                    "non_contiguous_frame_index",
                    format!("frames[{index}].frame_index"),
                    "frame indexes must be contiguous and zero based",
                );
            }
            if let Some(previous) = previous_timestamp
                && frame.timestamp_ns <= previous
            {
                report.error(
                    "non_monotonic_timestamp",
                    format!("frames[{index}].timestamp_ns"),
                    "camera-plan timestamps must increase strictly",
                );
            }
            previous_timestamp = Some(frame.timestamp_ns);
            let camera_position = frame.camera.world_from_camera.translation;
            validate_camera_position(
                &mut report,
                &format!("frames[{index}].camera.world_from_camera.translation"),
                camera_position,
                &self.scene,
                previous_position,
            );
            previous_position = Some(camera_position);
            let intrinsics = frame.camera.intrinsics;
            if intrinsics.width == 0
                || intrinsics.height == 0
                || !intrinsics.fx.is_finite()
                || !intrinsics.fy.is_finite()
                || intrinsics.fx <= 0.0
                || intrinsics.fy <= 0.0
                || !intrinsics.cx.is_finite()
                || !intrinsics.cy.is_finite()
                || !intrinsics.skew.is_finite()
                || !frame.camera.world_from_camera.is_valid()
            {
                report.error(
                    "invalid_planned_camera",
                    format!("frames[{index}].camera"),
                    "planned camera needs finite intrinsics and a valid rigid transform",
                );
            }
        }
        report
    }
}

fn validate_camera_motion_plan(
    report: &mut ValidationReport,
    path: &str,
    motion: CameraMotionPlan,
) {
    if !motion.target_center.is_finite()
        || !motion.framing_offset.is_finite()
        || !motion.target_width_fraction_goal.is_finite()
        || !(0.1..=1.5).contains(&motion.target_width_fraction_goal)
        || !motion.start_horizontal_fov_degrees.is_finite()
        || !motion.end_horizontal_fov_degrees.is_finite()
        || !(1.0..179.0).contains(&motion.start_horizontal_fov_degrees)
        || !(1.0..179.0).contains(&motion.end_horizontal_fov_degrees)
        || !motion.handheld_sway_m.is_finite()
        || !(0.0..=0.5).contains(&motion.handheld_sway_m)
    {
        report.error(
            "invalid_camera_motion_plan",
            path,
            "camera composition, FOV, target, and sway must be finite and physically bounded",
        );
    }
    let zoom_consistent = match motion.zoom_behavior {
        ZoomBehavior::Fixed => {
            (motion.start_horizontal_fov_degrees - motion.end_horizontal_fov_degrees).abs() < 0.1
        }
        ZoomBehavior::SmoothIn => {
            motion.end_horizontal_fov_degrees < motion.start_horizontal_fov_degrees
        }
        ZoomBehavior::SmoothOut => {
            motion.end_horizontal_fov_degrees > motion.start_horizontal_fov_degrees
        }
    };
    if !zoom_consistent {
        report.error(
            "inconsistent_zoom_behavior",
            format!("{path}.zoom_behavior"),
            "zoom category must agree with start and end horizontal FOV",
        );
    }
}

fn validate_camera_position(
    report: &mut ValidationReport,
    path: &str,
    position: Vec3,
    scene: &SampledScene,
    previous: Option<Vec3>,
) {
    let roof_envelope = actual_roof_envelope(scene);
    let ground_limit = scene.building.ground_half_extent_m - GROUND_EDGE_MARGIN_M;
    if position.x.abs() > ground_limit + 1.0e-3 || position.z.abs() > ground_limit + 1.0e-3 {
        report.error(
            "camera_outside_ground",
            path,
            "camera path must remain inside the finite ground plane with an edge margin",
        );
    }
    if camera_point_intersects_scene(
        position,
        scene.building,
        roof_envelope,
        &scene.composition.building_extensions,
        &scene.composition.background_buildings,
    ) {
        report.error(
            "camera_scene_intersection",
            path,
            "camera must clear the target wall volume and background-building bounds",
        );
    }
    if camera_view_intersects_background(
        position,
        scene.building,
        roof_envelope,
        &scene.composition.background_buildings,
    ) {
        report.error(
            "camera_view_background_intersection",
            path,
            "the target-roof sightline must clear background-building bounds",
        );
    }
    if let Some(start) = previous
        && camera_segment_intersects_scene(
            start,
            position,
            scene.building,
            roof_envelope,
            &scene.composition.building_extensions,
            &scene.composition.background_buildings,
        )
    {
        report.error(
            "camera_path_scene_intersection",
            path,
            "camera path segment must clear target and background-building bounds",
        );
    }
}

impl Validate for DatasetManifest {
    fn validate(&self) -> ValidationReport {
        let mut report = ValidationReport::default();
        if self.dataset_id.trim().is_empty() {
            report.error(
                "empty_id",
                "dataset_id",
                "dataset identifier must not be empty",
            );
        }
        if self.schema_version != DATASET_SCHEMA_VERSION {
            report.error(
                "unsupported_schema",
                "schema_version",
                format!("expected {DATASET_SCHEMA_VERSION}"),
            );
        }
        for (path, value) in [
            ("generator.name", self.generator.name.as_str()),
            ("generator.version", self.generator.version.as_str()),
            (
                "generator.rng_algorithm",
                self.generator.rng_algorithm.as_str(),
            ),
            (
                "generator.config_fingerprint",
                self.generator.config_fingerprint.as_str(),
            ),
        ] {
            if value.trim().is_empty() {
                report.error("empty_value", path, "value must not be empty");
            }
        }
        if let Err(error) = self
            .split_policy
            .assign(&SplitKey::procedural("manifest_validation", 0))
        {
            report.error("invalid_split_policy", "split_policy", error.to_string());
        }

        validate_classes(&mut report, "labels.keypoints", &self.labels.keypoints);
        validate_classes(&mut report, "labels.edges", &self.labels.edges);
        validate_classes(&mut report, "labels.parts", &self.labels.parts);
        validate_classes(&mut report, "labels.faces", &self.labels.faces);

        let mut asset_ids = HashSet::new();
        for (index, asset) in self.source_assets.iter().enumerate() {
            if asset.id.trim().is_empty() {
                report.error(
                    "empty_asset_id",
                    format!("source_assets[{index}].id"),
                    "source asset ID must not be empty",
                );
            } else if !asset_ids.insert(asset.id.as_str()) {
                report.error(
                    "duplicate_asset_id",
                    format!("source_assets[{index}].id"),
                    "source asset IDs must be unique",
                );
            }
            if asset.content_hash.trim().is_empty() || asset.license.trim().is_empty() {
                report.error(
                    "incomplete_asset_provenance",
                    format!("source_assets[{index}]"),
                    "content hash and licence are required",
                );
            }
        }

        let mut output_names = HashSet::new();
        for (index, output) in self.outputs.iter().enumerate() {
            if output.name.trim().is_empty()
                || output.media_type.trim().is_empty()
                || output.encoding.trim().is_empty()
            {
                report.error(
                    "incomplete_output",
                    format!("outputs[{index}]"),
                    "name, media type, and encoding are required",
                );
            } else if !output_names.insert(output.name.as_str()) {
                report.error(
                    "duplicate_output",
                    format!("outputs[{index}].name"),
                    "output names must be unique",
                );
            }
        }
        report
    }
}

/// Context-aware validator for final frame and sequence records.
pub struct DatasetValidator<'a> {
    manifest: &'a DatasetManifest,
}

impl<'a> DatasetValidator<'a> {
    /// Creates a validator using the manifest's schema, taxonomy, and split policy.
    #[must_use]
    pub const fn new(manifest: &'a DatasetManifest) -> Self {
        Self { manifest }
    }

    /// Validates a single frame independently of its parent sequence file.
    #[must_use]
    pub fn validate_frame(&self, frame: &FrameRecord) -> ValidationReport {
        let mut report = ValidationReport::default();
        if frame.schema_version != self.manifest.schema_version {
            report.error(
                "schema_mismatch",
                "schema_version",
                "frame schema differs from dataset manifest",
            );
        }
        if frame.sample_key.trim().is_empty() || frame.sequence_id.trim().is_empty() {
            report.error(
                "empty_record_id",
                "sample_key",
                "sample and sequence IDs must not be empty",
            );
        }
        validate_camera(&mut report, frame);
        validate_asset(&mut report, "assets.rgb", &frame.assets.rgb);
        if let Some(asset) = &frame.assets.surface_normals {
            validate_asset(&mut report, "assets.surface_normals", asset);
        }
        if let Some(asset) = &frame.assets.motion_vectors {
            validate_asset(&mut report, "assets.motion_vectors", asset);
        }
        if let Some(profile) = frame.appearance.photometric_profile
            && let Err(error) = profile.validate()
        {
            report.error(
                "invalid_photometric_profile",
                format!("appearance.photometric_profile.{}", error.field()),
                error.to_string(),
            );
        }
        validate_locator(&mut report, &frame.locator);

        if matches!(
            frame.locator.target_kind,
            TargetKind::Target | TargetKind::NearMiss
        ) {
            let requires_amodal = self
                .manifest
                .outputs
                .iter()
                .any(|output| output.name == "amodal_roof_mask" && output.required_for_targets);
            if frame.roof.is_none() {
                report.error(
                    "missing_roof_instance",
                    "roof",
                    "target or near-miss frame requires roof geometry",
                );
            }
            if frame.locator.visible_fraction > 0.0 && frame.locator.bounding_box.is_none() {
                report.error(
                    "missing_target_box",
                    "locator.bounding_box",
                    "a visible target or near-miss frame requires a bounding box",
                );
            }
            if requires_amodal && frame.locator.amodal_bounding_box.is_none() {
                report.error(
                    "missing_amodal_target_box",
                    "locator.amodal_bounding_box",
                    "current target records require bounds before scene occlusion",
                );
            }
            for (path, present) in [
                (
                    "labels.dense.roof_mask",
                    frame.labels.dense.roof_mask.is_some(),
                ),
                (
                    "labels.dense.amodal_roof_mask",
                    !requires_amodal || frame.labels.dense.amodal_roof_mask.is_some(),
                ),
                (
                    "labels.dense.part_mask",
                    frame.labels.dense.part_mask.is_some(),
                ),
                (
                    "labels.dense.face_id_map",
                    frame.labels.dense.face_id_map.is_some(),
                ),
                (
                    "labels.dense.face_coordinates",
                    frame.labels.dense.face_coordinates.is_some(),
                ),
            ] {
                if !present {
                    report.error(
                        "missing_dense_target",
                        path,
                        "target or near-miss frame requires this dense annotation",
                    );
                }
            }
            if frame.labels.keypoints.is_empty() || frame.labels.edges.is_empty() {
                report.error(
                    "missing_structural_labels",
                    "labels",
                    "target or near-miss frame requires keypoint and edge annotations",
                );
            }
        } else if frame.locator.target_kind == TargetKind::Negative {
            if frame.locator.bounding_box.is_some() || frame.locator.amodal_bounding_box.is_some() {
                report.error(
                    "negative_has_target_box",
                    "locator.bounding_box",
                    "ordinary-building negative frame cannot have a target bounding box",
                );
            }
            if frame.roof.is_some()
                || !frame.labels.keypoints.is_empty()
                || !frame.labels.edges.is_empty()
                || frame.labels.dense.roof_mask.is_some()
                || frame.labels.dense.amodal_roof_mask.is_some()
                || frame.labels.dense.part_mask.is_some()
                || frame.labels.dense.face_id_map.is_some()
                || frame.labels.dense.face_coordinates.is_some()
            {
                report.error(
                    "negative_has_target_geometry",
                    "labels",
                    "ordinary-building negatives must have no target roof instance or structural labels",
                );
            }
            if frame.locator.visible_fraction != 0.0
                || frame.locator.occluded_fraction != 0.0
                || frame.locator.truncated
            {
                report.error(
                    "negative_has_target_coverage",
                    "locator",
                    "ordinary-building negatives must have zero target coverage and no target truncation",
                );
            }
        }

        if let Some(roof) = &frame.roof {
            if roof.family.trim().is_empty() || !roof.world_from_roof.is_valid() {
                report.error(
                    "invalid_roof_instance",
                    "roof",
                    "roof family and world transform must be valid",
                );
            }
            if roof.parameters.values().any(|value| !value.is_finite()) {
                report.error(
                    "non_finite_roof_parameter",
                    "roof.parameters",
                    "all roof parameters must be finite",
                );
            }
        }

        for (path, asset) in [
            ("labels.dense.roof_mask", &frame.labels.dense.roof_mask),
            (
                "labels.dense.amodal_roof_mask",
                &frame.labels.dense.amodal_roof_mask,
            ),
            ("labels.dense.part_mask", &frame.labels.dense.part_mask),
            ("labels.dense.face_id_map", &frame.labels.dense.face_id_map),
            (
                "labels.dense.face_coordinates",
                &frame.labels.dense.face_coordinates,
            ),
        ] {
            if let Some(asset) = asset {
                validate_asset(&mut report, path, asset);
            }
        }
        if frame.labels.dense.face_coordinates.is_some() && frame.labels.dense.face_id_map.is_none()
        {
            report.error(
                "orphan_face_coordinates",
                "labels.dense.face_coordinates",
                "face-coordinate output requires a face-ID map",
            );
        }

        let keypoint_ids = self
            .manifest
            .labels
            .keypoints
            .iter()
            .map(|class| class.id)
            .collect::<HashSet<_>>();
        let mut instances = HashSet::new();
        for (index, keypoint) in frame.labels.keypoints.iter().enumerate() {
            let path = format!("labels.keypoints[{index}]");
            if !keypoint_ids.contains(&keypoint.class_id) {
                report.error(
                    "unknown_keypoint_class",
                    format!("{path}.class_id"),
                    "class is absent from manifest taxonomy",
                );
            }
            if !instances.insert(keypoint.instance_id) {
                report.error(
                    "duplicate_keypoint_instance",
                    format!("{path}.instance_id"),
                    "keypoint instance IDs must be unique within a frame",
                );
            }
            if !keypoint.roof_position.is_finite() {
                report.error(
                    "non_finite_keypoint",
                    format!("{path}.roof_position"),
                    "roof position must be finite",
                );
            }
            validate_keypoint_projection(&mut report, &path, keypoint);
        }

        let edge_ids = self
            .manifest
            .labels
            .edges
            .iter()
            .map(|class| class.id)
            .collect::<HashSet<_>>();
        let mut edge_instances = HashSet::new();
        for (index, edge) in frame.labels.edges.iter().enumerate() {
            let path = format!("labels.edges[{index}]");
            if !edge_ids.contains(&edge.class_id) {
                report.error(
                    "unknown_edge_class",
                    format!("{path}.class_id"),
                    "class is absent from manifest taxonomy",
                );
            }
            if !edge_instances.insert(edge.instance_id) {
                report.error(
                    "duplicate_edge_instance",
                    format!("{path}.instance_id"),
                    "edge instance IDs must be unique within a frame",
                );
            }
            let requires_polyline = matches!(
                edge.visibility,
                EdgeVisibility::Visible | EdgeVisibility::PartiallyOccluded
            );
            if (requires_polyline && edge.polyline.len() < 2) || edge.polyline.len() == 1 {
                report.error(
                    "short_edge_polyline",
                    format!("{path}.polyline"),
                    "projected edge requires at least two points",
                );
            }
            if edge.polyline.iter().any(|point| !point.is_finite()) {
                report.error(
                    "non_finite_edge",
                    format!("{path}.polyline"),
                    "edge points must be finite",
                );
            }
            if edge.visibility == EdgeVisibility::Visible
                && edge.polyline.iter().any(|point| !inside_image(*point))
            {
                report.error(
                    "visible_edge_outside_image",
                    format!("{path}.polyline"),
                    "fully visible edge points must lie inside normalized image bounds",
                );
            }
        }
        report
    }

    /// Validates sequence ordering, split stability, and all supplied frame records.
    #[must_use]
    pub fn validate_sequence(
        &self,
        sequence: &SequenceRecord,
        frames: &[FrameRecord],
    ) -> ValidationReport {
        let mut report = ValidationReport::default();
        if sequence.schema_version != self.manifest.schema_version {
            report.error(
                "schema_mismatch",
                "schema_version",
                "sequence schema differs from dataset manifest",
            );
        }
        if sequence.sequence_id.trim().is_empty() || sequence.building_family.trim().is_empty() {
            report.error(
                "empty_sequence_identity",
                "sequence_id",
                "sequence and building-family IDs must not be empty",
            );
        }
        if sequence.config_fingerprint != self.manifest.generator.config_fingerprint {
            report.error(
                "config_fingerprint_mismatch",
                "config_fingerprint",
                "sequence was not generated with the manifest's recorded configuration",
            );
        }
        validate_sampled_scene(&mut report, &sequence.scene);
        match (sequence.target_kind, sequence.scene.ordinary_roof) {
            (TargetKind::Negative, None) => report.error(
                "missing_ordinary_negative_roof",
                "scene.ordinary_roof",
                "negative sequences must identify the ordinary roof they render",
            ),
            (TargetKind::Target | TargetKind::NearMiss, Some(_)) => report.error(
                "target_has_ordinary_roof",
                "scene.ordinary_roof",
                "target sequences cannot replace target geometry with an ordinary roof",
            ),
            _ => {}
        }
        validate_camera_motion_plan(&mut report, "camera_motion", sequence.camera_motion);
        for (index, source_id) in sequence
            .scene
            .composition
            .source_asset_ids
            .iter()
            .enumerate()
        {
            match self
                .manifest
                .source_assets
                .iter()
                .find(|asset| asset.id == *source_id)
            {
                None => report.error(
                    "unknown_source_asset_id",
                    format!("scene.composition.source_asset_ids[{index}]"),
                    "resolved source asset ID is not declared in the dataset manifest",
                ),
                Some(asset)
                    if asset.split_group.is_some()
                        && asset.split_group != sequence.source_asset_group =>
                {
                    report.error(
                        "source_asset_group_mismatch",
                        format!("scene.composition.source_asset_ids[{index}]"),
                        "source asset split group must match the sequence split-group key",
                    );
                }
                Some(_) => {}
            }
        }
        let expected_split = self.manifest.split_policy.assign(&SplitKey {
            building_family: sequence.building_family.clone(),
            building_seed: sequence.building_seed,
            source_asset_group: sequence.source_asset_group.clone(),
        });
        match expected_split {
            Ok(split) if split != sequence.split => report.error(
                "split_mismatch",
                "split",
                "sequence split does not match its building-level key",
            ),
            Err(error) => report.error("invalid_split_policy", "split", error.to_string()),
            Ok(_) => {}
        }
        if sequence.frames.is_empty() {
            report.error(
                "empty_sequence",
                "frames",
                "sequence must contain at least one frame reference",
            );
        }
        if sequence.frames.len() != frames.len() {
            report.error(
                "frame_count_mismatch",
                "frames",
                "frame-reference and frame-record counts differ",
            );
        }

        let mut sample_keys = HashSet::new();
        let mut last_timestamp = None;
        for (position, frame_ref) in sequence.frames.iter().enumerate() {
            if frame_ref.frame_index as usize != position {
                report.error(
                    "non_contiguous_frame_index",
                    format!("frames[{position}].frame_index"),
                    "frame indexes must be contiguous and zero based",
                );
            }
            if !sample_keys.insert(frame_ref.sample_key.as_str()) {
                report.error(
                    "duplicate_sample_key",
                    format!("frames[{position}].sample_key"),
                    "sample keys must be unique",
                );
            }
            if let Some(previous) = last_timestamp
                && frame_ref.timestamp_ns <= previous
            {
                report.error(
                    "non_monotonic_timestamp",
                    format!("frames[{position}].timestamp_ns"),
                    "timestamps must increase strictly",
                );
            }
            last_timestamp = Some(frame_ref.timestamp_ns);
        }

        let mut previous_camera_position = None;
        for (index, frame) in frames.iter().enumerate() {
            report.append(self.validate_frame(frame));
            let camera_position = frame.camera.world_from_camera.translation;
            validate_camera_position(
                &mut report,
                &format!("frame_records[{index}].camera.world_from_camera.translation"),
                camera_position,
                &sequence.scene,
                previous_camera_position,
            );
            previous_camera_position = Some(camera_position);
            if frame.sequence_id != sequence.sequence_id
                || frame.split != sequence.split
                || frame.locator.target_kind != sequence.target_kind
            {
                report.error(
                    "frame_sequence_mismatch",
                    format!("frame_records[{index}]"),
                    "frame identity, split, and target kind must match the sequence",
                );
            }
            if let Some(frame_ref) = sequence.frames.get(index)
                && (frame.sample_key != frame_ref.sample_key
                    || frame.frame_index != frame_ref.frame_index
                    || frame.timestamp_ns != frame_ref.timestamp_ns)
            {
                report.error(
                    "frame_reference_mismatch",
                    format!("frame_records[{index}]"),
                    "frame record does not match the ordered sequence reference",
                );
            }
        }
        report
    }
}

fn validate_roof_profiles(report: &mut ValidationReport, config: &GeneratorConfig) {
    let profiles = &config.roof.profiles;
    if profiles.is_empty() || profiles.iter().all(|profile| profile.weight == 0) {
        report.error(
            "empty_roof_morphology_distribution",
            "roof.profiles",
            "at least one positive-weight roof morphology profile is required",
        );
    }

    let mut seen = HashSet::new();
    let globally_feasible_aspect = if config.scene.footprint_width_m.is_valid()
        && config.scene.footprint_depth_m.is_valid()
        && config.scene.footprint_width_m.min > 0.0
        && config.scene.footprint_depth_m.min > 0.0
    {
        Some(FloatRange::new(
            config.scene.footprint_width_m.min / config.scene.footprint_depth_m.max,
            config.scene.footprint_width_m.max / config.scene.footprint_depth_m.min,
        ))
    } else {
        None
    };

    for (index, profile) in profiles.iter().enumerate() {
        let base = format!("roof.profiles[{index}]");
        if !seen.insert(profile.morphology) {
            report.error(
                "duplicate_roof_morphology",
                format!("{base}.morphology"),
                "each roof morphology may appear at most once",
            );
        }

        for (name, range, positive) in [
            (
                "footprint_aspect_ratio",
                profile.footprint_aspect_ratio,
                true,
            ),
            ("overhang_m", profile.overhang_m, true),
            (
                "shoulder_width_fraction",
                profile.shoulder_width_fraction,
                true,
            ),
            (
                "shoulder_depth_fraction",
                profile.shoulder_depth_fraction,
                true,
            ),
            ("lower_rise_m", profile.lower_rise_m, true),
            ("upper_rise_m", profile.upper_rise_m, true),
            (
                "crown_top_width_fraction",
                profile.crown_top_width_fraction,
                true,
            ),
            (
                "crown_top_depth_fraction",
                profile.crown_top_depth_fraction,
                true,
            ),
        ] {
            validate_range(report, &format!("{base}.{name}"), range, positive);
        }
        for (name, range) in [
            ("shoulder_width_fraction", profile.shoulder_width_fraction),
            ("shoulder_depth_fraction", profile.shoulder_depth_fraction),
            ("crown_top_width_fraction", profile.crown_top_width_fraction),
            ("crown_top_depth_fraction", profile.crown_top_depth_fraction),
        ] {
            let path = format!("{base}.{name}");
            validate_unit_range(report, &path, range);
            if range.is_valid() && range.max >= 1.0 {
                report.error(
                    "non_nested_roof_profile",
                    path,
                    "roof profile fractions must remain strictly below one",
                );
            }
        }

        for (name, global, profile_range) in [
            ("overhang_m", config.roof.overhang_m, profile.overhang_m),
            (
                "shoulder_width_fraction",
                config.roof.shoulder_width_fraction,
                profile.shoulder_width_fraction,
            ),
            (
                "shoulder_depth_fraction",
                config.roof.shoulder_depth_fraction,
                profile.shoulder_depth_fraction,
            ),
            (
                "lower_rise_m",
                config.roof.lower_rise_m,
                profile.lower_rise_m,
            ),
            (
                "upper_rise_m",
                config.roof.upper_rise_m,
                profile.upper_rise_m,
            ),
            (
                "crown_top_width_fraction",
                config.roof.crown_top_width_fraction,
                profile.crown_top_width_fraction,
            ),
            (
                "crown_top_depth_fraction",
                config.roof.crown_top_depth_fraction,
                profile.crown_top_depth_fraction,
            ),
        ] {
            if global.is_valid()
                && profile_range.is_valid()
                && global.intersection(profile_range).is_none()
            {
                report.error(
                    "unsupported_roof_profile_range",
                    format!("{base}.{name}"),
                    "profile range must overlap the corresponding global roof range",
                );
            }
        }

        if let Some(global_aspect) = globally_feasible_aspect
            && profile.footprint_aspect_ratio.is_valid()
            && profile
                .footprint_aspect_ratio
                .intersection(global_aspect)
                .is_none()
        {
            report.error(
                "unsupported_footprint_aspect",
                format!("{base}.footprint_aspect_ratio"),
                "profile aspect ratio must admit width and depth inside the global scene bounds",
            );
        }
    }
}

fn validate_composition_config(report: &mut ValidationReport, config: &GeneratorConfig) {
    let composition = &config.composition;
    validate_materials(
        report,
        "composition.ground_materials",
        &composition.ground_materials,
    );
    validate_materials(
        report,
        "composition.facade.glazing_materials",
        &composition.facade.glazing_materials,
    );
    validate_range(
        report,
        "composition.facade.glazing_fraction",
        composition.facade.glazing_fraction,
        false,
    );
    validate_unit_range(
        report,
        "composition.facade.glazing_fraction",
        composition.facade.glazing_fraction,
    );
    validate_range(
        report,
        "composition.facade.entrance_width_m",
        composition.facade.entrance_width_m,
        true,
    );
    validate_range(
        report,
        "composition.facade.weathering",
        composition.facade.weathering,
        false,
    );
    validate_unit_range(
        report,
        "composition.facade.weathering",
        composition.facade.weathering,
    );

    let extensions = composition.building_extensions;
    if [
        extensions.none_weight,
        extensions.one_weight,
        extensions.two_weight,
    ]
    .iter()
    .all(|weight| *weight == 0)
    {
        report.error(
            "empty_building_extension_distribution",
            "composition.building_extensions",
            "at least one extension-count weight must be positive",
        );
    }
    if [
        extensions.dining_wing_weight,
        extensions.entrance_vestibule_weight,
        extensions.service_annex_weight,
    ]
    .iter()
    .all(|weight| *weight == 0)
        || [extensions.flat_roof_weight, extensions.shed_roof_weight]
            .iter()
            .all(|weight| *weight == 0)
    {
        report.error(
            "empty_building_extension_kind_distribution",
            "composition.building_extensions",
            "extension kind and roof distributions each need a positive weight",
        );
    }
    for (name, range, positive) in [
        (
            "facade_width_fraction",
            extensions.facade_width_fraction,
            true,
        ),
        ("projection_m", extensions.projection_m, true),
        (
            "wall_height_fraction",
            extensions.wall_height_fraction,
            true,
        ),
        (
            "facade_offset_fraction",
            extensions.facade_offset_fraction,
            false,
        ),
        ("shed_rise_m", extensions.shed_rise_m, true),
    ] {
        validate_range(
            report,
            &format!("composition.building_extensions.{name}"),
            range,
            positive,
        );
    }
    validate_unit_range(
        report,
        "composition.building_extensions.facade_width_fraction",
        extensions.facade_width_fraction,
    );
    validate_unit_range(
        report,
        "composition.building_extensions.wall_height_fraction",
        extensions.wall_height_fraction,
    );
    if extensions.facade_offset_fraction.min < -1.0
        || extensions.facade_offset_fraction.max > 1.0
        || extensions.projection_m.max > 12.0
        || extensions.shed_rise_m.max > 2.0
    {
        report.error(
            "implausible_building_extension_distribution",
            "composition.building_extensions",
            "extension offset, projection, and roof rise exceed supported physical bounds",
        );
    }

    let signage = composition.signage;
    if [
        signage.none_weight,
        signage.pizza_hut_weight,
        signage.removed_ghost_weight,
        signage.rebranded_tenant_weight,
    ]
    .iter()
    .all(|weight| *weight == 0)
    {
        report.error(
            "empty_signage_distribution",
            "composition.signage",
            "at least one blank or sign-state weight must be positive",
        );
    }
    for (path, range, positive) in [
        (
            "composition.signage.width_fraction",
            signage.width_fraction,
            true,
        ),
        ("composition.signage.height_m", signage.height_m, true),
        (
            "composition.signage.vertical_fraction",
            signage.vertical_fraction,
            true,
        ),
        (
            "composition.signage.emissive_strength",
            signage.emissive_strength,
            false,
        ),
    ] {
        validate_range(report, path, range, positive);
    }
    validate_unit_range(
        report,
        "composition.signage.width_fraction",
        signage.width_fraction,
    );
    validate_unit_range(
        report,
        "composition.signage.vertical_fraction",
        signage.vertical_fraction,
    );

    if composition.day_phase.profiles.is_empty()
        || composition
            .day_phase
            .profiles
            .iter()
            .all(|profile| profile.weight == 0)
    {
        report.error(
            "empty_day_phase_distribution",
            "composition.day_phase.profiles",
            "at least one positive-weight day-phase profile is required",
        );
    }
    let has_day = composition
        .day_phase
        .profiles
        .iter()
        .any(|profile| profile.phase == DayPhase::Day && profile.weight > 0);
    let has_night = composition
        .day_phase
        .profiles
        .iter()
        .any(|profile| profile.phase == DayPhase::Night && profile.weight > 0);
    if !has_day || !has_night {
        report.error(
            "missing_day_or_night_regime",
            "composition.day_phase.profiles",
            "default training generation must include positive-weight day and night regimes",
        );
    }
    for (index, profile) in composition.day_phase.profiles.iter().enumerate() {
        let base = format!("composition.day_phase.profiles[{index}]");
        validate_range(
            report,
            &format!("{base}.sun_elevation_degrees"),
            profile.sun_elevation_degrees,
            false,
        );
        validate_range(
            report,
            &format!("{base}.camera_exposure_ev100"),
            profile.camera_exposure_ev100,
            false,
        );
        validate_range(
            report,
            &format!("{base}.artificial_light_strength"),
            profile.artificial_light_strength,
            false,
        );
        if profile.artificial_light_strength.min < 0.0
            || !ranges_overlap(
                profile.sun_elevation_degrees,
                config.lighting.sun_elevation_degrees,
            )
        {
            report.error(
                "invalid_day_phase_profile",
                &base,
                "day-phase light must be nonnegative and solar elevation must overlap global bounds",
            );
        }
        match profile.phase {
            DayPhase::Day if profile.sun_elevation_degrees.min < 0.0 => report.error(
                "day_sun_below_horizon",
                format!("{base}.sun_elevation_degrees"),
                "day profile cannot place the sun below the horizon",
            ),
            DayPhase::Night if profile.sun_elevation_degrees.max > -4.0 => report.error(
                "night_sun_too_high",
                format!("{base}.sun_elevation_degrees"),
                "night profile must keep the sun at least four degrees below the horizon",
            ),
            _ => {}
        }
    }

    if composition.weather.profiles.is_empty()
        || composition
            .weather
            .profiles
            .iter()
            .all(|profile| profile.weight == 0)
    {
        report.error(
            "empty_weather_distribution",
            "composition.weather.profiles",
            "at least one positive-weight weather profile is required",
        );
    }
    for (index, profile) in composition.weather.profiles.iter().enumerate() {
        let base = format!("composition.weather.profiles[{index}]");
        for (name, range, positive) in [
            ("cloud_coverage", profile.cloud_coverage, false),
            ("haze", profile.haze, false),
            ("sun_intensity", profile.sun_intensity, false),
            ("sky_intensity", profile.sky_intensity, false),
            ("shadow_softness", profile.shadow_softness, false),
            ("ground_wetness", profile.ground_wetness, false),
            ("visibility_km", profile.visibility_km, true),
            ("color_temperature_k", profile.color_temperature_k, true),
        ] {
            validate_range(report, &format!("{base}.{name}"), range, positive);
        }
        for (name, range) in [
            ("cloud_coverage", profile.cloud_coverage),
            ("haze", profile.haze),
            ("shadow_softness", profile.shadow_softness),
            ("ground_wetness", profile.ground_wetness),
        ] {
            validate_unit_range(report, &format!("{base}.{name}"), range);
        }
        if profile.sun_intensity.min < 0.0
            || profile.sky_intensity.min < 0.0
            || profile.color_temperature_k.min < 1_800.0
            || profile.color_temperature_k.max > 12_000.0
            || !ranges_overlap(profile.cloud_coverage, config.lighting.cloud_coverage)
            || !ranges_overlap(profile.haze, config.lighting.haze)
            || !ranges_overlap(profile.sun_intensity, config.lighting.sun_intensity)
            || !ranges_overlap(profile.sky_intensity, config.lighting.sky_intensity)
        {
            report.error(
                "invalid_weather_profile",
                base,
                "weather profile must overlap global lighting bounds and remain physically plausible",
            );
        }
    }

    if composition.domains.profiles.is_empty()
        || composition
            .domains
            .profiles
            .iter()
            .all(|profile| profile.weight == 0)
    {
        report.error(
            "empty_domain_distribution",
            "composition.domains.profiles",
            "at least one positive-weight site domain is required",
        );
    }
    for (index, profile) in composition.domains.profiles.iter().enumerate() {
        let base = format!("composition.domains.profiles[{index}]");
        for (name, range) in [
            ("background_buildings", profile.background_buildings),
            ("vegetation", profile.vegetation),
            ("parking_bays", profile.parking_bays),
            ("road_lanes", profile.road_lanes),
            ("utility_poles", profile.utility_poles),
        ] {
            validate_u32_range(report, &format!("{base}.{name}"), range);
        }
        validate_probability(
            report,
            &format!("{base}.curb_probability"),
            profile.curb_probability,
        );
        validate_probability(
            report,
            &format!("{base}.overhead_line_probability"),
            profile.overhead_line_probability,
        );
        if profile.road_lanes.min == 0
            || profile.utility_poles.max > u32::from(u16::MAX)
            || intersect_u32(
                profile.background_buildings,
                composition.background_buildings.count,
            )
            .is_none()
            || intersect_u32(profile.vegetation, composition.vegetation.count).is_none()
        {
            report.error(
                "invalid_domain_profile",
                base,
                "domain counts must overlap global composition bounds, roads need a lane, and pole indexes must fit u16",
            );
        }
    }

    let background = composition.background_buildings;
    validate_u32_range(
        report,
        "composition.background_buildings.count",
        background.count,
    );
    for (name, range, positive) in [
        ("setback_m", background.setback_m, true),
        ("width_m", background.width_m, true),
        ("depth_m", background.depth_m, true),
        ("height_m", background.height_m, true),
        ("yaw_jitter_degrees", background.yaw_jitter_degrees, false),
    ] {
        validate_range(
            report,
            &format!("composition.background_buildings.{name}"),
            range,
            positive,
        );
    }

    let vegetation = composition.vegetation;
    validate_u32_range(report, "composition.vegetation.count", vegetation.count);
    if [
        vegetation.deciduous_weight,
        vegetation.evergreen_weight,
        vegetation.palm_weight,
        vegetation.shrub_weight,
    ]
    .iter()
    .all(|weight| *weight == 0)
    {
        report.error(
            "empty_vegetation_distribution",
            "composition.vegetation",
            "at least one plant-family weight must be positive",
        );
    }
    for (name, range, positive) in [
        ("building_setback_m", vegetation.building_setback_m, true),
        ("site_distance_m", vegetation.site_distance_m, false),
        ("tree_height_m", vegetation.tree_height_m, true),
        ("shrub_height_m", vegetation.shrub_height_m, true),
        (
            "canopy_radius_fraction",
            vegetation.canopy_radius_fraction,
            true,
        ),
    ] {
        validate_range(
            report,
            &format!("composition.vegetation.{name}"),
            range,
            positive,
        );
    }
    if vegetation.site_distance_m.min < 0.0 || vegetation.canopy_radius_fraction.max > 0.75 {
        report.error(
            "implausible_vegetation_layout",
            "composition.vegetation",
            "site distance cannot be negative and canopy radius must stay below 75% of height",
        );
    }
}

fn validate_probability(report: &mut ValidationReport, path: &str, value: f32) {
    if !value.is_finite() || !(0.0..=1.0).contains(&value) {
        report.error(
            "invalid_probability",
            path,
            "probability must be finite and inside [0, 1]",
        );
    }
}

fn ranges_overlap(left: FloatRange, right: FloatRange) -> bool {
    left.is_valid() && right.is_valid() && left.min.max(right.min) <= left.max.min(right.max)
}

fn intersect_u32(left: U32Range, right: U32Range) -> Option<U32Range> {
    let min = left.min.max(right.min);
    let max = left.max.min(right.max);
    (min <= max).then(|| U32Range::new(min, max))
}

fn validate_range(report: &mut ValidationReport, path: &str, range: FloatRange, positive: bool) {
    if !range.is_valid() {
        report.error(
            "invalid_range",
            path,
            "range bounds must be finite and ordered",
        );
    } else if positive && range.min <= 0.0 {
        report.error("non_positive_range", path, "range must remain positive");
    }
}

fn validate_u32_range(report: &mut ValidationReport, path: &str, range: U32Range) {
    if !range.is_valid() {
        report.error("invalid_range", path, "range bounds must be ordered");
    }
}

fn validate_unit_range(report: &mut ValidationReport, path: &str, range: FloatRange) {
    if range.is_valid() && (range.min < 0.0 || range.max > 1.0) {
        report.error(
            "outside_unit_range",
            path,
            "range must remain inside [0, 1]",
        );
    }
}

fn validate_materials(report: &mut ValidationReport, path: &str, materials: &[MaterialChoice]) {
    if materials.is_empty() || materials.iter().all(|material| material.weight == 0) {
        report.error(
            "empty_weighted_distribution",
            path,
            "at least one positive-weight material is required",
        );
        return;
    }
    let mut ids = HashSet::new();
    for (index, material) in materials.iter().enumerate() {
        let item_path = format!("{path}[{index}]");
        if material.id.trim().is_empty() || !ids.insert(material.id.as_str()) {
            report.error(
                "invalid_material_id",
                format!("{item_path}.id"),
                "material ID must be non-empty and unique",
            );
        }
        if material
            .base_color_srgb
            .iter()
            .any(|channel| !channel.is_finite() || !(0.0..=1.0).contains(channel))
        {
            report.error(
                "invalid_material_color",
                format!("{item_path}.base_color_srgb"),
                "sRGB channels must be finite and inside [0, 1]",
            );
        }
        validate_range(
            report,
            &format!("{item_path}.roughness"),
            material.roughness,
            false,
        );
        validate_range(
            report,
            &format!("{item_path}.weathering"),
            material.weathering,
            false,
        );
        validate_unit_range(
            report,
            &format!("{item_path}.roughness"),
            material.roughness,
        );
        validate_unit_range(
            report,
            &format!("{item_path}.weathering"),
            material.weathering,
        );
    }
}

fn validate_classes(report: &mut ValidationReport, path: &str, classes: &[LabelClass]) {
    let mut ids = HashSet::new();
    let mut names = HashSet::new();
    for (index, class) in classes.iter().enumerate() {
        if !ids.insert(class.id) {
            report.error(
                "duplicate_class_id",
                format!("{path}[{index}].id"),
                "class IDs must be unique within a taxonomy",
            );
        }
        if class.name.trim().is_empty() || !names.insert(class.name.as_str()) {
            report.error(
                "invalid_class_name",
                format!("{path}[{index}].name"),
                "class names must be non-empty and unique",
            );
        }
    }
}

fn validate_camera(report: &mut ValidationReport, frame: &FrameRecord) {
    let camera = frame.camera;
    let intrinsics = camera.intrinsics;
    if intrinsics.width == 0
        || intrinsics.height == 0
        || !intrinsics.fx.is_finite()
        || !intrinsics.fy.is_finite()
        || intrinsics.fx <= 0.0
        || intrinsics.fy <= 0.0
        || !intrinsics.cx.is_finite()
        || !intrinsics.cy.is_finite()
        || !intrinsics.skew.is_finite()
    {
        report.error(
            "invalid_camera_intrinsics",
            "camera.intrinsics",
            "dimensions and all finite pinhole terms are required",
        );
    }
    if !camera.world_from_camera.is_valid() {
        report.error(
            "invalid_camera_transform",
            "camera.world_from_camera",
            "camera transform must be finite with a unit quaternion",
        );
    }
    if camera
        .output_from_sensor
        .0
        .iter()
        .any(|value| !value.is_finite())
    {
        report.error(
            "invalid_image_transform",
            "camera.output_from_sensor",
            "image transform must contain only finite values",
        );
    }
    let matrix = camera.output_from_sensor.0;
    let determinant = matrix[0] * (matrix[4] * matrix[8] - matrix[5] * matrix[7])
        - matrix[1] * (matrix[3] * matrix[8] - matrix[5] * matrix[6])
        + matrix[2] * (matrix[3] * matrix[7] - matrix[4] * matrix[6]);
    if determinant.is_finite() && determinant.abs() <= f32::EPSILON {
        report.error(
            "singular_image_transform",
            "camera.output_from_sensor",
            "image transform must be invertible",
        );
    }
    match camera.distortion {
        crate::DistortionModel::None => {}
        crate::DistortionModel::BrownConrady { k1, k2, p1, p2, k3 }
            if [k1, k2, p1, p2, k3].iter().any(|value| !value.is_finite()) =>
        {
            report.error(
                "invalid_distortion",
                "camera.distortion",
                "lens-distortion coefficients must be finite",
            );
        }
        crate::DistortionModel::BrownConrady { .. } => {}
    }
}

fn validate_sampled_scene(report: &mut ValidationReport, scene: &SampledScene) {
    let building = scene.building;
    if [
        building.footprint_width_m,
        building.footprint_depth_m,
        building.wall_height_m,
        building.ground_half_extent_m,
    ]
    .iter()
    .any(|value| !value.is_finite() || *value <= 0.0)
    {
        report.error(
            "invalid_sampled_building",
            "scene.building",
            "all sampled building dimensions must be finite and positive",
        );
    }
    let roof = scene.roof;
    if [
        roof.eave_width_m,
        roof.eave_depth_m,
        roof.shoulder_width_m,
        roof.shoulder_depth_m,
        roof.lower_rise_m,
        roof.upper_rise_m,
        roof.crown_top_width_m,
        roof.crown_top_depth_m,
    ]
    .iter()
    .any(|value| !value.is_finite() || *value <= 0.0)
        || !roof.asymmetry_fraction.is_finite()
        || roof.shoulder_width_m >= roof.eave_width_m
        || roof.shoulder_depth_m >= roof.eave_depth_m
        || roof.crown_top_width_m >= roof.shoulder_width_m
        || roof.crown_top_depth_m >= roof.shoulder_depth_m
    {
        report.error(
            "invalid_sampled_roof",
            "scene.roof",
            "roof dimensions must be finite, positive, and form three nested rectangles",
        );
    }
    if let Some(ordinary) = scene.ordinary_roof
        && ([
            ordinary.eave_width_m,
            ordinary.eave_depth_m,
            ordinary.rise_m,
            ordinary.cap_height_m,
        ]
        .iter()
        .any(|value| !value.is_finite() || *value < 0.0)
            || ordinary.eave_width_m <= building.footprint_width_m
            || ordinary.eave_depth_m <= building.footprint_depth_m
            || ordinary.rise_m <= 0.0
            || !ordinary.ridge_length_fraction.is_finite()
            || !(0.0..=1.0).contains(&ordinary.ridge_length_fraction)
            || !ordinary.inset_fraction.is_finite()
            || !(0.0..=1.0).contains(&ordinary.inset_fraction))
    {
        report.error(
            "invalid_sampled_ordinary_roof",
            "scene.ordinary_roof",
            "ordinary-roof dimensions and topology fractions must be finite and physically bounded",
        );
    }
    for (path, material) in [
        ("scene.roof_material", &scene.roof_material),
        ("scene.wall_material", &scene.wall_material),
    ] {
        if material.id.trim().is_empty()
            || material
                .base_color_srgb
                .iter()
                .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
            || !material.roughness.is_finite()
            || !(0.0..=1.0).contains(&material.roughness)
            || !material.weathering.is_finite()
            || !(0.0..=1.0).contains(&material.weathering)
        {
            report.error(
                "invalid_sampled_material",
                path,
                "sampled material values must be finite and physically bounded",
            );
        }
    }
    let light = scene.lighting;
    if [
        light.sun_azimuth_degrees,
        light.sun_elevation_degrees,
        light.sun_intensity,
        light.sky_intensity,
        light.cloud_coverage,
        light.haze,
    ]
    .iter()
    .any(|value| !value.is_finite())
        || light.sun_intensity < 0.0
        || light.sky_intensity < 0.0
        || !(0.0..=1.0).contains(&light.cloud_coverage)
        || !(0.0..=1.0).contains(&light.haze)
    {
        report.error(
            "invalid_sampled_lighting",
            "scene.lighting",
            "sampled lighting must be finite with bounded cloud and haze values",
        );
    }
    for (index, occluder) in scene.occluders.iter().enumerate() {
        if !occluder.position.is_finite()
            || !occluder.yaw_degrees.is_finite()
            || !occluder.scale.is_finite()
            || occluder.scale <= 0.0
            || !occluder.nominal_size_m.is_finite()
            || occluder.nominal_size_m.x <= 0.0
            || occluder.nominal_size_m.y <= 0.0
            || occluder.nominal_size_m.z <= 0.0
        {
            report.error(
                "invalid_sampled_occluder",
                format!("scene.occluders[{index}]"),
                "sampled occluder placement and scale must be finite",
            );
        }
        let (half_x, half_z) = rotated_half_extents(occluder.nominal_size_m, occluder.yaw_degrees);
        if occluder.placement == OccluderPlacement::Rooftop {
            let roof_top = building.wall_height_m + roof.lower_rise_m + roof.upper_rise_m;
            if occluder.position.y < roof_top - 1.0e-3
                || occluder.position.x.abs() + half_x + ROOFTOP_EDGE_CLEARANCE_M
                    > roof.crown_top_width_m * 0.5 + 1.0e-3
                || occluder.position.z.abs() + half_z + ROOFTOP_EDGE_CLEARANCE_M
                    > roof.crown_top_depth_m * 0.5 + 1.0e-3
            {
                report.error(
                    "rooftop_occluder_outside_roof",
                    format!("scene.occluders[{index}].position"),
                    "rooftop equipment base and footprint must remain on the flat crown top",
                );
            }
        } else if occluder.position.x.abs()
            < building.footprint_width_m * 0.5 + half_x + OCCLUDER_TARGET_CLEARANCE_M
            && occluder.position.z.abs()
                < building.footprint_depth_m * 0.5 + half_z + OCCLUDER_TARGET_CLEARANCE_M
        {
            report.error(
                "occluder_intersects_target",
                format!("scene.occluders[{index}].position"),
                "non-rooftop occluders must clear the target footprint",
            );
        }
    }
    validate_sampled_composition(report, scene);
    let required_ground = sampled_content_half_extent(
        building,
        actual_roof_envelope(scene),
        &scene.composition,
        &scene.occluders,
        &[],
    ) + GROUND_EDGE_MARGIN_M;
    if building.ground_half_extent_m + 1.0e-3 < required_ground {
        report.error(
            "scene_outside_ground",
            "scene.building.ground_half_extent_m",
            "finite ground must contain all sampled scene content with an edge margin",
        );
    }
}

fn validate_sampled_composition(report: &mut ValidationReport, scene: &SampledScene) {
    let composition = &scene.composition;
    let mut source_ids = HashSet::new();
    for (index, id) in composition.source_asset_ids.iter().enumerate() {
        if id.trim().is_empty() || !source_ids.insert(id.as_str()) {
            report.error(
                "invalid_source_asset_id",
                format!("scene.composition.source_asset_ids[{index}]"),
                "resolved source asset IDs must be non-empty and unique",
            );
        }
    }
    let Some(environment) = composition.environment else {
        report.error(
            "missing_sampled_environment",
            "scene.composition.environment",
            "new sequence plans require a correlated environment regime",
        );
        return;
    };
    if [
        environment.shadow_softness,
        environment.ground_wetness,
        environment.visibility_km,
        environment.color_temperature_k,
        environment.camera_exposure_ev100,
        environment.artificial_light_strength,
    ]
    .iter()
    .any(|value| !value.is_finite())
        || !(0.0..=1.0).contains(&environment.shadow_softness)
        || !(0.0..=1.0).contains(&environment.ground_wetness)
        || environment.visibility_km <= 0.0
        || !(1_800.0..=12_000.0).contains(&environment.color_temperature_k)
        || environment.artificial_light_strength < 0.0
    {
        report.error(
            "invalid_sampled_environment",
            "scene.composition.environment",
            "environment values must be finite and physically bounded",
        );
    }
    match environment.day_phase {
        DayPhase::Day if scene.lighting.sun_elevation_degrees < 0.0 => report.error(
            "day_lighting_mismatch",
            "scene.lighting.sun_elevation_degrees",
            "day regime cannot have a below-horizon sun",
        ),
        DayPhase::Night
            if scene.lighting.sun_elevation_degrees > -4.0
                || scene.lighting.sun_intensity != 0.0 =>
        {
            report.error(
                "night_lighting_mismatch",
                "scene.lighting",
                "night regime requires a below-horizon sun and zero direct solar intensity",
            );
        }
        _ => {}
    }

    match &composition.ground_material {
        Some(material) => {
            validate_sampled_material(report, "scene.composition.ground_material", material)
        }
        None => report.error(
            "missing_ground_material",
            "scene.composition.ground_material",
            "new sequence plans require a resolved ground material",
        ),
    }
    match &composition.facade {
        Some(facade) => {
            validate_sampled_material(
                report,
                "scene.composition.facade.glazing_material",
                &facade.glazing_material,
            );
            if !facade.glazing_fraction.is_finite()
                || !(0.0..=1.0).contains(&facade.glazing_fraction)
                || !facade.entrance_width_m.is_finite()
                || facade.entrance_width_m <= 0.0
                || facade.entrance_width_m
                    >= scene
                        .building
                        .footprint_width_m
                        .min(scene.building.footprint_depth_m)
                || !facade.weathering.is_finite()
                || !(0.0..=1.0).contains(&facade.weathering)
            {
                report.error(
                    "invalid_sampled_facade",
                    "scene.composition.facade",
                    "façade fractions and entrance dimensions must be physically bounded",
                );
            }
        }
        None => report.error(
            "missing_sampled_facade",
            "scene.composition.facade",
            "new sequence plans require resolved façade detail",
        ),
    }

    let mut extension_facades = HashSet::new();
    for (index, extension) in composition.building_extensions.iter().enumerate() {
        let (expected_normal_position, actual_normal_position, tangential_limit, tangential_extent) =
            match extension.facade {
                crate::FacadeSide::Front => (
                    -scene.building.footprint_depth_m * 0.5 - extension.size_m.z * 0.5 + 0.12,
                    extension.position.z,
                    scene.building.footprint_width_m * 0.5,
                    extension.position.x.abs() + extension.size_m.x * 0.5,
                ),
                crate::FacadeSide::Back => (
                    scene.building.footprint_depth_m * 0.5 + extension.size_m.z * 0.5 - 0.12,
                    extension.position.z,
                    scene.building.footprint_width_m * 0.5,
                    extension.position.x.abs() + extension.size_m.x * 0.5,
                ),
                crate::FacadeSide::Right => (
                    scene.building.footprint_width_m * 0.5 + extension.size_m.x * 0.5 - 0.12,
                    extension.position.x,
                    scene.building.footprint_depth_m * 0.5,
                    extension.position.z.abs() + extension.size_m.z * 0.5,
                ),
                crate::FacadeSide::Left => (
                    -scene.building.footprint_width_m * 0.5 - extension.size_m.x * 0.5 + 0.12,
                    extension.position.x,
                    scene.building.footprint_depth_m * 0.5,
                    extension.position.z.abs() + extension.size_m.z * 0.5,
                ),
            };
        let roof_consistent = match extension.roof {
            crate::BuildingExtensionRoof::Flat => extension.roof_rise_m == 0.0,
            crate::BuildingExtensionRoof::Shed => extension.roof_rise_m > 0.0,
        };
        if !extension_facades.insert(extension.facade)
            || !extension.position.is_finite()
            || extension.position.y != 0.0
            || !extension.size_m.is_finite()
            || extension.size_m.x <= 0.0
            || extension.size_m.y <= 0.0
            || extension.size_m.z <= 0.0
            || extension.size_m.y >= scene.building.wall_height_m
            || !extension.roof_rise_m.is_finite()
            || extension.roof_rise_m < 0.0
            || extension.size_m.y + extension.roof_rise_m >= scene.building.wall_height_m
            || !roof_consistent
            || (expected_normal_position - actual_normal_position).abs() > 1.0e-3
            || tangential_extent > tangential_limit + 1.0e-3
        {
            report.error(
                "invalid_sampled_building_extension",
                format!("scene.composition.building_extensions[{index}]"),
                "extensions must be finite, below the main eave, attached to distinct façades, and remain within the host façade span",
            );
        }
    }

    for (index, sign) in composition.signage.iter().enumerate() {
        if !sign.center.is_finite()
            || !sign.size_m.is_finite()
            || sign.size_m.x <= 0.0
            || sign.size_m.y <= 0.0
            || sign.size_m.y >= scene.building.wall_height_m
            || !sign.emissive_strength.is_finite()
            || sign.emissive_strength < 0.0
            || !sign.weathering.is_finite()
            || !(0.0..=1.0).contains(&sign.weathering)
            || (sign.kind == crate::SignageKind::RemovedGhost && sign.emissive_strength != 0.0)
        {
            report.error(
                "invalid_sampled_signage",
                format!("scene.composition.signage[{index}]"),
                "sign dimensions, material state, and emission must be physically plausible",
            );
        }
    }

    let target_radius = libm::sqrtf(
        (scene.building.footprint_width_m * 0.5).powi(2)
            + (scene.building.footprint_depth_m * 0.5).powi(2),
    );
    for (index, background) in composition.background_buildings.iter().enumerate() {
        validate_sampled_material(
            report,
            &format!("scene.composition.background_buildings[{index}].material"),
            &background.material,
        );
        let own_radius =
            libm::sqrtf((background.size_m.x * 0.5).powi(2) + (background.size_m.z * 0.5).powi(2));
        let distance = libm::sqrtf(
            background.position.x * background.position.x
                + background.position.z * background.position.z,
        );
        if !background.position.is_finite()
            || !background.size_m.is_finite()
            || background.size_m.x <= 0.0
            || background.size_m.y <= 0.0
            || background.size_m.z <= 0.0
            || !background.yaw_degrees.is_finite()
            || distance <= target_radius + own_radius
        {
            report.error(
                "invalid_background_building",
                format!("scene.composition.background_buildings[{index}]"),
                "background buildings must be finite, positive, and outside the target footprint",
            );
        }
    }

    for (index, vegetation) in composition.vegetation.iter().enumerate() {
        let distance = libm::sqrtf(
            vegetation.position.x * vegetation.position.x
                + vegetation.position.z * vegetation.position.z,
        );
        if !vegetation.position.is_finite()
            || !vegetation.height_m.is_finite()
            || vegetation.height_m <= 0.0
            || !vegetation.canopy_radius_m.is_finite()
            || vegetation.canopy_radius_m <= 0.0
            || !vegetation.yaw_degrees.is_finite()
            || !vegetation.seasonal_variation.is_finite()
            || !(-1.0..=1.0).contains(&vegetation.seasonal_variation)
            || distance <= target_radius
        {
            report.error(
                "invalid_sampled_vegetation",
                format!("scene.composition.vegetation[{index}]"),
                "vegetation must have finite dimensions and remain outside the target footprint",
            );
        }
    }

    match &composition.infrastructure {
        Some(infrastructure) => {
            if infrastructure.domain != environment.domain
                || !infrastructure.parking_center.is_finite()
                || !infrastructure.parking_bay_size_m.is_finite()
                || infrastructure.parking_bay_size_m.x <= 0.0
                || infrastructure.parking_bay_size_m.y <= 0.0
                || !infrastructure.parking_yaw_degrees.is_finite()
                || infrastructure.road_lanes == 0
                || !infrastructure.lane_width_m.is_finite()
                || !(2.6..=4.2).contains(&infrastructure.lane_width_m)
                || !infrastructure.road_center_z_m.is_finite()
            {
                report.error(
                    "invalid_site_infrastructure",
                    "scene.composition.infrastructure",
                    "site infrastructure must match its domain and use plausible dimensions",
                );
            }
            for (index, pole) in infrastructure.utility_poles.iter().enumerate() {
                if !pole.position.is_finite()
                    || !pole.height_m.is_finite()
                    || !(3.0..=18.0).contains(&pole.height_m)
                {
                    report.error(
                        "invalid_utility_pole",
                        format!("scene.composition.infrastructure.utility_poles[{index}]"),
                        "utility poles require finite placement and plausible height",
                    );
                }
            }
            for (index, line) in infrastructure.utility_lines.iter().enumerate() {
                if usize::from(line.start_pole) >= infrastructure.utility_poles.len()
                    || usize::from(line.end_pole) >= infrastructure.utility_poles.len()
                    || line.start_pole == line.end_pole
                    || !line.sag_m.is_finite()
                    || line.sag_m <= 0.0
                {
                    report.error(
                        "invalid_utility_line",
                        format!("scene.composition.infrastructure.utility_lines[{index}]"),
                        "utility lines must connect distinct valid poles with positive sag",
                    );
                }
            }
        }
        None => report.error(
            "missing_site_infrastructure",
            "scene.composition.infrastructure",
            "new sequence plans require a correlated parking, road, curb, and utility layout",
        ),
    }
}

fn validate_sampled_material(
    report: &mut ValidationReport,
    path: &str,
    material: &SampledMaterial,
) {
    if material.id.trim().is_empty()
        || material
            .base_color_srgb
            .iter()
            .any(|value| !value.is_finite() || !(0.0..=1.0).contains(value))
        || !material.roughness.is_finite()
        || !(0.0..=1.0).contains(&material.roughness)
        || !material.weathering.is_finite()
        || !(0.0..=1.0).contains(&material.weathering)
    {
        report.error(
            "invalid_sampled_material",
            path,
            "sampled material values must be finite and physically bounded",
        );
    }
}

fn validate_locator(report: &mut ValidationReport, locator: &LocatorLabel) {
    for (path, value) in [
        ("locator.visible_fraction", locator.visible_fraction),
        ("locator.occluded_fraction", locator.occluded_fraction),
    ] {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            report.error(
                "invalid_fraction",
                path,
                "fraction must be finite and inside [0, 1]",
            );
        }
    }
    if let Some(bounds) = locator.bounding_box {
        validate_bounds(report, "locator.bounding_box", bounds);
    }
    if let Some(bounds) = locator.amodal_bounding_box {
        validate_bounds(report, "locator.amodal_bounding_box", bounds);
    }
}

fn validate_bounds(report: &mut ValidationReport, path: &str, bounds: NormalizedBoundingBox) {
    if !inside_image(bounds.min)
        || !inside_image(bounds.max)
        || bounds.min.x >= bounds.max.x
        || bounds.min.y >= bounds.max.y
    {
        report.error(
            "invalid_bounding_box",
            path,
            "normalized bounds must be finite, ordered, and inside [0, 1]",
        );
    }
}

fn validate_keypoint_projection(
    report: &mut ValidationReport,
    path: &str,
    keypoint: &crate::KeypointLabel,
) {
    match (keypoint.visibility, keypoint.image_position) {
        (Visibility::Visible | Visibility::Occluded, Some(point)) if inside_image(point) => {}
        (Visibility::Truncated, Some(point)) if point.is_finite() => {}
        (Visibility::BehindCamera, None) => {}
        (Visibility::Visible | Visibility::Occluded, _) => report.error(
            "invalid_visible_projection",
            format!("{path}.image_position"),
            "visible or occluded point requires an in-frame projection",
        ),
        (Visibility::Truncated, _) => report.error(
            "invalid_truncated_projection",
            format!("{path}.image_position"),
            "truncated point requires its finite pre-clip projection",
        ),
        (Visibility::BehindCamera, _) => report.error(
            "invalid_behind_camera_projection",
            format!("{path}.image_position"),
            "behind-camera point must not have an image projection",
        ),
    }
}

fn validate_asset(report: &mut ValidationReport, path: &str, asset: &AssetRef) {
    let asset_path = Path::new(&asset.path);
    if asset.path.trim().is_empty()
        || asset_path.is_absolute()
        || asset_path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        report.error(
            "unsafe_asset_path",
            format!("{path}.path"),
            "asset path must be non-empty, relative, and contain no parent traversal",
        );
    }
    if asset.media_type.trim().is_empty() || asset.encoding.trim().is_empty() {
        report.error(
            "incomplete_asset_encoding",
            path,
            "asset media type and encoding are required",
        );
    }
    if asset.content_hash.is_none() {
        report.warning(
            "missing_content_hash",
            format!("{path}.content_hash"),
            "writer should attach a digest before accepting the final shard",
        );
    }
}

fn inside_image(point: crate::Vec2) -> bool {
    point.is_finite() && (0.0..=1.0).contains(&point.x) && (0.0..=1.0).contains(&point.y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_generator_configuration_is_valid() {
        assert!(GeneratorConfig::default().validate().is_valid());
    }

    #[test]
    fn configuration_collects_multiple_failures() {
        let mut config = GeneratorConfig::default();
        config.image.width = 0;
        config.roof.lower_rise_m = FloatRange::new(4.0, 2.0);
        config.materials.roof.clear();
        let report = config.validate();
        assert!(report.error_count() >= 3);
    }
}
