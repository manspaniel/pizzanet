//! Held-out synthetic metrics for the complete perspective-fitted roof mesh.

use anyhow::{Context, Result, bail};
use roof_fit::{
    FocalLengthConfig, KEYPOINT_COUNT, KeypointObservation, SingleViewFitConfig,
    SingleViewObservation, fit_single_view,
};
use roof_geometry::{RoofParameters, generate_roof};
use roof_training::{SYMMETRY_COUNT, symmetry_target_slot};
use serde::Serialize;
use synth_data::{FrameRecord, RigidTransform};

const SILHOUETTE_LONG_EDGE: usize = 256;

/// Metrics for one synthetic positive whose detector observations produced a fit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct SyntheticFitEvaluation {
    /// D4-aligned corresponding-vertex RMSE, divided by the image diagonal.
    pub mesh_rmse: f32,
    /// IoU of complete fitted and exact synthetic roof silhouettes.
    pub silhouette_iou: f32,
    /// Whether the fitter's own observation/inlier/RMSE gate accepted the result.
    pub accepted: bool,
}

/// Aggregate fit metrics serialized alongside held-out observation metrics.
#[derive(Clone, Debug, Default, Serialize)]
pub(super) struct SyntheticFitMetrics {
    /// Synthetic positive frames passed to the fitter.
    pub attempted: usize,
    /// Frames for which nonlinear fitting returned a finite mesh.
    pub fitted: usize,
    /// Returned fits accepted by the fitter's confidence gate.
    pub accepted: usize,
    /// Fraction of attempts for which nonlinear fitting returned a mesh.
    pub fit_success_rate: f32,
    /// Fraction of attempts passing the fitter confidence gate.
    pub accepted_rate: f32,
    /// Median projected mesh error for all successfully fitted frames.
    pub median_mesh_rmse: f32,
    /// Median complete-roof silhouette IoU for all successfully fitted frames.
    pub median_silhouette_iou: f32,
    #[serde(skip)]
    mesh_errors: Vec<f32>,
    #[serde(skip)]
    silhouette_ious: Vec<f32>,
}

impl SyntheticFitMetrics {
    pub(super) fn record_attempt(&mut self) {
        self.attempted += 1;
    }

    pub(super) fn record_fit(&mut self, evaluation: SyntheticFitEvaluation) {
        self.fitted += 1;
        self.accepted += usize::from(evaluation.accepted);
        self.mesh_errors.push(evaluation.mesh_rmse);
        self.silhouette_ious.push(evaluation.silhouette_iou);
    }

    pub(super) fn finish(&mut self) {
        self.fit_success_rate = self.fitted as f32 / self.attempted.max(1) as f32;
        self.accepted_rate = self.accepted as f32 / self.attempted.max(1) as f32;
        self.median_mesh_rmse = median(&mut self.mesh_errors);
        self.median_silhouette_iou = median(&mut self.silhouette_ious);
    }
}

/// Fits decoded source-image keypoints and compares the complete inferred mesh
/// with the exact synthetic camera/roof record. Offscreen predictions are `None`.
pub(super) fn evaluate_synthetic_fit(
    predictions: &[Option<[f32; 2]>; KEYPOINT_COUNT],
    confidences: Option<&[f32; KEYPOINT_COUNT]>,
    frame: &FrameRecord,
    shape_prior: Option<[f32; 7]>,
) -> Result<SyntheticFitEvaluation> {
    let unit_confidences = [1.0; KEYPOINT_COUNT];
    let confidences = confidences.unwrap_or(&unit_confidences);
    let intrinsics = frame.camera.intrinsics;
    if intrinsics.width == 0 || intrinsics.height == 0 || intrinsics.fx <= 0.0 {
        bail!("synthetic frame has invalid camera intrinsics");
    }
    let observation = SingleViewObservation {
        keypoints: std::array::from_fn(|index| {
            predictions[index].map(|position| KeypointObservation {
                position,
                weight: confidences[index].clamp(0.01, 1.0),
            })
        }),
    };
    let config = SingleViewFitConfig {
        image_aspect_ratio: intrinsics.width as f32 / intrinsics.height as f32,
        principal_point: [
            intrinsics.cx / intrinsics.width as f32,
            intrinsics.cy / intrinsics.height as f32,
        ],
        // Match the common runtime path for photos without usable EXIF: focal
        // length is recovered from the 45/60/75 degree hypotheses. The exact
        // known-intrinsics path has its own stricter roof-fit oracle tests.
        focal_length: FocalLengthConfig::default(),
        shape_prior,
        maximum_reprojection_rmse: 0.05,
        ..SingleViewFitConfig::default()
    };
    let fit = fit_single_view(&observation, config).context("fit decoded synthetic keypoints")?;
    let exact = exact_projected_mesh(frame)?;
    let fitted = fit
        .projected_mesh
        .vertices
        .iter()
        .map(|vertex| vertex.position)
        .collect::<Vec<_>>();
    if fitted.iter().flatten().any(|value| !value.is_finite()) {
        bail!("fitter returned a non-finite projected mesh");
    }
    let fitted_geometry = generate_roof(&fit.parameters).context("regenerate fitted roof")?;
    let alignment = best_d4_alignment(
        &fit.projected_keypoints,
        &exact.keypoints,
        intrinsics.width,
        intrinsics.height,
    );
    let mesh_rmse = corresponding_vertex_rmse(
        &fitted,
        &fitted_geometry,
        &exact.keypoints,
        alignment,
        intrinsics.width,
        intrinsics.height,
    )?;
    let silhouette_iou = silhouette_iou(
        &fitted,
        &fit.projected_mesh.indices,
        &exact.vertices,
        &exact.indices,
        intrinsics.width,
        intrinsics.height,
    );
    Ok(SyntheticFitEvaluation {
        mesh_rmse,
        silhouette_iou,
        accepted: fit.confidence.accepted,
    })
}

struct ExactProjectedMesh {
    vertices: Vec<[f32; 2]>,
    keypoints: [[f32; 2]; KEYPOINT_COUNT],
    indices: Vec<u32>,
}

fn exact_projected_mesh(frame: &FrameRecord) -> Result<ExactProjectedMesh> {
    let roof = frame
        .roof
        .as_ref()
        .context("synthetic target has no roof instance")?;
    let parameter = |name: &str| {
        roof.parameters
            .get(name)
            .copied()
            .with_context(|| format!("synthetic roof is missing {name}"))
    };
    let parameters = RoofParameters::new(
        parameter("eave_width")?,
        parameter("eave_depth")?,
        parameter("shoulder_width")?,
        parameter("shoulder_depth")?,
        parameter("crown_top_width")?,
        parameter("crown_top_depth")?,
        parameter("lower_rise")?,
        parameter("upper_rise")?,
    )
    .context("invalid exact synthetic roof parameters")?;
    let geometry = generate_roof(&parameters).context("generate exact synthetic roof")?;
    let vertices = geometry
        .mesh
        .vertices
        .iter()
        .map(|vertex| project_exact(frame, roof.world_from_roof, vertex.position))
        .collect::<Result<Vec<_>>>()?;
    let mut keypoints = [[0.0; 2]; KEYPOINT_COUNT];
    for (index, keypoint) in geometry.keypoints.iter().enumerate() {
        keypoints[index] = project_exact(frame, roof.world_from_roof, keypoint.position)?;
    }
    Ok(ExactProjectedMesh {
        vertices,
        keypoints,
        indices: geometry.mesh.indices,
    })
}

fn project_exact(
    frame: &FrameRecord,
    world_from_roof: RigidTransform,
    point: [f32; 3],
) -> Result<[f32; 2]> {
    let world = transform_point(world_from_roof, point);
    let camera_origin = frame.camera.world_from_camera.translation;
    let relative = [
        world[0] - camera_origin.x,
        world[1] - camera_origin.y,
        world[2] - camera_origin.z,
    ];
    let camera = rotate_vector(
        conjugate(frame.camera.world_from_camera.rotation_xyzw),
        relative,
    );
    // The synthetic renderer is right-handed and looks down camera-local -Z.
    let depth = -camera[2];
    if !depth.is_finite() || depth <= 1.0e-5 {
        bail!("exact synthetic roof vertex is behind the camera");
    }
    let intrinsics = frame.camera.intrinsics;
    Ok([
        (intrinsics.fx * camera[0] / depth + intrinsics.skew * camera[1] / depth + intrinsics.cx)
            / intrinsics.width as f32,
        (intrinsics.cy - intrinsics.fy * camera[1] / depth) / intrinsics.height as f32,
    ])
}

fn transform_point(transform: RigidTransform, point: [f32; 3]) -> [f32; 3] {
    let rotated = rotate_vector(transform.rotation_xyzw, point);
    [
        rotated[0] + transform.translation.x,
        rotated[1] + transform.translation.y,
        rotated[2] + transform.translation.z,
    ]
}

fn conjugate([x, y, z, w]: [f32; 4]) -> [f32; 4] {
    [-x, -y, -z, w]
}

fn rotate_vector([qx, qy, qz, qw]: [f32; 4], vector: [f32; 3]) -> [f32; 3] {
    // q * v * conjugate(q), expanded to avoid a math dependency in the CLI.
    let q = [qx, qy, qz];
    let uv = cross(q, vector);
    let uuv = cross(q, uv);
    [
        vector[0] + 2.0 * (qw * uv[0] + uuv[0]),
        vector[1] + 2.0 * (qw * uv[1] + uuv[1]),
        vector[2] + 2.0 * (qw * uv[2] + uuv[2]),
    ]
}

fn cross(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn diagonal_distance(left: [f32; 2], right: [f32; 2], width: u32, height: u32) -> f32 {
    let dx = (left[0] - right[0]) * width as f32;
    let dy = (left[1] - right[1]) * height as f32;
    (dx * dx + dy * dy).sqrt() / (width as f32).hypot(height as f32)
}

fn corresponding_vertex_rmse(
    fitted_projection: &[[f32; 2]],
    fitted_geometry: &roof_geometry::RoofGeometry,
    exact_keypoints: &[[f32; 2]; KEYPOINT_COUNT],
    alignment: usize,
    width: u32,
    height: u32,
) -> Result<f32> {
    if fitted_projection.len() != fitted_geometry.mesh.vertices.len() {
        bail!("fitted projected and local mesh vertex counts differ");
    }
    let mut squared_error = 0.0;
    for (projected, vertex) in fitted_projection.iter().zip(&fitted_geometry.mesh.vertices) {
        let model_index = fitted_geometry
            .keypoints
            .iter()
            .position(|keypoint| {
                keypoint
                    .position
                    .iter()
                    .zip(vertex.position)
                    .all(|(left, right)| (*left - right).abs() < 1.0e-5)
            })
            .context("fitted mesh vertex is not a structural control point")?;
        let ring = model_index / 4;
        let target_index = ring * 4 + symmetry_target_slot(alignment, model_index % 4);
        squared_error +=
            diagonal_distance(*projected, exact_keypoints[target_index], width, height).powi(2);
    }
    Ok((squared_error / fitted_projection.len().max(1) as f32).sqrt())
}

fn best_d4_alignment(
    fitted: &[[f32; 2]; KEYPOINT_COUNT],
    exact: &[[f32; 2]; KEYPOINT_COUNT],
    width: u32,
    height: u32,
) -> usize {
    (0..SYMMETRY_COUNT)
        .min_by(|left, right| {
            let error = |hypothesis| {
                (0..KEYPOINT_COUNT)
                    .map(|index| {
                        let ring = index / 4;
                        let target = ring * 4 + symmetry_target_slot(hypothesis, index % 4);
                        diagonal_distance(fitted[index], exact[target], width, height).powi(2)
                    })
                    .sum::<f32>()
            };
            error(*left).total_cmp(&error(*right))
        })
        .unwrap_or(0)
}

fn silhouette_iou(
    left_vertices: &[[f32; 2]],
    left_indices: &[u32],
    right_vertices: &[[f32; 2]],
    right_indices: &[u32],
    source_width: u32,
    source_height: u32,
) -> f32 {
    let (width, height) = raster_dimensions(source_width, source_height);
    let left = rasterize_mesh(left_vertices, left_indices, width, height);
    let right = rasterize_mesh(right_vertices, right_indices, width, height);
    let mut intersection = 0usize;
    let mut union = 0usize;
    for (left, right) in left.into_iter().zip(right) {
        intersection += usize::from(left && right);
        union += usize::from(left || right);
    }
    if union == 0 {
        0.0
    } else {
        intersection as f32 / union as f32
    }
}

fn raster_dimensions(source_width: u32, source_height: u32) -> (usize, usize) {
    if source_width >= source_height {
        (
            SILHOUETTE_LONG_EDGE,
            ((SILHOUETTE_LONG_EDGE as f32 * source_height as f32 / source_width as f32).round()
                as usize)
                .max(1),
        )
    } else {
        (
            ((SILHOUETTE_LONG_EDGE as f32 * source_width as f32 / source_height as f32).round()
                as usize)
                .max(1),
            SILHOUETTE_LONG_EDGE,
        )
    }
}

fn rasterize_mesh(
    vertices: &[[f32; 2]],
    indices: &[u32],
    width: usize,
    height: usize,
) -> Vec<bool> {
    let mut mask = vec![false; width * height];
    for triangle in indices.chunks_exact(3) {
        let Some(points) = triangle
            .iter()
            .map(|index| vertices.get(*index as usize).copied())
            .collect::<Option<Vec<_>>>()
        else {
            continue;
        };
        let min_x = points
            .iter()
            .map(|point| point[0])
            .fold(f32::INFINITY, f32::min);
        let max_x = points
            .iter()
            .map(|point| point[0])
            .fold(f32::NEG_INFINITY, f32::max);
        let min_y = points
            .iter()
            .map(|point| point[1])
            .fold(f32::INFINITY, f32::min);
        let max_y = points
            .iter()
            .map(|point| point[1])
            .fold(f32::NEG_INFINITY, f32::max);
        let x0 = (min_x * width as f32)
            .floor()
            .clamp(0.0, width as f32 - 1.0) as usize;
        let x1 = (max_x * width as f32).ceil().clamp(0.0, width as f32 - 1.0) as usize;
        let y0 = (min_y * height as f32)
            .floor()
            .clamp(0.0, height as f32 - 1.0) as usize;
        let y1 = (max_y * height as f32)
            .ceil()
            .clamp(0.0, height as f32 - 1.0) as usize;
        for y in y0..=y1 {
            for x in x0..=x1 {
                let point = [
                    (x as f32 + 0.5) / width as f32,
                    (y as f32 + 0.5) / height as f32,
                ];
                if point_in_triangle(point, points[0], points[1], points[2]) {
                    mask[y * width + x] = true;
                }
            }
        }
    }
    mask
}

fn point_in_triangle(point: [f32; 2], a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> bool {
    let edge = |first: [f32; 2], second: [f32; 2]| {
        (point[0] - second[0]) * (first[1] - second[1])
            - (first[0] - second[0]) * (point[1] - second[1])
    };
    let d1 = edge(a, b);
    let d2 = edge(b, c);
    let d3 = edge(c, a);
    let has_negative = d1 < 0.0 || d2 < 0.0 || d3 < 0.0;
    let has_positive = d1 > 0.0 || d2 > 0.0 || d3 > 0.0;
    !(has_negative && has_positive)
}

fn median(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(f32::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use synth_data::{
        AssetRef, CameraIntrinsics, CameraModel, DatasetSplit, DistortionModel, FrameAssets,
        FrameIdentity, ImageTransform, LocatorLabel, RoofInstanceRecord, TargetKind, Vec3,
    };

    fn oracle_frame() -> FrameRecord {
        let parameters = RoofParameters {
            eave_width: 1.0,
            eave_depth: 0.72,
            shoulder_width: 0.61,
            shoulder_depth: 0.37,
            crown_top_width: 0.49,
            crown_top_depth: 0.27,
            lower_rise: 0.12,
            upper_rise: 0.15,
        };
        let width = 800;
        let height = 600;
        let horizontal_fov = 62.0_f32.to_radians();
        let fx = width as f32 / (2.0 * (horizontal_fov * 0.5).tan());
        let camera = CameraModel {
            intrinsics: CameraIntrinsics {
                width,
                height,
                fx,
                fy: fx,
                cx: width as f32 * 0.5,
                cy: height as f32 * 0.5,
                skew: 0.0,
            },
            distortion: DistortionModel::None,
            world_from_camera: RigidTransform::IDENTITY,
            output_from_sensor: ImageTransform::IDENTITY,
        };
        let mut frame = FrameRecord::new(
            FrameIdentity::new("oracle", "oracle", 0, 0),
            DatasetSplit::Test,
            camera,
            LocatorLabel {
                target_kind: TargetKind::Target,
                bounding_box: None,
                amodal_bounding_box: None,
                visible_fraction: 1.0,
                occluded_fraction: 0.0,
                truncated: false,
            },
            FrameAssets {
                rgb: AssetRef::new("oracle.jpg", "image/jpeg", "jpeg"),
                surface_normals: None,
                motion_vectors: None,
            },
        );
        frame.roof = Some(RoofInstanceRecord {
            family: "classic_two_tier".to_owned(),
            world_from_roof: RigidTransform {
                translation: Vec3::new(0.0, -0.05, -3.0),
                rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
            },
            parameters: BTreeMap::from([
                ("eave_width".to_owned(), parameters.eave_width),
                ("eave_depth".to_owned(), parameters.eave_depth),
                ("shoulder_width".to_owned(), parameters.shoulder_width),
                ("shoulder_depth".to_owned(), parameters.shoulder_depth),
                ("crown_top_width".to_owned(), parameters.crown_top_width),
                ("crown_top_depth".to_owned(), parameters.crown_top_depth),
                ("lower_rise".to_owned(), parameters.lower_rise),
                ("upper_rise".to_owned(), parameters.upper_rise),
            ]),
        });
        frame
    }

    #[test]
    fn oracle_annotations_pass_mesh_and_silhouette_acceptance() {
        let frame = oracle_frame();
        let roof = frame.roof.as_ref().unwrap();
        let parameters = RoofParameters::new(
            roof.parameters["eave_width"],
            roof.parameters["eave_depth"],
            roof.parameters["shoulder_width"],
            roof.parameters["shoulder_depth"],
            roof.parameters["crown_top_width"],
            roof.parameters["crown_top_depth"],
            roof.parameters["lower_rise"],
            roof.parameters["upper_rise"],
        )
        .unwrap();
        let geometry = generate_roof(&parameters).unwrap();
        let predictions = std::array::from_fn(|index| {
            Some(
                project_exact(
                    &frame,
                    roof.world_from_roof,
                    geometry.keypoints[index].position,
                )
                .unwrap(),
            )
        });
        let shape_prior = Some([
            parameters.eave_depth / parameters.eave_width,
            parameters.shoulder_width / parameters.eave_width,
            parameters.shoulder_depth / parameters.eave_depth,
            parameters.crown_top_width / parameters.shoulder_width,
            parameters.crown_top_depth / parameters.shoulder_depth,
            parameters.lower_rise / parameters.eave_width,
            parameters.upper_rise / parameters.eave_width,
        ]);
        let evaluation = evaluate_synthetic_fit(&predictions, None, &frame, shape_prior).unwrap();
        assert!(evaluation.mesh_rmse < 0.005, "{:?}", evaluation);
        assert!(evaluation.silhouette_iou > 0.95, "{:?}", evaluation);
        assert!(evaluation.accepted, "{:?}", evaluation);
    }

    #[test]
    fn aggregate_reports_medians_and_counts() {
        let mut metrics = SyntheticFitMetrics::default();
        metrics.record_attempt();
        metrics.record_fit(SyntheticFitEvaluation {
            mesh_rmse: 0.04,
            silhouette_iou: 0.7,
            accepted: false,
        });
        metrics.record_attempt();
        metrics.record_fit(SyntheticFitEvaluation {
            mesh_rmse: 0.02,
            silhouette_iou: 0.9,
            accepted: true,
        });
        metrics.finish();
        assert_eq!(metrics.attempted, 2);
        assert_eq!(metrics.fitted, 2);
        assert_eq!(metrics.accepted, 1);
        assert!((metrics.median_mesh_rmse - 0.03).abs() < 1.0e-6);
        assert!((metrics.median_silhouette_iou - 0.8).abs() < 1.0e-6);
    }
}
