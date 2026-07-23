//! Constraint-based perspective fitting of a complete parametric roof.
//!
//! A detector supplies possibly occluded or offscreen structural landmarks in
//! normalized image coordinates. This crate jointly estimates a pinhole camera
//! and the seven scale-free proportions of the classic two-tier roof. The
//! resulting mesh is complete even when only part of the roof was observed.

#![forbid(unsafe_code)]

mod solver;

use roof_geometry::{KeypointId, MeshFace, RoofParameters};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Stable number of structural landmarks in the roof geometry contract.
pub const KEYPOINT_COUNT: usize = KeypointId::ALL.len();

/// Number of fitted scale-free roof proportions.
pub const SHAPE_PARAMETER_COUNT: usize = 7;

/// Population prior used until a corpus-derived prior is supplied by training.
pub const DEFAULT_SHAPE_PRIOR: [f32; SHAPE_PARAMETER_COUNT] =
    [0.75, 0.60, 0.50, 0.83, 0.72, 0.117, 0.133];

/// One possibly occluded or offscreen 2D landmark supplied to the fitter.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeypointObservation {
    /// Normalized source-image position, with the origin at the top left.
    /// Values outside `[0, 1]` deliberately represent offscreen landmarks.
    pub position: [f32; 2],
    /// Relative confidence used by the robust objective.
    pub weight: f32,
}

impl KeypointObservation {
    /// Constructs an equally weighted observation.
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self {
            position: [x, y],
            weight: 1.0,
        }
    }
}

/// Detector evidence for one image in [`KeypointId::ALL`] order.
#[derive(Clone, Debug, PartialEq)]
pub struct SingleViewObservation {
    /// `None` denotes a hidden or rejected landmark. Missing landmarks are
    /// reconstructed from the fitted parametric roof.
    pub keypoints: [Option<KeypointObservation>; KEYPOINT_COUNT],
}

impl Default for SingleViewObservation {
    fn default() -> Self {
        Self {
            keypoints: [None; KEYPOINT_COUNT],
        }
    }
}

/// How horizontal focal length enters the single-view optimization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FocalLengthConfig {
    /// Estimate focal length from several deterministic initial hypotheses.
    Estimate {
        /// Horizontal field-of-view seeds in degrees.
        hypotheses_degrees: [f32; 3],
        /// Inclusive horizontal field-of-view limits in degrees.
        bounds_degrees: [f32; 2],
    },
    /// Use camera metadata as a normally distributed focal-length prior.
    ///
    /// A small uncertainty effectively fixes calibrated synthetic intrinsics;
    /// a larger value is appropriate for EXIF-derived estimates.
    Known {
        /// Horizontal field of view derived from calibration or EXIF metadata.
        horizontal_fov_degrees: f32,
        /// One-standard-deviation uncertainty of the metadata estimate.
        uncertainty_degrees: f32,
    },
}

impl Default for FocalLengthConfig {
    fn default() -> Self {
        Self::Estimate {
            hypotheses_degrees: [45.0, 60.0, 75.0],
            bounds_degrees: [30.0, 110.0],
        }
    }
}

/// Tunable robust-fit settings shared by native and WASM callers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SingleViewFitConfig {
    /// Minimum number of finite landmarks needed for a constrained fit.
    /// Sparse four- and five-point fits rely more heavily on the population
    /// shape prior and still need observations from multiple rings/corners.
    pub minimum_observations: usize,
    /// Normalized-image Huber transition for landmark residuals.
    pub huber_delta: f32,
    /// Strength of the population roof-proportion prior.
    pub shape_prior_weight: f32,
    /// Optional population prior in fitter parameter order: eave depth,
    /// shoulder width, shoulder depth/eave depth, crown width/shoulder width,
    /// crown depth/shoulder depth, lower rise, and upper rise.
    pub shape_prior: Option<[f32; SHAPE_PARAMETER_COUNT]>,
    /// Source image width divided by source image height.
    pub image_aspect_ratio: f32,
    /// Fixed normalized principal point. A centered lens is `[0.5, 0.5]`.
    pub principal_point: [f32; 2],
    /// Focal-length estimate or calibration prior.
    pub focal_length: FocalLengthConfig,
    /// LM evaluation multiplier for each continuous/discrete hypothesis.
    pub solver_patience: usize,
    /// Largest normalized-image RMSE accepted as a confident mesh fit.
    pub maximum_reprojection_rmse: f32,
    /// Smallest combined coverage/inlier/RMSE/plausibility score accepted.
    pub minimum_confidence_score: f32,
}

impl Default for SingleViewFitConfig {
    fn default() -> Self {
        Self {
            minimum_observations: 4,
            huber_delta: 0.03,
            shape_prior_weight: 0.01,
            shape_prior: None,
            image_aspect_ratio: 1.0,
            principal_point: [0.5, 0.5],
            focal_length: FocalLengthConfig::default(),
            solver_patience: 64,
            maximum_reprojection_rmse: 0.05,
            minimum_confidence_score: 0.25,
        }
    }
}

/// Pinhole camera recovered together with the relative roof shape.
///
/// View coordinates use positive X to the right, positive Y up, and positive Z
/// forward. Focal lengths and principal point are normalized by image width and
/// height, so projection directly produces normalized image coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerspectiveCamera {
    /// Normalized horizontal and vertical focal lengths (`fx / width`, `fy / height`).
    pub focal_length: [f32; 2],
    /// Normalized principal point (`cx / width`, `cy / height`).
    pub principal_point: [f32; 2],
    /// Roof-local origin expressed in camera coordinates after rotation.
    pub translation: [f32; 3],
    /// Roof rotation around its local up axis.
    pub yaw_radians: f32,
    /// Roof rotation around the camera X axis.
    pub pitch_radians: f32,
    /// In-plane rotation around the camera Z axis.
    pub roll_radians: f32,
    /// Discrete ring-corner rotation resolving single-view label ambiguity.
    pub corner_shift: u8,
    /// Whether detector-to-model corner correspondence is reflected.
    pub reflected: bool,
}

impl PerspectiveCamera {
    /// Transforms one roof-local point into positive-depth camera coordinates.
    #[must_use]
    pub fn view_point(self, point: [f32; 3]) -> [f32; 3] {
        let [x, y, z] = point;
        let (sin_yaw, cos_yaw) = self.yaw_radians.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch_radians.sin_cos();
        let (sin_roll, cos_roll) = self.roll_radians.sin_cos();

        let yaw_x = cos_yaw * x + sin_yaw * z;
        let yaw_z = -sin_yaw * x + cos_yaw * z;
        let pitch_y = cos_pitch * y - sin_pitch * yaw_z;
        let pitch_z = sin_pitch * y + cos_pitch * yaw_z;
        [
            cos_roll * yaw_x - sin_roll * pitch_y + self.translation[0],
            sin_roll * yaw_x + cos_roll * pitch_y + self.translation[1],
            pitch_z + self.translation[2],
        ]
    }

    /// Projects one roof-local point into normalized image coordinates.
    ///
    /// Invalid or behind-camera points produce NaNs for backward-compatible
    /// rendering code. New callers can use [`Self::try_project`] instead.
    #[must_use]
    pub fn project(self, point: [f32; 3]) -> [f32; 2] {
        self.try_project(point)
            .map_or([f32::NAN; 2], |projected| projected.position)
    }

    /// Projects one point and retains positive camera-space depth.
    #[must_use]
    pub fn try_project(self, point: [f32; 3]) -> Option<ProjectedPoint> {
        let view = self.view_point(point);
        if !view.iter().all(|value| value.is_finite()) || view[2] <= 1.0e-5 {
            return None;
        }
        let position = [
            self.principal_point[0] + self.focal_length[0] * view[0] / view[2],
            self.principal_point[1] - self.focal_length[1] * view[1] / view[2],
        ];
        position
            .iter()
            .all(|value| value.is_finite())
            .then_some(ProjectedPoint {
                position,
                depth: view[2],
            })
    }

    /// Horizontal field of view implied by the normalized focal length.
    #[must_use]
    pub fn horizontal_fov_degrees(self) -> f32 {
        (2.0 * (0.5 / self.focal_length[0]).atan()).to_degrees()
    }
}

/// A projected mesh point with its camera-space depth.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectedPoint {
    /// Normalized image coordinates, which may be outside the source frame.
    pub position: [f32; 2],
    /// Positive camera-space depth in eave-width units.
    pub depth: f32,
}

/// A complete indexed roof mesh projected into the source image.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProjectedMesh {
    /// Projected vertices in the same order as the generated roof mesh.
    pub vertices: Vec<ProjectedPoint>,
    /// Triangle indices copied from the generated roof mesh.
    pub indices: Vec<u32>,
    /// Stable semantic face ranges copied from the generated roof mesh.
    pub faces: Vec<MeshFace>,
}

/// Normalized bounds of the complete fitted mesh.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FittedBounds {
    /// Top-left extent; it may be outside `[0, 1]` for a truncated roof.
    pub min: [f32; 2],
    /// Bottom-right extent; it may be outside `[0, 1]` for a truncated roof.
    pub max: [f32; 2],
}

/// Confidence diagnostics used to decide whether an overlay should be drawn.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FitConfidence {
    /// Calibratable score in `[0, 1]` combining coverage, inliers, RMSE, and extrapolation.
    pub score: f32,
    /// Whether the result passes all observation, residual, and plausibility gates.
    pub accepted: bool,
    /// Number of landmarks within two Huber transitions of the final model.
    pub inlier_count: usize,
    /// Confidence-weighted fraction of observations classified as inliers.
    pub weighted_inlier_ratio: f32,
    /// Complete projected-mesh span divided by the observed landmark span.
    pub extrapolation_ratio: f32,
}

/// One complete single-view roof estimate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SingleViewRoofFit {
    /// Versioned serialized fit contract.
    pub schema_version: String,
    /// Scale-free roof parameters. `eave_width` is fixed to one relative unit.
    pub parameters: RoofParameters,
    /// Perspective camera estimated jointly with the shape.
    pub camera: PerspectiveCamera,
    /// All twelve generated landmarks, including landmarks absent from input.
    pub projected_keypoints: [[f32; 2]; KEYPOINT_COUNT],
    /// Full projected indexed mesh, including occluded and offscreen vertices.
    pub projected_mesh: ProjectedMesh,
    /// Bounds derived from the generated complete mesh.
    pub bounding_box: FittedBounds,
    /// Weighted landmark reprojection RMSE in normalized image coordinates.
    pub reprojection_rmse: f32,
    /// Number of detector observations accepted by the fit.
    pub observation_count: usize,
    /// Overlay acceptance diagnostics.
    pub confidence: FitConfidence,
}

/// An invalid, insufficient, or unsolved single-view observation.
#[derive(Clone, Debug, Error, PartialEq)]
pub enum FitError {
    /// Fewer than the configured number of finite, positive-weight points were supplied.
    #[error("roof fit needs at least {minimum} usable keypoints, received {actual}")]
    InsufficientObservations {
        /// Required observation count.
        minimum: usize,
        /// Supplied usable observation count.
        actual: usize,
    },
    /// Sparse landmarks do not cover enough roof tiers to constrain a full mesh.
    #[error(
        "sparse roof fit needs observations from all three rings and at least two corner slots; got {ring_count} rings and {corner_count} slots"
    )]
    DegenerateObservations {
        /// Number of eave/shoulder/crown rings represented.
        ring_count: usize,
        /// Number of cyclic corner slots represented.
        corner_count: usize,
    },
    /// A caller supplied an invalid fitting configuration.
    #[error("invalid roof fit configuration: {0}")]
    InvalidConfiguration(&'static str),
    /// No discrete/continuous hypothesis produced a positive-depth roof.
    #[error("roof fit could not find a finite positive-depth perspective solution")]
    NoValidSolution,
}

/// Fits a scale-free two-tier roof and perspective camera to one detector frame.
///
/// The solver evaluates all eight cyclic/reflected corner correspondences and
/// minimizes a Huber-weighted nonlinear least-squares objective. Focal length
/// is either estimated from the configured hypotheses or softly constrained by
/// camera metadata. The returned mesh always includes hidden roof geometry.
pub fn fit_single_view(
    observation: &SingleViewObservation,
    config: SingleViewFitConfig,
) -> Result<SingleViewRoofFit, FitError> {
    solver::fit_single_view(observation, config)
}
