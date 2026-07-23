use core::f64::consts::PI;

use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{Dyn, OMatrix, OVector, U14, storage::Owned};
use roof_geometry::{RoofParameters, generate_roof};

use crate::{
    DEFAULT_SHAPE_PRIOR, FitConfidence, FitError, FittedBounds, FocalLengthConfig, KEYPOINT_COUNT,
    KeypointObservation, PerspectiveCamera, ProjectedMesh, SHAPE_PARAMETER_COUNT,
    SingleViewFitConfig, SingleViewObservation, SingleViewRoofFit,
};

const PARAMETER_COUNT: usize = 14;
const SHAPE_BOUNDS: [(f64, f64); SHAPE_PARAMETER_COUNT] = [
    (0.40, 1.25),
    (0.38, 0.82),
    (0.30, 0.82),
    (0.55, 0.96),
    (0.50, 0.96),
    (0.04, 0.24),
    (0.04, 0.30),
];
const PITCH_BOUNDS: (f64, f64) = (-1.35, 0.35);
const ROLL_BOUNDS: (f64, f64) = (-0.65, 0.65);
const CENTER_BOUNDS: (f64, f64) = (-1.0, 2.0);
const DEPTH_BOUNDS: (f64, f64) = (0.25, 100.0);
const YAW_SEEDS: [f64; 3] = [-0.65, 0.0, 0.65];
const PITCH_SEEDS: [f64; 3] = [-0.75, -0.38, -0.05];
const SPARSE_OBSERVATION_LIMIT: usize = 6;
const MINIMUM_RELATIVE_VERTEX_DEPTH: f64 = 0.12;
const MAXIMUM_CONFIDENT_EXTRAPOLATION: f32 = 3.5;

pub(super) fn fit_single_view(
    observation: &SingleViewObservation,
    config: SingleViewFitConfig,
) -> Result<SingleViewRoofFit, FitError> {
    validate_config(config)?;
    let observation_count = observation
        .keypoints
        .iter()
        .filter(|point| is_usable(**point))
        .count();
    if observation_count < config.minimum_observations {
        return Err(FitError::InsufficientObservations {
            minimum: config.minimum_observations,
            actual: observation_count,
        });
    }
    if observation_count < SPARSE_OBSERVATION_LIMIT {
        let (ring_count, corner_count) = observation_coverage(observation);
        if ring_count < 3 || corner_count < 2 {
            return Err(FitError::DegenerateObservations {
                ring_count,
                corner_count,
            });
        }
    }

    let (center, span) = observation_extent(observation);
    let shape_prior = config.shape_prior.unwrap_or(DEFAULT_SHAPE_PRIOR);
    let optimization_config = SingleViewFitConfig {
        shape_prior_weight: if observation_count < SPARSE_OBSERVATION_LIMIT {
            config.shape_prior_weight.max(0.05)
        } else {
            config.shape_prior_weight
        },
        ..config
    };
    let fov_seeds = focal_hypotheses(config.focal_length);
    let fov_bounds = focal_bounds(config.focal_length);
    let solver = LevenbergMarquardt::new()
        .with_patience(config.solver_patience)
        .with_tol(1.0e-10)
        .with_stepbound(20.0);

    let mut best: Option<Candidate> = None;
    for reflected in [false, true] {
        for corner_shift in 0_u8..4 {
            for fov_degrees in &fov_seeds {
                let focal_x = 0.5 / (0.5 * fov_degrees.to_radians()).tan();
                let initial_depth = (focal_x / span as f64).clamp(0.5, 30.0);
                for yaw in YAW_SEEDS {
                    for pitch in PITCH_SEEDS {
                        let problem = FitProblem::new(
                            observation,
                            optimization_config,
                            corner_shift,
                            reflected,
                            shape_prior,
                            fov_bounds,
                            center,
                            initial_depth,
                            *fov_degrees,
                            yaw,
                            pitch,
                        );
                        let (problem, _report) = solver.minimize(problem);
                        let Some(candidate) = Candidate::from_problem(problem) else {
                            continue;
                        };
                        if best
                            .as_ref()
                            .is_none_or(|current| candidate.loss < current.loss)
                        {
                            best = Some(candidate);
                        }
                    }
                }
            }
        }
    }

    best.ok_or(FitError::NoValidSolution)?
        .finish(observation, config)
}

fn validate_config(config: SingleViewFitConfig) -> Result<(), FitError> {
    if !(4..=KEYPOINT_COUNT).contains(&config.minimum_observations) {
        return Err(FitError::InvalidConfiguration(
            "minimum_observations must be between 4 and 12",
        ));
    }
    if !config.huber_delta.is_finite() || config.huber_delta <= 0.0 {
        return Err(FitError::InvalidConfiguration(
            "huber_delta must be finite and positive",
        ));
    }
    if !config.shape_prior_weight.is_finite() || config.shape_prior_weight < 0.0 {
        return Err(FitError::InvalidConfiguration(
            "shape_prior_weight must be finite and nonnegative",
        ));
    }
    if !config.image_aspect_ratio.is_finite() || config.image_aspect_ratio <= 0.0 {
        return Err(FitError::InvalidConfiguration(
            "image_aspect_ratio must be finite and positive",
        ));
    }
    if !config
        .principal_point
        .iter()
        .all(|coordinate| coordinate.is_finite())
    {
        return Err(FitError::InvalidConfiguration(
            "principal_point must be finite",
        ));
    }
    if config.solver_patience == 0 {
        return Err(FitError::InvalidConfiguration(
            "solver_patience must be nonzero",
        ));
    }
    if !config.maximum_reprojection_rmse.is_finite() || config.maximum_reprojection_rmse <= 0.0 {
        return Err(FitError::InvalidConfiguration(
            "maximum_reprojection_rmse must be finite and positive",
        ));
    }
    if !config.minimum_confidence_score.is_finite()
        || !(0.0..=1.0).contains(&config.minimum_confidence_score)
    {
        return Err(FitError::InvalidConfiguration(
            "minimum_confidence_score must lie between zero and one",
        ));
    }
    if let Some(shape_prior) = config.shape_prior
        && !shape_prior.iter().enumerate().all(|(index, value)| {
            value.is_finite()
                && f64::from(*value) >= SHAPE_BOUNDS[index].0
                && f64::from(*value) <= SHAPE_BOUNDS[index].1
        })
    {
        return Err(FitError::InvalidConfiguration(
            "shape_prior must stay within the fitter's seven shape bounds",
        ));
    }
    match config.focal_length {
        FocalLengthConfig::Estimate {
            hypotheses_degrees,
            bounds_degrees,
        } => {
            if !valid_fov_bounds(bounds_degrees)
                || !hypotheses_degrees.iter().all(|fov| {
                    fov.is_finite() && *fov >= bounds_degrees[0] && *fov <= bounds_degrees[1]
                })
            {
                return Err(FitError::InvalidConfiguration(
                    "estimated FOV hypotheses must be finite and inside valid bounds",
                ));
            }
        }
        FocalLengthConfig::Known {
            horizontal_fov_degrees,
            uncertainty_degrees,
        } => {
            if !horizontal_fov_degrees.is_finite()
                || !(5.0..175.0).contains(&horizontal_fov_degrees)
                || !uncertainty_degrees.is_finite()
                || uncertainty_degrees <= 0.0
            {
                return Err(FitError::InvalidConfiguration(
                    "known FOV and its uncertainty must be finite and physically valid",
                ));
            }
        }
    }
    Ok(())
}

fn valid_fov_bounds(bounds: [f32; 2]) -> bool {
    bounds.iter().all(|value| value.is_finite())
        && bounds[0] >= 5.0
        && bounds[1] <= 175.0
        && bounds[0] < bounds[1]
}

fn focal_hypotheses(config: FocalLengthConfig) -> Vec<f64> {
    match config {
        FocalLengthConfig::Estimate {
            hypotheses_degrees, ..
        } => hypotheses_degrees.map(f64::from).to_vec(),
        FocalLengthConfig::Known {
            horizontal_fov_degrees,
            ..
        } => vec![f64::from(horizontal_fov_degrees)],
    }
}

fn focal_bounds(config: FocalLengthConfig) -> [f64; 2] {
    match config {
        FocalLengthConfig::Estimate { bounds_degrees, .. } => bounds_degrees.map(f64::from),
        FocalLengthConfig::Known {
            horizontal_fov_degrees,
            uncertainty_degrees,
        } => {
            let radius = (8.0 * uncertainty_degrees).max(1.0);
            [
                f64::from((horizontal_fov_degrees - radius).max(5.0)),
                f64::from((horizontal_fov_degrees + radius).min(175.0)),
            ]
        }
    }
}

fn is_usable(observation: Option<KeypointObservation>) -> bool {
    observation.is_some_and(|point| {
        point.position[0].is_finite()
            && point.position[1].is_finite()
            && point.weight.is_finite()
            && point.weight > 0.0
    })
}

fn observation_extent(observation: &SingleViewObservation) -> ([f32; 2], f32) {
    let mut min = [f32::INFINITY; 2];
    let mut max = [f32::NEG_INFINITY; 2];
    for point in observation.keypoints.iter().flatten().copied() {
        if !is_usable(Some(point)) {
            continue;
        }
        for axis in 0..2 {
            min[axis] = min[axis].min(point.position[axis]);
            max[axis] = max[axis].max(point.position[axis]);
        }
    }
    let center = [(min[0] + max[0]) * 0.5, (min[1] + max[1]) * 0.5];
    let span = (max[0] - min[0]).max(max[1] - min[1]).clamp(0.02, 2.5);
    (center, span)
}

fn observation_coverage(observation: &SingleViewObservation) -> (usize, usize) {
    let mut rings = [false; 3];
    let mut corners = [false; 4];
    for (index, point) in observation.keypoints.iter().enumerate() {
        if is_usable(*point) {
            rings[index / 4] = true;
            corners[index % 4] = true;
        }
    }
    (
        rings.into_iter().filter(|present| *present).count(),
        corners.into_iter().filter(|present| *present).count(),
    )
}

#[derive(Clone)]
struct FitProblem {
    params: OVector<f64, U14>,
    observation: SingleViewObservation,
    config: SingleViewFitConfig,
    corner_shift: u8,
    reflected: bool,
    shape_prior: [f64; SHAPE_PARAMETER_COUNT],
    fov_bounds: [f64; 2],
    fov_prior_degrees: f64,
}

#[allow(clippy::too_many_arguments)]
impl FitProblem {
    fn new(
        observation: &SingleViewObservation,
        config: SingleViewFitConfig,
        corner_shift: u8,
        reflected: bool,
        shape_prior: [f32; SHAPE_PARAMETER_COUNT],
        fov_bounds: [f64; 2],
        center: [f32; 2],
        depth: f64,
        fov_degrees: f64,
        yaw: f64,
        pitch: f64,
    ) -> Self {
        let mut params = OVector::<f64, U14>::zeros();
        for index in 0..SHAPE_PARAMETER_COUNT {
            params[index] = encode_bounded(f64::from(shape_prior[index]), SHAPE_BOUNDS[index]);
        }
        params[7] = yaw;
        params[8] = encode_bounded(pitch, PITCH_BOUNDS);
        params[9] = encode_bounded(0.0, ROLL_BOUNDS);
        params[10] = encode_bounded(f64::from(center[0]), CENTER_BOUNDS);
        params[11] = encode_bounded(f64::from(center[1]), CENTER_BOUNDS);
        params[12] = encode_bounded(depth, DEPTH_BOUNDS);
        params[13] = encode_bounded(fov_degrees, (fov_bounds[0], fov_bounds[1]));
        Self {
            params,
            observation: observation.clone(),
            config,
            corner_shift,
            reflected,
            shape_prior: shape_prior.map(f64::from),
            fov_bounds,
            fov_prior_degrees: fov_degrees,
        }
    }

    fn decoded_at(&self, params: &OVector<f64, U14>) -> DecodedFit {
        let shape =
            core::array::from_fn(|index| decode_bounded(params[index], SHAPE_BOUNDS[index]));
        let parameters = parameters_from_shape(shape);
        let yaw = wrap_angle(params[7]);
        let pitch = decode_bounded(params[8], PITCH_BOUNDS);
        let roll = decode_bounded(params[9], ROLL_BOUNDS);
        let center = [
            decode_bounded(params[10], CENTER_BOUNDS),
            decode_bounded(params[11], CENTER_BOUNDS),
        ];
        let depth = decode_bounded(params[12], DEPTH_BOUNDS);
        let horizontal_fov_degrees =
            decode_bounded(params[13], (self.fov_bounds[0], self.fov_bounds[1]));
        let focal_x = 0.5 / (0.5 * horizontal_fov_degrees.to_radians()).tan();
        let focal_y = focal_x * f64::from(self.config.image_aspect_ratio);
        let principal_point = self.config.principal_point.map(f64::from);
        let translation = [
            (center[0] - principal_point[0]) * depth / focal_x,
            (principal_point[1] - center[1]) * depth / focal_y,
            depth,
        ];
        DecodedFit {
            shape,
            horizontal_fov_degrees,
            parameters,
            camera: PerspectiveCamera {
                focal_length: [focal_x as f32, focal_y as f32],
                principal_point: self.config.principal_point,
                translation: translation.map(|value| value as f32),
                yaw_radians: yaw as f32,
                pitch_radians: pitch as f32,
                roll_radians: roll as f32,
                corner_shift: self.corner_shift,
                reflected: self.reflected,
            },
        }
    }

    fn residuals_at(&self, params: &OVector<f64, U14>) -> OVector<f64, Dyn> {
        let decoded = self.decoded_at(params);
        let roof = generate_roof(&decoded.parameters)
            .expect("bounded scale-free parameters must generate a valid roof");
        let usable_count = self
            .observation
            .keypoints
            .iter()
            .filter(|point| is_usable(**point))
            .count();
        let residual_count = usable_count * 2 + KEYPOINT_COUNT + SHAPE_PARAMETER_COUNT + 1;
        let mut residuals = Vec::with_capacity(residual_count);
        let huber_delta = f64::from(self.config.huber_delta);

        for (model_index, keypoint) in roof.keypoints.iter().enumerate() {
            let observed_index = correspondence(model_index, self.corner_shift, self.reflected);
            let Some(observed) = self.observation.keypoints[observed_index] else {
                continue;
            };
            if !is_usable(Some(observed)) {
                continue;
            }
            let projected = decoded.camera.try_project(keypoint.position);
            let (dx, dy) = projected.map_or((2.0, 2.0), |point| {
                (
                    f64::from(point.position[0] - observed.position[0]),
                    f64::from(point.position[1] - observed.position[1]),
                )
            });
            let distance = (dx * dx + dy * dy).sqrt();
            let huber_weight = if distance <= huber_delta || distance == 0.0 {
                1.0
            } else {
                huber_delta / distance
            };
            let weight = (f64::from(observed.weight) * huber_weight).sqrt();
            residuals.push(weight * dx);
            residuals.push(weight * dy);
        }

        let minimum_depth =
            (MINIMUM_RELATIVE_VERTEX_DEPTH * f64::from(decoded.camera.translation[2])).max(0.02);
        for keypoint in &roof.keypoints {
            let depth = f64::from(decoded.camera.view_point(keypoint.position)[2]);
            residuals.push((minimum_depth - depth).max(0.0) * 10.0);
        }
        let shape_weight = f64::from(self.config.shape_prior_weight).sqrt();
        for index in 0..SHAPE_PARAMETER_COUNT {
            residuals.push(shape_weight * (decoded.shape[index] - self.shape_prior[index]));
        }
        let focal_residual = match self.config.focal_length {
            FocalLengthConfig::Estimate { .. } => {
                0.04 * (decoded.horizontal_fov_degrees - self.fov_prior_degrees) / 15.0
            }
            FocalLengthConfig::Known {
                horizontal_fov_degrees,
                uncertainty_degrees,
            } => {
                0.02 * (decoded.horizontal_fov_degrees - f64::from(horizontal_fov_degrees))
                    / f64::from(uncertainty_degrees)
            }
        };
        residuals.push(focal_residual);

        debug_assert_eq!(residuals.len(), residual_count);
        OVector::<f64, Dyn>::from_vec(residuals)
    }
}

impl LeastSquaresProblem<f64, Dyn, U14> for FitProblem {
    type ParameterStorage = Owned<f64, U14>;
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, U14>;

    fn set_params(&mut self, params: &OVector<f64, U14>) {
        self.params.copy_from(params);
    }

    fn params(&self) -> OVector<f64, U14> {
        self.params
    }

    fn residuals(&self) -> Option<OVector<f64, Dyn>> {
        Some(self.residuals_at(&self.params))
    }

    fn jacobian(&self) -> Option<OMatrix<f64, Dyn, U14>> {
        let residual_count = self.residuals_at(&self.params).len();
        let mut jacobian = OMatrix::<f64, Dyn, U14>::zeros_generic(Dyn(residual_count), U14);
        for parameter_index in 0..PARAMETER_COUNT {
            let epsilon = 1.0e-5 * (1.0 + self.params[parameter_index].abs());
            let mut plus = self.params;
            let mut minus = self.params;
            plus[parameter_index] += epsilon;
            minus[parameter_index] -= epsilon;
            let plus_residuals = self.residuals_at(&plus);
            let minus_residuals = self.residuals_at(&minus);
            for residual_index in 0..residual_count {
                jacobian[(residual_index, parameter_index)] = (plus_residuals[residual_index]
                    - minus_residuals[residual_index])
                    / (2.0 * epsilon);
            }
        }
        Some(jacobian)
    }
}

struct DecodedFit {
    shape: [f64; SHAPE_PARAMETER_COUNT],
    horizontal_fov_degrees: f64,
    parameters: RoofParameters,
    camera: PerspectiveCamera,
}

struct Candidate {
    loss: f64,
    decoded: DecodedFit,
}

impl Candidate {
    fn from_problem(problem: FitProblem) -> Option<Self> {
        let decoded = problem.decoded_at(&problem.params);
        let roof = generate_roof(&decoded.parameters).ok()?;
        if roof
            .mesh
            .vertices
            .iter()
            .any(|vertex| decoded.camera.try_project(vertex.position).is_none())
        {
            return None;
        }
        let residuals = problem.residuals_at(&problem.params);
        let loss = residuals.dot(&residuals);
        loss.is_finite().then_some(Self { loss, decoded })
    }

    fn finish(
        self,
        observation: &SingleViewObservation,
        config: SingleViewFitConfig,
    ) -> Result<SingleViewRoofFit, FitError> {
        let parameters = self.decoded.parameters;
        let camera = self.decoded.camera;
        let roof = generate_roof(&parameters).map_err(|_| FitError::NoValidSolution)?;
        let mut projected_keypoints = [[0.0; 2]; KEYPOINT_COUNT];
        for (index, keypoint) in roof.keypoints.iter().enumerate() {
            projected_keypoints[index] = camera
                .try_project(keypoint.position)
                .ok_or(FitError::NoValidSolution)?
                .position;
        }
        let vertices = roof
            .mesh
            .vertices
            .iter()
            .map(|vertex| {
                camera
                    .try_project(vertex.position)
                    .ok_or(FitError::NoValidSolution)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut min = [f32::INFINITY; 2];
        let mut max = [f32::NEG_INFINITY; 2];
        for vertex in &vertices {
            for axis in 0..2 {
                min[axis] = min[axis].min(vertex.position[axis]);
                max[axis] = max[axis].max(vertex.position[axis]);
            }
        }

        let mut squared_error = 0.0_f32;
        let mut weight_sum = 0.0_f32;
        let mut inlier_weight = 0.0_f32;
        let mut observation_count = 0;
        let mut inlier_count = 0;
        for (model_index, projected) in projected_keypoints.iter().enumerate() {
            let observed_index = correspondence(model_index, camera.corner_shift, camera.reflected);
            let Some(observed) = observation.keypoints[observed_index] else {
                continue;
            };
            if !is_usable(Some(observed)) {
                continue;
            }
            let dx = projected[0] - observed.position[0];
            let dy = projected[1] - observed.position[1];
            let distance_squared = dx * dx + dy * dy;
            squared_error += observed.weight * distance_squared;
            weight_sum += observed.weight;
            observation_count += 1;
            if distance_squared.sqrt() <= 2.0 * config.huber_delta {
                inlier_count += 1;
                inlier_weight += observed.weight;
            }
        }
        let reprojection_rmse = (squared_error / weight_sum.max(f32::EPSILON)).sqrt();
        let inlier_ratio = inlier_weight / weight_sum.max(f32::EPSILON);
        let coverage = observation_count as f32 / KEYPOINT_COUNT as f32;
        let (_, observed_span) = observation_extent(observation);
        let fitted_span = (max[0] - min[0]).max(max[1] - min[1]);
        let extrapolation_ratio = fitted_span / observed_span.max(0.02);
        let extrapolation_score = if extrapolation_ratio <= 1.75 {
            1.0
        } else {
            ((MAXIMUM_CONFIDENT_EXTRAPOLATION - extrapolation_ratio)
                / (MAXIMUM_CONFIDENT_EXTRAPOLATION - 1.75))
                .clamp(0.0, 1.0)
        };
        let rmse_score =
            (1.0 - reprojection_rmse / config.maximum_reprojection_rmse).clamp(0.0, 1.0);
        let score = (rmse_score * inlier_ratio * extrapolation_score * (0.5 + 0.5 * coverage))
            .clamp(0.0, 1.0);
        let accepted = observation_count >= config.minimum_observations
            && reprojection_rmse <= config.maximum_reprojection_rmse
            && inlier_ratio >= 2.0 / 3.0
            && extrapolation_ratio <= MAXIMUM_CONFIDENT_EXTRAPOLATION
            && score >= config.minimum_confidence_score;

        Ok(SingleViewRoofFit {
            schema_version: "single-view-roof-fit/v3".to_owned(),
            parameters,
            camera,
            projected_keypoints,
            projected_mesh: ProjectedMesh {
                vertices,
                indices: roof.mesh.indices,
                faces: roof.mesh.faces,
            },
            bounding_box: FittedBounds { min, max },
            reprojection_rmse,
            observation_count,
            confidence: FitConfidence {
                score,
                accepted,
                inlier_count,
                weighted_inlier_ratio: inlier_ratio,
                extrapolation_ratio,
            },
        })
    }
}

fn parameters_from_shape(shape: [f64; SHAPE_PARAMETER_COUNT]) -> RoofParameters {
    let eave_depth = shape[0] as f32;
    let shoulder_width = shape[1] as f32;
    let shoulder_depth = eave_depth * shape[2] as f32;
    RoofParameters {
        eave_width: 1.0,
        eave_depth,
        shoulder_width,
        shoulder_depth,
        crown_top_width: shoulder_width * shape[3] as f32,
        crown_top_depth: shoulder_depth * shape[4] as f32,
        lower_rise: shape[5] as f32,
        upper_rise: shape[6] as f32,
    }
}

fn correspondence(model_index: usize, corner_shift: u8, reflected: bool) -> usize {
    let ring = model_index / 4;
    let corner = model_index % 4;
    let oriented = if reflected {
        (4 - corner + usize::from(corner_shift)) % 4
    } else {
        (corner + usize::from(corner_shift)) % 4
    };
    ring * 4 + oriented
}

fn decode_bounded(value: f64, bounds: (f64, f64)) -> f64 {
    let sigmoid = if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    };
    bounds.0 + (bounds.1 - bounds.0) * sigmoid
}

fn encode_bounded(value: f64, bounds: (f64, f64)) -> f64 {
    let fraction = ((value - bounds.0) / (bounds.1 - bounds.0)).clamp(1.0e-6, 1.0 - 1.0e-6);
    (fraction / (1.0 - fraction)).ln()
}

fn wrap_angle(angle: f64) -> f64 {
    (angle + PI).rem_euclid(2.0 * PI) - PI
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FocalLengthConfig, KeypointObservation};

    const WIDTH: u32 = 800;
    const HEIGHT: u32 = 600;
    const HORIZONTAL_FOV_DEGREES: f32 = 62.0;

    fn oracle_parameters() -> RoofParameters {
        RoofParameters {
            eave_width: 1.0,
            eave_depth: 0.72,
            shoulder_width: 0.61,
            shoulder_depth: 0.37,
            crown_top_width: 0.49,
            crown_top_depth: 0.27,
            lower_rise: 0.12,
            upper_rise: 0.15,
        }
    }

    fn shape_ratios(parameters: RoofParameters) -> [f32; SHAPE_PARAMETER_COUNT] {
        [
            parameters.eave_depth,
            parameters.shoulder_width,
            parameters.shoulder_depth / parameters.eave_depth,
            parameters.crown_top_width / parameters.shoulder_width,
            parameters.crown_top_depth / parameters.shoulder_depth,
            parameters.lower_rise,
            parameters.upper_rise,
        ]
    }

    #[derive(Clone, Copy)]
    struct OracleCamera {
        position: [f32; 3],
        target: [f32; 3],
        horizontal_fov_degrees: f32,
        width: u32,
        height: u32,
    }

    impl OracleCamera {
        fn project(self, point: [f32; 3]) -> [f32; 2] {
            // Independent look-at pinhole projection matching the synthetic
            // renderer's world-space camera convention.
            let forward = normalize(subtract(self.target, self.position));
            let right = normalize(cross(forward, [0.0, 1.0, 0.0]));
            let up = cross(right, forward);
            let relative = subtract(point, self.position);
            let depth = dot(relative, forward);
            assert!(depth > 0.0);
            let focal_x = 0.5 / (0.5 * self.horizontal_fov_degrees.to_radians()).tan();
            let focal_y = focal_x * self.width as f32 / self.height as f32;
            [
                0.5 + focal_x * dot(relative, right) / depth,
                0.5 - focal_y * dot(relative, up) / depth,
            ]
        }
    }

    fn oracle_camera() -> OracleCamera {
        OracleCamera {
            position: [1.25, 0.95, -2.35],
            target: [0.04, 0.10, 0.02],
            horizontal_fov_degrees: HORIZONTAL_FOV_DEGREES,
            width: WIDTH,
            height: HEIGHT,
        }
    }

    fn subtract(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
        [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
    }

    fn dot(left: [f32; 3], right: [f32; 3]) -> f32 {
        left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
    }

    fn cross(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
        [
            left[1] * right[2] - left[2] * right[1],
            left[2] * right[0] - left[0] * right[2],
            left[0] * right[1] - left[1] * right[0],
        ]
    }

    fn normalize(vector: [f32; 3]) -> [f32; 3] {
        let length = dot(vector, vector).sqrt();
        vector.map(|component| component / length)
    }

    fn oracle_config(parameters: RoofParameters) -> SingleViewFitConfig {
        SingleViewFitConfig {
            image_aspect_ratio: WIDTH as f32 / HEIGHT as f32,
            focal_length: FocalLengthConfig::Known {
                horizontal_fov_degrees: HORIZONTAL_FOV_DEGREES,
                uncertainty_degrees: 0.05,
            },
            shape_prior: Some(shape_ratios(parameters)),
            shape_prior_weight: 0.0001,
            solver_patience: 80,
            ..SingleViewFitConfig::default()
        }
    }

    fn synthetic_observation(
        hidden: &[usize],
        corner_shift: u8,
        reflected: bool,
    ) -> SingleViewObservation {
        let parameters = oracle_parameters();
        let camera = oracle_camera();
        synthetic_observation_for(parameters, camera, hidden, corner_shift, reflected)
    }

    fn synthetic_observation_for(
        parameters: RoofParameters,
        camera: OracleCamera,
        hidden: &[usize],
        corner_shift: u8,
        reflected: bool,
    ) -> SingleViewObservation {
        let roof = generate_roof(&parameters).unwrap();
        let mut observation = SingleViewObservation::default();
        for (model_index, keypoint) in roof.keypoints.iter().enumerate() {
            if hidden.contains(&model_index) {
                continue;
            }
            let observed_index = correspondence(model_index, corner_shift, reflected);
            let point = camera.project(keypoint.position);
            observation.keypoints[observed_index] =
                Some(KeypointObservation::new(point[0], point[1]));
        }
        observation
    }

    fn fixed_prior_config(camera: OracleCamera) -> SingleViewFitConfig {
        SingleViewFitConfig {
            image_aspect_ratio: camera.width as f32 / camera.height as f32,
            focal_length: FocalLengthConfig::Known {
                horizontal_fov_degrees: camera.horizontal_fov_degrees,
                uncertainty_degrees: 0.05,
            },
            shape_prior: Some(DEFAULT_SHAPE_PRIOR),
            shape_prior_weight: 0.0001,
            solver_patience: 100,
            ..SingleViewFitConfig::default()
        }
    }

    #[test]
    fn exact_annotation_sweep_recovers_varied_roofs_from_one_population_prior() {
        let cases = [
            (
                "wide_deep_high",
                RoofParameters {
                    eave_width: 1.0,
                    eave_depth: 1.08,
                    shoulder_width: 0.78,
                    shoulder_depth: 0.75,
                    crown_top_width: 0.70,
                    crown_top_depth: 0.64,
                    lower_rise: 0.20,
                    upper_rise: 0.26,
                },
                OracleCamera {
                    position: [1.55, 1.18, -2.85],
                    target: [0.02, 0.13, -0.04],
                    horizontal_fov_degrees: 48.0,
                    width: 1280,
                    height: 720,
                },
            ),
            (
                "compact_shallow_low",
                RoofParameters {
                    eave_width: 1.0,
                    eave_depth: 0.46,
                    shoulder_width: 0.42,
                    shoulder_depth: 0.15,
                    crown_top_width: 0.25,
                    crown_top_depth: 0.08,
                    lower_rise: 0.055,
                    upper_rise: 0.065,
                },
                OracleCamera {
                    position: [-1.65, 0.62, -3.15],
                    target: [-0.05, 0.08, 0.03],
                    horizontal_fov_degrees: 74.0,
                    width: 900,
                    height: 900,
                },
            ),
            (
                "long_eave_steep_crown",
                RoofParameters {
                    eave_width: 1.0,
                    eave_depth: 1.18,
                    shoulder_width: 0.55,
                    shoulder_depth: 0.62,
                    crown_top_width: 0.44,
                    crown_top_depth: 0.36,
                    lower_rise: 0.09,
                    upper_rise: 0.28,
                },
                OracleCamera {
                    position: [0.45, 1.45, -2.55],
                    target: [0.08, 0.16, 0.05],
                    horizontal_fov_degrees: 58.0,
                    width: 1024,
                    height: 768,
                },
            ),
            (
                "broad_shoulder_narrow_crown",
                RoofParameters {
                    eave_width: 1.0,
                    eave_depth: 0.66,
                    shoulder_width: 0.80,
                    shoulder_depth: 0.50,
                    crown_top_width: 0.46,
                    crown_top_depth: 0.25,
                    lower_rise: 0.23,
                    upper_rise: 0.10,
                },
                OracleCamera {
                    position: [-1.20, 1.05, -2.10],
                    target: [0.10, 0.11, -0.06],
                    horizontal_fov_degrees: 66.0,
                    width: 750,
                    height: 1000,
                },
            ),
        ];

        for (name, parameters, camera) in cases {
            let target_shape = shape_ratios(parameters);
            assert!(
                target_shape
                    .iter()
                    .zip(DEFAULT_SHAPE_PRIOR)
                    .any(|(target, prior)| (target - prior).abs() > 0.08),
                "{name} must not duplicate the fixed population prior"
            );
            let observation = synthetic_observation_for(parameters, camera, &[], 0, false);
            let fit = fit_single_view(&observation, fixed_prior_config(camera))
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            let vertex_rmse = projected_vertex_rmse(&fit, parameters, camera);
            let iou = silhouette_iou(&fit, parameters, camera);

            assert!(fit.confidence.accepted, "{name}: {fit:#?}");
            assert!(
                vertex_rmse < 0.005,
                "{name}: projected-vertex RMSE {vertex_rmse:.6} is not below 0.5% of the image diagonal; {fit:#?}"
            );
            assert!(
                iou > 0.95,
                "{name}: amodal silhouette IoU {iou:.6} is not above 0.95; {fit:#?}"
            );
        }
    }

    #[test]
    fn calibrated_pinhole_observations_recover_a_subpixel_mesh() {
        let parameters = oracle_parameters();
        let observation = synthetic_observation(&[], 0, false);
        let fit = fit_single_view(&observation, oracle_config(parameters)).unwrap();

        assert!(fit.reprojection_rmse < 0.005, "{fit:#?}");
        assert!(fit.confidence.accepted, "{fit:#?}");
        assert_eq!(fit.observation_count, KEYPOINT_COUNT);
        assert_eq!(fit.projected_mesh.indices.len(), 60);
        assert_eq!(fit.projected_mesh.faces.len(), 10);
        assert!(silhouette_iou(&fit, parameters, oracle_camera()) > 0.95);
        for (actual, expected) in shape_ratios(fit.parameters)
            .into_iter()
            .zip(shape_ratios(parameters))
        {
            assert!((actual - expected).abs() < 0.025, "{fit:#?}");
        }
    }

    #[test]
    fn d4_permutation_is_resolved_consistently_across_all_rings() {
        let parameters = oracle_parameters();
        let observation = synthetic_observation(&[], 3, true);
        let fit = fit_single_view(&observation, oracle_config(parameters)).unwrap();

        assert!(fit.reprojection_rmse < 0.005, "{fit:#?}");
        assert!(fit.confidence.accepted, "{fit:#?}");
        assert_eq!(fit.confidence.inlier_count, KEYPOINT_COUNT);
    }

    #[test]
    fn unknown_focal_length_is_estimated_from_the_standard_hypotheses() {
        let parameters = oracle_parameters();
        let observation = synthetic_observation(&[], 0, false);
        let fit = fit_single_view(
            &observation,
            SingleViewFitConfig {
                image_aspect_ratio: WIDTH as f32 / HEIGHT as f32,
                shape_prior: Some(shape_ratios(parameters)),
                shape_prior_weight: 0.0001,
                solver_patience: 80,
                ..SingleViewFitConfig::default()
            },
        )
        .unwrap();

        assert!(fit.reprojection_rmse < 0.005, "{fit:#?}");
        assert!(fit.confidence.accepted, "{fit:#?}");
        assert!((30.0..=110.0).contains(&fit.camera.horizontal_fov_degrees()));
    }

    #[test]
    fn six_partial_observations_still_generate_hidden_geometry() {
        let parameters = oracle_parameters();
        let hidden = [2, 3, 6, 7, 10, 11];
        let observation = synthetic_observation(&hidden, 0, false);
        let fit = fit_single_view(&observation, oracle_config(parameters)).unwrap();

        assert_eq!(fit.observation_count, 6);
        assert!(fit.reprojection_rmse < 0.0075, "{fit:#?}");
        assert!(fit.confidence.accepted, "{fit:#?}");
        for index in hidden {
            assert!(
                fit.projected_keypoints[index]
                    .iter()
                    .all(|value| value.is_finite())
            );
        }
        assert!(fit.bounding_box.min[0] < fit.bounding_box.max[0]);
        assert!(fit.bounding_box.min[1] < fit.bounding_box.max[1]);
    }

    #[test]
    fn four_cross_tier_observations_produce_a_constrained_prior_guided_fit() {
        let parameters = oracle_parameters();
        let hidden = [2, 3, 5, 6, 7, 9, 10, 11];
        let observation = synthetic_observation(&hidden, 0, false);
        let fit = fit_single_view(
            &observation,
            SingleViewFitConfig {
                minimum_observations: 4,
                shape_prior_weight: 0.05,
                ..oracle_config(parameters)
            },
        )
        .unwrap();

        assert_eq!(fit.observation_count, 4);
        assert!(fit.confidence.accepted, "{fit:#?}");
        assert!(fit.reprojection_rmse < 0.02, "{fit:#?}");
        assert!(silhouette_iou(&fit, parameters, oracle_camera()) > 0.65);
    }

    #[test]
    fn sparse_observations_must_span_all_three_rings() {
        let mut observation = SingleViewObservation::default();
        for index in [0, 1, 4, 5, 6] {
            observation.keypoints[index] = Some(KeypointObservation::new(
                0.2 + index as f32 * 0.03,
                0.3 + index as f32 * 0.02,
            ));
        }
        let error = fit_single_view(
            &observation,
            SingleViewFitConfig {
                minimum_observations: 4,
                ..SingleViewFitConfig::default()
            },
        )
        .unwrap_err();

        assert!(matches!(
            error,
            FitError::DegenerateObservations { ring_count: 2, .. }
        ));
    }

    #[test]
    fn robust_objective_limits_one_low_confidence_outlier() {
        let parameters = oracle_parameters();
        let mut observation = synthetic_observation(&[], 0, false);
        observation.keypoints[5] = Some(KeypointObservation {
            position: [0.95, 0.05],
            weight: 0.1,
        });
        let fit = fit_single_view(&observation, oracle_config(parameters)).unwrap();

        assert!(fit.reprojection_rmse < 0.09, "{fit:#?}");
        assert!(fit.confidence.inlier_count >= 10, "{fit:#?}");
    }

    #[test]
    fn rejects_insufficient_observations_and_invalid_configuration() {
        let error = fit_single_view(
            &SingleViewObservation::default(),
            SingleViewFitConfig::default(),
        )
        .unwrap_err();
        assert!(matches!(
            error,
            FitError::InsufficientObservations { actual: 0, .. }
        ));

        let observation = synthetic_observation(&[], 0, false);
        let error = fit_single_view(
            &observation,
            SingleViewFitConfig {
                image_aspect_ratio: 0.0,
                ..SingleViewFitConfig::default()
            },
        )
        .unwrap_err();
        assert!(matches!(error, FitError::InvalidConfiguration(_)));
    }

    fn silhouette_iou(
        fit: &SingleViewRoofFit,
        oracle_parameters: RoofParameters,
        oracle_camera: OracleCamera,
    ) -> f32 {
        const SIZE: usize = 256;
        let oracle = generate_roof(&oracle_parameters).unwrap();
        let oracle_vertices = oracle
            .mesh
            .vertices
            .iter()
            .map(|vertex| oracle_camera.project(vertex.position))
            .collect::<Vec<_>>();
        let fitted_vertices = fit
            .projected_mesh
            .vertices
            .iter()
            .map(|vertex| vertex.position)
            .collect::<Vec<_>>();
        let oracle_mask = rasterize_triangles(&oracle_vertices, &oracle.mesh.indices, SIZE);
        let fitted_mask = rasterize_triangles(&fitted_vertices, &fit.projected_mesh.indices, SIZE);
        let mut intersection = 0_usize;
        let mut union = 0_usize;
        for (oracle, fitted) in oracle_mask.into_iter().zip(fitted_mask) {
            intersection += usize::from(oracle && fitted);
            union += usize::from(oracle || fitted);
        }
        intersection as f32 / union.max(1) as f32
    }

    fn projected_vertex_rmse(
        fit: &SingleViewRoofFit,
        oracle_parameters: RoofParameters,
        oracle_camera: OracleCamera,
    ) -> f32 {
        let oracle = generate_roof(&oracle_parameters).unwrap();
        let fitted_roof = generate_roof(&fit.parameters).unwrap();
        assert_eq!(
            fit.projected_mesh.vertices.len(),
            oracle.mesh.vertices.len()
        );
        let width = oracle_camera.width as f32;
        let height = oracle_camera.height as f32;
        let diagonal = width.hypot(height);
        let mean_squared_pixels = fit
            .projected_mesh
            .vertices
            .iter()
            .zip(&fitted_roof.mesh.vertices)
            .map(|(fitted, fitted_local)| {
                let model_index = fitted_roof
                    .keypoints
                    .iter()
                    .position(|keypoint| keypoint.position == fitted_local.position)
                    .expect("every roof mesh vertex is a structural keypoint");
                let oracle_index =
                    correspondence(model_index, fit.camera.corner_shift, fit.camera.reflected);
                let expected = oracle_camera.project(oracle.keypoints[oracle_index].position);
                let dx = (fitted.position[0] - expected[0]) * width;
                let dy = (fitted.position[1] - expected[1]) * height;
                dx * dx + dy * dy
            })
            .sum::<f32>()
            / fit.projected_mesh.vertices.len() as f32;
        mean_squared_pixels.sqrt() / diagonal
    }

    fn rasterize_triangles(vertices: &[[f32; 2]], indices: &[u32], size: usize) -> Vec<bool> {
        let mut mask = vec![false; size * size];
        for triangle in indices.chunks_exact(3) {
            let points = [
                vertices[triangle[0] as usize],
                vertices[triangle[1] as usize],
                vertices[triangle[2] as usize],
            ];
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
            let start_x = (min_x * size as f32).floor().max(0.0) as usize;
            let end_x = (max_x * size as f32).ceil().min(size as f32) as usize;
            let start_y = (min_y * size as f32).floor().max(0.0) as usize;
            let end_y = (max_y * size as f32).ceil().min(size as f32) as usize;
            for y in start_y..end_y {
                for x in start_x..end_x {
                    let point = [
                        (x as f32 + 0.5) / size as f32,
                        (y as f32 + 0.5) / size as f32,
                    ];
                    if point_in_triangle(point, points) {
                        mask[y * size + x] = true;
                    }
                }
            }
        }
        mask
    }

    fn point_in_triangle(point: [f32; 2], triangle: [[f32; 2]; 3]) -> bool {
        fn cross(a: [f32; 2], b: [f32; 2], c: [f32; 2]) -> f32 {
            (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        }
        let ab = cross(triangle[0], triangle[1], point);
        let bc = cross(triangle[1], triangle[2], point);
        let ca = cross(triangle[2], triangle[0], point);
        (ab >= 0.0 && bc >= 0.0 && ca >= 0.0) || (ab <= 0.0 && bc <= 0.0 && ca <= 0.0)
    }
}
