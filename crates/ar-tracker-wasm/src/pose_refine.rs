//! Small, bounded six-degree-of-freedom pose refinement.
//!
//! This module deliberately has no solver dependency. It is suitable for both
//! native replay and `wasm32`: each call performs a fixed number of dense 6x6
//! Gauss-Newton steps over a capped number of observations.
//!
//! Coordinate conventions match the rest of `ar-tracker-wasm`: `orientation`
//! rotates camera coordinates into the world, the camera looks along local
//! `-z`, and centered pixel `y` points down. Rotation increments are
//! camera-local small angles, applied on the right of the camera-to-world
//! quaternion.

use glam::{DQuat, DVec3};

const HARD_MAXIMUM_OBSERVATIONS: usize = 256;
const HARD_MAXIMUM_ITERATIONS: usize = 16;

/// One fixed-world landmark observed in the current camera frame.
#[derive(Clone, Copy, Debug)]
pub struct PoseRefineObservation {
    /// Landmark position in world metres.
    pub world: DVec3,
    /// Pixel x coordinate after subtracting the principal point.
    pub pixel_x: f64,
    /// Pixel y coordinate after subtracting the principal point.
    pub pixel_y: f64,
    /// Relative information weight in `[0, 1]`.
    pub weight: f64,
}

/// Safety and robustness settings for [`refine_pose_with_config`].
#[derive(Clone, Copy, Debug)]
pub struct PoseRefineConfig {
    /// Minimum usable observations required before and after trimming.
    pub minimum_observations: usize,
    /// Maximum observations entering a solve, up to a hard limit of 256.
    pub maximum_observations: usize,
    /// Robust iterations before residual trimming (16 total at most).
    pub initial_iterations: usize,
    /// Robust iterations after residual trimming.
    pub trimmed_iterations: usize,
    /// Nominal standard deviation of a tracked pixel.
    pub reprojection_sigma_pixels: f64,
    /// Huber transition in unweighted pixel residual magnitude.
    pub huber_pixels: f64,
    /// Multiple of the median residual used as the trimming threshold.
    pub trim_median_multiplier: f64,
    /// Lower clamp for the adaptive trimming threshold.
    pub trim_minimum_pixels: f64,
    /// Upper clamp for the adaptive trimming threshold.
    pub trim_maximum_pixels: f64,
    /// Standard deviation of the inertial position prediction, in metres.
    pub position_prior_sigma_metres: f64,
    /// Standard deviation of the camera-to-world IMU orientation prior.
    pub orientation_prior_sigma_radians: f64,
    /// Largest translation accepted from one linearized iteration.
    pub maximum_position_step_metres: f64,
    /// Largest small-angle rotation accepted from one iteration.
    pub maximum_orientation_step_radians: f64,
    /// Largest total translation allowed away from the inertial prediction.
    pub maximum_position_correction_metres: f64,
    /// Largest total rotation allowed away from the IMU orientation.
    pub maximum_orientation_correction_radians: f64,
    /// Points at or nearer than this optical-axis depth are ignored.
    pub minimum_depth_metres: f64,
    /// Scale-relative diagonal damping applied to the 6x6 normal matrix.
    pub damping: f64,
}

impl Default for PoseRefineConfig {
    fn default() -> Self {
        Self {
            minimum_observations: 6,
            maximum_observations: 128,
            initial_iterations: 3,
            trimmed_iterations: 3,
            reprojection_sigma_pixels: 1.5,
            huber_pixels: 3.0,
            trim_median_multiplier: 2.5,
            trim_minimum_pixels: 2.5,
            trim_maximum_pixels: 8.0,
            // Translation is allowed to follow a coherent visual correction,
            // but no individual frame may move far enough to visibly teleport.
            position_prior_sigma_metres: 0.12,
            maximum_position_step_metres: 0.04,
            maximum_position_correction_metres: 0.20,
            // DeviceOrientation is usually more reliable over one frame than
            // a sparse monocular rotation estimate. Vision therefore gets only
            // a small correction budget around a roughly one-degree prior.
            orientation_prior_sigma_radians: 0.015,
            maximum_orientation_step_radians: 0.01,
            maximum_orientation_correction_radians: 0.035,
            minimum_depth_metres: 0.08,
            damping: 1.0e-6,
        }
    }
}

/// Result of one bounded pose refinement.
#[derive(Clone, Debug)]
pub struct PoseRefinement {
    /// Refined camera position in world metres.
    pub position: DVec3,
    /// Refined camera-to-world orientation.
    pub orientation: DQuat,
    /// Indices into the caller's observation slice that survived trimming.
    pub inliers: Vec<usize>,
    /// Indices into the caller's observation slice that entered the bounded
    /// solve. Observations outside the fixed-cap selection were not evaluated
    /// and must not be classified as reprojection outliers by the caller.
    pub evaluated: Vec<usize>,
    /// Root-mean-square pixel error over `inliers`.
    pub reprojection_rmse_pixels: f64,
    /// Number of Gauss-Newton iterations actually attempted.
    pub iterations: usize,
}

/// Refine a camera pose with the conservative default settings.
///
/// Returns `None` when the inputs are invalid, too few landmarks project in
/// front of the camera, the geometry is singular, or trimming leaves too
/// little support. A caller should retain its inertial prediction in that case.
pub fn refine_pose(
    observations: &[PoseRefineObservation],
    orientation_prior: DQuat,
    predicted_position: DVec3,
    focal_length_pixels: f64,
) -> Option<PoseRefinement> {
    refine_pose_with_config(
        observations,
        orientation_prior,
        predicted_position,
        focal_length_pixels,
        &PoseRefineConfig::default(),
    )
}

/// Refine a camera pose with explicit tuning and safety settings.
///
/// The visual residual is Huber-weighted in both stages. After the first
/// stage, a median-derived threshold trims gross feature mismatches and the
/// second stage resolves only on the retained observations. Both translation
/// and rotation are clamped per iteration and relative to their input priors.
pub fn refine_pose_with_config(
    observations: &[PoseRefineObservation],
    orientation_prior: DQuat,
    predicted_position: DVec3,
    focal_length_pixels: f64,
    config: &PoseRefineConfig,
) -> Option<PoseRefinement> {
    if !valid_inputs(
        orientation_prior,
        predicted_position,
        focal_length_pixels,
        config,
    ) {
        return None;
    }

    let orientation_prior = orientation_prior.normalize();
    let selected = select_observations(observations, config.maximum_observations);
    if selected.len() < config.minimum_observations {
        return None;
    }

    let mut position = predicted_position;
    let mut orientation = orientation_prior;
    let mut attempted_iterations = 0;
    if !solve_iterations(
        observations,
        &selected,
        orientation_prior,
        predicted_position,
        focal_length_pixels,
        config,
        config.initial_iterations,
        &mut position,
        &mut orientation,
        &mut attempted_iterations,
    ) {
        return None;
    }

    let mut residuals = residuals_for_indices(
        observations,
        &selected,
        position,
        orientation,
        focal_length_pixels,
        config.minimum_depth_metres,
    );
    if residuals.len() < config.minimum_observations {
        return None;
    }
    residuals.sort_by(|left, right| left.1.total_cmp(&right.1));
    let median = residuals[residuals.len() / 2].1;
    let trim_threshold = (median * config.trim_median_multiplier)
        .clamp(config.trim_minimum_pixels, config.trim_maximum_pixels);
    let trimmed: Vec<usize> = residuals
        .iter()
        .filter(|(_, residual)| *residual <= trim_threshold)
        .map(|(index, _)| *index)
        .collect();
    if trimmed.len() < config.minimum_observations {
        return None;
    }

    if !solve_iterations(
        observations,
        &trimmed,
        orientation_prior,
        predicted_position,
        focal_length_pixels,
        config,
        config.trimmed_iterations,
        &mut position,
        &mut orientation,
        &mut attempted_iterations,
    ) {
        return None;
    }

    let final_inliers_with_residuals: Vec<(usize, f64)> = residuals_for_indices(
        observations,
        &trimmed,
        position,
        orientation,
        focal_length_pixels,
        config.minimum_depth_metres,
    )
    .into_iter()
    .filter(|(_, residual)| *residual <= trim_threshold)
    .collect();
    if final_inliers_with_residuals.len() < config.minimum_observations {
        return None;
    }
    let squared_error: f64 = final_inliers_with_residuals
        .iter()
        .map(|(_, residual)| residual * residual)
        .sum();
    let reprojection_rmse_pixels =
        (squared_error / final_inliers_with_residuals.len() as f64).sqrt();
    let mut inliers: Vec<usize> = final_inliers_with_residuals
        .into_iter()
        .map(|(index, _)| index)
        .collect();
    inliers.sort_unstable();

    Some(PoseRefinement {
        position,
        orientation,
        inliers,
        evaluated: selected,
        reprojection_rmse_pixels,
        iterations: attempted_iterations,
    })
}

fn valid_inputs(
    orientation_prior: DQuat,
    predicted_position: DVec3,
    focal_length_pixels: f64,
    config: &PoseRefineConfig,
) -> bool {
    orientation_prior.is_finite()
        && orientation_prior.length_squared().is_finite()
        && orientation_prior.length_squared() > 0.5
        && predicted_position.is_finite()
        && focal_length_pixels.is_finite()
        && focal_length_pixels > 0.0
        && config.minimum_observations >= 4
        && config.maximum_observations >= config.minimum_observations
        && config.maximum_observations <= HARD_MAXIMUM_OBSERVATIONS
        && config.initial_iterations > 0
        && config.trimmed_iterations > 0
        && config
            .initial_iterations
            .checked_add(config.trimmed_iterations)
            .is_some_and(|iterations| iterations <= HARD_MAXIMUM_ITERATIONS)
        && config.reprojection_sigma_pixels.is_finite()
        && config.reprojection_sigma_pixels > 0.0
        && config.huber_pixels.is_finite()
        && config.huber_pixels > 0.0
        && config.trim_median_multiplier.is_finite()
        && config.trim_median_multiplier >= 1.0
        && config.trim_minimum_pixels.is_finite()
        && config.trim_minimum_pixels > 0.0
        && config.trim_maximum_pixels.is_finite()
        && config.trim_maximum_pixels >= config.trim_minimum_pixels
        && config.position_prior_sigma_metres.is_finite()
        && config.position_prior_sigma_metres > 0.0
        && config.orientation_prior_sigma_radians.is_finite()
        && config.orientation_prior_sigma_radians > 0.0
        && config.maximum_position_step_metres.is_finite()
        && config.maximum_position_step_metres > 0.0
        && config.maximum_orientation_step_radians.is_finite()
        && config.maximum_orientation_step_radians > 0.0
        && config.maximum_position_correction_metres.is_finite()
        && config.maximum_position_correction_metres > 0.0
        && config.maximum_orientation_correction_radians.is_finite()
        && config.maximum_orientation_correction_radians > 0.0
        && config.minimum_depth_metres.is_finite()
        && config.minimum_depth_metres > 0.0
        && config.damping.is_finite()
        && config.damping >= 0.0
}

fn select_observations(
    observations: &[PoseRefineObservation],
    maximum_observations: usize,
) -> Vec<usize> {
    // Keep the strongest observations with fixed memory. Equal weights retain
    // earlier input order, which makes replay deterministic.
    let mut selected = Vec::with_capacity(maximum_observations);
    for (index, observation) in observations.iter().enumerate() {
        if !observation.world.is_finite()
            || !observation.pixel_x.is_finite()
            || !observation.pixel_y.is_finite()
            || !observation.weight.is_finite()
            || observation.weight <= 0.0
        {
            continue;
        }
        if selected.len() < maximum_observations {
            selected.push(index);
            continue;
        }
        let Some((weakest_slot, weakest_index)) =
            selected.iter().enumerate().min_by(|(_, left), (_, right)| {
                observations[**left]
                    .weight
                    .total_cmp(&observations[**right].weight)
                    .then_with(|| right.cmp(left))
            })
        else {
            continue;
        };
        if observation.weight > observations[*weakest_index].weight {
            selected[weakest_slot] = index;
        }
    }
    selected.sort_unstable();
    selected
}

#[allow(clippy::too_many_arguments)]
fn solve_iterations(
    observations: &[PoseRefineObservation],
    indices: &[usize],
    orientation_prior: DQuat,
    predicted_position: DVec3,
    focal_length_pixels: f64,
    config: &PoseRefineConfig,
    iteration_count: usize,
    position: &mut DVec3,
    orientation: &mut DQuat,
    attempted_iterations: &mut usize,
) -> bool {
    for _ in 0..iteration_count {
        *attempted_iterations += 1;
        let mut normal = [[0.0; 6]; 6];
        let mut right_hand_side = [0.0; 6];
        let mut usable = 0;

        for &index in indices {
            let observation = &observations[index];
            let Some((residual, jacobian)) = residual_and_jacobian(
                observation,
                *position,
                *orientation,
                focal_length_pixels,
                config.minimum_depth_metres,
            ) else {
                continue;
            };
            let residual_magnitude = residual[0].hypot(residual[1]);
            let robust_weight = if residual_magnitude <= config.huber_pixels {
                1.0
            } else {
                config.huber_pixels / residual_magnitude
            };
            let information = observation.weight.clamp(0.0, 1.0) * robust_weight
                / config.reprojection_sigma_pixels.powi(2);
            if !information.is_finite() || information <= 0.0 {
                continue;
            }
            usable += 1;
            for row in 0..2 {
                for column in 0..6 {
                    right_hand_side[column] -= information * jacobian[row][column] * residual[row];
                    for other in column..6 {
                        normal[column][other] +=
                            information * jacobian[row][column] * jacobian[row][other];
                    }
                }
            }
        }
        if usable < config.minimum_observations {
            return false;
        }
        mirror_upper_triangle(&mut normal);

        // Priors are expressed as the local update that would return the
        // current estimate to the inertial input pose.
        let desired_translation_local = orientation.conjugate() * (predicted_position - *position);
        add_isotropic_prior(
            &mut normal,
            &mut right_hand_side,
            0,
            desired_translation_local,
            config.position_prior_sigma_metres,
        );
        let desired_rotation_local =
            shortest_scaled_axis(orientation.conjugate() * orientation_prior);
        add_isotropic_prior(
            &mut normal,
            &mut right_hand_side,
            3,
            desired_rotation_local,
            config.orientation_prior_sigma_radians,
        );

        for (diagonal, row) in normal.iter_mut().enumerate() {
            row[diagonal] += config.damping * row[diagonal].max(1.0);
        }
        let Some(mut increment) = solve_symmetric_6x6(normal, right_hand_side) else {
            return false;
        };
        if increment.iter().any(|value| !value.is_finite()) {
            return false;
        }

        clamp_triplet(&mut increment, 0, config.maximum_position_step_metres);
        clamp_triplet(&mut increment, 3, config.maximum_orientation_step_radians);
        let translation_local = DVec3::new(increment[0], increment[1], increment[2]);
        let rotation_local = DVec3::new(increment[3], increment[4], increment[5]);
        *position += *orientation * translation_local;
        *orientation = (*orientation * DQuat::from_scaled_axis(rotation_local)).normalize();
        apply_total_bounds(
            position,
            orientation,
            predicted_position,
            orientation_prior,
            config,
        );

        if translation_local.length() < 1.0e-5 && rotation_local.length() < 1.0e-5 {
            break;
        }
    }
    position.is_finite() && orientation.is_finite()
}

fn residual_and_jacobian(
    observation: &PoseRefineObservation,
    position: DVec3,
    orientation: DQuat,
    focal_length_pixels: f64,
    minimum_depth_metres: f64,
) -> Option<([f64; 2], [[f64; 6]; 2])> {
    let camera = orientation.conjugate() * (observation.world - position);
    let depth = -camera.z;
    if !camera.is_finite() || depth <= minimum_depth_metres {
        return None;
    }
    let inverse_depth = depth.recip();
    let inverse_depth_squared = inverse_depth * inverse_depth;
    let projected = [
        focal_length_pixels * camera.x * inverse_depth,
        -focal_length_pixels * camera.y * inverse_depth,
    ];
    let residual = [
        projected[0] - observation.pixel_x,
        projected[1] - observation.pixel_y,
    ];

    let projection_jacobian = [
        [
            focal_length_pixels * inverse_depth,
            0.0,
            focal_length_pixels * camera.x * inverse_depth_squared,
        ],
        [
            0.0,
            -focal_length_pixels * inverse_depth,
            -focal_length_pixels * camera.y * inverse_depth_squared,
        ],
    ];
    // A local translation dt gives dc = -dt. A right-multiplied
    // small-angle rotation dtheta gives dc = camera x dtheta.
    let camera_jacobian = [
        [-1.0, 0.0, 0.0, 0.0, -camera.z, camera.y],
        [0.0, -1.0, 0.0, camera.z, 0.0, -camera.x],
        [0.0, 0.0, -1.0, -camera.y, camera.x, 0.0],
    ];
    let mut jacobian = [[0.0; 6]; 2];
    for row in 0..2 {
        for column in 0..6 {
            jacobian[row][column] = (0..3)
                .map(|axis| projection_jacobian[row][axis] * camera_jacobian[axis][column])
                .sum();
        }
    }
    Some((residual, jacobian))
}

fn residuals_for_indices(
    observations: &[PoseRefineObservation],
    indices: &[usize],
    position: DVec3,
    orientation: DQuat,
    focal_length_pixels: f64,
    minimum_depth_metres: f64,
) -> Vec<(usize, f64)> {
    indices
        .iter()
        .filter_map(|&index| {
            residual_and_jacobian(
                &observations[index],
                position,
                orientation,
                focal_length_pixels,
                minimum_depth_metres,
            )
            .map(|(residual, _)| (index, residual[0].hypot(residual[1])))
        })
        .collect()
}

fn mirror_upper_triangle(matrix: &mut [[f64; 6]; 6]) {
    for row in 1..6 {
        let (previous_rows, current_and_later) = matrix.split_at_mut(row);
        let current_row = &mut current_and_later[0];
        for (column, previous_row) in previous_rows.iter().enumerate() {
            current_row[column] = previous_row[row];
        }
    }
}

fn add_isotropic_prior(
    normal: &mut [[f64; 6]; 6],
    right_hand_side: &mut [f64; 6],
    offset: usize,
    desired: DVec3,
    sigma: f64,
) {
    let information = sigma.recip().powi(2);
    for axis in 0..3 {
        normal[offset + axis][offset + axis] += information;
        right_hand_side[offset + axis] += information * desired[axis];
    }
}

fn clamp_triplet(values: &mut [f64; 6], offset: usize, maximum_length: f64) {
    let length = values[offset..offset + 3]
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    if length > maximum_length {
        let scale = maximum_length / length;
        for value in &mut values[offset..offset + 3] {
            *value *= scale;
        }
    }
}

fn apply_total_bounds(
    position: &mut DVec3,
    orientation: &mut DQuat,
    predicted_position: DVec3,
    orientation_prior: DQuat,
    config: &PoseRefineConfig,
) {
    let translation = *position - predicted_position;
    if translation.length() > config.maximum_position_correction_metres {
        *position = predicted_position
            + translation.normalize() * config.maximum_position_correction_metres;
    }
    let prior_to_current = shortest_scaled_axis(orientation_prior.conjugate() * *orientation);
    if prior_to_current.length() > config.maximum_orientation_correction_radians {
        *orientation = (orientation_prior
            * DQuat::from_scaled_axis(
                prior_to_current.normalize() * config.maximum_orientation_correction_radians,
            ))
        .normalize();
    }
}

fn shortest_scaled_axis(mut rotation: DQuat) -> DVec3 {
    rotation = rotation.normalize();
    if rotation.w < 0.0 {
        rotation = -rotation;
    }
    rotation.to_scaled_axis()
}

fn solve_symmetric_6x6(matrix: [[f64; 6]; 6], right_hand_side: [f64; 6]) -> Option<[f64; 6]> {
    let mut augmented = [[0.0; 7]; 6];
    for row in 0..6 {
        augmented[row][..6].copy_from_slice(&matrix[row]);
        augmented[row][6] = right_hand_side[row];
    }
    for pivot in 0..6 {
        let mut best_row = pivot;
        for candidate in pivot + 1..6 {
            if augmented[candidate][pivot].abs() > augmented[best_row][pivot].abs() {
                best_row = candidate;
            }
        }
        let pivot_value = augmented[best_row][pivot];
        if !pivot_value.is_finite() || pivot_value.abs() < 1.0e-12 {
            return None;
        }
        if best_row != pivot {
            augmented.swap(best_row, pivot);
        }
        let pivot_row = augmented[pivot];
        for row in pivot + 1..6 {
            let factor = augmented[row][pivot] / augmented[pivot][pivot];
            for (target, source) in augmented[row].iter_mut().zip(pivot_row.iter()).skip(pivot) {
                *target -= factor * source;
            }
        }
    }

    let mut solution = [0.0; 6];
    for row in (0..6).rev() {
        let known: f64 = (row + 1..6)
            .map(|column| augmented[row][column] * solution[column])
            .sum();
        let diagonal = augmented[row][row];
        if !diagonal.is_finite() || diagonal.abs() < 1.0e-12 {
            return None;
        }
        solution[row] = (augmented[row][6] - known) / diagonal;
    }
    Some(solution)
}

#[cfg(test)]
mod tests {
    use super::*;

    const FOCAL: f64 = 360.0;

    fn angle_between(left: DQuat, right: DQuat) -> f64 {
        shortest_scaled_axis(left.conjugate() * right).length()
    }

    fn observations_for_pose(position: DVec3, orientation: DQuat) -> Vec<PoseRefineObservation> {
        let mut observations = Vec::new();
        for row in 0..5 {
            for column in 0..7 {
                let x = (column as f64 - 3.0) * 0.18;
                let y = (row as f64 - 2.0) * 0.14;
                let z = -2.0 - 0.25 * ((row + column) % 4) as f64;
                let camera = DVec3::new(x, y, z);
                let depth = -camera.z;
                observations.push(PoseRefineObservation {
                    world: position + orientation * camera,
                    pixel_x: FOCAL * camera.x / depth,
                    pixel_y: -FOCAL * camera.y / depth,
                    weight: 1.0,
                });
            }
        }
        observations
    }

    #[test]
    fn analytic_jacobian_matches_local_finite_difference() {
        let position = DVec3::new(0.3, -0.2, 0.1);
        let orientation = DQuat::from_scaled_axis(DVec3::new(0.08, -0.04, 0.03));
        let observation = PoseRefineObservation {
            world: position + orientation * DVec3::new(0.4, -0.3, -2.7),
            pixel_x: 12.0,
            pixel_y: -8.0,
            weight: 1.0,
        };
        let (base, analytic) =
            residual_and_jacobian(&observation, position, orientation, FOCAL, 0.08).unwrap();
        let epsilon = 1.0e-7;
        for column in 0..6 {
            let mut perturbed_position = position;
            let mut perturbed_orientation = orientation;
            if column < 3 {
                let mut delta = DVec3::ZERO;
                delta[column] = epsilon;
                perturbed_position += orientation * delta;
            } else {
                let mut delta = DVec3::ZERO;
                delta[column - 3] = epsilon;
                perturbed_orientation = (orientation * DQuat::from_scaled_axis(delta)).normalize();
            }
            let (changed, _) = residual_and_jacobian(
                &observation,
                perturbed_position,
                perturbed_orientation,
                FOCAL,
                0.08,
            )
            .unwrap();
            for row in 0..2 {
                let numerical = (changed[row] - base[row]) / epsilon;
                assert!(
                    (numerical - analytic[row][column]).abs() < 1.0e-4,
                    "row={row} column={column}: numerical={numerical}, analytic={}",
                    analytic[row][column]
                );
            }
        }
    }

    #[test]
    fn recovers_small_position_and_attitude_error() {
        let truth_position = DVec3::new(0.045, -0.025, 0.035);
        let truth_orientation = DQuat::from_scaled_axis(DVec3::new(0.009, -0.012, 0.006));
        let observations = observations_for_pose(truth_position, truth_orientation);
        let result = refine_pose(&observations, DQuat::IDENTITY, DVec3::ZERO, FOCAL).unwrap();

        assert!((result.position - truth_position).length() < 0.012);
        assert!(angle_between(result.orientation, truth_orientation) < 0.008);
        assert_eq!(result.inliers.len(), observations.len());
        assert!(result.reprojection_rmse_pixels < 0.8);
    }

    #[test]
    fn trims_large_pixel_outliers() {
        let truth_position = DVec3::new(0.035, 0.01, -0.025);
        let truth_orientation = DQuat::from_scaled_axis(DVec3::new(-0.006, 0.01, 0.004));
        let mut observations = observations_for_pose(truth_position, truth_orientation);
        let outliers = [2, 9, 17, 28];
        for (ordinal, index) in outliers.iter().copied().enumerate() {
            observations[index].pixel_x += 70.0 + ordinal as f64 * 8.0;
            observations[index].pixel_y -= 55.0;
        }
        let result = refine_pose(&observations, DQuat::IDENTITY, DVec3::ZERO, FOCAL).unwrap();

        assert!((result.position - truth_position).length() < 0.02);
        assert!(angle_between(result.orientation, truth_orientation) < 0.01);
        for outlier in outliers {
            assert!(!result.inliers.contains(&outlier));
        }
        assert!(result.inliers.len() >= observations.len() - outliers.len() - 1);
    }

    #[test]
    fn reports_only_observations_that_entered_a_capped_solve_as_evaluated() {
        let observations = observations_for_pose(DVec3::ZERO, DQuat::IDENTITY);
        let config = PoseRefineConfig {
            maximum_observations: 12,
            ..PoseRefineConfig::default()
        };
        let result =
            refine_pose_with_config(&observations, DQuat::IDENTITY, DVec3::ZERO, FOCAL, &config)
                .unwrap();

        assert_eq!(result.evaluated.len(), config.maximum_observations);
        assert!(
            result
                .inliers
                .iter()
                .all(|index| result.evaluated.contains(index))
        );
        assert!(
            (config.maximum_observations..observations.len())
                .any(|index| !result.evaluated.contains(&index))
        );
    }

    #[test]
    fn total_visual_correction_is_bounded() {
        let observations = observations_for_pose(
            DVec3::new(1.0, -0.5, 0.8),
            DQuat::from_scaled_axis(DVec3::new(0.2, -0.15, 0.1)),
        );
        let config = PoseRefineConfig {
            trim_maximum_pixels: 1_000.0,
            ..PoseRefineConfig::default()
        };
        // Avoid trimming every point solely because this deliberately starts
        // far outside the local solver's capture range.
        let result =
            refine_pose_with_config(&observations, DQuat::IDENTITY, DVec3::ZERO, FOCAL, &config)
                .unwrap();

        assert!(result.position.length() <= config.maximum_position_correction_metres + 1.0e-12);
        assert!(
            angle_between(DQuat::IDENTITY, result.orientation)
                <= config.maximum_orientation_correction_radians + 1.0e-12
        );
    }

    #[test]
    fn rejects_invalid_or_insufficient_inputs() {
        let observations = observations_for_pose(DVec3::ZERO, DQuat::IDENTITY);
        assert!(refine_pose(&observations[..5], DQuat::IDENTITY, DVec3::ZERO, FOCAL).is_none());
        assert!(refine_pose(&observations, DQuat::IDENTITY, DVec3::ZERO, 0.0).is_none());
        assert!(
            refine_pose(
                &observations,
                DQuat::from_xyzw(f64::NAN, 0.0, 0.0, 1.0),
                DVec3::ZERO,
                FOCAL,
            )
            .is_none()
        );
    }
}
