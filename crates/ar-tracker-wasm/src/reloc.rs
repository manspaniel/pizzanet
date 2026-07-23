//! Appearance-based relocalization, ported from the previous tracker
//! iteration (validated against the recorded sessions there).
//!
//! When the camera returns to a previously mapped view, a coarse global
//! descriptor shortlists candidate keyframes and a rotation-compensated direct
//! photometric alignment estimates the translation between the stored and the
//! live frame, with block-correlation and edge-structure verification guarding
//! against false positives. The scene-depth value used for the projection now
//! comes from the live map's converged landmarks instead of a hardcoded prior.

use crate::geometry::focal_length_pixels;
use vio_core::{DQuat, DVec3};

pub const APPEARANCE_LOOP_MINIMUM_FRAME_GAP: u32 = 45;
const APPEARANCE_LOOP_MAX_ORIENTATION_DEGREES: f64 = 8.0;
const APPEARANCE_LOOP_MIN_CORRELATION: f64 = 0.85;
const APPEARANCE_DIRECT_MIN_CORRELATION: f64 = 0.9;
pub const APPEARANCE_DIRECT_SMOOTH_MIN_CORRELATION: f64 = 0.94;
const APPEARANCE_DIRECT_SEARCH_RADIUS_PIXELS: isize = 16;
const APPEARANCE_DIRECT_MIN_BLOCK_INLIERS: usize = 2;
const APPEARANCE_DIRECT_MAX_TRANSLATION_METRES: f64 = 0.5;
const VISUAL_VERTICAL_TRANSLATION_GAIN: f64 = 0.5;

/// The view of a stored keyframe (or the live frame) the relocalizer needs.
pub struct RelocView<'a> {
    pub frame_id: u32,
    pub pixels: &'a [u8],
    pub width: usize,
    pub height: usize,
    pub orientation: DQuat,
    pub position: DVec3,
    pub descriptor: &'a [i8],
}

pub struct RelocMatch {
    /// Index of the matched keyframe in the candidate slice.
    pub keyframe_index: usize,
    pub camera_delta_world: DVec3,
    pub matches: usize,
    pub inliers: usize,
    pub spatially_verified: bool,
}

#[derive(Clone, Copy)]
struct TrackedFeature {
    observed_x: f64,
    observed_y: f64,
}

fn feature_coverage(features: &[TrackedFeature]) -> usize {
    let mut occupied = [false; 4];
    for feature in features {
        let column = usize::from(feature.observed_x >= 0.0);
        let row = usize::from(feature.observed_y >= 0.0);
        occupied[row * 2 + column] = true;
    }
    occupied.into_iter().filter(|value| *value).count()
}

#[derive(Clone, Copy, Default)]
struct CorrelationAccumulator {
    count: u64,
    left_sum: f64,
    right_sum: f64,
    left_squared_sum: f64,
    right_squared_sum: f64,
    cross_sum: f64,
}

impl CorrelationAccumulator {
    fn push(&mut self, left: f64, right: f64) {
        self.count = self.count.saturating_add(1);
        self.left_sum += left;
        self.right_sum += right;
        self.left_squared_sum += left * left;
        self.right_squared_sum += right * right;
        self.cross_sum += left * right;
    }

    fn correlation(self, minimum_standard_deviation: f64) -> Option<f64> {
        if self.count < 2 {
            return None;
        }
        let count = self.count as f64;
        let left_variance = self.left_squared_sum - self.left_sum * self.left_sum / count;
        let right_variance = self.right_squared_sum - self.right_sum * self.right_sum / count;
        if left_variance < minimum_standard_deviation.powi(2) * count
            || right_variance < minimum_standard_deviation.powi(2) * count
        {
            return None;
        }
        let covariance = self.cross_sum - self.left_sum * self.right_sum / count;
        Some(covariance / (left_variance * right_variance).sqrt())
    }
}

fn descriptor_correlation(left: &[i8], right: &[i8]) -> f64 {
    let mut dot = 0.0;
    let mut left_squared = 0.0;
    let mut right_squared = 0.0;
    for (left, right) in left.iter().zip(right) {
        let left = f64::from(*left);
        let right = f64::from(*right);
        dot += left * right;
        left_squared += left * left;
        right_squared += right * right;
    }
    let denominator = (left_squared * right_squared).sqrt();
    if denominator <= f64::EPSILON {
        0.0
    } else {
        dot / denominator
    }
}


pub fn visual_frame_descriptor(pixels: &[u8], width: usize, height: usize) -> Vec<i8> {
    const COLUMNS: usize = 8;
    const ROWS: usize = 12;
    let mut means = Vec::with_capacity(COLUMNS * ROWS);
    for row in 0..ROWS {
        let start_y = row * height / ROWS;
        let end_y = ((row + 1) * height / ROWS).max(start_y + 1);
        for column in 0..COLUMNS {
            let start_x = column * width / COLUMNS;
            let end_x = ((column + 1) * width / COLUMNS).max(start_x + 1);
            let mut sum = 0_u64;
            let mut count = 0_u64;
            for y in start_y..end_y.min(height) {
                for x in start_x..end_x.min(width) {
                    sum += u64::from(pixels[y * width + x]);
                    count += 1;
                }
            }
            means.push(sum as f64 / count.max(1) as f64);
        }
    }
    let mean = means.iter().sum::<f64>() / means.len().max(1) as f64;
    let variance = means
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / means.len().max(1) as f64;
    let scale = variance.sqrt().max(4.0);
    means
        .into_iter()
        .map(|value| {
            (((value - mean) / scale) * 32.0)
                .round()
                .clamp(-127.0, 127.0) as i8
        })
        .collect()
}

pub fn find_appearance_relocalization(
    keyframes: &[RelocView<'_>],
    current: &RelocView<'_>,
    scene_depth_metres: f64,
    long_axis_field_of_view_degrees: f64,
) -> Option<RelocMatch> {
    let minimum_orientation_dot = (APPEARANCE_LOOP_MAX_ORIENTATION_DEGREES * 0.5)
        .to_radians()
        .cos();
    let mut candidates = keyframes
        .iter()
        .enumerate()
        .filter(|(_, keyframe)| {
            current.frame_id.abs_diff(keyframe.frame_id) >= APPEARANCE_LOOP_MINIMUM_FRAME_GAP
                && keyframe.width == current.width
                && keyframe.height == current.height
                && keyframe.orientation.dot(current.orientation).abs() >= minimum_orientation_dot
        })
        .filter_map(|(index, keyframe)| {
            let correlation = descriptor_correlation(keyframe.descriptor, current.descriptor);
            (correlation >= APPEARANCE_LOOP_MIN_CORRELATION)
                .then_some((correlation, index, keyframe))
        })
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| right.0.total_cmp(&left.0));
    candidates
        .into_iter()
        .take(4)
        .filter_map(|(_, index, keyframe)| {
            direct_appearance_translation(
                keyframe,
                current,
                scene_depth_metres,
                long_axis_field_of_view_degrees,
            )
            .map(|(correlation, matched, spatially_verified)| {
                (correlation, index, matched, spatially_verified)
            })
        })
        .max_by(|left, right| left.0.total_cmp(&right.0))
        .map(|(_, index, matched, spatially_verified)| RelocMatch {
            keyframe_index: index,
            camera_delta_world: matched.0,
            matches: matched.1,
            inliers: matched.2,
            spatially_verified,
        })
}

fn direct_appearance_translation(
    previous: &RelocView<'_>,
    current: &RelocView<'_>,
    scene_depth_metres: f64,
    long_axis_field_of_view_degrees: f64,
) -> Option<(f64, (DVec3, usize, usize), bool)> {
    const SAMPLE_STEP: usize = 4;
    const BLOCK_COLUMNS: usize = 2;
    const BLOCK_ROWS: usize = 3;
    const BLOCK_COUNT: usize = BLOCK_COLUMNS * BLOCK_ROWS;
    let width = previous.width;
    let height = previous.height;
    if width != current.width || height != current.height || width < 24 || height < 24 {
        return None;
    }
    let focal = focal_length_pixels(width, height, long_axis_field_of_view_degrees);
    let center_x = (width.saturating_sub(1)) as f64 * 0.5;
    let center_y = (height.saturating_sub(1)) as f64 * 0.5;
    let mut best = None;
    for offset_y in (-APPEARANCE_DIRECT_SEARCH_RADIUS_PIXELS
        ..=APPEARANCE_DIRECT_SEARCH_RADIUS_PIXELS)
        .step_by(2)
    {
        for offset_x in (-APPEARANCE_DIRECT_SEARCH_RADIUS_PIXELS
            ..=APPEARANCE_DIRECT_SEARCH_RADIUS_PIXELS)
            .step_by(2)
        {
            let mut accumulator = CorrelationAccumulator::default();
            for previous_y in (4..height - 4).step_by(SAMPLE_STEP) {
                for previous_x in (4..width - 4).step_by(SAMPLE_STEP) {
                    let Some((current_x, current_y, _)) = rotation_compensated_projection(
                        previous,
                        current.orientation,
                        previous_x as f64,
                        previous_y as f64,
                        scene_depth_metres,
                        focal,
                        center_x,
                        center_y,
                        offset_x as f64,
                        offset_y as f64,
                    ) else {
                        continue;
                    };
                    let rounded_x = current_x.round() as isize;
                    let rounded_y = current_y.round() as isize;
                    if rounded_x < 0
                        || rounded_y < 0
                        || rounded_x >= width as isize
                        || rounded_y >= height as isize
                    {
                        continue;
                    }
                    accumulator.push(
                        f64::from(previous.pixels[previous_y * width + previous_x]),
                        f64::from(current.pixels[rounded_y as usize * width + rounded_x as usize]),
                    );
                }
            }
            let Some(correlation) = accumulator.correlation(5.0) else {
                continue;
            };
            if best.is_none_or(|(best_correlation, _, _, _)| correlation > best_correlation) {
                best = Some((correlation, offset_x, offset_y, accumulator));
            }
        }
    }
    let (correlation, offset_x, offset_y, best_accumulator) = best?;
    if correlation < APPEARANCE_DIRECT_MIN_CORRELATION {
        return None;
    }

    let mut blocks = [CorrelationAccumulator::default(); BLOCK_COUNT];
    let mut previous_edge_count = 0_usize;
    let mut overlapping_edge_count = 0_usize;
    let mut aligned_edge_count = 0_usize;
    let mut edge_dot = 0.0;
    let mut previous_edge_squared = 0.0;
    let mut current_edge_squared = 0.0;
    let mut edge_quadrants = [false; 4];
    let normalization_count = best_accumulator.count as f64;
    let previous_mean = best_accumulator.left_sum / normalization_count;
    let current_mean = best_accumulator.right_sum / normalization_count;
    let previous_standard_deviation = (best_accumulator.left_squared_sum / normalization_count
        - previous_mean.powi(2))
    .max(0.0)
    .sqrt();
    let current_standard_deviation = (best_accumulator.right_squared_sum / normalization_count
        - current_mean.powi(2))
    .max(0.0)
    .sqrt();
    let mut photometric_matches = 0_usize;
    let mut photometric_inliers = 0_usize;
    for previous_y in (4..height - 4).step_by(SAMPLE_STEP) {
        for previous_x in (4..width - 4).step_by(SAMPLE_STEP) {
            let Some((current_x, current_y, _)) = rotation_compensated_projection(
                previous,
                current.orientation,
                previous_x as f64,
                previous_y as f64,
                scene_depth_metres,
                focal,
                center_x,
                center_y,
                offset_x as f64,
                offset_y as f64,
            ) else {
                continue;
            };
            let rounded_x = current_x.round() as isize;
            let rounded_y = current_y.round() as isize;
            if rounded_x < 0
                || rounded_y < 0
                || rounded_x >= width as isize
                || rounded_y >= height as isize
            {
                continue;
            }
            let column = (previous_x * BLOCK_COLUMNS / width).min(BLOCK_COLUMNS - 1);
            let row = (previous_y * BLOCK_ROWS / height).min(BLOCK_ROWS - 1);
            let previous_value = f64::from(previous.pixels[previous_y * width + previous_x]);
            let current_value =
                f64::from(current.pixels[rounded_y as usize * width + rounded_x as usize]);
            blocks[row * BLOCK_COLUMNS + column].push(previous_value, current_value);
            photometric_matches += 1;
            let normalized_previous =
                (previous_value - previous_mean) / previous_standard_deviation;
            let normalized_current = (current_value - current_mean) / current_standard_deviation;
            if (normalized_previous - normalized_current).abs() <= 0.75 {
                photometric_inliers += 1;
            }
            if rounded_x <= 0
                || rounded_y <= 0
                || rounded_x + 1 >= width as isize
                || rounded_y + 1 >= height as isize
            {
                continue;
            }
            let previous_gradient_x = f64::from(
                i16::from(previous.pixels[previous_y * width + previous_x + 1])
                    - i16::from(previous.pixels[previous_y * width + previous_x - 1]),
            );
            let previous_gradient_y = f64::from(
                i16::from(previous.pixels[(previous_y + 1) * width + previous_x])
                    - i16::from(previous.pixels[(previous_y - 1) * width + previous_x]),
            );
            let current_index = rounded_y as usize * width + rounded_x as usize;
            let current_gradient_x = f64::from(
                i16::from(current.pixels[current_index + 1])
                    - i16::from(current.pixels[current_index - 1]),
            );
            let current_gradient_y = f64::from(
                i16::from(current.pixels[current_index + width])
                    - i16::from(current.pixels[current_index - width]),
            );
            let previous_magnitude_squared =
                previous_gradient_x.powi(2) + previous_gradient_y.powi(2);
            if previous_magnitude_squared < 64.0 {
                continue;
            }
            previous_edge_count += 1;
            let current_magnitude_squared = current_gradient_x.powi(2) + current_gradient_y.powi(2);
            if current_magnitude_squared < 64.0 {
                continue;
            }
            overlapping_edge_count += 1;
            let dot =
                previous_gradient_x * current_gradient_x + previous_gradient_y * current_gradient_y;
            edge_dot += dot;
            previous_edge_squared += previous_magnitude_squared;
            current_edge_squared += current_magnitude_squared;
            let local_similarity =
                dot / (previous_magnitude_squared * current_magnitude_squared).sqrt();
            if local_similarity >= 0.5 {
                aligned_edge_count += 1;
                let quadrant_column = usize::from(previous_x >= width / 2);
                let quadrant_row = usize::from(previous_y >= height / 2);
                edge_quadrants[quadrant_row * 2 + quadrant_column] = true;
            }
        }
    }

    let mut verified_features = Vec::with_capacity(BLOCK_COUNT);
    let mut matches = 0_usize;
    for (index, block) in blocks.into_iter().enumerate() {
        let Some(block_correlation) = block.correlation(4.0) else {
            continue;
        };
        matches += 1;
        if block_correlation < 0.5 {
            continue;
        }
        let column = index % BLOCK_COLUMNS;
        let row = index / BLOCK_COLUMNS;
        let previous_x = (column * 2 + 1) as f64 * width as f64 / (BLOCK_COLUMNS * 2) as f64;
        let previous_y = (row * 2 + 1) as f64 * height as f64 / (BLOCK_ROWS * 2) as f64;
        if let Some((current_x, current_y, base_current_camera)) = rotation_compensated_projection(
            previous,
            current.orientation,
            previous_x,
            previous_y,
            scene_depth_metres,
            focal,
            center_x,
            center_y,
            offset_x as f64,
            offset_y as f64,
        ) {
            let _ = base_current_camera;
            verified_features.push(TrackedFeature {
                observed_x: (current_x - center_x) / focal,
                observed_y: (current_y - center_y) / focal,
            });
        }
    }
    let block_coverage = feature_coverage(&verified_features);
    let blocks_verified = matches >= APPEARANCE_DIRECT_MIN_BLOCK_INLIERS
        && verified_features.len() >= APPEARANCE_DIRECT_MIN_BLOCK_INLIERS
        && verified_features.len() * 2 >= matches
        && block_coverage >= 2;
    let edge_similarity = if previous_edge_squared > 0.0 && current_edge_squared > 0.0 {
        edge_dot / (previous_edge_squared * current_edge_squared).sqrt()
    } else {
        0.0
    };
    let edge_coverage = edge_quadrants.into_iter().filter(|value| *value).count();
    let edges_verified = previous_edge_count >= 20
        && overlapping_edge_count * 2 >= previous_edge_count
        && aligned_edge_count >= 12
        && aligned_edge_count * 2 >= overlapping_edge_count
        && edge_similarity >= 0.35
        && edge_coverage >= 2;
    let spatially_verified = blocks_verified || edges_verified;
    if !spatially_verified && correlation < APPEARANCE_DIRECT_SMOOTH_MIN_CORRELATION {
        return None;
    }
    let camera_delta_current_three = DVec3::new(
        -(offset_x as f64) * scene_depth_metres / focal,
        (offset_y as f64) * scene_depth_metres / focal,
        0.0,
    );
    let mut camera_delta_world = current.orientation * camera_delta_current_three;
    camera_delta_world.y *= VISUAL_VERTICAL_TRANSLATION_GAIN;
    if !camera_delta_world.is_finite()
        || camera_delta_world.length() > APPEARANCE_DIRECT_MAX_TRANSLATION_METRES
    {
        return None;
    }
    Some((
        correlation,
        (
            camera_delta_world,
            matches.max(overlapping_edge_count).max(photometric_matches),
            verified_features
                .len()
                .max(aligned_edge_count)
                .max(photometric_inliers),
        ),
        spatially_verified,
    ))
}

#[allow(clippy::too_many_arguments)]
fn rotation_compensated_projection(
    previous: &RelocView<'_>,
    current_orientation: DQuat,
    previous_x: f64,
    previous_y: f64,
    scene_depth_metres: f64,
    focal: f64,
    center_x: f64,
    center_y: f64,
    offset_x: f64,
    offset_y: f64,
) -> Option<(f64, f64, DVec3)> {
    let previous_ray_cv = DVec3::new(
        (previous_x - center_x) / focal,
        (previous_y - center_y) / focal,
        1.0,
    );
    let previous_point_three = DVec3::new(
        previous_ray_cv.x * scene_depth_metres,
        -previous_ray_cv.y * scene_depth_metres,
        -scene_depth_metres,
    );
    let point_world = previous.orientation * previous_point_three;
    let base_three = current_orientation.conjugate() * point_world;
    let base_current_camera = DVec3::new(base_three.x, -base_three.y, -base_three.z);
    if base_current_camera.z <= 0.1 {
        return None;
    }
    Some((
        focal * base_current_camera.x / base_current_camera.z + center_x + offset_x,
        focal * base_current_camera.y / base_current_camera.z + center_y + offset_y,
        base_current_camera,
    ))
}
