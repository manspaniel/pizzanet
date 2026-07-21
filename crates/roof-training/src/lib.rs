//! Converts exact synthetic records into sparse keypoint-distribution targets.
//!
//! The runtime network predicts the twelve amodal structural corners of the
//! three roof rings. Dense renderer products such as masks and edge maps remain
//! useful for validation, but they are deliberately not neural outputs.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;

use roof_geometry::KeypointId;
use roof_model::{HEATMAP_SIZE, KEYPOINT_COUNT, LetterboxTransform};
use synth_data::{FrameRecord, Visibility};
use thiserror::Error;

/// Extra categorical state appended to every spatial heatmap.
pub const OFFSCREEN_INDEX: usize = HEATMAP_SIZE * HEATMAP_SIZE;
/// Number of categorical values per keypoint, including the offscreen state.
pub const KEYPOINT_DISTRIBUTION_SIZE: usize = OFFSCREEN_INDEX + 1;
/// Number of cyclic and reflected correspondences of a rectangular ring.
pub const SYMMETRY_COUNT: usize = 8;

const KEYPOINT_SIGMA: f32 = 1.45;

/// Spatial supervision for one target-positive synthetic frame.
#[derive(Clone, Debug, PartialEq)]
pub struct KeypointTrainingTarget {
    /// Twelve normalized categorical distributions in stable geometry order.
    /// Layout is `[KEYPOINT_COUNT, KEYPOINT_DISTRIBUTION_SIZE]`.
    pub distributions: Vec<f32>,
    /// Exact letterboxed normalized coordinates, used by evaluation metrics.
    pub positions: [[f32; 2]; KEYPOINT_COUNT],
    /// Whether the corresponding point is projectable inside the retained image.
    pub in_frame: [bool; KEYPOINT_COUNT],
}

/// Invalid or incomplete synthetic supervision.
#[derive(Debug, Error)]
pub enum TargetError {
    /// A required stable keypoint was absent.
    #[error("missing structural keypoint {0}")]
    MissingKeypoint(&'static str),
}

/// Builds twelve Gaussian spatial distributions plus explicit offscreen states.
///
/// Visible and occluded points both receive their exact amodal projection.
/// Truncated and behind-camera points train the offscreen state. Physical
/// front/back identities are retained only as a source ordering; the trainer
/// minimizes over all eight shared ring correspondences.
pub fn build_keypoint_target(
    frame: &FrameRecord,
    transform: LetterboxTransform,
) -> Result<KeypointTrainingTarget, TargetError> {
    let labels = frame
        .labels
        .keypoints
        .iter()
        .map(|point| (u32::from(point.instance_id), point))
        .collect::<BTreeMap<_, _>>();
    let mut distributions = vec![0.0; KEYPOINT_COUNT * KEYPOINT_DISTRIBUTION_SIZE];
    let mut positions = [[0.0; 2]; KEYPOINT_COUNT];
    let mut in_frame = [false; KEYPOINT_COUNT];

    for (index, id) in KeypointId::ALL.iter().enumerate() {
        let label = labels
            .get(&id.as_u32())
            .ok_or(TargetError::MissingKeypoint(id.as_str()))?;
        let offset = index * KEYPOINT_DISTRIBUTION_SIZE;
        let mapped = label
            .image_position
            .map(|position| transform.map_point([position.x, position.y]));
        let usable = matches!(label.visibility, Visibility::Visible | Visibility::Occluded)
            && mapped.is_some_and(inside_unit);
        if usable {
            let point = mapped.expect("usable projection has coordinates");
            positions[index] = point;
            in_frame[index] = true;
            draw_normalized_gaussian(
                &mut distributions[offset..offset + OFFSCREEN_INDEX],
                [
                    point[0] * HEATMAP_SIZE as f32,
                    point[1] * HEATMAP_SIZE as f32,
                ],
                KEYPOINT_SIGMA,
            );
        } else {
            distributions[offset + OFFSCREEN_INDEX] = 1.0;
        }
    }

    Ok(KeypointTrainingTarget {
        distributions,
        positions,
        in_frame,
    })
}

/// Maps one predicted ring slot to the target slot for a dihedral hypothesis.
///
/// Hypotheses zero through three are cyclic rotations. Four through seven are
/// reflected rotations. The same mapping is applied to eave, shoulder, and
/// crown rings so their vertical correspondences remain coherent.
#[must_use]
pub const fn symmetry_target_slot(hypothesis: usize, predicted_slot: usize) -> usize {
    let shift = hypothesis % 4;
    if hypothesis < 4 {
        (predicted_slot + shift) % 4
    } else {
        (shift + 4 - predicted_slot) % 4
    }
}

/// Copies a base target into `[8, 12, 4097]` symmetry-hypothesis order.
#[must_use]
pub fn symmetry_distributions(base: &[f32]) -> Vec<f32> {
    assert_eq!(
        base.len(),
        KEYPOINT_COUNT * KEYPOINT_DISTRIBUTION_SIZE,
        "invalid base keypoint target"
    );
    let mut output = vec![0.0; SYMMETRY_COUNT * base.len()];
    for hypothesis in 0..SYMMETRY_COUNT {
        for predicted_index in 0..KEYPOINT_COUNT {
            let ring = predicted_index / 4;
            let slot = predicted_index % 4;
            let target_index = ring * 4 + symmetry_target_slot(hypothesis, slot);
            let source = &base[target_index * KEYPOINT_DISTRIBUTION_SIZE
                ..(target_index + 1) * KEYPOINT_DISTRIBUTION_SIZE];
            let destination_offset =
                (hypothesis * KEYPOINT_COUNT + predicted_index) * KEYPOINT_DISTRIBUTION_SIZE;
            output[destination_offset..destination_offset + KEYPOINT_DISTRIBUTION_SIZE]
                .copy_from_slice(source);
        }
    }
    output
}

/// Mirrors a target horizontally in the model-input coordinate system.
///
/// The heatmap channels do not need to be renamed because the training loss is
/// correspondence invariant; applying the same pixel transform to all rings is
/// sufficient.
pub fn flip_target_horizontal(target: &mut KeypointTrainingTarget) {
    for keypoint in 0..KEYPOINT_COUNT {
        let offset = keypoint * KEYPOINT_DISTRIBUTION_SIZE;
        let map = &mut target.distributions[offset..offset + OFFSCREEN_INDEX];
        for y in 0..HEATMAP_SIZE {
            let row = &mut map[y * HEATMAP_SIZE..(y + 1) * HEATMAP_SIZE];
            row.reverse();
        }
        if target.in_frame[keypoint] {
            target.positions[keypoint][0] = 1.0 - target.positions[keypoint][0];
        }
    }
}

fn draw_normalized_gaussian(target: &mut [f32], center: [f32; 2], sigma: f32) {
    debug_assert_eq!(target.len(), OFFSCREEN_INDEX);
    let radius = (sigma * 4.0).ceil() as i32;
    let center_x = center[0].floor() as i32;
    let center_y = center[1].floor() as i32;
    let mut total = 0.0;
    for y in center_y - radius..=center_y + radius {
        for x in center_x - radius..=center_x + radius {
            if x < 0 || y < 0 || x >= HEATMAP_SIZE as i32 || y >= HEATMAP_SIZE as i32 {
                continue;
            }
            let dx = x as f32 + 0.5 - center[0];
            let dy = y as f32 + 0.5 - center[1];
            let value = (-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).exp();
            target[y as usize * HEATMAP_SIZE + x as usize] = value;
            total += value;
        }
    }
    assert!(
        total > 0.0,
        "in-frame Gaussian must contain probability mass"
    );
    for value in target {
        *value /= total;
    }
}

fn inside_unit(point: [f32; 2]) -> bool {
    point[0].is_finite()
        && point[1].is_finite()
        && (0.0..1.0).contains(&point[0])
        && (0.0..1.0).contains(&point[1])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_is_a_probability_distribution() {
        let mut target = vec![0.0; OFFSCREEN_INDEX];
        draw_normalized_gaussian(&mut target, [20.5, 30.5], 1.0);
        let maximum = target
            .iter()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(right.1))
            .unwrap();
        assert_eq!(maximum.0, 30 * HEATMAP_SIZE + 20);
        assert!((target.iter().sum::<f32>() - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn correspondence_hypotheses_cover_rotations_and_reflections() {
        let mappings = (0..SYMMETRY_COUNT)
            .map(|hypothesis| {
                (0..4)
                    .map(|slot| symmetry_target_slot(hypothesis, slot))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(mappings[0], [0, 1, 2, 3]);
        assert_eq!(mappings[1], [1, 2, 3, 0]);
        assert_eq!(mappings[4], [0, 3, 2, 1]);
        assert_eq!(mappings[7], [3, 2, 1, 0]);
        let mut unique = mappings.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(unique.len(), SYMMETRY_COUNT);
    }

    #[test]
    fn symmetry_expansion_uses_one_mapping_for_every_ring() {
        let mut base = vec![0.0; KEYPOINT_COUNT * KEYPOINT_DISTRIBUTION_SIZE];
        for index in 0..KEYPOINT_COUNT {
            base[index * KEYPOINT_DISTRIBUTION_SIZE] = index as f32;
        }
        let expanded = symmetry_distributions(&base);
        let value = |hypothesis: usize, index: usize| {
            expanded[(hypothesis * KEYPOINT_COUNT + index) * KEYPOINT_DISTRIBUTION_SIZE]
        };
        for ring in 0..3 {
            assert_eq!(value(5, ring * 4), (ring * 4 + 1) as f32);
            assert_eq!(value(5, ring * 4 + 1), (ring * 4) as f32);
            assert_eq!(value(5, ring * 4 + 2), (ring * 4 + 3) as f32);
            assert_eq!(value(5, ring * 4 + 3), (ring * 4 + 2) as f32);
        }
    }

    #[test]
    fn horizontal_flip_updates_maps_and_metric_positions() {
        let mut distributions = vec![0.0; KEYPOINT_COUNT * KEYPOINT_DISTRIBUTION_SIZE];
        distributions[4] = 1.0;
        let mut target = KeypointTrainingTarget {
            distributions,
            positions: [[0.0; 2]; KEYPOINT_COUNT],
            in_frame: [false; KEYPOINT_COUNT],
        };
        target.positions[0] = [0.1, 0.2];
        target.in_frame[0] = true;
        flip_target_horizontal(&mut target);
        assert_eq!(target.distributions[HEATMAP_SIZE - 1 - 4], 1.0);
        assert!((target.positions[0][0] - 0.9).abs() < 1.0e-6);
        assert_eq!(target.positions[0][1], 0.2);
    }
}
