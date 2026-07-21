//! Decoding for the primary keypoint observation network.

use burn::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    HEATMAP_SIZE, KEYPOINT_COUNT, KeypointRoofOutput, LetterboxTransform, POINTS_PER_RING,
    ROOF_RING_COUNT,
};

/// Default minimum categorical mass for an in-frame point passed to the
/// constrained roof fitter. Training acceptance and runtime inference share
/// this value unless a promoted model manifest supplies an override.
pub const DEFAULT_FIT_KEYPOINT_CONFIDENCE: f32 = 0.15;
/// Default posterior threshold for the mutually exclusive offscreen state.
///
/// The other 4,096 categorical values jointly represent the in-frame state,
/// so the offscreen token must be compared with their summed probability, not
/// with the probability of one spatial cell.
pub const DEFAULT_OFFSCREEN_THRESHOLD: f32 = 0.5;

/// Normalized source-image bounds derived from accepted keypoints.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BoundingBoxPrediction {
    /// Minimum horizontal coordinate.
    pub min_x: f32,
    /// Minimum vertical coordinate.
    pub min_y: f32,
    /// Maximum horizontal coordinate.
    pub max_x: f32,
    /// Maximum vertical coordinate.
    pub max_y: f32,
}

/// One of the three rectangular control rings in the two-tier roof mesh.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RoofRing {
    /// Outer eave perimeter at the bottom of the shallow lower skirt.
    Eave,
    /// Pitch-break perimeter shared by the lower skirt and upper crown.
    Shoulder,
    /// Perimeter of the flat crown top.
    Crown,
}

impl RoofRing {
    /// Rings in model-channel order.
    pub const ALL: [Self; ROOF_RING_COUNT] = [Self::Eave, Self::Shoulder, Self::Crown];

    /// Stable serialized/debugging name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Eave => "eave",
            Self::Shoulder => "shoulder",
            Self::Crown => "crown",
        }
    }
}

/// One amodal corner slot decoded from a heatmap and offscreen classifier.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AmodalKeypointPrediction {
    /// Cyclic slot within the ring, in the range `0..4`.
    ///
    /// This is not a physical front/left identity. The geometric fitter must
    /// resolve cyclic rotation and reflection consistently across all rings.
    pub slot: usize,
    /// Normalized source-image coordinate, absent when classified offscreen.
    pub position: Option<[f32; 2]>,
    /// Joint in-frame confidence: probability mass in the peak's 5x5
    /// neighbourhood, or zero when the point falls outside source content.
    pub confidence: f32,
    /// Probability assigned by the independent offscreen classifier.
    pub offscreen_probability: f32,
    /// Whether the model selected the offscreen token or the decoded spatial
    /// peak fell in letterbox padding/outside the source image.
    pub offscreen: bool,
}

/// Four cyclic corner slots belonging to one structural roof ring.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoofRingPrediction {
    /// Which parametric roof ring these points describe.
    pub ring: RoofRing,
    /// Four cyclic slots; rotation and reflection remain intentionally open.
    pub points: [AmodalKeypointPrediction; POINTS_PER_RING],
}

/// Decoded output for one source image.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeypointRoofDetection {
    /// Versioned output contract.
    pub schema_version: String,
    /// Probability that a current or former Pizza Hut two-tier roof is present.
    pub probability: f32,
    /// Presence threshold applied by the caller.
    pub threshold: f32,
    /// Whether the presence probability met the threshold.
    pub detected: bool,
    /// Bounds derived from accepted in-frame keypoints, not a learned box.
    pub bounding_box: Option<BoundingBoxPrediction>,
    /// Eave, shoulder, and crown rings in stable order.
    pub rings: [RoofRingPrediction; ROOF_RING_COUNT],
}

/// Decodes one model forward pass into source-image ring observations.
///
/// Each heatmap's 4,096 cells and its offscreen logit are normalized as one
/// 4,097-way categorical distribution, matching the training objective. The
/// offscreen token is selected when its posterior reaches
/// `offscreen_threshold`. The spatial cells partition the alternative
/// in-frame event, so comparing the token only with the highest individual
/// cell would incorrectly reject broad but confident spatial distributions.
///
/// The function deliberately accepts one image. Batched decoding should split
/// the raw tensors first so each result keeps its own letterbox transform.
#[must_use]
pub fn decode_keypoint_prediction<B: Backend>(
    output: KeypointRoofOutput<B>,
    transform: LetterboxTransform,
    presence_threshold: f32,
    offscreen_threshold: f32,
) -> KeypointRoofDetection {
    let [batch] = output.presence_logits.dims();
    assert_eq!(batch, 1, "decode_keypoint_prediction expects one image");
    assert_eq!(
        output.keypoint_logits.dims(),
        [1, KEYPOINT_COUNT, HEATMAP_SIZE, HEATMAP_SIZE],
        "keypoint output contract changed"
    );
    assert_eq!(
        output.offscreen_logits.dims(),
        [1, KEYPOINT_COUNT],
        "offscreen output contract changed"
    );

    let presence_logits = output
        .presence_logits
        .to_data()
        .to_vec::<f32>()
        .expect("presence tensor must be f32");
    let probability = logistic(presence_logits[0]);
    let heatmaps = output
        .keypoint_logits
        .to_data()
        .to_vec::<f32>()
        .expect("keypoint tensor must be f32");
    let offscreen_logits = output
        .offscreen_logits
        .to_data()
        .to_vec::<f32>()
        .expect("offscreen tensor must be f32");

    let rings = std::array::from_fn(|ring_index| RoofRingPrediction {
        ring: RoofRing::ALL[ring_index],
        points: std::array::from_fn(|slot| {
            let channel_index = ring_index * POINTS_PER_RING + slot;
            let distribution = local_peak(
                channel(&heatmaps, channel_index),
                offscreen_logits[channel_index],
            );
            let model_offscreen = distribution.offscreen_probability >= offscreen_threshold;
            let source_position = transform.unmap_point_if_in_source(distribution.point);
            let offscreen = model_offscreen || source_position.is_none();
            AmodalKeypointPrediction {
                slot,
                position: (!offscreen).then_some(source_position).flatten(),
                confidence: if offscreen {
                    0.0
                } else {
                    distribution.local_probability
                },
                offscreen_probability: distribution.offscreen_probability,
                offscreen,
            }
        }),
    });
    let bounding_box = point_bounds(&rings);

    KeypointRoofDetection {
        schema_version: "keypoint-roof-detection/v1".to_owned(),
        probability,
        threshold: presence_threshold,
        detected: probability >= presence_threshold,
        bounding_box,
        rings,
    }
}

fn channel(values: &[f32], index: usize) -> &[f32] {
    let area = HEATMAP_SIZE * HEATMAP_SIZE;
    &values[index * area..(index + 1) * area]
}

struct KeypointDistribution {
    point: [f32; 2],
    local_probability: f32,
    offscreen_probability: f32,
}

fn local_peak(logits: &[f32], offscreen_logit: f32) -> KeypointDistribution {
    let (peak_index, peak_logit) = logits
        .iter()
        .copied()
        .enumerate()
        .max_by(|left, right| left.1.total_cmp(&right.1))
        .expect("heatmap cannot be empty");
    let peak_x = peak_index % HEATMAP_SIZE;
    let peak_y = peak_index / HEATMAP_SIZE;
    let maximum = peak_logit.max(offscreen_logit);

    let mut weighted = [0.0_f32; 2];
    let mut local_weight_sum = 0.0;
    for y in peak_y.saturating_sub(2)..=(peak_y + 2).min(HEATMAP_SIZE - 1) {
        for x in peak_x.saturating_sub(2)..=(peak_x + 2).min(HEATMAP_SIZE - 1) {
            let weight = (logits[y * HEATMAP_SIZE + x] - maximum).exp();
            weighted[0] += weight * (x as f32 + 0.5);
            weighted[1] += weight * (y as f32 + 0.5);
            local_weight_sum += weight;
        }
    }
    let spatial_weight_sum = logits
        .iter()
        .map(|value| (*value - maximum).exp())
        .sum::<f32>();
    let offscreen_weight = (offscreen_logit - maximum).exp();
    let categorical_weight_sum = spatial_weight_sum + offscreen_weight;
    KeypointDistribution {
        point: [
            weighted[0] / local_weight_sum / HEATMAP_SIZE as f32,
            weighted[1] / local_weight_sum / HEATMAP_SIZE as f32,
        ],
        local_probability: local_weight_sum / categorical_weight_sum,
        offscreen_probability: offscreen_weight / categorical_weight_sum,
    }
}

fn point_bounds(rings: &[RoofRingPrediction; ROOF_RING_COUNT]) -> Option<BoundingBoxPrediction> {
    let mut points = rings
        .iter()
        .flat_map(|ring| ring.points.iter())
        .filter_map(|point| point.position);
    let first = points.next()?;
    Some(points.fold(
        BoundingBoxPrediction {
            min_x: first[0],
            min_y: first[1],
            max_x: first[0],
            max_y: first[1],
        },
        |mut bounds, point| {
            bounds.min_x = bounds.min_x.min(point[0]);
            bounds.min_y = bounds.min_y.min(point[1]);
            bounds.max_x = bounds.max_x.max(point[0]);
            bounds.max_y = bounds.max_y.max(point[1]);
            bounds
        },
    ))
}

fn logistic(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    fn output_with_peak(x: usize, y: usize) -> KeypointRoofOutput<TestBackend> {
        let device: burn::tensor::Device<TestBackend> = Default::default();
        let mut heatmaps = vec![-10.0; KEYPOINT_COUNT * HEATMAP_SIZE * HEATMAP_SIZE];
        for channel_index in 0..KEYPOINT_COUNT {
            heatmaps[channel_index * HEATMAP_SIZE * HEATMAP_SIZE + y * HEATMAP_SIZE + x] = 10.0;
        }
        KeypointRoofOutput {
            presence_logits: Tensor::from_floats([4.0], &device),
            keypoint_logits: Tensor::from_data(
                TensorData::new(
                    heatmaps,
                    Shape::new([1, KEYPOINT_COUNT, HEATMAP_SIZE, HEATMAP_SIZE]),
                ),
                &device,
            ),
            offscreen_logits: Tensor::from_data(
                TensorData::new(vec![-10.0; KEYPOINT_COUNT], Shape::new([1, KEYPOINT_COUNT])),
                &device,
            ),
        }
    }

    #[test]
    fn decoder_groups_channels_and_omits_offscreen_positions() {
        let device: burn::tensor::Device<TestBackend> = Default::default();
        let mut heatmaps = vec![-10.0; KEYPOINT_COUNT * HEATMAP_SIZE * HEATMAP_SIZE];
        for channel_index in 0..KEYPOINT_COUNT {
            let x = 8 + channel_index;
            let y = 28 + channel_index;
            heatmaps[channel_index * HEATMAP_SIZE * HEATMAP_SIZE + y * HEATMAP_SIZE + x] = 10.0;
        }
        let mut offscreen = vec![-10.0; KEYPOINT_COUNT];
        offscreen[5] = 12.0;
        let output: KeypointRoofOutput<TestBackend> = KeypointRoofOutput {
            presence_logits: Tensor::from_floats([4.0], &device),
            keypoint_logits: Tensor::from_data(
                TensorData::new(
                    heatmaps,
                    Shape::new([1, KEYPOINT_COUNT, HEATMAP_SIZE, HEATMAP_SIZE]),
                ),
                &device,
            ),
            offscreen_logits: Tensor::from_data(
                TensorData::new(offscreen, Shape::new([1, KEYPOINT_COUNT])),
                &device,
            ),
        };

        let detection = decode_keypoint_prediction(
            output,
            LetterboxTransform::for_input_size(200, 100, crate::SPATIAL_INPUT_SIZE),
            0.5,
            0.5,
        );
        assert!(detection.detected);
        assert_eq!(
            std::array::from_fn(|index| detection.rings[index].ring),
            RoofRing::ALL
        );
        assert_eq!(detection.rings[1].points[1].slot, 1);
        assert!(detection.rings[1].points[1].offscreen);
        assert_eq!(detection.rings[1].points[1].position, None);
        assert!(detection.rings[1].points[1].offscreen_probability > 0.85);
        assert!(detection.rings[0].points[0].confidence > 0.99);
        assert!(detection.bounding_box.is_some());
    }

    #[test]
    fn offscreen_state_compares_against_total_spatial_probability() {
        let device: burn::tensor::Device<TestBackend> = Default::default();
        let heatmaps = vec![0.0; KEYPOINT_COUNT * HEATMAP_SIZE * HEATMAP_SIZE];
        // This token beats every individual spatial cell, but carries far less
        // than half of the normalized categorical probability because all
        // spatial cells together represent the in-frame alternative.
        let offscreen = vec![1.0; KEYPOINT_COUNT];
        let output: KeypointRoofOutput<TestBackend> = KeypointRoofOutput {
            presence_logits: Tensor::from_floats([4.0], &device),
            keypoint_logits: Tensor::from_data(
                TensorData::new(
                    heatmaps,
                    Shape::new([1, KEYPOINT_COUNT, HEATMAP_SIZE, HEATMAP_SIZE]),
                ),
                &device,
            ),
            offscreen_logits: Tensor::from_data(
                TensorData::new(offscreen, Shape::new([1, KEYPOINT_COUNT])),
                &device,
            ),
        };

        let detection = decode_keypoint_prediction(
            output,
            LetterboxTransform::for_input_size(256, 256, crate::SPATIAL_INPUT_SIZE),
            0.5,
            DEFAULT_OFFSCREEN_THRESHOLD,
        );
        for point in detection.rings.iter().flat_map(|ring| ring.points.iter()) {
            assert!(!point.offscreen);
            assert!(point.position.is_some());
            assert!(point.offscreen_probability < DEFAULT_OFFSCREEN_THRESHOLD);
        }
    }

    #[test]
    fn landscape_letterbox_peaks_are_not_clamped_to_image_border() {
        let detection = decode_keypoint_prediction(
            output_with_peak(32, 4),
            LetterboxTransform::for_input_size(200, 100, crate::SPATIAL_INPUT_SIZE),
            0.5,
            0.5,
        );
        for point in detection.rings.iter().flat_map(|ring| ring.points.iter()) {
            assert!(point.offscreen);
            assert_eq!(point.position, None);
            assert_eq!(point.confidence, 0.0);
            assert!(point.offscreen_probability < 0.01);
        }
        assert_eq!(detection.bounding_box, None);
    }

    #[test]
    fn portrait_letterbox_peaks_are_not_clamped_to_image_border() {
        let detection = decode_keypoint_prediction(
            output_with_peak(4, 32),
            LetterboxTransform::for_input_size(100, 200, crate::SPATIAL_INPUT_SIZE),
            0.5,
            0.5,
        );
        for point in detection.rings.iter().flat_map(|ring| ring.points.iter()) {
            assert!(point.offscreen);
            assert_eq!(point.position, None);
            assert_eq!(point.confidence, 0.0);
            assert!(point.offscreen_probability < 0.01);
        }
        assert_eq!(detection.bounding_box, None);
    }

    #[test]
    fn all_offscreen_points_have_no_bounds() {
        let rings = std::array::from_fn(|ring_index| RoofRingPrediction {
            ring: RoofRing::ALL[ring_index],
            points: std::array::from_fn(|slot| AmodalKeypointPrediction {
                slot,
                position: None,
                confidence: 0.0,
                offscreen_probability: 1.0,
                offscreen: true,
            }),
        });
        assert_eq!(point_bounds(&rings), None);
    }
}
