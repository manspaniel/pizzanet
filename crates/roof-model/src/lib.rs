//! Burn network and portable contracts for one-frame roof recognition.
//!
//! The crate owns no camera, browser, or renderer integration. It accepts a
//! normalized RGB tensor and predicts the 2D structural observations consumed
//! by the roof fitter and diagnostic overlay.

mod keypoint;
mod keypoint_prediction;
mod mobilenet;
mod preprocess;

pub use keypoint::{
    HEATMAP_SIZE, KEYPOINT_COUNT, KeypointParameterGroups, KeypointRoofGeometryOutput,
    KeypointRoofNet, KeypointRoofNetConfig, KeypointRoofOutput, KeypointTrainingOptions,
    POINTS_PER_RING, ROOF_RING_COUNT, SPATIAL_INPUT_SIZE,
};
pub use keypoint_prediction::{
    AmodalKeypointPrediction, BoundingBoxPrediction, DEFAULT_FIT_KEYPOINT_CONFIDENCE,
    DEFAULT_OFFSCREEN_THRESHOLD, KeypointRoofDetection, RoofRing, RoofRingPrediction,
    decode_keypoint_prediction,
};
pub use preprocess::{
    INPUT_SIZE, LetterboxTransform, PrepareError, PreparedInput, prepare_rgb8, prepare_rgb8_sized,
};
