//! Browser-independent data contracts for visual-inertial tracking.
//!
//! This crate deliberately knows nothing about browser events, JavaScript clocks, camera
//! acquisition, or rendering. An acquisition adapter is responsible for converting raw platform
//! values into the canonical coordinate system and units documented by
//! [`NormalizedMotionSample`] before passing them into this crate.

#![forbid(unsafe_code)]

mod frame;
mod preintegration;
mod sensor;
mod time;

pub use frame::{
    CalibrationRevision, CameraCalibration, CameraFrameMetadata, FrameId, FrameMetadataError,
    FrameTimestamps, ImageOrientation, ImageTransform, NormalizedRect, PixelDimensions,
};
pub use glam::{DQuat, DVec2, DVec3};
pub use preintegration::{
    ImuBias, NavState, PreintegratedImu, PreintegrationConfig, PreintegrationDiagnostics,
    PreintegrationError, PreintegrationUncertainty, preintegrate,
};
pub use sensor::{
    BatchDiagnostics, BufferError, BufferStats, FrameInterval, NormalizedMotionSample, PushOutcome,
    SampleError, SensorBatch, SensorBuffer, SensorSequence, SensorTimeBasis,
    SourceScreenOrientation,
};
pub use time::{MonotonicDuration, MonotonicTimestamp, TimeValueError};
