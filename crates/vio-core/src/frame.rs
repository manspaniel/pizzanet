use crate::{DQuat, DVec2, MonotonicTimestamp};
use serde::{Deserialize, Serialize};
use std::num::NonZeroU32;
use thiserror::Error;

/// A monotonically increasing camera-frame identifier within one capture epoch.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct FrameId(u64);

impl FrameId {
    /// Constructs an identifier.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the underlying identifier.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Identifies the immutable calibration snapshot used to interpret a frame.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct CalibrationRevision(u64);

impl CalibrationRevision {
    /// Constructs a revision identifier.
    #[must_use]
    pub const fn new(value: u64) -> Self {
        Self(value)
    }

    /// Returns the underlying revision number.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// Non-zero pixel dimensions.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PixelDimensions {
    width: NonZeroU32,
    height: NonZeroU32,
}

impl PixelDimensions {
    /// Constructs validated dimensions.
    pub fn new(width: u32, height: u32) -> Result<Self, FrameMetadataError> {
        Ok(Self {
            width: NonZeroU32::new(width).ok_or(FrameMetadataError::ZeroDimensions)?,
            height: NonZeroU32::new(height).ok_or(FrameMetadataError::ZeroDimensions)?,
        })
    }

    /// Returns the width in pixels.
    #[must_use]
    pub const fn width(self) -> u32 {
        self.width.get()
    }

    /// Returns the height in pixels.
    #[must_use]
    pub const fn height(self) -> u32 {
        self.height.get()
    }
}

/// A crop rectangle in normalized source-image coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct NormalizedRect {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
}

impl NormalizedRect {
    /// The full source image.
    pub const FULL: Self = Self {
        x: 0.0,
        y: 0.0,
        width: 1.0,
        height: 1.0,
    };

    /// Constructs a crop contained by `[0, 1] x [0, 1]`.
    pub fn new(x: f64, y: f64, width: f64, height: f64) -> Result<Self, FrameMetadataError> {
        let values = [x, y, width, height];
        if values.iter().any(|value| !value.is_finite()) {
            return Err(FrameMetadataError::NonFiniteValue);
        }
        if x < 0.0
            || y < 0.0
            || width <= 0.0
            || height <= 0.0
            || x + width > 1.0
            || y + height > 1.0
        {
            return Err(FrameMetadataError::InvalidCrop);
        }

        Ok(Self {
            x,
            y,
            width,
            height,
        })
    }

    /// Returns the left coordinate.
    #[must_use]
    pub const fn x(self) -> f64 {
        self.x
    }

    /// Returns the top coordinate.
    #[must_use]
    pub const fn y(self) -> f64 {
        self.y
    }

    /// Returns the normalized width.
    #[must_use]
    pub const fn width(self) -> f64 {
        self.width
    }

    /// Returns the normalized height.
    #[must_use]
    pub const fn height(self) -> f64 {
        self.height
    }
}

/// Clockwise image rotation required to display the captured pixels upright.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageOrientation {
    /// No rotation.
    #[default]
    Degrees0,
    /// Ninety degrees clockwise.
    Degrees90,
    /// One hundred and eighty degrees.
    Degrees180,
    /// Two hundred and seventy degrees clockwise.
    Degrees270,
}

/// A projective transform from source-pixel coordinates to model-input pixel coordinates.
///
/// The matrix is stored in row-major order and multiplies the homogeneous column vector
/// `[source_x, source_y, 1]`. Keeping this exact transform with a frame prevents crop, rotation,
/// resize, and mirroring decisions from becoming implicit mutable state.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImageTransform {
    source_dimensions: PixelDimensions,
    output_dimensions: PixelDimensions,
    source_to_output: [f64; 9],
}

impl ImageTransform {
    /// Constructs and validates a non-singular finite projective transform.
    pub fn new(
        source_dimensions: PixelDimensions,
        output_dimensions: PixelDimensions,
        source_to_output: [f64; 9],
    ) -> Result<Self, FrameMetadataError> {
        if source_to_output.iter().any(|value| !value.is_finite()) {
            return Err(FrameMetadataError::NonFiniteValue);
        }

        let [a, b, c, d, e, f, g, h, i] = source_to_output;
        let determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
        if determinant.abs() <= f64::EPSILON {
            return Err(FrameMetadataError::SingularImageTransform);
        }

        Ok(Self {
            source_dimensions,
            output_dimensions,
            source_to_output,
        })
    }

    /// Constructs an identity transform between equally sized images.
    #[must_use]
    pub const fn identity(dimensions: PixelDimensions) -> Self {
        Self {
            source_dimensions: dimensions,
            output_dimensions: dimensions,
            source_to_output: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Applies the homogeneous transform to one source pixel.
    #[must_use]
    pub fn transform_point(self, source_pixel: DVec2) -> Option<DVec2> {
        let [a, b, c, d, e, f, g, h, i] = self.source_to_output;
        let denominator = g * source_pixel.x + h * source_pixel.y + i;
        if !denominator.is_finite() || denominator.abs() <= f64::EPSILON {
            return None;
        }
        Some(DVec2::new(
            (a * source_pixel.x + b * source_pixel.y + c) / denominator,
            (d * source_pixel.x + e * source_pixel.y + f) / denominator,
        ))
    }

    /// Returns the source-image dimensions.
    #[must_use]
    pub const fn source_dimensions(self) -> PixelDimensions {
        self.source_dimensions
    }

    /// Returns the model-input dimensions.
    #[must_use]
    pub const fn output_dimensions(self) -> PixelDimensions {
        self.output_dimensions
    }

    /// Returns the row-major homography.
    #[must_use]
    pub const fn source_to_output(self) -> [f64; 9] {
        self.source_to_output
    }
}

/// Camera timestamps converted into one monotonic clock domain by the acquisition adapter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FrameTimestamps {
    capture: Option<MonotonicTimestamp>,
    media: Option<MonotonicTimestamp>,
    presentation: Option<MonotonicTimestamp>,
    callback: MonotonicTimestamp,
}

impl FrameTimestamps {
    /// Constructs the available timestamps. The callback timestamp is always required.
    #[must_use]
    pub const fn new(
        capture: Option<MonotonicTimestamp>,
        media: Option<MonotonicTimestamp>,
        presentation: Option<MonotonicTimestamp>,
        callback: MonotonicTimestamp,
    ) -> Self {
        Self {
            capture,
            media,
            presentation,
            callback,
        }
    }

    /// Returns the capture timestamp, when supplied by the camera API.
    #[must_use]
    pub const fn capture(self) -> Option<MonotonicTimestamp> {
        self.capture
    }

    /// Returns the converted media timestamp, when available.
    #[must_use]
    pub const fn media(self) -> Option<MonotonicTimestamp> {
        self.media
    }

    /// Returns the presentation timestamp, when available.
    #[must_use]
    pub const fn presentation(self) -> Option<MonotonicTimestamp> {
        self.presentation
    }

    /// Returns the callback receipt timestamp.
    #[must_use]
    pub const fn callback(self) -> MonotonicTimestamp {
        self.callback
    }

    /// Selects the best available estimate of exposure time.
    #[must_use]
    pub fn best_exposure_time(self) -> MonotonicTimestamp {
        self.capture.or(self.presentation).unwrap_or(self.callback)
    }
}

/// An immutable pinhole calibration snapshot.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CameraCalibration {
    revision: CalibrationRevision,
    image_dimensions: PixelDimensions,
    focal_length_pixels: DVec2,
    principal_point_pixels: DVec2,
    radial_tangential_distortion: [f64; 5],
    device_to_camera_rotation: DQuat,
}

impl CameraCalibration {
    /// Constructs a calibrated camera model.
    pub fn new(
        revision: CalibrationRevision,
        image_dimensions: PixelDimensions,
        focal_length_pixels: DVec2,
        principal_point_pixels: DVec2,
        radial_tangential_distortion: [f64; 5],
        device_to_camera_rotation: DQuat,
    ) -> Result<Self, FrameMetadataError> {
        let finite = focal_length_pixels.is_finite()
            && principal_point_pixels.is_finite()
            && radial_tangential_distortion
                .iter()
                .all(|value| value.is_finite())
            && device_to_camera_rotation.is_finite();
        if !finite {
            return Err(FrameMetadataError::NonFiniteValue);
        }
        if focal_length_pixels.x <= 0.0 || focal_length_pixels.y <= 0.0 {
            return Err(FrameMetadataError::InvalidFocalLength);
        }
        let norm = device_to_camera_rotation.length();
        if norm <= f64::EPSILON {
            return Err(FrameMetadataError::InvalidRotation);
        }

        Ok(Self {
            revision,
            image_dimensions,
            focal_length_pixels,
            principal_point_pixels,
            radial_tangential_distortion,
            device_to_camera_rotation: device_to_camera_rotation.normalize(),
        })
    }

    /// Returns the revision identifier.
    #[must_use]
    pub const fn revision(self) -> CalibrationRevision {
        self.revision
    }

    /// Returns the image dimensions for which the intrinsics are expressed.
    #[must_use]
    pub const fn image_dimensions(self) -> PixelDimensions {
        self.image_dimensions
    }

    /// Returns `(fx, fy)` in pixels.
    #[must_use]
    pub const fn focal_length_pixels(self) -> DVec2 {
        self.focal_length_pixels
    }

    /// Returns `(cx, cy)` in pixels.
    #[must_use]
    pub const fn principal_point_pixels(self) -> DVec2 {
        self.principal_point_pixels
    }

    /// Returns Brown-Conrady coefficients `[k1, k2, p1, p2, k3]`.
    #[must_use]
    pub const fn radial_tangential_distortion(self) -> [f64; 5] {
        self.radial_tangential_distortion
    }

    /// Returns the normalized device-body to camera-frame rotation.
    #[must_use]
    pub const fn device_to_camera_rotation(self) -> DQuat {
        self.device_to_camera_rotation
    }
}

/// Immutable metadata associated with one captured camera frame.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CameraFrameMetadata {
    id: FrameId,
    timestamps: FrameTimestamps,
    orientation: ImageOrientation,
    mirrored: bool,
    crop: NormalizedRect,
    model_input_transform: ImageTransform,
    calibration_revision: CalibrationRevision,
}

impl CameraFrameMetadata {
    /// Constructs frame metadata with all geometry decisions captured by value.
    #[must_use]
    pub const fn new(
        id: FrameId,
        timestamps: FrameTimestamps,
        orientation: ImageOrientation,
        mirrored: bool,
        crop: NormalizedRect,
        model_input_transform: ImageTransform,
        calibration_revision: CalibrationRevision,
    ) -> Self {
        Self {
            id,
            timestamps,
            orientation,
            mirrored,
            crop,
            model_input_transform,
            calibration_revision,
        }
    }

    /// Returns the frame identifier.
    #[must_use]
    pub const fn id(self) -> FrameId {
        self.id
    }

    /// Returns all frame timestamps.
    #[must_use]
    pub const fn timestamps(self) -> FrameTimestamps {
        self.timestamps
    }

    /// Returns the display orientation.
    #[must_use]
    pub const fn orientation(self) -> ImageOrientation {
        self.orientation
    }

    /// Reports whether source pixels were mirrored.
    #[must_use]
    pub const fn mirrored(self) -> bool {
        self.mirrored
    }

    /// Returns the normalized source crop.
    #[must_use]
    pub const fn crop(self) -> NormalizedRect {
        self.crop
    }

    /// Returns the exact source-to-model transform.
    #[must_use]
    pub const fn model_input_transform(self) -> ImageTransform {
        self.model_input_transform
    }

    /// Returns the calibration revision that must be used for this frame.
    #[must_use]
    pub const fn calibration_revision(self) -> CalibrationRevision {
        self.calibration_revision
    }
}

/// Invalid frame geometry or calibration data.
#[derive(Clone, Copy, Debug, PartialEq, Error)]
pub enum FrameMetadataError {
    /// An image dimension was zero.
    #[error("image dimensions must be non-zero")]
    ZeroDimensions,
    /// A floating-point value was NaN or infinite.
    #[error("frame metadata values must be finite")]
    NonFiniteValue,
    /// A normalized crop was empty or outside the source image.
    #[error("crop must be non-empty and contained by normalized image coordinates")]
    InvalidCrop,
    /// The image transform cannot be inverted.
    #[error("image transform must be non-singular")]
    SingularImageTransform,
    /// Focal lengths were not positive.
    #[error("camera focal lengths must be positive")]
    InvalidFocalLength,
    /// A camera rotation had zero length.
    #[error("camera rotation must have non-zero length")]
    InvalidRotation,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn projective_image_transform_has_an_explicit_direction() {
        let source = PixelDimensions::new(640, 480).unwrap();
        let output = PixelDimensions::new(320, 240).unwrap();
        let transform = ImageTransform::new(
            source,
            output,
            [0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
        )
        .unwrap();

        let point = transform.transform_point(DVec2::new(200.0, 100.0)).unwrap();
        assert_relative_eq!(point.x, 100.0);
        assert_relative_eq!(point.y, 50.0);
    }

    #[test]
    fn invalid_frame_geometry_is_rejected() {
        assert_eq!(
            PixelDimensions::new(0, 480),
            Err(FrameMetadataError::ZeroDimensions)
        );
        assert_eq!(
            NormalizedRect::new(0.8, 0.0, 0.3, 1.0),
            Err(FrameMetadataError::InvalidCrop)
        );
        let dimensions = PixelDimensions::new(1, 1).unwrap();
        assert_eq!(
            ImageTransform::new(dimensions, dimensions, [0.0; 9]),
            Err(FrameMetadataError::SingularImageTransform)
        );
    }

    #[test]
    fn frame_metadata_round_trips_through_serde() {
        let dimensions = PixelDimensions::new(960, 720).unwrap();
        let metadata = CameraFrameMetadata::new(
            FrameId::new(42),
            FrameTimestamps::new(
                Some(MonotonicTimestamp::from_nanos(1_000)),
                None,
                Some(MonotonicTimestamp::from_nanos(1_010)),
                MonotonicTimestamp::from_nanos(1_020),
            ),
            ImageOrientation::Degrees90,
            false,
            NormalizedRect::FULL,
            ImageTransform::identity(dimensions),
            CalibrationRevision::new(7),
        );

        let encoded = serde_json::to_string(&metadata).unwrap();
        let decoded: CameraFrameMetadata = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, metadata);
        assert_eq!(decoded.calibration_revision(), CalibrationRevision::new(7));
    }
}
