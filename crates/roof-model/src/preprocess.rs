use thiserror::Error;

/// Square pixel size expected by [`crate::KeypointRoofNet`].
pub const INPUT_SIZE: usize = crate::SPATIAL_INPUT_SIZE;

/// Mapping between source-image normalized coordinates and model-input pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LetterboxTransform {
    /// Square model-input side length in pixels.
    pub input_size: usize,
    /// Source image width in pixels.
    pub source_width: u32,
    /// Source image height in pixels.
    pub source_height: u32,
    /// Uniform source-to-input scale.
    pub scale: f32,
    /// Horizontal model-input padding in pixels.
    pub pad_x: f32,
    /// Vertical model-input padding in pixels.
    pub pad_y: f32,
}

impl LetterboxTransform {
    /// Creates a centered square letterbox transform.
    #[must_use]
    pub fn new(source_width: u32, source_height: u32) -> Self {
        Self::for_input_size(source_width, source_height, INPUT_SIZE)
    }

    /// Creates a centered square letterbox transform for an explicit model size.
    #[must_use]
    pub fn for_input_size(source_width: u32, source_height: u32, input_size: usize) -> Self {
        let scale = input_size as f32 / source_width.max(source_height) as f32;
        let scaled_width = source_width as f32 * scale;
        let scaled_height = source_height as f32 * scale;
        Self {
            input_size,
            source_width,
            source_height,
            scale,
            pad_x: (input_size as f32 - scaled_width) * 0.5,
            pad_y: (input_size as f32 - scaled_height) * 0.5,
        }
    }

    /// Maps a normalized source-image point into normalized model-input space.
    #[must_use]
    pub fn map_point(self, point: [f32; 2]) -> [f32; 2] {
        [
            (point[0] * self.source_width as f32 * self.scale + self.pad_x)
                / self.input_size as f32,
            (point[1] * self.source_height as f32 * self.scale + self.pad_y)
                / self.input_size as f32,
        ]
    }

    /// Maps a normalized model-input point back into normalized source space.
    ///
    /// This compatibility helper clamps points from letterbox padding to the
    /// nearest source-image edge. New decoding code should use
    /// [`Self::unmap_point_unclamped`] or [`Self::unmap_point_if_in_source`]
    /// when padding must remain distinguishable from image content.
    #[must_use]
    pub fn unmap_point(self, point: [f32; 2]) -> [f32; 2] {
        self.unmap_point_unclamped(point)
            .map(|coordinate| coordinate.clamp(0.0, 1.0))
    }

    /// Maps a normalized model-input point into normalized source space
    /// without clamping letterbox padding to the source-image border.
    #[must_use]
    pub fn unmap_point_unclamped(self, point: [f32; 2]) -> [f32; 2] {
        [
            ((point[0] * self.input_size as f32 - self.pad_x)
                / (self.source_width as f32 * self.scale)),
            ((point[1] * self.input_size as f32 - self.pad_y)
                / (self.source_height as f32 * self.scale)),
        ]
    }

    /// Maps a normalized model-input point only when it lies on source-image
    /// content rather than in the surrounding letterbox padding.
    #[must_use]
    pub fn unmap_point_if_in_source(self, point: [f32; 2]) -> Option<[f32; 2]> {
        let source = self.unmap_point_unclamped(point);
        source
            .iter()
            .all(|coordinate| coordinate.is_finite() && (0.0..=1.0).contains(coordinate))
            .then_some(source)
    }
}

/// Tensor-ready image plus the exact coordinate transform used to create it.
#[derive(Clone, Debug, PartialEq)]
pub struct PreparedInput {
    /// Channel-first floats normalized with ImageNet channel statistics.
    pub chw: Vec<f32>,
    /// Source-to-model letterbox transform.
    pub transform: LetterboxTransform,
}

/// Invalid raw RGB input.
#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum PrepareError {
    /// Model input side length must be positive.
    #[error("model input size must be nonzero")]
    EmptyInput,
    /// Images must have nonzero dimensions.
    #[error("image dimensions must be nonzero")]
    EmptyImage,
    /// RGB byte count disagrees with width and height.
    #[error("expected {expected} RGB bytes, received {actual}")]
    ByteCount {
        /// Expected byte count.
        expected: usize,
        /// Actual byte count.
        actual: usize,
    },
}

/// Letterboxes and bilinearly resamples packed RGB8 into normalized CHW floats.
pub fn prepare_rgb8(width: u32, height: u32, rgb: &[u8]) -> Result<PreparedInput, PrepareError> {
    prepare_rgb8_sized(width, height, rgb, INPUT_SIZE)
}

/// Letterboxes and resamples packed RGB8 for an explicit square model size.
pub fn prepare_rgb8_sized(
    width: u32,
    height: u32,
    rgb: &[u8],
    input_size: usize,
) -> Result<PreparedInput, PrepareError> {
    if input_size == 0 {
        return Err(PrepareError::EmptyInput);
    }
    if width == 0 || height == 0 {
        return Err(PrepareError::EmptyImage);
    }
    let expected = width as usize * height as usize * 3;
    if rgb.len() != expected {
        return Err(PrepareError::ByteCount {
            expected,
            actual: rgb.len(),
        });
    }

    let transform = LetterboxTransform::for_input_size(width, height, input_size);
    let mut chw = vec![0.0; 3 * input_size * input_size];
    for output_y in 0..input_size {
        for output_x in 0..input_size {
            let source_x = (output_x as f32 + 0.5 - transform.pad_x) / transform.scale - 0.5;
            let source_y = (output_y as f32 + 0.5 - transform.pad_y) / transform.scale - 0.5;
            if source_x < -0.5
                || source_y < -0.5
                || source_x > width as f32 - 0.5
                || source_y > height as f32 - 0.5
            {
                continue;
            }
            for channel in 0..3 {
                let value = bilinear(rgb, width, height, source_x, source_y, channel);
                const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
                const STD: [f32; 3] = [0.229, 0.224, 0.225];
                chw[channel * input_size * input_size + output_y * input_size + output_x] =
                    (value / 255.0 - MEAN[channel]) / STD[channel];
            }
        }
    }
    Ok(PreparedInput { chw, transform })
}

fn bilinear(rgb: &[u8], width: u32, height: u32, x: f32, y: f32, channel: usize) -> f32 {
    let x0 = x.floor().clamp(0.0, width.saturating_sub(1) as f32) as u32;
    let y0 = y.floor().clamp(0.0, height.saturating_sub(1) as f32) as u32;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let tx = (x - x0 as f32).clamp(0.0, 1.0);
    let ty = (y - y0 as f32).clamp(0.0, 1.0);
    let sample = |sx: u32, sy: u32| rgb[((sy * width + sx) * 3) as usize + channel] as f32;
    let top = sample(x0, y0) * (1.0 - tx) + sample(x1, y0) * tx;
    let bottom = sample(x0, y1) * (1.0 - tx) + sample(x1, y1) * tx;
    top * (1.0 - ty) + bottom * ty
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn letterbox_round_trip_is_stable() {
        let transform = LetterboxTransform::new(1600, 900);
        let point = [0.27, 0.81];
        let round_trip = transform.unmap_point(transform.map_point(point));
        assert!((round_trip[0] - point[0]).abs() < 1.0e-6);
        assert!((round_trip[1] - point[1]).abs() < 1.0e-6);
    }

    #[test]
    fn unclamped_inverse_preserves_landscape_padding() {
        let transform = LetterboxTransform::for_input_size(200, 100, 256);
        let source = transform.unmap_point_unclamped([0.5, 0.1]);
        assert!((source[0] - 0.5).abs() < 1.0e-6);
        assert!(source[1] < 0.0);
        assert_eq!(transform.unmap_point_if_in_source([0.5, 0.1]), None);
        assert_eq!(transform.unmap_point([0.5, 0.1]), [0.5, 0.0]);
    }

    #[test]
    fn unclamped_inverse_preserves_portrait_padding() {
        let transform = LetterboxTransform::for_input_size(100, 200, 256);
        let source = transform.unmap_point_unclamped([0.9, 0.5]);
        assert!(source[0] > 1.0);
        assert!((source[1] - 0.5).abs() < 1.0e-6);
        assert_eq!(transform.unmap_point_if_in_source([0.9, 0.5]), None);
        assert_eq!(transform.unmap_point([0.9, 0.5]), [1.0, 0.5]);
    }

    #[test]
    fn input_is_channel_first_and_imagenet_normalized() {
        let prepared = prepare_rgb8(1, 1, &[255, 128, 0]).expect("valid pixel");
        let center = (INPUT_SIZE / 2) * INPUT_SIZE + INPUT_SIZE / 2;
        assert!((prepared.chw[center] - (1.0 - 0.485) / 0.229).abs() < 1.0e-6);
        assert!(
            (prepared.chw[INPUT_SIZE * INPUT_SIZE + center] - (128.0 / 255.0 - 0.456) / 0.224)
                .abs()
                < 1.0e-6
        );
        assert!(
            (prepared.chw[2 * INPUT_SIZE * INPUT_SIZE + center] + 0.406 / 0.225).abs() < 1.0e-6
        );
    }

    #[test]
    fn explicit_spatial_size_controls_tensor_and_mapping() {
        let prepared = prepare_rgb8_sized(2, 1, &[255; 6], 64).unwrap();
        assert_eq!(prepared.chw.len(), 3 * 64 * 64);
        assert_eq!(prepared.transform.input_size, 64);
        let point = [0.2, 0.8];
        let round_trip = prepared
            .transform
            .unmap_point(prepared.transform.map_point(point));
        assert!((round_trip[0] - point[0]).abs() < 1.0e-6);
        assert!((round_trip[1] - point[1]).abs() < 1.0e-6);
    }
}
