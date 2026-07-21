//! Deterministic, geometry-preserving phone-camera appearance simulation.
//!
//! The transform operates only on RGB samples. It never moves pixels, changes
//! dimensions, or touches alpha, so renderer-produced geometric labels remain
//! aligned with the resulting image. A sampled [`PhotometricProfile`] is fully
//! serializable and can be recorded beside a generated frame.

use std::{error::Error, fmt};

use synth_data::{DayPhase, PhotometricProfileValidationError, SampledEnvironment};

pub use synth_data::{PHOTOMETRIC_PROFILE_VERSION, PhotometricProfile};

/// An invalid input to the photometric appearance transform.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PhotometricError {
    /// Width and height must both be non-zero.
    ZeroDimensions,
    /// Pixel-count or byte-length arithmetic overflowed the current platform.
    DimensionOverflow,
    /// The byte slice is neither packed RGB8 nor packed RGBA8.
    InvalidBufferLength {
        /// Required length for a packed RGB8 image.
        expected_rgb: usize,
        /// Required length for a packed RGBA8 image.
        expected_rgba: usize,
        /// Actual byte-slice length.
        actual: usize,
    },
    /// A sampled-environment field was not finite or within its documented domain.
    InvalidEnvironment(&'static str),
    /// A profile field was not finite or within its supported safety bounds.
    InvalidProfile(&'static str),
    /// The serialized profile uses an unsupported schema version.
    UnsupportedProfileVersion(u32),
    /// Temporary image storage could not be reserved.
    AllocationFailed,
}

impl fmt::Display for PhotometricError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDimensions => formatter.write_str("image dimensions must be non-zero"),
            Self::DimensionOverflow => formatter.write_str("image dimensions overflowed"),
            Self::InvalidBufferLength {
                expected_rgb,
                expected_rgba,
                actual,
            } => write!(
                formatter,
                "pixel buffer has {actual} bytes; expected {expected_rgb} for RGB8 or {expected_rgba} for RGBA8"
            ),
            Self::InvalidEnvironment(field) => {
                write!(formatter, "sampled environment has invalid {field}")
            }
            Self::InvalidProfile(field) => {
                write!(formatter, "photometric profile has invalid {field}")
            }
            Self::UnsupportedProfileVersion(version) => {
                write!(
                    formatter,
                    "unsupported photometric profile version {version}"
                )
            }
            Self::AllocationFailed => {
                formatter.write_str("could not allocate photometric image storage")
            }
        }
    }
}

impl Error for PhotometricError {}

impl From<PhotometricProfileValidationError> for PhotometricError {
    fn from(error: PhotometricProfileValidationError) -> Self {
        match error {
            PhotometricProfileValidationError::UnsupportedSchemaVersion(version) => {
                Self::UnsupportedProfileVersion(version)
            }
            PhotometricProfileValidationError::InvalidValue(field) => Self::InvalidProfile(field),
        }
    }
}

/// Samples a deterministic profile correlated with the supplied environment.
///
/// The same seed and environment produce the same profile. Callers should derive
/// a distinct stable seed per frame when a sequence needs changing sensor noise.
pub fn sample_photometric_profile(
    seed: u64,
    environment: SampledEnvironment,
) -> Result<PhotometricProfile, PhotometricError> {
    validate_environment(environment)?;

    let phase = PhaseRanges::for_phase(environment.day_phase);
    let phase_tag = match environment.day_phase {
        DayPhase::Day => 0x0d41_7e5c_04a2_813b,
        DayPhase::Twilight => 0x7f4a_7c15_9e37_79b9,
        DayPhase::Night => 0xa5a3_56d1_0f2c_8b71,
    };
    let mut rng = StableRng::new(seed ^ phase_tag);

    // Treat EV100 as context for a small auto-exposure residual, not as an
    // instruction to re-expose the renderer's already displayable result.
    let exposure_residual = (phase.reference_ev100 - environment.camera_exposure_ev100) * 0.035;
    let exposure_compensation_stops = (exposure_residual
        + rng.symmetric(phase.exposure_jitter_stops))
    .clamp(-phase.exposure_limit_stops, phase.exposure_limit_stops);

    // A phone's automatic white balance corrects most of the illuminant. These
    // gains model only its small residual temperature and green-magenta error.
    let illuminant_bias = ((6_500.0 - environment.color_temperature_k) / 4_500.0).clamp(-1.0, 1.0);
    let temperature_residual = illuminant_bias * 0.035 + rng.symmetric(phase.temperature_jitter);
    let white_balance_rgb = [
        (1.0 - temperature_residual).clamp(0.94, 1.06),
        1.0,
        (1.0 + temperature_residual * 1.15).clamp(0.93, 1.07),
    ];

    let visibility_contrast = ((environment.visibility_km - 20.0) / 100.0).clamp(-0.015, 0.015);
    let artificial_light_factor = (environment.artificial_light_strength * 0.025).min(0.07);
    let wet_highlight_factor = environment.ground_wetness * 0.16;
    let bloom_strength = (rng.range(phase.bloom_strength)
        * (1.0 + artificial_light_factor + wet_highlight_factor))
        .min(0.08);
    let bloom_threshold = (rng.range(phase.bloom_threshold)
        - environment.ground_wetness * 0.025
        - artificial_light_factor * 0.2)
        .clamp(0.62, 1.05);
    let mut jpeg_rng = StableRng::new(seed ^ phase_tag ^ 0xd6e8_feb8_6659_fd93);
    let jpeg_quality = jpeg_rng.inclusive_u8(86, 96);

    let profile = PhotometricProfile {
        schema_version: PHOTOMETRIC_PROFILE_VERSION,
        day_phase: environment.day_phase,
        seed,
        exposure_compensation_stops,
        white_balance_rgb,
        tint_green: 1.0 + rng.symmetric(phase.tint_jitter),
        response_contrast: (rng.range(phase.response_contrast) + visibility_contrast)
            .clamp(0.94, 1.06),
        response_shoulder: rng.range(phase.response_shoulder),
        vignette_strength: rng.range(phase.vignette_strength),
        vignette_power: rng.range(phase.vignette_power),
        shot_noise_stddev: rng.range(phase.shot_noise),
        read_noise_stddev: rng.range(phase.read_noise),
        sharpening_amount: rng.range(phase.sharpening),
        bloom_strength,
        bloom_threshold,
        jpeg_quality,
        noise_seed: rng.next_u64() ^ seed.rotate_left(29),
    };
    profile.validate().map_err(PhotometricError::from)?;
    Ok(profile)
}

/// Samples and applies deterministic phone-camera appearance to RGB8 or RGBA8.
///
/// The pixel format is inferred from the exact byte length. RGB channels are
/// modified in place; an RGBA8 alpha channel is preserved byte-for-byte. The
/// returned profile can be stored with dataset metadata or replayed later.
pub fn apply_phone_camera_appearance(
    width: u32,
    height: u32,
    pixels: &mut [u8],
    seed: u64,
    environment: SampledEnvironment,
) -> Result<PhotometricProfile, PhotometricError> {
    let profile = sample_photometric_profile(seed, environment)?;
    apply_photometric_profile(width, height, pixels, &profile)?;
    Ok(profile)
}

/// Applies an existing profile to packed RGB8 or RGBA8 pixels.
///
/// Reapplying the same profile to the same bytes is deterministic. This function
/// is useful when replaying a profile loaded from dataset metadata.
pub fn apply_photometric_profile(
    width: u32,
    height: u32,
    pixels: &mut [u8],
    profile: &PhotometricProfile,
) -> Result<(), PhotometricError> {
    profile.validate().map_err(PhotometricError::from)?;
    let image = ImageLayout::new(width, height, pixels.len())?;

    let mut linear = allocate_pixels(image.pixel_count)?;
    let exposure = 2.0_f32.powf(profile.exposure_compensation_stops);
    for y in 0..image.height {
        for x in 0..image.width {
            let pixel_index = y * image.width + x;
            let byte_index = pixel_index * image.channels;
            let vignette = vignette_factor(x, y, &image, profile);
            let gains = [
                profile.white_balance_rgb[0],
                profile.white_balance_rgb[1] * profile.tint_green,
                profile.white_balance_rgb[2],
            ];
            for channel in 0..3 {
                linear[pixel_index][channel] = srgb_to_linear(pixels[byte_index + channel])
                    * exposure
                    * gains[channel]
                    * vignette;
            }
        }
    }

    apply_bloom(&mut linear, &image, profile)?;

    let mut noise = StableRng::new(profile.noise_seed);
    let mut display = allocate_pixels(image.pixel_count)?;
    for (pixel_index, signal) in linear.iter().enumerate() {
        for channel in 0..3 {
            let noise_stddev = profile.read_noise_stddev
                + profile.shot_noise_stddev * signal[channel].max(0.0).sqrt();
            let noisy_signal =
                (signal[channel] + noise.approx_standard_normal() * noise_stddev).max(0.0);
            display[pixel_index][channel] = encode_response(noisy_signal, profile);
        }
    }

    for y in 0..image.height {
        for x in 0..image.width {
            let pixel_index = y * image.width + x;
            let byte_index = pixel_index * image.channels;
            let left = y * image.width + x.saturating_sub(1);
            let right = y * image.width + (x + 1).min(image.width - 1);
            let up = y.saturating_sub(1) * image.width + x;
            let down = (y + 1).min(image.height - 1) * image.width + x;

            for channel in 0..3 {
                let centre = display[pixel_index][channel];
                let local_blur = (centre * 4.0
                    + display[left][channel]
                    + display[right][channel]
                    + display[up][channel]
                    + display[down][channel])
                    / 8.0;
                let sharpened = centre + profile.sharpening_amount * (centre - local_blur);
                pixels[byte_index + channel] = quantize_u8(sharpened);
            }
        }
    }

    Ok(())
}

#[derive(Clone, Copy)]
struct ImageLayout {
    width: usize,
    height: usize,
    pixel_count: usize,
    channels: usize,
}

impl ImageLayout {
    fn new(width: u32, height: u32, actual_bytes: usize) -> Result<Self, PhotometricError> {
        if width == 0 || height == 0 {
            return Err(PhotometricError::ZeroDimensions);
        }
        let width = usize::try_from(width).map_err(|_| PhotometricError::DimensionOverflow)?;
        let height = usize::try_from(height).map_err(|_| PhotometricError::DimensionOverflow)?;
        let pixel_count = width
            .checked_mul(height)
            .ok_or(PhotometricError::DimensionOverflow)?;
        let expected_rgb = pixel_count
            .checked_mul(3)
            .ok_or(PhotometricError::DimensionOverflow)?;
        let expected_rgba = pixel_count
            .checked_mul(4)
            .ok_or(PhotometricError::DimensionOverflow)?;
        let channels = if actual_bytes == expected_rgb {
            3
        } else if actual_bytes == expected_rgba {
            4
        } else {
            return Err(PhotometricError::InvalidBufferLength {
                expected_rgb,
                expected_rgba,
                actual: actual_bytes,
            });
        };

        Ok(Self {
            width,
            height,
            pixel_count,
            channels,
        })
    }
}

#[derive(Clone, Copy)]
struct PhaseRanges {
    reference_ev100: f32,
    exposure_jitter_stops: f32,
    exposure_limit_stops: f32,
    temperature_jitter: f32,
    tint_jitter: f32,
    response_contrast: (f32, f32),
    response_shoulder: (f32, f32),
    vignette_strength: (f32, f32),
    vignette_power: (f32, f32),
    shot_noise: (f32, f32),
    read_noise: (f32, f32),
    sharpening: (f32, f32),
    bloom_strength: (f32, f32),
    bloom_threshold: (f32, f32),
}

impl PhaseRanges {
    fn for_phase(day_phase: DayPhase) -> Self {
        match day_phase {
            DayPhase::Day => Self {
                reference_ev100: 13.0,
                exposure_jitter_stops: 0.055,
                exposure_limit_stops: 0.16,
                temperature_jitter: 0.010,
                tint_jitter: 0.007,
                response_contrast: (0.99, 1.025),
                response_shoulder: (0.07, 0.13),
                vignette_strength: (0.035, 0.085),
                vignette_power: (1.45, 2.15),
                shot_noise: (0.0015, 0.0030),
                read_noise: (0.0004, 0.0010),
                sharpening: (0.055, 0.115),
                bloom_strength: (0.004, 0.012),
                bloom_threshold: (0.88, 1.03),
            },
            DayPhase::Twilight => Self {
                reference_ev100: 7.5,
                exposure_jitter_stops: 0.085,
                exposure_limit_stops: 0.22,
                temperature_jitter: 0.016,
                tint_jitter: 0.011,
                response_contrast: (0.98, 1.03),
                response_shoulder: (0.10, 0.18),
                vignette_strength: (0.045, 0.105),
                vignette_power: (1.35, 2.10),
                shot_noise: (0.0035, 0.0065),
                read_noise: (0.0012, 0.0028),
                sharpening: (0.045, 0.105),
                bloom_strength: (0.008, 0.022),
                bloom_threshold: (0.76, 0.94),
            },
            DayPhase::Night => Self {
                reference_ev100: 2.2,
                exposure_jitter_stops: 0.12,
                exposure_limit_stops: 0.30,
                temperature_jitter: 0.024,
                tint_jitter: 0.017,
                response_contrast: (0.965, 1.035),
                response_shoulder: (0.14, 0.24),
                vignette_strength: (0.055, 0.13),
                vignette_power: (1.25, 2.0),
                shot_noise: (0.007, 0.013),
                read_noise: (0.003, 0.006),
                sharpening: (0.035, 0.085),
                bloom_strength: (0.014, 0.038),
                bloom_threshold: (0.66, 0.84),
            },
        }
    }
}

struct StableRng {
    state: u64,
}

impl StableRng {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x6a09_e667_f3bc_c909,
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.state;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn unit_f32(&mut self) -> f32 {
        let fraction = (self.next_u64() >> 40) as u32;
        fraction as f32 * (1.0 / 16_777_216.0)
    }

    fn symmetric(&mut self, magnitude: f32) -> f32 {
        (self.unit_f32() * 2.0 - 1.0) * magnitude
    }

    fn range(&mut self, bounds: (f32, f32)) -> f32 {
        bounds.0 + (bounds.1 - bounds.0) * self.unit_f32()
    }

    fn inclusive_u8(&mut self, minimum: u8, maximum: u8) -> u8 {
        let span = u64::from(maximum - minimum) + 1;
        minimum + (self.next_u64() % span) as u8
    }

    fn approx_standard_normal(&mut self) -> f32 {
        // The sum of two independent uniform variates has variance 1/6.
        (self.unit_f32() + self.unit_f32() - 1.0) * 6.0_f32.sqrt()
    }
}

fn validate_environment(environment: SampledEnvironment) -> Result<(), PhotometricError> {
    validate_environment_range("shadow_softness", environment.shadow_softness, 0.0, 1.0)?;
    validate_environment_range("ground_wetness", environment.ground_wetness, 0.0, 1.0)?;
    validate_environment_range("visibility_km", environment.visibility_km, 0.01, 500.0)?;
    validate_environment_range(
        "color_temperature_k",
        environment.color_temperature_k,
        1_500.0,
        12_000.0,
    )?;
    validate_environment_range(
        "camera_exposure_ev100",
        environment.camera_exposure_ev100,
        -20.0,
        30.0,
    )?;
    validate_environment_range(
        "artificial_light_strength",
        environment.artificial_light_strength,
        0.0,
        100.0,
    )?;
    Ok(())
}

fn validate_environment_range(
    field: &'static str,
    value: f32,
    minimum: f32,
    maximum: f32,
) -> Result<(), PhotometricError> {
    if value.is_finite() && (minimum..=maximum).contains(&value) {
        Ok(())
    } else {
        Err(PhotometricError::InvalidEnvironment(field))
    }
}

fn allocate_pixels(pixel_count: usize) -> Result<Vec<[f32; 3]>, PhotometricError> {
    let mut pixels = Vec::new();
    pixels
        .try_reserve_exact(pixel_count)
        .map_err(|_| PhotometricError::AllocationFailed)?;
    pixels.resize(pixel_count, [0.0; 3]);
    Ok(pixels)
}

fn vignette_factor(x: usize, y: usize, image: &ImageLayout, profile: &PhotometricProfile) -> f32 {
    let normalized_x = ((x as f32 + 0.5) / image.width as f32) * 2.0 - 1.0;
    let normalized_y = ((y as f32 + 0.5) / image.height as f32) * 2.0 - 1.0;
    let corner_radius_squared = (normalized_x * normalized_x + normalized_y * normalized_y) * 0.5;
    1.0 - profile.vignette_strength * corner_radius_squared.powf(profile.vignette_power)
}

fn apply_bloom(
    linear: &mut [[f32; 3]],
    image: &ImageLayout,
    profile: &PhotometricProfile,
) -> Result<(), PhotometricError> {
    if profile.bloom_strength == 0.0 {
        return Ok(());
    }

    let mut highlights = allocate_pixels(image.pixel_count)?;
    for (highlight, signal) in highlights.iter_mut().zip(linear.iter()) {
        let luminance = signal[0] * 0.2126 + signal[1] * 0.7152 + signal[2] * 0.0722;
        if luminance > profile.bloom_threshold {
            let fraction = (luminance - profile.bloom_threshold) / luminance.max(0.000_001);
            for channel in 0..3 {
                highlight[channel] = signal[channel] * fraction;
            }
        }
    }

    for y in 0..image.height {
        let rows = [y.saturating_sub(1), y, (y + 1).min(image.height - 1)];
        for x in 0..image.width {
            let columns = [x.saturating_sub(1), x, (x + 1).min(image.width - 1)];
            let pixel_index = y * image.width + x;
            let mut blurred = [0.0_f32; 3];
            for (kernel_y, source_y) in rows.iter().enumerate() {
                for (kernel_x, source_x) in columns.iter().enumerate() {
                    let weight_y = if kernel_y == 1 { 2.0 } else { 1.0 };
                    let weight_x = if kernel_x == 1 { 2.0 } else { 1.0 };
                    let weight = weight_x * weight_y;
                    let source = highlights[*source_y * image.width + *source_x];
                    for channel in 0..3 {
                        blurred[channel] += source[channel] * weight;
                    }
                }
            }
            for channel in 0..3 {
                linear[pixel_index][channel] += blurred[channel] * (profile.bloom_strength / 16.0);
            }
        }
    }
    Ok(())
}

fn srgb_to_linear(value: u8) -> f32 {
    let encoded = f32::from(value) / 255.0;
    if encoded <= 0.040_45 {
        encoded / 12.92
    } else {
        ((encoded + 0.055) / 1.055).powf(2.4)
    }
}

fn encode_response(signal: f32, profile: &PhotometricProfile) -> f32 {
    let shoulder = profile.response_shoulder;
    let mapped = signal * (1.0 + shoulder) / (1.0 + shoulder * signal);
    let encoded = if mapped <= 0.003_130_8 {
        mapped * 12.92
    } else {
        1.055 * mapped.powf(1.0 / 2.4) - 0.055
    };
    ((encoded - 0.5) * profile.response_contrast + 0.5).clamp(0.0, 1.0)
}

fn quantize_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use synth_data::{SceneDomain, WeatherPreset};

    fn environment(day_phase: DayPhase) -> SampledEnvironment {
        let (color_temperature_k, camera_exposure_ev100, artificial_light_strength) =
            match day_phase {
                DayPhase::Day => (6_200.0, 13.2, 0.02),
                DayPhase::Twilight => (5_100.0, 7.4, 0.7),
                DayPhase::Night => (3_200.0, 2.1, 1.8),
            };
        SampledEnvironment {
            day_phase,
            domain: SceneDomain::Urban,
            weather: WeatherPreset::PartlyCloudy,
            shadow_softness: 0.45,
            ground_wetness: 0.2,
            visibility_km: 36.0,
            color_temperature_k,
            camera_exposure_ev100,
            artificial_light_strength,
        }
    }

    fn rgb_gradient(width: usize, height: usize) -> Vec<u8> {
        let mut pixels = Vec::with_capacity(width * height * 3);
        for y in 0..height {
            for x in 0..width {
                pixels.extend_from_slice(&[
                    ((x * 31 + y * 7) % 256) as u8,
                    ((x * 11 + y * 29 + 40) % 256) as u8,
                    ((x * 17 + y * 13 + 90) % 256) as u8,
                ]);
            }
        }
        pixels
    }

    #[test]
    fn same_seed_and_input_are_identical() {
        let source = rgb_gradient(12, 8);
        let mut first = source.clone();
        let mut second = source;
        let first_profile = apply_phone_camera_appearance(
            12,
            8,
            &mut first,
            0x1234_5678,
            environment(DayPhase::Twilight),
        )
        .expect("first transform should succeed");
        let second_profile = apply_phone_camera_appearance(
            12,
            8,
            &mut second,
            0x1234_5678,
            environment(DayPhase::Twilight),
        )
        .expect("second transform should succeed");

        assert_eq!(first_profile, second_profile);
        assert_eq!(first, second);
    }

    #[test]
    fn different_seeds_change_profile_and_pixels() {
        let source = rgb_gradient(12, 8);
        let mut first = source.clone();
        let mut second = source;
        let first_profile =
            apply_phone_camera_appearance(12, 8, &mut first, 17, environment(DayPhase::Day))
                .expect("first transform should succeed");
        let second_profile =
            apply_phone_camera_appearance(12, 8, &mut second, 18, environment(DayPhase::Day))
                .expect("second transform should succeed");

        assert_ne!(first_profile, second_profile);
        assert_ne!(first, second);
    }

    #[test]
    fn rgba_alpha_is_preserved() {
        let rgb = rgb_gradient(9, 7);
        let mut rgba = Vec::with_capacity(9 * 7 * 4);
        let mut expected_alpha = Vec::with_capacity(9 * 7);
        for (pixel_index, color) in rgb.chunks_exact(3).enumerate() {
            let alpha = ((pixel_index * 37 + 19) % 256) as u8;
            rgba.extend_from_slice(&[color[0], color[1], color[2], alpha]);
            expected_alpha.push(alpha);
        }

        apply_phone_camera_appearance(9, 7, &mut rgba, 99, environment(DayPhase::Night))
            .expect("RGBA transform should succeed");
        let actual_alpha: Vec<_> = rgba.chunks_exact(4).map(|pixel| pixel[3]).collect();
        assert_eq!(actual_alpha, expected_alpha);
    }

    #[test]
    fn day_and_night_use_distinct_correlated_profiles() {
        let day = sample_photometric_profile(42, environment(DayPhase::Day))
            .expect("day profile should sample");
        let night = sample_photometric_profile(42, environment(DayPhase::Night))
            .expect("night profile should sample");

        assert_ne!(day, night);
        assert!(night.shot_noise_stddev > day.shot_noise_stddev);
        assert!(night.read_noise_stddev > day.read_noise_stddev);
        assert!(night.bloom_strength > day.bloom_strength);
    }

    #[test]
    fn profile_round_trips_through_json() {
        let profile = sample_photometric_profile(7, environment(DayPhase::Twilight))
            .expect("profile should sample");
        let encoded = serde_json::to_string(&profile).expect("profile should serialize");
        let decoded: PhotometricProfile =
            serde_json::from_str(&encoded).expect("profile should deserialize");
        assert_eq!(decoded, profile);
    }

    #[test]
    fn jpeg_quality_is_deterministic_bounded_and_validated() {
        let sampled = sample_photometric_profile(73, environment(DayPhase::Day))
            .expect("profile should sample");
        let repeated = sample_photometric_profile(73, environment(DayPhase::Day))
            .expect("profile should sample again");
        assert_eq!(sampled.jpeg_quality, repeated.jpeg_quality);
        assert!((86..=96).contains(&sampled.jpeg_quality));

        for seed in 0..64 {
            let profile = sample_photometric_profile(seed, environment(DayPhase::Night))
                .expect("profile should sample");
            assert!((86..=96).contains(&profile.jpeg_quality));
        }

        let mut invalid = sampled;
        invalid.jpeg_quality = 85;
        assert_eq!(
            invalid.validate(),
            Err(PhotometricProfileValidationError::InvalidValue(
                "jpeg_quality"
            ))
        );
    }

    #[test]
    fn rejects_bad_buffer_lengths_and_zero_dimensions() {
        let profile = sample_photometric_profile(1, environment(DayPhase::Day))
            .expect("profile should sample");
        let mut short = vec![0; 47];
        assert_eq!(
            apply_photometric_profile(4, 4, &mut short, &profile),
            Err(PhotometricError::InvalidBufferLength {
                expected_rgb: 48,
                expected_rgba: 64,
                actual: 47,
            })
        );
        assert_eq!(
            apply_photometric_profile(0, 4, &mut [], &profile),
            Err(PhotometricError::ZeroDimensions)
        );
    }

    #[test]
    fn rejects_non_finite_environment_fields() {
        let mut invalid = environment(DayPhase::Day);
        invalid.color_temperature_k = f32::NAN;
        assert_eq!(
            sample_photometric_profile(5, invalid),
            Err(PhotometricError::InvalidEnvironment("color_temperature_k"))
        );
    }
}
