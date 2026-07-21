//! Portable metadata describing how a rendered frame became model-input RGB.

use std::{error::Error, fmt};

use serde::{Deserialize, Serialize};

use crate::DayPhase;

/// Current serialized photometric-profile schema.
pub const PHOTOMETRIC_PROFILE_VERSION: u32 = 1;

/// Exact, deterministic phone-camera appearance parameters applied to one frame.
///
/// This contract contains only portable data. Pixel transforms live in producer
/// tooling, while dataset labels retain the values needed to audit or replay the
/// transform independently.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PhotometricProfile {
    /// Version of the serialized profile contract.
    pub schema_version: u32,
    /// Capture-time regime that selected the sensor-response ranges.
    pub day_phase: DayPhase,
    /// Stable caller-provided seed used to sample this profile.
    pub seed: u64,
    /// Residual automatic-exposure correction in photographic stops.
    pub exposure_compensation_stops: f32,
    /// Per-channel linear RGB gains representing residual white balance.
    pub white_balance_rgb: [f32; 3],
    /// Additional green-channel multiplier representing sensor tint error.
    pub tint_green: f32,
    /// Display-response contrast around middle grey.
    pub response_contrast: f32,
    /// Strength of the highlight shoulder in the linear response curve.
    pub response_shoulder: f32,
    /// Relative falloff from the optical centre towards image corners.
    pub vignette_strength: f32,
    /// Exponent controlling how close to the frame edge vignetting appears.
    pub vignette_power: f32,
    /// Signal-dependent noise coefficient in linear RGB units.
    pub shot_noise_stddev: f32,
    /// Signal-independent noise floor in linear RGB units.
    pub read_noise_stddev: f32,
    /// Amount of a small, display-space unsharp mask.
    pub sharpening_amount: f32,
    /// Strength of the local highlight bloom contribution.
    pub bloom_strength: f32,
    /// Linear luminance at which highlight bloom begins.
    pub bloom_threshold: f32,
    /// JPEG quality selected for encoding this frame, in `86..=96`.
    pub jpeg_quality: u8,
    /// Independent stable seed for the per-pixel sensor noise stream.
    pub noise_seed: u64,
}

impl PhotometricProfile {
    /// Checks that every parameter is finite, supported, and safely bounded.
    pub fn validate(&self) -> Result<(), PhotometricProfileValidationError> {
        if self.schema_version != PHOTOMETRIC_PROFILE_VERSION {
            return Err(PhotometricProfileValidationError::UnsupportedSchemaVersion(
                self.schema_version,
            ));
        }

        validate_value(
            "exposure_compensation_stops",
            self.exposure_compensation_stops,
            -1.0,
            1.0,
        )?;
        for gain in self.white_balance_rgb {
            validate_value("white_balance_rgb", gain, 0.75, 1.25)?;
        }
        validate_value("tint_green", self.tint_green, 0.9, 1.1)?;
        validate_value("response_contrast", self.response_contrast, 0.8, 1.2)?;
        validate_value("response_shoulder", self.response_shoulder, 0.0, 1.0)?;
        validate_value("vignette_strength", self.vignette_strength, 0.0, 0.5)?;
        validate_value("vignette_power", self.vignette_power, 0.5, 4.0)?;
        validate_value("shot_noise_stddev", self.shot_noise_stddev, 0.0, 0.1)?;
        validate_value("read_noise_stddev", self.read_noise_stddev, 0.0, 0.1)?;
        validate_value("sharpening_amount", self.sharpening_amount, 0.0, 0.5)?;
        validate_value("bloom_strength", self.bloom_strength, 0.0, 0.2)?;
        validate_value("bloom_threshold", self.bloom_threshold, 0.5, 2.0)?;
        if !(86..=96).contains(&self.jpeg_quality) {
            return Err(PhotometricProfileValidationError::InvalidValue(
                "jpeg_quality",
            ));
        }
        Ok(())
    }
}

/// Why a serialized photometric profile cannot be accepted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PhotometricProfileValidationError {
    /// The profile uses a schema newer or older than this crate understands.
    UnsupportedSchemaVersion(u32),
    /// The named field is non-finite or outside its supported bounds.
    InvalidValue(&'static str),
}

impl PhotometricProfileValidationError {
    /// Field within `PhotometricProfile` responsible for the error.
    #[must_use]
    pub const fn field(&self) -> &'static str {
        match self {
            Self::UnsupportedSchemaVersion(_) => "schema_version",
            Self::InvalidValue(field) => field,
        }
    }
}

impl fmt::Display for PhotometricProfileValidationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion(version) => write!(
                formatter,
                "unsupported photometric profile schema version {version}; expected {PHOTOMETRIC_PROFILE_VERSION}"
            ),
            Self::InvalidValue(field) => {
                write!(formatter, "photometric profile has invalid {field}")
            }
        }
    }
}

impl Error for PhotometricProfileValidationError {}

/// Optional appearance transforms applied after geometry-aligned rendering.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FrameAppearance {
    /// Exact phone-camera profile used to produce stored RGB, when applied.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub photometric_profile: Option<PhotometricProfile>,
}

impl FrameAppearance {
    /// Returns `true` when no post-render appearance transform is recorded.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.photometric_profile.is_none()
    }
}

fn validate_value(
    field: &'static str,
    value: f32,
    minimum: f32,
    maximum: f32,
) -> Result<(), PhotometricProfileValidationError> {
    if value.is_finite() && (minimum..=maximum).contains(&value) {
        Ok(())
    } else {
        Err(PhotometricProfileValidationError::InvalidValue(field))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_profile() -> PhotometricProfile {
        PhotometricProfile {
            schema_version: PHOTOMETRIC_PROFILE_VERSION,
            day_phase: DayPhase::Twilight,
            seed: 42,
            exposure_compensation_stops: 0.1,
            white_balance_rgb: [0.98, 1.0, 1.03],
            tint_green: 1.01,
            response_contrast: 1.02,
            response_shoulder: 0.15,
            vignette_strength: 0.08,
            vignette_power: 1.7,
            shot_noise_stddev: 0.004,
            read_noise_stddev: 0.002,
            sharpening_amount: 0.08,
            bloom_strength: 0.02,
            bloom_threshold: 0.8,
            jpeg_quality: 92,
            noise_seed: 84,
        }
    }

    #[test]
    fn profile_round_trips_without_changing_field_names() {
        let profile = valid_profile();
        let value = serde_json::to_value(profile).expect("profile should serialize");
        assert_eq!(value["schema_version"], PHOTOMETRIC_PROFILE_VERSION);
        assert_eq!(
            value["white_balance_rgb"]
                .as_array()
                .expect("white balance should be an array")
                .len(),
            3
        );
        assert_eq!(value["jpeg_quality"], 92);
        assert_eq!(
            serde_json::from_value::<PhotometricProfile>(value)
                .expect("profile should deserialize"),
            profile
        );
    }

    #[test]
    fn validation_rejects_versions_non_finite_values_and_bounds() {
        let mut profile = valid_profile();
        profile.schema_version += 1;
        assert_eq!(
            profile.validate(),
            Err(PhotometricProfileValidationError::UnsupportedSchemaVersion(
                2
            ))
        );

        profile = valid_profile();
        profile.white_balance_rgb[1] = f32::NAN;
        assert_eq!(
            profile.validate(),
            Err(PhotometricProfileValidationError::InvalidValue(
                "white_balance_rgb"
            ))
        );

        profile = valid_profile();
        profile.jpeg_quality = 85;
        assert_eq!(
            profile.validate(),
            Err(PhotometricProfileValidationError::InvalidValue(
                "jpeg_quality"
            ))
        );
    }
}
