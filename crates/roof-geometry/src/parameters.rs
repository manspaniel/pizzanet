use core::fmt;

use serde::{Deserialize, Serialize};

const MIN_FEATURE_SIZE: f32 = 0.001;
const MAX_FEATURE_SIZE: f32 = 1_000_000.0;

/// A named field in [`RoofParameters`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParameterField {
    /// Overall eave width along X.
    EaveWidth,
    /// Overall eave depth along Z.
    EaveDepth,
    /// Width of the break between the shallow skirt and steep crown.
    ShoulderWidth,
    /// Depth of the break between the shallow skirt and steep crown.
    ShoulderDepth,
    /// Width of the flat crown top along X.
    CrownTopWidth,
    /// Depth of the flat crown top along Z.
    CrownTopDepth,
    /// Vertical rise from the eave to the shoulder.
    LowerRise,
    /// Vertical rise from the shoulder to the crown top.
    UpperRise,
}

impl ParameterField {
    /// Stable snake-case field name used by configuration and diagnostics.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::EaveWidth => "eave_width",
            Self::EaveDepth => "eave_depth",
            Self::ShoulderWidth => "shoulder_width",
            Self::ShoulderDepth => "shoulder_depth",
            Self::CrownTopWidth => "crown_top_width",
            Self::CrownTopDepth => "crown_top_depth",
            Self::LowerRise => "lower_rise",
            Self::UpperRise => "upper_rise",
        }
    }
}

impl fmt::Display for ParameterField {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Invalid parametric roof dimensions.
#[derive(Debug, Clone, PartialEq)]
pub enum ParameterError {
    /// A field contained NaN or infinity.
    NonFinite {
        /// Invalid field.
        field: ParameterField,
        /// Invalid value.
        value: f32,
    },
    /// A feature was too small or too large for robust mesh generation.
    OutsideRange {
        /// Invalid field.
        field: ParameterField,
        /// Invalid value.
        value: f32,
        /// Inclusive minimum.
        minimum: f32,
        /// Inclusive maximum.
        maximum: f32,
    },
    /// An inner ring was not sufficiently inset from its outer ring.
    InsufficientInset {
        /// Inner dimension field.
        inner_field: ParameterField,
        /// Inner dimension value.
        inner_value: f32,
        /// Outer dimension field.
        outer_field: ParameterField,
        /// Outer dimension value.
        outer_value: f32,
        /// Required difference between the full dimensions.
        minimum_difference: f32,
    },
    /// Adding the two rises overflowed the supported coordinate range.
    CombinedRiseTooLarge {
        /// Lower rise.
        lower_rise: f32,
        /// Upper rise.
        upper_rise: f32,
        /// Largest supported combined rise.
        maximum: f32,
    },
}

impl fmt::Display for ParameterError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFinite { field, value } => {
                write!(formatter, "{field} must be finite, got {value}")
            }
            Self::OutsideRange {
                field,
                value,
                minimum,
                maximum,
            } => write!(
                formatter,
                "{field} must be in [{minimum}, {maximum}], got {value}"
            ),
            Self::InsufficientInset {
                inner_field,
                inner_value,
                outer_field,
                outer_value,
                minimum_difference,
            } => write!(
                formatter,
                "{inner_field} ({inner_value}) must be at least {minimum_difference} smaller than {outer_field} ({outer_value})"
            ),
            Self::CombinedRiseTooLarge {
                lower_rise,
                upper_rise,
                maximum,
            } => write!(
                formatter,
                "lower_rise + upper_rise ({lower_rise} + {upper_rise}) must not exceed {maximum}"
            ),
        }
    }
}

impl std::error::Error for ParameterError {}

/// Dimensions of a symmetric classic two-pitch Pizza Hut roof.
///
/// The broad outer eave rectangle rises with a shallow pitch to the inset
/// shoulder rectangle. A steeper crown frustum then terminates at an inset,
/// flat rectangular top. This small parameter set preserves the recognizable
/// silhouette while remaining cheap to fit against multi-view observations.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoofParameters {
    /// Full outer width along X.
    pub eave_width: f32,
    /// Full outer depth along Z.
    pub eave_depth: f32,
    /// Full width of the inset pitch-break rectangle.
    pub shoulder_width: f32,
    /// Full depth of the inset pitch-break rectangle.
    pub shoulder_depth: f32,
    /// Full width of the inset flat crown top.
    pub crown_top_width: f32,
    /// Full depth of the inset flat crown top.
    pub crown_top_depth: f32,
    /// Height from the eave plane to the pitch break.
    pub lower_rise: f32,
    /// Height from the pitch break to the crown top.
    pub upper_rise: f32,
}

impl Default for RoofParameters {
    fn default() -> Self {
        Self {
            eave_width: 24.0,
            eave_depth: 18.0,
            shoulder_width: 14.4,
            shoulder_depth: 9.0,
            crown_top_width: 12.0,
            crown_top_depth: 6.5,
            lower_rise: 2.8,
            upper_rise: 3.2,
        }
    }
}

impl RoofParameters {
    /// Constructs and validates a parameter set.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        eave_width: f32,
        eave_depth: f32,
        shoulder_width: f32,
        shoulder_depth: f32,
        crown_top_width: f32,
        crown_top_depth: f32,
        lower_rise: f32,
        upper_rise: f32,
    ) -> Result<Self, ParameterError> {
        let parameters = Self {
            eave_width,
            eave_depth,
            shoulder_width,
            shoulder_depth,
            crown_top_width,
            crown_top_depth,
            lower_rise,
            upper_rise,
        };
        parameters.validate()?;
        Ok(parameters)
    }

    /// Validates finiteness, useful feature size, and ring nesting.
    pub fn validate(&self) -> Result<(), ParameterError> {
        let fields = [
            (ParameterField::EaveWidth, self.eave_width),
            (ParameterField::EaveDepth, self.eave_depth),
            (ParameterField::ShoulderWidth, self.shoulder_width),
            (ParameterField::ShoulderDepth, self.shoulder_depth),
            (ParameterField::CrownTopWidth, self.crown_top_width),
            (ParameterField::CrownTopDepth, self.crown_top_depth),
            (ParameterField::LowerRise, self.lower_rise),
            (ParameterField::UpperRise, self.upper_rise),
        ];

        for (field, value) in fields {
            if !value.is_finite() {
                return Err(ParameterError::NonFinite { field, value });
            }
            if !(MIN_FEATURE_SIZE..=MAX_FEATURE_SIZE).contains(&value) {
                return Err(ParameterError::OutsideRange {
                    field,
                    value,
                    minimum: MIN_FEATURE_SIZE,
                    maximum: MAX_FEATURE_SIZE,
                });
            }
        }

        validate_inset(
            ParameterField::ShoulderWidth,
            self.shoulder_width,
            ParameterField::EaveWidth,
            self.eave_width,
        )?;
        validate_inset(
            ParameterField::ShoulderDepth,
            self.shoulder_depth,
            ParameterField::EaveDepth,
            self.eave_depth,
        )?;
        validate_inset(
            ParameterField::CrownTopWidth,
            self.crown_top_width,
            ParameterField::ShoulderWidth,
            self.shoulder_width,
        )?;
        validate_inset(
            ParameterField::CrownTopDepth,
            self.crown_top_depth,
            ParameterField::ShoulderDepth,
            self.shoulder_depth,
        )?;

        let total_rise = self.lower_rise + self.upper_rise;
        if !total_rise.is_finite() || total_rise > MAX_FEATURE_SIZE {
            return Err(ParameterError::CombinedRiseTooLarge {
                lower_rise: self.lower_rise,
                upper_rise: self.upper_rise,
                maximum: MAX_FEATURE_SIZE,
            });
        }

        Ok(())
    }
}

fn validate_inset(
    inner_field: ParameterField,
    inner_value: f32,
    outer_field: ParameterField,
    outer_value: f32,
) -> Result<(), ParameterError> {
    if outer_value - inner_value < MIN_FEATURE_SIZE {
        return Err(ParameterError::InsufficientInset {
            inner_field,
            inner_value,
            outer_field,
            outer_value,
            minimum_difference: MIN_FEATURE_SIZE,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_and_constructor_validate() {
        let default = RoofParameters::default();
        assert_eq!(default.validate(), Ok(()));
        assert_eq!(
            RoofParameters::new(
                default.eave_width,
                default.eave_depth,
                default.shoulder_width,
                default.shoulder_depth,
                default.crown_top_width,
                default.crown_top_depth,
                default.lower_rise,
                default.upper_rise,
            ),
            Ok(default)
        );
    }

    #[test]
    fn rejects_non_finite_and_out_of_range_values() {
        let parameters = RoofParameters {
            eave_width: f32::NAN,
            ..RoofParameters::default()
        };
        assert!(matches!(
            parameters.validate(),
            Err(ParameterError::NonFinite {
                field: ParameterField::EaveWidth,
                ..
            })
        ));

        let parameters = RoofParameters {
            upper_rise: 0.0,
            ..RoofParameters::default()
        };
        assert!(matches!(
            parameters.validate(),
            Err(ParameterError::OutsideRange {
                field: ParameterField::UpperRise,
                ..
            })
        ));

        let parameters = RoofParameters {
            eave_depth: MAX_FEATURE_SIZE * 2.0,
            ..RoofParameters::default()
        };
        assert!(matches!(
            parameters.validate(),
            Err(ParameterError::OutsideRange {
                field: ParameterField::EaveDepth,
                ..
            })
        ));
    }

    #[test]
    fn rejects_non_nested_rings() {
        let defaults = RoofParameters::default();
        let parameters = RoofParameters {
            shoulder_width: defaults.eave_width,
            ..defaults
        };
        assert!(matches!(
            parameters.validate(),
            Err(ParameterError::InsufficientInset {
                inner_field: ParameterField::ShoulderWidth,
                outer_field: ParameterField::EaveWidth,
                ..
            })
        ));

        let parameters = RoofParameters {
            shoulder_depth: defaults.eave_depth + 1.0,
            ..defaults
        };
        assert!(matches!(
            parameters.validate(),
            Err(ParameterError::InsufficientInset {
                inner_field: ParameterField::ShoulderDepth,
                outer_field: ParameterField::EaveDepth,
                ..
            })
        ));

        let parameters = RoofParameters {
            crown_top_width: defaults.shoulder_width,
            ..defaults
        };
        assert!(matches!(
            parameters.validate(),
            Err(ParameterError::InsufficientInset {
                inner_field: ParameterField::CrownTopWidth,
                outer_field: ParameterField::ShoulderWidth,
                ..
            })
        ));

        let parameters = RoofParameters {
            crown_top_depth: defaults.shoulder_depth + 1.0,
            ..defaults
        };
        assert!(matches!(
            parameters.validate(),
            Err(ParameterError::InsufficientInset {
                inner_field: ParameterField::CrownTopDepth,
                outer_field: ParameterField::ShoulderDepth,
                ..
            })
        ));
    }

    #[test]
    fn rejects_derived_height_overflow() {
        let parameters = RoofParameters {
            lower_rise: MAX_FEATURE_SIZE * 0.75,
            upper_rise: MAX_FEATURE_SIZE * 0.75,
            ..RoofParameters::default()
        };
        assert!(matches!(
            parameters.validate(),
            Err(ParameterError::CombinedRiseTooLarge { .. })
        ));
    }

    #[test]
    fn parameter_json_rejects_unknown_fields() {
        let json = r#"{
            "eave_width": 24.0,
            "eave_depth": 18.0,
            "shoulder_width": 14.4,
            "shoulder_depth": 9.0,
            "crown_top_width": 12.0,
            "crown_top_depth": 6.5,
            "lower_rise": 2.8,
            "upper_rise": 3.2,
            "typo": 1.0
        }"#;
        assert!(serde_json::from_str::<RoofParameters>(json).is_err());
    }

    #[test]
    fn schema_one_ridge_parameters_do_not_deserialize_as_schema_two() {
        let json = r#"{
            "eave_width": 24.0,
            "eave_depth": 18.0,
            "shoulder_width": 14.4,
            "shoulder_depth": 9.0,
            "lower_rise": 2.8,
            "upper_rise": 4.6,
            "ridge_length": 8.0
        }"#;
        assert!(serde_json::from_str::<RoofParameters>(json).is_err());
    }
}
