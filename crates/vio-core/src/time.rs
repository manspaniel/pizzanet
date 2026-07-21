use serde::{Deserialize, Serialize};
use std::fmt;
use thiserror::Error;

/// A timestamp on one monotonically increasing clock, represented as integer nanoseconds.
///
/// The epoch is intentionally unspecified. Values may only be compared when they came from the
/// same clock domain. Browser integration should convert both event and receipt timestamps to a
/// shared monotonic domain before constructing this value.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct MonotonicTimestamp(u64);

impl MonotonicTimestamp {
    /// The zero point of an arbitrary monotonic clock.
    pub const ZERO: Self = Self(0);

    /// Constructs a timestamp from integer nanoseconds.
    #[must_use]
    pub const fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }

    /// Converts a non-negative finite millisecond value without silently accepting invalid input.
    pub fn try_from_millis_f64(milliseconds: f64) -> Result<Self, TimeValueError> {
        nanos_from_f64(milliseconds, 1_000_000.0).map(Self)
    }

    /// Returns the timestamp as integer nanoseconds.
    #[must_use]
    pub const fn as_nanos(self) -> u64 {
        self.0
    }

    /// Returns the timestamp in milliseconds for interop and diagnostics.
    #[must_use]
    pub fn as_millis_f64(self) -> f64 {
        self.0 as f64 / 1_000_000.0
    }

    /// Computes `self - earlier`, returning `None` rather than wrapping for reversed timestamps.
    #[must_use]
    pub const fn checked_duration_since(self, earlier: Self) -> Option<MonotonicDuration> {
        match self.0.checked_sub(earlier.0) {
            Some(value) => Some(MonotonicDuration::from_nanos(value)),
            None => None,
        }
    }

    /// Adds a duration, returning `None` on integer overflow.
    #[must_use]
    pub const fn checked_add(self, duration: MonotonicDuration) -> Option<Self> {
        match self.0.checked_add(duration.as_nanos()) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }
}

impl fmt::Display for MonotonicTimestamp {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}ns", self.0)
    }
}

/// A non-negative monotonic duration represented as integer nanoseconds.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct MonotonicDuration(u64);

impl MonotonicDuration {
    /// A zero-length duration.
    pub const ZERO: Self = Self(0);

    /// Constructs a duration from integer nanoseconds.
    #[must_use]
    pub const fn from_nanos(nanos: u64) -> Self {
        Self(nanos)
    }

    /// Converts non-negative finite seconds to the nearest integer nanosecond.
    pub fn try_from_secs_f64(seconds: f64) -> Result<Self, TimeValueError> {
        nanos_from_f64(seconds, 1_000_000_000.0).map(Self)
    }

    /// Converts non-negative finite milliseconds to the nearest integer nanosecond.
    pub fn try_from_millis_f64(milliseconds: f64) -> Result<Self, TimeValueError> {
        nanos_from_f64(milliseconds, 1_000_000.0).map(Self)
    }

    /// Returns the duration as integer nanoseconds.
    #[must_use]
    pub const fn as_nanos(self) -> u64 {
        self.0
    }

    /// Returns the duration in seconds.
    #[must_use]
    pub fn as_secs_f64(self) -> f64 {
        self.0 as f64 / 1_000_000_000.0
    }

    /// Adds two durations, returning `None` on integer overflow.
    #[must_use]
    pub const fn checked_add(self, other: Self) -> Option<Self> {
        match self.0.checked_add(other.0) {
            Some(value) => Some(Self(value)),
            None => None,
        }
    }
}

impl fmt::Display for MonotonicDuration {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}ns", self.0)
    }
}

/// An invalid floating-point time value supplied at an external boundary.
#[derive(Clone, Copy, Debug, PartialEq, Error)]
pub enum TimeValueError {
    /// The value was NaN or infinite.
    #[error("time value must be finite")]
    NonFinite,
    /// The value was negative.
    #[error("time value must not be negative")]
    Negative,
    /// The value could not fit in the nanosecond representation.
    #[error("time value exceeds the supported nanosecond range")]
    Overflow,
}

fn nanos_from_f64(value: f64, scale: f64) -> Result<u64, TimeValueError> {
    if !value.is_finite() {
        return Err(TimeValueError::NonFinite);
    }
    if value < 0.0 {
        return Err(TimeValueError::Negative);
    }

    let nanos = value * scale;
    if nanos > u64::MAX as f64 {
        return Err(TimeValueError::Overflow);
    }

    Ok(nanos.round() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floating_point_boundaries_are_validated_and_rounded() {
        assert_eq!(
            MonotonicTimestamp::try_from_millis_f64(1.234_567).unwrap(),
            MonotonicTimestamp::from_nanos(1_234_567)
        );
        assert_eq!(
            MonotonicDuration::try_from_secs_f64(0.25).unwrap(),
            MonotonicDuration::from_nanos(250_000_000)
        );
        assert_eq!(
            MonotonicDuration::try_from_secs_f64(f64::NAN),
            Err(TimeValueError::NonFinite)
        );
        assert_eq!(
            MonotonicTimestamp::try_from_millis_f64(-1.0),
            Err(TimeValueError::Negative)
        );
    }

    #[test]
    fn reversed_duration_is_not_allowed_to_wrap() {
        let earlier = MonotonicTimestamp::from_nanos(20);
        let later = MonotonicTimestamp::from_nanos(10);
        assert_eq!(later.checked_duration_since(earlier), None);
    }
}
