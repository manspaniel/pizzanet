use crate::{
    BatchDiagnostics, DQuat, DVec3, MonotonicDuration, MonotonicTimestamp, NormalizedMotionSample,
    SensorBatch,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Constant gyroscope and accelerometer biases used for one preintegration interval.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImuBias {
    gyroscope_rad_s: DVec3,
    accelerometer_mps2: DVec3,
}

impl ImuBias {
    /// A zero-bias estimate.
    pub const ZERO: Self = Self {
        gyroscope_rad_s: DVec3::ZERO,
        accelerometer_mps2: DVec3::ZERO,
    };

    /// Constructs a finite bias estimate.
    pub fn new(
        gyroscope_rad_s: DVec3,
        accelerometer_mps2: DVec3,
    ) -> Result<Self, PreintegrationError> {
        if !gyroscope_rad_s.is_finite() || !accelerometer_mps2.is_finite() {
            return Err(PreintegrationError::NonFiniteInput);
        }
        Ok(Self {
            gyroscope_rad_s,
            accelerometer_mps2,
        })
    }

    /// Returns gyroscope bias in radians/second.
    #[must_use]
    pub const fn gyroscope_rad_s(self) -> DVec3 {
        self.gyroscope_rad_s
    }

    /// Returns accelerometer bias in metres/second squared.
    #[must_use]
    pub const fn accelerometer_mps2(self) -> DVec3 {
        self.accelerometer_mps2
    }
}

impl Default for ImuBias {
    fn default() -> Self {
        Self::ZERO
    }
}

/// Deterministic preintegration and uncertainty parameters.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreintegrationConfig {
    max_sample_gap: MonotonicDuration,
    gyroscope_noise_density: f64,
    accelerometer_noise_density: f64,
}

impl PreintegrationConfig {
    /// Constructs validated noise and gap parameters.
    ///
    /// Noise densities use `rad/s/sqrt(Hz)` and `m/s^2/sqrt(Hz)` respectively.
    pub fn new(
        max_sample_gap: MonotonicDuration,
        gyroscope_noise_density: f64,
        accelerometer_noise_density: f64,
    ) -> Result<Self, PreintegrationError> {
        if max_sample_gap == MonotonicDuration::ZERO
            || !gyroscope_noise_density.is_finite()
            || !accelerometer_noise_density.is_finite()
            || gyroscope_noise_density < 0.0
            || accelerometer_noise_density < 0.0
        {
            return Err(PreintegrationError::InvalidConfig);
        }
        Ok(Self {
            max_sample_gap,
            gyroscope_noise_density,
            accelerometer_noise_density,
        })
    }

    /// Returns the interval above which missing sensor cadence is diagnosed as a gap.
    #[must_use]
    pub const fn max_sample_gap(self) -> MonotonicDuration {
        self.max_sample_gap
    }

    /// Returns the gyroscope white-noise density.
    #[must_use]
    pub const fn gyroscope_noise_density(self) -> f64 {
        self.gyroscope_noise_density
    }

    /// Returns the accelerometer white-noise density.
    #[must_use]
    pub const fn accelerometer_noise_density(self) -> f64 {
        self.accelerometer_noise_density
    }
}

impl Default for PreintegrationConfig {
    fn default() -> Self {
        Self {
            max_sample_gap: MonotonicDuration::from_nanos(50_000_000),
            gyroscope_noise_density: 0.005,
            accelerometer_noise_density: 0.05,
        }
    }
}

/// Diagonal uncertainty propagated alongside an IMU delta.
///
/// This is a compact health/weighting summary, not a replacement for the full covariance and bias
/// Jacobians required by the later fixed-lag optimiser.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PreintegrationUncertainty {
    rotation_variance: DVec3,
    velocity_variance: DVec3,
    position_variance: DVec3,
}

impl PreintegrationUncertainty {
    /// Returns the approximate rotation-vector variance in radians squared.
    #[must_use]
    pub const fn rotation_variance(self) -> DVec3 {
        self.rotation_variance
    }

    /// Returns approximate delta-velocity variance in `(m/s)^2`.
    #[must_use]
    pub const fn velocity_variance(self) -> DVec3 {
        self.velocity_variance
    }

    /// Returns approximate delta-position variance in metres squared.
    #[must_use]
    pub const fn position_variance(self) -> DVec3 {
        self.position_variance
    }
}

/// Integration-coverage and sensor-health information for one camera interval.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PreintegrationDiagnostics {
    interval_sample_count: usize,
    integrated_segment_count: usize,
    gap_count: usize,
    max_segment_duration: MonotonicDuration,
    start_extrapolation: MonotonicDuration,
    end_extrapolation: MonotonicDuration,
    buffer: BatchDiagnostics,
}

impl PreintegrationDiagnostics {
    /// Returns the number of samples whose timestamp was inside the camera interval.
    #[must_use]
    pub const fn interval_sample_count(self) -> usize {
        self.interval_sample_count
    }

    /// Returns the number of positive-duration numerical integration segments.
    #[must_use]
    pub const fn integrated_segment_count(self) -> usize {
        self.integrated_segment_count
    }

    /// Returns the number of integration segments longer than the configured maximum gap.
    #[must_use]
    pub const fn gap_count(self) -> usize {
        self.gap_count
    }

    /// Returns the longest integration segment.
    #[must_use]
    pub const fn max_segment_duration(self) -> MonotonicDuration {
        self.max_segment_duration
    }

    /// Returns how far the first available measurement was extended backwards.
    #[must_use]
    pub const fn start_extrapolation(self) -> MonotonicDuration {
        self.start_extrapolation
    }

    /// Returns how far the latest available measurement was extended forwards.
    #[must_use]
    pub const fn end_extrapolation(self) -> MonotonicDuration {
        self.end_extrapolation
    }

    /// Returns loss diagnostics reported by the bounded sensor buffer.
    #[must_use]
    pub const fn buffer(self) -> BatchDiagnostics {
        self.buffer
    }

    /// Reports any known gap, extrapolation, capacity loss, or late-arrival loss.
    #[must_use]
    pub const fn is_degraded(self) -> bool {
        self.gap_count > 0
            || self.start_extrapolation.as_nanos() > 0
            || self.end_extrapolation.as_nanos() > 0
            || self.buffer.capacity_drops > 0
            || self.buffer.late_rejections > 0
    }
}

/// Bias-corrected inertial motion accumulated between two camera timestamps.
///
/// Rotation maps the ending body frame into the starting body frame. Velocity and position deltas
/// are expressed in the starting body frame and exclude gravity. [`Self::predict`] applies gravity
/// and the initial body-to-world orientation exactly once.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PreintegratedImu {
    interval: crate::FrameInterval,
    delta_time: MonotonicDuration,
    delta_rotation: DQuat,
    delta_velocity: DVec3,
    delta_position: DVec3,
    bias: ImuBias,
    uncertainty: PreintegrationUncertainty,
    diagnostics: PreintegrationDiagnostics,
}

impl PreintegratedImu {
    /// Returns the integrated frame interval.
    #[must_use]
    pub const fn interval(self) -> crate::FrameInterval {
        self.interval
    }

    /// Returns the integrated duration.
    #[must_use]
    pub const fn delta_time(self) -> MonotonicDuration {
        self.delta_time
    }

    /// Returns the ending-body to starting-body orientation delta.
    #[must_use]
    pub const fn delta_rotation(self) -> DQuat {
        self.delta_rotation
    }

    /// Returns the gravity-free delta velocity in the starting body frame.
    #[must_use]
    pub const fn delta_velocity(self) -> DVec3 {
        self.delta_velocity
    }

    /// Returns the gravity-free delta position in the starting body frame.
    #[must_use]
    pub const fn delta_position(self) -> DVec3 {
        self.delta_position
    }

    /// Returns the fixed bias estimate used for integration.
    #[must_use]
    pub const fn bias(self) -> ImuBias {
        self.bias
    }

    /// Returns the propagated diagonal uncertainty summary.
    #[must_use]
    pub const fn uncertainty(self) -> PreintegrationUncertainty {
        self.uncertainty
    }

    /// Returns sample coverage and gap diagnostics.
    #[must_use]
    pub const fn diagnostics(self) -> PreintegrationDiagnostics {
        self.diagnostics
    }

    /// Propagates a navigation state using this delta and an explicit world gravity vector.
    pub fn predict(
        self,
        initial: NavState,
        gravity_world_mps2: DVec3,
    ) -> Result<NavState, PreintegrationError> {
        if !gravity_world_mps2.is_finite() {
            return Err(PreintegrationError::NonFiniteInput);
        }

        let seconds = self.delta_time.as_secs_f64();
        let body_to_world = initial.body_to_world;
        NavState::new(
            initial.position_world
                + initial.velocity_world * seconds
                + gravity_world_mps2 * (0.5 * seconds * seconds)
                + body_to_world * self.delta_position,
            initial.velocity_world
                + gravity_world_mps2 * seconds
                + body_to_world * self.delta_velocity,
            body_to_world * self.delta_rotation,
        )
    }
}

/// Position, velocity, and body orientation in a gravity-aligned world frame.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct NavState {
    position_world: DVec3,
    velocity_world: DVec3,
    body_to_world: DQuat,
}

impl NavState {
    /// A zero-position, zero-velocity, identity-orientation state.
    pub const ORIGIN: Self = Self {
        position_world: DVec3::ZERO,
        velocity_world: DVec3::ZERO,
        body_to_world: DQuat::IDENTITY,
    };

    /// Constructs a finite state and normalizes its quaternion.
    pub fn new(
        position_world: DVec3,
        velocity_world: DVec3,
        body_to_world: DQuat,
    ) -> Result<Self, PreintegrationError> {
        if !position_world.is_finite()
            || !velocity_world.is_finite()
            || !body_to_world.is_finite()
            || body_to_world.length() <= f64::EPSILON
        {
            return Err(PreintegrationError::NonFiniteInput);
        }
        Ok(Self {
            position_world,
            velocity_world,
            body_to_world: body_to_world.normalize(),
        })
    }

    /// Returns position in world coordinates.
    #[must_use]
    pub const fn position_world(self) -> DVec3 {
        self.position_world
    }

    /// Returns velocity in world coordinates.
    #[must_use]
    pub const fn velocity_world(self) -> DVec3 {
        self.velocity_world
    }

    /// Returns the body-to-world orientation.
    #[must_use]
    pub const fn body_to_world(self) -> DQuat {
        self.body_to_world
    }
}

impl Default for NavState {
    fn default() -> Self {
        Self::ORIGIN
    }
}

/// Preintegrates one validated sensor batch with midpoint integration.
///
/// Values at interval boundaries are linearly interpolated when both bracketing measurements are
/// available. Otherwise the nearest sample is held and the extrapolation is reported. Segments
/// beyond `max_sample_gap` are still integrated deterministically, but are diagnosed and receive
/// inflated uncertainty.
pub fn preintegrate(
    batch: &SensorBatch,
    bias: ImuBias,
    config: PreintegrationConfig,
) -> Result<PreintegratedImu, PreintegrationError> {
    validate_inputs(bias, config)?;

    let interval = batch.interval();
    let basis = batch.time_basis();
    let start = interval.start_exclusive();
    let end = interval.end_inclusive();
    let first_inside = batch.samples().first();
    let last_inside = batch.samples().last();

    let (start_measurement, start_extrapolation) = match (batch.leading_sample(), first_inside) {
        (Some(leading), Some(first)) => (
            interpolate_at(leading, first, start, basis),
            MonotonicDuration::ZERO,
        ),
        (Some(leading), None) => (Measurement::from_sample(leading), MonotonicDuration::ZERO),
        (None, Some(first)) => (
            Measurement::from_sample(first),
            first
                .timestamp(basis)
                .checked_duration_since(start)
                .ok_or(PreintegrationError::InvalidTimeline)?,
        ),
        (None, None) => match batch.trailing_sample() {
            Some(trailing) => (Measurement::from_sample(trailing), interval.duration()),
            None => return Err(PreintegrationError::NoSamples),
        },
    };

    let mut knots = Vec::with_capacity(batch.samples().len() + 2);
    knots.push(TimedMeasurement {
        timestamp: start,
        measurement: start_measurement,
    });
    for sample in batch.samples() {
        let timed = TimedMeasurement {
            timestamp: sample.timestamp(basis),
            measurement: Measurement::from_sample(sample),
        };
        if knots
            .last()
            .is_some_and(|previous| previous.timestamp == timed.timestamp)
        {
            *knots
                .last_mut()
                .expect("the previous knot was just observed") = timed;
        } else {
            knots.push(timed);
        }
    }

    let latest_actual = last_inside.or(batch.leading_sample());
    let end_extrapolation = if knots
        .last()
        .is_some_and(|measurement| measurement.timestamp == end)
    {
        MonotonicDuration::ZERO
    } else {
        let end_measurement = match (latest_actual, batch.trailing_sample()) {
            (Some(before), Some(after)) => interpolate_at(before, after, end, basis),
            (Some(before), None) => Measurement::from_sample(before),
            (None, Some(after)) => Measurement::from_sample(after),
            (None, None) => return Err(PreintegrationError::NoSamples),
        };
        knots.push(TimedMeasurement {
            timestamp: end,
            measurement: end_measurement,
        });

        if batch.trailing_sample().is_some() {
            MonotonicDuration::ZERO
        } else {
            let latest_timestamp = latest_actual
                .map(|sample| sample.timestamp(basis))
                .unwrap_or(start)
                .max(start);
            end.checked_duration_since(latest_timestamp)
                .ok_or(PreintegrationError::InvalidTimeline)?
        }
    };

    let mut delta_rotation = DQuat::IDENTITY;
    let mut delta_velocity = DVec3::ZERO;
    let mut delta_position = DVec3::ZERO;
    let mut uncertainty = PreintegrationUncertainty::default();
    let mut gap_count = 0_usize;
    let mut max_segment_duration = MonotonicDuration::ZERO;
    let mut integrated_segment_count = 0_usize;

    for pair in knots.windows(2) {
        let segment_duration = pair[1]
            .timestamp
            .checked_duration_since(pair[0].timestamp)
            .ok_or(PreintegrationError::InvalidTimeline)?;
        if segment_duration == MonotonicDuration::ZERO {
            continue;
        }
        integrated_segment_count += 1;
        max_segment_duration = max_segment_duration.max(segment_duration);
        if segment_duration > config.max_sample_gap {
            gap_count += 1;
        }

        let seconds = segment_duration.as_secs_f64();
        let angular_velocity =
            (pair[0].measurement.angular_velocity + pair[1].measurement.angular_velocity) * 0.5
                - bias.gyroscope_rad_s;
        let specific_force =
            (pair[0].measurement.specific_force + pair[1].measurement.specific_force) * 0.5
                - bias.accelerometer_mps2;

        let half_rotation = DQuat::from_scaled_axis(angular_velocity * (0.5 * seconds));
        let full_rotation = DQuat::from_scaled_axis(angular_velocity * seconds);
        let acceleration_in_start = (delta_rotation * half_rotation) * specific_force;

        delta_position +=
            delta_velocity * seconds + acceleration_in_start * (0.5 * seconds * seconds);
        delta_velocity += acceleration_in_start * seconds;
        delta_rotation = (delta_rotation * full_rotation).normalize();

        propagate_uncertainty(
            &mut uncertainty,
            seconds,
            specific_force.length(),
            segment_duration,
            config,
        );
    }

    let loss_count = batch
        .diagnostics()
        .capacity_drops
        .saturating_add(batch.diagnostics().late_rejections);
    if loss_count > 0 {
        let loss_factor = 1.0 + loss_count as f64;
        uncertainty.rotation_variance *= loss_factor;
        uncertainty.velocity_variance *= loss_factor;
        uncertainty.position_variance *= loss_factor;
    }

    Ok(PreintegratedImu {
        interval,
        delta_time: interval.duration(),
        delta_rotation,
        delta_velocity,
        delta_position,
        bias,
        uncertainty,
        diagnostics: PreintegrationDiagnostics {
            interval_sample_count: batch.samples().len(),
            integrated_segment_count,
            gap_count,
            max_segment_duration,
            start_extrapolation,
            end_extrapolation,
            buffer: batch.diagnostics(),
        },
    })
}

/// Invalid preintegration configuration, state, or sensor coverage.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum PreintegrationError {
    /// No leading, interval, or trailing sample was available.
    #[error("at least one motion sample is required for preintegration")]
    NoSamples,
    /// A vector, quaternion, noise density, or bias was not finite.
    #[error("preintegration inputs must be finite")]
    NonFiniteInput,
    /// Noise density or maximum-gap configuration was invalid.
    #[error("preintegration configuration is invalid")]
    InvalidConfig,
    /// Timestamp arithmetic contradicted the validated batch ordering.
    #[error("sensor batch timeline is invalid")]
    InvalidTimeline,
}

#[derive(Clone, Copy, Debug)]
struct Measurement {
    angular_velocity: DVec3,
    specific_force: DVec3,
}

impl Measurement {
    fn from_sample(sample: &NormalizedMotionSample) -> Self {
        Self {
            angular_velocity: sample.angular_velocity_rad_s(),
            specific_force: sample.specific_force_mps2(),
        }
    }

    fn lerp(self, other: Self, fraction: f64) -> Self {
        Self {
            angular_velocity: self.angular_velocity.lerp(other.angular_velocity, fraction),
            specific_force: self.specific_force.lerp(other.specific_force, fraction),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct TimedMeasurement {
    timestamp: MonotonicTimestamp,
    measurement: Measurement,
}

fn interpolate_at(
    before: &NormalizedMotionSample,
    after: &NormalizedMotionSample,
    timestamp: MonotonicTimestamp,
    basis: crate::SensorTimeBasis,
) -> Measurement {
    let before_time = before.timestamp(basis);
    let after_time = after.timestamp(basis);
    let total = after_time
        .checked_duration_since(before_time)
        .map_or(0.0, MonotonicDuration::as_secs_f64);
    if total <= f64::EPSILON {
        return Measurement::from_sample(after);
    }
    let elapsed = timestamp
        .checked_duration_since(before_time)
        .map_or(0.0, MonotonicDuration::as_secs_f64);
    Measurement::from_sample(before).lerp(Measurement::from_sample(after), elapsed / total)
}

fn validate_inputs(bias: ImuBias, config: PreintegrationConfig) -> Result<(), PreintegrationError> {
    if !bias.gyroscope_rad_s.is_finite() || !bias.accelerometer_mps2.is_finite() {
        return Err(PreintegrationError::NonFiniteInput);
    }
    if config.max_sample_gap == MonotonicDuration::ZERO
        || !config.gyroscope_noise_density.is_finite()
        || !config.accelerometer_noise_density.is_finite()
        || config.gyroscope_noise_density < 0.0
        || config.accelerometer_noise_density < 0.0
    {
        return Err(PreintegrationError::InvalidConfig);
    }
    Ok(())
}

fn propagate_uncertainty(
    uncertainty: &mut PreintegrationUncertainty,
    seconds: f64,
    specific_force_magnitude: f64,
    segment_duration: MonotonicDuration,
    config: PreintegrationConfig,
) {
    let gap_ratio = segment_duration.as_secs_f64() / config.max_sample_gap.as_secs_f64();
    let inflation = gap_ratio.max(1.0).powi(2);
    let gyro_variance = config.gyroscope_noise_density.powi(2) * seconds * inflation;
    let acceleration_variance = config.accelerometer_noise_density.powi(2) * seconds * inflation;

    uncertainty.rotation_variance += DVec3::splat(gyro_variance);
    let orientation_coupling =
        uncertainty.rotation_variance * specific_force_magnitude.powi(2) * seconds.powi(2) / 3.0;
    uncertainty.position_variance += uncertainty.velocity_variance * seconds.powi(2)
        + DVec3::splat(
            config.accelerometer_noise_density.powi(2) * seconds.powi(3) * inflation / 3.0,
        );
    uncertainty.velocity_variance += DVec3::splat(acceleration_variance) + orientation_coupling;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        BatchDiagnostics, FrameInterval, SensorSequence, SensorTimeBasis, SourceScreenOrientation,
    };
    use approx::assert_relative_eq;

    const GRAVITY: f64 = 9.806_65;

    fn timestamp(milliseconds: u64) -> MonotonicTimestamp {
        MonotonicTimestamp::from_nanos(milliseconds * 1_000_000)
    }

    fn sample(
        sequence: u64,
        milliseconds: u64,
        angular_velocity: DVec3,
        specific_force: DVec3,
    ) -> NormalizedMotionSample {
        NormalizedMotionSample::new(
            SensorSequence::new(sequence),
            timestamp(milliseconds),
            timestamp(milliseconds + 1),
            Some(MonotonicDuration::from_nanos(10_000_000)),
            angular_velocity,
            specific_force,
            None,
            SourceScreenOrientation::PortraitPrimary,
        )
        .unwrap()
    }

    fn batch(
        leading: NormalizedMotionSample,
        samples: Vec<NormalizedMotionSample>,
        end_ms: u64,
    ) -> SensorBatch {
        SensorBatch::new(
            SensorTimeBasis::Event,
            FrameInterval::new(timestamp(0), timestamp(end_ms)).unwrap(),
            Some(leading),
            samples,
            None,
            BatchDiagnostics::default(),
        )
        .unwrap()
    }

    #[test]
    fn constant_rotation_integrates_to_expected_orientation() {
        let angular_velocity = DVec3::Z;
        let leading = sample(0, 0, angular_velocity, DVec3::ZERO);
        let samples = (1..=100)
            .map(|index| sample(index, index * 10, angular_velocity, DVec3::ZERO))
            .collect();
        let result = preintegrate(
            &batch(leading, samples, 1_000),
            ImuBias::ZERO,
            PreintegrationConfig::default(),
        )
        .unwrap();

        let scaled_axis = result.delta_rotation().to_scaled_axis();
        assert_relative_eq!(scaled_axis.x, 0.0, epsilon = 1.0e-12);
        assert_relative_eq!(scaled_axis.y, 0.0, epsilon = 1.0e-12);
        assert_relative_eq!(scaled_axis.z, 1.0, epsilon = 1.0e-10);
        assert_eq!(result.diagnostics().gap_count(), 0);
    }

    #[test]
    fn stationary_specific_force_is_cancelled_by_world_gravity_once() {
        let specific_force = DVec3::Z * GRAVITY;
        let leading = sample(0, 0, DVec3::ZERO, specific_force);
        let samples = (1..=100)
            .map(|index| sample(index, index * 10, DVec3::ZERO, specific_force))
            .collect();
        let result = preintegrate(
            &batch(leading, samples, 1_000),
            ImuBias::ZERO,
            PreintegrationConfig::default(),
        )
        .unwrap();

        assert_relative_eq!(result.delta_velocity().z, GRAVITY, epsilon = 1.0e-10);
        assert_relative_eq!(result.delta_position().z, 0.5 * GRAVITY, epsilon = 1.0e-10);

        let predicted = result
            .predict(NavState::ORIGIN, -DVec3::Z * GRAVITY)
            .unwrap();
        assert_relative_eq!(predicted.velocity_world().length(), 0.0, epsilon = 1.0e-10);
        assert_relative_eq!(predicted.position_world().length(), 0.0, epsilon = 1.0e-10);
    }

    #[test]
    fn configured_bias_is_removed_before_integration() {
        let gyro_bias = DVec3::new(0.01, -0.02, 0.03);
        let accel_bias = DVec3::new(0.2, -0.1, 0.05);
        let leading = sample(0, 0, gyro_bias, accel_bias);
        let samples = (1..=10)
            .map(|index| sample(index, index * 10, gyro_bias, accel_bias))
            .collect();
        let result = preintegrate(
            &batch(leading, samples, 100),
            ImuBias::new(gyro_bias, accel_bias).unwrap(),
            PreintegrationConfig::default(),
        )
        .unwrap();

        assert_relative_eq!(
            result.delta_rotation().to_scaled_axis().length(),
            0.0,
            epsilon = 1.0e-12
        );
        assert_relative_eq!(result.delta_velocity().length(), 0.0, epsilon = 1.0e-12);
        assert_relative_eq!(result.delta_position().length(), 0.0, epsilon = 1.0e-12);
    }

    #[test]
    fn long_sensor_gaps_are_diagnosed_and_inflate_uncertainty() {
        let leading = sample(0, 0, DVec3::ZERO, DVec3::ZERO);
        let dense_samples = (1..=10)
            .map(|index| sample(index, index * 10, DVec3::ZERO, DVec3::ZERO))
            .collect();
        let dense = preintegrate(
            &batch(leading.clone(), dense_samples, 100),
            ImuBias::ZERO,
            PreintegrationConfig::default(),
        )
        .unwrap();
        let sparse = preintegrate(
            &batch(leading, vec![sample(1, 100, DVec3::ZERO, DVec3::ZERO)], 100),
            ImuBias::ZERO,
            PreintegrationConfig::default(),
        )
        .unwrap();

        assert_eq!(sparse.diagnostics().gap_count(), 1);
        assert_eq!(
            sparse.diagnostics().max_segment_duration(),
            MonotonicDuration::from_nanos(100_000_000)
        );
        assert!(
            sparse.uncertainty().rotation_variance().x > dense.uncertainty().rotation_variance().x
        );
        assert!(sparse.diagnostics().is_degraded());
    }

    #[test]
    fn preintegrated_result_round_trips_through_serde() {
        let result = preintegrate(
            &batch(
                sample(0, 0, DVec3::ZERO, DVec3::Z * GRAVITY),
                vec![sample(1, 10, DVec3::ZERO, DVec3::Z * GRAVITY)],
                10,
            ),
            ImuBias::ZERO,
            PreintegrationConfig::default(),
        )
        .unwrap();

        let encoded = serde_json::to_string(&result).unwrap();
        let decoded: PreintegratedImu = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, result);
    }
}
