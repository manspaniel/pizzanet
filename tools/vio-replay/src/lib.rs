//! Deterministic offline sensor generation and preintegration reporting.

#![forbid(unsafe_code)]

use anyhow::{Context, Result, bail};
use serde::Serialize;
use vio_core::{
    BatchDiagnostics, DQuat, DVec3, FrameInterval, ImuBias, MonotonicDuration, MonotonicTimestamp,
    NormalizedMotionSample, PreintegratedImu, PreintegrationConfig, SensorBatch, SensorBuffer,
    SensorSequence, SensorTimeBasis, SourceScreenOrientation, preintegrate,
};

const REPORT_SCHEMA_VERSION: u32 = 1;
const MAX_SIMULATED_SAMPLES: u64 = 10_000_000;

/// Constant-motion parameters for one deterministic IMU simulation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimulationConfig {
    duration_seconds: f64,
    sample_rate_hz: f64,
    angular_velocity_rad_s: DVec3,
    specific_force_mps2: DVec3,
}

impl SimulationConfig {
    /// Creates a simulation after validating its duration, cadence, and vectors.
    pub fn new(
        duration_seconds: f64,
        sample_rate_hz: f64,
        angular_velocity_rad_s: DVec3,
        specific_force_mps2: DVec3,
    ) -> Result<Self> {
        if !duration_seconds.is_finite() || duration_seconds <= 0.0 {
            bail!("duration must be finite and greater than zero");
        }
        if !sample_rate_hz.is_finite() || sample_rate_hz <= 0.0 {
            bail!("sample rate must be finite and greater than zero");
        }
        if !angular_velocity_rad_s.is_finite() || !specific_force_mps2.is_finite() {
            bail!("motion vectors must contain only finite values");
        }

        let duration = MonotonicDuration::try_from_secs_f64(duration_seconds)
            .context("duration cannot be represented in integer nanoseconds")?;
        let sample_period = MonotonicDuration::try_from_secs_f64(1.0 / sample_rate_hz)
            .context("sample period cannot be represented in integer nanoseconds")?;
        if duration == MonotonicDuration::ZERO {
            bail!("duration rounds to zero nanoseconds");
        }
        if sample_period == MonotonicDuration::ZERO {
            bail!("sample rate exceeds nanosecond timestamp resolution");
        }

        let intervals = duration.as_nanos().div_ceil(sample_period.as_nanos());
        if intervals >= MAX_SIMULATED_SAMPLES {
            bail!("simulation would exceed the limit of {MAX_SIMULATED_SAMPLES} samples");
        }

        Ok(Self {
            duration_seconds,
            sample_rate_hz,
            angular_velocity_rad_s,
            specific_force_mps2,
        })
    }
}

/// A generated sensor batch together with its preintegration report.
#[derive(Clone, Debug)]
pub struct SimulationOutput {
    batch: SensorBatch,
    report: ReplayReport,
}

impl SimulationOutput {
    /// Returns the generated batch, suitable for serializing as replay input.
    #[must_use]
    pub const fn batch(&self) -> &SensorBatch {
        &self.batch
    }

    /// Returns the diagnostics generated from the batch.
    #[must_use]
    pub const fn report(&self) -> &ReplayReport {
        &self.report
    }
}

/// Stable, unit-labelled JSON report produced by simulation and inspection.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ReplayReport {
    schema_version: u32,
    source: ReportSource,
    batch: BatchReport,
    preintegration: PreintegrationReport,
}

/// Generates constant normalized measurements, batches them, and preintegrates the interval.
pub fn simulate(config: SimulationConfig) -> Result<SimulationOutput> {
    let duration = MonotonicDuration::try_from_secs_f64(config.duration_seconds)
        .context("duration cannot be represented in integer nanoseconds")?;
    let sample_period = MonotonicDuration::try_from_secs_f64(1.0 / config.sample_rate_hz)
        .context("sample period cannot be represented in integer nanoseconds")?;
    let interval_count = duration.as_nanos().div_ceil(sample_period.as_nanos());
    let generated_sample_count = interval_count
        .checked_add(1)
        .context("generated sample count overflowed")?;
    let capacity = usize::try_from(generated_sample_count)
        .context("generated sample count does not fit this platform")?;
    let mut buffer = SensorBuffer::new(capacity, SensorTimeBasis::Event)
        .context("failed to create sensor buffer")?;

    for sequence in 0..=interval_count {
        let timestamp_nanos = sequence
            .checked_mul(sample_period.as_nanos())
            .context("simulated timestamp overflowed")?;
        let timestamp = MonotonicTimestamp::from_nanos(timestamp_nanos);
        let sample = NormalizedMotionSample::new(
            SensorSequence::new(sequence),
            timestamp,
            timestamp,
            Some(sample_period),
            config.angular_velocity_rad_s,
            config.specific_force_mps2,
            None,
            SourceScreenOrientation::PortraitPrimary,
        )
        .context("failed to construct normalized motion sample")?;
        let _ = buffer.push(sample);
    }

    let interval = FrameInterval::new(
        MonotonicTimestamp::ZERO,
        MonotonicTimestamp::from_nanos(duration.as_nanos()),
    )
    .context("failed to construct simulation interval")?;
    let batch = buffer
        .drain_interval(interval)
        .context("failed to batch simulated samples")?;
    let source = ReportSource::Simulation {
        requested_duration_seconds: config.duration_seconds,
        requested_sample_rate_hz: config.sample_rate_hz,
        effective_sample_period_ns: sample_period.as_nanos(),
        generated_sample_count,
        angular_velocity_rad_s: Vector3::from(config.angular_velocity_rad_s),
        specific_force_mps2: Vector3::from(config.specific_force_mps2),
    };
    let report = report_for_batch(&batch, source)?;

    Ok(SimulationOutput { batch, report })
}

/// Preintegrates a deserialized sensor batch using the same settings as [`simulate`].
pub fn inspect(batch: &SensorBatch) -> Result<ReplayReport> {
    report_for_batch(batch, ReportSource::SensorBatch)
}

fn report_for_batch(batch: &SensorBatch, source: ReportSource) -> Result<ReplayReport> {
    let config = PreintegrationConfig::default();
    let bias = ImuBias::ZERO;
    let integrated = preintegrate(batch, bias, config).context("failed to preintegrate batch")?;

    Ok(ReplayReport {
        schema_version: REPORT_SCHEMA_VERSION,
        source,
        batch: BatchReport::from(batch),
        preintegration: PreintegrationReport::new(integrated, config),
    })
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ReportSource {
    Simulation {
        requested_duration_seconds: f64,
        requested_sample_rate_hz: f64,
        effective_sample_period_ns: u64,
        generated_sample_count: u64,
        angular_velocity_rad_s: Vector3,
        specific_force_mps2: Vector3,
    },
    SensorBatch,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct BatchReport {
    time_basis: SensorTimeBasis,
    interval_start_exclusive_ns: u64,
    interval_end_inclusive_ns: u64,
    interval_duration_ns: u64,
    interval_sample_count: usize,
    leading_sample_sequence: Option<u64>,
    trailing_sample_sequence: Option<u64>,
    buffer_diagnostics: BatchDiagnostics,
}

impl From<&SensorBatch> for BatchReport {
    fn from(batch: &SensorBatch) -> Self {
        let interval = batch.interval();
        Self {
            time_basis: batch.time_basis(),
            interval_start_exclusive_ns: interval.start_exclusive().as_nanos(),
            interval_end_inclusive_ns: interval.end_inclusive().as_nanos(),
            interval_duration_ns: interval.duration().as_nanos(),
            interval_sample_count: batch.samples().len(),
            leading_sample_sequence: batch.leading_sample().map(|sample| sample.sequence().get()),
            trailing_sample_sequence: batch
                .trailing_sample()
                .map(|sample| sample.sequence().get()),
            buffer_diagnostics: batch.diagnostics(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct PreintegrationReport {
    settings: PreintegrationSettings,
    delta_time_seconds: f64,
    delta_rotation_xyzw: Quaternion,
    delta_rotation_scaled_axis_rad: Vector3,
    delta_velocity_mps: Vector3,
    delta_position_m: Vector3,
    uncertainty: UncertaintyReport,
    diagnostics: PreintegrationDiagnosticsReport,
}

impl PreintegrationReport {
    fn new(integrated: PreintegratedImu, config: PreintegrationConfig) -> Self {
        let uncertainty = integrated.uncertainty();
        let diagnostics = integrated.diagnostics();
        Self {
            settings: PreintegrationSettings {
                gyroscope_bias_rad_s: Vector3::from(integrated.bias().gyroscope_rad_s()),
                accelerometer_bias_mps2: Vector3::from(integrated.bias().accelerometer_mps2()),
                max_sample_gap_ns: config.max_sample_gap().as_nanos(),
                gyroscope_noise_density_rad_s_sqrt_hz: config.gyroscope_noise_density(),
                accelerometer_noise_density_mps2_sqrt_hz: config.accelerometer_noise_density(),
            },
            delta_time_seconds: integrated.delta_time().as_secs_f64(),
            delta_rotation_xyzw: Quaternion::from(integrated.delta_rotation()),
            delta_rotation_scaled_axis_rad: Vector3::from(
                integrated.delta_rotation().to_scaled_axis(),
            ),
            delta_velocity_mps: Vector3::from(integrated.delta_velocity()),
            delta_position_m: Vector3::from(integrated.delta_position()),
            uncertainty: UncertaintyReport {
                rotation_variance_rad2: Vector3::from(uncertainty.rotation_variance()),
                velocity_variance_m2_s2: Vector3::from(uncertainty.velocity_variance()),
                position_variance_m2: Vector3::from(uncertainty.position_variance()),
            },
            diagnostics: PreintegrationDiagnosticsReport {
                interval_sample_count: diagnostics.interval_sample_count(),
                integrated_segment_count: diagnostics.integrated_segment_count(),
                gap_count: diagnostics.gap_count(),
                max_segment_duration_ns: diagnostics.max_segment_duration().as_nanos(),
                start_extrapolation_ns: diagnostics.start_extrapolation().as_nanos(),
                end_extrapolation_ns: diagnostics.end_extrapolation().as_nanos(),
                degraded: diagnostics.is_degraded(),
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct PreintegrationSettings {
    gyroscope_bias_rad_s: Vector3,
    accelerometer_bias_mps2: Vector3,
    max_sample_gap_ns: u64,
    gyroscope_noise_density_rad_s_sqrt_hz: f64,
    accelerometer_noise_density_mps2_sqrt_hz: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct UncertaintyReport {
    rotation_variance_rad2: Vector3,
    velocity_variance_m2_s2: Vector3,
    position_variance_m2: Vector3,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct PreintegrationDiagnosticsReport {
    interval_sample_count: usize,
    integrated_segment_count: usize,
    gap_count: usize,
    max_segment_duration_ns: u64,
    start_extrapolation_ns: u64,
    end_extrapolation_ns: u64,
    degraded: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
struct Vector3 {
    x: f64,
    y: f64,
    z: f64,
}

impl From<DVec3> for Vector3 {
    fn from(value: DVec3) -> Self {
        Self {
            x: value.x,
            y: value.y,
            z: value.z,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
struct Quaternion {
    x: f64,
    y: f64,
    z: f64,
    w: f64,
}

impl From<DQuat> for Quaternion {
    fn from(value: DQuat) -> Self {
        Self {
            x: value.x,
            y: value.y,
            z: value.z,
            w: value.w,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(duration_seconds: f64, sample_rate_hz: f64) -> SimulationConfig {
        SimulationConfig::new(
            duration_seconds,
            sample_rate_hz,
            DVec3::new(0.0, 0.0, 0.5),
            DVec3::ZERO,
        )
        .unwrap()
    }

    #[test]
    fn aligned_simulation_has_expected_samples_and_segments() {
        let output = simulate(config(1.0, 100.0)).unwrap();

        assert_eq!(output.batch().samples().len(), 100);
        assert!(output.batch().leading_sample().is_some());
        assert!(output.batch().trailing_sample().is_none());
        assert_eq!(output.report.batch.interval_sample_count, 100);
        assert_eq!(
            output
                .report
                .preintegration
                .diagnostics
                .integrated_segment_count,
            100
        );
    }

    #[test]
    fn non_aligned_simulation_keeps_a_trailing_interpolation_sample() {
        let output = simulate(config(0.025, 100.0)).unwrap();

        assert_eq!(output.batch().samples().len(), 2);
        assert_eq!(
            output.batch().trailing_sample().unwrap().sequence().get(),
            3
        );
        assert_eq!(
            output
                .report
                .preintegration
                .diagnostics
                .integrated_segment_count,
            3
        );
        assert_eq!(
            output
                .report
                .preintegration
                .diagnostics
                .end_extrapolation_ns,
            0
        );
    }

    #[test]
    fn simulation_and_integration_are_deterministic() {
        let first = simulate(config(1.0, 100.0)).unwrap();
        let second = simulate(config(1.0, 100.0)).unwrap();

        assert_eq!(first.batch(), second.batch());
        assert_eq!(first.report(), second.report());
        assert_eq!(
            serde_json::to_vec_pretty(first.report()).unwrap(),
            serde_json::to_vec_pretty(second.report()).unwrap()
        );

        let rotation = &first.report.preintegration.delta_rotation_scaled_axis_rad;
        assert!(rotation.x.abs() < 1.0e-12);
        assert!(rotation.y.abs() < 1.0e-12);
        assert!((rotation.z - 0.5).abs() < 1.0e-10);
    }

    #[test]
    fn inspection_reuses_the_simulation_report_path() {
        let output = simulate(config(1.0, 50.0)).unwrap();
        let inspected = inspect(output.batch()).unwrap();

        assert_eq!(inspected.batch, output.report.batch);
        assert_eq!(inspected.preintegration, output.report.preintegration);
        assert_eq!(inspected.source, ReportSource::SensorBatch);
    }
}
