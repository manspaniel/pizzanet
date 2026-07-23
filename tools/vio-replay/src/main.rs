//! Command-line interface for offline VIO sensor replay.

#![forbid(unsafe_code)]

use anyhow::{Context, Result, bail};
use ar_tracker_wasm::{ArTracker, TRACKING_STATE_TRACKING};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::{
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    str::FromStr,
};
use vio_core::{DQuat, DVec3, SensorBatch};
use vio_replay::{SimulationConfig, inspect, simulate};

#[derive(Debug, Parser)]
#[command(about = "Generate and inspect deterministic VIO sensor replays")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Generate constant normalized IMU measurements and preintegrate them.
    Simulate {
        /// Simulated interval duration in seconds.
        #[arg(long, default_value_t = 1.0)]
        duration_seconds: f64,

        /// IMU sample rate in hertz.
        #[arg(long, default_value_t = 100.0)]
        sample_rate_hz: f64,

        /// Constant body angular velocity in radians/second as X,Y,Z.
        #[arg(long, value_name = "X,Y,Z", default_value = "0,0,0")]
        angular_velocity_rad_s: Vector3Argument,

        /// Constant body specific force in metres/second squared as X,Y,Z.
        #[arg(long, value_name = "X,Y,Z", default_value = "0,0,9.80665")]
        specific_force_mps2: Vector3Argument,

        /// Pretty JSON report path; omit to write to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Also write the generated raw SensorBatch as pretty JSON.
        #[arg(long)]
        batch_output: Option<PathBuf>,
    },

    /// Read a JSON SensorBatch and report its preintegration diagnostics.
    Inspect {
        /// SensorBatch JSON produced by this tool or another acquisition adapter.
        input: PathBuf,

        /// Pretty JSON report path; omit to write to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Replay a browser recording from raw grayscale frames and captured sensor sidecars.
    Recording {
        /// Recording directory containing manifest and NDJSON sidecars.
        recording: PathBuf,

        /// Headerless GRAY8 frames decoded at 10 fps in tracker dimensions.
        #[arg(long)]
        frames: PathBuf,

        /// Override the camera field of view along the longer frame axis.
        #[arg(long)]
        long_axis_fov_degrees: Option<f64>,

        /// Hold back sensor events relative to tracker frames for timing calibration.
        #[arg(long, default_value_t = 0.0)]
        sensor_delay_milliseconds: f64,

        /// Camera-to-DeviceOrientation timing offset used for rotation compensation.
        #[arg(long, default_value_t = 40.0)]
        visual_orientation_delay_milliseconds: f64,

        /// Pretty JSON report path; omit to write to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Optional NDJSON file containing one estimator state per replayed frame.
        #[arg(long)]
        trace_output: Option<PathBuf>,
    },
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RecordingManifest {
    camera: RecordingCamera,
    device: RecordingDevice,
    recording_id: String,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RecordingCamera {
    tracker_frame_height: usize,
    tracker_frame_width: usize,
}

#[derive(Debug, Deserialize)]
struct RecordingDevice {
    platform: String,
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
struct OptionalVector3 {
    x: Option<f64>,
    y: Option<f64>,
    z: Option<f64>,
}

#[derive(Clone, Copy, Debug, Default, Deserialize)]
struct RotationRate {
    alpha: Option<f64>,
    beta: Option<f64>,
    gamma: Option<f64>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum RecordedSensorEvent {
    DeviceMotion {
        #[serde(rename = "eventTimestampMilliseconds")]
        event_timestamp_milliseconds: f64,
        #[serde(rename = "receiptTimestampMilliseconds")]
        receipt_timestamp_milliseconds: f64,
        #[serde(rename = "intervalMilliseconds")]
        interval: f64,
        acceleration: OptionalVector3,
        #[serde(rename = "accelerationIncludingGravity")]
        acceleration_including_gravity: OptionalVector3,
        #[serde(rename = "rotationRateDegreesPerSecond")]
        rotation_rate: RotationRate,
        #[serde(rename = "screenAngleDegrees")]
        screen_angle_degrees: f64,
    },
    DeviceOrientation {
        #[serde(rename = "eventTimestampMilliseconds")]
        event_timestamp_milliseconds: f64,
        #[serde(rename = "alphaDegrees")]
        alpha_degrees: Option<f64>,
        #[serde(rename = "betaDegrees")]
        beta_degrees: Option<f64>,
        #[serde(rename = "gammaDegrees")]
        gamma_degrees: Option<f64>,
        #[serde(rename = "screenAngleDegrees")]
        screen_angle_degrees: f64,
    },
}

impl RecordedSensorEvent {
    fn timestamp_milliseconds(&self) -> f64 {
        match self {
            Self::DeviceMotion {
                event_timestamp_milliseconds,
                ..
            }
            | Self::DeviceOrientation {
                event_timestamp_milliseconds,
                ..
            } => *event_timestamp_milliseconds,
        }
    }

    fn replay_timestamp_milliseconds(&self, sensor_delay_milliseconds: f64) -> f64 {
        delayed_timestamp_milliseconds(self.timestamp_milliseconds(), sensor_delay_milliseconds)
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RecordedTrackerFrame {
    frame_id: u32,
    performance_timestamp_milliseconds: f64,
    recording_time_milliseconds: f64,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct RecordingReplayReport {
    recording_id: String,
    long_axis_field_of_view_degrees: f64,
    sensor_delay_milliseconds: f64,
    visual_orientation_delay_milliseconds: f64,
    replayed_frames: usize,
    accepted_motion_samples: u64,
    keyframes_selected: u64,
    relocalizations: u64,
    limited_frame_fraction: f64,
    path_length_metres: f64,
    net_displacement_metres: f64,
    maximum_displacement_metres: f64,
    closure_ratio: f64,
    endpoint_orientation_error_degrees: f64,
    vertical_range_metres: f64,
    median_visual_matches: u32,
    median_visual_inliers: u32,
}

#[derive(Clone, Copy, Debug)]
struct ReplayedPose {
    position: DVec3,
    orientation: DQuat,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SensorReplayOutcome {
    Pushed,
    SkippedIncomplete,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ReplayTraceFrame {
    frame_id: u32,
    recording_time_milliseconds: f64,
    position: [f64; 3],
    inertial_velocity_metres_per_second: [f64; 3],
    matches: u32,
    inliers: u32,
    keyframe_id: u32,
    keyframe_count: u64,
    relocalization_count: u64,
    tracking: bool,
    stationary_candidate: bool,
}

#[derive(Clone, Copy, Debug)]
struct Vector3Argument(DVec3);

impl FromStr for Vector3Argument {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let components = value
            .split(',')
            .map(str::trim)
            .map(|component| {
                component
                    .parse::<f64>()
                    .map_err(|error| format!("invalid component `{component}`: {error}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let [x, y, z] = components.as_slice() else {
            return Err("expected exactly three comma-separated components: X,Y,Z".to_owned());
        };
        let vector = DVec3::new(*x, *y, *z);
        if !vector.is_finite() {
            return Err("vector components must be finite".to_owned());
        }
        Ok(Self(vector))
    }
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Simulate {
            duration_seconds,
            sample_rate_hz,
            angular_velocity_rad_s,
            specific_force_mps2,
            output,
            batch_output,
        } => {
            if let (Some(report_path), Some(batch_path)) = (&output, &batch_output)
                && report_path == batch_path
            {
                bail!("report and batch output paths must be different");
            }

            let config = SimulationConfig::new(
                duration_seconds,
                sample_rate_hz,
                angular_velocity_rad_s.0,
                specific_force_mps2.0,
            )?;
            let simulation = simulate(config)?;
            if let Some(path) = batch_output {
                write_json_file(&path, simulation.batch())
                    .with_context(|| format!("failed to write batch to {}", path.display()))?;
            }
            write_json(output.as_deref(), simulation.report())?;
        }
        Command::Inspect { input, output } => {
            let file = File::open(&input)
                .with_context(|| format!("failed to open batch {}", input.display()))?;
            let batch: SensorBatch =
                serde_json::from_reader(BufReader::new(file)).with_context(|| {
                    format!("failed to parse SensorBatch JSON in {}", input.display())
                })?;
            let report = inspect(&batch)?;
            write_json(output.as_deref(), &report)?;
        }
        Command::Recording {
            recording,
            frames,
            long_axis_fov_degrees,
            sensor_delay_milliseconds,
            visual_orientation_delay_milliseconds,
            output,
            trace_output,
        } => {
            let (report, trace) = replay_recording(
                &recording,
                &frames,
                long_axis_fov_degrees,
                sensor_delay_milliseconds,
                visual_orientation_delay_milliseconds,
            )?;
            if let Some(path) = trace_output {
                write_ndjson_file(&path, &trace).with_context(|| {
                    format!("failed to write replay trace to {}", path.display())
                })?;
            }
            write_json(output.as_deref(), &report)?;
        }
    }

    Ok(())
}

fn replay_recording(
    recording: &Path,
    frames_path: &Path,
    long_axis_fov_degrees: Option<f64>,
    sensor_delay_milliseconds: f64,
    visual_orientation_delay_milliseconds: f64,
) -> Result<(RecordingReplayReport, Vec<ReplayTraceFrame>)> {
    if !sensor_delay_milliseconds.is_finite() || sensor_delay_milliseconds < 0.0 {
        bail!("sensor delay must be a finite non-negative number of milliseconds");
    }
    if !tracker_delay_is_valid(visual_orientation_delay_milliseconds) {
        bail!("visual orientation delay must be between 0 and 250 milliseconds");
    }
    let manifest: RecordingManifest =
        serde_json::from_reader(BufReader::new(File::open(recording.join("manifest.json"))?))?;
    let sensors = read_ndjson::<RecordedSensorEvent>(&recording.join("sensor-events.ndjson"))?;
    let tracker_frames =
        read_ndjson::<RecordedTrackerFrame>(&recording.join("tracker-frames.ndjson"))?;
    let frame_size = manifest
        .camera
        .tracker_frame_width
        .checked_mul(manifest.camera.tracker_frame_height)
        .context("tracker frame dimensions overflow")?;
    let mut frame_reader = BufReader::new(File::open(frames_path)?);
    let mut pixels = vec![0_u8; frame_size];
    let mut sensor_index = 0;
    let mut tracker = ArTracker::new();
    if !tracker.set_visual_orientation_delay_milliseconds(visual_orientation_delay_milliseconds) {
        bail!("visual orientation delay must be between 0 and 250 milliseconds");
    }
    if let Some(degrees) = long_axis_fov_degrees
        && !tracker.set_long_axis_field_of_view_degrees(degrees)
    {
        bail!("long-axis field of view must be finite and between 30 and 130 degrees");
    }
    let apple_sign = if is_apple_platform(&manifest.device.platform) {
        -1.0
    } else {
        1.0
    };
    let mut positions = Vec::with_capacity(tracker_frames.len());
    let mut poses = Vec::with_capacity(tracker_frames.len());
    let mut inliers = Vec::with_capacity(tracker_frames.len());
    let mut matches = Vec::with_capacity(tracker_frames.len());
    let mut tracking_frames = 0_usize;
    let mut trace = Vec::with_capacity(tracker_frames.len());

    for frame in &tracker_frames {
        while let Some(sensor) = sensors.get(sensor_index)
            && sensor.timestamp_milliseconds() <= frame.performance_timestamp_milliseconds
        {
            push_recorded_sensor(&mut tracker, sensor, apple_sign, sensor_delay_milliseconds)
                .with_context(|| format!("tracker rejected sensor event {sensor_index}"))?;
            sensor_index += 1;
        }
        frame_reader.read_exact(&mut pixels).with_context(|| {
            format!(
                "raw frame file ended before tracker frame {}",
                frame.frame_id
            )
        })?;
        push_recorded_luma_frame(&mut tracker, frame, manifest.camera, &pixels)?;
        let pose = tracker.pose();
        let replayed_pose = ReplayedPose {
            position: DVec3::new(pose[0], pose[1], pose[2]),
            orientation: DQuat::from_xyzw(pose[3], pose[4], pose[5], pose[6]),
        };
        positions.push(replayed_pose.position);
        poses.push(replayed_pose);
        inliers.push(tracker.visual_inlier_count());
        matches.push(tracker.visual_match_count());
        if tracker.tracking_state() == TRACKING_STATE_TRACKING {
            tracking_frames += 1;
        }
        let inertial_velocity = tracker.inertial_velocity();
        trace.push(ReplayTraceFrame {
            frame_id: frame.frame_id,
            recording_time_milliseconds: frame.recording_time_milliseconds,
            position: replayed_pose.position.to_array(),
            inertial_velocity_metres_per_second: [
                inertial_velocity[0],
                inertial_velocity[1],
                inertial_velocity[2],
            ],
            matches: tracker.visual_match_count(),
            inliers: tracker.visual_inlier_count(),
            keyframe_id: tracker.latest_visual_keyframe_id(),
            keyframe_count: tracker.visual_keyframe_count(),
            relocalization_count: tracker.visual_relocalization_count(),
            tracking: tracker.tracking_state() == TRACKING_STATE_TRACKING,
            stationary_candidate: tracker.inertial_stationary_candidate(),
        });
    }

    inliers.sort_unstable();
    matches.sort_unstable();
    let path_length_metres = positions
        .windows(2)
        .map(|pair| pair[0].distance(pair[1]))
        .sum();
    let net_displacement_metres = positions
        .first()
        .zip(positions.last())
        .map_or(0.0, |(first, last)| first.distance(*last));
    let maximum_displacement_metres = positions.first().map_or(0.0, |first| {
        positions
            .iter()
            .map(|position| first.distance(*position))
            .fold(0.0, f64::max)
    });
    let closure_ratio = if maximum_displacement_metres > 1.0e-9 {
        net_displacement_metres / maximum_displacement_metres
    } else {
        0.0
    };
    let endpoint_orientation_error_degrees =
        poses
            .first()
            .zip(poses.last())
            .map_or(0.0, |(first, last)| {
                (2.0 * first
                    .orientation
                    .dot(last.orientation)
                    .abs()
                    .clamp(-1.0, 1.0)
                    .acos())
                .to_degrees()
            });
    let minimum_y = positions
        .iter()
        .map(|value| value.y)
        .fold(f64::INFINITY, f64::min);
    let maximum_y = positions
        .iter()
        .map(|value| value.y)
        .fold(f64::NEG_INFINITY, f64::max);
    let report = RecordingReplayReport {
        recording_id: manifest.recording_id,
        long_axis_field_of_view_degrees: tracker.long_axis_field_of_view_degrees(),
        sensor_delay_milliseconds,
        visual_orientation_delay_milliseconds,
        replayed_frames: positions.len(),
        accepted_motion_samples: tracker.motion_sample_count(),
        keyframes_selected: tracker.visual_keyframe_count(),
        relocalizations: tracker.visual_relocalization_count(),
        limited_frame_fraction: 1.0 - tracking_frames as f64 / positions.len().max(1) as f64,
        path_length_metres,
        net_displacement_metres,
        maximum_displacement_metres,
        closure_ratio,
        endpoint_orientation_error_degrees,
        vertical_range_metres: if minimum_y.is_finite() && maximum_y.is_finite() {
            maximum_y - minimum_y
        } else {
            0.0
        },
        median_visual_matches: matches.get(matches.len() / 2).copied().unwrap_or(0),
        median_visual_inliers: inliers.get(inliers.len() / 2).copied().unwrap_or(0),
    };
    Ok((report, trace))
}

fn tracker_delay_is_valid(milliseconds: f64) -> bool {
    milliseconds.is_finite() && (0.0..=250.0).contains(&milliseconds)
}

fn read_ndjson<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>> {
    let file = File::open(path)
        .with_context(|| format!("failed to open NDJSON input {}", path.display()))?;
    parse_ndjson_lines(BufReader::new(file).lines())
        .with_context(|| format!("failed to read NDJSON input {}", path.display()))
}

fn parse_ndjson_lines<T, I>(lines: I) -> Result<Vec<T>>
where
    T: for<'de> Deserialize<'de>,
    I: IntoIterator<Item = io::Result<String>>,
{
    let mut values = Vec::new();
    for (index, line) in lines.into_iter().enumerate() {
        let line = line.with_context(|| format!("failed to read NDJSON line {}", index + 1))?;
        if line.is_empty() {
            continue;
        }
        values.push(
            serde_json::from_str(&line)
                .with_context(|| format!("failed to parse NDJSON line {}", index + 1))?,
        );
    }
    Ok(values)
}

fn normalized_motion_interval_milliseconds(interval: f64) -> f64 {
    if interval.is_finite() && interval > 0.0 && interval < 1.0 {
        interval * 1_000.0
    } else {
        interval
    }
}

fn delayed_timestamp_milliseconds(timestamp: f64, sensor_delay_milliseconds: f64) -> f64 {
    timestamp + sensor_delay_milliseconds
}

fn delayed_motion_timestamps(
    event_timestamp_milliseconds: f64,
    receipt_timestamp_milliseconds: f64,
    sensor_delay_milliseconds: f64,
) -> (f64, f64) {
    (
        delayed_timestamp_milliseconds(event_timestamp_milliseconds, sensor_delay_milliseconds),
        delayed_timestamp_milliseconds(receipt_timestamp_milliseconds, sensor_delay_milliseconds),
    )
}

fn is_apple_platform(platform: &str) -> bool {
    let normalized = platform.trim().to_ascii_lowercase();
    normalized.contains("iphone")
        || normalized.contains("ipad")
        || normalized.contains("ipod")
        || normalized.starts_with("mac")
}

fn push_recorded_sensor(
    tracker: &mut ArTracker,
    sensor: &RecordedSensorEvent,
    apple_sign: f64,
    sensor_delay_milliseconds: f64,
) -> Result<SensorReplayOutcome> {
    match sensor {
        RecordedSensorEvent::DeviceOrientation {
            alpha_degrees: Some(alpha),
            beta_degrees: Some(beta),
            gamma_degrees: Some(gamma),
            screen_angle_degrees,
            ..
        } => {
            let accepted = tracker.push_device_orientation(
                *alpha,
                *beta,
                *gamma,
                *screen_angle_degrees,
                sensor.replay_timestamp_milliseconds(sensor_delay_milliseconds),
            );
            if !accepted {
                bail!("device-orientation sample was rejected");
            }
            Ok(SensorReplayOutcome::Pushed)
        }
        RecordedSensorEvent::DeviceMotion {
            event_timestamp_milliseconds,
            receipt_timestamp_milliseconds,
            interval,
            acceleration,
            acceleration_including_gravity: force,
            rotation_rate,
            screen_angle_degrees,
        } => {
            let Some((gyro_alpha, gyro_beta, gyro_gamma)) = rotation_rate
                .alpha
                .zip(rotation_rate.beta)
                .zip(rotation_rate.gamma)
                .map(|((alpha, beta), gamma)| (alpha, beta, gamma))
            else {
                return Ok(SensorReplayOutcome::SkippedIncomplete);
            };
            let Some((force_x, force_y, force_z)) = force
                .x
                .zip(force.y)
                .zip(force.z)
                .map(|((x, y), z)| (x, y, z))
            else {
                return Ok(SensorReplayOutcome::SkippedIncomplete);
            };
            let degrees_to_radians = std::f64::consts::PI / 180.0;
            let (gyro_x, gyro_y, gyro_z) = if apple_sign < 0.0 {
                (gyro_alpha, gyro_beta, gyro_gamma)
            } else {
                (gyro_beta, gyro_gamma, gyro_alpha)
            };
            let (delayed_event_timestamp, delayed_receipt_timestamp) = delayed_motion_timestamps(
                *event_timestamp_milliseconds,
                *receipt_timestamp_milliseconds,
                sensor_delay_milliseconds,
            );
            let accepted = tracker.push_motion_sample(
                delayed_event_timestamp,
                delayed_receipt_timestamp,
                normalized_motion_interval_milliseconds(*interval),
                gyro_x * degrees_to_radians,
                gyro_y * degrees_to_radians,
                gyro_z * degrees_to_radians,
                force_x * apple_sign,
                force_y * apple_sign,
                force_z * apple_sign,
                acceleration.x.map_or(f64::NAN, |value| value * apple_sign),
                acceleration.y.map_or(f64::NAN, |value| value * apple_sign),
                acceleration.z.map_or(f64::NAN, |value| value * apple_sign),
                screen_orientation_code(*screen_angle_degrees),
            );
            if !accepted {
                bail!("device-motion sample was rejected");
            }
            Ok(SensorReplayOutcome::Pushed)
        }
        RecordedSensorEvent::DeviceOrientation { .. } => Ok(SensorReplayOutcome::SkippedIncomplete),
    }
}

fn push_recorded_luma_frame(
    tracker: &mut ArTracker,
    frame: &RecordedTrackerFrame,
    camera: RecordingCamera,
    pixels: &[u8],
) -> Result<()> {
    let texture_score = tracker.push_luma_frame(
        frame.frame_id,
        frame.performance_timestamp_milliseconds,
        u32::try_from(camera.tracker_frame_width)?,
        u32::try_from(camera.tracker_frame_height)?,
        pixels,
    );
    if !texture_score.is_finite() || texture_score < 0.0 {
        bail!("tracker rejected luma frame {}", frame.frame_id);
    }
    Ok(())
}

fn screen_orientation_code(angle_degrees: f64) -> u8 {
    match (angle_degrees.rem_euclid(360.0).round() as i32).rem_euclid(360) {
        90 => 1,
        180 => 2,
        270 => 3,
        _ => 0,
    }
}

fn write_json<T: Serialize>(path: Option<&Path>, value: &T) -> Result<()> {
    match path {
        Some(path) => write_json_file(path, value)
            .with_context(|| format!("failed to write report to {}", path.display())),
        None => {
            let stdout = io::stdout();
            let mut writer = BufWriter::new(stdout.lock());
            write_pretty_json(&mut writer, value).context("failed to write report to stdout")
        }
    }
}

fn write_json_file<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    write_pretty_json(&mut writer, value)
}

fn write_ndjson_file<T: Serialize>(path: &Path, values: &[T]) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    for value in values {
        serde_json::to_writer(&mut writer, value)?;
        writeln!(writer)?;
    }
    writer.flush()?;
    Ok(())
}

fn write_pretty_json<T: Serialize>(writer: &mut impl Write, value: &T) -> Result<()> {
    serde_json::to_writer_pretty(&mut *writer, value)?;
    writeln!(writer)?;
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_argument_requires_three_finite_components() {
        assert_eq!(
            "1, 2, 3".parse::<Vector3Argument>().unwrap().0,
            DVec3::new(1.0, 2.0, 3.0)
        );
        assert!("1,2".parse::<Vector3Argument>().is_err());
        assert!("1,2,NaN".parse::<Vector3Argument>().is_err());
    }

    #[test]
    fn ios_fractional_motion_interval_is_normalized_to_milliseconds() {
        assert!(
            (normalized_motion_interval_milliseconds(0.016_666_667) - 16.666_667).abs() < 1.0e-6
        );
        assert_eq!(normalized_motion_interval_milliseconds(16.0), 16.0);
    }

    #[test]
    fn sensor_delay_shifts_event_and_receipt_clocks_together() {
        assert_eq!(
            delayed_motion_timestamps(100.0, 104.0, 25.0),
            (125.0, 129.0)
        );
        assert_eq!(delayed_motion_timestamps(100.0, 104.0, 0.0), (100.0, 104.0));

        let orientation = RecordedSensorEvent::DeviceOrientation {
            event_timestamp_milliseconds: 100.0,
            alpha_degrees: Some(0.0),
            beta_degrees: Some(0.0),
            gamma_degrees: Some(0.0),
            screen_angle_degrees: 0.0,
        };
        assert_eq!(orientation.replay_timestamp_milliseconds(25.0), 125.0);
    }

    #[test]
    fn apple_platform_detection_covers_mobile_and_desktop_identifiers() {
        for platform in [
            "iPhone",
            "iPhone Simulator",
            "iPad",
            "iPod touch",
            "MacIntel",
            " Macintosh ",
        ] {
            assert!(
                is_apple_platform(platform),
                "expected Apple platform: {platform}"
            );
        }
        for platform in ["Linux armv8l", "Android", "Win32"] {
            assert!(
                !is_apple_platform(platform),
                "unexpected Apple platform: {platform}"
            );
        }
    }

    #[test]
    fn ndjson_line_io_errors_are_not_silently_dropped() {
        let lines = vec![
            Ok(r#"{"value":1}"#.to_owned()),
            Err(io::Error::other("injected line read failure")),
        ];
        let error = parse_ndjson_lines::<serde_json::Value, _>(lines).unwrap_err();
        let message = format!("{error:#}");
        assert!(message.contains("failed to read NDJSON line 2"));
        assert!(message.contains("injected line read failure"));
    }

    #[test]
    fn incomplete_sensor_events_are_skipped_but_rejections_fail() {
        let mut tracker = ArTracker::new();
        let incomplete = RecordedSensorEvent::DeviceOrientation {
            event_timestamp_milliseconds: 10.0,
            alpha_degrees: None,
            beta_degrees: Some(0.0),
            gamma_degrees: Some(0.0),
            screen_angle_degrees: 0.0,
        };
        assert_eq!(
            push_recorded_sensor(&mut tracker, &incomplete, -1.0, 0.0).unwrap(),
            SensorReplayOutcome::SkippedIncomplete
        );

        let invalid = RecordedSensorEvent::DeviceMotion {
            event_timestamp_milliseconds: 10.0,
            receipt_timestamp_milliseconds: f64::MAX,
            interval: 16.0,
            acceleration: OptionalVector3 {
                x: Some(0.0),
                y: Some(0.0),
                z: Some(0.0),
            },
            acceleration_including_gravity: OptionalVector3 {
                x: Some(0.0),
                y: Some(0.0),
                z: Some(9.806_65),
            },
            rotation_rate: RotationRate {
                alpha: Some(0.0),
                beta: Some(0.0),
                gamma: Some(0.0),
            },
            screen_angle_degrees: 0.0,
        };
        assert!(push_recorded_sensor(&mut tracker, &invalid, -1.0, 0.0).is_err());
    }

    #[test]
    fn rejected_luma_frames_fail_replay() {
        let mut tracker = ArTracker::new();
        let camera = RecordingCamera {
            tracker_frame_height: 2,
            tracker_frame_width: 2,
        };
        let first = RecordedTrackerFrame {
            frame_id: 1,
            performance_timestamp_milliseconds: 10.0,
            recording_time_milliseconds: 0.0,
        };
        let duplicate_timestamp = RecordedTrackerFrame {
            frame_id: 2,
            performance_timestamp_milliseconds: 10.0,
            recording_time_milliseconds: 1.0,
        };
        let pixels = [0_u8; 4];
        push_recorded_luma_frame(&mut tracker, &first, camera, &pixels).unwrap();
        assert!(
            push_recorded_luma_frame(&mut tracker, &duplicate_timestamp, camera, &pixels).is_err()
        );
    }
}
