//! Command-line interface for offline VIO sensor replay.

#![forbid(unsafe_code)]

use anyhow::{Context, Result, bail};
use ar_tracker_wasm::{ArTracker, TRACKING_STATE_TRACKING};
use clap::{Parser, Subcommand};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, HashSet},
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

        /// Headerless contiguous GRAY8 frames in tracker dimensions.
        #[arg(long)]
        frames: PathBuf,

        /// Override the recorded camera field of view along the longer frame axis.
        #[arg(long)]
        long_axis_fov_degrees: Option<f64>,

        /// Shift sensor events relative to tracker frames for timing calibration.
        #[arg(long, default_value_t = 0.0)]
        sensor_delay_milliseconds: f64,

        /// Camera-to-DeviceOrientation timing offset used for rotation compensation.
        #[arg(long, default_value_t = 0.0)]
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
    long_axis_field_of_view_degrees: Option<f64>,
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

    fn is_complete_for_replay(&self) -> bool {
        match self {
            Self::DeviceMotion {
                event_timestamp_milliseconds,
                receipt_timestamp_milliseconds,
                interval,
                acceleration_including_gravity,
                rotation_rate,
                screen_angle_degrees,
                ..
            } => {
                [
                    *event_timestamp_milliseconds,
                    *receipt_timestamp_milliseconds,
                    *interval,
                    *screen_angle_degrees,
                ]
                .into_iter()
                .all(f64::is_finite)
                    && [
                        acceleration_including_gravity.x,
                        acceleration_including_gravity.y,
                        acceleration_including_gravity.z,
                        rotation_rate.alpha,
                        rotation_rate.beta,
                        rotation_rate.gamma,
                    ]
                    .into_iter()
                    .all(|value| value.is_some_and(f64::is_finite))
            }
            Self::DeviceOrientation {
                event_timestamp_milliseconds,
                alpha_degrees,
                beta_degrees,
                gamma_degrees,
                screen_angle_degrees,
            } => {
                [*event_timestamp_milliseconds, *screen_angle_degrees]
                    .into_iter()
                    .all(f64::is_finite)
                    && [*alpha_degrees, *beta_degrees, *gamma_degrees]
                        .into_iter()
                        .all(|value| value.is_some_and(f64::is_finite))
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RecordedTrackerFrame {
    frame_id: u32,
    performance_timestamp_milliseconds: f64,
    recording_time_milliseconds: f64,
}

#[derive(Clone, Copy, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RecordedBrowserFrameTiming {
    accepted: bool,
    frame_id: u32,
    performance_timestamp_milliseconds: Option<f64>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct RecordedArkitPose {
    frame_id: u32,
    recording_time_milliseconds: f64,
    timestamp_seconds: f64,
    position: [f64; 3],
    #[serde(rename = "quaternionXYZW")]
    quaternion_xyzw: [f64; 4],
    tracking_state: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct RecordingReplayReport {
    recording_id: String,
    long_axis_field_of_view_degrees: f64,
    sensor_delay_milliseconds: f64,
    visual_orientation_delay_milliseconds: f64,
    skipped_leading_frames_before_sensors: usize,
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
    metric_scale_initialized: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    metric_scale_initialization_frame: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arkit_comparison: Option<ArkitComparisonReport>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
struct ArkitComparisonReport {
    matched_normal_frames: usize,
    position_rmse_metres: f64,
    position_median_metres: f64,
    position_p95_metres: f64,
    horizontal_position_rmse_metres: f64,
    vertical_position_rmse_metres: f64,
    orientation_rmse_degrees: f64,
    orientation_median_degrees: f64,
    orientation_p95_degrees: f64,
    endpoint_position_error_metres: f64,
    endpoint_orientation_error_degrees: f64,
    arkit_path_length_metres: f64,
    estimator_path_length_metres: f64,
    path_length_ratio: f64,
    arkit_maximum_displacement_metres: f64,
    estimator_maximum_displacement_metres: f64,
    maximum_displacement_ratio: f64,
    least_squares_scale_ratio: f64,
    frame_delta_rmse_metres: f64,
    one_second_translation_rmse_metres: f64,
    one_second_scale_samples: usize,
    one_second_scale_p10: f64,
    one_second_scale_median: f64,
    one_second_scale_p90: f64,
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

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct ReplayTraceFrame {
    frame_id: u32,
    recording_time_milliseconds: f64,
    position: [f64; 3],
    orientation_xyzw: [f64; 4],
    inertial_velocity_metres_per_second: [f64; 3],
    matches: u32,
    inliers: u32,
    keyframe_id: u32,
    keyframe_count: u64,
    landmark_count: u32,
    converged_landmark_count: u32,
    mean_scene_depth_metres: f64,
    relocalization_count: u64,
    tracking: bool,
    stationary_candidate: bool,
    metric_scale_initialized: bool,
    latest_metric_scale_ratio: f64,
    metric_scale_confidence: f64,
    latest_window_end_cost: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    arkit_position_aligned: Option<[f64; 3]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    arkit_orientation_xyzw_aligned: Option<[f64; 4]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    position_error_metres: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    orientation_error_degrees: Option<f64>,
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
    if !sensor_delay_milliseconds.is_finite() || sensor_delay_milliseconds.abs() > 250.0 {
        bail!("sensor delay must be finite and between -250 and 250 milliseconds");
    }
    if !tracker_delay_is_valid(visual_orientation_delay_milliseconds) {
        bail!("visual orientation delay must be between 0 and 250 milliseconds");
    }
    let manifest: RecordingManifest =
        serde_json::from_reader(BufReader::new(File::open(recording.join("manifest.json"))?))?;
    let sensors = read_ndjson::<RecordedSensorEvent>(&recording.join("sensor-events.ndjson"))?;
    let tracker_frames =
        read_ndjson::<RecordedTrackerFrame>(&recording.join("tracker-frames.ndjson"))?;
    let browser_timing_path = recording.join("wk-frame-timing.ndjson");
    let browser_frame_timing = browser_timing_path
        .exists()
        .then(|| read_ndjson::<RecordedBrowserFrameTiming>(&browser_timing_path))
        .transpose()?;
    let browser_timing_by_frame: HashMap<u32, RecordedBrowserFrameTiming> = browser_frame_timing
        .unwrap_or_default()
        .into_iter()
        .map(|timing| (timing.frame_id, timing))
        .collect();
    let uses_browser_frame_clock = !browser_timing_by_frame.is_empty();
    let arkit_path = recording.join("arkit-poses.ndjson");
    let arkit_poses = arkit_path
        .exists()
        .then(|| read_ndjson::<RecordedArkitPose>(&arkit_path))
        .transpose()?;
    if let Some(arkit_poses) = arkit_poses.as_deref() {
        validate_arkit_pairing(&tracker_frames, arkit_poses)?;
    }
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
    let replay_fov_degrees =
        long_axis_fov_degrees.or(manifest.camera.long_axis_field_of_view_degrees);
    if let Some(degrees) = replay_fov_degrees
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
    let mut replayed_tracker_frames = Vec::with_capacity(tracker_frames.len());
    let sensor_ready_timestamp = first_sensor_ready_timestamp(&sensors, sensor_delay_milliseconds);
    let mut skipped_leading_frames = 0_usize;

    for native_frame in &tracker_frames {
        frame_reader.read_exact(&mut pixels).with_context(|| {
            format!(
                "raw frame file ended before tracker frame {}",
                native_frame.frame_id
            )
        })?;
        let mut frame = *native_frame;
        if uses_browser_frame_clock {
            let Some(timing) = browser_timing_by_frame.get(&frame.frame_id) else {
                skipped_leading_frames += 1;
                continue;
            };
            let Some(timestamp) = timing
                .accepted
                .then_some(timing.performance_timestamp_milliseconds)
                .flatten()
                .filter(|timestamp| timestamp.is_finite())
            else {
                continue;
            };
            frame.performance_timestamp_milliseconds = timestamp;
        }
        if sensor_ready_timestamp
            .is_some_and(|ready| frame.performance_timestamp_milliseconds < ready)
        {
            skipped_leading_frames += 1;
            continue;
        }
        while let Some(sensor) = sensors.get(sensor_index)
            && sensor.replay_timestamp_milliseconds(sensor_delay_milliseconds)
                <= frame.performance_timestamp_milliseconds
        {
            push_recorded_sensor(&mut tracker, sensor, apple_sign, sensor_delay_milliseconds)
                .with_context(|| format!("tracker rejected sensor event {sensor_index}"))?;
            sensor_index += 1;
        }
        push_recorded_luma_frame(&mut tracker, &frame, manifest.camera, &pixels)?;
        replayed_tracker_frames.push(frame);
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
        let map_stats = tracker.map_stats();
        trace.push(ReplayTraceFrame {
            frame_id: frame.frame_id,
            recording_time_milliseconds: frame.recording_time_milliseconds,
            position: replayed_pose.position.to_array(),
            orientation_xyzw: replayed_pose.orientation.to_array(),
            inertial_velocity_metres_per_second: [
                inertial_velocity[0],
                inertial_velocity[1],
                inertial_velocity[2],
            ],
            matches: tracker.visual_match_count(),
            inliers: tracker.visual_inlier_count(),
            keyframe_id: tracker.latest_visual_keyframe_id(),
            keyframe_count: tracker.visual_keyframe_count(),
            landmark_count: tracker.landmark_count(),
            converged_landmark_count: map_stats.get(2).copied().unwrap_or(0.0) as u32,
            mean_scene_depth_metres: map_stats.get(3).copied().unwrap_or(0.0),
            relocalization_count: tracker.visual_relocalization_count(),
            tracking: tracker.tracking_state() == TRACKING_STATE_TRACKING,
            stationary_candidate: tracker.inertial_stationary_candidate(),
            metric_scale_initialized: tracker.metric_scale_initialized(),
            latest_metric_scale_ratio: tracker.latest_metric_scale_ratio(),
            metric_scale_confidence: tracker.metric_scale_confidence(),
            latest_window_end_cost: tracker.latest_window_end_cost(),
            arkit_position_aligned: None,
            arkit_orientation_xyzw_aligned: None,
            position_error_metres: None,
            orientation_error_degrees: None,
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
    let arkit_comparison = arkit_poses
        .as_deref()
        .and_then(|arkit| compare_with_arkit(&replayed_tracker_frames, &poses, arkit, &mut trace));
    let report = RecordingReplayReport {
        recording_id: manifest.recording_id,
        long_axis_field_of_view_degrees: tracker.long_axis_field_of_view_degrees(),
        sensor_delay_milliseconds,
        visual_orientation_delay_milliseconds,
        skipped_leading_frames_before_sensors: skipped_leading_frames,
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
        metric_scale_initialized: tracker.metric_scale_initialized(),
        metric_scale_initialization_frame: trace
            .iter()
            .find(|frame| frame.metric_scale_initialized)
            .map(|frame| frame.frame_id),
        arkit_comparison,
    };
    Ok((report, trace))
}

#[derive(Clone, Copy)]
struct AlignedPosePair {
    recording_time_milliseconds: f64,
    estimator: ReplayedPose,
    arkit: ReplayedPose,
}

fn compare_with_arkit(
    tracker_frames: &[RecordedTrackerFrame],
    replayed_poses: &[ReplayedPose],
    arkit_poses: &[RecordedArkitPose],
    trace: &mut [ReplayTraceFrame],
) -> Option<ArkitComparisonReport> {
    let arkit_by_frame: HashMap<u32, &RecordedArkitPose> = arkit_poses
        .iter()
        .filter(|pose| pose.tracking_state.eq_ignore_ascii_case("normal"))
        .map(|pose| (pose.frame_id, pose))
        .collect();
    let origin = tracker_frames
        .iter()
        .zip(replayed_poses)
        .find_map(|(frame, replayed)| {
            let arkit = arkit_by_frame.get(&frame.frame_id)?;
            let arkit = decoded_arkit_pose(arkit)?;
            Some((*replayed, arkit))
        })?;
    // Native recordings can contain one or two camera frames before the first
    // motion/orientation sample. Use the first non-negative-time pair for the
    // world rotation, while preserving the actual first camera center as the
    // translation origin.
    let orientation_anchor = tracker_frames
        .iter()
        .zip(replayed_poses)
        .find_map(|(frame, replayed)| {
            if frame.recording_time_milliseconds < 0.0 {
                return None;
            }
            let arkit = arkit_by_frame.get(&frame.frame_id)?;
            let arkit = decoded_arkit_pose(arkit)?;
            Some((*replayed, arkit))
        })
        .unwrap_or(origin);
    let world_alignment = (orientation_anchor.0.orientation
        * orientation_anchor.1.orientation.conjugate())
    .normalize();
    let aligned_origin = origin.0.position;
    let arkit_origin = origin.1.position;

    let mut pairs = Vec::new();
    for ((frame, replayed), trace_frame) in tracker_frames
        .iter()
        .zip(replayed_poses)
        .zip(trace.iter_mut())
    {
        let Some(arkit) = arkit_by_frame
            .get(&frame.frame_id)
            .and_then(|pose| decoded_arkit_pose(pose))
        else {
            continue;
        };
        let aligned = ReplayedPose {
            position: aligned_origin + world_alignment * (arkit.position - arkit_origin),
            orientation: (world_alignment * arkit.orientation).normalize(),
        };
        let position_error = replayed.position.distance(aligned.position);
        let orientation_error = quaternion_error_degrees(replayed.orientation, aligned.orientation);
        trace_frame.arkit_position_aligned = Some(aligned.position.to_array());
        trace_frame.arkit_orientation_xyzw_aligned = Some(aligned.orientation.to_array());
        trace_frame.position_error_metres = Some(position_error);
        trace_frame.orientation_error_degrees = Some(orientation_error);
        pairs.push(AlignedPosePair {
            recording_time_milliseconds: frame.recording_time_milliseconds,
            estimator: *replayed,
            arkit: aligned,
        });
    }
    if pairs.len() < 2 {
        return None;
    }

    let position_errors: Vec<f64> = pairs
        .iter()
        .map(|pair| pair.estimator.position.distance(pair.arkit.position))
        .collect();
    let horizontal_errors: Vec<f64> = pairs
        .iter()
        .map(|pair| {
            let error = pair.estimator.position - pair.arkit.position;
            error.x.hypot(error.z)
        })
        .collect();
    let vertical_errors: Vec<f64> = pairs
        .iter()
        .map(|pair| (pair.estimator.position.y - pair.arkit.position.y).abs())
        .collect();
    let orientation_errors: Vec<f64> = pairs
        .iter()
        .map(|pair| quaternion_error_degrees(pair.estimator.orientation, pair.arkit.orientation))
        .collect();

    let estimator_path_length_metres: f64 = pairs
        .windows(2)
        .map(|pair| {
            pair[0]
                .estimator
                .position
                .distance(pair[1].estimator.position)
        })
        .sum();
    let arkit_path_length_metres: f64 = pairs
        .windows(2)
        .map(|pair| pair[0].arkit.position.distance(pair[1].arkit.position))
        .sum();
    let frame_delta_errors: Vec<f64> = pairs
        .windows(2)
        .map(|pair| {
            let estimator_delta = pair[1].estimator.position - pair[0].estimator.position;
            let arkit_delta = pair[1].arkit.position - pair[0].arkit.position;
            (estimator_delta - arkit_delta).length()
        })
        .collect();

    let first_pair = pairs[0];
    let estimator_maximum_displacement_metres = pairs
        .iter()
        .map(|pair| {
            pair.estimator
                .position
                .distance(first_pair.estimator.position)
        })
        .fold(0.0, f64::max);
    let arkit_maximum_displacement_metres = pairs
        .iter()
        .map(|pair| pair.arkit.position.distance(first_pair.arkit.position))
        .fold(0.0, f64::max);
    let mut scale_numerator = 0.0;
    let mut scale_denominator = 0.0;
    for pair in pairs.iter().skip(1) {
        let estimator_delta = pair.estimator.position - first_pair.estimator.position;
        let arkit_delta = pair.arkit.position - first_pair.arkit.position;
        scale_numerator += estimator_delta.dot(arkit_delta);
        scale_denominator += arkit_delta.length_squared();
    }
    let least_squares_scale_ratio = finite_ratio(scale_numerator, scale_denominator);

    let mut one_second_translation_errors = Vec::new();
    let mut one_second_scale_ratios = Vec::new();
    for (index, pair) in pairs.iter().enumerate() {
        let target_time = pair.recording_time_milliseconds + 1_000.0;
        let Some(later) = pairs[index + 1..]
            .iter()
            .find(|candidate| candidate.recording_time_milliseconds >= target_time)
        else {
            continue;
        };
        if later.recording_time_milliseconds - target_time > 75.0 {
            continue;
        }
        let estimator_delta = later.estimator.position - pair.estimator.position;
        let arkit_delta = later.arkit.position - pair.arkit.position;
        one_second_translation_errors.push((estimator_delta - arkit_delta).length());
        let arkit_distance = arkit_delta.length();
        if arkit_distance >= 0.05 {
            one_second_scale_ratios.push(estimator_delta.length() / arkit_distance);
        }
    }
    one_second_scale_ratios.sort_by(f64::total_cmp);

    let last_pair = *pairs.last()?;
    Some(ArkitComparisonReport {
        matched_normal_frames: pairs.len(),
        position_rmse_metres: root_mean_square(&position_errors),
        position_median_metres: quantile(&position_errors, 0.5),
        position_p95_metres: quantile(&position_errors, 0.95),
        horizontal_position_rmse_metres: root_mean_square(&horizontal_errors),
        vertical_position_rmse_metres: root_mean_square(&vertical_errors),
        orientation_rmse_degrees: root_mean_square(&orientation_errors),
        orientation_median_degrees: quantile(&orientation_errors, 0.5),
        orientation_p95_degrees: quantile(&orientation_errors, 0.95),
        endpoint_position_error_metres: last_pair
            .estimator
            .position
            .distance(last_pair.arkit.position),
        endpoint_orientation_error_degrees: quaternion_error_degrees(
            last_pair.estimator.orientation,
            last_pair.arkit.orientation,
        ),
        arkit_path_length_metres,
        estimator_path_length_metres,
        path_length_ratio: finite_ratio(estimator_path_length_metres, arkit_path_length_metres),
        arkit_maximum_displacement_metres,
        estimator_maximum_displacement_metres,
        maximum_displacement_ratio: finite_ratio(
            estimator_maximum_displacement_metres,
            arkit_maximum_displacement_metres,
        ),
        least_squares_scale_ratio,
        frame_delta_rmse_metres: root_mean_square(&frame_delta_errors),
        one_second_translation_rmse_metres: root_mean_square(&one_second_translation_errors),
        one_second_scale_samples: one_second_scale_ratios.len(),
        one_second_scale_p10: sorted_quantile(&one_second_scale_ratios, 0.1),
        one_second_scale_median: sorted_quantile(&one_second_scale_ratios, 0.5),
        one_second_scale_p90: sorted_quantile(&one_second_scale_ratios, 0.9),
    })
}

fn validate_arkit_pairing(
    tracker_frames: &[RecordedTrackerFrame],
    arkit_poses: &[RecordedArkitPose],
) -> Result<()> {
    if tracker_frames.len() != arkit_poses.len() {
        bail!(
            "tracker/ARKit frame count mismatch: {} tracker frames, {} ARKit poses",
            tracker_frames.len(),
            arkit_poses.len()
        );
    }
    let mut tracker_ids = HashSet::with_capacity(tracker_frames.len());
    for frame in tracker_frames {
        if !tracker_ids.insert(frame.frame_id) {
            bail!("duplicate tracker frame id {}", frame.frame_id);
        }
    }
    let mut arkit_by_id = HashMap::with_capacity(arkit_poses.len());
    for pose in arkit_poses {
        if arkit_by_id.insert(pose.frame_id, pose).is_some() {
            bail!("duplicate ARKit frame id {}", pose.frame_id);
        }
    }
    for window in tracker_frames.windows(2) {
        if window[1].frame_id <= window[0].frame_id
            || window[1].performance_timestamp_milliseconds
                <= window[0].performance_timestamp_milliseconds
            || window[1].recording_time_milliseconds <= window[0].recording_time_milliseconds
        {
            bail!("tracker frame order is not strictly monotonic");
        }
    }
    for window in arkit_poses.windows(2) {
        if window[1].frame_id <= window[0].frame_id
            || window[1].timestamp_seconds <= window[0].timestamp_seconds
            || window[1].recording_time_milliseconds <= window[0].recording_time_milliseconds
        {
            bail!("ARKit pose order is not strictly monotonic");
        }
    }
    const TIMESTAMP_TOLERANCE_MILLISECONDS: f64 = 0.1;
    for frame in tracker_frames {
        let Some(pose) = arkit_by_id.get(&frame.frame_id) else {
            bail!("tracker frame {} has no ARKit pose", frame.frame_id);
        };
        let absolute_difference =
            (frame.performance_timestamp_milliseconds - pose.timestamp_seconds * 1_000.0).abs();
        let recording_difference =
            (frame.recording_time_milliseconds - pose.recording_time_milliseconds).abs();
        if !absolute_difference.is_finite()
            || !recording_difference.is_finite()
            || absolute_difference > TIMESTAMP_TOLERANCE_MILLISECONDS
            || recording_difference > TIMESTAMP_TOLERANCE_MILLISECONDS
        {
            bail!(
                "tracker/ARKit timestamp mismatch at frame {}: absolute {:.6} ms, recording {:.6} ms",
                frame.frame_id,
                absolute_difference,
                recording_difference
            );
        }
    }
    Ok(())
}

fn decoded_arkit_pose(pose: &RecordedArkitPose) -> Option<ReplayedPose> {
    let position = DVec3::from_array(pose.position);
    let raw_orientation = DQuat::from_xyzw(
        pose.quaternion_xyzw[0],
        pose.quaternion_xyzw[1],
        pose.quaternion_xyzw[2],
        pose.quaternion_xyzw[3],
    );
    // ARCamera.transform carries the camera axes of ARKit's landscape sensor
    // image. The recorded tracker luma is rotated 90 degrees clockwise into
    // portrait, so compare the estimator with the corresponding portrait
    // camera frame (a +90 degree local-Z extrinsic).
    let portrait_orientation =
        raw_orientation * DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2);
    (position.is_finite()
        && portrait_orientation.is_finite()
        && portrait_orientation.length_squared() > f64::EPSILON)
        .then_some(ReplayedPose {
            position,
            orientation: portrait_orientation.normalize(),
        })
}

fn quaternion_error_degrees(left: DQuat, right: DQuat) -> f64 {
    (2.0 * left.dot(right).abs().clamp(-1.0, 1.0).acos()).to_degrees()
}

fn root_mean_square(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    (values.iter().map(|value| value * value).sum::<f64>() / values.len() as f64).sqrt()
}

fn quantile(values: &[f64], fraction: f64) -> f64 {
    let mut sorted: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    sorted.sort_by(f64::total_cmp);
    sorted_quantile(&sorted, fraction)
}

fn sorted_quantile(sorted: &[f64], fraction: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f64 * fraction.clamp(0.0, 1.0)).round() as usize;
    sorted[index]
}

fn finite_ratio(numerator: f64, denominator: f64) -> f64 {
    if numerator.is_finite() && denominator.is_finite() && denominator.abs() > 1.0e-12 {
        numerator / denominator
    } else {
        0.0
    }
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

fn first_sensor_ready_timestamp(
    sensors: &[RecordedSensorEvent],
    sensor_delay_milliseconds: f64,
) -> Option<f64> {
    let first_motion = sensors.iter().find_map(|sensor| match sensor {
        RecordedSensorEvent::DeviceMotion { .. } if sensor.is_complete_for_replay() => {
            Some(sensor.replay_timestamp_milliseconds(sensor_delay_milliseconds))
        }
        _ => None,
    });
    let first_orientation = sensors.iter().find_map(|sensor| match sensor {
        RecordedSensorEvent::DeviceOrientation { .. } if sensor.is_complete_for_replay() => {
            Some(sensor.replay_timestamp_milliseconds(sensor_delay_milliseconds))
        }
        _ => None,
    });
    first_motion
        .zip(first_orientation)
        .map(|(motion, orientation)| motion.max(orientation))
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
        assert_eq!(orientation.replay_timestamp_milliseconds(-10.0), 90.0);
    }

    #[test]
    fn replay_waits_until_both_motion_and_orientation_are_available() {
        let sensors = vec![
            RecordedSensorEvent::DeviceOrientation {
                event_timestamp_milliseconds: 90.0,
                alpha_degrees: None,
                beta_degrees: Some(0.0),
                gamma_degrees: Some(0.0),
                screen_angle_degrees: 0.0,
            },
            RecordedSensorEvent::DeviceMotion {
                event_timestamp_milliseconds: 100.0,
                receipt_timestamp_milliseconds: 100.0,
                interval: 20.0,
                acceleration: OptionalVector3::default(),
                acceleration_including_gravity: OptionalVector3 {
                    x: Some(0.0),
                    y: Some(9.806_65),
                    z: Some(0.0),
                },
                rotation_rate: RotationRate {
                    alpha: Some(0.0),
                    beta: Some(0.0),
                    gamma: Some(0.0),
                },
                screen_angle_degrees: 0.0,
            },
            RecordedSensorEvent::DeviceOrientation {
                event_timestamp_milliseconds: 102.0,
                alpha_degrees: Some(0.0),
                beta_degrees: Some(0.0),
                gamma_degrees: Some(0.0),
                screen_angle_degrees: 0.0,
            },
        ];
        assert_eq!(first_sensor_ready_timestamp(&sensors, 0.0), Some(102.0));
        assert_eq!(first_sensor_ready_timestamp(&sensors, -5.0), Some(97.0));
    }

    #[test]
    fn arkit_pose_contract_preserves_capitalized_xyzw_and_portrait_extrinsic() {
        let pose: RecordedArkitPose = serde_json::from_str(
            r#"{
                "frameId": 7,
                "recordingTimeMilliseconds": 123.0,
                "timestampSeconds": 456.0,
                "position": [1.0, 2.0, 3.0],
                "quaternionXYZW": [0.0, 0.0, 0.0, 1.0],
                "trackingState": "normal"
            }"#,
        )
        .unwrap();
        let decoded = decoded_arkit_pose(&pose).unwrap();
        assert_eq!(decoded.position, DVec3::new(1.0, 2.0, 3.0));
        let expected = DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2);
        assert!(decoded.orientation.abs_diff_eq(expected, 1.0e-12));
    }

    #[test]
    fn literal_wk_sensor_and_frame_timing_contracts_parse() {
        let sensor: RecordedSensorEvent = serde_json::from_str(
            r#"{
                "kind": "device_motion",
                "captureSource": "wkwebview",
                "eventTimestampMilliseconds": 120.0,
                "receiptTimestampMilliseconds": 123.0,
                "performanceTimeOriginMilliseconds": 1000000.0,
                "nativeMessageReceiptUptimeMilliseconds": 900000.0,
                "intervalMilliseconds": 20.0,
                "reportedInterval": 0.02,
                "acceleration": {"x": 0.1, "y": 0.2, "z": 0.3},
                "accelerationIncludingGravity": {"x": 0.1, "y": 9.9, "z": 0.3},
                "rotationRateDegreesPerSecond": {
                    "alpha": 1.0, "beta": 2.0, "gamma": 3.0
                },
                "screenAngleDegrees": 0.0,
                "screenOrientation": "portrait-primary",
                "isTrusted": true,
                "sequence": 4
            }"#,
        )
        .unwrap();
        assert!(sensor.is_complete_for_replay());
        assert_eq!(sensor.timestamp_milliseconds(), 120.0);

        let timing: RecordedBrowserFrameTiming = serde_json::from_str(
            r#"{
                "kind": "native_frame_timing",
                "accepted": true,
                "frameId": 42,
                "nativeTimestampMilliseconds": 900100.0,
                "receiptTimestampMilliseconds": 220.0,
                "performanceTimestampMilliseconds": 200.0
            }"#,
        )
        .unwrap();
        assert!(timing.accepted);
        assert_eq!(timing.frame_id, 42);
        assert_eq!(timing.performance_timestamp_milliseconds, Some(200.0));
    }

    fn tracker_frame(frame_id: u32, recording_time_milliseconds: f64) -> RecordedTrackerFrame {
        RecordedTrackerFrame {
            frame_id,
            performance_timestamp_milliseconds: 1_000.0 + recording_time_milliseconds,
            recording_time_milliseconds,
        }
    }

    fn arkit_pose(
        frame_id: u32,
        recording_time_milliseconds: f64,
        position: DVec3,
    ) -> RecordedArkitPose {
        RecordedArkitPose {
            frame_id,
            recording_time_milliseconds,
            timestamp_seconds: (1_000.0 + recording_time_milliseconds) / 1_000.0,
            position: position.to_array(),
            quaternion_xyzw: DQuat::IDENTITY.to_array(),
            tracking_state: "normal".to_owned(),
        }
    }

    #[test]
    fn arkit_pairing_rejects_duplicate_ids_and_timestamp_mismatches() {
        let frames = [tracker_frame(1, 0.0), tracker_frame(2, 1_000.0)];
        let valid = [
            arkit_pose(1, 0.0, DVec3::ZERO),
            arkit_pose(2, 1_000.0, DVec3::X),
        ];
        validate_arkit_pairing(&frames, &valid).unwrap();

        let duplicate = [
            arkit_pose(1, 0.0, DVec3::ZERO),
            arkit_pose(1, 1_000.0, DVec3::X),
        ];
        assert!(validate_arkit_pairing(&frames, &duplicate).is_err());

        let mut mismatched = valid;
        mismatched[1].timestamp_seconds += 0.01;
        assert!(validate_arkit_pairing(&frames, &mismatched).is_err());
    }

    #[test]
    fn arkit_comparison_preserves_a_known_scale_error() {
        let frames = [
            tracker_frame(1, 0.0),
            tracker_frame(2, 1_000.0),
            tracker_frame(3, 2_000.0),
        ];
        let arkit = [
            arkit_pose(1, 0.0, DVec3::ZERO),
            arkit_pose(2, 1_000.0, DVec3::X),
            arkit_pose(3, 2_000.0, DVec3::X * 2.0),
        ];
        let portrait = DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2);
        let estimator = [
            ReplayedPose {
                position: DVec3::ZERO,
                orientation: portrait,
            },
            ReplayedPose {
                position: DVec3::X * 2.0,
                orientation: portrait,
            },
            ReplayedPose {
                position: DVec3::X * 4.0,
                orientation: portrait,
            },
        ];
        let mut trace = vec![ReplayTraceFrame::default(); frames.len()];
        let report = compare_with_arkit(&frames, &estimator, &arkit, &mut trace).unwrap();

        assert!((report.least_squares_scale_ratio - 2.0).abs() < 1.0e-12);
        assert!(report.position_rmse_metres > 1.0);
        assert!((report.frame_delta_rmse_metres - 1.0).abs() < 1.0e-12);
        assert!((report.one_second_scale_median - 2.0).abs() < 1.0e-12);
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
            long_axis_field_of_view_degrees: None,
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
