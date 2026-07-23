//! Browser-facing WebAssembly boundary for the owned AR tracker.
//!
//! JavaScript owns browser APIs and normalizes their values at the acquisition
//! boundary. This crate owns the evolving tracking state and returns a
//! renderer-independent camera pose.
//!
//! Architecture (visual-primary, IMU-hinted):
//! - Orientation comes from iOS/W3C device-orientation fusion, yaw-recentered;
//!   it is never optimized visually.
//! - A continuous Lucas-Kanade front-end ([`frontend`]) advances long-lived
//!   pixel tracks every frame, seeded by IMU-rotation-compensated predictions.
//! - A keyframe map ([`map`]) anchors tracks as inverse-depth landmarks — no
//!   global scene-depth assumption; every landmark owns its depth.
//! - An `arael`-based estimator ([`estimator`]) refines the camera position
//!   every frame against the landmark map and runs sliding-window
//!   visual-inertial bundle adjustment at keyframe rate, where preintegrated
//!   accelerometer factors make the monocular reconstruction metric (scale
//!   starts at a prior and converges silently as the phone translates).
//! - Appearance relocalization ([`reloc`]) cancels drift when a previously
//!   mapped view is revisited.

#![forbid(unsafe_code)]

mod estimator;
mod frontend;
mod geometry;
mod map;
mod reloc;
mod scale;

use estimator::{FrameObservation, solve_frame_pose, solve_window};
use frontend::{FrontEnd, TrackState};
use geometry::Intrinsics;
use glam::EulerRot;
use image::GrayImage;
use map::{Keyframe, Map, Preintegration};
use std::{
    collections::VecDeque,
    f64::consts::{FRAC_PI_2, PI},
};
use vio_core::{
    DQuat, DVec3, FrameInterval, MonotonicDuration, MonotonicTimestamp, NormalizedMotionSample,
    PushOutcome, SensorBuffer, SensorSequence, SensorTimeBasis, SourceScreenOrientation,
};
use wasm_bindgen::prelude::*;

const CAMERA_HEIGHT_METRES: f64 = 1.6;
const ESTIMATED_LONG_AXIS_FIELD_OF_VIEW_DEGREES: f64 = 68.0;
const DEFAULT_FEATURE_BUDGET: usize = 130;
const MIN_VISUAL_INLIERS: usize = 8;
const SENSOR_BUFFER_CAPACITY: usize = 512;
const ORIENTATION_HISTORY_CAPACITY: usize = 256;
const DEFAULT_VISUAL_ORIENTATION_DELAY_MILLISECONDS: f64 = 40.0;
const INITIAL_ACCELEROMETER_BIAS_SAMPLES: u64 = 45;
const MAX_INERTIAL_SPEED_METRES_PER_SECOND: f64 = 2.0;
const INERTIAL_VELOCITY_DAMPING_PER_SECOND: f64 = 0.35;
const INERTIAL_VERTICAL_ACCELERATION_GAIN: f64 = 0.35;
const INERTIAL_VERTICAL_DAMPING_PER_SECOND: f64 = 1.0;
const VISUAL_VELOCITY_CORRECTION_GAIN: f64 = 0.25;
const STATIONARY_TRANSLATION_DEADBAND_METRES: f64 = 0.02;
const GRAVITY_METRES_PER_SECOND_SQUARED: f64 = 9.806_65;

/// Keyframe creation policy.
const KEYFRAME_MIN_FRAME_GAP: u64 = 3;
/// Median pixel flow (fraction of frame width) since the last keyframe that
/// triggers a new one.
const KEYFRAME_FLOW_FRACTION: f64 = 0.06;
/// Anchored-inlier starvation threshold that also triggers a keyframe.
const KEYFRAME_MIN_ANCHORED: usize = 24;
const MIN_TRACK_AGE_FOR_LANDMARK: u32 = 2;
const LANDMARK_OUTLIER_STREAK_LIMIT: u32 = 3;
const UNCONVERGED_OUTLIER_STREAK_LIMIT: u32 = 8;

/// Relocalization pacing. Appearance relocalization is a drift-recovery tool,
/// not continuous loop closure: it only runs during visual outages, only
/// against keyframes well in the past, and with a cooldown — each accepted
/// match snaps the position, and doing that against recent keyframes while
/// tracking is healthy injects jumps instead of removing drift.
const RELOCALIZATION_INTERVAL_FRAMES: u64 = 10;
const RELOCALIZATION_COOLDOWN_FRAMES: u32 = 45;
/// A keyframe must be at least this many keyframes older than the newest to be
/// a relocalization candidate.
const RELOCALIZATION_MIN_KEYFRAME_AGE: u32 = 12;
/// Stored keyframe luma is downsampled by this factor for relocalization.
const RELOC_LUMA_DOWNSAMPLE: usize = 2;

/// Numeric value returned for a tracker that does not yet have enough observations.
pub const TRACKING_STATE_INITIALIZING: u8 = 0;
/// Numeric value returned when orientation is available but metric 6DoF is not yet solved.
pub const TRACKING_STATE_LIMITED: u8 = 1;
/// Numeric value reserved for a visual-inertial 6DoF solution.
pub const TRACKING_STATE_TRACKING: u8 = 2;

#[derive(Clone, Copy)]
struct TimedOrientation {
    timestamp_milliseconds: f64,
    absolute: DQuat,
}

/// Browser-facing state for the owned visual-inertial tracker.
#[wasm_bindgen]
pub struct ArTracker {
    // Orientation and inertial pipeline.
    orientation_reference: Option<DQuat>,
    absolute_orientation: Option<DQuat>,
    camera_orientation: DQuat,
    camera_position: DVec3,
    sensor_buffer: SensorBuffer,
    next_sensor_sequence: u64,
    regularized_motion_timestamp_milliseconds: Option<f64>,
    orientation_samples: u64,
    orientation_history: VecDeque<TimedOrientation>,
    motion_samples: u64,
    late_motion_samples: u64,
    latest_linear_acceleration_mps2: f64,
    has_linear_acceleration: bool,
    linear_acceleration_bias_device_mps2: DVec3,
    inertial_velocity_world_mps: DVec3,
    position_before_inertial_prediction: DVec3,
    latest_frame_delta_seconds: f64,
    inertial_stationary_candidate: bool,
    consecutive_stationary_frames: u32,

    // Frames.
    frame_count: u64,
    latest_frame_id: u32,
    latest_frame_timestamp: Option<MonotonicTimestamp>,
    latest_texture_score: f64,
    previous_frame_orientation: Option<DQuat>,

    // Visual pipeline.
    frontend: FrontEnd,
    map: Map,
    preintegration: Preintegration,
    preintegration_valid: bool,
    last_keyframe_frame_count: u64,
    track_anchor_pixels: Vec<(u64, (f32, f32))>,
    visual_matches: u32,
    visual_inliers: u32,
    frames_since_visual_update: u32,
    visual_relocalization_count: u64,
    last_relocalization_frame_id: Option<u32>,
    pending_appearance_frame_id: Option<u32>,
    pending_appearance_position: DVec3,
    pending_appearance_confirmations: u8,
    relocalization_enabled: bool,
    latest_window_end_cost: f64,
    scale_initialized: bool,
    latest_scale_ratio: f64,
    scale_confidence: f64,

    // Calibration.
    long_axis_field_of_view_degrees: f64,
    visual_orientation_delay_milliseconds: f64,
}

#[wasm_bindgen]
impl ArTracker {
    #[wasm_bindgen(constructor)]
    #[allow(clippy::new_without_default)]
    /// Creates a tracker with default calibration and an empty map.
    pub fn new() -> Self {
        #[cfg(target_arch = "wasm32")]
        console_error_panic_hook::set_once();
        Self {
            orientation_reference: None,
            absolute_orientation: None,
            camera_orientation: DQuat::IDENTITY,
            camera_position: DVec3::new(0.0, CAMERA_HEIGHT_METRES, 0.0),
            sensor_buffer: SensorBuffer::new(SENSOR_BUFFER_CAPACITY, SensorTimeBasis::Event)
                .expect("the fixed sensor buffer capacity is non-zero"),
            next_sensor_sequence: 0,
            regularized_motion_timestamp_milliseconds: None,
            orientation_samples: 0,
            orientation_history: VecDeque::new(),
            motion_samples: 0,
            late_motion_samples: 0,
            latest_linear_acceleration_mps2: 0.0,
            has_linear_acceleration: false,
            linear_acceleration_bias_device_mps2: DVec3::ZERO,
            inertial_velocity_world_mps: DVec3::ZERO,
            position_before_inertial_prediction: DVec3::new(0.0, CAMERA_HEIGHT_METRES, 0.0),
            latest_frame_delta_seconds: 0.0,
            inertial_stationary_candidate: false,
            consecutive_stationary_frames: 0,
            frame_count: 0,
            latest_frame_id: 0,
            latest_frame_timestamp: None,
            latest_texture_score: 0.0,
            previous_frame_orientation: None,
            frontend: FrontEnd::new(DEFAULT_FEATURE_BUDGET),
            map: Map::new(),
            preintegration: Preintegration::default(),
            preintegration_valid: false,
            last_keyframe_frame_count: 0,
            track_anchor_pixels: Vec::new(),
            visual_matches: 0,
            visual_inliers: 0,
            frames_since_visual_update: u32::MAX,
            visual_relocalization_count: 0,
            last_relocalization_frame_id: None,
            pending_appearance_frame_id: None,
            pending_appearance_position: DVec3::ZERO,
            pending_appearance_confirmations: 0,
            relocalization_enabled: true,
            latest_window_end_cost: 0.0,
            scale_initialized: false,
            latest_scale_ratio: 1.0,
            scale_confidence: 0.0,
            long_axis_field_of_view_degrees: ESTIMATED_LONG_AXIS_FIELD_OF_VIEW_DEGREES,
            visual_orientation_delay_milliseconds: DEFAULT_VISUAL_ORIENTATION_DELAY_MILLISECONDS,
        }
    }

    /// Adds an absolute W3C Device Orientation observation in degrees.
    pub fn push_device_orientation(
        &mut self,
        alpha_degrees: f64,
        beta_degrees: f64,
        gamma_degrees: f64,
        screen_angle_degrees: f64,
        timestamp_milliseconds: f64,
    ) -> bool {
        if [
            alpha_degrees,
            beta_degrees,
            gamma_degrees,
            screen_angle_degrees,
            timestamp_milliseconds,
        ]
        .iter()
        .any(|value| !value.is_finite())
        {
            return false;
        }

        let absolute = device_orientation_quaternion(
            alpha_degrees,
            beta_degrees,
            gamma_degrees,
            screen_angle_degrees,
        );
        self.absolute_orientation = Some(absolute);
        self.orientation_history.push_back(TimedOrientation {
            timestamp_milliseconds,
            absolute,
        });
        if self.orientation_history.len() > ORIENTATION_HISTORY_CAPACITY {
            self.orientation_history.pop_front();
        }
        self.orientation_reference
            .get_or_insert_with(|| yaw_reference(absolute));
        self.update_camera_orientation();
        self.orientation_samples += 1;
        true
    }

    /// Adds one normalized motion sample using SI units in the portrait-primary device frame.
    #[allow(clippy::too_many_arguments)]
    pub fn push_motion_sample(
        &mut self,
        event_timestamp_milliseconds: f64,
        receipt_timestamp_milliseconds: f64,
        interval_milliseconds: f64,
        gyro_x_radians_per_second: f64,
        gyro_y_radians_per_second: f64,
        gyro_z_radians_per_second: f64,
        specific_force_x_metres_per_second_squared: f64,
        specific_force_y_metres_per_second_squared: f64,
        specific_force_z_metres_per_second_squared: f64,
        linear_acceleration_x_metres_per_second_squared: f64,
        linear_acceleration_y_metres_per_second_squared: f64,
        linear_acceleration_z_metres_per_second_squared: f64,
        screen_orientation_code: u8,
    ) -> bool {
        let regularized_timestamp_milliseconds = self
            .next_regularized_motion_timestamp(event_timestamp_milliseconds, interval_milliseconds);
        let Ok(event_timestamp) =
            MonotonicTimestamp::try_from_millis_f64(regularized_timestamp_milliseconds)
        else {
            return false;
        };
        let Ok(receipt_timestamp) =
            MonotonicTimestamp::try_from_millis_f64(receipt_timestamp_milliseconds)
        else {
            return false;
        };
        let reported_interval = if interval_milliseconds.is_finite() && interval_milliseconds > 0.0
        {
            MonotonicDuration::try_from_millis_f64(interval_milliseconds).ok()
        } else {
            None
        };
        let orientation = match screen_orientation_code {
            0 => SourceScreenOrientation::PortraitPrimary,
            1 => SourceScreenOrientation::LandscapePrimary,
            2 => SourceScreenOrientation::PortraitSecondary,
            3 => SourceScreenOrientation::LandscapeSecondary,
            _ => SourceScreenOrientation::Unknown,
        };
        let linear_acceleration = DVec3::new(
            linear_acceleration_x_metres_per_second_squared,
            linear_acceleration_y_metres_per_second_squared,
            linear_acceleration_z_metres_per_second_squared,
        );
        let linear_acceleration = linear_acceleration
            .is_finite()
            .then_some(linear_acceleration);
        let Ok(sample) = NormalizedMotionSample::new(
            SensorSequence::new(self.next_sensor_sequence),
            event_timestamp,
            receipt_timestamp,
            reported_interval,
            DVec3::new(
                gyro_x_radians_per_second,
                gyro_y_radians_per_second,
                gyro_z_radians_per_second,
            ),
            DVec3::new(
                specific_force_x_metres_per_second_squared,
                specific_force_y_metres_per_second_squared,
                specific_force_z_metres_per_second_squared,
            ),
            linear_acceleration,
            orientation,
        ) else {
            return false;
        };

        self.regularized_motion_timestamp_milliseconds = Some(regularized_timestamp_milliseconds);
        self.next_sensor_sequence = self.next_sensor_sequence.saturating_add(1);
        let outcome = self.sensor_buffer.push(sample);
        match outcome {
            PushOutcome::Inserted | PushOutcome::InsertedAndDroppedOldest { .. } => {}
            PushOutcome::RejectedLate { .. } => {
                // iOS delivers devicemotion in delayed bursts; at 30 Hz frame
                // draining, a burst can straddle a frame boundary and arrive
                // behind the drain watermark. That is platform behavior, not
                // caller error — count the sample and move on.
                self.late_motion_samples = self.late_motion_samples.saturating_add(1);
                self.motion_samples += 1;
                return true;
            }
            _ => return false,
        }
        if let Some(linear_acceleration) = linear_acceleration {
            let magnitude = linear_acceleration.length();
            if self.motion_samples < INITIAL_ACCELEROMETER_BIAS_SAMPLES && magnitude < 1.5 {
                let sample_count = (self.motion_samples + 1) as f64;
                let gain = 1.0 / sample_count;
                self.linear_acceleration_bias_device_mps2 = self
                    .linear_acceleration_bias_device_mps2
                    .lerp(linear_acceleration, gain);
            }
            self.latest_linear_acceleration_mps2 = if self.has_linear_acceleration {
                self.latest_linear_acceleration_mps2 * 0.9 + magnitude * 0.1
            } else {
                magnitude
            };
            self.has_linear_acceleration = true;
        }
        self.motion_samples += 1;
        true
    }

    /// Adds one downscaled grayscale camera frame and returns its normalized texture score.
    pub fn push_luma_frame(
        &mut self,
        frame_id: u32,
        timestamp_milliseconds: f64,
        width: u32,
        height: u32,
        pixels: &[u8],
    ) -> f64 {
        let Some(expected_len) = width
            .checked_mul(height)
            .and_then(|value| usize::try_from(value).ok())
        else {
            return -1.0;
        };
        if expected_len == 0 || pixels.len() != expected_len {
            return -1.0;
        }
        let Ok(timestamp) = MonotonicTimestamp::try_from_millis_f64(timestamp_milliseconds) else {
            return -1.0;
        };
        if self
            .latest_frame_timestamp
            .is_some_and(|previous| timestamp <= previous)
        {
            return -1.0;
        }

        self.latest_texture_score = texture_score(width as usize, pixels);
        self.position_before_inertial_prediction = self.camera_position;
        self.predict_inertial_translation(timestamp);
        let visual_orientation = self.visual_orientation_at(timestamp_milliseconds);
        self.process_frame(
            frame_id,
            width as usize,
            height as usize,
            pixels,
            visual_orientation,
        );
        self.previous_frame_orientation = Some(visual_orientation);
        self.latest_frame_id = frame_id;
        self.latest_frame_timestamp = Some(timestamp);
        self.frame_count += 1;
        self.latest_texture_score
    }

    /// Recenters the horizontal world heading while preserving gravity-aligned pitch and roll.
    pub fn recenter(&mut self) {
        if let Some(absolute) = self.absolute_orientation {
            self.orientation_reference = Some(yaw_reference(absolute));
            self.update_camera_orientation();
        }
        self.camera_position = DVec3::new(0.0, CAMERA_HEIGHT_METRES, 0.0);
        self.inertial_velocity_world_mps = DVec3::ZERO;
        self.position_before_inertial_prediction = self.camera_position;
        self.latest_frame_delta_seconds = 0.0;
        self.inertial_stationary_candidate = false;
        self.consecutive_stationary_frames = 0;
        self.frontend.reset();
        self.map.reset();
        self.preintegration = Preintegration::default();
        self.preintegration_valid = false;
        self.last_keyframe_frame_count = 0;
        self.track_anchor_pixels.clear();
        self.previous_frame_orientation = None;
        self.visual_matches = 0;
        self.visual_inliers = 0;
        self.frames_since_visual_update = u32::MAX;
        self.visual_relocalization_count = 0;
        self.last_relocalization_frame_id = None;
        self.clear_pending_appearance();
        self.latest_window_end_cost = 0.0;
        self.scale_initialized = false;
        self.latest_scale_ratio = 1.0;
        self.scale_confidence = 0.0;
    }

    /// Returns `[x, y, z, qx, qy, qz, qw, confidence]` for the Three.js camera.
    pub fn pose(&self) -> Vec<f64> {
        vec![
            self.camera_position.x,
            self.camera_position.y,
            self.camera_position.z,
            self.camera_orientation.x,
            self.camera_orientation.y,
            self.camera_orientation.z,
            self.camera_orientation.w,
            self.confidence(),
        ]
    }

    /// Flat `[x, y, state]` per live track, in tracker-frame pixels, for the
    /// debug overlay. State: 0 new, 1 tracked, 2 anchored, 3 anchored with a
    /// converged depth.
    pub fn tracked_points(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.frontend.tracks.len() * 3);
        for track in &self.frontend.tracks {
            let state = match track.state {
                TrackState::New => 0.0_f32,
                TrackState::Tracked => 1.0,
                TrackState::Anchored => {
                    let converged = track
                        .landmark
                        .and_then(|id| self.map.landmark(id))
                        .is_some_and(map::Landmark::converged);
                    if converged { 3.0 } else { 2.0 }
                }
            };
            out.push(track.pixel.0);
            out.push(track.pixel.1);
            out.push(state);
        }
        out
    }

    /// `[keyframes, landmarks, converged_landmarks, mean_scene_depth_metres,
    /// window_end_cost, scale_ratio, scale_confidence]` for the debug panel.
    /// `scale_ratio` is the latest metric-scale correction observation (1.0 =
    /// map already metric); `scale_confidence` grows with the acceleration
    /// excitation that made scale observable.
    pub fn map_stats(&self) -> Vec<f64> {
        vec![
            self.map.keyframes.len() as f64,
            self.map.landmarks.len() as f64,
            self.map.converged_landmark_count() as f64,
            self.map.mean_scene_depth(),
            self.latest_window_end_cost,
            self.latest_scale_ratio,
            self.scale_confidence,
        ]
    }

    /// Coarse tracking state: 0 initializing, 1 limited (orientation only), 2 tracking (6DoF).
    pub fn tracking_state(&self) -> u8 {
        if self.orientation_samples == 0 {
            return TRACKING_STATE_INITIALIZING;
        }
        if self.visual_inliers as usize >= MIN_VISUAL_INLIERS
            && self.frames_since_visual_update <= 3
        {
            TRACKING_STATE_TRACKING
        } else {
            TRACKING_STATE_LIMITED
        }
    }

    /// Blended 0..1 confidence from visual inliers, texture, and freshness.
    pub fn confidence(&self) -> f64 {
        if self.orientation_samples == 0 {
            return 0.0;
        }
        let inlier_confidence =
            (f64::from(self.visual_inliers) / (MIN_VISUAL_INLIERS as f64 * 2.5)).clamp(0.0, 1.0);
        let texture_confidence = (self.latest_texture_score / 0.08).clamp(0.0, 1.0);
        let freshness = if self.frames_since_visual_update <= 3 {
            1.0
        } else {
            0.4
        };
        (0.25 + 0.75 * inlier_confidence * freshness).min(0.25 + 0.75 * texture_confidence.max(0.4))
    }

    /// Total accepted camera frames.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Total accepted motion samples.
    pub fn motion_sample_count(&self) -> u64 {
        self.motion_samples
    }

    /// Motion samples that arrived behind the frame drain watermark and were
    /// dropped (delayed sensor delivery bursts).
    pub fn late_motion_sample_count(&self) -> u64 {
        self.late_motion_samples
    }

    /// Identifier of the most recent accepted frame.
    pub fn latest_frame_id(&self) -> u32 {
        self.latest_frame_id
    }

    /// Normalized texture score of the most recent frame.
    pub fn latest_texture_score(&self) -> f64 {
        self.latest_texture_score
    }

    /// Landmark observations attempted in the latest pose solve.
    pub fn visual_match_count(&self) -> u32 {
        self.visual_matches
    }

    /// Reprojection inliers in the latest pose solve.
    pub fn visual_inlier_count(&self) -> u32 {
        self.visual_inliers
    }

    /// Number of keyframes currently retained in the map.
    pub fn visual_keyframe_count(&self) -> u64 {
        self.map.keyframes.len() as u64
    }

    /// Identifier of the newest keyframe.
    pub fn latest_visual_keyframe_id(&self) -> u32 {
        self.map.keyframes.last().map_or(0, |keyframe| keyframe.id)
    }

    /// Total accepted appearance relocalizations.
    pub fn visual_relocalization_count(&self) -> u64 {
        self.visual_relocalization_count
    }

    /// Number of landmarks currently in the map.
    pub fn landmark_count(&self) -> u32 {
        self.map.landmarks.len() as u32
    }

    /// Smoothed magnitude of the most recent linear acceleration, m/s^2.
    pub fn linear_acceleration_magnitude(&self) -> f64 {
        self.latest_linear_acceleration_mps2
    }

    /// Current world-frame inertial velocity estimate `[x, y, z]`, m/s.
    pub fn inertial_velocity(&self) -> Vec<f64> {
        vec![
            self.inertial_velocity_world_mps.x,
            self.inertial_velocity_world_mps.y,
            self.inertial_velocity_world_mps.z,
        ]
    }

    /// True when the IMU suggests the device is currently stationary.
    pub fn inertial_stationary_candidate(&self) -> bool {
        self.inertial_stationary_candidate
    }

    /// Assumed field of view across the longer frame axis, degrees.
    pub fn long_axis_field_of_view_degrees(&self) -> f64 {
        self.long_axis_field_of_view_degrees
    }

    /// Horizontal field of view implied by the long-axis value at this frame size.
    pub fn horizontal_field_of_view_degrees(&self, width: u32, height: u32) -> f64 {
        geometry::horizontal_field_of_view_degrees(
            width as usize,
            height as usize,
            self.long_axis_field_of_view_degrees,
        )
    }

    /// Overrides the long-axis field of view (30..=130 degrees). Returns false on out-of-range input.
    pub fn set_long_axis_field_of_view_degrees(&mut self, degrees: f64) -> bool {
        if !(30.0..=130.0).contains(&degrees) {
            return false;
        }
        self.long_axis_field_of_view_degrees = degrees;
        true
    }

    /// Overrides the camera-vs-orientation latency compensation (0..=250 ms).
    pub fn set_visual_orientation_delay_milliseconds(&mut self, milliseconds: f64) -> bool {
        if !(0.0..=250.0).contains(&milliseconds) {
            return false;
        }
        self.visual_orientation_delay_milliseconds = milliseconds;
        true
    }

    /// Overrides the tracked-feature budget (20..=400).
    pub fn set_feature_budget(&mut self, budget: u32) -> bool {
        if !(20..=400).contains(&budget) {
            return false;
        }
        self.frontend.feature_budget = budget as usize;
        true
    }

    /// Enables or disables appearance relocalization (debug control).
    pub fn set_relocalization_enabled(&mut self, enabled: bool) {
        self.relocalization_enabled = enabled;
        if !enabled {
            self.clear_pending_appearance();
        }
    }
}

impl ArTracker {
    fn update_camera_orientation(&mut self) {
        let (Some(reference), Some(absolute)) =
            (self.orientation_reference, self.absolute_orientation)
        else {
            return;
        };
        self.camera_orientation = (reference.conjugate() * absolute).normalize();
    }

    fn visual_orientation_at(&self, frame_timestamp_milliseconds: f64) -> DQuat {
        let target_timestamp =
            frame_timestamp_milliseconds - self.visual_orientation_delay_milliseconds;
        let absolute = self.absolute_orientation_at(target_timestamp);
        self.orientation_reference
            .map_or(self.camera_orientation, |reference| {
                (reference.conjugate() * absolute).normalize()
            })
    }

    fn absolute_orientation_at(&self, timestamp_milliseconds: f64) -> DQuat {
        let Some(first) = self.orientation_history.front() else {
            return self.absolute_orientation.unwrap_or(self.camera_orientation);
        };
        let mut before = first;
        for after in self.orientation_history.iter().skip(1) {
            if after.timestamp_milliseconds >= timestamp_milliseconds {
                let duration = after.timestamp_milliseconds - before.timestamp_milliseconds;
                if duration <= f64::EPSILON {
                    return after.absolute;
                }
                let fraction = ((timestamp_milliseconds - before.timestamp_milliseconds)
                    / duration)
                    .clamp(0.0, 1.0);
                return before.absolute.slerp(after.absolute, fraction).normalize();
            }
            before = after;
        }
        before.absolute
    }

    fn next_regularized_motion_timestamp(
        &self,
        raw_timestamp_milliseconds: f64,
        interval_milliseconds: f64,
    ) -> f64 {
        match self.regularized_motion_timestamp_milliseconds {
            Some(previous)
                if interval_milliseconds.is_finite()
                    && (1.0..=50.0).contains(&interval_milliseconds) =>
            {
                let predicted = previous + interval_milliseconds;
                if !raw_timestamp_milliseconds.is_finite()
                    || raw_timestamp_milliseconds <= previous + interval_milliseconds * 0.25
                {
                    predicted
                } else {
                    let error = raw_timestamp_milliseconds - predicted;
                    if error.abs() > interval_milliseconds * 0.75 {
                        raw_timestamp_milliseconds.max(previous + 0.001)
                    } else {
                        predicted + error * 0.1
                    }
                }
            }
            Some(previous) if raw_timestamp_milliseconds.is_finite() => {
                raw_timestamp_milliseconds.max(previous + 0.001)
            }
            _ => raw_timestamp_milliseconds,
        }
    }

    /// Bounded inertial position/velocity propagation over the sensor samples
    /// between the previous and current frame. Also feeds the world-frame
    /// accelerometer preintegration that the bundle adjustment uses as its
    /// metric-scale hint.
    fn predict_inertial_translation(&mut self, frame_timestamp: MonotonicTimestamp) {
        let Some(previous_timestamp) = self.latest_frame_timestamp else {
            self.latest_frame_delta_seconds = 0.0;
            self.inertial_stationary_candidate = false;
            return;
        };
        let Ok(interval) = FrameInterval::new(previous_timestamp, frame_timestamp) else {
            return;
        };
        self.latest_frame_delta_seconds = interval.duration().as_secs_f64();
        let Ok(batch) = self.sensor_buffer.drain_interval(interval) else {
            self.inertial_stationary_candidate = false;
            self.consecutive_stationary_frames = 0;
            self.preintegration_valid = false;
            return;
        };

        let samples = batch.samples();
        if samples.is_empty() {
            self.inertial_stationary_candidate = false;
            self.consecutive_stationary_frames = 0;
            self.inertial_velocity_world_mps *=
                (-INERTIAL_VELOCITY_DAMPING_PER_SECOND * self.latest_frame_delta_seconds).exp();
            self.camera_position +=
                self.inertial_velocity_world_mps * self.latest_frame_delta_seconds;
            return;
        }

        let mut corrected_acceleration_sum = 0.0;
        let mut gyro_sum = 0.0;
        let mut raw_acceleration_sum = DVec3::ZERO;
        let mut used_sample_count = 0_u64;
        let mut position = self.camera_position;
        let mut velocity = self.inertial_velocity_world_mps;
        let mut cursor = previous_timestamp;
        let mut latest_acceleration_world = DVec3::ZERO;
        for sample in samples {
            let Some(linear_acceleration_device) = sample.linear_acceleration_mps2() else {
                continue;
            };
            let timestamp = sample.event_timestamp().min(frame_timestamp);
            let seconds = timestamp
                .checked_duration_since(cursor)
                .map_or(0.0, MonotonicDuration::as_secs_f64)
                .min(0.05);
            let absolute = self.absolute_orientation_at(sample.event_timestamp().as_millis_f64());
            let body_to_world = self
                .orientation_reference
                .map_or(self.camera_orientation, |reference| {
                    (reference.conjugate() * absolute).normalize()
                });
            let corrected_specific_force =
                sample.specific_force_mps2() - self.linear_acceleration_bias_device_mps2;
            let acceleration_world_raw = body_to_world * corrected_specific_force
                - DVec3::Y * GRAVITY_METRES_PER_SECOND_SQUARED;
            if acceleration_world_raw.is_finite() && acceleration_world_raw.length() <= 8.0 {
                // Preintegration keeps the physically consistent value; the
                // per-frame propagation applies the vertical damping hack that
                // keeps the rendered pose calm.
                self.preintegration.push(acceleration_world_raw, seconds);
                let mut acceleration_world = acceleration_world_raw;
                acceleration_world.y *= INERTIAL_VERTICAL_ACCELERATION_GAIN;
                position += velocity * seconds + acceleration_world * (0.5 * seconds * seconds);
                velocity += acceleration_world * seconds;
                latest_acceleration_world = acceleration_world;
            }
            corrected_acceleration_sum += acceleration_world_raw.length();
            gyro_sum += sample.angular_velocity_rad_s().length();
            raw_acceleration_sum += linear_acceleration_device;
            used_sample_count = used_sample_count.saturating_add(1);
            cursor = timestamp;
        }
        let remaining_seconds = frame_timestamp
            .checked_duration_since(cursor)
            .map_or(0.0, MonotonicDuration::as_secs_f64)
            .min(0.05);
        position += velocity * remaining_seconds
            + latest_acceleration_world * (0.5 * remaining_seconds * remaining_seconds);
        velocity += latest_acceleration_world * remaining_seconds;

        let sample_count = used_sample_count as f64;
        if used_sample_count == 0 {
            self.inertial_stationary_candidate = false;
            self.consecutive_stationary_frames = 0;
            self.camera_position = position;
            self.inertial_velocity_world_mps = velocity;
            return;
        }
        self.preintegration_valid = true;
        let mean_acceleration = corrected_acceleration_sum / sample_count;
        let mean_gyro = gyro_sum / sample_count;
        self.inertial_stationary_candidate = mean_acceleration < 0.18 && mean_gyro < 0.06;
        if self.inertial_stationary_candidate {
            self.consecutive_stationary_frames =
                self.consecutive_stationary_frames.saturating_add(1);
            let mean_raw_acceleration = raw_acceleration_sum / sample_count;
            self.linear_acceleration_bias_device_mps2 = self
                .linear_acceleration_bias_device_mps2
                .lerp(mean_raw_acceleration, 0.02);
        } else {
            self.consecutive_stationary_frames = 0;
        }

        velocity *= (-INERTIAL_VELOCITY_DAMPING_PER_SECOND * self.latest_frame_delta_seconds).exp();
        velocity.y *=
            (-INERTIAL_VERTICAL_DAMPING_PER_SECOND * self.latest_frame_delta_seconds).exp();
        if self.consecutive_stationary_frames >= 3 {
            velocity = DVec3::ZERO;
        } else if velocity.length() > MAX_INERTIAL_SPEED_METRES_PER_SECOND {
            velocity = velocity.normalize() * MAX_INERTIAL_SPEED_METRES_PER_SECOND;
        }
        self.camera_position = position;
        self.inertial_velocity_world_mps = velocity;
    }

    fn accept_visual_position(&mut self, target_position: DVec3, relocalized: bool) {
        if self.latest_frame_delta_seconds > 1.0e-4 {
            let visual_velocity = (target_position - self.position_before_inertial_prediction)
                / self.latest_frame_delta_seconds;
            if visual_velocity.is_finite()
                && visual_velocity.length() <= MAX_INERTIAL_SPEED_METRES_PER_SECOND * 1.5
            {
                let gain = if relocalized {
                    VISUAL_VELOCITY_CORRECTION_GAIN * 0.5
                } else {
                    VISUAL_VELOCITY_CORRECTION_GAIN
                };
                self.inertial_velocity_world_mps =
                    self.inertial_velocity_world_mps.lerp(visual_velocity, gain);
            }
        }
        if self.inertial_stationary_candidate
            && target_position.distance(self.position_before_inertial_prediction)
                < STATIONARY_TRANSLATION_DEADBAND_METRES
        {
            self.inertial_velocity_world_mps = DVec3::ZERO;
        }
        if relocalized {
            self.inertial_velocity_world_mps = DVec3::ZERO;
        }
        self.camera_position = target_position;
    }

    /// The visual pipeline for one frame: advance tracks, refine the camera
    /// position against the map, attempt relocalization, and manage keyframes.
    fn process_frame(
        &mut self,
        frame_id: u32,
        width: usize,
        height: usize,
        pixels: &[u8],
        orientation: DQuat,
    ) {
        let Ok(image_width) = u32::try_from(width) else {
            return;
        };
        let Ok(image_height) = u32::try_from(height) else {
            return;
        };
        let Some(image) = GrayImage::from_raw(image_width, image_height, pixels.to_vec()) else {
            return;
        };
        let intrinsics = Intrinsics::new(width, height, self.long_axis_field_of_view_degrees);

        // 1. Advance tracks, seeding each with a geometric prediction.
        let predicted_position = self.camera_position;
        let previous_orientation = self.previous_frame_orientation.unwrap_or(orientation);
        let mean_depth = self.map.mean_scene_depth();
        let map = &self.map;
        let seed = |track: &frontend::FeatureTrack| -> Option<(f32, f32)> {
            let world = match track.landmark.and_then(|id| map.landmark(id)) {
                Some(landmark) => map.landmark_world(landmark)?,
                None => {
                    // No depth yet: rotate the previous bearing at the mean
                    // scene depth (exact for pure rotation, close enough for
                    // seeding otherwise).
                    let bearing = intrinsics
                        .bearing(f64::from(track.pixel.0), f64::from(track.pixel.1));
                    predicted_position + previous_orientation * (bearing * mean_depth)
                }
            };
            let camera = geometry::world_to_camera(world, predicted_position, orientation);
            let (x, y) = intrinsics.project(camera)?;
            intrinsics
                .contains(x, y, 2.0)
                .then_some((x as f32, y as f32))
        };
        self.frontend.advance(image, seed);

        // 1b. Re-acquire lost landmarks from the newest keyframe when the
        // anchored set runs thin. Frame-to-frame LK dies under fast motion;
        // matching the stored keyframe image directly into the current frame
        // (pyramids + geometric seed) restores bindings without waiting for a
        // new keyframe — the resilience the exhaustive-search matcher had.
        let anchored_count = self
            .frontend
            .tracks
            .iter()
            .filter(|track| track.landmark.is_some())
            .count();
        if anchored_count < KEYFRAME_MIN_ANCHORED
            && let Some(newest) = self.map.keyframes.last()
        {
            let bound: Vec<u32> = self
                .frontend
                .tracks
                .iter()
                .filter_map(|track| track.landmark)
                .collect();
            let mut candidates = Vec::new();
            for observation in &newest.observations {
                if bound.contains(&observation.landmark) {
                    continue;
                }
                let Some(landmark) = self.map.landmark(observation.landmark) else {
                    continue;
                };
                let Some(world) = self.map.landmark_world(landmark) else {
                    continue;
                };
                let camera =
                    geometry::world_to_camera(world, self.camera_position, orientation);
                let Some((seed_x, seed_y)) = intrinsics.project(camera) else {
                    continue;
                };
                if !intrinsics.contains(seed_x, seed_y, 4.0) {
                    continue;
                }
                candidates.push((
                    observation.landmark,
                    observation.pixel,
                    (seed_x as f32, seed_y as f32),
                ));
            }
            if !candidates.is_empty()
                && let (Ok(reference_width), Ok(reference_height)) = (
                    u32::try_from(newest.full_width),
                    u32::try_from(newest.full_height),
                )
                && let Some(reference) = GrayImage::from_raw(
                    reference_width,
                    reference_height,
                    newest.full_luma.clone(),
                )
            {
                self.frontend.reacquire(&reference, &candidates);
            }
        }

        // 2. Per-frame position refinement against anchored landmarks.
        let mut observations = Vec::new();
        for track in &self.frontend.tracks {
            let Some(landmark_id) = track.landmark else {
                continue;
            };
            let Some(landmark) = self.map.landmark(landmark_id) else {
                continue;
            };
            let Some(world) = self.map.landmark_world(landmark) else {
                continue;
            };
            observations.push(FrameObservation {
                landmark: landmark_id,
                world,
                pixel_x: f64::from(track.pixel.0) - intrinsics.center_x,
                pixel_y: f64::from(track.pixel.1) - intrinsics.center_y,
            });
        }

        match solve_frame_pose(
            &observations,
            orientation,
            self.camera_position,
            &intrinsics,
        ) {
            Some(solution) => {
                self.visual_matches = u32::try_from(solution.matches).unwrap_or(u32::MAX);
                self.visual_inliers = u32::try_from(solution.inliers).unwrap_or(u32::MAX);
                // A yaw-orientation error during fast rotation is geometrically
                // indistinguishable from lateral translation at near-uniform
                // scene depth, so an oversized correction is far more likely a
                // rotation artifact than real motion. Reject it outright (the
                // previous tracker's keyframe-translation gate did the same) —
                // a clamped version would still inject drift every frame.
                let correction =
                    (solution.position - self.camera_position).length();
                let correction_bound =
                    (1.2 * self.latest_frame_delta_seconds).clamp(0.08, 0.35);
                // Strong consensus (many inliers, high inlier fraction) may
                // override the bound: that is the recovery case where the
                // *prediction* had drifted. The rotation artifact never has
                // strong consensus — the misfit is not rigid.
                let strong_consensus = solution.inliers >= 20
                    && solution.inliers * 10 >= solution.matches * 6;
                let accepted = solution.inliers >= MIN_VISUAL_INLIERS
                    && (correction <= correction_bound || strong_consensus);
                if accepted {
                    self.frames_since_visual_update = 0;
                    // Vertical visual corrections are the least trustworthy
                    // (depth-prior misfit projects mostly into y); apply them
                    // at half gain, as the previous tracker iteration did.
                    let mut target = solution.position;
                    target.y = self.camera_position.y
                        + (target.y - self.camera_position.y) * 0.5;
                    self.accept_visual_position(target, false);
                } else {
                    self.frames_since_visual_update =
                        self.frames_since_visual_update.saturating_add(1);
                }
                // Landmark bookkeeping only means something when the solve
                // itself was sane.
                if accepted {
                    for landmark_id in &solution.outlier_landmarks {
                        if let Some(landmark) = self.map.landmark_mut(*landmark_id) {
                            landmark.outlier_streak =
                                landmark.outlier_streak.saturating_add(1);
                        }
                    }
                    let outliers = &solution.outlier_landmarks;
                    for observation in &observations {
                        if !outliers.contains(&observation.landmark)
                            && let Some(landmark) =
                                self.map.landmark_mut(observation.landmark)
                        {
                            landmark.outlier_streak = 0;
                        }
                    }
                    let dropped = self.map.cull_outlier_landmarks(
                        LANDMARK_OUTLIER_STREAK_LIMIT,
                        UNCONVERGED_OUTLIER_STREAK_LIMIT,
                    );
                    if !dropped.is_empty() {
                        self.frontend.drop_landmarks(&dropped);
                    }
                }
            }
            None => {
                self.visual_matches = observations.len() as u32;
                self.visual_inliers = 0;
                self.frames_since_visual_update =
                    self.frames_since_visual_update.saturating_add(1);
            }
        }
        // During a prolonged visual outage, dead-reckoning velocity is more
        // likely stale than real: decay it hard so the pose parks instead of
        // sailing away and poisoning whatever the map does next.
        if self.frames_since_visual_update != u32::MAX && self.frames_since_visual_update > 4 {
            self.inertial_velocity_world_mps *= 0.5;
        }

        // 3. Appearance relocalization against stored keyframes — only while
        // the per-frame solve is failing (recovery), never during healthy
        // tracking.
        if self.relocalization_enabled
            && self.frames_since_visual_update > 2
            && self.frame_count.is_multiple_of(RELOCALIZATION_INTERVAL_FRAMES)
            && self.relocalization_ready(frame_id)
        {
            self.attempt_relocalization(frame_id, width, height, pixels, orientation);
        }

        // 4. Keyframe policy.
        if self.should_create_keyframe() {
            self.create_keyframe(width, height, pixels, orientation, &intrinsics);
            let report = solve_window(&mut self.map, &intrinsics);
            if let Some(report) = report {
                self.latest_window_end_cost = report.end_cost;
                // The newest keyframe was created at the current camera
                // position; carry any BA correction of it into the live pose.
                if let Some(newest) = self.map.keyframes.last() {
                    let correction = newest.position - self.camera_position;
                    if correction.is_finite() && correction.length() < 0.5 {
                        self.camera_position += correction;
                    }
                }
            }
            self.update_metric_scale();
        }
    }

    /// Metric-scale maintenance, run at keyframe rate. The closed-form
    /// estimator ([`scale`]) is only observable under acceleration excitation;
    /// the first confident estimate is applied in full (initialization), later
    /// ones as small bounded steps so the world never visibly breathes.
    fn update_metric_scale(&mut self) {
        let Some(estimate) = scale::estimate_scale(&self.map) else {
            return;
        };
        self.latest_scale_ratio = estimate.ratio;
        let confidence = (estimate.excitation / 2.0).clamp(0.0, 1.0);
        self.scale_confidence = self.scale_confidence * 0.9 + confidence * 0.1;
        if !self.scale_initialized {
            if confidence >= 0.4 {
                let ratio = estimate.ratio.clamp(0.2, 5.0);
                self.apply_map_scale(ratio);
                self.scale_initialized = true;
            }
            return;
        }
        // Maintenance: correct residual scale drift slowly and only when the
        // observation is meaningfully away from unity.
        if (estimate.ratio - 1.0).abs() > 0.05 {
            let step = estimate.ratio.clamp(0.95, 1.05);
            self.apply_map_scale(step);
        }
    }

    /// Rescales the entire metric state by `ratio` about the current camera
    /// position (so the on-screen view does not jump): keyframe positions and
    /// velocities, landmark depths, and the inertial velocity.
    fn apply_map_scale(&mut self, ratio: f64) {
        if !ratio.is_finite() || ratio <= 0.0 || (ratio - 1.0).abs() < 1.0e-6 {
            return;
        }
        let pivot = self.camera_position;
        for keyframe in &mut self.map.keyframes {
            keyframe.position = pivot + (keyframe.position - pivot) * ratio;
            keyframe.velocity *= ratio;
        }
        for landmark in &mut self.map.landmarks {
            landmark.inverse_depth = (landmark.inverse_depth / ratio)
                .clamp(map::MIN_INVERSE_DEPTH, map::MAX_INVERSE_DEPTH);
        }
        self.inertial_velocity_world_mps *= ratio;
        self.position_before_inertial_prediction =
            pivot + (self.position_before_inertial_prediction - pivot) * ratio;
        self.clear_pending_appearance();
    }

    fn should_create_keyframe(&self) -> bool {
        if self.map.keyframes.is_empty() {
            // Bootstrap as soon as the front-end has anything to anchor.
            return self.frontend.tracks.len() >= MIN_VISUAL_INLIERS && self.orientation_samples > 0;
        }
        if self.frame_count - self.last_keyframe_frame_count < KEYFRAME_MIN_FRAME_GAP {
            return false;
        }
        let anchored = self
            .frontend
            .tracks
            .iter()
            .filter(|track| track.landmark.is_some())
            .count();
        // A keyframe stamps the current camera position into the map as a new
        // anchor. During a visual outage that position is dead-reckoned and
        // drifting — anchoring landmarks there would make the map follow the
        // drift instead of correcting it. Only a total-loss recovery (no
        // anchored tracks at all) may restart the local map, and
        // `create_keyframe` zeroes the velocity when it does.
        let visual_fresh = self.frames_since_visual_update <= 2;
        if !visual_fresh {
            // Recovery restart: re-acquisition and relocalization both failed
            // to restore a usable anchored set, so restart the local map here
            // (with dead-reckoning halted by `create_keyframe`).
            return anchored < MIN_VISUAL_INLIERS && !self.frontend.tracks.is_empty();
        }
        if anchored < KEYFRAME_MIN_ANCHORED {
            return true;
        }
        // Median pixel flow since the last keyframe.
        let mut flows: Vec<f64> = self
            .frontend
            .tracks
            .iter()
            .filter_map(|track| {
                self.track_anchor_pixels
                    .iter()
                    .find(|(id, _)| *id == track.id)
                    .map(|(_, pixel)| {
                        (f64::from(track.pixel.0) - f64::from(pixel.0))
                            .hypot(f64::from(track.pixel.1) - f64::from(pixel.1))
                    })
            })
            .collect();
        if flows.is_empty() {
            return false;
        }
        flows.sort_by(f64::total_cmp);
        let median = flows[flows.len() / 2];
        let width = self
            .map
            .keyframes
            .last()
            .map_or(240.0, |keyframe| (keyframe.luma_width * RELOC_LUMA_DOWNSAMPLE) as f64);
        median >= width * KEYFRAME_FLOW_FRACTION
    }

    fn create_keyframe(
        &mut self,
        width: usize,
        height: usize,
        pixels: &[u8],
        orientation: DQuat,
        intrinsics: &Intrinsics,
    ) {
        // The bootstrap keyframe anchors even brand-new detections — there is
        // no age evidence yet, and the per-frame outlier culling cleans up any
        // that turn out flaky.
        let minimum_track_age = if self.map.keyframes.is_empty() {
            0
        } else {
            MIN_TRACK_AGE_FOR_LANDMARK
        };
        // Total-loss recovery keyframe: the position is dead-reckoned, so stop
        // extrapolating it any further.
        if self.frames_since_visual_update > 2 && !self.map.keyframes.is_empty() {
            self.inertial_velocity_world_mps = DVec3::ZERO;
        }
        let (luma, luma_width, luma_height) =
            downsample_luma(pixels, width, height, RELOC_LUMA_DOWNSAMPLE);
        let descriptor = reloc::visual_frame_descriptor(&luma, luma_width, luma_height);
        let preintegration = (self.preintegration_valid
            && estimator::preintegration_is_usable(&self.preintegration)
            && !self.map.keyframes.is_empty())
        .then_some(self.preintegration);

        let keyframe_id = self.map.push_keyframe(Keyframe {
            id: 0,
            position: self.camera_position,
            velocity: self.inertial_velocity_world_mps,
            orientation,
            observations: Vec::new(),
            preintegration,
            luma,
            luma_width,
            luma_height,
            descriptor,
            full_luma: pixels.to_vec(),
            full_width: width,
            full_height: height,
        });

        // Anchor mature unanchored tracks as new landmarks; record
        // observations for already-anchored tracks.
        let mut observations = Vec::new();
        let mut landmark_updates = Vec::new();
        for track in &mut self.frontend.tracks {
            let pixel = track.pixel;
            match track.landmark {
                Some(landmark_id) => {
                    observations.push(map::Observation {
                        landmark: landmark_id,
                        pixel,
                    });
                    landmark_updates.push((landmark_id, pixel));
                }
                None if track.age >= minimum_track_age => {
                    let bearing =
                        intrinsics.bearing(f64::from(pixel.0), f64::from(pixel.1));
                    let landmark_id = self.map.create_landmark(keyframe_id, bearing);
                    track.landmark = Some(landmark_id);
                    track.state = TrackState::Anchored;
                    observations.push(map::Observation {
                        landmark: landmark_id,
                        pixel,
                    });
                }
                None => {}
            }
        }
        let observer_position = self.camera_position;
        for (landmark_id, pixel) in landmark_updates {
            self.map
                .record_observation(landmark_id, observer_position, intrinsics, pixel);
        }
        if let Some(keyframe) = self.map.keyframes.last_mut() {
            keyframe.observations = observations;
        }

        self.track_anchor_pixels = self
            .frontend
            .tracks
            .iter()
            .map(|track| (track.id, track.pixel))
            .collect();
        self.last_keyframe_frame_count = self.frame_count;
        self.preintegration = Preintegration::default();
    }

    fn relocalization_ready(&self, frame_id: u32) -> bool {
        self.last_relocalization_frame_id
            .is_none_or(|last| frame_id.abs_diff(last) >= RELOCALIZATION_COOLDOWN_FRAMES)
    }

    fn attempt_relocalization(
        &mut self,
        frame_id: u32,
        width: usize,
        height: usize,
        pixels: &[u8],
        orientation: DQuat,
    ) {
        if self
            .pending_appearance_frame_id
            .is_some_and(|previous| frame_id.abs_diff(previous) > 2 * RELOCALIZATION_INTERVAL_FRAMES as u32)
        {
            self.clear_pending_appearance();
        }
        let (current_luma, current_width, current_height) =
            downsample_luma(pixels, width, height, RELOC_LUMA_DOWNSAMPLE);
        let current_descriptor =
            reloc::visual_frame_descriptor(&current_luma, current_width, current_height);
        let current_view = reloc::RelocView {
            frame_id,
            pixels: &current_luma,
            width: current_width,
            height: current_height,
            orientation,
            position: self.camera_position,
            descriptor: &current_descriptor,
        };
        let newest_keyframe_id = self.map.keyframes.last().map_or(0, |keyframe| keyframe.id);
        let keyframe_views: Vec<reloc::RelocView<'_>> = self
            .map
            .keyframes
            .iter()
            .filter(|keyframe| {
                newest_keyframe_id.saturating_sub(keyframe.id) >= RELOCALIZATION_MIN_KEYFRAME_AGE
            })
            .map(|keyframe| reloc::RelocView {
                // Age gating happens above; give the reloc module's own
                // frame-gap filter ids that always pass.
                frame_id: 0,
                pixels: &keyframe.luma,
                width: keyframe.luma_width,
                height: keyframe.luma_height,
                orientation: keyframe.orientation,
                position: keyframe.position,
                descriptor: &keyframe.descriptor,
            })
            .collect();
        if keyframe_views.is_empty() {
            return;
        }
        let Some(matched) = reloc::find_appearance_relocalization(
            &keyframe_views,
            &reloc::RelocView {
                frame_id: 1000,
                ..current_view
            },
            self.map.mean_scene_depth(),
            self.long_axis_field_of_view_degrees,
        ) else {
            return;
        };
        let matched_position =
            keyframe_views[matched.keyframe_index].position + matched.camera_delta_world;
        let accepted = matched.spatially_verified
            || self.confirm_smooth_appearance(frame_id, matched_position);
        if !accepted {
            return;
        }
        self.clear_pending_appearance();
        self.accept_visual_position(matched_position, true);
        self.visual_matches = u32::try_from(matched.matches).unwrap_or(u32::MAX);
        self.visual_inliers = u32::try_from(matched.inliers).unwrap_or(u32::MAX);
        self.visual_relocalization_count = self.visual_relocalization_count.saturating_add(1);
        self.last_relocalization_frame_id = Some(frame_id);
        self.frames_since_visual_update = 0;
    }

    fn confirm_smooth_appearance(&mut self, frame_id: u32, matched_position: DVec3) -> bool {
        match self.pending_appearance_frame_id {
            Some(_previous)
                if self.pending_appearance_position.distance(matched_position) < 0.3 =>
            {
                self.pending_appearance_confirmations =
                    self.pending_appearance_confirmations.saturating_add(1);
                self.pending_appearance_frame_id = Some(frame_id);
                self.pending_appearance_position = matched_position;
                self.pending_appearance_confirmations >= 2
            }
            _ => {
                self.pending_appearance_frame_id = Some(frame_id);
                self.pending_appearance_position = matched_position;
                self.pending_appearance_confirmations = 1;
                false
            }
        }
    }

    fn clear_pending_appearance(&mut self) {
        self.pending_appearance_frame_id = None;
        self.pending_appearance_position = DVec3::ZERO;
        self.pending_appearance_confirmations = 0;
    }
}

/// 2x2 (or n×n) box downsample of a GRAY8 frame.
fn downsample_luma(
    pixels: &[u8],
    width: usize,
    height: usize,
    factor: usize,
) -> (Vec<u8>, usize, usize) {
    if factor <= 1 {
        return (pixels.to_vec(), width, height);
    }
    let out_width = (width / factor).max(1);
    let out_height = (height / factor).max(1);
    let mut out = Vec::with_capacity(out_width * out_height);
    for out_y in 0..out_height {
        for out_x in 0..out_width {
            let mut sum = 0_u32;
            let mut count = 0_u32;
            for dy in 0..factor {
                for dx in 0..factor {
                    let x = out_x * factor + dx;
                    let y = out_y * factor + dy;
                    if x < width && y < height {
                        sum += u32::from(pixels[y * width + x]);
                        count += 1;
                    }
                }
            }
            out.push((sum / count.max(1)) as u8);
        }
    }
    (out, out_width, out_height)
}

/// Normalized mean absolute neighbor difference — a cheap texture proxy.
fn texture_score(width: usize, pixels: &[u8]) -> f64 {
    if width < 2 || pixels.len() <= width {
        return 0.0;
    }
    let mut total_difference = 0_u64;
    let mut comparisons = 0_u64;
    for (index, pixel) in pixels.iter().copied().enumerate() {
        if index % width != 0 {
            total_difference += u64::from(pixel.abs_diff(pixels[index - 1]));
            comparisons += 1;
        }
        if index >= width {
            total_difference += u64::from(pixel.abs_diff(pixels[index - width]));
            comparisons += 1;
        }
    }
    total_difference as f64 / comparisons.max(1) as f64 / 255.0
}

/// W3C device orientation (with screen-angle compensation) to a three.js-style
/// camera orientation quaternion. Matches DeviceOrientationControls.
fn device_orientation_quaternion(
    alpha_degrees: f64,
    beta_degrees: f64,
    gamma_degrees: f64,
    screen_angle_degrees: f64,
) -> DQuat {
    let degrees_to_radians = PI / 180.0;
    let device = DQuat::from_euler(
        EulerRot::YXZ,
        alpha_degrees * degrees_to_radians,
        beta_degrees * degrees_to_radians,
        -gamma_degrees * degrees_to_radians,
    );
    let camera_from_device = DQuat::from_rotation_x(-FRAC_PI_2);
    let screen_correction = DQuat::from_rotation_z(-screen_angle_degrees * degrees_to_radians);
    (device * camera_from_device * screen_correction).normalize()
}

/// Extracts the yaw-only component of an orientation for heading recentering.
fn yaw_reference(orientation: DQuat) -> DQuat {
    let forward = orientation * -DVec3::Z;
    let horizontal = DVec3::new(forward.x, 0.0, forward.z);
    if horizontal.length_squared() <= 1.0e-12 {
        DQuat::IDENTITY
    } else {
        let normalized = horizontal.normalize();
        DQuat::from_rotation_y((-normalized.x).atan2(-normalized.z))
    }
}

#[cfg(test)]
mod tests;
