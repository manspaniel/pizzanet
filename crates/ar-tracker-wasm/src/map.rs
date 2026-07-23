//! The persistent map: keyframes, anchored inverse-depth landmarks, and the
//! sliding optimization window.
//!
//! Landmarks are parameterized as an anchor keyframe plus a fixed camera-frame
//! bearing and a single optimizable inverse depth — the standard monocular
//! choice, well conditioned at low parallax and graceful for distant points.
//! Keyframes hold IMU orientation (never optimized), an optimizable position
//! and velocity, the pixel observations that feed bundle adjustment, and a
//! downsampled luma copy for appearance relocalization.

use crate::geometry::{Intrinsics, landmark_world_position};
use vio_core::{DQuat, DVec3};

/// Depth prior a fresh landmark starts at, in metres along the optical axis.
/// Chosen for indoor AR; bundle adjustment individualizes it as parallax
/// accumulates ("prior + converge silently").
pub const INITIAL_DEPTH_METRES: f64 = 3.5;
pub const MIN_INVERSE_DEPTH: f64 = 1.0 / 40.0;
pub const MAX_INVERSE_DEPTH: f64 = 1.0 / 0.15;
/// Parallax (degrees, from the anchor bearing) above which a landmark's depth
/// is considered observed rather than prior-driven.
pub const CONVERGED_PARALLAX_DEGREES: f64 = 1.5;
/// Bundle-adjustment window size in keyframes.
pub const WINDOW_KEYFRAMES: usize = 6;
/// Total retained keyframes (older ones serve relocalization and as frozen
/// landmark anchors). Bounded by stored keyframe imagery memory.
pub const MAX_KEYFRAMES: usize = 24;

/// World-frame accelerometer preintegration between two consecutive keyframes:
/// `dv = ∫a dt`, `dp = ∫∫a dt²`, both gravity-removed and bias-corrected.
/// Rotation is taken from the IMU per sample, so these are plain world-frame
/// integrals with no re-linearization state.
#[derive(Clone, Copy, Debug, Default)]
pub struct Preintegration {
    pub duration_seconds: f64,
    pub delta_velocity: DVec3,
    pub delta_position: DVec3,
    pub sample_count: u32,
}

impl Preintegration {
    pub fn push(&mut self, acceleration_world: DVec3, seconds: f64) {
        let seconds = seconds.clamp(0.0, 0.05);
        self.delta_position +=
            self.delta_velocity * seconds + acceleration_world * (0.5 * seconds * seconds);
        self.delta_velocity += acceleration_world * seconds;
        self.duration_seconds += seconds;
        self.sample_count += 1;
    }
}

#[derive(Clone, Debug)]
pub struct Observation {
    pub landmark: u32,
    pub pixel: (f32, f32),
}

#[derive(Clone)]
pub struct Keyframe {
    pub id: u32,
    pub position: DVec3,
    pub velocity: DVec3,
    /// Camera orientation from the IMU at capture time. Fixed forever.
    pub orientation: DQuat,
    pub observations: Vec<Observation>,
    /// Preintegrated accelerometer between the previous retained keyframe and
    /// this one. `None` for the first keyframe or after an IMU gap.
    pub preintegration: Option<Preintegration>,
    /// Downsampled luma for appearance relocalization.
    pub luma: Vec<u8>,
    pub luma_width: usize,
    pub luma_height: usize,
    pub descriptor: Vec<i8>,
    /// Processing-resolution luma for landmark re-acquisition.
    pub full_luma: Vec<u8>,
    pub full_width: usize,
    pub full_height: usize,
}

#[derive(Clone, Debug)]
pub struct Landmark {
    pub id: u32,
    pub anchor: u32,
    /// Anchor-camera-frame bearing `(bx, by, -1)`.
    pub bearing: DVec3,
    /// Inverse of the optical-axis depth in the anchor frame, 1/m.
    pub inverse_depth: f64,
    /// Keyframes that observed this landmark (including the anchor).
    pub observation_count: u32,
    /// Largest bearing separation seen from the anchor, degrees. The depth
    /// convergence proxy.
    pub max_parallax_degrees: f64,
    /// Consecutive pose-solve rounds this landmark was a reprojection outlier.
    pub outlier_streak: u32,
}

impl Landmark {
    pub fn converged(&self) -> bool {
        self.max_parallax_degrees >= CONVERGED_PARALLAX_DEGREES && self.observation_count >= 3
    }
}

pub struct Map {
    pub keyframes: Vec<Keyframe>,
    pub landmarks: Vec<Landmark>,
    next_keyframe_id: u32,
    next_landmark_id: u32,
}

impl Map {
    pub fn new() -> Self {
        Self {
            keyframes: Vec::new(),
            landmarks: Vec::new(),
            next_keyframe_id: 0,
            next_landmark_id: 0,
        }
    }

    pub fn reset(&mut self) {
        self.keyframes.clear();
        self.landmarks.clear();
    }

    pub fn keyframe(&self, id: u32) -> Option<&Keyframe> {
        self.keyframes.iter().find(|keyframe| keyframe.id == id)
    }

    pub fn landmark(&self, id: u32) -> Option<&Landmark> {
        self.landmarks.iter().find(|landmark| landmark.id == id)
    }

    pub fn landmark_mut(&mut self, id: u32) -> Option<&mut Landmark> {
        self.landmarks.iter_mut().find(|landmark| landmark.id == id)
    }

    pub fn landmark_world(&self, landmark: &Landmark) -> Option<DVec3> {
        let anchor = self.keyframe(landmark.anchor)?;
        Some(landmark_world_position(
            anchor.position,
            anchor.orientation,
            landmark.bearing,
            landmark.inverse_depth,
        ))
    }

    /// Adds a keyframe and returns its id. Evicts the oldest non-window
    /// keyframe (and landmarks left without a live anchor) past capacity.
    pub fn push_keyframe(&mut self, mut keyframe: Keyframe) -> u32 {
        let id = self.next_keyframe_id;
        self.next_keyframe_id += 1;
        keyframe.id = id;
        self.keyframes.push(keyframe);

        if self.keyframes.len() > MAX_KEYFRAMES {
            let evicted = self.keyframes.remove(0);
            let anchor_gone: Vec<u32> = self
                .landmarks
                .iter()
                .filter(|landmark| landmark.anchor == evicted.id)
                .map(|landmark| landmark.id)
                .collect();
            self.landmarks
                .retain(|landmark| landmark.anchor != evicted.id);
            // Also drop the evicted keyframe's observations of surviving
            // landmarks — they refer to a pose that no longer exists.
            let _ = anchor_gone;
        }
        id
    }

    /// Creates a landmark anchored at `anchor` observing pixel bearing
    /// `bearing`, at the depth prior.
    pub fn create_landmark(&mut self, anchor: u32, bearing: DVec3) -> u32 {
        let id = self.next_landmark_id;
        self.next_landmark_id += 1;
        self.landmarks.push(Landmark {
            id,
            anchor,
            bearing,
            inverse_depth: 1.0 / INITIAL_DEPTH_METRES,
            observation_count: 1,
            max_parallax_degrees: 0.0,
            outlier_streak: 0,
        });
        id
    }

    /// Records that `landmark` was observed at `pixel` in the keyframe being
    /// built, and updates its parallax bookkeeping given the observing pose.
    pub fn record_observation(
        &mut self,
        landmark_id: u32,
        observer_position: DVec3,
        intrinsics: &Intrinsics,
        pixel: (f32, f32),
    ) {
        let Some(world) = self
            .landmark(landmark_id)
            .and_then(|landmark| self.landmark_world(landmark))
        else {
            return;
        };
        let anchor_position = self
            .landmark(landmark_id)
            .and_then(|landmark| self.keyframe(landmark.anchor))
            .map(|keyframe| keyframe.position);
        let Some(anchor_position) = anchor_position else {
            return;
        };
        let _ = intrinsics;
        let _ = pixel;
        let ray_anchor = (world - anchor_position).normalize_or_zero();
        let ray_observer = (world - observer_position).normalize_or_zero();
        let parallax = ray_anchor
            .dot(ray_observer)
            .clamp(-1.0, 1.0)
            .acos()
            .to_degrees();
        if let Some(landmark) = self.landmark_mut(landmark_id) {
            landmark.observation_count += 1;
            if parallax > landmark.max_parallax_degrees {
                landmark.max_parallax_degrees = parallax;
            }
        }
    }

    /// Removes landmarks with a persistent outlier streak; returns their ids
    /// so the front-end can unbind tracks. Unconverged landmarks get a longer
    /// leash — their depth is still being learned, so misfits are expected.
    pub fn cull_outlier_landmarks(&mut self, converged_streak: u32, unconverged_streak: u32) -> Vec<u32> {
        let over_limit = |landmark: &Landmark| {
            let limit = if landmark.converged() {
                converged_streak
            } else {
                unconverged_streak
            };
            landmark.outlier_streak >= limit
        };
        let dropped: Vec<u32> = self
            .landmarks
            .iter()
            .filter(|landmark| over_limit(landmark))
            .map(|landmark| landmark.id)
            .collect();
        if !dropped.is_empty() {
            self.landmarks.retain(|landmark| !over_limit(landmark));
        }
        dropped
    }

    /// Ids of the keyframes inside the optimization window (the most recent
    /// `WINDOW_KEYFRAMES`).
    pub fn window_ids(&self) -> Vec<u32> {
        let start = self.keyframes.len().saturating_sub(WINDOW_KEYFRAMES);
        self.keyframes[start..]
            .iter()
            .map(|keyframe| keyframe.id)
            .collect()
    }

    pub fn converged_landmark_count(&self) -> usize {
        self.landmarks
            .iter()
            .filter(|landmark| landmark.converged())
            .count()
    }

    pub fn mean_scene_depth(&self) -> f64 {
        let converged: Vec<f64> = self
            .landmarks
            .iter()
            .filter(|landmark| landmark.converged())
            .map(|landmark| 1.0 / landmark.inverse_depth.max(MIN_INVERSE_DEPTH))
            .collect();
        if converged.is_empty() {
            INITIAL_DEPTH_METRES
        } else {
            converged.iter().sum::<f64>() / converged.len() as f64
        }
    }
}
