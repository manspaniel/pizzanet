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
/// landmark anchors). Metric-scale calibration keeps a separate lightweight
/// history, so this image-heavy map can stay bounded.
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
        self.push_hold(acceleration_world, seconds);
        self.sample_count += 1;
    }

    /// Integrates a held (extrapolated) acceleration over a sub-interval
    /// without counting a measured sample. Used for the trailing segment of a
    /// frame interval, so `duration_seconds` matches the real keyframe
    /// interval — the estimator and scale solver divide by it.
    pub fn push_hold(&mut self, acceleration_world: DVec3, seconds: f64) {
        let seconds = seconds.clamp(0.0, 0.05);
        self.delta_position +=
            self.delta_velocity * seconds + acceleration_world * (0.5 * seconds * seconds);
        self.delta_velocity += acceleration_world * seconds;
        self.duration_seconds += seconds;
    }

    /// Advances the time base over a sub-interval with no usable acceleration
    /// estimate at all (coast at the current delta velocity).
    pub fn push_gap(&mut self, seconds: f64) {
        self.push_hold(DVec3::ZERO, seconds);
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
            self.landmarks
                .retain(|landmark| landmark.anchor != evicted.id);
        }
        id
    }

    /// Creates a landmark anchored at `anchor` observing pixel bearing
    /// `bearing`. New landmarks inherit the map's current mean converged
    /// depth rather than the bootstrap prior — otherwise every fresh landmark
    /// injects prior-scale back into a map whose gauge has already been
    /// corrected, and the scale re-inflates continuously.
    pub fn create_landmark(&mut self, anchor: u32, bearing: DVec3) -> u32 {
        let id = self.next_landmark_id;
        self.next_landmark_id += 1;
        let initial_depth = self.mean_scene_depth();
        self.landmarks.push(Landmark {
            id,
            anchor,
            bearing,
            inverse_depth: 1.0 / initial_depth.clamp(0.2, 30.0),
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
        observer_orientation: DQuat,
        intrinsics: &Intrinsics,
        pixel: (f32, f32),
    ) {
        let Some((anchor_orientation, anchor_bearing)) =
            self.landmark(landmark_id).and_then(|landmark| {
                self.keyframe(landmark.anchor)
                    .map(|anchor| (anchor.orientation, landmark.bearing))
            })
        else {
            return;
        };
        // Parallax is an angular observation and must come from the measured
        // rays. Deriving the observer ray from the landmark's current
        // prior-depth world point makes convergence circular: a wrong depth
        // can falsely declare itself observed and release its prior.
        let ray_anchor = (anchor_orientation * anchor_bearing).normalize_or_zero();
        let ray_observer = (observer_orientation
            * intrinsics.bearing(f64::from(pixel.0), f64::from(pixel.1)))
        .normalize_or_zero();
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
    pub fn cull_outlier_landmarks(
        &mut self,
        converged_streak: u32,
        unconverged_streak: u32,
    ) -> Vec<u32> {
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
            initial_depth_metres()
        } else {
            converged.iter().sum::<f64>() / converged.len() as f64
        }
    }
}

fn initial_depth_metres() -> f64 {
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(depth) = std::env::var("AR_INITIAL_DEPTH_METRES")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| value.is_finite() && (0.2..=30.0).contains(value))
    {
        return depth;
    }
    INITIAL_DEPTH_METRES
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keyframe(orientation: DQuat) -> Keyframe {
        Keyframe {
            id: 0,
            position: DVec3::new(0.0, 1.6, 0.0),
            velocity: DVec3::ZERO,
            orientation,
            observations: Vec::new(),
            preintegration: None,
            luma: Vec::new(),
            luma_width: 0,
            luma_height: 0,
            descriptor: Vec::new(),
            full_luma: Vec::new(),
            full_width: 0,
            full_height: 0,
        }
    }

    #[test]
    fn parallax_uses_measured_observer_ray_instead_of_depth_prior() {
        let intrinsics = Intrinsics::new(240, 427, 73.7);
        let mut shallow = Map::new();
        let anchor = shallow.push_keyframe(keyframe(DQuat::IDENTITY));
        let landmark = shallow.create_landmark(anchor, DVec3::new(0.0, 0.0, -1.0));
        shallow.landmark_mut(landmark).unwrap().inverse_depth = 1.0 / 0.2;

        let mut deep = Map::new();
        let anchor = deep.push_keyframe(keyframe(DQuat::IDENTITY));
        let deep_landmark = deep.create_landmark(anchor, DVec3::new(0.0, 0.0, -1.0));
        deep.landmark_mut(deep_landmark).unwrap().inverse_depth = 1.0 / 30.0;

        let pixel = (
            (intrinsics.center_x + intrinsics.focal * 0.1) as f32,
            intrinsics.center_y as f32,
        );
        shallow.record_observation(landmark, DQuat::IDENTITY, &intrinsics, pixel);
        deep.record_observation(deep_landmark, DQuat::IDENTITY, &intrinsics, pixel);

        let shallow_angle = shallow.landmark(landmark).unwrap().max_parallax_degrees;
        let deep_angle = deep.landmark(deep_landmark).unwrap().max_parallax_degrees;
        assert!((shallow_angle - deep_angle).abs() < 1.0e-9);
        assert!((shallow_angle - 0.1_f64.atan().to_degrees()).abs() < 0.01);
    }

    #[test]
    fn keyframe_eviction_removes_landmarks_with_dead_anchors() {
        let mut map = Map::new();
        let first = map.push_keyframe(keyframe(DQuat::IDENTITY));
        let landmark = map.create_landmark(first, DVec3::new(0.0, 0.0, -1.0));
        for _ in 1..=MAX_KEYFRAMES {
            map.push_keyframe(keyframe(DQuat::IDENTITY));
        }
        assert!(map.keyframe(first).is_none());
        assert!(map.landmark(landmark).is_none());
    }
}
