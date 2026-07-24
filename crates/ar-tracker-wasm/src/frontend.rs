//! Continuous frame-to-frame Lucas-Kanade feature tracking.
//!
//! This is the crate's "eyes": long-lived point tracks advanced every frame by
//! pyramidal LK (seeded with a geometric prediction from the IMU), culled by a
//! forward-backward consistency check, and topped up with grid Shi-Tomasi
//! detection so the tracked set never runs dry. All state is in pixels; the
//! estimator owns everything metric.

use image::GrayImage;
use optical_flow_lk::{
    DEFAULT_MIN_EIGEN_THRESHOLD, TrackStatus, TrackerContext, build_pyramid,
    calc_optical_flow_ex, good_features_to_track_grid,
};

const LK_PYRAMID_LEVELS: usize = 4;
const LK_WINDOW_SIZE: usize = 13;
const LK_MAX_ITERATIONS: usize = 20;
const LK_FB_THRESHOLD_PIXELS: f32 = 1.0;
/// Drop a track whose mean photometric residual (0..255) exceeds this.
const LK_MAX_PHOTOMETRIC_ERROR: f32 = 40.0;
/// Detection grid cell edge in pixels; one strongest corner per cell keeps
/// coverage uniform, which the translation solve depends on.
const DETECT_CELL_PIXELS: u32 = 20;
const DETECT_QUALITY_LEVEL: f32 = 0.05;
const DETECT_MIN_DISTANCE_PIXELS: u32 = 8;
/// Keep detections and tracks clear of the border so LK windows fit.
const BORDER_PIXELS: f32 = 8.0;

/// One landmark re-acquisition request: `(landmark_id, keyframe_pixel, seed_pixel)`.
pub type ReacquireCandidate = (u32, (f32, f32), (f32, f32));

/// Overlay/debug state of a track, mirrored into JS as a small integer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrackState {
    /// Newly detected this frame; not yet confirmed by a second frame.
    New = 0,
    /// Survived LK + forward-backward this frame.
    Tracked = 1,
    /// Carries a triangulating/triangulated landmark.
    Anchored = 2,
}

#[derive(Clone, Debug)]
pub struct FeatureTrack {
    pub id: u64,
    /// Current position in tracker-frame pixels.
    pub pixel: (f32, f32),
    /// Position on the previous frame (before this frame's advance).
    pub previous_pixel: (f32, f32),
    /// Frames survived.
    pub age: u32,
    /// Landmark this track observes, once anchored by the map.
    pub landmark: Option<u32>,
    pub state: TrackState,
    /// Smoothed LK photometric residual (0..255) — the flow's own certainty
    /// signal; low means the patch is being followed crisply.
    pub smoothed_error: f32,
}

pub struct FrontEndStats {
    pub advanced: usize,
    pub culled: usize,
    pub detected: usize,
}

pub struct FrontEnd {
    context: TrackerContext,
    previous: Option<GrayImage>,
    pub tracks: Vec<FeatureTrack>,
    next_track_id: u64,
    pub feature_budget: usize,
}

impl FrontEnd {
    pub fn new(feature_budget: usize) -> Self {
        Self {
            context: TrackerContext::new(),
            previous: None,
            tracks: Vec::new(),
            next_track_id: 0,
            feature_budget,
        }
    }

    pub fn reset(&mut self) {
        self.previous = None;
        self.tracks.clear();
    }

    /// Advances all tracks into `image`. `seed` supplies the predicted pixel
    /// for a track (IMU-rotation-compensated, landmark-projected when depth is
    /// known); `None` falls back to the track's current position.
    pub fn advance(
        &mut self,
        image: GrayImage,
        seed: impl Fn(&FeatureTrack) -> Option<(f32, f32)>,
    ) -> FrontEndStats {
        let mut stats = FrontEndStats {
            advanced: 0,
            culled: 0,
            detected: 0,
        };
        let Some(previous) = self.previous.take() else {
            self.previous = Some(image);
            self.detect(&mut stats);
            return stats;
        };

        if previous.dimensions() != image.dimensions() {
            // Resolution change (debug control): drop everything and restart.
            self.tracks.clear();
            self.previous = Some(image);
            self.detect(&mut stats);
            return stats;
        }

        if !self.tracks.is_empty() {
            self.context
                .prepare(&previous, &image, LK_PYRAMID_LEVELS);
            let points: Vec<(f32, f32)> = self.tracks.iter().map(|track| track.pixel).collect();
            let predicted: Vec<(f32, f32)> = self
                .tracks
                .iter()
                .map(|track| seed(track).unwrap_or(track.pixel))
                .collect();
            let results = self.context.track_fb(
                &points,
                Some(&predicted),
                LK_WINDOW_SIZE,
                LK_MAX_ITERATIONS,
                DEFAULT_MIN_EIGEN_THRESHOLD,
                LK_FB_THRESHOLD_PIXELS,
            );

            let width = image.width() as f32;
            let height = image.height() as f32;
            let mut kept = Vec::with_capacity(self.tracks.len());
            for (track, result) in self.tracks.drain(..).zip(results) {
                let usable = matches!(
                    result.status,
                    TrackStatus::Tracked | TrackStatus::Diverged
                ) && result.error <= LK_MAX_PHOTOMETRIC_ERROR
                    && result.pos.0 >= BORDER_PIXELS
                    && result.pos.1 >= BORDER_PIXELS
                    && result.pos.0 <= width - 1.0 - BORDER_PIXELS
                    && result.pos.1 <= height - 1.0 - BORDER_PIXELS;
                // Diverged (hit the iteration cap without formally converging)
                // is retained: the forward-backward round trip is the filter
                // that matters, and it already passed inside track_fb.
                if usable && result.status != TrackStatus::FbInconsistent {
                    let mut track = track;
                    track.previous_pixel = track.pixel;
                    track.pixel = result.pos;
                    track.age = track.age.saturating_add(1);
                    track.smoothed_error = track.smoothed_error * 0.7 + result.error * 0.3;
                    track.state = if track.landmark.is_some() {
                        TrackState::Anchored
                    } else {
                        TrackState::Tracked
                    };
                    kept.push(track);
                    stats.advanced += 1;
                } else {
                    stats.culled += 1;
                }
            }
            self.tracks = kept;
        }

        self.previous = Some(image);
        self.detect(&mut stats);
        stats
    }

    /// Tops the tracked set back up to the feature budget with grid Shi-Tomasi
    /// corners, avoiding existing tracks.
    fn detect(&mut self, stats: &mut FrontEndStats) {
        let Some(image) = self.previous.as_ref() else {
            return;
        };
        if self.tracks.len() >= self.feature_budget {
            return;
        }
        let columns = (image.width() / DETECT_CELL_PIXELS).max(1);
        let rows = (image.height() / DETECT_CELL_PIXELS).max(1);
        let existing: Vec<(f32, f32)> = self.tracks.iter().map(|track| track.pixel).collect();
        let corners = good_features_to_track_grid(
            image,
            columns,
            rows,
            1,
            DETECT_QUALITY_LEVEL,
            DETECT_MIN_DISTANCE_PIXELS,
            &existing,
        );
        let max_x = image.width() as f32 - 1.0 - BORDER_PIXELS;
        let max_y = image.height() as f32 - 1.0 - BORDER_PIXELS;
        let room = self.feature_budget.saturating_sub(self.tracks.len());
        for (x, y, _quality) in corners.into_iter().take(room) {
            let (x, y) = (x as f32, y as f32);
            if x < BORDER_PIXELS || y < BORDER_PIXELS || x > max_x || y > max_y {
                continue;
            }
            self.tracks.push(FeatureTrack {
                id: self.next_track_id,
                pixel: (x, y),
                previous_pixel: (x, y),
                age: 0,
                landmark: None,
                state: TrackState::New,
                smoothed_error: 10.0,
            });
            self.next_track_id += 1;
            stats.detected += 1;
        }
    }

    /// Re-acquires lost landmarks by tracking them from a stored keyframe
    /// image directly into the current frame — the large-displacement mode:
    /// pyramidal LK seeded by the geometric prediction, verified by a
    /// forward-backward round trip. `candidates` holds
    /// `(landmark_id, keyframe_pixel, seed_pixel)` for landmarks not currently
    /// bound to a live track. Returns how many were recovered as new tracks.
    pub fn reacquire(
        &mut self,
        reference: &GrayImage,
        candidates: &[ReacquireCandidate],
    ) -> usize {
        let Some(current) = self.previous.as_ref() else {
            return 0;
        };
        if candidates.is_empty() || reference.dimensions() != current.dimensions() {
            return 0;
        }
        let reference_pyramid = build_pyramid(reference, LK_PYRAMID_LEVELS);
        let current_pyramid = build_pyramid(current, LK_PYRAMID_LEVELS);
        let points: Vec<(f32, f32)> = candidates.iter().map(|(_, pixel, _)| *pixel).collect();
        let seeds: Vec<(f32, f32)> = candidates.iter().map(|(_, _, seed)| *seed).collect();
        let forward = calc_optical_flow_ex(
            &reference_pyramid,
            &current_pyramid,
            &points,
            Some(&seeds),
            LK_WINDOW_SIZE,
            LK_MAX_ITERATIONS,
            DEFAULT_MIN_EIGEN_THRESHOLD,
        );
        let forward_positions: Vec<(f32, f32)> =
            forward.iter().map(|result| result.pos).collect();
        let backward = calc_optical_flow_ex(
            &current_pyramid,
            &reference_pyramid,
            &forward_positions,
            Some(&points),
            LK_WINDOW_SIZE,
            LK_MAX_ITERATIONS,
            DEFAULT_MIN_EIGEN_THRESHOLD,
        );
        let width = current.width() as f32;
        let height = current.height() as f32;
        let usable = |status: TrackStatus| {
            matches!(status, TrackStatus::Tracked | TrackStatus::Diverged)
        };
        let mut recovered = 0;
        for (index, (landmark, keyframe_pixel, _)) in candidates.iter().enumerate() {
            let result = &forward[index];
            let reverse = &backward[index];
            if !usable(result.status)
                || !usable(reverse.status)
                || result.error > LK_MAX_PHOTOMETRIC_ERROR
            {
                continue;
            }
            let round_trip = (reverse.pos.0 - keyframe_pixel.0)
                .hypot(reverse.pos.1 - keyframe_pixel.1);
            if round_trip > LK_FB_THRESHOLD_PIXELS * 1.5 {
                continue;
            }
            let position = result.pos;
            if position.0 < BORDER_PIXELS
                || position.1 < BORDER_PIXELS
                || position.0 > width - 1.0 - BORDER_PIXELS
                || position.1 > height - 1.0 - BORDER_PIXELS
            {
                continue;
            }
            self.tracks.push(FeatureTrack {
                id: self.next_track_id,
                pixel: position,
                previous_pixel: position,
                age: 1,
                landmark: Some(*landmark),
                state: TrackState::Anchored,
                smoothed_error: 15.0,
            });
            self.next_track_id += 1;
            recovered += 1;
        }
        recovered
    }

    /// Removes tracks whose landmark ids appear in `landmarks` (used when the
    /// map discards bad landmarks).
    pub fn drop_landmarks(&mut self, landmarks: &[u32]) {
        for track in &mut self.tracks {
            if let Some(id) = track.landmark
                && landmarks.contains(&id)
            {
                track.landmark = None;
                track.state = TrackState::Tracked;
            }
        }
    }
}
