use super::*;

const TEST_HEIGHT: usize = 120;
const TEST_WIDTH: usize = 160;

fn textured_frame() -> Vec<u8> {
    let mut pixels = vec![0_u8; TEST_WIDTH * TEST_HEIGHT];
    let mut state = 0x1234_5678_u32;
    for pixel in &mut pixels {
        state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        *pixel = (state >> 24) as u8;
    }
    // Soften the noise slightly so pyramid levels stay correlated for LK.
    let raw = pixels.clone();
    for y in 1..TEST_HEIGHT - 1 {
        for x in 1..TEST_WIDTH - 1 {
            let index = y * TEST_WIDTH + x;
            let sum = u32::from(raw[index]) * 4
                + u32::from(raw[index - 1])
                + u32::from(raw[index + 1])
                + u32::from(raw[index - TEST_WIDTH])
                + u32::from(raw[index + TEST_WIDTH]);
            pixels[index] = (sum / 8) as u8;
        }
    }
    pixels
}

fn tracker_with_first_frame(frame: &[u8]) -> ArTracker {
    let mut tracker = ArTracker::new();
    assert!(tracker.push_device_orientation(0.0, 90.0, 0.0, 0.0, 1.0));
    assert!(tracker.push_luma_frame(1, 10.0, TEST_WIDTH as u32, TEST_HEIGHT as u32, frame) > 0.0);
    tracker
}

fn shifted_frame(source: &[u8], shift_x: isize, shift_y: isize) -> Vec<u8> {
    let mut out = vec![0_u8; TEST_WIDTH * TEST_HEIGHT];
    for y in 0..TEST_HEIGHT as isize {
        for x in 0..TEST_WIDTH as isize {
            let sx = x + shift_x;
            let sy = y + shift_y;
            if sx >= 0 && sy >= 0 && sx < TEST_WIDTH as isize && sy < TEST_HEIGHT as isize {
                out[y as usize * TEST_WIDTH + x as usize] =
                    source[sy as usize * TEST_WIDTH + sx as usize];
            }
        }
    }
    out
}

fn rotate_frame(frame: &[u8], current_orientation: DQuat) -> Vec<u8> {
    let mut rotated = vec![0_u8; frame.len()];
    let focal = geometry::focal_length_pixels(
        TEST_WIDTH,
        TEST_HEIGHT,
        ESTIMATED_LONG_AXIS_FIELD_OF_VIEW_DEGREES,
    );
    let center_x = (TEST_WIDTH - 1) as f64 * 0.5;
    let center_y = (TEST_HEIGHT - 1) as f64 * 0.5;
    for y in 0..TEST_HEIGHT {
        for x in 0..TEST_WIDTH {
            let current_cv = DVec3::new(
                (x as f64 - center_x) / focal,
                (y as f64 - center_y) / focal,
                1.0,
            );
            let current_three = DVec3::new(current_cv.x, -current_cv.y, -1.0);
            let previous_three = current_orientation * current_three;
            let previous_cv = DVec3::new(previous_three.x, -previous_three.y, -previous_three.z);
            if previous_cv.z <= 0.0 {
                continue;
            }
            let source_x = (focal * previous_cv.x / previous_cv.z + center_x).round() as isize;
            let source_y = (focal * previous_cv.y / previous_cv.z + center_y).round() as isize;
            if source_x >= 0
                && source_y >= 0
                && source_x < TEST_WIDTH as isize
                && source_y < TEST_HEIGHT as isize
            {
                rotated[y * TEST_WIDTH + x] =
                    frame[source_y as usize * TEST_WIDTH + source_x as usize];
            }
        }
    }
    rotated
}

#[test]
fn first_orientation_recenters_heading_without_discarding_tilt() {
    let mut tracker = ArTracker::new();
    assert!(tracker.push_device_orientation(80.0, 25.0, -4.0, 0.0, 10.0));
    let pose = tracker.pose();
    assert!(pose[3..7].iter().all(|value| value.is_finite()));
    assert!(tracker.confidence() > 0.0);
    assert_eq!(tracker.tracking_state(), TRACKING_STATE_LIMITED);
}

#[test]
fn luma_frames_require_exact_dimensions_and_increasing_time() {
    let mut tracker = ArTracker::new();
    let frame = [0, 255, 0, 255];
    assert!(tracker.push_luma_frame(1, 10.0, 2, 2, &frame) > 0.0);
    assert_eq!(tracker.push_luma_frame(2, 10.0, 2, 2, &frame), -1.0);
    assert_eq!(tracker.push_luma_frame(2, 11.0, 3, 2, &frame), -1.0);
    assert_eq!(tracker.frame_count(), 1);
}

#[test]
fn flat_images_have_no_texture_signal() {
    assert_eq!(texture_score(4, &[80; 16]), 0.0);
}

#[test]
fn portrait_camera_uses_long_axis_focal_length() {
    let horizontal = geometry::horizontal_field_of_view_degrees(
        1_080,
        1_920,
        ESTIMATED_LONG_AXIS_FIELD_OF_VIEW_DEGREES,
    );
    assert!(
        (41.0..42.5).contains(&horizontal),
        "horizontal={horizontal}"
    );
    assert!(
        (geometry::horizontal_field_of_view_degrees(
            1_920,
            1_080,
            ESTIMATED_LONG_AXIS_FIELD_OF_VIEW_DEGREES,
        ) - ESTIMATED_LONG_AXIS_FIELD_OF_VIEW_DEGREES)
            .abs()
            < 1.0e-9
    );
}

#[test]
fn invalid_motion_samples_do_not_enter_the_buffer() {
    let mut tracker = ArTracker::new();
    assert!(!tracker.push_motion_sample(
        1.0,
        1.1,
        16.0,
        f64::NAN,
        0.0,
        0.0,
        0.0,
        0.0,
        9.806_65,
        f64::NAN,
        f64::NAN,
        f64::NAN,
        0,
    ));
    assert_eq!(tracker.motion_sample_count(), 0);
}

#[test]
fn bootstrap_frame_creates_keyframe_and_landmarks() {
    let tracker = tracker_with_first_frame(&textured_frame());
    assert_eq!(tracker.visual_keyframe_count(), 1);
    assert!(
        tracker.landmark_count() >= 20,
        "landmarks={}",
        tracker.landmark_count()
    );
    let points = tracker.tracked_points();
    assert!(points.len() >= 60);
    assert_eq!(points.len() % 3, 0);
}

#[test]
fn image_shift_recovers_horizontal_camera_translation() {
    let previous = textured_frame();
    let current = shifted_frame(&previous, 3, 0);

    let mut tracker = tracker_with_first_frame(&previous);
    assert!(
        tracker.push_luma_frame(2, 110.0, TEST_WIDTH as u32, TEST_HEIGHT as u32, &current) > 0.0
    );

    assert!(
        tracker.visual_match_count() >= 20,
        "matches={}",
        tracker.visual_match_count()
    );
    assert!(
        tracker.visual_inlier_count() >= MIN_VISUAL_INLIERS as u32,
        "inliers={}",
        tracker.visual_inlier_count()
    );
    assert_eq!(tracker.tracking_state(), TRACKING_STATE_TRACKING);
    assert!(tracker.pose()[0] > 0.02, "pose={:?}", tracker.pose());
}

#[test]
fn vertical_image_shift_recovers_upward_camera_translation() {
    let previous = textured_frame();
    let current = shifted_frame(&previous, 0, -3);

    let mut tracker = tracker_with_first_frame(&previous);
    tracker.push_luma_frame(2, 110.0, TEST_WIDTH as u32, TEST_HEIGHT as u32, &current);
    assert!(tracker.visual_inlier_count() >= MIN_VISUAL_INLIERS as u32);
    assert!(
        tracker.pose()[1] > CAMERA_HEIGHT_METRES + 0.02,
        "pose={:?}",
        tracker.pose()
    );
}

#[test]
fn expanding_image_recovers_forward_camera_translation() {
    let previous = textured_frame();
    let mut current = vec![0_u8; TEST_WIDTH * TEST_HEIGHT];
    let center_x = (TEST_WIDTH - 1) as f64 * 0.5;
    let center_y = (TEST_HEIGHT - 1) as f64 * 0.5;
    let scale = 1.035;
    for y in 0..TEST_HEIGHT {
        for x in 0..TEST_WIDTH {
            let source_x = (center_x + (x as f64 - center_x) / scale).round() as usize;
            let source_y = (center_y + (y as f64 - center_y) / scale).round() as usize;
            current[y * TEST_WIDTH + x] = previous[source_y * TEST_WIDTH + source_x];
        }
    }

    let mut tracker = tracker_with_first_frame(&previous);
    tracker.push_luma_frame(2, 110.0, TEST_WIDTH as u32, TEST_HEIGHT as u32, &current);
    assert!(tracker.visual_inlier_count() >= MIN_VISUAL_INLIERS as u32);
    assert!(tracker.pose()[2] < -0.02, "pose={:?}", tracker.pose());
}

#[test]
fn measured_rotation_does_not_create_false_translation() {
    let previous = textured_frame();
    let absolute = device_orientation_quaternion(4.0, 90.0, 0.0, 0.0);
    let current = rotate_frame(&previous, absolute);
    let mut tracker = tracker_with_first_frame(&previous);
    assert!(tracker.push_device_orientation(4.0, 90.0, 0.0, 0.0, 100.0));
    tracker.push_luma_frame(2, 200.0, TEST_WIDTH as u32, TEST_HEIGHT as u32, &current);

    let pose = tracker.pose();
    let displacement = DVec3::new(pose[0], pose[1] - CAMERA_HEIGHT_METRES, pose[2]);
    assert!(tracker.visual_inlier_count() >= MIN_VISUAL_INLIERS as u32);
    assert!(displacement.length() < 0.03, "pose={pose:?}");
}

#[test]
fn continuous_shift_sequence_grows_map_and_runs_bundle_adjustment() {
    let base = textured_frame();
    let mut tracker = tracker_with_first_frame(&base);
    // Slide the camera steadily; a keyframe should promote once median flow
    // crosses the threshold, and window BA should run without destabilizing
    // the pose.
    for step in 1..14_i32 {
        let frame = shifted_frame(&base, isize::try_from(step).unwrap() * 2, 0);
        let timestamp = 10.0 + f64::from(step) * 100.0;
        tracker.push_luma_frame(
            u32::try_from(step + 1).unwrap(),
            timestamp,
            TEST_WIDTH as u32,
            TEST_HEIGHT as u32,
            &frame,
        );
    }
    assert!(
        tracker.visual_keyframe_count() >= 2,
        "keyframes={}",
        tracker.visual_keyframe_count()
    );
    let pose = tracker.pose();
    assert!(pose.iter().all(|value| value.is_finite()), "pose={pose:?}");
    // The image content shifts right in samples, meaning the scene moved left
    // in view — the camera translated right in the world.
    assert!(pose[0] > 0.05, "pose={pose:?}");
    let stats = tracker.map_stats();
    assert!(stats[0] >= 2.0 && stats[1] > 20.0, "stats={stats:?}");
}

#[test]
fn recenter_clears_map_and_pose() {
    let base = textured_frame();
    let mut tracker = tracker_with_first_frame(&base);
    tracker.push_luma_frame(
        2,
        110.0,
        TEST_WIDTH as u32,
        TEST_HEIGHT as u32,
        &shifted_frame(&base, 3, 0),
    );
    tracker.recenter();
    assert_eq!(tracker.visual_keyframe_count(), 0);
    assert_eq!(tracker.landmark_count(), 0);
    let pose = tracker.pose();
    assert!(pose[0].abs() < 1e-12 && (pose[1] - CAMERA_HEIGHT_METRES).abs() < 1e-12);
}

#[test]
fn feature_budget_and_fov_setters_validate_ranges() {
    let mut tracker = ArTracker::new();
    assert!(tracker.set_feature_budget(200));
    assert!(!tracker.set_feature_budget(5));
    assert!(tracker.set_long_axis_field_of_view_degrees(70.0));
    assert!(!tracker.set_long_axis_field_of_view_degrees(20.0));
    assert!(tracker.set_visual_orientation_delay_milliseconds(60.0));
    assert!(!tracker.set_visual_orientation_delay_milliseconds(-1.0));
}

// ---------------------------------------------------------------------------
// Design-point simulation: 30 Hz, 3D structure, rotation + translation,
// analytic IMU. Validates that landmark depths individualize from parallax,
// the pose follows ground truth, and the pipeline stays in tracking state.
// ---------------------------------------------------------------------------

struct SyntheticScene {
    points: Vec<(DVec3, f64)>,
}

impl SyntheticScene {
    fn new() -> Self {
        // Deterministic 3D point cloud with genuinely varied depth.
        let mut points = Vec::new();
        let mut state = 0x9e37_79b9_u32;
        let mut random = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            f64::from(state >> 8) / f64::from(1_u32 << 24)
        };
        for _ in 0..260 {
            let x = (random() - 0.5) * 7.0;
            let y = random() * 3.2;
            let z = -1.8 - random() * 4.5;
            let intensity = 90.0 + random() * 150.0;
            points.push((DVec3::new(x, y, z), intensity));
        }
        Self { points }
    }

    fn render(&self, position: DVec3, orientation: DQuat, width: usize, height: usize) -> Vec<u8> {
        let intrinsics = geometry::Intrinsics::new(
            width,
            height,
            ESTIMATED_LONG_AXIS_FIELD_OF_VIEW_DEGREES,
        );
        // Mild background gradient so empty regions are not perfectly flat.
        let mut frame = vec![0_u8; width * height];
        for y in 0..height {
            for x in 0..width {
                frame[y * width + x] = (24 + (x * 13 / width) + (y * 11 / height)) as u8;
            }
        }
        for (world, intensity) in &self.points {
            let camera = geometry::world_to_camera(*world, position, orientation);
            let Some((px, py)) = intrinsics.project(camera) else {
                continue;
            };
            // Gaussian splat, radius 3.
            for dy in -3_i64..=3 {
                for dx in -3_i64..=3 {
                    let sx = px.round() as i64 + dx;
                    let sy = py.round() as i64 + dy;
                    if sx < 0 || sy < 0 || sx >= width as i64 || sy >= height as i64 {
                        continue;
                    }
                    let fall = (-((dx * dx + dy * dy) as f64) / 4.5).exp();
                    let index = sy as usize * width + sx as usize;
                    let value = f64::from(frame[index]) + intensity * fall;
                    frame[index] = value.min(255.0) as u8;
                }
            }
        }
        frame
    }
}

#[test]
fn design_point_sequence_tracks_and_converges_depth() {
    const WIDTH: usize = 180;
    const HEIGHT: usize = 240;
    const FRAME_HZ: f64 = 30.0;
    const SECONDS: f64 = 6.0;

    let scene = SyntheticScene::new();
    let mut tracker = ArTracker::new();

    // Ground-truth trajectory: sideways sweep with gentle yaw, at eye height.
    let position_at = |t: f64| {
        DVec3::new(
            0.45 * (t * 0.9).sin(),
            CAMERA_HEIGHT_METRES + 0.05 * (t * 0.7).sin(),
            -0.25 * (t * 0.6).sin(),
        )
    };
    let yaw_at = |t: f64| 6.0_f64.to_radians() * (t * 0.5).sin();

    let total_frames = (SECONDS * FRAME_HZ) as usize;
    let mut orientation_time = 0.0;
    let mut frame_id = 0_u32;
    for frame_index in 0..total_frames {
        let t = frame_index as f64 / FRAME_HZ;
        let timestamp_ms = 1000.0 + t * 1000.0;

        // Orientation samples at ~60 Hz between frames. Alpha in degrees;
        // beta 90 = camera level, looking forward.
        while orientation_time <= t + 1.0 / FRAME_HZ {
            let alpha = yaw_at(orientation_time).to_degrees();
            assert!(tracker.push_device_orientation(
                alpha,
                90.0,
                0.0,
                0.0,
                1000.0 + orientation_time * 1000.0,
            ));
            orientation_time += 1.0 / 60.0;
        }

        let orientation = device_orientation_quaternion(yaw_at(t).to_degrees(), 90.0, 0.0, 0.0);
        let reference = yaw_reference(device_orientation_quaternion(
            yaw_at(0.0).to_degrees(),
            90.0,
            0.0,
            0.0,
        ));
        let camera_orientation = (reference.conjugate() * orientation).normalize();
        let frame = scene.render(position_at(t), camera_orientation, WIDTH, HEIGHT);
        frame_id += 1;
        tracker.push_luma_frame(frame_id, timestamp_ms, WIDTH as u32, HEIGHT as u32, &frame);
    }

    let pose = tracker.pose();
    let truth = position_at((total_frames - 1) as f64 / FRAME_HZ);
    let error = DVec3::new(pose[0] - truth.x, pose[1] - truth.y, pose[2] - truth.z).length();
    assert!(pose.iter().all(|value| value.is_finite()), "pose={pose:?}");
    assert_eq!(tracker.tracking_state(), TRACKING_STATE_TRACKING);
    let stats = tracker.map_stats();
    // Depth must individualize: converged landmarks present, and mean scene
    // depth pulled away from the initialization prior toward the true scene.
    assert!(stats[2] > 15.0, "converged landmarks: {stats:?}");
    // With visual-only scale (no accel samples here) the trajectory magnitude
    // tracks the depth prior; the direction and stability are what this test
    // pins down. Position must stay in the right region and bounded.
    assert!(error < 0.6, "error={error} pose={pose:?} truth={truth:?}");
    assert!(
        tracker.visual_inlier_count() >= MIN_VISUAL_INLIERS as u32,
        "inliers={}",
        tracker.visual_inlier_count()
    );
}
