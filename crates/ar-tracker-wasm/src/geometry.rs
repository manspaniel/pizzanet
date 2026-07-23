//! Pinhole camera geometry shared by the front-end, map, and estimator.
//!
//! Conventions:
//! - World frame: three.js — x right, y up, z toward the viewer; the camera
//!   looks down -z. Gravity is along -y.
//! - Camera frame (three-style): a point in front of the camera has negative z.
//! - Pixels: origin at the top-left, x right, y down.
//! - A landmark "bearing" is the three-style camera-frame direction scaled so
//!   its optical-axis component is exactly -1: `(bx, by, -1)`. Multiplying the
//!   bearing by the optical-axis depth `d` (or dividing by inverse depth
//!   `rho = 1/d`) gives the camera-frame point; this makes the projection of a
//!   landmark into its own anchor exact regardless of depth.

use vio_core::{DQuat, DVec3};

/// Focal length in pixels for a frame whose long axis spans the given field of
/// view.
pub fn focal_length_pixels(
    width: usize,
    height: usize,
    long_axis_field_of_view_degrees: f64,
) -> f64 {
    width.max(height) as f64 * 0.5 / (long_axis_field_of_view_degrees * 0.5).to_radians().tan()
}

/// Horizontal field of view implied by the long-axis field of view at a given
/// aspect.
pub fn horizontal_field_of_view_degrees(
    width: usize,
    height: usize,
    long_axis_field_of_view_degrees: f64,
) -> f64 {
    let focal = focal_length_pixels(width, height, long_axis_field_of_view_degrees);
    (2.0 * (width as f64 * 0.5 / focal).atan()).to_degrees()
}

/// Intrinsics for one processed frame size.
#[derive(Clone, Copy, Debug)]
pub struct Intrinsics {
    pub focal: f64,
    pub center_x: f64,
    pub center_y: f64,
    pub width: usize,
    pub height: usize,
}

impl Intrinsics {
    pub fn new(width: usize, height: usize, long_axis_field_of_view_degrees: f64) -> Self {
        Self {
            focal: focal_length_pixels(width, height, long_axis_field_of_view_degrees),
            center_x: (width.saturating_sub(1)) as f64 * 0.5,
            center_y: (height.saturating_sub(1)) as f64 * 0.5,
            width,
            height,
        }
    }

    /// Camera-frame bearing `(bx, by, -1)` for a pixel.
    pub fn bearing(&self, pixel_x: f64, pixel_y: f64) -> DVec3 {
        DVec3::new(
            (pixel_x - self.center_x) / self.focal,
            -(pixel_y - self.center_y) / self.focal,
            -1.0,
        )
    }

    /// Projects a three-style camera-frame point to a pixel. `None` when the
    /// point is at or behind the camera plane.
    pub fn project(&self, point_camera: DVec3) -> Option<(f64, f64)> {
        let depth = -point_camera.z;
        if depth <= 0.05 {
            return None;
        }
        Some((
            self.focal * point_camera.x / depth + self.center_x,
            self.focal * -point_camera.y / depth + self.center_y,
        ))
    }

    pub fn contains(&self, pixel_x: f64, pixel_y: f64, margin: f64) -> bool {
        pixel_x >= margin
            && pixel_y >= margin
            && pixel_x <= self.width as f64 - 1.0 - margin
            && pixel_y <= self.height as f64 - 1.0 - margin
    }
}

/// World position of an anchored inverse-depth landmark.
pub fn landmark_world_position(
    anchor_position: DVec3,
    anchor_orientation: DQuat,
    bearing: DVec3,
    inverse_depth: f64,
) -> DVec3 {
    anchor_position + anchor_orientation * (bearing / inverse_depth.max(1.0e-4))
}

/// A world point expressed in a camera's three-style local frame.
pub fn world_to_camera(point_world: DVec3, position: DVec3, orientation: DQuat) -> DVec3 {
    orientation.conjugate() * (point_world - position)
}
