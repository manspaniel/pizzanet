//! Deterministic source-image augmentation shared by pixels and annotations.

use image::{Rgb, RgbImage};
use synth_data::{FrameRecord, KeypointLabel, Visibility};

use super::sample_hash;

const MAX_ROLL_DEGREES: f32 = 8.0;
const ROLL_HASH_SALT: u64 = 0x524f_4c4c;

/// Resamples an image to the raster on which training augmentation runs.
///
/// The long edge is exactly `long_edge`; the short edge is rounded once and
/// remains at least one pixel. Performing roll, flip, blur, and colour jitter
/// on this bounded raster avoids repeatedly processing multi-megapixel source
/// photographs while preserving their normalized coordinate system.
#[must_use]
pub(super) fn resize_to_working_raster(source: &RgbImage, long_edge: usize) -> RgbImage {
    let (width, height) = source.dimensions();
    assert!(
        width > 0 && height > 0,
        "working raster requires a non-empty image"
    );
    assert!(long_edge > 0, "working raster long edge must be nonzero");
    let long_edge = u32::try_from(long_edge).expect("model input size fits in u32");
    let (working_width, working_height) = if width >= height {
        (
            long_edge,
            rounded_scaled_dimension(height, width, long_edge),
        )
    } else {
        (
            rounded_scaled_dimension(width, height, long_edge),
            long_edge,
        )
    };
    if (working_width, working_height) == (width, height) {
        return source.clone();
    }
    image::imageops::resize(
        source,
        working_width,
        working_height,
        image::imageops::FilterType::Triangle,
    )
}

fn rounded_scaled_dimension(short: u32, long: u32, target_long: u32) -> u32 {
    let numerator = u64::from(short) * u64::from(target_long) + u64::from(long) / 2;
    u32::try_from(numerator / u64::from(long))
        .expect("scaled image dimension fits in u32")
        .max(1)
}

/// Returns a deterministic camera-roll angle for every training source.
///
/// Applying roll to only the renderer-backed classes makes camera orientation
/// an avoidable synthetic/real cue. Reflection fill is source-derived, so the
/// same transform is safe for presence-only photographs and synthetic images.
/// The caller disables all augmentation for evaluation and the overfit gate.
pub(super) fn training_roll_radians(key: &str, epoch: usize, training: bool) -> Option<f32> {
    if !training {
        return None;
    }
    let hash = sample_hash(key, epoch, ROLL_HASH_SALT);
    let unit = (hash & 0x00ff_ffff) as f32 / 0x00ff_ffff_u64 as f32;
    Some((unit * 2.0 - 1.0) * MAX_ROLL_DEGREES.to_radians())
}

/// Rotates an RGB image around its centre without introducing flat-colour
/// triangles. Samples beyond an edge are reflected back into the source.
#[must_use]
pub(super) fn rotate_rgb_reflect(source: &RgbImage, radians: f32) -> RgbImage {
    let (width, height) = source.dimensions();
    if radians.abs() <= f32::EPSILON || width == 0 || height == 0 {
        return source.clone();
    }
    let center_x = width as f32 * 0.5;
    let center_y = height as f32 * 0.5;
    let (sin, cos) = radians.sin_cos();
    RgbImage::from_fn(width, height, |x, y| {
        // Work in pixel-edge coordinates. A raster sample is at x + 0.5,
        // which is the same convention used to rotate normalized labels.
        let destination_x = x as f32 + 0.5 - center_x;
        let destination_y = y as f32 + 0.5 - center_y;
        let source_x = center_x + cos * destination_x + sin * destination_y - 0.5;
        let source_y = center_y - sin * destination_x + cos * destination_y - 0.5;
        bilinear_reflect(source, source_x, source_y)
    })
}

/// Clones a frame record and applies the same source-pixel roll to all twelve
/// projected structural labels. Points crossing the actual image boundary are
/// marked truncated, which makes the target builder select the offscreen token.
#[must_use]
pub(super) fn rotate_frame_keypoints(
    source: &FrameRecord,
    width: u32,
    height: u32,
    radians: f32,
) -> FrameRecord {
    let mut rotated = source.clone();
    rotate_keypoint_labels(&mut rotated.labels.keypoints, width, height, radians);
    rotated
}

fn rotate_keypoint_labels(keypoints: &mut [KeypointLabel], width: u32, height: u32, radians: f32) {
    for keypoint in keypoints {
        let Some(point) = keypoint.image_position else {
            continue;
        };
        let mapped = rotate_normalized_point([point.x, point.y], width, height, radians);
        keypoint.image_position = Some(synth_data::Vec2 {
            x: mapped[0],
            y: mapped[1],
        });
        if matches!(
            keypoint.visibility,
            Visibility::Visible | Visibility::Occluded
        ) && !inside_source(mapped)
        {
            keypoint.visibility = Visibility::Truncated;
        }
        // A point which was already truncated remains offscreen even if the
        // affine projection enters the output rectangle: rotating a rendered
        // crop cannot reveal source pixels that were never captured.
    }
}

fn rotate_normalized_point(point: [f32; 2], width: u32, height: u32, radians: f32) -> [f32; 2] {
    let width = width as f32;
    let height = height as f32;
    let center_x = width * 0.5;
    let center_y = height * 0.5;
    let x = point[0] * width - center_x;
    let y = point[1] * height - center_y;
    let (sin, cos) = radians.sin_cos();
    [
        (center_x + cos * x - sin * y) / width,
        (center_y + sin * x + cos * y) / height,
    ]
}

fn inside_source(point: [f32; 2]) -> bool {
    point[0].is_finite()
        && point[1].is_finite()
        && (0.0..1.0).contains(&point[0])
        && (0.0..1.0).contains(&point[1])
}

fn bilinear_reflect(image: &RgbImage, x: f32, y: f32) -> Rgb<u8> {
    let x = reflect_coordinate(x, image.width());
    let y = reflect_coordinate(y, image.height());
    let x0 = x.floor() as u32;
    let y0 = y.floor() as u32;
    let x1 = (x0 + 1).min(image.width() - 1);
    let y1 = (y0 + 1).min(image.height() - 1);
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let top_left = image.get_pixel(x0, y0);
    let top_right = image.get_pixel(x1, y0);
    let bottom_left = image.get_pixel(x0, y1);
    let bottom_right = image.get_pixel(x1, y1);
    Rgb(std::array::from_fn(|channel| {
        let top = f32::from(top_left[channel]) * (1.0 - tx) + f32::from(top_right[channel]) * tx;
        let bottom =
            f32::from(bottom_left[channel]) * (1.0 - tx) + f32::from(bottom_right[channel]) * tx;
        (top * (1.0 - ty) + bottom * ty).round().clamp(0.0, 255.0) as u8
    }))
}

fn reflect_coordinate(value: f32, length: u32) -> f32 {
    if length <= 1 {
        return 0.0;
    }
    let maximum = length as f32 - 1.0;
    let period = maximum * 2.0;
    let wrapped = value.rem_euclid(period);
    if wrapped <= maximum {
        wrapped
    } else {
        period - wrapped
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use roof_geometry::KeypointId;
    use roof_model::{LetterboxTransform, prepare_rgb8_sized};
    use roof_training::{OFFSCREEN_INDEX, build_keypoint_target};
    use synth_data::{
        AssetRef, CameraIntrinsics, CameraModel, DatasetSplit, DistortionModel, FrameAssets,
        FrameIdentity, ImageTransform, LocatorLabel, RigidTransform, TargetKind, Vec2, Vec3,
    };

    #[test]
    fn roll_is_bounded_deterministic_and_training_only() {
        let first = training_roll_radians("roof-17", 3, true).unwrap();
        let repeated = training_roll_radians("roof-17", 3, true).unwrap();
        assert_eq!(first, repeated);
        assert!(first.abs() <= MAX_ROLL_DEGREES.to_radians());
        assert!(training_roll_radians("real-roof-17", 3, true).is_some());
        assert_eq!(training_roll_radians("building-2", 3, false), None);
    }

    #[test]
    fn working_raster_has_an_exact_long_edge_and_preserves_aspect() {
        let landscape = RgbImage::new(4_032, 3_024);
        let landscape = resize_to_working_raster(&landscape, 256);
        assert_eq!(landscape.dimensions(), (256, 192));

        let portrait = RgbImage::new(333, 1_000);
        let portrait = resize_to_working_raster(&portrait, 256);
        assert_eq!(portrait.dimensions(), (85, 256));
    }

    #[test]
    fn pixel_and_normalized_point_share_the_same_rotation() {
        let mut image = RgbImage::from_pixel(5, 5, Rgb([0, 0, 0]));
        image.put_pixel(4, 2, Rgb([255, 0, 0]));
        let rotated = rotate_rgb_reflect(&image, 90.0_f32.to_radians());
        assert_eq!(*rotated.get_pixel(2, 4), Rgb([255, 0, 0]));

        // The centre of source pixel (4, 2) uses normalized edge coordinates.
        let point = rotate_normalized_point([4.5 / 5.0, 2.5 / 5.0], 5, 5, 90.0_f32.to_radians());
        assert!((point[0] - 2.5 / 5.0).abs() < 1.0e-6);
        assert!((point[1] - 4.5 / 5.0).abs() < 1.0e-6);
    }

    #[test]
    fn downscaled_roll_keeps_pixels_and_keypoints_aligned_through_letterbox() {
        let source_width = 512;
        let source_height = 256;
        let point = [0.75, 0.375];
        let mut source = RgbImage::from_pixel(source_width, source_height, Rgb([0, 0, 0]));
        // A symmetric 16-pixel patch has its centre at the normalized point.
        for y in 88..104 {
            for x in 376..392 {
                source.put_pixel(x, y, Rgb([255, 0, 0]));
            }
        }
        let working = resize_to_working_raster(&source, 256);
        assert_eq!(working.dimensions(), (256, 128));
        let radians = 8.0_f32.to_radians();
        let pixels = rotate_rgb_reflect(&working, radians);

        let mut frame = FrameRecord::new(
            FrameIdentity::new("aligned", "aligned", 0, 0),
            DatasetSplit::Train,
            CameraModel {
                intrinsics: CameraIntrinsics {
                    width: source_width,
                    height: source_height,
                    fx: 300.0,
                    fy: 300.0,
                    cx: source_width as f32 * 0.5,
                    cy: source_height as f32 * 0.5,
                    skew: 0.0,
                },
                distortion: DistortionModel::None,
                world_from_camera: RigidTransform::IDENTITY,
                output_from_sensor: ImageTransform::IDENTITY,
            },
            LocatorLabel {
                target_kind: TargetKind::Target,
                bounding_box: None,
                amodal_bounding_box: None,
                visible_fraction: 1.0,
                occluded_fraction: 0.0,
                truncated: false,
            },
            FrameAssets {
                rgb: AssetRef::new("aligned.png", "image/png", "png"),
                surface_normals: None,
                motion_vectors: None,
            },
        );
        frame.labels.keypoints = KeypointId::ALL
            .iter()
            .map(|id| KeypointLabel {
                class_id: 1,
                instance_id: id.as_u32() as u16,
                roof_position: Vec3::default(),
                image_position: Some(Vec2::new(point[0], point[1])),
                visibility: Visibility::Visible,
            })
            .collect();
        let rotated = rotate_frame_keypoints(&frame, working.width(), working.height(), radians);
        let prepared =
            prepare_rgb8_sized(working.width(), working.height(), pixels.as_raw(), 256).unwrap();
        let target = build_keypoint_target(&rotated, prepared.transform).unwrap();

        // Compare the label with the red patch's centre of mass inside the
        // true image rectangle. Letterbox padding is deliberately excluded.
        let black = (0.0 - 0.485) / 0.229;
        let content_top = prepared.transform.pad_y.ceil() as usize;
        let content_bottom = (prepared.transform.pad_y + working.height() as f32).floor() as usize;
        let mut weighted = [0.0_f32; 2];
        let mut total = 0.0_f32;
        for y in content_top..content_bottom {
            for x in 0..256 {
                let weight = (prepared.chw[y * 256 + x] - black).max(0.0);
                weighted[0] += weight * (x as f32 + 0.5);
                weighted[1] += weight * (y as f32 + 0.5);
                total += weight;
            }
        }
        let centroid = [weighted[0] / total, weighted[1] / total];
        let labelled = [
            target.positions[0][0] * 256.0,
            target.positions[0][1] * 256.0,
        ];
        assert!((centroid[0] - labelled[0]).abs() < 0.75);
        assert!((centroid[1] - labelled[1]).abs() < 0.75);
    }

    #[test]
    fn reflection_fill_never_introduces_black_corners() {
        let image = RgbImage::from_pixel(11, 7, Rgb([23, 47, 89]));
        let rotated = rotate_rgb_reflect(&image, MAX_ROLL_DEGREES.to_radians());
        assert!(rotated.pixels().all(|pixel| *pixel == Rgb([23, 47, 89])));
    }

    #[test]
    fn aspect_aware_roll_can_move_a_point_offscreen() {
        let rotated = rotate_normalized_point([0.95, 0.5], 200, 100, 8.0_f32.to_radians());
        assert!(rotated[1] > 0.5);
        assert!(inside_source(rotated));

        let rotated = rotate_normalized_point([0.95, 0.5], 200, 100, 90.0_f32.to_radians());
        assert!(rotated[1] > 1.0);
        assert!(!inside_source(rotated));
    }

    #[test]
    fn point_crossing_source_boundary_becomes_truncated() {
        let mut labels = [KeypointLabel {
            class_id: 1,
            instance_id: 1,
            roof_position: synth_data::Vec3::default(),
            image_position: Some(synth_data::Vec2::new(0.95, 0.5)),
            visibility: Visibility::Visible,
        }];
        rotate_keypoint_labels(&mut labels, 200, 100, 90.0_f32.to_radians());
        assert_eq!(labels[0].visibility, Visibility::Truncated);
        assert!(labels[0].image_position.unwrap().y > 1.0);
    }

    #[test]
    fn crossed_boundary_builds_an_offscreen_training_token() {
        let width = 200;
        let height = 100;
        let mut frame = FrameRecord::new(
            FrameIdentity::new("roll-target", "roll-target", 0, 0),
            DatasetSplit::Train,
            CameraModel {
                intrinsics: CameraIntrinsics {
                    width,
                    height,
                    fx: 100.0,
                    fy: 100.0,
                    cx: 100.0,
                    cy: 50.0,
                    skew: 0.0,
                },
                distortion: DistortionModel::None,
                world_from_camera: RigidTransform::IDENTITY,
                output_from_sensor: ImageTransform::IDENTITY,
            },
            LocatorLabel {
                target_kind: TargetKind::Target,
                bounding_box: None,
                amodal_bounding_box: None,
                visible_fraction: 1.0,
                occluded_fraction: 0.0,
                truncated: false,
            },
            FrameAssets {
                rgb: AssetRef::new("roll-target.jpg", "image/jpeg", "jpeg"),
                surface_normals: None,
                motion_vectors: None,
            },
        );
        frame.labels.keypoints = KeypointId::ALL
            .iter()
            .enumerate()
            .map(|(index, id)| KeypointLabel {
                class_id: 1,
                instance_id: id.as_u32() as u16,
                roof_position: Vec3::default(),
                image_position: Some(if index == 0 {
                    Vec2::new(0.95, 0.5)
                } else {
                    Vec2::new(0.5, 0.5)
                }),
                visibility: Visibility::Visible,
            })
            .collect();

        let rotated = rotate_frame_keypoints(&frame, width, height, 90.0_f32.to_radians());
        let target = build_keypoint_target(
            &rotated,
            LetterboxTransform::for_input_size(width, height, 256),
        )
        .unwrap();
        assert!(!target.in_frame[0]);
        assert_eq!(target.distributions[OFFSCREEN_INDEX], 1.0);
        assert!(target.in_frame[1..].iter().all(|value| *value));
    }
}
