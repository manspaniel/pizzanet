//! Compatibility tests for per-frame phone-camera appearance metadata.

use synth_data::{
    AssetRef, CameraIntrinsics, CameraModel, DatasetManifest, DatasetSplit, DatasetValidator,
    DayPhase, DistortionModel, FrameAppearance, FrameAssets, FrameIdentity, FrameRecord,
    GeneratorDescriptor, ImageTransform, LocatorLabel, PHOTOMETRIC_PROFILE_VERSION,
    PhotometricProfile, RigidTransform, TargetKind,
};

fn profile() -> PhotometricProfile {
    PhotometricProfile {
        schema_version: PHOTOMETRIC_PROFILE_VERSION,
        day_phase: DayPhase::Night,
        seed: 7,
        exposure_compensation_stops: 0.12,
        white_balance_rgb: [0.97, 1.0, 1.04],
        tint_green: 0.99,
        response_contrast: 1.01,
        response_shoulder: 0.2,
        vignette_strength: 0.1,
        vignette_power: 1.5,
        shot_noise_stddev: 0.01,
        read_noise_stddev: 0.004,
        sharpening_amount: 0.06,
        bloom_strength: 0.03,
        bloom_threshold: 0.72,
        jpeg_quality: 90,
        noise_seed: 11,
    }
}

fn frame() -> FrameRecord {
    FrameRecord::new(
        FrameIdentity::new("sample-000", "sequence-000", 0, 0),
        DatasetSplit::Train,
        CameraModel {
            intrinsics: CameraIntrinsics {
                width: 640,
                height: 480,
                fx: 500.0,
                fy: 500.0,
                cx: 320.0,
                cy: 240.0,
                skew: 0.0,
            },
            distortion: DistortionModel::None,
            world_from_camera: RigidTransform::IDENTITY,
            output_from_sensor: ImageTransform::IDENTITY,
        },
        LocatorLabel {
            target_kind: TargetKind::Negative,
            bounding_box: None,
            amodal_bounding_box: None,
            visible_fraction: 0.0,
            occluded_fraction: 0.0,
            truncated: false,
        },
        FrameAssets {
            rgb: AssetRef::new("sample-000.rgb.jpg", "image/jpeg", "jpeg"),
            surface_normals: None,
            motion_vectors: None,
        },
    )
}

fn manifest() -> DatasetManifest {
    DatasetManifest::new(
        "appearance-contract-test",
        GeneratorDescriptor::chacha20("test", "0", "stable64:test"),
        0,
    )
}

#[test]
fn frame_round_trip_persists_the_exact_profile() {
    let mut expected = frame();
    expected.appearance = FrameAppearance {
        photometric_profile: Some(profile()),
    };

    let value = serde_json::to_value(&expected).expect("frame should serialize");
    assert_eq!(value["appearance"]["photometric_profile"]["noise_seed"], 11);
    let decoded: FrameRecord = serde_json::from_value(value).expect("frame should deserialize");
    assert_eq!(decoded, expected);
}

#[test]
fn legacy_frame_without_appearance_receives_an_empty_default() {
    let mut current = frame();
    current.appearance.photometric_profile = Some(profile());
    let mut value = serde_json::to_value(current).expect("frame should serialize");
    value
        .as_object_mut()
        .expect("frame should be an object")
        .remove("appearance");

    let decoded: FrameRecord = serde_json::from_value(value).expect("legacy frame should load");
    assert_eq!(decoded.appearance, FrameAppearance::default());
}

#[test]
fn dataset_validator_rejects_an_invalid_embedded_profile() {
    let mut invalid = frame();
    let mut invalid_profile = profile();
    invalid_profile.read_noise_stddev = f32::INFINITY;
    invalid.appearance.photometric_profile = Some(invalid_profile);

    let report = DatasetValidator::new(&manifest()).validate_frame(&invalid);
    assert!(report.issues.iter().any(|issue| {
        issue.code == "invalid_photometric_profile"
            && issue.path == "appearance.photometric_profile.read_noise_stddev"
    }));
}
