//! Cross-module reproducibility and validation contract tests.

use std::collections::{BTreeMap, BTreeSet};

use synth_data::{
    AssetRef, DatasetManifest, DatasetValidator, DenseLabelRefs, EdgeLabel, EdgeVisibility,
    FloatRange, FrameAssets, FrameIdentity, FrameRecord, FramingIntent, GeneratorConfig,
    GeneratorDescriptor, KeypointLabel, LabelClass, LocatorLabel, NormalizedBoundingBox,
    OccluderChoice, OccluderKind, OrdinaryRoofFamily, RigidTransform, SampledBuilding,
    SampledOrdinaryRoof, SampledRoof, SequencePlan, SequenceRequest, SequenceSampler, SourceAsset,
    SourceAssetKind, SplitPolicy, StructuralLabels, TargetKind, U32Range, Validate, Vec2, Vec3,
    Visibility, ZoomBehavior,
};

fn asset(path: impl Into<String>, media_type: &str, encoding: &str) -> AssetRef {
    AssetRef {
        path: path.into(),
        media_type: media_type.to_owned(),
        encoding: encoding.to_owned(),
        content_hash: Some("blake3:0123456789abcdef".to_owned()),
    }
}

fn sampler() -> SequenceSampler {
    SequenceSampler::new(GeneratorConfig::default()).expect("default config should be valid")
}

fn manifest(sampler: &SequenceSampler) -> DatasetManifest {
    let generator = GeneratorDescriptor::chacha20(
        "roof-synth",
        env!("CARGO_PKG_VERSION"),
        sampler.config_fingerprint(),
    );
    let mut manifest = DatasetManifest::new("roof-test-v001", generator, 0);
    manifest.labels.keypoints = vec![LabelClass {
        id: 1,
        name: "crown_top_corner".to_owned(),
    }];
    manifest.labels.edges = vec![LabelClass {
        id: 2,
        name: "crown_top_edge".to_owned(),
    }];
    manifest.labels.parts = vec![LabelClass {
        id: 1,
        name: "roof".to_owned(),
    }];
    manifest.labels.faces = vec![LabelClass {
        id: 1,
        name: "front_lower".to_owned(),
    }];
    manifest
}

fn target_frames(plan: &SequencePlan, split: synth_data::DatasetSplit) -> Vec<FrameRecord> {
    plan.frames
        .iter()
        .map(|frame| {
            let key = plan
                .frame_key(frame.frame_index)
                .expect("sampled frame index should be in range");
            let mut record = FrameRecord::new(
                FrameIdentity::new(
                    &key,
                    &plan.sequence_id,
                    frame.frame_index,
                    frame.timestamp_ns,
                ),
                split,
                frame.camera,
                LocatorLabel {
                    target_kind: TargetKind::Target,
                    bounding_box: Some(NormalizedBoundingBox {
                        min: Vec2::new(0.2, 0.25),
                        max: Vec2::new(0.8, 0.7),
                    }),
                    amodal_bounding_box: Some(NormalizedBoundingBox {
                        min: Vec2::new(0.15, 0.2),
                        max: Vec2::new(0.85, 0.75),
                    }),
                    visible_fraction: 0.8,
                    occluded_fraction: 0.1,
                    truncated: false,
                },
                FrameAssets {
                    rgb: asset(format!("{key}.rgb.jpg"), "image/jpeg", "jpeg"),
                    surface_normals: None,
                    motion_vectors: None,
                },
            );
            record.roof = plan.roof_instance();
            record.labels = StructuralLabels {
                keypoints: vec![KeypointLabel {
                    class_id: 1,
                    instance_id: 10,
                    roof_position: Vec3::new(
                        0.0,
                        plan.scene.roof.lower_rise_m + plan.scene.roof.upper_rise_m,
                        0.0,
                    ),
                    image_position: Some(Vec2::new(0.5, 0.4)),
                    visibility: Visibility::Visible,
                }],
                edges: vec![EdgeLabel {
                    class_id: 2,
                    instance_id: 20,
                    polyline: vec![Vec2::new(0.4, 0.4), Vec2::new(0.6, 0.4)],
                    visibility: EdgeVisibility::Visible,
                }],
                dense: DenseLabelRefs {
                    roof_mask: Some(asset(format!("{key}.roof-mask.png"), "image/png", "uint8")),
                    amodal_roof_mask: Some(asset(
                        format!("{key}.amodal-roof-mask.png"),
                        "image/png",
                        "uint8",
                    )),
                    part_mask: Some(asset(format!("{key}.parts.png"), "image/png", "uint16")),
                    face_id_map: Some(asset(format!("{key}.faces.png"), "image/png", "uint16")),
                    face_coordinates: Some(asset(
                        format!("{key}.face-coordinates.bin.zst"),
                        "application/octet-stream",
                        "rg16float-zstd",
                    )),
                },
            };
            record
        })
        .collect()
}

fn dot(left: Vec3, right: Vec3) -> f32 {
    left.x * right.x + left.y * right.y + left.z * right.z
}

fn roof_control_points(building: SampledBuilding, roof: SampledRoof) -> [Vec3; 12] {
    let rectangle = |width: f32, depth: f32, y: f32| {
        [
            Vec3::new(-width * 0.5, y, -depth * 0.5),
            Vec3::new(width * 0.5, y, -depth * 0.5),
            Vec3::new(width * 0.5, y, depth * 0.5),
            Vec3::new(-width * 0.5, y, depth * 0.5),
        ]
    };
    let eave = rectangle(roof.eave_width_m, roof.eave_depth_m, building.wall_height_m);
    let shoulder = rectangle(
        roof.shoulder_width_m,
        roof.shoulder_depth_m,
        building.wall_height_m + roof.lower_rise_m,
    );
    let crown = rectangle(
        roof.crown_top_width_m,
        roof.crown_top_depth_m,
        building.wall_height_m + roof.lower_rise_m + roof.upper_rise_m,
    );
    [
        eave[0],
        eave[1],
        eave[2],
        eave[3],
        shoulder[0],
        shoulder[1],
        shoulder[2],
        shoulder[3],
        crown[0],
        crown[1],
        crown[2],
        crown[3],
    ]
}

fn ordinary_roof_control_points(building: SampledBuilding, roof: SampledOrdinaryRoof) -> Vec<Vec3> {
    let x = roof.eave_width_m * 0.5;
    let z = roof.eave_depth_m * 0.5;
    let y = building.wall_height_m;
    let top = y + roof.rise_m;
    let mut points = vec![
        Vec3::new(-x, y, -z),
        Vec3::new(x, y, -z),
        Vec3::new(x, y, z),
        Vec3::new(-x, y, z),
    ];
    match roof.family {
        OrdinaryRoofFamily::Flat => points.extend([
            Vec3::new(-x, top, -z),
            Vec3::new(x, top, -z),
            Vec3::new(x, top, z),
            Vec3::new(-x, top, z),
        ]),
        OrdinaryRoofFamily::Gable => {
            points.extend([Vec3::new(-x, top, 0.0), Vec3::new(x, top, 0.0)])
        }
        OrdinaryRoofFamily::Hip => {
            let ridge_x = x * roof.ridge_length_fraction;
            points.extend([Vec3::new(-ridge_x, top, 0.0), Vec3::new(ridge_x, top, 0.0)]);
        }
        OrdinaryRoofFamily::Shed => points.extend([Vec3::new(-x, top, z), Vec3::new(x, top, z)]),
        OrdinaryRoofFamily::Mansard => {
            let inset_x = x * roof.inset_fraction;
            let inset_z = z * roof.inset_fraction;
            points.extend([
                Vec3::new(-inset_x, top, -inset_z),
                Vec3::new(inset_x, top, -inset_z),
                Vec3::new(inset_x, top, inset_z),
                Vec3::new(-inset_x, top, inset_z),
            ]);
        }
        OrdinaryRoofFamily::Pyramid => points.push(Vec3::new(0.0, top, 0.0)),
        OrdinaryRoofFamily::Cupola => {
            let cupola_x = x * roof.inset_fraction.clamp(0.12, 0.32);
            let cupola_z = z * roof.inset_fraction.clamp(0.12, 0.32);
            let body_top = top + roof.cap_height_m * 0.62;
            let cap_top = top + roof.cap_height_m;
            points.extend([
                Vec3::new(0.0, top, 0.0),
                Vec3::new(-cupola_x, top, -cupola_z),
                Vec3::new(cupola_x, top, -cupola_z),
                Vec3::new(cupola_x, top, cupola_z),
                Vec3::new(-cupola_x, top, cupola_z),
                Vec3::new(-cupola_x, body_top, -cupola_z),
                Vec3::new(cupola_x, body_top, -cupola_z),
                Vec3::new(cupola_x, body_top, cupola_z),
                Vec3::new(-cupola_x, body_top, cupola_z),
                Vec3::new(0.0, cap_top, 0.0),
            ]);
        }
    }
    points
}

fn projected_roof_bounds(plan: &SequencePlan, frame_index: usize) -> Option<(f32, f32, f32, f32)> {
    let frame = &plan.frames[frame_index];
    let camera = frame.camera;
    let [x, y, z, w] = camera.world_from_camera.rotation_xyzw;
    let right = Vec3::new(
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + z * w),
        2.0 * (x * z - y * w),
    );
    let up = Vec3::new(
        2.0 * (x * y - z * w),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + x * w),
    );
    let back = Vec3::new(
        2.0 * (x * z + y * w),
        2.0 * (y * z - x * w),
        1.0 - 2.0 * (x * x + y * y),
    );
    let position = camera.world_from_camera.translation;
    let mut bounds = (
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::INFINITY,
        f32::NEG_INFINITY,
    );
    let points = plan.scene.ordinary_roof.map_or_else(
        || roof_control_points(plan.scene.building, plan.scene.roof).to_vec(),
        |roof| ordinary_roof_control_points(plan.scene.building, roof),
    );
    for point in points {
        let relative = Vec3::new(
            point.x - position.x,
            point.y - position.y,
            point.z - position.z,
        );
        let depth = -dot(relative, back);
        if depth <= 0.0 {
            return None;
        }
        let image_x = camera.intrinsics.fx * dot(relative, right) / depth + camera.intrinsics.cx;
        let image_y = camera.intrinsics.cy - camera.intrinsics.fy * dot(relative, up) / depth;
        bounds.0 = bounds.0.min(image_x);
        bounds.1 = bounds.1.max(image_x);
        bounds.2 = bounds.2.min(image_y);
        bounds.3 = bounds.3.max(image_y);
    }
    Some(bounds)
}

#[test]
fn deterministic_sampling_is_byte_identical() {
    let sampler = sampler();
    let request = SequenceRequest::procedural("classic_two_stage", 42, TargetKind::Target);
    let first = sampler
        .sample(request.clone())
        .expect("sampling should succeed");
    let second = sampler.sample(request).expect("sampling should succeed");

    assert_eq!(first, second);
    assert_eq!(
        serde_json::to_vec(&first).unwrap(),
        serde_json::to_vec(&second).unwrap()
    );
    assert!(first.scene.roof.crown_top_width_m < first.scene.roof.shoulder_width_m);
    assert!(first.scene.roof.crown_top_depth_m < first.scene.roof.shoulder_depth_m);
    assert_eq!(first.sequence_id, "seq-fa9539a22cfcb329");
    assert_eq!(first.config_fingerprint, "stable64:402bbb3703bce52f");
    assert_eq!(first.scene.roof.eave_width_m.to_bits(), 1_103_896_300);
}

#[test]
fn independent_substreams_localize_configuration_changes() {
    let default_sampler = sampler();
    let mut changed_config = GeneratorConfig::default();
    changed_config.occluders.count = U32Range::new(0, 0);
    let changed_sampler = SequenceSampler::new(changed_config).unwrap();
    let request = SequenceRequest::procedural("classic_two_stage", 777, TargetKind::Target);
    let default_plan = default_sampler.sample(request.clone()).unwrap();
    let changed_plan = changed_sampler.sample(request).unwrap();

    assert_eq!(default_plan.scene.building, changed_plan.scene.building);
    assert_eq!(default_plan.scene.roof, changed_plan.scene.roof);
    assert_eq!(
        default_plan.scene.roof_material,
        changed_plan.scene.roof_material
    );
    assert_eq!(
        default_plan.scene.wall_material,
        changed_plan.scene.wall_material
    );
    assert_eq!(default_plan.scene.lighting, changed_plan.scene.lighting);
    assert_eq!(default_plan.frames, changed_plan.frames);
    assert_ne!(default_plan.scene.occluders, changed_plan.scene.occluders);
    assert_ne!(
        default_plan.config_fingerprint,
        changed_plan.config_fingerprint
    );
}

#[test]
fn records_round_trip_and_validate_as_a_sequence() {
    let sampler = sampler();
    let manifest = manifest(&sampler);
    assert!(manifest.validate().is_valid());

    let plan = sampler
        .sample(SequenceRequest::procedural(
            "classic_two_stage",
            9_821,
            TargetKind::Target,
        ))
        .unwrap();
    let split = plan.split(&manifest.split_policy).unwrap();
    let frames = target_frames(&plan, split);
    let camera_motion = plan.camera_motion;
    let sequence = plan.into_record(&manifest.split_policy).unwrap();
    assert_eq!(sequence.camera_motion, camera_motion);
    let validator = DatasetValidator::new(&manifest);
    let report = validator.validate_sequence(&sequence, &frames);
    assert!(report.is_valid(), "{report}");
    assert_eq!(report.warning_count(), 0);

    let encoded = serde_json::to_vec(&sequence).unwrap();
    let decoded: synth_data::SequenceRecord = serde_json::from_slice(&encoded).unwrap();
    assert_eq!(decoded, sequence);
}

#[test]
fn sequence_source_assets_must_resolve_in_the_manifest() {
    let sampler = sampler();
    let mut manifest = manifest(&sampler);
    let mut request = SequenceRequest::procedural("classic_two_stage", 9_822, TargetKind::Target);
    request.source_asset_group = Some("site-textures-a".to_owned());
    let plan = sampler.sample(request).unwrap();
    let split = plan.split(&manifest.split_policy).unwrap();
    let frames = target_frames(&plan, split);
    let mut sequence = plan.into_record(&manifest.split_policy).unwrap();
    sequence.scene.composition.source_asset_ids = vec!["ground-asphalt-01".to_owned()];

    let unresolved = DatasetValidator::new(&manifest).validate_sequence(&sequence, &frames);
    assert!(
        unresolved
            .issues
            .iter()
            .any(|issue| issue.code == "unknown_source_asset_id")
    );

    manifest.source_assets.push(SourceAsset {
        id: "ground-asphalt-01".to_owned(),
        kind: SourceAssetKind::Texture,
        content_hash: "blake3:0123456789abcdef".to_owned(),
        license: "internal-training".to_owned(),
        split_group: Some("wrong-group".to_owned()),
    });
    let mismatched = DatasetValidator::new(&manifest).validate_sequence(&sequence, &frames);
    assert!(
        mismatched
            .issues
            .iter()
            .any(|issue| issue.code == "source_asset_group_mismatch")
    );

    manifest.source_assets[0].split_group = Some("site-textures-a".to_owned());
    let resolved = DatasetValidator::new(&manifest).validate_sequence(&sequence, &frames);
    assert!(resolved.is_valid(), "{resolved}");
}

#[test]
fn legacy_generator_configs_fill_new_realism_defaults() {
    let expected = GeneratorConfig::default();
    let mut encoded = serde_json::to_value(&expected).unwrap();
    let root = encoded.as_object_mut().unwrap();
    root.remove("composition");

    let camera = root.get_mut("camera").unwrap().as_object_mut().unwrap();
    for field in [
        "orbit_weight",
        "lateral_walk_weight",
        "approach_arc_weight",
        "corner_reveal_weight",
        "zoom_probability",
        "zoom_ratio",
        "partial_crop_probability",
        "target_width_fraction",
        "distant_target_width_fraction",
        "close_target_width_fraction",
        "distant_view_weight",
        "normal_view_weight",
        "close_view_weight",
        "partial_target_width_fraction",
        "framing_offset_fraction",
        "handheld_sway_m",
    ] {
        camera.remove(field);
    }

    let occluders = root.get_mut("occluders").unwrap().as_object_mut().unwrap();
    for field in [
        "foreground_probability",
        "foreground_depth_fraction",
        "foreground_lateral_offset_m",
    ] {
        occluders.remove(field);
    }

    let decoded: GeneratorConfig = serde_json::from_value(encoded).unwrap();
    assert_eq!(decoded, expected);
    assert!(decoded.validate().is_valid());
}

#[test]
fn validator_reports_multiple_corruptions() {
    let sampler = sampler();
    let manifest = manifest(&sampler);
    let plan = sampler
        .sample(SequenceRequest::procedural(
            "classic_two_stage",
            12,
            TargetKind::Target,
        ))
        .unwrap();
    let split = plan.split(&manifest.split_policy).unwrap();
    let mut frame = target_frames(&plan, split).remove(0);
    frame.camera.world_from_camera = RigidTransform {
        translation: Vec3::new(f32::NAN, 0.0, 0.0),
        rotation_xyzw: [0.0, 0.0, 0.0, 1.0],
    };
    frame.assets.rgb.path = "../escape.jpg".to_owned();
    frame.labels.keypoints[0].image_position = Some(Vec2::new(1.5, 0.5));
    frame.labels.dense.face_id_map = None;

    let report = DatasetValidator::new(&manifest).validate_frame(&frame);
    let codes = report
        .issues
        .iter()
        .map(|issue| issue.code.as_str())
        .collect::<Vec<_>>();
    assert!(!report.is_valid());
    assert!(codes.contains(&"invalid_camera_transform"));
    assert!(codes.contains(&"unsafe_asset_path"));
    assert!(codes.contains(&"invalid_visible_projection"));
    assert!(codes.contains(&"orphan_face_coordinates"));
}

#[test]
fn validator_rejects_any_target_geometry_on_negative_frames() {
    let sampler = sampler();
    let manifest = manifest(&sampler);
    let plan = sampler
        .sample(SequenceRequest::procedural(
            "classic_two_stage",
            99,
            TargetKind::Target,
        ))
        .unwrap();
    let split = plan.split(&manifest.split_policy).unwrap();
    let mut frame = target_frames(&plan, split).remove(0);
    frame.locator = LocatorLabel {
        target_kind: TargetKind::Negative,
        bounding_box: None,
        amodal_bounding_box: None,
        visible_fraction: 0.0,
        occluded_fraction: 0.0,
        truncated: false,
    };

    let report = DatasetValidator::new(&manifest).validate_frame(&frame);
    assert!(!report.is_valid());
    assert!(
        report
            .issues
            .iter()
            .any(|issue| issue.code == "negative_has_target_geometry")
    );
}

#[test]
fn invalid_config_is_rejected_before_rng_is_touched() {
    let mut config = GeneratorConfig::default();
    config.camera.distance_m.min = f32::NAN;
    let error = SequenceSampler::new(config).expect_err("NaN config must fail");
    assert!(
        error
            .to_string()
            .contains("invalid generator configuration")
    );
}

#[test]
fn split_policy_is_independent_of_frame_count() {
    let policy = SplitPolicy::default();
    let request = SequenceRequest::procedural("classic_two_stage", 55, TargetKind::Target);
    let original = sampler().sample(request.clone()).unwrap();
    let mut config = GeneratorConfig::default();
    config.sequence.frame_count = 3;
    let shorter = SequenceSampler::new(config)
        .unwrap()
        .sample(request)
        .unwrap();
    assert_eq!(original.split(&policy), shorter.split(&policy));
}

#[test]
fn sequence_ids_cover_target_intent_and_generator_configuration() {
    let default_sampler = sampler();
    let target = default_sampler
        .sample(SequenceRequest::procedural(
            "classic_two_stage",
            55,
            TargetKind::Target,
        ))
        .unwrap();
    let near_miss = default_sampler
        .sample(SequenceRequest::procedural(
            "classic_two_stage",
            55,
            TargetKind::NearMiss,
        ))
        .unwrap();
    let negative = default_sampler
        .sample(SequenceRequest::procedural(
            "classic_two_stage",
            55,
            TargetKind::Negative,
        ))
        .unwrap();
    let mut changed_config = GeneratorConfig::default();
    changed_config.sequence.frame_interval_ms += 1;
    let changed = SequenceSampler::new(changed_config)
        .unwrap()
        .sample(SequenceRequest::procedural(
            "classic_two_stage",
            55,
            TargetKind::Target,
        ))
        .unwrap();

    let ids = BTreeSet::from([
        target.sequence_id,
        near_miss.sequence_id,
        negative.sequence_id,
        changed.sequence_id,
    ]);
    assert_eq!(ids.len(), 4);
}

#[test]
fn negative_requests_sample_only_explicit_ordinary_roofs() {
    let sampler = sampler();
    for family in OrdinaryRoofFamily::ALL {
        let plan = sampler
            .sample(SequenceRequest::procedural(
                format!("ordinary_{}", family.as_str()),
                10_000 + family as u64,
                TargetKind::Negative,
            ))
            .unwrap();

        assert_eq!(plan.scene.ordinary_roof.unwrap().family, family);
        assert!(plan.roof_instance().is_none());
        assert!(plan.validate().is_valid());
    }
    let target = sampler
        .sample(SequenceRequest::procedural(
            "classic_two_stage",
            22,
            TargetKind::Target,
        ))
        .unwrap();
    assert!(target.scene.ordinary_roof.is_none());
    assert!(target.roof_instance().is_some());
}

#[test]
fn paired_target_and_negative_share_scale_framing_and_occluder_distributions() {
    let sampler = sampler();
    let mut family_counts = BTreeMap::new();
    let mut target_kind_counts = BTreeMap::new();
    let mut negative_kind_counts = BTreeMap::new();

    for seed in 0..512_u64 {
        let family = OrdinaryRoofFamily::ALL[seed as usize % OrdinaryRoofFamily::ALL.len()];
        let target = sampler
            .sample(SequenceRequest::procedural(
                "classic_two_stage",
                seed,
                TargetKind::Target,
            ))
            .unwrap();
        let negative = sampler
            .sample(SequenceRequest::procedural(
                format!("ordinary_{}", family.as_str()),
                seed,
                TargetKind::Negative,
            ))
            .unwrap();
        let ordinary = negative.scene.ordinary_roof.unwrap();

        assert_eq!(
            target.scene.building.footprint_width_m,
            negative.scene.building.footprint_width_m
        );
        assert_eq!(
            target.scene.building.footprint_depth_m,
            negative.scene.building.footprint_depth_m
        );
        assert_eq!(
            target.scene.building.wall_height_m,
            negative.scene.building.wall_height_m
        );
        assert_eq!(target.scene.roof, negative.scene.roof);
        assert_eq!(ordinary.eave_width_m, target.scene.roof.eave_width_m);
        assert_eq!(ordinary.eave_depth_m, target.scene.roof.eave_depth_m);
        assert_eq!(
            (ordinary.eave_width_m - negative.scene.building.footprint_width_m) * 0.5,
            (target.scene.roof.eave_width_m - target.scene.building.footprint_width_m) * 0.5,
        );
        assert_eq!(
            negative.camera_motion.target_width_fraction_goal,
            target.camera_motion.target_width_fraction_goal,
        );
        assert_eq!(
            negative.camera_motion.apparent_scale,
            target.camera_motion.apparent_scale,
        );
        assert_eq!(
            negative.camera_motion.framing_intent,
            target.camera_motion.framing_intent,
        );

        let target_kinds = target
            .scene
            .occluders
            .iter()
            .map(|occluder| occluder.kind)
            .collect::<Vec<_>>();
        let negative_kinds = negative
            .scene
            .occluders
            .iter()
            .map(|occluder| occluder.kind)
            .collect::<Vec<_>>();
        assert_eq!(target_kinds, negative_kinds);
        assert!(
            target_kinds
                .iter()
                .all(|kind| *kind != OccluderKind::RooftopEquipment)
        );
        for kind in target_kinds {
            *target_kind_counts.entry(format!("{kind:?}")).or_insert(0) += 1;
        }
        for kind in negative_kinds {
            *negative_kind_counts.entry(format!("{kind:?}")).or_insert(0) += 1;
        }
        *family_counts.entry(ordinary.family).or_insert(0) += 1;
    }

    assert_eq!(target_kind_counts, negative_kind_counts);
    assert_eq!(family_counts.len(), OrdinaryRoofFamily::ALL.len());
    assert!(family_counts.values().all(|count| *count >= 73));
}

#[test]
fn rooftop_equipment_cannot_be_the_only_new_dataset_occluder() {
    let mut config = GeneratorConfig::default();
    config.occluders.choices = vec![OccluderChoice {
        kind: OccluderKind::RooftopEquipment,
        weight: 1,
    }];
    let error = SequenceSampler::new(config).unwrap_err().to_string();
    assert!(error.contains("non-rooftop occluder choice"));
}

#[test]
fn color_temperature_never_escapes_custom_weather_bounds() {
    let mut config = GeneratorConfig::default();
    for weather in &mut config.composition.weather.profiles {
        weather.color_temperature_k = FloatRange::new(1_800.0, 2_500.0);
    }
    let sampler = SequenceSampler::new(config).unwrap();
    let mut twilight_samples = 0;
    for seed in 0..512 {
        let plan = sampler
            .sample(SequenceRequest::procedural(
                "classic_two_stage",
                seed,
                TargetKind::Target,
            ))
            .unwrap();
        let environment = plan.scene.composition.environment.unwrap();
        assert!((1_800.0..=2_500.0).contains(&environment.color_temperature_k));
        if environment.day_phase == synth_data::DayPhase::Twilight {
            twilight_samples += 1;
        }
    }
    assert!(twilight_samples > 20);
}

#[test]
fn sampled_spatial_plans_are_clear_and_ground_contained() {
    let sampler = sampler();
    for seed in [2_u64, 13, 12_823] {
        let plan = sampler
            .sample(SequenceRequest::procedural(
                "classic_two_stage",
                seed,
                TargetKind::Target,
            ))
            .unwrap();
        let report = plan.validate();
        assert!(report.is_valid(), "known regression seed {seed}: {report}");
    }
    for seed in 0..5_000 {
        let plan = sampler
            .sample(SequenceRequest::procedural(
                "classic_two_stage",
                seed,
                TargetKind::Target,
            ))
            .unwrap();
        let report = plan.validate();
        assert!(report.is_valid(), "seed {seed}: {report}");
    }
}

#[test]
fn partial_framing_crops_the_declared_image_edge() {
    let sampler = sampler();
    let mut partial_sequences = 0_u32;
    let mut partial_frames = 0_u32;
    for seed in 0..5_000 {
        let plan = sampler
            .sample(SequenceRequest::procedural(
                "classic_two_stage",
                seed,
                TargetKind::Target,
            ))
            .unwrap();
        let intent = plan.camera_motion.framing_intent;
        if intent == FramingIntent::Centered {
            continue;
        }
        partial_sequences += 1;
        for (frame_index, frame) in plan.frames.iter().enumerate() {
            let bounds = projected_roof_bounds(&plan, frame_index).unwrap_or_else(|| {
                panic!(
                    "seed {seed}, frame {frame_index}: roof point behind camera; motion {:?}; camera {:?}",
                    plan.camera_motion, frame.camera,
                )
            });
            let cropped = match intent {
                FramingIntent::PartialLeft => bounds.0 < 0.0,
                FramingIntent::PartialRight => bounds.1 > frame.camera.intrinsics.width as f32,
                FramingIntent::PartialTop => bounds.2 < 0.0,
                FramingIntent::PartialBottom => bounds.3 > frame.camera.intrinsics.height as f32,
                FramingIntent::Centered => true,
            };
            assert!(
                cropped,
                "seed {seed}, frame {frame_index}, intent {intent:?}, bounds {bounds:?}"
            );
            partial_frames += 1;
        }
    }
    // The default deliberately-partial stratum is now 8%, down from 15%, to
    // keep useful offscreen examples without dominating target supervision.
    assert!(partial_sequences > 300);
    assert!(partial_sequences < 500);
    assert!(partial_frames > 7_200);
    assert!(partial_frames < 12_000);
}

#[test]
fn negative_partial_framing_uses_the_rendered_ordinary_roof() {
    let sampler = sampler();
    let mut checked = 0_u32;
    for seed in 0..1_000_u64 {
        let family = OrdinaryRoofFamily::ALL[seed as usize % OrdinaryRoofFamily::ALL.len()];
        let plan = sampler
            .sample(SequenceRequest::procedural(
                format!("ordinary_{}", family.as_str()),
                seed,
                TargetKind::Negative,
            ))
            .unwrap();
        let intent = plan.camera_motion.framing_intent;
        if intent == FramingIntent::Centered {
            continue;
        }
        for (frame_index, frame) in plan.frames.iter().enumerate() {
            let bounds = projected_roof_bounds(&plan, frame_index).unwrap();
            let cropped = match intent {
                FramingIntent::PartialLeft => bounds.0 < 0.0,
                FramingIntent::PartialRight => bounds.1 > frame.camera.intrinsics.width as f32,
                FramingIntent::PartialTop => bounds.2 < 0.0,
                FramingIntent::PartialBottom => bounds.3 > frame.camera.intrinsics.height as f32,
                FramingIntent::Centered => true,
            };
            assert!(
                cropped,
                "seed {seed}, family {family:?}, frame {frame_index}, bounds {bounds:?}"
            );
            checked += 1;
        }
    }
    assert!(checked > 1_200);
    assert!(checked < 2_600);
}

#[test]
fn realism_plans_stay_valid_and_camera_motion_is_coherent() {
    let sampler = sampler();
    for seed in 0..384 {
        let plan = sampler
            .sample(SequenceRequest::procedural(
                "classic_two_stage",
                seed,
                TargetKind::Target,
            ))
            .unwrap();
        let report = plan.validate();
        assert!(report.is_valid(), "seed {seed}: {report}");

        let fovs = plan
            .frames
            .iter()
            .map(|frame| {
                let intrinsics = frame.camera.intrinsics;
                2.0 * (intrinsics.width as f32 / (2.0 * intrinsics.fx)).atan()
            })
            .collect::<Vec<_>>();
        match plan.camera_motion.zoom_behavior {
            ZoomBehavior::Fixed => {
                assert!((fovs.first().unwrap() - fovs.last().unwrap()).abs() < 0.002);
            }
            ZoomBehavior::SmoothIn => {
                assert!(fovs.windows(2).all(|pair| pair[1] <= pair[0] + 1.0e-6));
            }
            ZoomBehavior::SmoothOut => {
                assert!(fovs.windows(2).all(|pair| pair[1] + 1.0e-6 >= pair[0]));
            }
        }
        let environment = plan.scene.composition.environment.unwrap();
        if environment.day_phase == synth_data::DayPhase::Night {
            assert_eq!(plan.scene.lighting.sun_intensity, 0.0);
            assert!(plan.scene.lighting.sun_elevation_degrees <= -6.0);
            assert!(environment.artificial_light_strength > 0.0);
        }
    }
}

#[test]
fn defaults_generate_all_required_categorical_regimes() {
    let sampler = sampler();
    let mut phases = BTreeSet::new();
    let mut domains = BTreeSet::new();
    let mut weather = BTreeSet::new();
    let mut roof_materials = BTreeSet::new();
    let mut signs = BTreeSet::new();
    let mut camera_paths = BTreeSet::new();
    let mut framing = BTreeSet::new();
    let mut apparent_scales = BTreeSet::new();
    let mut occluder_placements = BTreeSet::new();

    for seed in 0..2_000 {
        let plan = sampler
            .sample(SequenceRequest::procedural(
                "classic_two_stage",
                seed,
                TargetKind::Target,
            ))
            .unwrap();
        let environment = plan.scene.composition.environment.unwrap();
        phases.insert(format!("{:?}", environment.day_phase));
        domains.insert(format!("{:?}", environment.domain));
        weather.insert(format!("{:?}", environment.weather));
        roof_materials.insert(plan.scene.roof_material.id.clone());
        signs.insert(
            plan.scene
                .composition
                .signage
                .first()
                .map_or_else(|| "Blank".to_owned(), |sign| format!("{:?}", sign.kind)),
        );
        camera_paths.insert(format!("{:?}", plan.camera_motion.path_kind));
        framing.insert(format!("{:?}", plan.camera_motion.framing_intent));
        apparent_scales.insert(format!("{:?}", plan.camera_motion.apparent_scale));
        occluder_placements.extend(
            plan.scene
                .occluders
                .iter()
                .map(|item| format!("{:?}", item.placement)),
        );
    }

    assert_eq!(
        phases,
        BTreeSet::from(["Day".into(), "Night".into(), "Twilight".into()])
    );
    assert_eq!(
        domains,
        BTreeSet::from([
            "City".into(),
            "Remote".into(),
            "Roadside".into(),
            "Suburban".into(),
            "Urban".into(),
        ])
    );
    assert_eq!(weather.len(), 5);
    for required in [
        "original_red",
        "faded_red",
        "terracotta_orange",
        "weathered_tan_brown",
        "light_metal",
        "repainted_neutral",
        "repainted_dark",
        "repainted_blue",
        "repainted_green",
    ] {
        assert!(roof_materials.contains(required), "missing {required}");
    }
    assert_eq!(
        signs,
        BTreeSet::from([
            "Blank".into(),
            "PizzaHut".into(),
            "RebrandedTenant".into(),
            "RemovedGhost".into(),
        ])
    );
    assert_eq!(camera_paths.len(), 4);
    assert_eq!(framing.len(), 5);
    assert_eq!(
        apparent_scales,
        BTreeSet::from([
            "Close".into(),
            "Distant".into(),
            "Normal".into(),
            "Partial".into(),
        ])
    );
    assert_eq!(
        occluder_placements,
        BTreeSet::from(["Foreground".into(), "Site".into()])
    );
}

#[test]
fn building_extensions_are_class_independent_and_structurally_valid() {
    let mut config = GeneratorConfig::default();
    config.sequence.frame_count = 1;
    let sampler = SequenceSampler::new(config).unwrap();
    let mut plans_with_extensions = 0_u32;
    let mut kinds = BTreeSet::new();
    let mut roofs = BTreeSet::new();

    for seed in 0..1_000 {
        let target = sampler
            .sample(SequenceRequest::procedural(
                "classic_two_stage",
                seed,
                TargetKind::Target,
            ))
            .unwrap();
        let negative = sampler
            .sample(SequenceRequest::procedural(
                "ordinary_flat",
                seed,
                TargetKind::Negative,
            ))
            .unwrap();
        assert_eq!(
            target.scene.composition.building_extensions,
            negative.scene.composition.building_extensions,
            "target class changed extension sampling for seed {seed}"
        );
        assert!(target.validate().is_valid(), "target seed {seed}");
        assert!(negative.validate().is_valid(), "negative seed {seed}");

        if !target.scene.composition.building_extensions.is_empty() {
            plans_with_extensions += 1;
        }
        for extension in &target.scene.composition.building_extensions {
            kinds.insert(format!("{:?}", extension.kind));
            roofs.insert(format!("{:?}", extension.roof));
        }
    }

    assert!((400..=560).contains(&plans_with_extensions));
    assert_eq!(
        kinds,
        BTreeSet::from([
            "DiningWing".into(),
            "EntranceVestibule".into(),
            "ServiceAnnex".into(),
        ])
    );
    assert_eq!(roofs, BTreeSet::from(["Flat".into(), "Shed".into()]));
}

#[test]
fn site_domains_produce_distinct_correlated_density() {
    #[derive(Default)]
    struct Aggregate {
        samples: u32,
        backgrounds: u32,
        vegetation: u32,
        parking: u32,
    }

    let sampler = sampler();
    let mut by_domain = BTreeMap::<String, Aggregate>::new();
    for seed in 0..2_500 {
        let plan = sampler
            .sample(SequenceRequest::procedural(
                "classic_two_stage",
                seed,
                TargetKind::Target,
            ))
            .unwrap();
        let composition = &plan.scene.composition;
        let environment = composition.environment.unwrap();
        let infrastructure = composition.infrastructure.as_ref().unwrap();
        let aggregate = by_domain
            .entry(format!("{:?}", environment.domain))
            .or_default();
        aggregate.samples += 1;
        aggregate.backgrounds += composition.background_buildings.len() as u32;
        aggregate.vegetation += composition.vegetation.len() as u32;
        aggregate.parking += infrastructure.parking_bays;
    }
    let average = |domain: &str, field: fn(&Aggregate) -> u32| {
        let aggregate = &by_domain[domain];
        field(aggregate) as f32 / aggregate.samples as f32
    };
    assert!(
        average("City", |value| value.backgrounds)
            > average("Remote", |value| value.backgrounds) + 4.0
    );
    assert!(
        average("Remote", |value| value.vegetation)
            > average("City", |value| value.vegetation) + 7.0
    );
    assert!(
        average("Suburban", |value| value.parking) > average("City", |value| value.parking) + 10.0
    );
}
