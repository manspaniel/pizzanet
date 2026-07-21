//! Correlated roof-morphology sampling and compatibility contracts.

use std::collections::BTreeSet;

use synth_data::{
    GeneratorConfig, RoofMorphology, RoofMorphologyProfile, SequenceRequest, SequenceSampler,
    TargetKind, Validate,
};

fn assert_near(value: f32, expected: f32) {
    let tolerance = 2.0e-5 * expected.abs().max(1.0);
    assert!(
        (value - expected).abs() <= tolerance,
        "expected {value} to be within {tolerance} of {expected}"
    );
}

fn assert_profile_envelope(
    config: &GeneratorConfig,
    profile: RoofMorphologyProfile,
    plan: &synth_data::SequencePlan,
) {
    let building = plan.scene.building;
    let roof = plan.scene.roof;
    assert_eq!(roof.morphology, profile.morphology);

    let aspect = building.footprint_width_m / building.footprint_depth_m;
    assert!(profile.footprint_aspect_ratio.contains(aspect));
    assert!(
        config
            .scene
            .footprint_width_m
            .contains(building.footprint_width_m)
    );
    assert!(
        config
            .scene
            .footprint_depth_m
            .contains(building.footprint_depth_m)
    );

    let width_overhang = (roof.eave_width_m - building.footprint_width_m) * 0.5;
    let depth_overhang = (roof.eave_depth_m - building.footprint_depth_m) * 0.5;
    assert_near(width_overhang, depth_overhang);
    assert!(profile.overhang_m.contains(width_overhang));
    assert!(config.roof.overhang_m.contains(width_overhang));

    let shoulder_width_fraction = roof.shoulder_width_m / roof.eave_width_m;
    let shoulder_depth_fraction = roof.shoulder_depth_m / roof.eave_depth_m;
    let crown_width_fraction = roof.crown_top_width_m / roof.shoulder_width_m;
    let crown_depth_fraction = roof.crown_top_depth_m / roof.shoulder_depth_m;
    for (value, profile_range, global_range) in [
        (
            shoulder_width_fraction,
            profile.shoulder_width_fraction,
            config.roof.shoulder_width_fraction,
        ),
        (
            shoulder_depth_fraction,
            profile.shoulder_depth_fraction,
            config.roof.shoulder_depth_fraction,
        ),
        (
            crown_width_fraction,
            profile.crown_top_width_fraction,
            config.roof.crown_top_width_fraction,
        ),
        (
            crown_depth_fraction,
            profile.crown_top_depth_fraction,
            config.roof.crown_top_depth_fraction,
        ),
    ] {
        assert!(profile_range.contains(value));
        assert!(global_range.contains(value));
    }
    for (value, profile_range, global_range) in [
        (
            roof.lower_rise_m,
            profile.lower_rise_m,
            config.roof.lower_rise_m,
        ),
        (
            roof.upper_rise_m,
            profile.upper_rise_m,
            config.roof.upper_rise_m,
        ),
    ] {
        assert!(profile_range.contains(value));
        assert!(global_range.contains(value));
    }

    assert!(roof.crown_top_width_m < roof.shoulder_width_m);
    assert!(roof.shoulder_width_m < roof.eave_width_m);
    assert!(roof.crown_top_depth_m < roof.shoulder_depth_m);
    assert!(roof.shoulder_depth_m < roof.eave_depth_m);
}

#[test]
fn deterministic_sweep_covers_every_correlated_profile() {
    let mut config = GeneratorConfig::default();
    config.sequence.frame_count = 1;
    let sampler = SequenceSampler::new(config.clone()).unwrap();
    let mut observed = BTreeSet::new();

    for seed in 0..4_096 {
        let plan = sampler
            .sample(SequenceRequest::procedural(
                "classic_two_stage",
                seed,
                TargetKind::Target,
            ))
            .unwrap();
        let profile = config
            .roof
            .profiles
            .iter()
            .find(|profile| profile.morphology == plan.scene.roof.morphology)
            .copied()
            .unwrap();
        assert_profile_envelope(&config, profile, &plan);
        assert!(plan.validate().is_valid());
        observed.insert(plan.scene.roof.morphology);
    }

    assert_eq!(
        observed,
        BTreeSet::from([
            RoofMorphology::TallEarlyCrown,
            RoofMorphology::BalancedClassic,
            RoofMorphology::LowWideLate,
        ])
    );
}

#[test]
fn each_profile_is_byte_deterministic_for_fixed_seeds() {
    for morphology in [
        RoofMorphology::TallEarlyCrown,
        RoofMorphology::BalancedClassic,
        RoofMorphology::LowWideLate,
    ] {
        let mut config = GeneratorConfig::default();
        config.sequence.frame_count = 2;
        config
            .roof
            .profiles
            .retain(|profile| profile.morphology == morphology);
        config.roof.profiles[0].weight = 1;
        let sampler = SequenceSampler::new(config).unwrap();

        for seed in [0, 1, 42, 7_777, u32::MAX as u64] {
            let request =
                SequenceRequest::procedural("classic_two_stage", seed, TargetKind::Target);
            let first = sampler.sample(request.clone()).unwrap();
            let second = sampler.sample(request).unwrap();
            assert_eq!(first.scene.roof.morphology, morphology);
            assert_eq!(first, second);
            assert_eq!(
                serde_json::to_vec(&first).unwrap(),
                serde_json::to_vec(&second).unwrap()
            );
        }
    }
}

#[test]
fn legacy_configs_and_records_receive_morphology_defaults() {
    let expected_config = GeneratorConfig::default();
    let mut config_json = serde_json::to_value(&expected_config).unwrap();
    config_json["roof"]
        .as_object_mut()
        .unwrap()
        .remove("profiles");
    let decoded_config: GeneratorConfig = serde_json::from_value(config_json).unwrap();
    assert_eq!(decoded_config.roof.profiles, expected_config.roof.profiles);

    let sampler = SequenceSampler::new(expected_config).unwrap();
    let roof = sampler
        .sample(SequenceRequest::procedural(
            "classic_two_stage",
            42,
            TargetKind::Target,
        ))
        .unwrap()
        .scene
        .roof;
    let mut roof_json = serde_json::to_value(roof).unwrap();
    roof_json.as_object_mut().unwrap().remove("morphology");
    let decoded_roof: synth_data::SampledRoof = serde_json::from_value(roof_json).unwrap();
    let mut expected_roof = roof;
    expected_roof.morphology = RoofMorphology::BalancedClassic;
    assert_eq!(decoded_roof, expected_roof);
}

#[test]
fn invalid_profile_distributions_are_rejected_before_sampling() {
    let mut empty = GeneratorConfig::default();
    empty.roof.profiles.clear();
    assert!(SequenceSampler::new(empty).is_err());

    let mut duplicate = GeneratorConfig::default();
    duplicate.roof.profiles[1].morphology = duplicate.roof.profiles[0].morphology;
    assert!(SequenceSampler::new(duplicate).is_err());

    let mut disjoint = GeneratorConfig::default();
    disjoint.roof.profiles[0].overhang_m = synth_data::FloatRange::new(5.0, 6.0);
    assert!(SequenceSampler::new(disjoint).is_err());

    let mut impossible_aspect = GeneratorConfig::default();
    impossible_aspect.roof.profiles[0].footprint_aspect_ratio =
        synth_data::FloatRange::new(5.0, 6.0);
    assert!(SequenceSampler::new(impossible_aspect).is_err());
}

#[test]
#[ignore = "explicit 20k-plan stress validation"]
fn twenty_thousand_plan_validation_sweep() {
    let mut config = GeneratorConfig::default();
    config.sequence.frame_count = 1;
    let sampler = SequenceSampler::new(config).unwrap();

    for seed in 0..20_000 {
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
