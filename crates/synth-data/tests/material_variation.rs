//! Deterministic continuous material-colour sampling contracts.

use std::collections::HashSet;

use synth_data::{GeneratorConfig, SequenceRequest, SequenceSampler, TargetKind, Validate};

fn color_bits(color: [f32; 3]) -> [u32; 3] {
    color.map(f32::to_bits)
}

#[test]
fn one_named_material_produces_continuous_deterministic_colours() {
    let mut config = GeneratorConfig::default();
    config.sequence.frame_count = 1;
    config.materials.roof.truncate(1);
    config.materials.roof[0].weight = 1;
    config.materials.walls.truncate(1);
    config.materials.walls[0].weight = 1;
    let roof_choice = config.materials.roof[0].clone();
    let wall_choice = config.materials.walls[0].clone();
    let sampler = SequenceSampler::new(config).unwrap();
    let mut roof_colours = HashSet::new();
    let mut wall_colours = HashSet::new();

    for seed in 0..64 {
        let request = SequenceRequest::procedural("classic_two_stage", seed, TargetKind::Target);
        let first = sampler.sample(request.clone()).unwrap();
        let repeated = sampler.sample(request).unwrap();
        assert_eq!(first, repeated);

        for (sampled, choice) in [
            (&first.scene.roof_material, &roof_choice),
            (&first.scene.wall_material, &wall_choice),
        ] {
            assert_eq!(sampled.id, choice.id);
            for (channel, base) in sampled.base_color_srgb.iter().zip(choice.base_color_srgb) {
                let minimum = base * (1.0 - choice.base_color_variation);
                let maximum = (base * (1.0 + choice.base_color_variation)).min(1.0);
                assert!((*channel >= minimum) && (*channel <= maximum));
            }
        }
        roof_colours.insert(color_bits(first.scene.roof_material.base_color_srgb));
        wall_colours.insert(color_bits(first.scene.wall_material.base_color_srgb));
    }

    assert!(roof_colours.len() > 48);
    assert!(wall_colours.len() > 48);
}

#[test]
fn legacy_material_choices_decode_to_exact_fixed_colours() {
    let expected = GeneratorConfig::default();
    let mut encoded = serde_json::to_value(&expected).unwrap();
    let materials = encoded["materials"].as_object_mut().unwrap();
    for palette in ["roof", "walls"] {
        for choice in materials[palette].as_array_mut().unwrap() {
            choice
                .as_object_mut()
                .unwrap()
                .remove("base_color_variation");
        }
    }

    let mut decoded: GeneratorConfig = serde_json::from_value(encoded).unwrap();
    assert!(
        decoded
            .materials
            .roof
            .iter()
            .chain(&decoded.materials.walls)
            .all(|choice| choice.base_color_variation == 0.0)
    );
    assert!(decoded.validate().is_valid());
    decoded.sequence.frame_count = 1;
    let roof_palette = decoded.materials.roof.clone();
    let wall_palette = decoded.materials.walls.clone();
    let sampler = SequenceSampler::new(decoded).unwrap();

    for seed in 0..32 {
        let plan = sampler
            .sample(SequenceRequest::procedural(
                "classic_two_stage",
                seed,
                TargetKind::Target,
            ))
            .unwrap();
        let roof = roof_palette
            .iter()
            .find(|choice| choice.id == plan.scene.roof_material.id)
            .unwrap();
        let wall = wall_palette
            .iter()
            .find(|choice| choice.id == plan.scene.wall_material.id)
            .unwrap();
        assert_eq!(
            plan.scene.roof_material.base_color_srgb,
            roof.base_color_srgb
        );
        assert_eq!(
            plan.scene.wall_material.base_color_srgb,
            wall.base_color_srgb
        );
    }
}

#[test]
fn excessive_material_colour_variation_is_rejected() {
    let mut config = GeneratorConfig::default();
    config.materials.roof[0].base_color_variation = 0.51;
    let report = config.validate();
    assert!(!report.is_valid());
    assert!(report.to_string().contains("base_color_variation"));
}
