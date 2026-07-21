//! Resolution of sampled appearance choices to physical renderer assets.

use std::collections::BTreeSet;

use anyhow::{Context, Result};
use synth_data::SampledScene;
use synth_render::{MaterialSelection, MaterialSlot, SurfacePattern};

use crate::assets::{PBR_MATERIAL_IDS, PublicAssetCatalog};

const ROOF_TILE_LAYER: u32 = 0;
const CORRUGATED_LAYER: u32 = 1;
const BRICK_LAYER: u32 = 2;
const ASPHALT_LAYER: u32 = 3;
const PROCEDURAL_LAYER: u32 = 31;

/// Exact physical texture selection and provenance for one sampled sequence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolvedRenderAssets {
    /// Logical Poly Haven environment ID selected for this scene regime.
    pub environment_id: String,
    /// Mapping from procedural material roles to resident PBR array layers.
    pub materials: MaterialSelection,
    /// Exact file-level IDs copied into the sequence composition record.
    pub source_asset_ids: Vec<String>,
}

impl ResolvedRenderAssets {
    /// Resolves materials, HDR panorama, and exact source provenance.
    pub fn resolve(
        catalog: &PublicAssetCatalog,
        scene: &SampledScene,
        sequence_seed: u64,
    ) -> Result<Self> {
        let environment = scene
            .composition
            .environment
            .context("sampled composition contains no environment")?;
        let environment_id = catalog
            .environment_id_for(environment, sequence_seed)?
            .to_owned();

        let roof_layer = roof_layer(&scene.roof_material.id);
        let wall_layer = wall_layer(&scene.wall_material.id);
        let ground_layer = scene
            .composition
            .ground_material
            .as_ref()
            .map_or(PROCEDURAL_LAYER, |material| ground_layer(&material.id));
        let mut materials = MaterialSelection::default();
        materials.set(MaterialSlot::ROOF, roof_layer)?;
        materials.set_pattern(MaterialSlot::ROOF, SurfacePattern::ROOF_SEAMS);
        materials.set(MaterialSlot::WALL, wall_layer)?;
        materials.set_pattern(MaterialSlot::WALL, wall_pattern(&scene.wall_material.id));
        materials.set(MaterialSlot::GROUND, ground_layer)?;
        materials.set_pattern(
            MaterialSlot::GROUND,
            ground_pattern(
                scene
                    .composition
                    .ground_material
                    .as_ref()
                    .map_or("", |material| material.id.as_str()),
            ),
        );
        materials.set(MaterialSlot::ASPHALT, ASPHALT_LAYER)?;
        materials.set_pattern(MaterialSlot::ASPHALT, SurfacePattern::ASPHALT);
        // Background materials vary per building and are rendered procedurally;
        // they must not inherit semantics from one public texture-array index.
        materials.set(MaterialSlot::BACKGROUND_WALL, PROCEDURAL_LAYER)?;
        materials.set_pattern(
            MaterialSlot::BACKGROUND_WALL,
            SurfacePattern::BACKGROUND_WINDOWS,
        );

        let mut logical_asset_ids = BTreeSet::new();
        for layer in [roof_layer, wall_layer, ground_layer, ASPHALT_LAYER] {
            if let Some(asset_id) = PBR_MATERIAL_IDS.get(layer as usize) {
                logical_asset_ids.insert(*asset_id);
            }
        }

        let mut source_asset_ids = Vec::new();
        for asset_id in logical_asset_ids {
            source_asset_ids.extend(catalog.dataset_source_ids_for(asset_id)?);
        }
        source_asset_ids.extend(catalog.dataset_source_ids_for(&environment_id)?);
        source_asset_ids.sort();
        source_asset_ids.dedup();

        Ok(Self {
            environment_id,
            materials,
            source_asset_ids,
        })
    }
}

fn roof_layer(material_id: &str) -> u32 {
    if matches!(material_id, "original_red" | "faded_red") {
        ROOF_TILE_LAYER
    } else {
        CORRUGATED_LAYER
    }
}

fn wall_layer(material_id: &str) -> u32 {
    match material_id {
        "warm_brick" => BRICK_LAYER,
        "neutral_cladding" | "dark_repaint" | "blue_repaint" | "green_repaint" => CORRUGATED_LAYER,
        _ => PROCEDURAL_LAYER,
    }
}

fn wall_pattern(material_id: &str) -> SurfacePattern {
    match material_id {
        "warm_brick" => SurfacePattern::BRICK,
        "neutral_cladding" | "dark_repaint" | "blue_repaint" | "green_repaint" => {
            SurfacePattern::VERTICAL_CLADDING
        }
        _ => SurfacePattern::SMOOTH,
    }
}

fn ground_layer(material_id: &str) -> u32 {
    if material_id.contains("asphalt") {
        ASPHALT_LAYER
    } else {
        PROCEDURAL_LAYER
    }
}

fn ground_pattern(material_id: &str) -> SurfacePattern {
    if material_id.contains("asphalt") {
        SurfacePattern::ASPHALT
    } else {
        SurfacePattern::SMOOTH
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_repainted_roofs_structural_without_reusing_red_albedo() {
        assert_eq!(roof_layer("original_red"), ROOF_TILE_LAYER);
        assert_eq!(roof_layer("faded_red"), ROOF_TILE_LAYER);
        assert_eq!(roof_layer("repainted_blue"), CORRUGATED_LAYER);
        assert_eq!(roof_layer("repainted_green"), CORRUGATED_LAYER);
    }

    #[test]
    fn leaves_untextured_render_and_paving_on_procedural_surfaces() {
        assert_eq!(wall_layer("painted_render"), PROCEDURAL_LAYER);
        assert_eq!(ground_layer("light_concrete"), PROCEDURAL_LAYER);
        assert_eq!(ground_layer("aged_asphalt"), ASPHALT_LAYER);
    }

    #[test]
    fn logical_patterns_do_not_depend_on_public_texture_layer_numbers() {
        assert_eq!(wall_pattern("warm_brick"), SurfacePattern::BRICK);
        assert_eq!(
            wall_pattern("neutral_cladding"),
            SurfacePattern::VERTICAL_CLADDING
        );
        assert_eq!(wall_pattern("painted_render"), SurfacePattern::SMOOTH);
        assert_eq!(ground_pattern("aged_asphalt"), SurfacePattern::ASPHALT);
        assert_eq!(ground_pattern("dry_grass"), SurfacePattern::SMOOTH);
    }
}
