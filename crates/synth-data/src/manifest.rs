use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::SplitPolicy;

/// Current JSON annotation and manifest schema version.
pub const DATASET_SCHEMA_VERSION: &str = "1.0.0";

/// Dataset-level provenance and interpretation contract.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetManifest {
    /// Human- and machine-readable dataset identifier, such as `roof-synth-v003`.
    pub dataset_id: String,
    /// Schema understood by readers of every record in this dataset.
    pub schema_version: String,
    /// Exact generator implementation identity.
    pub generator: GeneratorDescriptor,
    /// Root seed or seed-range namespace used by the generation run.
    pub root_seed: u64,
    /// Spatial and image coordinate conventions.
    pub coordinates: CoordinateSystem,
    /// Stable building-level split allocation.
    pub split_policy: SplitPolicy,
    /// Semantic integer-ID definitions.
    pub labels: LabelTaxonomy,
    /// External assets that may influence rendered samples.
    pub source_assets: Vec<SourceAsset>,
    /// Dense and image outputs written beside JSON records.
    pub outputs: Vec<OutputDefinition>,
    /// Final accepted record counts.
    pub statistics: DatasetStatistics,
}

impl DatasetManifest {
    /// Creates a manifest with current schema and coordinate defaults.
    #[must_use]
    pub fn new(
        dataset_id: impl Into<String>,
        generator: GeneratorDescriptor,
        root_seed: u64,
    ) -> Self {
        Self {
            dataset_id: dataset_id.into(),
            schema_version: DATASET_SCHEMA_VERSION.to_owned(),
            generator,
            root_seed,
            coordinates: CoordinateSystem::default(),
            split_policy: SplitPolicy::default(),
            labels: LabelTaxonomy::default(),
            source_assets: Vec::new(),
            outputs: OutputDefinition::standard_outputs(),
            statistics: DatasetStatistics::default(),
        }
    }
}

/// Generator build and reproducibility information.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GeneratorDescriptor {
    /// Executable or pipeline name.
    pub name: String,
    /// Released generator version.
    pub version: String,
    /// Source-control revision, if generation occurred from a checkout.
    pub source_revision: Option<String>,
    /// Explicit PRNG algorithm contract.
    pub rng_algorithm: String,
    /// Hash of the complete serialized generator configuration.
    pub config_fingerprint: String,
    /// Sorted renderer and execution details needed to audit non-bit-exact RGB.
    pub execution_environment: BTreeMap<String, String>,
}

impl GeneratorDescriptor {
    /// Creates a descriptor for the crate's default ChaCha20 sampling contract.
    #[must_use]
    pub fn chacha20(
        name: impl Into<String>,
        version: impl Into<String>,
        config_fingerprint: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            version: version.into(),
            source_revision: None,
            rng_algorithm: "rand_chacha::ChaCha20Rng/0.9".to_owned(),
            config_fingerprint: config_fingerprint.into(),
            execution_environment: BTreeMap::new(),
        }
    }
}

/// Fixed transform and image conventions used throughout a dataset.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CoordinateSystem {
    /// World handedness and up axis.
    pub world: String,
    /// Camera forward and image-axis convention.
    pub camera: String,
    /// Image coordinate origin and units for compact annotations.
    pub image: String,
    /// Unit used for positions and dimensions.
    pub length_unit: String,
}

impl Default for CoordinateSystem {
    fn default() -> Self {
        Self {
            world: "right_handed_y_up".to_owned(),
            camera: "right_handed_negative_z_forward_x_right_y_up".to_owned(),
            image: "normalized_top_left_x_right_y_down".to_owned(),
            length_unit: "metre".to_owned(),
        }
    }
}

/// Integer semantic class stored in compact per-frame labels.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LabelClass {
    /// Stable integer value written to records or dense maps.
    pub id: u16,
    /// Stable snake-case name.
    pub name: String,
}

/// Dataset-wide class maps, avoiding repeated strings in each frame.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LabelTaxonomy {
    /// Structural point classes.
    pub keypoints: Vec<LabelClass>,
    /// Eave, pitch-break, crown-corner, crown-top, and silhouette edge classes.
    pub edges: Vec<LabelClass>,
    /// Semantic roof-part classes used by part masks.
    pub parts: Vec<LabelClass>,
    /// Parametric face identities used by face maps.
    pub faces: Vec<LabelClass>,
}

/// Kind of external source data used to build scenes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceAssetKind {
    /// Mesh or procedural geometry input.
    Geometry,
    /// Surface image or material scan.
    Texture,
    /// High-dynamic-range environment lighting.
    Environment,
    /// Real photograph used as a background or appearance source.
    Photograph,
    /// Camera or sensor profile measured from a device.
    DeviceProfile,
}

/// Provenance for an asset that may influence one or more generated scenes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceAsset {
    /// Stable asset identifier used in scene metadata.
    pub id: String,
    /// Coarse content category.
    pub kind: SourceAssetKind,
    /// Hex-encoded content digest including algorithm prefix.
    pub content_hash: String,
    /// Licence or internal-rights identifier.
    pub license: String,
    /// Group assigned atomically to one dataset split.
    pub split_group: Option<String>,
}

/// Encoding contract for one file associated with a frame sample key.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputDefinition {
    /// Stable suffix, such as `rgb` or `face_coordinates`.
    pub name: String,
    /// Internet media type.
    pub media_type: String,
    /// Codec or scalar layout.
    pub encoding: String,
    /// Whether target-positive records must contain this output.
    pub required_for_targets: bool,
}

impl OutputDefinition {
    fn standard_outputs() -> Vec<Self> {
        [
            ("rgb", "image/jpeg", "jpeg", true),
            ("roof_mask", "image/png", "uint8", true),
            ("amodal_roof_mask", "image/png", "uint8", true),
            ("part_mask", "image/png", "uint16", true),
            ("face_id_map", "image/png", "uint16", true),
            (
                "face_coordinates",
                "application/octet-stream",
                "rg16float-zstd",
                true,
            ),
        ]
        .into_iter()
        .map(|(name, media_type, encoding, required_for_targets)| Self {
            name: name.to_owned(),
            media_type: media_type.to_owned(),
            encoding: encoding.to_owned(),
            required_for_targets,
        })
        .collect()
    }
}

/// Accepted sample counts, split by partition.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DatasetStatistics {
    /// Accepted training frames.
    pub train_frames: u64,
    /// Accepted validation frames.
    pub validation_frames: u64,
    /// Accepted test frames.
    pub test_frames: u64,
    /// Accepted coherent sequences.
    pub sequences: u64,
}
