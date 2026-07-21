//! Native validation of generated dataset manifests, records, and shard contents.

use std::{
    collections::{BTreeMap, BTreeSet, btree_map::Entry},
    fs::File,
    io::{BufReader, Read},
    path::{Component, Path, PathBuf},
};

use anyhow::{Context, Result};
use serde::Serialize;
use sha2::{Digest, Sha256};
use synth_data::{
    AssetRef, DatasetManifest, DatasetSplit, DatasetStatistics, DatasetValidator, FrameRecord,
    OrdinaryRoofFamily, SequenceRecord, Severity, TargetKind, Validate, ValidationIssue,
    ValidationReport,
};

use crate::{
    coverage::{SceneRegimeCounts, scene_regime},
    shard::list_shards,
};

/// Machine-readable result of validating one generated dataset directory.
#[derive(Clone, Debug, Serialize)]
pub struct ValidationSummary {
    /// Dataset directory that was inspected.
    pub dataset: PathBuf,
    /// Whether the dataset contains no error-severity findings.
    pub valid: bool,
    /// Number of tar shards inspected.
    pub shard_count: usize,
    /// Number of distinct WebDataset sample keys found in shards.
    pub sample_count: usize,
    /// Number of regular artifact entries read, including duplicate entries.
    pub artifact_count: usize,
    /// Number of labels successfully parsed as frame records.
    pub frame_count: usize,
    /// Number of sequence records read from `sequences.json`.
    pub sequence_count: usize,
    /// Number of target-building sequences.
    pub target_sequence_count: usize,
    /// Number of ordinary-building negative sequences.
    pub negative_sequence_count: usize,
    /// Counts of each rendered ordinary negative roof family.
    pub ordinary_roof_families: BTreeMap<String, usize>,
    /// Counts declared by `dataset.json`.
    pub expected_statistics: DatasetStatistics,
    /// Counts reconstructed from parsed frame and sequence records.
    pub observed_statistics: DatasetStatistics,
    /// Number of error-severity findings.
    pub error_count: usize,
    /// Number of warning-severity findings.
    pub warning_count: usize,
    /// Deterministically ordered contract and content findings.
    pub issues: Vec<ValidationIssue>,
}

/// Validates manifests, sequence relationships, frame records, and artifact bytes.
///
/// The directory must contain `dataset.json`, `sequences.json`, and zero or more
/// top-level `*.tar` shards. JSON files that establish the dataset schema are treated
/// as required inputs; record-level and content problems are collected in the summary.
pub fn validate_dataset(dataset: impl AsRef<Path>) -> Result<ValidationSummary> {
    let dataset = dataset.as_ref();
    let manifest: DatasetManifest = read_json(&dataset.join("dataset.json"))?;
    let sequences: Vec<SequenceRecord> = read_json(&dataset.join("sequences.json"))?;
    let shard_paths = list_shards(dataset)
        .with_context(|| format!("failed to list shards in {}", dataset.display()))?;

    let mut issues = Vec::new();
    append_report(&mut issues, "dataset.json", manifest.validate());
    let (samples, artifact_count) = read_samples(&shard_paths, dataset, &mut issues)?;
    if shard_paths.is_empty() && total_frames(manifest.statistics) > 0 {
        error(
            &mut issues,
            "missing_shards",
            "*.tar",
            "manifest declares frames but the dataset contains no tar shards",
        );
    }

    let validator = DatasetValidator::new(&manifest);
    let mut frames_by_key = BTreeMap::new();
    let mut observed_statistics = DatasetStatistics::default();

    for (archive_key, sample) in &samples {
        let labels_path = format!("{archive_key}.labels.json");
        let Some(labels) = sample.files.get(&labels_path) else {
            error(
                &mut issues,
                "missing_frame_record",
                format!("samples.{archive_key}"),
                "sample has no labels.json frame record",
            );
            continue;
        };

        let frame: FrameRecord = match serde_json::from_slice(labels) {
            Ok(frame) => frame,
            Err(parse_error) => {
                error(
                    &mut issues,
                    "invalid_frame_record",
                    format!("samples.{archive_key}.labels.json"),
                    format!("failed to parse FrameRecord: {parse_error}"),
                );
                continue;
            }
        };

        append_report(
            &mut issues,
            &format!("samples.{archive_key}.labels.json"),
            validator.validate_frame(&frame),
        );
        if frame.sample_key != *archive_key {
            error(
                &mut issues,
                "sample_key_mismatch",
                format!("samples.{archive_key}.labels.json.sample_key"),
                format!(
                    "frame declares sample key {:?}, but its archive key is {archive_key:?}",
                    frame.sample_key
                ),
            );
        }
        validate_frame_assets(archive_key, sample, &frame, &manifest, &mut issues);
        increment_split_count(&mut observed_statistics, frame.split);

        match frames_by_key.entry(frame.sample_key.clone()) {
            Entry::Vacant(entry) => {
                entry.insert(frame);
            }
            Entry::Occupied(_) => error(
                &mut issues,
                "duplicate_frame_sample_key",
                format!("samples.{archive_key}.labels.json.sample_key"),
                "more than one frame record declares this sample key",
            ),
        }
    }

    validate_sequences(&manifest, &sequences, &frames_by_key, &mut issues);
    let (target_sequence_count, negative_sequence_count, ordinary_roof_families) =
        validate_corpus_composition(&manifest, &sequences, &mut issues);
    observed_statistics.sequences = sequences.len() as u64;
    validate_statistics(manifest.statistics, observed_statistics, &mut issues);

    let error_count = issues
        .iter()
        .filter(|issue| issue.severity == Severity::Error)
        .count();
    let warning_count = issues.len() - error_count;
    Ok(ValidationSummary {
        dataset: dataset.to_path_buf(),
        valid: error_count == 0,
        shard_count: shard_paths.len(),
        sample_count: samples.len(),
        artifact_count,
        frame_count: observed_frame_count(observed_statistics),
        sequence_count: sequences.len(),
        target_sequence_count,
        negative_sequence_count,
        ordinary_roof_families,
        expected_statistics: manifest.statistics,
        observed_statistics,
        error_count,
        warning_count,
        issues,
    })
}

fn validate_corpus_composition(
    manifest: &DatasetManifest,
    sequences: &[SequenceRecord],
    issues: &mut Vec<ValidationIssue>,
) -> (usize, usize, BTreeMap<String, usize>) {
    let target_count = sequences
        .iter()
        .filter(|sequence| sequence.target_kind == TargetKind::Target)
        .count();
    let negative_count = sequences
        .iter()
        .filter(|sequence| sequence.target_kind == TargetKind::Negative)
        .count();
    let mut families = BTreeMap::new();
    let mut target_regimes = SceneRegimeCounts::default();
    let mut negative_regimes = SceneRegimeCounts::default();
    for sequence in sequences {
        let Some(environment) = sequence.scene.composition.environment else {
            continue;
        };
        match sequence.target_kind {
            TargetKind::Target => target_regimes.increment(scene_regime(environment.domain)),
            TargetKind::Negative => negative_regimes.increment(scene_regime(environment.domain)),
            TargetKind::NearMiss => {}
        }
    }
    for sequence in sequences
        .iter()
        .filter(|sequence| sequence.target_kind == TargetKind::Negative)
    {
        if let Some(roof) = sequence.scene.ordinary_roof {
            *families.entry(roof.family.as_str().to_owned()).or_insert(0) += 1;
        }
    }

    for (key, observed) in [
        ("target_buildings", target_count),
        ("ordinary_negative_buildings", negative_count),
    ] {
        if let Some(declared) = manifest.generator.execution_environment.get(key) {
            match declared.parse::<usize>() {
                Ok(declared) if declared == observed => {}
                Ok(declared) => error(
                    issues,
                    "corpus_class_count_mismatch",
                    format!("dataset.json.generator.execution_environment.{key}"),
                    format!("manifest declares {declared}, but sequences contain {observed}"),
                ),
                Err(parse_error) => error(
                    issues,
                    "invalid_corpus_class_count",
                    format!("dataset.json.generator.execution_environment.{key}"),
                    format!("declared count is not an unsigned integer: {parse_error}"),
                ),
            }
        }
    }
    if manifest
        .generator
        .execution_environment
        .contains_key("scene_regime_balance_manifest")
    {
        let combined_regimes = target_regimes.plus(negative_regimes);
        for (class, counts) in [
            ("target", target_regimes),
            ("negative", negative_regimes),
            ("combined", combined_regimes),
        ] {
            if !counts.is_balanced() {
                error(
                    issues,
                    "unbalanced_scene_regimes",
                    "sequences.json",
                    format!(
                        "{class} urban/suburban/remote counts differ by more than one: {}/{}/{}",
                        counts.urban, counts.suburban, counts.remote
                    ),
                );
            }
        }
        for (key, observed) in [
            ("target_scene_regimes", target_regimes),
            ("negative_scene_regimes", negative_regimes),
            ("combined_scene_regimes", combined_regimes),
        ] {
            let expected = format!(
                "urban={},suburban={},remote={}",
                observed.urban, observed.suburban, observed.remote
            );
            if manifest
                .generator
                .execution_environment
                .get(key)
                .is_none_or(|declared| declared != &expected)
            {
                error(
                    issues,
                    "scene_regime_count_mismatch",
                    format!("dataset.json.generator.execution_environment.{key}"),
                    format!("expected exact sequence-derived counts {expected}"),
                );
            }
        }
    }
    if negative_count >= OrdinaryRoofFamily::ALL.len() {
        for family in OrdinaryRoofFamily::ALL {
            if !families.contains_key(family.as_str()) {
                error(
                    issues,
                    "missing_ordinary_roof_family",
                    "sequences.json",
                    format!(
                        "negative corpus is large enough to cover every family but contains no {} roof",
                        family.as_str()
                    ),
                );
            }
        }
    }
    (target_count, negative_count, families)
}

#[derive(Default)]
struct SampleArtifacts {
    shards: BTreeSet<PathBuf>,
    files: BTreeMap<String, Vec<u8>>,
}

fn read_samples(
    shard_paths: &[PathBuf],
    dataset: &Path,
    issues: &mut Vec<ValidationIssue>,
) -> Result<(BTreeMap<String, SampleArtifacts>, usize)> {
    let mut samples = BTreeMap::<String, SampleArtifacts>::new();
    let mut artifact_count = 0;

    for shard_path in shard_paths {
        let shard_name = relative_display(dataset, shard_path);
        let file = File::open(shard_path)
            .with_context(|| format!("failed to open shard {}", shard_path.display()))?;
        let mut archive = tar::Archive::new(BufReader::new(file));
        let entries = archive
            .entries()
            .with_context(|| format!("failed to read tar index for {shard_name}"))?;

        for (entry_index, entry) in entries.enumerate() {
            let mut entry = entry.with_context(|| {
                format!("failed to read entry {entry_index} from shard {shard_name}")
            })?;
            if !entry.header().entry_type().is_file() {
                warning(
                    issues,
                    "non_file_archive_entry",
                    format!("{shard_name}[{entry_index}]"),
                    "only regular artifact files are expected in dataset shards",
                );
                continue;
            }
            artifact_count += 1;

            let path = entry
                .path()
                .with_context(|| {
                    format!("entry {entry_index} in {shard_name} has an invalid path")
                })?
                .into_owned();
            if path.components().count() != 1
                || path.components().any(|component| {
                    matches!(
                        component,
                        Component::ParentDir
                            | Component::RootDir
                            | Component::Prefix(_)
                            | Component::CurDir
                    )
                })
            {
                error(
                    issues,
                    "invalid_archive_member_path",
                    format!("{shard_name}[{entry_index}]"),
                    format!("artifact path {:?} must be a top-level relative file", path),
                );
                continue;
            }
            let Some(member_name) = path.to_str().map(str::to_owned) else {
                error(
                    issues,
                    "non_utf8_archive_member",
                    format!("{shard_name}[{entry_index}]"),
                    "artifact path must be valid UTF-8",
                );
                continue;
            };
            let Some((sample_key, suffix)) = member_name.split_once('.') else {
                error(
                    issues,
                    "invalid_archive_member_name",
                    format!("{shard_name}.{member_name}"),
                    "artifact name must have the form sample_key.suffix",
                );
                continue;
            };
            if sample_key.is_empty() || suffix.is_empty() {
                error(
                    issues,
                    "invalid_archive_member_name",
                    format!("{shard_name}.{member_name}"),
                    "sample key and artifact suffix must both be non-empty",
                );
                continue;
            }
            let sample_key = sample_key.to_owned();

            let mut bytes = Vec::new();
            entry
                .read_to_end(&mut bytes)
                .with_context(|| format!("failed to read {member_name} from shard {shard_name}"))?;
            let sample = samples.entry(sample_key.clone()).or_default();
            let new_shard = sample.shards.insert(shard_path.clone());
            if new_shard && sample.shards.len() > 1 {
                error(
                    issues,
                    "sample_spans_shards",
                    format!("samples.{sample_key}"),
                    "artifacts for one sample must reside in exactly one shard",
                );
            }
            match sample.files.entry(member_name.clone()) {
                Entry::Vacant(file) => {
                    file.insert(bytes);
                }
                Entry::Occupied(_) => error(
                    issues,
                    "duplicate_artifact",
                    format!("samples.{sample_key}.{member_name}"),
                    "artifact path occurs more than once",
                ),
            }
        }
    }

    Ok((samples, artifact_count))
}

fn validate_frame_assets(
    archive_key: &str,
    sample: &SampleArtifacts,
    frame: &FrameRecord,
    manifest: &DatasetManifest,
    issues: &mut Vec<ValidationIssue>,
) {
    validate_asset_ref(archive_key, sample, "assets.rgb", &frame.assets.rgb, issues);
    if let Some(asset) = &frame.assets.surface_normals {
        validate_asset_ref(archive_key, sample, "assets.surface_normals", asset, issues);
    }
    if let Some(asset) = &frame.assets.motion_vectors {
        validate_asset_ref(archive_key, sample, "assets.motion_vectors", asset, issues);
    }
    for (path, asset) in [
        ("labels.dense.roof_mask", &frame.labels.dense.roof_mask),
        (
            "labels.dense.amodal_roof_mask",
            &frame.labels.dense.amodal_roof_mask,
        ),
        ("labels.dense.part_mask", &frame.labels.dense.part_mask),
        ("labels.dense.face_id_map", &frame.labels.dense.face_id_map),
        (
            "labels.dense.face_coordinates",
            &frame.labels.dense.face_coordinates,
        ),
    ] {
        if let Some(asset) = asset {
            validate_asset_ref(archive_key, sample, path, asset, issues);
        }
    }
    validate_dense_targets(archive_key, sample, frame, manifest, issues);
}

fn validate_dense_targets(
    archive_key: &str,
    sample: &SampleArtifacts,
    frame: &FrameRecord,
    manifest: &DatasetManifest,
    issues: &mut Vec<ValidationIssue>,
) {
    let intrinsics = frame.camera.intrinsics;
    let Some(pixel_count) = (intrinsics.width as usize).checked_mul(intrinsics.height as usize)
    else {
        error(
            issues,
            "dense_target_size_overflow",
            format!("samples.{archive_key}.camera.intrinsics"),
            "camera dimensions overflow the native address space",
        );
        return;
    };
    let roof_mask = frame
        .labels
        .dense
        .roof_mask
        .as_ref()
        .and_then(|asset| sample.files.get(&asset.path))
        .and_then(|bytes| {
            decode_png_u8(
                archive_key,
                "roof_mask",
                bytes,
                intrinsics.width,
                intrinsics.height,
                issues,
            )
        });
    let amodal_roof_mask = frame
        .labels
        .dense
        .amodal_roof_mask
        .as_ref()
        .and_then(|asset| sample.files.get(&asset.path))
        .and_then(|bytes| {
            decode_png_u8(
                archive_key,
                "amodal_roof_mask",
                bytes,
                intrinsics.width,
                intrinsics.height,
                issues,
            )
        });
    let part_mask = frame
        .labels
        .dense
        .part_mask
        .as_ref()
        .and_then(|asset| sample.files.get(&asset.path))
        .and_then(|bytes| {
            decode_png_u16(
                archive_key,
                "part_mask",
                bytes,
                intrinsics.width,
                intrinsics.height,
                issues,
            )
        });
    let face_ids = frame
        .labels
        .dense
        .face_id_map
        .as_ref()
        .and_then(|asset| sample.files.get(&asset.path))
        .and_then(|bytes| {
            decode_png_u16(
                archive_key,
                "face_id_map",
                bytes,
                intrinsics.width,
                intrinsics.height,
                issues,
            )
        });
    let face_coordinates = frame
        .labels
        .dense
        .face_coordinates
        .as_ref()
        .and_then(|asset| sample.files.get(&asset.path))
        .and_then(|bytes| decode_face_coordinates(archive_key, bytes, pixel_count, issues));

    let known_parts = manifest
        .labels
        .parts
        .iter()
        .map(|class| class.id)
        .collect::<BTreeSet<_>>();
    let known_faces = manifest
        .labels
        .faces
        .iter()
        .map(|class| class.id)
        .collect::<BTreeSet<_>>();
    if let Some(mask) = &roof_mask
        && let Some((index, value)) = mask
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !matches!(value, 0 | 255))
    {
        error(
            issues,
            "non_binary_roof_mask",
            format!("samples.{archive_key}.roof_mask[{index}]"),
            format!("roof mask value {value} is neither 0 nor 255"),
        );
    }
    if let Some(mask) = &amodal_roof_mask
        && let Some((index, value)) = mask
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| !matches!(value, 0 | 255))
    {
        error(
            issues,
            "non_binary_amodal_roof_mask",
            format!("samples.{archive_key}.amodal_roof_mask[{index}]"),
            format!("amodal roof mask value {value} is neither 0 nor 255"),
        );
    }
    if let (Some(visible), Some(amodal)) = (&roof_mask, &amodal_roof_mask)
        && let Some(index) = visible
            .iter()
            .zip(amodal)
            .position(|(&visible, &amodal)| visible != 0 && amodal == 0)
    {
        error(
            issues,
            "visible_outside_amodal_mask",
            format!("samples.{archive_key}.amodal_roof_mask[{index}]"),
            "every visible roof pixel must also belong to the roof-only silhouette",
        );
    }
    if let Some(mask) = &part_mask
        && let Some((index, value)) = mask
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| *value != 0 && !known_parts.contains(value))
    {
        error(
            issues,
            "unknown_part_mask_id",
            format!("samples.{archive_key}.part_mask[{index}]"),
            format!("part mask ID {value} is absent from the manifest taxonomy"),
        );
    }
    if let Some(mask) = &face_ids
        && let Some((index, value)) = mask
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| *value != 0 && !known_faces.contains(value))
    {
        error(
            issues,
            "unknown_face_mask_id",
            format!("samples.{archive_key}.face_id_map[{index}]"),
            format!("face ID {value} is absent from the manifest taxonomy"),
        );
    }

    if let (Some(roof), Some(parts), Some(faces), Some(coordinates)) =
        (&roof_mask, &part_mask, &face_ids, &face_coordinates)
    {
        for index in 0..pixel_count {
            let is_roof = roof[index] != 0;
            let coordinate = coordinates[index];
            if !is_roof {
                if parts[index] != 0
                    || faces[index] != 0
                    || coordinate[0].to_bits() != 0
                    || coordinate[1].to_bits() != 0
                {
                    error(
                        issues,
                        "dense_target_background_mismatch",
                        format!("samples.{archive_key}.dense_targets[{index}]"),
                        "non-roof pixels must have zero part, face, and face-coordinate values",
                    );
                    break;
                }
                continue;
            }
            let expected_part = roof_geometry::FaceId::from_u32(u32::from(faces[index]))
                .map(|face| face.class().as_u16());
            if expected_part != Some(parts[index]) {
                error(
                    issues,
                    "dense_target_semantic_mismatch",
                    format!("samples.{archive_key}.dense_targets[{index}]"),
                    "roof, part, and face IDs do not describe the same semantic pixel",
                );
                break;
            }
            if coordinate
                .iter()
                .any(|value| !value.is_finite() || !(-0.001..=1.001).contains(value))
            {
                error(
                    issues,
                    "invalid_face_coordinate",
                    format!("samples.{archive_key}.face_coordinates[{index}]"),
                    "roof face coordinates must be finite and normalized",
                );
                break;
            }
        }
    }
}

fn decode_png_u8(
    archive_key: &str,
    label: &str,
    bytes: &[u8],
    width: u32,
    height: u32,
    issues: &mut Vec<ValidationIssue>,
) -> Option<Vec<u8>> {
    let image = match image::load_from_memory_with_format(bytes, image::ImageFormat::Png) {
        Ok(image) => image,
        Err(error_message) => {
            error(
                issues,
                "invalid_dense_png",
                format!("samples.{archive_key}.{label}"),
                format!("failed to decode PNG: {error_message}"),
            );
            return None;
        }
    };
    if image.width() != width || image.height() != height || image.color() != image::ColorType::L8 {
        error(
            issues,
            "dense_target_layout_mismatch",
            format!("samples.{archive_key}.{label}"),
            format!(
                "expected {width}x{height} L8 PNG, got {}x{} {:?}",
                image.width(),
                image.height(),
                image.color()
            ),
        );
        return None;
    }
    Some(image.into_luma8().into_raw())
}

fn decode_png_u16(
    archive_key: &str,
    label: &str,
    bytes: &[u8],
    width: u32,
    height: u32,
    issues: &mut Vec<ValidationIssue>,
) -> Option<Vec<u16>> {
    let image = match image::load_from_memory_with_format(bytes, image::ImageFormat::Png) {
        Ok(image) => image,
        Err(error_message) => {
            error(
                issues,
                "invalid_dense_png",
                format!("samples.{archive_key}.{label}"),
                format!("failed to decode PNG: {error_message}"),
            );
            return None;
        }
    };
    if image.width() != width || image.height() != height || image.color() != image::ColorType::L16
    {
        error(
            issues,
            "dense_target_layout_mismatch",
            format!("samples.{archive_key}.{label}"),
            format!(
                "expected {width}x{height} L16 PNG, got {}x{} {:?}",
                image.width(),
                image.height(),
                image.color()
            ),
        );
        return None;
    }
    Some(image.into_luma16().into_raw())
}

fn decode_face_coordinates(
    archive_key: &str,
    bytes: &[u8],
    pixel_count: usize,
    issues: &mut Vec<ValidationIssue>,
) -> Option<Vec<[f32; 2]>> {
    let expected_length = match pixel_count.checked_mul(4) {
        Some(length) => length,
        None => {
            error(
                issues,
                "dense_target_size_overflow",
                format!("samples.{archive_key}.face_coordinates"),
                "face-coordinate byte length overflowed",
            );
            return None;
        }
    };
    let decoder = match zstd::stream::read::Decoder::new(bytes) {
        Ok(decoder) => decoder,
        Err(error_message) => {
            error(
                issues,
                "invalid_face_coordinate_stream",
                format!("samples.{archive_key}.face_coordinates"),
                format!("failed to create zstd decoder: {error_message}"),
            );
            return None;
        }
    };
    let mut decoded = Vec::with_capacity(expected_length);
    if let Err(error_message) = decoder
        .take(expected_length as u64 + 1)
        .read_to_end(&mut decoded)
    {
        error(
            issues,
            "invalid_face_coordinate_stream",
            format!("samples.{archive_key}.face_coordinates"),
            format!("failed to decode zstd stream: {error_message}"),
        );
        return None;
    }
    if decoded.len() != expected_length {
        error(
            issues,
            "face_coordinate_size_mismatch",
            format!("samples.{archive_key}.face_coordinates"),
            format!(
                "expected {expected_length} decoded bytes, got {}",
                decoded.len()
            ),
        );
        return None;
    }
    Some(
        decoded
            .chunks_exact(4)
            .map(|chunk| {
                [
                    half::f16::from_bits(u16::from_le_bytes([chunk[0], chunk[1]])).to_f32(),
                    half::f16::from_bits(u16::from_le_bytes([chunk[2], chunk[3]])).to_f32(),
                ]
            })
            .collect(),
    )
}

fn validate_asset_ref(
    archive_key: &str,
    sample: &SampleArtifacts,
    record_path: &str,
    asset: &AssetRef,
    issues: &mut Vec<ValidationIssue>,
) {
    let issue_path = format!("samples.{archive_key}.labels.json.{record_path}");
    let referenced_sample = asset.path.split_once('.').map(|(key, _)| key);
    if referenced_sample != Some(archive_key) {
        error(
            issues,
            "asset_outside_sample",
            format!("{issue_path}.path"),
            format!(
                "asset {:?} is not grouped under archive sample {archive_key:?}",
                asset.path
            ),
        );
    }

    let Some(bytes) = sample.files.get(&asset.path) else {
        error(
            issues,
            "missing_referenced_asset",
            format!("{issue_path}.path"),
            format!(
                "referenced artifact {:?} is absent from this sample",
                asset.path
            ),
        );
        return;
    };
    let Some(expected_hash) = &asset.content_hash else {
        error(
            issues,
            "missing_content_hash",
            format!("{issue_path}.content_hash"),
            "referenced artifacts require a sha256 content hash",
        );
        return;
    };
    let Some(expected_hex) = expected_hash.strip_prefix("sha256:") else {
        error(
            issues,
            "invalid_content_hash",
            format!("{issue_path}.content_hash"),
            "content hash must use the sha256:<hex> form",
        );
        return;
    };
    if expected_hex.len() != 64 || !expected_hex.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        error(
            issues,
            "invalid_content_hash",
            format!("{issue_path}.content_hash"),
            "SHA-256 digest must contain exactly 64 hexadecimal digits",
        );
        return;
    }

    let actual_hex = sha256_hex(bytes);
    if !expected_hex.eq_ignore_ascii_case(&actual_hex) {
        error(
            issues,
            "content_hash_mismatch",
            format!("{issue_path}.content_hash"),
            format!("expected sha256:{expected_hex}, calculated sha256:{actual_hex}"),
        );
    }
}

fn validate_sequences(
    manifest: &DatasetManifest,
    sequences: &[SequenceRecord],
    frames_by_key: &BTreeMap<String, FrameRecord>,
    issues: &mut Vec<ValidationIssue>,
) {
    let validator = DatasetValidator::new(manifest);
    let mut sequence_ids = BTreeSet::new();
    let mut referenced_samples = BTreeSet::new();

    for (sequence_index, sequence) in sequences.iter().enumerate() {
        let sequence_path = format!("sequences.json[{sequence_index}]");
        if !sequence_ids.insert(sequence.sequence_id.as_str()) {
            error(
                issues,
                "duplicate_sequence_id",
                format!("{sequence_path}.sequence_id"),
                "sequence IDs must be unique within the dataset",
            );
        }

        let mut frames = Vec::with_capacity(sequence.frames.len());
        for (frame_index, frame_ref) in sequence.frames.iter().enumerate() {
            if !referenced_samples.insert(frame_ref.sample_key.as_str()) {
                error(
                    issues,
                    "duplicate_sample_reference",
                    format!("{sequence_path}.frames[{frame_index}].sample_key"),
                    "a frame sample may be referenced by only one sequence position",
                );
            }
            match frames_by_key.get(&frame_ref.sample_key) {
                Some(frame) => frames.push(frame.clone()),
                None => error(
                    issues,
                    "missing_referenced_sample",
                    format!("{sequence_path}.frames[{frame_index}].sample_key"),
                    format!(
                        "sample {:?} has no parsed frame record",
                        frame_ref.sample_key
                    ),
                ),
            }
        }
        append_report(
            issues,
            &sequence_path,
            validator.validate_sequence(sequence, &frames),
        );
    }

    for sample_key in frames_by_key.keys() {
        if !referenced_samples.contains(sample_key.as_str()) {
            error(
                issues,
                "unreferenced_sample",
                format!("samples.{sample_key}"),
                "parsed frame record is absent from every sequence",
            );
        }
    }
}

fn validate_statistics(
    expected: DatasetStatistics,
    observed: DatasetStatistics,
    issues: &mut Vec<ValidationIssue>,
) {
    for (name, expected_count, observed_count) in [
        ("train_frames", expected.train_frames, observed.train_frames),
        (
            "validation_frames",
            expected.validation_frames,
            observed.validation_frames,
        ),
        ("test_frames", expected.test_frames, observed.test_frames),
        ("sequences", expected.sequences, observed.sequences),
    ] {
        if expected_count != observed_count {
            error(
                issues,
                "manifest_statistics_mismatch",
                format!("dataset.json.statistics.{name}"),
                format!("manifest declares {expected_count}, observed {observed_count}"),
            );
        }
    }
}

fn append_report(issues: &mut Vec<ValidationIssue>, prefix: &str, report: ValidationReport) {
    issues.extend(report.issues.into_iter().map(|mut issue| {
        issue.path = if issue.path.is_empty() {
            prefix.to_owned()
        } else {
            format!("{prefix}.{}", issue.path)
        };
        issue
    }));
}

fn error(
    issues: &mut Vec<ValidationIssue>,
    code: &str,
    path: impl Into<String>,
    message: impl Into<String>,
) {
    issue(issues, Severity::Error, code, path, message);
}

fn warning(
    issues: &mut Vec<ValidationIssue>,
    code: &str,
    path: impl Into<String>,
    message: impl Into<String>,
) {
    issue(issues, Severity::Warning, code, path, message);
}

fn issue(
    issues: &mut Vec<ValidationIssue>,
    severity: Severity,
    code: &str,
    path: impl Into<String>,
    message: impl Into<String>,
) {
    issues.push(ValidationIssue {
        severity,
        code: code.to_owned(),
        path: path.into(),
        message: message.into(),
    });
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T> {
    let file = File::open(path).with_context(|| format!("failed to open {}", path.display()))?;
    serde_json::from_reader(BufReader::new(file))
        .with_context(|| format!("failed to parse {}", path.display()))
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut encoded = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut encoded, "{byte:02x}").expect("writing to String is infallible");
    }
    encoded
}

fn increment_split_count(statistics: &mut DatasetStatistics, split: DatasetSplit) {
    match split {
        DatasetSplit::Train => statistics.train_frames += 1,
        DatasetSplit::Validation => statistics.validation_frames += 1,
        DatasetSplit::Test => statistics.test_frames += 1,
    }
}

fn total_frames(statistics: DatasetStatistics) -> u64 {
    statistics.train_frames + statistics.validation_frames + statistics.test_frames
}

fn observed_frame_count(statistics: DatasetStatistics) -> usize {
    usize::try_from(total_frames(statistics)).unwrap_or(usize::MAX)
}

fn relative_display(dataset: &Path, path: &Path) -> String {
    path.strip_prefix(dataset)
        .unwrap_or(path)
        .display()
        .to_string()
}

#[cfg(test)]
mod tests {
    use std::{
        fs,
        io::{BufWriter, Cursor},
    };

    use synth_data::{
        AssetRef, DenseLabelRefs, FrameAssets, FrameIdentity, GeneratorConfig, GeneratorDescriptor,
        LabelClass, LocatorLabel, SequenceRequest, SequenceSampler, TargetKind,
    };
    use tempfile::TempDir;

    use super::*;
    use crate::shard::{Artifact, ShardWriter};

    struct Fixture {
        _directory: TempDir,
        root: PathBuf,
        manifest: DatasetManifest,
        sequence: SequenceRecord,
        frame: FrameRecord,
        rgb: Vec<u8>,
    }

    impl Fixture {
        fn new() -> Self {
            let directory = tempfile::tempdir().unwrap();
            let root = directory.path().to_path_buf();
            let mut config = GeneratorConfig::default();
            config.sequence.frame_count = 1;
            let sampler = SequenceSampler::new(config).unwrap();
            let plan = sampler
                .sample(SequenceRequest::procedural(
                    "classic_two_stage",
                    42,
                    TargetKind::Negative,
                ))
                .unwrap();
            let split = plan.split(&Default::default()).unwrap();
            let sequence = plan.clone().into_record(&Default::default()).unwrap();
            let frame_plan = plan.frames[0];
            let sample_key = plan.frame_key(0).unwrap();
            let rgb = b"deterministic-rgb-placeholder".to_vec();
            let mut rgb_ref = AssetRef::new(format!("{sample_key}.rgb.jpg"), "image/jpeg", "jpeg");
            rgb_ref.content_hash = Some(format!("sha256:{}", sha256_hex(&rgb)));
            let frame = FrameRecord::new(
                FrameIdentity::new(
                    &sample_key,
                    &plan.sequence_id,
                    frame_plan.frame_index,
                    frame_plan.timestamp_ns,
                ),
                split,
                frame_plan.camera,
                LocatorLabel {
                    target_kind: TargetKind::Negative,
                    bounding_box: None,
                    amodal_bounding_box: None,
                    visible_fraction: 0.0,
                    occluded_fraction: 0.0,
                    truncated: false,
                },
                FrameAssets {
                    rgb: rgb_ref,
                    surface_normals: None,
                    motion_vectors: None,
                },
            );
            let mut manifest = DatasetManifest::new(
                "validator-test",
                GeneratorDescriptor::chacha20(
                    "roof-synth-test",
                    "0.1.0",
                    sampler.config_fingerprint(),
                ),
                42,
            );
            increment_split_count(&mut manifest.statistics, split);
            manifest.statistics.sequences = 1;

            Self {
                _directory: directory,
                root,
                manifest,
                sequence,
                frame,
                rgb,
            }
        }

        fn write(&self, include_rgb: bool) {
            write_json(&self.root.join("dataset.json"), &self.manifest);
            write_json(
                &self.root.join("sequences.json"),
                &vec![self.sequence.clone()],
            );
            let labels = serde_json::to_vec(&self.frame).unwrap();
            let prefix = match self.frame.split {
                DatasetSplit::Train => "train",
                DatasetSplit::Validation => "validation",
                DatasetSplit::Test => "test",
            };
            let mut writer = ShardWriter::new(&self.root, prefix, 16).unwrap();
            if include_rgb {
                writer
                    .append(
                        &self.frame.sample_key,
                        &[
                            Artifact {
                                suffix: "rgb.jpg",
                                bytes: &self.rgb,
                            },
                            Artifact {
                                suffix: "labels.json",
                                bytes: &labels,
                            },
                        ],
                    )
                    .unwrap();
            } else {
                writer
                    .append(
                        &self.frame.sample_key,
                        &[Artifact {
                            suffix: "labels.json",
                            bytes: &labels,
                        }],
                    )
                    .unwrap();
            }
            writer.finish().unwrap();
        }
    }

    fn write_json(path: &Path, value: &impl Serialize) {
        let file = File::create(path).unwrap();
        serde_json::to_writer_pretty(BufWriter::new(file), value).unwrap();
    }

    fn has_issue(summary: &ValidationSummary, code: &str) -> bool {
        summary.issues.iter().any(|issue| issue.code == code)
    }

    fn png(image: image::DynamicImage) -> Vec<u8> {
        let mut output = Cursor::new(Vec::new());
        image
            .write_to(&mut output, image::ImageFormat::Png)
            .unwrap();
        output.into_inner()
    }

    #[test]
    fn accepts_a_complete_dataset_and_reconstructs_statistics() {
        let fixture = Fixture::new();
        fixture.write(true);

        let summary = validate_dataset(&fixture.root).unwrap();

        assert!(summary.valid, "{:?}", summary.issues);
        assert_eq!(summary.shard_count, 1);
        assert_eq!(summary.sample_count, 1);
        assert_eq!(summary.artifact_count, 2);
        assert_eq!(summary.frame_count, 1);
        assert_eq!(summary.sequence_count, 1);
        assert_eq!(summary.expected_statistics, summary.observed_statistics);
    }

    #[test]
    fn reports_missing_referenced_artifacts() {
        let fixture = Fixture::new();
        fixture.write(false);

        let summary = validate_dataset(&fixture.root).unwrap();

        assert!(!summary.valid);
        assert!(has_issue(&summary, "missing_referenced_asset"));
    }

    #[test]
    fn reports_hash_and_manifest_statistics_mismatches() {
        let mut fixture = Fixture::new();
        fixture.frame.assets.rgb.content_hash = Some(format!("sha256:{}", "0".repeat(64)));
        fixture.manifest.statistics.train_frames += 7;
        fixture.write(true);

        let summary = validate_dataset(&fixture.root).unwrap();

        assert!(!summary.valid);
        assert!(has_issue(&summary, "content_hash_mismatch"));
        assert!(has_issue(&summary, "manifest_statistics_mismatch"));
    }

    #[test]
    fn reports_duplicate_artifacts_and_samples_spanning_shards() {
        let fixture = Fixture::new();
        fixture.write(true);
        let labels = serde_json::to_vec(&fixture.frame).unwrap();
        let mut duplicate = ShardWriter::new(&fixture.root, "duplicate", 16).unwrap();
        duplicate
            .append(
                &fixture.frame.sample_key,
                &[Artifact {
                    suffix: "labels.json",
                    bytes: &labels,
                }],
            )
            .unwrap();
        duplicate.finish().unwrap();

        let summary = validate_dataset(&fixture.root).unwrap();

        assert!(!summary.valid);
        assert!(has_issue(&summary, "sample_spans_shards"));
        assert!(has_issue(&summary, "duplicate_artifact"));
    }

    #[test]
    fn reports_sequence_references_to_missing_samples() {
        let mut fixture = Fixture::new();
        fixture.sequence.frames[0].sample_key = "missing_sample".to_owned();
        fixture.write(true);

        let summary = validate_dataset(&fixture.root).unwrap();

        assert!(!summary.valid);
        assert!(has_issue(&summary, "missing_referenced_sample"));
        assert!(has_issue(&summary, "unreferenced_sample"));
    }

    #[test]
    fn required_json_files_are_hard_errors() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("dataset.json"), b"not-json").unwrap();

        assert!(validate_dataset(directory.path()).is_err());
    }

    #[test]
    fn dense_validation_rejects_unknown_face_ids() {
        let mut fixture = Fixture::new();
        fixture.frame.camera.intrinsics.width = 1;
        fixture.frame.camera.intrinsics.height = 1;
        fixture.manifest.labels.parts = vec![LabelClass {
            id: 1,
            name: "lower_skirt".to_owned(),
        }];
        fixture.manifest.labels.faces = vec![LabelClass {
            id: roof_geometry::FaceId::LowerFront.as_u32() as u16,
            name: "lower_front".to_owned(),
        }];
        let sample_key = fixture.frame.sample_key.clone();
        let reference = |suffix: &str, encoding: &str| {
            AssetRef::new(
                format!("{sample_key}.{suffix}"),
                if suffix.ends_with("png") {
                    "image/png"
                } else {
                    "application/octet-stream"
                },
                encoding,
            )
        };
        fixture.frame.labels.dense = DenseLabelRefs {
            roof_mask: Some(reference("roof_mask.png", "uint8")),
            amodal_roof_mask: Some(reference("amodal_roof_mask.png", "uint8")),
            part_mask: Some(reference("part_mask.png", "uint16")),
            face_id_map: Some(reference("face_ids.png", "uint16")),
            face_coordinates: Some(reference("facecoords.bin.zst", "rg16float-zstd")),
        };
        let mut sample = SampleArtifacts::default();
        sample.files.insert(
            format!("{sample_key}.roof_mask.png"),
            png(image::DynamicImage::ImageLuma8(
                image::GrayImage::from_raw(1, 1, vec![255]).unwrap(),
            )),
        );
        sample.files.insert(
            format!("{sample_key}.amodal_roof_mask.png"),
            png(image::DynamicImage::ImageLuma8(
                image::GrayImage::from_raw(1, 1, vec![255]).unwrap(),
            )),
        );
        sample.files.insert(
            format!("{sample_key}.part_mask.png"),
            png(image::DynamicImage::ImageLuma16(
                image::ImageBuffer::from_raw(1, 1, vec![1_u16]).unwrap(),
            )),
        );
        sample.files.insert(
            format!("{sample_key}.face_ids.png"),
            png(image::DynamicImage::ImageLuma16(
                image::ImageBuffer::from_raw(1, 1, vec![999_u16]).unwrap(),
            )),
        );
        let coordinate_bytes = [
            half::f16::from_f32(0.5).to_bits().to_le_bytes(),
            half::f16::from_f32(0.5).to_bits().to_le_bytes(),
        ]
        .concat();
        sample.files.insert(
            format!("{sample_key}.facecoords.bin.zst"),
            zstd::stream::encode_all(Cursor::new(coordinate_bytes), 1).unwrap(),
        );

        let mut issues = Vec::new();
        validate_dense_targets(
            &sample_key,
            &sample,
            &fixture.frame,
            &fixture.manifest,
            &mut issues,
        );
        assert!(
            issues
                .iter()
                .any(|issue| issue.code == "unknown_face_mask_id")
        );
        assert!(
            issues
                .iter()
                .any(|issue| issue.code == "dense_target_semantic_mismatch")
        );
    }
}
