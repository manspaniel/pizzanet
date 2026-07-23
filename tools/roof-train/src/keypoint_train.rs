//! Training loop for the sparse amodal-keypoint observation network.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, HashMap},
    f64::consts::PI,
    fs,
    path::{Path, PathBuf},
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering as AtomicOrdering},
    },
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result, bail};
use burn::{
    module::{AutodiffModule, Module},
    optim::{AdamWConfig, GradientsParams, Optimizer},
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::{TensorData, activation::log_sigmoid, backend::AutodiffBackend},
};
use rand::{SeedableRng, seq::SliceRandom};
use rand_chacha::ChaCha8Rng;
use rayon::prelude::*;
use roof_model::{
    DEFAULT_FIT_KEYPOINT_CONFIDENCE, DEFAULT_OFFSCREEN_THRESHOLD, HEATMAP_SIZE, KEYPOINT_COUNT,
    KeypointRoofGeometryOutput, KeypointRoofNet, KeypointRoofNetConfig, KeypointRoofOutput,
    KeypointTrainingOptions, LetterboxTransform, SPATIAL_INPUT_SIZE, prepare_rgb8_sized,
};
use roof_training::{
    KEYPOINT_DISTRIBUTION_SIZE, OFFSCREEN_INDEX, SYMMETRY_COUNT, build_keypoint_target,
    flip_target_horizontal, symmetry_target_slot,
};
use serde::{Deserialize, Serialize};
use synth_data::FrameRecord;

use super::{
    Args, ImageSource, SampleOrigin, Split, TrainingSample, append_json_line,
    augmentation::{
        resize_to_working_raster, rotate_frame_keypoints, rotate_rgb_reflect, training_roll_radians,
    },
    fit_evaluation::{SyntheticFitMetrics, evaluate_synthetic_fit},
    sample_hash, sha256_file,
};

const PCK_STRICT_THRESHOLD: f32 = 0.03;
const PCK_STANDARD_THRESHOLD: f32 = 0.05;
// The 4,096 spatial classes jointly encode `in frame`, while the final class
// encodes `offscreen`. The categorical KL remains the primary objective; this
// auxiliary state term teaches that union explicitly so a broad spatial
// distribution cannot be calibrated like 4,096 unrelated negative classes.
const OFFSCREEN_STATE_LOSS_WEIGHT: f32 = 0.25;
// Presence is the deployment gate and is the only supervision supplied by the
// real photographs. Giving it more gradient budget prevents the synthetic-only
// geometry objective from moving the shared backbone into a renderer-specific
// feature space after the classifier has already separated the real images.
pub(super) const GEOMETRY_LOSS_WEIGHT: f32 = 0.5;
// A normal batch represents all four sources. Real photographs receive two
// thirds of the normalized presence mass; synthetic targets and negatives
// retain one third and continue to provide all geometry supervision.
pub(super) const PRESENCE_SOURCE_MASS: [f32; 4] = [0.5, 0.5, 1.0, 1.0];
// The v14 pairwise experiment is retained for reproducible diagnostics, but
// v15 disables it: the tiny real corpus makes batch-pair rankings too easy to
// overfit. Source-balanced BCE remains the complete presence objective.
pub(super) const REAL_PRESENCE_PAIRWISE_LOSS_WEIGHT: f32 = 0.0;
pub(super) const REAL_PRESENCE_PAIRWISE_MARGIN: f32 = 0.50;
pub(super) const REAL_POSITIVE_SAMPLING_STRATEGY: &str =
    "physical_building_balanced_deterministic_view_cycle/v1";
const REAL_POSITIVE_SAMPLING_SEED_SALT: u64 = 0x7265_616c_2d70_6f73;
pub(super) const PRESENCE_THRESHOLD_CALIBRATION: &str =
    "real_validation_building_and_view_recall_constrained/v1";
pub(super) const PRESENCE_FREEZE_POLICY: &str =
    "consecutive_real_and_synthetic_safe_then_geometry_only/v1";
pub(super) const MIN_REAL_POSITIVE_RECALL: f32 = 0.80;
pub(super) const MIN_REAL_POSITIVE_VIEW_RECALL: f32 = 0.75;
pub(super) const MIN_REAL_POSITIVE_BUILDING_RECALL: f32 = 0.80;
pub(super) const MIN_REAL_NEGATIVE_SPECIFICITY: f32 = 0.85;
pub(super) const MIN_SYNTHETIC_ROC_AUC: f32 = 0.95;
pub(super) const MIN_SYNTHETIC_AVERAGE_PRECISION: f32 = 0.95;
pub(super) const MIN_STANDARD_PCK: f32 = 0.90;
pub(super) const MIN_OFFSCREEN_ACCURACY: f32 = 0.90;
const MIN_AGGREGATE_TARGET_RECALL: f32 = 0.95;
const MIN_AGGREGATE_SPECIFICITY: f32 = 0.90;
const MAX_MEDIAN_FITTED_MESH_RMSE: f32 = 0.08;
const MIN_MEDIAN_AMODAL_SILHOUETTE_IOU: f32 = 0.50;
pub(super) const CHECKPOINT_SELECTION_POLICY: &str = "hard_gates_then_bucketed_real_robustness/v1";
pub(super) const CHECKPOINT_REAL_ROBUSTNESS_WEIGHTS: [f32; 4] = [0.40, 0.35, 0.20, 0.05];
pub(super) const REAL_ROBUSTNESS_BAND_BASIS_POINTS: u16 = 200;
pub(super) const REAL_GATE_MARGIN_BAND_BASIS_POINTS: u16 = 200;
pub(super) const SYNTHETIC_MARGIN_BAND_BASIS_POINTS: u16 = 100;
pub(super) const SYNTHETIC_QUALITY_BAND_BASIS_POINTS: u16 = 100;
// Loss reporting is telemetry only. Synchronizing a device scalar stalls the
// asynchronous backend, so align it with the normal 100-batch progress report.
const LOSS_SYNC_INTERVAL: usize = 100;
static PROMOTION_TEMP_ID: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum OptimizerStage {
    Joint,
    JointBackboneFrozen,
    GeometryOnlyPresenceLocked,
}

impl OptimizerStage {
    fn label(self) -> &'static str {
        match self {
            Self::Joint => "joint",
            Self::JointBackboneFrozen => "joint_backbone_frozen",
            Self::GeometryOnlyPresenceLocked => "geometry_only_presence_locked",
        }
    }
}

/// Sticky, validation-driven transition from joint training to geometry-only
/// refinement. A safe validation at epoch N can only affect epoch N+1.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PresenceLockState {
    required_safe_epochs: usize,
    safe_streak: usize,
    frozen_after_epoch: Option<usize>,
}

impl PresenceLockState {
    fn new(required_safe_epochs: usize) -> Self {
        Self {
            required_safe_epochs,
            safe_streak: 0,
            frozen_after_epoch: None,
        }
    }

    fn restore_locked(
        required_safe_epochs: usize,
        safe_streak: usize,
        frozen_after_epoch: usize,
    ) -> Self {
        Self {
            required_safe_epochs,
            safe_streak,
            frozen_after_epoch: Some(frozen_after_epoch),
        }
    }

    fn presence_trainable(self) -> bool {
        self.frozen_after_epoch.is_none()
    }

    /// Record the just-finished validation and report whether the sticky lock
    /// was entered. A zero-length policy remains disabled forever.
    fn observe_validation(&mut self, epoch: usize, real_safe: bool, synthetic_safe: bool) -> bool {
        if self.required_safe_epochs == 0 || self.frozen_after_epoch.is_some() {
            return false;
        }
        if real_safe && synthetic_safe {
            self.safe_streak = self.safe_streak.saturating_add(1);
            if self.safe_streak >= self.required_safe_epochs {
                self.frozen_after_epoch = Some(epoch);
                return true;
            }
        } else {
            self.safe_streak = 0;
        }
        false
    }
}

/// Training inputs and optimizer semantics that must remain identical when a
/// checkpoint is continued as geometry-only refinement. Unknown config fields
/// are intentionally ignored so this reader can authenticate both the
/// original joint run and a later refinement run without weakening the fields
/// that affect model updates.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct RefinementTrainingContract {
    schema_version: String,
    synthetic: String,
    negatives: String,
    real_positives: String,
    real_positive_repeat: usize,
    real_positive_sampling_strategy: String,
    epochs: usize,
    batch_size: usize,
    evaluation_batch_size: usize,
    head_learning_rate: f64,
    backbone_learning_rate: f64,
    backbone_freeze_epochs: usize,
    freeze_backbone_batch_norm: bool,
    detach_geometry_backbone: bool,
    presence_freeze_policy: String,
    presence_freeze_after_safe_epochs: usize,
    weight_decay: f64,
    warmup_fraction: f64,
    geometry_loss_weight: f32,
    presence_source_mass: [f32; 4],
    real_presence_pairwise_loss_weight: f32,
    real_presence_pairwise_margin: f32,
    seed: u64,
    limit_per_split: Option<usize>,
    disable_augmentation: bool,
    overfit: bool,
    input_size: usize,
    heatmap_size: usize,
    keypoint_count: usize,
    pretrained_backbone: String,
    train_samples: usize,
    train_real_positive_buildings: usize,
    validation_samples: usize,
    test_samples: usize,
}

/// The completed epoch is the authority for optimizer offsets and the sticky
/// presence lock. Reconstructing these values from the current dataset would
/// silently change the learning-rate phase after even a small data mismatch.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
struct RefinementEpochContract {
    epoch: usize,
    presence_freeze_policy: String,
    presence_freeze_after_safe_epochs: usize,
    presence_safe_streak: usize,
    presence_frozen_after_epoch: Option<usize>,
    presence_frozen_from_epoch: Option<usize>,
    presence_trainable: bool,
    optimizer_stage: OptimizerStage,
    backbone_trainable: bool,
    head_updates_completed: usize,
    head_total_updates: usize,
    joint_head_updates_completed: usize,
    geometry_only_head_updates_completed: usize,
    backbone_updates_completed: usize,
    backbone_total_updates: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
struct RefinementDerivedUpdateContract {
    batches_per_epoch: usize,
    head_total_updates: usize,
    backbone_total_updates: usize,
    head_updates_completed_at_source: usize,
    joint_head_updates_completed_at_source: usize,
    geometry_only_head_updates_completed_at_source: usize,
    backbone_updates_completed_at_source: usize,
}

#[derive(Clone, Debug, Serialize)]
struct AuthenticatedRefinementSource {
    source_checkpoint: String,
    source_checkpoint_sha256: String,
    source_training_config: String,
    source_training_config_sha256: String,
    source_metrics: String,
    source_metrics_sha256: String,
    source_metrics_line: usize,
    training_contract: RefinementTrainingContract,
    epoch_contract: RefinementEpochContract,
    derived_update_contract: RefinementDerivedUpdateContract,
}

fn require_contract_match<T>(field: &str, source: &T, current: &T) -> Result<()>
where
    T: PartialEq + std::fmt::Debug,
{
    if source != current {
        bail!(
            "geometry-refinement source contract mismatch for {field}: source={source:?} current={current:?}"
        );
    }
    Ok(())
}

fn refinement_metrics_path(checkpoint_parent: &Path) -> Result<PathBuf> {
    let canonical = checkpoint_parent.join("metrics.jsonl");
    if canonical.is_file() {
        return Ok(canonical);
    }

    let mut fallbacks = fs::read_dir(checkpoint_parent)
        .with_context(|| {
            format!(
                "read refinement checkpoint directory {}",
                checkpoint_parent.display()
            )
        })?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.is_file()
                && path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .is_some_and(|name| name.starts_with("metrics") && name.ends_with(".jsonl"))
        })
        .collect::<Vec<_>>();
    fallbacks.sort();
    match fallbacks.as_slice() {
        [only] => Ok(only.clone()),
        [] => bail!(
            "refinement checkpoint directory {} has no metrics.jsonl or metrics*.jsonl fallback",
            checkpoint_parent.display()
        ),
        _ => bail!(
            "refinement checkpoint directory {} has multiple metrics*.jsonl fallbacks; expected exactly one: {}",
            checkpoint_parent.display(),
            fallbacks
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn load_refinement_epoch_contract(
    metrics_path: &Path,
    source_epoch: usize,
) -> Result<(usize, RefinementEpochContract)> {
    let contents = fs::read_to_string(metrics_path)
        .with_context(|| format!("read refinement metrics {}", metrics_path.display()))?;
    let mut selected = None;
    let mut previous_epoch = None;
    for (line_index, line) in contents.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let record = serde_json::from_str::<RefinementEpochContract>(line).with_context(|| {
            format!(
                "parse refinement metrics {} line {}",
                metrics_path.display(),
                line_index + 1
            )
        })?;
        if let Some(previous_epoch) = previous_epoch
            && record.epoch <= previous_epoch
        {
            bail!(
                "refinement metrics {} are not strictly increasing at line {} (epoch {} after {})",
                metrics_path.display(),
                line_index + 1,
                record.epoch,
                previous_epoch
            );
        }
        previous_epoch = Some(record.epoch);
        if record.epoch == source_epoch {
            if selected.is_some() {
                bail!(
                    "refinement metrics {} contain duplicate epoch {} records",
                    metrics_path.display(),
                    source_epoch
                );
            }
            selected = Some((line_index + 1, record));
        }
    }
    let last_epoch = previous_epoch.with_context(|| {
        format!(
            "refinement metrics {} contain no epoch records",
            metrics_path.display()
        )
    })?;
    require_contract_match(
        "source epoch vs last metrics epoch",
        &source_epoch,
        &last_epoch,
    )?;
    selected.with_context(|| {
        format!(
            "refinement metrics {} have no source epoch {} record",
            metrics_path.display(),
            source_epoch
        )
    })
}

fn authenticate_refinement_source(
    args: &Args,
    checkpoint: &Path,
    source_epoch: usize,
    train: &[TrainingSample],
    validation: &[TrainingSample],
    test: &[TrainingSample],
) -> Result<AuthenticatedRefinementSource> {
    let checkpoint_parent = checkpoint
        .parent()
        .context("refinement checkpoint has no parent directory")?;
    let config_path = checkpoint_parent.join("training-config.json");
    let source_config = serde_json::from_slice::<RefinementTrainingContract>(
        &fs::read(&config_path)
            .with_context(|| format!("read refinement config {}", config_path.display()))?,
    )
    .with_context(|| format!("parse refinement config {}", config_path.display()))?;
    let metrics_path = refinement_metrics_path(checkpoint_parent)?;
    let (metrics_line, epoch_record) = load_refinement_epoch_contract(&metrics_path, source_epoch)?;

    require_contract_match(
        "schema_version",
        &source_config.schema_version,
        &"roof-training/v15".to_owned(),
    )?;
    require_contract_match(
        "synthetic dataset path",
        &source_config.synthetic,
        &args.synthetic.display().to_string(),
    )?;
    require_contract_match(
        "negative dataset path",
        &source_config.negatives,
        &args.negatives.display().to_string(),
    )?;
    require_contract_match(
        "real-positive dataset path",
        &source_config.real_positives,
        &args.real_positives.display().to_string(),
    )?;
    require_contract_match(
        "real_positive_repeat",
        &source_config.real_positive_repeat,
        &args.real_positive_repeat,
    )?;
    require_contract_match(
        "real_positive_sampling_strategy",
        &source_config.real_positive_sampling_strategy,
        &REAL_POSITIVE_SAMPLING_STRATEGY.to_owned(),
    )?;
    require_contract_match("epochs", &source_config.epochs, &args.epochs)?;
    require_contract_match("batch_size", &source_config.batch_size, &args.batch_size)?;
    require_contract_match(
        "evaluation_batch_size",
        &source_config.evaluation_batch_size,
        &args.evaluation_batch_size(),
    )?;
    require_contract_match(
        "head_learning_rate",
        &source_config.head_learning_rate,
        &args.head_learning_rate,
    )?;
    require_contract_match(
        "backbone_learning_rate",
        &source_config.backbone_learning_rate,
        &args.backbone_learning_rate,
    )?;
    require_contract_match(
        "backbone_freeze_epochs",
        &source_config.backbone_freeze_epochs,
        &args.backbone_freeze_epochs,
    )?;
    require_contract_match(
        "freeze_backbone_batch_norm",
        &source_config.freeze_backbone_batch_norm,
        &args.freeze_backbone_batch_norm,
    )?;
    require_contract_match(
        "detach_geometry_backbone",
        &source_config.detach_geometry_backbone,
        &args.detach_geometry_backbone,
    )?;
    require_contract_match(
        "presence_freeze_policy",
        &source_config.presence_freeze_policy,
        &PRESENCE_FREEZE_POLICY.to_owned(),
    )?;
    require_contract_match(
        "presence_freeze_after_safe_epochs",
        &source_config.presence_freeze_after_safe_epochs,
        &args.presence_freeze_after_safe_epochs,
    )?;
    require_contract_match(
        "weight_decay",
        &source_config.weight_decay,
        &args.weight_decay,
    )?;
    require_contract_match(
        "warmup_fraction",
        &source_config.warmup_fraction,
        &args.warmup_fraction,
    )?;
    require_contract_match(
        "geometry_loss_weight",
        &source_config.geometry_loss_weight,
        &GEOMETRY_LOSS_WEIGHT,
    )?;
    require_contract_match(
        "presence_source_mass",
        &source_config.presence_source_mass,
        &PRESENCE_SOURCE_MASS,
    )?;
    require_contract_match(
        "real_presence_pairwise_loss_weight",
        &source_config.real_presence_pairwise_loss_weight,
        &REAL_PRESENCE_PAIRWISE_LOSS_WEIGHT,
    )?;
    require_contract_match(
        "real_presence_pairwise_margin",
        &source_config.real_presence_pairwise_margin,
        &REAL_PRESENCE_PAIRWISE_MARGIN,
    )?;
    require_contract_match("seed", &source_config.seed, &args.seed)?;
    require_contract_match(
        "limit_per_split",
        &source_config.limit_per_split,
        &args.limit_per_split,
    )?;
    require_contract_match(
        "disable_augmentation",
        &source_config.disable_augmentation,
        &args.disable_augmentation,
    )?;
    require_contract_match("overfit", &source_config.overfit, &args.overfit)?;
    require_contract_match("input_size", &source_config.input_size, &SPATIAL_INPUT_SIZE)?;
    require_contract_match("heatmap_size", &source_config.heatmap_size, &HEATMAP_SIZE)?;
    require_contract_match(
        "keypoint_count",
        &source_config.keypoint_count,
        &KEYPOINT_COUNT,
    )?;
    require_contract_match(
        "pretrained_backbone",
        &source_config.pretrained_backbone,
        &"torchvision MobileNetV2 ImageNet1K V2".to_owned(),
    )?;
    require_contract_match("train_samples", &source_config.train_samples, &train.len())?;
    require_contract_match(
        "train_real_positive_buildings",
        &source_config.train_real_positive_buildings,
        &super::real_positive_building_count(train)?,
    )?;
    require_contract_match(
        "validation_samples",
        &source_config.validation_samples,
        &validation.len(),
    )?;
    require_contract_match("test_samples", &source_config.test_samples, &test.len())?;

    require_contract_match(
        "epoch record source epoch",
        &epoch_record.epoch,
        &source_epoch,
    )?;
    require_contract_match(
        "epoch presence_freeze_policy",
        &epoch_record.presence_freeze_policy,
        &source_config.presence_freeze_policy,
    )?;
    require_contract_match(
        "epoch presence_freeze_after_safe_epochs",
        &epoch_record.presence_freeze_after_safe_epochs,
        &source_config.presence_freeze_after_safe_epochs,
    )?;
    let frozen_after_epoch = epoch_record.presence_frozen_after_epoch.context(
        "refinement source epoch has no authenticated presence lock; continue joint training instead",
    )?;
    if frozen_after_epoch > source_epoch {
        bail!(
            "refinement source lock epoch {frozen_after_epoch} follows source epoch {source_epoch}"
        );
    }
    require_contract_match(
        "presence_frozen_from_epoch",
        &epoch_record.presence_frozen_from_epoch,
        &Some(frozen_after_epoch.saturating_add(1)),
    )?;
    if source_config.presence_freeze_after_safe_epochs > 0
        && epoch_record.presence_safe_streak < source_config.presence_freeze_after_safe_epochs
    {
        bail!(
            "refinement source safe streak {} is below the configured lock requirement {}",
            epoch_record.presence_safe_streak,
            source_config.presence_freeze_after_safe_epochs
        );
    }

    let batches_per_epoch = balanced_epoch_len(train, args.batch_size)
        .div_ceil(args.batch_size)
        .max(1);
    let head_total_updates = args
        .epochs
        .checked_mul(batches_per_epoch)
        .context("refinement head-update contract overflow")?;
    let backbone_total_updates = args
        .epochs
        .saturating_sub(args.backbone_freeze_epochs)
        .checked_mul(batches_per_epoch)
        .context("refinement backbone-update contract overflow")?;
    let head_updates_completed_at_source = source_epoch
        .checked_mul(batches_per_epoch)
        .context("refinement completed head-update contract overflow")?;
    let joint_epochs = frozen_after_epoch.min(source_epoch);
    let joint_head_updates_completed_at_source = joint_epochs
        .checked_mul(batches_per_epoch)
        .context("refinement completed joint-update contract overflow")?;
    let geometry_only_head_updates_completed_at_source = source_epoch
        .saturating_sub(joint_epochs)
        .checked_mul(batches_per_epoch)
        .context("refinement completed geometry-update contract overflow")?;
    let backbone_updates_completed_at_source = joint_epochs
        .saturating_sub(args.backbone_freeze_epochs)
        .checked_mul(batches_per_epoch)
        .context("refinement completed backbone-update contract overflow")?;
    let derived_update_contract = RefinementDerivedUpdateContract {
        batches_per_epoch,
        head_total_updates,
        backbone_total_updates,
        head_updates_completed_at_source,
        joint_head_updates_completed_at_source,
        geometry_only_head_updates_completed_at_source,
        backbone_updates_completed_at_source,
    };
    require_contract_match(
        "head_total_updates",
        &epoch_record.head_total_updates,
        &derived_update_contract.head_total_updates,
    )?;
    require_contract_match(
        "backbone_total_updates",
        &epoch_record.backbone_total_updates,
        &derived_update_contract.backbone_total_updates,
    )?;
    require_contract_match(
        "head_updates_completed",
        &epoch_record.head_updates_completed,
        &derived_update_contract.head_updates_completed_at_source,
    )?;
    require_contract_match(
        "joint_head_updates_completed",
        &epoch_record.joint_head_updates_completed,
        &derived_update_contract.joint_head_updates_completed_at_source,
    )?;
    require_contract_match(
        "geometry_only_head_updates_completed",
        &epoch_record.geometry_only_head_updates_completed,
        &derived_update_contract.geometry_only_head_updates_completed_at_source,
    )?;
    require_contract_match(
        "backbone_updates_completed",
        &epoch_record.backbone_updates_completed,
        &derived_update_contract.backbone_updates_completed_at_source,
    )?;
    let source_epoch_presence_trainable = source_epoch <= frozen_after_epoch;
    require_contract_match(
        "source epoch presence_trainable",
        &epoch_record.presence_trainable,
        &source_epoch_presence_trainable,
    )?;
    let expected_optimizer_stage = if !source_epoch_presence_trainable {
        OptimizerStage::GeometryOnlyPresenceLocked
    } else if source_epoch > args.backbone_freeze_epochs {
        OptimizerStage::Joint
    } else {
        OptimizerStage::JointBackboneFrozen
    };
    require_contract_match(
        "source epoch optimizer_stage",
        &epoch_record.optimizer_stage,
        &expected_optimizer_stage,
    )?;
    require_contract_match(
        "source epoch backbone_trainable",
        &epoch_record.backbone_trainable,
        &(source_epoch_presence_trainable && source_epoch > args.backbone_freeze_epochs),
    )?;

    let canonical_string = |path: &Path| {
        path.canonicalize()
            .unwrap_or_else(|_| path.to_path_buf())
            .display()
            .to_string()
    };
    Ok(AuthenticatedRefinementSource {
        source_checkpoint: canonical_string(checkpoint),
        source_checkpoint_sha256: sha256_file(checkpoint)?,
        source_training_config: canonical_string(&config_path),
        source_training_config_sha256: sha256_file(&config_path)?,
        source_metrics: canonical_string(&metrics_path),
        source_metrics_sha256: sha256_file(&metrics_path)?,
        source_metrics_line: metrics_line,
        training_contract: source_config,
        epoch_contract: epoch_record,
        derived_update_contract,
    })
}

fn next_patience_count(
    current: usize,
    checkpoint_improved: bool,
    presence_lock_transitioned: bool,
    epoch_consumes_patience: bool,
) -> usize {
    if checkpoint_improved || presence_lock_transitioned {
        0
    } else if epoch_consumes_patience {
        current.saturating_add(1)
    } else {
        current
    }
}

pub(super) fn train_model<B: AutodiffBackend>(
    args: &Args,
    train: Vec<TrainingSample>,
    validation: Vec<TrainingSample>,
    test: Vec<TrainingSample>,
    device: B::Device,
) -> Result<()> {
    B::seed(&device, args.seed);
    // These diagnostic candidates are always regenerated from the current
    // run. Remove stale copies before initialization so a failed startup can
    // never leave an older checkpoint looking current.
    for filename in ["candidate-presence.mpk", "candidate-geometry.mpk"] {
        let path = args.artifacts.join(filename);
        match fs::remove_file(&path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                return Err(error)
                    .with_context(|| format!("remove stale checkpoint {}", path.display()));
            }
        }
    }
    let fit_shape_prior = shape_prior(&train)?.mean;
    let refinement_source = match (
        args.geometry_refine_from.as_deref(),
        args.geometry_refine_source_epoch,
    ) {
        (Some(checkpoint), Some(source_epoch)) => Some((
            checkpoint,
            source_epoch,
            authenticate_refinement_source(
                args,
                checkpoint,
                source_epoch,
                &train,
                &validation,
                &test,
            )?,
        )),
        (None, None) => None,
        _ => bail!("incomplete geometry-refinement source contract reached training"),
    };
    let mut model = if let Some((checkpoint, _, _)) = &refinement_source {
        let record = CompactRecorder::new()
            .load((*checkpoint).to_path_buf(), &device)
            .with_context(|| format!("load refinement checkpoint {}", checkpoint.display()))?;
        KeypointRoofNetConfig::new()
            .init::<B>(&device)
            .load_record(record)
    } else {
        KeypointRoofNetConfig::new()
            .init_pretrained::<B>(&device)
            .context("initialize trainable MobileNetV2 ImageNet weights")?
    };
    let groups = model.parameter_groups();
    let mut backbone_optimizer = AdamWConfig::new()
        .with_weight_decay(args.weight_decay as f32)
        .init();
    let mut head_optimizer = AdamWConfig::new()
        .with_weight_decay(args.weight_decay as f32)
        .init();
    let batches_per_epoch = balanced_epoch_len(&train, args.batch_size)
        .div_ceil(args.batch_size)
        .max(1);
    let total_head_updates = args
        .epochs
        .checked_mul(batches_per_epoch)
        .context("logical head-update count overflow")?;
    let inherited_head_updates = refinement_source
        .as_ref()
        .map(|(_, _, source)| source.epoch_contract.head_updates_completed)
        .unwrap_or(0);
    let mut head_schedule = OptimizerSchedule::new(
        args.head_learning_rate,
        total_head_updates,
        args.warmup_fraction,
    )
    .with_updates_completed(inherited_head_updates)
    .context("inherited head-update offset exceeds the logical training schedule")?;
    let mut backbone_schedule = StagedBackboneSchedule::new(
        args.backbone_learning_rate,
        args.epochs,
        args.backbone_freeze_epochs,
        batches_per_epoch,
        args.warmup_fraction,
    );
    let inherited_backbone_updates = refinement_source
        .as_ref()
        .map(|(_, _, source)| source.epoch_contract.backbone_updates_completed)
        .unwrap_or(0);
    backbone_schedule = backbone_schedule
        .with_updates_completed(inherited_backbone_updates)
        .context("inherited backbone-update offset exceeds the logical training schedule")?;
    let metrics_path = args.artifacts.join("metrics.jsonl");
    fs::write(&metrics_path, [])?;
    let mut best_rank: Option<CheckpointRank> = None;
    let mut best_presence_rank: Option<PresenceCheckpointRank> = None;
    let mut best_geometry_score = f32::NEG_INFINITY;
    let mut epochs_without_improvement = 0usize;
    let mut presence_lock = PresenceLockState::new(args.presence_freeze_after_safe_epochs);
    let mut joint_head_updates_completed = refinement_source
        .as_ref()
        .map(|(_, _, source)| source.epoch_contract.joint_head_updates_completed)
        .unwrap_or(0);
    let mut geometry_only_head_updates_completed = refinement_source
        .as_ref()
        .map(|(_, _, source)| source.epoch_contract.geometry_only_head_updates_completed)
        .unwrap_or(0);
    let working_image_cache = WorkingImageCache::default();

    if let Some((checkpoint, source_epoch, source_contract)) = &refinement_source {
        let validation_metrics = evaluate(
            &model.clone().valid(),
            &validation,
            &device,
            args.evaluation_batch_size(),
            None,
            false,
            Some(fit_shape_prior),
        )?;
        let checkpoint_rank = validation_metrics.checkpoint_rank();
        let selection_score = checkpoint_rank.selection_score();
        let real_presence_safe = validation_metrics.real_presence_gate_passes();
        let synthetic_presence_safe = validation_metrics.synthetic_presence_gate_passes();
        let source_metrics = RefinementSourceMetrics {
            schema_version: "roof-geometry-refinement-source/v2",
            source_checkpoint: source_contract.source_checkpoint.clone(),
            source_epoch: *source_epoch,
            first_training_epoch: args.first_training_epoch(),
            source_contract: source_contract.clone(),
            real_presence_safe,
            synthetic_presence_safe,
            validation: validation_metrics,
            checkpoint_rank,
            selection_score,
        };
        fs::write(
            args.artifacts.join("refinement-source-metrics.json"),
            serde_json::to_vec_pretty(&source_metrics)?,
        )?;
        println!(
            "refinement source epoch {source_epoch:02}: threshold={:.3} real_recall={:.3} real_building_recall={:.3} real_specificity={:.3} synthetic_auc={} synthetic_ap={} pck@.05={:.3} offscreen={:.3} real_gate_passes={real_presence_safe} synthetic_gate_passes={synthetic_presence_safe} score={selection_score:.3}",
            source_metrics.validation.presence_threshold,
            source_metrics.validation.real_positive_recall,
            source_metrics.validation.real_positive_building_recall,
            source_metrics.validation.real_negative_specificity,
            format_optional_metric(
                source_metrics
                    .validation
                    .presence_diagnostics
                    .synthetic_roc_auc
            ),
            format_optional_metric(
                source_metrics
                    .validation
                    .presence_diagnostics
                    .synthetic_average_precision
            ),
            source_metrics.validation.pck_05,
            source_metrics.validation.offscreen_accuracy,
        );
        if !real_presence_safe || !synthetic_presence_safe {
            bail!(
                "refinement source checkpoint is not independently presence-safe (real_gate_passes={real_presence_safe}, synthetic_gate_passes={synthetic_presence_safe})"
            );
        }
        seed_refinement_candidates(checkpoint, &args.artifacts)?;
        best_rank = Some(checkpoint_rank);
        best_presence_rank = Some(source_metrics.validation.presence_checkpoint_rank());
        best_geometry_score = source_metrics.validation.geometry_quality();
        presence_lock = PresenceLockState::restore_locked(
            args.presence_freeze_after_safe_epochs,
            source_contract.epoch_contract.presence_safe_streak,
            source_contract
                .epoch_contract
                .presence_frozen_after_epoch
                .expect("authenticated refinement sources always carry a lock epoch"),
        );
    }
    println!(
        "optimizer schedule: backbone_freeze_epochs={} head_updates={} backbone_updates={} backbone_batch_norm={} geometry_backbone_gradients={}",
        args.backbone_freeze_epochs,
        head_schedule.total_updates(),
        backbone_schedule.total_updates(),
        if args.freeze_backbone_batch_norm {
            "frozen"
        } else {
            "adaptive"
        },
        if args.detach_geometry_backbone {
            "detached"
        } else {
            "enabled"
        },
    );
    println!(
        "checkpoint selection: policy={} real_robustness_weights={:?} real_band_bp={} real_margin_band_bp={} synthetic_margin_band_bp={} synthetic_quality_band_bp={}",
        CHECKPOINT_SELECTION_POLICY,
        CHECKPOINT_REAL_ROBUSTNESS_WEIGHTS,
        REAL_ROBUSTNESS_BAND_BASIS_POINTS,
        REAL_GATE_MARGIN_BAND_BASIS_POINTS,
        SYNTHETIC_MARGIN_BAND_BASIS_POINTS,
        SYNTHETIC_QUALITY_BAND_BASIS_POINTS,
    );
    println!(
        "presence lock: policy={} freeze_after_safe_epochs={} pairwise_weight={:.3}",
        PRESENCE_FREEZE_POLICY,
        args.presence_freeze_after_safe_epochs,
        REAL_PRESENCE_PAIRWISE_LOSS_WEIGHT,
    );

    for epoch in args.first_training_epoch()..=args.epochs {
        let epoch_samples = balanced_epoch_samples(&train, args.batch_size, args.seed, epoch)?;
        let epoch_batch_count = epoch_samples.len().div_ceil(args.batch_size);
        let presence_trainable = presence_lock.presence_trainable();
        let geometry_only = !presence_trainable;
        let backbone_trainable = presence_trainable && backbone_schedule.is_trainable(epoch);
        let backbone_stage = if backbone_trainable {
            "trainable"
        } else {
            "frozen"
        };
        let optimizer_stage = if geometry_only {
            OptimizerStage::GeometryOnlyPresenceLocked
        } else if backbone_trainable {
            OptimizerStage::Joint
        } else {
            OptimizerStage::JointBackboneFrozen
        };
        println!(
            "epoch {epoch:02} optimizer: stage={} presence_trainable={presence_trainable} backbone_stage={backbone_stage} head_updates={}/{} joint_head_updates={} geometry_only_head_updates={} backbone_updates={}/{} safe_streak={} frozen_after_epoch={:?}",
            optimizer_stage.label(),
            head_schedule.updates_completed(),
            head_schedule.total_updates(),
            joint_head_updates_completed,
            geometry_only_head_updates_completed,
            backbone_schedule.updates_completed(),
            backbone_schedule.total_updates(),
            presence_lock.safe_streak,
            presence_lock.frozen_after_epoch,
        );
        let epoch_started = Instant::now();
        let mut total_loss = 0.0;
        let mut total_presence_loss = 0.0;
        let mut total_real_pairwise_loss = 0.0;
        let mut total_geometry_loss = 0.0;
        let mut batches = 0usize;
        let mut pending_loss: Option<Tensor<B::InnerBackend, 1>> = None;
        let mut pending_presence_loss: Option<Tensor<B::InnerBackend, 1>> = None;
        let mut pending_real_pairwise_loss: Option<Tensor<B::InnerBackend, 1>> = None;
        let mut pending_geometry_loss: Option<Tensor<B::InnerBackend, 1>> = None;
        let mut pending_loss_count = 0usize;
        for (batch_index, chunk) in epoch_samples.chunks(args.batch_size).enumerate() {
            let batch = make_batch::<B>(
                chunk,
                &device,
                epoch,
                !args.disable_augmentation,
                Some(&working_image_cache),
            )?;
            let (
                backward_loss,
                detached_loss,
                detached_presence_loss,
                detached_real_pairwise_loss,
                detached_geometry_loss,
            ) = if geometry_only {
                // Select the twelve synthetic-positive rows before the
                // backbone. The other three source groups have no geometry
                // target, and running them here would waste most of the CUDA
                // memory and compute in a source-balanced batch.
                let geometry_images = Tensor::from_inner(
                    batch
                        .images
                        .clone()
                        .inner()
                        .select(0, batch.geometry_indices.clone().inner()),
                );
                let output = model.forward_geometry_training_with_frozen_backbone(geometry_images);
                let geometry_loss = geometry_only_observation_loss(&output, &batch)
                    .expect("source-balanced training batches always contain geometry targets");
                let weighted_geometry_loss = geometry_loss.clone() * GEOMETRY_LOSS_WEIGHT;
                (
                    weighted_geometry_loss.clone(),
                    weighted_geometry_loss.inner(),
                    None,
                    None,
                    geometry_loss.inner(),
                )
            } else {
                // v15 intentionally skips the v14 real-photo pairwise ablation.
                let output = if args.freeze_backbone_batch_norm || args.detach_geometry_backbone {
                    model.forward_training_with_options(
                        batch.images.clone(),
                        KeypointTrainingOptions {
                            freeze_backbone_batch_norm: args.freeze_backbone_batch_norm,
                            detach_geometry_backbone: args.detach_geometry_backbone,
                        },
                    )
                } else {
                    model.forward(batch.images.clone())
                };
                let losses = observation_loss(output, &batch, false);
                (
                    losses.total.clone(),
                    losses.total.inner(),
                    Some(losses.presence.inner()),
                    Some(losses.real_pairwise.inner()),
                    losses.geometry.inner(),
                )
            };
            pending_loss = Some(match pending_loss.take() {
                Some(accumulated) => accumulated + detached_loss,
                None => detached_loss,
            });
            if let Some(detached_presence_loss) = detached_presence_loss {
                pending_presence_loss = Some(match pending_presence_loss.take() {
                    Some(accumulated) => accumulated + detached_presence_loss,
                    None => detached_presence_loss,
                });
            }
            if let Some(detached_real_pairwise_loss) = detached_real_pairwise_loss {
                pending_real_pairwise_loss = Some(match pending_real_pairwise_loss.take() {
                    Some(accumulated) => accumulated + detached_real_pairwise_loss,
                    None => detached_real_pairwise_loss,
                });
            }
            pending_geometry_loss = Some(match pending_geometry_loss.take() {
                Some(accumulated) => accumulated + detached_geometry_loss,
                None => detached_geometry_loss,
            });
            pending_loss_count += 1;
            batches += 1;

            let mut gradients = backward_loss.backward();
            let backbone_gradients = if backbone_trainable {
                Some(GradientsParams::from_params(
                    &mut gradients,
                    &model,
                    &groups.backbone,
                ))
            } else {
                None
            };
            let head_gradients = GradientsParams::from_params(
                &mut gradients,
                &model,
                if geometry_only {
                    &groups.geometry_heads
                } else {
                    &groups.heads
                },
            );
            let (next_model, backbone_lr) = if geometry_only {
                (model, None)
            } else {
                step_staged_backbone(
                    &mut backbone_schedule,
                    epoch,
                    model,
                    |learning_rate, model| {
                        backbone_optimizer.step(
                            learning_rate,
                            model,
                            backbone_gradients
                                .expect("a trainable backbone epoch extracts backbone gradients"),
                        )
                    },
                )
            };
            model = next_model;
            let head_lr = head_schedule.next_learning_rate();
            model = head_optimizer.step(head_lr, model, head_gradients);
            if geometry_only {
                geometry_only_head_updates_completed += 1;
            } else {
                joint_head_updates_completed += 1;
            }
            let completed_batches = batch_index + 1;
            let reports_progress = completed_batches == 1
                || completed_batches % 100 == 0
                || completed_batches == epoch_batch_count;
            if pending_loss_count == LOSS_SYNC_INTERVAL || reports_progress {
                total_loss += pending_loss
                    .take()
                    .expect("a completed batch always has a pending loss")
                    .into_scalar()
                    .elem::<f32>();
                if let Some(loss) = pending_presence_loss.take() {
                    total_presence_loss += loss.into_scalar().elem::<f32>();
                }
                if let Some(loss) = pending_real_pairwise_loss.take() {
                    total_real_pairwise_loss += loss.into_scalar().elem::<f32>();
                }
                total_geometry_loss += pending_geometry_loss
                    .take()
                    .expect("a completed batch always has a pending geometry loss")
                    .into_scalar()
                    .elem::<f32>();
                pending_loss_count = 0;
            }
            if reports_progress {
                let backbone_learning_rate = backbone_lr
                    .map(|learning_rate| format!("{learning_rate:.3e}"))
                    .unwrap_or_else(|| "disabled".to_owned());
                println!(
                    "epoch {epoch:02} batch {completed_batches:04}/{epoch_batch_count:04}: elapsed={:.1}s optimizer_stage={} presence_trainable={presence_trainable} backbone_stage={backbone_stage} backbone_lr={backbone_learning_rate} head_lr={head_lr:.3e} mean_train_loss={:.4} presence={:.4} real_pairwise={:.4} geometry={:.4}",
                    epoch_started.elapsed().as_secs_f32(),
                    optimizer_stage.label(),
                    total_loss / batches.max(1) as f32,
                    total_presence_loss / batches.max(1) as f32,
                    total_real_pairwise_loss / batches.max(1) as f32,
                    total_geometry_loss / batches.max(1) as f32,
                );
            }
        }

        let train_loss = total_loss / batches.max(1) as f32;
        let train_presence_loss = total_presence_loss / batches.max(1) as f32;
        let train_real_pairwise_loss = total_real_pairwise_loss / batches.max(1) as f32;
        let train_geometry_loss = total_geometry_loss / batches.max(1) as f32;
        let valid_model = model.clone().valid();
        let validation_metrics = evaluate(
            &valid_model,
            &validation,
            &device,
            args.evaluation_batch_size(),
            None,
            false,
            Some(fit_shape_prior),
        )?;
        let checkpoint_rank = validation_metrics.checkpoint_rank();
        let score = checkpoint_rank.selection_score();
        if !score.is_finite() {
            bail!("epoch {epoch} produced a non-finite checkpoint selection score");
        }
        let presence_lock_transitioned = presence_lock.observe_validation(
            epoch,
            validation_metrics.real_presence_gate_passes(),
            validation_metrics.synthetic_presence_gate_passes(),
        );
        let record = EpochMetrics {
            epoch,
            presence_freeze_policy: PRESENCE_FREEZE_POLICY,
            presence_freeze_after_safe_epochs: args.presence_freeze_after_safe_epochs,
            presence_safe_streak: presence_lock.safe_streak,
            presence_frozen_after_epoch: presence_lock.frozen_after_epoch,
            presence_frozen_from_epoch: presence_lock
                .frozen_after_epoch
                .map(|frozen_after| frozen_after.saturating_add(1)),
            presence_trainable,
            optimizer_stage,
            backbone_trainable,
            head_updates_completed: head_schedule.updates_completed(),
            head_total_updates: head_schedule.total_updates(),
            joint_head_updates_completed,
            geometry_only_head_updates_completed,
            backbone_updates_completed: backbone_schedule.updates_completed(),
            backbone_total_updates: backbone_schedule.total_updates(),
            train_loss,
            train_presence_loss,
            train_real_pairwise_loss,
            train_geometry_loss,
            validation: validation_metrics,
            checkpoint_rank,
            selection_score: score,
        };
        append_json_line(&metrics_path, &record)?;
        println!(
            "epoch {epoch:02}: optimizer_stage={} presence_trainable={presence_trainable} safe_streak={} frozen_after_epoch={:?} backbone_stage={backbone_stage} head_updates={}/{} joint_head_updates={} geometry_only_head_updates={} backbone_updates={}/{} train={train_loss:.4} train_presence={train_presence_loss:.4} train_real_pairwise={train_real_pairwise_loss:.4} train_geometry={train_geometry_loss:.4} val={:.4} threshold={:.3} recall={:.3} real_recall={:.3} real_building_recall={:.3} real_buildings={}/{} specificity={:.3} real_specificity={:.3} pck@.05={:.3} offscreen={:.3} duplicates={} score={score:.3}",
            optimizer_stage.label(),
            record.presence_safe_streak,
            record.presence_frozen_after_epoch,
            record.head_updates_completed,
            record.head_total_updates,
            record.joint_head_updates_completed,
            record.geometry_only_head_updates_completed,
            record.backbone_updates_completed,
            record.backbone_total_updates,
            record.validation.loss,
            record.validation.presence_threshold,
            record.validation.recall,
            record.validation.real_positive_recall,
            record.validation.real_positive_building_recall,
            record.validation.real_positive_buildings_detected,
            record.validation.real_positive_building_count,
            record.validation.specificity,
            record.validation.real_negative_specificity,
            record.validation.pck_05,
            record.validation.offscreen_accuracy,
            record.validation.duplicate_point_pairs,
        );
        println!(
            "epoch {epoch:02} presence: roc_auc={} average_precision={} synthetic_auc={} synthetic_ap={} real_auc={} real_ap={}",
            format_optional_metric(record.validation.presence_diagnostics.roc_auc),
            format_optional_metric(record.validation.presence_diagnostics.average_precision),
            format_optional_metric(record.validation.presence_diagnostics.synthetic_roc_auc),
            format_optional_metric(
                record
                    .validation
                    .presence_diagnostics
                    .synthetic_average_precision
            ),
            format_optional_metric(record.validation.presence_diagnostics.real_roc_auc),
            format_optional_metric(
                record
                    .validation
                    .presence_diagnostics
                    .real_average_precision
            ),
        );
        println!(
            "epoch {epoch:02} presence cross-domain: real_positive_vs_synthetic_negative_auc={} real_positive_vs_synthetic_negative_ap={} synthetic_target_vs_real_negative_auc={} synthetic_target_vs_real_negative_ap={}",
            format_optional_metric(
                record
                    .validation
                    .presence_diagnostics
                    .real_positive_vs_synthetic_negative_roc_auc
            ),
            format_optional_metric(
                record
                    .validation
                    .presence_diagnostics
                    .real_positive_vs_synthetic_negative_average_precision
            ),
            format_optional_metric(
                record
                    .validation
                    .presence_diagnostics
                    .synthetic_target_vs_real_negative_roc_auc
            ),
            format_optional_metric(
                record
                    .validation
                    .presence_diagnostics
                    .synthetic_target_vs_real_negative_average_precision
            ),
        );
        println!(
            "epoch {epoch:02} presence gate: common_threshold_feasible={} best_gate_ratio={:.3} best_threshold={:.8} recall={:.3} specificity={:.3} real_recall={:.3} real_specificity={:.3}",
            record
                .validation
                .presence_gate_diagnostics
                .common_threshold_feasible,
            record.validation.presence_gate_diagnostics.best_gate_ratio,
            record.validation.presence_gate_diagnostics.best_threshold,
            record.validation.presence_gate_diagnostics.recall,
            record.validation.presence_gate_diagnostics.specificity,
            record
                .validation
                .presence_gate_diagnostics
                .real_positive_recall,
            record
                .validation
                .presence_gate_diagnostics
                .real_negative_specificity,
        );
        println!(
            "epoch {epoch:02} deployment presence: real_gate_passes={} real_gate_ratio={:.3} synthetic_gate_passes={} synthetic_gate_ratio={:.3}",
            record.validation.real_presence_gate_passes(),
            record.validation.real_presence_gate_ratio(),
            record.validation.synthetic_presence_gate_passes(),
            record.validation.synthetic_presence_gate_ratio(),
        );
        if presence_lock_transitioned {
            println!(
                "epoch {epoch:02} presence lock entered after {} consecutive safe validations; epoch {} begins geometry-only training and patience restarts",
                record.presence_safe_streak,
                epoch.saturating_add(1),
            );
        }
        println!(
            "epoch {epoch:02} checkpoint rank: geometry_gate_passes={} real_metrics_available={} robust_real_band={} real_gate_margin_band={} synthetic_margin_band={} synthetic_quality_band={}",
            checkpoint_rank.geometry_gate_passes,
            checkpoint_rank.presence.real_metrics_available,
            checkpoint_rank.presence.robust_real_band,
            checkpoint_rank.presence.real_gate_margin_band,
            checkpoint_rank.presence.synthetic_margin_band,
            checkpoint_rank.presence.synthetic_quality_band,
        );
        print_probability_quantiles(epoch, &record.validation.presence_diagnostics);
        model
            .clone()
            .valid()
            .save_file(args.artifacts.join("model-last"), &CompactRecorder::new())?;
        let presence_rank = record.validation.presence_checkpoint_rank();
        if best_presence_rank.is_none_or(|best| presence_rank.is_better_than(best)) {
            best_presence_rank = Some(presence_rank);
            model.clone().valid().save_file(
                args.artifacts.join("candidate-presence"),
                &CompactRecorder::new(),
            )?;
        }
        let geometry_score = record.validation.geometry_quality();
        if geometry_score > best_geometry_score + 1.0e-5 {
            best_geometry_score = geometry_score;
            model.clone().valid().save_file(
                args.artifacts.join("candidate-geometry"),
                &CompactRecorder::new(),
            )?;
        }
        let checkpoint_improved = best_rank.is_none_or(|best| checkpoint_rank.is_better_than(best));
        if checkpoint_improved {
            best_rank = Some(checkpoint_rank);
            model.clone().valid().save_file(
                args.artifacts.join("candidate-best"),
                &CompactRecorder::new(),
            )?;
        }
        let epoch_consumes_patience = backbone_trainable || geometry_only;
        epochs_without_improvement = next_patience_count(
            epochs_without_improvement,
            checkpoint_improved,
            presence_lock_transitioned,
            epoch_consumes_patience,
        );
        if !checkpoint_improved
            && epoch_consumes_patience
            && !presence_lock_transitioned
            && epochs_without_improvement >= args.patience
        {
            println!(
                "early stopping after {epoch} epochs ({epochs_without_improvement} patience-counted epochs without improvement; optimizer_stage={})",
                optimizer_stage.label(),
            );
            break;
        }
    }

    let best_record = CompactRecorder::new()
        .load(args.artifacts.join("candidate-best"), &device)
        .context("reload best keypoint checkpoint")?;
    let best_model = KeypointRoofNetConfig::new()
        .init::<B>(&device)
        .load_record(best_record);
    let compute_final_fit = !args.overfit;
    let validation_metrics = evaluate(
        &best_model.valid(),
        &validation,
        &device,
        args.evaluation_batch_size(),
        None,
        false,
        Some(fit_shape_prior),
    )?;
    let test_metrics = evaluate(
        &best_model.valid(),
        &test,
        &device,
        args.evaluation_batch_size(),
        Some(validation_metrics.presence_threshold),
        compute_final_fit,
        Some(fit_shape_prior),
    )?;
    let promotion = PromotionDecision::from_metrics(args, &validation_metrics, &test_metrics);
    let final_metrics = FinalMetrics {
        schema_version: "roof-training-metrics/v9".to_owned(),
        validation: validation_metrics,
        test: test_metrics,
        promotion: promotion.clone(),
    };
    if promotion.promoted {
        let shape_prior = shape_prior(&train)?;
        let manifest = ModelManifest {
            schema_version: "roof-keypoint-model/v3".to_owned(),
            checkpoint: "model.mpk".to_owned(),
            input_size: SPATIAL_INPUT_SIZE,
            heatmap_size: HEATMAP_SIZE,
            keypoint_count: KEYPOINT_COUNT,
            recommended_presence_threshold: final_metrics.validation.presence_threshold,
            recommended_offscreen_threshold: 0.5,
            recommended_keypoint_threshold: DEFAULT_FIT_KEYPOINT_CONFIDENCE,
            shape_prior,
        };
        publish_promoted_model(&best_model, &args.artifacts, &manifest, &final_metrics)?;
    } else {
        publish_unpromoted_metrics(&args.artifacts, &final_metrics)?;
    }
    println!(
        "test: loss={:.4} aggregate_recall={:.3} aggregate_specificity={:.3} real_recall={:.3} real_building_recall={:.3} real_buildings={}/{} real_specificity={:.3} synthetic_auc={} synthetic_ap={} pck@.05={:.3} offscreen={:.3} fit_rmse={:.3} fit_iou={:.3}; promoted={}; checkpoint={}",
        final_metrics.test.loss,
        final_metrics.test.recall,
        final_metrics.test.specificity,
        final_metrics.test.real_positive_recall,
        final_metrics.test.real_positive_building_recall,
        final_metrics.test.real_positive_buildings_detected,
        final_metrics.test.real_positive_building_count,
        final_metrics.test.real_negative_specificity,
        format_optional_metric(final_metrics.test.presence_diagnostics.synthetic_roc_auc),
        format_optional_metric(
            final_metrics
                .test
                .presence_diagnostics
                .synthetic_average_precision
        ),
        final_metrics.test.pck_05,
        final_metrics.test.offscreen_accuracy,
        final_metrics.test.synthetic_fit.median_mesh_rmse,
        final_metrics.test.synthetic_fit.median_silhouette_iou,
        promotion.promoted,
        if promotion.promoted {
            args.artifacts.join("model.mpk")
        } else {
            args.artifacts.join("candidate-best.mpk")
        }
        .display(),
    );
    if !promotion.promoted {
        println!("promotion blocked: {}", promotion.failures.join("; "));
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct PromotionPaths {
    checkpoint: PathBuf,
    manifest: PathBuf,
    metrics: PathBuf,
}

impl PromotionPaths {
    fn final_paths(artifacts: &Path) -> Self {
        Self {
            checkpoint: artifacts.join("model.mpk"),
            manifest: artifacts.join("model.json"),
            metrics: artifacts.join("final-metrics.json"),
        }
    }

    fn temporary(artifacts: &Path) -> Self {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos();
        let id = PROMOTION_TEMP_ID.fetch_add(1, AtomicOrdering::Relaxed);
        let prefix = format!(".roof-promotion-{}-{timestamp}-{id}", std::process::id());
        Self {
            checkpoint: artifacts.join(format!("{prefix}.mpk")),
            manifest: artifacts.join(format!("{prefix}.model.json")),
            metrics: artifacts.join(format!("{prefix}.metrics.json")),
        }
    }

    fn iter(&self) -> impl Iterator<Item = &Path> {
        [
            self.checkpoint.as_path(),
            self.manifest.as_path(),
            self.metrics.as_path(),
        ]
        .into_iter()
    }
}

struct TemporaryPromotionFiles {
    paths: PromotionPaths,
}

impl TemporaryPromotionFiles {
    fn new(artifacts: &Path) -> Self {
        Self {
            paths: PromotionPaths::temporary(artifacts),
        }
    }
}

impl Drop for TemporaryPromotionFiles {
    fn drop(&mut self) {
        for path in self.paths.iter() {
            match fs::remove_file(path) {
                Ok(()) => {}
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
                Err(_) => {}
            }
        }
    }
}

fn ensure_promotion_destinations_absent(paths: &PromotionPaths) -> Result<()> {
    let existing = paths
        .iter()
        .filter(|path| path.exists())
        .map(|path| path.display().to_string())
        .collect::<Vec<_>>();
    if !existing.is_empty() {
        bail!(
            "refusing to overwrite existing promotion output(s): {}",
            existing.join(", ")
        );
    }
    Ok(())
}

fn verify_prepared_file(path: &Path, expected: Option<&[u8]>) -> Result<()> {
    if !path.is_file() {
        bail!("prepared promotion file is missing: {}", path.display());
    }
    if let Some(expected) = expected {
        let observed = fs::read(path)
            .with_context(|| format!("verify prepared promotion file {}", path.display()))?;
        if observed != expected {
            bail!(
                "prepared promotion file changed before commit: {}",
                path.display()
            );
        }
    }
    Ok(())
}

fn cleanup_committed_promotion_after_error(
    final_paths: &PromotionPaths,
    manifest_committed: bool,
    checkpoint_committed: bool,
) -> Vec<String> {
    let mut cleanup_failures = Vec::new();
    let checkpoint_absent = if checkpoint_committed {
        match fs::remove_file(&final_paths.checkpoint) {
            Ok(()) => true,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => true,
            Err(error) => {
                cleanup_failures.push(format!(
                    "remove partial checkpoint {}: {error}",
                    final_paths.checkpoint.display()
                ));
                false
            }
        }
    } else {
        true
    };

    // Never remove a committed manifest while a checkpoint could remain. A
    // checkpoint plus its valid manifest is safer than an orphan checkpoint
    // when rollback itself encounters an I/O failure.
    if manifest_committed && checkpoint_absent {
        match fs::remove_file(&final_paths.manifest) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => cleanup_failures.push(format!(
                "remove partial manifest {}: {error}",
                final_paths.manifest.display()
            )),
        }
    }
    cleanup_failures
}

fn cleanup_temporary_promotion(paths: &PromotionPaths) -> Vec<String> {
    let mut cleanup_failures = Vec::new();
    for path in paths.iter() {
        match fs::remove_file(path) {
            Ok(()) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => cleanup_failures.push(format!(
                "remove temporary promotion file {}: {error}",
                path.display()
            )),
        }
    }
    cleanup_failures
}

fn promotion_error_with_cleanup(
    error: anyhow::Error,
    cleanup_failures: Vec<String>,
) -> anyhow::Error {
    if cleanup_failures.is_empty() {
        error
    } else {
        error.context(format!(
            "promotion cleanup also failed: {}",
            cleanup_failures.join("; ")
        ))
    }
}

fn commit_prepared_promotion_with(
    temporary: &PromotionPaths,
    final_paths: &PromotionPaths,
    mut rename: impl FnMut(&Path, &Path) -> std::io::Result<()>,
) -> Result<()> {
    if let Err(error) = ensure_promotion_destinations_absent(final_paths) {
        return Err(promotion_error_with_cleanup(
            error,
            cleanup_temporary_promotion(temporary),
        ));
    }
    for path in temporary.iter() {
        if let Err(error) = verify_prepared_file(path, None) {
            return Err(promotion_error_with_cleanup(
                error,
                cleanup_temporary_promotion(temporary),
            ));
        }
    }

    let mut manifest_committed = false;
    let mut checkpoint_committed = false;
    let commit_result = (|| -> Result<()> {
        rename(&temporary.manifest, &final_paths.manifest).with_context(|| {
            format!(
                "commit promotion manifest {}",
                final_paths.manifest.display()
            )
        })?;
        manifest_committed = true;
        rename(&temporary.checkpoint, &final_paths.checkpoint).with_context(|| {
            format!(
                "commit promotion checkpoint {}",
                final_paths.checkpoint.display()
            )
        })?;
        checkpoint_committed = true;
        // A report with `promoted=true` is deliberately the final commit. It
        // cannot become visible until both deployable files are in place.
        rename(&temporary.metrics, &final_paths.metrics).with_context(|| {
            format!(
                "commit promoted final metrics {}",
                final_paths.metrics.display()
            )
        })?;
        Ok(())
    })();

    if let Err(error) = commit_result {
        let mut cleanup_failures = cleanup_committed_promotion_after_error(
            final_paths,
            manifest_committed,
            checkpoint_committed,
        );
        cleanup_failures.extend(cleanup_temporary_promotion(temporary));
        return Err(promotion_error_with_cleanup(error, cleanup_failures));
    }
    Ok(())
}

fn commit_prepared_promotion(
    temporary: &PromotionPaths,
    final_paths: &PromotionPaths,
) -> Result<()> {
    commit_prepared_promotion_with(temporary, final_paths, |source, destination| {
        fs::rename(source, destination)
    })
}

fn publish_promoted_model<B: AutodiffBackend>(
    model: &KeypointRoofNet<B>,
    artifacts: &Path,
    manifest: &ModelManifest,
    metrics: &FinalMetrics,
) -> Result<()> {
    let final_paths = PromotionPaths::final_paths(artifacts);
    ensure_promotion_destinations_absent(&final_paths)?;
    let temporary = TemporaryPromotionFiles::new(artifacts);
    let manifest_bytes = serde_json::to_vec_pretty(manifest)?;
    let metrics_bytes = serde_json::to_vec_pretty(metrics)?;

    model
        .clone()
        .valid()
        .save_file(temporary.paths.checkpoint.clone(), &CompactRecorder::new())?;
    fs::write(&temporary.paths.manifest, &manifest_bytes).with_context(|| {
        format!(
            "prepare promotion manifest {}",
            temporary.paths.manifest.display()
        )
    })?;
    fs::write(&temporary.paths.metrics, &metrics_bytes).with_context(|| {
        format!(
            "prepare promoted final metrics {}",
            temporary.paths.metrics.display()
        )
    })?;
    verify_prepared_file(&temporary.paths.checkpoint, None)?;
    verify_prepared_file(&temporary.paths.manifest, Some(&manifest_bytes))?;
    verify_prepared_file(&temporary.paths.metrics, Some(&metrics_bytes))?;
    commit_prepared_promotion(&temporary.paths, &final_paths)
}

fn publish_unpromoted_metrics(artifacts: &Path, metrics: &FinalMetrics) -> Result<()> {
    let final_paths = PromotionPaths::final_paths(artifacts);
    ensure_promotion_destinations_absent(&final_paths)?;
    let temporary = TemporaryPromotionFiles::new(artifacts);
    let metrics_bytes = serde_json::to_vec_pretty(metrics)?;
    fs::write(&temporary.paths.metrics, &metrics_bytes).with_context(|| {
        format!(
            "prepare unpromoted final metrics {}",
            temporary.paths.metrics.display()
        )
    })?;
    verify_prepared_file(&temporary.paths.metrics, Some(&metrics_bytes))?;
    fs::rename(&temporary.paths.metrics, &final_paths.metrics).with_context(|| {
        format!(
            "commit unpromoted final metrics {}",
            final_paths.metrics.display()
        )
    })?;
    Ok(())
}

fn seed_refinement_candidates(source: &Path, artifacts: &Path) -> Result<()> {
    for filename in [
        "candidate-best.mpk",
        "candidate-presence.mpk",
        "candidate-geometry.mpk",
        "model-last.mpk",
    ] {
        let destination = artifacts.join(filename);
        if destination.exists() {
            bail!(
                "refinement baseline destination already exists: {}",
                destination.display()
            );
        }
        fs::copy(source, &destination).with_context(|| {
            format!(
                "seed refinement baseline {} from {}",
                destination.display(),
                source.display()
            )
        })?;
    }
    Ok(())
}

pub(super) fn evaluate_checkpoint<B: AutodiffBackend>(
    args: &Args,
    train: &[TrainingSample],
    validation: &[TrainingSample],
    test: &[TrainingSample],
    checkpoint: &Path,
    device: B::Device,
) -> Result<()> {
    B::seed(&device, args.seed);
    let record = CompactRecorder::new()
        .load(checkpoint.to_path_buf(), &device)
        .with_context(|| format!("load checkpoint {}", checkpoint.display()))?;
    let model = KeypointRoofNetConfig::new()
        .init::<B>(&device)
        .load_record(record)
        .valid();
    let fit_shape_prior = load_checkpoint_shape_prior(checkpoint)
        .or_else(|| shape_prior(train).ok().map(|prior| prior.mean));
    let compute_fit_metrics = !args.overfit;
    let validation_metrics = evaluate(
        &model,
        validation,
        &device,
        args.evaluation_batch_size(),
        None,
        false,
        fit_shape_prior,
    )?;
    let test_metrics = evaluate(
        &model,
        test,
        &device,
        args.evaluation_batch_size(),
        Some(validation_metrics.presence_threshold),
        compute_fit_metrics,
        fit_shape_prior,
    )?;
    let promotion = PromotionDecision::from_metrics(args, &validation_metrics, &test_metrics);
    let report = FinalMetrics {
        schema_version: "roof-checkpoint-evaluation/v5".to_owned(),
        validation: validation_metrics,
        test: test_metrics,
        promotion: promotion.clone(),
    };
    fs::create_dir_all(&args.artifacts)?;
    fs::write(
        args.artifacts.join("checkpoint-evaluation.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!(
        "checkpoint evaluation: aggregate_recall={:.3} aggregate_specificity={:.3} real_recall={:.3} real_building_recall={:.3} real_buildings={}/{} real_specificity={:.3} synthetic_auc={} synthetic_ap={} pck@.03={:.3} pck@.05={:.3} offscreen={:.3} fit_rmse={:.3} fit_iou={:.3} duplicates={} passes_gate={}",
        report.test.recall,
        report.test.specificity,
        report.test.real_positive_recall,
        report.test.real_positive_building_recall,
        report.test.real_positive_buildings_detected,
        report.test.real_positive_building_count,
        report.test.real_negative_specificity,
        format_optional_metric(report.test.presence_diagnostics.synthetic_roc_auc),
        format_optional_metric(report.test.presence_diagnostics.synthetic_average_precision),
        report.test.pck_03,
        report.test.pck_05,
        report.test.offscreen_accuracy,
        report.test.synthetic_fit.median_mesh_rmse,
        report.test.synthetic_fit.median_silhouette_iou,
        report.test.duplicate_point_pairs,
        promotion.promoted,
    );
    println!(
        "checkpoint validation presence: common_threshold_feasible={} best_gate_ratio={:.3} best_threshold={:.8} real_positive_vs_synthetic_negative_auc={} synthetic_target_vs_real_negative_auc={}",
        report
            .validation
            .presence_gate_diagnostics
            .common_threshold_feasible,
        report.validation.presence_gate_diagnostics.best_gate_ratio,
        report.validation.presence_gate_diagnostics.best_threshold,
        format_optional_metric(
            report
                .validation
                .presence_diagnostics
                .real_positive_vs_synthetic_negative_roc_auc
        ),
        format_optional_metric(
            report
                .validation
                .presence_diagnostics
                .synthetic_target_vs_real_negative_roc_auc
        ),
    );
    if !promotion.promoted {
        println!("gate failures: {}", promotion.failures.join("; "));
    }
    Ok(())
}

struct ObservationBatch<B: Backend> {
    images: Tensor<B, 4>,
    presence: Tensor<B, 1>,
    presence_weights: Tensor<B, 1>,
    real_positive_indices: Tensor<B, 1, Int>,
    real_positive_count: usize,
    real_negative_indices: Tensor<B, 1, Int>,
    real_negative_count: usize,
    keypoint_targets: Tensor<B, 3>,
    symmetry_masks: Tensor<B, 4>,
    geometry_indices: Tensor<B, 1, Int>,
    geometry_count: usize,
    positions: Vec<[[f32; 2]; KEYPOINT_COUNT]>,
    in_frame: Vec<[bool; KEYPOINT_COUNT]>,
    transforms: Vec<LetterboxTransform>,
}

struct PreparedSample {
    chw: Vec<f32>,
    presence: f32,
    keypoint_targets: Vec<f32>,
    geometry_mask: f32,
    positions: [[f32; 2]; KEYPOINT_COUNT],
    in_frame: [bool; KEYPOINT_COUNT],
    transform: LetterboxTransform,
}

fn make_batch<B: Backend>(
    samples: &[TrainingSample],
    device: &Device<B>,
    epoch: usize,
    augment: bool,
    working_image_cache: Option<&WorkingImageCache>,
) -> Result<ObservationBatch<B>> {
    let prepared = samples
        .par_iter()
        .map(|sample| prepare_sample(sample, epoch, augment, working_image_cache))
        .collect::<Result<Vec<_>>>()?;
    let mut images =
        Vec::with_capacity(samples.len() * 3 * SPATIAL_INPUT_SIZE * SPATIAL_INPUT_SIZE);
    let mut presence = Vec::with_capacity(samples.len());
    let target_values = KEYPOINT_COUNT * KEYPOINT_DISTRIBUTION_SIZE;
    let mut keypoint_targets = Vec::with_capacity(samples.len() * target_values);
    let mut geometry_indices = Vec::new();
    let mut positions = Vec::with_capacity(samples.len());
    let mut in_frame = Vec::with_capacity(samples.len());
    let mut transforms = Vec::with_capacity(samples.len());
    for (sample_index, item) in prepared.into_iter().enumerate() {
        images.extend(item.chw);
        presence.push(item.presence);
        if item.geometry_mask > 0.0 {
            geometry_indices.push(sample_index as i64);
            keypoint_targets.extend(item.keypoint_targets);
        }
        positions.push(item.positions);
        in_frame.push(item.in_frame);
        transforms.push(item.transform);
    }
    let batch = samples.len();
    let geometry_count = geometry_indices.len();
    // Burn backends do not uniformly support zero-sized tensor dimensions.
    // Presence-only batches return before reading these padded placeholders.
    let geometry_storage_count = geometry_count.max(1);
    if geometry_count == 0 {
        geometry_indices.push(0);
        keypoint_targets.resize(target_values, 0.0);
    }
    let presence_weights = source_balancing_weights(samples);
    let RealPresencePairIndices {
        mut positives,
        mut negatives,
    } = real_presence_pair_indices(samples);
    let real_positive_count = positives.len();
    let real_negative_count = negatives.len();
    // Like the geometry placeholders above, these avoid zero-sized tensors on
    // backends that do not support them. The pairwise loss returns before
    // selecting a placeholder when either real source is absent.
    if positives.is_empty() {
        positives.push(0);
    }
    if negatives.is_empty() {
        negatives.push(0);
    }
    Ok(ObservationBatch {
        images: Tensor::from_data(
            TensorData::new(
                images,
                Shape::new([batch, 3, SPATIAL_INPUT_SIZE, SPATIAL_INPUT_SIZE]),
            ),
            device,
        ),
        presence: Tensor::from_data(TensorData::new(presence, Shape::new([batch])), device),
        presence_weights: Tensor::from_data(
            TensorData::new(presence_weights, Shape::new([batch])),
            device,
        ),
        real_positive_indices: Tensor::from_data(
            TensorData::new(positives, Shape::new([real_positive_count.max(1)])),
            device,
        ),
        real_positive_count,
        real_negative_indices: Tensor::from_data(
            TensorData::new(negatives, Shape::new([real_negative_count.max(1)])),
            device,
        ),
        real_negative_count,
        keypoint_targets: Tensor::from_data(
            TensorData::new(
                keypoint_targets,
                Shape::new([
                    geometry_storage_count,
                    KEYPOINT_COUNT,
                    KEYPOINT_DISTRIBUTION_SIZE,
                ]),
            ),
            device,
        ),
        symmetry_masks: Tensor::from_data(
            TensorData::new(symmetry_masks(), Shape::new([SYMMETRY_COUNT, 3, 4, 4])),
            device,
        ),
        geometry_indices: Tensor::from_data(
            TensorData::new(geometry_indices, Shape::new([geometry_storage_count])),
            device,
        ),
        geometry_count,
        positions,
        in_frame,
        transforms,
    })
}

fn origin_index(origin: SampleOrigin) -> usize {
    match origin {
        SampleOrigin::SyntheticTarget => 0,
        SampleOrigin::SyntheticNegative => 1,
        SampleOrigin::RealPositive => 2,
        SampleOrigin::RealNegative => 3,
    }
}

#[derive(Debug, Default, PartialEq, Eq)]
struct RealPresencePairIndices {
    positives: Vec<i64>,
    negatives: Vec<i64>,
}

/// Selects ranking pairs by source identity, never by the binary target alone.
///
/// Synthetic targets and negatives remain supervised by BCE, but must not
/// influence the real-domain ranking offset this auxiliary loss corrects.
fn real_presence_pair_indices(samples: &[TrainingSample]) -> RealPresencePairIndices {
    let mut indices = RealPresencePairIndices::default();
    for (index, sample) in samples.iter().enumerate() {
        match sample.origin {
            SampleOrigin::RealPositive => indices.positives.push(index as i64),
            SampleOrigin::RealNegative => indices.negatives.push(index as i64),
            SampleOrigin::SyntheticTarget | SampleOrigin::SyntheticNegative => {}
        }
    }
    indices
}

/// Gives each represented source its configured total presence mass.
///
/// Counts still cancel within a source, preventing thousands of synthetic
/// frames from drowning out the real photographs. The unequal source masses
/// deliberately make deployment-domain calibration the stronger objective.
fn source_balancing_weights(samples: &[TrainingSample]) -> Vec<f32> {
    let mut counts = [0usize; 4];
    for sample in samples {
        counts[origin_index(sample.origin)] += 1;
    }
    samples
        .iter()
        .map(|sample| {
            let index = origin_index(sample.origin);
            PRESENCE_SOURCE_MASS[index] / counts[index].max(1) as f32
        })
        .collect()
}

fn symmetry_masks() -> Vec<f32> {
    let mut masks = vec![0.0; SYMMETRY_COUNT * 3 * 4 * 4];
    for hypothesis in 0..SYMMETRY_COUNT {
        for ring in 0..3 {
            for predicted_slot in 0..4 {
                let target_slot = symmetry_target_slot(hypothesis, predicted_slot);
                let index = (((hypothesis * 3 + ring) * 4 + predicted_slot) * 4) + target_slot;
                masks[index] = 1.0;
            }
        }
    }
    masks
}

fn batch_slots(samples: &[TrainingSample], batch_size: usize) -> [usize; 4] {
    let mut populated = [false; 4];
    for sample in samples {
        populated[origin_index(sample.origin)] = true;
    }
    if populated.iter().all(|value| *value) && batch_size >= 8 {
        let real_each = (batch_size / 8).max(1);
        let synthetic = batch_size - real_each * 2;
        return [synthetic.div_ceil(2), synthetic / 2, real_each, real_each];
    }

    let active = populated.iter().filter(|value| **value).count();
    let mut slots = [0usize; 4];
    if active == 0 {
        return slots;
    }
    for index in 0..batch_size {
        let selected = (0..4)
            .filter(|candidate| populated[*candidate])
            .nth(index % active)
            .expect("at least one sample source is populated");
        slots[selected] += 1;
    }
    slots
}

fn balanced_epoch_len(samples: &[TrainingSample], batch_size: usize) -> usize {
    let slots = batch_slots(samples, batch_size);
    let mut counts = [0usize; 4];
    for sample in samples {
        counts[origin_index(sample.origin)] += 1;
    }
    let batches = (0..4)
        .filter(|index| slots[*index] > 0)
        .map(|index| counts[index].div_ceil(slots[index]))
        .max()
        .unwrap_or(0);
    batches * batch_size
}

fn balanced_epoch_samples(
    samples: &[TrainingSample],
    batch_size: usize,
    seed: u64,
    epoch: usize,
) -> Result<Vec<TrainingSample>> {
    let slots = batch_slots(samples, batch_size);
    let mut groups: [Vec<TrainingSample>; 4] = std::array::from_fn(|_| Vec::new());
    for sample in samples {
        groups[origin_index(sample.origin)].push(sample.clone());
    }
    let mut rng = ChaCha8Rng::seed_from_u64(seed ^ (epoch as u64).wrapping_mul(0x9e37_79b9));
    for group in &mut groups {
        group.shuffle(&mut rng);
    }
    let batches = (0..4)
        .filter(|index| slots[*index] > 0)
        .map(|index| groups[index].len().div_ceil(slots[index]))
        .max()
        .unwrap_or(0);
    let real_positive_source = origin_index(SampleOrigin::RealPositive);
    let real_positive_draws = building_balanced_real_positive_draws(
        &groups[real_positive_source],
        batches * slots[real_positive_source],
        seed,
        epoch,
    )?;
    let mut output = Vec::with_capacity(batches * batch_size);
    for batch_index in 0..batches {
        let mut batch = Vec::with_capacity(batch_size);
        for source in 0..4 {
            for slot in 0..slots[source] {
                let draw = batch_index * slots[source] + slot;
                if source == real_positive_source {
                    batch.push(real_positive_draws[draw].clone());
                    continue;
                }
                let mut sample = groups[source][draw % groups[source].len()].clone();
                sample.key = format!("{}#epoch-{epoch}-draw-{draw}", sample.key);
                batch.push(sample);
            }
        }
        batch.shuffle(&mut rng);
        output.extend(batch);
    }
    Ok(output)
}

fn building_balanced_real_positive_draws(
    samples: &[TrainingSample],
    draw_count: usize,
    seed: u64,
    epoch: usize,
) -> Result<Vec<TrainingSample>> {
    if draw_count == 0 {
        return Ok(Vec::new());
    }

    let mut grouped = BTreeMap::<&str, Vec<&TrainingSample>>::new();
    for sample in samples {
        let building = sample
            .physical_building_id
            .as_deref()
            .filter(|building| !building.trim().is_empty())
            .with_context(|| {
                format!(
                    "real-positive sample {:?} has no physical_building_id",
                    sample.key
                )
            })?;
        grouped.entry(building).or_default().push(sample);
    }
    if grouped.is_empty() {
        bail!("real-positive epoch slots were allocated without any eligible buildings");
    }

    let epoch_seed =
        seed ^ (epoch as u64).wrapping_mul(0x9e37_79b9) ^ REAL_POSITIVE_SAMPLING_SEED_SALT;
    let mut rng = ChaCha8Rng::seed_from_u64(epoch_seed);
    let mut buildings = grouped.into_values().collect::<Vec<_>>();
    for views in &mut buildings {
        views.sort_unstable_by(|left, right| left.key.cmp(&right.key));
        views.shuffle(&mut rng);
    }
    buildings.shuffle(&mut rng);

    let mut building_draw_counts = vec![0usize; buildings.len()];
    let mut draws = Vec::with_capacity(draw_count);
    for draw in 0..draw_count {
        let building_index = draw % buildings.len();
        let views = &buildings[building_index];
        let view_index = building_draw_counts[building_index] % views.len();
        let mut sample = (*views[view_index]).clone();
        sample.key = format!("{}#epoch-{epoch}-draw-{draw}", sample.key);
        draws.push(sample);
        building_draw_counts[building_index] += 1;
    }
    Ok(draws)
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
enum WorkingImageKey {
    Encoded { pointer: usize, length: usize },
    Path(PathBuf),
}

impl WorkingImageKey {
    fn from_source(source: &ImageSource) -> Self {
        match source {
            ImageSource::Encoded(bytes) => Self::Encoded {
                pointer: bytes.as_ptr() as usize,
                length: bytes.len(),
            },
            ImageSource::Path(path) => Self::Path(path.clone()),
        }
    }
}

/// Lazily retains each decoded, bounded training raster. Balanced sampling can
/// draw the same synthetic buffer repeatedly and deliberately repeats each
/// real-positive path; keying on the underlying source avoids decoding those
/// images again for every draw and epoch.
#[derive(Default)]
struct WorkingImageCache {
    images: Mutex<HashMap<WorkingImageKey, Arc<image::RgbImage>>>,
}

impl WorkingImageCache {
    fn load(&self, sample: &TrainingSample) -> Result<Arc<image::RgbImage>> {
        let key = WorkingImageKey::from_source(&sample.image);
        if let Some(image) = self
            .images
            .lock()
            .map_err(|_| anyhow::anyhow!("working-image cache lock was poisoned"))?
            .get(&key)
            .cloned()
        {
            return Ok(image);
        }

        // Decode outside the cache lock so Rayon workers can prepare distinct
        // samples concurrently. A rare first-batch duplicate may do the work
        // twice, but the entry operation still stores only one raster.
        let decoded = decode_sample_image(sample)?;
        let working = Arc::new(resize_to_working_raster(&decoded, SPATIAL_INPUT_SIZE));
        let mut images = self
            .images
            .lock()
            .map_err(|_| anyhow::anyhow!("working-image cache lock was poisoned"))?;
        Ok(images.entry(key).or_insert(working).clone())
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.images.lock().expect("cache lock").len()
    }
}

fn decode_sample_image(sample: &TrainingSample) -> Result<image::RgbImage> {
    let decoded = match &sample.image {
        ImageSource::Encoded(bytes) => image::load_from_memory(bytes.as_ref()),
        ImageSource::Path(path) => {
            let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
            image::load_from_memory(&bytes)
        }
    };
    decoded
        .with_context(|| format!("decode {}", sample.key))
        .map(image::DynamicImage::into_rgb8)
}

fn prepare_sample(
    sample: &TrainingSample,
    epoch: usize,
    augment: bool,
    working_image_cache: Option<&WorkingImageCache>,
) -> Result<PreparedSample> {
    let training = augment && epoch > 0 && sample.split == Split::Train;
    let mut rgb = if training {
        if let Some(cache) = working_image_cache {
            cache.load(sample)?.as_ref().clone()
        } else {
            let decoded = decode_sample_image(sample)?;
            resize_to_working_raster(&decoded, SPATIAL_INPUT_SIZE)
        }
    } else {
        // Evaluation deliberately prepares the original raster directly. Its
        // LetterboxTransform therefore retains the true source dimensions used
        // by source-diagonal PCK and perspective-fit metrics.
        decode_sample_image(sample)?
    };
    let roll = training_roll_radians(&sample.key, epoch, training);
    if let Some(radians) = roll {
        rgb = rotate_rgb_reflect(&rgb, radians);
    }
    let flipped = training && sample_hash(&sample.key, epoch, 0x464c_4950) & 1 != 0;
    if flipped {
        image::imageops::flip_horizontal_in_place(&mut rgb);
    }
    if training {
        apply_photometric_jitter(&mut rgb, &sample.key, epoch);
    }
    let (width, height) = rgb.dimensions();
    let prepared = prepare_rgb8_sized(width, height, rgb.as_raw(), SPATIAL_INPUT_SIZE)?;
    let mut output = PreparedSample {
        chw: prepared.chw,
        presence: sample.presence as f32,
        keypoint_targets: Vec::new(),
        geometry_mask: 0.0,
        positions: [[0.0; 2]; KEYPOINT_COUNT],
        in_frame: [false; KEYPOINT_COUNT],
        transform: prepared.transform,
    };
    if let Some(record) = &sample.geometry {
        let rotated_record =
            roll.map(|radians| rotate_frame_keypoints(record, width, height, radians));
        let target_record = rotated_record.as_ref().unwrap_or(record);
        let mut target = build_keypoint_target(target_record, prepared.transform)
            .with_context(|| format!("build keypoint targets for {}", sample.key))?;
        if flipped {
            flip_target_horizontal(&mut target);
        }
        output.keypoint_targets = target.distributions;
        output.geometry_mask = 1.0;
        output.positions = target.positions;
        output.in_frame = target.in_frame;
    }
    Ok(output)
}

fn apply_photometric_jitter(image: &mut image::RgbImage, key: &str, epoch: usize) {
    let hash = sample_hash(key, epoch, 0x9e37_79b9);
    let gain = 0.72 + (hash & 0xff) as f32 / 255.0 * 0.58;
    let bias = (((hash >> 8) & 0xff) as f32 / 255.0 - 0.5) * 32.0;
    let gamma = 0.75 + ((hash >> 16) & 0xff) as f32 / 255.0 * 0.65;
    let saturation = if (hash >> 24) & 7 == 0 {
        0.0
    } else {
        0.55 + ((hash >> 27) & 0xff) as f32 / 255.0 * 0.9
    };
    let channel_gain = [
        0.88 + ((hash >> 33) & 0x3f) as f32 / 63.0 * 0.24,
        0.92 + ((hash >> 39) & 0x3f) as f32 / 63.0 * 0.16,
        0.88 + ((hash >> 45) & 0x3f) as f32 / 63.0 * 0.24,
    ];
    let noise_amplitude = if (hash >> 51) & 3 == 0 {
        2.0 + ((hash >> 53) & 0x1f) as f32 / 31.0 * 8.0
    } else {
        0.0
    };
    let vignette_strength = if (hash >> 58) & 3 == 0 {
        0.08 + ((hash >> 60) & 0x0f) as f32 / 15.0 * 0.18
    } else {
        0.0
    };
    let (width, height) = image.dimensions();
    let half_width = width.max(1) as f32 * 0.5;
    let half_height = height.max(1) as f32 * 0.5;
    for (x, y, pixel) in image.enumerate_pixels_mut() {
        let luminance =
            0.299 * f32::from(pixel[0]) + 0.587 * f32::from(pixel[1]) + 0.114 * f32::from(pixel[2]);
        let normalized_x = (x as f32 + 0.5 - half_width) / half_width;
        let normalized_y = (y as f32 + 0.5 - half_height) / half_height;
        let vignette = 1.0
            - vignette_strength
                * (normalized_x * normalized_x + normalized_y * normalized_y).min(1.0);
        let pixel_hash =
            hash ^ u64::from(x).wrapping_mul(0x9e37_79b9) ^ u64::from(y).wrapping_mul(0x85eb_ca6b);
        for (channel_index, channel) in pixel.0.iter_mut().enumerate() {
            let saturated = luminance + (f32::from(*channel) - luminance) * saturation;
            let noise_bits = (pixel_hash.rotate_left(channel_index as u32 * 19) & 0xff) as f32;
            let noise = (noise_bits / 255.0 - 0.5) * 2.0 * noise_amplitude;
            let normalized =
                (saturated * gain * channel_gain[channel_index] + bias).clamp(0.0, 255.0) / 255.0;
            *channel = (normalized.powf(gamma) * 255.0 * vignette + noise).clamp(0.0, 255.0) as u8;
        }
    }
    if (hash >> 40) & 3 == 0 {
        let sigma = 0.35 + ((hash >> 42) & 0xff) as f32 / 255.0 * 1.1;
        *image = image::imageops::blur(image, sigma);
    }
    if (hash >> 54) & 3 == 0 {
        let (width, height) = image.dimensions();
        let scale = 0.45 + ((hash >> 56) & 0xff) as f32 / 255.0 * 0.4;
        let small_width = ((width as f32 * scale).round() as u32).max(8);
        let small_height = ((height as f32 * scale).round() as u32).max(8);
        let small = image::imageops::resize(
            image,
            small_width,
            small_height,
            image::imageops::FilterType::Triangle,
        );
        *image =
            image::imageops::resize(&small, width, height, image::imageops::FilterType::Triangle);
    }
}

struct ObservationLoss<B: Backend> {
    total: Tensor<B, 1>,
    /// Source-balanced weighted binary cross-entropy.
    presence: Tensor<B, 1>,
    /// Raw real-positive versus real-negative pairwise ranking loss before
    /// [`REAL_PRESENCE_PAIRWISE_LOSS_WEIGHT`] is applied.
    real_pairwise: Tensor<B, 1>,
    /// Raw geometry loss before [`GEOMETRY_LOSS_WEIGHT`] is applied.
    geometry: Tensor<B, 1>,
}

/// Mean `softplus(margin - (positive_logit - negative_logit))` over every
/// real-positive/real-negative pair represented in one training batch.
///
/// `-log_sigmoid` is the numerically stable logistic softplus identity. A
/// graph-connected zero keeps backward valid for synthetic-only and partial
/// source batches without selecting their padded index placeholders.
fn real_presence_pairwise_loss<B: Backend>(
    presence_logits: Tensor<B, 1>,
    real_positive_indices: Tensor<B, 1, Int>,
    real_positive_count: usize,
    real_negative_indices: Tensor<B, 1, Int>,
    real_negative_count: usize,
) -> Tensor<B, 1> {
    if real_positive_count == 0 || real_negative_count == 0 {
        return presence_logits.sum() * 0.0;
    }

    let positive_logits = presence_logits
        .clone()
        .select(0, real_positive_indices)
        .unsqueeze_dim::<2>(1);
    let negative_logits = presence_logits
        .select(0, real_negative_indices)
        .unsqueeze_dim::<2>(0);
    let margin_satisfaction = positive_logits - negative_logits - REAL_PRESENCE_PAIRWISE_MARGIN;
    -log_sigmoid(margin_satisfaction).sum() / (real_positive_count * real_negative_count) as f32
}

fn observation_loss<B: Backend>(
    output: KeypointRoofOutput<B>,
    batch: &ObservationBatch<B>,
    include_real_pairwise: bool,
) -> ObservationLoss<B> {
    let positive = batch.presence.clone();
    let negative = positive.clone().neg() + 1.0;
    let presence_per_sample = -(log_sigmoid(output.presence_logits.clone()) * positive
        + log_sigmoid(output.presence_logits.clone().neg()) * negative);
    let presence_loss = (presence_per_sample * batch.presence_weights.clone()).sum()
        / batch.presence_weights.clone().sum().clamp_min(1.0e-6);
    let real_pairwise_loss = if include_real_pairwise {
        real_presence_pairwise_loss(
            output.presence_logits.clone(),
            batch.real_positive_indices.clone(),
            batch.real_positive_count,
            batch.real_negative_indices.clone(),
            batch.real_negative_count,
        )
    } else {
        presence_loss.clone() * 0.0
    };
    let presence_objective =
        presence_loss.clone() + real_pairwise_loss.clone() * REAL_PRESENCE_PAIRWISE_LOSS_WEIGHT;

    let Some(geometry_loss) = geometry_observation_loss(&output, batch) else {
        let geometry_loss = presence_loss.clone() * 0.0;
        return ObservationLoss {
            total: presence_objective,
            presence: presence_loss,
            real_pairwise: real_pairwise_loss,
            geometry: geometry_loss,
        };
    };

    let total = presence_objective + geometry_loss.clone() * GEOMETRY_LOSS_WEIGHT;
    ObservationLoss {
        total,
        presence: presence_loss,
        real_pairwise: real_pairwise_loss,
        geometry: geometry_loss,
    }
}

/// Geometry-only objective with no dependency on the presence logits.
///
/// Returning `None` instead of manufacturing a graph-connected zero is
/// important after the presence lock: a geometry-only optimizer step must not
/// accidentally traverse or update the presence branch.
fn geometry_observation_loss<B: Backend>(
    output: &KeypointRoofOutput<B>,
    batch: &ObservationBatch<B>,
) -> Option<Tensor<B, 1>> {
    if batch.geometry_count == 0 {
        return None;
    }

    // Presence-only photos and ordinary-building negatives do not need the
    // expensive 4,097-way D4 objective. Selecting synthetic-positive rows
    // before forming pairwise correspondences is exactly equivalent to
    // masking their losses afterward, without doing that work for the other
    // ten rows in the normal source-balanced batch.
    let keypoint_logits = output
        .keypoint_logits
        .clone()
        .select(0, batch.geometry_indices.clone());
    let offscreen_logits = output
        .offscreen_logits
        .clone()
        .select(0, batch.geometry_indices.clone());
    selected_geometry_observation_loss(keypoint_logits, offscreen_logits, batch)
}

/// Geometry objective for the low-memory training path, whose model input and
/// outputs have already been reduced to the geometry-bearing batch rows.
fn geometry_only_observation_loss<B: Backend>(
    output: &KeypointRoofGeometryOutput<B>,
    batch: &ObservationBatch<B>,
) -> Option<Tensor<B, 1>> {
    if batch.geometry_count == 0 {
        return None;
    }
    selected_geometry_observation_loss(
        output.keypoint_logits.clone(),
        output.offscreen_logits.clone(),
        batch,
    )
}

fn selected_geometry_observation_loss<B: Backend>(
    keypoint_logits: Tensor<B, 4>,
    offscreen_logits: Tensor<B, 2>,
    batch: &ObservationBatch<B>,
) -> Option<Tensor<B, 1>> {
    if batch.geometry_count == 0 {
        return None;
    }
    debug_assert_eq!(keypoint_logits.dims()[0], batch.geometry_count);
    debug_assert_eq!(offscreen_logits.dims()[0], batch.geometry_count);
    let targets = batch.keypoint_targets.clone();
    let batch_size = batch.geometry_count;
    let spatial = keypoint_logits.reshape([batch_size, KEYPOINT_COUNT, OFFSCREEN_INDEX]);
    let categorical_logits = Tensor::cat(vec![spatial, offscreen_logits.unsqueeze_dim(2)], 2);
    let log_probabilities = burn::tensor::activation::log_softmax(categorical_logits, 2).reshape([
        batch_size,
        3,
        4,
        KEYPOINT_DISTRIBUTION_SIZE,
    ]);
    let targets = targets.reshape([batch_size, 3, 4, KEYPOINT_DISTRIBUTION_SIZE]);
    let offscreen_log_probability = log_probabilities
        .clone()
        .slice_dim(3, OFFSCREEN_INDEX..KEYPOINT_DISTRIBUTION_SIZE)
        .squeeze_dim::<3>(3);
    let in_frame_log_probability = (offscreen_log_probability.clone().exp().neg() + 1.0)
        .clamp_min(1.0e-6)
        .log();
    let offscreen_targets = targets
        .clone()
        .slice_dim(3, OFFSCREEN_INDEX..KEYPOINT_DISTRIBUTION_SIZE)
        .squeeze_dim::<3>(3);
    let in_frame_targets = offscreen_targets.clone().neg() + 1.0;
    let pairwise_state = -(offscreen_log_probability.unsqueeze_dim::<4>(3)
        * offscreen_targets.unsqueeze_dim::<4>(2)
        + in_frame_log_probability.unsqueeze_dim::<4>(3) * in_frame_targets.unsqueeze_dim::<4>(2));
    // Pair every predicted slot with every target slot, then reduce only the
    // eight valid shared D4 mappings. This avoids materializing eight copies of
    // the 4,097-way target distributions.
    let pairwise = -(log_probabilities.unsqueeze_dim::<5>(3) * targets.unsqueeze_dim::<5>(2))
        .sum_dim(4)
        .squeeze_dim::<4>(4)
        + pairwise_state * OFFSCREEN_STATE_LOSS_WEIGHT;
    let mapped =
        pairwise.unsqueeze_dim::<5>(1) * batch.symmetry_masks.clone().unsqueeze_dim::<5>(0);
    let mapped = mapped
        .sum_dim(4)
        .squeeze_dim::<4>(4)
        .sum_dim(3)
        .squeeze_dim::<3>(3)
        .sum_dim(2)
        .squeeze_dim::<2>(2)
        / KEYPOINT_COUNT as f32;
    let best_correspondence = mapped.min_dim(1).squeeze_dim::<1>(1);
    let geometry_loss = best_correspondence.sum() / batch_size as f32;
    Some(geometry_loss)
}

#[derive(Clone, Debug, Default, Serialize)]
struct EvaluationMetrics {
    loss: f32,
    presence_threshold: f32,
    precision: f32,
    recall: f32,
    specificity: f32,
    synthetic_negative_specificity: f32,
    real_negative_specificity: f32,
    real_positive_recall: f32,
    real_positive_building_count: usize,
    real_positive_buildings_detected: usize,
    real_positive_building_recall: f32,
    real_positive_missing_building_id_count: usize,
    pck_03: f32,
    pck_05: f32,
    mean_keypoint_error: f32,
    true_positives: usize,
    false_positives: usize,
    true_negatives: usize,
    false_negatives: usize,
    evaluated_keypoints: usize,
    offscreen_precision: f32,
    offscreen_recall: f32,
    offscreen_accuracy: f32,
    offscreen_true_positives: usize,
    offscreen_false_positives: usize,
    offscreen_true_negatives: usize,
    offscreen_false_negatives: usize,
    duplicate_point_pairs: usize,
    real_positive_count: usize,
    synthetic_negative_count: usize,
    synthetic_negative_false_positives: usize,
    real_negative_count: usize,
    real_negative_false_positives: usize,
    presence_diagnostics: PresenceDiagnostics,
    presence_gate_diagnostics: PresenceGateDiagnostics,
    synthetic_fit: SyntheticFitMetrics,
}

impl EvaluationMetrics {
    fn deployment_presence_metrics(&self) -> (f32, f32, f32, f32) {
        if self.real_positive_count != 0 && self.real_negative_count != 0 {
            (
                self.real_positive_recall,
                self.real_negative_specificity,
                MIN_REAL_POSITIVE_RECALL,
                MIN_REAL_NEGATIVE_SPECIFICITY,
            )
        } else {
            // Synthetic-only memorisation runs and partially populated
            // diagnostics retain the historical aggregate contract.
            (
                self.recall,
                self.specificity,
                MIN_AGGREGATE_TARGET_RECALL,
                MIN_AGGREGATE_SPECIFICITY,
            )
        }
    }

    fn real_presence_gate_ratio(&self) -> f32 {
        let (recall, specificity, minimum_recall, minimum_specificity) =
            self.deployment_presence_metrics();
        let operating_ratio = (recall / minimum_recall)
            .min(specificity / minimum_specificity)
            .clamp(0.0, 1.0);
        if self.real_positive_count == 0 || self.real_negative_count == 0 {
            return operating_ratio;
        }
        if self.real_positive_missing_building_id_count != 0
            || self.real_positive_building_count == 0
        {
            return 0.0;
        }
        operating_ratio
            .min(self.real_positive_building_recall / MIN_REAL_POSITIVE_BUILDING_RECALL)
            .clamp(0.0, 1.0)
    }

    fn real_presence_gate_passes(&self) -> bool {
        let (recall, specificity, minimum_recall, minimum_specificity) =
            self.deployment_presence_metrics();
        let operating_point_passes = meets_minimum(recall, minimum_recall)
            && meets_minimum(specificity, minimum_specificity);
        if self.real_positive_count == 0 || self.real_negative_count == 0 {
            return operating_point_passes;
        }
        operating_point_passes
            && self.real_positive_missing_building_id_count == 0
            && self.real_positive_building_count != 0
            && meets_minimum(
                self.real_positive_building_recall,
                MIN_REAL_POSITIVE_BUILDING_RECALL,
            )
    }

    fn synthetic_ranking_metrics(&self) -> Option<(f32, f32)> {
        let roc_auc = self.presence_diagnostics.synthetic_roc_auc?;
        let average_precision = self.presence_diagnostics.synthetic_average_precision?;
        (roc_auc.is_finite() && average_precision.is_finite())
            .then_some((roc_auc, average_precision))
    }

    fn real_ranking_metrics(&self) -> Option<(f32, f32)> {
        let roc_auc = self.presence_diagnostics.real_roc_auc?;
        let average_precision = self.presence_diagnostics.real_average_precision?;
        (roc_auc.is_finite() && average_precision.is_finite())
            .then_some((roc_auc, average_precision))
    }

    fn synthetic_presence_gate_ratio(&self) -> f32 {
        self.synthetic_ranking_metrics()
            .map(|(roc_auc, average_precision)| {
                (roc_auc / MIN_SYNTHETIC_ROC_AUC)
                    .min(average_precision / MIN_SYNTHETIC_AVERAGE_PRECISION)
                    .clamp(0.0, 1.0)
            })
            .unwrap_or(0.0)
    }

    fn synthetic_presence_gate_passes(&self) -> bool {
        self.synthetic_ranking_metrics()
            .is_some_and(|(roc_auc, average_precision)| {
                meets_minimum(roc_auc, MIN_SYNTHETIC_ROC_AUC)
                    && meets_minimum(average_precision, MIN_SYNTHETIC_AVERAGE_PRECISION)
            })
    }

    fn geometry_quality(&self) -> f32 {
        let mut quality = 0.70 * self.pck_05 + 0.30 * self.offscreen_accuracy;
        if self.duplicate_point_pairs != 0 {
            quality -= 0.05 + 0.005 * self.duplicate_point_pairs.min(10) as f32;
        }
        quality
    }

    fn geometry_gate_passes(&self) -> bool {
        meets_minimum(self.pck_05, MIN_STANDARD_PCK)
            && meets_minimum(self.offscreen_accuracy, MIN_OFFSCREEN_ACCURACY)
    }

    fn presence_checkpoint_rank(&self) -> PresenceCheckpointRank {
        let (real_recall, real_specificity, minimum_recall, minimum_specificity) =
            self.deployment_presence_metrics();
        let real_ranking_metrics = self.real_ranking_metrics();
        let synthetic_ranking_metrics = self.synthetic_ranking_metrics();
        let synthetic_ranking_quality = synthetic_ranking_metrics
            .map(|(roc_auc, average_precision)| 0.5 * (roc_auc + average_precision))
            .unwrap_or(0.0);
        let robust_real_quality = real_ranking_metrics
            .map(|(roc_auc, average_precision)| {
                CHECKPOINT_REAL_ROBUSTNESS_WEIGHTS[0] * real_specificity
                    + CHECKPOINT_REAL_ROBUSTNESS_WEIGHTS[1] * roc_auc
                    + CHECKPOINT_REAL_ROBUSTNESS_WEIGHTS[2] * average_precision
                    + CHECKPOINT_REAL_ROBUSTNESS_WEIGHTS[3] * real_recall
            })
            .unwrap_or(0.0);
        let real_gate_margin = (real_recall - minimum_recall)
            .min(real_specificity - minimum_specificity)
            .max(0.0);
        let synthetic_margin = synthetic_ranking_metrics
            .map(|(roc_auc, average_precision)| {
                (roc_auc - MIN_SYNTHETIC_ROC_AUC)
                    .min(average_precision - MIN_SYNTHETIC_AVERAGE_PRECISION)
                    .max(0.0)
            })
            .unwrap_or(0.0);
        PresenceCheckpointRank {
            real_gate_passes: self.real_presence_gate_passes(),
            synthetic_gate_passes: self.synthetic_presence_gate_passes(),
            real_metrics_available: real_ranking_metrics.is_some(),
            robust_real_band: metric_basis_points(robust_real_quality)
                / REAL_ROBUSTNESS_BAND_BASIS_POINTS,
            real_gate_margin_band: metric_basis_points(real_gate_margin)
                / REAL_GATE_MARGIN_BAND_BASIS_POINTS,
            synthetic_margin_band: metric_basis_points(synthetic_margin)
                / SYNTHETIC_MARGIN_BAND_BASIS_POINTS,
            synthetic_quality_band: metric_basis_points(synthetic_ranking_quality)
                / SYNTHETIC_QUALITY_BAND_BASIS_POINTS,
            real_gate_ratio: self.real_presence_gate_ratio(),
            synthetic_gate_ratio: self.synthetic_presence_gate_ratio(),
            real_recall,
            real_specificity,
            synthetic_ranking_quality,
        }
    }

    fn checkpoint_rank(&self) -> CheckpointRank {
        CheckpointRank {
            presence: self.presence_checkpoint_rank(),
            geometry_gate_passes: self.geometry_gate_passes(),
            geometry_quality: self.geometry_quality(),
        }
    }
}

fn metric_basis_points(value: f32) -> u16 {
    if !value.is_finite() {
        return 0;
    }
    (value.clamp(0.0, 1.0) * 10_000.0).round() as u16
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
struct PresenceCheckpointRank {
    real_gate_passes: bool,
    synthetic_gate_passes: bool,
    real_metrics_available: bool,
    robust_real_band: u16,
    real_gate_margin_band: u16,
    synthetic_margin_band: u16,
    synthetic_quality_band: u16,
    real_gate_ratio: f32,
    synthetic_gate_ratio: f32,
    real_recall: f32,
    real_specificity: f32,
    synthetic_ranking_quality: f32,
}

impl PresenceCheckpointRank {
    fn ordering(self, other: Self) -> Ordering {
        let real_pass_order = self.real_gate_passes.cmp(&other.real_gate_passes);
        if !real_pass_order.is_eq() {
            return real_pass_order;
        }
        if !self.real_gate_passes {
            return self
                .real_gate_ratio
                .total_cmp(&other.real_gate_ratio)
                .then_with(|| self.real_recall.total_cmp(&other.real_recall))
                .then_with(|| self.real_specificity.total_cmp(&other.real_specificity))
                .then_with(|| {
                    self.synthetic_gate_ratio
                        .total_cmp(&other.synthetic_gate_ratio)
                })
                .then_with(|| {
                    self.synthetic_ranking_quality
                        .total_cmp(&other.synthetic_ranking_quality)
                });
        }

        let synthetic_pass_order = self.synthetic_gate_passes.cmp(&other.synthetic_gate_passes);
        if !synthetic_pass_order.is_eq() {
            return synthetic_pass_order;
        }
        if !self.synthetic_gate_passes {
            return self
                .synthetic_gate_ratio
                .total_cmp(&other.synthetic_gate_ratio)
                .then_with(|| {
                    self.synthetic_ranking_quality
                        .total_cmp(&other.synthetic_ranking_quality)
                })
                .then_with(|| self.real_specificity.total_cmp(&other.real_specificity))
                .then_with(|| self.real_recall.total_cmp(&other.real_recall));
        }

        self.real_metrics_available
            .cmp(&other.real_metrics_available)
            .then_with(|| self.robust_real_band.cmp(&other.robust_real_band))
            .then_with(|| self.real_gate_margin_band.cmp(&other.real_gate_margin_band))
            .then_with(|| self.synthetic_margin_band.cmp(&other.synthetic_margin_band))
            .then_with(|| {
                self.synthetic_quality_band
                    .cmp(&other.synthetic_quality_band)
            })
    }

    fn is_better_than(self, other: Self) -> bool {
        self.ordering(other).is_gt()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
struct CheckpointRank {
    presence: PresenceCheckpointRank,
    geometry_gate_passes: bool,
    geometry_quality: f32,
}

impl CheckpointRank {
    fn is_better_than(self, other: Self) -> bool {
        let both_presence_safe = self.presence.real_gate_passes
            && self.presence.synthetic_gate_passes
            && other.presence.real_gate_passes
            && other.presence.synthetic_gate_passes;
        let ordering = if !both_presence_safe {
            self.presence
                .ordering(other.presence)
                .then_with(|| self.geometry_quality.total_cmp(&other.geometry_quality))
        } else {
            let geometry_pass_order = self.geometry_gate_passes.cmp(&other.geometry_gate_passes);
            if !geometry_pass_order.is_eq() {
                geometry_pass_order
            } else if !self.geometry_gate_passes {
                // Before geometry is viable, keep moving toward the promotion
                // floor instead of freezing on an early presence checkpoint.
                self.geometry_quality
                    .total_cmp(&other.geometry_quality)
                    .then_with(|| self.presence.ordering(other.presence))
            } else {
                // Once every hard gate is viable, deployment-domain stability
                // outranks sub-band single-frame geometry improvements.
                self.presence
                    .ordering(other.presence)
                    .then_with(|| self.geometry_quality.total_cmp(&other.geometry_quality))
            }
        };
        ordering.is_gt()
    }

    fn selection_score(self) -> f32 {
        if self.presence.real_gate_passes && self.presence.synthetic_gate_passes {
            if self.geometry_gate_passes {
                30.0 + self.presence.robust_real_band as f32 / 100.0
                    + self.presence.real_gate_margin_band as f32 / 10_000.0
                    + self.presence.synthetic_margin_band as f32 / 1_000_000.0
            } else {
                20.0 + self.geometry_quality
            }
        } else if self.presence.real_gate_passes {
            10.0 + self.presence.synthetic_gate_ratio + 0.001 * self.geometry_quality
        } else {
            self.presence.real_gate_ratio
                + 0.001 * self.presence.synthetic_gate_ratio
                + 0.000_001 * self.geometry_quality
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct EpochMetrics {
    epoch: usize,
    presence_freeze_policy: &'static str,
    presence_freeze_after_safe_epochs: usize,
    presence_safe_streak: usize,
    presence_frozen_after_epoch: Option<usize>,
    presence_frozen_from_epoch: Option<usize>,
    presence_trainable: bool,
    optimizer_stage: OptimizerStage,
    backbone_trainable: bool,
    head_updates_completed: usize,
    head_total_updates: usize,
    joint_head_updates_completed: usize,
    geometry_only_head_updates_completed: usize,
    backbone_updates_completed: usize,
    backbone_total_updates: usize,
    train_loss: f32,
    train_presence_loss: f32,
    /// Raw pairwise ranking loss before
    /// [`REAL_PRESENCE_PAIRWISE_LOSS_WEIGHT`] is applied.
    train_real_pairwise_loss: f32,
    /// Raw geometry loss before [`GEOMETRY_LOSS_WEIGHT`] is applied.
    train_geometry_loss: f32,
    validation: EvaluationMetrics,
    checkpoint_rank: CheckpointRank,
    selection_score: f32,
}

#[derive(Clone, Debug, Serialize)]
struct RefinementSourceMetrics {
    schema_version: &'static str,
    source_checkpoint: String,
    source_epoch: usize,
    first_training_epoch: usize,
    source_contract: AuthenticatedRefinementSource,
    real_presence_safe: bool,
    synthetic_presence_safe: bool,
    validation: EvaluationMetrics,
    checkpoint_rank: CheckpointRank,
    selection_score: f32,
}

#[derive(Clone, Debug, Serialize)]
struct FinalMetrics {
    schema_version: String,
    validation: EvaluationMetrics,
    test: EvaluationMetrics,
    promotion: PromotionDecision,
}

#[derive(Clone, Copy, Debug)]
struct PresenceObservation {
    probability: f32,
    positive: bool,
    origin: SampleOrigin,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct RealPositiveBuildingMetrics {
    count: usize,
    detected: usize,
    recall: f32,
    missing_id_count: usize,
}

/// Collapse correlated views to one decision per physical building by taking
/// the strongest view. Missing identities invalidate building-level recall;
/// silently treating each missing view as a new site would inflate coverage.
fn real_positive_building_metrics(
    observations: &[(Option<Arc<str>>, f32)],
    threshold: f32,
) -> RealPositiveBuildingMetrics {
    let mut maxima = BTreeMap::<Arc<str>, f32>::new();
    let mut missing_id_count = 0usize;
    for (building_id, probability) in observations {
        let Some(building_id) = building_id else {
            missing_id_count += 1;
            continue;
        };
        maxima
            .entry(building_id.clone())
            .and_modify(|maximum| *maximum = maximum.max(*probability))
            .or_insert(*probability);
    }
    let count = maxima.len();
    let detected = maxima
        .values()
        .filter(|probability| **probability >= threshold)
        .count();
    let recall = if missing_id_count == 0 && count != 0 {
        detected as f32 / count as f32
    } else {
        0.0
    };
    RealPositiveBuildingMetrics {
        count,
        detected,
        recall,
        missing_id_count,
    }
}

#[derive(Clone, Copy, Debug, Default, Serialize)]
struct PresenceGateDiagnostics {
    common_threshold_feasible: bool,
    best_gate_ratio: f32,
    best_threshold: f32,
    recall: f32,
    specificity: f32,
    real_positive_recall: f32,
    real_negative_specificity: f32,
}

impl PresenceGateDiagnostics {
    fn from_observations(observations: &[PresenceObservation]) -> Self {
        if observations.is_empty() {
            return Self {
                best_threshold: 0.5,
                ..Self::default()
            };
        }

        let mut candidates = observations
            .iter()
            .map(|observation| observation.probability.clamp(0.0, 1.0))
            .collect::<Vec<_>>();
        candidates.extend([0.0, 0.5, 1.0]);
        candidates.sort_by(f32::total_cmp);
        candidates.dedup_by(|left, right| left.total_cmp(right).is_eq());

        let mut best: Option<Self> = None;
        for threshold in candidates {
            let rates = PresenceRates::from_observations(observations, threshold);
            let gate_ratio = rates.gate_ratio();
            let candidate = Self {
                common_threshold_feasible: gate_ratio >= 1.0,
                best_gate_ratio: gate_ratio,
                best_threshold: threshold,
                recall: rates.recall,
                specificity: rates.specificity,
                real_positive_recall: rates.real_positive_recall,
                real_negative_specificity: rates.real_negative_specificity,
            };
            let improves = best.is_none_or(|current| {
                candidate
                    .best_gate_ratio
                    .total_cmp(&current.best_gate_ratio)
                    .then_with(|| candidate.specificity.total_cmp(&current.specificity))
                    .then_with(|| {
                        candidate
                            .real_negative_specificity
                            .total_cmp(&current.real_negative_specificity)
                    })
                    .then_with(|| candidate.best_threshold.total_cmp(&current.best_threshold))
                    .is_gt()
            });
            if improves {
                best = Some(candidate);
            }
        }
        best.expect("non-empty observations always produce threshold candidates")
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct PresenceRates {
    recall: f32,
    specificity: f32,
    real_positive_recall: f32,
    real_negative_specificity: f32,
}

impl PresenceRates {
    fn from_observations(observations: &[PresenceObservation], threshold: f32) -> Self {
        let mut true_positives = 0usize;
        let mut positives = 0usize;
        let mut true_negatives = 0usize;
        let mut negatives = 0usize;
        let mut real_positive_correct = 0usize;
        let mut real_positive_count = 0usize;
        let mut real_negative_correct = 0usize;
        let mut real_negative_count = 0usize;
        for observation in observations {
            let predicted = observation.probability >= threshold;
            if observation.positive {
                positives += 1;
                true_positives += usize::from(predicted);
            } else {
                negatives += 1;
                true_negatives += usize::from(!predicted);
            }
            match observation.origin {
                SampleOrigin::RealPositive => {
                    real_positive_count += 1;
                    real_positive_correct += usize::from(predicted);
                }
                SampleOrigin::RealNegative => {
                    real_negative_count += 1;
                    real_negative_correct += usize::from(!predicted);
                }
                SampleOrigin::SyntheticTarget | SampleOrigin::SyntheticNegative => {}
            }
        }
        let recall = true_positives as f32 / positives.max(1) as f32;
        let specificity = true_negatives as f32 / negatives.max(1) as f32;
        Self {
            recall,
            specificity,
            real_positive_recall: if real_positive_count == 0 {
                1.0
            } else {
                real_positive_correct as f32 / real_positive_count as f32
            },
            real_negative_specificity: if real_negative_count == 0 {
                specificity
            } else {
                real_negative_correct as f32 / real_negative_count as f32
            },
        }
    }

    fn gate_ratio(self) -> f32 {
        (self.recall / MIN_AGGREGATE_TARGET_RECALL)
            .min(self.specificity / MIN_AGGREGATE_SPECIFICITY)
            .min(self.real_positive_recall / MIN_REAL_POSITIVE_RECALL)
            .min(self.real_negative_specificity / MIN_REAL_NEGATIVE_SPECIFICITY)
            .clamp(0.0, 1.0)
    }
}

#[derive(Clone, Debug, Default, Serialize)]
struct ProbabilityQuantiles {
    count: usize,
    minimum: f32,
    p10: f32,
    p25: f32,
    median: f32,
    p75: f32,
    p90: f32,
    maximum: f32,
}

impl ProbabilityQuantiles {
    fn from_values(mut values: Vec<f32>) -> Self {
        if values.is_empty() {
            return Self::default();
        }
        values.sort_by(f32::total_cmp);
        Self {
            count: values.len(),
            minimum: percentile(&values, 0.0),
            p10: percentile(&values, 0.10),
            p25: percentile(&values, 0.25),
            median: percentile(&values, 0.50),
            p75: percentile(&values, 0.75),
            p90: percentile(&values, 0.90),
            maximum: percentile(&values, 1.0),
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
struct PresenceDiagnostics {
    synthetic_targets: ProbabilityQuantiles,
    synthetic_negatives: ProbabilityQuantiles,
    real_positives: ProbabilityQuantiles,
    real_negatives: ProbabilityQuantiles,
    roc_auc: Option<f32>,
    average_precision: Option<f32>,
    synthetic_roc_auc: Option<f32>,
    synthetic_average_precision: Option<f32>,
    real_roc_auc: Option<f32>,
    real_average_precision: Option<f32>,
    real_positive_vs_synthetic_negative_roc_auc: Option<f32>,
    real_positive_vs_synthetic_negative_average_precision: Option<f32>,
    synthetic_target_vs_real_negative_roc_auc: Option<f32>,
    synthetic_target_vs_real_negative_average_precision: Option<f32>,
}

impl PresenceDiagnostics {
    fn from_observations(observations: &[PresenceObservation]) -> Self {
        let values_for = |origin| {
            observations
                .iter()
                .filter(|observation| observation.origin == origin)
                .map(|observation| observation.probability)
                .collect::<Vec<_>>()
        };
        let synthetic = observations
            .iter()
            .copied()
            .filter(|observation| {
                matches!(
                    observation.origin,
                    SampleOrigin::SyntheticTarget | SampleOrigin::SyntheticNegative
                )
            })
            .collect::<Vec<_>>();
        let real = observations
            .iter()
            .copied()
            .filter(|observation| {
                matches!(
                    observation.origin,
                    SampleOrigin::RealPositive | SampleOrigin::RealNegative
                )
            })
            .collect::<Vec<_>>();
        let real_positive_vs_synthetic_negative = observations
            .iter()
            .copied()
            .filter(|observation| {
                matches!(
                    observation.origin,
                    SampleOrigin::RealPositive | SampleOrigin::SyntheticNegative
                )
            })
            .collect::<Vec<_>>();
        let synthetic_target_vs_real_negative = observations
            .iter()
            .copied()
            .filter(|observation| {
                matches!(
                    observation.origin,
                    SampleOrigin::SyntheticTarget | SampleOrigin::RealNegative
                )
            })
            .collect::<Vec<_>>();
        let overall_metrics = binary_ranking_metrics(observations);
        let synthetic_metrics = binary_ranking_metrics(&synthetic);
        let real_metrics = binary_ranking_metrics(&real);
        let real_positive_vs_synthetic_negative_metrics =
            binary_ranking_metrics(&real_positive_vs_synthetic_negative);
        let synthetic_target_vs_real_negative_metrics =
            binary_ranking_metrics(&synthetic_target_vs_real_negative);
        Self {
            synthetic_targets: ProbabilityQuantiles::from_values(values_for(
                SampleOrigin::SyntheticTarget,
            )),
            synthetic_negatives: ProbabilityQuantiles::from_values(values_for(
                SampleOrigin::SyntheticNegative,
            )),
            real_positives: ProbabilityQuantiles::from_values(values_for(
                SampleOrigin::RealPositive,
            )),
            real_negatives: ProbabilityQuantiles::from_values(values_for(
                SampleOrigin::RealNegative,
            )),
            roc_auc: overall_metrics.map(|metrics| metrics.0),
            average_precision: overall_metrics.map(|metrics| metrics.1),
            synthetic_roc_auc: synthetic_metrics.map(|metrics| metrics.0),
            synthetic_average_precision: synthetic_metrics.map(|metrics| metrics.1),
            real_roc_auc: real_metrics.map(|metrics| metrics.0),
            real_average_precision: real_metrics.map(|metrics| metrics.1),
            real_positive_vs_synthetic_negative_roc_auc:
                real_positive_vs_synthetic_negative_metrics.map(|metrics| metrics.0),
            real_positive_vs_synthetic_negative_average_precision:
                real_positive_vs_synthetic_negative_metrics.map(|metrics| metrics.1),
            synthetic_target_vs_real_negative_roc_auc: synthetic_target_vs_real_negative_metrics
                .map(|metrics| metrics.0),
            synthetic_target_vs_real_negative_average_precision:
                synthetic_target_vs_real_negative_metrics.map(|metrics| metrics.1),
        }
    }
}

fn percentile(sorted: &[f32], probability: f32) -> f32 {
    let position = probability * sorted.len().saturating_sub(1) as f32;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    let fraction = position - lower as f32;
    sorted[lower] + fraction * (sorted[upper] - sorted[lower])
}

fn binary_ranking_metrics(observations: &[PresenceObservation]) -> Option<(f32, f32)> {
    let positives = observations
        .iter()
        .filter(|observation| observation.positive)
        .count();
    let negatives = observations.len().saturating_sub(positives);
    if positives == 0 || negatives == 0 {
        return None;
    }

    let mut ranked = observations.to_vec();
    ranked.sort_by(|left, right| left.probability.total_cmp(&right.probability));
    let mut positive_rank_sum = 0.0_f64;
    let mut start = 0usize;
    while start < ranked.len() {
        let mut end = start + 1;
        while end < ranked.len()
            && ranked[end]
                .probability
                .total_cmp(&ranked[start].probability)
                .is_eq()
        {
            end += 1;
        }
        let average_rank = (start + 1 + end) as f64 / 2.0;
        let positive_count = ranked[start..end]
            .iter()
            .filter(|observation| observation.positive)
            .count();
        positive_rank_sum += average_rank * positive_count as f64;
        start = end;
    }
    let roc_auc = (positive_rank_sum - (positives * (positives + 1) / 2) as f64)
        / (positives * negatives) as f64;

    ranked.sort_by(|left, right| right.probability.total_cmp(&left.probability));
    let mut seen = 0usize;
    let mut true_positives = 0usize;
    let mut precision_sum = 0.0_f64;
    let mut start = 0usize;
    while start < ranked.len() {
        let mut end = start + 1;
        while end < ranked.len()
            && ranked[end]
                .probability
                .total_cmp(&ranked[start].probability)
                .is_eq()
        {
            end += 1;
        }
        let group_positives = ranked[start..end]
            .iter()
            .filter(|observation| observation.positive)
            .count();
        seen += end - start;
        true_positives += group_positives;
        precision_sum += group_positives as f64 * true_positives as f64 / seen as f64;
        start = end;
    }
    Some((roc_auc as f32, (precision_sum / positives as f64) as f32))
}

fn format_optional_metric(metric: Option<f32>) -> String {
    metric.map_or_else(|| "n/a".to_owned(), |value| format!("{value:.3}"))
}

fn print_probability_quantiles(epoch: usize, diagnostics: &PresenceDiagnostics) {
    for (name, quantiles) in [
        ("synthetic_targets", &diagnostics.synthetic_targets),
        ("synthetic_negatives", &diagnostics.synthetic_negatives),
        ("real_positives", &diagnostics.real_positives),
        ("real_negatives", &diagnostics.real_negatives),
    ] {
        if quantiles.count != 0 {
            println!(
                "epoch {epoch:02} presence {name}: n={} min={:.3} p10={:.3} p50={:.3} p90={:.3} max={:.3}",
                quantiles.count,
                quantiles.minimum,
                quantiles.p10,
                quantiles.median,
                quantiles.p90,
                quantiles.maximum,
            );
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct RealThresholdCandidate {
    viable: bool,
    identities_complete: bool,
    specificity: f32,
    building_recall: f32,
    view_recall: f32,
    threshold: f32,
}

impl RealThresholdCandidate {
    fn coverage_ratio(self) -> f32 {
        if !self.identities_complete {
            return 0.0;
        }
        (self.view_recall / MIN_REAL_POSITIVE_RECALL)
            .min(self.building_recall / MIN_REAL_POSITIVE_BUILDING_RECALL)
            .clamp(0.0, 1.0)
    }

    fn is_better_than(self, other: Self) -> bool {
        let ordering = self.viable.cmp(&other.viable);
        let ordering = if !ordering.is_eq() {
            ordering
        } else if self.viable {
            // Once both coverage floors are met, specificity chooses the
            // deployed operating point. Equal-specificity thresholds retain
            // more physical-building and view coverage before preferring a
            // more conservative numerical cutoff.
            self.specificity
                .total_cmp(&other.specificity)
                .then_with(|| self.building_recall.total_cmp(&other.building_recall))
                .then_with(|| self.view_recall.total_cmp(&other.view_recall))
                .then_with(|| self.threshold.total_cmp(&other.threshold))
        } else {
            // Before the floors are viable, move toward complete, balanced
            // physical-site coverage rather than optimizing specificity at a
            // threshold that can never be promoted.
            self.identities_complete
                .cmp(&other.identities_complete)
                .then_with(|| self.coverage_ratio().total_cmp(&other.coverage_ratio()))
                .then_with(|| self.building_recall.total_cmp(&other.building_recall))
                .then_with(|| self.view_recall.total_cmp(&other.view_recall))
                .then_with(|| self.specificity.total_cmp(&other.specificity))
                .then_with(|| self.threshold.total_cmp(&other.threshold))
        };
        ordering.is_gt()
    }
}

fn calibrate_presence_threshold(
    observations: &[PresenceObservation],
    real_positive_building_observations: &[(Option<Arc<str>>, f32)],
) -> f32 {
    if observations.is_empty() {
        return 0.5;
    }
    let has_real_positives = observations
        .iter()
        .any(|observation| observation.origin == SampleOrigin::RealPositive);
    let has_real_negatives = observations
        .iter()
        .any(|observation| observation.origin == SampleOrigin::RealNegative);
    let calibrate_on_real_photos = has_real_positives && has_real_negatives;
    let calibration_observations = observations
        .iter()
        .copied()
        .filter(|observation| {
            !calibrate_on_real_photos
                || matches!(
                    observation.origin,
                    SampleOrigin::RealPositive | SampleOrigin::RealNegative
                )
        })
        .collect::<Vec<_>>();
    let mut candidates = calibration_observations
        .iter()
        .map(|observation| observation.probability.clamp(0.0, 1.0))
        .collect::<Vec<_>>();
    candidates.extend([0.0, 0.5, 1.0]);
    candidates.sort_by(f32::total_cmp);
    // Tiny probabilities are meaningful here: the failed run needed a cutoff
    // around 4e-7 to retain seven of eight validation positives. Approximate
    // deduplication merged that value into 0.0 and silently changed the model's
    // decision boundary. Remove only bit-identical candidates.
    candidates.dedup_by(|left, right| left.total_cmp(right).is_eq());

    let mut best_aggregate = (f32::NEG_INFINITY, 0.5_f32);
    let mut best_real: Option<RealThresholdCandidate> = None;
    for threshold in candidates {
        let mut true_positives = 0usize;
        let mut false_negatives = 0usize;
        let mut true_negatives = 0usize;
        let mut false_positives = 0usize;
        let mut real_correct = 0usize;
        let mut real_total = 0usize;
        let mut real_negative_correct = 0usize;
        let mut real_negative_total = 0usize;
        for observation in &calibration_observations {
            let predicted = observation.probability >= threshold;
            match (observation.positive, predicted) {
                (true, true) => true_positives += 1,
                (true, false) => false_negatives += 1,
                (false, true) => false_positives += 1,
                (false, false) => true_negatives += 1,
            }
            if observation.origin == SampleOrigin::RealPositive {
                real_total += 1;
                real_correct += usize::from(predicted);
            }
            if observation.origin == SampleOrigin::RealNegative {
                real_negative_total += 1;
                real_negative_correct += usize::from(!predicted);
            }
        }
        let recall = true_positives as f32 / (true_positives + false_negatives).max(1) as f32;
        let specificity = true_negatives as f32 / (true_negatives + false_positives).max(1) as f32;
        let real_recall = if has_real_positives {
            real_correct as f32 / real_total.max(1) as f32
        } else {
            1.0
        };
        let real_specificity = if has_real_negatives {
            real_negative_correct as f32 / real_negative_total.max(1) as f32
        } else {
            specificity
        };
        // A deployed frame is always a real photograph. When both real
        // validation classes exist, synthetic score offsets must not move the
        // operating threshold. Keep the previous aggregate behavior only as a
        // fallback for synthetic-only/partially populated diagnostic corpora.
        if calibrate_on_real_photos {
            let building_metrics =
                real_positive_building_metrics(real_positive_building_observations, threshold);
            let identities_complete =
                building_metrics.missing_id_count == 0 && building_metrics.count != 0;
            let viable = identities_complete
                && meets_minimum(real_recall, MIN_REAL_POSITIVE_RECALL)
                && meets_minimum(building_metrics.recall, MIN_REAL_POSITIVE_BUILDING_RECALL);
            let candidate = RealThresholdCandidate {
                viable,
                identities_complete,
                specificity: real_specificity,
                building_recall: building_metrics.recall,
                view_recall: real_recall,
                threshold,
            };
            let improves = best_real.is_none_or(|current| candidate.is_better_than(current));
            if improves {
                best_real = Some(candidate);
            }
        } else {
            let viable = recall >= MIN_AGGREGATE_TARGET_RECALL
                && (!has_real_positives || meets_minimum(real_recall, MIN_REAL_POSITIVE_RECALL));
            let score = if viable {
                10.0 + 0.5 * specificity
                    + 0.5 * real_specificity
                    + 0.1 * real_recall
                    + threshold * 1.0e-4
            } else {
                recall + 0.125 * specificity + 0.125 * real_specificity + 0.25 * real_recall
            };
            if score > best_aggregate.0 {
                best_aggregate = (score, threshold);
            }
        }
    }
    if calibrate_on_real_photos {
        best_real
            .expect("real calibration always has threshold candidates")
            .threshold
    } else {
        best_aggregate.1
    }
}

#[derive(Clone, Debug, Serialize)]
struct PromotionDecision {
    promoted: bool,
    failures: Vec<String>,
}

impl PromotionDecision {
    fn from_metrics(args: &Args, validation: &EvaluationMetrics, test: &EvaluationMetrics) -> Self {
        let mut failures = Vec::new();
        if let Some(limit) = args.limit_per_split {
            failures.push(format!(
                "diagnostic --limit-per-split {limit} is active; capped splits cannot promote a production checkpoint"
            ));
        }
        if args.overfit {
            check_minimum(&mut failures, "test recall", test.recall, 1.0);
            check_minimum(&mut failures, "test specificity", test.specificity, 1.0);
            check_minimum(&mut failures, "test PCK@3%", test.pck_03, 0.98);
            check_minimum(
                &mut failures,
                "test offscreen accuracy",
                test.offscreen_accuracy,
                0.98,
            );
            if test.duplicate_point_pairs != 0 {
                failures.push(format!(
                    "test has {} duplicated/collapsed keypoint pairs",
                    test.duplicate_point_pairs
                ));
            }
        } else {
            if validation.real_positive_count == 0 {
                failures.push("validation has no held-out real Pizza Hut positives".to_owned());
            } else {
                check_minimum(
                    &mut failures,
                    "validation real-photo recall",
                    validation.real_positive_recall,
                    MIN_REAL_POSITIVE_VIEW_RECALL,
                );
            }
            check_real_positive_building_coverage(&mut failures, "validation", validation);
            if validation.real_negative_count == 0 {
                failures.push("validation has no curated real-building negatives".to_owned());
            } else {
                check_minimum(
                    &mut failures,
                    "validation real-negative specificity",
                    validation.real_negative_specificity,
                    MIN_REAL_NEGATIVE_SPECIFICITY,
                );
            }
            if test.real_positive_count == 0 {
                failures.push("test has no held-out real Pizza Hut positives".to_owned());
            } else {
                check_minimum(
                    &mut failures,
                    "test real-photo recall",
                    test.real_positive_recall,
                    MIN_REAL_POSITIVE_VIEW_RECALL,
                );
            }
            check_real_positive_building_coverage(&mut failures, "test", test);
            if test.real_negative_count == 0 {
                failures.push("test has no curated real-building negatives".to_owned());
            } else {
                check_minimum(
                    &mut failures,
                    "test real-negative specificity",
                    test.real_negative_specificity,
                    MIN_REAL_NEGATIVE_SPECIFICITY,
                );
            }
            check_optional_minimum(
                &mut failures,
                "test synthetic ROC AUC",
                test.presence_diagnostics.synthetic_roc_auc,
                MIN_SYNTHETIC_ROC_AUC,
            );
            check_optional_minimum(
                &mut failures,
                "test synthetic average precision",
                test.presence_diagnostics.synthetic_average_precision,
                MIN_SYNTHETIC_AVERAGE_PRECISION,
            );
            check_minimum(&mut failures, "test PCK@5%", test.pck_05, MIN_STANDARD_PCK);
            check_minimum(
                &mut failures,
                "test offscreen accuracy",
                test.offscreen_accuracy,
                MIN_OFFSCREEN_ACCURACY,
            );
            if test.synthetic_fit.fitted == 0 {
                failures.push("test has no successfully fitted synthetic roofs".to_owned());
            } else {
                check_minimum(
                    &mut failures,
                    "test synthetic fit success rate",
                    test.synthetic_fit.fit_success_rate,
                    0.90,
                );
                check_minimum(
                    &mut failures,
                    "test accepted-fit coverage",
                    test.synthetic_fit.accepted_rate,
                    0.80,
                );
                if test.synthetic_fit.median_mesh_rmse > MAX_MEDIAN_FITTED_MESH_RMSE + 1.0e-6 {
                    failures.push(format!(
                        "test median fitted-mesh RMSE {:.3} exceeds required {:.3}",
                        test.synthetic_fit.median_mesh_rmse, MAX_MEDIAN_FITTED_MESH_RMSE,
                    ));
                }
                check_minimum(
                    &mut failures,
                    "test median amodal silhouette IoU",
                    test.synthetic_fit.median_silhouette_iou,
                    MIN_MEDIAN_AMODAL_SILHOUETTE_IOU,
                );
            }
        }
        Self {
            promoted: failures.is_empty(),
            failures,
        }
    }
}

fn check_real_positive_building_coverage(
    failures: &mut Vec<String>,
    split: &str,
    metrics: &EvaluationMetrics,
) {
    if metrics.real_positive_missing_building_id_count != 0 {
        failures.push(format!(
            "{split} has {} real-positive views without a physical building ID",
            metrics.real_positive_missing_building_id_count,
        ));
    }
    if metrics.real_positive_building_count == 0 {
        failures.push(format!(
            "{split} has no identified held-out real Pizza Hut buildings"
        ));
    } else {
        check_minimum(
            failures,
            &format!("{split} real-positive building recall"),
            metrics.real_positive_building_recall,
            MIN_REAL_POSITIVE_BUILDING_RECALL,
        );
    }
}

fn meets_minimum(actual: f32, required: f32) -> bool {
    actual + 1.0e-6 >= required
}

fn check_minimum(failures: &mut Vec<String>, name: &str, actual: f32, required: f32) {
    if !meets_minimum(actual, required) {
        failures.push(format!(
            "{name} {actual:.3} is below required {required:.3}"
        ));
    }
}

fn check_optional_minimum(
    failures: &mut Vec<String>,
    name: &str,
    actual: Option<f32>,
    required: f32,
) {
    match actual {
        Some(actual) if actual.is_finite() => {
            check_minimum(failures, name, actual, required);
        }
        Some(_) => failures.push(format!("{name} is non-finite")),
        None => failures.push(format!("{name} is unavailable")),
    }
}

fn evaluate<B: Backend>(
    model: &KeypointRoofNet<B>,
    samples: &[TrainingSample],
    device: &Device<B>,
    batch_size: usize,
    threshold: Option<f32>,
    compute_fit_metrics: bool,
    fit_shape_prior: Option<[f32; 7]>,
) -> Result<EvaluationMetrics> {
    let mut metrics = EvaluationMetrics::default();
    let mut loss_sum = 0.0;
    let mut batches = 0usize;
    let mut real_positive_correct = 0usize;
    let mut real_positive_count = 0usize;
    let mut strict_correct = 0usize;
    let mut standard_correct = 0usize;
    let mut keypoint_error = 0.0;
    let mut presence_observations = Vec::with_capacity(samples.len());
    let mut real_positive_building_observations = Vec::new();
    let mut pending_fits = Vec::<(
        [Option<[f32; 2]>; KEYPOINT_COUNT],
        [f32; KEYPOINT_COUNT],
        Arc<FrameRecord>,
    )>::new();

    for chunk in samples.chunks(batch_size) {
        let batch = make_batch::<B>(chunk, device, 0, false, None)?;
        let output = model.forward(batch.images.clone());
        let presence_logits = output.presence_logits.clone();
        let keypoint_logits = output.keypoint_logits.clone();
        let offscreen_logits = output.offscreen_logits.clone();
        // Validation/test batches preserve dataset order rather than the
        // source-balanced training composition, so their reported loss keeps
        // the established per-sample BCE + geometry definition. Ranking
        // quality is reported independently by the presence diagnostics.
        loss_sum += observation_loss(output, &batch, false)
            .total
            .into_scalar()
            .elem::<f32>();
        batches += 1;

        let probabilities = burn::tensor::activation::sigmoid(presence_logits)
            .to_data()
            .to_vec::<f32>()?;
        for (probability, sample) in probabilities.into_iter().zip(chunk) {
            if sample.origin == SampleOrigin::RealPositive {
                real_positive_building_observations
                    .push((sample.physical_building_id.clone(), probability));
            }
            presence_observations.push(PresenceObservation {
                probability,
                positive: sample.presence == 1,
                origin: sample.origin,
            });
        }

        let keypoint_values = keypoint_logits.to_data().to_vec::<f32>()?;
        let offscreen_values = offscreen_logits.to_data().to_vec::<f32>()?;
        let map_values = KEYPOINT_COUNT * OFFSCREEN_INDEX;
        for sample_index in 0..chunk.len() {
            if chunk[sample_index].geometry.is_none() {
                continue;
            }
            let maps = &keypoint_values[sample_index * map_values..(sample_index + 1) * map_values];
            let offscreen = &offscreen_values
                [sample_index * KEYPOINT_COUNT..(sample_index + 1) * KEYPOINT_COUNT];
            let (predictions, prediction_confidences) = decode_metric_points(maps, offscreen);
            let transform = batch.transforms[sample_index];
            let raw_source_predictions = predictions.map(|prediction| {
                prediction.and_then(|point| transform.unmap_point_if_in_source(point))
            });
            let source_predictions = std::array::from_fn(|index| {
                (prediction_confidences[index] >= DEFAULT_FIT_KEYPOINT_CONFIDENCE)
                    .then_some(raw_source_predictions[index])
                    .flatten()
            });
            let source_targets =
                batch.positions[sample_index].map(|point| transform.unmap_point_unclamped(point));
            let correspondence = best_correspondence(
                &source_predictions,
                &source_targets,
                &batch.in_frame[sample_index],
                transform.source_width,
                transform.source_height,
            );
            for (predicted_index, raw_prediction) in raw_source_predictions.iter().enumerate() {
                let ring = predicted_index / 4;
                let target_index =
                    ring * 4 + symmetry_target_slot(correspondence.hypothesis, predicted_index % 4);
                let predicted_offscreen = raw_prediction.is_none();
                let target_offscreen = !batch.in_frame[sample_index][target_index];
                match (target_offscreen, predicted_offscreen) {
                    (true, true) => metrics.offscreen_true_positives += 1,
                    (false, true) => metrics.offscreen_false_positives += 1,
                    (false, false) => metrics.offscreen_true_negatives += 1,
                    (true, false) => metrics.offscreen_false_negatives += 1,
                }
                if !target_offscreen {
                    let error = correspondence.errors[predicted_index];
                    metrics.evaluated_keypoints += 1;
                    keypoint_error += error;
                    strict_correct += usize::from(error <= PCK_STRICT_THRESHOLD);
                    standard_correct += usize::from(error <= PCK_STANDARD_THRESHOLD);
                }
            }
            metrics.duplicate_point_pairs += duplicate_point_pairs(
                &predictions,
                &batch.positions[sample_index],
                &batch.in_frame[sample_index],
                correspondence.hypothesis,
            );
            if compute_fit_metrics && chunk[sample_index].origin == SampleOrigin::SyntheticTarget {
                metrics.synthetic_fit.record_attempt();
                if let Some(frame) = &chunk[sample_index].geometry {
                    pending_fits.push((source_predictions, prediction_confidences, frame.clone()));
                }
            }
        }
    }

    metrics.presence_diagnostics = PresenceDiagnostics::from_observations(&presence_observations);
    metrics.presence_gate_diagnostics =
        PresenceGateDiagnostics::from_observations(&presence_observations);
    let threshold = threshold.unwrap_or_else(|| {
        calibrate_presence_threshold(&presence_observations, &real_positive_building_observations)
    });
    metrics.presence_threshold = threshold;
    let building_metrics =
        real_positive_building_metrics(&real_positive_building_observations, threshold);
    metrics.real_positive_building_count = building_metrics.count;
    metrics.real_positive_buildings_detected = building_metrics.detected;
    metrics.real_positive_building_recall = building_metrics.recall;
    metrics.real_positive_missing_building_id_count = building_metrics.missing_id_count;
    for observation in presence_observations {
        let predicted = observation.probability >= threshold;
        match (observation.positive, predicted) {
            (true, true) => metrics.true_positives += 1,
            (true, false) => metrics.false_negatives += 1,
            (false, true) => metrics.false_positives += 1,
            (false, false) => metrics.true_negatives += 1,
        }
        match observation.origin {
            SampleOrigin::RealPositive => {
                real_positive_count += 1;
                real_positive_correct += usize::from(predicted);
            }
            SampleOrigin::SyntheticNegative => {
                metrics.synthetic_negative_count += 1;
                metrics.synthetic_negative_false_positives += usize::from(predicted);
            }
            SampleOrigin::RealNegative => {
                metrics.real_negative_count += 1;
                metrics.real_negative_false_positives += usize::from(predicted);
            }
            SampleOrigin::SyntheticTarget => {}
        }
    }

    metrics.loss = loss_sum / batches.max(1) as f32;
    metrics.precision = metrics.true_positives as f32
        / (metrics.true_positives + metrics.false_positives).max(1) as f32;
    metrics.recall = metrics.true_positives as f32
        / (metrics.true_positives + metrics.false_negatives).max(1) as f32;
    metrics.specificity = metrics.true_negatives as f32
        / (metrics.true_negatives + metrics.false_positives).max(1) as f32;
    metrics.synthetic_negative_specificity = 1.0
        - metrics.synthetic_negative_false_positives as f32
            / metrics.synthetic_negative_count.max(1) as f32;
    metrics.real_negative_specificity = 1.0
        - metrics.real_negative_false_positives as f32 / metrics.real_negative_count.max(1) as f32;
    metrics.real_positive_recall = real_positive_correct as f32 / real_positive_count.max(1) as f32;
    metrics.real_positive_count = real_positive_count;
    metrics.pck_03 = strict_correct as f32 / metrics.evaluated_keypoints.max(1) as f32;
    metrics.pck_05 = standard_correct as f32 / metrics.evaluated_keypoints.max(1) as f32;
    metrics.mean_keypoint_error = keypoint_error / metrics.evaluated_keypoints.max(1) as f32;
    metrics.offscreen_precision = metrics.offscreen_true_positives as f32
        / (metrics.offscreen_true_positives + metrics.offscreen_false_positives).max(1) as f32;
    metrics.offscreen_recall = metrics.offscreen_true_positives as f32
        / (metrics.offscreen_true_positives + metrics.offscreen_false_negatives).max(1) as f32;
    metrics.offscreen_accuracy = (metrics.offscreen_true_positives
        + metrics.offscreen_true_negatives) as f32
        / (metrics.offscreen_true_positives
            + metrics.offscreen_false_positives
            + metrics.offscreen_true_negatives
            + metrics.offscreen_false_negatives)
            .max(1) as f32;
    if compute_fit_metrics {
        println!(
            "final synthetic fit evaluation: {} held-out roofs across {} CPU workers",
            pending_fits.len(),
            rayon::current_num_threads()
        );
        let fit_results = pending_fits
            .par_iter()
            .map(|(predictions, confidences, frame)| {
                evaluate_synthetic_fit(predictions, Some(confidences), frame, fit_shape_prior).ok()
            })
            .collect::<Vec<_>>();
        for evaluation in fit_results.into_iter().flatten() {
            metrics.synthetic_fit.record_fit(evaluation);
        }
        println!(
            "final synthetic fit evaluation complete: fitted={}/{} accepted={}",
            metrics.synthetic_fit.fitted,
            metrics.synthetic_fit.attempted,
            metrics.synthetic_fit.accepted
        );
    }
    metrics.synthetic_fit.finish();
    Ok(metrics)
}

fn decode_metric_points(
    maps: &[f32],
    offscreen_logits: &[f32],
) -> ([Option<[f32; 2]>; KEYPOINT_COUNT], [f32; KEYPOINT_COUNT]) {
    let mut confidences = [0.0; KEYPOINT_COUNT];
    let points = std::array::from_fn(|index| {
        let map = &maps[index * OFFSCREEN_INDEX..(index + 1) * OFFSCREEN_INDEX];
        let (peak_index, peak_logit) = map
            .iter()
            .copied()
            .enumerate()
            .max_by(|left, right| left.1.total_cmp(&right.1))?;
        let normalizer_logit = peak_logit.max(offscreen_logits[index]);
        let normalizer = map
            .iter()
            .map(|logit| (*logit - normalizer_logit).exp())
            .sum::<f32>()
            + (offscreen_logits[index] - normalizer_logit).exp();
        let offscreen_probability =
            (offscreen_logits[index] - normalizer_logit).exp() / normalizer.max(f32::EPSILON);
        if offscreen_probability >= DEFAULT_OFFSCREEN_THRESHOLD {
            confidences[index] = offscreen_probability;
            return None;
        }
        let peak_x = peak_index % HEATMAP_SIZE;
        let peak_y = peak_index / HEATMAP_SIZE;
        let mut weighted = [0.0; 2];
        let mut total = 0.0;
        for y in peak_y.saturating_sub(2)..=(peak_y + 2).min(HEATMAP_SIZE - 1) {
            for x in peak_x.saturating_sub(2)..=(peak_x + 2).min(HEATMAP_SIZE - 1) {
                let weight = (map[y * HEATMAP_SIZE + x] - peak_logit).exp();
                weighted[0] += weight * (x as f32 + 0.5);
                weighted[1] += weight * (y as f32 + 0.5);
                total += weight;
            }
        }
        confidences[index] = total / normalizer.max(f32::EPSILON);
        Some([
            weighted[0] / total / HEATMAP_SIZE as f32,
            weighted[1] / total / HEATMAP_SIZE as f32,
        ])
    });
    (points, confidences)
}

#[derive(Clone, Debug)]
struct CorrespondenceEvaluation {
    hypothesis: usize,
    errors: [f32; KEYPOINT_COUNT],
}

fn best_correspondence(
    predicted: &[Option<[f32; 2]>; KEYPOINT_COUNT],
    target: &[[f32; 2]; KEYPOINT_COUNT],
    in_frame: &[bool; KEYPOINT_COUNT],
    source_width: u32,
    source_height: u32,
) -> CorrespondenceEvaluation {
    let mut best = CorrespondenceEvaluation {
        hypothesis: 0,
        errors: [1.0; KEYPOINT_COUNT],
    };
    let mut best_mean = f32::INFINITY;
    for hypothesis in 0..SYMMETRY_COUNT {
        let mut errors = [1.0; KEYPOINT_COUNT];
        let mut state_aware_cost = 0.0;
        for predicted_index in 0..KEYPOINT_COUNT {
            let ring = predicted_index / 4;
            let target_index = ring * 4 + symmetry_target_slot(hypothesis, predicted_index % 4);
            errors[predicted_index] = if in_frame[target_index] {
                predicted[predicted_index].map_or(1.0, |point| {
                    let dx = (point[0] - target[target_index][0]) * source_width as f32;
                    let dy = (point[1] - target[target_index][1]) * source_height as f32;
                    (dx * dx + dy * dy).sqrt() / (source_width as f32).hypot(source_height as f32)
                })
            } else if predicted[predicted_index].is_none() {
                0.0
            } else {
                1.0
            };
            state_aware_cost += errors[predicted_index];
        }
        let mean = state_aware_cost / KEYPOINT_COUNT as f32;
        if mean < best_mean {
            best_mean = mean;
            best = CorrespondenceEvaluation { hypothesis, errors };
        }
    }
    best
}

fn duplicate_point_pairs(
    predicted: &[Option<[f32; 2]>; KEYPOINT_COUNT],
    target: &[[f32; 2]; KEYPOINT_COUNT],
    in_frame: &[bool; KEYPOINT_COUNT],
    hypothesis: usize,
) -> usize {
    let minimum_separation = 0.5 / HEATMAP_SIZE as f32;
    let unambiguously_distinct = 2.0 / HEATMAP_SIZE as f32;
    let mut duplicates = 0usize;
    for left in 0..KEYPOINT_COUNT {
        for right in left + 1..KEYPOINT_COUNT {
            let target_index = |predicted_index: usize| {
                let ring = predicted_index / 4;
                ring * 4 + symmetry_target_slot(hypothesis, predicted_index % 4)
            };
            let target_left = target_index(left);
            let target_right = target_index(right);
            if !in_frame[target_left] || !in_frame[target_right] {
                continue;
            }
            let target_separation = (target[target_left][0] - target[target_right][0])
                .hypot(target[target_left][1] - target[target_right][1]);
            if target_separation < unambiguously_distinct {
                // Extreme foreshortening can legitimately put same-ring or
                // cross-tier corners into one heatmap cell; that is not model
                // collapse when the source geometry does the same.
                continue;
            }
            let (Some(left), Some(right)) = (predicted[left], predicted[right]) else {
                continue;
            };
            duplicates +=
                usize::from((left[0] - right[0]).hypot(left[1] - right[1]) < minimum_separation);
        }
    }
    duplicates
}

fn scheduled_learning_rate(base: f64, update: usize, total: usize, warmup_fraction: f64) -> f64 {
    let warmup = ((total as f64 * warmup_fraction).round() as usize).max(1);
    if update < warmup {
        return base * (update + 1) as f64 / warmup as f64;
    }
    let progress = (update - warmup) as f64 / total.saturating_sub(warmup).max(1) as f64;
    base * 0.5 * (1.0 + (PI * progress.clamp(0.0, 1.0)).cos())
}

#[derive(Clone, Copy, Debug)]
struct OptimizerSchedule {
    base_learning_rate: f64,
    total_updates: usize,
    warmup_fraction: f64,
    updates_completed: usize,
}

impl OptimizerSchedule {
    fn new(base_learning_rate: f64, total_updates: usize, warmup_fraction: f64) -> Self {
        Self {
            base_learning_rate,
            total_updates,
            warmup_fraction,
            updates_completed: 0,
        }
    }

    fn with_updates_completed(mut self, updates_completed: usize) -> Option<Self> {
        if updates_completed > self.total_updates {
            return None;
        }
        self.updates_completed = updates_completed;
        Some(self)
    }

    fn next_learning_rate(&mut self) -> f64 {
        debug_assert!(self.updates_completed < self.total_updates);
        let learning_rate = scheduled_learning_rate(
            self.base_learning_rate,
            self.updates_completed,
            self.total_updates,
            self.warmup_fraction,
        );
        self.updates_completed += 1;
        learning_rate
    }

    fn updates_completed(&self) -> usize {
        self.updates_completed
    }

    fn total_updates(&self) -> usize {
        self.total_updates
    }
}

#[derive(Clone, Copy, Debug)]
struct StagedBackboneSchedule {
    freeze_epochs: usize,
    schedule: OptimizerSchedule,
}

impl StagedBackboneSchedule {
    fn new(
        base_learning_rate: f64,
        epochs: usize,
        freeze_epochs: usize,
        batches_per_epoch: usize,
        warmup_fraction: f64,
    ) -> Self {
        Self {
            freeze_epochs,
            schedule: OptimizerSchedule::new(
                base_learning_rate,
                epochs.saturating_sub(freeze_epochs) * batches_per_epoch,
                warmup_fraction,
            ),
        }
    }

    fn is_trainable(&self, epoch: usize) -> bool {
        epoch > self.freeze_epochs
    }

    fn with_updates_completed(mut self, updates_completed: usize) -> Option<Self> {
        self.schedule = self.schedule.with_updates_completed(updates_completed)?;
        Some(self)
    }

    fn next_learning_rate(&mut self, epoch: usize) -> Option<f64> {
        if self.is_trainable(epoch) {
            Some(self.schedule.next_learning_rate())
        } else {
            None
        }
    }

    fn updates_completed(&self) -> usize {
        self.schedule.updates_completed()
    }

    fn total_updates(&self) -> usize {
        self.schedule.total_updates()
    }
}

fn step_staged_backbone<M>(
    schedule: &mut StagedBackboneSchedule,
    epoch: usize,
    model: M,
    step: impl FnOnce(f64, M) -> M,
) -> (M, Option<f64>) {
    match schedule.next_learning_rate(epoch) {
        Some(learning_rate) => (step(learning_rate, model), Some(learning_rate)),
        None => (model, None),
    }
}

#[derive(Clone, Debug, Serialize)]
struct ShapePrior {
    mean: [f32; 7],
    standard_deviation: [f32; 7],
}

#[derive(Deserialize)]
struct StoredShapePrior {
    shape_prior: StoredShapePriorMean,
}

#[derive(Deserialize)]
struct StoredShapePriorMean {
    mean: [f32; 7],
}

fn load_checkpoint_shape_prior(checkpoint: &Path) -> Option<[f32; 7]> {
    let path = checkpoint.parent()?.join("model.json");
    let bytes = fs::read(path).ok()?;
    serde_json::from_slice::<StoredShapePrior>(&bytes)
        .ok()
        .map(|manifest| manifest.shape_prior.mean)
}

#[derive(Clone, Debug, Serialize)]
struct ModelManifest {
    schema_version: String,
    checkpoint: String,
    input_size: usize,
    heatmap_size: usize,
    keypoint_count: usize,
    recommended_presence_threshold: f32,
    recommended_offscreen_threshold: f32,
    recommended_keypoint_threshold: f32,
    shape_prior: ShapePrior,
}

fn shape_prior(samples: &[TrainingSample]) -> Result<ShapePrior> {
    let mut values = Vec::<[f32; 7]>::new();
    for sample in samples.iter().filter(|sample| sample.split == Split::Train) {
        let Some(record) = &sample.geometry else {
            continue;
        };
        let roof = record
            .roof
            .as_ref()
            .context("synthetic target has no roof parameter record")?;
        let get = |name: &str| {
            roof.parameters
                .get(name)
                .copied()
                .with_context(|| format!("synthetic roof has no {name}"))
        };
        let eave_width = get("eave_width")?;
        let eave_depth = get("eave_depth")?;
        let shoulder_width = get("shoulder_width")?;
        let shoulder_depth = get("shoulder_depth")?;
        values.push([
            eave_depth / eave_width,
            shoulder_width / eave_width,
            shoulder_depth / eave_depth,
            get("crown_top_width")? / shoulder_width,
            get("crown_top_depth")? / shoulder_depth,
            get("lower_rise")? / eave_width,
            get("upper_rise")? / eave_width,
        ]);
    }
    anyhow::ensure!(
        !values.is_empty(),
        "cannot calculate a shape prior without targets"
    );
    let mut mean = [0.0; 7];
    for value in &values {
        for index in 0..7 {
            mean[index] += value[index];
        }
    }
    for value in &mut mean {
        *value /= values.len() as f32;
    }
    let mut standard_deviation = [0.0; 7];
    for value in &values {
        for index in 0..7 {
            standard_deviation[index] += (value[index] - mean[index]).powi(2);
        }
    }
    for value in &mut standard_deviation {
        *value = (*value / values.len() as f32).sqrt().max(0.01);
    }
    Ok(ShapePrior {
        mean,
        standard_deviation,
    })
}

#[cfg(test)]
mod tests {
    use std::{
        sync::atomic::{AtomicU64, Ordering as AtomicOrdering},
        time::{SystemTime, UNIX_EPOCH},
    };

    use burn::backend::{Autodiff, Flex, flex::FlexDevice};
    use clap::Parser;
    use image::{ColorType, ImageEncoder, codecs::png::PngEncoder};

    use super::*;

    static TEMP_DIRECTORY_ID: AtomicU64 = AtomicU64::new(0);

    struct TempDirectory(PathBuf);

    impl TempDirectory {
        fn new() -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock")
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "roof-refinement-contract-test-{}-{nonce}-{}",
                std::process::id(),
                TEMP_DIRECTORY_ID.fetch_add(1, AtomicOrdering::Relaxed)
            ));
            fs::create_dir(&path).expect("create refinement test directory");
            Self(path)
        }
    }

    impl Drop for TempDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn refinement_test_sample(index: usize, origin: SampleOrigin) -> TrainingSample {
        TrainingSample {
            key: format!("refinement-{origin:?}-{index}"),
            split: Split::Train,
            image: ImageSource::Encoded(Arc::<[u8]>::from([])),
            presence: i32::from(matches!(
                origin,
                SampleOrigin::SyntheticTarget | SampleOrigin::RealPositive
            )),
            geometry: None,
            origin,
            physical_building_id: (origin == SampleOrigin::RealPositive)
                .then(|| Arc::<str>::from(format!("building-{index}"))),
        }
    }

    fn refinement_test_samples() -> (
        Vec<TrainingSample>,
        Vec<TrainingSample>,
        Vec<TrainingSample>,
    ) {
        let mut train = Vec::new();
        for (count, origin) in [
            (6, SampleOrigin::SyntheticTarget),
            (6, SampleOrigin::SyntheticNegative),
            (2, SampleOrigin::RealPositive),
            (4, SampleOrigin::RealNegative),
        ] {
            train.extend((0..count).map(|index| refinement_test_sample(index, origin)));
        }
        let validation = vec![
            refinement_test_sample(100, SampleOrigin::SyntheticTarget),
            refinement_test_sample(101, SampleOrigin::RealNegative),
        ];
        let test = vec![
            refinement_test_sample(200, SampleOrigin::SyntheticTarget),
            refinement_test_sample(201, SampleOrigin::SyntheticNegative),
            refinement_test_sample(202, SampleOrigin::RealNegative),
        ];
        (train, validation, test)
    }

    fn refinement_test_args(source: &TempDirectory, output: &TempDirectory) -> Args {
        let checkpoint = source.0.join("candidate-e06.mpk");
        fs::write(&checkpoint, b"test checkpoint").expect("write checkpoint");
        let mut args = Args::parse_from([
            "roof-train",
            "--synthetic",
            "datasets/test-synthetic-v2",
            "--negatives",
            "datasets/test-negatives",
            "--real-positives",
            "datasets/test-real-positives",
            "--artifacts",
            output.0.to_str().unwrap(),
            "--epochs",
            "8",
            "--batch-size",
            "8",
            "--evaluation-batch-size",
            "8",
            "--geometry-refine-from",
            checkpoint.to_str().unwrap(),
            "--geometry-refine-source-epoch",
            "6",
        ]);
        crate::validate_and_normalize_args(&mut args).expect("normalize refinement args");
        args
    }

    fn refinement_test_training_contract(
        args: &Args,
        train: &[TrainingSample],
        validation: &[TrainingSample],
        test: &[TrainingSample],
    ) -> RefinementTrainingContract {
        RefinementTrainingContract {
            schema_version: "roof-training/v15".to_owned(),
            synthetic: args.synthetic.display().to_string(),
            negatives: args.negatives.display().to_string(),
            real_positives: args.real_positives.display().to_string(),
            real_positive_repeat: args.real_positive_repeat,
            real_positive_sampling_strategy: REAL_POSITIVE_SAMPLING_STRATEGY.to_owned(),
            epochs: args.epochs,
            batch_size: args.batch_size,
            evaluation_batch_size: args.evaluation_batch_size(),
            head_learning_rate: args.head_learning_rate,
            backbone_learning_rate: args.backbone_learning_rate,
            backbone_freeze_epochs: args.backbone_freeze_epochs,
            freeze_backbone_batch_norm: args.freeze_backbone_batch_norm,
            detach_geometry_backbone: args.detach_geometry_backbone,
            presence_freeze_policy: PRESENCE_FREEZE_POLICY.to_owned(),
            presence_freeze_after_safe_epochs: args.presence_freeze_after_safe_epochs,
            weight_decay: args.weight_decay,
            warmup_fraction: args.warmup_fraction,
            geometry_loss_weight: GEOMETRY_LOSS_WEIGHT,
            presence_source_mass: PRESENCE_SOURCE_MASS,
            real_presence_pairwise_loss_weight: REAL_PRESENCE_PAIRWISE_LOSS_WEIGHT,
            real_presence_pairwise_margin: REAL_PRESENCE_PAIRWISE_MARGIN,
            seed: args.seed,
            limit_per_split: args.limit_per_split,
            disable_augmentation: args.disable_augmentation,
            overfit: args.overfit,
            input_size: SPATIAL_INPUT_SIZE,
            heatmap_size: HEATMAP_SIZE,
            keypoint_count: KEYPOINT_COUNT,
            pretrained_backbone: "torchvision MobileNetV2 ImageNet1K V2".to_owned(),
            train_samples: train.len(),
            train_real_positive_buildings: crate::real_positive_building_count(train).unwrap(),
            validation_samples: validation.len(),
            test_samples: test.len(),
        }
    }

    fn refinement_test_epoch_contract(epoch: usize) -> RefinementEpochContract {
        let batches_per_epoch = 4;
        let locked = epoch == 6;
        RefinementEpochContract {
            epoch,
            presence_freeze_policy: PRESENCE_FREEZE_POLICY.to_owned(),
            presence_freeze_after_safe_epochs: 2,
            presence_safe_streak: usize::from(epoch >= 5) + usize::from(epoch >= 6),
            presence_frozen_after_epoch: locked.then_some(6),
            presence_frozen_from_epoch: locked.then_some(7),
            presence_trainable: true,
            optimizer_stage: OptimizerStage::Joint,
            backbone_trainable: true,
            head_updates_completed: epoch * batches_per_epoch,
            head_total_updates: 8 * batches_per_epoch,
            joint_head_updates_completed: epoch * batches_per_epoch,
            geometry_only_head_updates_completed: 0,
            backbone_updates_completed: epoch * batches_per_epoch,
            backbone_total_updates: 8 * batches_per_epoch,
        }
    }

    fn write_refinement_test_provenance(
        source: &TempDirectory,
        args: &Args,
        train: &[TrainingSample],
        validation: &[TrainingSample],
        test: &[TrainingSample],
        metrics_name: &str,
    ) {
        fs::write(
            source.0.join("training-config.json"),
            serde_json::to_vec_pretty(&refinement_test_training_contract(
                args, train, validation, test,
            ))
            .unwrap(),
        )
        .expect("write source training config");
        let metrics = (1..=6)
            .map(|epoch| serde_json::to_string(&refinement_test_epoch_contract(epoch)).unwrap())
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(source.0.join(metrics_name), format!("{metrics}\n"))
            .expect("write source metrics");
    }

    fn prepared_promotion_files(directory: &TempDirectory) -> (PromotionPaths, PromotionPaths) {
        let temporary = PromotionPaths::temporary(&directory.0);
        let final_paths = PromotionPaths::final_paths(&directory.0);
        fs::write(&temporary.checkpoint, b"checkpoint").unwrap();
        fs::write(&temporary.manifest, br#"{"checkpoint":"model.mpk"}"#).unwrap();
        fs::write(&temporary.metrics, br#"{"promotion":{"promoted":true}}"#).unwrap();
        (temporary, final_paths)
    }

    fn encoded_test_image(width: u32, height: u32) -> Arc<[u8]> {
        let image = image::RgbImage::from_pixel(width, height, image::Rgb([31, 71, 127]));
        let mut bytes = Vec::new();
        PngEncoder::new(&mut bytes)
            .write_image(image.as_raw(), width, height, ColorType::Rgb8.into())
            .expect("encode test image");
        bytes.into()
    }

    #[test]
    fn schedule_warms_up_and_decays() {
        let rates = (0..100)
            .map(|step| scheduled_learning_rate(1.0, step, 100, 0.05))
            .collect::<Vec<_>>();
        assert!(rates[0] < rates[4]);
        assert!((rates[4] - 1.0).abs() < 1.0e-6);
        assert!(rates[99] < 0.001);
    }

    #[test]
    fn staged_backbone_schedule_starts_its_own_warmup_after_unfreeze() {
        let batches_per_epoch = 10;
        let mut head = OptimizerSchedule::new(3.0e-4, 4 * batches_per_epoch, 0.25);
        let mut backbone = StagedBackboneSchedule::new(3.0e-5, 4, 2, batches_per_epoch, 0.25);

        for epoch in 1..=2 {
            for _ in 0..batches_per_epoch {
                let _ = head.next_learning_rate();
                assert_eq!(backbone.next_learning_rate(epoch), None);
            }
        }
        assert_eq!(head.updates_completed(), 20);
        assert_eq!(backbone.updates_completed(), 0);
        assert_eq!(backbone.total_updates(), 20);

        let first_backbone_rate = backbone.next_learning_rate(3).unwrap();
        assert_eq!(backbone.updates_completed(), 1);
        assert!(
            (first_backbone_rate - scheduled_learning_rate(3.0e-5, 0, 20, 0.25)).abs()
                < f64::EPSILON
        );
        assert!(first_backbone_rate < 3.0e-5);
    }

    #[test]
    fn refinement_schedule_starts_at_the_logical_update_offset() {
        let total_updates = 40 * 402;
        let completed_updates = 6 * 402;
        let mut schedule = OptimizerSchedule::new(3.0e-4, total_updates, 0.05)
            .with_updates_completed(completed_updates)
            .unwrap();

        assert_eq!(schedule.updates_completed(), completed_updates);
        assert_eq!(
            schedule.next_learning_rate(),
            scheduled_learning_rate(3.0e-4, completed_updates, total_updates, 0.05)
        );
        assert!(
            OptimizerSchedule::new(1.0, 10, 0.0)
                .with_updates_completed(11)
                .is_none()
        );
    }

    #[test]
    fn promotion_commits_manifest_then_checkpoint_then_metrics() {
        let directory = TempDirectory::new();
        let (temporary, final_paths) = prepared_promotion_files(&directory);
        let mut destinations = Vec::new();

        commit_prepared_promotion_with(&temporary, &final_paths, |source, destination| {
            destinations.push(
                destination
                    .file_name()
                    .unwrap()
                    .to_string_lossy()
                    .into_owned(),
            );
            fs::rename(source, destination)
        })
        .unwrap();

        assert_eq!(
            destinations,
            ["model.json", "model.mpk", "final-metrics.json"]
        );
        assert_eq!(fs::read(&final_paths.checkpoint).unwrap(), b"checkpoint");
        assert_eq!(
            fs::read(&final_paths.manifest).unwrap(),
            br#"{"checkpoint":"model.mpk"}"#
        );
        assert_eq!(
            fs::read(&final_paths.metrics).unwrap(),
            br#"{"promotion":{"promoted":true}}"#
        );
        assert!(temporary.iter().all(|path| !path.exists()));
    }

    #[test]
    fn promotion_rolls_back_checkpoint_and_manifest_when_metrics_commit_fails() {
        let directory = TempDirectory::new();
        let (temporary, final_paths) = prepared_promotion_files(&directory);
        let mut rename_count = 0;

        let error =
            commit_prepared_promotion_with(&temporary, &final_paths, |source, destination| {
                rename_count += 1;
                if rename_count == 3 {
                    return Err(std::io::Error::other("injected metrics rename failure"));
                }
                fs::rename(source, destination)
            })
            .expect_err("the final report commit is required")
            .to_string();

        assert!(error.contains("commit promoted final metrics"), "{error}");
        assert!(final_paths.iter().all(|path| !path.exists()));
        assert!(temporary.iter().all(|path| !path.exists()));
    }

    #[test]
    fn promotion_removes_manifest_when_checkpoint_commit_fails() {
        let directory = TempDirectory::new();
        let (temporary, final_paths) = prepared_promotion_files(&directory);
        let mut rename_count = 0;

        let error =
            commit_prepared_promotion_with(&temporary, &final_paths, |source, destination| {
                rename_count += 1;
                if rename_count == 2 {
                    return Err(std::io::Error::other("injected checkpoint rename failure"));
                }
                fs::rename(source, destination)
            })
            .expect_err("a checkpoint commit failure must abort promotion")
            .to_string();

        assert!(error.contains("commit promotion checkpoint"), "{error}");
        assert!(!final_paths.manifest.exists());
        assert!(!final_paths.checkpoint.exists());
        assert!(!final_paths.metrics.exists());
        assert!(temporary.iter().all(|path| !path.exists()));
    }

    #[test]
    fn promotion_refuses_existing_outputs_and_removes_prepared_temporaries() {
        let directory = TempDirectory::new();
        let (temporary, final_paths) = prepared_promotion_files(&directory);
        fs::write(&final_paths.metrics, b"existing report").unwrap();

        let error = commit_prepared_promotion(&temporary, &final_paths)
            .expect_err("promotion must never overwrite a prior report")
            .to_string();

        assert!(error.contains("refusing to overwrite"), "{error}");
        assert_eq!(fs::read(&final_paths.metrics).unwrap(), b"existing report");
        assert!(!final_paths.checkpoint.exists());
        assert!(!final_paths.manifest.exists());
        assert!(temporary.iter().all(|path| !path.exists()));
    }

    #[test]
    fn refinement_authenticates_fallback_metrics_and_recorded_update_offsets() {
        let source = TempDirectory::new();
        let output = TempDirectory::new();
        let args = refinement_test_args(&source, &output);
        let (train, validation, test) = refinement_test_samples();
        write_refinement_test_provenance(
            &source,
            &args,
            &train,
            &validation,
            &test,
            "metrics-e01-e06.jsonl",
        );

        let contract = authenticate_refinement_source(
            &args,
            args.geometry_refine_from.as_deref().unwrap(),
            6,
            &train,
            &validation,
            &test,
        )
        .expect("the complete source contract should authenticate");

        assert!(contract.source_metrics.ends_with("metrics-e01-e06.jsonl"));
        assert_eq!(contract.source_metrics_line, 6);
        assert_eq!(contract.derived_update_contract.batches_per_epoch, 4);
        assert_eq!(contract.epoch_contract.head_updates_completed, 24);
        assert_eq!(contract.epoch_contract.head_total_updates, 32);
        assert_eq!(contract.epoch_contract.joint_head_updates_completed, 24);
        assert_eq!(
            contract.epoch_contract.geometry_only_head_updates_completed,
            0
        );
        assert_eq!(contract.epoch_contract.backbone_updates_completed, 24);
        assert_eq!(contract.epoch_contract.presence_frozen_after_epoch, Some(6));
        assert!(contract.source_checkpoint_sha256.starts_with("sha256:"));
        assert!(
            contract
                .source_training_config_sha256
                .starts_with("sha256:")
        );
        assert!(contract.source_metrics_sha256.starts_with("sha256:"));
    }

    #[test]
    fn refinement_rejects_capped_or_otherwise_mismatched_continuations() {
        let source = TempDirectory::new();
        let output = TempDirectory::new();
        let mut args = refinement_test_args(&source, &output);
        let (train, validation, test) = refinement_test_samples();
        write_refinement_test_provenance(
            &source,
            &args,
            &train,
            &validation,
            &test,
            "metrics.jsonl",
        );
        args.limit_per_split = Some(4);

        let error = authenticate_refinement_source(
            &args,
            args.geometry_refine_from.as_deref().unwrap(),
            6,
            &train,
            &validation,
            &test,
        )
        .expect_err("a capped continuation must not inherit full-run optimizer state")
        .to_string();
        assert!(error.contains("limit_per_split"), "{error}");
    }

    #[test]
    fn refinement_fallback_metrics_must_be_unambiguous() {
        let source = TempDirectory::new();
        let output = TempDirectory::new();
        let args = refinement_test_args(&source, &output);
        let (train, validation, test) = refinement_test_samples();
        write_refinement_test_provenance(
            &source,
            &args,
            &train,
            &validation,
            &test,
            "metrics-e01-e06.jsonl",
        );
        fs::write(source.0.join("metrics-copy.jsonl"), b"{}\n")
            .expect("write ambiguous metrics fallback");

        let error = authenticate_refinement_source(
            &args,
            args.geometry_refine_from.as_deref().unwrap(),
            6,
            &train,
            &validation,
            &test,
        )
        .expect_err("multiple fallback metrics files must fail closed")
        .to_string();
        assert!(
            error.contains("multiple metrics*.jsonl fallbacks"),
            "{error}"
        );
    }

    #[test]
    fn refinement_requires_the_requested_epoch_to_be_the_last_complete_record() {
        let source = TempDirectory::new();
        let output = TempDirectory::new();
        let args = refinement_test_args(&source, &output);
        let (train, validation, test) = refinement_test_samples();
        write_refinement_test_provenance(
            &source,
            &args,
            &train,
            &validation,
            &test,
            "metrics.jsonl",
        );
        append_json_line(
            &source.0.join("metrics.jsonl"),
            &refinement_test_epoch_contract(7),
        )
        .unwrap();

        let error = authenticate_refinement_source(
            &args,
            args.geometry_refine_from.as_deref().unwrap(),
            6,
            &train,
            &validation,
            &test,
        )
        .expect_err("a checkpoint without final-epoch provenance must fail closed")
        .to_string();
        assert!(
            error.contains("source epoch vs last metrics epoch"),
            "{error}"
        );
    }

    #[test]
    fn staged_backbone_step_leaves_parameters_untouched_while_frozen() {
        let mut schedule = StagedBackboneSchedule::new(0.1, 3, 1, 1, 0.0);
        let parameters = [1.0_f64, -1.0];
        let mut step_calls = 0;

        let (parameters, frozen_rate) =
            step_staged_backbone(&mut schedule, 1, parameters, |learning_rate, parameters| {
                step_calls += 1;
                parameters.map(|parameter| parameter - learning_rate)
            });
        assert_eq!(frozen_rate, None);
        assert_eq!(parameters, [1.0, -1.0]);
        assert_eq!(step_calls, 0);
        assert_eq!(schedule.updates_completed(), 0);

        let (parameters, trainable_rate) =
            step_staged_backbone(&mut schedule, 2, parameters, |learning_rate, parameters| {
                step_calls += 1;
                parameters.map(|parameter| parameter - learning_rate)
            });
        assert!(trainable_rate.is_some());
        assert_ne!(parameters, [1.0, -1.0]);
        assert_eq!(step_calls, 1);
        assert_eq!(schedule.updates_completed(), 1);
    }

    #[test]
    fn presence_lock_requires_consecutive_safe_epochs_and_is_sticky() {
        let mut state = PresenceLockState::new(2);
        assert!(state.presence_trainable());
        assert!(!state.observe_validation(1, true, true));
        assert_eq!(state.safe_streak, 1);

        assert!(!state.observe_validation(2, true, false));
        assert_eq!(state.safe_streak, 0, "either unsafe gate resets the streak");
        assert!(!state.observe_validation(3, true, true));
        assert!(state.observe_validation(4, true, true));
        assert_eq!(state.frozen_after_epoch, Some(4));
        assert!(!state.presence_trainable());

        assert!(!state.observe_validation(5, false, false));
        assert_eq!(state.safe_streak, 2, "the entered lock must never reopen");
        assert_eq!(state.frozen_after_epoch, Some(4));
    }

    #[test]
    fn refinement_restores_a_sticky_presence_lock() {
        let mut state = PresenceLockState::restore_locked(2, 2, 6);
        assert!(!state.presence_trainable());
        assert_eq!(state.safe_streak, 2);
        assert_eq!(state.frozen_after_epoch, Some(6));
        assert!(!state.observe_validation(7, false, false));
        assert_eq!(state.frozen_after_epoch, Some(6));

        let disabled_policy = PresenceLockState::restore_locked(0, 0, 6);
        assert!(
            !disabled_policy.presence_trainable(),
            "an explicit refinement contract stays locked even when automatic locking is disabled"
        );
    }

    #[test]
    fn presence_lock_and_checkpoint_rank_reject_two_of_three_real_buildings() {
        let metrics = EvaluationMetrics {
            real_positive_recall: 5.0 / 6.0,
            real_positive_count: 6,
            real_positive_building_count: 3,
            real_positive_buildings_detected: 2,
            real_positive_building_recall: 2.0 / 3.0,
            real_negative_specificity: 0.90,
            real_negative_count: 20,
            presence_diagnostics: PresenceDiagnostics {
                synthetic_roc_auc: Some(0.97),
                synthetic_average_precision: Some(0.97),
                ..PresenceDiagnostics::default()
            },
            ..EvaluationMetrics::default()
        };
        assert!(!metrics.real_presence_gate_passes());
        assert!(!metrics.checkpoint_rank().presence.real_gate_passes);
        assert!(
            (metrics.real_presence_gate_ratio() - (2.0 / 3.0) / MIN_REAL_POSITIVE_BUILDING_RECALL)
                .abs()
                < 1.0e-6
        );

        let mut lock = PresenceLockState::new(2);
        for epoch in 1..=2 {
            assert!(!lock.observe_validation(
                epoch,
                metrics.real_presence_gate_passes(),
                metrics.synthetic_presence_gate_passes(),
            ));
        }
        assert!(lock.presence_trainable());
        assert_eq!(lock.safe_streak, 0);
    }

    #[test]
    fn zero_presence_lock_policy_disables_the_transition() {
        let mut state = PresenceLockState::new(0);
        for epoch in 1..=20 {
            assert!(!state.observe_validation(epoch, true, true));
        }
        assert!(state.presence_trainable());
        assert_eq!(state.safe_streak, 0);
        assert_eq!(state.frozen_after_epoch, None);
    }

    #[test]
    fn presence_lock_resets_patience_but_not_optimizer_schedule() {
        let mut head_schedule = OptimizerSchedule::new(1.0, 10, 0.2);
        let _ = head_schedule.next_learning_rate();
        let _ = head_schedule.next_learning_rate();
        let completed_before_lock = head_schedule.updates_completed();

        let patience = next_patience_count(5, false, true, true);
        assert_eq!(patience, 0, "transition restarts early-stopping patience");
        let next_rate = head_schedule.next_learning_rate();
        assert_eq!(head_schedule.updates_completed(), completed_before_lock + 1);
        assert_eq!(next_rate, scheduled_learning_rate(1.0, 2, 10, 0.2));

        let patience = next_patience_count(patience, false, false, true);
        assert_eq!(patience, 1, "geometry-only epochs consume patience");
        assert_eq!(next_patience_count(patience, true, false, true), 0);
    }

    #[test]
    fn training_uses_cached_working_raster_but_evaluation_keeps_source_transform() {
        let image = encoded_test_image(1_000, 333);
        let sample = TrainingSample {
            key: "source-transform".to_owned(),
            split: Split::Train,
            image: ImageSource::Encoded(image),
            presence: 0,
            geometry: None,
            origin: SampleOrigin::RealNegative,
            physical_building_id: None,
        };
        let cache = WorkingImageCache::default();

        let evaluation = prepare_sample(&sample, 0, true, Some(&cache)).unwrap();
        assert_eq!(evaluation.transform.source_width, 1_000);
        assert_eq!(evaluation.transform.source_height, 333);
        assert_eq!(
            cache.len(),
            0,
            "evaluation must not populate training cache"
        );

        let training = prepare_sample(&sample, 1, true, Some(&cache)).unwrap();
        assert_eq!(training.transform.source_width, 256);
        assert_eq!(training.transform.source_height, 85);
        assert_eq!(cache.len(), 1);

        let mut repeated_draw = sample.clone();
        repeated_draw.key.push_str("#epoch-1-draw-99");
        let _ = prepare_sample(&repeated_draw, 1, true, Some(&cache)).unwrap();
        assert_eq!(
            cache.len(),
            1,
            "balanced draws sharing one encoded source reuse its working raster"
        );
    }

    #[test]
    fn evaluation_pck_is_measured_with_original_source_dimensions() {
        let sample = TrainingSample {
            key: "pck-transform".to_owned(),
            split: Split::Validation,
            image: ImageSource::Encoded(encoded_test_image(1_000, 333)),
            presence: 0,
            geometry: None,
            origin: SampleOrigin::RealNegative,
            physical_building_id: None,
        };
        let prepared = prepare_sample(&sample, 0, false, None).unwrap();
        let transform = prepared.transform;
        assert_eq!(
            (transform.source_width, transform.source_height),
            (1_000, 333)
        );

        let source_target = [0.5, 0.5];
        let source_prediction = [0.525, 0.5];
        let model_target = transform.map_point(source_target);
        let model_prediction = transform.map_point(source_prediction);
        let mut targets = [[0.0; 2]; KEYPOINT_COUNT];
        targets[0] = transform.unmap_point_unclamped(model_target);
        let mut predictions = [None; KEYPOINT_COUNT];
        predictions[0] = Some(transform.unmap_point_unclamped(model_prediction));
        let mut in_frame = [false; KEYPOINT_COUNT];
        in_frame[0] = true;

        let correspondence = best_correspondence(
            &predictions,
            &targets,
            &in_frame,
            transform.source_width,
            transform.source_height,
        );
        let expected = 25.0_f32 / 1_000.0_f32.hypot(333.0);
        assert!((correspondence.errors[0] - expected).abs() < 1.0e-6);
        assert!(correspondence.errors[0] < PCK_STRICT_THRESHOLD);
    }

    #[test]
    fn metric_matching_accepts_a_shared_reflection() {
        let target = std::array::from_fn(|index| {
            let ring = index / 4;
            let slot = index % 4;
            [slot as f32 * 0.2 + 0.1, ring as f32 * 0.2 + 0.1]
        });
        let in_frame = [true; KEYPOINT_COUNT];
        let predicted = std::array::from_fn(|index| {
            let ring = index / 4;
            let slot = symmetry_target_slot(6, index % 4);
            Some(target[ring * 4 + slot])
        });
        let correspondence = best_correspondence(&predicted, &target, &in_frame, 1, 1);
        assert!(correspondence.errors.iter().all(|error| *error < 1.0e-6));
    }

    fn distinct_real_positive_building_observations(
        observations: &[PresenceObservation],
    ) -> Vec<(Option<Arc<str>>, f32)> {
        observations
            .iter()
            .filter(|observation| observation.origin == SampleOrigin::RealPositive)
            .enumerate()
            .map(|(index, observation)| {
                (
                    Some(Arc::<str>::from(format!("building-{index}"))),
                    observation.probability,
                )
            })
            .collect()
    }

    #[test]
    fn presence_threshold_prefers_required_recall_and_best_specificity() {
        let observations = [
            PresenceObservation {
                probability: 0.9,
                positive: true,
                origin: SampleOrigin::SyntheticTarget,
            },
            PresenceObservation {
                probability: 0.8,
                positive: true,
                origin: SampleOrigin::RealPositive,
            },
            PresenceObservation {
                probability: 0.3,
                positive: false,
                origin: SampleOrigin::SyntheticNegative,
            },
            PresenceObservation {
                probability: 0.2,
                positive: false,
                origin: SampleOrigin::RealNegative,
            },
        ];
        let buildings = distinct_real_positive_building_observations(&observations);
        assert!((calibrate_presence_threshold(&observations, &buildings) - 0.8).abs() < 1.0e-6);
    }

    #[test]
    fn presence_threshold_ignores_synthetic_probability_offsets() {
        let observations = [
            PresenceObservation {
                probability: 0.01,
                positive: true,
                origin: SampleOrigin::SyntheticTarget,
            },
            PresenceObservation {
                probability: 0.99,
                positive: false,
                origin: SampleOrigin::SyntheticNegative,
            },
            PresenceObservation {
                probability: 0.9,
                positive: true,
                origin: SampleOrigin::RealPositive,
            },
            PresenceObservation {
                probability: 0.8,
                positive: true,
                origin: SampleOrigin::RealPositive,
            },
            PresenceObservation {
                probability: 0.7,
                positive: true,
                origin: SampleOrigin::RealPositive,
            },
            PresenceObservation {
                probability: 0.6,
                positive: true,
                origin: SampleOrigin::RealPositive,
            },
            PresenceObservation {
                probability: 0.5,
                positive: true,
                origin: SampleOrigin::RealPositive,
            },
            PresenceObservation {
                probability: 0.55,
                positive: false,
                origin: SampleOrigin::RealNegative,
            },
            PresenceObservation {
                probability: 0.4,
                positive: false,
                origin: SampleOrigin::RealNegative,
            },
        ];

        let buildings = distinct_real_positive_building_observations(&observations);
        assert!((calibrate_presence_threshold(&observations, &buildings) - 0.6).abs() < 1.0e-6);
    }

    #[test]
    fn presence_threshold_calibration_keeps_five_of_six_real_views() {
        let mut observations = [0.9, 0.8, 0.7, 0.6, 0.5, 0.1]
            .map(|probability| PresenceObservation {
                probability,
                positive: true,
                origin: SampleOrigin::RealPositive,
            })
            .to_vec();
        observations.extend([0.55, 0.2].map(|probability| PresenceObservation {
            probability,
            positive: false,
            origin: SampleOrigin::RealNegative,
        }));

        let buildings = distinct_real_positive_building_observations(&observations);
        let threshold = calibrate_presence_threshold(&observations, &buildings);
        let rates = PresenceRates::from_observations(&observations, threshold);
        assert!((threshold - 0.5).abs() < 1.0e-6);
        assert!((rates.real_positive_recall - 5.0 / 6.0).abs() < 1.0e-6);
        assert!(meets_minimum(
            rates.real_positive_recall,
            MIN_REAL_POSITIVE_RECALL
        ));
    }

    #[test]
    fn presence_threshold_prefers_building_coverage_before_a_higher_equal_specificity_cutoff() {
        let positive_probabilities = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4];
        let mut observations = positive_probabilities
            .map(|probability| PresenceObservation {
                probability,
                positive: true,
                origin: SampleOrigin::RealPositive,
            })
            .to_vec();
        observations.push(PresenceObservation {
            probability: 0.1,
            positive: false,
            origin: SampleOrigin::RealNegative,
        });
        let buildings = ["a", "a", "a", "a", "b", "c"]
            .into_iter()
            .zip(positive_probabilities)
            .map(|(building, probability)| (Some(Arc::<str>::from(building)), probability))
            .collect::<Vec<_>>();

        let threshold = calibrate_presence_threshold(&observations, &buildings);
        assert!((threshold - 0.4).abs() < 1.0e-6);
        let building_metrics = real_positive_building_metrics(&buildings, threshold);
        assert_eq!(building_metrics.detected, 3);
        assert_eq!(building_metrics.count, 3);
        assert_eq!(building_metrics.recall, 1.0);
    }

    #[test]
    fn real_positive_building_recall_uses_the_best_view_and_fails_closed() {
        let observations = vec![
            (Some(Arc::<str>::from("a")), 0.1),
            (Some(Arc::<str>::from("a")), 0.9),
            (Some(Arc::<str>::from("b")), 0.8),
            (Some(Arc::<str>::from("c")), 0.7),
        ];
        assert_eq!(
            real_positive_building_metrics(&observations, 0.5),
            RealPositiveBuildingMetrics {
                count: 3,
                detected: 3,
                recall: 1.0,
                missing_id_count: 0,
            }
        );

        let mut missing = observations;
        missing.push((None, 0.99));
        let metrics = real_positive_building_metrics(&missing, 0.5);
        assert_eq!(metrics.missing_id_count, 1);
        assert_eq!(metrics.recall, 0.0);
    }

    #[test]
    fn presence_threshold_falls_back_to_aggregate_without_real_pairs() {
        let observations = [
            PresenceObservation {
                probability: 0.9,
                positive: true,
                origin: SampleOrigin::SyntheticTarget,
            },
            PresenceObservation {
                probability: 0.8,
                positive: true,
                origin: SampleOrigin::SyntheticTarget,
            },
            PresenceObservation {
                probability: 0.4,
                positive: false,
                origin: SampleOrigin::SyntheticNegative,
            },
            PresenceObservation {
                probability: 0.3,
                positive: false,
                origin: SampleOrigin::SyntheticNegative,
            },
        ];

        assert!((calibrate_presence_threshold(&observations, &[]) - 0.8).abs() < 1.0e-6);
    }

    #[test]
    fn presence_threshold_preserves_sub_micro_probability_candidates() {
        let mut observations = Vec::new();
        observations.extend((0..12).map(|_| PresenceObservation {
            probability: 0.9,
            positive: true,
            origin: SampleOrigin::SyntheticTarget,
        }));
        observations.extend(
            [4.0e-8, 3.8e-7, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6].map(|probability| PresenceObservation {
                probability,
                positive: true,
                origin: SampleOrigin::RealPositive,
            }),
        );
        observations.push(PresenceObservation {
            probability: 1.0e-7,
            positive: false,
            origin: SampleOrigin::RealNegative,
        });
        observations.push(PresenceObservation {
            probability: 1.0e-8,
            positive: false,
            origin: SampleOrigin::SyntheticNegative,
        });

        let buildings = distinct_real_positive_building_observations(&observations);
        let threshold = calibrate_presence_threshold(&observations, &buildings);
        assert!(threshold > 0.0);
        assert!((threshold - 3.8e-7).abs() < 1.0e-10);
    }

    #[test]
    fn presence_diagnostics_report_source_quantiles_and_ranking_quality() {
        let observations = [
            PresenceObservation {
                probability: 0.9,
                positive: true,
                origin: SampleOrigin::SyntheticTarget,
            },
            PresenceObservation {
                probability: 0.8,
                positive: true,
                origin: SampleOrigin::RealPositive,
            },
            PresenceObservation {
                probability: 0.2,
                positive: false,
                origin: SampleOrigin::SyntheticNegative,
            },
            PresenceObservation {
                probability: 0.1,
                positive: false,
                origin: SampleOrigin::RealNegative,
            },
        ];
        let diagnostics = PresenceDiagnostics::from_observations(&observations);
        assert_eq!(diagnostics.synthetic_targets.count, 1);
        assert_eq!(diagnostics.real_negatives.count, 1);
        assert!((diagnostics.synthetic_targets.median - 0.9).abs() < 1.0e-6);
        assert_eq!(diagnostics.roc_auc, Some(1.0));
        assert_eq!(diagnostics.average_precision, Some(1.0));
        assert_eq!(diagnostics.real_roc_auc, Some(1.0));
        assert_eq!(diagnostics.real_average_precision, Some(1.0));
        assert_eq!(
            diagnostics.real_positive_vs_synthetic_negative_roc_auc,
            Some(1.0)
        );
        assert_eq!(
            diagnostics.synthetic_target_vs_real_negative_roc_auc,
            Some(1.0)
        );
    }

    #[test]
    fn cross_domain_metrics_expose_a_hidden_source_offset() {
        let observations = [
            PresenceObservation {
                probability: 0.9,
                positive: true,
                origin: SampleOrigin::SyntheticTarget,
            },
            PresenceObservation {
                probability: 0.8,
                positive: false,
                origin: SampleOrigin::SyntheticNegative,
            },
            PresenceObservation {
                probability: 0.2,
                positive: true,
                origin: SampleOrigin::RealPositive,
            },
            PresenceObservation {
                probability: 0.1,
                positive: false,
                origin: SampleOrigin::RealNegative,
            },
        ];
        let diagnostics = PresenceDiagnostics::from_observations(&observations);

        assert_eq!(diagnostics.synthetic_roc_auc, Some(1.0));
        assert_eq!(diagnostics.real_roc_auc, Some(1.0));
        assert_eq!(
            diagnostics.real_positive_vs_synthetic_negative_roc_auc,
            Some(0.0)
        );
        assert_eq!(
            diagnostics.synthetic_target_vs_real_negative_roc_auc,
            Some(1.0)
        );
    }

    #[test]
    fn common_threshold_diagnostics_report_feasibility_and_best_gate_ratio() {
        let feasible = [
            PresenceObservation {
                probability: 0.9,
                positive: true,
                origin: SampleOrigin::SyntheticTarget,
            },
            PresenceObservation {
                probability: 0.8,
                positive: true,
                origin: SampleOrigin::RealPositive,
            },
            PresenceObservation {
                probability: 0.2,
                positive: false,
                origin: SampleOrigin::SyntheticNegative,
            },
            PresenceObservation {
                probability: 0.1,
                positive: false,
                origin: SampleOrigin::RealNegative,
            },
        ];
        let feasible = PresenceGateDiagnostics::from_observations(&feasible);
        assert!(feasible.common_threshold_feasible);
        assert_eq!(feasible.best_gate_ratio, 1.0);
        assert!((feasible.best_threshold - 0.8).abs() < f32::EPSILON);

        let incompatible = [
            PresenceObservation {
                probability: 0.9,
                positive: true,
                origin: SampleOrigin::SyntheticTarget,
            },
            PresenceObservation {
                probability: 0.8,
                positive: false,
                origin: SampleOrigin::SyntheticNegative,
            },
            PresenceObservation {
                probability: 0.1,
                positive: true,
                origin: SampleOrigin::RealPositive,
            },
            PresenceObservation {
                probability: 0.05,
                positive: false,
                origin: SampleOrigin::RealNegative,
            },
        ];
        let incompatible = PresenceGateDiagnostics::from_observations(&incompatible);
        assert!(!incompatible.common_threshold_feasible);
        assert!((incompatible.best_gate_ratio - 0.5 / 0.9).abs() < 1.0e-6);
        assert!((incompatible.best_threshold - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn presence_checkpoint_rank_is_lexicographic() {
        let baseline = PresenceCheckpointRank {
            real_gate_passes: false,
            synthetic_gate_passes: true,
            real_gate_ratio: 0.8,
            synthetic_gate_ratio: 1.0,
            real_recall: 1.0,
            real_specificity: 0.68,
            synthetic_ranking_quality: 1.0,
            ..PresenceCheckpointRank::default()
        };
        assert!(
            PresenceCheckpointRank {
                real_gate_passes: true,
                synthetic_gate_passes: false,
                real_gate_ratio: 1.0,
                synthetic_gate_ratio: 0.8,
                real_recall: 0.80,
                real_specificity: 0.85,
                synthetic_ranking_quality: 0.76,
                ..PresenceCheckpointRank::default()
            }
            .is_better_than(baseline)
        );
        assert!(
            PresenceCheckpointRank {
                real_gate_passes: false,
                synthetic_gate_passes: false,
                real_gate_ratio: 0.9,
                synthetic_gate_ratio: 0.1,
                real_recall: 0.72,
                real_specificity: 0.90,
                synthetic_ranking_quality: 0.1,
                ..PresenceCheckpointRank::default()
            }
            .is_better_than(baseline)
        );

        let real_safe = PresenceCheckpointRank {
            real_gate_passes: true,
            synthetic_gate_passes: false,
            real_gate_ratio: 1.0,
            synthetic_gate_ratio: 0.8,
            real_recall: 0.80,
            real_specificity: 0.85,
            synthetic_ranking_quality: 0.76,
            ..PresenceCheckpointRank::default()
        };
        assert!(
            PresenceCheckpointRank {
                synthetic_gate_ratio: 0.9,
                synthetic_ranking_quality: 0.855,
                ..real_safe
            }
            .is_better_than(real_safe)
        );
    }

    #[test]
    fn presence_broken_checkpoint_cannot_outrank_presence_safe_checkpoint() {
        let broken = EvaluationMetrics {
            recall: 1.0,
            specificity: 0.10,
            real_positive_recall: 1.0,
            real_positive_count: 2,
            real_positive_building_count: 2,
            real_positive_buildings_detected: 2,
            real_positive_building_recall: 1.0,
            real_negative_specificity: 0.70,
            real_negative_count: 20,
            pck_05: 1.0,
            offscreen_accuracy: 1.0,
            ..EvaluationMetrics::default()
        };
        let safe = EvaluationMetrics {
            recall: 0.95,
            specificity: 0.90,
            real_positive_recall: 1.0,
            real_positive_count: 2,
            real_positive_building_count: 2,
            real_positive_buildings_detected: 2,
            real_positive_building_recall: 1.0,
            real_negative_specificity: 0.85,
            real_negative_count: 20,
            pck_05: 0.50,
            offscreen_accuracy: 0.50,
            ..EvaluationMetrics::default()
        };
        assert!(
            safe.checkpoint_rank()
                .is_better_than(broken.checkpoint_rank())
        );
    }

    #[test]
    fn geometry_selects_between_fully_presence_safe_checkpoints() {
        let metrics = |pck_05| EvaluationMetrics {
            real_positive_recall: MIN_REAL_POSITIVE_RECALL,
            real_positive_count: 5,
            real_positive_building_count: 3,
            real_positive_buildings_detected: 3,
            real_positive_building_recall: 1.0,
            real_negative_specificity: MIN_REAL_NEGATIVE_SPECIFICITY,
            real_negative_count: 20,
            pck_05,
            offscreen_accuracy: 0.95,
            presence_diagnostics: PresenceDiagnostics {
                synthetic_roc_auc: Some(MIN_SYNTHETIC_ROC_AUC),
                synthetic_average_precision: Some(MIN_SYNTHETIC_AVERAGE_PRECISION),
                ..PresenceDiagnostics::default()
            },
            ..EvaluationMetrics::default()
        };

        assert!(
            metrics(0.90)
                .checkpoint_rank()
                .is_better_than(metrics(0.80).checkpoint_rank())
        );
    }

    fn safe_checkpoint_metrics(
        real_specificity: f32,
        real_auc: Option<f32>,
        real_ap: Option<f32>,
        pck_05: f32,
    ) -> EvaluationMetrics {
        EvaluationMetrics {
            real_positive_recall: 5.0 / 6.0,
            real_positive_count: 6,
            real_positive_building_count: 3,
            real_positive_buildings_detected: 3,
            real_positive_building_recall: 1.0,
            real_negative_specificity: real_specificity,
            real_negative_count: 89,
            pck_05,
            offscreen_accuracy: 0.99,
            presence_diagnostics: PresenceDiagnostics {
                synthetic_roc_auc: Some(0.97),
                synthetic_average_precision: Some(0.97),
                real_roc_auc: real_auc,
                real_average_precision: real_ap,
                ..PresenceDiagnostics::default()
            },
            ..EvaluationMetrics::default()
        }
    }

    #[test]
    fn robust_real_band_outranks_marginal_geometry_after_all_gates_pass() {
        let robust = safe_checkpoint_metrics(0.876, Some(0.895), Some(0.600), 0.905);
        let geometry_only = safe_checkpoint_metrics(0.854, Some(0.861), Some(0.470), 0.950);

        assert!(
            robust
                .checkpoint_rank()
                .is_better_than(geometry_only.checkpoint_rank())
        );
    }

    #[test]
    fn geometry_breaks_ties_inside_the_same_presence_bands() {
        let lower_geometry = safe_checkpoint_metrics(0.876, Some(0.895), Some(0.600), 0.910);
        let higher_geometry = safe_checkpoint_metrics(0.877, Some(0.896), Some(0.601), 0.930);

        assert!(
            higher_geometry
                .checkpoint_rank()
                .is_better_than(lower_geometry.checkpoint_rank())
        );
    }

    #[test]
    fn geometry_gate_pass_outranks_presence_quality_until_geometry_is_viable() {
        let safer_but_not_viable = safe_checkpoint_metrics(0.95, Some(0.97), Some(0.90), 0.899);
        let viable = safe_checkpoint_metrics(0.85, Some(0.85), Some(0.45), 0.900);

        assert!(
            viable
                .checkpoint_rank()
                .is_better_than(safer_but_not_viable.checkpoint_rank())
        );
    }

    #[test]
    fn production_rank_fails_closed_when_real_ranking_metrics_are_missing() {
        let missing = safe_checkpoint_metrics(0.90, None, None, 0.92);
        let measured = safe_checkpoint_metrics(0.86, Some(0.86), Some(0.50), 0.91);

        assert!(
            measured
                .checkpoint_rank()
                .is_better_than(missing.checkpoint_rank())
        );
    }

    #[test]
    fn real_gate_progress_outranks_stronger_geometry() {
        let early = EvaluationMetrics {
            recall: 0.979,
            specificity: 0.167,
            real_positive_recall: 1.0,
            real_positive_count: 2,
            real_positive_building_count: 2,
            real_positive_buildings_detected: 2,
            real_positive_building_recall: 1.0,
            real_negative_specificity: 0.753,
            real_negative_count: 89,
            pck_05: 0.536,
            offscreen_accuracy: 0.972,
            duplicate_point_pairs: 325,
            presence_diagnostics: PresenceDiagnostics {
                roc_auc: Some(0.795),
                average_precision: Some(0.787),
                synthetic_roc_auc: Some(0.769),
                synthetic_average_precision: Some(0.790),
                real_roc_auc: Some(0.876),
                real_average_precision: Some(0.542),
                ..PresenceDiagnostics::default()
            },
            ..EvaluationMetrics::default()
        };
        let later = EvaluationMetrics {
            recall: 1.0,
            specificity: 0.086,
            real_positive_recall: 1.0,
            real_positive_count: 2,
            real_positive_building_count: 2,
            real_positive_buildings_detected: 2,
            real_positive_building_recall: 1.0,
            real_negative_specificity: 0.573,
            real_negative_count: 89,
            pck_05: 0.887,
            offscreen_accuracy: 0.992,
            duplicate_point_pairs: 48,
            presence_diagnostics: PresenceDiagnostics {
                roc_auc: Some(0.983),
                average_precision: Some(0.983),
                synthetic_roc_auc: Some(0.982),
                synthetic_average_precision: Some(0.984),
                real_roc_auc: Some(0.787),
                real_average_precision: Some(0.525),
                ..PresenceDiagnostics::default()
            },
            ..EvaluationMetrics::default()
        };

        assert!(
            early
                .checkpoint_rank()
                .is_better_than(later.checkpoint_rank())
        );
    }

    #[test]
    fn approximate_geometry_does_not_outrank_a_safer_operating_point() {
        let presence_safer = EvaluationMetrics {
            recall: 1.0,
            specificity: 0.249,
            real_positive_recall: 1.0,
            real_positive_count: 8,
            real_positive_building_count: 4,
            real_positive_buildings_detected: 4,
            real_positive_building_recall: 1.0,
            real_negative_specificity: 0.876,
            real_negative_count: 89,
            pck_05: 0.887,
            offscreen_accuracy: 0.99,
            presence_diagnostics: PresenceDiagnostics {
                real_roc_auc: Some(0.85),
                real_average_precision: Some(0.60),
                ..PresenceDiagnostics::default()
            },
            ..EvaluationMetrics::default()
        };
        let geometry_stronger = EvaluationMetrics {
            recall: 1.0,
            specificity: 0.109,
            real_positive_recall: 0.875,
            real_positive_count: 8,
            real_positive_building_count: 4,
            real_positive_buildings_detected: 4,
            real_positive_building_recall: 1.0,
            real_negative_specificity: 0.742,
            real_negative_count: 89,
            pck_05: 0.903,
            offscreen_accuracy: 0.99,
            presence_diagnostics: PresenceDiagnostics {
                real_roc_auc: Some(0.89),
                real_average_precision: Some(0.62),
                ..PresenceDiagnostics::default()
            },
            ..EvaluationMetrics::default()
        };

        assert!(
            presence_safer
                .checkpoint_rank()
                .is_better_than(geometry_stronger.checkpoint_rank())
        );
    }

    #[test]
    fn source_weights_sum_to_configured_mass_per_origin() {
        let sample = |origin| TrainingSample {
            key: "sample".to_owned(),
            split: Split::Train,
            image: ImageSource::Encoded(Vec::<u8>::new().into()),
            presence: i32::from(matches!(
                origin,
                SampleOrigin::SyntheticTarget | SampleOrigin::RealPositive
            )),
            geometry: None,
            origin,
            physical_building_id: (origin == SampleOrigin::RealPositive)
                .then(|| Arc::<str>::from("test-building")),
        };
        let samples = vec![
            sample(SampleOrigin::SyntheticTarget),
            sample(SampleOrigin::SyntheticTarget),
            sample(SampleOrigin::RealPositive),
            sample(SampleOrigin::RealNegative),
        ];
        let weights = source_balancing_weights(&samples);
        assert_eq!(weights, [0.25, 0.25, 1.0, 1.0]);
    }

    #[test]
    fn real_presence_pair_indices_exclude_both_synthetic_sources() {
        let samples = [
            SampleOrigin::SyntheticTarget,
            SampleOrigin::RealNegative,
            SampleOrigin::RealPositive,
            SampleOrigin::SyntheticNegative,
            SampleOrigin::RealPositive,
        ]
        .map(|origin| TrainingSample {
            key: format!("{origin:?}"),
            split: Split::Train,
            image: ImageSource::Encoded(Vec::<u8>::new().into()),
            presence: i32::from(matches!(
                origin,
                SampleOrigin::SyntheticTarget | SampleOrigin::RealPositive
            )),
            geometry: None,
            origin,
            physical_building_id: None,
        });

        assert_eq!(
            real_presence_pair_indices(&samples),
            RealPresencePairIndices {
                positives: vec![2, 4],
                negatives: vec![1],
            }
        );
    }

    #[test]
    fn v15_geometry_only_objective_has_no_presence_gradient_path() {
        type TestBackend = Autodiff<Flex>;

        let device = FlexDevice;
        let presence_logits = Tensor::<TestBackend, 1>::from_floats([0.25], &device).require_grad();
        let keypoint_logits = Tensor::<TestBackend, 4>::zeros(
            [1, KEYPOINT_COUNT, HEATMAP_SIZE, HEATMAP_SIZE],
            &device,
        )
        .require_grad();
        let offscreen_logits =
            Tensor::<TestBackend, 2>::zeros([1, KEYPOINT_COUNT], &device).require_grad();
        let output = KeypointRoofGeometryOutput {
            keypoint_logits: keypoint_logits.clone(),
            offscreen_logits: offscreen_logits.clone(),
        };
        let mut target = vec![0.0; KEYPOINT_COUNT * KEYPOINT_DISTRIBUTION_SIZE];
        for keypoint in 0..KEYPOINT_COUNT {
            target[keypoint * KEYPOINT_DISTRIBUTION_SIZE + OFFSCREEN_INDEX] = 1.0;
        }
        let batch = ObservationBatch {
            images: Tensor::<TestBackend, 4>::zeros([1, 3, 1, 1], &device),
            presence: Tensor::from_floats([1.0], &device),
            presence_weights: Tensor::from_floats([1.0], &device),
            real_positive_indices: Tensor::from_ints([0], &device),
            real_positive_count: 1,
            real_negative_indices: Tensor::from_ints([0], &device),
            real_negative_count: 0,
            keypoint_targets: Tensor::from_data(
                TensorData::new(
                    target,
                    Shape::new([1, KEYPOINT_COUNT, KEYPOINT_DISTRIBUTION_SIZE]),
                ),
                &device,
            ),
            symmetry_masks: Tensor::from_data(
                TensorData::new(symmetry_masks(), Shape::new([SYMMETRY_COUNT, 3, 4, 4])),
                &device,
            ),
            geometry_indices: Tensor::from_ints([0], &device),
            geometry_count: 1,
            positions: vec![[[0.0; 2]; KEYPOINT_COUNT]],
            in_frame: vec![[false; KEYPOINT_COUNT]],
            transforms: vec![LetterboxTransform::for_input_size(1, 1, 1)],
        };

        let weighted_loss =
            geometry_only_observation_loss(&output, &batch).unwrap() * GEOMETRY_LOSS_WEIGHT;
        let gradients = weighted_loss.backward();
        assert!(
            presence_logits.grad(&gradients).is_none(),
            "geometry-only backward must not create even a zero presence graph"
        );
        assert!(keypoint_logits.grad(&gradients).is_some());
        assert!(offscreen_logits.grad(&gradients).is_some());
        assert_eq!(REAL_PRESENCE_PAIRWISE_LOSS_WEIGHT, 0.0);
    }

    #[test]
    fn geometry_only_adamw_step_leaves_presence_output_unchanged_after_joint_momentum() {
        type TestBackend = Autodiff<Flex>;

        let device = FlexDevice;
        let mut model = KeypointRoofNetConfig::new().init::<TestBackend>(&device);
        let groups = model.parameter_groups();
        let input = Tensor::<TestBackend, 4>::random(
            [1, 3, 32, 32],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        let options = KeypointTrainingOptions {
            freeze_backbone_batch_norm: true,
            detach_geometry_backbone: true,
        };
        let mut optimizer = AdamWConfig::new().with_weight_decay(0.1).init();

        // Seed AdamW momentum for every head, including presence, exactly as a
        // preceding joint-training update would.
        let joint_output = model.forward_training_with_options(input.clone(), options);
        let mut joint_gradients = (joint_output.presence_logits.sum()
            + joint_output.keypoint_logits.sum()
            + joint_output.offscreen_logits.sum())
        .backward();
        let joint_head_gradients =
            GradientsParams::from_params(&mut joint_gradients, &model, &groups.heads);
        model = optimizer.step(0.01, model, joint_head_gradients);

        let geometry_output = model.forward_training_with_options(input.clone(), options);
        let presence_before = geometry_output
            .presence_logits
            .clone()
            .inner()
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        let mut geometry_gradients = (geometry_output.keypoint_logits.sum()
            + geometry_output.offscreen_logits.sum())
        .backward();
        let backbone_gradients =
            GradientsParams::from_params(&mut geometry_gradients, &model, &groups.backbone);
        let presence_gradients =
            GradientsParams::from_params(&mut geometry_gradients, &model, &groups.presence_heads);
        let geometry_gradients =
            GradientsParams::from_params(&mut geometry_gradients, &model, &groups.geometry_heads);
        assert!(backbone_gradients.is_empty());
        assert!(presence_gradients.is_empty());
        assert!(!geometry_gradients.is_empty());

        model = optimizer.step(0.01, model, geometry_gradients);
        let presence_after = model
            .forward_training_with_options(input, options)
            .presence_logits
            .inner()
            .to_data()
            .to_vec::<f32>()
            .unwrap();
        assert_eq!(
            presence_after, presence_before,
            "geometry-only AdamW must not apply stale presence momentum or weight decay"
        );
    }

    #[test]
    fn real_pairwise_loss_is_lower_when_positive_logits_outrank_negatives() {
        let device = FlexDevice;
        let positive_indices = Tensor::<Flex, 1, Int>::from_ints([0, 1], &device);
        let negative_indices = Tensor::<Flex, 1, Int>::from_ints([2, 3], &device);
        let ranked = real_presence_pairwise_loss(
            Tensor::<Flex, 1>::from_floats([2.0, 1.0, -1.0, -2.0], &device),
            positive_indices.clone(),
            2,
            negative_indices.clone(),
            2,
        )
        .into_scalar()
        .elem::<f32>();
        let reversed = real_presence_pairwise_loss(
            Tensor::<Flex, 1>::from_floats([-1.0, -2.0, 2.0, 1.0], &device),
            positive_indices,
            2,
            negative_indices,
            2,
        )
        .into_scalar()
        .elem::<f32>();

        assert!(ranked.is_finite());
        assert!(reversed.is_finite());
        assert!(ranked < reversed, "ranked={ranked}, reversed={reversed}");
    }

    #[test]
    fn real_pairwise_loss_has_finite_gradients_and_safe_empty_pairs() {
        type TestBackend = Autodiff<Flex>;

        let device = FlexDevice;
        let logits =
            Tensor::<TestBackend, 1>::from_floats([-10_000.0, 10_000.0], &device).require_grad();
        let loss = real_presence_pairwise_loss(
            logits.clone(),
            Tensor::from_ints([0], &device),
            1,
            Tensor::from_ints([1], &device),
            1,
        );
        let loss_value = loss.clone().inner().into_scalar().elem::<f32>();
        let gradients = loss.backward();
        let gradient = logits
            .grad(&gradients)
            .expect("ranking loss must remain connected to its logits")
            .to_data()
            .to_vec::<f32>()
            .expect("read ranking gradients");
        assert!(loss_value.is_finite());
        assert!(gradient.iter().all(|value| value.is_finite()));
        assert!(gradient[0] < 0.0);
        assert!(gradient[1] > 0.0);

        for (positive_count, negative_count) in [(0, 1), (1, 0), (0, 0)] {
            let logits = Tensor::<TestBackend, 1>::from_floats([3.0, -2.0], &device).require_grad();
            let loss = real_presence_pairwise_loss(
                logits.clone(),
                Tensor::from_ints([0], &device),
                positive_count,
                Tensor::from_ints([1], &device),
                negative_count,
            );
            let loss_value = loss.clone().inner().into_scalar().elem::<f32>();
            let gradients = loss.backward();
            let gradient = logits
                .grad(&gradients)
                .expect("empty-pair zero must remain connected to logits")
                .to_data()
                .to_vec::<f32>()
                .expect("read empty-pair gradients");
            assert_eq!(loss_value, 0.0);
            assert!(
                gradient
                    .iter()
                    .all(|value| value.is_finite() && *value == 0.0)
            );
        }
    }

    fn sampling_fixture_sample(
        key: impl Into<String>,
        origin: SampleOrigin,
        physical_building_id: Option<&str>,
    ) -> TrainingSample {
        TrainingSample {
            key: key.into(),
            split: Split::Train,
            image: ImageSource::Encoded(Vec::<u8>::new().into()),
            presence: i32::from(matches!(
                origin,
                SampleOrigin::SyntheticTarget | SampleOrigin::RealPositive
            )),
            geometry: None,
            origin,
            physical_building_id: physical_building_id.map(Arc::<str>::from),
        }
    }

    fn building_balancing_fixture() -> Vec<TrainingSample> {
        let mut samples = Vec::new();
        for index in 0..36 {
            samples.push(sampling_fixture_sample(
                format!("synthetic-target-{index:02}"),
                SampleOrigin::SyntheticTarget,
                None,
            ));
            samples.push(sampling_fixture_sample(
                format!("synthetic-negative-{index:02}"),
                SampleOrigin::SyntheticNegative,
                None,
            ));
        }
        for view in 0..3 {
            samples.push(sampling_fixture_sample(
                format!("building-a-view-{view}"),
                SampleOrigin::RealPositive,
                Some("building-a"),
            ));
        }
        samples.push(sampling_fixture_sample(
            "building-b-view-0",
            SampleOrigin::RealPositive,
            Some("building-b"),
        ));
        samples.push(sampling_fixture_sample(
            "real-negative",
            SampleOrigin::RealNegative,
            None,
        ));
        samples
    }

    #[test]
    fn balanced_epoch_allocates_equal_draws_to_physical_buildings() {
        let epoch = balanced_epoch_samples(&building_balancing_fixture(), 16, 42, 1).unwrap();
        let real_positives = epoch
            .iter()
            .filter(|sample| sample.origin == SampleOrigin::RealPositive)
            .collect::<Vec<_>>();

        assert_eq!(real_positives.len(), 12);
        assert_eq!(
            real_positives
                .iter()
                .filter(|sample| sample.physical_building_id.as_deref() == Some("building-a"))
                .count(),
            6
        );
        assert_eq!(
            real_positives
                .iter()
                .filter(|sample| sample.physical_building_id.as_deref() == Some("building-b"))
                .count(),
            6
        );
    }

    #[test]
    fn building_draw_remainder_differs_by_at_most_one() {
        let samples = [
            sampling_fixture_sample(
                "building-a-view-0",
                SampleOrigin::RealPositive,
                Some("building-a"),
            ),
            sampling_fixture_sample(
                "building-a-view-1",
                SampleOrigin::RealPositive,
                Some("building-a"),
            ),
            sampling_fixture_sample(
                "building-b-view-0",
                SampleOrigin::RealPositive,
                Some("building-b"),
            ),
            sampling_fixture_sample(
                "building-c-view-0",
                SampleOrigin::RealPositive,
                Some("building-c"),
            ),
            sampling_fixture_sample(
                "building-d-view-0",
                SampleOrigin::RealPositive,
                Some("building-d"),
            ),
        ];
        let draws = building_balanced_real_positive_draws(&samples, 11, 42, 1).unwrap();
        let mut counts = BTreeMap::<&str, usize>::new();
        for sample in &draws {
            *counts
                .entry(sample.physical_building_id.as_deref().unwrap())
                .or_default() += 1;
        }
        let minimum = counts.values().min().copied().unwrap();
        let maximum = counts.values().max().copied().unwrap();

        assert_eq!(counts.len(), 4);
        assert_eq!(counts.values().sum::<usize>(), 11);
        assert!(maximum - minimum <= 1, "{counts:?}");
    }

    #[test]
    fn balanced_epoch_cycles_building_views_deterministically() {
        let samples = building_balancing_fixture();
        let first = balanced_epoch_samples(&samples, 16, 42, 3).unwrap();
        let repeated = balanced_epoch_samples(&samples, 16, 42, 3).unwrap();
        assert_eq!(
            first.iter().map(|sample| &sample.key).collect::<Vec<_>>(),
            repeated
                .iter()
                .map(|sample| &sample.key)
                .collect::<Vec<_>>()
        );

        let building_a_views = first
            .iter()
            .filter(|sample| sample.physical_building_id.as_deref() == Some("building-a"))
            .map(|sample| sample.key.split('#').next().unwrap().to_owned())
            .collect::<Vec<_>>();
        assert_eq!(building_a_views.len(), 6);
        let mut unique_cycle = building_a_views[..3].to_vec();
        unique_cycle.sort();
        unique_cycle.dedup();
        assert_eq!(unique_cycle.len(), 3);
        assert_eq!(&building_a_views[..3], &building_a_views[3..]);
    }

    #[test]
    fn balanced_epoch_rejects_real_positive_without_building_identity() {
        let mut samples = building_balancing_fixture();
        let unidentified = samples
            .iter_mut()
            .find(|sample| sample.origin == SampleOrigin::RealPositive)
            .unwrap();
        unidentified.physical_building_id = None;

        let error = balanced_epoch_samples(&samples, 16, 42, 1)
            .err()
            .expect("real-positive grouping identity is mandatory")
            .to_string();
        assert!(error.contains("physical_building_id"), "{error}");
    }

    #[test]
    fn balanced_epoch_allocates_real_photo_slots() {
        let sample = |index, origin| TrainingSample {
            key: format!("sample-{index}"),
            split: Split::Train,
            image: ImageSource::Encoded(Vec::<u8>::new().into()),
            presence: i32::from(matches!(
                origin,
                SampleOrigin::SyntheticTarget | SampleOrigin::RealPositive
            )),
            geometry: None,
            origin,
            physical_building_id: (origin == SampleOrigin::RealPositive)
                .then(|| Arc::<str>::from("test-building")),
        };
        let mut samples = Vec::new();
        for index in 0..20 {
            samples.push(sample(index, SampleOrigin::SyntheticTarget));
            samples.push(sample(index + 20, SampleOrigin::SyntheticNegative));
        }
        samples.push(sample(40, SampleOrigin::RealPositive));
        samples.push(sample(41, SampleOrigin::RealNegative));
        let epoch = balanced_epoch_samples(&samples, 16, 42, 1).unwrap();
        for batch in epoch.chunks(16) {
            let mut counts = [0usize; 4];
            for item in batch {
                counts[origin_index(item.origin)] += 1;
            }
            assert_eq!(counts, [6, 6, 2, 2]);
        }
    }

    #[test]
    fn collapse_metric_ignores_genuinely_foreshortened_targets() {
        let mut predicted = [None; KEYPOINT_COUNT];
        let mut target = [[0.0; 2]; KEYPOINT_COUNT];
        let mut in_frame = [false; KEYPOINT_COUNT];
        predicted[0] = Some([0.5, 0.5]);
        predicted[1] = Some([0.5, 0.5]);
        target[0] = [0.5, 0.5];
        target[1] = [0.501, 0.5];
        in_frame[0] = true;
        in_frame[1] = true;
        assert_eq!(duplicate_point_pairs(&predicted, &target, &in_frame, 0), 0);

        target[1] = [0.7, 0.5];
        assert_eq!(duplicate_point_pairs(&predicted, &target, &in_frame, 0), 1);
    }

    #[test]
    fn collapse_metric_checks_corresponding_points_across_roof_tiers() {
        let mut predicted = [None; KEYPOINT_COUNT];
        let mut target = [[0.0; 2]; KEYPOINT_COUNT];
        let mut in_frame = [false; KEYPOINT_COUNT];
        predicted[0] = Some([0.5, 0.5]);
        predicted[4] = Some([0.5, 0.5]);
        target[0] = [0.4, 0.7];
        target[4] = [0.5, 0.5];
        in_frame[0] = true;
        in_frame[4] = true;

        assert_eq!(duplicate_point_pairs(&predicted, &target, &in_frame, 0), 1);
    }

    fn passing_promotion_metrics() -> (EvaluationMetrics, EvaluationMetrics) {
        let validation = EvaluationMetrics {
            recall: 0.10,
            specificity: 0.10,
            real_positive_recall: MIN_REAL_POSITIVE_VIEW_RECALL,
            real_positive_count: 4,
            real_positive_building_count: 3,
            real_positive_buildings_detected: 3,
            real_positive_building_recall: 1.0,
            real_negative_specificity: MIN_REAL_NEGATIVE_SPECIFICITY,
            real_negative_count: 20,
            ..EvaluationMetrics::default()
        };
        let mut test = EvaluationMetrics {
            // Aggregate threshold metrics intentionally fail the retired
            // cross-domain gate. Deployment metrics and synthetic ranking are
            // the source-aware promotion contract.
            recall: 0.10,
            specificity: 0.10,
            real_positive_recall: MIN_REAL_POSITIVE_VIEW_RECALL,
            real_positive_count: 4,
            real_positive_building_count: 3,
            real_positive_buildings_detected: 3,
            real_positive_building_recall: 1.0,
            real_negative_specificity: MIN_REAL_NEGATIVE_SPECIFICITY,
            real_negative_count: 20,
            pck_05: 0.90,
            offscreen_accuracy: 0.90,
            presence_diagnostics: PresenceDiagnostics {
                synthetic_roc_auc: Some(MIN_SYNTHETIC_ROC_AUC),
                synthetic_average_precision: Some(MIN_SYNTHETIC_AVERAGE_PRECISION),
                ..PresenceDiagnostics::default()
            },
            ..EvaluationMetrics::default()
        };
        test.synthetic_fit.attempted = 10;
        test.synthetic_fit.fitted = 10;
        test.synthetic_fit.accepted = 10;
        test.synthetic_fit.fit_success_rate = 1.0;
        test.synthetic_fit.accepted_rate = 1.0;
        test.synthetic_fit.median_mesh_rmse = 0.02;
        test.synthetic_fit.median_silhouette_iou = 0.90;
        (validation, test)
    }

    #[test]
    fn final_gate_uses_real_operating_metrics_not_aggregate_specificity() {
        use clap::Parser;

        let args = Args::try_parse_from(["roof-train"]).unwrap();
        let (validation, mut test) = passing_promotion_metrics();
        test.real_negative_specificity = 0.80;

        let rejected = PromotionDecision::from_metrics(&args, &validation, &test);
        assert!(!rejected.promoted);
        assert!(
            rejected
                .failures
                .iter()
                .any(|failure| failure.contains("real-negative specificity"))
        );

        test.real_negative_specificity = MIN_REAL_NEGATIVE_SPECIFICITY;
        assert!(PromotionDecision::from_metrics(&args, &validation, &test).promoted);
    }

    #[test]
    fn final_gate_never_promotes_a_capped_diagnostic_split() {
        use clap::Parser;

        let args = Args::try_parse_from(["roof-train", "--limit-per-split", "256"]).unwrap();
        let (validation, test) = passing_promotion_metrics();
        let decision = PromotionDecision::from_metrics(&args, &validation, &test);

        assert!(!decision.promoted);
        assert!(
            decision
                .failures
                .iter()
                .any(|failure| failure.contains("capped splits cannot promote"))
        );
    }

    #[test]
    fn final_gate_accepts_three_of_four_views_when_all_three_buildings_are_detected() {
        use clap::Parser;

        let args = Args::try_parse_from(["roof-train"]).unwrap();
        let (validation, test) = passing_promotion_metrics();
        assert_eq!(test.real_positive_recall, 3.0 / 4.0);
        assert_eq!(test.real_positive_buildings_detected, 3);
        assert_eq!(test.real_positive_building_count, 3);
        assert!(PromotionDecision::from_metrics(&args, &validation, &test).promoted);
    }

    #[test]
    fn final_gate_rejects_three_of_four_views_when_only_two_buildings_are_detected() {
        use clap::Parser;

        let args = Args::try_parse_from(["roof-train"]).unwrap();
        let (validation, mut test) = passing_promotion_metrics();
        test.real_positive_buildings_detected = 2;
        test.real_positive_building_recall = 2.0 / 3.0;

        let decision = PromotionDecision::from_metrics(&args, &validation, &test);
        assert!(!decision.promoted);
        assert!(
            decision
                .failures
                .iter()
                .any(|failure| failure.contains("test real-positive building recall"))
        );
    }

    #[test]
    fn final_gate_fails_closed_for_missing_real_positive_building_ids() {
        use clap::Parser;

        let args = Args::try_parse_from(["roof-train"]).unwrap();
        let (validation, mut test) = passing_promotion_metrics();
        test.real_positive_missing_building_id_count = 1;
        test.real_positive_building_recall = 0.0;

        let decision = PromotionDecision::from_metrics(&args, &validation, &test);
        assert!(!decision.promoted);
        assert!(
            decision
                .failures
                .iter()
                .any(|failure| failure.contains("without a physical building ID"))
        );
    }

    #[test]
    fn final_gate_enforces_validation_real_specificity() {
        use clap::Parser;

        let args = Args::try_parse_from(["roof-train"]).unwrap();
        let (mut validation, test) = passing_promotion_metrics();
        validation.real_negative_specificity = 0.84;

        let rejected = PromotionDecision::from_metrics(&args, &validation, &test);
        assert!(!rejected.promoted);
        assert!(
            rejected
                .failures
                .iter()
                .any(|failure| failure.contains("validation real-negative specificity"))
        );
    }

    #[test]
    fn final_gate_replaces_synthetic_specificity_with_roc_auc_and_ap() {
        use clap::Parser;

        let args = Args::try_parse_from(["roof-train"]).unwrap();
        let (validation, mut test) = passing_promotion_metrics();
        test.synthetic_negative_specificity = 0.0;
        assert!(PromotionDecision::from_metrics(&args, &validation, &test).promoted);

        test.presence_diagnostics.synthetic_roc_auc = Some(0.94);
        let rejected = PromotionDecision::from_metrics(&args, &validation, &test);
        assert!(!rejected.promoted);
        assert!(
            rejected
                .failures
                .iter()
                .any(|failure| failure.contains("synthetic ROC AUC"))
        );

        test.presence_diagnostics.synthetic_roc_auc = Some(0.95);
        test.presence_diagnostics.synthetic_average_precision = None;
        let rejected = PromotionDecision::from_metrics(&args, &validation, &test);
        assert!(
            rejected
                .failures
                .iter()
                .any(|failure| failure.contains("synthetic average precision is unavailable"))
        );
    }

    #[test]
    fn final_gate_accepts_approximate_single_frame_fit_quality() {
        use clap::Parser;

        let args = Args::try_parse_from(["roof-train"]).unwrap();
        let (validation, mut test) = passing_promotion_metrics();
        test.synthetic_fit.median_mesh_rmse = MAX_MEDIAN_FITTED_MESH_RMSE;
        test.synthetic_fit.median_silhouette_iou = MIN_MEDIAN_AMODAL_SILHOUETTE_IOU;
        assert!(PromotionDecision::from_metrics(&args, &validation, &test).promoted);

        test.synthetic_fit.median_mesh_rmse = MAX_MEDIAN_FITTED_MESH_RMSE + 0.001;
        let rejected = PromotionDecision::from_metrics(&args, &validation, &test);
        assert!(
            rejected
                .failures
                .iter()
                .any(|failure| failure.contains("fitted-mesh RMSE"))
        );

        test.synthetic_fit.median_mesh_rmse = MAX_MEDIAN_FITTED_MESH_RMSE;
        test.synthetic_fit.median_silhouette_iou = MIN_MEDIAN_AMODAL_SILHOUETTE_IOU - 0.001;
        let rejected = PromotionDecision::from_metrics(&args, &validation, &test);
        assert!(
            rejected
                .failures
                .iter()
                .any(|failure| failure.contains("amodal silhouette IoU"))
        );
    }
}
