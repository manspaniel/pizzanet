//! Trains the Burn still-image roof observation model.

mod augmentation;
mod fit_evaluation;
mod keypoint_train;

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    process::ExitCode,
    sync::Arc,
};

use anyhow::{Context, Result, bail};
use burn::backend::{Autodiff, Flex, Wgpu, flex::FlexDevice, wgpu::WgpuDevice};
#[cfg(feature = "cuda")]
use burn::backend::{Cuda, cuda::CudaDevice};
use clap::{Parser, ValueEnum};
use roof_model::{HEATMAP_SIZE, KEYPOINT_COUNT, SPATIAL_INPUT_SIZE};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use synth_data::{DatasetSplit, FrameRecord, TargetKind};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BackendChoice {
    Cuda,
    Wgpu,
    Flex,
}

#[derive(Debug, Parser)]
#[command(
    name = "roof-train",
    about = "Train the presence and amodal-keypoint roof model"
)]
struct Args {
    /// Synthetic WebDataset directory containing target and ordinary-building scenes.
    #[arg(long, default_value = "datasets/synthetic-training-keypoints")]
    synthetic: PathBuf,
    /// Curated Open Images ordinary-building negative dataset.
    #[arg(long, default_value = "datasets/open-images-negatives")]
    negatives: PathBuf,
    /// Manifest-backed Wikimedia historical sites with per-image roof-visibility review.
    #[arg(long, default_value = "datasets/wikimedia-positives")]
    real_positives: PathBuf,
    /// Legacy per-record multiplicity; source-balanced epochs already redraw fresh augmentations.
    #[arg(long, default_value_t = 1)]
    real_positive_repeat: usize,
    /// Output directory for the checkpoint, manifest, config, and metrics.
    #[arg(long, default_value = "artifacts/roof-model-keypoints")]
    artifacts: PathBuf,
    /// Training backend.
    #[arg(long, value_enum, default_value_t = BackendChoice::Wgpu)]
    backend: BackendChoice,
    /// Maximum number of complete training passes.
    #[arg(long, default_value_t = 40)]
    epochs: usize,
    /// Stop after this many epochs without a better validation checkpoint.
    #[arg(long, default_value_t = 6)]
    patience: usize,
    /// Samples per optimizer update.
    #[arg(long, default_value_t = 16)]
    batch_size: usize,
    /// Samples per validation/test forward pass; defaults to the training batch size.
    #[arg(long)]
    evaluation_batch_size: Option<usize>,
    /// AdamW learning rate for the FPN and output heads.
    #[arg(long, default_value_t = 3.0e-4)]
    head_learning_rate: f64,
    /// AdamW learning rate for the pretrained MobileNetV2 backbone.
    #[arg(long, default_value_t = 3.0e-5)]
    backbone_learning_rate: f64,
    /// Keep MobileNetV2 weights fixed for the first N complete epochs.
    ///
    /// The FPN and output heads still update during this stage. BatchNorm
    /// behavior remains independently controlled by --freeze-backbone-batch-norm.
    #[arg(long, default_value_t = 0)]
    backbone_freeze_epochs: usize,
    /// Keep MobileNetV2 BatchNorm running statistics at their pretrained values.
    ///
    /// Backbone convolutions and BatchNorm affine parameters remain trainable;
    /// newly initialized FPN BatchNorm layers continue adapting normally.
    #[arg(long)]
    freeze_backbone_batch_norm: bool,
    /// Prevent keypoint and offscreen losses from updating MobileNetV2.
    ///
    /// Presence gradients continue through the backbone. The FPN, keypoint,
    /// and offscreen heads remain fully trainable.
    #[arg(long)]
    detach_geometry_backbone: bool,
    /// Permanently freeze presence learning after this many consecutive safe validations.
    ///
    /// The following epoch trains only the geometry heads. Set to zero to
    /// disable the automatic presence lock.
    #[arg(long, default_value_t = 2)]
    presence_freeze_after_safe_epochs: usize,
    /// AdamW decoupled weight decay.
    #[arg(long, default_value_t = 1.0e-4)]
    weight_decay: f64,
    /// Fraction of optimizer updates used for linear warm-up.
    #[arg(long, default_value_t = 0.05)]
    warmup_fraction: f64,
    /// Deterministic shuffle and augmentation seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Optional deterministic cap per split for smoke tests.
    #[arg(long)]
    limit_per_split: Option<usize>,
    /// Disable image augmentation for the explicit small-set overfit gate.
    #[arg(long)]
    disable_augmentation: bool,
    /// Train and evaluate on exactly 32 synthetic targets plus 32 synthetic negatives.
    ///
    /// This gate does not read either real-photo dataset. It disables augmentation and early
    /// stopping so that it measures whether the model can memorise its geometric supervision.
    #[arg(long)]
    overfit: bool,
    /// Evaluate an existing checkpoint base path without performing optimizer updates.
    #[arg(long, value_name = "CHECKPOINT")]
    evaluate_checkpoint: Option<PathBuf>,
    /// Continue only geometry learning from an already presence-safe checkpoint.
    ///
    /// This loads model weights but deliberately starts a fresh geometry-head
    /// AdamW optimizer. The output --artifacts directory must be separate from
    /// the checkpoint's directory so the source run cannot be overwritten.
    #[arg(long, value_name = "CHECKPOINT")]
    geometry_refine_from: Option<PathBuf>,
    /// Last completed logical epoch represented by --geometry-refine-from.
    ///
    /// Refinement starts at the following epoch; --epochs remains the final
    /// logical epoch rather than a count of additional epochs.
    #[arg(long, requires = "geometry_refine_from")]
    geometry_refine_source_epoch: Option<usize>,
}

#[derive(Clone)]
enum ImageSource {
    Encoded(Arc<[u8]>),
    Path(PathBuf),
}

#[derive(Clone)]
struct TrainingSample {
    key: String,
    split: Split,
    image: ImageSource,
    presence: i32,
    geometry: Option<Arc<FrameRecord>>,
    origin: SampleOrigin,
    /// Stable physical-site identity used to balance eligible real positives.
    /// Other sources do not have this grouping contract.
    physical_building_id: Option<Arc<str>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SampleOrigin {
    SyntheticTarget,
    SyntheticNegative,
    RealPositive,
    RealNegative,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Split {
    Train,
    Validation,
    Test,
}

#[derive(Debug, Deserialize)]
struct NegativeManifest {
    schema_version: String,
    records: Vec<NegativeManifestRecord>,
}

#[derive(Debug, Deserialize)]
struct NegativeManifestRecord {
    image_id: String,
    split: String,
    relative_path: String,
    review_status: String,
}

#[derive(Debug, Deserialize)]
struct PositiveManifest {
    schema_version: String,
    records: Vec<PositiveManifestRecord>,
}

#[derive(Debug, Deserialize)]
struct PositiveManifestRecord {
    page_id: u64,
    physical_building_id: String,
    split: String,
    relative_path: String,
    review_status: String,
    characteristic_roof_visibility: CharacteristicRoofVisibility,
    visibility_review_reason: Option<String>,
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum CharacteristicRoofVisibility {
    Recognizable,
    NotRecognizable,
    Unreviewed,
}

struct LoadedPositiveRecord {
    page_id: u64,
    physical_building_id: String,
    split: Split,
    path: PathBuf,
}

#[derive(Default)]
struct PartialSyntheticSample {
    image: Option<Arc<[u8]>>,
    record: Option<FrameRecord>,
}

#[derive(Clone, Debug, Serialize)]
struct TrainingConfigRecord {
    schema_version: String,
    training_mode: String,
    geometry_refine_source_checkpoint: Option<String>,
    geometry_refine_source_checkpoint_sha256: Option<String>,
    geometry_refine_source_epoch: Option<usize>,
    first_training_epoch: usize,
    optimizer_state_policy: String,
    learning_rate_schedule_policy: String,
    synthetic: String,
    negatives: String,
    real_positives: String,
    real_positive_repeat: usize,
    real_positive_sampling_strategy: String,
    backend: String,
    epochs: usize,
    patience: usize,
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
    presence_threshold_calibration: String,
    checkpoint_selection_policy: String,
    checkpoint_real_robustness_weights: [f32; 4],
    checkpoint_real_robustness_band_basis_points: u16,
    checkpoint_real_gate_margin_band_basis_points: u16,
    checkpoint_synthetic_margin_band_basis_points: u16,
    checkpoint_synthetic_quality_band_basis_points: u16,
    minimum_real_positive_recall: f32,
    minimum_real_positive_view_recall: f32,
    minimum_real_positive_building_recall: f32,
    minimum_real_negative_specificity: f32,
    minimum_synthetic_roc_auc: f32,
    minimum_synthetic_average_precision: f32,
    minimum_standard_pck: f32,
    minimum_offscreen_accuracy: f32,
    seed: u64,
    limit_per_split: Option<usize>,
    disable_augmentation: bool,
    overfit: bool,
    evaluate_checkpoint: Option<String>,
    input_size: usize,
    heatmap_size: usize,
    keypoint_count: usize,
    pretrained_backbone: String,
    train_samples: usize,
    train_real_positive_buildings: usize,
    validation_samples: usize,
    test_samples: usize,
}

fn main() -> ExitCode {
    match run(Args::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("roof-train: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(mut args: Args) -> Result<()> {
    validate_and_normalize_args(&mut args)?;

    let samples = if args.overfit {
        load_synthetic(&args.synthetic)?
    } else {
        load_all_samples(&args)?
    };
    let (mut train, mut validation, mut test) = if args.overfit {
        // A generator request for 32+32 distributes those examples across normal dataset splits.
        // The memorisation gate intentionally ignores those splits and reuses the exact corpus.
        overfit_splits(&samples)?
    } else {
        (
            split_samples(&samples, Split::Train),
            split_samples(&samples, Split::Validation),
            split_samples(&samples, Split::Test),
        )
    };
    if let Some(limit) = args.limit_per_split {
        train = limit_stratified(train, limit);
        validation = limit_stratified(validation, limit);
        test = limit_stratified(test, limit);
    }
    if train.is_empty() || validation.is_empty() || test.is_empty() {
        bail!("training, validation, and test splits must all be non-empty");
    }
    if !train.iter().any(|sample| sample.geometry.is_some()) {
        bail!("training split has no synthetic target geometry");
    }
    let train_real_positive_buildings = real_positive_building_count(&train)?;

    fs::create_dir_all(&args.artifacts)?;
    ensure_geometry_refinement_artifact_safety(&args)?;
    if args.evaluate_checkpoint.is_none() {
        if args.geometry_refine_from.is_some() {
            require_fresh_refinement_artifacts(&args.artifacts)?;
        } else {
            archive_existing_run_artifacts(&args.artifacts)?;
        }
    }
    let geometry_refine_source_checkpoint = args
        .geometry_refine_from
        .as_ref()
        .map(|path| {
            path.canonicalize()
                .with_context(|| format!("resolve refinement checkpoint {}", path.display()))
        })
        .transpose()?;
    let geometry_refine_source_checkpoint_sha256 = geometry_refine_source_checkpoint
        .as_deref()
        .map(sha256_file)
        .transpose()?;
    let config = TrainingConfigRecord {
        schema_version: "roof-training/v15".to_owned(),
        training_mode: if args.geometry_refine_from.is_some() {
            "geometry_refinement".to_owned()
        } else if args.evaluate_checkpoint.is_some() {
            "checkpoint_evaluation".to_owned()
        } else {
            "joint_then_geometry".to_owned()
        },
        geometry_refine_source_checkpoint: geometry_refine_source_checkpoint
            .as_ref()
            .map(|path| path.display().to_string()),
        geometry_refine_source_checkpoint_sha256,
        geometry_refine_source_epoch: args.geometry_refine_source_epoch,
        first_training_epoch: args.first_training_epoch(),
        optimizer_state_policy: if args.evaluate_checkpoint.is_some() {
            "not_applicable".to_owned()
        } else if args.geometry_refine_from.is_some() {
            "fresh_geometry_head_adamw_no_restored_moments/v1".to_owned()
        } else {
            "fresh_joint_adamw/v1".to_owned()
        },
        learning_rate_schedule_policy: if args.evaluate_checkpoint.is_some() {
            "not_applicable".to_owned()
        } else if args.geometry_refine_from.is_some() {
            "original_cosine_schedule_at_logical_epoch_offset/v1".to_owned()
        } else {
            "cosine_over_logical_epochs/v1".to_owned()
        },
        synthetic: args.synthetic.display().to_string(),
        negatives: args.negatives.display().to_string(),
        real_positives: args.real_positives.display().to_string(),
        real_positive_repeat: args.real_positive_repeat,
        real_positive_sampling_strategy: keypoint_train::REAL_POSITIVE_SAMPLING_STRATEGY.to_owned(),
        backend: format!("{:?}", args.backend).to_lowercase(),
        epochs: args.epochs,
        patience: args.patience,
        batch_size: args.batch_size,
        evaluation_batch_size: args.evaluation_batch_size(),
        head_learning_rate: args.head_learning_rate,
        backbone_learning_rate: args.backbone_learning_rate,
        backbone_freeze_epochs: args.backbone_freeze_epochs,
        freeze_backbone_batch_norm: args.freeze_backbone_batch_norm,
        detach_geometry_backbone: args.detach_geometry_backbone,
        presence_freeze_policy: keypoint_train::PRESENCE_FREEZE_POLICY.to_owned(),
        presence_freeze_after_safe_epochs: args.presence_freeze_after_safe_epochs,
        weight_decay: args.weight_decay,
        warmup_fraction: args.warmup_fraction,
        geometry_loss_weight: keypoint_train::GEOMETRY_LOSS_WEIGHT,
        presence_source_mass: keypoint_train::PRESENCE_SOURCE_MASS,
        real_presence_pairwise_loss_weight: keypoint_train::REAL_PRESENCE_PAIRWISE_LOSS_WEIGHT,
        real_presence_pairwise_margin: keypoint_train::REAL_PRESENCE_PAIRWISE_MARGIN,
        presence_threshold_calibration: keypoint_train::PRESENCE_THRESHOLD_CALIBRATION.to_owned(),
        checkpoint_selection_policy: keypoint_train::CHECKPOINT_SELECTION_POLICY.to_owned(),
        checkpoint_real_robustness_weights: keypoint_train::CHECKPOINT_REAL_ROBUSTNESS_WEIGHTS,
        checkpoint_real_robustness_band_basis_points:
            keypoint_train::REAL_ROBUSTNESS_BAND_BASIS_POINTS,
        checkpoint_real_gate_margin_band_basis_points:
            keypoint_train::REAL_GATE_MARGIN_BAND_BASIS_POINTS,
        checkpoint_synthetic_margin_band_basis_points:
            keypoint_train::SYNTHETIC_MARGIN_BAND_BASIS_POINTS,
        checkpoint_synthetic_quality_band_basis_points:
            keypoint_train::SYNTHETIC_QUALITY_BAND_BASIS_POINTS,
        minimum_real_positive_recall: keypoint_train::MIN_REAL_POSITIVE_RECALL,
        minimum_real_positive_view_recall: keypoint_train::MIN_REAL_POSITIVE_VIEW_RECALL,
        minimum_real_positive_building_recall: keypoint_train::MIN_REAL_POSITIVE_BUILDING_RECALL,
        minimum_real_negative_specificity: keypoint_train::MIN_REAL_NEGATIVE_SPECIFICITY,
        minimum_synthetic_roc_auc: keypoint_train::MIN_SYNTHETIC_ROC_AUC,
        minimum_synthetic_average_precision: keypoint_train::MIN_SYNTHETIC_AVERAGE_PRECISION,
        minimum_standard_pck: keypoint_train::MIN_STANDARD_PCK,
        minimum_offscreen_accuracy: keypoint_train::MIN_OFFSCREEN_ACCURACY,
        seed: args.seed,
        limit_per_split: args.limit_per_split,
        disable_augmentation: args.disable_augmentation,
        overfit: args.overfit,
        evaluate_checkpoint: args
            .evaluate_checkpoint
            .as_ref()
            .map(|path| path.display().to_string()),
        input_size: SPATIAL_INPUT_SIZE,
        heatmap_size: HEATMAP_SIZE,
        keypoint_count: KEYPOINT_COUNT,
        pretrained_backbone: "torchvision MobileNetV2 ImageNet1K V2".to_owned(),
        train_samples: train.len(),
        train_real_positive_buildings,
        validation_samples: validation.len(),
        test_samples: test.len(),
    };
    fs::write(
        args.artifacts.join("training-config.json"),
        serde_json::to_vec_pretty(&config)?,
    )?;
    print_split_summary("train", &train);
    print_split_summary("validation", &validation);
    print_split_summary("test", &test);
    println!(
        "real-positive sampling: strategy={} train_views={} train_physical_buildings={}",
        keypoint_train::REAL_POSITIVE_SAMPLING_STRATEGY,
        train
            .iter()
            .filter(|sample| sample.origin == SampleOrigin::RealPositive)
            .count(),
        train_real_positive_buildings,
    );

    if let Some(checkpoint) = &args.evaluate_checkpoint {
        return match args.backend {
            BackendChoice::Cuda => {
                #[cfg(feature = "cuda")]
                {
                    keypoint_train::evaluate_checkpoint::<Autodiff<Cuda>>(
                        &args,
                        &train,
                        &validation,
                        &test,
                        checkpoint,
                        CudaDevice::default(),
                    )
                }
                #[cfg(not(feature = "cuda"))]
                {
                    bail!(
                        "CUDA support is not compiled; rerun Cargo with `--features cuda` before `-- --backend cuda`"
                    )
                }
            }
            BackendChoice::Wgpu => keypoint_train::evaluate_checkpoint::<Autodiff<Wgpu>>(
                &args,
                &train,
                &validation,
                &test,
                checkpoint,
                WgpuDevice::default(),
            ),
            BackendChoice::Flex => keypoint_train::evaluate_checkpoint::<Autodiff<Flex>>(
                &args,
                &train,
                &validation,
                &test,
                checkpoint,
                FlexDevice,
            ),
        };
    }

    match args.backend {
        BackendChoice::Cuda => {
            #[cfg(feature = "cuda")]
            {
                keypoint_train::train_model::<Autodiff<Cuda>>(
                    &args,
                    train,
                    validation,
                    test,
                    CudaDevice::default(),
                )
            }
            #[cfg(not(feature = "cuda"))]
            {
                bail!(
                    "CUDA support is not compiled; rerun Cargo with `--features cuda` before `-- --backend cuda`"
                )
            }
        }
        BackendChoice::Wgpu => keypoint_train::train_model::<Autodiff<Wgpu>>(
            &args,
            train,
            validation,
            test,
            WgpuDevice::default(),
        ),
        BackendChoice::Flex => keypoint_train::train_model::<Autodiff<Flex>>(
            &args, train, validation, test, FlexDevice,
        ),
    }
}

/// Moves every checkpoint/report that could be mistaken for the current run
/// out of the live artifact path before optimizer work starts. This includes
/// candidates, because a failed first epoch must never reload an older one.
fn archive_existing_run_artifacts(artifacts: &Path) -> Result<()> {
    let prior_run = live_run_artifacts(artifacts);
    if prior_run.iter().all(|path| !path.exists()) {
        return Ok(());
    }
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .context("system clock predates Unix epoch")?
        .as_nanos();
    let archive = artifacts
        .join("previous-promoted")
        .join(format!("run-{nonce}"));
    fs::create_dir_all(&archive)?;
    for source in prior_run.into_iter().filter(|path| path.exists()) {
        let name = source
            .file_name()
            .context("promoted artifact has no filename")?;
        fs::rename(&source, archive.join(name)).with_context(|| {
            format!(
                "archive previous run artifact {} before training",
                source.display()
            )
        })?;
    }
    Ok(())
}

fn live_run_artifacts(artifacts: &Path) -> [PathBuf; 11] {
    [
        artifacts.join("model.mpk"),
        artifacts.join("model.json"),
        artifacts.join("candidate-best.mpk"),
        artifacts.join("candidate-presence.mpk"),
        artifacts.join("candidate-geometry.mpk"),
        artifacts.join("model-last.mpk"),
        artifacts.join("final-metrics.json"),
        artifacts.join("checkpoint-evaluation.json"),
        artifacts.join("metrics.jsonl"),
        artifacts.join("refinement-source-metrics.json"),
        artifacts.join("training-config.json"),
    ]
}

/// Refinement is intentionally a new experiment, not an in-place mutation of
/// the safe source run. This also prevents the normal archive behavior from
/// moving the checkpoint immediately before it is loaded.
fn ensure_geometry_refinement_artifact_safety(args: &Args) -> Result<()> {
    let Some(checkpoint) = &args.geometry_refine_from else {
        return Ok(());
    };
    let checkpoint = checkpoint
        .canonicalize()
        .with_context(|| format!("resolve refinement checkpoint {}", checkpoint.display()))?;
    let artifacts = args
        .artifacts
        .canonicalize()
        .with_context(|| format!("resolve refinement artifacts {}", args.artifacts.display()))?;
    let checkpoint_parent = checkpoint
        .parent()
        .context("refinement checkpoint has no parent directory")?;
    if artifacts == checkpoint_parent || checkpoint.starts_with(&artifacts) {
        bail!(
            "--artifacts must be separate from the --geometry-refine-from checkpoint directory; source={} output={}",
            checkpoint.display(),
            artifacts.display(),
        );
    }
    Ok(())
}

fn require_fresh_refinement_artifacts(artifacts: &Path) -> Result<()> {
    let existing = live_run_artifacts(artifacts)
        .into_iter()
        .filter(|path| path.exists())
        .collect::<Vec<_>>();
    if !existing.is_empty() {
        bail!(
            "geometry refinement requires a fresh --artifacts directory; found existing run artifact(s): {}",
            existing
                .iter()
                .map(|path| path.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        );
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)
        .with_context(|| format!("open checkpoint for hashing {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("hash checkpoint {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("sha256:{:x}", hasher.finalize()))
}

fn validate_and_normalize_args(args: &mut Args) -> Result<()> {
    #[cfg(not(feature = "cuda"))]
    if matches!(args.backend, BackendChoice::Cuda) {
        bail!(
            "CUDA support is not compiled; rerun Cargo with `--features cuda` before `-- --backend cuda`"
        );
    }
    if args.epochs == 0
        || args.patience == 0
        || args.batch_size == 0
        || args.real_positive_repeat == 0
    {
        bail!("epochs, patience, batch size, and real-positive repeat must be greater than zero");
    }
    if args.evaluation_batch_size == Some(0) {
        bail!("--evaluation-batch-size must be greater than zero");
    }
    if args.limit_per_split == Some(0) {
        bail!("--limit-per-split must be greater than zero");
    }
    if args.overfit && args.limit_per_split.is_some() {
        bail!("--overfit selects an exact 32+32 set and cannot be combined with --limit-per-split");
    }
    match (
        args.geometry_refine_from.as_ref(),
        args.geometry_refine_source_epoch,
    ) {
        (Some(checkpoint), Some(source_epoch)) => {
            if args.evaluate_checkpoint.is_some() {
                bail!("--geometry-refine-from cannot be combined with --evaluate-checkpoint");
            }
            if args.overfit {
                bail!("--geometry-refine-from cannot be combined with --overfit");
            }
            if !checkpoint.is_file() {
                bail!(
                    "--geometry-refine-from must name an existing checkpoint file: {}",
                    checkpoint.display()
                );
            }
            if source_epoch == 0 {
                bail!("--geometry-refine-source-epoch must be greater than zero");
            }
            if source_epoch >= args.epochs {
                bail!(
                    "--geometry-refine-source-epoch ({source_epoch}) must be less than the final logical --epochs ({})",
                    args.epochs
                );
            }
            // These are invariants of the geometry-only path, not optional
            // hints. Persist their effective values truthfully in the config.
            args.freeze_backbone_batch_norm = true;
            args.detach_geometry_backbone = true;
        }
        (Some(_), None) => {
            bail!("--geometry-refine-from requires --geometry-refine-source-epoch")
        }
        (None, Some(_)) => {
            bail!("--geometry-refine-source-epoch requires --geometry-refine-from")
        }
        (None, None) => {}
    }
    if args.overfit && args.backbone_freeze_epochs != 0 {
        bail!(
            "--overfit cannot be combined with --backbone-freeze-epochs; the memorisation gate trains the complete model"
        );
    }
    if !args.overfit
        && args.evaluate_checkpoint.is_none()
        && args.backbone_freeze_epochs >= args.epochs
    {
        bail!("--backbone-freeze-epochs must be less than --epochs for normal training");
    }
    if args.head_learning_rate <= 0.0 || args.backbone_learning_rate <= 0.0 {
        bail!("learning rates must be greater than zero");
    }
    if !(0.0..1.0).contains(&args.warmup_fraction) {
        bail!("--warmup-fraction must lie between zero and one");
    }
    for (name, path) in [
        ("--synthetic", &args.synthetic),
        ("--negatives", &args.negatives),
        ("--real-positives", &args.real_positives),
    ] {
        if is_within_repository_samples(path)? {
            bail!(
                "samples/ is reserved for untouched qualitative evaluation and cannot be used as {name}"
            );
        }
    }

    if args.overfit {
        args.disable_augmentation = true;
        args.patience = args.epochs.saturating_add(1);
        args.presence_freeze_after_safe_epochs = 0;
    }
    Ok(())
}

impl Args {
    fn evaluation_batch_size(&self) -> usize {
        self.evaluation_batch_size.unwrap_or(self.batch_size)
    }

    fn first_training_epoch(&self) -> usize {
        self.geometry_refine_source_epoch
            .map_or(1, |epoch| epoch.saturating_add(1))
    }
}

fn is_within_repository_samples(path: &Path) -> Result<bool> {
    let repository_samples = std::env::current_dir()?.join("samples");
    let candidate = if path.is_absolute() {
        path.to_owned()
    } else {
        std::env::current_dir()?.join(path)
    };
    let candidate = candidate.canonicalize().unwrap_or(candidate);
    let repository_samples = repository_samples
        .canonicalize()
        .unwrap_or(repository_samples);
    Ok(candidate.starts_with(repository_samples))
}

fn split_samples(samples: &[TrainingSample], split: Split) -> Vec<TrainingSample> {
    samples
        .iter()
        .filter(|sample| sample.split == split)
        .cloned()
        .collect()
}

const OVERFIT_SAMPLES_PER_CLASS: usize = 32;
type TrainingSplits = (
    Vec<TrainingSample>,
    Vec<TrainingSample>,
    Vec<TrainingSample>,
);

/// Select the exact balanced set used to verify that the sparse training path can memorise.
fn select_overfit_samples(samples: &[TrainingSample]) -> Result<Vec<TrainingSample>> {
    let targets = samples
        .iter()
        .filter(|sample| sample.origin == SampleOrigin::SyntheticTarget)
        .take(OVERFIT_SAMPLES_PER_CLASS)
        .cloned()
        .collect::<Vec<_>>();
    let negatives = samples
        .iter()
        .filter(|sample| sample.origin == SampleOrigin::SyntheticNegative)
        .take(OVERFIT_SAMPLES_PER_CLASS)
        .cloned()
        .collect::<Vec<_>>();
    if targets.len() != OVERFIT_SAMPLES_PER_CLASS || negatives.len() != OVERFIT_SAMPLES_PER_CLASS {
        bail!(
            "--overfit requires at least {OVERFIT_SAMPLES_PER_CLASS} SyntheticTarget and {OVERFIT_SAMPLES_PER_CLASS} SyntheticNegative samples in the complete synthetic corpus; found {} and {}",
            targets.len(),
            negatives.len()
        );
    }

    let mut selected = Vec::with_capacity(OVERFIT_SAMPLES_PER_CLASS * 2);
    selected.extend(targets);
    selected.extend(negatives);
    Ok(selected)
}

fn overfit_splits(samples: &[TrainingSample]) -> Result<TrainingSplits> {
    let mut selected = select_overfit_samples(samples)?;
    for sample in &mut selected {
        sample.split = Split::Train;
    }
    Ok((selected.clone(), selected.clone(), selected))
}

fn limit_stratified(samples: Vec<TrainingSample>, limit: usize) -> Vec<TrainingSample> {
    if samples.len() <= limit {
        return samples;
    }
    let mut selected = Vec::with_capacity(limit);
    let positive_limit = limit.div_ceil(2);
    let negative_limit = limit - positive_limit;
    extend_source_balanced(
        &mut selected,
        &samples,
        &[SampleOrigin::SyntheticTarget, SampleOrigin::RealPositive],
        positive_limit,
    );
    extend_source_balanced(
        &mut selected,
        &samples,
        &[SampleOrigin::SyntheticNegative, SampleOrigin::RealNegative],
        negative_limit,
    );
    selected.truncate(limit);
    selected
}

fn extend_source_balanced(
    selected: &mut Vec<TrainingSample>,
    samples: &[TrainingSample],
    origins: &[SampleOrigin],
    limit: usize,
) {
    let buckets = origins
        .iter()
        .map(|origin| {
            samples
                .iter()
                .filter(|sample| sample.origin == *origin)
                .cloned()
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let mut offsets = vec![0usize; buckets.len()];
    let initial_len = selected.len();
    while selected.len() - initial_len < limit {
        let mut added = false;
        for (bucket, offset) in buckets.iter().zip(&mut offsets) {
            if selected.len() - initial_len == limit {
                break;
            }
            if let Some(sample) = bucket.get(*offset) {
                selected.push(sample.clone());
                *offset += 1;
                added = true;
            }
        }
        if !added {
            break;
        }
    }
}

fn print_split_summary(name: &str, samples: &[TrainingSample]) {
    let count = |origin| {
        samples
            .iter()
            .filter(|sample| sample.origin == origin)
            .count()
    };
    println!(
        "{name}: {} samples (synthetic targets={}, synthetic negatives={}, real positives={}, real negatives={})",
        samples.len(),
        count(SampleOrigin::SyntheticTarget),
        count(SampleOrigin::SyntheticNegative),
        count(SampleOrigin::RealPositive),
        count(SampleOrigin::RealNegative),
    );
}

fn real_positive_building_count(samples: &[TrainingSample]) -> Result<usize> {
    let mut buildings = BTreeSet::new();
    for sample in samples
        .iter()
        .filter(|sample| sample.origin == SampleOrigin::RealPositive)
    {
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
        buildings.insert(building);
    }
    Ok(buildings.len())
}

fn load_all_samples(args: &Args) -> Result<Vec<TrainingSample>> {
    let mut samples = load_synthetic(&args.synthetic)?;
    samples.extend(load_negatives(&args.negatives)?);
    samples.extend(load_real_positives(
        &args.real_positives,
        args.real_positive_repeat,
    )?);
    Ok(samples)
}

fn load_synthetic(dataset: &Path) -> Result<Vec<TrainingSample>> {
    let mut partials = BTreeMap::<String, PartialSyntheticSample>::new();
    let mut shards = fs::read_dir(dataset)
        .with_context(|| format!("read synthetic dataset {}", dataset.display()))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    shards.retain(|path| path.extension().is_some_and(|extension| extension == "tar"));
    shards.sort();
    if shards.is_empty() {
        bail!("{} contains no tar shards", dataset.display());
    }
    for shard in shards {
        let file = fs::File::open(&shard)?;
        let mut archive = tar::Archive::new(file);
        for entry in archive.entries()? {
            let mut entry = entry?;
            let path = entry.path()?.to_string_lossy().into_owned();
            if let Some(key) = path.strip_suffix(".rgb.jpg") {
                let mut bytes = Vec::new();
                entry.read_to_end(&mut bytes)?;
                let partial = partials.entry(key.to_owned()).or_default();
                if partial.image.is_some() {
                    bail!(
                        "duplicate synthetic image member for key {key} (latest occurrence in {})",
                        shard.display()
                    );
                }
                partial.image = Some(bytes.into());
            } else if let Some(key) = path.strip_suffix(".labels.json") {
                let mut bytes = Vec::new();
                entry.read_to_end(&mut bytes)?;
                let partial = partials.entry(key.to_owned()).or_default();
                if partial.record.is_some() {
                    bail!(
                        "duplicate synthetic label member for key {key} (latest occurrence in {})",
                        shard.display()
                    );
                }
                partial.record = Some(
                    serde_json::from_slice(&bytes)
                        .with_context(|| format!("decode synthetic labels for key {key}"))?,
                );
            }
        }
    }

    let mut samples = Vec::new();
    for (key, partial) in partials {
        let (image, record) = match (partial.image, partial.record) {
            (Some(image), Some(record)) => (image, record),
            (None, _) => {
                bail!("incomplete synthetic sample {key}: missing .rgb.jpg archive member")
            }
            (_, None) => {
                bail!("incomplete synthetic sample {key}: missing .labels.json archive member")
            }
        };
        let (presence, geometry, origin) = match record.locator.target_kind {
            TargetKind::Target if record.locator.visible_fraction >= 0.05 => (
                1,
                Some(Arc::new(record.clone())),
                SampleOrigin::SyntheticTarget,
            ),
            TargetKind::Target => continue,
            TargetKind::Negative if record.roof.is_none() => {
                (0, None, SampleOrigin::SyntheticNegative)
            }
            TargetKind::NearMiss => {
                bail!(
                    "synthetic sample {key} is labelled NearMiss; modified/former Pizza Hut roofs must be migrated to positive Target labels and are never training negatives"
                )
            }
            TargetKind::Negative => {
                bail!(
                    "synthetic sample {key} labels rendered target geometry as {:?}; regenerate with genuine ordinary-building geometry",
                    record.locator.target_kind
                )
            }
        };
        samples.push(TrainingSample {
            key,
            split: split_from_synthetic(record.split),
            image: ImageSource::Encoded(image),
            presence,
            geometry,
            origin,
            physical_building_id: None,
        });
    }
    Ok(samples)
}

fn split_from_synthetic(split: DatasetSplit) -> Split {
    match split {
        DatasetSplit::Train => Split::Train,
        DatasetSplit::Validation => Split::Validation,
        DatasetSplit::Test => Split::Test,
    }
}

fn load_negatives(dataset: &Path) -> Result<Vec<TrainingSample>> {
    let manifest_path = dataset.join("manifest.json");
    let manifest: NegativeManifest = serde_json::from_slice(&fs::read(&manifest_path)?)?;
    if manifest.schema_version != "roof-negative-dataset/v1" {
        bail!(
            "unsupported Open Images negative manifest schema {:?}",
            manifest.schema_version
        );
    }
    manifest
        .records
        .into_iter()
        .try_fold(Vec::new(), |mut samples, record| {
            if !review_status_is_accepted(
                "Open Images negative",
                &record.review_status,
                "visually_verified",
            )? {
                return Ok(samples);
            }
            let path = dataset.join(record.relative_path);
            if !path.is_file() {
                bail!("negative image is missing: {}", path.display());
            }
            samples.push(TrainingSample {
                key: format!("open-images-{}", record.image_id),
                split: parse_split(&record.split)?,
                image: ImageSource::Path(path),
                presence: 0,
                geometry: None,
                origin: SampleOrigin::RealNegative,
                physical_building_id: None,
            });
            Ok(samples)
        })
}

fn load_real_positives(dataset: &Path, repeat: usize) -> Result<Vec<TrainingSample>> {
    let manifest_path = dataset.join("manifest.json");
    let manifest: PositiveManifest = serde_json::from_slice(&fs::read(&manifest_path)?)?;
    if manifest.schema_version != "roof-positive-dataset/v2" {
        bail!(
            "unsupported Wikimedia positive manifest schema {:?}",
            manifest.schema_version
        );
    }
    let mut records = Vec::new();
    let mut building_splits = BTreeMap::<String, Split>::new();
    for record in manifest.records {
        if !review_status_is_accepted(
            "Wikimedia positive",
            &record.review_status,
            "category_screened",
        )? {
            continue;
        }
        if record.physical_building_id.trim().is_empty() {
            bail!(
                "Wikimedia positive page {} has no physical_building_id",
                record.page_id
            );
        }
        let split = parse_split(&record.split)?;
        if let Some(previous) = building_splits.insert(record.physical_building_id.clone(), split)
            && previous != split
        {
            bail!(
                "Wikimedia physical building {:?} occurs in more than one split",
                record.physical_building_id
            );
        }
        let path = dataset.join(&record.relative_path);
        if !path.is_file() {
            bail!("positive image is missing: {}", path.display());
        }
        if !characteristic_roof_is_eligible(&record)? {
            continue;
        }
        records.push(LoadedPositiveRecord {
            page_id: record.page_id,
            physical_building_id: record.physical_building_id,
            split,
            path,
        });
    }

    let mut samples = Vec::new();
    for split in [Split::Train, Split::Validation, Split::Test] {
        let ordered = building_balanced_positive_records(&records, split);
        let copies = if split == Split::Train { repeat } else { 1 };
        for augmentation in 0..copies {
            for record in &ordered {
                samples.push(TrainingSample {
                    key: format!("wikimedia-positive-{}-{augmentation:02}", record.page_id),
                    split,
                    image: ImageSource::Path(record.path.clone()),
                    presence: 1,
                    geometry: None,
                    origin: SampleOrigin::RealPositive,
                    physical_building_id: Some(Arc::from(record.physical_building_id.as_str())),
                });
            }
        }
    }
    Ok(samples)
}

fn characteristic_roof_is_eligible(record: &PositiveManifestRecord) -> Result<bool> {
    match record.characteristic_roof_visibility {
        CharacteristicRoofVisibility::Recognizable => Ok(true),
        CharacteristicRoofVisibility::NotRecognizable
        | CharacteristicRoofVisibility::Unreviewed => {
            let has_reason = record
                .visibility_review_reason
                .as_deref()
                .is_some_and(|reason| !reason.trim().is_empty());
            if !has_reason {
                bail!(
                    "Wikimedia positive page {} is {:?} but has no visibility_review_reason",
                    record.page_id,
                    record.characteristic_roof_visibility,
                );
            }
            Ok(false)
        }
    }
}

/// Puts one image from every eligible building before additional views of any building.
/// The caller then emits one full pass over these unique records before augmentation aliases,
/// so `--limit-per-split` cannot spend its real-positive quota on repeats of one early record.
fn building_balanced_positive_records(
    records: &[LoadedPositiveRecord],
    split: Split,
) -> Vec<&LoadedPositiveRecord> {
    let mut groups = BTreeMap::<&str, Vec<&LoadedPositiveRecord>>::new();
    for record in records.iter().filter(|record| record.split == split) {
        groups
            .entry(&record.physical_building_id)
            .or_default()
            .push(record);
    }
    let maximum_views = groups.values().map(Vec::len).max().unwrap_or(0);
    let mut ordered = Vec::new();
    for view_index in 0..maximum_views {
        for group in groups.values() {
            if let Some(record) = group.get(view_index) {
                ordered.push(*record);
            }
        }
    }
    ordered
}

/// Accept only the status emitted by the current manifest schema. `rejected` is the sole
/// explicit exclusion state. Unknown states are errors, preventing a future importer from
/// silently adding unreviewed images or silently dropping a newly accepted status.
fn review_status_is_accepted(dataset: &str, status: &str, accepted: &str) -> Result<bool> {
    match status {
        value if value == accepted => Ok(true),
        "rejected" => Ok(false),
        other => bail!(
            "unknown {dataset} review_status {other:?}; expected {accepted:?} or explicit \"rejected\""
        ),
    }
}

fn parse_split(split: &str) -> Result<Split> {
    match split {
        "train" => Ok(Split::Train),
        "validation" => Ok(Split::Validation),
        "test" => Ok(Split::Test),
        other => bail!("unknown dataset split {other}"),
    }
}

fn append_json_line(path: &Path, value: &impl Serialize) -> Result<()> {
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(&mut file, value)?;
    file.write_all(b"\n")?;
    Ok(())
}

fn sample_hash(key: &str, epoch: usize, salt: u64) -> u64 {
    key.bytes().fold(epoch as u64 ^ salt, |state, byte| {
        state.rotate_left(7) ^ u64::from(byte)
    })
}

#[cfg(test)]
mod tests {
    use std::{
        sync::atomic::{AtomicU64, Ordering},
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::*;

    static TEMP_ID: AtomicU64 = AtomicU64::new(0);

    struct TempDataset(PathBuf);

    impl TempDataset {
        fn new() -> Self {
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .expect("system clock")
                .as_nanos();
            let path = std::env::temp_dir().join(format!(
                "roof-train-test-{}-{nonce}-{}",
                std::process::id(),
                TEMP_ID.fetch_add(1, Ordering::Relaxed)
            ));
            fs::create_dir(&path).expect("create temporary dataset");
            Self(path)
        }

        fn append_tar(&self, shard: &str, members: &[(&str, &[u8])]) {
            let file = fs::File::create(self.0.join(shard)).expect("create shard");
            let mut archive = tar::Builder::new(file);
            for (path, bytes) in members {
                let mut header = tar::Header::new_gnu();
                header.set_size(bytes.len() as u64);
                header.set_mode(0o644);
                header.set_cksum();
                archive
                    .append_data(&mut header, path, *bytes)
                    .expect("append archive member");
            }
            archive.finish().expect("finish shard");
        }
    }

    impl Drop for TempDataset {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn synthetic_sample(index: usize, origin: SampleOrigin) -> TrainingSample {
        TrainingSample {
            key: format!("sample-{index:03}"),
            split: Split::Train,
            image: ImageSource::Encoded(Arc::<[u8]>::from([])),
            presence: i32::from(origin == SampleOrigin::SyntheticTarget),
            geometry: None,
            origin,
            physical_building_id: (origin == SampleOrigin::RealPositive)
                .then(|| Arc::<str>::from(format!("test-building-{index:03}"))),
        }
    }

    #[test]
    fn repository_samples_are_not_a_training_default() {
        let args = Args::parse_from(["roof-train"]);
        assert_eq!(
            args.real_positives,
            PathBuf::from("datasets/wikimedia-positives")
        );
        assert_ne!(args.real_positives, PathBuf::from("samples"));
        assert_eq!(args.real_positive_repeat, 1);
        assert_eq!(args.backbone_freeze_epochs, 0);
        assert!(!args.freeze_backbone_batch_norm);
        assert!(!args.detach_geometry_backbone);
        assert_eq!(args.presence_freeze_after_safe_epochs, 2);
        let enabled = Args::parse_from([
            "roof-train",
            "--backbone-freeze-epochs",
            "5",
            "--freeze-backbone-batch-norm",
            "--detach-geometry-backbone",
            "--presence-freeze-after-safe-epochs",
            "0",
        ]);
        assert_eq!(enabled.backbone_freeze_epochs, 5);
        assert!(enabled.freeze_backbone_batch_norm);
        assert!(enabled.detach_geometry_backbone);
        assert_eq!(enabled.presence_freeze_after_safe_epochs, 0);
    }

    #[test]
    fn geometry_refinement_contract_starts_at_the_next_logical_epoch() {
        let source = TempDataset::new();
        let output = TempDataset::new();
        let checkpoint = source.0.join("candidate-e06.mpk");
        fs::write(&checkpoint, b"safe epoch six checkpoint").unwrap();
        let mut args = Args::parse_from([
            "roof-train",
            "--geometry-refine-from",
            checkpoint.to_str().unwrap(),
            "--geometry-refine-source-epoch",
            "6",
            "--epochs",
            "40",
            "--artifacts",
            output.0.to_str().unwrap(),
        ]);

        validate_and_normalize_args(&mut args).unwrap();

        assert_eq!(
            args.geometry_refine_from.as_deref(),
            Some(checkpoint.as_path())
        );
        assert_eq!(args.geometry_refine_source_epoch, Some(6));
        assert_eq!(args.first_training_epoch(), 7);
        assert!(args.freeze_backbone_batch_norm);
        assert!(args.detach_geometry_backbone);
        ensure_geometry_refinement_artifact_safety(&args).unwrap();
    }

    #[test]
    fn geometry_refinement_requires_complete_non_overlapping_provenance() {
        let source = TempDataset::new();
        let checkpoint = source.0.join("candidate-e06.mpk");
        fs::write(&checkpoint, b"checkpoint").unwrap();

        let mut missing_epoch = Args::parse_from([
            "roof-train",
            "--geometry-refine-from",
            checkpoint.to_str().unwrap(),
        ]);
        let error = validate_and_normalize_args(&mut missing_epoch)
            .expect_err("the logical source epoch is mandatory")
            .to_string();
        assert!(
            error.contains("requires --geometry-refine-source-epoch"),
            "{error}"
        );

        let mut no_remaining_epochs = Args::parse_from([
            "roof-train",
            "--geometry-refine-from",
            checkpoint.to_str().unwrap(),
            "--geometry-refine-source-epoch",
            "6",
            "--epochs",
            "6",
        ]);
        let error = validate_and_normalize_args(&mut no_remaining_epochs)
            .expect_err("the final epoch must follow the source epoch")
            .to_string();
        assert!(
            error.contains("must be less than the final logical --epochs"),
            "{error}"
        );

        let mut same_directory = Args::parse_from([
            "roof-train",
            "--geometry-refine-from",
            checkpoint.to_str().unwrap(),
            "--geometry-refine-source-epoch",
            "6",
            "--artifacts",
            source.0.to_str().unwrap(),
        ]);
        validate_and_normalize_args(&mut same_directory).unwrap();
        let error = ensure_geometry_refinement_artifact_safety(&same_directory)
            .expect_err("refinement must not overwrite its source run")
            .to_string();
        assert!(error.contains("must be separate"), "{error}");
    }

    #[test]
    fn geometry_refinement_refuses_existing_live_run_artifacts() {
        let artifacts = TempDataset::new();
        fs::write(artifacts.0.join("training-config.json"), b"old run").unwrap();
        let error = require_fresh_refinement_artifacts(&artifacts.0)
            .expect_err("a refinement output cannot silently archive another run")
            .to_string();
        assert!(
            error.contains("requires a fresh --artifacts directory"),
            "{error}"
        );
        assert!(error.contains("training-config.json"), "{error}");
    }

    #[test]
    fn checkpoint_sha256_is_stable_provenance() {
        let source = TempDataset::new();
        let checkpoint = source.0.join("checkpoint.mpk");
        fs::write(&checkpoint, b"abc").unwrap();
        assert_eq!(
            sha256_file(&checkpoint).unwrap(),
            "sha256:ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn evaluation_batch_defaults_to_training_batch_and_accepts_an_override() {
        let default = Args::parse_from(["roof-train", "--batch-size", "24"]);
        assert_eq!(default.evaluation_batch_size(), 24);

        let overridden = Args::parse_from([
            "roof-train",
            "--batch-size",
            "24",
            "--evaluation-batch-size",
            "64",
        ]);
        assert_eq!(overridden.evaluation_batch_size(), 64);
    }

    #[cfg(not(feature = "cuda"))]
    #[test]
    fn cuda_backend_explains_the_required_cargo_feature() {
        let mut args = Args::parse_from(["roof-train", "--backend", "cuda"]);
        let error = validate_and_normalize_args(&mut args)
            .expect_err("CUDA must be explicitly compiled")
            .to_string();
        assert!(error.contains("--features cuda"), "{error}");
    }

    #[test]
    fn repository_samples_cannot_be_opted_into_training() {
        let mut args = Args::parse_from(["roof-train", "--real-positives", "samples"]);
        let error = validate_and_normalize_args(&mut args)
            .expect_err("samples must remain evaluation-only")
            .to_string();
        assert!(
            error.contains("untouched qualitative evaluation"),
            "{error}"
        );
    }

    #[test]
    fn repository_sample_descendants_cannot_enter_any_training_role() {
        for option in ["--synthetic", "--negatives", "--real-positives"] {
            let mut args =
                Args::parse_from(["roof-train", option, "samples/evaluation-subdirectory"]);
            let error = validate_and_normalize_args(&mut args)
                .expect_err("sample descendants must remain evaluation-only")
                .to_string();
            assert!(error.contains(option), "{error}");
            assert!(
                error.contains("untouched qualitative evaluation"),
                "{error}"
            );
        }
    }

    #[test]
    fn overfit_selection_is_exactly_balanced_and_deterministic() {
        let samples = (0..40)
            .map(|index| synthetic_sample(index, SampleOrigin::SyntheticTarget))
            .chain((40..80).map(|index| synthetic_sample(index, SampleOrigin::SyntheticNegative)))
            .chain((80..90).map(|index| synthetic_sample(index, SampleOrigin::RealNegative)))
            .collect::<Vec<_>>();

        let selected = select_overfit_samples(&samples).expect("select overfit set");
        assert_eq!(selected.len(), 64);
        assert_eq!(
            selected
                .iter()
                .filter(|sample| sample.origin == SampleOrigin::SyntheticTarget)
                .count(),
            32
        );
        assert_eq!(
            selected
                .iter()
                .filter(|sample| sample.origin == SampleOrigin::SyntheticNegative)
                .count(),
            32
        );
        assert_eq!(selected.first().unwrap().key, "sample-000");
        assert_eq!(selected.last().unwrap().key, "sample-071");
    }

    #[test]
    fn split_limit_preserves_every_available_training_source() {
        let origins = [
            SampleOrigin::SyntheticTarget,
            SampleOrigin::SyntheticNegative,
            SampleOrigin::RealPositive,
            SampleOrigin::RealNegative,
        ];
        let samples = origins
            .into_iter()
            .enumerate()
            .flat_map(|(origin_index, origin)| {
                (0..10).map(move |index| synthetic_sample(origin_index * 10 + index, origin))
            })
            .collect::<Vec<_>>();

        let limited = limit_stratified(samples, 8);
        assert_eq!(limited.len(), 8);
        for origin in origins {
            assert_eq!(
                limited
                    .iter()
                    .filter(|sample| sample.origin == origin)
                    .count(),
                2
            );
        }
    }

    #[test]
    fn split_limit_prioritizes_distinct_real_positive_buildings_before_repeats() {
        let dataset = TempDataset::new();
        for page_id in [11_u64, 12, 21, 31] {
            fs::write(dataset.0.join(format!("{page_id}.jpg")), b"image").unwrap();
        }
        let manifest = serde_json::json!({
            "schema_version": "roof-positive-dataset/v2",
            "records": [
                {
                    "page_id": 11,
                    "physical_building_id": "building-a",
                    "split": "train",
                    "relative_path": "11.jpg",
                    "review_status": "category_screened",
                    "characteristic_roof_visibility": "recognizable"
                },
                {
                    "page_id": 12,
                    "physical_building_id": "building-a",
                    "split": "train",
                    "relative_path": "12.jpg",
                    "review_status": "category_screened",
                    "characteristic_roof_visibility": "recognizable"
                },
                {
                    "page_id": 21,
                    "physical_building_id": "building-b",
                    "split": "train",
                    "relative_path": "21.jpg",
                    "review_status": "category_screened",
                    "characteristic_roof_visibility": "recognizable"
                },
                {
                    "page_id": 31,
                    "physical_building_id": "building-c",
                    "split": "train",
                    "relative_path": "31.jpg",
                    "review_status": "category_screened",
                    "characteristic_roof_visibility": "not_recognizable",
                    "visibility_review_reason": "roof is outside the crop"
                }
            ]
        });
        fs::write(
            dataset.0.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();

        let real_positives = load_real_positives(&dataset.0, 8).unwrap();
        assert_eq!(real_positives.len(), 24);
        assert_eq!(real_positives[0].key, "wikimedia-positive-11-00");
        assert_eq!(real_positives[1].key, "wikimedia-positive-21-00");
        assert_eq!(real_positives[2].key, "wikimedia-positive-12-00");
        assert_eq!(
            real_positives[0].physical_building_id.as_deref(),
            Some("building-a")
        );
        assert_eq!(
            real_positives[1].physical_building_id.as_deref(),
            Some("building-b")
        );
        assert_eq!(
            real_positives[2].physical_building_id.as_deref(),
            Some("building-a")
        );
        assert!(
            real_positives
                .iter()
                .all(|sample| !sample.key.contains("-31-"))
        );

        let samples = (100..110)
            .map(|index| synthetic_sample(index, SampleOrigin::SyntheticTarget))
            .chain((110..120).map(|index| synthetic_sample(index, SampleOrigin::SyntheticNegative)))
            .chain((120..130).map(|index| synthetic_sample(index, SampleOrigin::RealNegative)))
            .chain(real_positives)
            .collect::<Vec<_>>();
        let limited = limit_stratified(samples, 8);
        let selected_real = limited
            .iter()
            .filter(|sample| sample.origin == SampleOrigin::RealPositive)
            .map(|sample| sample.key.as_str())
            .collect::<Vec<_>>();
        assert_eq!(
            selected_real,
            ["wikimedia-positive-11-00", "wikimedia-positive-21-00"]
        );
    }

    #[test]
    fn provenance_only_positive_still_participates_in_building_split_validation() {
        let dataset = TempDataset::new();
        fs::write(dataset.0.join("41.jpg"), b"image").unwrap();
        fs::write(dataset.0.join("42.jpg"), b"image").unwrap();
        let manifest = serde_json::json!({
            "schema_version": "roof-positive-dataset/v2",
            "records": [
                {
                    "page_id": 41,
                    "physical_building_id": "same-building",
                    "split": "train",
                    "relative_path": "41.jpg",
                    "review_status": "category_screened",
                    "characteristic_roof_visibility": "recognizable"
                },
                {
                    "page_id": 42,
                    "physical_building_id": "same-building",
                    "split": "validation",
                    "relative_path": "42.jpg",
                    "review_status": "category_screened",
                    "characteristic_roof_visibility": "not_recognizable",
                    "visibility_review_reason": "roof is outside the crop"
                }
            ]
        });
        fs::write(
            dataset.0.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();

        let error = load_real_positives(&dataset.0, 8)
            .err()
            .expect("all views of a historical site must share one split")
            .to_string();
        assert!(error.contains("more than one split"), "{error}");
    }

    #[test]
    fn excluded_visibility_states_require_an_audit_reason() {
        let record = PositiveManifestRecord {
            page_id: 99,
            physical_building_id: "fixture".to_owned(),
            split: "train".to_owned(),
            relative_path: "fixture.jpg".to_owned(),
            review_status: "category_screened".to_owned(),
            characteristic_roof_visibility: CharacteristicRoofVisibility::Unreviewed,
            visibility_review_reason: None,
        };
        let error = characteristic_roof_is_eligible(&record)
            .expect_err("excluded records need review evidence")
            .to_string();
        assert!(error.contains("visibility_review_reason"), "{error}");
    }

    #[test]
    fn overfit_gate_uses_the_whole_corpus_and_reuses_one_set() {
        let mut samples = (0..32)
            .map(|index| synthetic_sample(index, SampleOrigin::SyntheticTarget))
            .chain((32..64).map(|index| synthetic_sample(index, SampleOrigin::SyntheticNegative)))
            .collect::<Vec<_>>();
        for (index, sample) in samples.iter_mut().enumerate() {
            sample.split = match index % 3 {
                0 => Split::Train,
                1 => Split::Validation,
                _ => Split::Test,
            };
        }

        let (train, validation, test) = overfit_splits(&samples).expect("build overfit splits");
        assert_eq!(train.len(), 64);
        assert_eq!(
            train.iter().map(|sample| &sample.key).collect::<Vec<_>>(),
            validation
                .iter()
                .map(|sample| &sample.key)
                .collect::<Vec<_>>()
        );
        assert_eq!(
            train.iter().map(|sample| &sample.key).collect::<Vec<_>>(),
            test.iter().map(|sample| &sample.key).collect::<Vec<_>>()
        );
        assert!(train.iter().all(|sample| sample.split == Split::Train));
    }

    #[test]
    fn overfit_selection_reports_each_missing_class() {
        let samples = (0..31)
            .map(|index| synthetic_sample(index, SampleOrigin::SyntheticTarget))
            .chain((31..62).map(|index| synthetic_sample(index, SampleOrigin::SyntheticNegative)))
            .collect::<Vec<_>>();
        let error = select_overfit_samples(&samples)
            .err()
            .expect("undersized overfit set must fail")
            .to_string();
        assert!(error.contains("found 31 and 31"), "{error}");
    }

    #[test]
    fn synthetic_loader_rejects_an_incomplete_image_label_pair() {
        let dataset = TempDataset::new();
        dataset.append_tar("shard-00000.tar", &[("building.rgb.jpg", b"jpeg")]);

        let error = load_synthetic(&dataset.0)
            .err()
            .expect("incomplete pair must fail")
            .to_string();
        assert!(
            error.contains("incomplete synthetic sample building"),
            "{error}"
        );
        assert!(error.contains(".labels.json"), "{error}");
    }

    #[test]
    fn synthetic_loader_rejects_duplicate_keys_across_shards() {
        let dataset = TempDataset::new();
        dataset.append_tar("shard-00000.tar", &[("building.rgb.jpg", b"first")]);
        dataset.append_tar("shard-00001.tar", &[("building.rgb.jpg", b"second")]);

        let error = load_synthetic(&dataset.0)
            .err()
            .expect("duplicate key must fail")
            .to_string();
        assert!(
            error.contains("duplicate synthetic image member"),
            "{error}"
        );
        assert!(error.contains("building"), "{error}");
    }

    #[test]
    fn review_statuses_are_schema_specific_and_fail_closed() {
        assert!(
            review_status_is_accepted(
                "Wikimedia positive",
                "category_screened",
                "category_screened"
            )
            .unwrap()
        );
        assert!(
            review_status_is_accepted(
                "Open Images negative",
                "visually_verified",
                "visually_verified"
            )
            .unwrap()
        );
        assert!(
            review_status_is_accepted(
                "Open Images negative",
                "metadata_screened",
                "visually_verified"
            )
            .is_err()
        );
        assert!(!review_status_is_accepted("dataset", "rejected", "metadata_screened").unwrap());
        assert!(review_status_is_accepted("dataset", "pending", "metadata_screened").is_err());
    }

    #[test]
    fn overfit_mode_forces_no_augmentation_and_no_early_stop() {
        let mut args = Args::try_parse_from(["roof-train", "--overfit", "--epochs", "40"]).unwrap();
        validate_and_normalize_args(&mut args).unwrap();
        assert!(args.overfit);
        assert!(args.disable_augmentation);
        assert_eq!(args.patience, 41);
        assert_eq!(args.presence_freeze_after_safe_epochs, 0);
    }

    #[test]
    fn overfit_mode_rejects_a_split_limit() {
        let mut args =
            Args::try_parse_from(["roof-train", "--overfit", "--limit-per-split", "64"]).unwrap();
        let error = validate_and_normalize_args(&mut args)
            .expect_err("ambiguous overfit limit must fail")
            .to_string();
        assert!(error.contains("cannot be combined"), "{error}");
    }

    #[test]
    fn normal_training_requires_an_epoch_after_the_backbone_freeze() {
        let mut valid = Args::try_parse_from([
            "roof-train",
            "--epochs",
            "4",
            "--backbone-freeze-epochs",
            "3",
        ])
        .unwrap();
        validate_and_normalize_args(&mut valid).unwrap();

        for frozen_epochs in [4, 5] {
            let frozen_epochs = frozen_epochs.to_string();
            let mut invalid = Args::try_parse_from([
                "roof-train",
                "--epochs",
                "4",
                "--backbone-freeze-epochs",
                &frozen_epochs,
            ])
            .unwrap();
            let error = validate_and_normalize_args(&mut invalid)
                .expect_err("the backbone must unfreeze during normal training")
                .to_string();
            assert!(error.contains("must be less than --epochs"), "{error}");
        }
    }

    #[test]
    fn overfit_mode_rejects_a_staged_backbone_freeze() {
        let mut args =
            Args::try_parse_from(["roof-train", "--overfit", "--backbone-freeze-epochs", "1"])
                .unwrap();
        let error = validate_and_normalize_args(&mut args)
            .expect_err("the memorisation gate must train the complete model")
            .to_string();
        assert!(error.contains("memorisation gate"), "{error}");
    }

    #[test]
    fn previous_promoted_checkpoint_is_archived_before_training() {
        let artifacts = TempDataset::new();
        fs::write(artifacts.0.join("model.mpk"), b"old checkpoint").unwrap();
        fs::write(artifacts.0.join("model.json"), b"old manifest").unwrap();

        fs::write(artifacts.0.join("candidate-best.mpk"), b"old candidate").unwrap();
        fs::write(
            artifacts.0.join("candidate-presence.mpk"),
            b"old presence candidate",
        )
        .unwrap();
        fs::write(
            artifacts.0.join("candidate-geometry.mpk"),
            b"old geometry candidate",
        )
        .unwrap();

        archive_existing_run_artifacts(&artifacts.0).unwrap();

        assert!(!artifacts.0.join("model.mpk").exists());
        assert!(!artifacts.0.join("model.json").exists());
        assert!(!artifacts.0.join("candidate-best.mpk").exists());
        assert!(!artifacts.0.join("candidate-presence.mpk").exists());
        assert!(!artifacts.0.join("candidate-geometry.mpk").exists());
        let archive_root = artifacts.0.join("previous-promoted");
        let runs = fs::read_dir(&archive_root)
            .unwrap()
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        assert_eq!(runs.len(), 1);
        assert_eq!(
            fs::read(runs[0].path().join("model.mpk")).unwrap(),
            b"old checkpoint"
        );
        assert_eq!(
            fs::read(runs[0].path().join("model.json")).unwrap(),
            b"old manifest"
        );
        assert_eq!(
            fs::read(runs[0].path().join("candidate-best.mpk")).unwrap(),
            b"old candidate"
        );
        assert_eq!(
            fs::read(runs[0].path().join("candidate-presence.mpk")).unwrap(),
            b"old presence candidate"
        );
        assert_eq!(
            fs::read(runs[0].path().join("candidate-geometry.mpk")).unwrap(),
            b"old geometry candidate"
        );
    }
}
