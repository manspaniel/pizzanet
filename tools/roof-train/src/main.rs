//! Trains the Burn still-image roof observation model.

mod augmentation;
mod fit_evaluation;
mod keypoint_train;

use std::{
    collections::BTreeMap,
    fs::{self, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    process::ExitCode,
    sync::Arc,
};

use anyhow::{Context, Result, bail};
use burn::backend::{Autodiff, Flex, Wgpu, flex::FlexDevice, wgpu::WgpuDevice};
use clap::{Parser, ValueEnum};
use roof_model::{HEATMAP_SIZE, KEYPOINT_COUNT, SPATIAL_INPUT_SIZE};
use serde::{Deserialize, Serialize};
use synth_data::{DatasetSplit, FrameRecord, TargetKind};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum BackendChoice {
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
    /// Manifest-backed Wikimedia current/former Pizza Hut positive dataset.
    #[arg(long, default_value = "datasets/wikimedia-positives")]
    real_positives: PathBuf,
    /// Independently augmented copies of each training-split real positive.
    #[arg(long, default_value_t = 8)]
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
    /// AdamW learning rate for the FPN and output heads.
    #[arg(long, default_value_t = 3.0e-4)]
    head_learning_rate: f64,
    /// AdamW learning rate for the pretrained MobileNetV2 backbone.
    #[arg(long, default_value_t = 3.0e-5)]
    backbone_learning_rate: f64,
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
    split: String,
    relative_path: String,
    review_status: String,
}

#[derive(Default)]
struct PartialSyntheticSample {
    image: Option<Arc<[u8]>>,
    record: Option<FrameRecord>,
}

#[derive(Clone, Debug, Serialize)]
struct TrainingConfigRecord {
    schema_version: String,
    synthetic: String,
    negatives: String,
    real_positives: String,
    real_positive_repeat: usize,
    backend: String,
    epochs: usize,
    patience: usize,
    batch_size: usize,
    head_learning_rate: f64,
    backbone_learning_rate: f64,
    weight_decay: f64,
    warmup_fraction: f64,
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

    fs::create_dir_all(&args.artifacts)?;
    if args.evaluate_checkpoint.is_none() {
        archive_existing_run_artifacts(&args.artifacts)?;
    }
    let config = TrainingConfigRecord {
        schema_version: "roof-training/v5".to_owned(),
        synthetic: args.synthetic.display().to_string(),
        negatives: args.negatives.display().to_string(),
        real_positives: args.real_positives.display().to_string(),
        real_positive_repeat: args.real_positive_repeat,
        backend: format!("{:?}", args.backend).to_lowercase(),
        epochs: args.epochs,
        patience: args.patience,
        batch_size: args.batch_size,
        head_learning_rate: args.head_learning_rate,
        backbone_learning_rate: args.backbone_learning_rate,
        weight_decay: args.weight_decay,
        warmup_fraction: args.warmup_fraction,
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

    if let Some(checkpoint) = &args.evaluate_checkpoint {
        return match args.backend {
            BackendChoice::Wgpu => keypoint_train::evaluate_checkpoint::<Autodiff<Wgpu>>(
                &args,
                &validation,
                &test,
                checkpoint,
                WgpuDevice::default(),
            ),
            BackendChoice::Flex => keypoint_train::evaluate_checkpoint::<Autodiff<Flex>>(
                &args,
                &validation,
                &test,
                checkpoint,
                FlexDevice,
            ),
        };
    }

    match args.backend {
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
    let prior_run = [
        artifacts.join("model.mpk"),
        artifacts.join("model.json"),
        artifacts.join("candidate-best.mpk"),
        artifacts.join("model-last.mpk"),
        artifacts.join("final-metrics.json"),
        artifacts.join("checkpoint-evaluation.json"),
        artifacts.join("metrics.jsonl"),
        artifacts.join("training-config.json"),
    ];
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

fn validate_and_normalize_args(args: &mut Args) -> Result<()> {
    if args.epochs == 0 || args.patience == 0 || args.batch_size == 0 {
        bail!("epochs, patience, and batch size must be greater than zero");
    }
    if args.limit_per_split == Some(0) {
        bail!("--limit-per-split must be greater than zero");
    }
    if args.overfit && args.limit_per_split.is_some() {
        bail!("--overfit selects an exact 32+32 set and cannot be combined with --limit-per-split");
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
    }
    Ok(())
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
    for origin in [SampleOrigin::SyntheticTarget, SampleOrigin::RealPositive] {
        let remaining = positive_limit.saturating_sub(selected.len());
        selected.extend(
            samples
                .iter()
                .filter(|sample| sample.origin == origin)
                .take(remaining)
                .cloned(),
        );
    }
    let positive_count = selected.len();
    for origin in [SampleOrigin::SyntheticNegative, SampleOrigin::RealNegative] {
        let selected_negatives = selected.len() - positive_count;
        let remaining = negative_limit.saturating_sub(selected_negatives);
        selected.extend(
            samples
                .iter()
                .filter(|sample| sample.origin == origin)
                .take(remaining)
                .cloned(),
        );
    }
    selected.truncate(limit);
    selected
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
            });
            Ok(samples)
        })
}

fn load_real_positives(dataset: &Path, repeat: usize) -> Result<Vec<TrainingSample>> {
    let manifest_path = dataset.join("manifest.json");
    let manifest: PositiveManifest = serde_json::from_slice(&fs::read(&manifest_path)?)?;
    if manifest.schema_version != "roof-positive-dataset/v1" {
        bail!(
            "unsupported Wikimedia positive manifest schema {:?}",
            manifest.schema_version
        );
    }
    let mut samples = Vec::new();
    for record in manifest.records {
        if !review_status_is_accepted(
            "Wikimedia positive",
            &record.review_status,
            "category_screened",
        )? {
            continue;
        }
        let split = parse_split(&record.split)?;
        let copies = if split == Split::Train { repeat } else { 1 };
        let path = dataset.join(&record.relative_path);
        if !path.is_file() {
            bail!("positive image is missing: {}", path.display());
        }
        for augmentation in 0..copies {
            samples.push(TrainingSample {
                key: format!("wikimedia-positive-{}-{augmentation:02}", record.page_id),
                split,
                image: ImageSource::Path(path.clone()),
                presence: 1,
                geometry: None,
                origin: SampleOrigin::RealPositive,
            });
        }
    }
    Ok(samples)
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
    fn previous_promoted_checkpoint_is_archived_before_training() {
        let artifacts = TempDataset::new();
        fs::write(artifacts.0.join("model.mpk"), b"old checkpoint").unwrap();
        fs::write(artifacts.0.join("model.json"), b"old manifest").unwrap();

        fs::write(artifacts.0.join("candidate-best.mpk"), b"old candidate").unwrap();

        archive_existing_run_artifacts(&artifacts.0).unwrap();

        assert!(!artifacts.0.join("model.mpk").exists());
        assert!(!artifacts.0.join("model.json").exists());
        assert!(!artifacts.0.join("candidate-best.mpk").exists());
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
    }
}
