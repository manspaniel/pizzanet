//! Training loop for the sparse amodal-keypoint observation network.

use std::{
    collections::HashMap,
    f64::consts::PI,
    fs,
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
    time::Instant,
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
    KeypointRoofNet, KeypointRoofNetConfig, KeypointRoofOutput, LetterboxTransform,
    SPATIAL_INPUT_SIZE, prepare_rgb8_sized,
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
        resize_to_working_raster, rotate_frame_keypoints, rotate_rgb_reflect,
        synthetic_roll_radians,
    },
    fit_evaluation::{SyntheticFitMetrics, evaluate_synthetic_fit},
    sample_hash,
};

const PCK_STRICT_THRESHOLD: f32 = 0.03;
const PCK_STANDARD_THRESHOLD: f32 = 0.05;
// The 4,096 spatial classes jointly encode `in frame`, while the final class
// encodes `offscreen`. The categorical KL remains the primary objective; this
// auxiliary state term teaches that union explicitly so a broad spatial
// distribution cannot be calibrated like 4,096 unrelated negative classes.
const OFFSCREEN_STATE_LOSS_WEIGHT: f32 = 0.25;
const LOSS_SYNC_INTERVAL: usize = 10;

pub(super) fn train_model<B: AutodiffBackend>(
    args: &Args,
    train: Vec<TrainingSample>,
    validation: Vec<TrainingSample>,
    test: Vec<TrainingSample>,
    device: B::Device,
) -> Result<()> {
    B::seed(&device, args.seed);
    let fit_shape_prior = shape_prior(&train)?.mean;
    let mut model = KeypointRoofNetConfig::new()
        .init_pretrained::<B>(&device)
        .context("initialize trainable MobileNetV2 ImageNet weights")?;
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
    let total_updates = args.epochs * batches_per_epoch;
    let metrics_path = args.artifacts.join("metrics.jsonl");
    fs::write(&metrics_path, [])?;
    let mut update = 0usize;
    let mut best_score = f32::NEG_INFINITY;
    let mut epochs_without_improvement = 0usize;
    let working_image_cache = WorkingImageCache::default();

    for epoch in 1..=args.epochs {
        let epoch_samples = balanced_epoch_samples(&train, args.batch_size, args.seed, epoch);
        let epoch_batch_count = epoch_samples.len().div_ceil(args.batch_size);
        let epoch_started = Instant::now();
        let mut total_loss = 0.0;
        let mut batches = 0usize;
        let mut pending_loss: Option<Tensor<B::InnerBackend, 1>> = None;
        let mut pending_loss_count = 0usize;
        for (batch_index, chunk) in epoch_samples.chunks(args.batch_size).enumerate() {
            let batch = make_batch::<B>(
                chunk,
                &device,
                epoch,
                !args.disable_augmentation,
                Some(&working_image_cache),
            )?;
            let output = model.forward(batch.images.clone());
            let loss = observation_loss(output, &batch);
            let detached_loss = loss.clone().inner();
            pending_loss = Some(match pending_loss.take() {
                Some(accumulated) => accumulated + detached_loss,
                None => detached_loss,
            });
            pending_loss_count += 1;
            batches += 1;

            let mut gradients = loss.backward();
            let backbone_gradients =
                GradientsParams::from_params(&mut gradients, &model, &groups.backbone);
            let head_gradients =
                GradientsParams::from_params(&mut gradients, &model, &groups.heads);
            let backbone_lr = scheduled_learning_rate(
                args.backbone_learning_rate,
                update,
                total_updates,
                args.warmup_fraction,
            );
            let head_lr = scheduled_learning_rate(
                args.head_learning_rate,
                update,
                total_updates,
                args.warmup_fraction,
            );
            model = backbone_optimizer.step(backbone_lr, model, backbone_gradients);
            model = head_optimizer.step(head_lr, model, head_gradients);
            update += 1;
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
                pending_loss_count = 0;
            }
            if reports_progress {
                println!(
                    "epoch {epoch:02} batch {completed_batches:04}/{epoch_batch_count:04}: elapsed={:.1}s mean_train_loss={:.4}",
                    epoch_started.elapsed().as_secs_f32(),
                    total_loss / batches.max(1) as f32,
                );
            }
        }

        let train_loss = total_loss / batches.max(1) as f32;
        let valid_model = model.clone().valid();
        let validation_metrics = evaluate(
            &valid_model,
            &validation,
            &device,
            args.batch_size,
            None,
            false,
            Some(fit_shape_prior),
        )?;
        let score = validation_metrics.selection_score();
        if !score.is_finite() {
            bail!("epoch {epoch} produced a non-finite checkpoint selection score");
        }
        let record = EpochMetrics {
            epoch,
            train_loss,
            validation: validation_metrics,
            selection_score: score,
        };
        append_json_line(&metrics_path, &record)?;
        println!(
            "epoch {epoch:02}: train={train_loss:.4} val={:.4} threshold={:.3} recall={:.3} real_recall={:.3} specificity={:.3} real_specificity={:.3} pck@.05={:.3} offscreen={:.3} duplicates={} score={score:.3}",
            record.validation.loss,
            record.validation.presence_threshold,
            record.validation.recall,
            record.validation.real_positive_recall,
            record.validation.specificity,
            record.validation.real_negative_specificity,
            record.validation.pck_05,
            record.validation.offscreen_accuracy,
            record.validation.duplicate_point_pairs,
        );
        model
            .clone()
            .valid()
            .save_file(args.artifacts.join("model-last"), &CompactRecorder::new())?;
        if score > best_score + 1.0e-5 {
            best_score = score;
            epochs_without_improvement = 0;
            model.clone().valid().save_file(
                args.artifacts.join("candidate-best"),
                &CompactRecorder::new(),
            )?;
        } else {
            epochs_without_improvement += 1;
            if epochs_without_improvement >= args.patience {
                println!(
                    "early stopping after {epoch} epochs ({epochs_without_improvement} without improvement)"
                );
                break;
            }
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
        args.batch_size,
        None,
        false,
        Some(fit_shape_prior),
    )?;
    let test_metrics = evaluate(
        &best_model.valid(),
        &test,
        &device,
        args.batch_size,
        Some(validation_metrics.presence_threshold),
        compute_final_fit,
        Some(fit_shape_prior),
    )?;
    let promotion = PromotionDecision::from_metrics(args, &validation_metrics, &test_metrics);
    let final_metrics = FinalMetrics {
        schema_version: "roof-training-metrics/v5".to_owned(),
        validation: validation_metrics,
        test: test_metrics,
        promotion: promotion.clone(),
    };
    fs::write(
        args.artifacts.join("final-metrics.json"),
        serde_json::to_vec_pretty(&final_metrics)?,
    )?;
    if promotion.promoted {
        best_model
            .clone()
            .valid()
            .save_file(args.artifacts.join("model"), &CompactRecorder::new())?;
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
        fs::write(
            args.artifacts.join("model.json"),
            serde_json::to_vec_pretty(&manifest)?,
        )?;
    }
    println!(
        "test: loss={:.4} recall={:.3} specificity={:.3} real_specificity={:.3} pck@.05={:.3} offscreen={:.3} fit_rmse={:.3} fit_iou={:.3}; promoted={}; checkpoint={}",
        final_metrics.test.loss,
        final_metrics.test.recall,
        final_metrics.test.specificity,
        final_metrics.test.real_negative_specificity,
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

pub(super) fn evaluate_checkpoint<B: AutodiffBackend>(
    args: &Args,
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
    let fit_shape_prior = load_checkpoint_shape_prior(checkpoint);
    let compute_fit_metrics = !args.overfit;
    let validation_metrics = evaluate(
        &model,
        validation,
        &device,
        args.batch_size,
        None,
        false,
        fit_shape_prior,
    )?;
    let test_metrics = evaluate(
        &model,
        test,
        &device,
        args.batch_size,
        Some(validation_metrics.presence_threshold),
        compute_fit_metrics,
        fit_shape_prior,
    )?;
    let promotion = PromotionDecision::from_metrics(args, &validation_metrics, &test_metrics);
    let report = FinalMetrics {
        schema_version: "roof-checkpoint-evaluation/v2".to_owned(),
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
        "checkpoint evaluation: recall={:.3} specificity={:.3} real_specificity={:.3} pck@.03={:.3} pck@.05={:.3} offscreen={:.3} fit_rmse={:.3} fit_iou={:.3} duplicates={} passes_gate={}",
        report.test.recall,
        report.test.specificity,
        report.test.real_negative_specificity,
        report.test.pck_03,
        report.test.pck_05,
        report.test.offscreen_accuracy,
        report.test.synthetic_fit.median_mesh_rmse,
        report.test.synthetic_fit.median_silhouette_iou,
        report.test.duplicate_point_pairs,
        promotion.promoted,
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

/// Gives every source represented in a minibatch the same total presence weight.
/// This prevents the thousands of synthetic frames from drowning out the much
/// smaller real-photo presence datasets.
fn source_balancing_weights(samples: &[TrainingSample]) -> Vec<f32> {
    let mut counts = [0usize; 4];
    for sample in samples {
        counts[origin_index(sample.origin)] += 1;
    }
    samples
        .iter()
        .map(|sample| 1.0 / counts[origin_index(sample.origin)].max(1) as f32)
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
) -> Vec<TrainingSample> {
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
    let mut output = Vec::with_capacity(batches * batch_size);
    for batch_index in 0..batches {
        let mut batch = Vec::with_capacity(batch_size);
        for source in 0..4 {
            for slot in 0..slots[source] {
                let draw = batch_index * slots[source] + slot;
                let mut sample = groups[source][draw % groups[source].len()].clone();
                sample.key = format!("{}#epoch-{epoch}-draw-{draw}", sample.key);
                batch.push(sample);
            }
        }
        batch.shuffle(&mut rng);
        output.extend(batch);
    }
    output
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
    let roll = synthetic_roll_radians(sample.origin, &sample.key, epoch, training);
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
    for pixel in image.pixels_mut() {
        let luminance =
            0.299 * f32::from(pixel[0]) + 0.587 * f32::from(pixel[1]) + 0.114 * f32::from(pixel[2]);
        for channel in &mut pixel.0 {
            let saturated = luminance + (f32::from(*channel) - luminance) * saturation;
            let normalized = (saturated * gain + bias).clamp(0.0, 255.0) / 255.0;
            *channel = (normalized.powf(gamma) * 255.0).clamp(0.0, 255.0) as u8;
        }
    }
    if (hash >> 40) & 3 == 0 {
        let sigma = 0.35 + ((hash >> 42) & 0xff) as f32 / 255.0 * 1.1;
        *image = image::imageops::blur(image, sigma);
    }
    if (hash >> 51) & 3 == 0 {
        let (width, height) = image.dimensions();
        let scale = 0.45 + ((hash >> 53) & 0xff) as f32 / 255.0 * 0.4;
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

fn observation_loss<B: Backend>(
    output: KeypointRoofOutput<B>,
    batch: &ObservationBatch<B>,
) -> Tensor<B, 1> {
    let positive = batch.presence.clone();
    let negative = positive.clone().neg() + 1.0;
    let presence_per_sample = -(log_sigmoid(output.presence_logits.clone()) * positive
        + log_sigmoid(output.presence_logits.clone().neg()) * negative);
    let presence_loss = (presence_per_sample * batch.presence_weights.clone()).sum()
        / batch.presence_weights.clone().sum().clamp_min(1.0e-6);

    if batch.geometry_count == 0 {
        return presence_loss;
    }

    // Presence-only photos and ordinary-building negatives do not need the
    // expensive 4,097-way D4 objective. Selecting synthetic-positive rows
    // before forming pairwise correspondences is exactly equivalent to
    // masking their losses afterward, without doing that work for the other
    // ten rows in the normal source-balanced batch.
    let keypoint_logits = output
        .keypoint_logits
        .select(0, batch.geometry_indices.clone());
    let offscreen_logits = output
        .offscreen_logits
        .select(0, batch.geometry_indices.clone());
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
    presence_loss + geometry_loss
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
    synthetic_fit: SyntheticFitMetrics,
}

impl EvaluationMetrics {
    fn selection_score(&self) -> f32 {
        let real_recall = if self.real_positive_count == 0 {
            1.0
        } else {
            self.real_positive_recall
        };
        let real_specificity = if self.real_negative_count == 0 {
            self.specificity
        } else {
            self.real_negative_specificity
        };
        let observation_score = 0.40 * self.pck_05
            + 0.15 * self.recall
            + 0.075 * self.specificity
            + 0.075 * real_specificity
            + 0.15 * real_recall
            + 0.15 * self.offscreen_accuracy;
        if self.duplicate_point_pairs == 0 {
            observation_score
        } else {
            // A collapsed solution must not outrank a slightly less accurate
            // checkpoint whose twelve structural points remain distinct.
            observation_score - 0.05 - 0.005 * self.duplicate_point_pairs.min(10) as f32
        }
    }
}

#[derive(Clone, Debug, Serialize)]
struct EpochMetrics {
    epoch: usize,
    train_loss: f32,
    validation: EvaluationMetrics,
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

fn calibrate_presence_threshold(observations: &[PresenceObservation]) -> f32 {
    if observations.is_empty() {
        return 0.5;
    }
    let mut candidates = observations
        .iter()
        .map(|observation| observation.probability.clamp(0.0, 1.0))
        .collect::<Vec<_>>();
    candidates.extend([0.0, 0.5, 1.0]);
    candidates.sort_by(f32::total_cmp);
    candidates.dedup_by(|left, right| (*left - *right).abs() < 1.0e-6);

    let has_real_positives = observations
        .iter()
        .any(|observation| observation.origin == SampleOrigin::RealPositive);
    let has_real_negatives = observations
        .iter()
        .any(|observation| observation.origin == SampleOrigin::RealNegative);
    let mut best = (f32::NEG_INFINITY, 0.5_f32);
    for threshold in candidates {
        let mut true_positives = 0usize;
        let mut false_negatives = 0usize;
        let mut true_negatives = 0usize;
        let mut false_positives = 0usize;
        let mut real_correct = 0usize;
        let mut real_total = 0usize;
        let mut real_negative_correct = 0usize;
        let mut real_negative_total = 0usize;
        for observation in observations {
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
        // Prefer thresholds satisfying the requested recall floor, then choose
        // the one with the strongest ordinary-building rejection. Falling back
        // remains well-defined for early, poorly separated epochs.
        let viable = recall >= 0.95 && real_recall >= 0.80;
        let score = if viable {
            10.0 + 0.5 * specificity
                + 0.5 * real_specificity
                + 0.1 * real_recall
                + threshold * 1.0e-4
        } else {
            recall + 0.125 * specificity + 0.125 * real_specificity + 0.25 * real_recall
        };
        if score > best.0 {
            best = (score, threshold);
        }
    }
    best.1
}

#[derive(Clone, Debug, Serialize)]
struct PromotionDecision {
    promoted: bool,
    failures: Vec<String>,
}

impl PromotionDecision {
    fn from_metrics(args: &Args, validation: &EvaluationMetrics, test: &EvaluationMetrics) -> Self {
        let mut failures = Vec::new();
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
            check_minimum(&mut failures, "test recall", test.recall, 0.95);
            if test.real_positive_count == 0 {
                failures.push("test has no held-out real Pizza Hut positives".to_owned());
            } else {
                check_minimum(
                    &mut failures,
                    "test real-photo recall",
                    test.real_positive_recall,
                    0.80,
                );
            }
            check_minimum(&mut failures, "test specificity", test.specificity, 0.90);
            if test.real_negative_count == 0 {
                failures.push("test has no curated real-building negatives".to_owned());
            } else {
                check_minimum(
                    &mut failures,
                    "test real-negative specificity",
                    test.real_negative_specificity,
                    0.85,
                );
            }
            check_minimum(&mut failures, "test PCK@5%", test.pck_05, 0.90);
            check_minimum(
                &mut failures,
                "test offscreen accuracy",
                test.offscreen_accuracy,
                0.90,
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
                    0.90,
                );
                if test.synthetic_fit.median_mesh_rmse > 0.03 + 1.0e-6 {
                    failures.push(format!(
                        "test median fitted-mesh RMSE {:.3} exceeds required 0.030",
                        test.synthetic_fit.median_mesh_rmse
                    ));
                }
                check_minimum(
                    &mut failures,
                    "test median amodal silhouette IoU",
                    test.synthetic_fit.median_silhouette_iou,
                    0.80,
                );
            }
            if validation.real_positive_count > 0 {
                check_minimum(
                    &mut failures,
                    "validation real-photo recall",
                    validation.real_positive_recall,
                    0.80,
                );
            }
        }
        Self {
            promoted: failures.is_empty(),
            failures,
        }
    }
}

fn check_minimum(failures: &mut Vec<String>, name: &str, actual: f32, required: f32) {
    if actual + 1.0e-6 < required {
        failures.push(format!(
            "{name} {actual:.3} is below required {required:.3}"
        ));
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
        loss_sum += observation_loss(output, &batch).into_scalar().elem::<f32>();
        batches += 1;

        let probabilities = burn::tensor::activation::sigmoid(presence_logits)
            .to_data()
            .to_vec::<f32>()?;
        for (probability, sample) in probabilities.into_iter().zip(chunk) {
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

    let threshold =
        threshold.unwrap_or_else(|| calibrate_presence_threshold(&presence_observations));
    metrics.presence_threshold = threshold;
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
    use image::{ColorType, ImageEncoder, codecs::png::PngEncoder};

    use super::*;

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
    fn training_uses_cached_working_raster_but_evaluation_keeps_source_transform() {
        let image = encoded_test_image(1_000, 333);
        let sample = TrainingSample {
            key: "source-transform".to_owned(),
            split: Split::Train,
            image: ImageSource::Encoded(image),
            presence: 0,
            geometry: None,
            origin: SampleOrigin::RealNegative,
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
        assert!((calibrate_presence_threshold(&observations) - 0.8).abs() < 1.0e-6);
    }

    #[test]
    fn source_weights_sum_to_one_per_origin() {
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
        };
        let samples = vec![
            sample(SampleOrigin::SyntheticTarget),
            sample(SampleOrigin::SyntheticTarget),
            sample(SampleOrigin::RealPositive),
            sample(SampleOrigin::RealNegative),
        ];
        let weights = source_balancing_weights(&samples);
        assert_eq!(weights, [0.5, 0.5, 1.0, 1.0]);
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
        };
        let mut samples = Vec::new();
        for index in 0..20 {
            samples.push(sample(index, SampleOrigin::SyntheticTarget));
            samples.push(sample(index + 20, SampleOrigin::SyntheticNegative));
        }
        samples.push(sample(40, SampleOrigin::RealPositive));
        samples.push(sample(41, SampleOrigin::RealNegative));
        let epoch = balanced_epoch_samples(&samples, 16, 42, 1);
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

    #[test]
    fn final_gate_enforces_curated_real_negative_specificity_separately() {
        use clap::Parser;

        let args = Args::try_parse_from(["roof-train"]).unwrap();
        let validation = EvaluationMetrics::default();
        let mut test = EvaluationMetrics {
            recall: 0.95,
            real_positive_recall: 0.80,
            real_positive_count: 5,
            specificity: 0.90,
            real_negative_specificity: 0.80,
            real_negative_count: 20,
            pck_05: 0.90,
            offscreen_accuracy: 0.90,
            ..EvaluationMetrics::default()
        };
        test.synthetic_fit.attempted = 10;
        test.synthetic_fit.fitted = 10;
        test.synthetic_fit.accepted = 10;
        test.synthetic_fit.fit_success_rate = 1.0;
        test.synthetic_fit.accepted_rate = 1.0;
        test.synthetic_fit.median_mesh_rmse = 0.02;
        test.synthetic_fit.median_silhouette_iou = 0.90;

        let rejected = PromotionDecision::from_metrics(&args, &validation, &test);
        assert!(!rejected.promoted);
        assert!(
            rejected
                .failures
                .iter()
                .any(|failure| failure.contains("real-negative specificity"))
        );

        test.real_negative_specificity = 0.85;
        assert!(PromotionDecision::from_metrics(&args, &validation, &test).promoted);
    }
}
