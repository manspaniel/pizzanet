//! End-to-end deterministic dataset generation.

use std::{
    collections::BTreeMap,
    fs::{self, File},
    io::{BufWriter, Cursor, Write},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use half::f16;
use image::{
    DynamicImage, GrayImage, ImageBuffer, ImageFormat, Luma, RgbImage, codecs::jpeg::JpegEncoder,
};
use roof_geometry::{
    EdgeCategory, FaceClass, FaceId, KeypointCategory, RoofGeometry, RoofParameters, generate_roof,
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use synth_data::{
    AssetRef, DatasetManifest, DatasetSplit, DatasetStatistics, DatasetValidator, DenseLabelRefs,
    EdgeLabel, EdgeVisibility, FloatRange, FrameAssets, FrameIdentity, FrameRecord,
    GeneratorConfig, GeneratorDescriptor, KeypointLabel, LabelClass, LabelTaxonomy, LocatorLabel,
    NormalizedBoundingBox, SequenceSampler, StructuralLabels, TargetKind, Validate, Vec2, Vec3,
    Visibility,
};
use synth_render::{
    OffscreenRenderer, PreparedRenderAssets, RenderCamera, RenderMesh, RenderSettings,
    RenderedFrame, SceneDescription,
};

use crate::{
    assets::PublicAssetCatalog,
    coverage::{
        CoverageSummary, SceneRegimeCounts, scene_regime, select_negative_sequence_plans,
        select_sequence_plans,
    },
    photometric::{PHOTOMETRIC_PROFILE_VERSION, apply_phone_camera_appearance},
    preview::{PreviewGallery, PreviewImages, PreviewMetadata, PreviewSummary},
    render_plan::ResolvedRenderAssets,
    shard::{Artifact, ShardWriter},
};

const COVERAGE_MANIFEST_FILE: &str = "coverage.json";
const SCENE_REGIME_BALANCE_FILE: &str = "scene-regime-balance.json";

/// Inputs controlling one deterministic local generation run.
#[derive(Clone, Debug)]
pub struct GenerateOptions {
    /// New or empty directory that will receive the manifest and shards.
    pub output: PathBuf,
    /// Dataset identifier written into the manifest.
    pub dataset_id: String,
    /// First building seed.
    pub seed: u64,
    /// Number of independently sampled two-tier target buildings.
    pub target_count: u32,
    /// Number of independently sampled ordinary-building negatives.
    pub negative_count: u32,
    /// Number of coherent views per building.
    pub frames_per_sequence: u32,
    /// Output width.
    pub width: u32,
    /// Output height.
    pub height: u32,
    /// Maximum frame samples in one tar shard.
    pub samples_per_shard: usize,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            output: PathBuf::from("datasets/synthetic/roof-local"),
            dataset_id: "roof-local".to_owned(),
            seed: 42,
            target_count: 1,
            negative_count: 1,
            frames_per_sequence: 1,
            width: 640,
            height: 480,
            samples_per_shard: 256,
        }
    }
}

/// Machine-readable result printed after a successful generation run.
#[derive(Clone, Debug, Serialize)]
pub struct GenerationSummary {
    /// Dataset output directory.
    pub output: PathBuf,
    /// WGPU adapter used for offscreen rendering.
    pub adapter: String,
    /// Number of generated coherent sequences.
    pub sequences: u32,
    /// Number of generated target buildings.
    pub targets: u32,
    /// Number of generated ordinary-building negatives.
    pub negatives: u32,
    /// Number of generated frame samples.
    pub frames: u64,
    /// Deterministic morphology by day-phase by domain selection coverage.
    pub coverage: CoverageSummary,
    /// Dataset-relative persisted copy of `coverage`.
    pub coverage_manifest: PathBuf,
    /// Target, negative, combined, and split-specific scene-regime counts.
    pub scene_regime_balance: SceneRegimeBalanceSummary,
    /// Dataset-relative persisted copy of `scene_regime_balance`.
    pub scene_regime_balance_manifest: PathBuf,
    /// Dataset-relative tar shard paths.
    pub shards: Vec<PathBuf>,
    /// Bounded, dataset-relative visual inspection output.
    pub preview: PreviewSummary,
}

/// Three-regime counts for one class, including the generated split assignment.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct ClassSceneRegimeBalance {
    /// Counts across every split.
    pub all: SceneRegimeCounts,
    /// Training-split counts.
    pub train: SceneRegimeCounts,
    /// Validation-split counts.
    pub validation: SceneRegimeCounts,
    /// Test-split counts.
    pub test: SceneRegimeCounts,
    /// Whether overall counts differ by at most one.
    pub balanced_overall: bool,
    /// Whether each non-empty split differs by at most one.
    pub balanced_within_splits: bool,
}

/// Auditable target/negative balance over urban, suburban, and remote regimes.
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct SceneRegimeBalanceSummary {
    /// Two-tier target counts.
    pub targets: ClassSceneRegimeBalance,
    /// Ordinary-building negative counts.
    pub negatives: ClassSceneRegimeBalance,
    /// Both classes combined.
    pub combined: ClassSceneRegimeBalance,
    /// True when target, negative, and combined overall counts are all balanced.
    pub balanced_overall: bool,
}

/// Generates a complete manifest, coherent sequence records, and frame shards.
pub fn generate_dataset(options: &GenerateOptions) -> Result<GenerationSummary> {
    validate_options(options)?;
    validate_output_destination(&options.output)?;
    let staging = staging_path(&options.output)?;
    if staging.exists() {
        bail!(
            "staging directory {} already exists; another generation may be active",
            staging.display()
        );
    }
    prepare_output_directory(&staging)?;

    let mut staged_options = options.clone();
    staged_options.output = staging.clone();
    let generated = generate_dataset_in_place(&staged_options);
    let mut summary = match generated {
        Ok(summary) => summary,
        Err(error) => {
            if let Err(cleanup_error) = fs::remove_dir_all(&staging) {
                return Err(error.context(format!(
                    "also failed to remove incomplete staging directory {}: {cleanup_error}",
                    staging.display()
                )));
            }
            return Err(error);
        }
    };

    if options.output.exists() {
        fs::remove_dir(&options.output).with_context(|| {
            format!(
                "failed to replace empty output directory {}",
                options.output.display()
            )
        })?;
    }
    fs::rename(&staging, &options.output).with_context(|| {
        format!(
            "failed to atomically publish {} as {}",
            staging.display(),
            options.output.display()
        )
    })?;
    summary.output = options.output.clone();
    Ok(summary)
}

fn generate_dataset_in_place(options: &GenerateOptions) -> Result<GenerationSummary> {
    let mut config = GeneratorConfig::default();
    config.image.width = options.width;
    config.image.height = options.height;
    config.sequence.frame_count = options.frames_per_sequence;
    // The shared roof schema does not yet encode asymmetric sides, so do not
    // claim a perturbation that the generated mesh cannot consume.
    config.roof.asymmetry_fraction = FloatRange::new(0.0, 0.0);
    let sampler = SequenceSampler::new(config.clone())?;
    let target_selection = select_sequence_plans(&sampler, options.seed, options.target_count)?;
    let coverage = target_selection.summary;
    let negative_selection = select_negative_sequence_plans(
        &sampler,
        options.seed,
        options.target_count,
        options.negative_count,
    )?;
    let mut targets = target_selection.sequences.into_iter();
    let mut negatives = negative_selection.into_iter();
    let mut selected_sequences =
        Vec::with_capacity((options.target_count + options.negative_count) as usize);
    loop {
        let target = targets.next();
        let negative = negatives.next();
        if target.is_none() && negative.is_none() {
            break;
        }
        selected_sequences.extend(target);
        selected_sequences.extend(negative);
    }
    let public_assets = PublicAssetCatalog::load_default()?;
    public_assets.verify_all()?;

    let mut manifest = DatasetManifest::new(
        &options.dataset_id,
        GeneratorDescriptor::chacha20(
            "roof-synth",
            env!("CARGO_PKG_VERSION"),
            sampler.config_fingerprint(),
        ),
        options.seed,
    );
    manifest.generator.source_revision = option_env!("ROOF_SYNTH_GIT_REVISION").map(str::to_owned);
    manifest.labels = label_taxonomy();
    manifest.source_assets = public_assets.dataset_sources();
    let manifest_report = manifest.validate();
    if !manifest_report.is_valid() {
        bail!("generated manifest contract is invalid: {manifest_report}");
    }

    let renderer = pollster::block_on(OffscreenRenderer::new())?;
    let adapter_info = renderer.adapter_info();
    let adapter = format!("{} ({:?})", adapter_info.name, adapter_info.backend);
    manifest
        .generator
        .execution_environment
        .insert("renderer_api".to_owned(), "wgpu/30.0.0".to_owned());
    manifest
        .generator
        .execution_environment
        .insert("adapter_name".to_owned(), adapter_info.name.clone());
    manifest.generator.execution_environment.insert(
        "adapter_backend".to_owned(),
        format!("{:?}", adapter_info.backend),
    );
    manifest.generator.execution_environment.insert(
        "adapter_device_type".to_owned(),
        format!("{:?}", adapter_info.device_type),
    );
    manifest
        .generator
        .execution_environment
        .insert("adapter_driver".to_owned(), adapter_info.driver.clone());
    manifest.generator.execution_environment.insert(
        "adapter_driver_info".to_owned(),
        adapter_info.driver_info.clone(),
    );
    manifest.generator.execution_environment.insert(
        "public_asset_pack".to_owned(),
        "pizzahut-synthetic-public-assets-v1".to_owned(),
    );
    manifest.generator.execution_environment.insert(
        "photometric_profile_version".to_owned(),
        PHOTOMETRIC_PROFILE_VERSION.to_string(),
    );
    manifest.generator.execution_environment.insert(
        "coverage_required_cells".to_owned(),
        coverage.required_cell_count.to_string(),
    );
    manifest.generator.execution_environment.insert(
        "coverage_covered_cells".to_owned(),
        coverage.covered_cell_count.to_string(),
    );
    manifest.generator.execution_environment.insert(
        "coverage_full_required".to_owned(),
        coverage.full_coverage_required.to_string(),
    );
    manifest.generator.execution_environment.insert(
        "coverage_manifest".to_owned(),
        COVERAGE_MANIFEST_FILE.to_owned(),
    );
    manifest.generator.execution_environment.insert(
        "target_buildings".to_owned(),
        options.target_count.to_string(),
    );
    manifest.generator.execution_environment.insert(
        "ordinary_negative_buildings".to_owned(),
        options.negative_count.to_string(),
    );
    manifest.generator.execution_environment.insert(
        "views_per_building".to_owned(),
        options.frames_per_sequence.to_string(),
    );
    let mut render_asset_bundle = public_assets.load_render_material_bundle()?;
    let mut prepared_assets: Option<PreparedRenderAssets> = None;
    let mut active_environment_id: Option<String> = None;
    let mut environment_cache = BTreeMap::new();
    let mut preview_gallery = PreviewGallery::default();
    let mut writers = SplitWriters::new(&options.output, options.samples_per_shard)?;
    let mut sequences =
        Vec::with_capacity((options.target_count + options.negative_count) as usize);

    for selected in selected_sequences {
        let building_seed = selected.seed;
        let mut plan = selected.plan;
        let resolved_assets =
            ResolvedRenderAssets::resolve(&public_assets, &plan.scene, building_seed)?;
        plan.scene.composition.source_asset_ids = resolved_assets.source_asset_ids.clone();
        let plan_report = plan.validate();
        if !plan_report.is_valid() {
            bail!(
                "sampled sequence {} failed validation after asset resolution: {plan_report}",
                plan.sequence_id
            );
        }
        let split = plan.split(&manifest.split_policy)?;
        let roof = if plan.request.target_kind == TargetKind::Negative {
            None
        } else {
            let roof_parameters = parameters_from_plan(&plan.scene.roof);
            Some(generate_roof(&roof_parameters)?)
        };
        let mut scene_description = SceneDescription::from_sampled(&plan.scene)?;
        scene_description.materials = resolved_assets.materials;

        if !environment_cache.contains_key(&resolved_assets.environment_id) {
            let environment =
                public_assets.load_render_environment(&resolved_assets.environment_id)?;
            environment_cache.insert(resolved_assets.environment_id.clone(), environment);
        }
        let environment = environment_cache
            .get(&resolved_assets.environment_id)
            .context("resolved environment disappeared from the decode cache")?;
        if plan.scene.lighting.sun_intensity > 0.0 {
            scene_description.environment.environment_yaw_radians = environment
                .yaw_to_align_dominant_light(
                    plan.scene.lighting.sun_azimuth_degrees.to_radians(),
                )?;
        }
        let scene_mesh = match (roof.as_ref(), plan.scene.ordinary_roof) {
            (Some(roof), None) => RenderMesh::from_scene(roof, &scene_description)?,
            (None, Some(ordinary_roof)) => {
                RenderMesh::from_ordinary_scene(ordinary_roof, &scene_description)?
            }
            _ => bail!(
                "sampled sequence {} has inconsistent target and ordinary roof geometry",
                plan.sequence_id
            ),
        };
        match prepared_assets.as_mut() {
            Some(assets)
                if active_environment_id.as_deref()
                    != Some(resolved_assets.environment_id.as_str()) =>
            {
                renderer.update_environment(assets, environment)?;
            }
            Some(_) => {}
            None => {
                render_asset_bundle.environment = Some(environment.clone());
                prepared_assets = Some(renderer.prepare_assets(&render_asset_bundle)?);
                render_asset_bundle.environment = None;
            }
        }
        active_environment_id = Some(resolved_assets.environment_id.clone());
        let prepared_assets = prepared_assets
            .as_ref()
            .context("renderer assets were not prepared")?;
        let sampled_environment = plan
            .scene
            .composition
            .environment
            .context("sampled scene contains no correlated environment")?;

        let mut frame_records = Vec::with_capacity(plan.frames.len());
        for frame_plan in &plan.frames {
            let sample_key = plan
                .frame_key(frame_plan.frame_index)
                .context("sampled frame index was outside its sequence")?;
            let settings = RenderSettings::from_sampled(&plan.scene, frame_plan.camera)?;
            let mut rendered =
                renderer.render_with_assets(&scene_mesh, &settings, prepared_assets)?;
            let photometric_profile = apply_phone_camera_appearance(
                rendered.width,
                rendered.height,
                &mut rendered.color_rgba8,
                photometric_seed(building_seed, frame_plan.frame_index),
                sampled_environment,
            )?;
            let encoded = EncodedTargets::new(&rendered, photometric_profile.jpeg_quality)
                .with_context(|| {
                    format!(
                        "encode visible targets for {sample_key} from building seed {building_seed}"
                    )
                })?;
            let mut record = build_frame_record(FrameBuildContext {
                sample_key: &sample_key,
                plan: &plan,
                frame_plan,
                split,
                roof: roof.as_ref(),
                camera: &settings.camera,
                rendered: &rendered,
                encoded: &encoded,
            })
            .with_context(|| {
                format!("build labels for {sample_key} from building seed {building_seed}")
            })?;
            record.roof = plan.roof_instance();
            record.appearance.photometric_profile = Some(photometric_profile);
            ensure_publishable_target_visibility(
                &record.sample_key,
                building_seed,
                plan.camera_motion.framing_intent,
                record.locator,
            )?;

            let report = DatasetValidator::new(&manifest).validate_frame(&record);
            if !report.is_valid() {
                bail!("generated frame {sample_key} failed validation: {report}");
            }

            preview_gallery.consider(
                PreviewMetadata {
                    sample_key: sample_key.clone(),
                    sequence_id: plan.sequence_id.clone(),
                    frame_index: frame_plan.frame_index,
                    target_kind: plan.request.target_kind,
                    ordinary_roof_family: plan.scene.ordinary_roof.map(|roof| roof.family),
                    roof_morphology: plan.scene.roof.morphology,
                    day_phase: format!("{:?}", sampled_environment.day_phase),
                    domain: format!("{:?}", sampled_environment.domain),
                    weather: format!("{:?}", sampled_environment.weather),
                    signage: plan
                        .scene
                        .composition
                        .signage
                        .iter()
                        .map(|sign| format!("{:?}", sign.kind))
                        .collect(),
                    building_extensions: plan
                        .scene
                        .composition
                        .building_extensions
                        .iter()
                        .map(|extension| format!("{:?}/{:?}", extension.kind, extension.roof))
                        .collect(),
                    roof_material: plan.scene.roof_material.id.clone(),
                    wall_material: plan.scene.wall_material.id.clone(),
                    ground_material: plan
                        .scene
                        .composition
                        .ground_material
                        .as_ref()
                        .map_or_else(|| "procedural".to_owned(), |material| material.id.clone()),
                    camera_path: format!("{:?}", plan.camera_motion.path_kind),
                    zoom_behavior: format!("{:?}", plan.camera_motion.zoom_behavior),
                    framing_intent: format!("{:?}", plan.camera_motion.framing_intent),
                    apparent_scale: format!("{:?}", plan.camera_motion.apparent_scale),
                    environment_asset_id: resolved_assets.environment_id.clone(),
                    horizontal_fov_degrees: 2.0
                        * ((frame_plan.camera.intrinsics.width as f32)
                            / (2.0 * frame_plan.camera.intrinsics.fx))
                            .atan()
                            .to_degrees(),
                    roof_bbox_width_fraction: record
                        .locator
                        .bounding_box
                        .map_or(0.0, |bounds| (bounds.max.x - bounds.min.x).max(0.0)),
                    occluder_kinds: {
                        let mut kinds = plan
                            .scene
                            .occluders
                            .iter()
                            .map(|occluder| format!("{:?}", occluder.kind))
                            .collect::<Vec<_>>();
                        kinds.sort();
                        kinds.dedup();
                        kinds
                    },
                    background_building_count: plan.scene.composition.background_buildings.len(),
                    vegetation_count: plan.scene.composition.vegetation.len(),
                    source_asset_ids: resolved_assets.source_asset_ids.clone(),
                    photometric_profile,
                    visible_fraction: record.locator.visible_fraction,
                    occluded_fraction: record.locator.occluded_fraction,
                    truncated: record.locator.truncated,
                },
                PreviewImages {
                    rgb_jpeg: &encoded.rgb_jpeg,
                    roof_mask_png: &encoded.roof_mask_png,
                    part_preview_png: &encoded.part_preview_png,
                    face_preview_png: &encoded.face_preview_png,
                },
            );

            let labels_json = serde_json::to_vec(&record)?;
            let mut artifacts = vec![
                Artifact {
                    suffix: "rgb.jpg",
                    bytes: &encoded.rgb_jpeg,
                },
                Artifact {
                    suffix: "labels.json",
                    bytes: &labels_json,
                },
            ];
            if plan.request.target_kind != TargetKind::Negative {
                artifacts.extend([
                    Artifact {
                        suffix: "roof_mask.png",
                        bytes: &encoded.roof_mask_png,
                    },
                    Artifact {
                        suffix: "amodal_roof_mask.png",
                        bytes: &encoded.amodal_roof_mask_png,
                    },
                    Artifact {
                        suffix: "part_mask.png",
                        bytes: &encoded.part_mask_png,
                    },
                    Artifact {
                        suffix: "face_ids.png",
                        bytes: &encoded.face_ids_png,
                    },
                    Artifact {
                        suffix: "facecoords.bin.zst",
                        bytes: &encoded.face_coordinates_zstd,
                    },
                ]);
            }
            writers.writer_mut(split).append(&sample_key, &artifacts)?;
            increment_split_count(&mut manifest.statistics, split);
            frame_records.push(record);
        }

        let sequence_record = plan.clone().into_record(&manifest.split_policy)?;
        let report =
            DatasetValidator::new(&manifest).validate_sequence(&sequence_record, &frame_records);
        if !report.is_valid() {
            bail!(
                "generated sequence {} failed validation: {report}",
                sequence_record.sequence_id
            );
        }
        manifest.statistics.sequences += 1;
        sequences.push(sequence_record);
    }

    let shards = writers.finish()?;
    let scene_regime_balance = summarize_scene_regime_balance(&sequences)?;
    if !scene_regime_balance.balanced_overall {
        bail!(
            "generated target, negative, or combined scene regimes differ by more than one: {:?}",
            scene_regime_balance
        );
    }
    manifest.generator.execution_environment.insert(
        "scene_regime_balance_manifest".to_owned(),
        SCENE_REGIME_BALANCE_FILE.to_owned(),
    );
    manifest.generator.execution_environment.insert(
        "target_scene_regimes".to_owned(),
        compact_regime_counts(scene_regime_balance.targets.all),
    );
    manifest.generator.execution_environment.insert(
        "negative_scene_regimes".to_owned(),
        compact_regime_counts(scene_regime_balance.negatives.all),
    );
    manifest.generator.execution_environment.insert(
        "combined_scene_regimes".to_owned(),
        compact_regime_counts(scene_regime_balance.combined.all),
    );
    write_pretty_json(options.output.join("generator-config.json"), &config)?;
    write_pretty_json(options.output.join("sequences.json"), &sequences)?;
    write_pretty_json(options.output.join("dataset.json"), &manifest)?;
    let coverage_manifest = write_coverage_summary(&options.output, &coverage)?;
    let scene_regime_balance_manifest =
        write_scene_regime_balance(&options.output, &scene_regime_balance)?;
    let preview = preview_gallery.write(&options.output)?;

    Ok(GenerationSummary {
        output: options.output.clone(),
        adapter,
        sequences: options.target_count + options.negative_count,
        targets: options.target_count,
        negatives: options.negative_count,
        frames: manifest.statistics.train_frames
            + manifest.statistics.validation_frames
            + manifest.statistics.test_frames,
        coverage,
        coverage_manifest,
        scene_regime_balance,
        scene_regime_balance_manifest,
        shards: shards
            .into_iter()
            .map(|path| {
                path.strip_prefix(&options.output)
                    .unwrap_or(&path)
                    .to_path_buf()
            })
            .collect(),
        preview,
    })
}

fn summarize_scene_regime_balance(
    sequences: &[synth_data::SequenceRecord],
) -> Result<SceneRegimeBalanceSummary> {
    let targets = summarize_class_regimes(sequences, TargetKind::Target)?;
    let negatives = summarize_class_regimes(sequences, TargetKind::Negative)?;
    let combined = combine_class_regimes(targets, negatives);
    Ok(SceneRegimeBalanceSummary {
        targets,
        negatives,
        combined,
        balanced_overall: targets.balanced_overall
            && negatives.balanced_overall
            && combined.balanced_overall,
    })
}

fn summarize_class_regimes(
    sequences: &[synth_data::SequenceRecord],
    target_kind: TargetKind,
) -> Result<ClassSceneRegimeBalance> {
    let mut summary = ClassSceneRegimeBalance::default();
    for sequence in sequences
        .iter()
        .filter(|sequence| sequence.target_kind == target_kind)
    {
        let environment = sequence
            .scene
            .composition
            .environment
            .context("generated sequence omitted its scene domain")?;
        let regime = scene_regime(environment.domain);
        summary.all.increment(regime);
        match sequence.split {
            DatasetSplit::Train => summary.train.increment(regime),
            DatasetSplit::Validation => summary.validation.increment(regime),
            DatasetSplit::Test => summary.test.increment(regime),
        }
    }
    summary.balanced_overall = summary.all.is_balanced();
    summary.balanced_within_splits = summary.train.is_balanced()
        && summary.validation.is_balanced()
        && summary.test.is_balanced();
    Ok(summary)
}

fn combine_class_regimes(
    targets: ClassSceneRegimeBalance,
    negatives: ClassSceneRegimeBalance,
) -> ClassSceneRegimeBalance {
    let all = targets.all.plus(negatives.all);
    let train = targets.train.plus(negatives.train);
    let validation = targets.validation.plus(negatives.validation);
    let test = targets.test.plus(negatives.test);
    ClassSceneRegimeBalance {
        all,
        train,
        validation,
        test,
        balanced_overall: all.is_balanced(),
        balanced_within_splits: train.is_balanced()
            && validation.is_balanced()
            && test.is_balanced(),
    }
}

fn compact_regime_counts(counts: SceneRegimeCounts) -> String {
    format!(
        "urban={},suburban={},remote={}",
        counts.urban, counts.suburban, counts.remote
    )
}

fn write_scene_regime_balance(
    output: &Path,
    summary: &SceneRegimeBalanceSummary,
) -> Result<PathBuf> {
    let relative_path = PathBuf::from(SCENE_REGIME_BALANCE_FILE);
    write_pretty_json(output.join(&relative_path), summary)?;
    Ok(relative_path)
}

fn write_coverage_summary(output: &Path, coverage: &CoverageSummary) -> Result<PathBuf> {
    let relative_path = PathBuf::from(COVERAGE_MANIFEST_FILE);
    write_pretty_json(output.join(&relative_path), coverage)?;
    Ok(relative_path)
}

fn validate_options(options: &GenerateOptions) -> Result<()> {
    if options.dataset_id.trim().is_empty()
        || !options
            .dataset_id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        bail!("dataset ID must contain only ASCII letters, digits, '-' and '_'");
    }
    if options.target_count == 0 && options.negative_count == 0 {
        bail!("at least one target or negative building must be requested");
    }
    if options.frames_per_sequence == 0
        || options.width == 0
        || options.height == 0
        || options.samples_per_shard == 0
    {
        bail!("frame, image, and shard sizes must all be non-zero");
    }
    Ok(())
}

fn prepare_output_directory(output: &Path) -> Result<()> {
    if output.exists() && fs::read_dir(output)?.next().is_some() {
        bail!(
            "output directory {} is not empty; choose a new directory",
            output.display()
        );
    }
    fs::create_dir_all(output)?;
    Ok(())
}

fn validate_output_destination(output: &Path) -> Result<()> {
    if output.file_name().is_none() {
        bail!("output must name a dataset directory, not a filesystem root");
    }
    if output.exists() {
        let metadata = fs::symlink_metadata(output)?;
        if metadata.file_type().is_symlink() || !metadata.is_dir() {
            bail!("output {} must be a real directory", output.display());
        }
        if fs::read_dir(output)?.next().is_some() {
            bail!(
                "output directory {} is not empty; choose a new directory",
                output.display()
            );
        }
    }
    if let Some(parent) = output.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    Ok(())
}

fn staging_path(output: &Path) -> Result<PathBuf> {
    let name = output
        .file_name()
        .and_then(|name| name.to_str())
        .context("output directory name must be valid UTF-8")?;
    let parent = output.parent().filter(|path| !path.as_os_str().is_empty());
    let staging_name = format!(".{name}.roof-synth-{}.staging", std::process::id());
    Ok(parent.unwrap_or_else(|| Path::new(".")).join(staging_name))
}

fn parameters_from_plan(roof: &synth_data::SampledRoof) -> RoofParameters {
    RoofParameters {
        eave_width: roof.eave_width_m,
        eave_depth: roof.eave_depth_m,
        shoulder_width: roof.shoulder_width_m,
        shoulder_depth: roof.shoulder_depth_m,
        crown_top_width: roof.crown_top_width_m,
        crown_top_depth: roof.crown_top_depth_m,
        lower_rise: roof.lower_rise_m,
        upper_rise: roof.upper_rise_m,
    }
}

fn photometric_seed(building_seed: u64, frame_index: u32) -> u64 {
    building_seed
        .wrapping_add(u64::from(frame_index).wrapping_mul(0x9e37_79b9_7f4a_7c15))
        .rotate_left(23)
        ^ 0x7068_6f6e_652d_7631
}

fn ensure_publishable_target_visibility(
    sample_key: &str,
    building_seed: u64,
    framing_intent: synth_data::FramingIntent,
    locator: LocatorLabel,
) -> Result<()> {
    if locator.target_kind == TargetKind::Target
        && (!locator.visible_fraction.is_finite()
            || locator.visible_fraction <= 0.0
            || locator.bounding_box.is_none())
    {
        bail!(
            "refusing to publish fully invisible target frame {} from building seed {} with {:?} framing (visible fraction {})",
            sample_key,
            building_seed,
            framing_intent,
            locator.visible_fraction
        );
    }
    Ok(())
}

struct FrameBuildContext<'a> {
    sample_key: &'a str,
    plan: &'a synth_data::SequencePlan,
    frame_plan: &'a synth_data::CameraFramePlan,
    split: DatasetSplit,
    roof: Option<&'a RoofGeometry>,
    camera: &'a RenderCamera,
    rendered: &'a RenderedFrame,
    encoded: &'a EncodedTargets,
}

fn build_frame_record(context: FrameBuildContext<'_>) -> Result<FrameRecord> {
    let FrameBuildContext {
        sample_key,
        plan,
        frame_plan,
        split,
        roof,
        camera,
        rendered,
        encoded,
    } = context;
    let is_target = plan.request.target_kind != TargetKind::Negative;
    let mask = if is_target {
        MaskStatistics::from_ids(rendered)?
    } else {
        if rendered.semantic_ids.iter().any(|id| *id != 0)
            || rendered.amodal_semantic_ids.iter().any(|id| *id != 0)
        {
            bail!("ordinary-building negative renderer leaked target semantic IDs");
        }
        MaskStatistics::empty()
    };
    let assets = FrameAssets {
        rgb: asset_ref(
            format!("{sample_key}.rgb.jpg"),
            "image/jpeg",
            "jpeg",
            &encoded.rgb_jpeg,
        ),
        surface_normals: None,
        motion_vectors: None,
    };
    let locator = LocatorLabel {
        target_kind: plan.request.target_kind,
        bounding_box: mask.bounding_box,
        amodal_bounding_box: mask.amodal_bounding_box,
        visible_fraction: mask.visible_fraction,
        occluded_fraction: mask.occluded_fraction,
        truncated: mask.truncated,
    };
    let identity = FrameIdentity::new(
        sample_key,
        &plan.sequence_id,
        frame_plan.frame_index,
        frame_plan.timestamp_ns,
    );
    let mut record = FrameRecord::new(identity, split, frame_plan.camera, locator, assets);
    if let Some(roof) = roof {
        record.labels = StructuralLabels {
            keypoints: keypoint_labels(roof, camera, rendered, plan.scene.building.wall_height_m)?,
            edges: edge_labels(roof, camera, rendered, plan.scene.building.wall_height_m)?,
            dense: DenseLabelRefs {
                roof_mask: Some(asset_ref(
                    format!("{sample_key}.roof_mask.png"),
                    "image/png",
                    "uint8",
                    &encoded.roof_mask_png,
                )),
                amodal_roof_mask: Some(asset_ref(
                    format!("{sample_key}.amodal_roof_mask.png"),
                    "image/png",
                    "uint8",
                    &encoded.amodal_roof_mask_png,
                )),
                part_mask: Some(asset_ref(
                    format!("{sample_key}.part_mask.png"),
                    "image/png",
                    "uint16",
                    &encoded.part_mask_png,
                )),
                face_id_map: Some(asset_ref(
                    format!("{sample_key}.face_ids.png"),
                    "image/png",
                    "uint16",
                    &encoded.face_ids_png,
                )),
                face_coordinates: Some(asset_ref(
                    format!("{sample_key}.facecoords.bin.zst"),
                    "application/octet-stream",
                    "rg16float-zstd",
                    &encoded.face_coordinates_zstd,
                )),
            },
        };
    }
    Ok(record)
}

fn keypoint_labels(
    roof: &RoofGeometry,
    camera: &RenderCamera,
    frame: &RenderedFrame,
    wall_height: f32,
) -> Result<Vec<KeypointLabel>> {
    roof.keypoints
        .iter()
        .map(|keypoint| {
            let world = [
                keypoint.position[0],
                keypoint.position[1] + wall_height,
                keypoint.position[2],
            ];
            let projected = camera.project(world, frame.width, frame.height)?;
            let (visibility, image_position) = projection_visibility(camera, projected, frame);
            Ok(KeypointLabel {
                class_id: keypoint_class(keypoint.category),
                instance_id: keypoint.id.as_u32() as u16,
                roof_position: Vec3::new(
                    keypoint.position[0],
                    keypoint.position[1],
                    keypoint.position[2],
                ),
                image_position,
                visibility,
            })
        })
        .collect()
}

fn edge_labels(
    roof: &RoofGeometry,
    camera: &RenderCamera,
    frame: &RenderedFrame,
    wall_height: f32,
) -> Result<Vec<EdgeLabel>> {
    roof.edges
        .iter()
        .map(|edge| {
            let (start, end) = roof
                .edge_positions(edge.id)
                .context("roof edge referred to a missing keypoint")?;
            let start = [start[0], start[1] + wall_height, start[2]];
            let end = [end[0], end[1] + wall_height, end[2]];
            let (polyline, visibility) = project_edge(camera, frame, start, end)?;
            Ok(EdgeLabel {
                class_id: edge_class(edge.category),
                instance_id: edge.id.as_u32() as u16,
                polyline,
                visibility,
            })
        })
        .collect()
}

fn project_edge(
    camera: &RenderCamera,
    frame: &RenderedFrame,
    start: [f32; 3],
    end: [f32; 3],
) -> Result<(Vec<Vec2>, EdgeVisibility)> {
    let projected_start = camera.project(start, frame.width, frame.height)?;
    let projected_end = camera.project(end, frame.width, frame.height)?;
    let projected_length = (projected_end.pixel[0] - projected_start.pixel[0])
        .hypot(projected_end.pixel[1] - projected_start.pixel[1]);
    let sample_count = if projected_length.is_finite() {
        (projected_length.ceil().min(4096.0) as u32 + 1).max(3)
    } else {
        33
    };
    let mut states = Vec::with_capacity(sample_count as usize);
    let mut projected_points = Vec::with_capacity(sample_count as usize);
    for index in 0..sample_count {
        let amount = index as f32 / (sample_count - 1) as f32;
        let world = [
            start[0] + (end[0] - start[0]) * amount,
            start[1] + (end[1] - start[1]) * amount,
            start[2] + (end[2] - start[2]) * amount,
        ];
        let projected = camera.project(world, frame.width, frame.height)?;
        let (state, point) = projection_visibility(camera, projected, frame);
        states.push(state);
        if let Some(point) = point {
            projected_points.push(point);
        }
    }

    let visibility = sampled_edge_visibility(&states);
    let polyline = projected_points
        .first()
        .copied()
        .zip(projected_points.last().copied())
        .and_then(|(start, end)| clip_segment_to_unit_square(start, end))
        .map_or_else(Vec::new, |[start, end]| vec![start, end]);
    Ok((polyline, visibility))
}

fn projection_visibility(
    camera: &RenderCamera,
    projected: synth_render::ProjectedPoint,
    frame: &RenderedFrame,
) -> (Visibility, Option<Vec2>) {
    if !projected.pixel[0].is_finite() || !projected.pixel[1].is_finite() {
        return (Visibility::BehindCamera, None);
    }
    let normalized = Vec2::new(
        projected.pixel[0] / frame.width as f32,
        projected.pixel[1] / frame.height as f32,
    );
    if !projected.in_frame {
        return (Visibility::Truncated, Some(normalized));
    }
    let x = projected.pixel[0]
        .floor()
        .clamp(0.0, frame.width.saturating_sub(1) as f32) as u32;
    let y = projected.pixel[1]
        .floor()
        .clamp(0.0, frame.height.saturating_sub(1) as f32) as u32;
    let Some(projected_distance) = camera.linearize_depth(projected.depth) else {
        return (Visibility::BehindCamera, None);
    };
    let world_units_per_pixel =
        2.0 * projected_distance * (camera.vertical_fov_radians() * 0.5).tan()
            / frame.height as f32;
    let depth_tolerance = (world_units_per_pixel * 1.5).max(0.03);
    let min_x = x.saturating_sub(1);
    let max_x = x.saturating_add(1).min(frame.width - 1);
    let min_y = y.saturating_sub(1);
    let max_y = y.saturating_add(1).min(frame.height - 1);
    let visible_in_footprint = (min_y..=max_y).any(|sample_y| {
        (min_x..=max_x).any(|sample_x| {
            let scene_depth = frame.depth[(sample_y * frame.width + sample_x) as usize];
            let scene_distance = camera.linearize_depth(scene_depth).unwrap_or(camera.far);
            projected_distance <= scene_distance + depth_tolerance
        })
    });
    let visibility = if visible_in_footprint {
        Visibility::Visible
    } else {
        Visibility::Occluded
    };
    (visibility, Some(normalized))
}

fn sampled_edge_visibility(states: &[Visibility]) -> EdgeVisibility {
    let behind = states
        .iter()
        .filter(|&&state| state == Visibility::BehindCamera)
        .count();
    if behind == states.len() {
        return EdgeVisibility::BehindCamera;
    }
    if behind > 0 || states.contains(&Visibility::Truncated) {
        return EdgeVisibility::Truncated;
    }
    let visible = states.contains(&Visibility::Visible);
    let occluded = states.contains(&Visibility::Occluded);
    match (visible, occluded) {
        (true, true) => EdgeVisibility::PartiallyOccluded,
        (true, false) => EdgeVisibility::Visible,
        (false, true) => EdgeVisibility::Occluded,
        (false, false) => EdgeVisibility::BehindCamera,
    }
}

fn clip_segment_to_unit_square(start: Vec2, end: Vec2) -> Option<[Vec2; 2]> {
    let dx = end.x - start.x;
    let dy = end.y - start.y;
    let mut minimum = 0.0_f32;
    let mut maximum = 1.0_f32;
    for (direction, distance) in [
        (-dx, start.x),
        (dx, 1.0 - start.x),
        (-dy, start.y),
        (dy, 1.0 - start.y),
    ] {
        if direction.abs() <= f32::EPSILON {
            if distance < 0.0 {
                return None;
            }
            continue;
        }
        let amount = distance / direction;
        if direction < 0.0 {
            minimum = minimum.max(amount);
        } else {
            maximum = maximum.min(amount);
        }
        if minimum > maximum {
            return None;
        }
    }
    let interpolate = |amount: f32| Vec2::new(start.x + dx * amount, start.y + dy * amount);
    Some([interpolate(minimum), interpolate(maximum)])
}

fn keypoint_class(category: KeypointCategory) -> u16 {
    match category {
        KeypointCategory::EaveCorner => 1,
        KeypointCategory::ShoulderCorner => 2,
        KeypointCategory::CrownTopCorner => 3,
    }
}

fn edge_class(category: EdgeCategory) -> u16 {
    match category {
        EdgeCategory::Eave => 1,
        EdgeCategory::ShoulderBreak => 2,
        EdgeCategory::LowerHip => 3,
        EdgeCategory::UpperHip => 4,
        EdgeCategory::CrownTopPerimeter => 5,
    }
}

fn face_class(face_id: u32) -> u16 {
    FaceId::from_u32(face_id).map_or(0, |face| face.class().as_u16())
}

fn label_taxonomy() -> LabelTaxonomy {
    LabelTaxonomy {
        keypoints: vec![
            LabelClass {
                id: 1,
                name: "eave_corner".to_owned(),
            },
            LabelClass {
                id: 2,
                name: "shoulder_corner".to_owned(),
            },
            LabelClass {
                id: 3,
                name: "crown_top_corner".to_owned(),
            },
        ],
        edges: vec![
            LabelClass {
                id: 1,
                name: "eave".to_owned(),
            },
            LabelClass {
                id: 2,
                name: "shoulder_break".to_owned(),
            },
            LabelClass {
                id: 3,
                name: "lower_hip".to_owned(),
            },
            LabelClass {
                id: 4,
                name: "upper_hip".to_owned(),
            },
            LabelClass {
                id: 5,
                name: "crown_top_perimeter".to_owned(),
            },
        ],
        parts: FaceClass::ALL
            .into_iter()
            .map(|class| LabelClass {
                id: class.as_u16(),
                name: class.as_str().to_owned(),
            })
            .collect(),
        faces: FaceId::ALL
            .into_iter()
            .map(|face| LabelClass {
                id: face.as_u32() as u16,
                name: face.as_str().to_owned(),
            })
            .collect(),
    }
}

struct EncodedTargets {
    rgb_jpeg: Vec<u8>,
    roof_mask_png: Vec<u8>,
    amodal_roof_mask_png: Vec<u8>,
    part_mask_png: Vec<u8>,
    face_ids_png: Vec<u8>,
    part_preview_png: Vec<u8>,
    face_preview_png: Vec<u8>,
    face_coordinates_zstd: Vec<u8>,
}

impl EncodedTargets {
    fn new(frame: &RenderedFrame, jpeg_quality: u8) -> Result<Self> {
        let mut rgb = Vec::with_capacity(frame.pixel_count() * 3);
        for pixel in frame.color_rgba8.chunks_exact(4) {
            rgb.extend_from_slice(&pixel[..3]);
        }
        let mut rgb_jpeg = Vec::new();
        JpegEncoder::new_with_quality(&mut rgb_jpeg, jpeg_quality).encode(
            &rgb,
            frame.width,
            frame.height,
            image::ExtendedColorType::Rgb8,
        )?;

        let roof_mask = frame
            .semantic_ids
            .iter()
            .map(|&id| u8::from(id != 0) * 255)
            .collect::<Vec<_>>();
        let amodal_roof_mask = frame
            .amodal_semantic_ids
            .iter()
            .map(|&id| u8::from(id != 0) * 255)
            .collect::<Vec<_>>();
        let part_mask = frame
            .semantic_ids
            .iter()
            .map(|&id| face_class(id))
            .collect::<Vec<_>>();
        let face_ids = frame
            .semantic_ids
            .iter()
            .map(|&id| u16::try_from(id).expect("roof semantic IDs fit in uint16"))
            .collect::<Vec<_>>();
        let roof_mask_png = encode_luma8(frame.width, frame.height, roof_mask)?;
        let amodal_roof_mask_png = encode_luma8(frame.width, frame.height, amodal_roof_mask)?;
        let part_mask_png = encode_luma16(frame.width, frame.height, part_mask)?;
        let face_ids_png = encode_luma16(frame.width, frame.height, face_ids)?;
        let part_preview_png = encode_rgb8(
            frame.width,
            frame.height,
            colorize_semantic_ids(&frame.semantic_ids, part_preview_color),
        )?;
        let face_preview_png = encode_rgb8(
            frame.width,
            frame.height,
            colorize_semantic_ids(&frame.semantic_ids, face_preview_color),
        )?;

        let mut coordinates = Vec::with_capacity(frame.pixel_count() * 4);
        for (&semantic_id, coordinate) in frame.semantic_ids.iter().zip(&frame.face_coordinates) {
            let value = if semantic_id == 0 {
                [f16::ZERO, f16::ZERO]
            } else {
                [f16::from_f32(coordinate[0]), f16::from_f32(coordinate[1])]
            };
            coordinates.extend_from_slice(&value[0].to_bits().to_le_bytes());
            coordinates.extend_from_slice(&value[1].to_bits().to_le_bytes());
        }
        let face_coordinates_zstd = zstd::stream::encode_all(Cursor::new(coordinates), 5)?;

        Ok(Self {
            rgb_jpeg,
            roof_mask_png,
            amodal_roof_mask_png,
            part_mask_png,
            face_ids_png,
            part_preview_png,
            face_preview_png,
            face_coordinates_zstd,
        })
    }
}

struct MaskStatistics {
    bounding_box: Option<NormalizedBoundingBox>,
    amodal_bounding_box: Option<NormalizedBoundingBox>,
    visible_fraction: f32,
    occluded_fraction: f32,
    truncated: bool,
}

impl MaskStatistics {
    const fn empty() -> Self {
        Self {
            bounding_box: None,
            amodal_bounding_box: None,
            visible_fraction: 0.0,
            occluded_fraction: 0.0,
            truncated: false,
        }
    }

    fn from_ids(frame: &RenderedFrame) -> Result<Self> {
        if frame.semantic_ids.len() != frame.pixel_count()
            || frame.amodal_semantic_ids.len() != frame.pixel_count()
        {
            bail!("renderer returned inconsistent semantic target lengths");
        }
        let mut truncated = false;
        let mut visible_pixels = 0_u64;
        let mut amodal_pixels = 0_u64;
        for &id in &frame.semantic_ids {
            if id == 0 {
                continue;
            }
            visible_pixels += 1;
        }
        for (index, &id) in frame.amodal_semantic_ids.iter().enumerate() {
            if id == 0 {
                continue;
            }
            amodal_pixels += 1;
            let x = index as u32 % frame.width;
            let y = index as u32 / frame.width;
            truncated |= x == 0 || y == 0 || x + 1 == frame.width || y + 1 == frame.height;
        }
        if amodal_pixels == 0 || visible_pixels > amodal_pixels {
            bail!("visible and roof-only semantic passes disagree");
        }
        let visible_fraction = visible_pixels as f32 / amodal_pixels as f32;
        Ok(Self {
            bounding_box: normalized_bounds(&frame.semantic_ids, frame.width, frame.height),
            amodal_bounding_box: normalized_bounds(
                &frame.amodal_semantic_ids,
                frame.width,
                frame.height,
            ),
            visible_fraction,
            occluded_fraction: 1.0 - visible_fraction,
            truncated,
        })
    }
}

fn normalized_bounds(ids: &[u32], width: u32, height: u32) -> Option<NormalizedBoundingBox> {
    let mut min_x = width;
    let mut min_y = height;
    let mut max_x = 0;
    let mut max_y = 0;
    let mut any = false;
    for (index, &id) in ids.iter().enumerate() {
        if id == 0 {
            continue;
        }
        any = true;
        let x = index as u32 % width;
        let y = index as u32 / width;
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    any.then_some(NormalizedBoundingBox {
        min: Vec2::new(min_x as f32 / width as f32, min_y as f32 / height as f32),
        max: Vec2::new(
            (max_x + 1) as f32 / width as f32,
            (max_y + 1) as f32 / height as f32,
        ),
    })
}

struct SplitWriters {
    train: ShardWriter,
    validation: ShardWriter,
    test: ShardWriter,
}

impl SplitWriters {
    fn new(output: &Path, samples_per_shard: usize) -> Result<Self> {
        Ok(Self {
            train: ShardWriter::new(output, "train", samples_per_shard)?,
            validation: ShardWriter::new(output, "validation", samples_per_shard)?,
            test: ShardWriter::new(output, "test", samples_per_shard)?,
        })
    }

    fn writer_mut(&mut self, split: DatasetSplit) -> &mut ShardWriter {
        match split {
            DatasetSplit::Train => &mut self.train,
            DatasetSplit::Validation => &mut self.validation,
            DatasetSplit::Test => &mut self.test,
        }
    }

    fn finish(self) -> Result<Vec<PathBuf>> {
        let mut shards = self.train.finish()?;
        shards.extend(self.validation.finish()?);
        shards.extend(self.test.finish()?);
        shards.sort();
        Ok(shards)
    }
}

fn asset_ref(path: String, media_type: &str, encoding: &str, bytes: &[u8]) -> AssetRef {
    let mut asset = AssetRef::new(path, media_type, encoding);
    asset.content_hash = Some(format!("sha256:{}", sha256_hex(bytes)));
    asset
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

fn encode_luma8(width: u32, height: u32, pixels: Vec<u8>) -> Result<Vec<u8>> {
    let image = GrayImage::from_vec(width, height, pixels).context("invalid uint8 label length")?;
    encode_dynamic_image(DynamicImage::ImageLuma8(image))
}

fn encode_luma16(width: u32, height: u32, pixels: Vec<u16>) -> Result<Vec<u8>> {
    let image = ImageBuffer::<Luma<u16>, Vec<u16>>::from_vec(width, height, pixels)
        .context("invalid uint16 label length")?;
    encode_dynamic_image(DynamicImage::ImageLuma16(image))
}

fn encode_rgb8(width: u32, height: u32, pixels: Vec<u8>) -> Result<Vec<u8>> {
    let image = RgbImage::from_vec(width, height, pixels).context("invalid RGB preview length")?;
    encode_dynamic_image(DynamicImage::ImageRgb8(image))
}

fn colorize_semantic_ids(ids: &[u32], color: fn(u32) -> [u8; 3]) -> Vec<u8> {
    ids.iter().flat_map(|id| color(*id)).collect()
}

fn part_preview_color(face_id: u32) -> [u8; 3] {
    match face_class(face_id) {
        1 => [235, 77, 75],
        2 => [247, 194, 72],
        3 => [72, 156, 230],
        _ => [12, 14, 18],
    }
}

fn face_preview_color(face_id: u32) -> [u8; 3] {
    const COLORS: [[u8; 3]; 13] = [
        [235, 77, 75],
        [238, 127, 64],
        [247, 194, 72],
        [166, 203, 76],
        [68, 183, 128],
        [55, 172, 198],
        [72, 131, 230],
        [123, 98, 221],
        [184, 87, 214],
        [226, 82, 151],
        [151, 119, 90],
        [154, 160, 166],
        [245, 245, 240],
    ];
    FaceId::from_u32(face_id)
        .and_then(|face| FaceId::ALL.iter().position(|candidate| *candidate == face))
        .map_or([12, 14, 18], |index| COLORS[index])
}

fn encode_dynamic_image(image: DynamicImage) -> Result<Vec<u8>> {
    let mut cursor = Cursor::new(Vec::new());
    image.write_to(&mut cursor, ImageFormat::Png)?;
    Ok(cursor.into_inner())
}

fn increment_split_count(statistics: &mut DatasetStatistics, split: DatasetSplit) {
    match split {
        DatasetSplit::Train => statistics.train_frames += 1,
        DatasetSplit::Validation => statistics.validation_frames += 1,
        DatasetSplit::Test => statistics.test_frames += 1,
    }
}

fn write_pretty_json(path: PathBuf, value: &impl Serialize) -> Result<()> {
    let file =
        File::create(&path).with_context(|| format!("failed to create {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    serde_json::to_writer_pretty(&mut writer, value)?;
    writeln!(writer)?;
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn taxonomy_contains_every_stable_face() {
        let taxonomy = label_taxonomy();
        assert_eq!(taxonomy.faces.len(), FaceId::ALL.len());
        for face in FaceId::ALL {
            assert!(
                taxonomy
                    .faces
                    .iter()
                    .any(|class| class.id == face.as_u32() as u16)
            );
        }
    }

    #[test]
    fn face_classes_reserve_zero_for_non_roof_geometry() {
        assert_eq!(face_class(0), 0);
        assert_eq!(face_class(FaceId::LowerFront.as_u32()), 1);
        assert_eq!(face_class(FaceId::UpperFront.as_u32()), 2);
        assert_eq!(face_class(FaceId::CrownTop.as_u32()), 2);
        assert_eq!(face_class(FaceId::Underside.as_u32()), 3);
    }

    #[test]
    fn output_directory_must_be_empty() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(directory.path().join("existing"), b"data").unwrap();
        assert!(prepare_output_directory(directory.path()).is_err());
    }

    #[test]
    fn persists_the_exact_coverage_summary_at_the_dataset_root() {
        let mut config = GeneratorConfig::default();
        config.sequence.frame_count = 1;
        let sampler = SequenceSampler::new(config).unwrap();
        let selection = select_sequence_plans(&sampler, 42, 3).unwrap();
        let directory = tempfile::tempdir().unwrap();

        let relative = write_coverage_summary(directory.path(), &selection.summary).unwrap();
        let bytes = fs::read(directory.path().join(&relative)).unwrap();
        let decoded: CoverageSummary = serde_json::from_slice(&bytes).unwrap();

        assert_eq!(relative, PathBuf::from(COVERAGE_MANIFEST_FILE));
        assert_eq!(decoded, selection.summary);
        assert!(bytes.ends_with(b"\n"));
    }

    #[test]
    fn clips_projected_edges_to_image_bounds() {
        let clipped =
            clip_segment_to_unit_square(Vec2::new(-0.5, 0.25), Vec2::new(1.5, 0.75)).unwrap();
        assert!((clipped[0].x - 0.0).abs() < 1.0e-6);
        assert!((clipped[0].y - 0.375).abs() < 1.0e-6);
        assert!((clipped[1].x - 1.0).abs() < 1.0e-6);
        assert!((clipped[1].y - 0.625).abs() < 1.0e-6);
        assert!(clip_segment_to_unit_square(Vec2::new(-2.0, 0.2), Vec2::new(-1.0, 0.8)).is_none());
    }

    #[test]
    fn aggregates_sampled_edge_visibility() {
        assert_eq!(
            sampled_edge_visibility(&[Visibility::Visible, Visibility::Occluded]),
            EdgeVisibility::PartiallyOccluded
        );
        assert_eq!(
            sampled_edge_visibility(&[Visibility::Visible, Visibility::Truncated]),
            EdgeVisibility::Truncated
        );
        assert_eq!(
            sampled_edge_visibility(&[Visibility::BehindCamera; 3]),
            EdgeVisibility::BehindCamera
        );
    }

    #[test]
    fn fully_invisible_target_frames_are_not_publishable() {
        let bounds = NormalizedBoundingBox {
            min: Vec2::new(0.2, 0.2),
            max: Vec2::new(0.8, 0.8),
        };
        let mut locator = LocatorLabel {
            target_kind: TargetKind::Target,
            bounding_box: None,
            amodal_bounding_box: Some(bounds),
            visible_fraction: 0.0,
            occluded_fraction: 1.0,
            truncated: false,
        };
        assert!(
            ensure_publishable_target_visibility(
                "sample",
                42,
                synth_data::FramingIntent::Centered,
                locator,
            )
            .is_err()
        );

        locator.bounding_box = Some(bounds);
        locator.visible_fraction = 0.01;
        locator.occluded_fraction = 0.99;
        assert!(
            ensure_publishable_target_visibility(
                "sample",
                42,
                synth_data::FramingIntent::Centered,
                locator,
            )
            .is_ok()
        );

        locator = LocatorLabel {
            target_kind: TargetKind::Negative,
            bounding_box: None,
            amodal_bounding_box: None,
            visible_fraction: 0.0,
            occluded_fraction: 0.0,
            truncated: false,
        };
        assert!(
            ensure_publishable_target_visibility(
                "sample",
                42,
                synth_data::FramingIntent::Centered,
                locator,
            )
            .is_ok()
        );
    }
}
