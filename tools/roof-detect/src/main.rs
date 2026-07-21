//! Image-only keypoint detector with a perspective structural-mesh overlay.

use std::{
    fs,
    io::Cursor,
    path::{Path, PathBuf},
    process::ExitCode,
};

use anyhow::{Context, Result, bail};
use burn::{
    backend::{Flex, Wgpu, flex::FlexDevice, wgpu::WgpuDevice},
    module::Module,
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::TensorData,
};
use clap::{Parser, ValueEnum};
use exif::{In, Reader, Tag};
use image::{DynamicImage, Rgba, RgbaImage, imageops};
use imageproc::drawing::{draw_filled_circle_mut, draw_line_segment_mut, draw_polygon_mut};
use imageproc::point::Point;
use roof_fit::{
    FitError, FocalLengthConfig, KeypointObservation, SingleViewFitConfig, SingleViewObservation,
    SingleViewRoofFit, fit_single_view,
};
use roof_geometry::{EdgeCategory, FaceClass, RoofParameters, generate_roof};
use roof_model::{
    DEFAULT_FIT_KEYPOINT_CONFIDENCE, HEATMAP_SIZE, KEYPOINT_COUNT, KeypointRoofDetection,
    KeypointRoofNetConfig, SPATIAL_INPUT_SIZE, decode_keypoint_prediction, prepare_rgb8_sized,
};
use serde::{Deserialize, Serialize};

const MODEL_MANIFEST_SCHEMA: &str = "roof-keypoint-model/v3";
const MINIMUM_CONFIDENT_FIT_OBSERVATIONS: usize = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum BackendChoice {
    Wgpu,
    Flex,
}

#[derive(Debug, Parser)]
#[command(
    name = "roof-detect",
    about = "Detect a retro Pizza Hut roof in one image and render its fitted two-tier mesh"
)]
struct Args {
    /// JPEG, PNG, or WebP input image.
    image: PathBuf,
    /// Burn checkpoint base path. CompactRecorder loads the corresponding `.mpk` file.
    #[arg(long, default_value = "artifacts/roof-model-keypoints/model")]
    model: PathBuf,
    /// Burn inference backend.
    #[arg(long, value_enum, default_value_t = BackendChoice::Wgpu)]
    backend: BackendChoice,
    /// Run both native backends and fail if decoded observations diverge.
    #[arg(long)]
    verify_backend_parity: bool,
    /// Probability required to accept roof presence.
    /// Defaults to the validation-calibrated value in the model manifest.
    #[arg(long)]
    threshold: Option<f32>,
    /// Probability required to classify a keypoint as offscreen.
    /// Defaults to the value in the model manifest.
    #[arg(long)]
    offscreen_threshold: Option<f32>,
    /// Minimum in-frame keypoint confidence accepted by the fitter.
    /// Defaults to the value used by the checkpoint promotion gate.
    #[arg(long)]
    keypoint_threshold: Option<f32>,
    /// Largest normalized-image reprojection RMSE accepted as a confident fit.
    #[arg(long, default_value_t = 0.05)]
    fit_threshold: f32,
    /// Overlay PNG path. Defaults beside the input.
    #[arg(long)]
    output: Option<PathBuf>,
    /// Structured prediction JSON path. Defaults beside the overlay.
    #[arg(long)]
    json: Option<PathBuf>,
    /// Draw a solved mesh even when detection or fit confidence is below threshold.
    #[arg(long)]
    show_all: bool,
    /// Draw the model's raw keypoints and their derived bounds.
    #[arg(long, alias = "raw-detector-debug")]
    raw_keypoint_debug: bool,
}

#[derive(Clone, Copy, Debug, Default)]
struct ExifMetadata {
    orientation: u32,
    focal_length_35mm: Option<f32>,
}

#[derive(Clone, Debug, Deserialize)]
struct ModelManifest {
    schema_version: String,
    input_size: usize,
    heatmap_size: usize,
    keypoint_count: usize,
    #[serde(default)]
    recommended_presence_threshold: Option<f32>,
    #[serde(default)]
    recommended_offscreen_threshold: Option<f32>,
    #[serde(default)]
    recommended_keypoint_threshold: Option<f32>,
    #[serde(default)]
    shape_prior: Option<ShapePrior>,
}

#[derive(Clone, Debug, Deserialize)]
struct ShapePrior {
    mean: [f32; 7],
}

#[derive(Clone, Debug, Serialize)]
#[serde(deny_unknown_fields)]
struct CameraMetadataReport {
    focal_length_35mm: Option<f32>,
    horizontal_fov_degrees: Option<f32>,
    focal_source: String,
}

/// The fitter's seven independent, scale-free shape values.
///
/// `SingleViewRoofFit::parameters` also retains the concrete normalized
/// dimensions used to regenerate the mesh. Keeping these ratios explicit in
/// the CLI contract avoids requiring consumers to reverse that conversion.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct InferredRoofRatios {
    eave_depth_to_eave_width: f32,
    shoulder_width_to_eave_width: f32,
    shoulder_depth_to_eave_depth: f32,
    crown_width_to_shoulder_width: f32,
    crown_depth_to_shoulder_depth: f32,
    lower_rise_to_eave_width: f32,
    upper_rise_to_eave_width: f32,
}

impl From<&RoofParameters> for InferredRoofRatios {
    fn from(parameters: &RoofParameters) -> Self {
        Self {
            eave_depth_to_eave_width: parameters.eave_depth / parameters.eave_width,
            shoulder_width_to_eave_width: parameters.shoulder_width / parameters.eave_width,
            shoulder_depth_to_eave_depth: parameters.shoulder_depth / parameters.eave_depth,
            crown_width_to_shoulder_width: parameters.crown_top_width / parameters.shoulder_width,
            crown_depth_to_shoulder_depth: parameters.crown_top_depth / parameters.shoulder_depth,
            lower_rise_to_eave_width: parameters.lower_rise / parameters.eave_width,
            upper_rise_to_eave_width: parameters.upper_rise / parameters.eave_width,
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum FitFailure {
    DetectionRejected {
        message: String,
        probability: f32,
        threshold: f32,
    },
    InsufficientObservations {
        message: String,
        minimum: usize,
        actual: usize,
    },
    InvalidConfiguration {
        message: String,
    },
    NoValidSolution {
        message: String,
    },
}

impl From<&FitError> for FitFailure {
    fn from(error: &FitError) -> Self {
        match error {
            FitError::InsufficientObservations { minimum, actual } => {
                Self::InsufficientObservations {
                    message: error.to_string(),
                    minimum: *minimum,
                    actual: *actual,
                }
            }
            FitError::InvalidConfiguration(_) => Self::InvalidConfiguration {
                message: error.to_string(),
            },
            FitError::NoValidSolution => Self::NoValidSolution {
                message: error.to_string(),
            },
        }
    }
}

#[derive(Debug, Serialize)]
struct InferenceReport {
    schema_version: String,
    image: String,
    model: String,
    backend: String,
    backend_parity_verified: bool,
    keypoint_threshold: f32,
    offscreen_threshold: f32,
    source_width: u32,
    source_height: u32,
    camera_metadata: CameraMetadataReport,
    accepted_keypoint_count: usize,
    /// Learned presence and three-ring amodal keypoint observations.
    detection: KeypointRoofDetection,
    fit_attempted: bool,
    /// Seven independent shape values inferred from the learned observations.
    inferred_roof_ratios: Option<InferredRoofRatios>,
    /// Camera, complete projected mesh, normalized dimensions, and confidence.
    fit: Option<SingleViewRoofFit>,
    #[serde(skip_serializing_if = "Option::is_none")]
    fit_error: Option<FitFailure>,
}

fn main() -> ExitCode {
    match run(Args::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("roof-detect: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Args) -> Result<()> {
    for (name, value) in [
        ("--threshold", args.threshold),
        ("--offscreen-threshold", args.offscreen_threshold),
        ("--keypoint-threshold", args.keypoint_threshold),
    ]
    .into_iter()
    .filter_map(|(name, value)| value.map(|value| (name, value)))
    {
        if !(0.0..=1.0).contains(&value) {
            bail!("{name} must lie between zero and one");
        }
    }
    if !args.fit_threshold.is_finite() || args.fit_threshold <= 0.0 {
        bail!("--fit-threshold must be finite and greater than zero");
    }
    ensure_model_exists(&args.model)?;
    let manifest_path = model_manifest_path(&args.model);
    let manifest = load_model_manifest(&manifest_path)?;
    if let Some(manifest) = &manifest {
        validate_model_manifest(manifest, &manifest_path)?;
    }
    let (presence_threshold, offscreen_threshold, keypoint_threshold) = resolve_thresholds(
        args.threshold,
        args.offscreen_threshold,
        args.keypoint_threshold,
        manifest.as_ref(),
    );
    for (name, value) in [
        ("presence threshold", presence_threshold),
        ("offscreen threshold", offscreen_threshold),
        ("keypoint threshold", keypoint_threshold),
    ] {
        if !(0.0..=1.0).contains(&value) {
            bail!("model {name} must lie between zero and one");
        }
    }

    let bytes = fs::read(&args.image)
        .with_context(|| format!("read input image {}", args.image.display()))?;
    let exif = read_exif_metadata(&bytes);
    let image = orient_image(image::load_from_memory(&bytes)?, exif.orientation);
    let rgb = image.to_rgb8();
    let (width, height) = rgb.dimensions();
    let prepared = prepare_rgb8_sized(width, height, rgb.as_raw(), SPATIAL_INPUT_SIZE)?;
    let detection = if args.verify_backend_parity {
        let wgpu = infer::<Wgpu>(
            &args.model,
            prepared.chw.clone(),
            prepared.transform,
            presence_threshold,
            offscreen_threshold,
            WgpuDevice::default(),
        )?;
        let flex = infer::<Flex>(
            &args.model,
            prepared.chw,
            prepared.transform,
            presence_threshold,
            offscreen_threshold,
            FlexDevice,
        )?;
        verify_backend_parity(&wgpu, &flex)?;
        match args.backend {
            BackendChoice::Wgpu => wgpu,
            BackendChoice::Flex => flex,
        }
    } else {
        match args.backend {
            BackendChoice::Wgpu => infer::<Wgpu>(
                &args.model,
                prepared.chw,
                prepared.transform,
                presence_threshold,
                offscreen_threshold,
                WgpuDevice::default(),
            )?,
            BackendChoice::Flex => infer::<Flex>(
                &args.model,
                prepared.chw,
                prepared.transform,
                presence_threshold,
                offscreen_threshold,
                FlexDevice,
            )?,
        }
    };

    let observation = keypoint_observation(&detection, keypoint_threshold);
    let accepted_keypoint_count = observation
        .keypoints
        .iter()
        .filter(|point| point.is_some())
        .count();
    let horizontal_fov_degrees = exif
        .focal_length_35mm
        .and_then(|focal| horizontal_fov_from_35mm(focal, width, height));
    let shape_prior = manifest
        .as_ref()
        .and_then(|manifest| manifest.shape_prior.as_ref())
        .map(|prior| prior.mean);
    let fit_config = SingleViewFitConfig {
        image_aspect_ratio: width as f32 / height as f32,
        shape_prior,
        focal_length: horizontal_fov_degrees.map_or_else(
            FocalLengthConfig::default,
            |horizontal_fov_degrees| FocalLengthConfig::Known {
                horizontal_fov_degrees,
                uncertainty_degrees: 5.0,
            },
        ),
        maximum_reprojection_rmse: args.fit_threshold,
        ..SingleViewFitConfig::default()
    };
    let fit_attempted = detection.detected || args.show_all;
    let (fit, fit_error) = fit_detection(&detection, &observation, fit_config, args.show_all);

    let output = args
        .output
        .unwrap_or_else(|| sibling_path(&args.image, "overlay", "png"));
    let json = args
        .json
        .unwrap_or_else(|| sibling_path(&args.image, "detection", "json"));
    create_parent_directory(&output)?;
    create_parent_directory(&json)?;

    let overlay = render_overlay(
        image.to_rgba8(),
        &detection,
        fit.as_ref(),
        keypoint_threshold,
        args.show_all,
        args.raw_keypoint_debug,
    );
    overlay
        .save(&output)
        .with_context(|| format!("save overlay {}", output.display()))?;
    let inferred_roof_ratios = fit
        .as_ref()
        .map(|fit| InferredRoofRatios::from(&fit.parameters));
    let report = InferenceReport {
        schema_version: "roof-inference-report/v7".to_owned(),
        image: args.image.display().to_string(),
        model: args.model.display().to_string(),
        backend: format!("{:?}", args.backend).to_lowercase(),
        backend_parity_verified: args.verify_backend_parity,
        keypoint_threshold,
        offscreen_threshold,
        source_width: width,
        source_height: height,
        camera_metadata: CameraMetadataReport {
            focal_length_35mm: exif.focal_length_35mm,
            horizontal_fov_degrees,
            focal_source: if horizontal_fov_degrees.is_some() {
                "exif_35mm_equivalent".to_owned()
            } else {
                "estimated".to_owned()
            },
        },
        accepted_keypoint_count,
        detection,
        fit_attempted,
        inferred_roof_ratios,
        fit,
        fit_error,
    };
    fs::write(&json, serde_json::to_vec_pretty(&report)?)
        .with_context(|| format!("write prediction JSON {}", json.display()))?;

    let fit_status = report.fit.as_ref().map_or_else(
        || {
            report
                .fit_error
                .as_ref()
                .map_or_else(|| "unavailable".to_owned(), |error| format!("{error:?}"))
        },
        |fit| {
            format!(
                "accepted={} rmse={:.4} confidence={:.3}",
                fit.confidence.accepted, fit.reprojection_rmse, fit.confidence.score
            )
        },
    );
    println!(
        "probability={:.4} detected={} keypoints={} fit=({}) overlay={} json={}",
        report.detection.probability,
        report.detection.detected,
        report.accepted_keypoint_count,
        fit_status,
        output.display(),
        json.display()
    );
    Ok(())
}

fn resolve_thresholds(
    presence_override: Option<f32>,
    offscreen_override: Option<f32>,
    keypoint_override: Option<f32>,
    manifest: Option<&ModelManifest>,
) -> (f32, f32, f32) {
    let presence = presence_override
        .or_else(|| manifest.and_then(|manifest| manifest.recommended_presence_threshold))
        .unwrap_or(0.5);
    let offscreen = offscreen_override
        .or_else(|| manifest.and_then(|manifest| manifest.recommended_offscreen_threshold))
        .unwrap_or(0.5);
    let keypoint = keypoint_override
        .or_else(|| manifest.and_then(|manifest| manifest.recommended_keypoint_threshold))
        .unwrap_or(DEFAULT_FIT_KEYPOINT_CONFIDENCE);
    (presence, offscreen, keypoint)
}

fn fit_detection(
    detection: &KeypointRoofDetection,
    observation: &SingleViewObservation,
    config: SingleViewFitConfig,
    force_diagnostic_fit: bool,
) -> (Option<SingleViewRoofFit>, Option<FitFailure>) {
    if !detection.detected && !force_diagnostic_fit {
        return (
            None,
            Some(FitFailure::DetectionRejected {
                message: "fit skipped because roof presence was below threshold".to_owned(),
                probability: detection.probability,
                threshold: detection.threshold,
            }),
        );
    }
    match fit_single_view(observation, config) {
        Ok(fit) => (Some(fit), None),
        Err(error) => (None, Some(FitFailure::from(&error))),
    }
}

fn ensure_model_exists(model_path: &Path) -> Result<()> {
    let checkpoint = checkpoint_path(model_path);
    if checkpoint.is_file() {
        return Ok(());
    }
    bail!(
        "keypoint model checkpoint does not exist: {}\ntrain it with `cargo run --release -p roof-train -- --artifacts {}` or pass `--model <checkpoint-base>`",
        checkpoint.display(),
        model_path
            .parent()
            .unwrap_or_else(|| Path::new("artifacts/roof-model-keypoints"))
            .display()
    )
}

fn checkpoint_path(model_path: &Path) -> PathBuf {
    let mut checkpoint = model_path.to_path_buf();
    checkpoint.set_extension("mpk");
    checkpoint
}

fn model_manifest_path(model_path: &Path) -> PathBuf {
    model_path.with_file_name("model.json")
}

fn load_model_manifest(manifest_path: &Path) -> Result<Option<ModelManifest>> {
    if !manifest_path.exists() {
        return Ok(None);
    }
    let bytes = fs::read(manifest_path)
        .with_context(|| format!("read model manifest {}", manifest_path.display()))?;
    let manifest: ModelManifest = serde_json::from_slice(&bytes)
        .with_context(|| format!("parse model manifest {}", manifest_path.display()))?;
    Ok(Some(manifest))
}

fn validate_model_manifest(manifest: &ModelManifest, manifest_path: &Path) -> Result<()> {
    if manifest.schema_version != MODEL_MANIFEST_SCHEMA {
        bail!(
            "model manifest {} uses {}, expected {} for the current keypoint network",
            manifest_path.display(),
            manifest.schema_version,
            MODEL_MANIFEST_SCHEMA
        );
    }
    if manifest.input_size != SPATIAL_INPUT_SIZE
        || manifest.heatmap_size != HEATMAP_SIZE
        || manifest.keypoint_count != KEYPOINT_COUNT
    {
        bail!(
            "model manifest {} describes input/heatmap/keypoints {}/{}/{}, expected {}/{}/{}",
            manifest_path.display(),
            manifest.input_size,
            manifest.heatmap_size,
            manifest.keypoint_count,
            SPATIAL_INPUT_SIZE,
            HEATMAP_SIZE,
            KEYPOINT_COUNT
        );
    }
    Ok(())
}

fn infer<B: Backend>(
    model_path: &Path,
    chw: Vec<f32>,
    transform: roof_model::LetterboxTransform,
    presence_threshold: f32,
    offscreen_threshold: f32,
    device: B::Device,
) -> Result<KeypointRoofDetection> {
    let record = CompactRecorder::new()
        .load(model_path.to_path_buf(), &device)
        .with_context(|| {
            format!(
                "load keypoint model {}",
                checkpoint_path(model_path).display()
            )
        })?;
    let model = KeypointRoofNetConfig::new()
        .init::<B>(&device)
        .load_record(record);
    let input = Tensor::<B, 4>::from_data(
        TensorData::new(
            chw,
            Shape::new([1, 3, SPATIAL_INPUT_SIZE, SPATIAL_INPUT_SIZE]),
        ),
        &device,
    );
    Ok(decode_keypoint_prediction(
        model.forward(input),
        transform,
        presence_threshold,
        offscreen_threshold,
    ))
}

fn keypoint_observation(
    detection: &KeypointRoofDetection,
    keypoint_threshold: f32,
) -> SingleViewObservation {
    let mut observation = SingleViewObservation::default();
    for (ring_index, ring) in detection.rings.iter().enumerate() {
        for point in &ring.points {
            let index = ring_index * 4 + point.slot;
            let Some(position) = point.position else {
                continue;
            };
            if point.offscreen
                || index >= KEYPOINT_COUNT
                || point.confidence < keypoint_threshold
                || !point.confidence.is_finite()
                || !position.iter().all(|coordinate| coordinate.is_finite())
            {
                continue;
            }
            observation.keypoints[index] = Some(KeypointObservation {
                position,
                weight: point.confidence,
            });
        }
    }
    observation
}

fn verify_backend_parity(wgpu: &KeypointRoofDetection, flex: &KeypointRoofDetection) -> Result<()> {
    const SCALAR_TOLERANCE: f32 = 2.0e-3;
    const POSITION_TOLERANCE: f32 = 1.0e-3;

    let close = |left: f32, right: f32, tolerance: f32| {
        left.is_finite() && right.is_finite() && (left - right).abs() <= tolerance
    };
    if !close(wgpu.probability, flex.probability, SCALAR_TOLERANCE) {
        bail!(
            "WGPU/Flex presence mismatch: {:.6} versus {:.6}",
            wgpu.probability,
            flex.probability
        );
    }
    for ring_index in 0..wgpu.rings.len() {
        for slot in 0..wgpu.rings[ring_index].points.len() {
            let left = &wgpu.rings[ring_index].points[slot];
            let right = &flex.rings[ring_index].points[slot];
            if left.offscreen != right.offscreen
                || left.position.is_some() != right.position.is_some()
            {
                bail!("WGPU/Flex state mismatch at ring {ring_index}, slot {slot}");
            }
            if !close(left.confidence, right.confidence, SCALAR_TOLERANCE)
                || !close(
                    left.offscreen_probability,
                    right.offscreen_probability,
                    SCALAR_TOLERANCE,
                )
            {
                bail!("WGPU/Flex confidence mismatch at ring {ring_index}, slot {slot}");
            }
            if let (Some(left), Some(right)) = (left.position, right.position)
                && (!close(left[0], right[0], POSITION_TOLERANCE)
                    || !close(left[1], right[1], POSITION_TOLERANCE))
            {
                bail!("WGPU/Flex position mismatch at ring {ring_index}, slot {slot}");
            }
        }
    }
    Ok(())
}

fn sibling_path(input: &Path, suffix: &str, extension: &str) -> PathBuf {
    let stem = input
        .file_stem()
        .and_then(|value| value.to_str())
        .unwrap_or("roof");
    input.with_file_name(format!("{stem}-{suffix}.{extension}"))
}

fn create_parent_directory(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create output directory {}", parent.display()))?;
    }
    Ok(())
}

fn read_exif_metadata(bytes: &[u8]) -> ExifMetadata {
    let Ok(exif) = Reader::new().read_from_container(&mut Cursor::new(bytes)) else {
        return ExifMetadata {
            orientation: 1,
            focal_length_35mm: None,
        };
    };
    let orientation = exif
        .get_field(Tag::Orientation, In::PRIMARY)
        .and_then(|field| field.value.get_uint(0))
        .unwrap_or(1);
    let focal_length_35mm = exif
        .get_field(Tag::FocalLengthIn35mmFilm, In::PRIMARY)
        .and_then(|field| field.value.get_uint(0))
        .map(|value| value as f32)
        .filter(|value| value.is_finite() && *value > 0.0);
    ExifMetadata {
        orientation,
        focal_length_35mm,
    }
}

fn horizontal_fov_from_35mm(focal_length_35mm: f32, width: u32, height: u32) -> Option<f32> {
    if !focal_length_35mm.is_finite() || focal_length_35mm <= 0.0 || width == 0 || height == 0 {
        return None;
    }
    // A 35 mm-equivalent focal length preserves diagonal field of view. Derive
    // the equivalent sensor width for this image's post-orientation aspect
    // ratio; this gives exactly 36 mm for the conventional 3:2 frame.
    const FULL_FRAME_DIAGONAL_MM: f32 = 43.266_617;
    let aspect = width as f32 / height as f32;
    let equivalent_sensor_width =
        FULL_FRAME_DIAGONAL_MM * aspect / (aspect.mul_add(aspect, 1.0)).sqrt();
    let fov = 2.0 * (equivalent_sensor_width / (2.0 * focal_length_35mm)).atan();
    let degrees = fov.to_degrees();
    (degrees.is_finite() && (5.0..175.0).contains(&degrees)).then_some(degrees)
}

fn orient_image(image: DynamicImage, orientation: u32) -> DynamicImage {
    match orientation {
        2 => image.fliph(),
        3 => image.rotate180(),
        4 => image.flipv(),
        5 => image.rotate90().fliph(),
        6 => image.rotate90(),
        7 => image.rotate270().fliph(),
        8 => image.rotate270(),
        _ => image,
    }
}

fn render_overlay(
    mut base: RgbaImage,
    detection: &KeypointRoofDetection,
    fit: Option<&SingleViewRoofFit>,
    keypoint_threshold: f32,
    show_all: bool,
    raw_keypoint_debug: bool,
) -> RgbaImage {
    if let Some(fit) = fit
        && should_draw_fitted_mesh(detection, fit, show_all)
    {
        draw_fitted_mesh(&mut base, fit);
    }
    if raw_keypoint_debug {
        let (width, height) = base.dimensions();
        let mut debug = RgbaImage::new(width, height);
        draw_keypoint_debug(&mut debug, detection, keypoint_threshold, width, height);
        imageops::overlay(&mut base, &debug, 0, 0);
    }
    base
}

fn should_draw_fitted_mesh(
    detection: &KeypointRoofDetection,
    fit: &SingleViewRoofFit,
    show_all: bool,
) -> bool {
    show_all
        || (detection.detected
            && fit.confidence.accepted
            && fit.observation_count >= MINIMUM_CONFIDENT_FIT_OBSERVATIONS)
}

fn draw_fitted_mesh(base: &mut RgbaImage, fit: &SingleViewRoofFit) {
    let (width, height) = base.dimensions();
    let roof = generate_roof(&fit.parameters).expect("fitted roof parameters must be valid");

    let mut fill = RgbaImage::new(width, height);
    let mut triangles = Vec::new();
    for face in &fit.projected_mesh.faces {
        let start = face.first_index as usize;
        let end = start + face.index_count as usize;
        for triangle in fit.projected_mesh.indices[start..end].chunks_exact(3) {
            let project = |index: u32| {
                let projected = fit.projected_mesh.vertices[index as usize];
                (
                    projected.position[0] * width as f32,
                    projected.position[1] * height as f32,
                )
            };
            let points = [
                project(triangle[0]),
                project(triangle[1]),
                project(triangle[2]),
            ];
            let mean_depth = triangle
                .iter()
                .map(|index| fit.projected_mesh.vertices[*index as usize].depth)
                .sum::<f32>()
                / 3.0;
            triangles.push(ProjectedTriangle {
                points,
                mean_depth,
                color: face_color(face.class),
            });
        }
    }
    triangles.sort_by(|left, right| right.mean_depth.total_cmp(&left.mean_depth));
    for triangle in triangles {
        draw_triangle(&mut fill, triangle.points, triangle.color);
    }
    imageops::overlay(base, &fill, 0, 0);

    let mut lines = RgbaImage::new(width, height);
    for edge in &roof.edges {
        let (start, end) = roof
            .edge_positions(edge.id)
            .expect("generated structural edge must resolve");
        let start = fit.camera.project(start);
        let end = fit.camera.project(end);
        draw_thick_line(
            &mut lines,
            (start[0] * width as f32, start[1] * height as f32),
            (end[0] * width as f32, end[1] * height as f32),
            edge_color(edge.category),
        );
    }
    for (index, point) in fit.projected_keypoints.iter().enumerate() {
        let (x, y) = (point[0] * width as f32, point[1] * height as f32);
        draw_filled_circle_mut(
            &mut lines,
            (x.round() as i32, y.round() as i32),
            5,
            ring_color(index / 4, 255),
        );
    }
    draw_fit_bounds(&mut lines, fit, width, height);
    imageops::overlay(base, &lines, 0, 0);
}

struct ProjectedTriangle {
    points: [(f32, f32); 3],
    mean_depth: f32,
    color: Rgba<u8>,
}

fn draw_triangle(image: &mut RgbaImage, points: [(f32, f32); 3], color: Rgba<u8>) {
    let polygon = points.map(|point| Point::new(point.0.round() as i32, point.1.round() as i32));
    draw_polygon_mut(image, &polygon, color);
}

fn face_color(class: FaceClass) -> Rgba<u8> {
    match class {
        FaceClass::LowerSkirt => Rgba([0, 215, 255, 38]),
        FaceClass::UpperCrown => Rgba([255, 72, 150, 48]),
        FaceClass::Underside => Rgba([255, 205, 0, 24]),
    }
}

fn edge_color(category: EdgeCategory) -> Rgba<u8> {
    match category {
        EdgeCategory::Eave => Rgba([0, 245, 255, 240]),
        EdgeCategory::ShoulderBreak => Rgba([255, 72, 150, 240]),
        EdgeCategory::CrownTopPerimeter => Rgba([255, 215, 0, 245]),
        EdgeCategory::LowerHip | EdgeCategory::UpperHip => Rgba([215, 245, 255, 220]),
    }
}

fn ring_color(ring_index: usize, alpha: u8) -> Rgba<u8> {
    match ring_index {
        0 => Rgba([0, 245, 255, alpha]),
        1 => Rgba([255, 72, 150, alpha]),
        _ => Rgba([255, 215, 0, alpha]),
    }
}

fn draw_thick_line(image: &mut RgbaImage, start: (f32, f32), end: (f32, f32), color: Rgba<u8>) {
    for offset in -1..=1 {
        let offset = offset as f32;
        draw_line_segment_mut(
            image,
            (start.0 + offset, start.1),
            (end.0 + offset, end.1),
            color,
        );
        draw_line_segment_mut(
            image,
            (start.0, start.1 + offset),
            (end.0, end.1 + offset),
            color,
        );
    }
}

fn draw_fit_bounds(image: &mut RgbaImage, fit: &SingleViewRoofFit, width: u32, height: u32) {
    let min = (
        fit.bounding_box.min[0] * width as f32,
        fit.bounding_box.min[1] * height as f32,
    );
    let max = (
        fit.bounding_box.max[0] * width as f32,
        fit.bounding_box.max[1] * height as f32,
    );
    let color = Rgba([110, 255, 120, 225]);
    draw_thick_line(image, min, (max.0, min.1), color);
    draw_thick_line(image, (max.0, min.1), max, color);
    draw_thick_line(image, max, (min.0, max.1), color);
    draw_thick_line(image, (min.0, max.1), min, color);
}

fn draw_keypoint_debug(
    image: &mut RgbaImage,
    detection: &KeypointRoofDetection,
    keypoint_threshold: f32,
    width: u32,
    height: u32,
) {
    if let Some(bounds) = detection.bounding_box {
        let min = (bounds.min_x * width as f32, bounds.min_y * height as f32);
        let max = (bounds.max_x * width as f32, bounds.max_y * height as f32);
        let color = Rgba([255, 255, 255, 150]);
        draw_thick_line(image, min, (max.0, min.1), color);
        draw_thick_line(image, (max.0, min.1), max, color);
        draw_thick_line(image, max, (min.0, max.1), color);
        draw_thick_line(image, (min.0, max.1), min, color);
    }
    for (ring_index, ring) in detection.rings.iter().enumerate() {
        for point in &ring.points {
            let Some(position) = point.position else {
                continue;
            };
            let alpha = if point.confidence >= keypoint_threshold {
                255
            } else {
                90
            };
            draw_filled_circle_mut(
                image,
                (
                    (position[0] * width as f32).round() as i32,
                    (position[1] * height as f32).round() as i32,
                ),
                3,
                ring_color(ring_index, alpha),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use roof_fit::{FitConfidence, FittedBounds, PerspectiveCamera, ProjectedMesh};
    use roof_model::{AmodalKeypointPrediction, RoofRing, RoofRingPrediction};

    fn detection_with_confidences(confidences: [f32; KEYPOINT_COUNT]) -> KeypointRoofDetection {
        KeypointRoofDetection {
            schema_version: "test".to_owned(),
            probability: 0.9,
            threshold: 0.5,
            detected: true,
            bounding_box: None,
            rings: std::array::from_fn(|ring_index| RoofRingPrediction {
                ring: RoofRing::ALL[ring_index],
                points: std::array::from_fn(|slot| {
                    let index = ring_index * 4 + slot;
                    AmodalKeypointPrediction {
                        slot,
                        position: Some([index as f32 / 20.0, index as f32 / 30.0]),
                        confidence: confidences[index],
                        offscreen_probability: 0.0,
                        offscreen: false,
                    }
                }),
            }),
        }
    }

    fn fitted_roof(observation_count: usize, accepted: bool) -> SingleViewRoofFit {
        SingleViewRoofFit {
            schema_version: "test".to_owned(),
            parameters: RoofParameters {
                eave_width: 1.0,
                eave_depth: 0.8,
                shoulder_width: 0.6,
                shoulder_depth: 0.4,
                crown_top_width: 0.48,
                crown_top_depth: 0.28,
                lower_rise: 0.12,
                upper_rise: 0.16,
            },
            camera: PerspectiveCamera {
                focal_length: [1.0, 1.0],
                principal_point: [0.5, 0.5],
                translation: [0.0, 0.0, 3.0],
                yaw_radians: 0.0,
                pitch_radians: 0.0,
                roll_radians: 0.0,
                corner_shift: 0,
                reflected: false,
            },
            projected_keypoints: [[0.5, 0.5]; KEYPOINT_COUNT],
            projected_mesh: ProjectedMesh {
                vertices: Vec::new(),
                indices: Vec::new(),
                faces: Vec::new(),
            },
            bounding_box: FittedBounds {
                min: [0.2, 0.2],
                max: [0.8, 0.8],
            },
            reprojection_rmse: 0.01,
            observation_count,
            confidence: FitConfidence {
                score: if accepted { 0.9 } else { 0.1 },
                accepted,
                inlier_count: observation_count,
            },
        }
    }

    #[test]
    fn image_only_invocation_uses_promoted_keypoint_checkpoint_defaults() {
        let args = Args::try_parse_from(["roof-detect", "photo.jpg"]).unwrap();
        assert_eq!(args.image, PathBuf::from("photo.jpg"));
        assert_eq!(
            args.model,
            PathBuf::from("artifacts/roof-model-keypoints/model")
        );
        assert_eq!(args.backend, BackendChoice::Wgpu);
        assert_eq!(args.threshold, None);
        assert_eq!(args.offscreen_threshold, None);
        assert_eq!(args.keypoint_threshold, None);
        assert_eq!(args.output, None);
        assert_eq!(args.json, None);
    }

    #[test]
    fn sibling_outputs_remain_beside_input() {
        assert_eq!(
            sibling_path(Path::new("/tmp/roof.jpg"), "overlay", "png"),
            PathBuf::from("/tmp/roof-overlay.png")
        );
    }

    #[test]
    fn checkpoint_extension_is_normalized() {
        assert_eq!(
            checkpoint_path(Path::new("artifacts/roof-model-keypoints/model")),
            PathBuf::from("artifacts/roof-model-keypoints/model.mpk")
        );
        assert_eq!(
            checkpoint_path(Path::new("artifacts/roof-model-keypoints/model.json")),
            PathBuf::from("artifacts/roof-model-keypoints/model.mpk")
        );
    }

    #[test]
    fn manifest_must_describe_the_current_keypoint_tensor_contract() {
        let current: ModelManifest = serde_json::from_value(serde_json::json!({
            "schema_version": MODEL_MANIFEST_SCHEMA,
            "input_size": SPATIAL_INPUT_SIZE,
            "heatmap_size": HEATMAP_SIZE,
            "keypoint_count": KEYPOINT_COUNT,
            "recommended_presence_threshold": 0.73,
            "recommended_offscreen_threshold": 0.61,
            "recommended_keypoint_threshold": 0.19
        }))
        .unwrap();
        validate_model_manifest(&current, Path::new("model.json")).unwrap();

        let stale = ModelManifest {
            heatmap_size: 32,
            ..current
        };
        let error = validate_model_manifest(&stale, Path::new("model.json"))
            .unwrap_err()
            .to_string();
        assert!(error.contains("expected 256/64/12"), "{error}");
    }

    #[test]
    fn observations_preserve_three_ring_slot_order_and_threshold_confidence() {
        let mut confidences = [0.9; KEYPOINT_COUNT];
        confidences[5] = 0.1;
        let mut detection = detection_with_confidences(confidences);
        detection.rings[2].points[3].position = None;
        detection.rings[2].points[3].offscreen = true;

        let observation = keypoint_observation(&detection, 0.5);
        assert_eq!(observation.keypoints.iter().flatten().count(), 10);
        assert_eq!(
            observation.keypoints[6].unwrap().position,
            [6.0 / 20.0, 6.0 / 30.0]
        );
        assert_eq!(observation.keypoints[5], None);
        assert_eq!(observation.keypoints[11], None);
    }

    #[test]
    fn backend_parity_check_rejects_material_observation_drift() {
        let wgpu = detection_with_confidences([0.8; KEYPOINT_COUNT]);
        let mut flex = wgpu.clone();
        verify_backend_parity(&wgpu, &flex).unwrap();

        flex.rings[1].points[2].position = Some([0.9, 0.9]);
        let error = verify_backend_parity(&wgpu, &flex).unwrap_err().to_string();
        assert!(error.contains("position mismatch"), "{error}");
    }

    #[test]
    fn horizontal_fov_respects_35mm_equivalent_and_output_aspect() {
        let landscape = horizontal_fov_from_35mm(36.0, 3_000, 2_000).unwrap();
        let portrait = horizontal_fov_from_35mm(36.0, 2_000, 3_000).unwrap();
        assert!((landscape - 53.130_1).abs() < 0.001);
        assert!(portrait < landscape);
        assert_eq!(horizontal_fov_from_35mm(0.0, 3_000, 2_000), None);
    }

    #[test]
    fn fit_failure_keeps_machine_readable_observation_counts() {
        let error = FitError::InsufficientObservations {
            minimum: 6,
            actual: 4,
        };
        let failure = FitFailure::from(&error);
        let json = serde_json::to_value(failure).unwrap();
        assert_eq!(json["kind"], "insufficient_observations");
        assert_eq!(json["minimum"], 6);
        assert_eq!(json["actual"], 4);
    }

    #[test]
    fn manifest_thresholds_are_defaults_but_cli_values_override_them() {
        let manifest: ModelManifest = serde_json::from_str(
            r#"{
                "schema_version": "roof-keypoint-model/v3",
                "input_size": 256,
                "heatmap_size": 64,
                "keypoint_count": 12,
                "recommended_presence_threshold": 0.73,
                "recommended_offscreen_threshold": 0.61,
                "recommended_keypoint_threshold": 0.19
            }"#,
        )
        .unwrap();
        assert_eq!(
            resolve_thresholds(None, None, None, Some(&manifest)),
            (0.73, 0.61, 0.19)
        );
        assert_eq!(
            resolve_thresholds(Some(0.8), Some(0.4), Some(0.12), Some(&manifest)),
            (0.8, 0.4, 0.12)
        );
        assert_eq!(
            resolve_thresholds(None, None, None, None),
            (0.5, 0.5, DEFAULT_FIT_KEYPOINT_CONFIDENCE)
        );
    }

    #[test]
    fn rejected_detection_skips_fit_with_a_structured_reason() {
        let mut detection = detection_with_confidences([0.9; KEYPOINT_COUNT]);
        detection.detected = false;
        detection.probability = 0.2;
        let (fit, failure) = fit_detection(
            &detection,
            &SingleViewObservation::default(),
            SingleViewFitConfig::default(),
            false,
        );
        assert!(fit.is_none());
        let json = serde_json::to_value(failure.unwrap()).unwrap();
        assert_eq!(json["kind"], "detection_rejected");
        assert!((json["probability"].as_f64().unwrap() - 0.2).abs() < 1.0e-6);
        assert!((json["threshold"].as_f64().unwrap() - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn show_all_explicitly_attempts_diagnostic_fit() {
        let mut detection = detection_with_confidences([0.9; KEYPOINT_COUNT]);
        detection.detected = false;
        let (fit, failure) = fit_detection(
            &detection,
            &SingleViewObservation::default(),
            SingleViewFitConfig::default(),
            true,
        );
        assert!(fit.is_none());
        assert!(matches!(
            failure,
            Some(FitFailure::InsufficientObservations { .. })
        ));
    }

    #[test]
    fn raw_debug_can_be_drawn_without_an_accepted_detection_or_fit() {
        let mut detection = detection_with_confidences([0.9; KEYPOINT_COUNT]);
        detection.detected = false;
        let image = RgbaImage::new(64, 64);
        let rendered = render_overlay(image, &detection, None, 0.5, false, true);
        assert!(rendered.pixels().any(|pixel| pixel.0[3] != 0));
    }

    #[test]
    fn confident_overlay_requires_detection_acceptance_and_six_observations() {
        let mut detection = detection_with_confidences([0.9; KEYPOINT_COUNT]);
        let underconstrained = fitted_roof(5, true);
        let rejected = fitted_roof(12, false);
        let accepted = fitted_roof(6, true);

        assert!(!should_draw_fitted_mesh(
            &detection,
            &underconstrained,
            false
        ));
        assert!(!should_draw_fitted_mesh(&detection, &rejected, false));
        assert!(should_draw_fitted_mesh(&detection, &accepted, false));

        detection.detected = false;
        assert!(!should_draw_fitted_mesh(&detection, &accepted, false));
        assert!(should_draw_fitted_mesh(&detection, &underconstrained, true));

        let base = RgbaImage::from_pixel(32, 32, Rgba([7, 11, 13, 255]));
        let rendered = render_overlay(
            base.clone(),
            &detection_with_confidences([0.9; KEYPOINT_COUNT]),
            Some(&underconstrained),
            0.5,
            false,
            false,
        );
        assert_eq!(rendered, base);
    }

    #[test]
    fn report_exposes_observations_ratios_camera_mesh_and_confidence() {
        let detection = detection_with_confidences([0.9; KEYPOINT_COUNT]);
        let fit = fitted_roof(12, true);
        let ratios = InferredRoofRatios::from(&fit.parameters);
        let report = InferenceReport {
            schema_version: "roof-inference-report/v7".to_owned(),
            image: "photo.jpg".to_owned(),
            model: "artifacts/roof-model-keypoints/model".to_owned(),
            backend: "wgpu".to_owned(),
            backend_parity_verified: false,
            keypoint_threshold: 0.15,
            offscreen_threshold: 0.5,
            source_width: 640,
            source_height: 480,
            camera_metadata: CameraMetadataReport {
                focal_length_35mm: None,
                horizontal_fov_degrees: None,
                focal_source: "estimated".to_owned(),
            },
            accepted_keypoint_count: 12,
            detection,
            fit_attempted: true,
            inferred_roof_ratios: Some(ratios),
            fit: Some(fit),
            fit_error: None,
        };
        let json = serde_json::to_value(report).unwrap();
        assert_eq!(json["detection"]["rings"].as_array().unwrap().len(), 3);
        assert_eq!(json["inferred_roof_ratios"].as_object().unwrap().len(), 7);
        let eave_depth_ratio = json["inferred_roof_ratios"]["eave_depth_to_eave_width"]
            .as_f64()
            .unwrap();
        assert!((eave_depth_ratio - 0.8).abs() < 1.0e-6);
        assert!(json["fit"]["camera"].is_object());
        assert!(json["fit"]["projected_mesh"].is_object());
        assert_eq!(json["fit"]["confidence"]["accepted"], true);
    }
}
