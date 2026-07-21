use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{
    CameraMotionPlan, DATASET_SCHEMA_VERSION, DatasetSplit, FrameAppearance, RigidTransform,
    SampledScene, Vec2, Vec3,
};

/// Whether a scene is a recognized target, a deliberate near miss, or empty of it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetKind {
    /// Accepted classic roof family.
    Target,
    /// Structurally similar roof outside the accepted family.
    NearMiss,
    /// Scene without a relevant roof.
    Negative,
}

/// Portable reference to a file grouped under a WebDataset sample key.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssetRef {
    /// Relative archive member name or dataset-relative path.
    pub path: String,
    /// Internet media type, such as `image/jpeg`.
    pub media_type: String,
    /// Codec or scalar layout, such as `jpeg` or `rg16float-zstd`.
    pub encoding: String,
    /// Optional content digest including algorithm prefix.
    pub content_hash: Option<String>,
}

impl AssetRef {
    /// Creates an unhashed asset reference. Writers should fill the digest before
    /// accepting the final shard.
    #[must_use]
    pub fn new(
        path: impl Into<String>,
        media_type: impl Into<String>,
        encoding: impl Into<String>,
    ) -> Self {
        Self {
            path: path.into(),
            media_type: media_type.into(),
            encoding: encoding.into(),
            content_hash: None,
        }
    }
}

/// Camera intrinsic matrix in pixels.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CameraIntrinsics {
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// Horizontal focal length.
    pub fx: f32,
    /// Vertical focal length.
    pub fy: f32,
    /// Horizontal principal point.
    pub cx: f32,
    /// Vertical principal point.
    pub cy: f32,
    /// Intrinsic skew, normally zero.
    pub skew: f32,
}

/// Lens distortion applied to the stored image and labels.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "model", rename_all = "snake_case")]
pub enum DistortionModel {
    /// Ideal pinhole camera.
    None,
    /// Brown-Conrady radial and tangential distortion.
    BrownConrady {
        /// First radial coefficient.
        k1: f32,
        /// Second radial coefficient.
        k2: f32,
        /// First tangential coefficient.
        p1: f32,
        /// Second tangential coefficient.
        p2: f32,
        /// Third radial coefficient.
        k3: f32,
    },
}

/// Row-major homogeneous mapping between two image-coordinate spaces.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ImageTransform(pub [f32; 9]);

impl ImageTransform {
    /// Identity image mapping.
    pub const IDENTITY: Self = Self([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
}

/// Camera pose, calibration, and geometric augmentation used for one frame.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CameraModel {
    /// Exact intrinsics for the stored output image.
    pub intrinsics: CameraIntrinsics,
    /// Lens model already reflected in the image and annotations.
    pub distortion: DistortionModel,
    /// Maps camera coordinates into the sampled scene world.
    pub world_from_camera: RigidTransform,
    /// Sensor-image to stored-output mapping after crop, rotation, and resize.
    pub output_from_sensor: ImageTransform,
}

/// Visibility state for a projected structural point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Visibility {
    /// Projected inside the frame and passes the scene depth test.
    Visible,
    /// Projected inside the frame but hidden by scene geometry.
    Occluded,
    /// Projection lies outside the retained image crop.
    Truncated,
    /// Point has non-positive camera-space depth.
    BehindCamera,
}

/// Compact projected structural point; names live in the manifest taxonomy.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeypointLabel {
    /// Dataset-wide keypoint class identifier.
    pub class_id: u16,
    /// Stable instance identifier within the roof geometry.
    pub instance_id: u16,
    /// Exact procedural point in roof-local metres.
    pub roof_position: Vec3,
    /// Projected normalized position, where the top-left is `(0, 0)`.
    pub image_position: Option<Vec2>,
    /// Occlusion and framing result.
    pub visibility: Visibility,
}

/// Visibility summary for a structural polyline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EdgeVisibility {
    /// Entire retained polyline passes depth testing.
    Visible,
    /// Retained polyline contains both visible and hidden spans.
    PartiallyOccluded,
    /// Projected polyline lies in the image but is hidden.
    Occluded,
    /// Projection crosses or lies outside an image boundary.
    Truncated,
    /// Edge lies behind the camera.
    BehindCamera,
}

/// Projected semantic roof edge.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EdgeLabel {
    /// Dataset-wide edge class identifier.
    pub class_id: u16,
    /// Stable edge identifier within the roof geometry.
    pub instance_id: u16,
    /// Clipped normalized image polyline. Tangents are derived when loading.
    pub polyline: Vec<Vec2>,
    /// Aggregate visibility of the projected edge.
    pub visibility: EdgeVisibility,
}

/// Dense ground-truth files whose contents remain outside compact JSON.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DenseLabelRefs {
    /// Binary visible-roof silhouette after scene occlusion.
    pub roof_mask: Option<AssetRef>,
    /// Binary in-frame roof silhouette before scene occlusion.
    #[serde(default)]
    pub amodal_roof_mask: Option<AssetRef>,
    /// Integer semantic-part image.
    pub part_mask: Option<AssetRef>,
    /// Integer stable face-identity image.
    pub face_id_map: Option<AssetRef>,
    /// Per-pixel normalized coordinates within each roof face.
    pub face_coordinates: Option<AssetRef>,
}

/// All exact compact and dense structural supervision for one frame.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StructuralLabels {
    /// Procedurally projected structural points.
    pub keypoints: Vec<KeypointLabel>,
    /// Procedurally projected structural edges.
    pub edges: Vec<EdgeLabel>,
    /// References to exact raster labels.
    pub dense: DenseLabelRefs,
}

/// Bounding box in normalized image coordinates with top-left origin.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NormalizedBoundingBox {
    /// Minimum X and Y.
    pub min: Vec2,
    /// Maximum X and Y.
    pub max: Vec2,
}

/// Locator and sample-quality supervision.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocatorLabel {
    /// Target class for the full image.
    pub target_kind: TargetKind,
    /// Retained visible target bounds, absent for negative scenes.
    pub bounding_box: Option<NormalizedBoundingBox>,
    /// In-frame target bounds before scene occlusion, absent for negative scenes.
    #[serde(default)]
    pub amodal_bounding_box: Option<NormalizedBoundingBox>,
    /// Fraction of the in-frame roof-only raster visible in the complete scene.
    /// Crop loss is represented independently by `truncated`.
    pub visible_fraction: f32,
    /// Fraction of the in-frame roof-only raster hidden by other scene geometry.
    pub occluded_fraction: f32,
    /// Whether the roof silhouette intersects the output boundary.
    pub truncated: bool,
}

/// Ground-truth fitted roof instance for target and near-miss frames.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RoofInstanceRecord {
    /// Stable procedural family.
    pub family: String,
    /// Maps roof-local points into scene world space.
    pub world_from_roof: RigidTransform,
    /// Named geometry values in metres or documented unitless fractions.
    pub parameters: BTreeMap<String, f32>,
}

/// Non-label image files associated with a frame.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrameAssets {
    /// Model input image after camera effects.
    pub rgb: AssetRef,
    /// Optional renderer diagnostic, never a runtime model requirement.
    pub surface_normals: Option<AssetRef>,
    /// Optional motion vectors for tracking regression data.
    pub motion_vectors: Option<AssetRef>,
}

/// Identity and timing fields used when constructing a frame record.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrameIdentity {
    /// Basename shared by files in this WebDataset sample.
    pub sample_key: String,
    /// Parent coherent camera sequence.
    pub sequence_id: String,
    /// Zero-based index within the sequence.
    pub frame_index: u32,
    /// Nominal sequence time in nanoseconds.
    pub timestamp_ns: u64,
}

impl FrameIdentity {
    /// Creates a frame identity.
    #[must_use]
    pub fn new(
        sample_key: impl Into<String>,
        sequence_id: impl Into<String>,
        frame_index: u32,
        timestamp_ns: u64,
    ) -> Self {
        Self {
            sample_key: sample_key.into(),
            sequence_id: sequence_id.into(),
            frame_index,
            timestamp_ns,
        }
    }
}

/// Complete training annotation stored beside one rendered frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FrameRecord {
    /// Dataset schema version.
    pub schema_version: String,
    /// Basename shared by files in this WebDataset sample.
    pub sample_key: String,
    /// Parent coherent camera sequence.
    pub sequence_id: String,
    /// Zero-based index within the sequence.
    pub frame_index: u32,
    /// Nominal sequence time in nanoseconds.
    pub timestamp_ns: u64,
    /// Building-level split, repeated to permit frame-only loading.
    pub split: DatasetSplit,
    /// Exact camera used by RGB and every label pass.
    pub camera: CameraModel,
    /// Parametric roof instance, absent for empty negative scenes.
    pub roof: Option<RoofInstanceRecord>,
    /// Full-frame locator target.
    pub locator: LocatorLabel,
    /// Compact and dense structural targets.
    pub labels: StructuralLabels,
    /// RGB and optional diagnostic output references.
    pub assets: FrameAssets,
    /// Exact post-render appearance transform applied to stored RGB.
    #[serde(default, skip_serializing_if = "FrameAppearance::is_empty")]
    pub appearance: FrameAppearance,
}

impl FrameRecord {
    /// Creates a record with the current schema version.
    #[must_use]
    pub fn new(
        identity: FrameIdentity,
        split: DatasetSplit,
        camera: CameraModel,
        locator: LocatorLabel,
        assets: FrameAssets,
    ) -> Self {
        Self {
            schema_version: DATASET_SCHEMA_VERSION.to_owned(),
            sample_key: identity.sample_key,
            sequence_id: identity.sequence_id,
            frame_index: identity.frame_index,
            timestamp_ns: identity.timestamp_ns,
            split,
            camera,
            roof: None,
            locator,
            labels: StructuralLabels::default(),
            assets,
            appearance: FrameAppearance::default(),
        }
    }
}

/// Lightweight reference from a coherent sequence to one frame sample.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequenceFrameRef {
    /// WebDataset sample basename.
    pub sample_key: String,
    /// Zero-based frame index.
    pub frame_index: u32,
    /// Nominal sequence time in nanoseconds.
    pub timestamp_ns: u64,
}

/// Sequence-level record stored once per generated building and camera path.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SequenceRecord {
    /// Dataset schema version.
    pub schema_version: String,
    /// Stable sequence identity derived from family and building seed.
    pub sequence_id: String,
    /// Procedural family used as a split group.
    pub building_family: String,
    /// Building-level seed shared by every frame.
    pub building_seed: u64,
    /// Optional source-asset grouping key used during split assignment.
    pub source_asset_group: Option<String>,
    /// Stable split assigned before rendering.
    pub split: DatasetSplit,
    /// Target, near-miss, or negative generation intent.
    pub target_kind: TargetKind,
    /// Hash of the complete generator configuration.
    pub config_fingerprint: String,
    /// Fully sampled scene state needed for exact replay.
    pub scene: SampledScene,
    /// Coherent path, zoom, and framing intent behind the exact frame cameras.
    #[serde(default)]
    pub camera_motion: CameraMotionPlan,
    /// Ordered references to frame annotations and images.
    pub frames: Vec<SequenceFrameRef>,
}
