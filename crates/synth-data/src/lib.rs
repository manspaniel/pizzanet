//! Reproducible scene plans and portable annotation records for synthetic data.
//!
//! This crate deliberately contains no renderer, GPU, filesystem, or browser code. A
//! producer samples a [`SequencePlan`] from a [`GeneratorConfig`], renders the plan,
//! and stores the result as [`SequenceRecord`] and [`FrameRecord`] values. Keeping
//! those contracts independent makes the same records usable by native generation,
//! training loaders, replay tools, and a future WASM build.

mod appearance;
mod composition;
mod config;
mod deterministic;
mod manifest;
mod record;
mod sampling;
mod split;
mod types;
mod validation;

pub use appearance::{
    FrameAppearance, PHOTOMETRIC_PROFILE_VERSION, PhotometricProfile,
    PhotometricProfileValidationError,
};
pub use composition::{
    BackgroundBuildingKind, BackgroundBuildingSamplingConfig, BuildingExtensionKind,
    BuildingExtensionRoof, BuildingExtensionSamplingConfig, CompositionSamplingConfig, DayPhase,
    DayPhaseProfile, DayPhaseSamplingConfig, FacadeSamplingConfig, FacadeSide, RoadKind,
    SampledBackgroundBuilding, SampledBuildingExtension, SampledComposition, SampledEnvironment,
    SampledFacade, SampledSignage, SampledSiteInfrastructure, SampledUtilityLine,
    SampledUtilityPole, SampledVegetation, SceneDomain, SceneDomainProfile,
    SceneDomainSamplingConfig, SignageKind, SignageSamplingConfig, VegetationKind,
    VegetationSamplingConfig, WeatherPreset, WeatherProfile, WeatherSamplingConfig,
};
pub use config::{
    CameraSamplingConfig, GeneratorConfig, ImageConfig, LightingSamplingConfig, MaterialChoice,
    MaterialSamplingConfig, OccluderChoice, OccluderKind, OccluderSamplingConfig, RoofMorphology,
    RoofMorphologyProfile, RoofSamplingConfig, SceneSamplingConfig, SequenceConfig,
};
pub use manifest::{
    CoordinateSystem, DATASET_SCHEMA_VERSION, DatasetManifest, DatasetStatistics,
    GeneratorDescriptor, LabelClass, LabelTaxonomy, OutputDefinition, SourceAsset, SourceAssetKind,
};
pub use record::{
    AssetRef, CameraIntrinsics, CameraModel, DenseLabelRefs, DistortionModel, EdgeLabel,
    EdgeVisibility, FrameAssets, FrameIdentity, FrameRecord, ImageTransform, KeypointLabel,
    LocatorLabel, NormalizedBoundingBox, RoofInstanceRecord, SequenceFrameRef, SequenceRecord,
    StructuralLabels, TargetKind, Visibility,
};
pub use sampling::{
    ApparentScale, CameraFramePlan, CameraMotionPlan, CameraPathKind, FramingIntent,
    OccluderPlacement, OrdinaryRoofFamily, SampledBuilding, SampledLighting, SampledMaterial,
    SampledOccluder, SampledOrdinaryRoof, SampledRoof, SampledScene, SamplingError, SequencePlan,
    SequenceRequest, SequenceSampler, ZoomBehavior,
};
pub use split::{DatasetSplit, SplitKey, SplitPolicy, SplitPolicyError};
pub use types::{FloatRange, RigidTransform, U32Range, Vec2, Vec3};
pub use validation::{DatasetValidator, Severity, Validate, ValidationIssue, ValidationReport};
