//! Compact spatial network for roof presence and amodal structural keypoints.

use std::collections::HashSet;

use burn::{
    module::{ParamId, list_param_ids},
    nn::{
        BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
        interpolate::{Interpolate2d, Interpolate2dConfig, InterpolateMode},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
    },
    prelude::*,
};

#[cfg(feature = "pretrained")]
use burn_store::PytorchStoreError;

use crate::mobilenet;

/// Number of rectangular structural rings predicted by the model.
pub const ROOF_RING_COUNT: usize = 3;
/// Number of cyclic corner slots in every structural ring.
pub const POINTS_PER_RING: usize = 4;
/// Total number of amodal structural keypoint distributions.
pub const KEYPOINT_COUNT: usize = ROOF_RING_COUNT * POINTS_PER_RING;
/// Square RGB input side used by the primary model.
pub const SPATIAL_INPUT_SIZE: usize = 256;
/// Side length of every keypoint distribution map.
pub const HEATMAP_SIZE: usize = SPATIAL_INPUT_SIZE / 4;
/// Side length retained by the spatial offscreen classifier.
const OFFSCREEN_POOL_SIZE: usize = 4;
/// Number of decoder channels consumed by the prediction heads.
const DECODER_CHANNELS: usize = 48;

/// Raw tensors produced by [`KeypointRoofNet`].
///
/// Keypoint channel `ring * 4 + slot` is one corner slot of the eave,
/// shoulder, or crown ring. The slots are deliberately not assigned physical
/// front/left names: training and fitting resolve the eight equivalent cyclic
/// and reflected correspondences.
pub struct KeypointRoofOutput<B: Backend> {
    /// Roof-presence logits with shape `[batch]`.
    pub presence_logits: Tensor<B, 1>,
    /// Amodal keypoint logits with shape `[batch, 12, 64, 64]`.
    pub keypoint_logits: Tensor<B, 4>,
    /// Per-keypoint offscreen logits with shape `[batch, 12]`.
    pub offscreen_logits: Tensor<B, 2>,
}

/// Parameter identifiers for applying different backbone and head rates.
///
/// A trainer can extract two gradient sets with
/// `GradientsParams::from_params` and step them using independent optimizers.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeypointParameterGroups {
    /// All MobileNetV2 feature-extractor parameters.
    pub backbone: Vec<ParamId>,
    /// FPN, keypoint, presence, and offscreen-head parameters.
    pub heads: Vec<ParamId>,
}

/// Configuration for the primary still-image roof observation network.
#[derive(Config, Debug)]
pub struct KeypointRoofNetConfig {
    /// Shared width of the presence and spatial offscreen MLP heads.
    #[config(default = 256)]
    pub global_hidden_size: usize,
}

/// MobileNetV2 FPN with keypoint-distribution, presence, and offscreen heads.
#[derive(Module, Debug)]
pub struct KeypointRoofNet<B: Backend> {
    features: mobilenet::MobileNetFeatures<B>,
    spatial_projection: Conv2d<B>,
    decoder_one: KeypointFusionStage<B>,
    decoder_two: KeypointFusionStage<B>,
    decoder_three: KeypointFusionStage<B>,
    keypoint_output: Conv2d<B>,
    avg_pool: AdaptiveAvgPool2d,
    global_hidden: Linear<B>,
    offscreen_pool: AdaptiveAvgPool2d,
    offscreen_hidden: Linear<B>,
    activation: Relu,
    presence_output: Linear<B>,
    offscreen_output: Linear<B>,
}

#[derive(Module, Debug)]
struct KeypointFusionStage<B: Backend> {
    upsample_projection: Conv2d<B>,
    upsample_normalization: BatchNorm<B>,
    upsample: Interpolate2d,
    skip_projection: Conv2d<B>,
    fusion_bottleneck: Conv2d<B>,
    fusion_bottleneck_normalization: BatchNorm<B>,
    fusion_spatial: Conv2d<B>,
    fusion_normalization: BatchNorm<B>,
    activation: Relu,
}

impl<B: Backend> KeypointFusionStage<B> {
    fn new(input: usize, skip: usize, output: usize, device: &B::Device) -> Self {
        let fused_channels = output * 2;
        Self {
            // Project on the smaller feature map before portable nearest
            // upsampling. Burn's WGPU backend implements the nearest backward
            // kernel (bilinear is forward-only), and the following learned
            // spatial block removes replication artefacts.
            upsample_projection: Conv2dConfig::new([input, output], [1, 1])
                .with_padding(PaddingConfig2d::Valid)
                .with_bias(false)
                .init(device),
            upsample_normalization: BatchNormConfig::new(output).init(device),
            upsample: Interpolate2dConfig::new()
                .with_scale_factor(Some([2.0, 2.0]))
                .with_mode(InterpolateMode::Nearest)
                .init(),
            skip_projection: Conv2dConfig::new([skip, output], [1, 1])
                .with_padding(PaddingConfig2d::Valid)
                .init(device),
            // Burn's current WGPU depthwise weight-gradient fallback is slower
            // than a smaller dense convolution on mobile-class channel counts.
            // Reduce the concatenated features first, then learn spatial
            // mixing at the output width.
            fusion_bottleneck: Conv2dConfig::new([fused_channels, output], [1, 1])
                .with_padding(PaddingConfig2d::Valid)
                .with_bias(false)
                .init(device),
            fusion_bottleneck_normalization: BatchNormConfig::new(output).init(device),
            fusion_spatial: Conv2dConfig::new([output, output], [3, 3])
                .with_padding(PaddingConfig2d::Same)
                .with_bias(false)
                .init(device),
            fusion_normalization: BatchNormConfig::new(output).init(device),
            activation: Relu::new(),
        }
    }

    fn forward(&self, input: Tensor<B, 4>, skip: Tensor<B, 4>) -> Tensor<B, 4> {
        let projected = self.upsample_projection.forward(input);
        let projected = self.upsample_normalization.forward(projected);
        let projected = self.activation.forward(projected);
        let upsampled = self.upsample.forward(projected);
        let skip = self.skip_projection.forward(skip);
        let combined = Tensor::cat(vec![upsampled, skip], 1);
        let combined = self.fusion_bottleneck.forward(combined);
        let combined = self.fusion_bottleneck_normalization.forward(combined);
        let combined = self.activation.forward(combined);
        let combined = self.fusion_spatial.forward(combined);
        let combined = self.fusion_normalization.forward(combined);
        self.activation.forward(combined)
    }
}

impl KeypointRoofNetConfig {
    /// Initializes a randomly weighted primary observation network.
    #[must_use]
    pub fn init<B: Backend>(&self, device: &B::Device) -> KeypointRoofNet<B> {
        KeypointRoofNet {
            features: mobilenet::random_features(device),
            spatial_projection: Conv2dConfig::new([1280, 128], [1, 1])
                .with_padding(PaddingConfig2d::Valid)
                .init(device),
            decoder_one: KeypointFusionStage::new(128, 96, 96, device),
            decoder_two: KeypointFusionStage::new(96, 32, 64, device),
            decoder_three: KeypointFusionStage::new(64, 24, DECODER_CHANNELS, device),
            keypoint_output: Conv2dConfig::new([DECODER_CHANNELS, KEYPOINT_COUNT], [1, 1])
                .with_padding(PaddingConfig2d::Valid)
                .init(device),
            avg_pool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
            global_hidden: LinearConfig::new(1280, self.global_hidden_size).init(device),
            offscreen_pool: AdaptiveAvgPool2dConfig::new([
                OFFSCREEN_POOL_SIZE,
                OFFSCREEN_POOL_SIZE,
            ])
            .init(),
            offscreen_hidden: LinearConfig::new(
                DECODER_CHANNELS * OFFSCREEN_POOL_SIZE * OFFSCREEN_POOL_SIZE,
                self.global_hidden_size,
            )
            .init(device),
            activation: Relu::new(),
            presence_output: LinearConfig::new(self.global_hidden_size, 1).init(device),
            offscreen_output: LinearConfig::new(self.global_hidden_size, KEYPOINT_COUNT)
                .init(device),
        }
    }

    /// Initializes a fully trainable feature extractor from torchvision
    /// ImageNet-1K V2 weights.
    ///
    /// This intentionally does not call `Module::no_grad`; callers control the
    /// lower backbone learning rate through [`KeypointRoofNet::parameter_groups`].
    #[cfg(feature = "pretrained")]
    pub fn init_pretrained<B: Backend>(
        &self,
        device: &B::Device,
    ) -> Result<KeypointRoofNet<B>, PytorchStoreError> {
        let pretrained = mobilenet::ImageNetMobileNet::pretrained(device)?;
        let mut model = self.init(device);
        model.features = pretrained.into_features();
        Ok(model)
    }
}

impl<B: Backend> KeypointRoofNet<B> {
    /// Predicts presence, amodal keypoint distributions, and offscreen states.
    #[must_use]
    pub fn forward(&self, images: Tensor<B, 4>) -> KeypointRoofOutput<B> {
        let features = self.features.forward_pyramid(images);
        let spatial = self
            .spatial_projection
            .forward(features.stride_thirty_two.clone());
        let spatial = self.decoder_one.forward(spatial, features.stride_sixteen);
        let spatial = self.decoder_two.forward(spatial, features.stride_eight);
        let spatial = self.decoder_three.forward(spatial, features.stride_four);
        let keypoint_logits = self.keypoint_output.forward(spatial.clone());

        // Offscreen classification needs the arrangement of the decoded roof
        // evidence: global average pooling the backbone cannot distinguish a
        // keypoint just outside one image edge from the same features elsewhere.
        // A small fixed grid retains coarse position while keeping this head
        // inexpensive and independent of the input resolution.
        let offscreen = self.offscreen_pool.forward(spatial).flatten(1, 3);
        let offscreen = self
            .activation
            .forward(self.offscreen_hidden.forward(offscreen));
        let offscreen_logits = self.offscreen_output.forward(offscreen);

        let batch = features.stride_thirty_two.dims()[0];
        let global = self
            .avg_pool
            .forward(features.stride_thirty_two)
            .flatten::<2>(1, 3);
        let global = self.activation.forward(self.global_hidden.forward(global));
        let presence_logits = self.presence_output.forward(global).reshape([batch]);

        debug_assert_eq!(keypoint_logits.dims()[1], KEYPOINT_COUNT);
        KeypointRoofOutput {
            presence_logits,
            keypoint_logits,
            offscreen_logits,
        }
    }

    /// Returns disjoint parameter groups for independent optimizer rates.
    #[must_use]
    pub fn parameter_groups(&self) -> KeypointParameterGroups {
        let backbone = list_param_ids(&self.features);
        let backbone_set = backbone.iter().copied().collect::<HashSet<_>>();
        let heads = list_param_ids(self)
            .into_iter()
            .filter(|id| !backbone_set.contains(id))
            .collect();
        KeypointParameterGroups { backbone, heads }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    type TestBackend = NdArray<f32>;

    #[test]
    fn output_contract_has_expected_channels_and_stride() {
        let device = Default::default();
        let config = KeypointRoofNetConfig::new();
        let model = config.init::<TestBackend>(&device);
        // A small input verifies the fully convolutional contract without
        // making the pure Rust ndarray backend run a full production frame.
        let input = Tensor::<TestBackend, 4>::zeros([1, 3, 32, 32], &device);
        let output = model.forward(input);
        assert_eq!(output.presence_logits.dims(), [1]);
        assert_eq!(output.keypoint_logits.dims(), [1, KEYPOINT_COUNT, 8, 8]);
        assert_eq!(output.offscreen_logits.dims(), [1, KEYPOINT_COUNT]);
        assert_eq!(crate::SPATIAL_INPUT_SIZE / 4, crate::HEATMAP_SIZE);
    }

    #[test]
    fn optimizer_parameter_groups_are_disjoint_and_complete() {
        let device = Default::default();
        let model = KeypointRoofNetConfig::new().init::<TestBackend>(&device);
        let groups = model.parameter_groups();
        assert!(!groups.backbone.is_empty());
        assert!(!groups.heads.is_empty());

        let backbone = groups.backbone.iter().copied().collect::<HashSet<_>>();
        let heads = groups.heads.iter().copied().collect::<HashSet<_>>();
        assert!(backbone.is_disjoint(&heads));
        assert_eq!(backbone.len() + heads.len(), list_param_ids(&model).len());
    }

    #[test]
    fn fusion_stage_uses_a_reduced_dense_spatial_block_and_doubles_resolution() {
        let device = Default::default();
        let stage = KeypointFusionStage::new(128, 96, 96, &device);
        assert_eq!(stage.fusion_bottleneck.weight.dims(), [96, 192, 1, 1]);
        assert_eq!(stage.fusion_spatial.groups, 1);
        assert_eq!(stage.fusion_spatial.weight.dims(), [96, 96, 3, 3]);

        let input = Tensor::<TestBackend, 4>::zeros([1, 128, 2, 3], &device);
        let skip = Tensor::<TestBackend, 4>::zeros([1, 96, 4, 6], &device);
        assert_eq!(stage.forward(input, skip).dims(), [1, 96, 4, 6]);
    }

    #[test]
    fn channels_are_three_four_point_rings() {
        assert_eq!(ROOF_RING_COUNT * POINTS_PER_RING, KEYPOINT_COUNT);
    }

    #[test]
    fn offscreen_head_retains_a_fixed_coarse_spatial_grid() {
        let device = Default::default();
        let config = KeypointRoofNetConfig::new();
        let model = config.init::<TestBackend>(&device);
        let spatial = Tensor::<TestBackend, 4>::zeros([2, DECODER_CHANNELS, 8, 8], &device);
        let pooled = model.offscreen_pool.forward(spatial);
        assert_eq!(
            pooled.dims(),
            [
                2,
                DECODER_CHANNELS,
                OFFSCREEN_POOL_SIZE,
                OFFSCREEN_POOL_SIZE
            ]
        );
        let hidden = model.offscreen_hidden.forward(pooled.flatten(1, 3));
        assert_eq!(hidden.dims(), [2, config.global_hidden_size]);
        let logits = model.offscreen_output.forward(hidden);
        assert_eq!(logits.dims(), [2, KEYPOINT_COUNT]);
    }
}
