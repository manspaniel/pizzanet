//! Compact spatial network for roof presence and amodal structural keypoints.

use std::collections::HashSet;

use burn::{
    module::{AutodiffModule, ParamId, list_param_ids},
    nn::{
        BatchNorm, BatchNormConfig, Linear, LinearConfig, PaddingConfig2d, Relu,
        conv::{Conv2d, Conv2dConfig},
        interpolate::{Interpolate2d, Interpolate2dConfig, InterpolateMode},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
    },
    prelude::*,
    tensor::backend::AutodiffBackend,
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

/// Geometry tensors produced without evaluating the roof-presence branch.
///
/// This narrower contract is used after presence has been locked during
/// training. It makes accidentally retaining an unused presence autodiff
/// graph impossible.
pub struct KeypointRoofGeometryOutput<B: Backend> {
    /// Amodal keypoint logits with shape `[batch, 12, 64, 64]`.
    pub keypoint_logits: Tensor<B, 4>,
    /// Per-keypoint offscreen logits with shape `[batch, 12]`.
    pub offscreen_logits: Tensor<B, 2>,
}

/// Parameter identifiers for applying different backbone and head rates.
///
/// A trainer can extract gradient sets with `GradientsParams::from_params` and
/// step the backbone, presence, and geometry branches independently. `heads`
/// is retained as the complete non-backbone group for callers that use one
/// optimizer for every prediction head.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KeypointParameterGroups {
    /// All MobileNetV2 feature-extractor parameters.
    pub backbone: Vec<ParamId>,
    /// FPN, keypoint, presence, and offscreen-head parameters.
    pub heads: Vec<ParamId>,
    /// Global hidden layer and roof-presence output parameters.
    pub presence_heads: Vec<ParamId>,
    /// FPN, keypoint-output, and spatial offscreen-head parameters.
    pub geometry_heads: Vec<ParamId>,
}

/// Training-only controls for how gradients and state cross the MobileNetV2
/// feature-pyramid boundary.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct KeypointTrainingOptions {
    /// Use the backbone's imported running mean and variance without updating
    /// them. Convolution and BatchNorm affine parameters remain differentiable.
    pub freeze_backbone_batch_norm: bool,
    /// Detach the pyramid consumed by the FPN while preserving the original
    /// stride-32 tensor, and its gradient graph, for roof presence.
    pub detach_geometry_backbone: bool,
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
        self.forward_from_features(features, false)
    }

    /// Training forward that preserves the pretrained MobileNetV2 running
    /// statistics while keeping its convolution and BatchNorm affine
    /// parameters trainable.
    ///
    /// Only the feature extractor uses frozen population statistics. The FPN
    /// and prediction heads continue to use normal backend-selected training
    /// behavior so their BatchNorm state can adapt from random initialization.
    #[must_use]
    pub fn forward_training_with_frozen_backbone_batch_norm(
        &self,
        images: Tensor<B, 4>,
    ) -> KeypointRoofOutput<B> {
        self.forward_training_with_options(
            images,
            KeypointTrainingOptions {
                freeze_backbone_batch_norm: true,
                detach_geometry_backbone: false,
            },
        )
    }

    /// Training forward with explicit MobileNetV2 state and gradient controls.
    ///
    /// When geometry detachment is enabled, presence consumes the original
    /// differentiable stride-32 feature tensor. Independent detached copies of
    /// all four pyramid levels feed the FPN, keypoint head, and offscreen head.
    /// Geometry-only losses therefore cannot update the backbone, while
    /// presence loss still can.
    #[must_use]
    pub fn forward_training_with_options(
        &self,
        images: Tensor<B, 4>,
        options: KeypointTrainingOptions,
    ) -> KeypointRoofOutput<B> {
        let features = if options.freeze_backbone_batch_norm {
            self.features.forward_pyramid_frozen_stats(images)
        } else {
            self.features.forward_pyramid(images)
        };
        self.forward_from_features(features, options.detach_geometry_backbone)
    }

    fn forward_from_features(
        &self,
        features: mobilenet::MobileNetPyramid<B>,
        detach_geometry_backbone: bool,
    ) -> KeypointRoofOutput<B> {
        let presence_features = features.stride_thirty_two.clone();
        let geometry_stride_four = if detach_geometry_backbone {
            features.stride_four.detach()
        } else {
            features.stride_four
        };
        let geometry_stride_eight = if detach_geometry_backbone {
            features.stride_eight.detach()
        } else {
            features.stride_eight
        };
        let geometry_stride_sixteen = if detach_geometry_backbone {
            features.stride_sixteen.detach()
        } else {
            features.stride_sixteen
        };
        let geometry_stride_thirty_two = if detach_geometry_backbone {
            features.stride_thirty_two.detach()
        } else {
            features.stride_thirty_two
        };
        let geometry = self.forward_geometry_from_features(mobilenet::MobileNetPyramid {
            stride_four: geometry_stride_four,
            stride_eight: geometry_stride_eight,
            stride_sixteen: geometry_stride_sixteen,
            stride_thirty_two: geometry_stride_thirty_two,
        });

        let batch = presence_features.dims()[0];
        let global = self.avg_pool.forward(presence_features).flatten::<2>(1, 3);
        let global = self.activation.forward(self.global_hidden.forward(global));
        let presence_logits = self.presence_output.forward(global).reshape([batch]);

        debug_assert_eq!(geometry.keypoint_logits.dims()[1], KEYPOINT_COUNT);
        KeypointRoofOutput {
            presence_logits,
            keypoint_logits: geometry.keypoint_logits,
            offscreen_logits: geometry.offscreen_logits,
        }
    }

    fn forward_geometry_from_features(
        &self,
        features: mobilenet::MobileNetPyramid<B>,
    ) -> KeypointRoofGeometryOutput<B> {
        let spatial = self.spatial_projection.forward(features.stride_thirty_two);
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

        debug_assert_eq!(keypoint_logits.dims()[1], KEYPOINT_COUNT);
        KeypointRoofGeometryOutput {
            keypoint_logits,
            offscreen_logits,
        }
    }

    /// Returns disjoint parameter groups for independent optimizer rates.
    #[must_use]
    pub fn parameter_groups(&self) -> KeypointParameterGroups {
        let backbone = list_param_ids(&self.features);
        let backbone_set = backbone.iter().copied().collect::<HashSet<_>>();
        let presence_set = list_param_ids(&self.global_hidden)
            .into_iter()
            .chain(list_param_ids(&self.presence_output))
            .collect::<HashSet<_>>();
        let heads = list_param_ids(self)
            .into_iter()
            .filter(|id| !backbone_set.contains(id))
            .collect::<Vec<_>>();
        let presence_heads = heads
            .iter()
            .copied()
            .filter(|id| presence_set.contains(id))
            .collect::<Vec<_>>();
        let geometry_heads = heads
            .iter()
            .copied()
            .filter(|id| !presence_set.contains(id))
            .collect::<Vec<_>>();
        KeypointParameterGroups {
            backbone,
            heads,
            presence_heads,
            geometry_heads,
        }
    }
}

impl<B: AutodiffBackend> KeypointRoofNet<B> {
    /// Runs geometry training with the backbone entirely outside autodiff.
    ///
    /// The frozen MobileNetV2 is evaluated on the inner backend, then its four
    /// feature levels are wrapped as untracked leaves before entering the
    /// trainable FPN and geometry heads. The presence head is not evaluated.
    /// This avoids retaining an unused backbone/presence graph between
    /// geometry-only optimizer steps.
    #[must_use]
    pub fn forward_geometry_training_with_frozen_backbone(
        &self,
        images: Tensor<B, 4>,
    ) -> KeypointRoofGeometryOutput<B> {
        let features = self
            .features
            .valid()
            .forward_pyramid_frozen_stats(images.inner());
        self.forward_geometry_from_features(mobilenet::MobileNetPyramid {
            stride_four: Tensor::from_inner(features.stride_four),
            stride_eight: Tensor::from_inner(features.stride_eight),
            stride_sixteen: Tensor::from_inner(features.stride_sixteen),
            stride_thirty_two: Tensor::from_inner(features.stride_thirty_two),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        backend::{Autodiff, Flex, NdArray, flex::FlexDevice},
        optim::GradientsParams,
        tensor::Tolerance,
    };

    type TestBackend = NdArray<f32>;
    type TestAutodiffBackend = Autodiff<TestBackend>;
    type TestFlexAutodiffBackend = Autodiff<Flex>;

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
        assert!(!groups.presence_heads.is_empty());
        assert!(!groups.geometry_heads.is_empty());

        let backbone = groups.backbone.iter().copied().collect::<HashSet<_>>();
        let heads = groups.heads.iter().copied().collect::<HashSet<_>>();
        let presence = groups
            .presence_heads
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        let geometry = groups
            .geometry_heads
            .iter()
            .copied()
            .collect::<HashSet<_>>();
        assert_eq!(backbone.len(), groups.backbone.len());
        assert_eq!(heads.len(), groups.heads.len());
        assert_eq!(presence.len(), groups.presence_heads.len());
        assert_eq!(geometry.len(), groups.geometry_heads.len());
        assert!(backbone.is_disjoint(&heads));
        assert!(backbone.is_disjoint(&presence));
        assert!(backbone.is_disjoint(&geometry));
        assert!(presence.is_disjoint(&geometry));
        assert_eq!(heads, presence.union(&geometry).copied().collect());
        assert_eq!(backbone.len() + heads.len(), list_param_ids(&model).len());

        let expected_presence = list_param_ids(&model.global_hidden)
            .into_iter()
            .chain(list_param_ids(&model.presence_output))
            .collect::<HashSet<_>>();
        assert_eq!(presence, expected_presence);

        let expected_geometry = list_param_ids(&model.spatial_projection)
            .into_iter()
            .chain(list_param_ids(&model.decoder_one))
            .chain(list_param_ids(&model.decoder_two))
            .chain(list_param_ids(&model.decoder_three))
            .chain(list_param_ids(&model.keypoint_output))
            .chain(list_param_ids(&model.offscreen_hidden))
            .chain(list_param_ids(&model.offscreen_output))
            .collect::<HashSet<_>>();
        assert_eq!(geometry, expected_geometry);
    }

    #[test]
    fn detached_geometry_and_offscreen_losses_do_not_reach_backbone() {
        let device = Default::default();
        let model = KeypointRoofNetConfig::new().init::<TestAutodiffBackend>(&device);
        let groups = model.parameter_groups();
        let input = Tensor::random(
            [1, 3, 32, 32],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        let output = model.forward_training_with_options(
            input,
            KeypointTrainingOptions {
                freeze_backbone_batch_norm: true,
                detach_geometry_backbone: true,
            },
        );

        let mut gradients =
            (output.keypoint_logits.sum() + output.offscreen_logits.sum()).backward();
        let backbone = GradientsParams::from_params(&mut gradients, &model, &groups.backbone);
        assert!(
            backbone.is_empty(),
            "detached geometry branches produced {} backbone gradients",
            backbone.len()
        );

        let presence = GradientsParams::from_params(&mut gradients, &model, &groups.presence_heads);
        assert!(
            presence.is_empty(),
            "geometry-only backward must not reach presence-head parameters"
        );

        let geometry_modules = [
            (
                "spatial projection",
                list_param_ids(&model.spatial_projection),
            ),
            ("decoder one", list_param_ids(&model.decoder_one)),
            ("decoder two", list_param_ids(&model.decoder_two)),
            ("decoder three", list_param_ids(&model.decoder_three)),
            ("keypoint output", list_param_ids(&model.keypoint_output)),
            ("offscreen hidden", list_param_ids(&model.offscreen_hidden)),
            ("offscreen output", list_param_ids(&model.offscreen_output)),
        ];
        for (name, ids) in geometry_modules {
            let branch = GradientsParams::from_params(&mut gradients, &model, &ids);
            assert!(
                !branch.is_empty(),
                "geometry and offscreen losses must reach {name} parameters"
            );
        }
    }

    #[test]
    fn frozen_backbone_geometry_forward_matches_detached_training_forward() {
        let device = FlexDevice;
        let model = KeypointRoofNetConfig::new().init::<TestFlexAutodiffBackend>(&device);
        let input = Tensor::random(
            [1, 3, 32, 32],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );

        let expected = model.forward_training_with_options(
            input.clone(),
            KeypointTrainingOptions {
                freeze_backbone_batch_norm: true,
                detach_geometry_backbone: true,
            },
        );
        let actual = model.forward_geometry_training_with_frozen_backbone(input);

        expected
            .keypoint_logits
            .to_data()
            .assert_approx_eq::<f32>(&actual.keypoint_logits.to_data(), Tolerance::default());
        expected
            .offscreen_logits
            .to_data()
            .assert_approx_eq::<f32>(&actual.offscreen_logits.to_data(), Tolerance::default());
    }

    #[test]
    fn inner_backbone_geometry_forward_only_tracks_geometry_parameters() {
        let device = FlexDevice;
        let model = KeypointRoofNetConfig::new().init::<TestFlexAutodiffBackend>(&device);
        let groups = model.parameter_groups();
        let input = Tensor::random(
            [1, 3, 32, 32],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        )
        .require_grad();
        let output = model.forward_geometry_training_with_frozen_backbone(input.clone());

        let mut gradients =
            (output.keypoint_logits.sum() + output.offscreen_logits.sum()).backward();
        assert!(
            input.grad(&gradients).is_none(),
            "the inner-backend backbone must sever the input autodiff graph"
        );
        assert!(
            GradientsParams::from_params(&mut gradients, &model, &groups.backbone).is_empty(),
            "the inner-backend backbone must not create backbone gradients"
        );
        assert!(
            GradientsParams::from_params(&mut gradients, &model, &groups.presence_heads).is_empty(),
            "the geometry-only forward must not evaluate or track the presence head"
        );
        assert!(
            !GradientsParams::from_params(&mut gradients, &model, &groups.geometry_heads)
                .is_empty(),
            "geometry-only backward must reach geometry parameters"
        );
    }

    #[test]
    fn presence_loss_keeps_backbone_and_presence_head_differentiable() {
        let device = Default::default();
        let model = KeypointRoofNetConfig::new().init::<TestAutodiffBackend>(&device);
        let groups = model.parameter_groups();
        let input = Tensor::random(
            [1, 3, 32, 32],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );
        let output = model.forward_training_with_options(
            input,
            KeypointTrainingOptions {
                freeze_backbone_batch_norm: true,
                detach_geometry_backbone: true,
            },
        );

        let mut gradients = output.presence_logits.sum().backward();
        let backbone = GradientsParams::from_params(&mut gradients, &model, &groups.backbone);
        assert!(
            !backbone.is_empty(),
            "presence loss must reach backbone parameters"
        );

        let presence = GradientsParams::from_params(&mut gradients, &model, &groups.presence_heads);
        assert_eq!(
            presence.len(),
            groups.presence_heads.len(),
            "presence loss must reach every presence-head parameter"
        );

        let geometry = GradientsParams::from_params(&mut gradients, &model, &groups.geometry_heads);
        assert!(
            geometry.is_empty(),
            "presence-only backward must not reach geometry-head parameters"
        );
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
