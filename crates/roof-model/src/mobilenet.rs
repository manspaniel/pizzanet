//! MobileNetV2 feature extractor adapted from tracel-ai/models.
//!
//! The upstream implementation is MIT OR Apache-2.0 licensed. Keeping the
//! module local lets the portable detector record use Burn 0.21 without a
//! runtime dependency on the research checkout under `references/`.

use std::cmp::max;

use burn::{
    config::Config,
    module::Module,
    nn::{
        BatchNorm, BatchNormConfig, Dropout, DropoutConfig, Linear, LinearConfig, PaddingConfig2d,
        conv::{Conv2d, Conv2dConfig},
        pool::{AdaptiveAvgPool2d, AdaptiveAvgPool2dConfig},
    },
    tensor::{Tensor, activation::relu, backend::Backend},
};

#[cfg(feature = "pretrained")]
use {
    burn::tensor::Device,
    burn_store::{ModuleSnapshot, PytorchStore, PytorchStoreError},
    std::{fs, io::Write, path::PathBuf},
};

const SETTINGS: [[usize; 4]; 7] = [
    [1, 16, 1, 1],
    [6, 24, 2, 2],
    [6, 32, 3, 2],
    [6, 64, 4, 2],
    [6, 96, 3, 1],
    [6, 160, 3, 2],
    [6, 320, 1, 1],
];

#[derive(Module, Debug)]
pub(crate) struct MobileNetFeatures<B: Backend> {
    blocks: Vec<ConvBlock<B>>,
}

pub(crate) struct MobileNetPyramid<B: Backend> {
    pub(crate) stride_four: Tensor<B, 4>,
    pub(crate) stride_eight: Tensor<B, 4>,
    pub(crate) stride_sixteen: Tensor<B, 4>,
    pub(crate) stride_thirty_two: Tensor<B, 4>,
}

impl<B: Backend> MobileNetFeatures<B> {
    pub(crate) fn forward_pyramid(&self, input: Tensor<B, 4>) -> MobileNetPyramid<B> {
        self.forward_pyramid_with_normalization(input, BackboneNormalization::BackendDefault)
    }

    /// Runs the pretrained feature extractor with its imported running
    /// statistics while leaving convolution and affine BatchNorm parameters
    /// differentiable.
    ///
    /// This is intentionally separate from [`Self::forward_pyramid`]: Burn
    /// selects training BatchNorm from the backend's autodiff state, but
    /// fine-tuning this backbone should not replace its ImageNet population
    /// statistics with the statistics of one small, source-balanced batch.
    pub(crate) fn forward_pyramid_frozen_stats(&self, input: Tensor<B, 4>) -> MobileNetPyramid<B> {
        self.forward_pyramid_with_normalization(input, BackboneNormalization::FrozenRunningStats)
    }

    fn forward_pyramid_with_normalization(
        &self,
        input: Tensor<B, 4>,
        normalization: BackboneNormalization,
    ) -> MobileNetPyramid<B> {
        let mut output = input;
        let mut stride_four = None;
        let mut stride_eight = None;
        let mut stride_sixteen = None;
        for (index, block) in self.blocks.iter().enumerate() {
            output = match block {
                ConvBlock::InvertedResidual(block) => {
                    block.forward_with_normalization(&output, normalization)
                }
                ConvBlock::Conv(block) => block.forward_with_normalization(output, normalization),
            };
            match index {
                3 => stride_four = Some(output.clone()),
                6 => stride_eight = Some(output.clone()),
                13 => stride_sixteen = Some(output.clone()),
                _ => {}
            }
        }
        MobileNetPyramid {
            stride_four: stride_four.expect("MobileNetV2 stride-four feature tap"),
            stride_eight: stride_eight.expect("MobileNetV2 stride-eight feature tap"),
            stride_sixteen: stride_sixteen.expect("MobileNetV2 stride-sixteen feature tap"),
            stride_thirty_two: output,
        }
    }
}

#[derive(Clone, Copy)]
enum BackboneNormalization {
    BackendDefault,
    FrozenRunningStats,
}

#[derive(Module, Debug)]
pub(crate) struct ImageNetMobileNet<B: Backend> {
    features: Vec<ConvBlock<B>>,
    classifier: Classifier<B>,
    avg_pool: AdaptiveAvgPool2d,
}

impl<B: Backend> ImageNetMobileNet<B> {
    fn init(device: &B::Device) -> Self {
        let features = build_features(device);
        Self {
            features,
            classifier: Classifier {
                dropout: DropoutConfig::new(0.2).init(),
                linear: LinearConfig::new(1280, 1000).init(device),
            },
            avg_pool: AdaptiveAvgPool2dConfig::new([1, 1]).init(),
        }
    }

    #[cfg(feature = "pretrained")]
    fn load_weights(&mut self) -> Result<(), PytorchStoreError> {
        let path =
            download_weights().map_err(|error| PytorchStoreError::Other(error.to_string()))?;
        let mut store = PytorchStore::from_file(path)
            .with_key_remapping("features\\.(0|18)\\.0.(.+)", "features.$1.conv.$2")
            .with_key_remapping("features\\.(0|18)\\.1.(.+)", "features.$1.norm.$2")
            .with_key_remapping("features\\.1\\.conv.0.0.(.+)", "features.1.dw.conv.$1")
            .with_key_remapping("features\\.1\\.conv.0.1.(.+)", "features.1.dw.norm.$1")
            .with_key_remapping("features\\.1\\.conv.1.(.+)", "features.1.pw_linear.conv.$1")
            .with_key_remapping("features\\.1\\.conv.2.(.+)", "features.1.pw_linear.norm.$1")
            .with_key_remapping(
                "features\\.([2-9]|1[0-7])\\.conv.0.0.(.+)",
                "features.$1.pw.conv.$2",
            )
            .with_key_remapping(
                "features\\.([2-9]|1[0-7])\\.conv.0.1.(.+)",
                "features.$1.pw.norm.$2",
            )
            .with_key_remapping(
                "features\\.([2-9]|1[0-7])\\.conv.1.0.(.+)",
                "features.$1.dw.conv.$2",
            )
            .with_key_remapping(
                "features\\.([2-9]|1[0-7])\\.conv.1.1.(.+)",
                "features.$1.dw.norm.$2",
            )
            .with_key_remapping(
                "features\\.([2-9]|1[0-7])\\.conv.2.(.+)",
                "features.$1.pw_linear.conv.$2",
            )
            .with_key_remapping(
                "features\\.([2-9]|1[0-7])\\.conv.3.(.+)",
                "features.$1.pw_linear.norm.$2",
            )
            .with_key_remapping("classifier.1.(.+)", "classifier.linear.$1");
        self.load_from(&mut store).map(|_| ())
    }

    #[cfg(feature = "pretrained")]
    pub(crate) fn pretrained(device: &Device<B>) -> Result<Self, PytorchStoreError> {
        let mut model = Self::init(device);
        model.load_weights()?;
        Ok(model)
    }

    pub(crate) fn into_features(self) -> MobileNetFeatures<B> {
        MobileNetFeatures {
            blocks: self.features,
        }
    }
}

pub(crate) fn random_features<B: Backend>(device: &B::Device) -> MobileNetFeatures<B> {
    ImageNetMobileNet::init(device).into_features()
}

#[derive(Module, Debug)]
#[allow(clippy::large_enum_variant)]
enum ConvBlock<B: Backend> {
    InvertedResidual(InvertedResidual<B>),
    Conv(ConvNormActivation<B>),
}

#[derive(Module, Debug)]
struct Classifier<B: Backend> {
    dropout: Dropout,
    linear: Linear<B>,
}

#[derive(Module, Debug)]
struct ConvNormActivation<B: Backend> {
    conv: Conv2d<B>,
    norm: BatchNorm<B>,
}

impl<B: Backend> ConvNormActivation<B> {
    fn forward_with_normalization(
        &self,
        input: Tensor<B, 4>,
        normalization: BackboneNormalization,
    ) -> Tensor<B, 4> {
        let output = self.conv.forward(input);
        let output = match normalization {
            BackboneNormalization::BackendDefault => self.norm.forward(output),
            BackboneNormalization::FrozenRunningStats => {
                forward_batch_norm_frozen_stats(&self.norm, output)
            }
        };
        relu(output).clamp_max(6)
    }
}

#[derive(Config, Debug)]
struct ConvNormConfig {
    in_channels: usize,
    out_channels: usize,
    #[config(default = 3)]
    kernel_size: usize,
    #[config(default = 1)]
    stride: usize,
    #[config(default = 1)]
    groups: usize,
}

impl ConvNormConfig {
    fn init<B: Backend>(&self, device: &B::Device) -> ConvNormActivation<B> {
        let padding = (self.kernel_size - 1) / 2;
        ConvNormActivation {
            conv: Conv2dConfig::new(
                [self.in_channels, self.out_channels],
                [self.kernel_size, self.kernel_size],
            )
            .with_padding(PaddingConfig2d::Explicit(
                padding, padding, padding, padding,
            ))
            .with_stride([self.stride, self.stride])
            .with_bias(false)
            .with_groups(self.groups)
            .init(device),
            norm: BatchNormConfig::new(self.out_channels).init(device),
        }
    }
}

#[derive(Module, Debug)]
struct PointWiseLinear<B: Backend> {
    conv: Conv2d<B>,
    norm: BatchNorm<B>,
}

impl<B: Backend> PointWiseLinear<B> {
    fn forward_with_normalization(
        &self,
        input: Tensor<B, 4>,
        normalization: BackboneNormalization,
    ) -> Tensor<B, 4> {
        let output = self.conv.forward(input);
        match normalization {
            BackboneNormalization::BackendDefault => self.norm.forward(output),
            BackboneNormalization::FrozenRunningStats => {
                forward_batch_norm_frozen_stats(&self.norm, output)
            }
        }
    }
}

fn forward_batch_norm_frozen_stats<B: Backend, const D: usize>(
    norm: &BatchNorm<B>,
    input: Tensor<B, D>,
) -> Tensor<B, D> {
    assert!(D >= 2, "BatchNorm input must have at least two dimensions");

    let device = input.device();
    let channels = input.dims()[1];
    let mut shape = [1; D];
    shape[1] = channels;
    let mean = norm.running_mean.value().to_device(&device).reshape(shape);
    let variance = norm.running_var.value().to_device(&device).reshape(shape);

    input
        .sub(mean)
        .div(variance.add_scalar(norm.epsilon).sqrt())
        .mul(norm.gamma.val().reshape(shape))
        .add(norm.beta.val().reshape(shape))
}

#[derive(Module, Debug)]
struct InvertedResidual<B: Backend> {
    use_residual: bool,
    pw: Option<ConvNormActivation<B>>,
    dw: ConvNormActivation<B>,
    pw_linear: PointWiseLinear<B>,
}

impl<B: Backend> InvertedResidual<B> {
    fn init(
        input: usize,
        output: usize,
        stride: usize,
        expansion: usize,
        device: &B::Device,
    ) -> Self {
        let hidden = input * expansion;
        let pw = (expansion != 1).then(|| {
            ConvNormConfig::new(input, hidden)
                .with_kernel_size(1)
                .init(device)
        });
        Self {
            use_residual: stride == 1 && input == output,
            pw,
            dw: ConvNormConfig::new(hidden, hidden)
                .with_stride(stride)
                .with_groups(hidden)
                .init(device),
            pw_linear: PointWiseLinear {
                conv: Conv2dConfig::new([hidden, output], [1, 1])
                    .with_padding(PaddingConfig2d::Explicit(0, 0, 0, 0))
                    .with_bias(false)
                    .init(device),
                norm: BatchNormConfig::new(output).init(device),
            },
        }
    }

    fn forward_with_normalization(
        &self,
        input: &Tensor<B, 4>,
        normalization: BackboneNormalization,
    ) -> Tensor<B, 4> {
        let mut output = input.clone();
        if let Some(pw) = &self.pw {
            output = pw.forward_with_normalization(output, normalization);
        }
        output = self.dw.forward_with_normalization(output, normalization);
        output = self
            .pw_linear
            .forward_with_normalization(output, normalization);
        if self.use_residual {
            output + input.clone()
        } else {
            output
        }
    }
}

fn build_features<B: Backend>(device: &B::Device) -> Vec<ConvBlock<B>> {
    let divisible = |value: f32| {
        let mut rounded = max(((value + 4.0) as usize / 8) * 8, 8);
        if (rounded as f32) < value * 0.9 {
            rounded += 8;
        }
        rounded
    };
    let mut input = divisible(32.0);
    let mut features = vec![ConvBlock::Conv(
        ConvNormConfig::new(3, input).with_stride(2).init(device),
    )];
    for [expansion, channels, count, stride] in SETTINGS {
        let output = divisible(channels as f32);
        for index in 0..count {
            features.push(ConvBlock::InvertedResidual(InvertedResidual::init(
                input,
                output,
                if index == 0 { stride } else { 1 },
                expansion,
                device,
            )));
            input = output;
        }
    }
    features.push(ConvBlock::Conv(
        ConvNormConfig::new(input, 1280)
            .with_kernel_size(1)
            .init(device),
    ));
    features
}

#[cfg(feature = "pretrained")]
fn download_weights() -> Result<PathBuf, std::io::Error> {
    const URL: &str = "https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth";
    let directory = dirs::cache_dir()
        .unwrap_or_else(std::env::temp_dir)
        .join("pizzahut-roof-model");
    fs::create_dir_all(&directory)?;
    let path = directory.join("mobilenet_v2-7ebf99e0.pth");
    if !path.exists() {
        let bytes = burn::data::network::downloader::download_file_as_bytes(
            URL,
            "mobilenet_v2-7ebf99e0.pth",
        );
        let mut file = fs::File::create(&path)?;
        file.write_all(&bytes)?;
    }
    Ok(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::{
        backend::{Autodiff, NdArray},
        tensor::TensorData,
    };

    type TestAutodiffBackend = Autodiff<NdArray<f32>>;

    fn values(tensor: Tensor<TestAutodiffBackend, 1>) -> Vec<f32> {
        tensor.to_data().to_vec::<f32>().expect("f32 tensor data")
    }

    fn running_stats(
        features: &MobileNetFeatures<TestAutodiffBackend>,
    ) -> Vec<(Vec<f32>, Vec<f32>)> {
        let mut stats = Vec::new();
        let mut push = |norm: &BatchNorm<TestAutodiffBackend>| {
            stats.push((
                values(norm.running_mean.value_sync()),
                values(norm.running_var.value_sync()),
            ));
        };
        for block in &features.blocks {
            match block {
                ConvBlock::Conv(block) => push(&block.norm),
                ConvBlock::InvertedResidual(block) => {
                    if let Some(pw) = &block.pw {
                        push(&pw.norm);
                    }
                    push(&block.dw.norm);
                    push(&block.pw_linear.norm);
                }
            }
        }
        stats
    }

    #[test]
    fn frozen_backbone_forward_does_not_mutate_any_running_statistics() {
        let device = Default::default();
        let features = random_features::<TestAutodiffBackend>(&device);
        let before = running_stats(&features);
        let input = Tensor::random(
            [2, 3, 32, 32],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        )
        .require_grad();

        let pyramid = features.forward_pyramid_frozen_stats(input.clone());

        assert_eq!(pyramid.stride_four.dims(), [2, 24, 8, 8]);
        assert_eq!(pyramid.stride_eight.dims(), [2, 32, 4, 4]);
        assert_eq!(pyramid.stride_sixteen.dims(), [2, 96, 2, 2]);
        assert_eq!(pyramid.stride_thirty_two.dims(), [2, 1280, 1, 1]);
        let gradients = pyramid.stride_thirty_two.backward();
        assert!(input.grad(&gradients).is_some());
        let ConvBlock::Conv(first) = &features.blocks[0] else {
            panic!("the first MobileNetV2 feature block must be a convolution");
        };
        assert!(first.conv.weight.grad(&gradients).is_some());
        assert!(first.norm.gamma.grad(&gradients).is_some());
        assert!(first.norm.beta.grad(&gradients).is_some());
        assert_eq!(running_stats(&features), before);
    }

    #[test]
    fn ordinary_autodiff_forward_updates_backbone_running_statistics() {
        let device = Default::default();
        let features = random_features::<TestAutodiffBackend>(&device);
        let before = running_stats(&features);
        let input = Tensor::random(
            [2, 3, 32, 32],
            burn::tensor::Distribution::Uniform(-1.0, 1.0),
            &device,
        );

        let _ = features.forward_pyramid(input);

        assert_ne!(running_stats(&features), before);
    }

    #[test]
    fn frozen_statistics_keep_input_and_affine_gradients() {
        let device = Default::default();
        let norm = BatchNormConfig::new(3).init::<TestAutodiffBackend>(&device);
        let input: Tensor<TestAutodiffBackend, 4> = Tensor::from_data(
            TensorData::from([
                [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]],
                [[[2.0, 3.0]], [[4.0, 5.0]], [[6.0, 7.0]]],
            ]),
            &device,
        )
        .require_grad();
        let before = (
            values(norm.running_mean.value_sync()),
            values(norm.running_var.value_sync()),
        );

        let output = forward_batch_norm_frozen_stats(&norm, input.clone());
        assert_eq!(output.dims(), input.dims());
        let gradients = output.backward();

        assert!(input.grad(&gradients).is_some());
        assert!(norm.gamma.grad(&gradients).is_some());
        assert!(norm.beta.grad(&gradients).is_some());
        assert_eq!(
            (
                values(norm.running_mean.value_sync()),
                values(norm.running_var.value_sync()),
            ),
            before
        );
    }
}
