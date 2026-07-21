//! Decoded texture contracts and native GPU preparation.

use std::collections::BTreeSet;

use half::f16;

use crate::RenderError;

pub(crate) const MAX_TEXTURE_LAYERS: usize = 32;
const SAFE_F16_COMPONENT: f32 = 60_000.0;

/// Stable physical layer within prepared PBR texture arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MaterialTextureLayer(u32);

impl MaterialTextureLayer {
    /// Creates an in-range texture layer identifier.
    pub fn new(layer: u32) -> Result<Self, RenderError> {
        if layer as usize >= MAX_TEXTURE_LAYERS {
            return Err(RenderError::InvalidTextureData);
        }
        Ok(Self(layer))
    }

    /// Returns the physical texture-array layer.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Decoded RGBA8 image supplied by the tool layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Rgba8Image {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Tightly packed row-major RGBA bytes.
    pub pixels: Vec<u8>,
}

impl Rgba8Image {
    /// Creates a decoded image after checking dimensions and byte count.
    pub fn new(width: u32, height: u32, pixels: Vec<u8>) -> Result<Self, RenderError> {
        let image = Self {
            width,
            height,
            pixels,
        };
        image.validate()?;
        Ok(image)
    }

    /// Revalidates dimensions and backing storage after public-field mutation.
    pub fn validate(&self) -> Result<(), RenderError> {
        validate_pixel_count(self.width, self.height, self.pixels.len(), 4)?;
        if self.width.checked_mul(4).is_none() {
            return Err(RenderError::InvalidTextureData);
        }
        Ok(())
    }
}

/// Decoded linear floating-point equirectangular environment image.
#[derive(Clone, Debug, PartialEq)]
pub struct EquirectangularImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Tightly packed row-major linear RGBA pixels.
    pub pixels: Vec<[f32; 4]>,
}

impl EquirectangularImage {
    /// Creates an environment image after checking dimensions and radiance pixels.
    pub fn new(width: u32, height: u32, pixels: Vec<[f32; 4]>) -> Result<Self, RenderError> {
        let image = Self {
            width,
            height,
            pixels,
        };
        image.validate()?;
        Ok(image)
    }

    /// Revalidates dimensions, finite non-negative radiance, and bounded alpha.
    ///
    /// RGB values may exceed the native `f16` range. GPU preparation applies one
    /// loss-minimising scale to the complete panorama and restores that scale in
    /// the shader, preserving relative HDR radiance rather than clamping it.
    pub fn validate(&self) -> Result<(), RenderError> {
        validate_pixel_count(self.width, self.height, self.pixels.len(), 1)?;
        if self.width.checked_mul(8).is_none()
            || self.pixels.iter().any(|pixel| {
                pixel[..3]
                    .iter()
                    .any(|component| !component.is_finite() || *component < 0.0)
                    || !pixel[3].is_finite()
                    || !(0.0..=1.0).contains(&pixel[3])
            })
        {
            return Err(RenderError::InvalidTextureData);
        }
        Ok(())
    }

    /// Longitude of the brightest luminance sample in equirectangular space.
    ///
    /// Longitude zero corresponds to the panorama's centre column, matching the
    /// renderer's equirectangular lookup convention.
    pub fn dominant_light_longitude_radians(&self) -> Result<f32, RenderError> {
        self.validate()?;
        let (index, _) = self
            .pixels
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| hdr_luminance(left).total_cmp(&hdr_luminance(right)))
            .ok_or(RenderError::InvalidTextureData)?;
        let x = (index % self.width as usize) as f32 + 0.5;
        Ok((x / self.width as f32 - 0.5) * std::f32::consts::TAU)
    }

    /// Environment yaw that places the dominant panorama light at a world azimuth.
    pub fn yaw_to_align_dominant_light(
        &self,
        world_azimuth_radians: f32,
    ) -> Result<f32, RenderError> {
        if !world_azimuth_radians.is_finite() {
            return Err(RenderError::InvalidTextureData);
        }
        Ok(wrap_angle(
            self.dominant_light_longitude_radians()? - world_azimuth_radians,
        ))
    }
}

fn hdr_luminance(pixel: &[f32; 4]) -> f32 {
    pixel[0] * 0.2126 + pixel[1] * 0.7152 + pixel[2] * 0.0722
}

fn wrap_angle(angle: f32) -> f32 {
    (angle + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI
}

/// Albedo, normal, and ambient-roughness-metallic maps for one physical layer.
#[derive(Clone, Debug, PartialEq)]
pub struct MaterialTextureSet {
    /// Physical layer, independent from a scene's logical material slots.
    pub layer: MaterialTextureLayer,
    /// Nonlinear sRGB base-colour map with a required fully opaque alpha channel.
    ///
    /// The current renderer is entirely opaque and does not implement cutout or
    /// blended material pipelines, so accepting non-opaque albedo would silently
    /// discard authored coverage.
    pub albedo: Rgba8Image,
    /// Linear tangent-space normal map.
    pub normal: Rgba8Image,
    /// Linear ambient-occlusion, roughness, metallic, and unused channels.
    pub arm: Rgba8Image,
    /// Horizontal and vertical repetitions over each geometry UV domain.
    pub uv_tiling: [f32; 2],
}

impl MaterialTextureSet {
    /// Creates one complete PBR material set.
    pub fn new(
        layer: MaterialTextureLayer,
        albedo: Rgba8Image,
        normal: Rgba8Image,
        arm: Rgba8Image,
        uv_tiling: [f32; 2],
    ) -> Result<Self, RenderError> {
        let material = Self {
            layer,
            albedo,
            normal,
            arm,
            uv_tiling,
        };
        material.validate()?;
        Ok(material)
    }

    /// Revalidates all maps and UV controls after public-field mutation.
    pub fn validate(&self) -> Result<(), RenderError> {
        self.albedo.validate()?;
        self.normal.validate()?;
        self.arm.validate()?;
        if self.albedo.width != self.normal.width
            || self.albedo.height != self.normal.height
            || self.albedo.width != self.arm.width
            || self.albedo.height != self.arm.height
        {
            return Err(RenderError::TextureDimensionMismatch);
        }
        if self
            .albedo
            .pixels
            .chunks_exact(4)
            .any(|pixel| pixel[3] != u8::MAX)
        {
            return Err(RenderError::InvalidTextureData);
        }
        if self
            .uv_tiling
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        {
            return Err(RenderError::InvalidTextureData);
        }
        Ok(())
    }
}

/// Optional decoded image assets for one renderer configuration.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RenderAssetBundle {
    /// Physical texture layers later mapped from logical material roles.
    pub materials: Vec<MaterialTextureSet>,
    /// Equirectangular sky/background image; procedural sky remains the fallback.
    pub environment: Option<EquirectangularImage>,
}

/// GPU-resident texture arrays reusable across many headless frames.
pub struct PreparedRenderAssets {
    pub(crate) _albedo: wgpu::Texture,
    pub(crate) albedo_view: wgpu::TextureView,
    pub(crate) _normal: wgpu::Texture,
    pub(crate) normal_view: wgpu::TextureView,
    pub(crate) _arm: wgpu::Texture,
    pub(crate) arm_view: wgpu::TextureView,
    pub(crate) material_sampler: wgpu::Sampler,
    pub(crate) _environment: wgpu::Texture,
    pub(crate) environment_view: wgpu::TextureView,
    pub(crate) environment_sampler: wgpu::Sampler,
    pub(crate) material_mask: u32,
    pub(crate) material_uv_tiling: [[f32; 4]; MAX_TEXTURE_LAYERS],
    pub(crate) has_environment: bool,
    pub(crate) environment_size: [u32; 2],
    pub(crate) environment_max_lod: u32,
    pub(crate) environment_radiance_scale: f32,
}

impl PreparedRenderAssets {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        bundle: &RenderAssetBundle,
        anisotropy_clamp: u16,
    ) -> Result<Self, RenderError> {
        validate_bundle(bundle)?;
        validate_bundle_device_limits(device, bundle)?;
        let texture_size = bundle.materials.first().map_or([1, 1], |material| {
            [material.albedo.width, material.albedo.height]
        });
        let layers = bundle
            .materials
            .iter()
            .map(|material| material.layer.get() + 1)
            .max()
            .unwrap_or(1);
        let albedo = create_array_texture(
            device,
            "roof-synth material albedo array",
            texture_size,
            layers,
            wgpu::TextureFormat::Rgba8UnormSrgb,
        );
        let normal = create_array_texture(
            device,
            "roof-synth material normal array",
            texture_size,
            layers,
            wgpu::TextureFormat::Rgba8Unorm,
        );
        let arm = create_array_texture(
            device,
            "roof-synth material ARM array",
            texture_size,
            layers,
            wgpu::TextureFormat::Rgba8Unorm,
        );
        let mut material_mask = 0_u32;
        let mut material_uv_tiling = [[1.0, 1.0, 0.0, 0.0]; MAX_TEXTURE_LAYERS];
        for material in &bundle.materials {
            let layer = material.layer.get();
            write_rgba8_mips(queue, &albedo, layer, &material.albedo, MipFilter::Srgb)?;
            write_rgba8_mips(queue, &normal, layer, &material.normal, MipFilter::Normal)?;
            write_rgba8_mips(queue, &arm, layer, &material.arm, MipFilter::Linear)?;
            material_mask |= 1 << layer;
            material_uv_tiling[layer as usize] =
                [material.uv_tiling[0], material.uv_tiling[1], 0.0, 0.0];
        }
        let array_view = |texture: &wgpu::Texture| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2Array),
                ..Default::default()
            })
        };
        let albedo_view = array_view(&albedo);
        let normal_view = array_view(&normal);
        let arm_view = array_view(&arm);
        let material_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("roof-synth repeating material sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp,
            ..Default::default()
        });

        let environment_size = bundle
            .environment
            .as_ref()
            .map_or([2, 1], |image| [image.width, image.height]);
        let environment_mip_levels = mip_level_count(environment_size[0], environment_size[1]);
        let environment = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("roof-synth equirectangular environment"),
            size: wgpu::Extent3d {
                width: environment_size[0],
                height: environment_size[1],
                depth_or_array_layers: 1,
            },
            mip_level_count: environment_mip_levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let environment_radiance_scale = if let Some(image) = &bundle.environment {
            write_environment_mips(queue, &environment, image)?
        } else {
            1.0
        };
        let environment_view = environment.create_view(&Default::default());
        let environment_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("roof-synth environment sampler"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });

        Ok(Self {
            _albedo: albedo,
            albedo_view,
            _normal: normal,
            normal_view,
            _arm: arm,
            arm_view,
            material_sampler,
            _environment: environment,
            environment_view,
            environment_sampler,
            material_mask,
            material_uv_tiling,
            has_environment: bundle.environment.is_some(),
            environment_size,
            environment_max_lod: environment_mip_levels - 1,
            environment_radiance_scale,
        })
    }

    pub(crate) fn update_environment(
        &mut self,
        queue: &wgpu::Queue,
        image: &EquirectangularImage,
    ) -> Result<(), RenderError> {
        validate_environment_replacement(self.environment_size, image)?;
        self.environment_radiance_scale = write_environment_mips(queue, &self._environment, image)?;
        self.has_environment = true;
        Ok(())
    }
}

fn write_environment_mips(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    image: &EquirectangularImage,
) -> Result<f32, RenderError> {
    image.validate()?;
    let maximum = image
        .pixels
        .iter()
        .flat_map(|pixel| pixel[..3].iter().copied())
        .fold(0.0_f32, f32::max);
    let upload_scale = if maximum > SAFE_F16_COMPONENT {
        SAFE_F16_COMPONENT / maximum
    } else {
        1.0
    };
    let radiance_scale = upload_scale.recip();
    let mut width = image.width;
    let mut height = image.height;
    let mut pixels = image.pixels.clone();
    for mip_level in 0..mip_level_count(image.width, image.height) {
        let half_pixels = pixels
            .iter()
            .flat_map(|pixel| {
                [
                    pixel[0] * upload_scale,
                    pixel[1] * upload_scale,
                    pixel[2] * upload_scale,
                    pixel[3],
                ]
                .map(f16::from_f32)
                .map(f16::to_bits)
            })
            .collect::<Vec<_>>();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&half_pixels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 8),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        if width == 1 && height == 1 {
            break;
        }
        let next_width = (width / 2).max(1);
        let next_height = (height / 2).max(1);
        pixels = downsample_hdr(&pixels, width, height, next_width, next_height);
        width = next_width;
        height = next_height;
    }
    Ok(radiance_scale)
}

fn downsample_hdr(
    source: &[[f32; 4]],
    source_width: u32,
    source_height: u32,
    width: u32,
    height: u32,
) -> Vec<[f32; 4]> {
    let mut output = Vec::with_capacity(width as usize * height as usize);
    for y in 0..height {
        for x in 0..width {
            let mut sum = [0.0; 4];
            for [sample_x, sample_y] in [
                [
                    (x * 2).min(source_width - 1),
                    (y * 2).min(source_height - 1),
                ],
                [
                    (x * 2 + 1).min(source_width - 1),
                    (y * 2).min(source_height - 1),
                ],
                [
                    (x * 2).min(source_width - 1),
                    (y * 2 + 1).min(source_height - 1),
                ],
                [
                    (x * 2 + 1).min(source_width - 1),
                    (y * 2 + 1).min(source_height - 1),
                ],
            ] {
                let sample = source[(sample_y * source_width + sample_x) as usize];
                for channel in 0..4 {
                    sum[channel] += sample[channel] * 0.25;
                }
            }
            output.push(sum);
        }
    }
    output
}

fn validate_bundle(bundle: &RenderAssetBundle) -> Result<(), RenderError> {
    let mut slots = BTreeSet::new();
    let expected_dimensions = bundle
        .materials
        .first()
        .map(|material| [material.albedo.width, material.albedo.height]);
    for material in &bundle.materials {
        material.validate()?;
        if !slots.insert(material.layer.get()) {
            return Err(RenderError::InvalidTextureData);
        }
        if expected_dimensions
            .is_some_and(|size| size != [material.albedo.width, material.albedo.height])
        {
            return Err(RenderError::TextureDimensionMismatch);
        }
    }
    if let Some(environment) = &bundle.environment {
        environment.validate()?;
    }
    Ok(())
}

fn validate_bundle_device_limits(
    device: &wgpu::Device,
    bundle: &RenderAssetBundle,
) -> Result<(), RenderError> {
    let limits = device.limits();
    let maximum_dimension = limits.max_texture_dimension_2d;
    let dimensions_fit =
        |width: u32, height: u32| width <= maximum_dimension && height <= maximum_dimension;
    if bundle
        .materials
        .iter()
        .any(|material| !dimensions_fit(material.albedo.width, material.albedo.height))
        || bundle
            .environment
            .as_ref()
            .is_some_and(|image| !dimensions_fit(image.width, image.height))
    {
        return Err(RenderError::InvalidTextureData);
    }
    let required_layers = bundle
        .materials
        .iter()
        .map(|material| material.layer.get() + 1)
        .max()
        .unwrap_or(1);
    if required_layers > limits.max_texture_array_layers {
        return Err(RenderError::InvalidTextureData);
    }
    Ok(())
}

fn validate_environment_replacement(
    expected_size: [u32; 2],
    image: &EquirectangularImage,
) -> Result<(), RenderError> {
    image.validate()?;
    if [image.width, image.height] != expected_size {
        return Err(RenderError::TextureDimensionMismatch);
    }
    Ok(())
}

fn validate_pixel_count(
    width: u32,
    height: u32,
    actual_values: usize,
    values_per_pixel: usize,
) -> Result<(), RenderError> {
    if width == 0 || height == 0 || values_per_pixel == 0 {
        return Err(RenderError::InvalidTextureData);
    }
    let expected = usize::try_from(width)
        .ok()
        .and_then(|width| {
            usize::try_from(height)
                .ok()
                .and_then(|height| width.checked_mul(height))
        })
        .and_then(|pixels| pixels.checked_mul(values_per_pixel));
    if expected != Some(actual_values) {
        return Err(RenderError::InvalidTextureData);
    }
    Ok(())
}

fn create_array_texture(
    device: &wgpu::Device,
    label: &'static str,
    size: [u32; 2],
    layers: u32,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: layers,
        },
        mip_level_count: mip_level_count(size[0], size[1]),
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    })
}

fn mip_level_count(width: u32, height: u32) -> u32 {
    u32::BITS - width.max(height).leading_zeros()
}

#[derive(Clone, Copy)]
enum MipFilter {
    Srgb,
    Normal,
    Linear,
}

fn write_rgba8_mips(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    layer: u32,
    image: &Rgba8Image,
    filter: MipFilter,
) -> Result<(), RenderError> {
    image.validate()?;
    let mut width = image.width;
    let mut height = image.height;
    let mut pixels = image.pixels.clone();
    for mip_level in 0..mip_level_count(image.width, image.height) {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level,
                origin: wgpu::Origin3d {
                    x: 0,
                    y: 0,
                    z: layer,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(width * 4),
                rows_per_image: Some(height),
            },
            wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        if width == 1 && height == 1 {
            break;
        }
        let next_width = (width / 2).max(1);
        let next_height = (height / 2).max(1);
        pixels = downsample_rgba8(&pixels, width, height, next_width, next_height, filter);
        width = next_width;
        height = next_height;
    }
    Ok(())
}

fn downsample_rgba8(
    source: &[u8],
    source_width: u32,
    source_height: u32,
    width: u32,
    height: u32,
    filter: MipFilter,
) -> Vec<u8> {
    let mut output = vec![0; width as usize * height as usize * 4];
    for y in 0..height {
        for x in 0..width {
            let coordinates = [
                [
                    (x * 2).min(source_width - 1),
                    (y * 2).min(source_height - 1),
                ],
                [
                    (x * 2 + 1).min(source_width - 1),
                    (y * 2).min(source_height - 1),
                ],
                [
                    (x * 2).min(source_width - 1),
                    (y * 2 + 1).min(source_height - 1),
                ],
                [
                    (x * 2 + 1).min(source_width - 1),
                    (y * 2 + 1).min(source_height - 1),
                ],
            ];
            let samples = coordinates.map(|[sample_x, sample_y]| {
                let offset = ((sample_y * source_width + sample_x) * 4) as usize;
                [
                    source[offset],
                    source[offset + 1],
                    source[offset + 2],
                    source[offset + 3],
                ]
            });
            let result = match filter {
                MipFilter::Srgb => average_srgb(samples),
                MipFilter::Normal => average_normal(samples),
                MipFilter::Linear => average_linear(samples),
            };
            let offset = ((y * width + x) * 4) as usize;
            output[offset..offset + 4].copy_from_slice(&result);
        }
    }
    output
}

fn average_linear(samples: [[u8; 4]; 4]) -> [u8; 4] {
    std::array::from_fn(|channel| {
        let sum = samples
            .iter()
            .map(|sample| u32::from(sample[channel]))
            .sum::<u32>();
        ((sum + 2) / 4) as u8
    })
}

fn average_srgb(samples: [[u8; 4]; 4]) -> [u8; 4] {
    let mut output = [0; 4];
    for channel in 0..3 {
        let linear = samples
            .iter()
            .map(|sample| srgb_u8_to_linear(sample[channel]))
            .sum::<f32>()
            * 0.25;
        output[channel] = linear_to_srgb_u8(linear);
    }
    output[3] = average_linear(samples)[3];
    output
}

fn average_normal(samples: [[u8; 4]; 4]) -> [u8; 4] {
    let mut normal = [0.0_f32; 3];
    for sample in samples {
        for channel in 0..3 {
            normal[channel] += f32::from(sample[channel]) / 127.5 - 1.0;
        }
    }
    let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2])
        .sqrt()
        .max(1.0e-6);
    [
        ((normal[0] / length * 0.5 + 0.5) * 255.0).round() as u8,
        ((normal[1] / length * 0.5 + 0.5) * 255.0).round() as u8,
        ((normal[2] / length * 0.5 + 0.5) * 255.0).round() as u8,
        average_linear(samples)[3],
    ]
}

fn srgb_u8_to_linear(value: u8) -> f32 {
    let value = f32::from(value) / 255.0;
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb_u8(value: f32) -> u8 {
    let encoded = if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    };
    (encoded.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn image(width: u32, height: u32) -> Rgba8Image {
        let mut pixels = vec![127; width as usize * height as usize * 4];
        for pixel in pixels.chunks_exact_mut(4) {
            pixel[3] = u8::MAX;
        }
        Rgba8Image::new(width, height, pixels).unwrap()
    }

    #[test]
    fn validates_decoded_pixel_counts() {
        assert!(Rgba8Image::new(2, 2, vec![0; 16]).is_ok());
        assert!(Rgba8Image::new(2, 2, vec![0; 15]).is_err());
        assert!(EquirectangularImage::new(1, 1, vec![[1.0; 4]]).is_ok());
        assert!(EquirectangularImage::new(1, 1, vec![[f32::NAN; 4]]).is_err());
    }

    #[test]
    fn accepts_finite_hdr_values_above_f16_range_for_scaled_upload() {
        assert!(EquirectangularImage::new(1, 1, vec![[421_888.0, 1.0, 0.0, 1.0]]).is_ok());
        assert!(EquirectangularImage::new(1, 1, vec![[f32::INFINITY, 1.0, 0.0, 1.0]]).is_err());
        assert!(EquirectangularImage::new(1, 1, vec![[-0.01, 1.0, 0.0, 1.0]]).is_err());
        assert!(EquirectangularImage::new(1, 1, vec![[1.0, 1.0, 1.0, 1.01]]).is_err());
    }

    #[test]
    fn revalidates_public_image_storage_after_mutation() {
        let mut rgba = image(2, 2);
        rgba.pixels.pop();
        assert!(matches!(
            rgba.validate(),
            Err(RenderError::InvalidTextureData)
        ));

        let mut environment = EquirectangularImage::new(2, 1, vec![[0.25; 4]; 2]).unwrap();
        environment.width = 3;
        assert!(matches!(
            environment.validate(),
            Err(RenderError::InvalidTextureData)
        ));
        environment.width = 2;
        environment.pixels[0][2] = f32::INFINITY;
        assert!(matches!(
            environment.validate(),
            Err(RenderError::InvalidTextureData)
        ));
    }

    #[test]
    fn bundle_validation_rechecks_mutated_material_and_environment_pixels() {
        let mut material = MaterialTextureSet::new(
            MaterialTextureLayer::new(0).unwrap(),
            image(2, 2),
            image(2, 2),
            image(2, 2),
            [1.0, 1.0],
        )
        .unwrap();
        material.normal.pixels.truncate(3);
        assert!(matches!(
            validate_bundle(&RenderAssetBundle {
                materials: vec![material],
                environment: None,
            }),
            Err(RenderError::InvalidTextureData)
        ));

        let mut environment = EquirectangularImage::new(2, 1, vec![[0.25; 4]; 2]).unwrap();
        environment.pixels.clear();
        assert!(matches!(
            validate_bundle(&RenderAssetBundle {
                materials: Vec::new(),
                environment: Some(environment),
            }),
            Err(RenderError::InvalidTextureData)
        ));
    }

    #[test]
    fn environment_replacement_validates_pixels_before_upload() {
        let valid = EquirectangularImage::new(2, 1, vec![[0.5; 4]; 2]).unwrap();
        assert!(validate_environment_replacement([2, 1], &valid).is_ok());

        let wrong_size = EquirectangularImage::new(1, 1, vec![[0.5; 4]]).unwrap();
        assert!(matches!(
            validate_environment_replacement([2, 1], &wrong_size),
            Err(RenderError::TextureDimensionMismatch)
        ));

        let mut invalid_pixels = valid;
        invalid_pixels.pixels.pop();
        assert!(matches!(
            validate_environment_replacement([2, 1], &invalid_pixels),
            Err(RenderError::InvalidTextureData)
        ));
    }

    #[test]
    fn extreme_dimensions_return_errors_without_overflowing() {
        assert!(Rgba8Image::new(u32::MAX, u32::MAX, Vec::new()).is_err());
        assert!(EquirectangularImage::new(u32::MAX, u32::MAX, Vec::new()).is_err());
        assert!(Rgba8Image::new(0, 1, Vec::new()).is_err());
        assert!(EquirectangularImage::new(1, 0, Vec::new()).is_err());
    }

    #[test]
    fn pbr_maps_must_share_dimensions() {
        assert!(
            MaterialTextureSet::new(
                MaterialTextureLayer::new(0).unwrap(),
                image(4, 4),
                image(4, 4),
                image(2, 2),
                [2.0, 2.0],
            )
            .is_err()
        );
    }

    #[test]
    fn mip_generation_reaches_one_pixel_and_preserves_flat_normals() {
        assert_eq!(mip_level_count(1024, 1024), 11);
        assert_eq!(mip_level_count(1024, 512), 11);
        let flat_normal = [128, 128, 255, 255];
        let mip = downsample_rgba8(&flat_normal.repeat(4), 2, 2, 1, 1, MipFilter::Normal);
        assert!((i16::from(mip[0]) - 128).abs() <= 1);
        assert!((i16::from(mip[1]) - 128).abs() <= 1);
        assert_eq!(mip[2], 255);
        assert_eq!(mip[3], 255);
    }

    #[test]
    fn hdr_mips_average_linear_radiance() {
        let source = vec![
            [4.0, 0.0, 0.0, 1.0],
            [0.0, 4.0, 0.0, 1.0],
            [0.0, 0.0, 4.0, 1.0],
            [4.0, 4.0, 4.0, 1.0],
        ];
        assert_eq!(
            downsample_hdr(&source, 2, 2, 1, 1),
            vec![[2.0, 2.0, 2.0, 1.0]]
        );
    }

    #[test]
    fn dominant_hdr_light_aligns_to_requested_world_azimuth() {
        let mut pixels = vec![[0.1, 0.1, 0.1, 1.0]; 8];
        pixels[6] = [100.0, 100.0, 100.0, 1.0];
        let image = EquirectangularImage::new(8, 1, pixels).unwrap();
        let source_longitude = image.dominant_light_longitude_radians().unwrap();
        assert!((source_longitude - 0.625 * std::f32::consts::PI).abs() < 1.0e-6);
        let world_azimuth = -0.3;
        let yaw = image.yaw_to_align_dominant_light(world_azimuth).unwrap();
        assert!(wrap_angle(world_azimuth + yaw - source_longitude).abs() < 1.0e-6);
    }

    #[test]
    fn rejects_non_opaque_albedo_in_opaque_pipeline() {
        let mut albedo = image(2, 2);
        albedo.pixels[3] = 254;
        assert!(
            MaterialTextureSet::new(
                MaterialTextureLayer::new(0).unwrap(),
                albedo,
                image(2, 2),
                image(2, 2),
                [1.0, 1.0],
            )
            .is_err()
        );
    }

    #[test]
    fn srgb_mips_average_in_linear_light() {
        let samples = [
            [0, 0, 0, 255],
            [255, 255, 255, 255],
            [0, 0, 0, 255],
            [255, 255, 255, 255],
        ];
        let result = average_srgb(samples);
        assert!((186..=189).contains(&result[0]));
        assert_eq!(result[0], result[1]);
        assert_eq!(result[1], result[2]);
        assert_eq!(result[3], 255);
    }
}
