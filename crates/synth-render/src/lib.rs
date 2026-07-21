//! Native, headless WGPU rendering for synthetic roof-training data.
//!
//! The renderer deliberately owns no window or swapchain. It renders one mesh
//! into aligned RGB, semantic-ID, face-coordinate, and depth targets and reads
//! those targets back for dataset encoding and annotation validation.

use std::sync::mpsc;

use bytemuck::{Pod, Zeroable};
use glam::{Mat3, Mat4, Quat, Vec3, Vec4};
use half::f16;
use serde::{Deserialize, Serialize};
use synth_data::{CameraIntrinsics, CameraModel, DistortionModel, ImageTransform, RigidTransform};
use thiserror::Error;
use wgpu::util::DeviceExt;

mod assets;
mod scene;

pub use assets::{
    EquirectangularImage, MaterialTextureLayer, MaterialTextureSet, PreparedRenderAssets,
    RenderAssetBundle, Rgba8Image,
};
pub use scene::{
    BackgroundBuilding, BackgroundBuildingKind, BuildingExtensionDescription,
    BuildingExtensionKind, BuildingExtensionRoof, BuildingSide, EnvironmentDomain,
    FacadeDescription, MaterialSelection, RenderEnvironment, SceneDescription, SceneLight,
    SceneOccluder, SceneOccluderKind, SignDescription, SignMount, SignStyle, SiteInfrastructure,
    SiteRoadKind, TimeOfDay, TreeInstance, TreeKind, UtilityLineDescription,
    UtilityPoleDescription, WeatherAppearance,
};

const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;
const SEMANTIC_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R32Uint;
const FACE_COORD_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rg16Float;
const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
const SHADOW_MAP_SIZE: u32 = 1_024;

const SHADER: &str = include_str!("render.wgsl");

/// Typed procedural surface treatment, independent from a physical texture layer.
///
/// Texture-array indices are an upload detail and must never be used as material
/// semantics. A vertex may inherit its logical slot's pattern or override it for
/// heterogeneous geometry such as neighbouring buildings and vehicle glazing.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SurfacePattern(u32);

impl SurfacePattern {
    /// Inherit the pattern selected for the vertex's logical material slot.
    pub const INHERIT: Self = Self(0);
    /// Smooth painted or unpatterned surface.
    pub const SMOOTH: Self = Self(1);
    /// Masonry brick bond.
    pub const BRICK: Self = Self(2);
    /// Narrow vertical metal or composite cladding panels.
    pub const VERTICAL_CLADDING: Self = Self(3);
    /// Repeated roof tile or sheet seams.
    pub const ROOF_SEAMS: Self = Self(4);
    /// Coarse asphalt aggregate.
    pub const ASPHALT: Self = Self(5);
    /// Restaurant or building window glass, eligible for phase-gated emission.
    pub const BUILDING_GLASS: Self = Self(6);
    /// Vehicle glazing, which reflects but never emits like occupied windows.
    pub const VEHICLE_GLASS: Self = Self(7);
    /// Background wall with a procedural window grid.
    pub const BACKGROUND_WINDOWS: Self = Self(8);
    /// Pizza Hut sign artwork.
    pub const PIZZA_HUT_SIGN: Self = Self(9);
    /// Generic unbranded road, pole, or luminaire panel.
    pub const GENERIC_SIGN: Self = Self(10);
    /// Brick background wall with an occupied-window grid on vertical faces.
    pub const BACKGROUND_WINDOWS_BRICK: Self = Self(11);
    /// Clad background wall with an occupied-window grid on vertical faces.
    pub const BACKGROUND_WINDOWS_CLADDING: Self = Self(12);

    /// Returns the stable shader representation.
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    const fn is_known(self) -> bool {
        self.0 <= Self::BACKGROUND_WINDOWS_CLADDING.0
    }
}

/// A vertex carrying both appearance geometry and exact structural labels.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct RenderVertex {
    /// World-space position, in the generator's shared coordinate system.
    pub position: [f32; 3],
    /// World-space surface normal.
    pub normal: [f32; 3],
    /// Normalized local coordinates within the semantic roof face.
    pub face_coord: [f32; 2],
    /// Stable semantic face identifier. Zero is reserved for background.
    pub semantic_id: u32,
    /// Appearance material selected independently from semantic labels.
    pub material: MaterialSlot,
    /// Per-instance RGB tint and material-specific scalar control.
    pub appearance: [f32; 4],
    /// Procedural treatment override, or [`SurfacePattern::INHERIT`].
    pub pattern: SurfacePattern,
}

impl RenderVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 7] = wgpu::vertex_attr_array![
        0 => Float32x3,
        1 => Float32x3,
        2 => Float32x2,
        3 => Uint32,
        4 => Uint32,
        5 => Float32x4,
        6 => Uint32
    ];

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

/// Appearance material used by the compact built-in scene shader.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct MaterialSlot(u32);

impl MaterialSlot {
    /// Roof finish.
    pub const ROOF: Self = Self(0);
    /// Exterior wall finish.
    pub const WALL: Self = Self(1);
    /// Ground or paving finish.
    pub const GROUND: Self = Self(2);
    /// Neutral foreground/background occluder finish.
    pub const OCCLUDER: Self = Self(3);
    /// Reflective window glazing.
    pub const GLASS: Self = Self(4);
    /// Light facade trim, mullions, and curb faces.
    pub const TRIM: Self = Self(5);
    /// Branded or generic illuminated signage.
    pub const SIGN: Self = Self(6);
    /// Parking lot and road asphalt.
    pub const ASPHALT: Self = Self(7);
    /// Tree canopies and planted shrubs.
    pub const FOLIAGE: Self = Self(8);
    /// Tree trunks and utility poles.
    pub const TRUNK: Self = Self(9);
    /// Distant commercial and city building shells.
    pub const BACKGROUND_WALL: Self = Self(10);
    /// Rooftop equipment, vehicles, and utility hardware.
    pub const METAL: Self = Self(11);
    /// Painted parking and road markings.
    pub const MARKING: Self = Self(12);
    /// Concrete curbs, paths, and foundations.
    pub const CONCRETE: Self = Self(13);
    /// Weathered outline left after original signage was removed.
    pub const GHOST_SIGN: Self = Self(14);
    /// Replacement tenant branding on a former restaurant.
    pub const TENANT_SIGN: Self = Self(15);

    const COUNT: usize = 16;

    /// Returns the stable integer consumed by the render shader.
    #[must_use]
    pub const fn as_u32(self) -> u32 {
        self.0
    }
}

/// Indexed geometry accepted by the synthetic renderer.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RenderMesh {
    /// Vertices with semantic attributes.
    pub vertices: Vec<RenderVertex>,
    /// Triangle-list indices.
    pub indices: Vec<u32>,
    /// Environment controls used by procedural surfaces and sky fallback.
    pub environment: RenderEnvironment,
    /// Logical material slots mapped to resident physical texture layers.
    pub materials: MaterialSelection,
    /// Local lights used primarily for credible night scenes.
    pub lights: Vec<SceneLight>,
}

impl RenderMesh {
    /// Converts the shared parametric roof mesh without changing vertex order,
    /// face coordinates, or stable semantic IDs.
    #[must_use]
    pub fn from_roof(roof: &roof_geometry::RoofGeometry) -> Self {
        Self {
            vertices: roof
                .mesh
                .vertices
                .iter()
                .map(|vertex| RenderVertex {
                    position: vertex.position,
                    normal: vertex.normal,
                    face_coord: vertex.face_coord,
                    semantic_id: vertex.face_id.as_u32(),
                    material: MaterialSlot::ROOF,
                    appearance: [0.0; 4],
                    pattern: SurfacePattern::INHERIT,
                })
                .collect(),
            indices: roof.mesh.indices.clone(),
            environment: RenderEnvironment::default(),
            materials: MaterialSelection::default(),
            lights: Vec::new(),
        }
    }

    /// Builds a simple complete scene from shared roof geometry, a building
    /// envelope, and a ground plane. Non-roof geometry writes semantic ID zero.
    pub fn roof_building_scene(
        roof: &roof_geometry::RoofGeometry,
        footprint_width: f32,
        footprint_depth: f32,
        wall_height: f32,
        ground_half_extent: f32,
    ) -> Result<Self, RenderError> {
        let description = SceneDescription::contextual(
            footprint_width,
            footprint_depth,
            wall_height,
            ground_half_extent,
            RenderEnvironment::default(),
        )?;
        Self::from_scene(roof, &description)
    }

    /// Checks the mesh before any GPU resources are allocated.
    pub fn validate(&self) -> Result<(), RenderError> {
        if self.vertices.is_empty() {
            return Err(RenderError::EmptyMesh);
        }
        if self.indices.is_empty() || !self.indices.len().is_multiple_of(3) {
            return Err(RenderError::InvalidIndexCount(self.indices.len()));
        }
        if let Some(index) = self
            .indices
            .iter()
            .copied()
            .find(|&index| index as usize >= self.vertices.len())
        {
            return Err(RenderError::IndexOutOfBounds {
                index,
                vertex_count: self.vertices.len(),
            });
        }
        if self.vertices.iter().any(|vertex| {
            vertex
                .position
                .iter()
                .chain(vertex.normal.iter())
                .chain(vertex.face_coord.iter())
                .chain(vertex.appearance.iter())
                .any(|value| !value.is_finite())
                || !vertex.pattern.is_known()
        }) {
            return Err(RenderError::NonFiniteVertex);
        }
        if self.lights.len() > SceneUniform::MAX_LIGHTS
            || self.lights.iter().any(|light| !light.is_valid())
            || !self.environment.is_valid()
        {
            return Err(RenderError::InvalidSceneDescription);
        }
        Ok(())
    }
}

/// Pinhole camera parameters for one synthetic frame.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RenderCamera {
    /// Exact camera-to-world transform used by the dataset record.
    pub world_from_camera: RigidTransform,
    /// Exact output-image pinhole calibration, in pixels.
    pub intrinsics: CameraIntrinsics,
    /// Near clipping plane.
    pub near: f32,
    /// Far clipping plane.
    pub far: f32,
}

impl RenderCamera {
    /// Creates an upright look-at camera with centered square pixels.
    #[must_use]
    pub fn look_at(
        position: [f32; 3],
        target: [f32; 3],
        width: u32,
        height: u32,
        vertical_fov_radians: f32,
    ) -> Self {
        let eye = Vec3::from_array(position);
        let target = Vec3::from_array(target);
        let forward = (target - eye).normalize_or_zero();
        let right = forward.cross(Vec3::Y).normalize_or_zero();
        let up = right.cross(forward);
        let rotation = if forward == Vec3::ZERO || right == Vec3::ZERO {
            Quat::from_array([f32::NAN; 4])
        } else {
            Quat::from_mat3(&Mat3::from_cols(right, up, -forward)).normalize()
        };
        let focal_length = height as f32 / (2.0 * (vertical_fov_radians * 0.5).tan());
        Self {
            world_from_camera: RigidTransform {
                translation: synth_data::Vec3::new(position[0], position[1], position[2]),
                rotation_xyzw: rotation.to_array(),
            },
            intrinsics: CameraIntrinsics {
                width,
                height,
                fx: focal_length,
                fy: focal_length,
                cx: width as f32 * 0.5,
                cy: height as f32 * 0.5,
                skew: 0.0,
            },
            near: 0.05,
            far: 250.0,
        }
    }

    /// Converts a schema camera into the exact native renderer camera.
    pub fn from_camera_model(model: CameraModel) -> Result<Self, RenderError> {
        if model.distortion != DistortionModel::None
            || model.output_from_sensor != ImageTransform::IDENTITY
        {
            return Err(RenderError::UnsupportedCameraEffects);
        }
        let camera = Self {
            world_from_camera: model.world_from_camera,
            intrinsics: model.intrinsics,
            near: 0.05,
            far: 250.0,
        };
        camera.validate()?;
        Ok(camera)
    }

    /// Camera origin in world coordinates.
    #[must_use]
    pub fn position(self) -> [f32; 3] {
        let position = self.world_from_camera.translation;
        [position.x, position.y, position.z]
    }

    /// Vertical field of view implied by the stored image calibration.
    #[must_use]
    pub fn vertical_fov_radians(self) -> f32 {
        2.0 * (self.intrinsics.height as f32 * 0.5 / self.intrinsics.fy).atan()
    }

    fn validate(self) -> Result<(), RenderError> {
        let intrinsics = self.intrinsics;
        if intrinsics.width == 0
            || intrinsics.height == 0
            || !intrinsics.fx.is_finite()
            || !intrinsics.fy.is_finite()
            || !intrinsics.cx.is_finite()
            || !intrinsics.cy.is_finite()
            || !intrinsics.skew.is_finite()
            || intrinsics.fx <= 0.0
            || intrinsics.fy <= 0.0
            || !self.world_from_camera.is_valid()
            || !self.near.is_finite()
            || !self.far.is_finite()
            || self.near <= 0.0
            || self.far <= self.near
        {
            return Err(RenderError::InvalidCamera);
        }
        Ok(())
    }

    fn view_projection(self, width: u32, height: u32) -> Result<Mat4, RenderError> {
        self.validate()?;
        if width != self.intrinsics.width || height != self.intrinsics.height {
            return Err(RenderError::CameraDimensionMismatch {
                camera_width: self.intrinsics.width,
                camera_height: self.intrinsics.height,
                render_width: width,
                render_height: height,
            });
        }

        let rotation = Quat::from_array(self.world_from_camera.rotation_xyzw).normalize();
        let position = Vec3::from_array(self.position());
        let view = Mat4::from_rotation_translation(rotation, position).inverse();
        let intrinsics = self.intrinsics;
        let width = width as f32;
        let height = height as f32;
        let depth_scale = self.far / (self.near - self.far);
        let projection = Mat4::from_cols(
            Vec4::new(2.0 * intrinsics.fx / width, 0.0, 0.0, 0.0),
            Vec4::new(
                -2.0 * intrinsics.skew / width,
                2.0 * intrinsics.fy / height,
                0.0,
                0.0,
            ),
            Vec4::new(
                1.0 - 2.0 * intrinsics.cx / width,
                2.0 * intrinsics.cy / height - 1.0,
                depth_scale,
                -1.0,
            ),
            Vec4::new(0.0, 0.0, depth_scale * self.near, 0.0),
        );
        Ok(projection * view)
    }

    /// Projects a world-space point into the exact output coordinates used by
    /// the renderer. This is the basis for keypoint and edge annotations.
    pub fn project(
        self,
        world_position: [f32; 3],
        width: u32,
        height: u32,
    ) -> Result<ProjectedPoint, RenderError> {
        if width == 0 || height == 0 {
            return Err(RenderError::InvalidDimensions { width, height });
        }
        let view_projection = self.view_projection(width, height)?;
        let clip = view_projection * Vec4::from((Vec3::from_array(world_position), 1.0));
        if !clip.is_finite() || clip.w <= 0.0 {
            return Ok(ProjectedPoint {
                pixel: [f32::NAN; 2],
                depth: f32::NAN,
                in_frame: false,
            });
        }
        let ndc = clip.truncate() / clip.w;
        let pixel = [
            (ndc.x * 0.5 + 0.5) * width as f32,
            (0.5 - ndc.y * 0.5) * height as f32,
        ];
        let in_frame = (-1.0..=1.0).contains(&ndc.x)
            && (-1.0..=1.0).contains(&ndc.y)
            && (0.0..=1.0).contains(&ndc.z);
        Ok(ProjectedPoint {
            pixel,
            depth: ndc.z,
            in_frame,
        })
    }

    /// Converts a zero-to-one WGPU depth value back to positive camera-space
    /// distance. This is useful for visibility tests because a fixed epsilon in
    /// nonlinear depth space becomes far too permissive at outdoor distances.
    #[must_use]
    pub fn linearize_depth(self, depth: f32) -> Option<f32> {
        if !depth.is_finite()
            || !(0.0..=1.0).contains(&depth)
            || !self.near.is_finite()
            || !self.far.is_finite()
            || self.near <= 0.0
            || self.far <= self.near
        {
            return None;
        }
        Some(self.far * self.near / (self.far - depth * (self.far - self.near)))
    }
}

/// Image-space projection of one world-space structural point.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProjectedPoint {
    /// Pixel coordinates measured from the output's top-left corner.
    pub pixel: [f32; 2],
    /// Normalized device depth in the WGPU zero-to-one range.
    pub depth: f32,
    /// Whether the projected point lies inside the camera frustum.
    pub in_frame: bool,
}

/// Appearance and output settings for one render.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RenderSettings {
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Camera used for all aligned render targets.
    pub camera: RenderCamera,
    /// Linear RGBA roof base colour.
    pub roof_color: [f32; 4],
    /// Linear RGBA exterior wall colour.
    pub wall_color: [f32; 4],
    /// Linear RGBA ground colour.
    pub ground_color: [f32; 4],
    /// Linear RGBA colour for simple generated occluders.
    pub occluder_color: [f32; 4],
    /// Linear RGBA clear colour.
    pub background_color: [f32; 4],
    /// Direction from the surface towards the key light.
    pub light_direction: [f32; 3],
    /// Relative strength of direct sun or moon illumination.
    pub direct_light: f32,
    /// Strength of shadow-independent illumination.
    pub ambient_light: f32,
    /// Sampled physical roughness for the target roof.
    pub roof_roughness: f32,
    /// Sampled physical roughness for the target exterior wall.
    pub wall_roughness: f32,
    /// Sampled physical roughness for the site ground material.
    pub ground_roughness: f32,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            width: 640,
            height: 480,
            camera: RenderCamera::look_at(
                [9.0, 6.0, 11.0],
                [0.0, 1.5, 0.0],
                640,
                480,
                52.0_f32.to_radians(),
            ),
            roof_color: [0.68, 0.035, 0.025, 1.0],
            wall_color: [0.46, 0.31, 0.2, 1.0],
            ground_color: [0.23, 0.25, 0.25, 1.0],
            occluder_color: [0.16, 0.27, 0.12, 1.0],
            background_color: [0.35, 0.58, 0.82, 1.0],
            light_direction: [-0.4, 0.8, 0.3],
            direct_light: 1.0,
            ambient_light: 0.28,
            roof_roughness: 0.78,
            wall_roughness: 0.85,
            ground_roughness: 0.92,
        }
    }
}

impl RenderSettings {
    /// Creates the canonical render settings for a sampled scene and camera.
    ///
    /// Sampled sun and sky values remain absolute renderer intensities. They are
    /// deliberately not converted to a ratio, so changing either magnitude has
    /// the expected photometric effect without changing the other.
    pub fn from_sampled(
        scene: &synth_data::SampledScene,
        camera: CameraModel,
    ) -> Result<Self, RenderError> {
        let intrinsics = camera.intrinsics;
        let sun_azimuth = scene.lighting.sun_azimuth_degrees.to_radians();
        let sun_elevation = scene.lighting.sun_elevation_degrees.to_radians();
        let ground = scene.composition.ground_material.as_ref();
        let settings = Self {
            width: intrinsics.width,
            height: intrinsics.height,
            camera: RenderCamera::from_camera_model(camera)?,
            roof_color: sampled_material_color(&scene.roof_material),
            wall_color: sampled_material_color(&scene.wall_material),
            ground_color: ground.map_or([0.22, 0.23, 0.21, 1.0], sampled_material_color),
            occluder_color: [0.12, 0.25, 0.09, 1.0],
            background_color: [0.34, 0.57, 0.82, 1.0],
            light_direction: [
                sun_elevation.cos() * sun_azimuth.cos(),
                sun_elevation.sin(),
                sun_elevation.cos() * sun_azimuth.sin(),
            ],
            direct_light: scene.lighting.sun_intensity,
            ambient_light: scene.lighting.sky_intensity,
            roof_roughness: scene.roof_material.roughness,
            wall_roughness: scene.wall_material.roughness,
            ground_roughness: ground.map_or(0.92, |material| material.roughness),
        };
        validate_settings(&settings)?;
        Ok(settings)
    }
}

fn sampled_material_color(material: &synth_data::SampledMaterial) -> [f32; 4] {
    let faded = material
        .base_color_srgb
        .map(|channel| channel + (0.5 - channel) * material.weathering * 0.24);
    [
        srgb_to_linear(faded[0]),
        srgb_to_linear(faded[1]),
        srgb_to_linear(faded[2]),
        1.0,
    ]
}

fn srgb_to_linear(channel: f32) -> f32 {
    if channel <= 0.040_45 {
        channel / 12.92
    } else {
        ((channel + 0.055) / 1.055).powf(2.4)
    }
}

/// CPU-owned outputs of one aligned offscreen render.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderedFrame {
    /// Width shared by every output.
    pub width: u32,
    /// Height shared by every output.
    pub height: u32,
    /// Row-major sRGB RGBA8 pixels.
    pub color_rgba8: Vec<u8>,
    /// Row-major semantic face identifiers. Zero denotes background.
    pub semantic_ids: Vec<u32>,
    /// Roof-only semantic IDs before building or scene occlusion, clipped to the frame.
    pub amodal_semantic_ids: Vec<u32>,
    /// Row-major normalized half-precision face coordinates; non-roof pixels are zero.
    pub face_coordinates: Vec<[f32; 2]>,
    /// Row-major normalized device depth, used for ground-truth visibility checks.
    pub depth: Vec<f32>,
}

impl RenderedFrame {
    /// Returns the number of target pixels.
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        self.width as usize * self.height as usize
    }
}

/// A native WGPU device and the immutable pipelines used for dataset rendering.
pub struct OffscreenRenderer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_info: wgpu::AdapterInfo,
    pipeline: wgpu::RenderPipeline,
    amodal_pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    sky_pipeline: wgpu::RenderPipeline,
    uniform_layout: wgpu::BindGroupLayout,
    shadow_uniform_layout: wgpu::BindGroupLayout,
    shadow_sampler: wgpu::Sampler,
    _shadow_map: wgpu::Texture,
    shadow_view: wgpu::TextureView,
    fallback_assets: PreparedRenderAssets,
    material_anisotropy_clamp: u16,
}

impl OffscreenRenderer {
    /// Creates a renderer using a high-performance native adapter when available.
    pub async fn new() -> Result<Self, RenderError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                ..Default::default()
            })
            .await?;
        let adapter_info = adapter.get_info();
        let material_anisotropy_clamp = if adapter
            .get_downlevel_capabilities()
            .flags
            .contains(wgpu::DownlevelFlags::ANISOTROPIC_FILTERING)
        {
            8
        } else {
            1
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("roof-synth device"),
                ..Default::default()
            })
            .await?;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("roof-synth aligned-target shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER.into()),
        });
        let uniform_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("roof-synth scene uniform layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                texture_layout_entry(
                    1,
                    wgpu::TextureSampleType::Depth,
                    wgpu::TextureViewDimension::D2,
                ),
                sampler_layout_entry(2, wgpu::SamplerBindingType::Comparison),
                texture_layout_entry(
                    3,
                    wgpu::TextureSampleType::Float { filterable: true },
                    wgpu::TextureViewDimension::D2Array,
                ),
                texture_layout_entry(
                    4,
                    wgpu::TextureSampleType::Float { filterable: true },
                    wgpu::TextureViewDimension::D2Array,
                ),
                texture_layout_entry(
                    5,
                    wgpu::TextureSampleType::Float { filterable: true },
                    wgpu::TextureViewDimension::D2Array,
                ),
                sampler_layout_entry(6, wgpu::SamplerBindingType::Filtering),
                texture_layout_entry(
                    7,
                    wgpu::TextureSampleType::Float { filterable: true },
                    wgpu::TextureViewDimension::D2,
                ),
                sampler_layout_entry(8, wgpu::SamplerBindingType::Filtering),
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("roof-synth pipeline layout"),
            bind_group_layouts: &[Some(&uniform_layout)],
            immediate_size: 0,
        });
        let shadow_uniform_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("roof-synth shadow uniform layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let shadow_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("roof-synth shadow pipeline layout"),
                bind_group_layouts: &[Some(&shadow_uniform_layout)],
                immediate_size: 0,
            });
        let targets = [
            Some(wgpu::ColorTargetState {
                format: COLOR_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            Some(wgpu::ColorTargetState {
                format: SEMANTIC_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
            Some(wgpu::ColorTargetState {
                format: FACE_COORD_FORMAT,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            }),
        ];
        let vertex_buffers = [Some(RenderVertex::layout())];
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("roof-synth aligned-target pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex_main"),
                compilation_options: Default::default(),
                buffers: &vertex_buffers,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment_main"),
                compilation_options: Default::default(),
                targets: &targets,
            }),
            multiview_mask: None,
            cache: None,
        });
        let amodal_targets = [Some(wgpu::ColorTargetState {
            format: SEMANTIC_FORMAT,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let amodal_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("roof-synth amodal roof pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex_main"),
                compilation_options: Default::default(),
                buffers: &vertex_buffers,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("amodal_fragment_main"),
                compilation_options: Default::default(),
                targets: &amodal_targets,
            }),
            multiview_mask: None,
            cache: None,
        });
        let shadow_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("roof-synth directional shadow pipeline"),
            layout: Some(&shadow_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("shadow_vertex_main"),
                compilation_options: Default::default(),
                buffers: &vertex_buffers,
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: Default::default(),
            fragment: None,
            multiview_mask: None,
            cache: None,
        });
        let sky_targets = [Some(wgpu::ColorTargetState {
            format: COLOR_FORMAT,
            blend: None,
            write_mask: wgpu::ColorWrites::ALL,
        })];
        let sky_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("roof-synth environment background pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("sky_vertex_main"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("sky_fragment_main"),
                compilation_options: Default::default(),
                targets: &sky_targets,
            }),
            multiview_mask: None,
            cache: None,
        });
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("roof-synth shadow comparison sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let shadow_map = create_target(
            &device,
            "roof-synth reusable directional shadow map",
            wgpu::Extent3d {
                width: SHADOW_MAP_SIZE,
                height: SHADOW_MAP_SIZE,
                depth_or_array_layers: 1,
            },
            DEPTH_FORMAT,
        );
        let shadow_view = shadow_map.create_view(&Default::default());
        let fallback_assets = PreparedRenderAssets::new(
            &device,
            &queue,
            &RenderAssetBundle::default(),
            material_anisotropy_clamp,
        )?;

        Ok(Self {
            device,
            queue,
            adapter_info,
            pipeline,
            amodal_pipeline,
            shadow_pipeline,
            sky_pipeline,
            uniform_layout,
            shadow_uniform_layout,
            shadow_sampler,
            _shadow_map: shadow_map,
            shadow_view,
            fallback_assets,
            material_anisotropy_clamp,
        })
    }

    /// Returns identifying information for the selected native graphics adapter.
    #[must_use]
    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.adapter_info
    }

    /// Uploads decoded PBR arrays and an optional equirectangular environment.
    ///
    /// Prepared assets remain reusable across frames and do not recreate the WGPU
    /// device or render pipelines. A caller may cheaply prepare one environment per
    /// sequence while retaining the same renderer.
    pub fn prepare_assets(
        &self,
        bundle: &RenderAssetBundle,
    ) -> Result<PreparedRenderAssets, RenderError> {
        PreparedRenderAssets::new(
            &self.device,
            &self.queue,
            bundle,
            self.material_anisotropy_clamp,
        )
    }

    /// Replaces a prepared equirectangular environment without re-uploading PBR arrays.
    ///
    /// The replacement must have the same dimensions as the environment used by
    /// [`Self::prepare_assets`]. This is intended for inexpensive per-sequence HDR
    /// changes when a catalog uses a uniform environment resolution.
    pub fn update_environment(
        &self,
        assets: &mut PreparedRenderAssets,
        image: &EquirectangularImage,
    ) -> Result<(), RenderError> {
        assets.update_environment(&self.queue, image)
    }

    /// Renders with procedural surfaces and sky into aligned structural targets.
    pub fn render(
        &self,
        mesh: &RenderMesh,
        settings: &RenderSettings,
    ) -> Result<RenderedFrame, RenderError> {
        self.render_with_assets(mesh, settings, &self.fallback_assets)
    }

    /// Renders with reusable decoded assets into the same aligned output targets.
    pub fn render_with_assets(
        &self,
        mesh: &RenderMesh,
        settings: &RenderSettings,
        assets: &PreparedRenderAssets,
    ) -> Result<RenderedFrame, RenderError> {
        mesh.validate()?;
        validate_settings(settings)?;
        let maximum_dimension = self.device.limits().max_texture_dimension_2d;
        if settings.width > maximum_dimension || settings.height > maximum_dimension {
            return Err(RenderError::DimensionsExceedDeviceLimit {
                width: settings.width,
                height: settings.height,
                maximum: maximum_dimension,
            });
        }

        let light_view_projection = directional_light_view_projection(mesh, settings)?;
        let uniform = SceneUniform::new(settings, mesh, assets, light_view_projection)?;
        let uniform_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("roof-synth scene uniform"),
                contents: bytemuck::bytes_of(&uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let uniform_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("roof-synth scene bind group"),
            layout: &self.uniform_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.shadow_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.shadow_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(&assets.albedo_view),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(&assets.normal_view),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(&assets.arm_view),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(&assets.material_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(&assets.environment_view),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::Sampler(&assets.environment_sampler),
                },
            ],
        });
        let shadow_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("roof-synth shadow uniform bind group"),
            layout: &self.shadow_uniform_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });
        let vertex_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("roof-synth vertices"),
                contents: bytemuck::cast_slice(&mesh.vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });
        let index_buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("roof-synth indices"),
                contents: bytemuck::cast_slice(&mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });

        let size = wgpu::Extent3d {
            width: settings.width,
            height: settings.height,
            depth_or_array_layers: 1,
        };
        let color = create_target(&self.device, "roof-synth color", size, COLOR_FORMAT);
        let semantic = create_target(
            &self.device,
            "roof-synth semantic ids",
            size,
            SEMANTIC_FORMAT,
        );
        let face_coordinates = create_target(
            &self.device,
            "roof-synth face coordinates",
            size,
            FACE_COORD_FORMAT,
        );
        let depth = create_target(&self.device, "roof-synth depth", size, DEPTH_FORMAT);
        let amodal_semantic = create_target(
            &self.device,
            "roof-synth amodal semantic ids",
            size,
            SEMANTIC_FORMAT,
        );
        let amodal_depth =
            create_target(&self.device, "roof-synth amodal depth", size, DEPTH_FORMAT);

        let color_readback = Readback::new(&self.device, settings.width, settings.height, 4)?;
        let semantic_readback = Readback::new(&self.device, settings.width, settings.height, 4)?;
        let face_readback = Readback::new(&self.device, settings.width, settings.height, 4)?;
        let depth_readback = Readback::new(&self.device, settings.width, settings.height, 4)?;
        let amodal_readback = Readback::new(&self.device, settings.width, settings.height, 4)?;

        let color_view = color.create_view(&Default::default());
        let semantic_view = semantic.create_view(&Default::default());
        let face_view = face_coordinates.create_view(&Default::default());
        let depth_view = depth.create_view(&Default::default());
        let amodal_semantic_view = amodal_semantic.create_view(&Default::default());
        let amodal_depth_view = amodal_depth.create_view(&Default::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("roof-synth render encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("roof-synth directional shadow pass"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadow_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&self.shadow_pipeline);
            pass.set_bind_group(0, &shadow_bind_group, &[]);
            pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.indices.len() as u32, 0, 0..1);
        }
        {
            let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                view: &color_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(to_wgpu_color(settings.background_color)),
                    store: wgpu::StoreOp::Store,
                },
            })];
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("roof-synth visible environment background pass"),
                color_attachments: &color_attachments,
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&self.sky_pipeline);
            pass.set_bind_group(0, &uniform_bind_group, &[]);
            pass.draw(0..3, 0..1);
        }
        {
            let color_attachments = [
                Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &semantic_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
                Some(wgpu::RenderPassColorAttachment {
                    view: &face_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                }),
            ];
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("roof-synth aligned-target pass"),
                color_attachments: &color_attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.indices.len() as u32, 0, 0..1);
        }
        {
            let color_attachments = [Some(wgpu::RenderPassColorAttachment {
                view: &amodal_semantic_view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })];
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("roof-synth amodal roof pass"),
                color_attachments: &color_attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &amodal_depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&self.amodal_pipeline);
            pass.set_bind_group(0, &uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            pass.set_index_buffer(index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..mesh.indices.len() as u32, 0, 0..1);
        }

        color_readback.copy_from(&mut encoder, &color, wgpu::TextureAspect::All);
        semantic_readback.copy_from(&mut encoder, &semantic, wgpu::TextureAspect::All);
        face_readback.copy_from(&mut encoder, &face_coordinates, wgpu::TextureAspect::All);
        depth_readback.copy_from(&mut encoder, &depth, wgpu::TextureAspect::DepthOnly);
        amodal_readback.copy_from(&mut encoder, &amodal_semantic, wgpu::TextureAspect::All);

        self.queue.submit([encoder.finish()]);

        let color_bytes = color_readback.read(&self.device)?;
        let semantic_bytes = semantic_readback.read(&self.device)?;
        let face_bytes = face_readback.read(&self.device)?;
        let depth_bytes = depth_readback.read(&self.device)?;
        let amodal_bytes = amodal_readback.read(&self.device)?;

        Ok(RenderedFrame {
            width: settings.width,
            height: settings.height,
            color_rgba8: color_readback.remove_padding(&color_bytes),
            semantic_ids: decode_u32(&semantic_readback.remove_padding(&semantic_bytes)),
            amodal_semantic_ids: decode_u32(&amodal_readback.remove_padding(&amodal_bytes)),
            face_coordinates: decode_vec2_f16(&face_readback.remove_padding(&face_bytes)),
            depth: decode_f32(&depth_readback.remove_padding(&depth_bytes)),
        })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct SceneUniform {
    view_projection: [[f32; 4]; 4],
    inverse_view_projection: [[f32; 4]; 4],
    light_view_projection: [[f32; 4]; 4],
    roof_color: [f32; 4],
    wall_color: [f32; 4],
    ground_color: [f32; 4],
    occluder_color: [f32; 4],
    background_color: [f32; 4],
    light_direction: [f32; 4],
    camera_position: [f32; 4],
    lighting: [f32; 4],
    environment: [f32; 4],
    color_grading: [f32; 4],
    atmosphere: [f32; 4],
    material_roughness: [f32; 4],
    local_light_positions: [[f32; 4]; SceneUniform::MAX_LIGHTS],
    local_light_colors: [[f32; 4]; SceneUniform::MAX_LIGHTS],
    material_uv_tiling: [[f32; 4]; assets::MAX_TEXTURE_LAYERS],
    material_layers: [[u32; 4]; 4],
    material_patterns: [[u32; 4]; 4],
    texture_flags: [u32; 4],
}

impl SceneUniform {
    const MAX_LIGHTS: usize = 4;

    fn new(
        settings: &RenderSettings,
        mesh: &RenderMesh,
        assets: &PreparedRenderAssets,
        light_view_projection: Mat4,
    ) -> Result<Self, RenderError> {
        let view_projection = settings
            .camera
            .view_projection(settings.width, settings.height)?;
        let light = Vec3::from_array(settings.light_direction).normalize_or_zero();
        if light == Vec3::ZERO {
            return Err(RenderError::InvalidLight);
        }
        let camera_position = settings.camera.position();
        let mut local_light_positions = [[0.0; 4]; Self::MAX_LIGHTS];
        let mut local_light_colors = [[0.0; 4]; Self::MAX_LIGHTS];
        for (index, local_light) in mesh.lights.iter().enumerate() {
            local_light_positions[index] = [
                local_light.position[0],
                local_light.position[1],
                local_light.position[2],
                local_light.range_m,
            ];
            local_light_colors[index] = [
                local_light.color[0],
                local_light.color[1],
                local_light.color[2],
                local_light.intensity,
            ];
        }
        let mut material_layers = [[0_u32; 4]; 4];
        for (index, layer) in mesh.materials.layers().into_iter().enumerate() {
            material_layers[index / 4][index % 4] = layer;
        }
        let mut material_patterns = [[0_u32; 4]; 4];
        for (index, pattern) in mesh.materials.patterns().into_iter().enumerate() {
            material_patterns[index / 4][index % 4] = pattern.as_u32();
        }
        let night = mesh.environment.time_of_day.emission_scale();
        let domain = match mesh.environment.domain {
            EnvironmentDomain::City => 0,
            EnvironmentDomain::Urban => 1,
            EnvironmentDomain::Remote => 2,
        };
        let weather = match mesh.environment.weather {
            scene::WeatherAppearance::Clear => 0.0,
            scene::WeatherAppearance::PartlyCloudy => 1.0,
            scene::WeatherAppearance::Overcast => 2.0,
            scene::WeatherAppearance::Hazy => 3.0,
            scene::WeatherAppearance::AfterRain => 4.0,
        };
        Ok(Self {
            view_projection: view_projection.to_cols_array_2d(),
            inverse_view_projection: view_projection.inverse().to_cols_array_2d(),
            light_view_projection: light_view_projection.to_cols_array_2d(),
            roof_color: settings.roof_color,
            wall_color: settings.wall_color,
            ground_color: settings.ground_color,
            occluder_color: settings.occluder_color,
            background_color: settings.background_color,
            light_direction: [light.x, light.y, light.z, settings.direct_light],
            camera_position: [
                camera_position[0],
                camera_position[1],
                camera_position[2],
                1.0,
            ],
            lighting: [
                settings.ambient_light,
                night,
                mesh.environment.cloud_cover,
                mesh.environment.haze,
            ],
            environment: [
                mesh.environment.environment_exposure,
                mesh.environment.seed as f32,
                mesh.environment.shadow_softness,
                mesh.environment.ground_wetness,
            ],
            color_grading: [
                mesh.environment.exposure_ev,
                mesh.environment.color_temperature_kelvin,
                0.0,
                0.0,
            ],
            atmosphere: [
                mesh.environment.visibility_km,
                mesh.environment.environment_yaw_radians,
                weather,
                assets.environment_radiance_scale,
            ],
            material_roughness: [
                settings.roof_roughness,
                settings.wall_roughness,
                settings.ground_roughness,
                assets.environment_max_lod as f32,
            ],
            local_light_positions,
            local_light_colors,
            material_uv_tiling: assets.material_uv_tiling,
            material_layers,
            material_patterns,
            texture_flags: [
                assets.material_mask,
                u32::from(assets.has_environment),
                mesh.lights.len() as u32,
                domain,
            ],
        })
    }
}

struct Readback {
    buffer: wgpu::Buffer,
    width: u32,
    height: u32,
    bytes_per_pixel: u32,
    padded_bytes_per_row: u32,
}

impl Readback {
    fn new(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        bytes_per_pixel: u32,
    ) -> Result<Self, RenderError> {
        let unpadded = width
            .checked_mul(bytes_per_pixel)
            .ok_or(RenderError::ReadbackLayoutOverflow)?;
        let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let padded_bytes_per_row = unpadded
            .checked_add(alignment - 1)
            .map(|value| value / alignment)
            .and_then(|rows| rows.checked_mul(alignment))
            .ok_or(RenderError::ReadbackLayoutOverflow)?;
        let buffer_size = u64::from(padded_bytes_per_row)
            .checked_mul(u64::from(height))
            .ok_or(RenderError::ReadbackLayoutOverflow)?;
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("roof-synth texture readback"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        Ok(Self {
            buffer,
            width,
            height,
            bytes_per_pixel,
            padded_bytes_per_row,
        })
    }

    fn copy_from(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        texture: &wgpu::Texture,
        aspect: wgpu::TextureAspect,
    ) {
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &self.buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
    }

    fn read(&self, device: &wgpu::Device) -> Result<Vec<u8>, RenderError> {
        let (sender, receiver) = mpsc::sync_channel(1);
        self.buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        device.poll(wgpu::PollType::wait_indefinitely())?;
        receiver
            .recv()
            .map_err(|_| RenderError::ReadbackChannel)??;
        let bytes = self.buffer.slice(..).get_mapped_range()?.to_vec();
        self.buffer.unmap();
        Ok(bytes)
    }

    fn remove_padding(&self, padded: &[u8]) -> Vec<u8> {
        let row_size = (self.width * self.bytes_per_pixel) as usize;
        let padded_size = self.padded_bytes_per_row as usize;
        let mut output = Vec::with_capacity(row_size * self.height as usize);
        for row in padded.chunks_exact(padded_size) {
            output.extend_from_slice(&row[..row_size]);
        }
        output
    }
}

fn create_target(
    device: &wgpu::Device,
    label: &'static str,
    size: wgpu::Extent3d,
    format: wgpu::TextureFormat,
) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT
            | wgpu::TextureUsages::COPY_SRC
            | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    })
}

fn texture_layout_entry(
    binding: u32,
    sample_type: wgpu::TextureSampleType,
    view_dimension: wgpu::TextureViewDimension,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_layout_entry(
    binding: u32,
    sampler_type: wgpu::SamplerBindingType,
) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(sampler_type),
        count: None,
    }
}

fn directional_light_view_projection(
    mesh: &RenderMesh,
    settings: &RenderSettings,
) -> Result<Mat4, RenderError> {
    let mut minimum = Vec3::splat(f32::INFINITY);
    let mut maximum = Vec3::splat(f32::NEG_INFINITY);
    for vertex in mesh.vertices.iter().filter(|vertex| {
        vertex.material != MaterialSlot::GROUND
            && vertex.material != MaterialSlot::ASPHALT
            && vertex.material != MaterialSlot::MARKING
    }) {
        let position = Vec3::from_array(vertex.position);
        minimum = minimum.min(position);
        maximum = maximum.max(position);
    }
    if !minimum.is_finite() || !maximum.is_finite() {
        return Err(RenderError::InvalidSceneDescription);
    }
    let center = (minimum + maximum) * 0.5;
    let radius = ((maximum - minimum).length() * 0.65).max(5.0);
    let light = Vec3::from_array(settings.light_direction).normalize_or_zero();
    if light == Vec3::ZERO {
        return Err(RenderError::InvalidLight);
    }
    let up = if light.dot(Vec3::Y).abs() > 0.96 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let eye = center + light * radius * 2.5;
    let view = glam::camera::rh::view::look_at_mat4(eye, center, up);
    let projection = glam::camera::rh::proj::directx::orthographic(
        -radius,
        radius,
        -radius,
        radius,
        0.1,
        radius * 5.0,
    );
    Ok(projection * view)
}

fn validate_settings(settings: &RenderSettings) -> Result<(), RenderError> {
    if settings.width == 0 || settings.height == 0 {
        return Err(RenderError::InvalidDimensions {
            width: settings.width,
            height: settings.height,
        });
    }
    if settings
        .roof_color
        .iter()
        .chain(settings.wall_color.iter())
        .chain(settings.ground_color.iter())
        .chain(settings.occluder_color.iter())
        .chain(settings.background_color.iter())
        .chain(settings.light_direction.iter())
        .any(|value| !value.is_finite())
        || [
            settings.roof_roughness,
            settings.wall_roughness,
            settings.ground_roughness,
        ]
        .iter()
        .any(|value| !value.is_finite())
        || !settings.direct_light.is_finite()
        || settings.direct_light < 0.0
        || !settings.ambient_light.is_finite()
        || settings.ambient_light < 0.0
        || !(0.0..=1.0).contains(&settings.roof_roughness)
        || !(0.0..=1.0).contains(&settings.wall_roughness)
        || !(0.0..=1.0).contains(&settings.ground_roughness)
    {
        return Err(RenderError::InvalidAppearance);
    }
    Ok(())
}

fn append_quad(mesh: &mut RenderMesh, positions: [[f32; 3]; 4], material: MaterialSlot) {
    append_quad_tinted(mesh, positions, material, [0.0; 4]);
}

fn append_quad_tinted(
    mesh: &mut RenderMesh,
    positions: [[f32; 3]; 4],
    material: MaterialSlot,
    appearance: [f32; 4],
) {
    append_quad_tinted_pattern(
        mesh,
        positions,
        material,
        appearance,
        SurfacePattern::INHERIT,
    );
}

fn append_quad_tinted_pattern(
    mesh: &mut RenderMesh,
    positions: [[f32; 3]; 4],
    material: MaterialSlot,
    appearance: [f32; 4],
    pattern: SurfacePattern,
) {
    let base = mesh.vertices.len() as u32;
    let a = Vec3::from_array(positions[1]) - Vec3::from_array(positions[0]);
    let b = Vec3::from_array(positions[2]) - Vec3::from_array(positions[0]);
    let normal = a.cross(b).normalize().to_array();
    let coords = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
    mesh.vertices.extend(
        positions
            .into_iter()
            .zip(coords)
            .map(|(position, face_coord)| RenderVertex {
                position,
                normal,
                face_coord,
                semantic_id: 0,
                material,
                appearance,
                pattern,
            }),
    );
    mesh.indices
        .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
}

fn to_wgpu_color(color: [f32; 4]) -> wgpu::Color {
    wgpu::Color {
        r: f64::from(color[0]),
        g: f64::from(color[1]),
        b: f64::from(color[2]),
        a: f64::from(color[3]),
    }
}

fn decode_u32(bytes: &[u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| u32::from_ne_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect()
}

fn decode_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("four-byte chunk")))
        .collect()
}

fn decode_vec2_f16(bytes: &[u8]) -> Vec<[f32; 2]> {
    bytes
        .chunks_exact(4)
        .map(|chunk| {
            [
                f16::from_bits(u16::from_ne_bytes(
                    chunk[0..2].try_into().expect("two-byte chunk"),
                ))
                .to_f32(),
                f16::from_bits(u16::from_ne_bytes(
                    chunk[2..4].try_into().expect("two-byte chunk"),
                ))
                .to_f32(),
            ]
        })
        .collect()
}

/// Errors produced during mesh validation, GPU setup, or readback.
#[derive(Debug, Error)]
pub enum RenderError {
    /// No compatible native graphics adapter was found.
    #[error(transparent)]
    RequestAdapter(#[from] wgpu::RequestAdapterError),
    /// WGPU could not create the requested logical device.
    #[error(transparent)]
    RequestDevice(#[from] wgpu::RequestDeviceError),
    /// The output dimensions were empty.
    #[error("invalid render dimensions {width}x{height}")]
    InvalidDimensions {
        /// Requested width.
        width: u32,
        /// Requested height.
        height: u32,
    },
    /// Dimensions exceed the selected device's maximum 2D texture extent.
    #[error("render dimensions {width}x{height} exceed device limit {maximum}")]
    DimensionsExceedDeviceLimit {
        /// Requested width.
        width: u32,
        /// Requested height.
        height: u32,
        /// Maximum width or height supported by the device.
        maximum: u32,
    },
    /// Row alignment or staging-buffer size could not be represented safely.
    #[error("render readback layout overflowed")]
    ReadbackLayoutOverflow,
    /// Camera projection or look-at parameters were invalid.
    #[error("invalid render camera")]
    InvalidCamera,
    /// Camera calibration dimensions differ from the requested target.
    #[error(
        "camera calibration is {camera_width}x{camera_height}, but render target is {render_width}x{render_height}"
    )]
    CameraDimensionMismatch {
        /// Width carried by the camera intrinsics.
        camera_width: u32,
        /// Height carried by the camera intrinsics.
        camera_height: u32,
        /// Requested render width.
        render_width: u32,
        /// Requested render height.
        render_height: u32,
    },
    /// The native renderer was given geometric image effects it does not yet apply.
    #[error("camera distortion or output transform is not supported by this render path")]
    UnsupportedCameraEffects,
    /// Lighting direction had no usable magnitude.
    #[error("invalid light direction")]
    InvalidLight,
    /// Appearance values were non-finite or outside their supported range.
    #[error("invalid appearance settings")]
    InvalidAppearance,
    /// No vertices were supplied.
    #[error("render mesh has no vertices")]
    EmptyMesh,
    /// Indices did not form a non-empty triangle list.
    #[error("render mesh index count {0} is not a non-empty triangle list")]
    InvalidIndexCount(usize),
    /// An index referred beyond the supplied vertex array.
    #[error("mesh index {index} exceeds vertex count {vertex_count}")]
    IndexOutOfBounds {
        /// Invalid index.
        index: u32,
        /// Number of supplied vertices.
        vertex_count: usize,
    },
    /// At least one vertex attribute was not finite.
    #[error("render mesh contains a non-finite vertex attribute")]
    NonFiniteVertex,
    /// Building or ground dimensions could not produce a valid scene shell.
    #[error("invalid building or ground dimensions")]
    InvalidSceneDimensions,
    /// A procedural scene object, environment, or light was invalid.
    #[error("invalid procedural scene description")]
    InvalidSceneDescription,
    /// Decoded texture dimensions, pixels, slots, or UV controls were invalid.
    #[error("invalid decoded texture data")]
    InvalidTextureData,
    /// Maps belonging to one PBR material did not share dimensions.
    #[error("PBR material maps must share identical dimensions")]
    TextureDimensionMismatch,
    /// WGPU polling failed while awaiting a mapped readback buffer.
    #[error(transparent)]
    Poll(#[from] wgpu::PollError),
    /// WGPU failed to map a readback buffer.
    #[error(transparent)]
    BufferMap(#[from] wgpu::BufferAsyncError),
    /// A mapped byte range was inconsistent with the requested buffer range.
    #[error(transparent)]
    MapRange(#[from] wgpu::MapRangeError),
    /// The map callback channel closed unexpectedly.
    #[error("readback callback channel closed")]
    ReadbackChannel,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn triangle() -> RenderMesh {
        RenderMesh {
            vertices: vec![
                RenderVertex {
                    position: [-1.0, -1.0, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    face_coord: [0.0, 0.0],
                    semantic_id: 7,
                    material: MaterialSlot::ROOF,
                    appearance: [0.0; 4],
                    pattern: SurfacePattern::INHERIT,
                },
                RenderVertex {
                    position: [1.0, -1.0, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    face_coord: [1.0, 0.0],
                    semantic_id: 7,
                    material: MaterialSlot::ROOF,
                    appearance: [0.0; 4],
                    pattern: SurfacePattern::INHERIT,
                },
                RenderVertex {
                    position: [0.0, 1.0, 0.0],
                    normal: [0.0, 0.0, 1.0],
                    face_coord: [0.5, 1.0],
                    semantic_id: 7,
                    material: MaterialSlot::ROOF,
                    appearance: [0.0; 4],
                    pattern: SurfacePattern::INHERIT,
                },
            ],
            indices: vec![0, 1, 2],
            ..RenderMesh::default()
        }
    }

    #[test]
    fn validates_triangle_mesh() {
        triangle().validate().unwrap();
    }

    #[test]
    fn sampled_render_settings_preserve_absolute_light_and_roughness() {
        let sampler = synth_data::SequenceSampler::new(synth_data::GeneratorConfig::default())
            .expect("default sampler");
        let mut plan = sampler
            .sample(synth_data::SequenceRequest::procedural(
                "classic_two_stage",
                91,
                synth_data::TargetKind::Target,
            ))
            .expect("sampled sequence");
        plan.scene.lighting.sun_intensity = 4.75;
        plan.scene.lighting.sky_intensity = 0.37;
        plan.scene.roof_material.roughness = 0.61;
        plan.scene.wall_material.roughness = 0.83;
        plan.scene
            .composition
            .ground_material
            .as_mut()
            .expect("sampled ground")
            .roughness = 0.92;

        let settings = RenderSettings::from_sampled(&plan.scene, plan.frames[0].camera).unwrap();
        assert_eq!(settings.direct_light, 4.75);
        assert_eq!(settings.ambient_light, 0.37);
        assert_eq!(settings.roof_roughness, 0.61);
        assert_eq!(settings.wall_roughness, 0.83);
        assert_eq!(settings.ground_roughness, 0.92);
    }

    #[test]
    fn preserves_shared_geometry_semantics() {
        let roof = roof_geometry::generate_roof(&roof_geometry::RoofParameters::default()).unwrap();
        let mesh = RenderMesh::from_roof(&roof);
        mesh.validate().unwrap();
        assert_eq!(mesh.vertices.len(), roof.mesh.vertices.len());
        assert_eq!(
            mesh.vertices[0].semantic_id,
            roof.mesh.vertices[0].face_id.as_u32()
        );
    }

    #[test]
    fn building_scene_keeps_non_roof_semantics_at_zero() {
        let roof = roof_geometry::generate_roof(&roof_geometry::RoofParameters::default()).unwrap();
        let scene = RenderMesh::roof_building_scene(&roof, 20.0, 14.0, 4.0, 40.0).unwrap();
        assert!(scene.vertices.iter().any(|vertex| vertex.semantic_id == 0));
        assert!(
            scene
                .vertices
                .iter()
                .any(|vertex| vertex.semantic_id == roof_geometry::FaceId::LowerFront.as_u32())
        );
        scene.validate().unwrap();
    }

    #[test]
    fn rejects_out_of_bounds_index() {
        let mut mesh = triangle();
        mesh.indices[2] = 3;
        assert!(matches!(
            mesh.validate(),
            Err(RenderError::IndexOutOfBounds { index: 3, .. })
        ));
    }

    #[test]
    fn rejects_invalid_camera() {
        let camera = RenderCamera::look_at([0.0; 3], [0.0; 3], 640, 480, 1.0);
        assert!(matches!(
            camera.view_projection(640, 480),
            Err(RenderError::InvalidCamera)
        ));
    }

    #[test]
    fn projects_camera_target_to_image_center() {
        let camera = RenderCamera::look_at([0.0, 0.0, 5.0], [0.0; 3], 640, 480, 1.0);
        let point = camera.project([0.0; 3], 640, 480).unwrap();
        assert!(point.in_frame);
        assert!((point.pixel[0] - 320.0).abs() < 1.0e-4);
        assert!((point.pixel[1] - 240.0).abs() < 1.0e-4);
    }

    #[test]
    fn projection_preserves_exact_recorded_intrinsics() {
        let camera = RenderCamera::from_camera_model(CameraModel {
            intrinsics: CameraIntrinsics {
                width: 640,
                height: 480,
                fx: 500.0,
                fy: 450.0,
                cx: 311.25,
                cy: 237.75,
                skew: 2.0,
            },
            distortion: DistortionModel::None,
            world_from_camera: RigidTransform::IDENTITY,
            output_from_sensor: ImageTransform::IDENTITY,
        })
        .unwrap();
        let center = camera.project([0.0, 0.0, -5.0], 640, 480).unwrap();
        assert!((center.pixel[0] - 311.25).abs() < 1.0e-4);
        assert!((center.pixel[1] - 237.75).abs() < 1.0e-4);

        let offset = camera.project([1.0, 1.0, -5.0], 640, 480).unwrap();
        assert!((offset.pixel[0] - (311.25 + 100.0 - 0.4)).abs() < 1.0e-4);
        assert!((offset.pixel[1] - (237.75 - 90.0)).abs() < 1.0e-4);
    }

    #[test]
    fn linearizes_projection_depth_at_clip_planes() {
        let camera = RenderCamera::look_at([0.0, 0.0, 5.0], [0.0; 3], 640, 480, 1.0);
        assert!((camera.linearize_depth(0.0).unwrap() - camera.near).abs() < 1.0e-6);
        assert!((camera.linearize_depth(1.0).unwrap() - camera.far).abs() < 0.02);
        assert!(camera.linearize_depth(f32::NAN).is_none());
    }

    #[test]
    fn wgsl_module_passes_frontend_validation() {
        let module = naga::front::wgsl::parse_str(SHADER).expect("WGSL should parse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("WGSL should validate");
        assert_eq!(
            SHADER.matches("surface = apply_ground_wetness").count(),
            1,
            "ground wetness must be applied once after texture resolution"
        );
        assert!(SHADER.contains("let transmittance = exp(-view_distance * extinction)"));
        assert!(SHADER.contains("roughness, 0.0, 1.0) * scene.material_roughness.w"));
        assert_eq!(
            SHADER.matches("final_display_transform(color)").count(),
            2,
            "sky and scene geometry must share one final display transform"
        );
        assert_eq!(
            SHADER.matches("apply_color_grade(linear_radiance)").count(),
            1
        );
        assert!(!SHADER.contains("vec3<f32>(1.0) - exp(-hdr"));
    }

    #[test]
    fn headless_gpu_pipeline_renders_aligned_targets() {
        let renderer = match pollster::block_on(OffscreenRenderer::new()) {
            Ok(renderer) => renderer,
            Err(error) => {
                eprintln!("skipping GPU smoke test because no adapter is available: {error}");
                return;
            }
        };
        let initial_environment = EquirectangularImage::new(4, 2, vec![[0.2; 4]; 8]).unwrap();
        let mut prepared = renderer
            .prepare_assets(&RenderAssetBundle {
                materials: Vec::new(),
                environment: Some(initial_environment),
            })
            .unwrap();
        let scaled_hdr =
            EquirectangularImage::new(4, 2, vec![[421_888.0, 10.0, 1.0, 1.0]; 8]).unwrap();
        renderer
            .update_environment(&mut prepared, &scaled_hdr)
            .unwrap();
        assert!(prepared.environment_radiance_scale > 1.0);
        renderer
            .update_environment(
                &mut prepared,
                &EquirectangularImage::new(4, 2, vec![[0.4; 4]; 8]).unwrap(),
            )
            .unwrap();
        assert!(
            renderer
                .update_environment(
                    &mut prepared,
                    &EquirectangularImage::new(2, 1, vec![[0.4; 4]; 2]).unwrap(),
                )
                .is_err()
        );
        let roof = roof_geometry::generate_roof(&roof_geometry::RoofParameters::default()).unwrap();
        let description = SceneDescription::contextual(
            20.0,
            14.0,
            4.0,
            42.0,
            RenderEnvironment {
                domain: EnvironmentDomain::City,
                time_of_day: TimeOfDay::Night,
                ..RenderEnvironment::default()
            },
        )
        .unwrap();
        let mesh = RenderMesh::from_scene(&roof, &description).unwrap();
        let width = 96;
        let height = 64;
        let settings = RenderSettings {
            width,
            height,
            camera: RenderCamera::look_at(
                [22.0, 9.0, 20.0],
                [0.0, 4.5, 0.0],
                width,
                height,
                52.0_f32.to_radians(),
            ),
            ..RenderSettings::default()
        };
        let frame = renderer
            .render_with_assets(&mesh, &settings, &prepared)
            .unwrap();

        assert_eq!(frame.color_rgba8.len(), frame.pixel_count() * 4);
        assert_eq!(frame.semantic_ids.len(), frame.pixel_count());
        assert_eq!(frame.amodal_semantic_ids.len(), frame.pixel_count());
        assert_eq!(frame.face_coordinates.len(), frame.pixel_count());
        assert_eq!(frame.depth.len(), frame.pixel_count());
        assert!(
            frame
                .color_rgba8
                .chunks_exact(4)
                .any(|pixel| pixel != [0, 0, 0, 0])
        );
        assert!(
            frame
                .amodal_semantic_ids
                .iter()
                .any(|semantic| *semantic != 0)
        );
        assert!(
            frame
                .semantic_ids
                .iter()
                .filter(|semantic| **semantic != 0)
                .count()
                <= frame
                    .amodal_semantic_ids
                    .iter()
                    .filter(|semantic| **semantic != 0)
                    .count()
        );
        assert!(frame.depth.iter().all(|depth| depth.is_finite()));
    }
}
