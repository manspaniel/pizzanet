//! Verified loading for the repository's redistributable rendering assets.

use std::{
    collections::BTreeSet,
    fs,
    path::{Component, Path, PathBuf},
};

use anyhow::{Context, Result, bail};
use image::ImageReader;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use synth_data::{
    DayPhase, SampledEnvironment, SceneDomain, SourceAsset, SourceAssetKind, WeatherPreset,
};
use synth_render::{
    EquirectangularImage, MaterialTextureLayer, MaterialTextureSet, RenderAssetBundle, Rgba8Image,
};

const MANIFEST_FILE: &str = "manifest.json";
const PACK_ID: &str = "pizzahut-synthetic-public-assets-v1";

/// Stable IDs for the complete built-in PBR surface library.
pub const PBR_MATERIAL_IDS: [&str; 4] = [
    "polyhaven_roof_07",
    "polyhaven_corrugated_iron_02",
    "polyhaven_brick_wall_001",
    "polyhaven_clean_asphalt",
];

/// Metadata and paths for the checked-in CC0 rendering pack.
#[derive(Clone, Debug)]
pub struct PublicAssetCatalog {
    root: PathBuf,
    manifest: AssetPackManifest,
}

impl PublicAssetCatalog {
    /// Loads the repository-owned asset pack used by `roof-synth`.
    pub fn load_default() -> Result<Self> {
        Self::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../assets/synthetic"))
    }

    /// Loads and structurally validates an asset manifest rooted at `root`.
    pub fn load(root: impl Into<PathBuf>) -> Result<Self> {
        let root = root.into();
        let manifest_path = root.join(MANIFEST_FILE);
        let manifest_bytes = fs::read(&manifest_path)
            .with_context(|| format!("failed to read {}", manifest_path.display()))?;
        let manifest: AssetPackManifest = serde_json::from_slice(&manifest_bytes)
            .with_context(|| format!("failed to parse {}", manifest_path.display()))?;
        if manifest.schema_version != 1 {
            bail!(
                "unsupported public-asset manifest schema {}",
                manifest.schema_version
            );
        }
        if manifest.pack_id != PACK_ID {
            bail!(
                "unsupported public-asset pack ID {}, expected {PACK_ID}",
                manifest.pack_id
            );
        }
        if manifest.license.spdx_id != "CC0-1.0" {
            bail!(
                "public-asset pack uses unsupported licence {}",
                manifest.license.spdx_id
            );
        }
        let mut ids = BTreeSet::new();
        let mut declared_bytes = 0_u64;
        for asset in &manifest.assets {
            if asset.id.trim().is_empty() || !ids.insert(asset.id.as_str()) {
                bail!("public-asset IDs must be non-empty and unique");
            }
            if asset.files.is_empty() {
                bail!("public asset {} contains no files", asset.id);
            }
            for file in &asset.files {
                validate_relative_path(&file.path)?;
                if file.byte_length == 0 {
                    bail!("public asset {} declares an empty file", file.path);
                }
                declared_bytes = declared_bytes
                    .checked_add(file.byte_length)
                    .context("public-asset byte-length total overflowed u64")?;
                if file.dimensions[0] == 0 || file.dimensions[1] == 0 {
                    bail!("public asset {} has empty dimensions", file.path);
                }
                if file.sha256.len() != 64
                    || !file.sha256.bytes().all(|byte| byte.is_ascii_hexdigit())
                {
                    bail!("public asset {} has an invalid SHA-256", file.path);
                }
            }
        }
        if declared_bytes != manifest.total_asset_bytes {
            bail!(
                "public-asset file lengths total {declared_bytes} bytes, manifest declares {}",
                manifest.total_asset_bytes
            );
        }
        Ok(Self { root, manifest })
    }

    /// Number of logical materials and environments in the pack.
    #[must_use]
    pub fn asset_count(&self) -> usize {
        self.manifest.assets.len()
    }

    /// Returns all logical IDs in deterministic manifest order.
    pub fn asset_ids(&self) -> impl ExactSizeIterator<Item = &str> {
        self.manifest.assets.iter().map(|asset| asset.id.as_str())
    }

    /// Returns environment IDs carrying every requested class tag.
    #[must_use]
    pub fn environment_ids(&self, required_classes: &[&str]) -> Vec<&str> {
        self.manifest
            .assets
            .iter()
            .filter(|asset| asset.kind == AssetKind::Environment)
            .filter(|asset| {
                required_classes.iter().all(|class| {
                    asset
                        .environment_classes
                        .iter()
                        .any(|candidate| candidate == class)
                })
            })
            .map(|asset| asset.id.as_str())
            .collect()
    }

    /// Resolves a correlated scene regime to a phase- and domain-compatible panorama.
    ///
    /// `variant` only selects between equally suitable surroundings; the same
    /// sampled sequence therefore receives the same environment on every run.
    pub fn environment_id_for(
        &self,
        environment: SampledEnvironment,
        variant: u64,
    ) -> Result<&str> {
        let urban = matches!(
            environment.domain,
            SceneDomain::City | SceneDomain::Urban | SceneDomain::Suburban
        );
        let environment_id = match environment.day_phase {
            DayPhase::Twilight if urban => "polyhaven_twilight_sunset",
            DayPhase::Twilight => "polyhaven_dikhololo_sunset",
            DayPhase::Night if urban => "polyhaven_modern_buildings_night",
            DayPhase::Night => "polyhaven_kloppenheim_02",
            DayPhase::Day if environment.weather == WeatherPreset::Overcast => {
                "polyhaven_snow_field_puresky"
            }
            DayPhase::Day if environment.weather == WeatherPreset::AfterRain && urban => {
                "polyhaven_urban_street_04"
            }
            DayPhase::Day if matches!(environment.domain, SceneDomain::Remote) => {
                if variant.is_multiple_of(3) {
                    "polyhaven_kloofendal_43d_clear_puresky"
                } else {
                    "polyhaven_goegap"
                }
            }
            DayPhase::Day
                if matches!(environment.domain, SceneDomain::City | SceneDomain::Urban) =>
            {
                if variant.is_multiple_of(3) {
                    "polyhaven_kloofendal_43d_clear_puresky"
                } else {
                    "polyhaven_urban_street_04"
                }
            }
            DayPhase::Day => match variant % 3 {
                0 => "polyhaven_kloofendal_43d_clear_puresky",
                1 => "polyhaven_goegap",
                _ => "polyhaven_urban_street_04",
            },
        };
        self.find_asset(environment_id, AssetKind::Environment)?;
        Ok(environment_id)
    }

    /// Verifies every checked-in byte stream against its recorded digest.
    pub fn verify_all(&self) -> Result<()> {
        for asset in &self.manifest.assets {
            for file in &asset.files {
                let bytes = self.read_verified(file)?;
                let decoded = ImageReader::new(std::io::Cursor::new(bytes))
                    .with_guessed_format()
                    .context("failed to identify public asset image format")?
                    .decode()
                    .with_context(|| format!("failed to decode {}", file.path))?;
                if [decoded.width(), decoded.height()] != file.dimensions {
                    bail!(
                        "public asset {} decoded as {}x{}, expected {}x{}",
                        file.path,
                        decoded.width(),
                        decoded.height(),
                        file.dimensions[0],
                        file.dimensions[1]
                    );
                }
            }
        }
        Ok(())
    }

    /// Decodes one named LDR material map as tightly packed RGBA8.
    pub fn load_material_map(&self, asset_id: &str, role: &str) -> Result<LoadedRgba8> {
        let asset = self.find_asset(asset_id, AssetKind::Material)?;
        let file = find_file(asset, role)?;
        let bytes = self.read_verified(file)?;
        let image = ImageReader::new(std::io::Cursor::new(bytes))
            .with_guessed_format()?
            .decode()
            .with_context(|| format!("failed to decode {}", file.path))?
            .to_rgba8();
        ensure_dimensions(file, image.width(), image.height())?;
        Ok(LoadedRgba8 {
            width: image.width(),
            height: image.height(),
            pixels: image.into_raw(),
        })
    }

    /// Decodes one complete Poly Haven albedo/normal/ARM material set.
    pub fn load_pbr_material(&self, asset_id: &str) -> Result<LoadedPbrMaterial> {
        let base_color = self.load_material_map(asset_id, "base_color")?;
        let normal = self.load_material_map(asset_id, "normal_opengl")?;
        let arm = self.load_material_map(asset_id, "ambient_occlusion_roughness_metalness")?;
        if (base_color.width, base_color.height) != (normal.width, normal.height)
            || (base_color.width, base_color.height) != (arm.width, arm.height)
        {
            bail!("public material {asset_id} maps have mismatched dimensions");
        }
        Ok(LoadedPbrMaterial {
            id: asset_id.to_owned(),
            base_color,
            normal,
            arm,
        })
    }

    /// Decodes one named Radiance environment as linear floating-point RGBA.
    pub fn load_environment(&self, asset_id: &str) -> Result<LoadedEnvironment> {
        let asset = self.find_asset(asset_id, AssetKind::Environment)?;
        let file = find_file(asset, "equirectangular_environment")?;
        let bytes = self.read_verified(file)?;
        let image = ImageReader::new(std::io::Cursor::new(bytes))
            .with_guessed_format()?
            .decode()
            .with_context(|| format!("failed to decode {}", file.path))?
            .to_rgb32f();
        ensure_dimensions(file, image.width(), image.height())?;
        let pixels = image
            .pixels()
            .map(|pixel| [pixel[0], pixel[1], pixel[2], 1.0])
            .collect();
        Ok(LoadedEnvironment {
            id: asset.id.clone(),
            classes: asset.environment_classes.clone(),
            width: image.width(),
            height: image.height(),
            pixels,
        })
    }

    /// Loads the four resident PBR layers shared by every generated sequence.
    pub fn load_render_material_bundle(&self) -> Result<RenderAssetBundle> {
        let tiling = [[7.0, 5.0], [5.0, 4.0], [9.0, 5.0], [10.0, 10.0]];
        let materials = PBR_MATERIAL_IDS
            .into_iter()
            .enumerate()
            .map(|(index, asset_id)| {
                let material = self.load_pbr_material(asset_id)?;
                MaterialTextureSet::new(
                    MaterialTextureLayer::new(index as u32)?,
                    render_rgba8(material.base_color)?,
                    render_rgba8(material.normal)?,
                    render_rgba8(material.arm)?,
                    tiling[index],
                )
                .map_err(anyhow::Error::from)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(RenderAssetBundle {
            materials,
            environment: None,
        })
    }

    /// Decodes one HDR panorama into the renderer's public upload contract.
    pub fn load_render_environment(&self, asset_id: &str) -> Result<EquirectangularImage> {
        let environment = self.load_environment(asset_id)?;
        Ok(EquirectangularImage::new(
            environment.width,
            environment.height,
            environment.pixels,
        )?)
    }

    /// Converts exact file provenance to the dataset manifest's compact schema.
    #[must_use]
    pub fn dataset_sources(&self) -> Vec<SourceAsset> {
        self.manifest
            .assets
            .iter()
            .flat_map(|asset| {
                asset.files.iter().map(|file| SourceAsset {
                    id: format!("{}:{}", asset.id, file.role),
                    kind: match asset.kind {
                        AssetKind::Material => SourceAssetKind::Texture,
                        AssetKind::Environment => SourceAssetKind::Environment,
                    },
                    content_hash: format!("sha256:{}", file.sha256),
                    license: self.manifest.license.spdx_id.clone(),
                    split_group: None,
                })
            })
            .collect()
    }

    /// Returns the exact dataset-manifest IDs for every file in one logical asset.
    pub fn dataset_source_ids_for(&self, asset_id: &str) -> Result<Vec<String>> {
        let asset = self
            .manifest
            .assets
            .iter()
            .find(|asset| asset.id == asset_id)
            .with_context(|| format!("unknown public asset {asset_id}"))?;
        Ok(asset
            .files
            .iter()
            .map(|file| format!("{}:{}", asset.id, file.role))
            .collect())
    }

    fn find_asset(&self, id: &str, expected: AssetKind) -> Result<&AssetEntry> {
        let asset = self
            .manifest
            .assets
            .iter()
            .find(|asset| asset.id == id)
            .with_context(|| format!("unknown public asset {id}"))?;
        if asset.kind != expected {
            bail!("public asset {id} has the wrong kind");
        }
        Ok(asset)
    }

    fn read_verified(&self, file: &AssetFile) -> Result<Vec<u8>> {
        let path = self.root.join(&file.path);
        let metadata = fs::symlink_metadata(&path)
            .with_context(|| format!("failed to inspect {}", path.display()))?;
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            bail!("public asset {} must be a regular file", path.display());
        }
        if metadata.len() != file.byte_length {
            bail!(
                "public asset {} has length {} bytes, expected {}",
                path.display(),
                metadata.len(),
                file.byte_length
            );
        }
        let bytes = fs::read(&path)
            .with_context(|| format!("failed to read public asset {}", path.display()))?;
        let actual_length =
            u64::try_from(bytes.len()).context("public-asset length exceeds u64")?;
        if actual_length != file.byte_length {
            bail!(
                "public asset {} changed length while reading: got {actual_length} bytes, expected {}",
                path.display(),
                file.byte_length
            );
        }
        let actual = hex_digest(&bytes);
        if actual != file.sha256 {
            bail!(
                "public asset {} failed SHA-256 verification: expected {}, got {}",
                path.display(),
                file.sha256,
                actual
            );
        }
        Ok(bytes)
    }
}

/// Decoded 8-bit colour or data texture.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadedRgba8 {
    /// Pixel width.
    pub width: u32,
    /// Pixel height.
    pub height: u32,
    /// Row-major RGBA bytes.
    pub pixels: Vec<u8>,
}

/// Complete decoded material maps with a shared texel grid.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadedPbrMaterial {
    /// Stable public-asset ID.
    pub id: String,
    /// sRGB albedo pixels.
    pub base_color: LoadedRgba8,
    /// Linear OpenGL-convention tangent-space normals.
    pub normal: LoadedRgba8,
    /// Linear ambient-occlusion, roughness, and metalness channels.
    pub arm: LoadedRgba8,
}

/// Decoded linear equirectangular environment.
#[derive(Clone, Debug, PartialEq)]
pub struct LoadedEnvironment {
    /// Stable public-asset ID.
    pub id: String,
    /// Scene-domain and illumination tags from the asset manifest.
    pub classes: Vec<String>,
    /// Pixel width.
    pub width: u32,
    /// Pixel height.
    pub height: u32,
    /// Row-major linear RGBA pixels.
    pub pixels: Vec<[f32; 4]>,
}

#[derive(Clone, Debug, Deserialize)]
struct AssetPackManifest {
    schema_version: u32,
    pack_id: String,
    total_asset_bytes: u64,
    license: LicenseEntry,
    assets: Vec<AssetEntry>,
}

#[derive(Clone, Debug, Deserialize)]
struct LicenseEntry {
    spdx_id: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum AssetKind {
    Material,
    Environment,
}

#[derive(Clone, Debug, Deserialize)]
struct AssetEntry {
    id: String,
    kind: AssetKind,
    #[serde(default)]
    environment_classes: Vec<String>,
    files: Vec<AssetFile>,
}

#[derive(Clone, Debug, Deserialize)]
struct AssetFile {
    role: String,
    path: String,
    dimensions: [u32; 2],
    byte_length: u64,
    sha256: String,
}

fn find_file<'a>(asset: &'a AssetEntry, role: &str) -> Result<&'a AssetFile> {
    asset
        .files
        .iter()
        .find(|file| file.role == role)
        .with_context(|| format!("public asset {} has no {role} file", asset.id))
}

fn ensure_dimensions(file: &AssetFile, width: u32, height: u32) -> Result<()> {
    if [width, height] != file.dimensions {
        bail!(
            "public asset {} decoded as {width}x{height}, expected {}x{}",
            file.path,
            file.dimensions[0],
            file.dimensions[1]
        );
    }
    Ok(())
}

fn validate_relative_path(path: &str) -> Result<()> {
    let path = Path::new(path);
    if path.as_os_str().is_empty()
        || path
            .components()
            .any(|component| !matches!(component, Component::Normal(_)))
    {
        bail!("public-asset path must be a non-empty relative path");
    }
    Ok(())
}

fn hex_digest(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn render_rgba8(image: LoadedRgba8) -> Result<Rgba8Image> {
    Ok(Rgba8Image::new(image.width, image.height, image.pixels)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_test_pack(root: &Path, pack_id: &str, total_asset_bytes: u64, file_byte_length: u64) {
        let relative_path = "environments/test.hdr";
        let contents = b"test asset bytes";
        fs::create_dir_all(root.join("environments")).unwrap();
        fs::write(root.join(relative_path), contents).unwrap();
        let manifest = serde_json::json!({
            "schema_version": 1,
            "pack_id": pack_id,
            "total_asset_bytes": total_asset_bytes,
            "license": { "spdx_id": "CC0-1.0" },
            "assets": [{
                "id": "test_environment",
                "kind": "environment",
                "environment_classes": ["day"],
                "files": [{
                    "role": "equirectangular_environment",
                    "path": relative_path,
                    "dimensions": [1, 1],
                    "byte_length": file_byte_length,
                    "sha256": hex_digest(contents),
                }],
            }],
        });
        fs::write(
            root.join(MANIFEST_FILE),
            serde_json::to_vec_pretty(&manifest).unwrap(),
        )
        .unwrap();
    }

    fn environment(
        day_phase: DayPhase,
        domain: SceneDomain,
        weather: WeatherPreset,
    ) -> SampledEnvironment {
        SampledEnvironment {
            day_phase,
            domain,
            weather,
            shadow_softness: 0.5,
            ground_wetness: 0.0,
            visibility_km: 30.0,
            color_temperature_k: 4_800.0,
            camera_exposure_ev100: 7.0,
            artificial_light_strength: 0.7,
        }
    }

    #[test]
    fn checked_in_pack_is_complete_and_decodable() {
        let catalog = PublicAssetCatalog::load_default().unwrap();
        assert_eq!(catalog.asset_count(), 12);
        assert_eq!(catalog.dataset_sources().len(), 20);
        assert_eq!(catalog.manifest.pack_id, PACK_ID);
        assert_eq!(catalog.manifest.total_asset_bytes, 18_328_942);
        catalog.verify_all().unwrap();
    }

    #[test]
    fn rejects_wrong_pack_id_and_inconsistent_declared_total() {
        let wrong_id = tempfile::tempdir().unwrap();
        write_test_pack(wrong_id.path(), "another-pack", 16, 16);
        assert!(
            PublicAssetCatalog::load(wrong_id.path())
                .unwrap_err()
                .to_string()
                .contains("unsupported public-asset pack ID")
        );

        let wrong_total = tempfile::tempdir().unwrap();
        write_test_pack(wrong_total.path(), PACK_ID, 17, 16);
        assert!(
            PublicAssetCatalog::load(wrong_total.path())
                .unwrap_err()
                .to_string()
                .contains("file lengths total 16 bytes, manifest declares 17")
        );
    }

    #[test]
    fn rejects_on_disk_length_mismatch_before_hash_verification() {
        let directory = tempfile::tempdir().unwrap();
        write_test_pack(directory.path(), PACK_ID, 17, 17);
        let catalog = PublicAssetCatalog::load(directory.path()).unwrap();
        let file = &catalog.manifest.assets[0].files[0];
        let error = catalog.read_verified(file).unwrap_err().to_string();
        assert!(error.contains("has length 16 bytes, expected 17"));
        assert!(!error.contains("SHA-256"));
    }

    #[test]
    fn finds_distinct_day_twilight_night_and_location_domains() {
        let catalog = PublicAssetCatalog::load_default().unwrap();
        assert!(!catalog.environment_ids(&["day", "urban"]).is_empty());
        assert_eq!(
            catalog.environment_ids(&["twilight", "urban"]),
            ["polyhaven_twilight_sunset"]
        );
        assert_eq!(
            catalog.environment_ids(&["twilight", "remote"]),
            ["polyhaven_dikhololo_sunset"]
        );
        assert_eq!(
            catalog
                .find_asset("polyhaven_twilight_sunset", AssetKind::Environment)
                .unwrap()
                .environment_classes
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            [
                "twilight",
                "urban",
                "suburban",
                "partly_cloudy",
                "low_contrast",
                "natural_light",
            ]
        );
        assert_eq!(
            catalog
                .find_asset("polyhaven_dikhololo_sunset", AssetKind::Environment)
                .unwrap()
                .environment_classes
                .iter()
                .map(String::as_str)
                .collect::<Vec<_>>(),
            [
                "twilight",
                "natural",
                "remote",
                "roadside",
                "partly_cloudy",
                "low_contrast",
            ]
        );
        assert!(!catalog.environment_ids(&["night", "urban"]).is_empty());
        assert!(!catalog.environment_ids(&["day", "remote"]).is_empty());
        assert!(!catalog.environment_ids(&["night", "remote"]).is_empty());
    }

    #[test]
    fn twilight_resolution_uses_only_twilight_domain_classes() {
        let catalog = PublicAssetCatalog::load_default().unwrap();
        let weather_presets = [
            WeatherPreset::Clear,
            WeatherPreset::PartlyCloudy,
            WeatherPreset::Overcast,
            WeatherPreset::Hazy,
            WeatherPreset::AfterRain,
        ];

        for domain in [SceneDomain::City, SceneDomain::Urban, SceneDomain::Suburban] {
            for weather in weather_presets {
                assert_eq!(
                    catalog
                        .environment_id_for(environment(DayPhase::Twilight, domain, weather), 0)
                        .unwrap(),
                    "polyhaven_twilight_sunset"
                );
            }
        }
        for domain in [SceneDomain::Roadside, SceneDomain::Remote] {
            for weather in weather_presets {
                assert_eq!(
                    catalog
                        .environment_id_for(environment(DayPhase::Twilight, domain, weather), 1)
                        .unwrap(),
                    "polyhaven_dikhololo_sunset"
                );
            }
        }

        for environment_id in ["polyhaven_twilight_sunset", "polyhaven_dikhololo_sunset"] {
            let classes = &catalog
                .find_asset(environment_id, AssetKind::Environment)
                .unwrap()
                .environment_classes;
            assert!(classes.iter().any(|class| class == "twilight"));
            assert!(!classes.iter().any(|class| class == "day"));
            assert!(!classes.iter().any(|class| class == "night"));
        }
    }

    #[test]
    fn decodes_surface_and_hdr_pixels() {
        let catalog = PublicAssetCatalog::load_default().unwrap();
        let surface = catalog
            .load_material_map("polyhaven_roof_07", "base_color")
            .unwrap();
        assert_eq!((surface.width, surface.height), (1024, 1024));
        assert_eq!(surface.pixels.len(), 1024 * 1024 * 4);

        let environment = catalog
            .load_environment("polyhaven_modern_buildings_night")
            .unwrap();
        assert_eq!((environment.width, environment.height), (1024, 512));
        assert_eq!(environment.pixels.len(), 1024 * 512);
        assert!(environment.pixels.iter().all(|pixel| {
            pixel
                .iter()
                .all(|channel| channel.is_finite() && *channel >= 0.0)
        }));

        let render_bundle = catalog.load_render_material_bundle().unwrap();
        assert_eq!(render_bundle.materials.len(), PBR_MATERIAL_IDS.len());
        let render_environment = catalog
            .load_render_environment("polyhaven_modern_buildings_night")
            .unwrap();
        assert_eq!(
            (render_environment.width, render_environment.height),
            (1024, 512)
        );
    }
}
