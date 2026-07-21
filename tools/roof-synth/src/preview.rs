//! Bounded, deterministic, static inspection output for generated datasets.

use std::{
    collections::BTreeSet,
    fs::{self, File},
    path::{Path, PathBuf},
};

use anyhow::{Context, Result};
use image::{
    ImageEncoder, Rgb, RgbImage,
    codecs::jpeg::JpegEncoder,
    imageops::{FilterType, replace, resize},
};
use serde::Serialize;

use crate::photometric::PhotometricProfile;

const PREVIEW_SCHEMA_VERSION: u32 = 1;
const DEFAULT_MAX_ENTRIES: usize = 64;
const LOW_OCCLUSION_UPPER_BOUND: f32 = 0.20;
const MEDIUM_OCCLUSION_UPPER_BOUND: f32 = 0.50;

/// Human-readable scene and coverage values shown beside one rendered frame.
#[derive(Clone, Debug, Serialize)]
pub struct PreviewMetadata {
    /// WebDataset sample key.
    pub sample_key: String,
    /// Coherent parent sequence.
    pub sequence_id: String,
    /// Zero-based frame index.
    pub frame_index: u32,
    /// Target or ordinary-building negative.
    pub target_kind: synth_data::TargetKind,
    /// Ordinary roof topology for negative scenes.
    pub ordinary_roof_family: Option<synth_data::OrdinaryRoofFamily>,
    /// Correlated two-stage roof proportion family.
    pub roof_morphology: synth_data::RoofMorphology,
    /// Day, twilight, or night.
    pub day_phase: String,
    /// City, urban, suburban, roadside, or remote.
    pub domain: String,
    /// Correlated weather preset.
    pub weather: String,
    /// Zero or more active, removed, or replacement sign states.
    pub signage: Vec<String>,
    /// Optional structural additions attached to the primary building.
    pub building_extensions: Vec<String>,
    /// Sampled roof material family.
    pub roof_material: String,
    /// Sampled wall material family.
    pub wall_material: String,
    /// Sampled parking/road surface material family.
    pub ground_material: String,
    /// Coherent camera path family.
    pub camera_path: String,
    /// Fixed, smooth-in, or smooth-out focal behavior.
    pub zoom_behavior: String,
    /// Centered or deliberate partial-edge framing.
    pub framing_intent: String,
    /// Distant, normal, close, or partial sampling stratum.
    pub apparent_scale: String,
    /// Public HDR logical ID.
    pub environment_asset_id: String,
    /// Horizontal field of view derived from the exact frame intrinsics.
    pub horizontal_fov_degrees: f32,
    /// Width of the visible, depth-tested roof bounds as a fraction of the frame.
    pub roof_bbox_width_fraction: f32,
    /// Foreground, site, and rooftop occluder families present in the scene.
    pub occluder_kinds: Vec<String>,
    /// Number of explicitly modelled neighbouring buildings.
    pub background_building_count: usize,
    /// Number of explicitly modelled trees and shrubs.
    pub vegetation_count: usize,
    /// Exact file-level source IDs represented in the dataset manifest.
    pub source_asset_ids: Vec<String>,
    /// Exact deterministic RGB-only phone-camera response used for this frame.
    pub photometric_profile: PhotometricProfile,
    /// Fraction of the in-frame amodal roof that remains visible.
    pub visible_fraction: f32,
    /// Fraction hidden behind scene geometry.
    pub occluded_fraction: f32,
    /// Whether the roof intersects an image boundary.
    pub truncated: bool,
}

impl PreviewMetadata {
    fn coverage_keys(&self) -> Vec<String> {
        let occlusion = if self.occluded_fraction < LOW_OCCLUSION_UPPER_BOUND {
            "low"
        } else if self.occluded_fraction < MEDIUM_OCCLUSION_UPPER_BOUND {
            "medium"
        } else {
            "high"
        };
        let mut keys = vec![
            format!("target_kind:{:?}", self.target_kind),
            format!("phase:{}", self.day_phase),
            format!("domain:{}", self.domain),
            format!("phase_domain:{}+{}", self.day_phase, self.domain),
            format!("weather:{}", self.weather),
            format!("roof_material:{}", self.roof_material),
            format!("wall_material:{}", self.wall_material),
            format!("ground_material:{}", self.ground_material),
            format!("environment:{}", self.environment_asset_id),
            format!("camera_path:{}", self.camera_path),
            format!("zoom:{}", self.zoom_behavior),
            format!("framing:{}", self.framing_intent),
            format!("framing_scale:{}", self.apparent_scale),
            format!("fov:{}", fov_bin(self.horizontal_fov_degrees)),
            format!(
                "apparent_scale:{}",
                scale_bin(self.roof_bbox_width_fraction)
            ),
            format!(
                "background_density:{}",
                count_bin(self.background_building_count)
            ),
            format!("vegetation_density:{}", count_bin(self.vegetation_count)),
            format!("occlusion:{occlusion}"),
            format!("truncated:{}", self.truncated),
        ];
        if self.target_kind == synth_data::TargetKind::Target {
            keys.push(format!(
                "roof_morphology:{}",
                roof_morphology_name(self.roof_morphology)
            ));
            keys.push(format!(
                "roof_phase_domain:{}+{}+{}",
                roof_morphology_name(self.roof_morphology),
                self.day_phase,
                self.domain
            ));
        } else if let Some(family) = self.ordinary_roof_family {
            keys.push(format!("ordinary_roof:{}", family.as_str()));
        }
        if self.signage.is_empty() {
            keys.push("signage:none".to_owned());
        } else {
            keys.extend(self.signage.iter().map(|state| format!("signage:{state}")));
        }
        if self.building_extensions.is_empty() {
            keys.push("building_extension:none".to_owned());
        } else {
            keys.extend(
                self.building_extensions
                    .iter()
                    .map(|extension| format!("building_extension:{extension}")),
            );
        }
        if self.occluder_kinds.is_empty() {
            keys.push("occluder:none".to_owned());
        } else {
            keys.extend(
                self.occluder_kinds
                    .iter()
                    .map(|kind| format!("occluder:{kind}")),
            );
        }
        keys
    }
}

const fn roof_morphology_name(morphology: synth_data::RoofMorphology) -> &'static str {
    match morphology {
        synth_data::RoofMorphology::TallEarlyCrown => "tall_early_crown",
        synth_data::RoofMorphology::BalancedClassic => "balanced_classic",
        synth_data::RoofMorphology::LowWideLate => "low_wide_late",
    }
}

fn fov_bin(horizontal_fov_degrees: f32) -> &'static str {
    if horizontal_fov_degrees < 45.0 {
        "narrow"
    } else if horizontal_fov_degrees < 65.0 {
        "standard"
    } else {
        "wide"
    }
}

fn scale_bin(roof_bbox_width_fraction: f32) -> &'static str {
    if roof_bbox_width_fraction < 0.45 {
        "distant"
    } else if roof_bbox_width_fraction < 0.80 {
        "medium"
    } else {
        "close_or_partial"
    }
}

const fn count_bin(count: usize) -> &'static str {
    match count {
        0 => "none",
        1..=2 => "sparse",
        3..=6 => "moderate",
        _ => "dense",
    }
}

/// Byte streams already produced for one selected frame.
#[derive(Clone, Copy, Debug)]
pub struct PreviewImages<'a> {
    /// Rendered RGB JPEG.
    pub rgb_jpeg: &'a [u8],
    /// Binary roof mask PNG.
    pub roof_mask_png: &'a [u8],
    /// Colourized semantic roof-part inspection PNG.
    pub part_preview_png: &'a [u8],
    /// Colourized stable face-ID inspection PNG.
    pub face_preview_png: &'a [u8],
}

/// Deterministically selected inspection frames and their covered categories.
#[derive(Debug)]
pub struct PreviewGallery {
    max_entries: usize,
    covered: BTreeSet<String>,
    temporal_showcase_sequences: BTreeSet<String>,
    entries: Vec<PendingEntry>,
}

impl Default for PreviewGallery {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_ENTRIES)
    }
}

impl PreviewGallery {
    /// Creates a bounded gallery. At least one entry is retained when offered.
    #[must_use]
    pub fn new(max_entries: usize) -> Self {
        Self {
            max_entries: max_entries.max(1),
            covered: BTreeSet::new(),
            temporal_showcase_sequences: BTreeSet::new(),
            entries: Vec::new(),
        }
    }

    /// Keeps a frame only when it adds a previously unseen coverage category.
    pub fn consider(&mut self, metadata: PreviewMetadata, images: PreviewImages<'_>) {
        if self.entries.len() >= self.max_entries {
            return;
        }
        if metadata.frame_index == 0
            && metadata.zoom_behavior != "Fixed"
            && self.temporal_showcase_sequences.len() < 2
        {
            self.temporal_showcase_sequences
                .insert(metadata.sequence_id.clone());
        }
        let coverage = metadata.coverage_keys();
        let temporal_showcase = self
            .temporal_showcase_sequences
            .contains(&metadata.sequence_id);
        if !self.entries.is_empty()
            && !temporal_showcase
            && !coverage.iter().any(|key| !self.covered.contains(key))
        {
            return;
        }
        self.covered.extend(coverage.iter().cloned());
        self.entries.push(PendingEntry {
            metadata,
            coverage,
            rgb_jpeg: images.rgb_jpeg.to_vec(),
            roof_mask_png: images.roof_mask_png.to_vec(),
            part_preview_png: images.part_preview_png.to_vec(),
            face_preview_png: images.face_preview_png.to_vec(),
        });
    }

    /// Writes a self-contained gallery shell and bounded image copies.
    pub fn write(&self, dataset_root: &Path) -> Result<PreviewSummary> {
        let root = dataset_root.join("preview");
        let images = root.join("images");
        fs::create_dir_all(&images)
            .with_context(|| format!("failed to create {}", images.display()))?;

        let mut records = Vec::with_capacity(self.entries.len());
        for entry in &self.entries {
            let key = &entry.metadata.sample_key;
            let rgb = format!("images/{key}.rgb.jpg");
            let roof_mask = format!("images/{key}.roof_mask.png");
            let part_mask = format!("images/{key}.parts.png");
            let face_ids = format!("images/{key}.faces.png");
            write_bytes(&root.join(&rgb), &entry.rgb_jpeg)?;
            write_bytes(&root.join(&roof_mask), &entry.roof_mask_png)?;
            write_bytes(&root.join(&part_mask), &entry.part_preview_png)?;
            write_bytes(&root.join(&face_ids), &entry.face_preview_png)?;
            records.push(PreviewRecord {
                metadata: entry.metadata.clone(),
                coverage: entry.coverage.clone(),
                rgb,
                roof_mask,
                part_mask,
                face_ids,
            });
        }

        let manifest = PreviewManifest {
            schema_version: PREVIEW_SCHEMA_VERSION,
            covered_categories: self.covered.iter().cloned().collect(),
            entries: records,
        };
        let json = serde_json::to_vec_pretty(&manifest)?;
        write_bytes(&root.join("preview.json"), &json)?;
        let embedded_json = serde_json::to_string(&manifest)?.replace('<', "\\u003c");
        let html = GALLERY_HTML.replace("__PREVIEW_DATA__", &embedded_json);
        write_bytes(&root.join("index.html"), html.as_bytes())?;
        let contact_sheet = PathBuf::from("contact-sheet.jpg");
        write_contact_sheet(&dataset_root.join(&contact_sheet), &self.entries)?;
        let stratified_contact_sheets =
            write_stratified_contact_sheets(dataset_root, &self.entries)?;

        Ok(PreviewSummary {
            index: PathBuf::from("preview/index.html"),
            manifest: PathBuf::from("preview/preview.json"),
            contact_sheet,
            stratified_contact_sheets,
            entries: self.entries.len(),
            covered_categories: self.covered.len(),
        })
    }
}

/// Paths and bounded counts reported after gallery publication.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct PreviewSummary {
    /// Dataset-relative HTML entry point.
    pub index: PathBuf,
    /// Dataset-relative machine-readable selection.
    pub manifest: PathBuf,
    /// Compact RGB-only inspection sheet at the dataset root.
    pub contact_sheet: PathBuf,
    /// Positive/negative by day/night inspection sheets that contain entries.
    pub stratified_contact_sheets: Vec<PathBuf>,
    /// Number of copied preview frames.
    pub entries: usize,
    /// Number of distinct stratification keys represented.
    pub covered_categories: usize,
}

#[derive(Clone, Debug)]
struct PendingEntry {
    metadata: PreviewMetadata,
    coverage: Vec<String>,
    rgb_jpeg: Vec<u8>,
    roof_mask_png: Vec<u8>,
    part_preview_png: Vec<u8>,
    face_preview_png: Vec<u8>,
}

#[derive(Clone, Debug, Serialize)]
struct PreviewRecord {
    #[serde(flatten)]
    metadata: PreviewMetadata,
    coverage: Vec<String>,
    rgb: String,
    roof_mask: String,
    part_mask: String,
    face_ids: String,
}

#[derive(Debug, Serialize)]
struct PreviewManifest {
    schema_version: u32,
    covered_categories: Vec<String>,
    entries: Vec<PreviewRecord>,
}

fn write_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    fs::write(path, bytes).with_context(|| format!("failed to write {}", path.display()))
}

fn write_contact_sheet(path: &Path, entries: &[PendingEntry]) -> Result<()> {
    const COLUMNS: u32 = 4;
    const CELL_WIDTH: u32 = 480;
    const CELL_HEIGHT: u32 = 360;
    const GUTTER: u32 = 10;

    let rows = (entries.len() as u32).div_ceil(COLUMNS).max(1);
    let width = COLUMNS * CELL_WIDTH + (COLUMNS + 1) * GUTTER;
    let height = rows * CELL_HEIGHT + (rows + 1) * GUTTER;
    let mut sheet = RgbImage::from_pixel(width, height, Rgb([16, 19, 24]));

    for (index, entry) in entries.iter().enumerate() {
        let source = image::load_from_memory(&entry.rgb_jpeg)
            .with_context(|| format!("failed to decode preview RGB {}", entry.metadata.sample_key))?
            .to_rgb8();
        let scale = (CELL_WIDTH as f32 / source.width() as f32)
            .min(CELL_HEIGHT as f32 / source.height() as f32);
        let target_width = (source.width() as f32 * scale).round().max(1.0) as u32;
        let target_height = (source.height() as f32 * scale).round().max(1.0) as u32;
        let thumbnail = resize(&source, target_width, target_height, FilterType::Lanczos3);
        let column = index as u32 % COLUMNS;
        let row = index as u32 / COLUMNS;
        let x = GUTTER + column * CELL_WIDTH + (CELL_WIDTH - target_width) / 2;
        let y = GUTTER + row * CELL_HEIGHT + (CELL_HEIGHT - target_height) / 2;
        replace(&mut sheet, &thumbnail, i64::from(x), i64::from(y));
    }

    let file =
        File::create(path).with_context(|| format!("failed to create {}", path.display()))?;
    JpegEncoder::new_with_quality(file, 90)
        .write_image(
            sheet.as_raw(),
            sheet.width(),
            sheet.height(),
            image::ExtendedColorType::Rgb8,
        )
        .with_context(|| format!("failed to encode {}", path.display()))
}

fn write_stratified_contact_sheets(
    dataset_root: &Path,
    entries: &[PendingEntry],
) -> Result<Vec<PathBuf>> {
    let strata = [
        (synth_data::TargetKind::Target, false, "target-day"),
        (synth_data::TargetKind::Target, true, "target-night"),
        (synth_data::TargetKind::Negative, false, "negative-day"),
        (synth_data::TargetKind::Negative, true, "negative-night"),
    ];
    let mut written = Vec::new();
    for (kind, night, name) in strata {
        let selected = entries
            .iter()
            .filter(|entry| {
                entry.metadata.target_kind == kind && (entry.metadata.day_phase == "Night") == night
            })
            .cloned()
            .collect::<Vec<_>>();
        if selected.is_empty() {
            continue;
        }
        let relative = PathBuf::from(format!("contact-sheet-{name}.jpg"));
        write_contact_sheet(&dataset_root.join(&relative), &selected)?;
        written.push(relative);
    }
    Ok(written)
}

const GALLERY_HTML: &str = r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Roof synthetic-data preview</title>
<style>
:root{color-scheme:dark;font:14px/1.45 system-ui,sans-serif;background:#111418;color:#e8edf2}body{margin:0;padding:24px}header{max-width:1100px;margin:0 auto 24px}h1{margin:0 0 6px;font-size:24px}.summary{color:#aab5c0}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:18px;max-width:1500px;margin:auto}.card{overflow:hidden;border:1px solid #323941;border-radius:10px;background:#1b2026;box-shadow:0 8px 24px #0005}.frame{aspect-ratio:4/3;background:#090b0e}.frame img{display:block;width:100%;height:100%;object-fit:contain}.body{padding:12px}.title{font-weight:700}.tags{display:flex;flex-wrap:wrap;gap:5px;margin:8px 0}.tag{padding:2px 7px;border-radius:999px;background:#29323b;color:#cbd6df;font-size:12px}.stats{color:#aeb8c2}.switcher{display:flex;gap:6px;margin-top:10px}.switcher button{border:1px solid #4a5661;border-radius:6px;background:#232a31;color:#eef3f7;padding:5px 8px;cursor:pointer}.switcher button:hover{background:#34404a}
</style>
</head>
<body>
<header><h1>Roof synthetic-data preview</h1><div class="summary" id="summary"></div></header>
<main class="grid" id="grid"></main>
<script>
const data=__PREVIEW_DATA__;
const grid=document.getElementById("grid");
document.getElementById("summary").textContent=`${data.entries.length} selected frames · ${data.covered_categories.length} coverage categories`;
for(const entry of data.entries){
 const card=document.createElement("article"); card.className="card";
 const frame=document.createElement("div"); frame.className="frame";
 const image=document.createElement("img"); image.src=entry.rgb; image.alt=entry.sample_key; frame.append(image); card.append(frame);
 const body=document.createElement("div"); body.className="body";
 const title=document.createElement("div"); title.className="title"; title.textContent=`${entry.sample_key} · frame ${entry.frame_index}`; body.append(title);
 const tags=document.createElement("div"); tags.className="tags";
 for(const value of [entry.target_kind,entry.ordinary_roof_family,entry.roof_morphology,entry.day_phase,entry.domain,entry.weather,entry.roof_material,entry.wall_material,entry.ground_material,entry.environment_asset_id,entry.camera_path,entry.zoom_behavior,entry.framing_intent,entry.apparent_scale,...entry.signage,...entry.building_extensions,...entry.occluder_kinds].filter(Boolean)){const tag=document.createElement("span");tag.className="tag";tag.textContent=value;tags.append(tag)} body.append(tags);
 const stats=document.createElement("div"); stats.className="stats"; stats.textContent=`FOV ${entry.horizontal_fov_degrees.toFixed(1)}° · roof width ${(entry.roof_bbox_width_fraction*100).toFixed(1)}% · visible ${(entry.visible_fraction*100).toFixed(1)}% · occluded ${(entry.occluded_fraction*100).toFixed(1)}% · truncated ${entry.truncated} · JPEG q${entry.photometric_profile.jpeg_quality}`; body.append(stats);
 const switcher=document.createElement("div"); switcher.className="switcher";
 for(const [label,source] of [["RGB",entry.rgb],["roof mask",entry.roof_mask],["parts",entry.part_mask],["face IDs",entry.face_ids]]){const button=document.createElement("button");button.type="button";button.textContent=label;button.onclick=()=>{image.src=source};switcher.append(button)} body.append(switcher);
 card.append(body); grid.append(card);
}
</script>
</body>
</html>
"#;

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    fn metadata(sample_key: &str, domain: &str) -> PreviewMetadata {
        PreviewMetadata {
            sample_key: sample_key.to_owned(),
            sequence_id: "seq-test".to_owned(),
            frame_index: 0,
            target_kind: synth_data::TargetKind::Target,
            ordinary_roof_family: None,
            roof_morphology: synth_data::RoofMorphology::BalancedClassic,
            day_phase: "Day".to_owned(),
            domain: domain.to_owned(),
            weather: "Clear".to_owned(),
            signage: vec!["RemovedGhost".to_owned()],
            building_extensions: vec!["DiningWing/Flat".to_owned()],
            roof_material: "original_red".to_owned(),
            wall_material: "warm_brick".to_owned(),
            ground_material: "clean_asphalt".to_owned(),
            camera_path: "Orbit".to_owned(),
            zoom_behavior: "Fixed".to_owned(),
            framing_intent: "Centered".to_owned(),
            apparent_scale: "Normal".to_owned(),
            environment_asset_id: "sky".to_owned(),
            horizontal_fov_degrees: 56.0,
            roof_bbox_width_fraction: 0.62,
            occluder_kinds: vec!["vehicle".to_owned()],
            background_building_count: 3,
            vegetation_count: 4,
            source_asset_ids: vec!["sky:file".to_owned()],
            photometric_profile: crate::photometric::sample_photometric_profile(
                7,
                synth_data::SampledEnvironment {
                    day_phase: synth_data::DayPhase::Day,
                    domain: synth_data::SceneDomain::Urban,
                    weather: synth_data::WeatherPreset::Clear,
                    shadow_softness: 0.2,
                    ground_wetness: 0.0,
                    visibility_km: 80.0,
                    color_temperature_k: 6_500.0,
                    camera_exposure_ev100: 13.0,
                    artificial_light_strength: 0.0,
                },
            )
            .unwrap(),
            visible_fraction: 0.8,
            occluded_fraction: 0.2,
            truncated: false,
        }
    }

    #[test]
    fn retains_only_frames_that_add_coverage() {
        let bytes = [1_u8, 2, 3];
        let images = PreviewImages {
            rgb_jpeg: &bytes,
            roof_mask_png: &bytes,
            part_preview_png: &bytes,
            face_preview_png: &bytes,
        };
        let mut gallery = PreviewGallery::new(8);
        gallery.consider(metadata("a", "Urban"), images);
        gallery.consider(metadata("b", "Urban"), images);
        gallery.consider(metadata("c", "Remote"), images);
        assert_eq!(gallery.entries.len(), 2);
    }

    #[test]
    fn stratifies_useful_outdoor_occlusion_ranges() {
        let mut low = metadata("low", "Urban");
        low.occluded_fraction = 0.19;
        let mut medium = metadata("medium", "Urban");
        medium.occluded_fraction = 0.34;
        let mut high = metadata("high", "Urban");
        high.occluded_fraction = 0.58;
        assert!(low.coverage_keys().contains(&"occlusion:low".to_owned()));
        assert!(
            medium
                .coverage_keys()
                .contains(&"occlusion:medium".to_owned())
        );
        assert!(high.coverage_keys().contains(&"occlusion:high".to_owned()));
    }

    #[test]
    fn retains_all_frames_from_two_non_fixed_zoom_showcases() {
        let bytes = [1_u8, 2, 3];
        let images = PreviewImages {
            rgb_jpeg: &bytes,
            roof_mask_png: &bytes,
            part_preview_png: &bytes,
            face_preview_png: &bytes,
        };
        let mut gallery = PreviewGallery::new(16);
        for (sequence, zoom) in [("smooth-in", "SmoothIn"), ("smooth-out", "SmoothOut")] {
            for frame_index in 0..3 {
                let mut frame = metadata(&format!("{sequence}-{frame_index}"), "Urban");
                frame.sequence_id = sequence.to_owned();
                frame.frame_index = frame_index;
                frame.zoom_behavior = zoom.to_owned();
                gallery.consider(frame, images);
            }
        }
        assert_eq!(gallery.entries.len(), 6);
    }

    #[test]
    fn writes_static_gallery_and_manifest() {
        let bytes = [1_u8, 2, 3];
        let mut rgb_jpeg = Vec::new();
        JpegEncoder::new_with_quality(&mut rgb_jpeg, 92)
            .write_image(
                &[220, 30, 20, 30, 200, 40, 20, 40, 210, 210, 190, 30],
                2,
                2,
                image::ExtendedColorType::Rgb8,
            )
            .unwrap();
        let mut gallery = PreviewGallery::default();
        gallery.consider(
            metadata("sample", "Urban"),
            PreviewImages {
                rgb_jpeg: &rgb_jpeg,
                roof_mask_png: &bytes,
                part_preview_png: &bytes,
                face_preview_png: &bytes,
            },
        );
        let directory = tempdir().unwrap();
        let summary = gallery.write(directory.path()).unwrap();
        assert_eq!(summary.entries, 1);
        assert_eq!(
            summary.stratified_contact_sheets,
            vec![PathBuf::from("contact-sheet-target-day.jpg")]
        );
        assert!(directory.path().join(summary.index).is_file());
        assert!(directory.path().join(summary.manifest).is_file());
        let contact_sheet = image::open(directory.path().join(summary.contact_sheet)).unwrap();
        assert_eq!(
            (contact_sheet.width(), contact_sheet.height()),
            (1_970, 380)
        );
    }
}
