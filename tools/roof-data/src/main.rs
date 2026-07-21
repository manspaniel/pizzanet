//! Curates licensed real-image examples for roof recognition.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Cursor,
    path::{Path, PathBuf},
    process::{Command as ProcessCommand, ExitCode},
    sync::atomic::{AtomicUsize, Ordering},
    thread,
    time::Duration,
};

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use image::{DynamicImage, ImageEncoder, RgbImage, codecs::jpeg::JpegEncoder, imageops};
use rayon::prelude::*;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

const BBOX_URL: &str =
    "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv";
const METADATA_URL: &str = "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv";
const IMAGE_URL_PREFIX: &str = "https://open-images-dataset.s3.amazonaws.com/validation";
const OPEN_IMAGES_VERSION: &str = "v7-validation-pixels-v5-boxes";
const COMMONS_API: &str = "https://commons.wikimedia.org/w/api.php";
const COMMONS_START_CATEGORY: &str = "Category:Former Pizza Hut restaurants";

const INCLUDED_CLASSES: [(&str, &str); 4] = [
    ("/m/03jm5", "House"),
    ("/m/0cgh4", "Building"),
    ("/m/021sj1", "Office building"),
    ("/m/079cl", "Skyscraper"),
];

#[derive(Debug, Parser)]
#[command(name = "roof-data", about = "Source and inspect roof training data")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Select and download ordinary-building negatives from Open Images.
    ImportOpenImages(ImportArgs),
    /// Download recognisable former Pizza Hut buildings from Wikimedia Commons.
    ImportWikimediaPositives(WikimediaArgs),
    /// Render every Open Images candidate into indexed review pages.
    PrepareOpenImagesReview(OpenImagesDatasetArgs),
    /// Apply a complete, digest-bound visual review ledger to the manifest.
    ApplyOpenImagesReview(ApplyOpenImagesReviewArgs),
    /// Validate reviewed Open Images records, pixels, and split coverage.
    ValidateOpenImagesReview(OpenImagesDatasetArgs),
}

#[derive(Debug, Args)]
struct OpenImagesDatasetArgs {
    /// Existing Open Images dataset directory.
    #[arg(long, default_value = "datasets/open-images-negatives")]
    dataset: PathBuf,
}

#[derive(Debug, Args)]
struct ApplyOpenImagesReviewArgs {
    /// Existing Open Images dataset directory.
    #[arg(long, default_value = "datasets/open-images-negatives")]
    dataset: PathBuf,
    /// Human-authored review ledger, bound to the ordered candidate IDs.
    #[arg(
        long,
        default_value = "datasets/open-images-negatives/review-ledger.json"
    )]
    ledger: PathBuf,
}

#[derive(Debug, Args)]
struct ImportArgs {
    /// New or existing output directory. Existing verified images are reused.
    #[arg(long, default_value = "datasets/open-images-negatives")]
    output: PathBuf,
    /// Maximum number of selected images.
    #[arg(long, default_value_t = 1_500)]
    limit: usize,
    /// Deterministic selection seed.
    #[arg(long, default_value_t = 0x5049_5a5a_4148_5554)]
    seed: u64,
    /// Parallel image downloads.
    #[arg(long, default_value_t = 8)]
    jobs: usize,
    /// Skip pixels and only write the proposed manifest.
    #[arg(long)]
    manifest_only: bool,
}

#[derive(Debug, Args)]
struct WikimediaArgs {
    /// New or existing output directory. Existing verified images are reused.
    #[arg(long, default_value = "datasets/wikimedia-positives")]
    output: PathBuf,
    /// Maximum number of selected files after metadata filtering.
    #[arg(long, default_value_t = 250)]
    limit: usize,
    /// Number of category levels followed from Former Pizza Hut restaurants.
    #[arg(long, default_value_t = 2)]
    category_depth: usize,
    /// Parallel image downloads.
    #[arg(long, default_value_t = 8)]
    jobs: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct BboxRow {
    #[serde(rename = "ImageID")]
    image_id: String,
    #[serde(rename = "LabelName")]
    label_name: String,
    #[serde(rename = "XMin")]
    x_min: f32,
    #[serde(rename = "XMax")]
    x_max: f32,
    #[serde(rename = "YMin")]
    y_min: f32,
    #[serde(rename = "YMax")]
    y_max: f32,
    #[serde(rename = "IsDepiction")]
    is_depiction: u8,
    #[serde(rename = "IsInside")]
    is_inside: u8,
}

#[derive(Clone, Debug, Deserialize)]
struct MetadataRow {
    #[serde(rename = "ImageID")]
    image_id: String,
    #[serde(rename = "OriginalURL")]
    original_url: String,
    #[serde(rename = "OriginalLandingURL")]
    original_landing_url: String,
    #[serde(rename = "License")]
    license: String,
    #[serde(rename = "AuthorProfileURL")]
    author_profile_url: String,
    #[serde(rename = "Author")]
    author: String,
    #[serde(rename = "Title")]
    title: String,
    #[serde(rename = "Rotation")]
    rotation: Option<f32>,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
struct BoxLabel {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
}

impl BoxLabel {
    fn area(self) -> f32 {
        (self.max_x - self.min_x).max(0.0) * (self.max_y - self.min_y).max(0.0)
    }
}

#[derive(Clone, Debug, Default)]
struct Candidate {
    classes: BTreeSet<String>,
    boxes: Vec<BoxLabel>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct NegativeRecord {
    image_id: String,
    split: String,
    relative_path: String,
    width: u32,
    height: u32,
    sha256: String,
    classes: Vec<String>,
    source_building_boxes: Vec<BoxLabel>,
    source_crop: BoxLabel,
    source_url: String,
    landing_page: String,
    license: String,
    author_profile: String,
    author: String,
    title: String,
    review_status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    review_reason: Option<String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct DatasetManifest {
    schema_version: String,
    dataset_id: String,
    source_version: String,
    source_bbox_url: String,
    source_metadata_url: String,
    selection_seed: u64,
    requested_limit: usize,
    records: Vec<NegativeRecord>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct OpenImagesReviewLedger {
    schema_version: String,
    dataset_id: String,
    ordered_image_ids_sha256: String,
    reviewed_record_count: usize,
    reviewer: String,
    reviewed_at: String,
    rejections: Vec<OpenImagesRejection>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct OpenImagesRejection {
    image_id: String,
    reason: String,
}

#[derive(Clone, Debug, Serialize)]
struct OpenImagesReviewSummary {
    schema_version: String,
    dataset_id: String,
    ordered_image_ids_sha256: String,
    reviewed_record_count: usize,
    accepted_by_split: BTreeMap<String, usize>,
    rejected_by_split: BTreeMap<String, usize>,
    rejected_by_reason: BTreeMap<String, usize>,
}

#[derive(Clone, Debug, Serialize)]
struct ReviewPageEntry {
    manifest_index: usize,
    page: usize,
    slot: usize,
    image_id: String,
    split: String,
    review_status: String,
    review_reason: Option<String>,
    relative_path: String,
}

#[derive(Clone, Debug, Serialize)]
struct PositiveRecord {
    page_id: u64,
    title: String,
    split: String,
    relative_path: String,
    width: u32,
    height: u32,
    sha256: String,
    source_url: String,
    landing_page: String,
    license: String,
    license_url: String,
    artist: String,
    description: String,
    source_categories: String,
    review_status: String,
}

#[derive(Clone, Debug, Serialize)]
struct PositiveDatasetManifest {
    schema_version: String,
    dataset_id: String,
    source_api: String,
    start_category: String,
    category_depth: usize,
    records: Vec<PositiveRecord>,
}

#[derive(Clone, Debug)]
struct CommonsCandidate {
    page_id: u64,
    title: String,
    thumbnail_url: String,
    source_url: String,
    landing_page: String,
    license: String,
    license_url: String,
    artist: String,
    description: String,
    source_categories: String,
}

#[derive(Clone)]
struct SelectedCandidate {
    metadata: MetadataRow,
    candidate: Candidate,
    split: String,
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("roof-data: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<()> {
    match cli.command {
        Command::ImportOpenImages(args) => import_open_images(args),
        Command::ImportWikimediaPositives(args) => import_wikimedia_positives(args),
        Command::PrepareOpenImagesReview(args) => prepare_open_images_review(&args.dataset),
        Command::ApplyOpenImagesReview(args) => {
            apply_open_images_review(&args.dataset, &args.ledger)
        }
        Command::ValidateOpenImagesReview(args) => {
            validate_open_images_review(&args.dataset).map(|_| ())
        }
    }
}

fn import_wikimedia_positives(args: WikimediaArgs) -> Result<()> {
    if args.limit == 0 || args.jobs == 0 {
        bail!("--limit and --jobs must be greater than zero");
    }
    fs::create_dir_all(args.output.join("images"))?;
    let client = Client::builder()
        .user_agent("pizzahut-roof-data/0.1 (research dataset curator)")
        .build()?;
    eprintln!("Enumerating Wikimedia Commons categories...");
    let titles = commons_file_titles(&client, COMMONS_START_CATEGORY, args.category_depth)?;
    eprintln!("Found {} candidate files", titles.len());
    let mut candidates = commons_image_info(&client, &titles)?;
    candidates.retain(|candidate| !rejected_positive_title(&candidate.title));
    candidates.sort_by_key(|candidate| stable_rank(&candidate.title, 0x0043_4f4d_4d4f_4e53));
    candidates.truncate(args.limit);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.jobs)
        .build()?;
    let completed = AtomicUsize::new(0);
    let total = candidates.len();
    let records = pool.install(|| {
        candidates
            .par_iter()
            .filter_map(
                |candidate| match materialize_positive(&client, &args.output, candidate) {
                    Ok(record) => {
                        let count = completed.fetch_add(1, Ordering::Relaxed) + 1;
                        if count.is_multiple_of(25) || count == total {
                            eprintln!("Prepared {count} images");
                        }
                        Some(record)
                    }
                    Err(error) => {
                        eprintln!("Skipping {}: {error:#}", candidate.title);
                        None
                    }
                },
            )
            .collect::<Vec<_>>()
    });
    let manifest = PositiveDatasetManifest {
        schema_version: "roof-positive-dataset/v1".to_owned(),
        dataset_id: "wikimedia-former-pizza-hut-positive".to_owned(),
        source_api: COMMONS_API.to_owned(),
        start_category: COMMONS_START_CATEGORY.to_owned(),
        category_depth: args.category_depth,
        records,
    };
    fs::write(
        args.output.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    write_positive_contact_sheet(&args.output, &manifest.records)?;
    write_positive_readme(&args.output, &manifest)?;
    println!("{}", args.output.join("manifest.json").display());
    Ok(())
}

fn commons_file_titles(client: &Client, start: &str, max_depth: usize) -> Result<Vec<String>> {
    let mut pending = vec![(start.to_owned(), 0usize)];
    let mut visited = BTreeSet::new();
    let mut files = BTreeSet::new();
    while let Some((category, depth)) = pending.pop() {
        if !visited.insert(category.clone()) {
            continue;
        }
        let mut continuation = None::<String>;
        loop {
            let mut request = client.get(COMMONS_API).query(&[
                ("action", "query"),
                ("list", "categorymembers"),
                ("cmtitle", category.as_str()),
                ("cmtype", "subcat|file"),
                ("cmlimit", "500"),
                ("format", "json"),
            ]);
            if let Some(value) = &continuation {
                request = request.query(&[("cmcontinue", value)]);
            }
            let json = response_json(request, "enumerate Commons category")?;
            for member in json["query"]["categorymembers"]
                .as_array()
                .into_iter()
                .flatten()
            {
                let namespace = member["ns"].as_u64().unwrap_or_default();
                let Some(title) = member["title"].as_str() else {
                    continue;
                };
                match namespace {
                    6 => {
                        files.insert(title.to_owned());
                    }
                    14 if depth < max_depth => pending.push((title.to_owned(), depth + 1)),
                    _ => {}
                }
            }
            continuation = json["continue"]["cmcontinue"].as_str().map(str::to_owned);
            if continuation.is_none() {
                break;
            }
        }
    }
    Ok(files.into_iter().collect())
}

fn commons_image_info(client: &Client, titles: &[String]) -> Result<Vec<CommonsCandidate>> {
    let mut candidates = Vec::new();
    for chunk in titles.chunks(50) {
        let joined = chunk.join("|");
        let json = response_json(
            client.get(COMMONS_API).query(&[
                ("action", "query"),
                ("prop", "imageinfo"),
                ("titles", joined.as_str()),
                ("iiprop", "url|mime|extmetadata"),
                // Commons maps this request to a standard cached 960 px rendition.
                ("iiurlwidth", "640"),
                ("format", "json"),
                ("formatversion", "2"),
            ]),
            "fetch Commons image metadata",
        )?;
        for page in json["query"]["pages"].as_array().into_iter().flatten() {
            let Some(info) = page["imageinfo"].as_array().and_then(|items| items.first()) else {
                continue;
            };
            let mime = info["mime"].as_str().unwrap_or_default();
            if !mime.starts_with("image/") {
                continue;
            }
            let metadata = &info["extmetadata"];
            let field = |name: &str| {
                metadata[name]["value"]
                    .as_str()
                    .unwrap_or_default()
                    .to_owned()
            };
            let license = field("LicenseShortName");
            let license_url = field("LicenseUrl");
            if license.is_empty()
                || (!license.to_ascii_lowercase().starts_with("cc")
                    && !license.to_ascii_lowercase().contains("public domain"))
            {
                continue;
            }
            let Some(thumbnail_url) = info["thumburl"].as_str() else {
                continue;
            };
            candidates.push(CommonsCandidate {
                page_id: page["pageid"].as_u64().unwrap_or_default(),
                title: page["title"].as_str().unwrap_or_default().to_owned(),
                thumbnail_url: thumbnail_url.to_owned(),
                source_url: info["url"].as_str().unwrap_or_default().to_owned(),
                landing_page: info["descriptionurl"]
                    .as_str()
                    .unwrap_or_default()
                    .to_owned(),
                license,
                license_url,
                artist: field("Artist"),
                description: field("ImageDescription"),
                source_categories: field("Categories"),
            });
        }
    }
    Ok(candidates)
}

fn response_json(request: reqwest::blocking::RequestBuilder, context: &str) -> Result<Value> {
    let text = response_with_retry(request, context)?.text()?;
    serde_json::from_str(&text).with_context(|| context.to_owned())
}

fn response_with_retry(
    request: reqwest::blocking::RequestBuilder,
    context: &str,
) -> Result<reqwest::blocking::Response> {
    for attempt in 0..6 {
        let response = request
            .try_clone()
            .context("request cannot be retried")?
            .send()
            .with_context(|| context.to_owned())?;
        let status = response.status();
        if status.as_u16() != 429 && !status.is_server_error() {
            return response
                .error_for_status()
                .with_context(|| context.to_owned());
        }
        if attempt == 5 {
            return response
                .error_for_status()
                .with_context(|| context.to_owned());
        }
        let delay = 2u64.pow(attempt + 1).min(20);
        eprintln!("{context}: server returned {status}; retrying in {delay}s");
        thread::sleep(Duration::from_secs(delay));
    }
    unreachable!("retry loop either returns a response or error")
}

fn rejected_positive_title(title: &str) -> bool {
    let title = title.to_ascii_lowercase();
    // These category members were visually reviewed and do not show enough of the
    // characteristic two-tier roof to supervise this detector. Excluding an image
    // here does not turn the former Pizza Hut into a negative example.
    [
        "foundation stone",
        "interior",
        "food",
        "pizza supreme",
        "aaron's store",
        "two rivers, wi",
        "fenced-off shell",
        "a story of yagoona",
        "briggate, leeds (4th",
        "סנטר פארק",
        "temporary topshop",
        "former pizza hut - panoramio",
        "church of our savior",
        "former pizza hut, monticello",
        "former pizza hut, dawson",
        "bud hut",
        "china hot buffet",
        "vj's",
        "brunswick square",
        "kfc hohe straße",
        "formerpizzahut, tequila's",
    ]
    .iter()
    .any(|needle| title.contains(needle))
}

fn materialize_positive(
    _client: &Client,
    output: &Path,
    candidate: &CommonsCandidate,
) -> Result<PositiveRecord> {
    let relative_path = format!("images/{}.jpg", candidate.page_id);
    let path = output.join(&relative_path);
    let (width, height, sha256) = if path.exists() {
        inspect_existing(&path)?
    } else {
        let context = format!("request {}", candidate.title);
        let bytes = download_commons_image(&candidate.thumbnail_url, &context)?;
        let rgb = image::load_from_memory(&bytes)
            .with_context(|| format!("decode {}", candidate.title))?
            .into_rgb8();
        let rgb = if rgb.width() > 1600 || rgb.height() > 1600 {
            imageops::thumbnail(&rgb, 1600, 1600)
        } else {
            rgb
        };
        let encoded = encode_jpeg(&rgb, 90)?;
        fs::write(&path, &encoded)?;
        // Keep this curator polite to Commons even when the caller asks for one worker.
        thread::sleep(Duration::from_millis(1_100));
        (rgb.width(), rgb.height(), hex_sha256(&encoded))
    };
    let split_bucket = stable_rank(&candidate.title, 0x0053_504c_4954) % 100;
    let split = match split_bucket {
        0..=79 => "train",
        80..=89 => "validation",
        _ => "test",
    };
    Ok(PositiveRecord {
        page_id: candidate.page_id,
        title: candidate.title.clone(),
        split: split.to_owned(),
        relative_path,
        width,
        height,
        sha256,
        source_url: candidate.source_url.clone(),
        landing_page: candidate.landing_page.clone(),
        license: candidate.license.clone(),
        license_url: candidate.license_url.clone(),
        artist: candidate.artist.clone(),
        description: candidate.description.clone(),
        source_categories: candidate.source_categories.clone(),
        review_status: "category_screened".to_owned(),
    })
}

fn download_commons_image(url: &str, context: &str) -> Result<Vec<u8>> {
    // curl negotiates Wikimedia's preferred HTTP/2 path reliably on macOS; metadata
    // requests remain native reqwest calls so the curation logic stays in Rust.
    let output = ProcessCommand::new("curl")
        .args([
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--retry",
            "5",
            "--retry-all-errors",
            "--retry-delay",
            "2",
            url,
        ])
        .output()
        .with_context(|| format!("{context}: launch curl"))?;
    if !output.status.success() {
        bail!(
            "{context}: curl failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(output.stdout)
}

fn import_open_images(args: ImportArgs) -> Result<()> {
    if args.limit == 0 {
        bail!("--limit must be greater than zero");
    }
    if args.jobs == 0 {
        bail!("--jobs must be greater than zero");
    }

    fs::create_dir_all(args.output.join("building-crops"))
        .with_context(|| format!("create {}", args.output.display()))?;
    let client = Client::builder()
        .user_agent("pizzahut-roof-data/0.1")
        .build()?;

    eprintln!("Fetching Open Images building annotations...");
    let bbox_csv = fetch_text(&client, BBOX_URL)?;
    let candidates = parse_candidates(&bbox_csv)?;
    eprintln!("Found {} candidate building images", candidates.len());

    eprintln!("Fetching per-image provenance and licence metadata...");
    let metadata_csv = fetch_text(&client, METADATA_URL)?;
    let selected = select_candidates(&metadata_csv, &candidates, args.seed, args.limit)?;
    eprintln!("Selected {} licence-filtered images", selected.len());

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(args.jobs)
        .build()?;
    let completed = AtomicUsize::new(0);
    let total = selected.len();
    let output = &args.output;
    let manifest_only = args.manifest_only;
    let records = pool.install(|| {
        selected
            .par_iter()
            .map(|selected| {
                let record = materialize_record(&client, output, selected, manifest_only)?;
                let count = completed.fetch_add(1, Ordering::Relaxed) + 1;
                if count.is_multiple_of(100) || count == total {
                    eprintln!("Prepared {count} images");
                }
                Ok(record)
            })
            .collect::<Result<Vec<_>>>()
    })?;

    let manifest = DatasetManifest {
        schema_version: "roof-negative-dataset/v1".to_owned(),
        dataset_id: "open-images-buildings-negative".to_owned(),
        source_version: OPEN_IMAGES_VERSION.to_owned(),
        source_bbox_url: BBOX_URL.to_owned(),
        source_metadata_url: METADATA_URL.to_owned(),
        selection_seed: args.seed,
        requested_limit: args.limit,
        records,
    };
    let manifest_path = args.output.join("manifest.json");
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)
        .with_context(|| format!("write {}", manifest_path.display()))?;

    if !manifest_only {
        write_contact_sheet(&args.output, &manifest.records)?;
    }
    write_readme(&args.output, &manifest)?;
    println!("{}", manifest_path.display());
    Ok(())
}

fn fetch_text(client: &Client, url: &str) -> Result<String> {
    client
        .get(url)
        .send()
        .with_context(|| format!("request {url}"))?
        .error_for_status()
        .with_context(|| format!("download {url}"))?
        .text()
        .with_context(|| format!("decode {url}"))
}

fn parse_candidates(csv_bytes: &str) -> Result<BTreeMap<String, Candidate>> {
    let class_names: BTreeMap<&str, &str> = INCLUDED_CLASSES.into_iter().collect();
    let mut candidates = BTreeMap::<String, Candidate>::new();
    let mut reader = csv::Reader::from_reader(csv_bytes.as_bytes());
    for row in reader.deserialize::<BboxRow>() {
        let row = row?;
        let Some(class_name) = class_names.get(row.label_name.as_str()) else {
            continue;
        };
        if row.is_depiction != 0 || row.is_inside != 0 {
            continue;
        }
        let bbox = BoxLabel {
            min_x: row.x_min,
            min_y: row.y_min,
            max_x: row.x_max,
            max_y: row.y_max,
        };
        if bbox.area() < 0.025 {
            continue;
        }
        let candidate = candidates.entry(row.image_id).or_default();
        candidate.classes.insert((*class_name).to_owned());
        candidate.boxes.push(bbox);
    }
    Ok(candidates)
}

fn select_candidates(
    metadata_csv: &str,
    candidates: &BTreeMap<String, Candidate>,
    seed: u64,
    limit: usize,
) -> Result<Vec<SelectedCandidate>> {
    let mut eligible = Vec::new();
    let mut reader = csv::Reader::from_reader(metadata_csv.as_bytes());
    for row in reader.deserialize::<MetadataRow>() {
        let row = row?;
        let Some(candidate) = candidates.get(&row.image_id) else {
            continue;
        };
        if !row
            .license
            .starts_with("https://creativecommons.org/licenses/by/")
        {
            continue;
        }
        let text = format!("{} {}", row.title, row.original_landing_url).to_lowercase();
        if rejected_title(&text) {
            continue;
        }
        let split_bucket = stable_rank(&row.image_id, seed ^ 0x0053_504c_4954) % 100;
        let split = match split_bucket {
            0..=79 => "train",
            80..=89 => "validation",
            _ => "test",
        };
        eligible.push((
            stable_rank(&row.image_id, seed),
            SelectedCandidate {
                metadata: row,
                candidate: candidate.clone(),
                split: split.to_owned(),
            },
        ));
    }
    eligible.sort_by_key(|(rank, _)| *rank);
    Ok(eligible
        .into_iter()
        .take(limit)
        .map(|(_, selected)| selected)
        .collect())
}

fn rejected_title(text: &str) -> bool {
    const REJECTED: [&str; 18] = [
        "pizza hut",
        "pizzahut",
        "interior",
        "lobby",
        "bedroom",
        "bathroom",
        "kitchen",
        "auditorium",
        "operating room",
        "waiting room",
        "conference room",
        "hotel room",
        "dining room",
        "living room",
        "lego",
        "dollhouse",
        "miniature",
        "scale model",
    ];
    REJECTED.iter().any(|needle| text.contains(needle))
}

fn stable_rank(image_id: &str, seed: u64) -> u64 {
    let mut hasher = Sha256::new();
    hasher.update(seed.to_le_bytes());
    hasher.update(image_id.as_bytes());
    let digest = hasher.finalize();
    u64::from_le_bytes(
        digest[..8]
            .try_into()
            .expect("SHA-256 prefix is eight bytes"),
    )
}

fn materialize_record(
    client: &Client,
    output: &Path,
    selected: &SelectedCandidate,
    manifest_only: bool,
) -> Result<NegativeRecord> {
    let image_id = &selected.metadata.image_id;
    let relative_path = format!("building-crops/{image_id}.jpg");
    let image_path = output.join(&relative_path);
    let crop = largest_expanded_crop(&selected.candidate.boxes);
    let (width, height, sha256) = if manifest_only {
        (0, 0, String::new())
    } else if image_path.exists() {
        inspect_existing(&image_path)?
    } else {
        download_normalized_image(
            client,
            image_id,
            selected.metadata.rotation,
            crop,
            &image_path,
        )?
    };

    Ok(NegativeRecord {
        image_id: image_id.clone(),
        split: selected.split.clone(),
        relative_path,
        width,
        height,
        sha256,
        classes: selected.candidate.classes.iter().cloned().collect(),
        source_building_boxes: selected.candidate.boxes.clone(),
        source_crop: crop,
        source_url: selected.metadata.original_url.clone(),
        landing_page: selected.metadata.original_landing_url.clone(),
        license: selected.metadata.license.clone(),
        author_profile: selected.metadata.author_profile_url.clone(),
        author: selected.metadata.author.clone(),
        title: selected.metadata.title.clone(),
        review_status: "metadata_screened".to_owned(),
        review_reason: None,
    })
}

fn download_normalized_image(
    client: &Client,
    image_id: &str,
    rotation: Option<f32>,
    crop: BoxLabel,
    path: &Path,
) -> Result<(u32, u32, String)> {
    let url = format!("{IMAGE_URL_PREFIX}/{image_id}.jpg");
    let bytes = client
        .get(&url)
        .send()
        .with_context(|| format!("request image {image_id}"))?
        .error_for_status()
        .with_context(|| format!("download image {image_id}"))?
        .bytes()?;
    let mut image =
        image::load_from_memory(&bytes).with_context(|| format!("decode image {image_id}"))?;
    image = apply_rotation(image, rotation);
    let crop_x = (crop.min_x * image.width() as f32).floor() as u32;
    let crop_y = (crop.min_y * image.height() as f32).floor() as u32;
    let crop_max_x = (crop.max_x * image.width() as f32).ceil() as u32;
    let crop_max_y = (crop.max_y * image.height() as f32).ceil() as u32;
    let crop_width = crop_max_x.saturating_sub(crop_x).max(1);
    let crop_height = crop_max_y.saturating_sub(crop_y).max(1);
    let rgb = image
        .crop_imm(crop_x, crop_y, crop_width, crop_height)
        .into_rgb8();
    let (width, height) = rgb.dimensions();
    let encoded = encode_jpeg(&rgb, 90)?;
    fs::write(path, &encoded).with_context(|| format!("write {}", path.display()))?;
    Ok((width, height, hex_sha256(&encoded)))
}

fn largest_expanded_crop(boxes: &[BoxLabel]) -> BoxLabel {
    let largest = boxes
        .iter()
        .copied()
        .max_by(|left, right| left.area().total_cmp(&right.area()))
        .expect("selected candidates always contain a box");
    let margin_x = (largest.max_x - largest.min_x) * 0.12;
    let margin_y = (largest.max_y - largest.min_y) * 0.12;
    BoxLabel {
        min_x: (largest.min_x - margin_x).max(0.0),
        min_y: (largest.min_y - margin_y).max(0.0),
        max_x: (largest.max_x + margin_x).min(1.0),
        max_y: (largest.max_y + margin_y).min(1.0),
    }
}

fn apply_rotation(image: DynamicImage, rotation: Option<f32>) -> DynamicImage {
    match rotation
        .map(|value| value.round() as i32)
        .unwrap_or(0)
        .rem_euclid(360)
    {
        90 => image.rotate90(),
        180 => image.rotate180(),
        270 => image.rotate270(),
        _ => image,
    }
}

fn inspect_existing(path: &Path) -> Result<(u32, u32, String)> {
    let bytes = fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let image =
        image::load_from_memory(&bytes).with_context(|| format!("decode {}", path.display()))?;
    Ok((image.width(), image.height(), hex_sha256(&bytes)))
}

fn encode_jpeg(image: &RgbImage, quality: u8) -> Result<Vec<u8>> {
    let mut output = Cursor::new(Vec::new());
    JpegEncoder::new_with_quality(&mut output, quality).write_image(
        image.as_raw(),
        image.width(),
        image.height(),
        image::ExtendedColorType::Rgb8,
    )?;
    Ok(output.into_inner())
}

fn hex_sha256(bytes: &[u8]) -> String {
    format!("sha256:{:x}", Sha256::digest(bytes))
}

fn write_contact_sheet(output: &Path, records: &[NegativeRecord]) -> Result<()> {
    const COLS: u32 = 8;
    const ROWS: u32 = 8;
    const CELL_W: u32 = 180;
    const CELL_H: u32 = 120;
    let mut sheet = RgbImage::new(COLS * CELL_W, ROWS * CELL_H);
    let sample_count = (COLS * ROWS) as usize;
    for (slot, record) in records.iter().take(sample_count).enumerate() {
        let image = image::open(output.join(&record.relative_path))?.into_rgb8();
        let thumb = imageops::thumbnail(&image, CELL_W, CELL_H);
        let x = (slot as u32 % COLS) * CELL_W + (CELL_W - thumb.width()) / 2;
        let y = (slot as u32 / COLS) * CELL_H + (CELL_H - thumb.height()) / 2;
        imageops::replace(&mut sheet, &thumb, i64::from(x), i64::from(y));
    }
    sheet.save(output.join("contact-sheet.jpg"))?;
    Ok(())
}

fn write_readme(output: &Path, manifest: &DatasetManifest) -> Result<()> {
    let mut splits = BTreeMap::<&str, usize>::new();
    let mut accepted_splits = BTreeMap::<&str, usize>::new();
    let mut rejected = 0usize;
    for record in &manifest.records {
        *splits.entry(&record.split).or_default() += 1;
        match record.review_status.as_str() {
            "visually_verified" => *accepted_splits.entry(&record.split).or_default() += 1,
            "rejected" => rejected += 1,
            _ => {}
        }
    }
    let accepted = accepted_splits.values().sum::<usize>();
    let review_state = if accepted + rejected == manifest.records.len() {
        format!(
            "complete visual review: {accepted} accepted exterior buildings; {rejected} rejected"
        )
    } else {
        "metadata screened; run `roof-data prepare-open-images-review` and apply a complete review ledger before training".to_owned()
    };
    let text = format!(
        "# Open Images ordinary-building negatives\n\nGenerated by `roof-data import-open-images` and curated with the digest-bound `review-ledger.json`. Pixels are excluded from Git.\n\n- Source candidates: {} (train {}, validation {}, test {})\n- Visually verified exteriors: {} (train {}, validation {}, test {})\n- Rejected: {}\n- Selection seed: {}\n- Review state: {}.\n- Audit artifacts: `review-ledger.json`, `review-summary.json`, `review-index.json`, and paginated `review-pages/`.\n- Contact sheets: `contact-sheet.jpg` is an accepted sample; all accepted/rejected pages are under `accepted-contact-sheets/` and `rejected-contact-sheets/`.\n- Licensing: retain and verify the per-image attribution and licence fields in `manifest.json`.\n",
        manifest.records.len(),
        splits.get("train").copied().unwrap_or(0),
        splits.get("validation").copied().unwrap_or(0),
        splits.get("test").copied().unwrap_or(0),
        accepted,
        accepted_splits.get("train").copied().unwrap_or(0),
        accepted_splits.get("validation").copied().unwrap_or(0),
        accepted_splits.get("test").copied().unwrap_or(0),
        rejected,
        manifest.selection_seed,
        review_state,
    );
    fs::write(output.join("README.md"), text)?;
    Ok(())
}

const REVIEW_REASONS: [&str; 6] = [
    "non_building",
    "interior",
    "building_too_small_or_obscured",
    "non_ordinary_structure",
    "pizza_hut_roof",
    "ambiguous_or_unusable",
];

fn read_negative_manifest(dataset: &Path) -> Result<DatasetManifest> {
    let path = dataset.join("manifest.json");
    let bytes = fs::read(&path).with_context(|| format!("read {}", path.display()))?;
    let manifest: DatasetManifest =
        serde_json::from_slice(&bytes).with_context(|| format!("parse {}", path.display()))?;
    if manifest.schema_version != "roof-negative-dataset/v1" {
        bail!(
            "unsupported negative manifest schema {:?}",
            manifest.schema_version
        );
    }
    if manifest.dataset_id != "open-images-buildings-negative" {
        bail!("unexpected dataset id {:?}", manifest.dataset_id);
    }
    Ok(manifest)
}

fn ordered_image_ids_sha256(records: &[NegativeRecord]) -> String {
    let mut hasher = Sha256::new();
    for (index, record) in records.iter().enumerate() {
        hasher.update(index.to_le_bytes());
        hasher.update(record.image_id.as_bytes());
        hasher.update([0]);
    }
    format!("sha256:{:x}", hasher.finalize())
}

fn prepare_open_images_review(dataset: &Path) -> Result<()> {
    let manifest = read_negative_manifest(dataset)?;
    validate_manifest_pixels(dataset, &manifest)?;
    let digest = ordered_image_ids_sha256(&manifest.records);
    write_review_pages(dataset, "review-pages", &manifest.records)?;
    write_review_index(dataset, &manifest.records)?;
    let template = OpenImagesReviewLedger {
        schema_version: "roof-negative-visual-review/v1".to_owned(),
        dataset_id: manifest.dataset_id.clone(),
        ordered_image_ids_sha256: digest.clone(),
        reviewed_record_count: manifest.records.len(),
        reviewer: "replace-with-reviewer".to_owned(),
        reviewed_at: "replace-with-ISO-8601-date".to_owned(),
        rejections: Vec::new(),
    };
    let path = dataset.join("review-ledger.template.json");
    fs::write(&path, serde_json::to_vec_pretty(&template)?)
        .with_context(|| format!("write {}", path.display()))?;
    println!(
        "prepared {} candidates in review-pages; ordered ID digest {}",
        manifest.records.len(),
        digest
    );
    Ok(())
}

fn apply_open_images_review(dataset: &Path, ledger_path: &Path) -> Result<()> {
    let mut manifest = read_negative_manifest(dataset)?;
    validate_manifest_pixels(dataset, &manifest)?;
    let ledger_bytes =
        fs::read(ledger_path).with_context(|| format!("read {}", ledger_path.display()))?;
    let ledger: OpenImagesReviewLedger = serde_json::from_slice(&ledger_bytes)
        .with_context(|| format!("parse {}", ledger_path.display()))?;
    validate_review_ledger(&manifest, &ledger)?;

    let rejections: BTreeMap<&str, &str> = ledger
        .rejections
        .iter()
        .map(|entry| (entry.image_id.as_str(), entry.reason.as_str()))
        .collect();
    for record in &mut manifest.records {
        if let Some(reason) = rejections.get(record.image_id.as_str()) {
            record.review_status = "rejected".to_owned();
            record.review_reason = Some((*reason).to_owned());
        } else {
            record.review_status = "visually_verified".to_owned();
            record.review_reason = None;
        }
    }

    let manifest_path = dataset.join("manifest.json");
    fs::write(&manifest_path, serde_json::to_vec_pretty(&manifest)?)
        .with_context(|| format!("write {}", manifest_path.display()))?;
    let canonical_ledger_path = dataset.join("review-ledger.json");
    fs::write(&canonical_ledger_path, serde_json::to_vec_pretty(&ledger)?)
        .with_context(|| format!("write {}", canonical_ledger_path.display()))?;

    let summary = summarize_review(&manifest)?;
    fs::write(
        dataset.join("review-summary.json"),
        serde_json::to_vec_pretty(&summary)?,
    )?;
    write_review_pages(dataset, "review-pages", &manifest.records)?;
    write_review_index(dataset, &manifest.records)?;
    let accepted = manifest
        .records
        .iter()
        .filter(|record| record.review_status == "visually_verified")
        .cloned()
        .collect::<Vec<_>>();
    let rejected = manifest
        .records
        .iter()
        .filter(|record| record.review_status == "rejected")
        .cloned()
        .collect::<Vec<_>>();
    write_review_pages(dataset, "accepted-contact-sheets", &accepted)?;
    write_review_pages(dataset, "rejected-contact-sheets", &rejected)?;
    write_contact_sheet(dataset, &accepted)?;
    write_readme(dataset, &manifest)?;
    print_review_summary(&summary);
    Ok(())
}

fn validate_review_ledger(
    manifest: &DatasetManifest,
    ledger: &OpenImagesReviewLedger,
) -> Result<()> {
    if ledger.schema_version != "roof-negative-visual-review/v1" {
        bail!(
            "unsupported review ledger schema {:?}",
            ledger.schema_version
        );
    }
    if ledger.dataset_id != manifest.dataset_id {
        bail!("review ledger dataset id does not match manifest");
    }
    if ledger.reviewed_record_count != manifest.records.len() {
        bail!(
            "review ledger covers {} records but manifest contains {}",
            ledger.reviewed_record_count,
            manifest.records.len()
        );
    }
    let digest = ordered_image_ids_sha256(&manifest.records);
    if ledger.ordered_image_ids_sha256 != digest {
        bail!(
            "review ledger candidate digest mismatch: expected {digest}, got {}",
            ledger.ordered_image_ids_sha256
        );
    }
    if ledger.reviewer.trim().is_empty() || ledger.reviewer.starts_with("replace-") {
        bail!("review ledger must name the reviewer");
    }
    if ledger.reviewed_at.trim().is_empty() || ledger.reviewed_at.starts_with("replace-") {
        bail!("review ledger must record the review date");
    }
    let ids = manifest
        .records
        .iter()
        .map(|record| record.image_id.as_str())
        .collect::<BTreeSet<_>>();
    let mut rejected = BTreeSet::new();
    for entry in &ledger.rejections {
        if !ids.contains(entry.image_id.as_str()) {
            bail!("review rejects unknown image id {}", entry.image_id);
        }
        if !rejected.insert(entry.image_id.as_str()) {
            bail!("review rejects image {} more than once", entry.image_id);
        }
        if !REVIEW_REASONS.contains(&entry.reason.as_str()) {
            bail!(
                "review rejection {} has unsupported reason {:?}",
                entry.image_id,
                entry.reason
            );
        }
    }
    Ok(())
}

fn validate_open_images_review(dataset: &Path) -> Result<OpenImagesReviewSummary> {
    let manifest = read_negative_manifest(dataset)?;
    validate_manifest_pixels(dataset, &manifest)?;
    for record in &manifest.records {
        match record.review_status.as_str() {
            "visually_verified" if record.review_reason.is_none() => {}
            "rejected"
                if record
                    .review_reason
                    .as_deref()
                    .is_some_and(|reason| REVIEW_REASONS.contains(&reason)) => {}
            status => bail!(
                "record {} has incomplete or unsupported review status {:?}",
                record.image_id,
                status
            ),
        }
    }
    let ledger_path = dataset.join("review-ledger.json");
    let ledger: OpenImagesReviewLedger = serde_json::from_slice(
        &fs::read(&ledger_path).with_context(|| format!("read {}", ledger_path.display()))?,
    )?;
    validate_review_ledger(&manifest, &ledger)?;
    let summary = summarize_review(&manifest)?;
    for (split, minimum) in [("train", 100usize), ("validation", 20), ("test", 20)] {
        let count = summary.accepted_by_split.get(split).copied().unwrap_or(0);
        if count < minimum {
            bail!(
                "reviewed exterior {split} split has only {count} records; require at least {minimum}"
            );
        }
    }
    print_review_summary(&summary);
    Ok(summary)
}

fn validate_manifest_pixels(dataset: &Path, manifest: &DatasetManifest) -> Result<()> {
    let mut ids = BTreeSet::new();
    let mut paths = BTreeSet::new();
    for record in &manifest.records {
        if !ids.insert(record.image_id.as_str()) {
            bail!("duplicate Open Images id {}", record.image_id);
        }
        if !paths.insert(record.relative_path.as_str()) {
            bail!("duplicate Open Images path {}", record.relative_path);
        }
        if !matches!(record.split.as_str(), "train" | "validation" | "test") {
            bail!(
                "record {} has invalid split {:?}",
                record.image_id,
                record.split
            );
        }
        if !record
            .license
            .starts_with("https://creativecommons.org/licenses/by/")
            || record.source_url.trim().is_empty()
            || record.landing_page.trim().is_empty()
            || record.sha256.trim().is_empty()
        {
            bail!(
                "record {} has incomplete provenance, licence, or checksum metadata",
                record.image_id
            );
        }
    }
    manifest.records.par_iter().try_for_each(|record| {
        let path = dataset.join(&record.relative_path);
        let (width, height, sha256) = inspect_existing(&path)?;
        if (width, height) != (record.width, record.height) {
            bail!(
                "record {} dimension mismatch: manifest {}x{}, pixels {}x{}",
                record.image_id,
                record.width,
                record.height,
                width,
                height
            );
        }
        if sha256 != record.sha256 {
            bail!("record {} checksum mismatch", record.image_id);
        }
        Ok(())
    })?;
    Ok(())
}

fn summarize_review(manifest: &DatasetManifest) -> Result<OpenImagesReviewSummary> {
    let mut accepted_by_split = BTreeMap::new();
    let mut rejected_by_split = BTreeMap::new();
    let mut rejected_by_reason = BTreeMap::new();
    for record in &manifest.records {
        match record.review_status.as_str() {
            "visually_verified" => {
                *accepted_by_split.entry(record.split.clone()).or_default() += 1;
            }
            "rejected" => {
                *rejected_by_split.entry(record.split.clone()).or_default() += 1;
                let reason = record
                    .review_reason
                    .clone()
                    .context("rejected record has no reason")?;
                *rejected_by_reason.entry(reason).or_default() += 1;
            }
            status => bail!("cannot summarize incomplete review status {status:?}"),
        }
    }
    Ok(OpenImagesReviewSummary {
        schema_version: "roof-negative-review-summary/v1".to_owned(),
        dataset_id: manifest.dataset_id.clone(),
        ordered_image_ids_sha256: ordered_image_ids_sha256(&manifest.records),
        reviewed_record_count: manifest.records.len(),
        accepted_by_split,
        rejected_by_split,
        rejected_by_reason,
    })
}

fn print_review_summary(summary: &OpenImagesReviewSummary) {
    let accepted = summary.accepted_by_split.values().sum::<usize>();
    let rejected = summary.rejected_by_split.values().sum::<usize>();
    println!(
        "review valid: {} accepted, {} rejected; accepted train={} validation={} test={}",
        accepted,
        rejected,
        summary.accepted_by_split.get("train").copied().unwrap_or(0),
        summary
            .accepted_by_split
            .get("validation")
            .copied()
            .unwrap_or(0),
        summary.accepted_by_split.get("test").copied().unwrap_or(0),
    );
}

fn write_review_index(dataset: &Path, records: &[NegativeRecord]) -> Result<()> {
    const PAGE_SIZE: usize = 64;
    let entries = records
        .iter()
        .enumerate()
        .map(|(index, record)| ReviewPageEntry {
            manifest_index: index,
            page: index / PAGE_SIZE + 1,
            slot: index % PAGE_SIZE + 1,
            image_id: record.image_id.clone(),
            split: record.split.clone(),
            review_status: record.review_status.clone(),
            review_reason: record.review_reason.clone(),
            relative_path: record.relative_path.clone(),
        })
        .collect::<Vec<_>>();
    fs::write(
        dataset.join("review-index.json"),
        serde_json::to_vec_pretty(&entries)?,
    )?;
    Ok(())
}

fn write_review_pages(dataset: &Path, directory: &str, records: &[NegativeRecord]) -> Result<()> {
    const COLS: u32 = 8;
    const ROWS: u32 = 8;
    const CELL_W: u32 = 180;
    const CELL_H: u32 = 140;
    const IMAGE_H: u32 = 120;
    const PAGE_SIZE: usize = (COLS * ROWS) as usize;
    let output = dataset.join(directory);
    fs::create_dir_all(&output)?;
    records.par_chunks(PAGE_SIZE).enumerate().try_for_each(
        |(page_index, page_records)| -> Result<()> {
            let mut sheet =
                RgbImage::from_pixel(COLS * CELL_W, ROWS * CELL_H, image::Rgb([24, 24, 24]));
            for (slot, record) in page_records.iter().enumerate() {
                let image = image::open(dataset.join(&record.relative_path))?.into_rgb8();
                let thumb = imageops::thumbnail(&image, CELL_W - 4, IMAGE_H - 4);
                let cell_x = (slot as u32 % COLS) * CELL_W;
                let cell_y = (slot as u32 / COLS) * CELL_H;
                let x = cell_x + 2 + (CELL_W - 4 - thumb.width()) / 2;
                let y = cell_y + 2 + (IMAGE_H - 4 - thumb.height()) / 2;
                imageops::replace(&mut sheet, &thumb, i64::from(x), i64::from(y));
                let color = match record.review_status.as_str() {
                    "visually_verified" => image::Rgb([20, 220, 80]),
                    "rejected" => image::Rgb([245, 55, 55]),
                    _ => image::Rgb([245, 190, 30]),
                };
                draw_cell_border(&mut sheet, cell_x, cell_y, CELL_W, IMAGE_H, color);
                let split = match record.split.as_str() {
                    "train" => 'T',
                    "validation" => 'V',
                    "test" => 'E',
                    _ => '?',
                };
                let status = match record.review_status.as_str() {
                    "visually_verified" => 'A',
                    "rejected" => 'R',
                    _ => 'P',
                };
                let global_index = page_index * PAGE_SIZE + slot;
                let short_id = record
                    .image_id
                    .chars()
                    .take(8)
                    .collect::<String>()
                    .to_ascii_uppercase();
                let label = format!("#{global_index:04} {short_id} {split}{status}",);
                draw_text_5x7(
                    &mut sheet,
                    cell_x + 5,
                    cell_y + IMAGE_H + 6,
                    &label,
                    image::Rgb([245, 245, 245]),
                );
            }
            sheet.save(output.join(format!("page-{:03}.jpg", page_index + 1)))?;
            Ok(())
        },
    )?;
    Ok(())
}

fn draw_cell_border(
    image: &mut RgbImage,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
    color: image::Rgb<u8>,
) {
    for inset in 0..2 {
        for px in x + inset..x + width - inset {
            image.put_pixel(px, y + inset, color);
            image.put_pixel(px, y + height - 1 - inset, color);
        }
        for py in y + inset..y + height - inset {
            image.put_pixel(x + inset, py, color);
            image.put_pixel(x + width - 1 - inset, py, color);
        }
    }
}

fn draw_text_5x7(image: &mut RgbImage, x: u32, y: u32, text: &str, color: image::Rgb<u8>) {
    let mut cursor = x;
    for character in text.chars() {
        let glyph = glyph_5x7(character);
        for (row, bits) in glyph.into_iter().enumerate() {
            for column in 0..5u32 {
                if bits & (1 << (4 - column)) != 0 {
                    let px = cursor + column;
                    let py = y + row as u32;
                    if px < image.width() && py < image.height() {
                        image.put_pixel(px, py, color);
                    }
                }
            }
        }
        cursor += 6;
    }
}

fn glyph_5x7(character: char) -> [u8; 7] {
    match character.to_ascii_uppercase() {
        '0' => [14, 17, 19, 21, 25, 17, 14],
        '1' => [4, 12, 4, 4, 4, 4, 14],
        '2' => [14, 17, 1, 2, 4, 8, 31],
        '3' => [30, 1, 1, 14, 1, 1, 30],
        '4' => [2, 6, 10, 18, 31, 2, 2],
        '5' => [31, 16, 16, 30, 1, 1, 30],
        '6' => [14, 16, 16, 30, 17, 17, 14],
        '7' => [31, 1, 2, 4, 8, 8, 8],
        '8' => [14, 17, 17, 14, 17, 17, 14],
        '9' => [14, 17, 17, 15, 1, 1, 14],
        'A' => [14, 17, 17, 31, 17, 17, 17],
        'B' => [30, 17, 17, 30, 17, 17, 30],
        'C' => [14, 17, 16, 16, 16, 17, 14],
        'D' => [30, 17, 17, 17, 17, 17, 30],
        'E' => [31, 16, 16, 30, 16, 16, 31],
        'F' => [31, 16, 16, 30, 16, 16, 16],
        'P' => [30, 17, 17, 30, 16, 16, 16],
        'R' => [30, 17, 17, 30, 20, 18, 17],
        'T' => [31, 4, 4, 4, 4, 4, 4],
        'V' => [17, 17, 17, 17, 17, 10, 4],
        '#' => [10, 31, 10, 10, 31, 10, 0],
        '?' => [14, 17, 1, 2, 4, 0, 4],
        _ => [0; 7],
    }
}

fn write_positive_contact_sheet(output: &Path, records: &[PositiveRecord]) -> Result<()> {
    const COLS: u32 = 6;
    const ROWS: u32 = 6;
    const CELL_W: u32 = 240;
    const CELL_H: u32 = 160;
    let mut sheet = RgbImage::new(COLS * CELL_W, ROWS * CELL_H);
    for (slot, record) in records.iter().take((COLS * ROWS) as usize).enumerate() {
        let image = image::open(output.join(&record.relative_path))?.into_rgb8();
        let thumb = imageops::thumbnail(&image, CELL_W, CELL_H);
        let x = (slot as u32 % COLS) * CELL_W + (CELL_W - thumb.width()) / 2;
        let y = (slot as u32 / COLS) * CELL_H + (CELL_H - thumb.height()) / 2;
        imageops::replace(&mut sheet, &thumb, i64::from(x), i64::from(y));
    }
    sheet.save(output.join("contact-sheet.jpg"))?;
    Ok(())
}

fn write_positive_readme(output: &Path, manifest: &PositiveDatasetManifest) -> Result<()> {
    let mut splits = BTreeMap::<&str, usize>::new();
    for record in &manifest.records {
        *splits.entry(&record.split).or_default() += 1;
    }
    let text = format!(
        "# Wikimedia Commons former-Pizza-Hut positives\n\nGenerated by `roof-data import-wikimedia-positives`. These are current uses of buildings categorised by Commons contributors as former Pizza Hut restaurants; they remain positive examples when the characteristic roof is recognisable.\n\n- Records: {}\n- Train: {}\n- Validation: {}\n- Test: {}\n- Review state: category and metadata screened; use `contact-sheet.jpg` for visual review.\n- Licensing: retain the per-image source, artist, licence, and landing-page fields in `manifest.json`.\n",
        manifest.records.len(),
        splits.get("train").copied().unwrap_or(0),
        splits.get("validation").copied().unwrap_or(0),
        splits.get("test").copied().unwrap_or(0),
    );
    fs::write(output.join("README.md"), text)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_parser_filters_small_and_unrelated_boxes() {
        let input = "ImageID,Source,LabelName,Confidence,XMin,XMax,YMin,YMax,IsOccluded,IsTruncated,IsGroupOf,IsDepiction,IsInside\n\
                     house,x,/m/03jm5,1,0.1,0.8,0.2,0.9,0,0,0,0,0\n\
                     tiny,x,/m/0cgh4,1,0.1,0.2,0.1,0.2,0,0,0,0,0\n\
                     inside,x,/m/0cgh4,1,0.1,0.9,0.1,0.9,0,0,0,0,1\n\
                     depiction,x,/m/0cgh4,1,0.1,0.9,0.1,0.9,0,0,0,1,0\n\
                     cat,x,/m/01yrx,1,0.1,0.9,0.1,0.9,0,0,0,0,0\n";
        let candidates = parse_candidates(input).expect("valid fixture");
        assert_eq!(candidates.len(), 1);
        assert!(candidates.contains_key("house"));
    }

    #[test]
    fn deterministic_rank_changes_with_seed() {
        assert_eq!(stable_rank("abc", 7), stable_rank("abc", 7));
        assert_ne!(stable_rank("abc", 7), stable_rank("abc", 8));
    }

    fn review_fixture() -> DatasetManifest {
        DatasetManifest {
            schema_version: "roof-negative-dataset/v1".to_owned(),
            dataset_id: "open-images-buildings-negative".to_owned(),
            source_version: "fixture".to_owned(),
            source_bbox_url: "fixture".to_owned(),
            source_metadata_url: "fixture".to_owned(),
            selection_seed: 1,
            requested_limit: 1,
            records: vec![NegativeRecord {
                image_id: "abc".to_owned(),
                split: "test".to_owned(),
                relative_path: "abc.jpg".to_owned(),
                width: 1,
                height: 1,
                sha256: "sha256:fixture".to_owned(),
                classes: vec!["House".to_owned()],
                source_building_boxes: vec![BoxLabel {
                    min_x: 0.0,
                    min_y: 0.0,
                    max_x: 1.0,
                    max_y: 1.0,
                }],
                source_crop: BoxLabel {
                    min_x: 0.0,
                    min_y: 0.0,
                    max_x: 1.0,
                    max_y: 1.0,
                },
                source_url: "fixture".to_owned(),
                landing_page: "fixture".to_owned(),
                license: "fixture".to_owned(),
                author_profile: "fixture".to_owned(),
                author: "fixture".to_owned(),
                title: "fixture".to_owned(),
                review_status: "metadata_screened".to_owned(),
                review_reason: None,
            }],
        }
    }

    #[test]
    fn visual_review_is_bound_to_ordered_candidate_ids() {
        let manifest = review_fixture();
        let mut ledger = OpenImagesReviewLedger {
            schema_version: "roof-negative-visual-review/v1".to_owned(),
            dataset_id: manifest.dataset_id.clone(),
            ordered_image_ids_sha256: ordered_image_ids_sha256(&manifest.records),
            reviewed_record_count: 1,
            reviewer: "reviewer".to_owned(),
            reviewed_at: "2026-07-21".to_owned(),
            rejections: vec![],
        };
        validate_review_ledger(&manifest, &ledger).expect("matching ledger");
        ledger.ordered_image_ids_sha256 = "sha256:wrong".to_owned();
        assert!(validate_review_ledger(&manifest, &ledger).is_err());
    }

    #[test]
    fn visual_review_rejections_are_explicit_and_unique() {
        let manifest = review_fixture();
        let rejection = OpenImagesRejection {
            image_id: "abc".to_owned(),
            reason: "interior".to_owned(),
        };
        let mut ledger = OpenImagesReviewLedger {
            schema_version: "roof-negative-visual-review/v1".to_owned(),
            dataset_id: manifest.dataset_id.clone(),
            ordered_image_ids_sha256: ordered_image_ids_sha256(&manifest.records),
            reviewed_record_count: 1,
            reviewer: "reviewer".to_owned(),
            reviewed_at: "2026-07-21".to_owned(),
            rejections: vec![rejection.clone(), rejection],
        };
        assert!(validate_review_ledger(&manifest, &ledger).is_err());
        ledger.rejections = vec![OpenImagesRejection {
            image_id: "abc".to_owned(),
            reason: "looks-bad".to_owned(),
        }];
        assert!(validate_review_ledger(&manifest, &ledger).is_err());
    }
}
