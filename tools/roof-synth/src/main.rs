//! Command-line entry point for synthetic roof dataset generation.

use std::{path::PathBuf, process::ExitCode};

use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use roof_synth::{
    generate::{GenerateOptions, generate_dataset},
    validate::validate_dataset,
};

#[derive(Debug, Parser)]
#[command(
    name = "roof-synth",
    about = "Generate and validate deterministic synthetic roof datasets"
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Render coherent target-building sequences into WebDataset tar shards.
    Generate(GenerateArgs),
    /// Verify manifests, records, asset hashes, and sequence relationships.
    Validate {
        /// Generated dataset directory containing dataset.json.
        dataset: PathBuf,
    },
}

#[derive(Debug, Args)]
struct GenerateArgs {
    /// New or empty output directory.
    #[arg(long, default_value = "datasets/synthetic/roof-local")]
    output: PathBuf,
    /// Manifest-safe dataset identifier.
    #[arg(long, default_value = "roof-local")]
    dataset_id: String,
    /// First deterministic building seed.
    #[arg(long, default_value_t = 42)]
    seed: u64,
    /// Number of independently sampled two-tier target buildings.
    #[arg(long, default_value_t = 1)]
    targets: u32,
    /// Number of independently sampled ordinary-building negatives.
    #[arg(long, default_value_t = 1)]
    negatives: u32,
    /// Number of coherent camera views per building.
    #[arg(long, default_value_t = 1)]
    frames: u32,
    /// Output image width in pixels.
    #[arg(long, default_value_t = 640)]
    width: u32,
    /// Output image height in pixels.
    #[arg(long, default_value_t = 480)]
    height: u32,
    /// Maximum frame samples per tar shard.
    #[arg(long, default_value_t = 256)]
    samples_per_shard: usize,
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(code) => code,
        Err(error) => {
            eprintln!("roof-synth: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<ExitCode> {
    match cli.command {
        Command::Generate(args) => {
            let summary = generate_dataset(&GenerateOptions {
                output: args.output,
                dataset_id: args.dataset_id,
                seed: args.seed,
                target_count: args.targets,
                negative_count: args.negatives,
                frames_per_sequence: args.frames,
                width: args.width,
                height: args.height,
                samples_per_shard: args.samples_per_shard,
            })?;
            serde_json::to_writer_pretty(std::io::stdout().lock(), &summary)?;
            println!();
            Ok(ExitCode::SUCCESS)
        }
        Command::Validate { dataset } => {
            let summary = validate_dataset(dataset)?;
            let valid = summary.valid;
            serde_json::to_writer_pretty(std::io::stdout().lock(), &summary)?;
            println!();
            Ok(if valid {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(2)
            })
        }
    }
}
