//! Command-line interface for offline VIO sensor replay.

#![forbid(unsafe_code)]

use anyhow::{Context, Result, bail};
use clap::{Parser, Subcommand};
use serde::Serialize;
use std::{
    fs::File,
    io::{self, BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    str::FromStr,
};
use vio_core::{DVec3, SensorBatch};
use vio_replay::{SimulationConfig, inspect, simulate};

#[derive(Debug, Parser)]
#[command(about = "Generate and inspect deterministic VIO sensor replays")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Generate constant normalized IMU measurements and preintegrate them.
    Simulate {
        /// Simulated interval duration in seconds.
        #[arg(long, default_value_t = 1.0)]
        duration_seconds: f64,

        /// IMU sample rate in hertz.
        #[arg(long, default_value_t = 100.0)]
        sample_rate_hz: f64,

        /// Constant body angular velocity in radians/second as X,Y,Z.
        #[arg(long, value_name = "X,Y,Z", default_value = "0,0,0")]
        angular_velocity_rad_s: Vector3Argument,

        /// Constant body specific force in metres/second squared as X,Y,Z.
        #[arg(long, value_name = "X,Y,Z", default_value = "0,0,9.80665")]
        specific_force_mps2: Vector3Argument,

        /// Pretty JSON report path; omit to write to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Also write the generated raw SensorBatch as pretty JSON.
        #[arg(long)]
        batch_output: Option<PathBuf>,
    },

    /// Read a JSON SensorBatch and report its preintegration diagnostics.
    Inspect {
        /// SensorBatch JSON produced by this tool or another acquisition adapter.
        input: PathBuf,

        /// Pretty JSON report path; omit to write to stdout.
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
}

#[derive(Clone, Copy, Debug)]
struct Vector3Argument(DVec3);

impl FromStr for Vector3Argument {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        let components = value
            .split(',')
            .map(str::trim)
            .map(|component| {
                component
                    .parse::<f64>()
                    .map_err(|error| format!("invalid component `{component}`: {error}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let [x, y, z] = components.as_slice() else {
            return Err("expected exactly three comma-separated components: X,Y,Z".to_owned());
        };
        let vector = DVec3::new(*x, *y, *z);
        if !vector.is_finite() {
            return Err("vector components must be finite".to_owned());
        }
        Ok(Self(vector))
    }
}

fn main() -> Result<()> {
    match Cli::parse().command {
        Command::Simulate {
            duration_seconds,
            sample_rate_hz,
            angular_velocity_rad_s,
            specific_force_mps2,
            output,
            batch_output,
        } => {
            if let (Some(report_path), Some(batch_path)) = (&output, &batch_output)
                && report_path == batch_path
            {
                bail!("report and batch output paths must be different");
            }

            let config = SimulationConfig::new(
                duration_seconds,
                sample_rate_hz,
                angular_velocity_rad_s.0,
                specific_force_mps2.0,
            )?;
            let simulation = simulate(config)?;
            if let Some(path) = batch_output {
                write_json_file(&path, simulation.batch())
                    .with_context(|| format!("failed to write batch to {}", path.display()))?;
            }
            write_json(output.as_deref(), simulation.report())?;
        }
        Command::Inspect { input, output } => {
            let file = File::open(&input)
                .with_context(|| format!("failed to open batch {}", input.display()))?;
            let batch: SensorBatch =
                serde_json::from_reader(BufReader::new(file)).with_context(|| {
                    format!("failed to parse SensorBatch JSON in {}", input.display())
                })?;
            let report = inspect(&batch)?;
            write_json(output.as_deref(), &report)?;
        }
    }

    Ok(())
}

fn write_json<T: Serialize>(path: Option<&Path>, value: &T) -> Result<()> {
    match path {
        Some(path) => write_json_file(path, value)
            .with_context(|| format!("failed to write report to {}", path.display())),
        None => {
            let stdout = io::stdout();
            let mut writer = BufWriter::new(stdout.lock());
            write_pretty_json(&mut writer, value).context("failed to write report to stdout")
        }
    }
}

fn write_json_file<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    let file = File::create(path)?;
    let mut writer = BufWriter::new(file);
    write_pretty_json(&mut writer, value)
}

fn write_pretty_json<T: Serialize>(writer: &mut impl Write, value: &T) -> Result<()> {
    serde_json::to_writer_pretty(&mut *writer, value)?;
    writeln!(writer)?;
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vector_argument_requires_three_finite_components() {
        assert_eq!(
            "1, 2, 3".parse::<Vector3Argument>().unwrap().0,
            DVec3::new(1.0, 2.0, 3.0)
        );
        assert!("1,2".parse::<Vector3Argument>().is_err());
        assert!("1,2,NaN".parse::<Vector3Argument>().is_err());
    }
}
