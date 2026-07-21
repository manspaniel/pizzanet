//! Native tooling shared by the `roof-synth` command-line operations.

pub mod assets;
mod coverage;
pub mod generate;
pub mod photometric;
pub mod preview;
pub mod render_plan;
pub mod shard;
pub mod validate;

pub use coverage::{CoverageCell, CoverageSummary};
