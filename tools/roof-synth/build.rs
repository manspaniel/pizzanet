//! Captures source-control provenance for generated dataset manifests.

use std::{env, path::Path, process::Command};

fn main() {
    let manifest_directory = env::var_os("CARGO_MANIFEST_DIR").expect("Cargo sets manifest dir");
    let repository = Path::new(&manifest_directory).join("../..");
    let output = Command::new("git")
        .args(["rev-parse", "--verify", "HEAD"])
        .current_dir(repository)
        .output();
    if let Ok(output) = output
        && output.status.success()
        && let Ok(revision) = String::from_utf8(output.stdout)
    {
        println!(
            "cargo:rustc-env=ROOF_SYNTH_GIT_REVISION={}",
            revision.trim()
        );
    }
}
