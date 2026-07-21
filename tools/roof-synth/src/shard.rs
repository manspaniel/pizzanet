//! Deterministic WebDataset-compatible tar shard output.

use std::{
    fs::{self, File},
    io,
    path::{Path, PathBuf},
};

/// One named file belonging to a sample in a WebDataset shard.
#[derive(Clone, Copy, Debug)]
pub struct Artifact<'a> {
    /// Suffix appended after the shared sample key, such as `rgb.png`.
    pub suffix: &'a str,
    /// Encoded file contents.
    pub bytes: &'a [u8],
}

/// Sequential writer which rotates deterministic tar shards by sample count.
pub struct ShardWriter {
    output_dir: PathBuf,
    prefix: String,
    samples_per_shard: usize,
    shard_index: u32,
    sample_count: usize,
    builder: Option<tar::Builder<File>>,
    completed: Vec<PathBuf>,
}

impl ShardWriter {
    /// Creates a writer. The first shard is opened only when a sample arrives.
    pub fn new(
        output_dir: impl Into<PathBuf>,
        prefix: impl Into<String>,
        samples_per_shard: usize,
    ) -> io::Result<Self> {
        if samples_per_shard == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "samples_per_shard must be non-zero",
            ));
        }
        let prefix = prefix.into();
        validate_component(&prefix)?;
        let output_dir = output_dir.into();
        fs::create_dir_all(&output_dir)?;
        Ok(Self {
            output_dir,
            prefix,
            samples_per_shard,
            shard_index: 0,
            sample_count: 0,
            builder: None,
            completed: Vec::new(),
        })
    }

    /// Adds all artifacts for one sample under their shared basename.
    pub fn append(&mut self, sample_key: &str, artifacts: &[Artifact<'_>]) -> io::Result<()> {
        validate_component(sample_key)?;
        if artifacts.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "sample must contain at least one artifact",
            ));
        }
        for artifact in artifacts {
            validate_suffix(artifact.suffix)?;
        }

        if self.sample_count == self.samples_per_shard {
            self.finish_current()?;
        }
        self.ensure_open()?;
        let builder = self.builder.as_mut().expect("builder opened above");
        for artifact in artifacts {
            let path = format!("{sample_key}.{}", artifact.suffix);
            let mut header = tar::Header::new_gnu();
            header.set_size(artifact.bytes.len() as u64);
            header.set_mode(0o644);
            header.set_mtime(0);
            header.set_uid(0);
            header.set_gid(0);
            header.set_cksum();
            builder.append_data(&mut header, path, artifact.bytes)?;
        }
        self.sample_count += 1;
        Ok(())
    }

    /// Finishes the current shard and returns every completed shard path.
    pub fn finish(mut self) -> io::Result<Vec<PathBuf>> {
        self.finish_current()?;
        Ok(self.completed)
    }

    fn ensure_open(&mut self) -> io::Result<()> {
        if self.builder.is_none() {
            let path = self.current_path();
            let file = File::create(path)?;
            self.builder = Some(tar::Builder::new(file));
            self.sample_count = 0;
        }
        Ok(())
    }

    fn finish_current(&mut self) -> io::Result<()> {
        if let Some(mut builder) = self.builder.take() {
            builder.finish()?;
            let file = builder.into_inner()?;
            file.sync_all()?;
            self.completed.push(self.current_path());
            self.shard_index += 1;
            self.sample_count = 0;
        }
        Ok(())
    }

    fn current_path(&self) -> PathBuf {
        self.output_dir
            .join(format!("{}-{:06}.tar", self.prefix, self.shard_index))
    }
}

fn validate_component(component: &str) -> io::Result<()> {
    if component.is_empty()
        || !component
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid dataset path component: {component:?}"),
        ));
    }
    Ok(())
}

fn validate_suffix(suffix: &str) -> io::Result<()> {
    if suffix.is_empty()
        || suffix.starts_with('.')
        || suffix.ends_with('.')
        || suffix
            .split('.')
            .any(|part| validate_component(part).is_err())
    {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("invalid artifact suffix: {suffix:?}"),
        ));
    }
    Ok(())
}

/// Lists regular files in a generated shard directory in stable order.
pub fn list_shards(directory: &Path) -> io::Result<Vec<PathBuf>> {
    let mut paths = fs::read_dir(directory)?
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .filter(|path| path.extension().is_some_and(|extension| extension == "tar"))
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use super::*;

    #[test]
    fn groups_artifacts_and_rotates_shards() {
        let directory = tempfile::tempdir().unwrap();
        let mut writer = ShardWriter::new(directory.path(), "train", 1).unwrap();
        writer
            .append(
                "sample_a",
                &[
                    Artifact {
                        suffix: "rgb.png",
                        bytes: b"rgb-a",
                    },
                    Artifact {
                        suffix: "labels.json",
                        bytes: b"{}",
                    },
                ],
            )
            .unwrap();
        writer
            .append(
                "sample_b",
                &[Artifact {
                    suffix: "rgb.png",
                    bytes: b"rgb-b",
                }],
            )
            .unwrap();
        let paths = writer.finish().unwrap();
        assert_eq!(paths.len(), 2);

        let mut archive = tar::Archive::new(File::open(&paths[0]).unwrap());
        let mut entries = archive.entries().unwrap();
        let mut first = entries.next().unwrap().unwrap();
        assert_eq!(
            first.path().unwrap().as_ref(),
            Path::new("sample_a.rgb.png")
        );
        let mut bytes = Vec::new();
        first.read_to_end(&mut bytes).unwrap();
        assert_eq!(bytes, b"rgb-a");
        assert_eq!(entries.count(), 1);
    }

    #[test]
    fn rejects_paths_and_empty_samples() {
        let directory = tempfile::tempdir().unwrap();
        assert!(ShardWriter::new(directory.path(), "../train", 1).is_err());

        let mut writer = ShardWriter::new(directory.path(), "train", 1).unwrap();
        assert!(writer.append("../escape", &[]).is_err());
        assert!(writer.append("sample", &[]).is_err());
        assert!(
            writer
                .append(
                    "sample",
                    &[Artifact {
                        suffix: "../json",
                        bytes: b"{}",
                    }],
                )
                .is_err()
        );
    }
}
