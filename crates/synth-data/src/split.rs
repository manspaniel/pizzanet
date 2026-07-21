use std::{error::Error, fmt};

use serde::{Deserialize, Serialize};

use crate::deterministic::stable_hash64;

const BASIS_POINTS: u32 = 10_000;

/// Dataset partition assigned to every view of one generated building.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DatasetSplit {
    /// Model-fitting data.
    Train,
    /// Data used for tuning without fitting model weights.
    Validation,
    /// Held-out evaluation data.
    Test,
}

/// Stable grouping key for split assignment.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitKey {
    /// Procedural or real-world building family.
    pub building_family: String,
    /// Seed shared by every frame of the generated building instance.
    pub building_seed: u64,
    /// Optional source-asset group used to prevent asset leakage.
    pub source_asset_group: Option<String>,
}

impl SplitKey {
    /// Creates a key for a wholly procedural building.
    #[must_use]
    pub fn procedural(building_family: impl Into<String>, building_seed: u64) -> Self {
        Self {
            building_family: building_family.into(),
            building_seed,
            source_asset_group: None,
        }
    }
}

/// Versioned basis-point allocation for deterministic dataset splits.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SplitPolicy {
    /// Policy version; change this when hashing semantics change.
    pub version: u32,
    /// Dataset-specific namespace mixed into the hash.
    pub salt: String,
    /// Training allocation in basis points.
    pub train_basis_points: u32,
    /// Validation allocation in basis points.
    pub validation_basis_points: u32,
    /// Test allocation in basis points.
    pub test_basis_points: u32,
}

impl Default for SplitPolicy {
    fn default() -> Self {
        Self {
            version: 1,
            salt: "pizza-hut-roof-v1".to_owned(),
            train_basis_points: 8_000,
            validation_basis_points: 1_000,
            test_basis_points: 1_000,
        }
    }
}

impl SplitPolicy {
    /// Creates and validates a split policy.
    pub fn new(
        salt: impl Into<String>,
        train_basis_points: u32,
        validation_basis_points: u32,
        test_basis_points: u32,
    ) -> Result<Self, SplitPolicyError> {
        let policy = Self {
            version: 1,
            salt: salt.into(),
            train_basis_points,
            validation_basis_points,
            test_basis_points,
        };
        policy.check()?;
        Ok(policy)
    }

    /// Assigns a stable split from building identity, never from a frame index.
    pub fn assign(&self, key: &SplitKey) -> Result<DatasetSplit, SplitPolicyError> {
        self.check()?;
        if key.building_family.trim().is_empty() {
            return Err(SplitPolicyError::EmptyBuildingFamily);
        }

        let seed = key.building_seed.to_le_bytes();
        let bucket_hash = match key.source_asset_group.as_deref() {
            Some(group) if group.trim().is_empty() => {
                return Err(SplitPolicyError::EmptySourceAssetGroup);
            }
            // Source assets take precedence over generated instance identity: every
            // scene derived from one photograph/mesh group must stay in one split.
            Some(group) => stable_hash64(&[
                b"synth-data-split-v1/source-asset",
                self.salt.as_bytes(),
                group.as_bytes(),
            ]),
            None => stable_hash64(&[
                b"synth-data-split-v1/procedural-building",
                self.salt.as_bytes(),
                key.building_family.as_bytes(),
                &seed,
            ]),
        };
        let bucket = bucket_hash % u64::from(BASIS_POINTS);
        let validation_start = u64::from(self.train_basis_points);
        let test_start = validation_start + u64::from(self.validation_basis_points);

        Ok(if bucket < validation_start {
            DatasetSplit::Train
        } else if bucket < test_start {
            DatasetSplit::Validation
        } else {
            DatasetSplit::Test
        })
    }

    fn check(&self) -> Result<(), SplitPolicyError> {
        if self.version != 1 {
            return Err(SplitPolicyError::UnsupportedVersion(self.version));
        }
        if self.salt.trim().is_empty() {
            return Err(SplitPolicyError::EmptySalt);
        }
        let total = self
            .train_basis_points
            .checked_add(self.validation_basis_points)
            .and_then(|sum| sum.checked_add(self.test_basis_points))
            .ok_or(SplitPolicyError::InvalidAllocation)?;
        if total != BASIS_POINTS {
            return Err(SplitPolicyError::InvalidAllocation);
        }
        Ok(())
    }
}

/// Invalid split policy or grouping key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SplitPolicyError {
    /// The policy format is newer than this crate understands.
    UnsupportedVersion(u32),
    /// The policy namespace is empty.
    EmptySalt,
    /// Allocations do not total exactly 10,000 basis points.
    InvalidAllocation,
    /// A building cannot be grouped without a family identifier.
    EmptyBuildingFamily,
    /// Present asset groups must contain a non-empty stable identifier.
    EmptySourceAssetGroup,
}

impl fmt::Display for SplitPolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported split policy version {version}")
            }
            Self::EmptySalt => formatter.write_str("split policy salt must not be empty"),
            Self::InvalidAllocation => {
                formatter.write_str("split allocations must total 10,000 basis points")
            }
            Self::EmptyBuildingFamily => {
                formatter.write_str("split key building family must not be empty")
            }
            Self::EmptySourceAssetGroup => {
                formatter.write_str("split key source asset group must not be empty")
            }
        }
    }
}

impl Error for SplitPolicyError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_is_stable_for_a_building() {
        let policy = SplitPolicy::default();
        let key = SplitKey::procedural("classic_two_stage", 9_821);
        assert_eq!(policy.assign(&key), policy.assign(&key));
    }

    #[test]
    fn policy_rejects_bad_totals() {
        assert_eq!(
            SplitPolicy::new("test", 8_000, 1_000, 999),
            Err(SplitPolicyError::InvalidAllocation)
        );
    }

    #[test]
    fn source_asset_group_prevents_cross_seed_leakage() {
        let policy = SplitPolicy::default();
        let key = |building_seed| SplitKey {
            building_family: "classic_two_stage".to_owned(),
            building_seed,
            source_asset_group: Some("real-building-0042".to_owned()),
        };
        assert_eq!(policy.assign(&key(1)), policy.assign(&key(999_999)));
    }
}
