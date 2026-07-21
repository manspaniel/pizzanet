//! Deterministic coverage-controlled sequence selection.

use std::{collections::BTreeSet, error::Error, fmt};

use serde::{Deserialize, Serialize};
use synth_data::{
    DayPhase, GeneratorConfig, OrdinaryRoofFamily, RoofMorphology, SceneDomain, SequencePlan,
    SequenceRequest, SequenceSampler, TargetKind,
};

/// Maximum number of consecutive candidate seeds inspected for categorical coverage.
///
/// One million candidates is deliberately much larger than the expected discovery
/// interval for the default 45 positive-weight cells, while still making a broken
/// or incompatible distribution terminate deterministically.
pub const CANDIDATE_SCAN_LIMIT: u64 = 1 << 20;

/// Keeps negative seed streams independent from target buildings even when the
/// caller uses the same root seed for one balanced corpus.
const NEGATIVE_SEED_NAMESPACE: u64 = 1 << 32;

/// Three scene regimes required by the training plan.
///
/// The sampler retains five detailed domains. City and urban form the urban
/// regime, roadside and remote form the remote regime, and suburban remains
/// its own regime.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SceneRegime {
    /// City and urban detailed domains.
    Urban,
    /// Suburban detailed domain.
    Suburban,
    /// Roadside and remote detailed domains.
    Remote,
}

impl SceneRegime {
    const ALL: [Self; 3] = [Self::Urban, Self::Suburban, Self::Remote];

    const fn index(self) -> usize {
        match self {
            Self::Urban => 0,
            Self::Suburban => 1,
            Self::Remote => 2,
        }
    }
}

/// Exact counts for the three balanced scene regimes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SceneRegimeCounts {
    /// City plus urban detailed-domain scenes.
    pub urban: usize,
    /// Suburban scenes.
    pub suburban: usize,
    /// Roadside plus remote detailed-domain scenes.
    pub remote: usize,
}

impl SceneRegimeCounts {
    pub(crate) fn increment(&mut self, regime: SceneRegime) {
        match regime {
            SceneRegime::Urban => self.urban += 1,
            SceneRegime::Suburban => self.suburban += 1,
            SceneRegime::Remote => self.remote += 1,
        }
    }

    /// Adds corresponding regime counts.
    #[must_use]
    pub const fn plus(self, other: Self) -> Self {
        Self {
            urban: self.urban + other.urban,
            suburban: self.suburban + other.suburban,
            remote: self.remote + other.remote,
        }
    }

    /// Returns whether all three counts differ by at most one.
    #[must_use]
    pub fn is_balanced(self) -> bool {
        let values = [self.urban, self.suburban, self.remote];
        let maximum = values.into_iter().max().unwrap_or(0);
        let minimum = values.into_iter().min().unwrap_or(0);
        maximum - minimum <= 1
    }
}

/// One morphology by day-phase by site-domain coverage cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoverageCell {
    /// Architectural roof proportion family.
    pub morphology: RoofMorphology,
    /// Capture-time regime.
    pub day_phase: DayPhase,
    /// Surrounding site regime.
    pub domain: SceneDomain,
}

impl fmt::Display for CoverageCell {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "{:?}/{:?}/{:?}",
            self.morphology, self.day_phase, self.domain
        )
    }
}

/// Auditable outcome of deterministic categorical sequence selection.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CoverageSummary {
    /// Hard upper bound on consecutive candidate seeds inspected.
    pub candidate_scan_limit: u64,
    /// Whether the requested sequence count was large enough to require every cell.
    pub full_coverage_required: bool,
    /// Whether every supported cell is represented.
    pub full_coverage_achieved: bool,
    /// Number of unique supported positive-weight cells.
    pub required_cell_count: usize,
    /// Number of unique cells represented by selected sequences.
    pub covered_cell_count: usize,
    /// Supported cells in stable generator-configuration order.
    pub required_cells: Vec<CoverageCell>,
    /// Represented cells in the same stable order as `required_cells`.
    pub covered_cells: Vec<CoverageCell>,
    /// Exact selected target counts in the three plan-level scene regimes.
    pub scene_regime_counts: SceneRegimeCounts,
    /// Whether the three selected counts differ by at most one.
    pub scene_regime_balance_achieved: bool,
}

/// One selected seed and the plan sampled from it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SeededSequencePlan {
    /// Building seed supplied to the sampler.
    pub seed: u64,
    /// Exact plan produced for `seed`.
    pub plan: SequencePlan,
}

/// Exact plans plus the coverage statement they satisfy.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CoverageSelection {
    /// Plans sorted by ascending seed for stable serialized output.
    pub sequences: Vec<SeededSequencePlan>,
    /// Categorical coverage represented by `sequences`.
    pub summary: CoverageSummary,
}

/// Failure to select the requested deterministic sequence set.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum CoverageSelectionError {
    /// More output sequences were requested than the bounded candidate range contains.
    RequestedCountExceedsLimit { requested: u32, limit: u64 },
    /// Consecutive candidate-seed arithmetic exceeded `u64`.
    SeedRangeOverflow { start_seed: u64, offset: u64 },
    /// Sampling one deterministic candidate failed.
    SamplingFailed { seed: u64, message: String },
    /// A sampled scene omitted its correlated environment classification.
    MissingEnvironment { seed: u64 },
    /// The sampler emitted a cell excluded by its own positive-weight configuration.
    UnsupportedSampledCell { seed: u64, cell: CoverageCell },
    /// The bounded range did not contain every cell that the requested count requires.
    MissingRequiredCoverage {
        start_seed: u64,
        scanned: u64,
        required: usize,
        missing: Vec<CoverageCell>,
    },
    /// An undersized request could not reach its maximum possible distinct-cell count.
    InsufficientDistinctCoverage {
        start_seed: u64,
        scanned: u64,
        requested: usize,
        covered: usize,
    },
}

impl fmt::Display for CoverageSelectionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RequestedCountExceedsLimit { requested, limit } => write!(
                formatter,
                "requested {requested} sequences, but the deterministic candidate range is bounded to {limit} seeds"
            ),
            Self::SeedRangeOverflow { start_seed, offset } => write!(
                formatter,
                "candidate seed overflowed while adding offset {offset} to starting seed {start_seed}"
            ),
            Self::SamplingFailed { seed, message } => {
                write!(
                    formatter,
                    "failed to sample candidate seed {seed}: {message}"
                )
            }
            Self::MissingEnvironment { seed } => write!(
                formatter,
                "candidate seed {seed} did not contain a correlated environment classification"
            ),
            Self::UnsupportedSampledCell { seed, cell } => write!(
                formatter,
                "candidate seed {seed} sampled unsupported coverage cell {cell}"
            ),
            Self::MissingRequiredCoverage {
                start_seed,
                scanned,
                required,
                missing,
            } => {
                write!(
                    formatter,
                    "coverage scan from seed {start_seed} exhausted {scanned} candidates; missing {} of {required} required morphology/day-phase/domain cells",
                    missing.len()
                )?;
                for cell in missing {
                    write!(formatter, " [{cell}]")?;
                }
                Ok(())
            }
            Self::InsufficientDistinctCoverage {
                start_seed,
                scanned,
                requested,
                covered,
            } => write!(
                formatter,
                "coverage scan from seed {start_seed} exhausted {scanned} candidates; requested {requested} distinct morphology/day-phase/domain cells but found {covered}"
            ),
        }
    }
}

impl Error for CoverageSelectionError {}

/// Selects exactly `sequence_count` plans from a bounded consecutive seed range.
pub(crate) fn select_sequence_plans(
    sampler: &SequenceSampler,
    start_seed: u64,
    sequence_count: u32,
) -> Result<CoverageSelection, CoverageSelectionError> {
    select_sequence_plans_with_limit(sampler, start_seed, sequence_count, CANDIDATE_SCAN_LIMIT)
}

/// Samples exactly `sequence_count` ordinary-building negative plans while
/// balancing the seven roof topologies to within one example.
pub(crate) fn select_negative_sequence_plans(
    sampler: &SequenceSampler,
    start_seed: u64,
    preceding_target_count: u32,
    sequence_count: u32,
) -> Result<Vec<SeededSequencePlan>, CoverageSelectionError> {
    let negative_start = start_seed.checked_add(NEGATIVE_SEED_NAMESPACE).ok_or(
        CoverageSelectionError::SeedRangeOverflow {
            start_seed,
            offset: NEGATIVE_SEED_NAMESPACE,
        },
    )?;
    let target_quotas = balanced_regime_quotas(preceding_target_count as usize);
    let combined_quotas =
        balanced_regime_quotas(preceding_target_count as usize + sequence_count as usize);
    let negative_quotas = [
        combined_quotas[0] - target_quotas[0],
        combined_quotas[1] - target_quotas[1],
        combined_quotas[2] - target_quotas[2],
    ];
    let desired_regimes = interleaved_regime_schedule(negative_quotas);
    let mut selected = Vec::with_capacity(sequence_count as usize);
    let mut candidate_offset = 0_u64;
    for (index, desired_regime) in desired_regimes.into_iter().enumerate() {
        let family = OrdinaryRoofFamily::ALL[index % OrdinaryRoofFamily::ALL.len()];
        loop {
            if candidate_offset >= CANDIDATE_SCAN_LIMIT {
                return Err(CoverageSelectionError::InsufficientDistinctCoverage {
                    start_seed: negative_start,
                    scanned: candidate_offset,
                    requested: sequence_count as usize,
                    covered: selected.len(),
                });
            }
            let seed = checked_candidate_seed(negative_start, candidate_offset)?;
            candidate_offset += 1;
            let plan = sampler
                .sample(SequenceRequest::procedural(
                    format!("ordinary_{}", family.as_str()),
                    seed,
                    TargetKind::Negative,
                ))
                .map_err(|error| CoverageSelectionError::SamplingFailed {
                    seed,
                    message: error.to_string(),
                })?;
            if scene_regime_from_plan(seed, &plan)? == desired_regime {
                selected.push(SeededSequencePlan { seed, plan });
                break;
            }
        }
    }
    Ok(selected)
}

fn select_sequence_plans_with_limit(
    sampler: &SequenceSampler,
    start_seed: u64,
    sequence_count: u32,
    candidate_scan_limit: u64,
) -> Result<CoverageSelection, CoverageSelectionError> {
    if u64::from(sequence_count) > candidate_scan_limit {
        return Err(CoverageSelectionError::RequestedCountExceedsLimit {
            requested: sequence_count,
            limit: candidate_scan_limit,
        });
    }

    let required_cells = supported_cells(sampler.config());
    let requested = sequence_count as usize;
    let regime_quotas = balanced_regime_quotas(requested);
    let required_regime_cells = required_cell_counts_by_regime(&required_cells);
    let distinct_regime_quotas = [
        regime_quotas[0].min(required_regime_cells[0]),
        regime_quotas[1].min(required_regime_cells[1]),
        regime_quotas[2].min(required_regime_cells[2]),
    ];
    let distinct_target = distinct_regime_quotas.iter().sum();
    let full_coverage_required = distinct_target == required_cells.len();
    let mut selected = Vec::with_capacity(requested);
    let mut covered = Vec::with_capacity(distinct_target);
    let mut selected_regime_counts = [0_usize; 3];
    let mut scanned = 0_u64;

    for offset in 0..candidate_scan_limit {
        if covered.len() == distinct_target {
            break;
        }
        let candidate = sample_candidate(sampler, start_seed, offset)?;
        scanned = offset + 1;
        let cell = cell_from_plan(candidate.seed, &candidate.plan)?;
        if !required_cells.contains(&cell) {
            return Err(CoverageSelectionError::UnsupportedSampledCell {
                seed: candidate.seed,
                cell,
            });
        }
        let regime_index = scene_regime(cell.domain).index();
        if !covered.contains(&cell)
            && selected_regime_counts[regime_index] < distinct_regime_quotas[regime_index]
        {
            covered.push(cell);
            selected.push(candidate);
            selected_regime_counts[regime_index] += 1;
        }
    }

    if covered.len() != distinct_target {
        if full_coverage_required {
            let missing = required_cells
                .iter()
                .copied()
                .filter(|cell| !covered.contains(cell))
                .collect();
            return Err(CoverageSelectionError::MissingRequiredCoverage {
                start_seed,
                scanned,
                required: required_cells.len(),
                missing,
            });
        }
        return Err(CoverageSelectionError::InsufficientDistinctCoverage {
            start_seed,
            scanned,
            requested: distinct_target,
            covered: covered.len(),
        });
    }

    let mut selected_seeds = selected
        .iter()
        .map(|candidate| candidate.seed)
        .collect::<BTreeSet<_>>();
    for offset in 0..candidate_scan_limit {
        if selected.len() == requested {
            break;
        }
        let seed = checked_candidate_seed(start_seed, offset)?;
        if selected_seeds.contains(&seed) {
            continue;
        }
        let candidate = sample_seed(sampler, seed)?;
        let regime = scene_regime_from_plan(candidate.seed, &candidate.plan)?;
        let regime_index = regime.index();
        if selected_regime_counts[regime_index] < regime_quotas[regime_index] {
            selected_seeds.insert(seed);
            selected.push(candidate);
            selected_regime_counts[regime_index] += 1;
        }
    }

    if selected.len() != requested {
        return Err(CoverageSelectionError::RequestedCountExceedsLimit {
            requested: sequence_count,
            limit: candidate_scan_limit,
        });
    }
    selected.sort_by_key(|candidate| candidate.seed);

    let covered_cells = required_cells
        .iter()
        .copied()
        .filter(|required| {
            selected.iter().any(|candidate| {
                cell_from_plan(candidate.seed, &candidate.plan).is_ok_and(|cell| cell == *required)
            })
        })
        .collect::<Vec<_>>();
    let scene_regime_counts = SceneRegimeCounts {
        urban: selected_regime_counts[0],
        suburban: selected_regime_counts[1],
        remote: selected_regime_counts[2],
    };
    let summary = CoverageSummary {
        candidate_scan_limit,
        full_coverage_required,
        full_coverage_achieved: covered_cells.len() == required_cells.len(),
        required_cell_count: required_cells.len(),
        covered_cell_count: covered_cells.len(),
        required_cells,
        covered_cells,
        scene_regime_counts,
        scene_regime_balance_achieved: scene_regime_counts.is_balanced(),
    };

    Ok(CoverageSelection {
        sequences: selected,
        summary,
    })
}

fn supported_cells(config: &GeneratorConfig) -> Vec<CoverageCell> {
    let mut morphologies = Vec::new();
    for profile in &config.roof.profiles {
        if profile.weight > 0 && !morphologies.contains(&profile.morphology) {
            morphologies.push(profile.morphology);
        }
    }
    let mut day_phases = Vec::new();
    for profile in &config.composition.day_phase.profiles {
        if profile.weight > 0 && !day_phases.contains(&profile.phase) {
            day_phases.push(profile.phase);
        }
    }
    let mut domains = Vec::new();
    for profile in &config.composition.domains.profiles {
        if profile.weight > 0 && !domains.contains(&profile.domain) {
            domains.push(profile.domain);
        }
    }

    let mut cells = Vec::new();
    for morphology in morphologies {
        for day_phase in &day_phases {
            for domain in &domains {
                cells.push(CoverageCell {
                    morphology,
                    day_phase: *day_phase,
                    domain: *domain,
                });
            }
        }
    }
    cells
}

fn balanced_regime_quotas(count: usize) -> [usize; 3] {
    let base = count / SceneRegime::ALL.len();
    let remainder = count % SceneRegime::ALL.len();
    [
        base + usize::from(remainder > 0),
        base + usize::from(remainder > 1),
        base,
    ]
}

fn required_cell_counts_by_regime(cells: &[CoverageCell]) -> [usize; 3] {
    let mut counts = [0_usize; 3];
    for cell in cells {
        counts[scene_regime(cell.domain).index()] += 1;
    }
    counts
}

fn interleaved_regime_schedule(mut quotas: [usize; 3]) -> Vec<SceneRegime> {
    let mut schedule = Vec::with_capacity(quotas.iter().sum());
    while quotas.iter().any(|count| *count > 0) {
        for regime in SceneRegime::ALL {
            let index = regime.index();
            if quotas[index] > 0 {
                quotas[index] -= 1;
                schedule.push(regime);
            }
        }
    }
    schedule
}

/// Maps the five detailed generator domains onto the three balanced regimes.
#[must_use]
pub const fn scene_regime(domain: SceneDomain) -> SceneRegime {
    match domain {
        SceneDomain::City | SceneDomain::Urban => SceneRegime::Urban,
        SceneDomain::Suburban => SceneRegime::Suburban,
        SceneDomain::Roadside | SceneDomain::Remote => SceneRegime::Remote,
    }
}

fn sample_candidate(
    sampler: &SequenceSampler,
    start_seed: u64,
    offset: u64,
) -> Result<SeededSequencePlan, CoverageSelectionError> {
    let seed = checked_candidate_seed(start_seed, offset)?;
    sample_seed(sampler, seed)
}

fn checked_candidate_seed(start_seed: u64, offset: u64) -> Result<u64, CoverageSelectionError> {
    start_seed
        .checked_add(offset)
        .ok_or(CoverageSelectionError::SeedRangeOverflow { start_seed, offset })
}

fn sample_seed(
    sampler: &SequenceSampler,
    seed: u64,
) -> Result<SeededSequencePlan, CoverageSelectionError> {
    let plan = sampler
        .sample(SequenceRequest::procedural(
            "classic_two_stage",
            seed,
            TargetKind::Target,
        ))
        .map_err(|error| CoverageSelectionError::SamplingFailed {
            seed,
            message: error.to_string(),
        })?;
    Ok(SeededSequencePlan { seed, plan })
}

fn cell_from_plan(seed: u64, plan: &SequencePlan) -> Result<CoverageCell, CoverageSelectionError> {
    let environment = plan
        .scene
        .composition
        .environment
        .ok_or(CoverageSelectionError::MissingEnvironment { seed })?;
    Ok(CoverageCell {
        morphology: plan.scene.roof.morphology,
        day_phase: environment.day_phase,
        domain: environment.domain,
    })
}

fn scene_regime_from_plan(
    seed: u64,
    plan: &SequencePlan,
) -> Result<SceneRegime, CoverageSelectionError> {
    Ok(scene_regime(cell_from_plan(seed, plan)?.domain))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sampler() -> SequenceSampler {
        let mut config = GeneratorConfig::default();
        config.sequence.frame_count = 1;
        SequenceSampler::new(config).unwrap()
    }

    #[test]
    fn default_configuration_covers_all_forty_five_cells_without_regime_skew() {
        // The five detailed domains produce 45 cells, while exact balance over
        // their three plan-level regimes requires 54 examples: 18 per regime.
        let selection = select_sequence_plans(&sampler(), 0, 54).unwrap();

        assert_eq!(selection.sequences.len(), 54);
        assert_eq!(selection.summary.required_cell_count, 45);
        assert_eq!(selection.summary.covered_cell_count, 45);
        assert!(selection.summary.full_coverage_required);
        assert!(selection.summary.full_coverage_achieved);
        assert_eq!(
            selection.summary.scene_regime_counts,
            SceneRegimeCounts {
                urban: 18,
                suburban: 18,
                remote: 18,
            }
        );
        assert!(selection.summary.scene_regime_balance_achieved);
        assert_eq!(
            selection.summary.covered_cells,
            selection.summary.required_cells
        );
    }

    #[test]
    fn repeated_selection_is_exactly_stable() {
        let sampler = sampler();
        let first = select_sequence_plans(&sampler, 1_234, 52).unwrap();
        let second = select_sequence_plans(&sampler, 1_234, 52).unwrap();

        assert_eq!(first, second);
    }

    #[test]
    fn undersized_selection_is_exact_and_uses_unique_seeds_and_cells() {
        let selection = select_sequence_plans(&sampler(), 77, 12).unwrap();
        let seeds = selection
            .sequences
            .iter()
            .map(|candidate| candidate.seed)
            .collect::<BTreeSet<_>>();

        assert_eq!(selection.sequences.len(), 12);
        assert_eq!(seeds.len(), 12);
        assert_eq!(selection.summary.covered_cell_count, 12);
        assert_eq!(
            selection.summary.scene_regime_counts,
            SceneRegimeCounts {
                urban: 4,
                suburban: 4,
                remote: 4,
            }
        );
        assert!(selection.summary.scene_regime_balance_achieved);
        assert!(!selection.summary.full_coverage_required);
        assert!(!selection.summary.full_coverage_achieved);
    }

    #[test]
    fn bounded_scan_and_seed_overflow_fail_cleanly() {
        let sampler = sampler();
        let bound_error = select_sequence_plans_with_limit(&sampler, 0, 54, 54).unwrap_err();
        assert!(matches!(
            bound_error,
            CoverageSelectionError::MissingRequiredCoverage { scanned: 54, .. }
        ));
        assert!(bound_error.to_string().contains("missing"));

        let overflow_error =
            select_sequence_plans_with_limit(&sampler, u64::MAX, 2, 2).unwrap_err();
        assert_eq!(
            overflow_error,
            CoverageSelectionError::SeedRangeOverflow {
                start_seed: u64::MAX,
                offset: 1,
            }
        );
        assert!(overflow_error.to_string().contains("overflowed"));
    }

    #[test]
    fn negative_selection_balances_genuine_ordinary_roof_families() {
        let plans = select_negative_sequence_plans(&sampler(), 42, 0, 15).unwrap();
        let counts = OrdinaryRoofFamily::ALL.map(|family| {
            plans
                .iter()
                .filter(|candidate| {
                    candidate.plan.scene.ordinary_roof.map(|roof| roof.family) == Some(family)
                })
                .count()
        });

        assert_eq!(plans.len(), 15);
        assert_eq!(counts.iter().sum::<usize>(), 15);
        assert!(counts.iter().all(|count| (2..=3).contains(count)));
        assert!(plans.iter().all(|candidate| {
            candidate.plan.request.target_kind == TargetKind::Negative
                && candidate.plan.roof_instance().is_none()
        }));
        assert_eq!(regime_counts(&plans), [5, 5, 5]);
    }

    #[test]
    fn target_and_negative_selection_balance_each_class_and_the_combined_corpus() {
        let sampler = sampler();
        let targets = select_sequence_plans(&sampler, 4_200, 32).unwrap();
        let negatives = select_negative_sequence_plans(&sampler, 4_200, 32, 32).unwrap();
        let target_counts = regime_counts(&targets.sequences);
        let negative_counts = regime_counts(&negatives);
        let combined = [
            target_counts[0] + negative_counts[0],
            target_counts[1] + negative_counts[1],
            target_counts[2] + negative_counts[2],
        ];

        assert_eq!(target_counts, [11, 11, 10]);
        assert_eq!(negative_counts, [11, 10, 11]);
        assert_eq!(combined, [22, 21, 21]);
    }

    fn regime_counts(plans: &[SeededSequencePlan]) -> [usize; 3] {
        let mut counts = [0_usize; 3];
        for candidate in plans {
            let regime = scene_regime_from_plan(candidate.seed, &candidate.plan).unwrap();
            counts[regime.index()] += 1;
        }
        counts
    }
}
