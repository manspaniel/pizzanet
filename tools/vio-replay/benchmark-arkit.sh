#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$script_dir/../.." && pwd)"
recordings_root="${1:-"$repo_root/datasets/ar-recordings"}"
output_dir="${2:-"$repo_root/target/vio-arkit-benchmark"}"

mkdir -p "$output_dir"
cargo build --release -p vio-replay --manifest-path "$repo_root/Cargo.toml"

for recording in "$recordings_root"/*; do
  [[ -d "$recording" ]] || continue
  recording_id="${recording##*/}"
  "$repo_root/target/release/vio-replay" recording "$recording" \
    --frames "$recording/tracker-luma.gray" \
    --trace-output "$output_dir/$recording_id.ndjson" \
    --output "$output_dir/$recording_id.json"
done

git_revision="$(git -C "$repo_root" rev-parse HEAD)"
if [[ -z "$(git -C "$repo_root" status --porcelain)" ]]; then
  git_dirty=false
else
  git_dirty=true
fi

jq -s \
  --arg gitRevision "$git_revision" \
  --argjson gitDirty "$git_dirty" \
  '
  def mean: if length == 0 then null else add / length end;
  def rms: if length == 0 then null else (map(. * .) | add / length | sqrt) end;
  def summary:
    {
      sessions: length,
      meanPositionRmseMetres: (map(.arkitComparison.positionRmseMetres) | mean),
      meanOneSecondTranslationRmseMetres:
        (map(.arkitComparison.oneSecondTranslationRmseMetres) | mean),
      meanFrameDeltaRmseMetres: (map(.arkitComparison.frameDeltaRmseMetres) | mean),
      meanOrientationRmseDegrees: (map(.arkitComparison.orientationRmseDegrees) | mean),
      meanTrajectoryScaleRatio: (map(.arkitComparison.leastSquaresScaleRatio) | mean),
      meanAbsoluteTrajectoryScaleError:
        (map((.arkitComparison.leastSquaresScaleRatio - 1) | abs) | mean),
      rmsTrajectoryScaleError:
        (map(.arkitComparison.leastSquaresScaleRatio - 1) | rms),
      upperMiddleOneSecondLocalScale:
        (map(.arkitComparison.oneSecondScaleMedian) | sort | .[length / 2]),
      metricScaleInitializedSessions: (map(select(.metricScaleInitialized)) | length),
      meanMetricScaleInitializationFrame:
        (map(select(.metricScaleInitializationFrame != null)
          | .metricScaleInitializationFrame) | mean)
    };

  [ .[] | select(.arkitComparison != null) ] as $scored
  | [ $scored[] | select(.arkitComparison.arkitMaximumDisplacementMetres >= 0.05) ] as $motion
  | [ $scored[] | select(.arkitComparison.arkitMaximumDisplacementMetres < 0.05) ] as $stationary
  | {
      schemaVersion: 1,
      reference: "ARKit",
      alignment: "first-pose rigid SE(3), no fitted scale",
      stationaryThresholdMetres: 0.05,
      gitRevision: $gitRevision,
      gitDirty: $gitDirty,
      replayConfiguration: {
        sensorDelayMilliseconds: ($scored | map(.sensorDelayMilliseconds) | unique),
        visualOrientationDelayMilliseconds:
          ($scored | map(.visualOrientationDelayMilliseconds) | unique),
        longAxisFieldOfViewDegrees:
          ($scored | map(.longAxisFieldOfViewDegrees) | unique)
      },
      allScoredSessions: ($scored | length),
      skippedLeadingFramesBeforeSensors:
        ($scored | map(.skippedLeadingFramesBeforeSensors) | add),
      motion: ($motion | summary),
      initializedMotion: ($motion | map(select(.metricScaleInitialized)) | summary),
      uninitializedMotion: ($motion | map(select(.metricScaleInitialized | not)) | summary),
      excludedStationarySessions:
        ($stationary | map({
          recordingId,
          arkitMaximumDisplacementMetres:
            .arkitComparison.arkitMaximumDisplacementMetres,
          estimatorMaximumDisplacementMetres:
            .arkitComparison.estimatorMaximumDisplacementMetres,
          positionRmseMetres: .arkitComparison.positionRmseMetres,
          frameDeltaRmseMetres: .arkitComparison.frameDeltaRmseMetres,
          metricScaleInitialized
        }))
    }
  ' "$output_dir"/*.json > "$output_dir/aggregate.json"

jq . "$output_dir/aggregate.json"
