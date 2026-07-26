# ARKit-reference VIO tuning — 2026-07-24

This iteration uses the 11 bundles in `datasets/ar-recordings/` as synchronized
raw-input/ARKit-reference pairs. The Rust tracker consumes only:

- `tracker-luma.gray`;
- raw frame timestamps and dimensions from `tracker-frames.ndjson`;
- synthesized browser-shaped CoreMotion events from `sensor-events.ndjson`;
- camera calibration from `manifest.json`.

`arkit-poses.ndjson` is never passed to the tracker. It is used only after
replay to score and tune the result.

## 2026-07-27 presentation-pose jitter correction

The live fallback mixed two different camera times in one published pose.
Every luma callback assigned DeviceOrientation sampled at
`frame_timestamp - 40 ms`, then the next sensor callback overwrote that
quaternion with the newest attitude. At 30 Hz camera and roughly 60 Hz
orientation/rendering, steady pans therefore produced a repeated
old-frame/current-sensor sawtooth even while image-space LK points moved
smoothly.

Presentation and estimator state are now separate:

- sensor callbacks update only the latest propagation attitude;
- each accepted camera frame latches one frame-aligned presentation attitude,
  which remains unchanged until the next displayed frame;
- the default camera/orientation offset is 0 ms because literal browser events
  and camera timestamps share the performance timeline;
- non-zero delay remains a measured diagnostic override, not a delivery-latency
  assumption;
- a constant-velocity alpha-beta observer filters only published translation.
  The SLAM position, map, visual corrections, and keyframe decisions remain
  unsmoothed.

Across the 10 moving ARKit-reference sessions, the presentation observer
reduces mean frame-delta RMSE from 9.696 mm to 8.072 mm (16.8%). Mean
rigid-aligned position RMSE changes by only 0.14 mm
(0.416646 m to 0.416789 m), and one-second translation RMSE changes by
0.82 mm (0.135317 m to 0.136136 m). On the literal browser recording from
2026-07-26, median/p95 position second difference falls from
1.84/16.84 mm to 1.01/9.44 mm while endpoint displacement is preserved
(24.64 mm versus 24.99 mm).

## 2026-07-25 startup and WKWebView correction

The 5/10 late metric-scale publication policy below was rejected after the
device check. It made the cube unavailable for 6.58–9.06 seconds and the
accepted scale observations were not reliable enough to rebase a visible
world. Production behavior is now continuity-first:

- world content is visible from the first usable pose;
- healthy visual 6DoF no longer depends on metric-scale certification;
- the closed-form scale estimate remains diagnostic, but production WASM never
  applies a late whole-map rescale;
- `AR_APPLY_EXPERIMENTAL_SCALE=1` restores the old behavior for native replay
  ablations only.

With the fixed 3.5 m provisional gauge, all 10 motion runs have immediate world
availability. Relative to the late-rescale policy, mean frame-delta error
improves from 11.55 mm to 9.70 mm, while rigid-aligned ATE changes from 0.365 m
to 0.417 m and scale MAE from 14.4% to 20.5%. The latter is the honest cost of
not silently resizing existing content. Sweeping fixed initial depths from
2–5 m found no dominant constant: the best scale MAE was 20.5% at 3.5 m, while
individual-session scale remained widely variable.

This does not make frame-one scale metrically observable. A single monocular
frame plus a stationary IMU has no distance baseline. A separate short
stationary-start bootstrap experiment reached 9/10 sessions at a median
2.37 seconds and 13.9% in-sample scale MAE, but it is not yet enabled: it needs
held-out validation using literal WKWebView events and must never rescale an
already-visible anchor.

The native capture path now records those literal inputs:

- the page requests WebKit's combined motion/orientation permission once,
  directly from the `Start AR` click;
- Swift returns WebKit's normal `.prompt` decision only for the expected HTTPS
  top-level origin;
- actual `devicemotion` and `deviceorientation` values, page receipt clocks,
  compass diagnostics, and exact native-frame page-clock mappings are batched
  through `WKScriptMessageHandler`;
- `sensor-events.ndjson` is now the WKWebView stream;
  `coremotion-events.ndjson` is a separate diagnostic sidecar;
- `wk-frame-timing.ndjson` lets replay use the same page-clock frame timestamps
  and frame drops as the live tracker;
- the native Record button remains disabled until both valid browser event
  streams have arrived.

Old bundles remain valid and replay through their synthesized CoreMotion
contract. Fresh bundles automatically use the WK timing sidecar. The new Swift
path still requires an Xcode/iPhone validation because this workspace is
Linux-only.

## Recording contract audit

All 11 bundles have `source: "native-arkit"`. They contain 6,503 luma frames
and matching ARKit poses, 10,799 CoreMotion samples (each serialized as motion
plus orientation), 216.7 seconds, and 635.6 MiB of luma. Every frame timestamp
matches its ARKit pose timestamp exactly and every ARKit pose reports normal
tracking.

These are not literal WKWebView captures. The sensor events are synthesized
from `CMDeviceMotion`; receipt time equals event time. The files do not contain
WK scheduling delay, native-frame JS receipt/mapped timestamps, browser camera
video, or WASM output. Therefore:

- raw/offline replay timing is calibratable from these bundles;
- live native-to-JS bridge latency is not, and remains a separate app setting;
- future recordings should add browser receipt/mapping sidecars before live
  bridge-delay tuning.

Important contract fixes:

- Actual CoreMotion cadence is 20.0688 ms (49.83 Hz), despite the old recorded
  16.667 ms nominal interval. Valid monotonic event timestamps are now
  preserved instead of warped toward the nominal interval.
- Replay uses 0 ms visual-orientation delay. ARKit-reference frame-to-frame
  orientation RMSE is 0.054 degrees at 0 ms versus 0.218 degrees at 40 ms.
- Native camera FOV is 73.16–73.89 degrees, not the old 68-degree default.
  Per-frame FOV now crosses the native bridge; new bundles also preserve
  per-frame `fx`, `fy`, `cx`, `cy`, source dimensions, and derived FOV.
- The portrait-luma comparison transform is
  `q_arkit_portrait = q_arkit_raw * RotZ(+90 degrees)`.
- Replay skips the 21 total leading luma frames that predate the first complete
  motion/orientation pair. New native recordings retain every post-tap camera
  frame; replay uses the WK timing sidecar and skips only frames before its
  first complete browser sensor pair.
- Replay now rejects duplicate frame IDs, non-monotonic sidecars, unequal
  tracker/ARKit cardinality, and frame timestamp mismatches before scoring.
- Native bridge frame IDs remain monotonic across Record taps.
- The added FOV bridge argument is trailing and optional, so either the native
  host or hosted page can be deployed first without breaking the old luma
  callback contract.
- The test app's build settings are locked to portrait, matching the 90-degree
  luma transform and sensor-axis contract instead of silently allowing
  landscape configurations.
- The recorder writes measured motion intervals and uses a full source-footprint
  area average instead of an aliasing 2x2 sample during roughly 4.5x luma
  reduction. The heavier conversion now runs outside the shared recording lock
  so it cannot block CoreMotion serialization.

## Estimator changes

- Added robust bounded 6DoF per-frame refinement over mapped 3D landmarks.
  DeviceOrientation remains the published smooth attitude; visual rotation is
  an internal nuisance variable so small attitude residuals do not become
  false translation.
- Accelerometer bias initialization now accepts genuinely stationary samples
  only (`|a| < 0.2 m/s²`, `|gyro| < 0.08 rad/s`) instead of absorbing normal
  startup movement.
- Corrected landmark parallax bookkeeping to use measured anchor/observer rays,
  independent of the current depth prior.
- Preserved a lightweight 128-keyframe scale-calibration history separately
  from the 24 image-heavy map keyframes.
- Metric scale now requires both inertial excitation and visual/inertial motion
  correlation. Three consistent estimates are required. This rejects calm
  noise instead of guessing scale.
- The one-time metric bootstrap scales the complete state about the session
  origin. The previous current-camera pivot changed map depth without fixing
  the already-travelled camera trajectory.
- Preintegration factors are weighted 4x relative to the prior configuration.
  Full-data sweeps of 1x, 2x, 4x, 6x, and 8x selected 4x as the best balance of
  global scale, relative error, and jitter.
- Bundle adjustment candidates are transactional: cost must not regress and
  the newest-keyframe correction must remain bounded before any optimized
  position, velocity, or inverse depth is committed.
- BA/relocalization corrections receive an 80 ms correction-only presentation
  offset. Ordinary camera motion is not low-pass filtered. Whole-pose Three.js
  smoothing is off by default.
- Evicted landmark IDs are removed from live feature tracks, preventing stale
  "anchored" tracks that the pose solver could not actually use.
- Reordered IMU events are rejected rather than assigned manufactured future
  timestamps; only genuine duplicate timestamps use the cadence fallback.
- Only observations that entered the bounded pose solve can accrue an outlier
  streak. Strong landmarks outside the 128-point cap are no longer penalized.
- Native FOV is frozen from the first calibrated frame because the current map
  has one session-wide intrinsics model; exact per-frame values remain recorded
  for a future variable-intrinsics implementation.
- All metric world content (cube, grid, and origin marker) remains hidden until
  excitation-gated metric scale is initialized, so no world object is visible
  through the one-time gauge change. After about 300 frames the UI reports an
  explicit unobservable-scale state and asks for firmer calibration motion.

## ARKit comparison

The position comparison uses one rigid world alignment from the first valid
pose and preserves metric scale; it does not similarity-scale the estimate.
The 0.918-second session with only 4 mm of ARKit movement is excluded from the
10-session motion aggregate. Both columns use the audited manifest FOV,
portrait-camera transform, and same ARKit scorer. “Branch start” is the full
starting pipeline: it retains the old 40 ms orientation delay, nominal-interval
timestamp warping, and leading pre-sensor frames. The table is therefore the
overall raw-replay pipeline improvement, not an isolated estimator ablation.

| Metric (equal-session mean, 10 motion sessions) | Branch-start pipeline | Tuned pipeline |
|---|---:|---:|
| Mean rigid-aligned position RMSE | 0.792 m | 0.365 m |
| Mean 1-second relative translation RMSE | 0.219 m | 0.134 m |
| Mean frame-delta RMSE | 15.1 mm | 11.5 mm |
| Mean trajectory scale, estimator / ARKit (target 1.0) | 1.904 | 1.028 |
| Mean absolute trajectory-scale error | 96.1% | 14.4% |
| RMS trajectory-scale error | 134.1% | 20.3% |
| Upper-middle session 1-second local scale (target 1.0) | 1.561 | 0.765 |
| Mean orientation RMSE | 2.49 degrees | 2.04 degrees |

This is a 54% reduction in position RMSE, 39% reduction in one-second relative
error, and 23% reduction in frame-step error. The signed mean scale bias is
2.8% high, but session errors do not cancel in the accuracy figure: mean
absolute trajectory-scale error falls from 96.1% to 14.4% (20.3% RMS, tuned
range 0.717–1.447). Five of ten motion sessions contain enough high-correlation
acceleration excitation to certify and publish metric scale; the others
intentionally withhold metric world content rather than displaying a cube at
an untrusted size.

The five scale-initialized motion sessions reach metric scale in 6.58–9.06
seconds (7.87 seconds mean). Their mean ATE is 0.519 m and scale MAE is 12.7%;
the five withheld sessions have 0.210 m mean ATE and 16.2% scale MAE. These
groups represent different motion difficulty, so the split is operational
coverage—not a causal comparison. The excluded 0.918-second stationary run
moves 4.7 mm in ARKit; the estimator has 14.6 mm position RMSE and 29.7 mm
maximum displacement, while correctly refusing to initialize scale.

These are tuning-set results: the same 11 recordings guided the fixed 4x
preintegration weight and scale gates. There is no runtime ARKit-pose leakage,
but the improvement is not an independent generalization estimate. Validate
the defaults on fresh held-out sessions, ideally from another device, before
treating the percentages as production expectations.

The weakest remaining run is the longest 34.6-second path (1.09 m RMSE).
Long-horizon drift and session-to-session local scale variation remain the
main estimator limitations. DeviceOrientation is excellent frame-to-frame but
can yaw-drift by up to 6.6 degrees over 39 seconds; a future visual yaw factor
should correct that slowly without replacing the smooth IMU attitude.

## Reproduction

Build once:

```bash
cargo build --release -p vio-replay
```

Replay one bundle and write the ARKit-comparison report plus per-frame trace:

```bash
recording=datasets/ar-recordings/<recording-id>
target/release/vio-replay recording "$recording" \
  --frames "$recording/tracker-luma.gray" \
  --trace-output /tmp/vio-trace.ndjson \
  --output /tmp/vio-report.json
```

The report includes rigid-aligned ATE, orientation error, endpoint error,
path/max-displacement ratios, frame-delta error, one-second relative error and
scale distribution, scale-initialization frame, and tracker coverage.

Replay and aggregate all bundles deterministically:

```bash
tools/vio-replay/benchmark-arkit.sh \
  datasets/ar-recordings \
  target/vio-arkit-benchmark
```

This writes per-session reports/traces and `aggregate.json`, including the
stationary exclusion, initialized/uninitialized split, replay configuration,
Git revision, and dirty-worktree flag. The saved branch-start comparison above
is from revision `cb166d6759b9df9a4972bc09d2e434325cd1d188`; it is retained
here as a historical pipeline snapshot rather than claimed as a separately
reproducible estimator ablation.

Validation used:

```bash
cargo test -p ar-tracker-wasm -p vio-replay
cargo clippy -p ar-tracker-wasm -p vio-replay --all-targets -- -D warnings
cargo check --target wasm32-unknown-unknown -p ar-tracker-wasm
cd app
pnpm lint
pnpm build
```

The Swift capture changes were reviewed but not compiled in this Linux
workspace; they still require an Xcode/iPhone build before deployment.
