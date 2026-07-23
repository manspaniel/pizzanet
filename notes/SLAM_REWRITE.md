# SLAM tracker rewrite — 2026-07-23

One-shot replacement of the approximate keyframe/ZNCC tracker with a
visual-primary SLAM/VIO architecture. Crate layout: `crates/ar-tracker-wasm/src/`
split into `frontend.rs` (LK tracking), `map.rs` (keyframes + landmarks),
`estimator.rs` (arael solvers), `reloc.rs` (appearance relocalization, ported),
`geometry.rs`, `lib.rs` (orchestration + wasm API).

## Architecture

- **Orientation**: iOS/W3C device-orientation fusion, yaw-recentered. Never
  optimized visually (better short-term than anything recoverable from these
  frames; removes SO(3) params and the rotation gauge from every solve).
- **Front-end** (`optical-flow-lk`): continuous frame-to-frame pyramidal LK
  tracks, gyro-seeded, forward-backward-culled, grid-Shi-Tomasi top-up every
  frame. Plus **keyframe re-acquisition**: lost landmarks are re-tracked from
  the stored keyframe image directly into the current frame (large-displacement
  LK with geometric seeding) — this is what survives fast-motion track
  collapse.
- **Map**: keyframes (position + velocity + fixed IMU orientation + pixel
  observations + luma for reloc/reacquisition) and **anchored inverse-depth
  landmarks** — each landmark owns its depth; no global scene-depth constant.
  Depths start at a 3.5 m prior and individualize via bundle adjustment
  ("prior + converge silently").
- **Estimator** (`arael` 0.7, f32, symbolic differentiation):
  - Per frame: 3-DOF camera-position refinement against the landmark map
    (rotation fixed from IMU), soft prior toward the inertial prediction,
    adaptive median-based outlier trimming, one trimmed re-solve. Solutions are
    **rejected** (not clamped) when the correction exceeds
    `1.2 m/s × frame-interval` unless inlier consensus is strong — a yaw
    orientation error during fast rotation is geometrically indistinguishable
    from lateral translation at near-uniform depth, and accepting it injects
    drift.
  - Per keyframe: sliding-window (6 keyframes) visual-inertial bundle
    adjustment — keyframe positions/velocities + landmark inverse depths,
    pixel-reprojection factors (TripletBlock when the anchor is in-window,
    CrossBlock when frozen), **world-frame accelerometer preintegration
    factors** between consecutive keyframes (the metric-scale anchor), weak
    priors for gauge/degeneracy protection. Dense LM solve (~220 params;
    arael 0.7's sparse indexed assembly does not cover symbolic TripletBlocks).
- **Relocalization**: the previous iteration's appearance machinery
  (descriptor shortlist + rotation-compensated direct alignment with block and
  edge verification), operating on stored keyframes, scene depth from the live
  map's converged landmarks.

## arael integration notes

- Consumed as a path dependency on `references/arael` (0.7.2 tree, excluded
  from the workspace in the root Cargo.toml): the published 0.7.1 panics on
  wasm32 (`std::time` reached inside the solver); 0.7.2 carries the complete
  wasm clock handling. Compiles clean on the workspace's pinned 1.93.1 (the
  repo's 1.94 pin is only for its own trybuild CI). A `console_error_panic_hook`
  (wasm-only dep) surfaces any future Rust panic in the browser console
  instead of an opaque `unreachable` + poisoned wasm-bindgen borrow flags.
- Constraint-body binding names are the *lowercased struct name* (`camnode`,
  `preintpair`), and remote-block constraints cannot read their own struct
  fields — constants must live on a referenced/containing entity (see
  `ObsDetail` and arael's own `loc_demo`).
- Symbolic `TripletBlock` factors work with `calc_cost`/dense/band solves but
  break `solve_sparse`'s cached-pattern assembly in 0.7.1 — use the dense path
  for the window.
- The Starship robust loss is patented; we use plain residuals + explicit
  trimming instead.

## Validation status

- 14 unit tests including a **design-point simulation**: 6 s at 30 Hz,
  180×240 frames, 260-point 3D scene with 1.8–6.3 m depths, sinusoidal
  translation + yaw. Asserts: tracking state locked, >15 landmarks converge
  depth via parallax, bounded pose error, healthy inliers. Passes.
- Old 10 Hz/160px recordings replay without divergence (bounded, max
  displacement < 3 m, no runaway) but with chaotic per-recording variance
  (endpoint nets 0.3–2.6 m shuffle under any small parameter change — keyframe
  decisions cascade) and more limited-state frames than the old tracker —
  **expected**: frame-to-frame LK at 10 Hz has 3× the inter-frame motion the
  design targets, and the old exhaustive matcher remains better in exactly that
  regime. The capture pipeline now feeds 240px @ 30 Hz; fresh recordings at
  those settings are the real evaluation and next tuning target.
- Native-only ablation env vars for replay experiments: `AR_DISABLE_BA`,
  `AR_DISABLE_PREINT`, `AR_DEBUG_FRAME` (per-frame residual distribution).

## Known limitations / next steps

- Scale converges only with translation; before convergence the depth prior
  sets the magnitude (chosen behavior).
- Approximate marginalization (prior on the oldest window keyframe) instead of
  true Schur marginalization of departed states.
- Preintegration weights are conservative; tune against 30 Hz recordings.
- The old recordings' regression numbers live in `AR_RECORDING_ANALYSIS.md`;
  re-baseline once new recordings exist.

## First 30 Hz recordings — findings (2026-07-23, evening)

Three 240px@30Hz sessions (2× closed-loop). Fixes landed from them:
- **Late motion samples**: iOS delivers devicemotion in delayed bursts; at
  30 Hz frame draining they can arrive behind the drain watermark. Now
  tolerated as counted drops (`late_motion_sample_count`) instead of hard
  rejection (which aborted replays and silently poisoned on-device runs).
- **Relocalization storm**: frame-count pacing constants assumed 10 Hz; at
  30 Hz appearance relocalization fired every ~0.6 s against 1.5-second-old
  keyframes, snapping the pose constantly. Now a recovery-only tool: runs only
  during visual outages, only against keyframes ≥12 keyframes old, 45-frame
  cooldown. Result: limited fraction 0.2–12% (median inliers 65–86), zero
  spurious relocs.
- **Open problem — scale**: trajectories replay ~3–4× too large and breathe
  (the on-device "cube changes size"). Diagnosis: reprojection factors are
  scale-free, the velocity states let preintegration absorb any scale during
  smooth motion, so the inverse-depth priors decide the gauge — and the true
  scene is closer than the 3.5 m prior. Releasing converged-depth priors and
  strengthening preint made it worse (accel too weak to hold the gauge alone).
  Next lever: proper VIO scale initialization (closed-form scale from
  up-to-scale visual odometry + integrated accel over an excitation window),
  and/or gravity-aligned known-height constraints. Ground-truth walk distances
  from the operator would calibrate this directly.

## Metric-scale initializer — 2026-07-23 (late)

`src/scale.rs`: closed-form scale estimation. Over the newest contiguous chain
of preintegrated keyframe pairs, `s·(p_{i+1}−p_i) = v_i·dt + Δp_i` with
`v_{i+1} = v_i + Δv_i` is linear in `(s, v_0)` — solved by 4×4 normal
equations, accepted only with ≥5 pairs, ≥1.5 s span, and ≥0.5 m/s summed |Δv|
(scale is unobservable during smooth motion; the estimator refuses rather than
guesses). First confident estimate applies in full (initialization, rescaling
the whole map about the current camera so the view doesn't jump); later ones
as ≤5% bounded maintenance steps. Device-independent by construction — the
accelerometer measures m/s² the same everywhere; nothing is fitted to a
specific phone or room. `map_stats()` gained `[5] scale_ratio`,
`[6] scale_confidence`.

Unit tests: recovers a synthetic 3× scale mismatch to <5%; refuses
constant-velocity motion. On the 2026-07-23 recordings it compresses the
07-24-33 trajectory 2.3× (max displacement 5.1→2.25 m); the two loop sessions
respond less and remain noisy. Blocked on ground truth: a tape-measured
out-and-back recording will validate absolute scale without tuning to the
device.
