# Closed-loop AR recording analysis — 2026-07-21

The exact-luma iPhone captures provide four structurally valid replay bundles. The three substantive
closed loops last 15.1 s, 14.6 s, and 19.4 s and contain 427 tracker frames plus 5,888 sensor events.
A fourth 6.7 s capture is retained as a short auxiliary loop. Every luma file is exactly
`frame_count * 160 * 284` bytes, frame identifiers and timestamps are monotonic, and the one-blob
MP4 files are valid. Replay uses `tracker-luma.gray`, not decoded video timestamps.

The estimator does not use the lamp's dimensions or a detected lamp coordinate. Its supervised
constraints are:

- the camera begins and ends at the same position and orientation;
- the trajectory must not collapse to zero motion;
- the roughly 2.5 m reported maximum excursion is a plausibility bound, not exact scale;
- start/end appearance correlation and raw orientation closure independently verify the loop.

The three substantive captures have start/end image correlations of 0.879–0.924 and raw orientation
closure errors of 0.61–3.52 degrees. This makes endpoint closure a useful metric even though the
start/end close-up is too texture-poor for ordinary corner tracking.

## What the recordings exposed

- Estimator state was low-pass filtered before keyframe promotion, then filtered again for Three.js
  rendering. Promotions permanently stored the lagged pose, causing the weak forward response. The
  tracker now keeps an unsmoothed estimator pose and leaves smoothing to presentation only.
- Safari reports a stable 16.667 ms motion cadence but 4.5–12.3% of event timestamps are duplicated.
  Motion timestamps are now regularized from the reported cadence and the sensor buffer is drained
  for every camera interval instead of overflowing after roughly 8.5 s.
- Recorded iPhone gyro axes correlate with camera angular velocity as `[alpha, beta, gamma]`, not the
  previous `[beta, gamma, alpha]` permutation. Acceleration retains the verified Apple sign flip.
- Absolute orientation is now interpolated at camera time. Exact-luma sweeps selected a 40 ms visual
  timing offset; the former 100 ms value came from the old fragmented-video approximation.
- Motion samples now propagate position and velocity using bias-corrected specific force, gravity,
  damping, stationary zero-velocity updates, and quality-weighted visual velocity corrections.
- Raw 5x5 SSD patches were too exposure-sensitive. The visual front end now uses 7x7 zero-mean
  normalized correlation, forward/backward checks, deterministic translation RANSAC, spatial
  coverage checks, and lower-contrast full-frame feature selection.
- Textureless frames can no longer replace the active visual keyframe. Reliable origin keyframes are
  pinned, and old views are queried during successful tracking as well as after failures.
- A bounded, exposure-normalized appearance map proposes candidates for the low-texture return
  close-up. A proposal must have similar orientation, at least 45 frames of temporal separation,
  high coarse correlation, and dense rotation-compensated photometric agreement. Textured matches
  also require spatial support; exceptionally smooth views require confirmation on consecutive
  frames. The correction estimates an image-space residual rather than snapping exactly to a
  stored pose. On these captures accepted candidates occur only during the final return. Geometric
  feature matching remains the preferred path for textured views.
- The fixed-depth monocular scale prior is now 3.6 m. Vertical visual and inertial motion use a weaker
  handheld prior because gravity-removal error otherwise dominated height. Metric scale remains
  approximate by design.

## Deterministic replay result

The baseline below is the fresh-state exact replay before this iteration. The optimized result is
the current Rust estimator; no lamp size or lamp reprojection score participates.

| Session | Baseline limited | Optimized limited | Baseline median inliers | Optimized median inliers | Max radius | Endpoint residual | Orientation residual | Closure/radius | Vertical range |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 15.1 s | 71.2% | 35.6% | 2 | 21 | 2.32 m | 0.105 m | 0.74° | 4.5% | 1.03 m |
| 14.6 s | 79.2% | 36.8% | 2 | 27 | 1.85 m | 0.223 m | 3.69° | 12.0% | 1.11 m |
| 19.4 s | 75.3% | 28.2% | 2 | 25 | 2.54 m | 0.196 m | 1.65° | 7.7% | 0.81 m |
| 6.7 s auxiliary | 82.8% | 63.8% | 0 | 4 | 1.80 m | 1.802 m | 19.99° | 100.0% | 0.61 m |

The substantive runs therefore track on 63–72% of frames instead of 21–29%, retain non-trivial
1.85–2.54 m prior-scaled excursions, and close within 22.3 cm, 12.1% of maximum radius, and 3.69
degrees. Every accepted return has non-zero dense photometric support; no endpoint is forced to the
stored origin. The short low-texture auxiliary capture does not close in position or orientation
under these stricter rules and is not counted as one of the three substantive results.

These are replay constraints, not a claim of ground-truth trajectory accuracy between endpoints.
The next device check should look for responsive forward motion, reduced cube sliding during normal
movement, and a smooth correction when a previously mapped view is revisited. Longer unseen loops
remain the important hold-out test; sparse inverse-depth landmarks and a pose graph are still future
work.

Run structural analysis with:

```bash
cd app
pnpm recordings:analyze
```

Run deterministic native replay, optionally writing a per-frame NDJSON trace, with:

```bash
cargo run --release -p vio-replay -- recording datasets/ar-recordings/<id> \
  --frames datasets/ar-recordings/<id>/tracker-luma.gray \
  --trace-output /tmp/ar-trace.ndjson
```

## Lucas-Kanade (`optical-flow-lk`) evaluation — 2026-07-23

The `optical-flow-lk` crate (source vendored at `references/lucas-shi-rust`) was integrated and
evaluated against all four recordings via deterministic replay. Three architectures were measured:

1. **Full pyramidal LK replacing ZNCC** (Shi-Tomasi corners, forward-backward check, prediction
   seeding, exposure normalization variants). At best on par with the ZNCC matcher; the aggregate
   loop-closure ratio was equal or slightly worse in every parameter configuration swept (pyramid
   levels 2–4, windows 9–21, iterations 25–50). Keyframe-to-frame matching spans auto-exposure
   swings, where normalized cross-correlation is structurally more robust than LK's
   brightness-constancy assumption; global exposure pre-matching helped some sessions and hurt
   others (scene-content-driven statistics).
2. **ZNCC + single-level LK sub-pixel refinement** seeded at each integer ZNCC match. No measurable
   improvement in endpoint closure, and no measurable reduction in rendered-position jitter
   (median second-difference ~55–64 mm both ways — the inertial blend dominates, so ±0.5 px
   measurement quantization is not the limiting noise source). Removed.
3. **Shi-Tomasi grid detection only** (`good_features_to_track_grid`, 24 px cells, 2 corners per
   cell, quality 0.03, 6 px spacing, 5.5 px border) feeding the existing ZNCC matcher. This is the
   adopted configuration.

Adopted result versus the previous min-gradient corner picker:

| Session | Limited before → after | Median matches/inliers before → after | Endpoint residual before → after | Closure/radius before → after |
|---|---|---|---|---|
| 15.1 s | 35.6% → 37.1% | 31/21 → 37/22 | 0.105 m → 0.105 m | 4.5% → 4.4% |
| 14.6 s | 36.8% → 36.8% | 37/27 → 39/21 | 0.223 m → 0.297 m | 12.0% → 9.6% |
| 19.4 s | 28.2% → 27.6% | 39/25 → 39/24 | 0.196 m → 0.196 m | 7.7% → 8.9% |
| 6.7 s auxiliary | 63.8% → 58.6% | 9/4 → 21/6 | 1.802 m → 0.548 m | 100.0% → 25.5% |

The three substantive loops hold parity (worst-case closure ratio improves 12.0% → 9.6%), and the
previously failing low-texture auxiliary capture improves 3.3× in endpoint residual because
Shi-Tomasi finds more than twice as many trackable corners there. Detector parameters matter more
than the matcher: coarse cells with per-cell budgets >2 cluster features and bias the
single-depth-prior translation solve; very fine cells (16 px) flipped which session regressed.
All sweeps live in the session scratchpad workflow; re-run any variant with the replay command
above.

## SLAM rewrite replay status — 2026-07-23

The tracker was rewritten around continuous LK tracks, per-landmark inverse
depth, and arael-based per-frame/window optimization (see `SLAM_REWRITE.md`).
Replaying these 10 Hz/160px recordings through the new pipeline stays bounded
(net endpoint 0.4–1.6 m, vertical range ~1 m) but spends 45–97% of frames in
limited state versus 28–64% for the old tracker: frame-to-frame optical flow at
10 Hz sees ~3× the inter-frame motion the new front-end is designed for
(240px @ 30 Hz). These recordings remain a divergence/regression guard, not a
tuning target — capture new sessions at the new settings for that.
