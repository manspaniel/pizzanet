# Linux / RTX Handoff

Last updated: 2026-07-22

This is the standalone context for continuing the Retro Pizza Hut Roof AR
project in a fresh Codex session on the Linux/RTX 4070 machine. Read `BRIEF.md`,
this file, and `notes/UPDATED_PLAN.md` before changing the design. Detailed data
and command references live in `notes/TRAINING_AND_INFERENCE.md` and
`notes/SYNTHETIC_TRAINING_DATA.md`.

## Product constraints

- Detect the recognisable two-tier roof on current **and former** Pizza Hut
  buildings. Repainting, missing signs, extensions, and ordinary deterioration
  do not make one a negative.
- Negatives are ordinary houses, shops, offices, and unrelated buildings.
- Recognition must depend mainly on architectural form, not red paint, signs,
  or a Pizza Hut-like background.
- Infer a complete simplified two-tier mesh, including obscured portions. The
  user supplies camera frames, not camera pose or roof parameters.
- Absolute scale and depth-sensor data are not required. Multi-view parallax and
  visual-inertial tracking will later refine relative pose and scale enough for
  a stable overlay.
- The eventual iOS Safari experience must materially use camera **and** motion
  sensors. Still-image work is the recognition/fitting slice, not a substitute
  for SLAM.
- Burn inference, Rust/WASM tracking/fitting, and Three.js rendering stay
  separate and do not need to share a graphics context.

## Current architecture

```text
RGB image
  -> Burn MobileNetV2 feature pyramid
  -> presence logit + 12 amodal keypoint maps + 12 offscreen logits
  -> symmetry-aware keypoint decoder
  -> pinhole perspective fitter (all eight D4 correspondences)
  -> seven scale-free roof ratios + camera + complete projected mesh
  -> roof-detect JSON and PNG overlay
```

| Path | Responsibility |
| --- | --- |
| `crates/roof-geometry/` | Stable semantic IDs and complete parametric two-tier mesh |
| `crates/roof-fit/` | Robust single-view pinhole fitting and fit rejection |
| `crates/roof-model/` | Portable Burn network, preprocessing, and decoding |
| `crates/roof-training/` | Gaussian keypoint targets and D4-invariant supervision |
| `crates/synth-data/` | Deterministic scene/dataset records and validation |
| `crates/synth-render/` | Native headless WGPU RGB and aligned-label rendering |
| `tools/roof-synth/` | Generate, shard, validate, and preview synthetic data |
| `tools/roof-data/` | Source and curate real positives/negatives |
| `tools/roof-train/` | Balanced training, metrics, fitting evaluation, promotion |
| `tools/roof-detect/` | Image-only detection, complete mesh fitting, PNG/JSON output |
| `crates/vio-core/` | Sensor records and IMU preintegration only; not complete SLAM |

The React application now has a first Three.js AR vertical slice. Supported Android browsers use
WebXR `immersive-ar`/`local-floor` tracking and optional hit tests; other browsers use rear-camera
capture, motion permissions, and the new `ar-tracker-wasm` bundle. The WASM fallback accepts
timestamped luma frames and normalized IMU observations. It provides recenterable gravity-aligned
orientation, frame-interval inertial position/velocity propagation, stationary zero-velocity
updates, and rotation-compensated sparse visual odometry. The visual front end uses
brightness-normalized patch matching, deterministic translation RANSAC, spatial coverage,
quality-gated keyframes, and bounded geometric plus coarse-appearance relocalization. Four
exact-luma iPhone loops exposed and now cover portrait-axis calibration, iOS gyro axes and duplicate
motion timestamps, interpolated camera/orientation timing, non-collapsing scale, and loop closure.
Across the three substantive replays the estimator tracks on 63–72% of frames, reaches a
prior-scaled 1.85–2.54 m from the origin, and has replay-estimated closure residuals below 22.3 cm,
12.1% of maximum radius, and 3.69 degrees without using the lamp's size. Appearance only proposes
return candidates; dense rotation-compensated photometric evidence verifies and estimates each
accepted correction, so endpoints are not snapped exactly to the origin. The comparison is in
`notes/AR_RECORDING_ANALYSIS.md`. Sparse inverse-depth landmarks, a pose graph, independently
calibrated metric scale, live Burn inference, multi-view roof fitting, and estimator-driven
guidance remain unimplemented.

## Corrected Still-Image Roof Detection Plan

This section folds in the active plan called **Corrected Still-Image Roof
Detection Plan**. It supersedes the rejected global-pooling and 19-channel
spatial models.

### Required model and loss

- Keep a 256×256 letterboxed input and ImageNet-pretrained MobileNetV2 FPN.
- Predict one global roof-presence logit.
- Predict twelve 64×64 amodal keypoint maps: four eave, four shoulder, and four
  crown corners, plus one offscreen logit per point.
- Treat each map's 4,096 cells plus offscreen as one categorical distribution.
  Train an in-frame point against a small Gaussian and an unavailable point
  against the offscreen token.
- Minimise over all eight cyclic/reflected rectangular correspondences. Apply
  the selected permutation consistently to all three rings.
- Use source-balanced BCE as the complete presence objective. The v14
  real-domain pairwise experiment is retained only for reproducibility; v15
  sets its weight to `0.0`. Geometry loss applies only to synthetic target
  roofs, with weight `0.5` while training jointly.
- Train MobileNetV2 from epoch 1 at `1e-5` and the FPN/heads at `3e-4`. Both
  schedules use 5% warm-up followed by cosine decay across the original
  40-epoch horizon; neither schedule resets at a phase transition.
- Keep the imported MobileNetV2 BatchNorm running mean and variance fixed
  during training. Detach every feature-pyramid tensor before the FPN so
  keypoint and offscreen losses update the decoder/heads but not MobileNetV2;
  the presence branch retains the original differentiable stride-32 tensor and
  is the only source of MobileNetV2 gradients. Backbone convolutions and
  BatchNorm affine parameters therefore remain presence-trainable, while the
  newly initialized FPN BatchNorm layers retain normal adaptive training
  behavior. Once two consecutive validations pass both the real presence gate
  (0.80 view recall, 0.80 physical-building recall, and 0.85 real-negative
  specificity) and the synthetic AUC/AP gate, lock the presence solution. The
  next and all later epochs freeze MobileNetV2 and the presence heads and
  optimize only the geometry heads from `0.5 * geometry_loss`. This lock is
  sticky, resets early-stopping patience once when entered, and subsequent
  geometry-only epochs consume patience normally. Use AdamW with `1e-4` weight
  decay, batch size 32, at most 40 epochs, and ten-epoch early stopping.

Do **not** add a centre/extent head, bounding-box head, mask head, edge head,
visibility head, or direct shape-parameter head. Keypoints locate the roof; the
perspective fitter returns its shape. Synthetic masks remain validation data.
Real photos need presence labels, not manually supplied pose, intrinsics, roof
parameters, or localisation labels.

### Perspective fitting

- Fix eave width to one relative unit and fit camera rotation, translation,
  focal length, and seven bounded scale-free ratios.
- Use EXIF focal information when available; otherwise initialise from 45, 60,
  and 75 degree field-of-view hypotheses.
- Use robust Levenberg-Marquardt residuals, a population shape prior, and all
  eight correspondence hypotheses.
- Return inferred ratios, camera, projected mesh, bounds, reprojection error,
  and confidence.
- Require at least four accepted observations. Four- and five-point fits must
  span all three rings and at least two cyclic corner slots; denser fits retain
  the normal robust coverage path. Reject a confident fit when normalised RMSE
  exceeds 0.05, confidence-weighted inliers fall below two thirds, the complete
  mesh extrapolates implausibly far beyond the observations, or the combined
  fit score is below 0.25.

Exact-annotation fitter tests already satisfy projected-vertex RMSE below 0.5%
of the image diagonal and silhouette IoU above 0.95 with known intrinsics.

### Data roles

| Data | Role |
| --- | --- |
| Synthetic two-tier roofs | Positive presence and complete geometry supervision |
| Synthetic ordinary roofs | Negative presence only |
| Visibility-eligible Wikimedia current/former Pizza Huts | Positive presence only |
| Curated Open Images ordinary buildings | Negative presence only |
| `samples/` | Untouched qualitative evaluation; never training or threshold selection |

The retained `datasets/synthetic-training-keypoints/` corpus is the historical
v1 render: 6,000 independent targets and 6,000 independent ordinary-building
negatives, one view per building, split into 9,619 train, 1,152 validation, and
1,229 test frames in 48 shards. It predates the current five-profile generator
and must not be reused for the v2 full run.

The completed `datasets/synthetic-training-keypoints-v2/` corpus contains
12,000 one-frame buildings (12,000 frames) in 48 shards and 54,000 aligned
artifacts. Its hash-assigned split is 9,547 train, 1,226 validation, and 1,227
test frames. Coverage reaches all 75/75 required positive cells across five roof
morphologies, three day phases, and five scene domains. Each class is balanced
exactly overall at 2,000 urban, 2,000 suburban, and 2,000 remote scenes. The
negative set has 858 flat roofs and 857 each of gable, hip, shed, mansard,
pyramid, and ordinary cupola roofs. `roof-synth validate` completed with zero
warnings. A full target-visibility audit found zero
invisible frames, 27 below 25% visible, 474 below 50% visible, 1,195 truncated,
520 full-width, and a median visible fraction of 0.718; these categories may
overlap. The trainer excludes the three train-split targets below its 5%
visible-fraction floor, so the canonical run uses 10,257 training samples
(9,544 eligible synthetic frames, 705 real negatives, and 8 real positives).
Selection permits at most one foreground occluder, never combines
partial framing with an occluder, and the publisher refuses a target frame with
zero visible roof pixels or no visible bounding box.

The curated real data contains 873 accepted Open Images negatives (705/89/79)
and 25 historical-site Wikimedia records across 17 physical buildings. The
`roof-positive-dataset/v2` manifest separately reviews whether the
characteristic two-tier roof is recognizable in each image. Only 18 photos
across 12 buildings are eligible for positive supervision: 8 train images
across 6 buildings, 6 validation images across 3 buildings, and 4 test images
across 3 buildings. The other 7
records are provenance-only, never negatives, and are excluded from training,
calibration, and evaluation. `roof-train` rejects any manifest that puts views
of one physical building in different splits. Each eligible real-positive image
enters the source pool once by default; balanced epoch draws provide fresh
deterministic augmentations without inflating the dataset manifest. Within the
real-positive source slots, the current sampler cycles physical buildings
round-robin before repeating one, and deterministically rotates through the
available views of each building. This prevents a multi-view site from
outweighing a single-view site.

### Promotion gates

A full checkpoint becomes deployable only when its operating threshold is
calibrated from held-out real-camera positives and negatives, and held-out
evaluation reaches:

- validation threshold calibration and presence-checkpoint selection require
  at least 0.80 real-positive view recall, 0.80 physical-building recall, and
  0.85 real-negative specificity;
- final promotion requires at least 0.75 real-positive view recall and 0.80
  physical-building recall on both validation and test, with complete building
  IDs;
- test curated-real-negative specificity at least 0.85;
- test synthetic presence ROC AUC and average precision at least 0.95;
- permutation-matched PCK@5% at least 0.90;
- offscreen accuracy at least 0.90;
- synthetic fit success at least 0.90 and accepted-fit coverage at least 0.80;
- median fitted-mesh RMSE at most 0.08 of the image diagonal;
- median amodal silhouette IoU at least 0.50.

Aggregate and cross-domain same-threshold recall/specificity remain useful
diagnostics, but they do not define the v2 promotion operating point because
synthetic and real-camera probabilities need not share one calibration.

Checkpoint selection uses
`hard_gates_then_bucketed_real_robustness/v1`. It first protects the real
validation recall/specificity gate, then the synthetic ROC AUC/AP gate, and
then moves geometry over its PCK/offscreen gate. Once every hard gate passes,
real-domain robustness outranks sub-band geometry gains. Its score is weighted
from real specificity, ROC AUC, average precision, and recall at
`0.40/0.35/0.20/0.05`; that score and the real gate margin use 200-basis-point
bands, while synthetic gate margin and synthetic ranking quality use
100-basis-point bands. This avoids selecting a nominally sharper single-frame
geometry epoch when its deployment-domain separation has materially regressed.

The untouched `samples/` contact sheet then requires at least 80% detection
recall, plausible placement of both tiers on at least 70%, and no more than 15%
false positives on the curated real-negative test split.

## What has been proven

The exact 32-target + 32-negative memorisation gate passed on ordinary WGPU and
Flex with identical saved weights/observations:

- presence recall/specificity: 1.000 / 1.000;
- PCK@3%: 0.9803371;
- PCK@5%: 0.98595506;
- offscreen accuracy: 1.000;
- duplicated/collapsed pairs: 0.

The retained diagnostic checkpoint is
`artifacts/roof-model-overfit/model.mpk`, with its manifest and metrics beside
it. This proves that the current network, supervision, portable Burn
serialization, decoder, fitter, and CLI work together. It is not a production
detector.

The Linux trainer also has an opt-in Burn CUDA backend (`--features cuda` with
`--backend cuda`). On the RTX 4070 with CUDA 12.0, CUDA evaluation reproduced
the retained WGPU checkpoint metrics exactly, and a fresh CUDA smoke epoch
completed forward/backward execution, AdamW updates, checkpoint save/reload,
and gate evaluation. A separate `--evaluation-batch-size` permits larger
validation batches without changing training batches or learned weights. A
fair throughput comparison must wait until no other trainer is occupying the
GPU.

An explicit CLI/parity run on a memorised synthetic frame produced presence
`1.0000`, ten accepted keypoints, an accepted complete two-tier fit, normalised
RMSE `0.0074`, and fit confidence `0.780`:

```bash
mkdir -p /tmp/pizzahut-overfit-cli
tar -xf datasets/synthetic-overfit-balanced/train-000000.tar \
  -C /tmp/pizzahut-overfit-cli \
  seq-f854ea7661e6efb8-000000.rgb.jpg

cargo run --release -p roof-detect -- \
  /tmp/pizzahut-overfit-cli/seq-f854ea7661e6efb8-000000.rgb.jpg \
  --model artifacts/roof-model-overfit/model \
  --verify-backend-parity \
  --output /tmp/pizzahut-overfit-cli/overlay.png \
  --json /tmp/pizzahut-overfit-cli/prediction.json
```

## Stopped historical v1 full-corpus run

The Mac WGPU run used the historical v1 synthetic/positive contracts and was
stopped at the start of epoch 5 as requested. Epoch 4 had:

| Metric | Value |
| --- | ---: |
| Recall | 1.000 |
| Overall specificity | 0.112 |
| Synthetic-negative specificity | 0.023 |
| Real-negative specificity | 0.685 |
| PCK@5% | 0.859 |
| Offscreen accuracy | 0.988 |
| Duplicated pairs | 33 |

The falling threshold and specificity mean this was not a valid detector. Its
checkpoint and logs were removed after these metrics were recorded here.

- `artifacts/roof-model-keypoints/model.mpk` does not exist.
- `artifacts/roof-model-keypoints/model.json` does not exist.
- The v2 `roof-training/v15` CUDA run completed and promoted
  `artifacts/roof-model-keypoints-v2/model.mpk` plus `model.json`; the exact
  held-out result is recorded below.
- Ordinary `roof-detect image.jpg` now uses that promoted v2 checkpoint by
  default when pointed at `artifacts/roof-model-keypoints-v2/model`.
- The trainer saves model weights, not AdamW/update state, so Linux must restart
  rather than resume epoch 5.

The geometry side improved while the presence decision collapsed. The trainer
now logs source-wise probability quantiles plus overall, synthetic, real, and
cross-domain ROC AUC and average precision. It writes separate
`candidate-presence.mpk` and `candidate-geometry.mpk` diagnostics in addition
to `candidate-best.mpk` and `model-last.mpk`; all are archived before a new run
and none is a promoted model. Under the v2 contract, held-out real-camera recall
and specificity calibrate the operating threshold and rank the presence
checkpoint, while synthetic ROC AUC/AP check threshold-free learned
separation. A capped four-source WGPU smoke run on the RTX 4070 exercised fresh
optimizer updates and confirmed that synthetic targets, synthetic negatives,
real positives, and real negatives are all retained by stratified smoke-test
limits. More visibility-eligible real Pizza Hut photos would still improve
domain grounding without requiring geometry annotations.

The definitive `roof-training/v15` configuration makes the transfer-learning
safeguards explicit and reproducible: `backbone_freeze_epochs` is zero,
`freeze_backbone_batch_norm` preserves imported ImageNet population
statistics, and `detach_geometry_backbone` restricts MobileNetV2 updates to
presence gradients. It also records the physical-building-balanced
real-positive sampling strategy, number of train buildings, disabled pairwise
weight `0.0`, building-and-view threshold policy, two-safe-validation presence
lock, and bucketed checkpoint policy. After the lock, the backbone and presence
heads receive neither gradients nor optimizer updates; only geometry heads are
updated, while the original learning-rate schedules continue. Earlier
diagnostic runs are retained at
`artifacts/roof-model-keypoints-v2-backbone-3e-5/` and
`artifacts/roof-model-keypoints-v2-backbone-1e-5-unfrozen-bn/`, while the
superseded five-epoch frozen-backbone run is retained at
`artifacts/roof-model-keypoints-v2-frozen-bn-freeze5/`.

The immediately preceding no-ranking full diagnostic is preserved at
`artifacts/roof-model-keypoints-v2-v12-no-ranking/`. Its epoch-6 candidate met
all validation gates, but validation real-negative specificity then fell to
`0.730`, `0.730`, and `0.618` at epochs 7, 8, and 9. The run was stopped rather
than spending the remaining epochs on a repeatable real-domain regression. A
frozen copy of epoch 6 was evaluated at
`artifacts/roof-model-keypoints-v2-probe-e06/`: validation reached real recall
`0.833`, real specificity `0.888`, and PCK@5% `0.906`; test reached real recall
`0.750` (3/4 images), real specificity `0.873`, synthetic ROC AUC/AP
`0.975/0.979`, PCK@5% `0.888`, and offscreen accuracy `0.991`. The missed test
positive was the rear Adel view while its front view succeeded, so every one
of the 3/3 held-out buildings still had a detected view. The fitter completed
612/612 roofs, accepted 505, and reported median RMSE `0.018` and IoU `0.790`.
Re-evaluation under the v15 building-aware gate detected all 3/3 test buildings
from 3/4 views and passed the presence and fitter requirements. It remained
correctly unpromoted because PCK@5% was `0.888`, below `0.900`.

The capped two-epoch v14 smoke run at
`artifacts/roof-model-keypoints-v2-v14-pairwise-smoke/` is mechanical evidence
only: it exercised the pairwise loss, new checkpoint ranking/configuration,
CUDA updates, save/reload, and final evaluation on limited splits. Its metrics
are not evidence of detector quality. The corresponding full v14 pairwise run
at `artifacts/roof-model-keypoints-v2-v14-pairwise/` was stopped after epoch 6:
it never passed the real validation gate and underperformed the v12 epoch-6
checkpoint. Pairwise training is therefore disabled in v15. The v15 capped
smoke at `artifacts/roof-model-keypoints-v2-v15-smoke/` is likewise mechanical
evidence only. None of these diagnostic paths is the active artifact directory
or a promoted model.

The definitive v15 joint run reached two consecutive safe validations at
epoch 6. Reloaded validation metrics were real-view recall `0.833`, physical-
building recall `1.000`, real-negative specificity `0.888`, synthetic ROC
AUC/AP `0.971/0.974`, PCK@5% `0.906`, and offscreen accuracy `0.993`. The first
implementation of the epoch-7 geometry-only transition retained an unused
autodiff MobileNet/presence graph and exhausted CUDA memory on its second
batch. That complete failed run, including the large CubeCL panic log, is
preserved at `artifacts/roof-model-keypoints-v2-v15-joint-cuda-oom/`; the safe
epoch-6 weights and provenance are preserved separately at
`artifacts/roof-model-keypoints-v2-v15-e06-safe/`.

Geometry refinement now runs MobileNet on the non-autodiff inner backend,
never evaluates the presence branch, selects only the 12 synthetic-positive
rows before the backbone, and updates only FPN/keypoint/offscreen parameters.
The continuation fails closed unless the checkpoint's sibling
`training-config.json` and final metrics record match the current datasets,
sample counts, hyperparameters, augmentation seed, presence-lock state, and
optimizer-update horizon. It records hashes and the authenticated 401-batch,
2,406-of-16,040 update offset in `refinement-source-metrics.json`. Capped split
runs are diagnostic-only and can no longer create a promoted checkpoint.

The authenticated continuation stopped at logical epoch 33 after ten epochs
without improving the epoch-23 candidate. Epoch 23 validation reached PCK@5%
`0.938` and offscreen accuracy `0.997`, with the frozen presence metrics
unchanged. The uncapped held-out test reached PCK@5% `0.928`, offscreen accuracy
`0.996`, real-view recall `0.750`, all `3/3` physical buildings detected,
real-negative specificity `0.873`, and synthetic ROC AUC/AP `0.975/0.979`.
Perspective fitting succeeded on `612/612` roofs and accepted `544` (`0.889`),
with median mesh RMSE `0.014` and silhouette IoU `0.823`. Promotion passed with
no failures. Independent WGPU and Flex reloads reproduced the detector metrics
and passed the gate. Flex's downstream fitter accepted `545` rather than `544`
and reported IoU `0.822` rather than `0.823`, within expected backend
floating-point variation. Reports are at
`artifacts/roof-model-keypoints-v2-verify-wgpu/` and
`artifacts/roof-model-keypoints-v2-verify-flex/`.

## Linux and dataset transfer

### Clone correctly

The reference checkouts are study-only submodules. `.gitmodules` now records all
seven upstream URLs. After the Mac changes are committed and pushed:

```bash
git clone --recurse-submodules <remote-url> pizzahut
cd pizzahut
git submodule update --init --recursive
```

Before pushing from the Mac, ensure the previously untracked
`references/burn-models/` directory is staged as a Git submodule. Do not build
or edit anything inside `references/`.

### Transfer the ignored datasets

Git alone is not enough. `datasets/` is ignored. The retained historical data
is about 1.9 GiB:

- `datasets/synthetic-training-keypoints/` — about 1.7 GiB, historical v1;
- `datasets/open-images-negatives/` — about 210 MiB;
- `datasets/wikimedia-positives/` — about 4.4 MiB;
- `datasets/synthetic-overfit-balanced/` — about 18 MiB.

Copy those four directories to the same paths on Linux with `rsync`, `scp`, an
external drive, or another non-Git channel. Copying preserves the exact reviewed
pixels and manifests. Transfer the completed
`datasets/synthetic-training-keypoints-v2/` separately; do not replace or
silently relabel the historical corpus.

If copying is impossible, regenerate synthetic data with the commands in
`notes/TRAINING_AND_INFERENCE.md`. Reconstruct real data with:

```bash
cargo run --release -p roof-data -- import-open-images
cargo run --release -p roof-data -- apply-open-images-review \
  --ledger assets/training/open-images-review-ledger.json
cargo run --release -p roof-data -- validate-open-images-review
cargo run --release -p roof-data -- import-wikimedia-positives \
  --category-depth 0 --jobs 1
```

The checked-in ledger is digest-bound to all 1,161 selected Open Images
candidates and preserves the 288 rejections. Upstream availability can still
change, another reason to copy the existing data.

## RTX backend reality

The trainer supports portable `--backend wgpu` and `--backend flex` builds and
an opt-in Burn CUDA build. On Linux, WGPU uses the RTX 4070 through Vulkan;
native CUDA requires the CUDA toolkit and NVRTC, and is invoked with
`cargo run --release -p roof-train --features cuda -- ... --backend cuda`.
Checkpoints remain portable, so evaluate a retained CUDA checkpoint through
WGPU and Flex before trusting it.

Do not enable Burn/CubeCL autotune as a shortcut. It previously produced
numerically divergent WGPU results that did not match ordinary WGPU, Flex, or
`roof-detect`. Benchmark native CUDA and Vulkan separately with exact parity.

## Recommended continuation order

1. Confirm all repository work is committed, clone recursively, and transfer
   the four retained dataset directories plus the completed v2 corpus.
2. Run dataset validation and the 32+32 overfit CLI/parity command above.
3. Inspect the v2 real-positive contract explicitly: 18 eligible photos with
   an 8/6/4 image split, grouped as 6/3/3 physical buildings, plus 7
   provenance-only records excluded from model roles.
4. Confirm the completed five-profile corpus at
   `datasets/synthetic-training-keypoints-v2/` still validates with 75/75
   coverage cells and zero warnings. Do not train from the historical 45-cell
   corpus by accident.
5. The complete CUDA run and its authenticated geometry continuation are now
   finished. The original joint command was:

   ```bash
   target/release/roof-train \
     --synthetic datasets/synthetic-training-keypoints-v2 \
     --negatives datasets/open-images-negatives \
     --real-positives datasets/wikimedia-positives \
     --real-positive-repeat 1 \
     --artifacts artifacts/roof-model-keypoints-v2 \
     --backend cuda --epochs 40 --patience 10 \
     --batch-size 32 --evaluation-batch-size 64 \
     --head-learning-rate 3e-4 --backbone-learning-rate 1e-5 \
     --backbone-freeze-epochs 0 --freeze-backbone-batch-norm \
     --detach-geometry-backbone --presence-freeze-after-safe-epochs 2 \
     --weight-decay 1e-4 --warmup-fraction 0.05 --seed 42 \
     2>&1 | tee artifacts/roof-model-keypoints-v2/training.log
   ```

   The joint phase locked presence after epoch 6. Its authenticated continuation
   added these arguments, with every other data, schedule, and optimizer
   argument unchanged:

   ```text
   --geometry-refine-from artifacts/roof-model-keypoints-v2-v15-e06-safe/candidate-e06.mpk
   --geometry-refine-source-epoch 6
   ```

   Do not rerun it unless a new experiment is intended; the canonical directory
   contains the promoted result.
6. The hard promotion gate created
   `artifacts/roof-model-keypoints-v2/model.mpk` and `model.json`. Treat
   candidates as diagnostics even when their individual metrics look good.
   Future promotions stage all outputs in the artifact directory, publish the
   manifest before the checkpoint, and expose `promoted=true` metrics last.
7. WGPU and Flex checkpoint parity are complete. Run the qualitative fixed
   contact sheet over all 18 untouched photos in `samples/`, then inspect
   representative `roof-detect --model artifacts/roof-model-keypoints-v2/model`
   overlays.
8. Integrate the promoted observation model with browser capture, VIO/SLAM,
   multi-frame convergence, guidance, and Three.js rendering without merging
   those timestamped contracts.

## Validation

```bash
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo check --target wasm32-unknown-unknown -p roof-model --no-default-features
cargo check --target wasm32-unknown-unknown -p roof-fit
cargo run --release -p roof-synth -- validate \
  datasets/synthetic-overfit-balanced
cargo run --release -p roof-synth -- validate \
  datasets/synthetic-training-keypoints-v2
```

On 2026-07-22, the Linux workspace passed formatting, all workspace tests (with
the explicit 20k-plan stress test ignored), strict workspace Clippy, strict
CUDA-feature Clippy for `roof-model`/`roof-train`, both WASM checks, and v2
dataset validation. The promoted checkpoint also passed independent WGPU and
Flex full-split evaluation plus a live WGPU `roof-detect` smoke.

## Cleanup and repository-transfer checklist

Cleanup removed only reproducible or rejected material:

- approximately 11.9 GiB of Cargo output and the Node dependency cache;
- seven obsolete/pre-correction synthetic corpora;
- the stopped full-run checkpoint, old global/spatial/smoke models, benchmark
  artifacts, and misleading legacy overlays;
- duplicate overfit checkpoints;
- five uncompiled legacy Rust source files;
- `.DS_Store` files and rejected root CLI outputs.

Retained:

- all seven `references/` checkouts;
- all original and generated content under `samples/`;
- the historical v1 full corpus, the overfit corpus, and both reviewed real
  datasets;
- the complete Rust workspace and public synthetic asset pack;
- the minimal valid overfit checkpoint and metrics.

Almost all Rust implementation files are new relative to the old repository
HEAD. **Do not use `git clean`.** Before pushing, inspect `git status` and
deliberately stage `Cargo.toml`, `Cargo.lock`, `rust-toolchain.toml`, `crates/`,
`tools/`, `assets/`, `.gitignore`, `.gitmodules`, `HANDOFF.md`, the new/updated
notes, the minimal overfit artifact, and `references/burn-models` as a
submodule. The 18 original photos are already tracked. The local
`samples/synthetic-generated/` gallery is retained but intentionally ignored as
generated data.
