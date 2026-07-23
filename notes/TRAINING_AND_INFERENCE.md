# Roof Training and Single-Frame Inference

> The Mac full-corpus run was stopped after epoch 4 and did not pass promotion.
> See [`HANDOFF.md`](../HANDOFF.md) for its metrics, cleanup state, Linux/RTX
> continuation steps, and the folded corrected detection plan. That run and
> `datasets/synthetic-training-keypoints/` use the historical v1 contract. The
> v2 rendering and training are complete. The `roof-training/v15` CUDA run
> promoted `artifacts/roof-model-keypoints-v2/model.mpk` and `model.json`; the
> exact held-out metrics and continuation history are recorded below.

## Data roles

The still-image path uses each dataset for one specific job:

| Source | Training role |
| --- | --- |
| `datasets/synthetic-training-keypoints-v2/` | Completed v2 generated Pizza Hut roofs: presence plus complete amodal geometry |
| Same v2 synthetic dataset | Generated ordinary roofs: negative presence only |
| Visibility-eligible records in `datasets/wikimedia-positives/` | Current and former Pizza Hut roofs: positive presence only |
| `datasets/open-images-negatives/` | Visually verified exterior ordinary buildings: negative presence only |
| `samples/` | Untouched qualitative evaluation; never read by `roof-train` |

The Wikimedia positive manifest uses `roof-positive-dataset/v2`. Category
membership establishes historical-site provenance; it does not by itself make
an image positive supervision. Each record has a separate
`characteristic_roof_visibility` review. Only `recognizable` records may enter
training, threshold calibration, or evaluation. `not_recognizable` and
`unreviewed` records stay available for provenance and review, are never
negatives, and fail closed out of every model role. Modified former Pizza Huts
remain positives only when the characteristic two-tier roof is recognizable in
the supplied image.

The manifest retains 25 photos across 17 physical buildings. Eighteen photos
across 12 buildings are eligible: train has 8 images across 6 buildings,
validation has 6 across 3, and test has 4 across 3. The other 7 records are
provenance-only. Every record carries a `physical_building_id`; all views of a
building, including excluded views, must stay in one split, and `roof-train`
rejects cross-split leakage before applying visibility filters. Synthetic
buildings are likewise split before their images are rendered.

All 1,161 Open Images candidates were reviewed across 19 indexed sheets. The
digest-bound ledger accepts 873 clear exteriors (705 train, 89 validation, 79
test) and explicitly rejects 288 interiors, objects, unusual structures,
obscured crops, and ambiguous frames. `roof-train` fails closed unless accepted
records carry `visually_verified`; metadata screening alone is insufficient.

The checked workspace includes the deliberately small
`datasets/synthetic-overfit-balanced/` memorisation corpus (32 targets and 32
ordinary-building negatives) and the historical v1 6,000 + 6,000 corpus at
`datasets/synthetic-training-keypoints/`. The latter contains one view per
building and splits into 9,619 train, 1,152 validation, and 1,229 test frames.
Its three plan-level scene regimes are balanced exactly **overall** for each
class: 2,000 urban, 2,000 suburban, and 2,000 remote targets, with the same
counts for negatives. Hash-assigned splits are intentionally not rebalanced;
their exact counts are recorded in `scene-regime-balance.json`. These figures
describe the retained v1 corpus.

The completed v2 corpus contains 12,000 one-frame buildings (12,000 frames) in
48 shards and 54,000 aligned artifacts. It splits into 9,547 train, 1,226
validation, and 1,227 test frames. Its coverage report records all 75/75
required positive cells. Targets and negatives are each balanced exactly
overall at 2,000 urban, 2,000 suburban, and 2,000 remote scenes, while
hash-assigned splits remain intentionally unbalanced. The negative set has 858
flat roofs and 857 each of gable, hip, shed, mansard, pyramid, and ordinary
cupola roofs. The dataset validator completed with zero warnings.

The historical v1 full corpus has 6,208 buildings with no attached addition,
5,073 with one, and 719 with two. The 6,511 additions comprise 2,891 dining
wings, 1,489 entrance vestibules, and 2,131 service annexes; 3,777 use flat
roofs and 2,734 use shed roofs. Generated data is local and gitignored. A v2
final model is available only when its recorded promotion gate has produced
`artifacts/roof-model-keypoints-v2/model.mpk` and `model.json`. The v2
artifact path remains separate from historical results.

Regenerate the sourced photographs with:

```sh
cargo run --release -p roof-data -- import-open-images
cargo run --release -p roof-data -- apply-open-images-review \
  --ledger assets/training/open-images-review-ledger.json
cargo run --release -p roof-data -- validate-open-images-review
cargo run --release -p roof-data -- import-wikimedia-positives \
  --category-depth 0 --jobs 1
```

## Model contract

`roof-model` contains one backend-independent Burn network with a trainable,
ImageNet-pretrained MobileNetV2 feature pyramid. A 256×256 letterboxed image
produces:

- one global roof-presence logit;
- twelve 64×64 amodal keypoint distributions: four eave, four shoulder, and
  four crown corners;
- one offscreen logit for each keypoint.

Each map's 4,096 cells and its offscreen value form one categorical
distribution. Synthetic in-frame keypoints target a small Gaussian at the exact
projection; truncated or behind-camera points target the offscreen token. The
loss considers all eight cyclic/reflected `D4` correspondences and uses one
shared correspondence for all three rings. This removes arbitrary
front/left naming from a symmetric roof.

Presence is trained for every image. Geometry loss is applied only to synthetic
Pizza Hut roofs. Presence examples are balanced by source within each minibatch
so the smaller real-photo dataset cannot disappear beneath generated examples.
Source-balanced BCE is the complete v15 presence objective. The v14 pairwise
soft-margin experiment is retained in code and configuration for reproducible
diagnostics, but its v15 weight is `0.0`; the small real corpus made those batch
comparisons easy to memorize without improving held-out separation.

The default `--real-positive-repeat 1` inserts each eligible Wikimedia image
once into the real-positive source pool. Source-balanced epoch draws revisit
that pool with fresh deterministic augmentation keys, so increasing the repeat
does not create new physical buildings or independent evidence. When a
per-split limit is used, real positives are selected building-first before
additional views. Normal epoch construction also balances real-positive draws
by `physical_building_id`: a deterministic seed/epoch permutation cycles once
through every building before repeating one, then cycles that building's views.
The recorded strategy is
`physical_building_balanced_deterministic_view_cycle/v1`; it prevents sites
with more eligible photographs from dominating the real-positive source.

Before the presence lock, the FPN/heads train at `3e-4` and MobileNetV2 at
`1e-5`; both use 5% warm-up followed by cosine decay over the original
40-epoch horizon. The active configuration keeps imported MobileNetV2
BatchNorm running mean and variance fixed. Before the FPN, all four pyramid
tensors are detached, so the keypoint and offscreen objectives update the
decoder and their heads without pushing renderer-specific geometry gradients
into MobileNetV2. Presence uses the original differentiable stride-32 tensor
and is the only loss that updates backbone convolutions and BatchNorm affine
parameters. Newly initialized FPN BatchNorm layers collect adaptive training
statistics normally during this joint phase.

With `--presence-freeze-after-safe-epochs 2`, two consecutive validations must
pass the real gate (0.80 view recall, 0.80 physical-building recall, and 0.85
real-negative specificity) and the synthetic AUC/AP gate. The lock then becomes
sticky: from the next epoch onward MobileNetV2 and the presence heads are
frozen and omitted from optimizer updates, and backpropagation uses only
`0.5 * geometry_loss` through the geometry heads. Entering the lock resets
early-stopping patience once. Later geometry-only epochs consume patience, and
the learning-rate schedules continue from their existing counters rather than
restarting.

The production geometry-only implementation does not merely mask a full joint
forward. It selects the synthetic-positive rows before MobileNetV2, executes
the frozen backbone on the backend's non-autodiff inner tensors, omits the
presence branch entirely, and creates gradients only for the FPN, keypoint,
and offscreen parameters. This avoids retaining an unused backbone/presence
graph across backward and reduces a normal 32-image source-balanced batch to
12 geometry-bearing images for that phase.

Training reports calibrated presence threshold, overall and real-photo recall,
aggregate, synthetic, and curated-real ordinary-building specificity,
source-image-diagonal permutation-matched PCK, offscreen metrics, and
duplicated/collapsed points. Final evaluation also runs perspective fitting on
every held-out synthetic positive and reports fit coverage, corresponding-mesh
RMSE, and complete-silhouette IoU. Masks, semantic IDs, edges, camera state, and
exact roof parameters remain useful generator validation records, but are not
learned heads in this model.

Training augmentation applies the same deterministic in-plane camera roll to
synthetic RGB and all twelve labels, followed by a shared horizontal flip.
Points rotated outside the true source image target the offscreen token;
letterbox padding is never treated as image content.

## Generate the synthetic sets

The small memorisation corpus is reproducible with:

```sh
cargo run --release -p roof-synth -- generate \
  --output datasets/synthetic-overfit-balanced \
  --dataset-id synthetic-overfit-balanced \
  --seed 4242 --targets 32 --negatives 32 --frames 1 \
  --width 640 --height 480 --samples-per-shard 64

cargo run --release -p roof-synth -- validate \
  datasets/synthetic-overfit-balanced
```

Generate the complete v2 independent-building corpus into a new directory:

```sh
cargo run --release -p roof-synth -- generate \
  --output datasets/synthetic-training-keypoints-v2 \
  --dataset-id synthetic-training-keypoints-v2 \
  --seed 42 --targets 6000 --negatives 6000 --frames 1 \
  --width 640 --height 480 --samples-per-shard 256

cargo run --release -p roof-synth -- validate \
  datasets/synthetic-training-keypoints-v2
```

Generation refuses to overwrite a non-empty output. It writes ordinary flat,
gable, hip, shed, mansard, pyramid, and cupola roofs with the same rendering,
lighting, signage, background, and occlusion distributions as the targets. It
also writes target/negative and day/night contact sheets for visual review. The
v2 target contract covers five correlated roof profiles—`tall_early_crown`,
`near_square_tall`, `balanced_classic`, `low_wide_late`, and
`shallow_remodelled`—across three day phases and five detailed scene domains.
All 75 positive-weight morphology/phase/domain cells are required.

Visibility is guarded at both plan selection and publication. A selected scene
may have at most one foreground occluder; deliberately partial framing may not
be combined with any occluder. The default partial-crop and foreground-occluder
ranges are also reduced, and generation aborts instead of publishing a target
with zero visible roof pixels or no visible bounding box. An audit of all 6,000
v2 target frames found zero invisible frames, 27 below 25% visible, 474 below
50% visible, 1,195 truncated, 520 full-width, and a median visible fraction of
0.718; these categories may overlap. The trainer deliberately excludes the
three train-split targets below its 5% visible-fraction floor. Consequently,
the canonical run uses 9,544 eligible synthetic frames plus 705 real negatives
and 8 real positives, or 10,257 training samples in total.

## Train and promote a checkpoint

First prove that the model and loss can memorise exact supervision:

```sh
cargo run --release -p roof-train -- \
  --synthetic datasets/synthetic-overfit-balanced \
  --artifacts artifacts/roof-model-overfit \
  --backend wgpu --overfit --epochs 40 --batch-size 8
```

`--overfit` ignores both real-photo datasets, selects exactly 32 generated
targets and 32 generated negatives, disables augmentation and early stopping,
and evaluates the same fixed set. It cannot be combined with
`--limit-per-split`.

The current portable result is recorded at
`artifacts/roof-model-overfit/final-metrics.json`. After removing backend
autotune from this portable path, standard WGPU and Flex evaluation produced
the same result: 1.000 recall, 1.000 specificity, PCK@3% 0.9803371, PCK@5%
0.98595506, 1.000 offscreen accuracy, and zero duplicated/collapsed point
pairs. This proves the observation model can learn the supplied supervision;
it is not substituted for the final checkpoint.

The portable WGPU equivalent of the full training command is:

```sh
cargo run --release -p roof-train -- \
  --synthetic datasets/synthetic-training-keypoints-v2 \
  --negatives datasets/open-images-negatives \
  --real-positives datasets/wikimedia-positives \
  --real-positive-repeat 1 \
  --artifacts artifacts/roof-model-keypoints-v2 \
  --backend wgpu --epochs 40 --patience 10 --batch-size 32 \
  --evaluation-batch-size 64 --seed 42 \
  --head-learning-rate 3e-4 --backbone-learning-rate 1e-5 \
  --backbone-freeze-epochs 0 --freeze-backbone-batch-norm \
  --detach-geometry-backbone --presence-freeze-after-safe-epochs 2 \
  --weight-decay 1e-4 --warmup-fraction 0.05
```

On an NVIDIA training host with the CUDA toolkit and NVRTC installed, use
Burn's opt-in CUDA backend. CUDA changes only tensor execution; checkpoints
remain portable to the WGPU and Flex inference backends. Evaluation can use a
larger batch without changing optimizer updates or learned weights:

```sh
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

This CUDA configuration completed its joint phase through epoch 6, where the
second consecutive safe validation locked presence. The original geometry-only
forward retained an unused autodiff backbone/presence graph and hit a CubeCL
CUDA allocation failure on epoch 7 batch 2. The failed directory is preserved
at `artifacts/roof-model-keypoints-v2-v15-joint-cuda-oom/`, and the complete
safe source checkpoint is at
`artifacts/roof-model-keypoints-v2-v15-e06-safe/candidate-e06.mpk`.

The corrected continuation authenticated the source checkpoint, sibling
configuration, and final epoch record before any update. It restored 401
batches per logical epoch, 2,406 completed joint/backbone updates, the original
epoch-6 sticky lock, and the 16,040-update cosine horizon. It then ran with the
same arguments plus:

```sh
--geometry-refine-from \
  artifacts/roof-model-keypoints-v2-v15-e06-safe/candidate-e06.mpk \
--geometry-refine-source-epoch 6
```

`refinement-source-metrics.json` schema
`roof-geometry-refinement-source/v2` records the checkpoint, config, and
metrics SHA-256 hashes, metrics line, source counters, derived update contract,
and independent source validation. Any mismatch in datasets, sample counts,
batching, schedule, seed/augmentation, split cap, or lock state fails closed.
Runs using `--limit-per-split` are always diagnostic and cannot promote.

The CUDA feature is deliberately not part of the default build, so WGPU/Flex
training and portable checks do not require a local CUDA toolkit. Loss logging
synchronizes only at the normal 100-batch progress interval; this avoids
stalling asynchronous GPU execution for telemetry that does not affect the
gradient or optimizer.

Every ordinary training run writes `model-last.mpk`, `candidate-best.mpk`,
`candidate-presence.mpk`, `candidate-geometry.mpk`, the configuration, and
metrics. The presence and geometry candidates are diagnostic snapshots of the
best epoch for each concern, not independently deployable models. Before
optimizer work, all live candidates and an older promoted default are moved
under `previous-promoted/`; a failed or interrupted run cannot masquerade as a
fresh success. Geometry refinement is stricter: it requires a fresh artifact
directory and fails rather than moving any existing live artifacts. A candidate
is copied to `model.mpk` and receives `model.json` only if the applicable
acceptance gate passes. Promotion first prepares and verifies same-directory
temporary files, then publishes `model.json`, `model.mpk`, and finally the
`promoted=true` metrics report; a failed commit rolls partial outputs back.
`--evaluate-checkpoint
CHECKPOINT` re-runs the gate without optimizer updates.

The authenticated production refinement early-stopped at logical epoch 33,
ten non-improving epochs after selecting epoch 23. Its promoted files are:

- `artifacts/roof-model-keypoints-v2/model.mpk` (SHA-256
  `f9f24362270e7867e9784a61177d071d0eaf41163aa00f1685f38581f3568dfe`);
- `artifacts/roof-model-keypoints-v2/model.json` (SHA-256
  `e8e8e6c0b1d5f768acadbfa03f49b509dd24e31ae1928eae802e918b08e34599`);
- `artifacts/roof-model-keypoints-v2/final-metrics.json` (SHA-256
  `9c9ec5d653c8c70bcc5482b8107bb8f554afa5929a42fe77cc9038679a754874`).

Epoch-23 validation PCK@5% was `0.938` and offscreen accuracy was `0.997`.
The uncapped held-out test reached PCK@5% `0.928`, offscreen accuracy `0.996`,
real-view recall `0.750`, physical-building recall `1.000` (`3/3`),
real-negative specificity `0.873`, and synthetic ROC AUC/AP `0.975/0.979`.
All `612/612` synthetic roofs fitted; `544` were accepted (`0.889`), with
median mesh RMSE `0.014` and silhouette IoU `0.823`. Promotion passed with no
failures. Reloading the promoted file through WGPU reproduced the same metrics
and passed the gate. Flex also passed with the same detector metrics; its
downstream fitter accepted `545` instead of `544` and reported median IoU
`0.822` instead of `0.823`, within expected backend floating-point variation.
The independent reports are under
`artifacts/roof-model-keypoints-v2-verify-wgpu/` and
`artifacts/roof-model-keypoints-v2-verify-flex/`.

The superseded v2 diagnostic runs were moved intact to
`artifacts/roof-model-keypoints-v2-backbone-3e-5/` and
`artifacts/roof-model-keypoints-v2-backbone-1e-5-unfrozen-bn/`. The superseded
five-epoch frozen-backbone diagnostic is at
`artifacts/roof-model-keypoints-v2-frozen-bn-freeze5/`. These paths preserve
their configurations, logs, metrics, and any diagnostic candidates for audit;
they are not promoted outputs and are not read by the active run.

The stopped no-ranking full diagnostic is retained at
`artifacts/roof-model-keypoints-v2-v12-no-ranking/`. Epoch 6 had passed the
validation presence and geometry gates, but validation real-negative
specificity fell to `0.730`, `0.730`, and `0.618` at epochs 7, 8, and 9. That
consistent deployment-domain regression justified stopping instead of spending
the rest of the 40-epoch budget.

The frozen epoch-6 checkpoint and its independent CUDA evaluation are at
`artifacts/roof-model-keypoints-v2-probe-e06/`. Validation real recall,
real-negative specificity, and PCK@5% were `0.833`, `0.888`, and `0.906`.
Test real recall was `0.750` (3/4 images), real-negative specificity `0.873`,
synthetic ROC AUC/AP `0.975/0.979`, PCK@5% `0.888`, and offscreen accuracy
`0.991`. The missed image was the rear Adel view while the front Adel view
succeeded; all 3/3 test buildings therefore had at least one detected view.
Synthetic fitting completed 612/612 roofs, accepted 505, and produced median
mesh RMSE `0.018` and silhouette IoU `0.790`. Re-evaluation under the v15 gate
at `artifacts/roof-model-keypoints-v2-v15-e06-eval/` detected all 3/3 test
buildings from 3/4 views and passed the presence and fitter requirements. It
failed only PCK@5% (`0.888` versus the required `0.900`), so it remains a
diagnostic checkpoint rather than a promoted model.

`artifacts/roof-model-keypoints-v2-v14-pairwise-smoke/` contains a capped
two-epoch v14 smoke run. It is mechanical evidence only that the pairwise loss,
bucketed rank, CUDA update path, serialization, reload, and final gate execute;
its deliberately limited-split metrics are not a quality result.

The corresponding full pairwise run is retained at
`artifacts/roof-model-keypoints-v2-v14-pairwise/`. It was stopped after epoch 6
because it never reached the real validation gate and underperformed the v12
epoch-6 checkpoint. Pairwise training is disabled with weight `0.0` in v15.
The capped run at `artifacts/roof-model-keypoints-v2-v15-smoke/` mechanically
exercised the v15 update, serialization, reload, and gate paths; its limited
splits are not evidence of model quality.

The v2 operating threshold is calibrated only from validation real-camera
positives and curated real-building negatives. Presence-checkpoint selection
requires that validation operating point to reach at least 0.80 real-positive
view recall, 0.80 physical-building recall, and 0.85 real-negative specificity.
Building recall groups views by `physical_building_id` and counts a building as
detected when any of its views clears the threshold; missing IDs fail closed.
Source probability quantiles and overall, per-domain, and cross-domain ROC
AUC/AP remain recorded diagnostics; aggregate or cross-domain same-threshold
recall/specificity does not control promotion because synthetic and real-camera
scores need not share one calibration.

Checkpoint selection is recorded as
`hard_gates_then_bucketed_real_robustness/v1`. It first protects the real
validation recall/specificity gate, then the synthetic ROC AUC/AP gate, and
then advances geometry to its PCK/offscreen gate. Once every hard gate is
viable, the primary tie-breaker is bucketed real-domain robustness:

- `0.40 * real specificity + 0.35 * real ROC AUC + 0.20 * real average
  precision + 0.05 * real recall`;
- 200-basis-point bands for that robustness score and for minimum real-gate
  margin;
- 100-basis-point bands for synthetic gate margin and synthetic ranking
  quality.

Only after those bands tie does single-frame geometry quality decide between
fully viable epochs. This deliberately values deployment-domain stability over
small PCK movement that can converge across later video frames.

The full-model v2 promotion gate is numerical, source-aware, and multi-view
aware. At the frozen validation-calibrated threshold, both validation and test
must reach at least 0.75 held-out real-positive view recall and 0.80
physical-building recall with complete building IDs; test must also reach 0.85
curated-real-negative specificity. Threshold-free test synthetic presence ROC
AUC and average precision must each reach 0.95. Geometry must reach 0.90
PCK@5% and 0.90 offscreen accuracy. Synthetic perspective fitting must succeed
on at least 90% of attempted roofs, accept at least 80%, keep median fitted-mesh
RMSE at or below 0.08 of the image diagonal, and reach median amodal silhouette
IoU of at least 0.50. The strict 32+32 gate instead requires 1.000 recall and
specificity, PCK@3% and offscreen accuracy of at least 0.98, and zero
duplicated/collapsed pairs.

The historical v1 Mac full-corpus run was stopped after epoch 4. It reached
PCK@5% 0.859 and offscreen accuracy 0.988, but only 0.112 overall specificity
and was not promoted. Its diagnostic checkpoint was removed during handoff
cleanup. The completed v2 result above supersedes that historical run; only
the gate-created `artifacts/roof-model-keypoints-v2/model.mpk` plus its
`model.json` manifest are production artifacts.

## Playable single-frame detector

After promotion, run one image through inference and perspective fitting:

```sh
cargo run --release -p roof-detect -- path/to/photo.jpg \
  --output photo-overlay.png --json photo-prediction.json
```

The CLI's legacy default checkpoint base remains
`artifacts/roof-model-keypoints/model`. A promoted v2 run is intentionally kept
separate, so pass the explicit model base
`artifacts/roof-model-keypoints-v2/model` with `--model`; Burn loads the
corresponding `.mpk`. Use `--model` to inspect an explicit candidate,
`--backend flex` for CPU inference, `--verify-backend-parity` to compare WGPU
and Flex observations, `--show-all` for rejected fits, and
`--raw-keypoint-debug` to draw the learned observations. Presence, offscreen,
and minimum keypoint-confidence defaults come from the promoted model manifest.
An explicit candidate such as `model-last.mpk` only uses a same-base
`model-last.json`; it never borrows calibration from `model.json`. The CLI warns
and uses generic defaults when that matching manifest is absent.

The only required input is the image. Burn predicts presence and the twelve
amodal observations. `roof-fit` then evaluates all eight correspondences and
robustly fits camera rotation, translation, focal length, and seven bounded,
scale-free roof ratios with a pinhole camera model. EXIF 35 mm focal information
is used when present; otherwise the fitter starts from several field-of-view
hypotheses. `roof-geometry` generates the complete two-tier mesh, including
occluded parts.

The PNG draws a mesh only when presence and fit checks accept it unless
`--show-all` is supplied. The JSON contains the raw observations, camera
metadata, inferred roof parameters, projected mesh, bounds, reprojection error,
fit confidence, or a structured fit failure. Users never supply camera pose,
roof parameters, depth, or sensor data to this still-image command.

The default fitter requires at least four accepted keypoints. Four- and
five-point fits must cover all three structural rings and at least two cyclic
corner slots. A fit is rejected when normalized reprojection RMSE exceeds 0.05,
confidence-weighted inliers fall below two thirds, the projected full mesh
extrapolates implausibly far beyond the observations, or its combined score is
below 0.25. The normal CLI therefore does not draw a confident mesh from a
tight point cluster or geometrically inconsistent evidence.

After promotion, the qualitative acceptance run processes all 18 untouched
images in `samples/` into one fixed contact sheet. Those photographs remain
outside training and threshold selection. The target is at least 80% detection
recall and plausible placement of both roof tiers on at least 70% of them. The
separate curated-real-negative test split must remain at or below 15% false
positives, equivalent to the 0.85 specificity promotion floor.

Check the portable boundary with:

```sh
cargo check --target wasm32-unknown-unknown -p roof-model --no-default-features
cargo check --target wasm32-unknown-unknown -p roof-fit
```

This command deliberately covers one frame only. Live visual-inertial tracking
and multi-view fusion remain separate browser work.
