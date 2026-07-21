# Roof Training and Single-Frame Inference

> The Mac full-corpus run was stopped after epoch 4 and did not pass promotion.
> See [`HANDOFF.md`](../HANDOFF.md) for its metrics, cleanup state, Linux/RTX
> continuation steps, and the folded corrected detection plan.

## Data roles

The still-image path uses each dataset for one specific job:

| Source | Training role |
| --- | --- |
| `datasets/synthetic-training-keypoints/` | Generated Pizza Hut roofs: presence plus complete amodal geometry |
| Same synthetic dataset | Generated ordinary roofs: negative presence only |
| `datasets/wikimedia-positives/` | Current and former Pizza Hut roofs: positive presence only |
| `datasets/open-images-negatives/` | Visually verified exterior ordinary buildings: negative presence only |
| `samples/` | Untouched qualitative evaluation; never read by `roof-train` |

The Wikimedia and Open Images manifests retain their supplied splits. Modified
former Pizza Huts remain positives. Synthetic buildings are split before their
images are rendered, so views of one building cannot leak across splits.

All 1,161 Open Images candidates were reviewed across 19 indexed sheets. The
digest-bound ledger accepts 873 clear exteriors (705 train, 89 validation, 79
test) and explicitly rejects 288 interiors, objects, unusual structures,
obscured crops, and ambiguous frames. `roof-train` fails closed unless accepted
records carry `visually_verified`; metadata screening alone is insufficient.

The checked workspace includes both the deliberately small
`datasets/synthetic-overfit-balanced/` memorisation corpus (32 targets and 32
ordinary-building negatives) and the 6,000 + 6,000 independent-building corpus
at `datasets/synthetic-training-keypoints/`. The full corpus contains one view
per building and splits into 9,619 train, 1,152 validation, and 1,229 test
frames. Its three plan-level scene regimes are balanced exactly **overall** for
each class: 2,000 urban, 2,000 suburban, and 2,000 remote targets, with the same
counts for negatives. Hash-assigned splits are intentionally not rebalanced;
their exact counts are recorded in `scene-regime-balance.json`.

The current full corpus has 6,208 buildings with no attached addition, 5,073
with one, and 719 with two. The 6,511 additions comprise 2,891 dining wings,
1,489 entrance vestibules, and 2,131 service annexes; 3,777 use flat roofs and
2,734 use shed roofs. Generated data is local and gitignored. A final model is
available only when its recorded promotion gate has produced
`artifacts/roof-model-keypoints/model.mpk` and `model.json`.

Regenerate the sourced photographs with:

```sh
cargo run --release -p roof-data -- import-open-images
cargo run --release -p roof-data -- import-wikimedia-positives --category-depth 0 --jobs 1
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
The backbone and FPN/heads are fine-tuned with separate AdamW learning rates.

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

Generate the complete independent-building corpus with:

```sh
cargo run --release -p roof-synth -- generate \
  --output datasets/synthetic-training-keypoints \
  --dataset-id synthetic-training-keypoints \
  --seed 42 --targets 6000 --negatives 6000 --frames 1 \
  --width 640 --height 480 --samples-per-shard 256

cargo run --release -p roof-synth -- validate \
  datasets/synthetic-training-keypoints
```

Generation refuses to overwrite a non-empty output. It writes ordinary flat,
gable, hip, shed, mansard, pyramid, and cupola roofs with the same rendering,
lighting, signage, background, and occlusion distributions as the targets. It
also writes target/negative and day/night contact sheets for visual review.
The recorded full-corpus validation reports 12,000 one-frame buildings, 48
shards, 54,000 aligned artifacts, all 45 required coverage cells, and zero
errors or warnings. The seven negative roof families contain 858 flat roofs and
857 examples each of gable, hip, shed, mansard, pyramid, and ordinary cupola.

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

The complete training command is:

```sh
cargo run --release -p roof-train -- \
  --synthetic datasets/synthetic-training-keypoints \
  --negatives datasets/open-images-negatives \
  --real-positives datasets/wikimedia-positives \
  --artifacts artifacts/roof-model-keypoints \
  --backend wgpu --epochs 40 --patience 6 --batch-size 16 \
  --head-learning-rate 3e-4 --backbone-learning-rate 3e-5 \
  --weight-decay 1e-4 --warmup-fraction 0.05
```

Every run writes `model-last.mpk`, `candidate-best.mpk`, the configuration, and
metrics. Before optimizer work, an older promoted default is moved under
`previous-promoted/`; a failed or interrupted run cannot masquerade as a fresh
success. A candidate is copied to `model.mpk` and receives `model.json` only if
the applicable acceptance gate passes. Failed candidates stay diagnostic and
must not become the CLI default. `--evaluate-checkpoint CHECKPOINT` re-runs the
gate without optimizer updates.

The full-model promotion gate is numerical and source-aware. Test results must
reach at least 0.95 overall target recall, 0.80 held-out real-photo recall, 0.90
overall specificity, 0.85 curated-real-negative specificity, 0.90 PCK@5%, and
0.90 offscreen accuracy. Synthetic perspective fitting must succeed on at least
90% of attempted roofs, accept at least 90%, keep median fitted-mesh RMSE at or
below 0.03 of the image diagonal, and reach median amodal silhouette IoU of at
least 0.80. Validation real-photo recall must also reach 0.80 when that source
is present. The strict 32+32 gate instead requires 1.000 recall and specificity,
PCK@3% and offscreen accuracy of at least 0.98, and zero duplicated/collapsed
pairs.

The Mac full-corpus run was stopped after epoch 4. It reached PCK@5% 0.859 and
offscreen accuracy 0.988, but only 0.112 overall specificity and was not
promoted. Its diagnostic checkpoint was removed during handoff cleanup. No
final full-corpus checkpoint or performance result is claimed until a new run
passes the recorded gate.

## Playable single-frame detector

After promotion, run one image through inference and perspective fitting:

```sh
cargo run --release -p roof-detect -- path/to/photo.jpg \
  --output photo-overlay.png --json photo-prediction.json
```

The default checkpoint base is `artifacts/roof-model-keypoints/model`; Burn
loads the corresponding `.mpk`. Use `--model` to inspect an explicit candidate,
`--backend flex` for CPU inference, `--verify-backend-parity` to compare WGPU
and Flex observations, `--show-all` for rejected fits, and
`--raw-keypoint-debug` to draw the learned observations. Presence, offscreen,
and minimum keypoint-confidence defaults come from the promoted model manifest.

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

The default fitter requires at least six accepted keypoints. A fit is rejected
when normalized reprojection RMSE exceeds 0.05 or fewer than two thirds of the
accepted observations are inliers, so the normal CLI does not draw a confident
mesh from insufficient or geometrically inconsistent evidence.

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
