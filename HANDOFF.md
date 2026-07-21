# Linux / RTX Handoff

Last updated: 2026-07-21

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

The React application is still a scaffold. Browser capture, live Burn
inference, full VIO/SLAM, multi-view roof fitting, guidance, and Three.js AR
integration have not been implemented.

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
- Use balanced BCE for presence. Geometry loss applies only to synthetic target
  roofs.
- Fine-tune the backbone at `3e-5` and the FPN/heads at `3e-4` with AdamW,
  `1e-4` weight decay, 5% warm-up, cosine decay, batch size 16, at most 40
  epochs, and six-epoch early stopping.

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
- Require at least six accepted observations. Reject a confident fit when
  normalised RMSE exceeds 0.05 or fewer than two thirds are inliers.

Exact-annotation fitter tests already satisfy projected-vertex RMSE below 0.5%
of the image diagonal and silhouette IoU above 0.95 with known intrinsics.

### Data roles

| Data | Role |
| --- | --- |
| Synthetic two-tier roofs | Positive presence and complete geometry supervision |
| Synthetic ordinary roofs | Negative presence only |
| Wikimedia current/former Pizza Huts | Positive presence only |
| Curated Open Images ordinary buildings | Negative presence only |
| `samples/` | Untouched qualitative evaluation; never training or threshold selection |

The retained synthetic corpus has 6,000 independent targets and 6,000
independent ordinary-building negatives, one view per building. It contains
9,619 train, 1,152 validation, and 1,229 test frames in 48 shards. Each class
has exactly 2,000 urban, 2,000 suburban, and 2,000 remote scenes. It covers day,
twilight, night, distant/normal/close/partial framing, seven ordinary-negative
roof families, repainting, additions, lighting, weather, backgrounds, signage,
and occlusion under class-independent distributions.

The curated real data contains 873 accepted Open Images negatives (705/89/79)
and only 19 unique Wikimedia positives (14/2/3). The latter is enough to wire
presence-only domain supervision, but its tiny validation/test counts make
metrics and threshold calibration fragile.

### Promotion gates

A full checkpoint becomes the default only when held-out evaluation reaches:

- target recall at least 0.95;
- real-photo recall at least 0.80;
- overall specificity at least 0.90;
- curated-real-negative specificity at least 0.85;
- permutation-matched PCK@5% at least 0.90;
- offscreen accuracy at least 0.90;
- synthetic fit success and accepted-fit coverage at least 0.90;
- median fitted-mesh RMSE at most 0.03 of the image diagonal;
- median amodal silhouette IoU at least 0.80.

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

## Stopped full-corpus run

The Mac WGPU run was stopped at the start of epoch 5 as requested. Epoch 4 had:

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
- Ordinary `roof-detect image.jpg` therefore fails by design until a checkpoint
  passes promotion.
- The trainer saves model weights, not AdamW/update state, so Linux must restart
  rather than resume epoch 5.

The geometry side improved while the presence decision collapsed. Do not start
another unattended 40-epoch run until presence scores are diagnosed. First log
source-wise probability quantiles and ROC/PR metrics for synthetic targets,
synthetic negatives, real positives, and real negatives. Confirm whether two
low-scoring real validation positives force the calibrated threshold close to
zero. Then make checkpoint selection presence-safe: PCK improvement must not
outweigh unusable specificity. More presence-only real Pizza Hut photos would
improve domain grounding and still require no geometry annotations.

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

Git alone is not enough. `datasets/` is ignored and the retained training data
is about 1.9 GiB:

- `datasets/synthetic-training-keypoints/` — about 1.7 GiB;
- `datasets/open-images-negatives/` — about 210 MiB;
- `datasets/wikimedia-positives/` — about 4.4 MiB;
- `datasets/synthetic-overfit-balanced/` — about 18 MiB.

Copy those four directories to the same paths on Linux with `rsync`, `scp`, an
external drive, or another non-Git channel. Copying preserves the exact reviewed
pixels and manifests.

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

The trainer currently supports only `--backend wgpu` and `--backend flex`.
On Linux, WGPU should use the RTX 4070 through Vulkan and should already be much
faster than the M1 run. CUDA is **not** implemented merely because optional CUDA
packages appear in `Cargo.lock`.

If native Burn CUDA is desired before the full run:

1. Add a `cuda` crate feature in `tools/roof-train/Cargo.toml` that enables
   Burn's `cuda` feature.
2. Under `cfg(feature = "cuda")`, add `Cuda` and `cuda::CudaDevice`, a `Cuda`
   CLI variant, and `Autodiff<Cuda>` train/evaluate dispatch.
3. Build on Linux with CUDA 12.x and invoke the trainer with
   `--features cuda -- --backend cuda`.
4. Repeat the 32+32 overfit gate and evaluate the saved CUDA checkpoint through
   WGPU and Flex before trusting a long run.

Do not enable Burn/CubeCL autotune as a shortcut. It previously produced
numerically divergent WGPU results that did not match ordinary WGPU, Flex, or
`roof-detect`. Benchmark native CUDA and Vulkan separately with exact parity.

## Recommended continuation order

1. Confirm all repository work is committed, clone recursively, and transfer
   the four retained dataset directories.
2. Run dataset validation and the 32+32 overfit CLI/parity command above.
3. Add source-wise presence diagnostics and make checkpoint selection reject
   presence collapse. Inspect the 14/2/3 real-positive split explicitly.
4. Add and parity-test Burn CUDA if its measured speed justifies it; otherwise
   use WGPU/Vulkan.
5. Run a short, fully evaluated training diagnostic. Continue only when both
   source-wise separation and PCK improve; aggregate BCE is insufficient.
6. Start the complete run:

   ```bash
   cargo run --release -p roof-train -- \
     --synthetic datasets/synthetic-training-keypoints \
     --negatives datasets/open-images-negatives \
     --real-positives datasets/wikimedia-positives \
     --artifacts artifacts/roof-model-keypoints \
     --backend wgpu --epochs 40 --patience 6 --batch-size 16 \
     --head-learning-rate 3e-4 --backbone-learning-rate 3e-5 \
     --weight-decay 1e-4 --warmup-fraction 0.05
   ```

7. Let the hard promotion gate—not `candidate-best.mpk` existing—create the
   default `model.mpk` and `model.json`.
8. Run `roof-detect` with only an image filename, verify WGPU/Flex parity, and
   build the fixed contact sheet over all 18 untouched photos in `samples/`.
9. Only after still-image acceptance should browser capture, VIO/SLAM,
   multi-frame fusion, guidance, and Three.js rendering be integrated.

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
  datasets/synthetic-training-keypoints
```

At the Mac handoff point, formatting, workspace checking, workspace clippy,
all 230 workspace tests (with one explicit stress test ignored), and both WASM
checks passed after obsolete source removal. The build cache was then cleaned,
so the Linux clone should still run the complete matrix afresh.

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
- the current full, overfit, Open Images, and Wikimedia datasets;
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
