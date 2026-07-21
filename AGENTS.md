# Repository Guidelines

## Project Structure & Module Organization

- `BRIEF.md` defines the product; `notes/UPDATED_PLAN.md` is the architecture plan.
- `crates/roof-geometry/` owns the parametric two-tier roof and semantic IDs.
- `crates/roof-fit/` converts noisy image observations into relative roof parameters and a coherent projected mesh.
- `crates/roof-model/` owns portable Burn preprocessing, MobileNetV2 inference, and prediction contracts.
- `crates/roof-training/` builds symmetry-aware amodal-keypoint and offscreen targets.
- `crates/vio-core/` owns browser-independent sensor buffering and IMU preintegration.
- `crates/synth-data/` and `crates/synth-render/` own deterministic records and native WGPU rendering.
- `tools/roof-synth/`, `roof-data/`, `roof-train/`, and `roof-detect/` generate, source, train, and inspect data.
- `app/` is the React/TypeScript client. `references/` is study-only; never edit or build its checkouts unless explicitly requested.

Keep learned inference, VIO, roof fitting, synthetic tooling, and Three.js rendering separated through timestamped, serializable contracts.

## Build, Test, and Development Commands

From the repository root:

```bash
cargo fmt --all -- --check
cargo test --workspace
cargo clippy --workspace --all-targets -- -D warnings
cargo check --target wasm32-unknown-unknown -p roof-model --no-default-features
cargo check --target wasm32-unknown-unknown -p roof-fit
cargo run --release -p roof-train -- --help
cargo run --release -p roof-synth -- validate datasets/synthetic-overfit-balanced
cargo run --release -p roof-detect -- photo.jpg
```

See `notes/TRAINING_AND_INFERENCE.md` for reproducible import and training commands. From `app/`, use `pnpm dev`, `pnpm lint`, and `pnpm build`.

## Coding Style & Naming Conventions

Use Rust 2024, rustfmt, documented public APIs, and no unsafe code. Prefer `snake_case` functions/modules, `PascalCase` types, explicit units (`timestamp_ns`, `width_m`), checked arithmetic, and deterministic serialized output. TypeScript uses strict ES modules, two-space indentation, PascalCase components, and camelCase values.

## Testing Guidelines

Place unit tests beside modules and contract tests under `tests/`. Cover stable IDs, geometry invariants, deterministic seeds, corrupt inputs, model tensor shapes, and native/WASM portability. Visually inspect dataset contact sheets and CLI overlays; record iPhone/iOS and permission state for device tests.

## Commits & Pull Requests

Use short imperative subjects such as `Add pretrained roof detector`. Keep unrelated changes separate. Pull requests should explain architecture impact, list validation commands, link the relevant note, and include screenshots/video for visual or device behavior. Never commit secrets, generated datasets, `target/`, `app/dist/`, or `.DS_Store`.
