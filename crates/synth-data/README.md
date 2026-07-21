# `synth-data`

Renderer-independent data contracts and deterministic procedural foundations for
the Pizza Hut roof training dataset.

The crate has no GPU, filesystem, or browser dependencies. It is suitable for
native batch generation and a later WASM build. A generation job follows this
flow:

```rust
use synth_data::{GeneratorConfig, SequenceRequest, SequenceSampler, TargetKind};

let sampler = SequenceSampler::new(GeneratorConfig::default())?;
let plan = sampler.sample(SequenceRequest::procedural(
    "classic_two_stage",
    42,
    TargetKind::Target,
))?;
```

`SequencePlan` contains a completely resolved scene and coherent camera path. Its
composition plan records a correlated day/twilight/night environment, weather,
site domain, façade, signage state, ground finish, background buildings,
attached dining/entrance/service additions, vegetation, parking/road/curb
infrastructure, utility poles and lines, and coarse occluders. Camera plans include orbit, lateral-walk, approach-arc, and
corner-reveal motion with smooth zoom and deliberate partial framing.

A renderer consumes the exact plan and writes standalone `FrameRecord`
annotations. Requested framing is only an intent: masks, bounding boxes,
visibility, truncation, and occlusion must be measured from the rendered passes.
After resolving texture, environment, mesh, or device-profile inputs, the
producer records their manifest IDs in `scene.composition.source_asset_ids`.
The plan converts to a `SequenceRecord` after applying a `SplitPolicy`.
That record retains `camera_motion` so replay and preview tools do not lose path,
zoom, or framing intent.

The sampler rejects camera paths through the primary shell, attached additions,
or neighbouring-building bounds and keeps clutter clear of the target footprint.
It expands the finite ground plane to contain the
resolved composition and camera path with a safety margin. Partial framing is
solved independently in each camera's image-space right/up basis so the declared
edge is actually truncated throughout a curved sequence.

## Reproducibility contract

- Sampling uses explicitly versioned `ChaCha20Rng` streams.
- Roof, camera, lighting, material, clutter, environment, façade, attached-addition, signage, and
  site-layout streams use independent derived seeds. Adding clutter choices does
  not reshuffle roof dimensions.
- Dataset splits hash building family plus building seed. When a source-asset
  group is present, that group takes precedence so one source cannot leak across
  splits.
- The complete generator configuration is serialized and fingerprinted. Plans,
  sequence records, and the dataset manifest carry that fingerprint; sequence
  IDs also include target intent and the fingerprint.
- Every sampled float and camera transform is stored in the plan; rendering does
  not need to repeat random sampling.

`DatasetValidator` checks manifest taxonomies, source-asset references and split
groups,
camera models, transforms, safe asset paths, required target files, structural
projections, physically plausible sampled composition, sequence ordering,
configuration provenance, and stable split assignment. It returns all findings
in a serializable `ValidationReport` rather than stopping at the first failure.

Run the crate checks from the repository root:

```bash
cargo test -p synth-data
cargo clippy -p synth-data --all-targets -- -D warnings
cargo check -p synth-data --target wasm32-unknown-unknown
```
