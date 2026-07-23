# PizzaNet AR client

The Vite application is a cross-platform mobile AR vertical slice. It renders the same basic
experience as `references/threejs-world-effects-example`: a live view, Three.js scene, tracked
one-metre cube, floor shadow, and a recenter/place interaction.

It selects one of two tracking paths at runtime:

- **WebXR:** requests an `immersive-ar` session with `local-floor` tracking and optional hit tests.
  A detected floor automatically positions the cube, and a tap can reposition it.
- **Rust/WASM fallback:** uses the rear camera plus device motion/orientation events. The
  `ar-tracker-wasm` crate consumes timestamped downscaled luma frames and normalized IMU samples
  and returns the Three.js camera pose.

The fallback provides gravity-aligned orientation plus sparse visual-inertial translation. It
drains motion samples for each camera interval, propagates a bounded inertial position/velocity
state, and corrects that state with rotation-compensated, forward/backward-consistent visual
matches. Brightness-normalized patches, robust spatial consensus, quality-gated keyframes, and
bounded geometric plus appearance relocalization improve tracking through exposure changes and
revisited views. Camera frames retain their source aspect ratio, and the Three.js projection uses
the same field-of-view assumption as the tracker. Monocular scale currently uses a 3.6-metre
scene-depth prior, so this remains an approximate world-effects tracker rather than finished
map-scale SLAM. Triangulated inverse-depth landmarks and pose-graph optimization remain future
estimator stages.

## Run locally

Install dependencies and start Vite from this directory:

```bash
pnpm install
pnpm dev
```

`pnpm dev` and `pnpm build` first compile `../crates/ar-tracker-wasm` through the project-local
`wasm-pack`. The generated bindings live under `src/generated/` and are ignored because they are
reproducible build output.

Local desktop access is available at `http://localhost:5555`. A phone cannot use camera, motion,
or WebXR APIs from plain `http://danlinux:5555`, because that network origin is not a secure
context.

For an iPhone or another Tailscale-connected device, leave Vite running and expose it through
Tailscale's trusted HTTPS endpoint in a second terminal:

```bash
tailscale serve --bg 5555
tailscale serve status
```

Open the `https://danlinux.<tailnet>.ts.net` address printed by Tailscale. The Vite server accepts
the forwarded host and sends the cross-origin isolation and sensor permission headers needed by
the application. Run `tailscale serve reset` when that proxy is no longer wanted.

## Validation

```bash
pnpm lint
pnpm build
cargo test -p ar-tracker-wasm
cargo check --target wasm32-unknown-unknown -p ar-tracker-wasm
```

Physical device validation should record the browser, OS/device, selected backend, camera and
motion permission outcome, and whether floor placement or fallback recentering behaved as
expected.

## Development recordings

When the Rust/WASM fallback is running from `pnpm dev`, **Record tracking run** captures the encoded
camera stream, raw device orientation/motion events, and timestamped tracker-frame/keyframe records.
Press **Done** to upload and atomically save the session under `../datasets/ar-recordings/`. Failed
uploads can be retried without re-recording while the page remains open. Recordings also retain the
exact contiguous `GRAY8` frames submitted to WASM for deterministic native replay. See
[`notes/AR_RECORDINGS.md`](../notes/AR_RECORDINGS.md) for the data contract and closed-loop replay
constraints. The lamp label is recording metadata only; its dimensions are not used for scale.
