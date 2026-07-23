# AR device recordings

The Vite development app can capture synchronized real-device sessions for the Rust/WASM fallback.
While an AR session is running, **Record tracking run** starts the camera encoder and raw sensor log;
**Done** stops the recording, uploads it to the Vite development server, and stores it under:

```text
datasets/ar-recordings/<timestamp>-<uuid>/
  manifest.json
  camera.mp4                 # or camera.webm
  sensor-events.ndjson
  tracker-frames.ndjson
  tracker-luma.gray
```

`manifest.json` records the browser/device, video and tracker dimensions, media-track settings,
clock origins, duration, and event counts. Schema 2 recordings are scene-agnostic. Existing schema
1 captures retain their historical lamp label, but neither that label nor any target dimension is a
scale constraint.

`sensor-events.ndjson` retains raw browser values rather than only the normalized values accepted
by WASM. Each line is either `device_motion` or `device_orientation` and includes the event
timestamp, handler receipt timestamp, time since recording start, screen orientation, and every
available acceleration, acceleration-with-gravity, rotation-rate, or orientation component. Schema
2 stores both the browser's raw reported motion interval and its normalized millisecond value;
schema 1 stored the raw value in the millisecond-named field.

`tracker-frames.ndjson` records the timestamp and dimensions of each luma frame submitted to WASM,
the returned pose and diagnostics, and the current keyframe ID/count. `isKeyframe` identifies frames
selected as new visual keyframes. The encoded video contains the complete camera stream rather than
only those selected frames, allowing the exact keyframes and additional frames to be decoded later.
`tracker-luma.gray` contains those submitted `GRAY8` frames consecutively with no header, providing
an exact deterministic replay input without depending on video-container timestamps.

## Ground-truth limitation

These recordings provide synchronized visual and inertial evidence, but neither the lamp's image
coordinate nor its physical size is ground truth. The closed-loop recordings instead provide a
target-independent constraint: their beginning and ending camera poses are the same. Replay must
combine endpoint closure with tracking coverage and non-zero excursion so a frozen estimator cannot
score well. The reported roughly 2.5 m maximum excursion is only a plausibility bound, not exact
metric supervision.

The upload route is supplied by the Hono Vite development server and is intentionally absent from a
production static build. The `datasets/` tree is gitignored.
