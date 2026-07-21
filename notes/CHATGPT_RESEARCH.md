The most practical architecture is a small learned landmark detector followed by a deterministic parametric 3D fit. I would not use a generic monocular-depth model: it would be larger, slower, and less stable than exploiting the roof’s distinctive, constrained shape.

### Recommended pipeline

```text
webcam frame
    ↓
crop / resize / normalize
    ↓
tiny CNN:
  - Pizza Hut roof confidence
  - bounding box
  - 10–16 roof landmarks
  - visibility/confidence per landmark
  - optional silhouette mask
    ↓
robust PnP pose estimate
    ↓
nonlinear parametric roof fitting
    ↓
temporally filtered 3D geometry
```

For the roof in the example, useful landmarks would include the visible main-eave corners, hip/ridge junctions, lower and upper corners of the raised central “hat”, and the top-ridge endpoints. Give each keypoint a visibility output because several will inevitably be behind the building or outside the frame.

## Crates

### Browser and frame acquisition

Use `wasm-bindgen`, `web-sys`, and `js-sys`/`wasm-bindgen-futures`.

`web-sys` exposes `MediaDevices`, `MediaStream`, `HtmlVideoElement`, canvas APIs, and the surrounding browser interfaces. `wasm-bindgen-futures` bridges JavaScript promises into Rust futures, which is useful around `getUserMedia()`. ([Docs.rs][1])

In practice, I would keep the camera plumbing thin:

```text
getUserMedia()
→ HtmlVideoElement
→ drawImage(video, offscreen canvas)
→ ImageData / Uint8Array
→ WASM inference function
```

Webcam access itself is through the browser’s `MediaDevices.getUserMedia()` API and requires user permission. ([MDN Web Docs][2])

### Preprocessing

Use:

- `fast_image_resize` for converting the webcam frame to a fixed inference size such as 256×256 or 320×320. It has Wasm SIMD128 implementations for common pixel formats. ([GitHub][3])
- `image` for basic image buffers and pixel types.
- `imageproc` for optional grayscale conversion, filtering, Canny edges, morphology, contours, Hough-style line processing, and debug overlays. It is a general-purpose Rust image-processing library rather than a browser-specific wrapper. Disable its default `rayon`, `text`, and `fft` features unless required, to reduce the WASM bundle and avoid unnecessary dependencies. ([Docs.rs][4])

I would not make edge detection the primary detector, but it can be useful for refining predicted roof edges after the neural model has isolated the correct building.

### Model inference: first choice

Use `rten`.

RTen is a pure-Rust ONNX runtime designed to compile to WebAssembly. It supports WebAssembly SIMD, float models, and int8/uint8 quantized weights, and it provides both a built-in JavaScript/WASM API and the option to perform preprocessing and postprocessing inside your own Rust WASM module. It is CPU-only, which is probably acceptable—and simpler—for a very small keypoint network. ([GitHub][5])

This would be my initial deployment target:

```text
256×256 RGB
MobileNetV3-small-like backbone
one low-resolution heatmap per landmark
one objectness output
optional low-resolution mask
INT8 or UINT8 weights
```

A roof-specific keypoint network can be substantially smaller than a general-purpose object detector because it only needs to represent one architectural family.

### Model inference: alternatives

`tract` is the strongest CPU-WASM alternative. It loads ONNX and NNEF, supports browser deployment through standard WASM targets, and includes browser benchmarking and computer-vision examples. It is a good option where its model conversion and optimization pipeline works better for your chosen architecture. ([GitHub][6])

`burn` plus its WGPU backend is the option for WebGPU acceleration. Burn supports browser inference using its CPU Flex backend or WGPU/WebGPU backend. Its ONNX importer generates native Burn code, but the importer still supports a limited set of ONNX operators, so model compatibility should be tested before committing to it. ([GitHub][7])

I would begin with RTen and only move to Burn/WebGPU when measurements show that CPU SIMD is not sufficient. For a very small fixed-input CNN, GPU dispatch and texture-transfer overhead can erase some of the benefit of WebGPU.

### Camera and 3D geometry

Use the Rust-CV crates:

- `cv-pinhole` for camera intrinsics, lens distortion and converting image coordinates to 3D bearing vectors. ([GitHub][8])
- `lambda-twist` for fast P3P pose hypotheses from known 3D-to-2D correspondences. ([GitHub][9])
- `sample-consensus` and `arrsac` for rejecting erroneous or low-confidence landmark correspondences. ARRSAC is intended as a faster adaptive alternative to conventional RANSAC. ([GitHub][10])
- `nalgebra` for vectors, matrices, rotations, transforms and small linear systems.
- `levenberg-marquardt` for refining pose, camera focal length and roof parameters by minimizing reprojection error. The crate provides a Rust nonlinear least-squares solver based on the MINPACK Levenberg–Marquardt implementation. ([GitHub][11])

The geometry stage could look like:

```rust
struct RoofParameters {
    width: f64,
    depth: f64,
    main_eave_height: f64,
    main_roof_pitch: f64,
    hat_width_ratio: f64,
    hat_depth_ratio: f64,
    hat_height: f64,
    hat_side_slope: f64,
}

struct CameraPose {
    rotation: nalgebra::UnitQuaternion<f64>,
    translation: nalgebra::Vector3<f64>,
}
```

Start with a canonical roof mesh and solve pose using the high-confidence landmarks. Then refine:

```text
minimize Σ robust_loss(
    project(K, distortion, pose, roof_vertex(parameters))
    - detected_keypoint
)^2
```

The parameter bounds are important. For example, constrain roof pitch, upper-hat width ratio and upper-hat height to plausible ranges. This prevents a bad landmark from producing a physically absurd roof.

### Temporal tracking

`optical-flow-lk` is an optional crate for Lucas–Kanade tracking and Shi–Tomasi feature detection, and is explicitly aimed at real-time and WASM-compatible use. It could track landmark-adjacent image points between neural detections. ([GitHub][12])

A useful scheduling arrangement would be:

```text
CNN inference: every 2–4 frames
optical-flow/keypoint propagation: intervening frames
PnP + geometry refinement: every frame
full redetection: after confidence loss or large motion
```

The geometry optimization itself should be very cheap because it is only solving perhaps 10–15 parameters against a small number of observations.

## Day and night

Day/night reliability will come mostly from the data, not the crate selection.

Train the detector to recognize geometry rather than “a big red roof”. Include:

- Actual night photos, not only darkened daytime photos.
- Gamma and exposure changes.
- Local shadows and bright signage.
- Sensor noise, motion blur and defocus.
- Sodium-vapour, LED and fluorescent colour casts.
- Grayscale and near-monochrome examples.
- Wet roofs and specular highlights.
- Snow, faded paint and roofs repainted another colour.
- Partial occlusion by trees, signs and power lines.
- Hard negatives drawn from ordinary houses and unrelated buildings, including mansard roofs, ordinary hipped roofs, flat-roofed commercial buildings and petrol-station canopies.

Synthetic training data would be particularly effective here. Build a parameterized 3D roof once, render thousands of camera angles with randomized materials, lighting, backgrounds and occluders, and then fine-tune on real photographs. The synthetic renders give you exact landmark, mask, pose and geometry labels essentially for free.

No RGB algorithm can reliably operate in genuinely dark conditions where the sensor contains no roof information. For that case you need adequate ambient lighting, an IR-sensitive camera plus IR illumination, or a thermal/low-light sensor—and training material from that sensor domain.

## The unavoidable monocular limitation

From one ordinary webcam image, absolute physical scale is unobservable. You can estimate:

- Camera-relative orientation.
- Camera-relative translation up to scale.
- Roof proportions.
- Roof pitches and angles.
- A canonical mesh fitted to the image.

You cannot determine that the roof is, for example, exactly 18.3 metres wide unless at least one scale constraint is available. That could be a known canonical dimension, known camera height, a known object in the scene, stereo/depth hardware, or camera movement across substantially different viewpoints.

Camera calibration also matters. A one-time calibration per webcam/device ID will produce noticeably better pitch and depth estimates than assuming an arbitrary focal length. With unknown intrinsics, you can add focal length to the nonlinear optimization, but pose and geometry become less well constrained.

## Suggested initial stack

```text
wasm-bindgen
web-sys
js-sys / wasm-bindgen-futures

rten
fast_image_resize
image
imageproc

nalgebra
cv-core
cv-pinhole
lambda-twist
sample-consensus
arrsac
levenberg-marquardt

optical-flow-lk        # optional
wgpu                   # optional 3D overlay
```

The smallest credible prototype would be:

1. Train a 256×256 model predicting object confidence and approximately 12 landmarks.
2. Export it to ONNX and run it with RTen.
3. Define one normalized Pizza Hut roof mesh.
4. Use `cv-pinhole` + `lambda-twist` + `arrsac` for initial pose.
5. Refine pose and four to eight roof-shape parameters with `levenberg-marquardt`.
6. Draw the reprojected mesh over the webcam frame.
7. Add night imagery and synthetic domain randomization after the daylight pipeline is stable.

[1]: https://docs.rs/web-sys/latest/web_sys/struct.HtmlVideoElement.html?utm_source=chatgpt.com "HtmlVideoElement in web_sys - Rust"
[2]: https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices?utm_source=chatgpt.com "MediaDevices - Web APIs | MDN"
[3]: https://github.com/cykooz/fast_image_resize?utm_source=chatgpt.com "Cykooz/fast_image_resize: Rust library for fast image ..."
[4]: https://docs.rs/imageproc?utm_source=chatgpt.com "imageproc - Rust"
[5]: https://github.com/robertknight/rten "GitHub - robertknight/rten: ONNX neural network inference engine · GitHub"
[6]: https://github.com/sonos/tract "GitHub - sonos/tract: Tiny, no-nonsense, self-contained, Tensorflow and ONNX inference · GitHub"
[7]: https://github.com/tracel-ai/burn "GitHub - tracel-ai/burn: Burn is a next generation tensor library and Deep Learning Framework that doesn't compromise on flexibility, efficiency and portability. · GitHub"
[8]: https://github.com/rust-cv/cv/tree/main/cv-pinhole "cv/cv-pinhole at main · rust-cv/cv · GitHub"
[9]: https://github.com/rust-cv/cv/tree/main/lambda-twist "cv/lambda-twist at main · rust-cv/cv · GitHub"

[10]: https://github.com/rust-cv/arrsac?utm_source=chatgpt.com "Implements ARRSAC from the paper \"A ..."
[11]: https://github.com/rust-cv/levenberg-marquardt?utm_source=chatgpt.com "rust-cv/levenberg-marquardt: Provides abstractions to run ..."
[12]: https://github.com/den59k/lucas-shi-rust?utm_source=chatgpt.com "den59k/lucas-shi-rust: High-performance implementation ..."
