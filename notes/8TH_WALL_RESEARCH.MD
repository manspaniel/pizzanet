# Pizza Hut Roof Recognition AR Research: Feasibility and Architecture

## Executive summary

This is feasible, but it should be built as category recognition plus model-based pose and shape fitting over several camera frames—not as an image target, a pure edge detector, or unconstrained 3D reconstruction.

Recommended first architecture:

- React for product UI and lifecycle
- Three.js controlled through one XR8 canvas
- 8th Wall SLAM initially, subject to its separate licence
- a lightweight learned detector for roof masks and semantic keypoints
- robust geometric fitting of a parameterised Pizza Hut roof
- multi-frame refinement using camera poses from XR8
- ONNX Runtime Web/WASM initially, with a route to move inference into 8th Wall's existing TFLite-style machinery
- Rust/WASM for geometry and optimisation if desired, but not a new SLAM implementation first

The most important business and product variable is whether Pizza Hut can provide a list of eligible sites plus architectural plans or approximate roof dimensions. “Recognise a known Pizza Hut roof at a known site” is much easier than “discover any former Pizza Hut anywhere and infer its full geometry from one view.”

## What is actually in the 8th Wall repositories

### Open repository versus SLAM

The current 8th Wall repository is MIT licensed, but its SLAM is not part of that open code.

- `references/8thwall/README.md` says the framework is MIT licensed but SLAM is distributed separately.
- `references/8thwall/packages/engine/README.md` explicitly says SLAM, VPS, and hand tracking are omitted.
- The documented hybrid loads the open WebAR framework alongside a separate `xr-slam.js`.
- The separate `@8thwall/engine` package has its own licence and should be reviewed for a commercial Pizza Hut deployment.

Therefore “fork 8th Wall” and “own or modify its SLAM” are different proposals. The open repository can be adapted heavily, but the tracking core would remain a black-box dependency unless replaced.

### The Three.js integration

The current example is intentionally thin:

1. install the GL camera renderer
2. install the Three.js pipeline module
3. install the XR controller/SLAM module
4. install helpers and the application's own pipeline module
5. call `XR8.run({canvas})`

`XR8.Threejs` then copies the camera projection, quaternion, and position into the Three camera each update and renders the scene in its render callback.

React should sit outside this loop. It should own permissions, onboarding, UI, state, and lifecycle, while XR8 and Three retain a single imperative canvas.

### Why image targets are the wrong abstraction

The image-target implementation supports only:

- planar targets
- curvy cylindrical or conical targets

Its detector is based on ORB/GORB-like local image features and uses the known target geometry to recover pose.

That is suitable for a poster, package, mural, label, or other known textured surface. It does not:

- recognise a semantic building category
- infer a deformable roof shape
- generalise across paint and material changes
- account for the roof's multiple planes and self-occlusion
- distinguish roof edges from the building, trees, signs, roads, gutters, and other line-rich clutter

An image target could still be used as a controlled fallback or at a specifically instrumented location, but it is not the foundation for this idea.

### The useful extension points

The engine already exposes a pipeline:

`onProcessGpu -> onProcessCpu -> onUpdate -> onRender`

A module can receive camera pixels and later expose its result to other modules.

The important details:

- `CameraPixelArray` can resize the camera feed and return luminance or RGBA.
- It performs a `readPixels` operation back to CPU memory.
- It reads the previously rendered texture before rendering the current texture, so its image is already one frame behind.
- Pipeline callbacks are synchronous; returning a Promise would not integrate asynchronous inference correctly.
- Every inference must be tagged with the exact source frame and matched to that frame's camera pose.
- The current semantics/layers implementation already demonstrates an appropriate worker pattern: only launch inference when the previous job is complete and discard work rather than building a queue.

The repository also contains reusable geometric primitives including PnP/AP3P, robust estimation, reprojection, and bundle adjustment, plus existing TFLite inference code for face and semantic models.

## What the roof-recognition problem really is

The roofs are visually distinctive, but not identical. The stable signal is their geometry:

- a broad low roof
- a raised central roof or sign volume
- characteristic transitions between those sections
- strong eaves and converging roof seams
- broadly consistent proportions, with meaningful variation

Unstable signals include:

- red, black, blue, yellow, or faded materials
- tile versus metal versus shingle textures
- logos and signage
- vents, chimneys, additions, damage, and renovations
- foreground cars, foliage, poles, and adjoining buildings

The system should therefore solve four separate problems:

1. detection: is a plausible Pizza Hut roof present?
2. correspondence: where are its structural landmarks?
3. pose: where is a roof model relative to the camera?
4. shape: which variant and dimensions best explain the observations?

XR tracking then keeps that solved roof stable as the user moves.

## Recommended recognition model

Use a compact multi-head model, initially at a few inferences per second rather than camera frame rate.

Suggested outputs:

- roof bounding region
- roof and foreground mask
- separate segmentation of the lower roof and raised central section
- 12–24 semantic keypoints
- confidence and visibility per keypoint
- roof-variant probabilities
- optional coarse parameter estimates

Candidate semantic points:

- visible outer eave corners
- intersections between the raised section and lower roof
- top corners or ridge endpoints of the raised section
- visible roof seam intersections
- optional facade anchors where they are consistent enough

Keypoints should be trained with visible, occluded, and out-of-frame states. A PVNet-like direction-voting approach is attractive if direct heatmaps prove brittle under occlusion, but a simpler heatmap model should be tested first.

The model should use colour images, but be trained so colour is not required:

- aggressive hue and material variation
- replacement of red roofs with other finishes
- non-Pizza-Hut red roofs as hard negatives
- lighting, weather, and background randomisation
- simulated occluders and partial crops

Using grayscale alone would throw away information. The objective is not to ignore appearance; it is to prevent one superficial feature from dominating.

## Parameterised geometry

Do not use a single immutable roof mesh. Use a small family of parameterised models.

A useful initial parameter set:

- overall width and depth
- eave height
- lower-roof rise or slope
- central-section width and depth
- central-section rise and top dimensions
- overhang
- limited asymmetry or variant selection

For a frame with enough high-confidence keypoints, solve camera-to-object pose with PnP/RANSAC against a nominal roof. Then refine pose and shape over several frames.

With camera poses `T_world_camera(t)` from XR8 and object pose `T_camera_roof(t)` from vision:

`T_world_roof = T_world_camera(t) * T_camera_roof(t)`

A multi-frame objective should optimise one `T_world_roof` and one shape parameter vector across all accepted observations using:

- robust keypoint reprojection error
- silhouette or mask alignment
- edge alignment after detection
- priors over realistic roof proportions
- consistency with gravity
- outlier rejection

Raw Canny/Hough-style edges can help refine the final projected model, but should not be responsible for deciding what object is present.

## Scale and observability

A single monocular image cannot uniquely determine both object scale and distance. The same projection can result from a small nearby roof or a larger, more distant one.

Use one or more of:

- known dimensions for a specific site
- a known Pizza Hut prototype or variant
- multi-view parallax combined with XR8's metric tracking
- ground or facade constraints
- limited user assistance as a fallback

The experience should ask the user to move several metres or sweep laterally for a few seconds. That makes depth and scale better conditioned and gathers alternate views when landmarks are occluded.

If exact locations are known, GPS can select the site and roof family, but it is not accurate enough to place the effect by itself.

## Runtime architecture

```text
React application
  onboarding / permissions / progress / failure UI
                  |
                  v
XR8 + Three.js pipeline
  camera pose at display rate
  image capture at reduced rate
                  |
                  v
Vision worker
  detection / mask / keypoints / confidence
                  |
                  v
Geometry solver
  PnP / shape refinement / multi-frame optimisation
                  |
                  v
Stable roof transform in XR8/Three world coordinates
                  |
                  v
AR content and roof-based occlusion
```

Each submitted frame record should include:

- unique frame ID
- video or frame timestamp
- pixels and image dimensions
- image rotation and orientation
- mapping between model pixels and original pixels
- camera projection and intrinsics
- camera position and quaternion for that source frame
- tracking status and reason
- session or configuration generation

Never combine a delayed inference with the most recent camera pose.

Suggested state machine:

- `searching`
- `candidate`
- `collecting_views`
- `solving`
- `locked`
- `refining`
- `degraded`
- `lost`

The application should reject uncertain solutions rather than snapping content to a plausible-looking false positive.

## Inference technology

### First implementation

ONNX Runtime Web in a worker is a pragmatic starting point:

- broad WASM support
- model iteration without rebuilding the 8th Wall engine
- predictable fallback across iOS and Android
- WebGPU can be tested opportunistically

WASM should be the guaranteed baseline. WebGPU support varies by browser, OS, and runtime even where the browser API itself is available.

If threaded WASM is used, the deployment needs appropriate COOP/COEP headers and compatible loading for every cross-origin dependency.

### Performance-oriented implementation

If CPU readback becomes a bottleneck:

- adapt the 8th Wall semantics pipeline pattern
- avoid `readPixels` except when a worker is ready
- consider integrating a TFLite model into the existing engine/WebAssembly build
- reduce allocation and copies
- keep model inference around 2–5 Hz while XR tracking and rendering continue at display rate

The model does not need to run at 30 or 60 Hz once the roof has a stable world transform.

## Should the whole stack be rewritten in Rust?

Rust/WASM is reasonable now for:

- pose and shape fitting
- robust estimation
- coordinate conversions
- multi-view optimisation
- edge and silhouette refinement

It is not the highest-leverage first use of time for SLAM.

A new mobile WebAR tracking stack additionally requires:

- camera calibration across devices
- synchronised camera and IMU ingestion
- visual-inertial initialisation
- feature tracking and mapping
- scale and gravity estimation
- relocalisation and recovery
- browser lifecycle and permission handling
- rendering and camera-texture integration
- extensive iOS and Android compatibility testing

That is a separate core-technology programme. It can be architected behind an interface so XR8 can be replaced later, but it does not solve the roof-recognition uncertainty first.

Recommended sequence:

1. use XR8 tracking, subject to licence review
2. isolate it behind a pose-provider interface
3. implement the category detector and roof solver independently
4. prove product value and field reliability
5. replace SLAM only if licensing, access, performance, or strategic ownership justifies it

## Data requirements

The most valuable inputs Pizza Hut could provide:

- prototype or architectural drawings
- known roof variants and dimensions
- current and former site list
- historic image collections
- access to representative locations
- any known modification history

For an initial feasibility programme:

- approximately 20–30 distinct roofs
- many views of each, but split evaluation by building
- accurate reference geometry for a subset
- at least as many hard-negative non-Pizza-Hut buildings
- a synthetic rendering pipeline to generate labelled keypoints and masks

For a more general public deployment:

- likely 100+ distinct eligible roofs
- substantially larger negative coverage
- representation across countries, eras, renovations, materials, lighting, and backgrounds

Never put frames from the same building in both training and test data. That would measure memorisation rather than category generalisation.

## Failure cases that need explicit design

- heavy tree, vehicle, or sign occlusion
- roof too small in the frame
- only the symmetric front or side visible
- visually similar non-Pizza-Hut buildings
- significant roof modification
- insufficient user movement for scale
- motion blur or poor weather and lighting
- XR tracking degradation
- frame and pose timestamp mismatch
- camera intrinsics or image-rotation mismatch
- inference contention with SLAM and rendering

If content has a meaningful front/back orientation, roof geometry alone may be ambiguous. Resolve this with site metadata, facade or sign cues, or a short user-guided viewpoint change.

## Work plan

### Phase 0: definition and assets

- determine whether sites are known or arbitrary
- obtain plans, site list, and reference photography
- define required placement accuracy and target phones
- review the `@8thwall/engine` licence

### Phase 1: geometry and calibration

- build the parameterised roof
- create manual keypoint annotations
- solve pose from manually selected points
- verify XR8/Three coordinate conventions on device
- validate delayed frame-to-pose association

### Phase 2: offline detector

- train the roof mask, keypoint, and variant model
- validate on building-level holdouts
- build the hard-negative set
- measure pose after geometric fitting, not only ML metrics

### Phase 3: integrated AR

- worker-based inference pipeline
- guided multi-view collection
- robust multi-frame solve
- state machine and rejection behaviour
- continuous refinement

### Phase 4: optimisation and field testing

- test representative iPhones and Android devices
- measure sustained performance, not one-frame benchmarks
- improve hard negatives and difficult sites
- move inference into the engine/TFLite path if required

### Phase 5: optional tracking independence

- select or build an alternative SLAM/VIO foundation
- preserve the same pose-provider contract
- validate it independently of roof detection

A realistic first integrated feasibility pass is an 8–12 week effort once representative data and geometry are available. Production reliability would require further months of data collection, refinement, and device testing. A replacement cross-platform SLAM stack is a much larger parallel programme.

## Recommendation

Proceed with:

- XR8, Three.js, and React for the first implementation
- learned semantic detection plus keypoints
- a parameterised roof model
- multi-frame geometry fitting
- Rust/WASM for optimisation if desired
- an explicit abstraction around XR8 so it can be replaced
- strong rejection and user-guided movement

Do not proceed with:

- image targets as the primary recogniser
- edge matching as the only recognition system
- one-shot scale or geometry claims from one frame
- a from-scratch SLAM rewrite before the category and pose problem has been proven

The single most important next decision is whether Pizza Hut's eligible locations are known in advance. A known-site experience with roof plans is a high-feasibility engineering project; unrestricted discovery is a materially larger computer-vision product.
