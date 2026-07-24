import type { ArTracker } from "../generated/ar-tracker-wasm/ar_tracker_wasm";
import { requestMotionPermissions } from "./capabilities";
import { DevRecording } from "./DevRecording";
import { isNativeCameraMode } from "./nativeCameraMode";
import { ThreeArScene } from "./ThreeArScene";
import type {
  ArSessionController,
  ArStatus,
  RecordingUploadResult,
  TrackerDebugSettings,
  TrackingState,
} from "./types";
import { defaultTrackerDebugSettings } from "./types";

const radiansPerDegree = Math.PI / 180;
// Captures fire from requestVideoFrameCallback, so frame spacing jitters by a
// few milliseconds around the camera rate. The tolerance keeps a 30 Hz target
// from skipping every other 30 fps camera frame.
const captureIntervalToleranceMilliseconds = 5;
// A native-frame clock offset this far below the running minimum cannot be
// explained by delivery jitter; treat it as a clock jump and re-estimate.
const nativeClockJumpMilliseconds = 500;
const trackedPointRadiusCssPixels = 2.5;
const trackedPointStateColors = [
  "#9e9e9e", // 0: new detection
  "#3fe27f", // 1: tracked
  "#ffa53d", // 2: anchored landmark
  "#4fd8e8", // 3: anchored with converged depth
];

let wasmModulePromise:
  | Promise<typeof import("../generated/ar-tracker-wasm/ar_tracker_wasm")>
  | undefined;

async function loadWasm() {
  wasmModulePromise ??= import(
    "../generated/ar-tracker-wasm/ar_tracker_wasm"
  ).then(async (module) => {
    await module.default();
    return module;
  });
  return wasmModulePromise;
}

export function preloadFallbackTracker(): void {
  void loadWasm();
}

function screenAngle(): number {
  const legacyOrientation = window as Window & { orientation?: number };
  return window.screen.orientation?.angle ?? legacyOrientation.orientation ?? 0;
}

function screenOrientationCode(): number {
  const angle = ((screenAngle() % 360) + 360) % 360;
  if (angle === 90) return 1;
  if (angle === 180) return 2;
  if (angle === 270) return 3;
  return 0;
}

function screenOrientationType(): string {
  return window.screen.orientation?.type ?? "unknown";
}

function isAppleMobile(): boolean {
  return (
    /iPad|iPhone|iPod/.test(navigator.userAgent) ||
    (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1)
  );
}

function normalizedMotionIntervalMilliseconds(interval: number): number {
  return Number.isFinite(interval) && interval > 0 && interval < 1
    ? interval * 1_000
    : interval;
}

function trackingState(code: number): TrackingState {
  if (code === 2) return "tracking";
  if (code === 1) return "limited";
  return "initializing";
}

export class FallbackArSession implements ArSessionController {
  private animationFrame = 0;
  private backdropContext: CanvasRenderingContext2D | null = null;
  private backdropImageData: ImageData | null = null;
  private backdropSourceContext: CanvasRenderingContext2D | null = null;
  private captureContext: CanvasRenderingContext2D | null = null;
  private debugSettings: TrackerDebugSettings;
  private devRecording: DevRecording | null = null;
  private frameId = 0;
  private frameHeight = 135;
  private frameWidth: number;
  private lastCaptureMilliseconds = 0;
  private lastPushedNativeTimestampMilliseconds = Number.NEGATIVE_INFINITY;
  private lastStatusMilliseconds = 0;
  private minimumNativeClockOffsetMilliseconds = Number.POSITIVE_INFINITY;
  private minimumCaptureIntervalMilliseconds: number;
  private motionPermissionGranted = false;
  private pointOverlayContext: CanvasRenderingContext2D | null = null;
  private running = false;
  private scene: ThreeArScene | null = null;
  private stream: MediaStream | null = null;
  private tracker: ArTracker | null = null;
  private videoFrameCallback = 0;
  private readonly backdropCanvas: HTMLCanvasElement | null;
  private readonly canvas: HTMLCanvasElement;
  private readonly nativeMode = isNativeCameraMode();
  private readonly onStatus: (status: ArStatus) => void;
  private readonly pointOverlayCanvas: HTMLCanvasElement;
  private readonly video: HTMLVideoElement | null;

  private readonly onDeviceMotion = (event: DeviceMotionEvent) => {
    const receiptTimestampMilliseconds = performance.now();
    this.devRecording?.recordDeviceMotion(
      event,
      receiptTimestampMilliseconds,
      screenAngle(),
      screenOrientationType(),
    );
    if (!this.tracker || !event.rotationRate || !event.accelerationIncludingGravity) {
      return;
    }
    const rotation = event.rotationRate;
    const force = event.accelerationIncludingGravity;
    const acceleration = event.acceleration;
    if (
      rotation.alpha === null ||
      rotation.beta === null ||
      rotation.gamma === null ||
      force.x === null ||
      force.y === null ||
      force.z === null
    ) {
      return;
    }
    const sign = isAppleMobile() ? -1 : 1;
    const gyro = isAppleMobile()
      ? [rotation.alpha, rotation.beta, rotation.gamma]
      : [rotation.beta, rotation.gamma, rotation.alpha];
    this.tracker.push_motion_sample(
      event.timeStamp,
      receiptTimestampMilliseconds,
      normalizedMotionIntervalMilliseconds(event.interval),
      gyro[0] * radiansPerDegree,
      gyro[1] * radiansPerDegree,
      gyro[2] * radiansPerDegree,
      force.x * sign,
      force.y * sign,
      force.z * sign,
      acceleration?.x === null || acceleration?.x === undefined
        ? Number.NaN
        : acceleration.x * sign,
      acceleration?.y === null || acceleration?.y === undefined
        ? Number.NaN
        : acceleration.y * sign,
      acceleration?.z === null || acceleration?.z === undefined
        ? Number.NaN
        : acceleration.z * sign,
      screenOrientationCode(),
    );
  };

  private readonly onDeviceOrientation = (event: DeviceOrientationEvent) => {
    this.devRecording?.recordDeviceOrientation(
      event,
      performance.now(),
      screenAngle(),
      screenOrientationType(),
    );
    if (
      !this.tracker ||
      event.alpha === null ||
      event.beta === null ||
      event.gamma === null
    ) {
      return;
    }
    this.tracker.push_device_orientation(
      event.alpha,
      event.beta,
      event.gamma,
      screenAngle(),
      event.timeStamp,
    );
  };

  private readonly onResize = () => this.scene?.resize();

  private readonly onVideoFrame = (
    _nowMilliseconds: DOMHighResTimeStamp,
    metadata: VideoFrameCallbackMetadata,
  ) => {
    if (!this.running || !this.video) {
      this.videoFrameCallback = 0;
      return;
    }
    const captureTime = metadata.captureTime;
    const frameTimestampMilliseconds =
      typeof captureTime === "number" && Number.isFinite(captureTime)
        ? captureTime
        : metadata.presentationTime;
    if (
      frameTimestampMilliseconds - this.lastCaptureMilliseconds >=
      this.minimumCaptureIntervalMilliseconds
    ) {
      this.captureFrame(frameTimestampMilliseconds);
      this.lastCaptureMilliseconds = frameTimestampMilliseconds;
    }
    if (this.running) {
      this.videoFrameCallback = this.video.requestVideoFrameCallback(this.onVideoFrame);
    } else {
      this.videoFrameCallback = 0;
    }
  };

  /**
   * Bridge entry point for native-camera mode. The native ARKit host pushes
   * already-downscaled grayscale frames (throttled to 30 Hz on the native
   * side), so every call is processed.
   *
   * Frames are stamped by mapping the native capture timestamp onto the
   * page's `performance.now()` clock instead of using receipt time: bridge
   * delivery adds ~50-100 ms of jittery latency, and folding that into the
   * frame clock misaligns frames with the page's devicemotion /
   * deviceorientation events — during rotation the tracker then fuses
   * wrong-instant orientation and manufactures false translation. The running
   * minimum of `receipt - nativeTimestamp` estimates the constant clock
   * offset, because the minimum-latency delivery carries the least bridge
   * delay.
   */
  private readonly onNativeFrame = (
    frameId: number,
    nativeTimestampMilliseconds: number,
    width: number,
    height: number,
    base64Luma: string,
  ): void => {
    const receiptTimestampMilliseconds = performance.now();
    if (!this.running || !this.tracker || width <= 0 || height <= 0) {
      return;
    }
    const luma = Uint8Array.from(atob(base64Luma), (character) =>
      character.charCodeAt(0),
    );
    if (luma.length < width * height) {
      return;
    }
    if (width !== this.frameWidth || height !== this.frameHeight) {
      this.frameWidth = width;
      this.frameHeight = height;
      this.configureSceneProjection();
    }
    const offsetMilliseconds =
      receiptTimestampMilliseconds - nativeTimestampMilliseconds;
    if (
      offsetMilliseconds <
      this.minimumNativeClockOffsetMilliseconds - nativeClockJumpMilliseconds
    ) {
      // An offset far below the running minimum means one of the clocks
      // jumped; restart the estimate rather than mapping across the jump.
      this.minimumNativeClockOffsetMilliseconds = offsetMilliseconds;
    } else if (offsetMilliseconds < this.minimumNativeClockOffsetMilliseconds) {
      this.minimumNativeClockOffsetMilliseconds = offsetMilliseconds;
    }
    if (this.debugSettings.nativeBackdropEnabled) {
      this.drawBackdrop(luma, width, height);
    }
    const frameTimestampMilliseconds =
      nativeTimestampMilliseconds + this.minimumNativeClockOffsetMilliseconds;
    // The tracker requires strictly increasing frame timestamps, and a drop
    // in the offset estimate can map one frame slightly behind the previous
    // push; skip it.
    if (frameTimestampMilliseconds <= this.lastPushedNativeTimestampMilliseconds) {
      return;
    }
    this.lastPushedNativeTimestampMilliseconds = frameTimestampMilliseconds;
    this.tracker.push_luma_frame(
      frameId,
      frameTimestampMilliseconds,
      width,
      height,
      luma,
    );
    if (this.debugSettings.pointOverlayEnabled) {
      this.drawTrackedPoints();
    }
  };

  constructor(
    video: HTMLVideoElement | null,
    canvas: HTMLCanvasElement,
    pointOverlayCanvas: HTMLCanvasElement,
    backdropCanvas: HTMLCanvasElement | null,
    onStatus: (status: ArStatus) => void,
    initialDebugSettings: TrackerDebugSettings = defaultTrackerDebugSettings(),
  ) {
    this.video = video;
    this.canvas = canvas;
    this.pointOverlayCanvas = pointOverlayCanvas;
    this.backdropCanvas = backdropCanvas;
    this.onStatus = onStatus;
    this.debugSettings = { ...initialDebugSettings };
    this.frameWidth = this.debugSettings.trackerFrameWidth;
    this.minimumCaptureIntervalMilliseconds = FallbackArSession.captureIntervalFor(
      this.debugSettings.captureRateHz,
    );
  }

  private static captureIntervalFor(captureRateHz: number): number {
    return 1_000 / captureRateHz - captureIntervalToleranceMilliseconds;
  }

  async start(): Promise<void> {
    if (this.nativeMode) {
      await this.startWithNativeFrames();
      return;
    }
    const video = this.video;
    if (!video) {
      throw new Error("Camera mode requires a video element.");
    }
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("This browser does not expose camera capture.");
    }

    // Start every protected request before awaiting so iOS sees the originating user gesture.
    const streamPromise = navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: "environment" },
        height: { ideal: 1080 },
        width: { ideal: 1920 },
      },
    });
    const motionPermissionPromise = requestMotionPermissions();
    const modulePromise = loadWasm();

    let stream: MediaStream;
    let motionPermissionGranted: boolean;
    let wasmModule: Awaited<ReturnType<typeof loadWasm>>;
    try {
      [stream, motionPermissionGranted, wasmModule] = await Promise.all([
        streamPromise,
        motionPermissionPromise,
        modulePromise,
      ]);
    } catch (error) {
      void streamPromise
        .then((pendingStream) =>
          pendingStream.getTracks().forEach((track) => track.stop()),
        )
        .catch(() => undefined);
      throw error;
    }
    this.stream = stream;
    this.motionPermissionGranted = motionPermissionGranted;
    this.tracker = new wasmModule.ArTracker();

    video.srcObject = stream;
    video.muted = true;
    video.playsInline = true;
    await video.play();

    const captureCanvas = document.createElement("canvas");
    this.captureContext = captureCanvas.getContext("2d", {
      alpha: false,
      willReadFrequently: true,
    });
    if (!this.captureContext) {
      throw new Error("The browser could not create a camera capture surface.");
    }
    this.applyTrackerFrameSize(this.debugSettings.trackerFrameWidth);
    this.pointOverlayContext = this.pointOverlayCanvas.getContext("2d");

    this.scene = new ThreeArScene(this.canvas);
    this.configureSceneProjection();
    this.applyDebugSettings(this.debugSettings);
    window.addEventListener("devicemotion", this.onDeviceMotion);
    window.addEventListener("deviceorientation", this.onDeviceOrientation);
    window.addEventListener("resize", this.onResize);
    this.canvas.addEventListener("click", this.recenter);
    this.running = true;
    if ("requestVideoFrameCallback" in video) {
      this.videoFrameCallback = video.requestVideoFrameCallback(this.onVideoFrame);
    }
    this.animationFrame = requestAnimationFrame(this.renderFrame);
  }

  /**
   * Native-camera mode start: no getUserMedia and no video element. The native
   * ARKit host owns the camera and pushes grayscale frames through
   * `window.__pizzanetNativeFrame`; the scene renders with a transparent clear
   * color over the bridged-frame backdrop canvas (or, when the backdrop is
   * toggled off, directly over the live native camera view).
   */
  private async startWithNativeFrames(): Promise<void> {
    const [motionPermissionGranted, wasmModule] = await Promise.all([
      requestMotionPermissions(),
      loadWasm(),
    ]);
    this.motionPermissionGranted = motionPermissionGranted;
    this.tracker = new wasmModule.ArTracker();
    this.pointOverlayContext = this.pointOverlayCanvas.getContext("2d");
    if (this.backdropCanvas) {
      this.backdropContext = this.backdropCanvas.getContext("2d");
      this.backdropSourceContext = document
        .createElement("canvas")
        .getContext("2d");
    }

    this.scene = new ThreeArScene(this.canvas);
    // The renderer already clears to transparent; keep the scene background
    // unset so nothing paints over the native camera.
    this.scene.scene.background = null;
    this.scene.renderer.setClearColor(0x000000, 0);
    // Frame dimensions arrive with the first pushed frame; the projection is
    // configured then.
    this.frameWidth = 0;
    this.frameHeight = 0;
    this.applyDebugSettings(this.debugSettings);
    window.addEventListener("devicemotion", this.onDeviceMotion);
    window.addEventListener("deviceorientation", this.onDeviceOrientation);
    window.addEventListener("resize", this.onResize);
    this.canvas.addEventListener("click", this.recenter);
    window.__pizzanetNativeFrame = this.onNativeFrame;
    this.running = true;
    this.animationFrame = requestAnimationFrame(this.renderFrame);
  }

  /**
   * Applies the current field of view to the virtual camera projection using
   * the source frame dimensions: the video element in camera mode, or the
   * dimensions of the last pushed frame in native-camera mode.
   */
  private configureSceneProjection(): void {
    if (!this.tracker || !this.scene) {
      return;
    }
    const { width, height } = this.sourceDimensions();
    if (width === 0 || height === 0) {
      return;
    }
    this.scene.configureVideoProjection(
      this.tracker.horizontal_field_of_view_degrees(width, height),
      width / Math.max(height, 1),
    );
  }

  /** Full-resolution source dimensions that display cover-fit math maps from. */
  private sourceDimensions(): { width: number; height: number } {
    if (this.nativeMode) {
      return { width: this.frameWidth, height: this.frameHeight };
    }
    return {
      width: this.video?.videoWidth ?? 0,
      height: this.video?.videoHeight ?? 0,
    };
  }

  recenter = (): void => {
    this.tracker?.recenter();
  };

  /**
   * Applies the full debug-panel state. Every setter is idempotent, so the UI
   * can call this on any single change. A tracker resolution change is
   * deferred while a recording is active because the recording format requires
   * constant luma dimensions.
   */
  applyDebugSettings(settings: TrackerDebugSettings): void {
    this.debugSettings = { ...settings };
    this.minimumCaptureIntervalMilliseconds = FallbackArSession.captureIntervalFor(
      settings.captureRateHz,
    );
    if (!settings.pointOverlayEnabled) {
      this.clearPointOverlay();
    }
    if (!settings.nativeBackdropEnabled) {
      this.clearBackdrop();
    }
    this.scene?.setPoseSmoothingEnabled(settings.renderSmoothingEnabled);
    if (!this.devRecording) {
      this.applyTrackerFrameSize(settings.trackerFrameWidth);
    }
    if (!this.tracker) {
      return;
    }
    this.tracker.set_visual_orientation_delay_milliseconds(
      settings.visualOrientationDelayMilliseconds,
    );
    this.tracker.set_feature_budget(settings.featureBudget);
    this.tracker.set_relocalization_enabled(settings.relocalizationEnabled);
    if (
      this.tracker.set_long_axis_field_of_view_degrees(
        settings.longAxisFieldOfViewDegrees,
      )
    ) {
      // Keep the virtual camera projection consistent with the source FOV.
      this.configureSceneProjection();
    }
  }

  startDevRecording(): void {
    if (!import.meta.env.DEV) {
      throw new Error("Recording is only available from the Vite development server.");
    }
    if (!this.stream || !this.tracker || !this.video) {
      throw new Error("Start the camera session before recording.");
    }
    if (this.devRecording) {
      throw new Error("A recording is already active or waiting to upload.");
    }
    this.devRecording = new DevRecording(this.stream, {
      horizontalFieldOfViewDegrees: this.tracker.horizontal_field_of_view_degrees(
        this.video.videoWidth,
        this.video.videoHeight,
      ),
      longAxisFieldOfViewDegrees: this.tracker.long_axis_field_of_view_degrees(),
      targetCaptureRateHz: this.debugSettings.captureRateHz,
      trackerFrameHeight: this.frameHeight,
      trackerFrameWidth: this.frameWidth,
      videoHeight: this.video.videoHeight,
      videoWidth: this.video.videoWidth,
    });
    this.devRecording.start();
  }

  async finishDevRecording(): Promise<RecordingUploadResult> {
    if (!this.devRecording) {
      throw new Error("There is no recording to finish.");
    }
    const result = await this.devRecording.finishAndUpload();
    this.devRecording = null;
    // Apply any resolution change that arrived while the recording was locked.
    this.applyTrackerFrameSize(this.debugSettings.trackerFrameWidth);
    return result;
  }

  async stop(): Promise<void> {
    this.running = false;
    cancelAnimationFrame(this.animationFrame);
    if (this.videoFrameCallback !== 0 && this.video) {
      this.video.cancelVideoFrameCallback(this.videoFrameCallback);
      this.videoFrameCallback = 0;
    }
    if (window.__pizzanetNativeFrame === this.onNativeFrame) {
      delete window.__pizzanetNativeFrame;
    }
    this.devRecording?.cancel();
    this.devRecording = null;
    window.removeEventListener("devicemotion", this.onDeviceMotion);
    window.removeEventListener("deviceorientation", this.onDeviceOrientation);
    window.removeEventListener("resize", this.onResize);
    this.canvas.removeEventListener("click", this.recenter);
    this.stream?.getTracks().forEach((track) => track.stop());
    if (this.video) {
      this.video.pause();
      this.video.srcObject = null;
    }
    this.clearPointOverlay();
    this.clearBackdrop();
    this.scene?.dispose();
    this.tracker?.free();
    this.backdropContext = null;
    this.backdropImageData = null;
    this.backdropSourceContext = null;
    this.captureContext = null;
    this.pointOverlayContext = null;
    this.scene = null;
    this.stream = null;
    this.tracker = null;
  }

  private readonly renderFrame = (timestampMilliseconds: number) => {
    if (!this.running || !this.scene || !this.tracker) {
      return;
    }

    if (
      !this.nativeMode &&
      this.videoFrameCallback === 0 &&
      timestampMilliseconds - this.lastCaptureMilliseconds >=
        this.minimumCaptureIntervalMilliseconds
    ) {
      this.captureFrame(timestampMilliseconds);
      this.lastCaptureMilliseconds = timestampMilliseconds;
    }

    const pose = this.tracker.pose();
    this.scene.setCameraPose(pose);
    this.scene.render(timestampMilliseconds);

    if (timestampMilliseconds - this.lastStatusMilliseconds >= 250) {
      const state = trackingState(this.tracker.tracking_state());
      const mapStats = this.tracker.map_stats();
      this.onStatus({
        backend: "wasm",
        confidence: this.tracker.confidence(),
        convergedLandmarks: mapStats[2] ?? 0,
        frames: Number(this.tracker.frame_count()),
        inliers: this.tracker.visual_inlier_count(),
        keyframes: mapStats[0] ?? 0,
        landmarks: mapStats[1] ?? 0,
        linearAcceleration: this.tracker.linear_acceleration_magnitude(),
        matches: this.tracker.visual_match_count(),
        meanSceneDepthMetres: mapStats[3] ?? 0,
        message: this.statusMessage(state),
        motionSamples: Number(this.tracker.motion_sample_count()),
        position: [pose[0], pose[1], pose[2]],
        relocalizations: Number(this.tracker.visual_relocalization_count()),
        state,
        textureScore: this.tracker.latest_texture_score(),
      });
      this.lastStatusMilliseconds = timestampMilliseconds;
    }

    this.animationFrame = requestAnimationFrame(this.renderFrame);
  };

  private applyTrackerFrameSize(targetFrameWidth: number): void {
    const captureCanvas = this.captureContext?.canvas;
    if (!captureCanvas || !this.video || this.video.videoWidth === 0) {
      return;
    }
    this.frameWidth = targetFrameWidth;
    this.frameHeight = Math.max(
      90,
      Math.round(
        this.frameWidth *
          (this.video.videoHeight / Math.max(this.video.videoWidth, 1)),
      ),
    );
    if (
      captureCanvas.width !== this.frameWidth ||
      captureCanvas.height !== this.frameHeight
    ) {
      captureCanvas.width = this.frameWidth;
      captureCanvas.height = this.frameHeight;
    }
  }

  private captureFrame(timestampMilliseconds: number): void {
    if (
      !this.captureContext ||
      !this.tracker ||
      !this.video ||
      this.video.readyState < 2
    ) {
      return;
    }
    this.captureContext.drawImage(
      this.video,
      0,
      0,
      this.frameWidth,
      this.frameHeight,
    );
    const rgba = this.captureContext.getImageData(
      0,
      0,
      this.frameWidth,
      this.frameHeight,
    ).data;
    const luma = new Uint8Array(this.frameWidth * this.frameHeight);
    for (let source = 0, target = 0; source < rgba.length; source += 4, target += 1) {
      luma[target] =
        (rgba[source] * 77 + rgba[source + 1] * 150 + rgba[source + 2] * 29) >>
        8;
    }
    const textureScore = this.tracker.push_luma_frame(
      this.frameId,
      timestampMilliseconds,
      this.frameWidth,
      this.frameHeight,
      luma,
    );
    const pose = this.tracker.pose();
    this.devRecording?.recordTrackerFrame({
      confidence: this.tracker.confidence(),
      frameHeight: this.frameHeight,
      frameId: this.frameId,
      frameWidth: this.frameWidth,
      inliers: this.tracker.visual_inlier_count(),
      isKeyframe: this.tracker.latest_visual_keyframe_id() === this.frameId,
      keyframeCount: Number(this.tracker.visual_keyframe_count()),
      keyframeId: this.tracker.latest_visual_keyframe_id(),
      matches: this.tracker.visual_match_count(),
      performanceTimestampMilliseconds: timestampMilliseconds,
      pose: Array.from(pose),
      relocalizationCount: Number(this.tracker.visual_relocalization_count()),
      textureScore,
      trackingState: this.tracker.tracking_state(),
    }, luma);
    this.frameId += 1;
    if (this.debugSettings.pointOverlayEnabled) {
      this.drawTrackedPoints();
    }
  }

  /**
   * Draws tracker.tracked_points() onto the overlay canvas in display space.
   *
   * The video element uses `object-fit: cover`, so the camera frame is
   * uniformly scaled by `max(displayWidth / videoWidth, displayHeight /
   * videoHeight)` and centre-cropped. The tracker frame is a downscaled copy
   * of the full camera frame, so tracker coordinates first scale up by
   * `videoWidth / frameWidth` (and `videoHeight / frameHeight`, which only
   * differs by the height rounding) before the cover transform applies.
   *
   * In native-camera mode there is no video element: the native camera view
   * fills the viewport behind the page, so the same cover math applies with
   * the pushed frame's dimensions as the source (the tracker-to-source scale
   * is then 1).
   */
  private drawTrackedPoints(): void {
    const context = this.pointOverlayContext;
    const tracker = this.tracker;
    if (!context || !tracker) {
      return;
    }
    const overlayCanvas = context.canvas;
    const displayWidth = overlayCanvas.clientWidth;
    const displayHeight = overlayCanvas.clientHeight;
    const { width: videoWidth, height: videoHeight } = this.sourceDimensions();
    if (
      displayWidth === 0 ||
      displayHeight === 0 ||
      videoWidth === 0 ||
      videoHeight === 0
    ) {
      return;
    }
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    const deviceWidth = Math.round(displayWidth * pixelRatio);
    const deviceHeight = Math.round(displayHeight * pixelRatio);
    if (overlayCanvas.width !== deviceWidth || overlayCanvas.height !== deviceHeight) {
      overlayCanvas.width = deviceWidth;
      overlayCanvas.height = deviceHeight;
    }
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.clearRect(0, 0, displayWidth, displayHeight);

    const coverScale = Math.max(
      displayWidth / videoWidth,
      displayHeight / videoHeight,
    );
    const coverOffsetX = (displayWidth - videoWidth * coverScale) / 2;
    const coverOffsetY = (displayHeight - videoHeight * coverScale) / 2;
    const trackerToDisplayX = (videoWidth / this.frameWidth) * coverScale;
    const trackerToDisplayY = (videoHeight / this.frameHeight) * coverScale;

    const points = tracker.tracked_points();
    for (let index = 0; index + 2 < points.length; index += 3) {
      const displayX = coverOffsetX + points[index] * trackerToDisplayX;
      const displayY = coverOffsetY + points[index + 1] * trackerToDisplayY;
      const state = points[index + 2];
      context.fillStyle =
        trackedPointStateColors[state] ?? trackedPointStateColors[0];
      context.beginPath();
      context.arc(displayX, displayY, trackedPointRadiusCssPixels, 0, Math.PI * 2);
      context.fill();
    }
  }

  /**
   * Draws a received bridged luma frame as a grayscale backdrop behind the
   * Three.js canvas and point overlay. The overlay dots and cube pose derive
   * from bridged frames that are ~2 frames older than the live native camera
   * view, so compositing them over the live view makes them visibly lag;
   * drawing the bridged frame itself as the backdrop keeps everything the
   * user sees derived from the same frame. Uses the same cover-fit math as
   * drawTrackedPoints so backdrop and dots align pixel-for-pixel.
   */
  private drawBackdrop(luma: Uint8Array, width: number, height: number): void {
    const context = this.backdropContext;
    const sourceContext = this.backdropSourceContext;
    if (!context || !sourceContext) {
      return;
    }
    const sourceCanvas = sourceContext.canvas;
    if (sourceCanvas.width !== width || sourceCanvas.height !== height) {
      sourceCanvas.width = width;
      sourceCanvas.height = height;
      this.backdropImageData = null;
    }
    this.backdropImageData ??= sourceContext.createImageData(width, height);
    const rgba = this.backdropImageData.data;
    const pixelCount = width * height;
    for (let index = 0; index < pixelCount; index += 1) {
      const value = luma[index];
      const offset = index * 4;
      rgba[offset] = value;
      rgba[offset + 1] = value;
      rgba[offset + 2] = value;
      rgba[offset + 3] = 255;
    }
    sourceContext.putImageData(this.backdropImageData, 0, 0);

    const backdropCanvas = context.canvas;
    const displayWidth = backdropCanvas.clientWidth;
    const displayHeight = backdropCanvas.clientHeight;
    if (displayWidth === 0 || displayHeight === 0) {
      return;
    }
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    const deviceWidth = Math.round(displayWidth * pixelRatio);
    const deviceHeight = Math.round(displayHeight * pixelRatio);
    if (
      backdropCanvas.width !== deviceWidth ||
      backdropCanvas.height !== deviceHeight
    ) {
      backdropCanvas.width = deviceWidth;
      backdropCanvas.height = deviceHeight;
    }
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    const coverScale = Math.max(displayWidth / width, displayHeight / height);
    const coverOffsetX = (displayWidth - width * coverScale) / 2;
    const coverOffsetY = (displayHeight - height * coverScale) / 2;
    context.drawImage(
      sourceCanvas,
      coverOffsetX,
      coverOffsetY,
      width * coverScale,
      height * coverScale,
    );
  }

  private clearBackdrop(): void {
    const context = this.backdropContext;
    if (context) {
      context.setTransform(1, 0, 0, 1, 0, 0);
      context.clearRect(0, 0, context.canvas.width, context.canvas.height);
    }
  }

  private clearPointOverlay(): void {
    const context = this.pointOverlayContext;
    if (context) {
      context.setTransform(1, 0, 0, 1, 0, 0);
      context.clearRect(0, 0, context.canvas.width, context.canvas.height);
    }
  }

  private statusMessage(state: TrackingState): string {
    if (!this.motionPermissionGranted) {
      return "Camera active; motion access was denied.";
    }
    if (state === "initializing") {
      return "Move the phone gently while orientation initializes.";
    }
    if (state === "tracking") {
      return "Visual-inertial translation is active. Move slowly around the cube.";
    }
    return "Heading is live. Point at a textured scene and move slowly to initialize translation.";
  }
}
