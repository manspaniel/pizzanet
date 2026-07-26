import type { ArTracker } from "../generated/ar-tracker-wasm/ar_tracker_wasm";
import {
  requestMotionPermissions,
  type MotionPermissionOutcome,
} from "./capabilities";
import { DevRecording } from "./DevRecording";
import {
  isNativeCameraMode,
  notifyNativeHost,
  recordNativeRawInput,
} from "./nativeCameraMode";
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
  private motionPermissionOutcome: MotionPermissionOutcome | null = null;
  private motionEventReceived = false;
  private nativeLongAxisFieldOfViewDegrees: number | null = null;
  private orientationEventReceived = false;
  private pointOverlayContext: CanvasRenderingContext2D | null = null;
  private rawInputSequence = 0;
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
    recordNativeRawInput({
      acceleration: {
        x: event.acceleration?.x ?? null,
        y: event.acceleration?.y ?? null,
        z: event.acceleration?.z ?? null,
      },
      accelerationIncludingGravity: {
        x: event.accelerationIncludingGravity?.x ?? null,
        y: event.accelerationIncludingGravity?.y ?? null,
        z: event.accelerationIncludingGravity?.z ?? null,
      },
      eventTimestampMilliseconds: event.timeStamp,
      intervalMilliseconds: normalizedMotionIntervalMilliseconds(event.interval),
      isTrusted: event.isTrusted,
      kind: "device_motion",
      receiptTimestampMilliseconds,
      reportedInterval: event.interval,
      rotationRateDegreesPerSecond: {
        alpha: event.rotationRate?.alpha ?? null,
        beta: event.rotationRate?.beta ?? null,
        gamma: event.rotationRate?.gamma ?? null,
      },
      screenAngleDegrees: screenAngle(),
      screenOrientation: screenOrientationType(),
      sequence: this.rawInputSequence++,
    });
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
    if (!this.motionEventReceived) {
      this.motionEventReceived = true;
      this.reportNativeSensorReadiness();
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
    const receiptTimestampMilliseconds = performance.now();
    const compassEvent = event as DeviceOrientationEvent & {
      webkitCompassAccuracy?: number;
      webkitCompassHeading?: number;
    };
    recordNativeRawInput({
      absolute: event.absolute,
      alphaDegrees: event.alpha,
      betaDegrees: event.beta,
      eventTimestampMilliseconds: event.timeStamp,
      gammaDegrees: event.gamma,
      isTrusted: event.isTrusted,
      kind: "device_orientation",
      receiptTimestampMilliseconds,
      screenAngleDegrees: screenAngle(),
      screenOrientation: screenOrientationType(),
      sequence: this.rawInputSequence++,
      webkitCompassAccuracy: compassEvent.webkitCompassAccuracy ?? null,
      webkitCompassHeading: compassEvent.webkitCompassHeading ?? null,
    });
    this.devRecording?.recordDeviceOrientation(
      event,
      receiptTimestampMilliseconds,
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
    if (!this.orientationEventReceived) {
      this.orientationEventReceived = true;
      this.reportNativeSensorReadiness();
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
    longAxisFieldOfViewDegrees?: number,
  ): void => {
    const receiptTimestampMilliseconds = performance.now();
    if (!this.running || !this.tracker || width <= 0 || height <= 0) {
      recordNativeRawInput({
        accepted: false,
        frameHeight: height,
        frameId,
        frameWidth: width,
        kind: "native_frame_timing",
        longAxisFieldOfViewDegrees: longAxisFieldOfViewDegrees ?? null,
        nativeTimestampMilliseconds,
        receiptTimestampMilliseconds,
        rejectionReason: "tracker_not_ready",
        sequence: this.rawInputSequence++,
      });
      return;
    }
    const luma = Uint8Array.from(atob(base64Luma), (character) =>
      character.charCodeAt(0),
    );
    if (luma.length < width * height) {
      recordNativeRawInput({
        accepted: false,
        frameHeight: height,
        frameId,
        frameWidth: width,
        kind: "native_frame_timing",
        longAxisFieldOfViewDegrees: longAxisFieldOfViewDegrees ?? null,
        nativeTimestampMilliseconds,
        receiptTimestampMilliseconds,
        rejectionReason: "short_luma",
        sequence: this.rawInputSequence++,
      });
      return;
    }
    const calibrationChanged =
      typeof longAxisFieldOfViewDegrees === "number" &&
      Number.isFinite(longAxisFieldOfViewDegrees) &&
      longAxisFieldOfViewDegrees >= 30 &&
      longAxisFieldOfViewDegrees <= 130 &&
      // Landmark bearings currently share one session-wide intrinsics model.
      // Freeze the first native calibration rather than rewriting that model
      // for tiny per-frame ARKit intrinsics fluctuations.
      this.nativeLongAxisFieldOfViewDegrees === null;
    if (
      calibrationChanged &&
      this.tracker.set_long_axis_field_of_view_degrees(
        longAxisFieldOfViewDegrees,
      )
    ) {
      this.nativeLongAxisFieldOfViewDegrees = longAxisFieldOfViewDegrees;
    }
    if (
      width !== this.frameWidth ||
      height !== this.frameHeight ||
      calibrationChanged
    ) {
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
      recordNativeRawInput({
        accepted: false,
        frameHeight: height,
        frameId,
        frameWidth: width,
        kind: "native_frame_timing",
        longAxisFieldOfViewDegrees: longAxisFieldOfViewDegrees ?? null,
        minimumNativeClockOffsetMilliseconds:
          this.minimumNativeClockOffsetMilliseconds,
        nativeTimestampMilliseconds,
        performanceTimestampMilliseconds: frameTimestampMilliseconds,
        receiptTimestampMilliseconds,
        rejectionReason: "non_monotonic_mapped_timestamp",
        sequence: this.rawInputSequence++,
      });
      return;
    }
    this.lastPushedNativeTimestampMilliseconds = frameTimestampMilliseconds;
    recordNativeRawInput({
      accepted: true,
      frameHeight: height,
      frameId,
      frameWidth: width,
      kind: "native_frame_timing",
      longAxisFieldOfViewDegrees: longAxisFieldOfViewDegrees ?? null,
      minimumNativeClockOffsetMilliseconds:
        this.minimumNativeClockOffsetMilliseconds,
      nativeTimestampMilliseconds,
      performanceTimestampMilliseconds: frameTimestampMilliseconds,
      receiptTimestampMilliseconds,
      sequence: this.rawInputSequence++,
    });
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
    let motionPermissionOutcome: MotionPermissionOutcome;
    let wasmModule: Awaited<ReturnType<typeof loadWasm>>;
    try {
      [stream, motionPermissionOutcome, wasmModule] = await Promise.all([
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
    this.motionPermissionOutcome = motionPermissionOutcome;
    this.motionPermissionGranted = motionPermissionOutcome.state === "granted";
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
    const [motionPermissionOutcome, wasmModule] = await Promise.all([
      requestMotionPermissions(),
      loadWasm(),
    ]);
    this.motionPermissionOutcome = motionPermissionOutcome;
    this.motionPermissionGranted = motionPermissionOutcome.state === "granted";
    this.reportNativeSensorReadiness();
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
    const calibratedFieldOfView =
      this.nativeMode && this.nativeLongAxisFieldOfViewDegrees !== null
        ? this.nativeLongAxisFieldOfViewDegrees
        : settings.longAxisFieldOfViewDegrees;
    if (
      this.tracker.set_long_axis_field_of_view_degrees(
        calibratedFieldOfView,
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
    this.motionEventReceived = false;
    this.orientationEventReceived = false;
    this.reportNativeSensorReadiness();
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
    this.nativeLongAxisFieldOfViewDegrees = null;
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
    const state = trackingState(this.tracker.tracking_state());
    const metricScaleInitialized = this.tracker.metric_scale_initialized();
    this.scene.setCameraPose(pose);
    // Monocular metric scale cannot be observed at a stationary first frame.
    // Publish the stable provisional gauge immediately; background scale
    // diagnostics must never hide or silently resize an existing world.
    this.scene.setWorldContentVisible(true);
    this.scene.render(timestampMilliseconds);

    if (timestampMilliseconds - this.lastStatusMilliseconds >= 250) {
      const mapStats = this.tracker.map_stats();
      const frameCount = Number(this.tracker.frame_count());
      this.onStatus({
        backend: "wasm",
        confidence: this.tracker.confidence(),
        convergedLandmarks: mapStats[2] ?? 0,
        frames: frameCount,
        inliers: this.tracker.visual_inlier_count(),
        keyframes: mapStats[0] ?? 0,
        landmarks: mapStats[1] ?? 0,
        linearAcceleration: this.tracker.linear_acceleration_magnitude(),
        matches: this.tracker.visual_match_count(),
        meanSceneDepthMetres: mapStats[3] ?? 0,
        message: this.statusMessage(state, metricScaleInitialized, frameCount),
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

  private statusMessage(
    state: TrackingState,
    metricScaleInitialized: boolean,
    frameCount: number,
  ): string {
    if (!this.motionPermissionGranted) {
      const detail = this.motionPermissionOutcome?.errorName;
      return detail
        ? `Motion access failed (${detail}). Tap Exit, then Start AR and allow Motion & Orientation.`
        : "Motion access was denied. Tap Exit, then Start AR and allow Motion & Orientation.";
    }
    if (frameCount >= 30 && !this.motionEventReceived) {
      return "Motion permission succeeded, but WKWebView sent no devicemotion events.";
    }
    if (frameCount >= 30 && !this.orientationEventReceived) {
      return "Motion arrived, but WKWebView sent no deviceorientation events.";
    }
    if (state === "initializing") {
      return "Move the phone gently while orientation initializes.";
    }
    if (state === "tracking") {
      return metricScaleInitialized
        ? "Visual-inertial translation is active; metric scale is certified."
        : "Visual-inertial translation is active with a provisional scale; no late resize will occur.";
    }
    return frameCount < 30
      ? "The cube is ready. Point at texture and begin moving slowly."
      : "The cube remains available; visual tracking is limited. Point at texture and move slowly.";
  }

  private reportNativeSensorReadiness(): void {
    if (!this.nativeMode) {
      return;
    }
    notifyNativeHost({
      kind: "sensor_readiness",
      motionEventReceived: this.motionEventReceived,
      orientationEventReceived: this.orientationEventReceived,
      permissionApi: this.motionPermissionOutcome?.api ?? null,
      permissionErrorMessage: this.motionPermissionOutcome?.errorMessage ?? null,
      permissionErrorName: this.motionPermissionOutcome?.errorName ?? null,
      permissionState: this.motionPermissionOutcome?.state ?? "pending",
      receiptTimestampMilliseconds: performance.now(),
    });
  }
}
