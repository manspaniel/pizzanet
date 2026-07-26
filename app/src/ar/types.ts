export type ArBackend = "webxr" | "wasm";

export type TrackingState = "initializing" | "limited" | "tracking";

export interface RecordingUploadResult {
  recordingId: string;
  savedPath: string;
}

export interface ArStatus {
  backend: ArBackend;
  confidence: number;
  convergedLandmarks: number;
  frames: number;
  inliers: number;
  keyframes: number;
  landmarks: number;
  linearAcceleration: number;
  matches: number;
  meanSceneDepthMetres: number;
  message: string;
  motionSamples: number;
  position: [number, number, number];
  relocalizations: number;
  state: TrackingState;
  textureScore: number;
}

export type TrackerFrameWidth = 160 | 240 | 320;

export type CaptureRateHz = 10 | 20 | 30;

export interface TrackerDebugSettings {
  captureRateHz: CaptureRateHz;
  featureBudget: number;
  longAxisFieldOfViewDegrees: number;
  /** Native-camera mode only: draw each bridged luma frame as a grayscale
   * backdrop behind the AR canvas so the visible background matches the frame
   * the overlay and pose derive from, instead of the fresher live native
   * camera view. */
  nativeBackdropEnabled: boolean;
  pointOverlayEnabled: boolean;
  relocalizationEnabled: boolean;
  renderSmoothingEnabled: boolean;
  trackerFrameWidth: TrackerFrameWidth;
  visualOrientationDelayMilliseconds: number;
}

/**
 * Camera frames and browser sensor events use the same performance timeline.
 * Keep their default offset at zero; a non-zero diagnostic override must come
 * from measured acquisition latency, never from bridge delivery time.
 */
export function defaultTrackerDebugSettings(): TrackerDebugSettings {
  return {
    captureRateHz: 30,
    featureBudget: 130,
    longAxisFieldOfViewDegrees: 68,
    nativeBackdropEnabled: true,
    pointOverlayEnabled: import.meta.env.DEV,
    relocalizationEnabled: true,
    // The Rust tracker eases in optimization corrections without lagging
    // genuine camera motion. Whole-pose renderer smoothing remains available
    // as a diagnostic override, but is intentionally off by default.
    renderSmoothingEnabled: false,
    trackerFrameWidth: 240,
    visualOrientationDelayMilliseconds: 0,
  };
}

export interface ArSessionController {
  finishDevRecording?(): Promise<RecordingUploadResult>;
  recenter(): void;
  startDevRecording?(): void;
  stop(): Promise<void>;
}
