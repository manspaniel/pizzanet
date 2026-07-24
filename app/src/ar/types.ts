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
 * Mirrors the wasm tracker defaults (68 degree long-axis field of view, 40 ms
 * visual orientation delay, 130 feature budget) so the debug panel starts in
 * sync with the tracker without needing wasm getters for every value.
 */
export function defaultTrackerDebugSettings(): TrackerDebugSettings {
  return {
    captureRateHz: 30,
    featureBudget: 130,
    longAxisFieldOfViewDegrees: 68,
    nativeBackdropEnabled: true,
    pointOverlayEnabled: import.meta.env.DEV,
    relocalizationEnabled: true,
    renderSmoothingEnabled: true,
    trackerFrameWidth: 240,
    visualOrientationDelayMilliseconds: 40,
  };
}

export interface ArSessionController {
  finishDevRecording?(): Promise<RecordingUploadResult>;
  recenter(): void;
  startDevRecording?(): void;
  stop(): Promise<void>;
}
