import { useEffect, useRef, useState } from "react";
import { FallbackArSession, preloadFallbackTracker } from "../../ar/FallbackArSession";
import { secureContextMessage, supportsImmersiveAr } from "../../ar/capabilities";
import { isNativeCameraMode } from "../../ar/nativeCameraMode";
import type { ArSessionController, ArStatus, TrackerDebugSettings } from "../../ar/types";
import { defaultTrackerDebugSettings } from "../../ar/types";
import { WebXrArSession } from "../../ar/WebXrArSession";
import { ArDebugPanel } from "../ArDebugPanel";

// Native-camera mode: the page runs in a WKWebView above a native ARKit camera
// view that pushes frames in; there is no camera access and the page renders
// with a transparent background.
const nativeCamera = isNativeCameraMode();

type ExperienceState = "checking" | "idle" | "starting" | "running" | "error";
type RecordingState = "idle" | "recording" | "uploading" | "uploaded" | "error";

const initialStatus: ArStatus = {
  backend: "wasm",
  confidence: 0,
  convergedLandmarks: 0,
  frames: 0,
  inliers: 0,
  keyframes: 0,
  landmarks: 0,
  linearAcceleration: 0,
  matches: 0,
  meanSceneDepthMetres: 0,
  message: "Waiting to start",
  motionSamples: 0,
  position: [0, 1.6, 0],
  relocalizations: 0,
  state: "initializing",
  textureScore: 0,
};

function errorMessage(error: unknown): string {
  if (error instanceof DOMException && error.name === "NotAllowedError") {
    return "Camera or sensor permission was denied. Allow access in browser settings, then try again.";
  }
  if (error instanceof Error) {
    return error.message;
  }
  return "The AR session could not be started.";
}

export function Home() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLDivElement>(null);
  const pointOverlayRef = useRef<HTMLCanvasElement>(null);
  const sessionRef = useRef<ArSessionController | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const [experienceState, setExperienceState] = useState<ExperienceState>("checking");
  const [status, setStatus] = useState<ArStatus>(initialStatus);
  const [recordingState, setRecordingState] = useState<RecordingState>("idle");
  const [recordingMessage, setRecordingMessage] = useState<string | null>(null);
  const [webXrAvailable, setWebXrAvailable] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [debugSettings, setDebugSettings] = useState<TrackerDebugSettings>(
    defaultTrackerDebugSettings,
  );
  // Camera and WebXR capability checks do not apply in native-camera mode:
  // the native host owns the camera and pushes frames into the page.
  const secureMessage = nativeCamera ? null : secureContextMessage();

  useEffect(() => {
    let active = true;
    preloadFallbackTracker();
    if (nativeCamera) {
      document.documentElement.classList.add("native-camera-mode");
      setExperienceState("idle");
    } else {
      void supportsImmersiveAr().then((supported) => {
        if (active) {
          setWebXrAvailable(supported);
          setExperienceState("idle");
        }
      });
    }
    return () => {
      active = false;
      document.documentElement.classList.remove("native-camera-mode");
      void sessionRef.current?.stop();
      sessionRef.current = null;
    };
  }, []);

  const updateDebugSettings = (partial: Partial<TrackerDebugSettings>) => {
    const next = { ...debugSettings, ...partial };
    setDebugSettings(next);
    const session = sessionRef.current;
    if (session instanceof FallbackArSession) {
      session.applyDebugSettings(next);
    }
  };

  const start = async () => {
    const canvas = canvasRef.current;
    const overlay = overlayRef.current;
    const pointOverlay = pointOverlayRef.current;
    const video = videoRef.current;
    if (!canvas || !overlay || !pointOverlay || (!nativeCamera && !video)) {
      return;
    }

    setError(null);
    setRecordingMessage(null);
    setRecordingState("idle");
    setExperienceState("starting");
    const session =
      !nativeCamera && webXrAvailable
        ? new WebXrArSession(canvas, overlay, setStatus)
        : new FallbackArSession(video, canvas, pointOverlay, setStatus, debugSettings);
    sessionRef.current = session;

    try {
      await session.start();
      setExperienceState("running");
    } catch (caught) {
      await session.stop().catch(() => undefined);
      sessionRef.current = null;
      if (session instanceof WebXrArSession) {
        setWebXrAvailable(false);
        setError(`${errorMessage(caught)} Try again to use the Rust/WASM fallback.`);
      } else {
        setError(errorMessage(caught));
      }
      setExperienceState("error");
    }
  };

  const stop = async () => {
    await sessionRef.current?.stop();
    sessionRef.current = null;
    setStatus(initialStatus);
    setRecordingMessage(null);
    setRecordingState("idle");
    setExperienceState("idle");
  };

  const startRecording = () => {
    try {
      sessionRef.current?.startDevRecording?.();
      setRecordingMessage(
        "Recording camera, tracker frames, and raw sensors. Move around, turn away, then return.",
      );
      setRecordingState("recording");
    } catch (caught) {
      setRecordingMessage(errorMessage(caught));
      setRecordingState("idle");
    }
  };

  const finishRecording = async () => {
    const session = sessionRef.current;
    if (!session?.finishDevRecording) return;
    setRecordingMessage("Encoding and uploading the recording…");
    setRecordingState("uploading");
    try {
      const result = await session.finishDevRecording();
      setRecordingMessage(`Saved ${result.recordingId} to ${result.savedPath}`);
      setRecordingState("uploaded");
    } catch (caught) {
      setRecordingMessage(`${errorMessage(caught)} Tap Retry upload to try again.`);
      setRecordingState("error");
    }
  };

  const isRunning = experienceState === "running";
  const isStarting = experienceState === "starting";
  const isWasmSession = status.backend === "wasm";
  const canRecord =
    import.meta.env.DEV &&
    isWasmSession &&
    Boolean(sessionRef.current?.startDevRecording);

  return (
    <main className="experience-shell">
      {!nativeCamera && (
        <video
          ref={videoRef}
          aria-hidden="true"
          className={`camera-feed ${isWasmSession && isRunning ? "is-visible" : ""}`}
        />
      )}
      <canvas ref={canvasRef} className="ar-canvas" />
      <canvas
        ref={pointOverlayRef}
        aria-hidden="true"
        className={`point-overlay ${
          isWasmSession && isRunning && debugSettings.pointOverlayEnabled
            ? "is-visible"
            : ""
        }`}
      />

      <div ref={overlayRef} className="interface-layer">
        {!isRunning && (
          <section className="launch-card" aria-labelledby="launch-title">
            <h1 id="launch-title">PizzaNet AR</h1>
            <p className="capability-line">
              {experienceState === "checking"
                ? "Checking spatial tracking…"
                : nativeCamera
                  ? "Native ARKit camera bridge"
                  : webXrAvailable
                    ? "WebXR spatial tracking available"
                    : "Rust/WASM camera + motion fallback"}
            </p>
            {secureMessage && <p className="notice notice-warning">{secureMessage}</p>}
            {error && <p className="notice notice-error">{error}</p>}
            <button
              type="button"
              className="primary-button"
              disabled={experienceState === "checking" || isStarting || Boolean(secureMessage)}
              onClick={() => void start()}
            >
              {isStarting ? "Starting camera…" : "Start AR"}
            </button>
          </section>
        )}

        {isRunning && (
          <>
            <header className="status-bar">
              <div className="status-pill">
                <span className={`status-light state-${status.state}`} />
                <span>{status.backend === "webxr" ? "WebXR 6DoF" : "Rust/WASM"}</span>
              </div>
              <div className="status-actions">
                {canRecord &&
                  (nativeCamera ? (
                    <button type="button" className="icon-button" disabled>
                      rec n/a (native)
                    </button>
                  ) : (
                    <button
                      type="button"
                      className={`icon-button ${recordingState === "recording" ? "is-recording" : ""}`}
                      disabled={recordingState === "uploading"}
                      onClick={() =>
                        recordingState === "recording" || recordingState === "error"
                          ? void finishRecording()
                          : startRecording()
                      }
                    >
                      {recordingState === "recording"
                        ? "Done"
                        : recordingState === "uploading"
                          ? "Uploading…"
                          : recordingState === "error"
                            ? "Retry upload"
                            : "Record"}
                    </button>
                  ))}
                <button
                  className="icon-button"
                  type="button"
                  onClick={() => sessionRef.current?.recenter()}
                >
                  {status.backend === "webxr" ? "Place cube" : "Recenter"}
                </button>
                <button className="icon-button" type="button" onClick={() => void stop()}>
                  Exit
                </button>
              </div>
            </header>

            <section className="guidance-panel" aria-live="polite">
              <p>{status.message}</p>
              {recordingMessage && (
                <p className={`recording-status state-${recordingState}`}>
                  {recordingMessage}
                </p>
              )}
            </section>

            {isWasmSession && (
              <ArDebugPanel
                nativeCamera={nativeCamera}
                onChange={updateDebugSettings}
                settings={debugSettings}
                status={status}
              />
            )}
          </>
        )}
      </div>
    </main>
  );
}
