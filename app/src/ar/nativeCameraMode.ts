/**
 * "Native camera" mode: the page runs inside a WKWebView stacked on top of a
 * native ARKit camera view. The native side owns the camera (getUserMedia is
 * unavailable) and pushes grayscale frames into the page through
 * `window.__pizzanetNativeFrame`, while the page renders with a fully
 * transparent background so the native camera shows through.
 */

export type NativeFrameCallback = (
  frameId: number,
  nativeTimestampMilliseconds: number,
  width: number,
  height: number,
  base64Luma: string,
  longAxisFieldOfViewDegrees?: number,
) => void;

type NativeCaptureMessageHandler = {
  postMessage(message: Record<string, unknown>): void;
};

declare global {
  interface Window {
    /** Bridge installed at session start; the native host calls it per frame. */
    __pizzanetNativeFrame?: NativeFrameCallback;
    /** Set by the native recorder only while it wants exact WKWebView input. */
    __pizzanetNativeRawCaptureEnabled?: boolean;
    /** Drains the short recording batch before Swift disables capture. */
    __pizzanetFlushNativeRawCapture?: () => void;
    /** WKWebView's page-world script-message bridge. Absent in normal browsers. */
    webkit?: {
      messageHandlers?: {
        pizzanetRawCapture?: NativeCaptureMessageHandler;
      };
    };
  }
}

export function isNativeCameraMode(): boolean {
  return new URLSearchParams(window.location.search).get("nativeCamera") === "1";
}

const rawCaptureBatchIntervalMilliseconds = 40;
let rawCaptureBatch: Array<Record<string, unknown>> = [];
let rawCaptureTimer: number | undefined;

function flushNativeRawCapture(): void {
  if (rawCaptureTimer !== undefined) {
    window.clearTimeout(rawCaptureTimer);
    rawCaptureTimer = undefined;
  }
  const handler = window.webkit?.messageHandlers?.pizzanetRawCapture;
  if (!handler || rawCaptureBatch.length === 0) {
    rawCaptureBatch = [];
    return;
  }
  const events = rawCaptureBatch;
  rawCaptureBatch = [];
  try {
    handler.postMessage({
      events,
      kind: "raw_input_batch",
      performanceTimeOriginMilliseconds: performance.timeOrigin,
    });
  } catch {
    // Recording diagnostics must never interrupt the live tracker.
  }
}

window.__pizzanetFlushNativeRawCapture = flushNativeRawCapture;

/**
 * Copies the exact values and page-clock timing seen by the live tracker back
 * to the native recorder. This is deliberately gated by Swift so normal
 * tracking pays no per-event script-message overhead. A short batch avoids
 * flooding WKWebView's main-thread script bridge at the sensor event rate.
 */
export function recordNativeRawInput(message: Record<string, unknown>): void {
  if (!window.__pizzanetNativeRawCaptureEnabled) {
    return;
  }
  if (!window.webkit?.messageHandlers?.pizzanetRawCapture) {
    return;
  }
  rawCaptureBatch.push({
    ...message,
    captureSource: "wkwebview",
  });
  if (rawCaptureTimer === undefined) {
    rawCaptureTimer = window.setTimeout(
      flushNativeRawCapture,
      rawCaptureBatchIntervalMilliseconds,
    );
  }
}

/** Sends low-rate lifecycle/permission diagnostics even when capture is idle. */
export function notifyNativeHost(message: Record<string, unknown>): void {
  const handler = window.webkit?.messageHandlers?.pizzanetRawCapture;
  if (!handler) {
    return;
  }
  try {
    handler.postMessage({
      ...message,
      performanceTimeOriginMilliseconds: performance.timeOrigin,
    });
  } catch {
    // The native host is optional; ordinary browsers have no receiver.
  }
}
