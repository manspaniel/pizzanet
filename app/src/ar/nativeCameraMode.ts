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
) => void;

declare global {
  interface Window {
    /** Bridge installed at session start; the native host calls it per frame. */
    __pizzanetNativeFrame?: NativeFrameCallback;
  }
}

export function isNativeCameraMode(): boolean {
  return new URLSearchParams(window.location.search).get("nativeCamera") === "1";
}
