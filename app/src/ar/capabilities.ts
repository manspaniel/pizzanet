type PermissionResult = "denied" | "granted";

interface PermissionCapableEventConstructor {
  requestPermission?: () => Promise<PermissionResult>;
}

export interface MotionPermissionOutcome {
  api: "DeviceMotionEvent" | "DeviceOrientationEvent" | "implicit";
  errorMessage?: string;
  errorName?: string;
  state: PermissionResult;
}

export async function supportsImmersiveAr(): Promise<boolean> {
  if (!navigator.xr || !window.isSecureContext) {
    return false;
  }

  try {
    return await navigator.xr.isSessionSupported("immersive-ar");
  } catch {
    return false;
  }
}

export async function requestMotionPermissions(): Promise<MotionPermissionOutcome> {
  // WebKit exposes one combined per-origin motion/orientation permission. Two
  // concurrent requests race the same controller and can make one reject even
  // after the other succeeds. Prefer motion and use orientation only as a
  // compatibility fallback.
  const motion = window.DeviceMotionEvent as typeof DeviceMotionEvent &
    PermissionCapableEventConstructor;
  const orientation = window.DeviceOrientationEvent as typeof DeviceOrientationEvent &
    PermissionCapableEventConstructor;
  const request = motion.requestPermission
    ? {
        api: "DeviceMotionEvent" as const,
        run: () => motion.requestPermission!(),
      }
    : orientation.requestPermission
      ? {
          api: "DeviceOrientationEvent" as const,
          run: () => orientation.requestPermission!(),
        }
      : null;
  if (!request) {
    return { api: "implicit", state: "granted" };
  }
  try {
    return { api: request.api, state: await request.run() };
  } catch (error) {
    return {
      api: request.api,
      errorMessage: error instanceof Error ? error.message : String(error),
      errorName: error instanceof Error ? error.name : "UnknownError",
      state: "denied",
    };
  }
}

export function secureContextMessage(): string | null {
  if (window.isSecureContext) {
    return null;
  }
  return "Camera, motion sensors, and WebXR require HTTPS on a phone. Use Tailscale Serve instead of plain http://danlinux:5555.";
}
