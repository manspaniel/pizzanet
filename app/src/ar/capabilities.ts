type PermissionResult = "denied" | "granted";

interface PermissionCapableEventConstructor {
  requestPermission?: () => Promise<PermissionResult>;
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

export async function requestMotionPermissions(): Promise<boolean> {
  const constructors = [
    window.DeviceMotionEvent as typeof DeviceMotionEvent &
      PermissionCapableEventConstructor,
    window.DeviceOrientationEvent as typeof DeviceOrientationEvent &
      PermissionCapableEventConstructor,
  ];
  const requests = constructors.flatMap((constructor) =>
    constructor.requestPermission ? [constructor.requestPermission()] : [],
  );

  if (requests.length === 0) {
    return true;
  }

  try {
    const results = await Promise.all(requests);
    return results.every((result) => result === "granted");
  } catch {
    return false;
  }
}

export function secureContextMessage(): string | null {
  if (window.isSecureContext) {
    return null;
  }
  return "Camera, motion sensors, and WebXR require HTTPS on a phone. Use Tailscale Serve instead of plain http://danlinux:5555.";
}
