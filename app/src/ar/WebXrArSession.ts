import { ThreeArScene } from "./ThreeArScene";
import type { ArSessionController, ArStatus } from "./types";

export class WebXrArSession implements ArSessionController {
  private readonly canvas: HTMLCanvasElement;
  private hitTestSource: XRHitTestSource | null = null;
  private readonly onStatus: (status: ArStatus) => void;
  private readonly overlayRoot: HTMLElement;
  private placedFromHitTest = false;
  private scene: ThreeArScene | null = null;
  private session: XRSession | null = null;

  private readonly onResize = () => this.scene?.resize();

  constructor(
    canvas: HTMLCanvasElement,
    overlayRoot: HTMLElement,
    onStatus: (status: ArStatus) => void,
  ) {
    this.canvas = canvas;
    this.overlayRoot = overlayRoot;
    this.onStatus = onStatus;
  }

  async start(): Promise<void> {
    if (!navigator.xr) {
      throw new Error("WebXR is unavailable in this browser.");
    }

    this.scene = new ThreeArScene(this.canvas);
    this.scene.renderer.xr.enabled = true;
    this.scene.renderer.xr.setReferenceSpaceType("local-floor");
    const session = await navigator.xr.requestSession("immersive-ar", {
      domOverlay: { root: this.overlayRoot },
      optionalFeatures: ["hit-test", "dom-overlay"],
      requiredFeatures: ["local-floor"],
    });
    this.session = session;
    session.addEventListener("end", this.onSessionEnd);
    session.addEventListener("select", this.onSelect);
    await this.scene.renderer.xr.setSession(session);

    try {
      const viewerSpace = await session.requestReferenceSpace("viewer");
      this.hitTestSource = session.requestHitTestSource
        ? ((await session.requestHitTestSource({ space: viewerSpace })) ?? null)
        : null;
    } catch {
      this.hitTestSource = null;
    }

    window.addEventListener("resize", this.onResize);
    this.scene.renderer.setAnimationLoop(this.renderFrame);
    this.onStatus({
      backend: "webxr",
      confidence: 1,
      convergedLandmarks: 0,
      frames: 0,
      inliers: 0,
      keyframes: 0,
      landmarks: 0,
      linearAcceleration: 0,
      matches: 0,
      meanSceneDepthMetres: 0,
      message: this.hitTestSource
        ? "Move slowly while a floor surface is found. Tap to reposition the cube."
        : "WebXR tracking is active; floor hit testing is unavailable.",
      motionSamples: 0,
      position: [0, 0, 0],
      relocalizations: 0,
      state: "tracking",
      textureScore: 0,
    });
  }

  recenter = (): void => {
    this.scene?.placeAnchorAtReticle();
  };

  async stop(): Promise<void> {
    const session = this.session;
    this.cleanup();
    if (session) {
      await session.end().catch(() => undefined);
    }
  }

  private readonly onSelect = () => {
    this.scene?.placeAnchorAtReticle();
  };

  private readonly onSessionEnd = () => {
    this.cleanup();
  };

  private readonly renderFrame = (time: DOMHighResTimeStamp, frame?: XRFrame) => {
    if (!this.scene) {
      return;
    }
    const referenceSpace = this.scene.renderer.xr.getReferenceSpace();
    if (frame && referenceSpace && this.hitTestSource) {
      const result = frame.getHitTestResults(this.hitTestSource)[0];
      const pose = result?.getPose(referenceSpace);
      if (pose) {
        this.scene.setReticle(pose.transform.matrix);
        if (!this.placedFromHitTest) {
          this.scene.placeAnchorAtReticle();
          this.placedFromHitTest = true;
        }
      } else {
        this.scene.hideReticle();
      }
    }
    this.scene.render(time);
  };

  private cleanup(): void {
    this.hitTestSource?.cancel();
    this.hitTestSource = null;
    this.session?.removeEventListener("end", this.onSessionEnd);
    this.session?.removeEventListener("select", this.onSelect);
    window.removeEventListener("resize", this.onResize);
    this.scene?.dispose();
    this.scene = null;
    this.session = null;
  }
}
