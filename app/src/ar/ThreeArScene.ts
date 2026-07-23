import * as THREE from "three";

const cubeColor = new THREE.Color(0x8d35ff);
// Device orientation is already smooth, and smoothing it here lags the video
// during rotation, so orientation is applied directly. Position keeps a short
// time constant to hide translation-solve jitter.
const smoothedPositionTimeConstantSeconds = 0.05;
const smoothedOrientationTimeConstantSeconds = 0;

export class ThreeArScene {
  readonly camera = new THREE.PerspectiveCamera(60, 1, 0.01, 100);
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();

  private readonly anchor = new THREE.Group();
  private readonly reticle: THREE.Mesh;
  private readonly targetCameraPosition = new THREE.Vector3(0, 1.6, 0);
  private readonly targetCameraQuaternion = new THREE.Quaternion();
  private lastRenderTimestampMilliseconds: number | null = null;
  private positionSmoothingTimeSeconds = smoothedPositionTimeConstantSeconds;
  private orientationSmoothingTimeSeconds = smoothedOrientationTimeConstantSeconds;
  private sourceHorizontalFovDegrees: number | null = null;
  private sourceVideoAspect = 1;
  private usesExternalCameraPose = false;

  constructor(canvas: HTMLCanvasElement) {
    this.renderer = new THREE.WebGLRenderer({
      alpha: true,
      antialias: true,
      canvas,
      powerPreference: "high-performance",
      premultipliedAlpha: false,
    });
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;

    this.camera.position.set(0, 1.6, 0);
    this.scene.add(this.camera);

    const hemisphere = new THREE.HemisphereLight(0xffffff, 0x36205f, 1.8);
    this.scene.add(hemisphere);

    const directional = new THREE.DirectionalLight(0xffffff, 2.2);
    directional.position.set(4, 8, 5);
    directional.castShadow = true;
    directional.shadow.mapSize.set(1024, 1024);
    this.scene.add(directional);

    const cube = new THREE.Mesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshStandardMaterial({
        color: cubeColor,
        metalness: 0.15,
        roughness: 0.3,
      }),
    );
    cube.castShadow = true;
    cube.position.y = 0.5;
    this.anchor.add(cube);

    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(cube.geometry),
      new THREE.LineBasicMaterial({ color: 0xe9d8ff }),
    );
    edges.position.copy(cube.position);
    this.anchor.add(edges);

    const shadow = new THREE.Mesh(
      new THREE.PlaneGeometry(2000, 2000),
      new THREE.ShadowMaterial({ color: 0x16072b, opacity: 0.42 }),
    );
    shadow.rotation.x = -Math.PI / 2;
    shadow.receiveShadow = true;
    this.anchor.add(shadow);
    this.anchor.position.set(0, 0, -2.5);
    this.scene.add(this.anchor);

    this.reticle = new THREE.Mesh(
      new THREE.RingGeometry(0.11, 0.15, 40).rotateX(-Math.PI / 2),
      new THREE.MeshBasicMaterial({ color: 0xcfa8ff, side: THREE.DoubleSide }),
    );
    this.reticle.matrixAutoUpdate = false;
    this.reticle.visible = false;
    this.scene.add(this.reticle);

    this.resize();
  }

  resize(): void {
    const width = window.innerWidth;
    const height = window.innerHeight;
    this.camera.aspect = width / Math.max(height, 1);
    if (this.sourceHorizontalFovDegrees !== null) {
      const horizontalHalfTangent = Math.tan(
        THREE.MathUtils.degToRad(this.sourceHorizontalFovDegrees) * 0.5,
      );
      const displayedVerticalHalfTangent =
        horizontalHalfTangent /
        Math.max(this.sourceVideoAspect, this.camera.aspect);
      this.camera.fov = THREE.MathUtils.radToDeg(
        2 * Math.atan(displayedVerticalHalfTangent),
      );
    }
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  }

  configureVideoProjection(
    horizontalFovDegrees: number,
    sourceAspect: number,
  ): void {
    if (
      Number.isFinite(horizontalFovDegrees) &&
      horizontalFovDegrees > 0 &&
      horizontalFovDegrees < 180 &&
      Number.isFinite(sourceAspect) &&
      sourceAspect > 0
    ) {
      this.sourceHorizontalFovDegrees = horizontalFovDegrees;
      this.sourceVideoAspect = sourceAspect;
      this.resize();
    }
  }

  setPoseSmoothingEnabled(enabled: boolean): void {
    this.positionSmoothingTimeSeconds = enabled
      ? smoothedPositionTimeConstantSeconds
      : 0;
    this.orientationSmoothingTimeSeconds = enabled
      ? smoothedOrientationTimeConstantSeconds
      : 0;
  }

  setCameraPose(pose: Float64Array): void {
    if (pose.length < 7) {
      return;
    }
    this.targetCameraPosition.set(pose[0], pose[1], pose[2]);
    this.targetCameraQuaternion.set(pose[3], pose[4], pose[5], pose[6]).normalize();
    this.usesExternalCameraPose = true;
  }

  setReticle(matrix: Float32Array): void {
    this.reticle.matrix.fromArray(matrix);
    this.reticle.visible = true;
  }

  hideReticle(): void {
    this.reticle.visible = false;
  }

  placeAnchorAtReticle(): boolean {
    if (!this.reticle.visible) {
      return false;
    }
    this.anchor.matrixAutoUpdate = false;
    this.anchor.matrix.copy(this.reticle.matrix);
    return true;
  }

  render(timestampMilliseconds: number = performance.now()): void {
    if (this.usesExternalCameraPose) {
      const elapsedSeconds =
        this.lastRenderTimestampMilliseconds === null
          ? 1 / 60
          : Math.min(
              Math.max(
                (timestampMilliseconds - this.lastRenderTimestampMilliseconds) / 1_000,
                0,
              ),
              0.1,
            );
      if (this.positionSmoothingTimeSeconds > 0) {
        const positionAlpha =
          1 - Math.exp(-elapsedSeconds / this.positionSmoothingTimeSeconds);
        this.camera.position.lerp(this.targetCameraPosition, positionAlpha);
      } else {
        this.camera.position.copy(this.targetCameraPosition);
      }
      if (this.orientationSmoothingTimeSeconds > 0) {
        const orientationAlpha =
          1 - Math.exp(-elapsedSeconds / this.orientationSmoothingTimeSeconds);
        this.camera.quaternion.slerp(this.targetCameraQuaternion, orientationAlpha);
      } else {
        this.camera.quaternion.copy(this.targetCameraQuaternion);
      }
    }
    this.lastRenderTimestampMilliseconds = timestampMilliseconds;
    this.renderer.render(this.scene, this.camera);
  }

  dispose(): void {
    this.renderer.setAnimationLoop(null);
    this.scene.traverse((object) => {
      if (object instanceof THREE.Mesh || object instanceof THREE.LineSegments) {
        object.geometry.dispose();
        const materials = Array.isArray(object.material)
          ? object.material
          : [object.material];
        materials.forEach((material) => material.dispose());
      }
    });
    this.renderer.dispose();
  }
}
