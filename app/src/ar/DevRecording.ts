import type { RecordingUploadResult } from "./types";

const recordingSchemaVersion = 2;

function normalizedMotionIntervalMilliseconds(interval: number): number {
  return Number.isFinite(interval) && interval > 0 && interval < 1
    ? interval * 1_000
    : interval;
}

/**
 * Written verbatim into `manifest.json` under `camera`. `trackerFrameWidth`
 * and `trackerFrameHeight` are load-bearing: `tools/vio-replay` and the dev
 * upload server both slice `tracker-luma.gray` as
 * `trackerFrameWidth * trackerFrameHeight` bytes per frame, so the dimensions
 * must stay constant for the whole recording.
 */
export interface RecordingCameraMetadata {
  horizontalFieldOfViewDegrees: number;
  longAxisFieldOfViewDegrees: number;
  targetCaptureRateHz: number;
  trackerFrameHeight: number;
  trackerFrameWidth: number;
  videoHeight: number;
  videoWidth: number;
}

export interface RecordedTrackerFrame {
  confidence: number;
  frameHeight: number;
  frameId: number;
  frameWidth: number;
  inliers: number;
  isKeyframe: boolean;
  keyframeCount: number;
  keyframeId: number;
  matches: number;
  performanceTimestampMilliseconds: number;
  pose: number[];
  relocalizationCount: number;
  textureScore: number;
  trackingState: number;
}

interface ActiveRecording {
  chunks: Blob[];
  frameEvents: string[];
  mediaRecorder: MediaRecorder;
  lumaFrameBuffers: ArrayBuffer[];
  sensorEvents: string[];
  startedAtIso: string;
  startedAtPerformanceMilliseconds: number;
}

interface CompletedRecording {
  frameEvents: string[];
  manifest: Record<string, unknown>;
  sensorEvents: string[];
  trackerLuma: Blob;
  video: Blob;
}

function supportedVideoMimeType(): string | undefined {
  const candidates = [
    "video/mp4;codecs=avc1.42E01E",
    "video/mp4",
    "video/webm;codecs=vp8",
    "video/webm",
  ];
  return candidates.find((candidate) => MediaRecorder.isTypeSupported(candidate));
}

function eventTime(
  active: ActiveRecording,
  eventTimestampMilliseconds: number,
  receiptTimestampMilliseconds: number,
) {
  return {
    eventTimestampMilliseconds,
    receiptTimestampMilliseconds,
    recordingTimeMilliseconds:
      receiptTimestampMilliseconds - active.startedAtPerformanceMilliseconds,
  };
}

function ndjson(lines: string[]): string {
  return lines.length === 0 ? "" : `${lines.join("\n")}\n`;
}

export class DevRecording {
  private active: ActiveRecording | null = null;
  private readonly camera: RecordingCameraMetadata;
  private completed: CompletedRecording | null = null;
  private readonly stream: MediaStream;

  constructor(stream: MediaStream, camera: RecordingCameraMetadata) {
    this.stream = stream;
    this.camera = camera;
  }

  start(): void {
    if (this.active || this.completed) {
      throw new Error("A recording is already active or waiting to upload.");
    }
    if (typeof MediaRecorder === "undefined") {
      throw new Error("This browser does not support camera recording.");
    }

    const mimeType = supportedVideoMimeType();
    const mediaRecorder = mimeType
      ? new MediaRecorder(this.stream, { mimeType })
      : new MediaRecorder(this.stream);
    const active: ActiveRecording = {
      chunks: [],
      frameEvents: [],
      mediaRecorder,
      lumaFrameBuffers: [],
      sensorEvents: [],
      startedAtIso: new Date().toISOString(),
      startedAtPerformanceMilliseconds: performance.now(),
    };
    mediaRecorder.addEventListener("dataavailable", (event) => {
      if (event.data.size > 0) {
        active.chunks.push(event.data);
      }
    });
    this.active = active;
    mediaRecorder.start();
  }

  recordDeviceMotion(
    event: DeviceMotionEvent,
    receiptTimestampMilliseconds: number,
    screenAngleDegrees: number,
    screenOrientation: string,
  ): void {
    const active = this.active;
    if (!active) return;
    active.sensorEvents.push(
      JSON.stringify({
        ...eventTime(active, event.timeStamp, receiptTimestampMilliseconds),
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
        intervalMilliseconds: normalizedMotionIntervalMilliseconds(event.interval),
        kind: "device_motion",
        reportedInterval: event.interval,
        rotationRateDegreesPerSecond: {
          alpha: event.rotationRate?.alpha ?? null,
          beta: event.rotationRate?.beta ?? null,
          gamma: event.rotationRate?.gamma ?? null,
        },
        screenAngleDegrees,
        screenOrientation,
      }),
    );
  }

  recordDeviceOrientation(
    event: DeviceOrientationEvent,
    receiptTimestampMilliseconds: number,
    screenAngleDegrees: number,
    screenOrientation: string,
  ): void {
    const active = this.active;
    if (!active) return;
    active.sensorEvents.push(
      JSON.stringify({
        ...eventTime(active, event.timeStamp, receiptTimestampMilliseconds),
        absolute: event.absolute,
        alphaDegrees: event.alpha,
        betaDegrees: event.beta,
        gammaDegrees: event.gamma,
        kind: "device_orientation",
        screenAngleDegrees,
        screenOrientation,
      }),
    );
  }

  recordTrackerFrame(frame: RecordedTrackerFrame, luma: Uint8Array): void {
    const active = this.active;
    if (!active) return;
    active.frameEvents.push(
      JSON.stringify({
        ...frame,
        recordingTimeMilliseconds:
          frame.performanceTimestampMilliseconds -
          active.startedAtPerformanceMilliseconds,
      }),
    );
    active.lumaFrameBuffers.push(luma.slice().buffer as ArrayBuffer);
  }

  async finishAndUpload(): Promise<RecordingUploadResult> {
    if (this.active) {
      this.completed = await this.finishActive(this.active);
      this.active = null;
    }
    if (!this.completed) {
      throw new Error("There is no recording to upload.");
    }

    const form = new FormData();
    form.append("manifest", JSON.stringify(this.completed.manifest));
    form.append("sensorEvents", ndjson(this.completed.sensorEvents));
    form.append("frameEvents", ndjson(this.completed.frameEvents));
    form.append("trackerLuma", this.completed.trackerLuma, "tracker-luma.gray");
    form.append(
      "video",
      this.completed.video,
      this.completed.video.type.includes("mp4") ? "camera.mp4" : "camera.webm",
    );
    const response = await fetch("/api/dev/recordings", {
      body: form,
      method: "POST",
    });
    const body = (await response.json().catch(() => null)) as
      | (Partial<RecordingUploadResult> & { error?: string })
      | null;
    if (!response.ok || !body?.recordingId || !body.savedPath) {
      throw new Error(body?.error ?? `Recording upload failed (${response.status}).`);
    }
    this.completed = null;
    return {
      recordingId: body.recordingId,
      savedPath: body.savedPath,
    };
  }

  cancel(): void {
    if (this.active?.mediaRecorder.state !== "inactive") {
      this.active?.mediaRecorder.stop();
    }
    this.active = null;
    this.completed = null;
  }

  private async finishActive(active: ActiveRecording): Promise<CompletedRecording> {
    const stopped = new Promise<void>((resolve, reject) => {
      active.mediaRecorder.addEventListener("stop", () => resolve(), { once: true });
      active.mediaRecorder.addEventListener(
        "error",
        () => reject(new Error("The browser failed while encoding the camera recording.")),
        { once: true },
      );
    });
    active.mediaRecorder.stop();
    await stopped;

    const endedAtPerformanceMilliseconds = performance.now();
    const track = this.stream.getVideoTracks()[0];
    const mimeType = active.mediaRecorder.mimeType || active.chunks[0]?.type || "video/webm";
    return {
      frameEvents: active.frameEvents,
      manifest: {
        camera: {
          ...this.camera,
          mediaTrackSettings: track?.getSettings() ?? null,
          trackerLumaFormat: "GRAY8_contiguous",
          videoMimeType: mimeType,
        },
        clock: {
          eventTimestampBasis: "DOMHighResTimeStamp",
          performanceTimeOriginMilliseconds: performance.timeOrigin,
          receiptTimestampBasis: "performance.now",
        },
        counts: {
          sensorEvents: active.sensorEvents.length,
          trackerFrames: active.frameEvents.length,
        },
        device: {
          language: navigator.language,
          platform: navigator.platform,
          screenHeight: window.screen.height,
          screenWidth: window.screen.width,
          userAgent: navigator.userAgent,
        },
        durationMilliseconds:
          endedAtPerformanceMilliseconds - active.startedAtPerformanceMilliseconds,
        endedAtIso: new Date().toISOString(),
        kind: "pizzanet_ar_tracking_recording",
        schemaVersion: recordingSchemaVersion,
        startedAtIso: active.startedAtIso,
        startedAtPerformanceMilliseconds: active.startedAtPerformanceMilliseconds,
      },
      sensorEvents: active.sensorEvents,
      trackerLuma: new Blob(active.lumaFrameBuffers, {
        type: "application/octet-stream",
      }),
      video: new Blob(active.chunks, { type: mimeType }),
    };
  }
}
