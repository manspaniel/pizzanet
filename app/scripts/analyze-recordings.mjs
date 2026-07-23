import { readFile, readdir } from "node:fs/promises";
import { resolve } from "node:path";

const recordingsRoot = resolve(process.cwd(), "../datasets/ar-recordings");

function quantile(sorted, fraction) {
  if (sorted.length === 0) return null;
  const index = Math.min(sorted.length - 1, Math.floor(sorted.length * fraction));
  return sorted[index];
}

function summary(values) {
  const sorted = values.filter(Number.isFinite).toSorted((left, right) => left - right);
  return {
    maximum: sorted.at(-1) ?? null,
    median: quantile(sorted, 0.5),
    p10: quantile(sorted, 0.1),
    p90: quantile(sorted, 0.9),
    p95: quantile(sorted, 0.95),
  };
}

function vectorLength(values) {
  return Math.hypot(...values);
}

function position(frame) {
  return frame.pose.slice(0, 3);
}

function distance(left, right) {
  return vectorLength(left.map((value, index) => value - right[index]));
}

async function ndjson(path) {
  return (await readFile(path, "utf8"))
    .split("\n")
    .filter(Boolean)
    .map((line) => JSON.parse(line));
}

async function analyze(directory) {
  const manifest = JSON.parse(await readFile(resolve(directory, "manifest.json"), "utf8"));
  const frames = await ndjson(resolve(directory, "tracker-frames.ndjson"));
  const sensors = await ndjson(resolve(directory, "sensor-events.ndjson"));
  const motion = sensors.filter((event) => event.kind === "device_motion");
  const orientation = sensors.filter((event) => event.kind === "device_orientation");
  const frameSteps = frames.slice(1).map((frame, index) =>
    distance(position(frame), position(frames[index])),
  );
  const pathLength = frameSteps.reduce((total, value) => total + value, 0);
  const firstPosition = position(frames[0]);
  const lastPosition = position(frames.at(-1));
  const keyframesSelected =
    frames.at(-1).keyframeCount - frames[0].keyframeCount + Number(frames[0].isKeyframe);
  const videoAspect = manifest.camera.videoWidth / manifest.camera.videoHeight;
  const longAxisFieldOfViewDegrees =
    manifest.camera.longAxisFieldOfViewDegrees ??
    manifest.camera.horizontalFieldOfViewDegrees;
  const longAxisHalfTangent = Math.tan(
    (longAxisFieldOfViewDegrees * Math.PI) / 360,
  );
  const correctedHorizontalFov =
    (2 * Math.atan(longAxisHalfTangent * Math.min(videoAspect, 1)) * 180) / Math.PI;
  const oldFocalPixels =
    manifest.camera.trackerFrameWidth / (2 * longAxisHalfTangent);
  const correctedFocalPixels =
    Math.max(
      manifest.camera.trackerFrameWidth,
      manifest.camera.trackerFrameHeight,
    ) /
    (2 * longAxisHalfTangent);

  return {
    recordingId: manifest.recordingId,
    durationSeconds: manifest.durationMilliseconds / 1_000,
    camera: {
      correctedHorizontalFovDegrees: correctedHorizontalFov,
      focalScaleCorrection: correctedFocalPixels / oldFocalPixels,
      oldFocalPixels,
      correctedFocalPixels,
      trackerDimensions: [
        manifest.camera.trackerFrameWidth,
        manifest.camera.trackerFrameHeight,
      ],
      videoDimensions: [manifest.camera.videoWidth, manifest.camera.videoHeight],
    },
    tracker: {
      frameCount: frames.length,
      frameDeltaMilliseconds: summary(
        frames.slice(1).map(
          (frame, index) =>
            frame.performanceTimestampMilliseconds -
            frames[index].performanceTimestampMilliseconds,
        ),
      ),
      inliers: summary(frames.map((frame) => frame.inliers)),
      keyframesSelected,
      keyframeIntervalFrames: frames.length / Math.max(keyframesSelected, 1),
      limitedFrameFraction:
        frames.filter((frame) => frame.trackingState !== 2).length / frames.length,
      matches: summary(frames.map((frame) => frame.matches)),
      netDisplacementMetres: distance(firstPosition, lastPosition),
      pathLengthMetres: pathLength,
      positionStepMetres: summary(frameSteps),
      textureScore: summary(frames.map((frame) => frame.textureScore)),
      verticalRangeMetres:
        Math.max(...frames.map((frame) => frame.pose[1])) -
        Math.min(...frames.map((frame) => frame.pose[1])),
    },
    sensors: {
      motionCount: motion.length,
      motionDeltaMilliseconds: summary(
        motion.slice(1).map(
          (event, index) =>
            event.eventTimestampMilliseconds - motion[index].eventTimestampMilliseconds,
        ),
      ),
      orientationCount: orientation.length,
      reportedMotionInterval: summary(
        motion.map((event) => event.intervalMilliseconds),
      ),
      linearAccelerationMagnitude: summary(
        motion.map((event) =>
          vectorLength([
            event.acceleration.x ?? 0,
            event.acceleration.y ?? 0,
            event.acceleration.z ?? 0,
          ]),
        ),
      ),
      rotationRateDegreesPerSecond: summary(
        motion.map((event) =>
          vectorLength([
            event.rotationRateDegreesPerSecond.alpha ?? 0,
            event.rotationRateDegreesPerSecond.beta ?? 0,
            event.rotationRateDegreesPerSecond.gamma ?? 0,
          ]),
        ),
      ),
    },
  };
}

const directories = (await readdir(recordingsRoot, { withFileTypes: true }))
  .filter((entry) => entry.isDirectory() && !entry.name.startsWith("."))
  .map((entry) => resolve(recordingsRoot, entry.name))
  .toSorted();
const reports = [];
for (const directory of directories) {
  reports.push(await analyze(directory));
}
process.stdout.write(`${JSON.stringify(reports, null, 2)}\n`);
