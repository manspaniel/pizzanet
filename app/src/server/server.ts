import { randomUUID } from "node:crypto";
import { mkdir, readFile, rename, rm, writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { relative, resolve } from "node:path";
import { Hono } from "hono";
import type { ViteDevServer } from "vite";

const repositoryRoot = fileURLToPath(new URL("../../../", import.meta.url));
const recordingsRoot = resolve(repositoryRoot, "datasets", "ar-recordings");
const maximumRequestBytes = 750 * 1024 * 1024;
const maximumSidecarCharacters = 64 * 1024 * 1024;

const app = new Hono<{ Bindings: { vite: ViteDevServer } }>();

function requiredText(
  form: FormData,
  field: string,
  maximumCharacters: number,
): string {
  const value = form.get(field);
  if (typeof value !== "string" || value.length > maximumCharacters) {
    throw new Error(`Invalid ${field} field.`);
  }
  return value;
}

function videoExtension(mimeType: string): "mp4" | "webm" {
  if (mimeType.toLowerCase().includes("mp4")) return "mp4";
  if (mimeType.toLowerCase().includes("webm")) return "webm";
  throw new Error("The camera recording must be MP4 or WebM video.");
}

function recordingIdentifier(): string {
  return `${new Date().toISOString().replaceAll(":", "-")}-${randomUUID()}`;
}

app.get("/", async (c) => {
  const vite = c.env.vite;
  const html = await vite.transformIndexHtml(
    c.req.url,
    await readFile("./index.html", "utf8"),
    "./",
  );
  return c.html(html);
});

app.post("/api/dev/recordings", async (c) => {
  const contentLength = Number(c.req.header("content-length") ?? 0);
  if (!Number.isFinite(contentLength) || contentLength > maximumRequestBytes) {
    return c.json({ error: "Recording upload is too large." }, 413);
  }

  let temporaryDirectory: string | null = null;
  try {
    const form = await c.req.raw.formData();
    const manifestText = requiredText(form, "manifest", 1024 * 1024);
    const sensorEvents = requiredText(
      form,
      "sensorEvents",
      maximumSidecarCharacters,
    );
    const frameEvents = requiredText(
      form,
      "frameEvents",
      maximumSidecarCharacters,
    );
    // Native ARKit recordings carry no camera video (the luma stream is the
    // camera record) but do carry ground-truth poses.
    const video = form.get("video");
    const hasVideo = video !== null && typeof video !== "string" && video.size > 0;
    const arkitPoses = form.get("arkitPoses");
    const arkitPosesText =
      typeof arkitPoses === "string" && arkitPoses.length > 0 ? arkitPoses : null;
    if (!hasVideo && arkitPosesText === null) {
      throw new Error("A non-empty camera video is required.");
    }
    const extension = hasVideo ? videoExtension(video.type) : null;
    const trackerLuma = form.get("trackerLuma");
    if (
      trackerLuma === null ||
      typeof trackerLuma === "string" ||
      trackerLuma.size === 0
    ) {
      throw new Error("A non-empty tracker luma stream is required.");
    }
    const clientManifest = JSON.parse(manifestText) as Record<string, unknown>;
    const supportedManifest =
      (clientManifest.schemaVersion === 1 &&
        clientManifest.kind === "pizzanet_lamp_tracking_recording") ||
      (clientManifest.schemaVersion === 2 &&
        clientManifest.kind === "pizzanet_ar_tracking_recording");
    if (!supportedManifest) {
      throw new Error("Unsupported recording manifest.");
    }
    const camera = clientManifest.camera as
      | { trackerFrameHeight?: unknown; trackerFrameWidth?: unknown }
      | undefined;
    const counts = clientManifest.counts as { trackerFrames?: unknown } | undefined;
    const expectedLumaBytes =
      Number(camera?.trackerFrameWidth) *
      Number(camera?.trackerFrameHeight) *
      Number(counts?.trackerFrames);
    if (
      !Number.isSafeInteger(expectedLumaBytes) ||
      expectedLumaBytes <= 0 ||
      trackerLuma.size !== expectedLumaBytes
    ) {
      throw new Error("Tracker luma size does not match the recording manifest.");
    }

    const recordingId = recordingIdentifier();
    const finalDirectory = resolve(recordingsRoot, recordingId);
    temporaryDirectory = resolve(recordingsRoot, `.${recordingId}.tmp`);
    await mkdir(temporaryDirectory, { recursive: true });
    const videoName = extension === null ? null : `camera.${extension}`;
    const storedManifest = {
      ...clientManifest,
      files: {
        ...(videoName === null ? {} : { cameraVideo: videoName }),
        frameEvents: "tracker-frames.ndjson",
        sensorEvents: "sensor-events.ndjson",
        trackerLuma: "tracker-luma.gray",
        ...(arkitPosesText === null ? {} : { arkitPoses: "arkit-poses.ndjson" }),
      },
      receivedAtIso: new Date().toISOString(),
      recordingId,
    };
    const writes: Promise<void>[] = [
      writeFile(
        resolve(temporaryDirectory, "manifest.json"),
        `${JSON.stringify(storedManifest, null, 2)}\n`,
      ),
      writeFile(resolve(temporaryDirectory, "sensor-events.ndjson"), sensorEvents),
      writeFile(resolve(temporaryDirectory, "tracker-frames.ndjson"), frameEvents),
      writeFile(
        resolve(temporaryDirectory, "tracker-luma.gray"),
        new Uint8Array(await trackerLuma.arrayBuffer()),
      ),
    ];
    if (hasVideo && videoName !== null) {
      writes.push(
        writeFile(
          resolve(temporaryDirectory, videoName),
          new Uint8Array(await video.arrayBuffer()),
        ),
      );
    }
    if (arkitPosesText !== null) {
      writes.push(
        writeFile(resolve(temporaryDirectory, "arkit-poses.ndjson"), arkitPosesText),
      );
    }
    await Promise.all(writes);
    await rename(temporaryDirectory, finalDirectory);
    temporaryDirectory = null;
    return c.json(
      {
        recordingId,
        savedPath: relative(repositoryRoot, finalDirectory),
      },
      201,
    );
  } catch (error) {
    if (temporaryDirectory) {
      await rm(temporaryDirectory, { force: true, recursive: true }).catch(
        () => undefined,
      );
    }
    const message = error instanceof Error ? error.message : "Recording upload failed.";
    const status = error instanceof SyntaxError ? 400 : 422;
    return c.json({ error: message }, status);
  }
});

export default app;
