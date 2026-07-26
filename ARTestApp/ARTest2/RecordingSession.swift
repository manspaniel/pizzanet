//
//  RecordingSession.swift
//  ARTest2
//
//  Records an ARKit session as BOTH:
//  1. Ground truth: per-frame ARKit camera poses -> arkit-poses.ndjson
//  2. A pizzanet web-format tracking recording (manifest + sensor-events
//     + tracker-frames + tracker-luma.gray), with camera luma taken from the
//     same ARKit frames and sensor-events copied verbatim from the actual
//     WKWebView Device Motion / Device Orientation callbacks.
//
//  Native CoreMotion remains a separate diagnostic sidecar. WK frame timing
//  records preserve the page-clock mapping used by the live tracker, so replay
//  does not confuse bridge receipt jitter with capture time.
//
//  Concurrency: `RecordingCore` is explicitly nonisolated (the project uses
//  default main-actor isolation) and guards all recording state with a lock —
//  ARKit and CoreMotion callbacks arrive on their own queues. The main-actor
//  `RecordingSession` holds only the SwiftUI-facing state, updated via
//  `onUpdate` hops to the main queue.
//

import ARKit
import Combine
import CoreMotion
import Foundation

/// SwiftUI-facing state.
final class RecordingSession: ObservableObject {
    nonisolated enum Phase: Equatable {
        case idle
        case recording
        case uploading
        case done
        case failed(String)
    }

    @Published var phase: Phase = .idle
    @Published var frameCount = 0
    @Published var sensorEventCount = 0
    @Published var arkitTrackingState = "—"
    @Published var browserSensorsReady = false

    let core: RecordingCore

    init() {
        core = RecordingCore()
        core.onUpdate = { [weak self] update in
            DispatchQueue.main.async {
                guard let self else { return }
                if let phase = update.phase { self.phase = phase }
                if let frames = update.frameCount { self.frameCount = frames }
                if let sensors = update.sensorEventCount { self.sensorEventCount = sensors }
                if let tracking = update.trackingState { self.arkitTrackingState = tracking }
                if let ready = update.browserSensorsReady {
                    self.browserSensorsReady = ready
                }
            }
        }
    }

    func toggleRecording() {
        if phase == .recording {
            core.stopAndUpload()
        } else {
            core.start()
        }
    }
}

/// All recording state and sensor plumbing, off the main actor.
nonisolated final class RecordingCore: NSObject, ARSessionDelegate, @unchecked Sendable {
    static let uploadURL = URL(string: "https://danlinux.warg-balance.ts.net/api/dev/recordings")!
    static let trackerFrameWidth = 240
    static let targetFrameIntervalSeconds = 1.0 / 30.0

    struct Update: Sendable {
        var phase: RecordingSession.Phase?
        var frameCount: Int?
        var sensorEventCount: Int?
        var trackingState: String?
        var browserSensorsReady: Bool?
    }

    var onUpdate: (@Sendable (Update) -> Void)?
    /// Fed every throttled frame (recording or not) for the webview bridge:
    /// (frameId, timestampMilliseconds, width, height, base64Luma,
    /// longAxisFovDegrees). Keeping luma as argument five preserves
    /// compatibility with a host page that has not deployed the FOV extension.
    var onLumaFrame: (@Sendable (UInt32, Double, Int, Int, String, Double) -> Void)?
    /// Enables the page-to-native raw capture bridge only while recording.
    var onBrowserCaptureRecordingState: (@Sendable (Bool) -> Void)?

    private let motionManager = CMMotionManager()
    private let motionQueue = OperationQueue()
    private let lock = NSLock()

    // Guarded by `lock`.
    private var isRecording = false
    private var startUptimeSeconds: TimeInterval = 0
    private var startWallClock = Date()
    private var nextFrameAtSeconds: TimeInterval = 0
    private var trackerFrameHeight = 0
    private var sensorEventLines: [String] = []
    private var browserSensorEventLines: [String] = []
    private var browserFrameTimingLines: [String] = []
    private var browserMotionEventCount = 0
    private var browserOrientationEventCount = 0
    private var frameEventLines: [String] = []
    private var arkitPoseLines: [String] = []
    private var lumaFileHandle: FileHandle?
    private var lumaFileURL: URL?
    private var recordedFrameCount = 0
    private var nextFrameId: UInt32 = 1
    private var cameraFocalPixels: Double = 0
    private var cameraImageWidth = 0
    private var cameraImageHeight = 0
    private var previousMotionTimestampSeconds: TimeInterval?
    private var browserSensorsReady = false

    // MARK: - Controls

    func start() {
        lock.lock()
        guard !isRecording else {
            lock.unlock()
            return
        }
        guard browserSensorsReady else {
            lock.unlock()
            onUpdate?(Update(phase: .failed("Start AR and allow WK motion sensors first.")))
            return
        }

        sensorEventLines = []
        browserSensorEventLines = []
        browserFrameTimingLines = []
        browserMotionEventCount = 0
        browserOrientationEventCount = 0
        frameEventLines = []
        arkitPoseLines = []
        recordedFrameCount = 0
        trackerFrameHeight = 0
        startUptimeSeconds = ProcessInfo.processInfo.systemUptime
        startWallClock = Date()
        // Reject any ARFrame captured before the Record tap but still queued
        // for delegate delivery.
        nextFrameAtSeconds = startUptimeSeconds
        previousMotionTimestampSeconds = nil

        let lumaURL = FileManager.default.temporaryDirectory
            .appendingPathComponent("tracker-luma-\(UUID().uuidString).gray")
        FileManager.default.createFile(atPath: lumaURL.path, contents: nil)
        lumaFileURL = lumaURL
        lumaFileHandle = try? FileHandle(forWritingTo: lumaURL)

        motionQueue.maxConcurrentOperationCount = 1
        motionManager.deviceMotionUpdateInterval = 1.0 / 60.0
        motionManager.startDeviceMotionUpdates(
            using: .xArbitraryZVertical,
            to: motionQueue
        ) { [weak self] motion, _ in
            guard let motion else { return }
            self?.appendMotion(motion)
        }

        isRecording = true
        lock.unlock()
        onBrowserCaptureRecordingState?(true)
        onUpdate?(Update(phase: .recording, frameCount: 0, sensorEventCount: 0))
    }

    func stopAndUpload() {
        lock.lock()
        guard isRecording else {
            lock.unlock()
            return
        }
        isRecording = false
        motionManager.stopDeviceMotionUpdates()
        try? lumaFileHandle?.close()
        lumaFileHandle = nil
        let durationMilliseconds =
            (ProcessInfo.processInfo.systemUptime - startUptimeSeconds) * 1000.0
        let sensorEvents = browserSensorEventLines.joined(separator: "\n")
        let coreMotionEvents = sensorEventLines.joined(separator: "\n")
        let browserFrameTiming = browserFrameTimingLines.joined(separator: "\n")
        let frameEvents = frameEventLines.joined(separator: "\n")
        let arkitPoses = arkitPoseLines.joined(separator: "\n")
        let manifest = buildManifestLocked(durationMilliseconds: durationMilliseconds)
        let lumaURL = lumaFileURL
        let frames = recordedFrameCount
        let browserSensors = browserSensorEventLines.count
        let browserMotionEvents = browserMotionEventCount
        let browserOrientationEvents = browserOrientationEventCount
        lock.unlock()
        onBrowserCaptureRecordingState?(false)

        guard
            frames > 10,
            browserMotionEvents > 5,
            browserOrientationEvents > 5,
            let lumaURL
        else {
            onUpdate?(
                Update(
                    phase: .failed(
                        browserSensors == 0
                            ? "No WKWebView sensor events were captured."
                            : "Recording too short."
                    )
                )
            )
            return
        }
        onUpdate?(Update(phase: .uploading))
        let update = onUpdate
        Task.detached(priority: .userInitiated) {
            do {
                try await RecordingCore.upload(
                    manifest: manifest,
                    sensorEvents: sensorEvents,
                    frameEvents: frameEvents,
                    arkitPoses: arkitPoses,
                    coreMotionEvents: coreMotionEvents,
                    browserFrameTiming: browserFrameTiming,
                    lumaFileURL: lumaURL
                )
                update?(Update(phase: .done))
            } catch {
                update?(Update(phase: .failed(error.localizedDescription)))
            }
        }
    }

    // MARK: - ARKit frames

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let trackingLabel: String
        switch frame.camera.trackingState {
        case .normal: trackingLabel = "normal"
        case .notAvailable: trackingLabel = "unavailable"
        case .limited: trackingLabel = "limited"
        }

        lock.lock()
        guard frame.timestamp >= nextFrameAtSeconds else {
            lock.unlock()
            onUpdate?(Update(trackingState: trackingLabel))
            return
        }
        nextFrameAtSeconds =
            max(nextFrameAtSeconds + Self.targetFrameIntervalSeconds, frame.timestamp - 0.005)
        let feedFrameId = nextFrameId
        // The WK tracker remains alive across Record taps, so bridge ids must
        // remain globally monotonic even while the native recorder is idle.
        nextFrameId += 1
        lock.unlock()

        let eventTimestampMilliseconds = frame.timestamp * 1000.0
        let intrinsics = frame.camera.intrinsics
        let focalPixels = Double(intrinsics.columns.0.x)
        let focalYPixels = Double(intrinsics.columns.1.y)
        let principalPointXPixels = Double(intrinsics.columns.2.x)
        let principalPointYPixels = Double(intrinsics.columns.2.y)
        let imageWidth = Int(frame.camera.imageResolution.width)
        let imageHeight = Int(frame.camera.imageResolution.height)
        let longAxisFovDegrees = 2.0
            * atan(Double(max(imageWidth, imageHeight)) / (2.0 * focalPixels))
            * 180.0 / Double.pi

        guard let (luma, width, height) = RecordingCore.portraitLuma(
            from: frame.capturedImage,
            targetWidth: Self.trackerFrameWidth
        ) else {
            onUpdate?(Update(trackingState: trackingLabel))
            return
        }
        lock.lock()
        guard isRecording else {
            lock.unlock()
            onLumaFrame?(
                feedFrameId, eventTimestampMilliseconds, width, height,
                luma.base64EncodedString(),
                longAxisFovDegrees
            )
            onUpdate?(Update(trackingState: trackingLabel))
            return
        }
        // A Record tap can race with luma conversion, which deliberately runs
        // outside the shared lock. Reject that pre-tap capture after
        // reacquiring the recording state.
        guard frame.timestamp >= startUptimeSeconds else {
            lock.unlock()
            onLumaFrame?(
                feedFrameId, eventTimestampMilliseconds, width, height,
                luma.base64EncodedString(),
                longAxisFovDegrees
            )
            onUpdate?(Update(trackingState: trackingLabel))
            return
        }
        let recordingTimeMilliseconds = (frame.timestamp - startUptimeSeconds) * 1000.0
        // Keep every post-tap frame. Exact WK frame timing may start a few
        // milliseconds later; replay skips frames before its first complete
        // browser sensor pair instead of making camera capture depend on the
        // separate CoreMotion diagnostic stream.
        if trackerFrameHeight == 0 {
            trackerFrameHeight = height
            cameraFocalPixels = focalPixels
            cameraImageWidth = imageWidth
            cameraImageHeight = imageHeight
        }
        guard height == trackerFrameHeight else {
            lock.unlock()
            return
        }

        lumaFileHandle?.write(luma)
        let frameId = feedFrameId
        recordedFrameCount += 1
        let frames = recordedFrameCount

        frameEventLines.append(
            jsonLine([
                "frameId": frameId,
                "performanceTimestampMilliseconds": eventTimestampMilliseconds,
                "recordingTimeMilliseconds": recordingTimeMilliseconds,
                "frameWidth": width,
                "frameHeight": height,
                // Preserve exact per-frame calibration for future replay
                // models. The current tracker consumes the manifest-level
                // long-axis FOV because its landmark bearings share one
                // global intrinsics object.
                "cameraFocalXPixels": focalPixels,
                "cameraFocalYPixels": focalYPixels,
                "cameraPrincipalPointXPixels": principalPointXPixels,
                "cameraPrincipalPointYPixels": principalPointYPixels,
                "cameraImageWidth": imageWidth,
                "cameraImageHeight": imageHeight,
                "longAxisFieldOfViewDegrees": longAxisFovDegrees,
            ])
        )

        // Ground truth: ARKit camera pose. ARKit camera space matches the
        // three.js convention (x right, y up, camera looks down -z); the world
        // is gravity-aligned with y up.
        let transform = frame.camera.transform
        let position = transform.columns.3
        let rotation = simd_quatf(transform)
        arkitPoseLines.append(
            jsonLine([
                "recordingTimeMilliseconds": recordingTimeMilliseconds,
                "timestampSeconds": frame.timestamp,
                "frameId": frameId,
                "position": [position.x, position.y, position.z],
                "quaternionXYZW": [
                    rotation.imag.x, rotation.imag.y, rotation.imag.z, rotation.real,
                ],
                "trackingState": trackingLabel,
            ])
        )
        lock.unlock()

        onLumaFrame?(
            frameId, eventTimestampMilliseconds, width, height,
            luma.base64EncodedString(),
            longAxisFovDegrees
        )
        onUpdate?(Update(frameCount: frames, trackingState: trackingLabel))
    }

    /// Extracts the Y (luma) plane, rotates landscape -> portrait (90° CW),
    /// and area-downsamples to `targetWidth` columns.
    private static func portraitLuma(
        from pixelBuffer: CVPixelBuffer,
        targetWidth: Int
    ) -> (Data, Int, Int)? {
        CVPixelBufferLockBaseAddress(pixelBuffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(pixelBuffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddressOfPlane(pixelBuffer, 0) else {
            return nil
        }
        let sourceWidth = CVPixelBufferGetWidthOfPlane(pixelBuffer, 0)
        let sourceHeight = CVPixelBufferGetHeightOfPlane(pixelBuffer, 0)
        let stride = CVPixelBufferGetBytesPerRowOfPlane(pixelBuffer, 0)
        let source = base.assumingMemoryBound(to: UInt8.self)

        let destWidth = targetWidth
        let destHeight = Int(
            (Double(targetWidth) * Double(sourceWidth) / Double(sourceHeight)).rounded()
        )
        var out = Data(count: destWidth * destHeight)
        out.withUnsafeMutableBytes { (buffer: UnsafeMutableRawBufferPointer) in
            let dest = buffer.bindMemory(to: UInt8.self).baseAddress!
            for destY in 0..<destHeight {
                // A portrait row covers a horizontal interval in the
                // landscape source. Average the entire footprint rather than
                // a 2x2 patch at one sample location; the reduction is about
                // 4.5x per axis, so the old sampler aliased roof edges and
                // texture into unstable LK features.
                let sourceXStart = destY * sourceWidth / destHeight
                let sourceXEnd = min(
                    sourceWidth,
                    ((destY + 1) * sourceWidth + destHeight - 1) / destHeight
                )
                for destX in 0..<destWidth {
                    // 90° clockwise: portrait columns traverse source rows in
                    // reverse. These are half-open source bounds.
                    let sourceYStart = max(
                        0,
                        sourceHeight
                            - ((destX + 1) * sourceHeight + destWidth - 1) / destWidth
                    )
                    let sourceYEnd = min(
                        sourceHeight,
                        sourceHeight - destX * sourceHeight / destWidth
                    )
                    var sum = 0
                    var samples = 0
                    for sourceY in sourceYStart..<sourceYEnd {
                        let row = sourceY * stride
                        for sourceX in sourceXStart..<sourceXEnd {
                            sum += Int(source[row + sourceX])
                            samples += 1
                        }
                    }
                    dest[destY * destWidth + destX] =
                        samples > 0 ? UInt8(sum / samples) : 0
                }
            }
        }
        return (out, destWidth, destHeight)
    }

    // MARK: - Exact WKWebView inputs

    /// Receives low-rate readiness messages and 40 ms batches containing the
    /// untouched values seen by the page's actual event handlers.
    func appendBrowserCaptureMessage(_ body: Any) {
        guard
            let envelope = body as? [String: Any],
            let kind = envelope["kind"] as? String
        else {
            return
        }

        if kind == "sensor_readiness" {
            let permissionGranted = envelope["permissionState"] as? String == "granted"
            let motionReceived = envelope["motionEventReceived"] as? Bool == true
            let orientationReceived = envelope["orientationEventReceived"] as? Bool == true
            let ready = permissionGranted && motionReceived && orientationReceived
            lock.lock()
            browserSensorsReady = ready
            lock.unlock()
            onUpdate?(Update(browserSensorsReady: ready))
            return
        }

        guard
            kind == "raw_input_batch",
            let events = envelope["events"] as? [Any]
        else {
            return
        }
        let nativeReceiptUptimeMilliseconds =
            ProcessInfo.processInfo.systemUptime * 1000.0
        let performanceTimeOriginMilliseconds =
            envelope["performanceTimeOriginMilliseconds"] as? Double

        lock.lock()
        guard isRecording else {
            lock.unlock()
            return
        }
        let recordingTimeMilliseconds =
            nativeReceiptUptimeMilliseconds - startUptimeSeconds * 1000.0
        for rawEvent in events {
            guard let eventValue = rawEvent as? [String: Any] else { continue }
            guard let eventKind = eventValue["kind"] as? String else { continue }
            var event = eventValue
            event["nativeMessageReceiptUptimeMilliseconds"] =
                nativeReceiptUptimeMilliseconds
            event["recordingTimeMilliseconds"] = recordingTimeMilliseconds
            if let performanceTimeOriginMilliseconds {
                event["performanceTimeOriginMilliseconds"] =
                    performanceTimeOriginMilliseconds
            }
            guard JSONSerialization.isValidJSONObject(event) else { continue }
            switch eventKind {
            case "device_motion":
                browserSensorEventLines.append(jsonLine(event))
                browserMotionEventCount += 1
            case "device_orientation":
                browserSensorEventLines.append(jsonLine(event))
                browserOrientationEventCount += 1
            case "native_frame_timing":
                browserFrameTimingLines.append(jsonLine(event))
            default:
                continue
            }
        }
        let count = browserSensorEventLines.count
        lock.unlock()
        onUpdate?(Update(sensorEventCount: count))
    }

    // MARK: - Native CoreMotion diagnostics

    private func appendMotion(_ motion: CMDeviceMotion) {
        let gravityConstant = 9.80665
        let degreesPerRadian = 180.0 / Double.pi
        let eventTimestampMilliseconds = motion.timestamp * 1000.0

        lock.lock()
        guard isRecording else {
            lock.unlock()
            return
        }
        let recordingTimeMilliseconds = (motion.timestamp - startUptimeSeconds) * 1000.0
        let measuredIntervalSeconds = previousMotionTimestampSeconds
            .map { motion.timestamp - $0 }
            .flatMap { interval in
                interval.isFinite && interval > 0 && interval <= 0.1 ? interval : nil
            }
            ?? motionManager.deviceMotionUpdateInterval
        previousMotionTimestampSeconds = motion.timestamp

        // Safari-convention values, verified empirically against ARKit ground
        // truth (integrated dv correlates 0.98 with ARKit velocity):
        // CMDeviceMotion.userAcceleration uses the accelerometer sign
        // convention, so the user term ADDS to gravity here.
        //   acceleration                 =  9.81 * userAcceleration
        //   accelerationIncludingGravity =  9.81 * (gravity + userAcceleration)
        let user = motion.userAcceleration
        let gravity = motion.gravity
        sensorEventLines.append(
            jsonLine([
                "kind": "device_motion",
                "eventTimestampMilliseconds": eventTimestampMilliseconds,
                "receiptTimestampMilliseconds": eventTimestampMilliseconds,
                "recordingTimeMilliseconds": recordingTimeMilliseconds,
                "acceleration": [
                    "x": gravityConstant * user.x,
                    "y": gravityConstant * user.y,
                    "z": gravityConstant * user.z,
                ],
                "accelerationIncludingGravity": [
                    "x": gravityConstant * (gravity.x + user.x),
                    "y": gravityConstant * (gravity.y + user.y),
                    "z": gravityConstant * (gravity.z + user.z),
                ],
                // Safari's iOS rotationRate quirk: alpha/beta/gamma carry the
                // device x/y/z rates (deg/s).
                "rotationRateDegreesPerSecond": [
                    "alpha": motion.rotationRate.x * degreesPerRadian,
                    "beta": motion.rotationRate.y * degreesPerRadian,
                    "gamma": motion.rotationRate.z * degreesPerRadian,
                ],
                "intervalMilliseconds": measuredIntervalSeconds * 1000.0,
                "reportedInterval": measuredIntervalSeconds,
                "screenAngleDegrees": 0,
                "screenOrientation": "portrait-primary",
            ])
        )

        // W3C deviceorientation from the attitude matrix. CMRotationMatrix is
        // reference->device (verified against ARKit ground truth: with the
        // transposed elements below, the synthesized orientation tracks ARKit
        // to within a few degrees; the untransposed variant wanders 180°), so
        // the device->earth matrix is its transpose. Decomposing
        // transpose(m) = Rz(a)·Rx(b)·Ry(g):
        //   b = asin(m23), a = atan2(-m21, m22), g = atan2(-m13, m33)
        let m = motion.attitude.rotationMatrix
        let beta = asin(max(-1.0, min(1.0, m.m23))) * degreesPerRadian
        var alpha = atan2(-m.m21, m.m22) * degreesPerRadian
        if alpha < 0 { alpha += 360.0 }
        let gamma = atan2(-m.m13, m.m33) * degreesPerRadian
        sensorEventLines.append(
            jsonLine([
                "kind": "device_orientation",
                "eventTimestampMilliseconds": eventTimestampMilliseconds,
                "receiptTimestampMilliseconds": eventTimestampMilliseconds,
                "recordingTimeMilliseconds": recordingTimeMilliseconds,
                "alphaDegrees": alpha,
                "betaDegrees": beta,
                "gammaDegrees": gamma,
                "screenAngleDegrees": 0,
                "screenOrientation": "portrait-primary",
            ])
        )
        lock.unlock()
    }

    // MARK: - Manifest + upload

    /// Caller must hold `lock`.
    private func buildManifestLocked(durationMilliseconds: Double) -> String {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        let longAxisPixels = Double(max(cameraImageWidth, cameraImageHeight))
        let longAxisFov = cameraFocalPixels > 0
            ? 2.0 * atan(longAxisPixels / (2.0 * cameraFocalPixels)) * 180.0 / Double.pi
            : 68.0
        let manifest: [String: Any] = [
            "kind": "pizzanet_ar_tracking_recording",
            "schemaVersion": 2,
            "source": "native-arkit",
            "startedAtIso": formatter.string(from: startWallClock),
            "endedAtIso": formatter.string(
                from: startWallClock.addingTimeInterval(durationMilliseconds / 1000.0)
            ),
            "durationMilliseconds": durationMilliseconds,
            "startedAtNativeUptimeMilliseconds": startUptimeSeconds * 1000.0,
            // Retained for schema-v2 readers of older native bundles.
            "startedAtPerformanceMilliseconds": startUptimeSeconds * 1000.0,
            "camera": [
                "trackerFrameWidth": Self.trackerFrameWidth,
                "trackerFrameHeight": trackerFrameHeight,
                "trackerLumaFormat": "GRAY8_contiguous",
                "targetCaptureRateHz": 30,
                "videoWidth": cameraImageHeight,
                "videoHeight": cameraImageWidth,
                "longAxisFieldOfViewDegrees": longAxisFov,
            ],
            "arkitIntrinsics": [
                "focalPixels": cameraFocalPixels,
                "imageWidth": cameraImageWidth,
                "imageHeight": cameraImageHeight,
            ],
            "clock": [
                "sensorEventTimestampBasis": "wkwebviewPerformanceMilliseconds",
                "sensorReceiptTimestampBasis": "wkwebviewPerformanceMilliseconds",
                "frameNativeTimestampBasis": "boottimeMilliseconds",
                "frameReplayTimestampBasis": "wkwebviewPerformanceMilliseconds",
                "nativeMessageReceiptBasis": "boottimeMilliseconds",
            ],
            "counts": [
                "sensorEvents": browserSensorEventLines.count,
                "browserMotionEvents": browserMotionEventCount,
                "browserOrientationEvents": browserOrientationEventCount,
                "coreMotionEvents": sensorEventLines.count,
                "browserFrameTimingEvents": browserFrameTimingLines.count,
                "trackerFrames": recordedFrameCount,
            ],
            "device": [
                "platform": "iPhone",
                "userAgent": "ARTest2 native ARKit recorder",
            ],
            "sensorSource": "wkwebview-device-events",
            "files": [
                "sensorEvents": "sensor-events.ndjson",
                "coreMotionEvents": "coremotion-events.ndjson",
                "browserFrameTiming": "wk-frame-timing.ndjson",
                "frameEvents": "tracker-frames.ndjson",
                "trackerLuma": "tracker-luma.gray",
                "arkitPoses": "arkit-poses.ndjson",
            ],
        ]
        let data = try? JSONSerialization.data(withJSONObject: manifest)
        return data.flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
    }

    private static func upload(
        manifest: String,
        sensorEvents: String,
        frameEvents: String,
        arkitPoses: String,
        coreMotionEvents: String,
        browserFrameTiming: String,
        lumaFileURL: URL
    ) async throws {
        let boundary = "pizzanet-\(UUID().uuidString)"
        var body = Data()
        func addField(_ name: String, _ value: String) {
            body.append(Data("--\(boundary)\r\n".utf8))
            body.append(
                Data("Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n".utf8)
            )
            body.append(Data(value.utf8))
            body.append(Data("\r\n".utf8))
        }
        addField("manifest", manifest)
        addField("sensorEvents", sensorEvents)
        addField("frameEvents", frameEvents)
        addField("arkitPoses", arkitPoses)
        addField("coreMotionEvents", coreMotionEvents)
        addField("browserFrameTiming", browserFrameTiming)
        let luma = try Data(contentsOf: lumaFileURL)
        body.append(Data("--\(boundary)\r\n".utf8))
        body.append(
            Data(
                "Content-Disposition: form-data; name=\"trackerLuma\"; filename=\"tracker-luma.gray\"\r\nContent-Type: application/octet-stream\r\n\r\n"
                    .utf8
            )
        )
        body.append(luma)
        body.append(Data("\r\n--\(boundary)--\r\n".utf8))

        var request = URLRequest(url: uploadURL)
        request.httpMethod = "POST"
        request.setValue(
            "multipart/form-data; boundary=\(boundary)",
            forHTTPHeaderField: "Content-Type"
        )
        request.timeoutInterval = 300
        let (responseData, response) = try await URLSession.shared.upload(
            for: request,
            from: body
        )
        guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode)
        else {
            let text = String(data: responseData, encoding: .utf8) ?? ""
            throw NSError(
                domain: "upload",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "Upload failed: \(text.prefix(200))"]
            )
        }
        try? FileManager.default.removeItem(at: lumaFileURL)
    }

    /// Serializes a dictionary to a single NDJSON line with stable key order.
    private func jsonLine(_ object: [String: Any]) -> String {
        let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        return data.flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
    }
}
