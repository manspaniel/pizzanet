//
//  RecordingSession.swift
//  ARTest2
//
//  Records an ARKit session as BOTH:
//  1. Ground truth: per-frame ARKit camera poses -> arkit-poses.ndjson
//  2. A pizzanet web-format tracking recording (manifest + sensor-events
//     + tracker-frames + tracker-luma.gray), with camera luma taken from the
//     same ARKit frames and Safari-convention sensor events synthesized from
//     CoreMotion.
//
//  Both streams share the boot-time clock (ARFrame.timestamp and
//  CMDeviceMotion.timestamp use it), so they are inherently synchronized.
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
    }

    var onUpdate: (@Sendable (Update) -> Void)?
    /// Fed every throttled frame (recording or not) for the webview bridge:
    /// (frameId, timestampMilliseconds, width, height, base64Luma).
    var onLumaFrame: (@Sendable (UInt32, Double, Int, Int, String) -> Void)?

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
    private var frameEventLines: [String] = []
    private var arkitPoseLines: [String] = []
    private var lumaFileHandle: FileHandle?
    private var lumaFileURL: URL?
    private var recordedFrameCount = 0
    private var nextFrameId: UInt32 = 1
    private var cameraFocalPixels: Double = 0
    private var cameraImageWidth = 0
    private var cameraImageHeight = 0

    // MARK: - Controls

    func start() {
        lock.lock()
        defer { lock.unlock() }
        guard !isRecording else { return }

        sensorEventLines = []
        frameEventLines = []
        arkitPoseLines = []
        recordedFrameCount = 0
        nextFrameId = 1
        trackerFrameHeight = 0
        startUptimeSeconds = ProcessInfo.processInfo.systemUptime
        startWallClock = Date()
        nextFrameAtSeconds = 0

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
        let sensorEvents = sensorEventLines.joined(separator: "\n")
        let frameEvents = frameEventLines.joined(separator: "\n")
        let arkitPoses = arkitPoseLines.joined(separator: "\n")
        let manifest = buildManifestLocked(durationMilliseconds: durationMilliseconds)
        let lumaURL = lumaFileURL
        let frames = recordedFrameCount
        lock.unlock()

        guard frames > 10, let lumaURL else {
            onUpdate?(Update(phase: .failed("Recording too short.")))
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

        let recordingTimeMilliseconds = (frame.timestamp - startUptimeSeconds) * 1000.0
        let eventTimestampMilliseconds = frame.timestamp * 1000.0

        guard let (luma, width, height) = RecordingCore.portraitLuma(
            from: frame.capturedImage,
            targetWidth: Self.trackerFrameWidth
        ) else {
            lock.unlock()
            return
        }
        let feedFrameId = nextFrameId
        guard isRecording else {
            lock.unlock()
            onLumaFrame?(
                feedFrameId, eventTimestampMilliseconds, width, height,
                luma.base64EncodedString()
            )
            onUpdate?(Update(trackingState: trackingLabel))
            return
        }
        if trackerFrameHeight == 0 {
            trackerFrameHeight = height
            cameraFocalPixels = Double(frame.camera.intrinsics.columns.0.x)
            cameraImageWidth = Int(frame.camera.imageResolution.width)
            cameraImageHeight = Int(frame.camera.imageResolution.height)
        }
        guard height == trackerFrameHeight else {
            lock.unlock()
            return
        }

        lumaFileHandle?.write(luma)
        let frameId = nextFrameId
        nextFrameId += 1
        recordedFrameCount += 1
        let frames = recordedFrameCount

        frameEventLines.append(
            jsonLine([
                "frameId": frameId,
                "performanceTimestampMilliseconds": eventTimestampMilliseconds,
                "recordingTimeMilliseconds": recordingTimeMilliseconds,
                "frameWidth": width,
                "frameHeight": height,
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
            luma.base64EncodedString()
        )
        onUpdate?(Update(frameCount: frames, trackingState: trackingLabel))
    }

    /// Extracts the Y (luma) plane, rotates landscape -> portrait (90° CW),
    /// and box-downsamples to `targetWidth` columns.
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
                let sourceX = min(
                    sourceWidth - 1,
                    Int(Double(destY) / Double(destHeight) * Double(sourceWidth))
                )
                for destX in 0..<destWidth {
                    // 90° clockwise: portrait column samples the sensor's y
                    // axis, reversed.
                    let sourceY = min(
                        sourceHeight - 1,
                        sourceHeight - 1
                            - Int(Double(destX) / Double(destWidth) * Double(sourceHeight))
                    )
                    let x1 = min(sourceX + 1, sourceWidth - 1)
                    let y1 = min(sourceY + 1, sourceHeight - 1)
                    let sum =
                        Int(source[sourceY * stride + sourceX])
                        + Int(source[sourceY * stride + x1])
                        + Int(source[y1 * stride + sourceX])
                        + Int(source[y1 * stride + x1])
                    dest[destY * destWidth + destX] = UInt8(sum / 4)
                }
            }
        }
        return (out, destWidth, destHeight)
    }

    // MARK: - CoreMotion -> Safari-convention sensor events

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

        // Safari on iOS reports the NEGATED spec values (the long-standing
        // WebKit sign inversion the web pipeline already corrects for):
        //   acceleration                 = -9.81 * userAcceleration
        //   accelerationIncludingGravity =  9.81 * (gravity - userAcceleration)
        let user = motion.userAcceleration
        let gravity = motion.gravity
        sensorEventLines.append(
            jsonLine([
                "kind": "device_motion",
                "eventTimestampMilliseconds": eventTimestampMilliseconds,
                "receiptTimestampMilliseconds": eventTimestampMilliseconds,
                "recordingTimeMilliseconds": recordingTimeMilliseconds,
                "acceleration": [
                    "x": -gravityConstant * user.x,
                    "y": -gravityConstant * user.y,
                    "z": -gravityConstant * user.z,
                ],
                "accelerationIncludingGravity": [
                    "x": gravityConstant * (gravity.x - user.x),
                    "y": gravityConstant * (gravity.y - user.y),
                    "z": gravityConstant * (gravity.z - user.z),
                ],
                // Safari's iOS rotationRate quirk: alpha/beta/gamma carry the
                // device x/y/z rates (deg/s).
                "rotationRateDegreesPerSecond": [
                    "alpha": motion.rotationRate.x * degreesPerRadian,
                    "beta": motion.rotationRate.y * degreesPerRadian,
                    "gamma": motion.rotationRate.z * degreesPerRadian,
                ],
                "intervalMilliseconds": 1000.0 / 60.0,
                "reportedInterval": 1.0 / 60.0,
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
        let count = sensorEventLines.count
        lock.unlock()
        onUpdate?(Update(sensorEventCount: count))
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
                "eventTimestampBasis": "boottimeMilliseconds",
                "receiptTimestampBasis": "boottimeMilliseconds",
            ],
            "counts": [
                "sensorEvents": sensorEventLines.count,
                "trackerFrames": recordedFrameCount,
            ],
            "device": [
                "platform": "iPhone",
                "userAgent": "ARTest2 native ARKit recorder",
            ],
            "files": [
                "sensorEvents": "sensor-events.ndjson",
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
