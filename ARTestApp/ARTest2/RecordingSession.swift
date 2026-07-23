//
//  RecordingSession.swift
//  ARTest2
//
//  Records an ARKit session as BOTH:
//  1. Ground truth: per-frame ARKit camera poses (centimetre-accurate VIO) →
//     arkit-poses.ndjson
//  2. A pizzanet web-format tracking recording (manifest + sensor-events
//     + tracker-frames + tracker-luma.gray), with camera luma taken from the
//     same ARKit frames and Safari-convention sensor events synthesized from
//     CoreMotion — so the Rust tracker can be replayed offline against the
//     ARKit truth for the exact same motion.
//
//  Everything is stamped with the same boot-time clock (ARFrame.timestamp and
//  CMDeviceMotion.timestamp share it), so the two streams are inherently
//  synchronized.
//

import ARKit
import CoreMotion
import Foundation

@MainActor
final class RecordingSession: NSObject, ObservableObject, ARSessionDelegate {
    /// Where the pizzanet dev server accepts recordings (Vite proxies /api).
    static let uploadURL = URL(string: "https://danlinux.warg-balance.ts.net/api/dev/recordings")!

    /// Tracker frame geometry, matching the web app's capture settings.
    static let trackerFrameWidth = 240
    static let targetFrameIntervalSeconds = 1.0 / 30.0

    enum Phase: Equatable {
        case idle
        case recording
        case uploading
        case done(String)
        case failed(String)
    }

    @Published var phase: Phase = .idle
    @Published var frameCount = 0
    @Published var sensorEventCount = 0
    @Published var arkitTrackingState = "—"

    private let motionManager = CMMotionManager()
    private let motionQueue = OperationQueue()

    // Recording state. Sensor callbacks land on motionQueue; frame callbacks
    // on the session's queue; both funnel through the lock below.
    private let stateLock = NSLock()
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
    private var cameraIntrinsics: simd_float3x3?
    private var cameraImageSize = CGSize.zero

    // MARK: - Controls

    func toggleRecording() {
        if isRecording {
            stopAndUpload()
        } else {
            start()
        }
    }

    private func start() {
        stateLock.lock()
        defer { stateLock.unlock() }
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
        phase = .recording
        frameCount = 0
        sensorEventCount = 0
    }

    private func stopAndUpload() {
        stateLock.lock()
        guard isRecording else {
            stateLock.unlock()
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
        let manifest = buildManifest(durationMilliseconds: durationMilliseconds)
        let lumaURL = lumaFileURL
        let frames = recordedFrameCount
        stateLock.unlock()

        guard frames > 10, let lumaURL else {
            phase = .failed("Recording too short.")
            return
        }
        phase = .uploading
        Task {
            do {
                let response = try await Self.upload(
                    manifest: manifest,
                    sensorEvents: sensorEvents,
                    frameEvents: frameEvents,
                    arkitPoses: arkitPoses,
                    lumaFileURL: lumaURL
                )
                self.phase = .done(response)
            } catch {
                self.phase = .failed(error.localizedDescription)
            }
        }
    }

    // MARK: - ARKit frames

    nonisolated func session(_ session: ARSession, didUpdate frame: ARFrame) {
        let trackingLabel: String
        switch frame.camera.trackingState {
        case .normal: trackingLabel = "normal"
        case .notAvailable: trackingLabel = "unavailable"
        case .limited(let reason): trackingLabel = "limited(\(reason))"
        }
        Task { @MainActor in
            self.arkitTrackingState = trackingLabel
        }
        processFrameForRecording(frame, trackingLabel: trackingLabel)
    }

    private nonisolated func processFrameForRecording(_ frame: ARFrame, trackingLabel: String) {
        stateLock.lock()
        defer { stateLock.unlock() }
        guard isRecording else { return }
        guard frame.timestamp >= nextFrameAtSeconds else { return }
        nextFrameAtSeconds =
            max(nextFrameAtSeconds + Self.targetFrameIntervalSeconds, frame.timestamp - 0.005)

        let recordingTimeMilliseconds = (frame.timestamp - startUptimeSeconds) * 1000.0
        let eventTimestampMilliseconds = frame.timestamp * 1000.0

        guard let (luma, width, height) = Self.portraitLuma(
            from: frame.capturedImage,
            targetWidth: Self.trackerFrameWidth
        ) else {
            return
        }
        if trackerFrameHeight == 0 {
            trackerFrameHeight = height
            cameraIntrinsics = frame.camera.intrinsics
            cameraImageSize = frame.camera.imageResolution
        }
        guard height == trackerFrameHeight else { return }

        lumaFileHandle?.write(luma)
        let frameId = nextFrameId
        nextFrameId += 1
        recordedFrameCount += 1

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
        // three.js convention (x right, y up, camera looks down -z); world is
        // gravity-aligned with y up.
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

        let frames = recordedFrameCount
        Task { @MainActor in
            self.frameCount = frames
        }
    }

    /// Extracts the Y (luma) plane, rotates landscape → portrait (90° CW), and
    /// box-downsamples to `targetWidth` columns.
    private nonisolated static func portraitLuma(
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

        // Portrait dimensions: the landscape sensor image rotated 90° CW.
        let destWidth = targetWidth
        let destHeight = Int(
            (Double(targetWidth) * Double(sourceWidth) / Double(sourceHeight)).rounded()
        )
        var out = Data(count: destWidth * destHeight)
        out.withUnsafeMutableBytes { (buffer: UnsafeMutableRawBufferPointer) in
            let dest = buffer.bindMemory(to: UInt8.self).baseAddress!
            for destY in 0..<destHeight {
                // Portrait row destY samples along the sensor's x axis.
                let sourceX = min(
                    sourceWidth - 1,
                    Int(Double(destY) / Double(destHeight) * Double(sourceWidth))
                )
                for destX in 0..<destWidth {
                    // Portrait column destX samples the sensor's y axis,
                    // reversed (90° clockwise rotation).
                    let sourceY = min(
                        sourceHeight - 1,
                        sourceHeight - 1
                            - Int(Double(destX) / Double(destWidth) * Double(sourceHeight))
                    )
                    // 2x2 box sample for mild antialiasing.
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

    // MARK: - CoreMotion → Safari-convention sensor events

    private nonisolated func appendMotion(_ motion: CMDeviceMotion) {
        let gravityConstant = 9.80665
        let eventTimestampMilliseconds = motion.timestamp * 1000.0

        stateLock.lock()
        defer { stateLock.unlock() }
        guard isRecording else { return }
        let recordingTimeMilliseconds = (motion.timestamp - startUptimeSeconds) * 1000.0

        // Safari on iOS reports the NEGATED spec values (the long-standing
        // WebKit sign inversion the web pipeline already corrects for):
        //   acceleration               = -9.81 * userAcceleration
        //   accelerationIncludingGravity = 9.81 * (gravity - userAcceleration)
        // (CMDeviceMotion gravity/userAcceleration are in g, device axes match
        // the W3C device frame: x right, y top, z out of screen.)
        let userAcceleration = motion.userAcceleration
        let gravity = motion.gravity
        let acceleration: [String: Double] = [
            "x": -gravityConstant * userAcceleration.x,
            "y": -gravityConstant * userAcceleration.y,
            "z": -gravityConstant * userAcceleration.z,
        ]
        let accelerationIncludingGravity: [String: Double] = [
            "x": gravityConstant * (gravity.x - userAcceleration.x),
            "y": gravityConstant * (gravity.y - userAcceleration.y),
            "z": gravityConstant * (gravity.z - userAcceleration.z),
        ]
        // Safari's iOS rotationRate quirk: alpha/beta/gamma carry the device
        // x/y/z rates (deg/s) — the web pipeline expects exactly that order.
        let degreesPerRadian = 180.0 / Double.pi
        let rotationRate: [String: Double] = [
            "alpha": motion.rotationRate.x * degreesPerRadian,
            "beta": motion.rotationRate.y * degreesPerRadian,
            "gamma": motion.rotationRate.z * degreesPerRadian,
        ]
        sensorEventLines.append(
            jsonLine([
                "kind": "device_motion",
                "eventTimestampMilliseconds": eventTimestampMilliseconds,
                "receiptTimestampMilliseconds": eventTimestampMilliseconds,
                "recordingTimeMilliseconds": recordingTimeMilliseconds,
                "acceleration": acceleration,
                "accelerationIncludingGravity": accelerationIncludingGravity,
                "rotationRateDegreesPerSecond": rotationRate,
                "intervalMilliseconds": motionManager.deviceMotionUpdateInterval * 1000.0,
                "reportedInterval": motionManager.deviceMotionUpdateInterval,
                "screenAngleDegrees": 0,
                "screenOrientation": "portrait-primary",
            ])
        )

        // W3C deviceorientation: decompose the attitude (device → earth,
        // reference frame Z vertical, X arbitrary — matching Safari's
        // gyro-relative alpha) as R = Rz(alpha)·Rx(beta)·Ry(gamma):
        //   beta  = asin(m32), alpha = atan2(-m12, m22), gamma = atan2(-m31, m33)
        // Sanity anchor: a phone held upright in portrait must give beta ≈ 90.
        // If it reads ≈ 0 instead, CMRotationMatrix is the transpose of what
        // this assumes — swap to the transposed elements.
        let m = motion.attitude.rotationMatrix
        let beta = asin(max(-1.0, min(1.0, m.m32))) * degreesPerRadian
        var alpha = atan2(-m.m12, m.m22) * degreesPerRadian
        if alpha < 0 { alpha += 360.0 }
        let gamma = atan2(-m.m31, m.m33) * degreesPerRadian
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
        Task { @MainActor in
            self.sensorEventCount = count
        }
    }

    // MARK: - Manifest + upload

    private func buildManifest(durationMilliseconds: Double) -> String {
        let intrinsics = cameraIntrinsics
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        var manifest: [String: Any] = [
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
                "videoWidth": Int(cameraImageSize.height),
                "videoHeight": Int(cameraImageSize.width),
                "longAxisFieldOfViewDegrees": arkitLongAxisFieldOfViewDegrees() ?? 68,
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
        if let intrinsics {
            manifest["arkitIntrinsics"] = [
                "fx": intrinsics.columns.0.x,
                "fy": intrinsics.columns.1.y,
                "cx": intrinsics.columns.2.x,
                "cy": intrinsics.columns.2.y,
                "imageWidth": Int(cameraImageSize.width),
                "imageHeight": Int(cameraImageSize.height),
            ]
        }
        let data = try? JSONSerialization.data(withJSONObject: manifest)
        return data.flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
    }

    /// The true long-axis FOV from ARKit's calibrated intrinsics — this also
    /// calibrates the web tracker's FOV assumption for this exact device.
    private func arkitLongAxisFieldOfViewDegrees() -> Double? {
        guard let intrinsics = cameraIntrinsics, cameraImageSize.width > 0 else {
            return nil
        }
        let longAxisPixels = Double(max(cameraImageSize.width, cameraImageSize.height))
        let focal = Double(intrinsics.columns.0.x)
        return 2.0 * atan(longAxisPixels / (2.0 * focal)) * 180.0 / Double.pi
    }

    private nonisolated static func upload(
        manifest: String,
        sensorEvents: String,
        frameEvents: String,
        arkitPoses: String,
        lumaFileURL: URL
    ) async throws -> String {
        let boundary = "pizzanet-\(UUID().uuidString)"
        var body = Data()
        func addField(_ name: String, _ value: String, filename: String? = nil) {
            body.append("--\(boundary)\r\n".data(using: .utf8)!)
            if let filename {
                body.append(
                    "Content-Disposition: form-data; name=\"\(name)\"; filename=\"\(filename)\"\r\n\r\n"
                        .data(using: .utf8)!
                )
            } else {
                body.append(
                    "Content-Disposition: form-data; name=\"\(name)\"\r\n\r\n"
                        .data(using: .utf8)!
                )
            }
            body.append(value.data(using: .utf8)!)
            body.append("\r\n".data(using: .utf8)!)
        }
        addField("manifest", manifest)
        addField("sensorEvents", sensorEvents)
        addField("frameEvents", frameEvents)
        addField("arkitPoses", arkitPoses)
        let luma = try Data(contentsOf: lumaFileURL)
        body.append("--\(boundary)\r\n".data(using: .utf8)!)
        body.append(
            "Content-Disposition: form-data; name=\"trackerLuma\"; filename=\"tracker-luma.gray\"\r\nContent-Type: application/octet-stream\r\n\r\n"
                .data(using: .utf8)!
        )
        body.append(luma)
        body.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)

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
        return String(data: responseData, encoding: .utf8) ?? "uploaded"
    }
}

/// Serializes a dictionary to a single NDJSON line with stable key order.
private func jsonLine(_ object: [String: Any]) -> String {
    let data = try? JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    return data.flatMap { String(data: $0, encoding: .utf8) } ?? "{}"
}
