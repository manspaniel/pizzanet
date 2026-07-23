//
//  ContentView.swift
//  ARTest2
//
//  ARKit ground-truth recorder for the pizzanet web tracker: shows the live
//  AR session (with feature points), and a Record button that captures both
//  the ARKit poses and a web-format tracking recording from the same frames,
//  then uploads everything to the dev server over Tailscale.
//

import ARKit
import RealityKit
import SwiftUI

struct ContentView: View {
    @StateObject private var recorder = RecordingSession()

    var body: some View {
        ZStack(alignment: .bottom) {
            ARSessionView(recorder: recorder)
                .edgesIgnoringSafeArea(.all)
            VStack(spacing: 10) {
                statusLine
                recordButton
            }
            .padding(.bottom, 40)
        }
    }

    private var statusLine: some View {
        Text(statusText)
            .font(.system(.footnote, design: .monospaced))
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .background(.black.opacity(0.55), in: Capsule())
            .foregroundStyle(.white)
    }

    private var statusText: String {
        switch recorder.phase {
        case .idle:
            return "arkit \(recorder.arkitTrackingState)"
        case .recording:
            return "REC \(recorder.frameCount) frames · \(recorder.sensorEventCount) imu · arkit \(recorder.arkitTrackingState)"
        case .uploading:
            return "uploading…"
        case .done:
            return "uploaded ✓"
        case .failed(let message):
            return "failed: \(message)"
        }
    }

    private var recordButton: some View {
        Button {
            recorder.toggleRecording()
        } label: {
            Text(recorder.phase == .recording ? "Stop + Upload" : "Record")
                .font(.headline)
                .padding(.horizontal, 28)
                .padding(.vertical, 12)
                .background(
                    recorder.phase == .recording ? Color.red : Color.blue,
                    in: Capsule()
                )
                .foregroundStyle(.white)
        }
        .disabled(recorder.phase == .uploading)
    }
}

/// Hosts an ARView with a manually-run world-tracking session so the recorder
/// receives every ARFrame. Prefers a 16:9 video format to match the web app's
/// camera aspect.
struct ARSessionView: UIViewRepresentable {
    let recorder: RecordingSession

    func makeUIView(context: Context) -> ARView {
        let view = ARView(frame: .zero)
        let configuration = ARWorldTrackingConfiguration()
        configuration.worldAlignment = .gravity
        if let wideFormat = ARWorldTrackingConfiguration.supportedVideoFormats.first(where: {
            $0.imageResolution.width == 1920 && $0.imageResolution.height == 1080
        }) {
            configuration.videoFormat = wideFormat
        }
        view.session.delegate = recorder
        view.debugOptions = [.showFeaturePoints, .showWorldOrigin]
        view.session.run(configuration)
        return view
    }

    func updateUIView(_ uiView: ARView, context: Context) {}
}

#Preview {
    ContentView()
}
