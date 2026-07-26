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
import WebKit

struct ContentView: View {
    @StateObject private var recorder = RecordingSession()

    var body: some View {
        ZStack(alignment: .bottom) {
            ARSessionView(recorder: recorder)
                .edgesIgnoringSafeArea(.all)
            WebOverlayView(recorder: recorder)
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
            return recorder.browserSensorsReady
                ? "WK sensors ready · arkit \(recorder.arkitTrackingState)"
                : "Tap Start AR, then allow Motion & Orientation"
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
        .disabled(recorder.phase == .uploading || !recorder.browserSensorsReady)
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
        view.session.delegate = recorder.core
        view.debugOptions = [.showFeaturePoints, .showWorldOrigin]
        view.session.run(configuration)
        return view
    }

    func updateUIView(_ uiView: ARView, context: Context) {}
}

/// Transparent webview running the pizzanet web app in native-camera mode,
/// stacked over the ARKit view: the native camera is the backdrop, the page
/// renders its Three.js content on a clear background, and the recorder
/// pushes each throttled ARKit luma frame into the page.
struct WebOverlayView: UIViewRepresentable {
    let recorder: RecordingSession

    /// Owns both WebKit permission handling and the page-to-native recording
    /// bridge. Only the app's HTTPS top-level origin is trusted.
    final class Coordinator: NSObject, WKUIDelegate, WKScriptMessageHandler {
        static let allowedHost = "danlinux.warg-balance.ts.net"
        weak var recordingCore: RecordingCore?

        init(recordingCore: RecordingCore) {
            self.recordingCore = recordingCore
        }

        func webView(
            _ webView: WKWebView,
            requestDeviceOrientationAndMotionPermissionFor origin: WKSecurityOrigin,
            initiatedByFrame frame: WKFrameInfo,
            decisionHandler: @escaping (WKPermissionDecision) -> Void
        ) {
            guard
                frame.isMainFrame,
                origin.protocol == "https",
                origin.host == Self.allowedHost
            else {
                decisionHandler(.deny)
                return
            }
            // Let WebKit present its normal per-origin permission UI. The page
            // invokes requestPermission directly from its Start AR click.
            decisionHandler(.prompt)
        }

        func userContentController(
            _ userContentController: WKUserContentController,
            didReceive message: WKScriptMessage
        ) {
            guard
                message.name == "pizzanetRawCapture",
                message.frameInfo.isMainFrame,
                message.frameInfo.securityOrigin.protocol == "https",
                message.frameInfo.securityOrigin.host == Self.allowedHost
            else {
                return
            }
            recordingCore?.appendBrowserCaptureMessage(message.body)
        }
    }

    func makeCoordinator() -> Coordinator {
        Coordinator(recordingCore: recorder.core)
    }

    func makeUIView(context: Context) -> WKWebView {
        let configuration = WKWebViewConfiguration()
        configuration.allowsInlineMediaPlayback = true
        configuration.userContentController.add(
            context.coordinator,
            name: "pizzanetRawCapture"
        )
        let webView = WKWebView(frame: .zero, configuration: configuration)
        webView.uiDelegate = context.coordinator
        webView.isOpaque = false
        webView.backgroundColor = .clear
        webView.scrollView.backgroundColor = .clear
        webView.scrollView.isScrollEnabled = false
        if #available(iOS 16.4, *) {
            webView.isInspectable = true
        }
        let url = URL(string: "https://danlinux.warg-balance.ts.net/?nativeCamera=1")!
        webView.load(URLRequest(url: url))

        recorder.core.onLumaFrame = {
            [weak webView] frameId, timestamp, width, height, base64, longAxisFov in
            DispatchQueue.main.async {
                let script =
                    "window.__pizzanetNativeFrame && window.__pizzanetNativeFrame(\(frameId), \(timestamp), \(width), \(height), '\(base64)', \(longAxisFov));"
                webView?.evaluateJavaScript(script, completionHandler: nil)
            }
        }
        recorder.core.onBrowserCaptureRecordingState = { [weak webView] enabled in
            DispatchQueue.main.async {
                let script = enabled
                    ? "window.__pizzanetNativeRawCaptureEnabled = true;"
                    : "window.__pizzanetFlushNativeRawCapture && window.__pizzanetFlushNativeRawCapture(); window.__pizzanetNativeRawCaptureEnabled = false;"
                webView?.evaluateJavaScript(script, completionHandler: nil)
            }
        }
        return webView
    }

    func updateUIView(_ uiView: WKWebView, context: Context) {}

    static func dismantleUIView(_ uiView: WKWebView, coordinator: Coordinator) {
        uiView.configuration.userContentController.removeScriptMessageHandler(
            forName: "pizzanetRawCapture"
        )
        coordinator.recordingCore?.onBrowserCaptureRecordingState = nil
        coordinator.recordingCore?.onLumaFrame = nil
    }
}

#Preview {
    ContentView()
}
