//
// Copyright Amazon.com Inc. or its affiliates.
// All Rights Reserved.
//
// SPDX-License-Identifier: Apache-2.0
//

import UIKit
import AVFoundation

class LivenessCaptureSession {
    let captureDevice: LivenessCaptureDevice
    private let captureQueue = DispatchQueue(label: "com.amazonaws.faceliveness.cameracapturequeue")
    private let configurationQueue = DispatchQueue(label: "com.amazonaws.faceliveness.sessionconfiguration", qos: .userInteractive)
    let outputDelegate: AVCaptureVideoDataOutputSampleBufferDelegate
    var captureSession: AVCaptureSession?
    
    var outputSampleBufferCapturer: OutputSampleBufferCapturer? {
        return outputDelegate as? OutputSampleBufferCapturer
    }

    init(captureDevice: LivenessCaptureDevice, outputDelegate: AVCaptureVideoDataOutputSampleBufferDelegate) {
        self.captureDevice = captureDevice
        self.outputDelegate = outputDelegate
    }

    func configureCamera(frame: CGRect) throws -> CALayer {
        try configureCamera()

        guard let captureSession = captureSession else {
            throw LivenessCaptureSessionError.captureSessionUnavailable
        }
        
        let previewLayer = previewLayer(
            frame: frame,
            for: captureSession
        )

        return previewLayer
    }
    
    func configureCamera() throws {
        guard let camera = captureDevice.avCaptureDevice
        else { throw LivenessCaptureSessionError.cameraUnavailable }

        let cameraInput = try AVCaptureDeviceInput(device: camera)
        let videoOutput = AVCaptureVideoDataOutput()

        teardownExistingSession(input: cameraInput)
        captureSession = AVCaptureSession()

        guard let captureSession = captureSession else {
            throw LivenessCaptureSessionError.captureSessionUnavailable
        }

        try setupInput(cameraInput, for: captureSession)
        captureSession.sessionPreset = captureDevice.preset
        try setupOutput(videoOutput, for: captureSession)
        try captureDevice.configure()

        videoOutput.setSampleBufferDelegate(
            outputDelegate,
            queue: captureQueue
        )
    }

    func startSession() {
        guard let session = captureSession else { return }
        configurationQueue.async {
            session.startRunning()
        }
    }

    /// Stops the session on `configurationQueue`, the queue `startSession()` starts it on.
    /// `stopRunning()` and removing inputs commit the configuration synchronously, and called on
    /// the main thread they held it for up to 10 s (Frienda FRD-4429). One serial queue for both
    /// also keeps the order: a stop that comes while the start is still queued runs after it,
    /// instead of finding the session not yet running and leaving it to start with no owner.
    func stopRunning() {
        guard let session = captureSession else { return }
        captureSession = nil
        configurationQueue.async {
            Self.tearDown(session)
        }
    }

    /// Before a new session is configured for the same camera: the old one is gone when this
    /// returns, because the new one adds its input on the calling thread right after.
    private func teardownExistingSession(input: AVCaptureDeviceInput) {
        guard let session = captureSession else { return }
        captureSession = nil
        configurationQueue.sync {
            Self.tearDown(session)
        }
    }

    private static func tearDown(_ session: AVCaptureSession) {
        if session.isRunning {
            session.stopRunning()
        }

        for input in session.inputs {
            session.removeInput(input)
        }

        for output in session.outputs {
            session.removeOutput(output)
        }
    }

    private func setupInput(
        _ input: AVCaptureDeviceInput,
        for captureSession: AVCaptureSession
    ) throws {
        if captureSession.canAddInput(input) {
            captureSession.addInput(input)
        } else {
            throw LivenessCaptureSessionError.captureSessionInputUnavailable
        }
    }

    private func setupOutput(
        _ output: AVCaptureVideoDataOutput,
        for captureSession: AVCaptureSession
    ) throws {
        if captureSession.canAddOutput(output) {
            captureSession.addOutput(output)
        } else {
            throw LivenessCaptureSessionError.captureSessionOutputUnavailable
        }
        output.videoSettings = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA
        ]

        output.connections
            .filter(\.isVideoOrientationSupported)
            .forEach {
                $0.videoOrientation = .portrait
                $0.isVideoMirrored = true
        }
    }

    private func previewLayer(
        frame: CGRect,
        for captureSession: AVCaptureSession
    ) -> AVCaptureVideoPreviewLayer {
        let previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        previewLayer.videoGravity = .resizeAspectFill
        previewLayer.connection?.videoOrientation = .portrait
        previewLayer.frame = frame
        return previewLayer
    }
}
