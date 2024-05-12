//
//  AudioPlayer.swift
//  bark.swiftui
//
//  Created by Pierre-Antoine BANNIER on 10/05/2024.
//

import Foundation
import AVFoundation

class AudioPlayer {
    private var audioEngine: AVAudioEngine
    private var playerNode: AVAudioPlayerNode
    private var audioFormat: AVAudioFormat
    private var buffer: AVAudioPCMBuffer

    init(samples: [Float], sampleRate: Double = 24000.0) {
        // Initialize the AVFoundation objects
        audioEngine = AVAudioEngine()
        playerNode = AVAudioPlayerNode()
        audioFormat = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
        
        // Configure the audio session for playback
        let audioSession = AVAudioSession.sharedInstance()
        do {
            try audioSession.setCategory(.playback, mode: .default)
            try audioSession.setActive(true)
        } catch {
            print("Failed to configure audio session: \(error)")
        }

        // Set up the audio engine
        audioEngine.attach(playerNode)
        audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: audioFormat)
        
        do {
            try audioEngine.start()
        } catch {
            print("Error starting audio engine: \(error)")
        }
        
        // Prepare the buffer
        buffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: UInt32(samples.count))!
        buffer.frameLength = buffer.frameCapacity
        let channelData = buffer.floatChannelData![0]
        
        // Copy samples to buffer
        for i in 0..<samples.count {
            channelData[i] = samples[i]
        }
    }

    func playAudio() {
        // Schedule the buffer and play
        playerNode.scheduleBuffer(buffer, at: nil, options: [], completionHandler: nil)
        playerNode.play()
    }

    func stop() {
        playerNode.stop()
        // Optionally deactivate the audio session
        do {
            try AVAudioSession.sharedInstance().setActive(false)
        } catch {
            print("Failed to deactivate audio session: \(error)")
        }
    }
}

