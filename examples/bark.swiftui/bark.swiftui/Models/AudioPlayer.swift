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
        audioEngine = AVAudioEngine()
        playerNode = AVAudioPlayerNode()
        audioFormat = AVAudioFormat(standardFormatWithSampleRate: sampleRate, channels: 1)!
        
        // Setup the audio engine
        audioEngine.attach(playerNode)
        audioEngine.connect(playerNode, to: audioEngine.mainMixerNode, format: audioFormat)
        
        do {
            try audioEngine.start()
        } catch {
            print("Error starting audio engine: \(error)")
        }
        
        // Copy samples to the buffer
        buffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: UInt32(samples.count))!
        buffer.frameLength = buffer.frameCapacity
        let channelData = buffer.floatChannelData![0]
        
        for i in 0..<samples.count {
            channelData[i] = samples[i]
        }
    }

    func playAudio() {
        // Schedule the buffer and play
        playerNode.scheduleBuffer(buffer, at: nil, options: .loops, completionHandler: nil)
        playerNode.play()
    }

    func stop() {
        playerNode.stop()
    }
}

