//
//  BarkState.swift
//  BarkCppDemo
//
//  Created by Pierre-Antoine BANNIER on 08/05/2024.
//

import Foundation
import AVFoundation

let kSampleRate: Double = 44100;
let kNumChannels: UInt32 = 1;


@MainActor
class BarkState: NSObject, ObservableObject {
    @Published var isModelLoaded = false
    @Published var messageLog = ""
    @Published var canGenerate = false
    @Published var isGenerating = false
    @Published var isPlaying = false
    
    private var barkContext: BarkContext?
    private var audioPlayer: AVAudioPlayer?
    
    private var modelUrl: URL? {
        Bundle.main.url(forResource: "ggml-weight", withExtension: "bin", subdirectory: "models")
    }
    
    private var generatedAudioUrl: URL? {
        Bundle.main.url(forResource: "output", withExtension: "wav", subdirectory: "generated")
    }
    
    private enum LoadError: Error {
        case couldNotLocateModel
    }
    
    override init() {
        super.init()
        do {
            try loadModel()
            canGenerate = true
        } catch {
            print(error.localizedDescription)
            messageLog += "\(error.localizedDescription)"
        }
    }
    
    private func loadModel() throws {
        messageLog += "Loading model...\n"
        if let modelUrl {
            barkContext = try BarkContext.createContext(path: modelUrl.path(), seed: 0)
            messageLog += "Loaded model \(modelUrl.lastPathComponent)"
        } else {
            messageLog += "Could not locate model\n"
        }
    }
    
    func generateAudioFromText(prompt: String) async {
        if (!canGenerate) {
            return
        }
        guard let barkContext else {
            return
        }
        
        do {
            canGenerate = false
            messageLog += "Generating audio...\n"
            let audioArray = try await barkContext.generateAudio(text: prompt)
            messageLog += "Audio generated!\n"
            loadAudioInPlayer(samples: audioArray)
        } catch {
            print(error.localizedDescription)
            messageLog += "\(error.localizedDescription)"
        }
        
        canGenerate = true
    }
    
    private func loadAudioInPlayer(samples: [Float]) {
        let audioFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: kSampleRate, channels: kNumChannels, interleaved: false)!
        var pcmBuffer = AVAudioPCMBuffer(pcmFormat: audioFormat, frameCapacity: AVAudioFrameCount(samples.count))!
        pcmBuffer.frameLength = pcmBuffer.frameCapacity

        let channelMemory = pcmBuffer.floatChannelData!
        for channel in 0..<Int(kNumChannels) {
            for i in 0..<samples.count {
                channelMemory[channel][i] = samples[i]
            }
        }

        // Create the audio player
        do {
            let audioFile = try AVAudioFile(forWriting: FileManager.default.temporaryDirectory.appendingPathComponent("temp.wav"), settings: audioFormat.settings)
            try audioFile.write(from: pcmBuffer)
            
            // Rewind and prepare to play
            audioPlayer = try AVAudioPlayer(contentsOf: audioFile.url, fileTypeHint: AVFileType.wav.rawValue)
            audioPlayer?.prepareToPlay()
        } catch {
            print("Failed to create audio player: \(error.localizedDescription)")
            messageLog += "\(error.localizedDescription)"
        }
    }
    
    func startPlayback() {
        audioPlayer?.play()
    }
    
    func stopPlayback() {
        audioPlayer?.pause()
    }
}
