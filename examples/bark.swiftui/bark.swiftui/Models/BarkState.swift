//
//  BarkState.swift
//  BarkCppDemo
//
//  Created by Pierre-Antoine BANNIER on 08/05/2024.
//

import Foundation
import AVFoundation


@MainActor
class BarkState: NSObject, ObservableObject {
    @Published var isModelLoaded = false
    @Published var messageLog = ""
    @Published var canGenerate = false
    @Published var isGenerating = false
    @Published var isPlaying = false
    
    private var barkContext: BarkContext?
    private var audioPlayer: AudioPlayer?
    
    private var modelUrl: URL? {
        Bundle.main.url(forResource: "ggml_weights", withExtension: "bin")
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
        }
    }
    
    private func loadModel() throws {
        messageLog += "Loading model...\n"
        if let modelUrl {
            barkContext = try BarkContext.createContext(path: modelUrl.path(), seed: 0)
            messageLog += "Loaded model \(modelUrl.lastPathComponent)\n"
        } else {
            messageLog += "Could not locate model\n"
            throw LoadError.couldNotLocateModel
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
            audioPlayer = AudioPlayer(samples: audioArray)
        } catch {
            print(error.localizedDescription)
            messageLog += "\(error.localizedDescription)"
        }
        
        canGenerate = true
    }
    
    func startPlayback() {
        if (audioPlayer != nil) {
            audioPlayer!.playAudio()
        }
    }
    
    func stopPlayback() {
        if (audioPlayer != nil) {
            audioPlayer!.stop()
        }
    }
    
}
