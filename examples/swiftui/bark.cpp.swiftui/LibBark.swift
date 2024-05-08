import Foundation
import bark

enum BarkError: Error {
    case couldNotInitializeContext
    case couldNotGenerateAudio
}

actor BarkContext {
    private var context: OpaquePointer

    init(context: OpaquePointer) {
        self.context = context
    }

    deinit {
        bark_free(context)
    }

    func generateAudio(text: String) throws -> [Float] {
        // Leave 2 processors free (i.e. the high-efficiency cores).
        let maxThreads = max(1, min(8, cpuCount() - 2))
        print("Using \(maxThreads) threads for audio generation")
        if (bark_generate_audio(self.context, text, maxThreads)) {
            let audio = bark_get_audio_data(self.context)
            let audioLength = bark_get_audio_data_size(self.context)
            return Array(UnsafeBufferPointer(start: audio, count: audioLength))
        } else {
            throw BarkError.couldNotGenerateAudio
        }
    }

    static func createContext(path: String, seed: Int) throws -> BarkContext {
        let context = bark_load_model(path, 0 /* verbosity level (low) */, seed)
        if let context {
            return BarkContext(context: context)
        } else {
            print("Couldn't load model at \(path)")
            throw BarkError.couldNotInitializeContext
        }
    }
}

fileprivate func cpuCount() -> Int {
    ProcessInfo.processInfo.processorCount
}