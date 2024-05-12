//
//  ProgressData.swift
//  bark.swiftui
//
//  Created by Pierre-Antoine BANNIER on 10/05/2024.
//

import Foundation
import Combine
import bark

class ProgressData: ObservableObject {
    static let shared = ProgressData()

    @Published var progress: Float = 0.0
    @Published var stepTitle: String = "Progress (0%)"

    private init() {}
}

func cCallbackBridge(bctx: OpaquePointer?, step: bark_encoding_step, progress: Int32, userData: UnsafeMutableRawPointer?) {
    DispatchQueue.main.async {
        let progressValue = Float(progress) / 100.0
        var stepTitle: String

        switch step {
        case bark_encoding_step(rawValue: 0):
            stepTitle = "Semantic tokens (1/3)"
        case bark_encoding_step(rawValue: 1):
            stepTitle = "Coarse tokens (2/3)"
        default:
            stepTitle = "Fine tokens (3/3)"
        }

        // Update the shared observable object
        ProgressData.shared.progress = progressValue
        ProgressData.shared.stepTitle = stepTitle
    }
}
