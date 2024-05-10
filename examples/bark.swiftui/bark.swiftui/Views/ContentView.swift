//
//  ContentView.swift
//  BarkCppDemo
//
//  Created by Pierre-Antoine BANNIER on 08/05/2024.
//

import SwiftUI

struct ContentView: View {
    @StateObject var barkState = BarkState()
    @ObservedObject var progressData = ProgressData.shared
    @State private var textInput: String = ""
    
    var body: some View {
        NavigationStack {
            VStack {
                TextField("Enter your text here", text: $textInput)
                    .padding()
                    .textFieldStyle(RoundedBorderTextFieldStyle())
                    .multilineTextAlignment(.leading)
                    .frame(maxWidth: .infinity)
                
                HStack {
                    Button("Generate Audio", action: {
                        Task {
                            await barkState.generateAudioFromText(prompt: textInput)
                        }
                    })
                    .buttonStyle(.bordered)
                    .disabled(!barkState.canGenerate || barkState.isGenerating)

                    Button(barkState.isPlaying ? "Pause" : "Play", action: {
                        if barkState.isPlaying {
                            barkState.stopPlayback()
                        } else {
                            barkState.startPlayback()
                        }
                        barkState.isPlaying.toggle()
                    })
                    .buttonStyle(.bordered)
                    .disabled(!barkState.canGenerate)
                }
                
                HStack {
                    Text(verbatim: progressData.stepTitle)
                    
                    Spacer()
                    
                    ProgressView(value: progressData.progress)
                        .frame(width: 150)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding()
                
                ScrollView {
                    Text(verbatim: barkState.messageLog)
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .navigationTitle("Bark SwiftUI Demo")
            .padding()
        }
    }
}

#Preview {
    ContentView()
}
