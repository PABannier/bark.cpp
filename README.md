# bark.cpp (coming soon!)

Inference of SunoAI's bark model in pure C/C++ using [ggml](https://github.com/ggerganov/ggml).

## Description

The main goal of `bark.cpp` is to synthesize audio from a textual input with the [Bark](https://github.com/suno-ai/bark) model.

Bark has essentially 4 components:
- [x] Semantic model to encode the text input
- [x] Coarse model
- [x] Fine model
- [ ] Encoder (quantizer + decoder) to generate the waveform from the tokens

## Roadmap

- [ ] Quantization
- [ ] FP16
- [ ] Swift package for iOS devices
