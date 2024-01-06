# bark.cpp

![bark.cpp](./assets/banner.png)

[![Actions Status](https://github.com/PABannier/bark.cpp/actions/workflows/build.yml/badge.svg)](https://github.com/PABannier/bark.cpp/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

[Roadmap](https://github.com/users/PABannier/projects/1) / [encodec.cpp](https://github.com/PABannier/encodec.cpp) / [ggml](https://github.com/ggerganov/ggml)

Inference of [SunoAI's bark model](https://github.com/suno-ai/bark) in pure C/C++.

## Description

With `bark.cpp`, our goal is to bring **real-time realistic** text-to-speech generation to the community.
Currently, we are focused on porting the [Bark](https://github.com/suno-ai/bark) model in C++.

- [X] Plain C/C++ implementation without dependencies
- [X] AVX, AVX2 and AVX512 for x86 architectures
- [X] Mixed F16 / F32 precision
- [X] 4-bit, 5-bit and 8-bit integer quantization
- [ ] Optimized via ARM NEON, Accelerate and Metal frameworks
- [ ] iOS on-device deployment using CoreML

The original implementation of `bark.cpp` is the bark's 24Khz English model. We expect to support multiple encoders in the future (see [this](https://github.com/PABannier/bark.cpp/issues/36) and [this](https://github.com/PABannier/bark.cpp/issues/6)), as well as music generation model (see [this](https://github.com/PABannier/bark.cpp/issues/62)). This project is for educational purposes.

Demo on [Google Colab](https://colab.research.google.com/drive/1JVtJ6CDwxtKfFmEd8J4FGY2lzdL0d0jT?usp=sharing) ([#95](https://github.com/PABannier/bark.cpp/issues/95))

**Supported platforms:**

- [X] Mac OS
- [X] Linux
- [X] Windows

**Supported models:**

- [X] Bark
- [ ] Vocos
- [ ] AudioCraft

---

Here is a typical run using `bark.cpp`:

```java
make -j && ./main -p "This is an audio generated by bark.cpp"

   __               __
   / /_  ____ ______/ /__        _________  ____
  / __ \/ __ `/ ___/ //_/       / ___/ __ \/ __ \
 / /_/ / /_/ / /  / ,<    _    / /__/ /_/ / /_/ /
/_.___/\__,_/_/  /_/|_|  (_)   \___/ .___/ .___/
                                  /_/   /_/


bark_tokenize_input: prompt: 'this is a dog barking.'
bark_tokenize_input: number of tokens in prompt = 513, first 8 tokens: 20579 20172 10217 27883 28169 25677 10167 129595

Generating semantic tokens: [========>                                          ] (17%)

bark_print_statistics: mem per token =     0.00 MB
bark_print_statistics:   sample time =     9.90 ms / 138 tokens
bark_print_statistics:  predict time =  3163.78 ms / 22.92 ms per token
bark_print_statistics:    total time =  3188.37 ms

Generating coarse tokens: [==================================================>] (100%)

bark_print_statistics: mem per token =     0.00 MB
bark_print_statistics:   sample time =     3.96 ms / 410 tokens
bark_print_statistics:  predict time = 14303.32 ms / 34.89 ms per token
bark_print_statistics:    total time = 14315.52 ms

Generating fine tokens: [==================================================>] (100%)

bark_print_statistics: mem per token =     0.00 MB
bark_print_statistics:   sample time =    41.93 ms / 6144 tokens
bark_print_statistics:  predict time = 15234.38 ms / 2.48 ms per token
bark_print_statistics:    total time = 15282.15 ms

Number of frames written = 51840.

main:     load time =  1436.36 ms
main:     eval time = 34520.53 ms
main:    total time = 32786.04 ms
```

Here are typical audio pieces generated by `bark.cpp`:

https://github.com/PABannier/bark.cpp/assets/12958149/f9f240fd-975f-4d69-9bb3-b295a61daaff

https://github.com/PABannier/bark.cpp/assets/12958149/c0caadfd-bed9-4a48-8c17-3215963facc1

## Usage

Here are the steps for the bark model.

### Get the code

```bash
git clone --recursive https://github.com/PABannier/bark.cpp.git
cd bark.cpp
git submodule update --init --recursive
```

### Build

In order to build bark.cpp you must use `CMake`:

```bash
mkdir bark/build
cd bark/build
cmake ..
cmake --build . --config Release
```

### Prepare data & Run

```bash
# install Python dependencies
python3 -m pip install -r bark/requirements.txt

# obtain the original bark and encodec weights and place them in ./models
python3 bark/download_weights.py --download-dir ./models

# convert the model to ggml format
python3 bark/convert.py \
        --dir-model ./models \
        --vocab-path ./ggml_weights/ \
        --out-dir ./ggml_weights/

# run the inference
./bark/build/examples/main/main -m ./ggml_weights/ -p "this is an audio"
```

### (Optional) Quantize weights

Weights can be quantized using the following strategy: `q4_0`, `q4_1`, `q5_0`, `q5_1`, `q8_0`.

Note that to preserve audio quality, we do not quantize the codec model. The bulk of the
computation is in the forward pass of the GPT models.

```bash
mkdir ggml_weights_q4
cp ggml_weights/*vocab* ggml_weights_q4
./bark/build/examples/quantize/quantize ./ggml_weights/ggml_weights_text.bin ./ggml_weights_q4/ggml_weights_text.bin q4_0
./bark/build/examples/quantize/quantize ./ggml_weights/ggml_weights_coarse.bin ./ggml_weights_q4/ggml_weights_coarse.bin q4_0
./bark/build/examples/quantize/quantize ./ggml_weights/ggml_weights_fine.bin ./ggml_weights_q4/ggml_weights_fine.bin q4_0
```

### Seminal papers and background on models

- Bark
    - [Text Prompted Generative Audio](https://github.com/suno-ai/bark)
- Encodec
    - [High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438)
- GPT-3
    - [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)

### Contributing

`bark.cpp` is a continuous endeavour that relies on the community efforts to last and evolve. Your contribution is welcome and highly valuable. It can be

- bug report: you may encounter a bug while using `bark.cpp`. Don't hesitate to report it on the issue section.
- feature request: you want to add a new model or support a new platform. You can use the issue section to make suggestions.
- pull request: you may have fixed a bug, added a features, or even fixed a small typo in the documentation, ... you can submit a pull request and a reviewer will reach out to you.

### Coding guidelines

- Avoid adding third-party dependencies, extra files, extra headers, etc.
- Always consider cross-compatibility with other operating systems and architectures
- Avoid fancy looking modern STL constructs, keep it simple
- Clean-up any trailing whitespaces, use 4 spaces for indentation, brackets on the same line, `void * ptr`, `int & ref`
