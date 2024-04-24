## bark.cpp WebAssembly demonstration

This demo runs `bark.cpp` in your browser using WebAssembly.

**Important Note: Your browser must support 128-bit SIMD instructions.**

To build the example, run from the root of the repository:

```bash
mkdir build && cd build
emcmake cmake ..
make
```

You'll need [emscripten](https://emscripten.org). Follow the instructions [here](https://emscripten.org/docs/getting_started/downloads.html#sdk-download-and-install) to install it.
