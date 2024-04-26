## bark.cpp WebAssembly demonstration

This demo runs `bark.cpp` in your browser using WebAssembly.

**Import Note: The forward pass is currently extremely slow. This is because ggml only has `WASM_SIMD` implementation for the matrix multiplication operation. This is left as an open experiment.**
See [155](https://github.com/PABannier/bark.cpp/pull/155) for more details.
**Important Note 2: Your browser must support 128-bit SIMD instructions.**

To build the example, run from the root of the repository:

```bash
mkdir build && cd build
emcmake cmake ..
make
```

You'll need [emscripten](https://emscripten.org). Follow the instructions [here](https://emscripten.org/docs/getting_started/downloads.html#sdk-download-and-install) to install it.

### Docker image

To prevent any dependency problems, you should build the Docker image specified by the Dockerfile located at the repository's root. This image compiles a WASM library that exposes segments of the Bark API and establishes an Nginx web server to serve static content for interacting with Bark.

```bash
docker build -t bark_wasm .
docker run -p 80:80 bark_wasm
```
