#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include "bark.h"
#include "common.h"
#include "httplib.h"
#include "json.hpp"

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267)  // possible loss of data
#endif

#ifndef SERVER_VERBOSE
#define SERVER_VERBOSE 1
#endif

using namespace httplib;
using json = nlohmann::json;

struct server_params {
    std::string hostname = "127.0.0.1";
    std::string public_path = "examples/server/public";
    int32_t port = 1337;
    int32_t read_timeout = 600;
    int32_t write_timeout = 600;
};

void bark_print_usage(char **argv, const bark_params &params, const server_params &server_params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help              show this help message and exit\n");
    fprintf(stderr, "  -m MODEL, --model MODEL path to model\n");
    fprintf(stderr, "  -t N, --threads N       number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -p PORT, --port PORT    port number\n");
    fprintf(stderr, "  -a IP, --adress IP      ip adress\n");
    fprintf(stderr, "\n");
}

void bark_params_parse(int argc, char **argv, bark_params &params, server_params &server_params) {
    bool model_req = false;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-t" || arg == "--threads") {
            params.n_threads = std::stoi(argv[++i]);
        } else if (arg == "-m" || arg == "--model") {
            params.model_path = argv[++i];
            model_req = true;
        } else if (arg == "-t" || arg == "--thread") {
            params.n_threads = std::stoi(argv[++i]);
        } else if (arg == "-p" || arg == "--port") {
            server_params.port = std::stoi(argv[++i]);
        } else if (arg == "-a" || arg == "--address") {
            server_params.hostname = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            bark_print_usage(argv, params, server_params);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            bark_print_usage(argv, params, server_params);
            exit(1);
        }
    }
    if (!model_req) {
        fprintf(stderr, "error: no model path specified\n");
        bark_print_usage(argv, params, server_params);
        exit(1);
    }
}

bool generate_audio(int n_threads, bark_context *bctx, std::string text, std::string dest) {
    if (!bark_generate_audio(bctx, text.c_str(), n_threads)) {
        fprintf(stderr, "%s: An error occured. If the problem persists, feel free to open an issue to report it.\n", __func__);
        return false;
    }

    const float *audio_data = bark_get_audio_data(bctx);
    if (audio_data == NULL) {
        fprintf(stderr, "%s: Could not get audio data\n", __func__);
        return false;
    }

    const int audio_arr_size = bark_get_audio_data_size(bctx);

    std::vector<float> audio_arr(audio_data, audio_data + audio_arr_size);

    write_wav_on_disk(audio_arr, dest);
    return true;
}

int main(int argc, char **argv) {
    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();

    bark_params params;
    server_params server_params;
    bark_verbosity_level verbosity = bark_verbosity_level::LOW;

    bark_params_parse(argc, argv, params, server_params);

    struct bark_context_params ctx_params = bark_context_default_params();
    ctx_params.verbosity = verbosity;

    struct bark_context *bctx = bark_load_model(params.model_path.c_str(), ctx_params, params.seed);
    if (!bctx) {
        fprintf(stderr, "%s: Could not load model\n", __func__);
        return 1;
    }

    // bark_seed_rng(bctx, params.seed);

    std::mutex bark_mutex;

    Server svr;

    std::string default_content = "<html>hello</html>";

    // this is only called if no index.html is found in the public --path
    svr.Get("/", [&default_content](const Request &, Response &res) {
        res.set_content(default_content.c_str(), default_content.size(), "text/html");
        return false; });

    svr.Post("/bark", [&](const Request &req, Response &res) {
        // aquire bark model mutex lock
        bark_mutex.lock();

        json jreq = json::parse(req.body);
        std::string text = jreq.at("text");

        std::string dest_wav_path = "/tmp/bark_tmp.wav";
        
        // generate audio
        bool generated = generate_audio(params.n_threads, bctx, text, dest_wav_path);

        // read audio as binary
        std::ifstream wav_file(dest_wav_path, std::ios::binary);

        if (generated && wav_file.is_open()) {
            // Read the contents of the WAV file
            std::string wav_contents((std::istreambuf_iterator<char>(wav_file)),
                                     std::istreambuf_iterator<char>());

            // Set the response content type to audio/wav
            res.set_header("Content-Type", "audio/wav");

            // Set the response body to the WAV file contents
            res.set_content(wav_contents, "audio/wav");
        } else {
            // If the file cannot be opened, set a 500 Internal Server Error response
            res.status = 500;
            res.set_content("Internal Server Error", "text/plain");
        }

        // clean up
        std::remove("/tmp/bark_tmp.wav");

        // return bark model mutex lock
        bark_mutex.unlock(); });

    svr.set_read_timeout(server_params.read_timeout);
    svr.set_write_timeout(server_params.write_timeout);

    if (!svr.bind_to_port(server_params.hostname, server_params.port)) {
        fprintf(stderr, "\ncouldn't bind to server socket: hostname=%s port=%d\n\n",
                server_params.hostname.c_str(), server_params.port);
        return 1;
    }

    // Set the base directory for serving static files
    svr.set_base_dir(server_params.public_path);

    // to make it ctrl+clickable:
    printf("\nbark server listening at http://%s:%d\n\n",
           server_params.hostname.c_str(), server_params.port);

    if (!svr.listen_after_bind()) {
        return 1;
    }

    bark_free(bctx);

    return 0;
}
