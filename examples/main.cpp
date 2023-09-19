#include "ggml/include/ggml/ggml.h"
#include "bark.h"

#include <tuple>

struct bark_params {
    int32_t n_threads = std::min(4, (int32_t) std::thread::hardware_concurrency());

    // user prompt
    std::string prompt = "this is an audio";

    // paths
    std::string model_path = "./ggml_weights";
    std::string dest_wav_path = "output.wav";

    int32_t seed = 0;
};

void bark_print_usage(char ** argv, const bark_params & params) {
    fprintf(stderr, "usage: %s [options]\n", argv[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "options:\n");
    fprintf(stderr, "  -h, --help            show this help message and exit\n");
    fprintf(stderr, "  -t N, --threads N     number of threads to use during computation (default: %d)\n", params.n_threads);
    fprintf(stderr, "  -s N, --seed N        seed for random number generator (default: %d)\n", params.seed);
    fprintf(stderr, "  -p PROMPT, --prompt PROMPT\n");
    fprintf(stderr, "                        prompt to start generation with (default: random)\n");
    fprintf(stderr, "  -m FNAME, --model FNAME\n");
    fprintf(stderr, "                        model path (default: %s)\n", params.model_path.c_str());
    fprintf(stderr, "  -o FNAME, --outwav FNAME\n");
    fprintf(stderr, "                        output generated wav (default: %s)\n", params.dest_wav_path.c_str());
    fprintf(stderr, "\n");
}

int bark_params_parse(int argc, char ** argv, bark_params & params) {
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-t" || arg == "--threads") {
            params.n_threads = std::stoi(argv[++i]);
        } else if (arg == "-p" || arg == "--prompt") {
            params.prompt = argv[++i];
        } else if (arg == "-m" || arg == "--model") {
            params.model_path = argv[++i];
        } else if (arg == "-s" || arg == "--seed") {
            params.seed = std::stoi(argv[++i]);
        } else if (arg == "-o" || arg == "--outwav") {
            params.dest_wav_path = argv[++i];
        } else if (arg == "-h" || arg == "--help") {
            bark_print_usage(argv, params);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg.c_str());
            bark_print_usage(argv, params);
            exit(0);
        }
    }

    return 0;
}

std::tuple<struct bark_model *, struct bark_context *> bark_init_from_params(bark_params & params) {
    bark_model * model = bark_load_model_from_file(params.model_path.c_str());
    if (model == NULL) {
        fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model_path.c_str());
        return std::make_tuple(nullptr, nullptr);
    }

    bark_context_params bctx_params = bark_context_default_params();
    bark_context * bctx = bark_new_context_with_model(model, bctx_params);
    if (bctx == NULL) {
        fprintf(stderr, "%s: error: failed to create context with model '%s'\n", __func__, params.model_path.c_str());
        bark_free_model(model);
        return std::make_tuple(nullptr, nullptr);
    }

    return std::make_tuple(model, bctx);
}

int main(int argc, char **argv) {
    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();

    bark_params params;

    if (bark_params_parse(argc, argv, params) > 0) {
        fprintf(stderr, "%s: Could not parse arguments\n", __func__);
        return 1;
    }

    int64_t t_load_us = 0;
    int64_t t_eval_us = 0;

    bark_context * bctx;
    bark_model * model;

    // load the model
    const int64_t t_start_us = ggml_time_us();
    std::tie(model, bctx) = bark_init_from_params(params);
    t_load_us = ggml_time_us() - t_start_us;

    printf("\n");

    bark_seed_rng(bctx, params.seed);

    const int64_t t_eval_us_start = ggml_time_us();
    bark_generate_audio(bctx, params.prompt.data(), params.dest_wav_path.c_str(), params.n_threads);
    t_eval_us = ggml_time_us() - t_eval_us_start;

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n\n");
        printf("%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
        printf("%s:     eval time = %8.2f ms\n", __func__, t_eval_us/1000.0f);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    bark_free(bctx);

    return 0;
}
