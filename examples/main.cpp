#include "ggml.h"
#include "bark.h"

#include <tuple>

std::tuple<struct bark_model *, struct bark_context *> bark_init_from_params(bark_params & params) {
    bark_model * model = bark_load_model_from_file(params.model_path);
    if (model == NULL) {
        fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model_path);
        return std::make_tuple(nullptr, nullptr);
    }

    bark_context * bctx = bark_new_context_with_model(model);
    if (bctx == NULL) {
        fprintf(stderr, "%s: error: failed to create context with model '%s'\n", __func__, params.model_path);
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

    std::string fname = "./ggml_weights";
    if (params.model_path) {
        fname = std::string(params.model_path);
    }

    std::string prompt = "this is an audio";
    if (params.prompt) {
        prompt = params.prompt;
    }

    std::string out_path = "./ggml_out.wav";
    if (params.dest_wav_path) {
        out_path = std::string(params.dest_wav_path);
    }

    // load the model
    const int64_t t_start_us = ggml_time_us();
    std::tie(model, bctx) = bark_init_from_params(params);
    t_load_us = ggml_time_us() - t_start_us;

    printf("\n");

    const int64_t t_eval_us_start = ggml_time_us();
    bark_generate_audio(bctx, prompt.data(), out_path.c_str(), params.n_threads);
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
