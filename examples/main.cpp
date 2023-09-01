#include "ggml.h"
#include "bark.h"


int main(int argc, char **argv) {
    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();

    bark_params params;

    if (bark_params_parse(argc, argv, params) == false) {
        return 1;
    }

    int64_t t_load_us = 0;
    int64_t t_eval_us = 0;

    bark_model model;
    std::string fname = "./ggml_weights";

    if (!params.model.empty()) {
        fname = params.model;
    }

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if(!bark_model_load(fname, model)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, fname.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }

    // create a context
    bark_context * ctx = bark_new_context_with_model(&model);
    if (ctx == nullptr) {
        fprintf(stderr, "%s: failed to create context\n", __func__);
        return 1;
    }

    printf("\n");

    std::string prompt = "this is an audio";
    if (!params.prompt.empty()) {
        prompt = params.prompt;
    }

    const int64_t t_eval_us_start = ggml_time_us();
    bark_generate_audio(ctx, prompt.data(), params.dest_wav_path, params.n_threads);
    t_eval_us = ggml_time_us() - t_eval_us_start;

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n\n");
        printf("%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
        printf("%s:     eval time = %8.2f ms\n", __func__, t_eval_us/1000.0f);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    bark_free(ctx);

    return 0;
}
