#include <iostream>
#include <tuple>

#include "bark.h"
#include "common.h"
#include "ggml.h"

void bark_print_progress_callback(struct bark_context *bctx, enum bark_encoding_step step, int progress, void *user_data) {
    if (step == bark_encoding_step::SEMANTIC) {
        printf("\rGenerating semantic tokens... %d%%", progress);
    } else if (step == bark_encoding_step::COARSE) {
        printf("\rGenerating coarse tokens... %d%%", progress);
    } else if (step == bark_encoding_step::FINE) {
        printf("\rGenerating fine tokens... %d%%", progress);
    }
    fflush(stdout);
}

int main(int argc, char **argv) {
    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();

    bark_params params;
    bark_verbosity_level verbosity = bark_verbosity_level::LOW;

    if (bark_params_parse(argc, argv, params) > 0) {
        fprintf(stderr, "%s: Could not parse arguments\n", __func__);
        return 1;
    }

    std::cout << R"( _                   _                           )" << "\n"
              << R"(| |                 | |                          )" << "\n"
              << R"(| |__    __ _  _ __ | | __     ___  _ __   _ __  )" << "\n"
              << R"(| '_ \  / _` || '__|| |/ /    / __|| '_ \ | '_ \ )" << "\n"
              << R"(| |_) || (_| || |   |   <  _ | (__ | |_) || |_) |)" << "\n"
              << R"(|_.__/  \__,_||_|   |_|\_\(_) \___|| .__/ | .__/ )" << "\n"
              << R"(                                   | |    | |    )" << "\n"
              << R"(                                   |_|    |_|    )" << "\n";

    // initialize bark context
    struct bark_context_params ctx_params = bark_context_default_params();

    ctx_params.verbosity = verbosity;
    ctx_params.progress_callback = bark_print_progress_callback;
    ctx_params.progress_callback_user_data = nullptr;

    struct bark_context *bctx = bark_load_model(params.model_path.c_str(), ctx_params, params.seed);
    if (!bctx) {
        fprintf(stderr, "%s: Could not load model\n", __func__);
        exit(1);
    }

    // generate audio
    if (!bark_generate_audio(bctx, params.prompt.c_str(), params.n_threads)) {
        fprintf(stderr, "%s: An error occured. If the problem persists, feel free to open an issue to report it.\n", __func__);
        exit(1);
    }

    const float *audio_data = bark_get_audio_data(bctx);
    if (audio_data == NULL) {
        fprintf(stderr, "%s: Could not get audio data\n", __func__);
        exit(1);
    }

    const int audio_arr_size = bark_get_audio_data_size(bctx);

    std::vector<float> audio_arr(audio_data, audio_data + audio_arr_size);

    write_wav_on_disk(audio_arr, params.dest_wav_path);

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();
        const int64_t t_load_us = bark_get_load_time(bctx);
        const int64_t t_eval_us = bark_get_eval_time(bctx);

        printf("\n\n");
        printf("%s:     load time = %8.2f ms\n", __func__, t_load_us / 1000.0f);
        printf("%s:     eval time = %8.2f ms\n", __func__, t_eval_us / 1000.0f);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us) / 1000.0f);
    }

    bark_free(bctx);

    return 0;
}
