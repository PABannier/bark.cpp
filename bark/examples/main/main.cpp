#include <tuple>

#include "ggml.h"
#include "bark.h"
#include "common.h"


int main(int argc, char **argv) {
    ggml_time_init();
    const int64_t t_main_start_us = ggml_time_us();

    bark_params params;

    if (bark_params_parse(argc, argv, params) > 0) {
        fprintf(stderr, "%s: Could not parse arguments\n", __func__);
        return 1;
    }

    // initialize bark context
    struct bark_context * bctx = bark_load_model(params.model_path);
    if (!bctx) {
        fprintf(stderr, "%s: Could not load model\n", __func__);
        exit(1);
    }

    // generate audio
    if (!bark_generate_audio(bctx, params.prompt, params.dest_wav_path, params.n_threads)) {
        fprintf(stderr, "%s: Could not generate audio\n", __func__);
        exit(1);
    }

    auto & audio_arr = bctx->audio_arr;
    write_wav_on_disk(audio_arr, params.dest_wav_path);

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n\n");
        printf("%s:     load time = %8.2f ms\n", __func__, bctx->t_load_us/1000.0f);
        printf("%s:     eval time = %8.2f ms\n", __func__, bctx->t_eval_us/1000.0f);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    bark_free(bctx);

    return 0;
}
