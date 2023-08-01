#include "ggml.h"
#include "bark.h"

int main() {
    const int64_t t_main_start_us = ggml_time_us();

    int64_t t_load_us = 0;
    int64_t t_eval_us = 0;

    bark_model model;
    std::string fname = "./ggml_weights";

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if(!bark_model_load(fname, model)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, fname.c_str());
            return 1;
        }

        t_load_us = ggml_time_us() - t_start_us;
    }

    printf("\n");

    // forward pass
    const std::string prompt = "this is an audio";
    {
        const int64_t t_eval_us_start = ggml_time_us();

        // call to generate audio
        bark_generate_audio(model, model.vocab, prompt.data(), 4);

        t_eval_us = ggml_time_us() - t_eval_us_start;
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n\n");
        printf("%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
        printf("%s:     eval time = %8.2f ms\n", __func__, t_eval_us/1000.0f);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
    }

    // TODO: write wrapper
    ggml_free(model.coarse_model.ctx);
    ggml_free(model.fine_model.ctx);
    ggml_free(model.text_model.ctx);
    ggml_free(model.codec_model.ctx);

    return 0;
}
