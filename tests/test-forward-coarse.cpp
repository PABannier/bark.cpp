#include <cstdio>
#include <string>
#include <random>
#include <vector>

#include "bark.h"
#include "common.h"

static const std::vector<std::string> test_data = {
    "./data/coarse/test_pass_coarse_1.bin",   // prompt: The amount of random conversations that lead to culture-shifting ideas is insane.
    "./data/coarse/test_pass_coarse_2.bin",   // prompt: Des Teufels liebstes Möbelstück ist die lange Bank
    "./data/coarse/test_pass_coarse_3.bin",   // prompt: खुदा ने बहुत सी अच्छी चीज बनाई है उस में एक हमारा दिमाग भी है बस उसे Use करने के लिए बता देता तो हम भी करोड़पति बन जाते I
};

static const int n_threads = 4;
static const int sliding_window_size = 60;
static const int max_coarse_history  = 630;
static const float temp = 0.0f;

int main() {
    const std::string fname = "../ggml_weights/ggml_weights_coarse.bin";

    std::mt19937 rng(0);

    bark_model model;

    if(!gpt_model_load(fname, model.coarse_model)) {
        fprintf(stderr, "%s: invalid model file '%s'\n", __func__, fname.c_str());
        return 1;
    }

    bark_context * ctx = bark_new_context_with_model(&model);
    ctx->rng = rng;

    bark_sequence input;
    bark_codes gt_tokens;

    for (int i = 0; i < (int) test_data.size(); i++) {
        input.clear();
        gt_tokens.clear();

        std::string path = test_data[i];
        load_test_data(path, input, gt_tokens);
        ctx->semantic_tokens = input;

        bark_forward_coarse_encoder(ctx, max_coarse_history, sliding_window_size, temp, n_threads);

        printf("\n");
        printf("%s: %s\n", __func__, path.c_str());
        if (!run_test(transpose(gt_tokens), ctx->coarse_tokens)) {
            printf("%s:     test %d failed.\n", __func__, i+1);
        } else {
            printf("%s:     test %d passed.\n", __func__, i+1);
        }
    }

    bark_free(ctx);

    return 0;
}
