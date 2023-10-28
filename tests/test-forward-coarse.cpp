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

const int n_threads = 4;
const int sliding_window_size = 60;
const int max_coarse_history  = 630;
const float temp = 0.0f;

int main() {
    const std::string fname = "../ggml_weights/";

    std::mt19937 rng(0);

    struct bark_context * bctx = bark_load_model(fname);
    if (!bctx) {
        fprintf(stderr, "%s: Could not load model\n", __func__);
        exit(1);
    }

    bark_sequence input;
    bark_codes gt_tokens;

    for (int i = 0; i < (int) test_data.size(); i++) {
        input.clear();
        gt_tokens.clear();

        std::string path = test_data[i];
        load_test_data(path, input, gt_tokens);
        bctx->semantic_tokens = input;

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
