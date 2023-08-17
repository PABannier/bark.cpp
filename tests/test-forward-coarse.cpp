/* This test checks that the forward pass as defined in `bark_forward_coarse_encoder`
yields the same output as the original Bark implementation when using a deterministic
sampling: the argmax sampling.
Note that this sampling does not yield good quality audio, and is used solely for testing
purposes to remove the stochasticity from sampling.
*/
#include <cstdio>
#include <string>
#include <random>
#include <vector>

#include "bark.h"
#include "common.h"

static const std::vector<std::string> test_data = {
    "./data/coarse/test_pass_coarse_1.bin",   // prompt:
    "./data/coarse/test_pass_coarse_2.bin",   // prompt:
    "./data/coarse/test_pass_coarse_3.bin",   // prompt:
};

static const int n_threads = 4;
static const int sliding_window_size = 60;
static const int max_coarse_history  = 630;
static const float temp = 0.0f;

int main() {
    const std::string fname = "../ggml_weights/ggml_weights_fine.bin";

    std::mt19937 rng(0);

    gpt_model model;
    if(!gpt_model_load(fname, model)) {
        fprintf(stderr, "%s: invalid model file '%s'\n", __func__, fname.c_str());
        return 1;
    }

    bark_sequence input;
    bark_codes gt_tokens;

    for (int i = 0; i < (int) test_data.size(); i++) {
        input.clear();
        gt_tokens.clear();

        std::string path = test_data[i];
        load_test_data(path, input, gt_tokens);

        bark_codes tokens = bark_forward_coarse_encoder(
            input, model, rng, n_threads, temp, max_coarse_history, sliding_window_size);

        printf("\n");
        printf("%s: %s\n", __func__, path.c_str());
        if (!run_test(transpose(gt_tokens), tokens)) {
            printf("%s:     test %d failed.\n", __func__, i+1);
        } else {
            printf("%s:     test %d passed.\n", __func__, i+1);
        }
    }

    return 0;
}
