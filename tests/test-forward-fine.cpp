#include <cstdio>
#include <string>
#include <random>
#include <vector>

#include "bark.h"
#include "common.h"

static const std::vector<std::string> test_data = {
    "./data/fine/test_pass_fine_1.bin",   // prompt:
    "./data/fine/test_pass_fine_2.bin",   // prompt:
    "./data/fine/test_pass_fine_3.bin",   // prompt:
};

static const int n_threads = 4;
static const float temp = 0.0f;

int main() {
    const std::string fname = "../ggml_weights/ggml_weights_fine.bin";

    std::mt19937 rng(0);

    gpt_model model;
    if(!gpt_model_load(fname, model)) {
        fprintf(stderr, "%s: invalid model file '%s'\n", __func__, fname.c_str());
        return 1;
    }

    bark_codes input, gt_tokens, tokens;

    for (int i = 0; i < (int) test_data.size(); i++) {
        input.clear();
        gt_tokens.clear();
        tokens.clear();

        std::string path = test_data[i];
        load_test_data(path, input, gt_tokens);

        // TODO: need to remove transpose
        bark_codes input_t = transpose(input);
        bark_codes tokens  = transpose(bark_forward_fine_encoder(input_t, model, rng, n_threads, temp));

        printf("\n");
        printf("%s: %s\n", __func__, path.c_str());
        if (!run_test(gt_tokens, tokens)) {
            printf("%s:     test %d failed.\n", __func__, i+1);
        } else {
            printf("%s:     test %d passed.\n", __func__, i+1);
        }
    }

    return 0;
}
