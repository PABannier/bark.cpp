#include <cstdio>
#include <string>
#include <random>
#include <vector>

#include "bark.h"
#include "common.h"

static const std::vector<std::string> test_data = {
    "./data/semantic/test_pass_semantic_1.bin",   // prompt: Ceci est un texte en français pour tester le bon fonctionnement de bark.
    "./data/semantic/test_pass_semantic_2.bin",   // prompt: Sometimes the heart sees what is invisible to the eye
    "./data/semantic/test_pass_semantic_3.bin",   // prompt: El Arte de Vencer se Aprende en las Derrotas
};

static const int   n_threads = 4;
static const float min_eos_p = 0.2;
static const float temp      = 0.0f;  // deterministic sampling

int main() {
    const std::string fname = "../ggml_weights/ggml_weights_text.bin";

    std::mt19937 rng(0);

    gpt_model model;
    if(!gpt_model_load(fname, model)) {
        fprintf(stderr, "%s: invalid model file '%s'\n", __func__, fname.c_str());
        return 1;
    }

    bark_sequence input;
    bark_sequence gt_tokens;

    for (int i = 0; i < (int) test_data.size(); i++) {
        input.clear();
        gt_tokens.clear();

        std::string path = test_data[i];
        load_test_data(path, input, gt_tokens);

        bark_sequence tokens = bark_forward_text_encoder(
            input, model, rng, n_threads, temp, min_eos_p);

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
