#include <cstdio>
#include <string>
#include <random>
#include <vector>

#include "bark.h"
#include "common.h"

const std::vector<std::string> test_data = {
    "../tests/data/semantic/test_pass_semantic_1.bin",   // prompt: Ceci est un texte en français pour tester le bon fonctionnement de bark.
    "../tests/data/semantic/test_pass_semantic_2.bin",   // prompt: Sometimes the heart sees what is invisible to the eye
    "../tests/data/semantic/test_pass_semantic_3.bin",   // prompt: El Arte de Vencer se Aprende en las Derrotas
};

const int   n_threads = 4;
const float min_eos_p = 0.2;
const float temp      = 0.0f;  // deterministic sampling

int main() {
    const std::string dirname = "../ggml_weights/";

    bark_sequence input, gt_tokens;

    std::mt19937 rng(0);

    // initialize bark context
    struct bark_context * bctx = bark_load_model(dirname);
    if (!bctx) {
        fprintf(stderr, "%s: Could not load model\n", __func__);
        exit(1);
    }
    bctx->rng = rng;

    for (int i = 0; i < (int) test_data.size(); i++) {
        input.clear();
        gt_tokens.clear();

        std::string path = test_data[i];
        load_test_data(path, input, gt_tokens);
        bctx->tokens = input;

        if (!bark_forward_text_encoder(bctx, n_threads)) {
            fprintf(stderr, "%s: failed to forward text encoder\n", __func__);
            exit(1);
        }

        printf("\n");
        printf("%s: %s\n", __func__, path.c_str());
        if (!run_test(gt_tokens, bctx->semantic_tokens)) {
            printf("%s:     test %d failed.\n", __func__, i+1);
        } else {
            printf("%s:     test %d passed.\n", __func__, i+1);
        }
    }

    bark_free(bctx);

    return 0;
}
