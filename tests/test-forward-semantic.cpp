/* This test checks that the forward pass as defined in `bark_forward_semantic_encoder`
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
    "./data/semantic/test_pass_semantic_1.bin",
    "./data/semantic/test_pass_semantic_2.bin",
};

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model-file>\n", argv[0]);
        return 1;
    }

    const std::string fname = argv[1];

    gpt_model model;
    std::mt19937 rng(0);
    const int n_threads = 4;
    const float min_eos_p = 0.2;

    bool success = true;

    printf("%s: reading bark text model\n", __func__);
    if(!gpt_model_load(fname, model)) {
        fprintf(stderr, "%s: invalid model file '%s'\n", __func__, fname.c_str());
        return 1;
    }

    for (int i = 0; i < (int) test_data.size(); i++) {
        std::vector<int32_t> input, truth;
        std::string path = test_data[i];

        load_test_data(path, input, truth);
        bark_sequence output = bark_forward_text_encoder(input, model, rng, n_threads, 0.0f, min_eos_p);

        fprintf(stderr, "%s", path.c_str());
        if (!run_test_on_sequence(truth, output)) {
            success = false;
            fprintf(stderr, "   TEST %d FAILED.\n", i+1);
        } else {
            fprintf(stderr, "   TEST %d PASSED.\n", i+1);
        }
    }

    if (success)
        fprintf(stderr, "%s : tests passed successfully.\n", __func__);

    return 0;
}