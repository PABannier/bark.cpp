/* These tests test the forward pass of the GPT models as defined in the `gpt_eval`
function. The purpose of these tests is to isolate the `gpt_eval` function to test
whether it outputs the correct logits from a pre-defined input: the (padded) sequence
of tokens.
*/
#include <cstdio>
#include <string>
#include <tuple>
#include <random>
#include <vector>

#include "bark.h"
#include "common.h"

static const std::vector<std::string> test_data = {
    "./data/coarse/test_coarse_1.bin",
    "./data/coarse/test_coarse_2.bin",
};

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model-file>\n", argv[0]);
        return 1;
    }

    const std::string fname = argv[1];

    gpt_model model;
    const int n_threads = 4;

    size_t mem_per_token = 0;

    logit_sequence logits;

    printf("%s: reading bark coarse model\n", __func__);
    if(!gpt_model_load(fname, model)) {
        fprintf(stderr, "%s: invalid model file '%s'\n", __func__, fname.c_str());
        return 1;
    }

    bool success = true;

    // dry run to estimate mem_per_token
    gpt_eval(model, n_threads, 0, false, { 0, 1, 2, 3 }, logits, mem_per_token);

    for (int i = 0; i < (int) test_data.size(); i++) {
        bark_sequence input;
        logit_sequence truth;
        std::string path = test_data[i];

        load_test_data(path, input, truth);
        gpt_eval(model, n_threads, 0, false, input, logits, mem_per_token);

        fprintf(stderr, "%s", path.c_str());
        if (!run_test_on_sequence(truth, logits)) {
            success = false;
            fprintf(stderr, "   TEST %d FAILED.\n", i+1);
        } else {
            fprintf(stderr, "   TEST %d PASSED.\n", i+1);
        }

        logits.clear(); 
    }

    if (success)
        fprintf(stderr, "%s : tests passed successfully.\n", __func__);

    return 0;
}
