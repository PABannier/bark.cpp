/* These tests test the forward pass of the GPT models as defined in the `gpt_eval`
function. The purpose of these tests is to isolate the `gpt_eval` function to test
whether it outputs the correct logits from a pre-defined input: the (padded) sequence
of tokens.
*/
#include <cstdio>
#include <string>
#include <map>
#include <random>
#include <vector>

#include "bark.h"
#include "common.h"

static const std::vector<std::string> test_data = {
    "./data/fine/test_fine_1.bin",
    "./data/fine/test_fine_2.bin",
    "./data/debug_test.bin",
    "./data/fine/long_prompt_nn_2.bin"
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

    std::vector<std::vector<float>> logits;

    printf("%s: reading bark fine model\n", __func__);
    if(!gpt_model_load(fname, model)) {
        fprintf(stderr, "%s: invalid model file '%s'\n", __func__, fname.c_str());
        return 1;
    }

    bool success = true;

    // dry run to estimate mem_per_token
    fine_gpt_eval(model, n_threads, 2, { {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 8} }, logits, mem_per_token);

    for (int i = 0; i < (int) test_data.size(); i++) {
        bark_codes input;
        std::vector<std::vector<float>> truth;

        std::string path = test_data[i];

        load_nested_test_data(path, input, truth);
        bark_codes input_t = transpose(input);
        fine_gpt_eval(model, n_threads, 2, input_t, logits, mem_per_token);

        // fprintf(stderr, "truth : ");
        // for (int i = 0; i < 8; i++) { fprintf(stderr, "%.4f ", truth[0][i]); }
        // fprintf(stderr, "\n");
        // fprintf(stderr, "logits: ");
        // for (int i = 0; i < 8; i++) { fprintf(stderr, "%.4f ", logits[0][i]); }
        // fprintf(stderr, "\n");

        fprintf(stderr, "%s", path.c_str());
        if (!run_test_on_codes(truth, logits)) {
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
