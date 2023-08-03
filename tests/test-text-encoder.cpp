/* These tests test the forward pass of the GPT models as defined in the `gpt_eval`
function. The purpose of these tests is to isolate the `gpt_eval` function to test
whether it outputs the correct logits from a pre-defined input: the (padded) sequence
of tokens.

This file tests two configurations: merge_ctx is True and merge_ctx is False.
*/
#include <cstdio>
#include <string>
#include <vector>

#include "bark.h"
#include "common.h"

static const std::vector<std::tuple<std::string, bool>> test_data = {
    { "./data/semantic/test_semantic_merge_1.bin"   , true  },
    { "./data/semantic/test_semantic_no_merge_1.bin", false },
    { "./data/semantic/test_semantic_merge_2.bin"   , true  },
    { "./data/semantic/test_semantic_no_merge_2.bin", false },
};

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model-file>\n", argv[0]);
        return 1;
    }

    const std::string fname = argv[1];

    gpt_model model;
    const int n_threads = 4;

    bool success = true;

    size_t mem_per_token = 0;
    logit_sequence logits;

    printf("%s: reading bark text model\n", __func__);
    if(!gpt_model_load(fname, model)) {
        fprintf(stderr, "%s: invalid model file '%s'\n", __func__, fname.c_str());
        return 1;
    }

    // dry run to estimate mem_per_token
    {
        int n_past = 0;
        gpt_eval(model, n_threads, &n_past, false, { 0, 1, 2, 3 }, logits, mem_per_token);
    }

    for (int i = 0; i < (int) test_data.size(); i++) {
        bark_sequence input;
        logit_sequence truth;

        std::string path = std::get<0>(test_data[i]);
        bool merge_ctx = std::get<1>(test_data[i]);

        int n_past = 0;

        load_test_data(path, input, truth);
        gpt_eval(model, n_threads, &n_past, merge_ctx, input, logits, mem_per_token);

        fprintf(stderr, "%s (merge context=%d)", path.c_str(), merge_ctx);
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
