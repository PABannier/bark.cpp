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

static const std::vector<test_data_t> & k_tests()
{
    static std::vector<test_data_t> _k_tests;

    // test 1: hello world
    {
        std::vector<int> input;
        logit_sequence logits_merge, logits_no_merge;

        load_test_data("./data/semantic/test1_merge.bin", input, logits_merge);
        load_test_data("./data/semantic/test1_no_merge.bin", input, logits_no_merge);
        _k_tests.push_back({input, logits_merge, logits_no_merge});
    }

    // test 2: How far that little candle throws its beams! So shines a good deed in a naughty world.
    {
        std::vector<int> input;
        logit_sequence logits_merge, logits_no_merge;

        load_test_data("./data/semantic/test2_merge.bin", input, logits_merge);
        load_test_data("./data/semantic/test2_no_merge.bin", input, logits_no_merge);
        _k_tests.push_back({input, logits_merge, logits_no_merge});
    }

    return _k_tests;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model-file>\n", argv[0]);
        return 1;
    }

    const std::string fname = argv[1];

    gpt_model model;
    const int n_threads = 4;

    const int n_past = 0;

    size_t mem_per_token = 0;

    logit_sequence logits;

    printf("%s: reading bark text model\n", __func__);
    if(!gpt_model_load(fname, model)) {
        fprintf(stderr, "%s: invalid model file '%s'\n", __func__, fname.c_str());
        return 1;
    }

    // dry run to estimate mem_per_token
    gpt_eval(model, n_threads, 0, false, { 0, 1, 2, 3 }, logits, mem_per_token);

    for (const auto & test_data : k_tests()) {
        bark_sequence input = std::get<0>(test_data);

        // merge_ctx = True
        gpt_eval(model, n_threads, n_past, true, input, logits, mem_per_token);
        if (!run_test_on_sequence(std::get<1>(test_data), logits, true)) {
            return 3;
        }

        logits.clear();

        // merge_ctx = False
        gpt_eval(model, n_threads, n_past, false, input, logits, mem_per_token);
        if (!run_test_on_sequence(std::get<2>(test_data), logits, false)) {
            return 3;
        }
    }

    fprintf(stderr, "%s : tests passed successfully.\n", __func__);

    return 0;
}
