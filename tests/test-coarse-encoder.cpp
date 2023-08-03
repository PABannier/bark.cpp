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

static const std::vector<std::tuple<bark_sequence, logit_sequence>> & k_tests()
{
    static std::vector<std::tuple<bark_sequence, logit_sequence>> _k_tests;

    // test 1: hello world
    {
        std::vector<int> input;
        logit_sequence logits;

        load_test_data("./data/coarse/test1.bin", input, logits);
        _k_tests.push_back({input, logits});
    }

    // test 2: this is an audio
    {
        std::vector<int> input;
        logit_sequence logits;

        load_test_data("./data/coarse/test2.bin", input, logits);
        _k_tests.push_back({input, logits});
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

    size_t mem_per_token = 0;

    logit_sequence logits;

    printf("%s: reading bark coarse model\n", __func__);
    if(!gpt_model_load(fname, model)) {
        fprintf(stderr, "%s: invalid model file '%s'\n", __func__, fname.c_str());
        return 1;
    }

    // dry run to estimate mem_per_token
    gpt_eval(model, n_threads, 0, false, { 0, 1, 2, 3 }, logits, mem_per_token);

    for (const auto & test_data : k_tests()) {
        bark_sequence input = std::get<0>(test_data);

        gpt_eval(model, n_threads, 0, false, input, logits, mem_per_token);

        if (!run_test_on_sequence(std::get<1>(test_data), logits, false)) {
            return 3;
        }
    }

    fprintf(stderr, "%s : tests passed successfully.\n", __func__);

    return 0;
}
