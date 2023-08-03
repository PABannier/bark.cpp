/* These tests test the forward pass of the GPT models as defined in the `gpt_eval`
function. The purpose of these tests is to isolate the `gpt_eval` function to test
whether it outputs the correct logits from a pre-defined input: the (padded) sequence
of tokens.

Only the first and last 50 logits are tested.
*/
#include <cstdio>
#include <string>
#include <map>
#include <random>
#include <vector>

#include "bark.h"
#include "common.h"

static const std::vector<std::tuple<bark_codes, std::vector<logit_sequence>>> & k_tests()
{
    static std::vector<std::tuple<bark_codes, std::vector<logit_sequence>>> _k_tests;

    // test 1: hello world
    {
        bark_codes input = {};

        logit_sequence first = {
            -0.2504, -0.8357, -0.7817, -3.4325, -1.3497, -0.0246, -3.9537,  1.0414,
            -4.8329, -1.3045, -1.6172, -2.0307, -3.4241, -0.3468, -1.1227, -4.1510,
            -0.9693, -1.2971, -2.4425, -2.2676, -2.4385, -2.1456, -1.0770,  4.3147,
            -0.3306,  7.6165, -2.2822,  0.0097, -3.2219, -0.9281, -0.0448, -1.8040,
            -0.3715, -0.9066, -1.1935,  1.0319, -2.2509, -3.0084,  3.5436, -2.4745,
            1.0158, -0.8026, -0.9665, -2.9495, -0.0318, -1.0646, -0.8192,  2.3201,
            -3.5992, -6.0473
        };

        logit_sequence last = {
            -0.0447,  -2.0397,   4.3142,   0.0667,  -2.9984,   2.9137,  -2.0471,
            -1.0145,   2.3341,  -0.6549,  -1.6525,   1.1172,  -0.3786,  -2.6145,
            -0.7574,  -0.4769,  -1.5991,   0.9172,  -1.3099,   0.4555,  -1.1762,
            -1.6306,  -1.9082,  -4.4277,  -3.5168,   4.8780,  -1.2237,   2.2364,
            -1.9533,  -1.4511,   1.3850,  -7.2282,  -2.1588,  -0.6294,   2.6044,
            -4.0282,   2.3695,   3.3589,  -4.5038,  -4.8646,   1.2393, -22.4992,
            -2.1164,   6.5350,  -1.5946,   6.6539,  -2.5565,  -0.2885,   3.6583,
            -1.7904
        };

        first.insert(first.end(), last.begin(), last.end());
        _k_tests.push_back({input, first});
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

    std::vector<logit_sequence> logits;

    printf("%s: reading bark fine model\n", __func__);
    if(!gpt_model_load(fname, model)) {
        fprintf(stderr, "%s: invalid model file '%s'\n", __func__, fname.c_str());
        return 1;
    }

    // dry run to estimate mem_per_token
    fine_gpt_eval(model, n_threads, 2, { {0, 1}, {1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6}, {6, 7}, {7, 8} }, logits, mem_per_token);

    for (const auto & test_data: k_tests()) {
        bark_codes input = std::get<0>(test_data);

        fine_gpt_eval(model, n_threads, 2, input, logits, mem_per_token);
        if (!run_test_on_codes(std::get<1>(test_data), logits, true)) {
            return 3;
        }

    }

    fprintf(stderr, "%s : tests passed successfully.\n", __func__);

    return 0;
}
