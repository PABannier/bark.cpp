/* These tests test the forward pass of the GPT models as defined in the `gpt_eval`
function. The purpose of these tests is to isolate the `gpt_eval` function to test
whether it outputs the correct logits from a pre-defined input: the (padded) sequence
of tokens.

Note that the input tokens (seq1, seq2 and seq3) have already been offset by
`TEXT_ENCODING_OFFSET`.

Only the first 50 logits and last 50 logits are tested due to the size of the original
logit vector (10,048).

This file tests two configurations: merge_ctx is True and merge_ctx is False.
*/
#include <cstdio>
#include <string>
#include <vector>

#include "bark.h"
#include "common.cpp"

static const std::vector<test_data_t> & k_tests()
{
    static std::vector<test_data_t> _k_tests;

    // test 1: hello world
    {
        static const bark_sequence input = { 71742,  20181,  21404 };

        // merge_ctx = True
        logit_sequence gt_merge_ctx;
        {
            logit_sequence first = {
                -6.7850,  -8.4204,  -6.9174,  -6.6163,  -6.1079,  -6.7111,  -7.5943,
                -7.2360,  -6.2813,  -9.2174,  -1.3561, -10.1586,  -7.1028,  -8.2125,
                -8.1532,  -9.2955, -10.7875,  -6.4029,  -7.9612, -10.8515,  -7.0013,
                -9.5818,  -4.9015,  -7.1582,  -8.2076,  -9.3205,  -5.9254,  -4.6250,
                -4.9591,  -9.3749,  -9.0533,  -7.3062,  -8.0011,  -6.8268,  -6.7611,
                -7.5841,  -6.5029,  -6.9205,  -7.7602,  -9.2439,  -7.6442,  -1.7321,
                -7.7471,  -6.9885,  -6.8584,  -7.8602,  -7.9086,  -4.9766,  -4.7366,
                -9.5923
            };

            logit_sequence last = {
                -10.7830,  -9.9664, -10.7215, -19.7928, -19.7928, -19.7926, -19.7927,
                -19.7930, -19.7931, -19.7930, -19.7928, -19.7929, -19.7923, -19.7931,
                -19.7927, -19.7929, -19.7931, -19.7932, -19.7926, -19.7929, -19.7934,
                -19.7929, -19.7928, -19.7928, -19.7930, -19.7928, -19.7929, -19.7932,
                -19.7930, -19.7925, -19.7929, -19.7932, -19.7928, -19.7930, -19.7929,
                -19.7928, -19.7931, -19.7929, -19.7929, -19.7928, -19.7930, -19.7928,
                -19.7925, -19.7934, -19.7925, -19.7934, -19.7929, -19.7927, -19.7930,
                -19.7930
            };

            first.insert(first.end(), last.begin(), last.end());
            gt_merge_ctx = first;
        }

        // merge_ctx = False
        logit_sequence gt_no_merge;
        {
            logit_sequence first = {
                -4.6237,  -8.9065, -10.8395,  -7.7626,  -6.6053,  -5.6599,  -5.1323,
                -9.4369,  -7.5160, -10.3806,  -0.7840,  -7.9517,  -7.3649,  -5.5768,
                -10.0623,  -8.8393, -16.6438,  -7.1485,  -5.9515,  -8.9547,  -6.9725,
                -9.3813,  -6.6845,  -6.4903,  -7.8976,  -8.8185,  -7.5417,  -5.2861,
                -2.9462, -10.3543,  -8.0235,  -7.2701,  -8.1734,  -7.8364,  -6.2430,
                -8.0917,  -8.1107,  -7.7524,  -5.5639,  -9.0928,  -7.1926,  -2.2350,
                -6.2111,  -6.0619,  -7.4119,  -6.3649,  -8.1705,  -8.7320,  -5.4250,
                -9.9603
            };

            logit_sequence last = {
                -7.7061,  -8.0450,  11.7950, -20.8908, -20.8914, -20.8913, -20.8909,
                -20.8915, -20.8915, -20.8911, -20.8909, -20.8910, -20.8911, -20.8916,
                -20.8912, -20.8918, -20.8911, -20.8916, -20.8914, -20.8915, -20.8915,
                -20.8912, -20.8913, -20.8911, -20.8911, -20.8916, -20.8912, -20.8913,
                -20.8916, -20.8910, -20.8913, -20.8910, -20.8912, -20.8909, -20.8916,
                -20.8908, -20.8912, -20.8916, -20.8914, -20.8911, -20.8914, -20.8909,
                -20.8914, -20.8912, -20.8914, -20.8915, -20.8909, -20.8912, -20.8913,
                -20.8917
            };

            first.insert(first.end(), last.begin(), last.end());
            gt_no_merge = first;
        }

        _k_tests.push_back({input, gt_merge_ctx, gt_no_merge});
    }

    // test 2: How far that little candle throws its beams! So shines a good deed in a naughty world.
    {
        static const bark_sequence input = {
            25010,  23349,  20237,  26793,  95425,  20332,  83744,  20155,  20522,
            77246,  20155,  10154,  22930,  67715,  21544,  10217,  25246,  47294,
            20154,  10217,  70090,  30735,  20205,  21404,  10167, };

        // merge_ctx = True
        logit_sequence gt_merge_ctx;
        {
            logit_sequence first = {
                -5.3793, -6.1819, -5.7294, -3.9748, -3.2352, -5.0669, -4.7945, -5.9294,
                -5.0572, -6.8415,  1.0149, -8.1913, -5.5375, -5.6902, -7.2809, -6.2684,
                -8.8602, -4.6138, -5.8980, -8.7850, -4.3053, -8.6675, -4.0874, -5.7161,
                -5.6619, -6.8635, -3.4520, -2.5501, -2.1214, -8.1724, -6.8350, -5.2371,
                -6.4175, -5.0723, -3.6968, -4.4802, -0.4470, -5.6713, -5.4908, -5.7577,
                -5.0717, -1.5586, -5.2244, -4.3490, -4.7702, -5.9671, -6.5616, -4.2143,
                -2.4564, -7.1832
            };

            logit_sequence last = {
                -8.2814,  -8.3309,  -7.9492, -14.1677, -14.1675, -14.1675, -14.1675,
                -14.1679, -14.1677, -14.1678, -14.1677, -14.1678, -14.1674, -14.1681,
                -14.1676, -14.1677, -14.1677, -14.1679, -14.1674, -14.1675, -14.1682,
                -14.1678, -14.1676, -14.1679, -14.1679, -14.1679, -14.1677, -14.1678,
                -14.1677, -14.1677, -14.1675, -14.1679, -14.1675, -14.1675, -14.1676,
                -14.1675, -14.1675, -14.1677, -14.1675, -14.1677, -14.1680, -14.1676,
                -14.1678, -14.1682, -14.1677, -14.1679, -14.1678, -14.1676, -14.1678,
                -14.1677
            };

            first.insert(first.end(), last.begin(), last.end());
            gt_merge_ctx = first;
        }

        // merge_ctx = False
        logit_sequence gt_no_merge;
        {
            logit_sequence first = {
                 0.0533, -3.3788, -6.4929, -2.3461,  0.1204, -1.2127, -2.2630, -1.5615,
                -0.1647, -6.9080,  2.4658, -2.0545, -1.4231, -1.8261, -3.3888, -1.7102,
                -4.0333, -0.1014, -3.3560, -0.8281, -0.5758, -1.5550, -2.8358, -0.3463,
                -0.4991, -1.7289, -0.7232,  2.2476,  0.5060, -5.8686, -1.8501, -1.5646,
                -2.2055, -0.0267, -1.8045, -1.0386,  0.2744, -1.8290, -2.3269, -2.3316,
                 0.0888,  4.7798, -0.8191,  1.7266, -0.8885, -2.4267, -2.7426, -3.3305,
                -0.4763, -3.9719
            };

            logit_sequence last = {
                -3.3547, -1.1211, -7.4785, -8.6918, -8.6915, -8.6911, -8.6913, -8.6916,
                -8.6914, -8.6912, -8.6919, -8.6912, -8.6908, -8.6916, -8.6910, -8.6916,
                -8.6915, -8.6921, -8.6910, -8.6913, -8.6914, -8.6914, -8.6908, -8.6911,
                -8.6916, -8.6914, -8.6918, -8.6914, -8.6917, -8.6912, -8.6914, -8.6915,
                -8.6916, -8.6916, -8.6913, -8.6913, -8.6913, -8.6915, -8.6913, -8.6912,
                -8.6919, -8.6911, -8.6915, -8.6911, -8.6914, -8.6914, -8.6913, -8.6915,
                -8.6922, -8.6915
            };

            first.insert(first.end(), last.begin(), last.end());
            gt_no_merge = first;
        }

        _k_tests.push_back({ input, gt_merge_ctx, gt_no_merge });
    }

    return _k_tests;
};

bark_sequence pad_input(const bark_sequence & input) {
    int original_sz = input.size();
    bark_sequence pad(input.begin(), input.end());
    pad.resize(513);

    for (int i = original_sz; i < 256; i++)
        pad[i] = TEXT_PAD_TOKEN;
    for (int i = 256; i < 512; i++)
        pad[i] = SEMANTIC_PAD_TOKEN;
    pad[512] = SEMANTIC_INFER_TOKEN;

    return pad;
}

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
        bark_sequence input = pad_input(std::get<0>(test_data));

        // merge_ctx = True
        gpt_eval(model, n_threads, n_past, true, input, logits, mem_per_token);
        if (!run_test(std::get<1>(test_data), logits, true)) {
            return 3;
        }

        logits.clear();

        // merge_ctx = False
        gpt_eval(model, n_threads, n_past, false, input, logits, mem_per_token);
        if (!run_test(std::get<2>(test_data), logits, false)) {
            return 3;
        }

    }

    fprintf(stderr, "%s : tests passed successfully.\n", __func__);

    return 0;
}
