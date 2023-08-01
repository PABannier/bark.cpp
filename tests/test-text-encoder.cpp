#include "bark.h"

#include <cstdio>
#include <string>
#include <map>
#include <random>
#include <vector>

static const std::map<bark_sequence, bark_sequence> & k_tests()
{
    static const bark_sequence seq1 = { 71742, 20181, 21404 };
    static const bark_sequence seq2 = { 20579, 20172, 20199, 33733 };
    static const bark_sequence seq3 = { 21113, 35307, 10165, 62571, 10165, 23622, 20236, 20959, 52867 };

    static const bark_sequence ans1 = { 215, 215, 215, 2315, 2315, 10, 8924, 10, 5934, 2015, 334, 334, 334, 2015, 278, 278, 3255, 3255, 278, 278, 3255, 278, 278, 278, 278, 9163, 10, 9163, 10, 9163, 10, 4717, 10, 302, 1075, 3490, 2584, 2584, 7196, 7196, 129, 326, 292, 292, 5770, 5770, 624, 624, 2877, 6303, 6303, 6303, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 1667, 8228, 2469, 215, 215, 2825, 5588, 5588, 9628, 3062, 9628, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062, 3062 };
    static const bark_sequence ans2 = { 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 206, 3208, 733, 215, 215, 1988, 4004, 4004, 206, 206, 5934, 206, 2015, 334, 221, 334, 2015, 278, 278, 278, 278, 9425, 206, 206, 206, 239, 206, 206, 239, 206, 206, 239, 206, 206, 239, 206, 206, 239, 206, 206, 239, 206, 206, 239, 206, 206, 239, 206, 206, 2003, 2003, 3208, 302,  57,  57, 5851, 1011, 1011, 321,  41,  41, 1682, 1769, 1769, 9303, 9303, 9303, 2834, 2834, 812,  23, 171, 171, 171, 1367, 1595, 7594, 7594, 2365, 2365, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157, 157 };
    static const bark_sequence ans3 = { 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 215, 41, 215, 1293, 2315, 5607, 5607, 9372, 9372, 222, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221, 221 };

    static std::map<bark_sequence, bark_sequence> _k_tests = {
        { seq1, ans1 },  // hello world
        { seq2, ans2 },  // this is an audio
        { seq3, ans3 },  // You cannot, sir, take from me anything
    };
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
    std::mt19937 rng(0);

    const int   n_threads = 4;
    const float min_eos_p = 0.2f;
    const float temp      = 0.0f;  // deterministic sampling

    printf("%s: reading bark text model\n", __func__);
    if(!gpt_model_load(fname, model)) {
        fprintf(stderr, "%s: invalid model file '%s'\n", __func__, fname.c_str());
        return 1;
    }

    for (const auto & test_kv : k_tests()) {
        bark_sequence pad = pad_input(test_kv.first);
        bark_sequence res = bark_forward_text_encoder(
            pad, model, rng, n_threads, temp, true, min_eos_p);

        bool correct = res.size() == test_kv.second.size();

        for (int i = 0; i < (int) res.size() && correct; ++i) {
            if (res[i] != test_kv.second[i]) {
                correct = false;
            }
        }

        if (!correct) {
            fprintf(stderr, "%s : failed test \n", __func__);
            fprintf(stderr, "%s : expected tokens (n=%zu): ", __func__, test_kv.second.size());
            for (const auto & t : test_kv.second) {
                fprintf(stderr, "%d ", t);
            }
            fprintf(stderr, "\n");
            fprintf(stderr, "%s : got tokens (n=%zu):      ", __func__, res.size());
            for (const auto & t : res) {
                fprintf(stderr, "%d ", t);
            }
            fprintf(stderr, "\n");

            return 3;
        }
    }

    fprintf(stderr, "%s : tests passed successfully.\n", __func__);

    return 0;
}
