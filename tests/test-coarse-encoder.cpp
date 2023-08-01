#include "bark.h"

#include <cstdio>
#include <string>
#include <map>
#include <random>
#include <vector>

static const std::map<std::vector<bark_vocab::id>, std::vector<std::vector<bark_vocab::id>>> & k_tests()
{
    static const std::vector<bark_vocab::id> seq1 = { 71742, 20181, 21404        };
    static const std::vector<bark_vocab::id> seq2 = { 20579, 20172, 20199, 33733 };
    static const std::vector<bark_vocab::id> seq3 = { 21113,  35307,  10165,  62571,  10165,  23622,  20236,  20959,  52867 };

    static const std::vector<std::vector<bark_vocab::id>> ans1 = { {}, {} };
    static const std::vector<std::vector<bark_vocab::id>> ans2 = { {}, {} };
    static const std::vector<std::vector<bark_vocab::id>> ans3 = { {}, {} };

    static std::map<std::vector<bark_vocab::id>, std::vector<std::vector<bark_vocab::id>>> _k_tests = {
        { seq1, ans1 },  // hello world
        { seq2, ans2 },  // this is an audio
        { seq3, ans3 },  // You cannot, sir, take from me anything
    };
    return _k_tests;
};

std::vector<bark_vocab::id> pad_input(const std::vector<bark_vocab::id> & input) {
    int original_sz = input.size();
    std::vector<bark_vocab::id> pad(input.begin(), input.end());
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
    const int   n_threads = 4;
    const float min_eos_p = 0.2;
    const float temp = 0.7;
    const int max_coarse_history  = 630;
    const int sliding_window_size = 60;

    printf("%s: reading bark text model\n", __func__);
    if(!gpt_model_load(fname, model)) {
        fprintf(stderr, "%s: invalid model file '%s'\n", __func__, fname.c_str());
        return 1;
    }

    for (const auto & test_kv : k_tests()) {
        std::vector<std::vector<bark_vocab::id>> res = bark_forward_coarse_encoder(
            test_kv.first, model, n_threads, temp, true, min_eos_p, max_coarse_history, sliding_window_size);

        bool correct = res.size() == test_kv.second.size();

        for (int i = 0; i < (int) res.size() && correct; ++i) {
            for (int j = 0; j < (int) res[i].size() && correct; j++) {
                if (res[i][j] != test_kv.second[i][j]) {
                    correct = false;
                }
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
