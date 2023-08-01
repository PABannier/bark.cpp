#include "bark.h"

#include <cstdio>
#include <string>
#include <map>
#include <random>
#include <vector>

static const std::map<std::vector<std::vector<bark_vocab::id>>, std::vector<std::vector<bark_vocab::id>>> & k_tests()
{
    static const std::vector<std::vector<bark_vocab::id>> seq1 = {};
    static const std::vector<std::vector<bark_vocab::id>> seq2 = {};
    static const std::vector<std::vector<bark_vocab::id>> seq3 = {};

    static const std::vector<std::vector<bark_vocab::id>> ans1 = { {}, {} };
    static const std::vector<std::vector<bark_vocab::id>> ans2 = { {}, {} };
    static const std::vector<std::vector<bark_vocab::id>> ans3 = { {}, {} };

    static std::map<std::vector<std::vector<bark_vocab::id>>, std::vector<std::vector<bark_vocab::id>>> _k_tests = {
        // { seq1, ans1 },  // hello world
        // { seq2, ans2 },  // this is an audio
        { seq3, ans3 },  // You cannot, sir, take from me anything
    };
    return _k_tests;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model-file>\n", argv[0]);
        return 1;
    }

    const std::string fname = argv[1];

    gpt_model model;
    std::mt19937 rng(0);

    const int   n_threads = 4;
    const float temp      = 0.0f;  // deterministic sampling

    printf("%s: reading bark coarse model\n", __func__);
    if(!gpt_model_load(fname, model)) {
        fprintf(stderr, "%s: invalid model file '%s'\n", __func__, fname.c_str());
        return 1;
    }

    for (const auto & test_kv : k_tests()) {
        std::vector<std::vector<bark_vocab::id>> res = bark_forward_fine_encoder(
            test_kv.first, model, rng, n_threads, temp);

        bool correct = res.size() == test_kv.second.size();

        for (int i = 0; i < (int) res.size() && correct; ++i) {
            correct = res[i].size() == test_kv.second[i].size();
            for (int j = 0; j < (int) res[i].size() && correct; j++) {
                if (res[i][j] != test_kv.second[i][j]) {
                    correct = false;
                }
            }
        }

        if (!correct) {
            fprintf(stderr, "%s : failed test \n", __func__);
            fprintf(stderr, "%s : expected tokens (n=%zu): ", __func__, test_kv.second.size());
            for (int i = 0; i < (int) test_kv.second.size(); i++) {
                for (int j = 0; j < (int) test_kv.second[i].size(); j++) {
                    fprintf(stderr, "%d ", test_kv.second[i][j]);
                }
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "\n");
            fprintf(stderr, "%s : got tokens (n=%zu):      ", __func__, res.size());
            for (int i = 0; i < (int) res.size(); i++) {
                for (int j = 0; j < (int) res[i].size(); j++) {
                    fprintf(stderr, "%d ", res[i][j]);
                }
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "\n");

            return 3;
        }
    }

    fprintf(stderr, "%s : tests passed successfully.\n", __func__);

    return 0;
}
