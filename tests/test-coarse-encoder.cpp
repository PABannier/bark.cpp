#include "bark.h"

#include <cstdio>
#include <string>
#include <map>
#include <random>
#include <vector>

static const std::map<std::vector<bark_vocab::id>, std::vector<std::vector<bark_vocab::id>>> & k_tests()
{
    static const std::vector<bark_vocab::id> seq1 = { 215, 1988, 3275, 1898, 1898, 1898, 9372, 9372, 222, 334, 8568, 8568, 7963, 222, 8568,  55, 7963, 1270,  55, 1283, 1283, 222, 1283, 1283, 1283,  55, 1283, 5960, 5960, 5960, 5960, 5960, 5960, 5960, 231, 5960, 5960, 5960, 5960, 5960, 5960, 5960, 5960, 5960, 5960, 5960, 5960, 5960, 340, 5960, 5960, 5960, 5960, 1374, 4193, 4193, 9323, 1374, 1374, 1374, 1374, 4193, 1374, 4193, 1374, 1374, 4193, 1374, 231, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 8328, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 9318, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374, 1374 };
    static const std::vector<bark_vocab::id> seq2 = { 59, 28, 28, 107, 7409, 1999, 7695, 6486, 6486, 5836, 5836, 5836, 873, 2585, 92, 92, 59, 28, 28, 107, 315, 5623, 1025, 10, 173, 125, 7385, 147, 147, 3689, 302, 9600, 6876, 6876, 321, 41, 164, 1367, 739, 41, 10, 140, 140, 6202, 6051, 6051, 4071, 9804, 8583, 677, 3, 17, 113, 9414, 5419, 5419, 3831, 3663, 3663, 3663, 2224, 2224, 2224, 73, 9144, 9144, 1667, 1997, 1957, 1093, 825, 175, 175, 1087, 736, 1233, 230, 147, 147, 230, 230, 230, 230, 230, 528, 528, 528, 528, 528, 528, 528, 528, 528, 528, 528, 528, 528, 528, 528, 1613, 528, 1613, 1613, 1613, 1613, 1613, 1613, 1613, 1613, 1613, 1613, 1613, 2009, 2009 };
    static const std::vector<bark_vocab::id> seq3 = { 10, 10, 560, 10, 9602, 10, 10, 10, 302, 2363, 2919, 6860, 5127, 7134, 7134, 3934, 3934, 3352, 3352, 3507, 50, 10, 27, 27, 3320, 6107, 9891, 9891, 9891, 321, 41, 4287, 5667, 6152, 6152, 557, 1228, 12, 12, 200, 59, 28, 28, 28, 28, 1133, 9569, 5920, 1424, 1424, 51, 51, 682, 3820, 2107, 6059, 348, 210, 10, 10, 5, 2187, 7842, 988, 1728, 1728, 438, 366, 50, 27, 27, 181, 181, 7352, 9725, 4431, 6445, 2428, 41, 41, 41, 5119, 6557, 4212, 3963, 26, 26, 934, 1025, 1024, 173, 10, 41, 5467, 6684, 6684, 6684, 4958, 41, 298, 5982, 5982, 526, 3219, 122, 181, 10, 10, 884, 3446, 2599, 4478, 4478, 2549 };

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

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model-file>\n", argv[0]);
        return 1;
    }

    const std::string fname = argv[1];

    gpt_model model;

    const int   n_threads = 4;
    const float min_eos_p = 0.2;
    const float temp      = 0.7;

    const int max_coarse_history  = 630;
    const int sliding_window_size = 60;

    printf("%s: reading bark coarse model\n", __func__);
    if(!gpt_model_load(fname, model)) {
        fprintf(stderr, "%s: invalid model file '%s'\n", __func__, fname.c_str());
        return 1;
    }

    for (const auto & test_kv : k_tests()) {
        std::vector<std::vector<bark_vocab::id>> res = bark_forward_coarse_encoder(
            test_kv.first, model, n_threads, temp, true, min_eos_p, max_coarse_history, sliding_window_size);

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
