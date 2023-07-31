#include "bark.h"

#include <cstdio>
#include <string>
#include <map>
#include <random>
#include <vector>

static const std::map<std::vector<bark_vocab::id>, std::vector<bark_vocab::id>> & k_tests()
{
    static const std::vector<bark_vocab::id> seq1 = { 71742, 20181, 21404        };
    static const std::vector<bark_vocab::id> seq2 = { 20579, 20172, 20199, 33733 };
    static const std::vector<bark_vocab::id> seq3 = { 21113,  35307,  10165,  62571,  10165,  23622,  20236,  20959,  52867 };

    static const std::vector<bark_vocab::id> ans1 = { 3264, 6121, 6414, 7799, 6121, 1907, 1888, 206, 1888, 1888, 6143, 2131, 728, 6328, 3393, 5990, 2992, 8837, 206, 7799, 2533, 7374, 2992, 4059, 5990, 5990, 5236, 5939, 10, 2137, 2137, 6021, 176, 176, 2584, 499, 2382, 499, 8051, 4218, 8051, 8909, 306, 6804, 6804, 292, 292, 675, 3927, 3819, 624, 6583, 8843, 6583, 6583, 9628, 3062, 6652, 6652, 6652, 4792, 6608, 4792, 6608, 9025, 8534, 709, 7431, 7431, 9693, 8858, 8858, 3820, 8858, 682, 4076, 8996, 4909, 5682, 6139, 6139, 9133, 445, 971, 7542, 4564, 9931, 9931, 785, 785, 157, 5897, 5897, 9527, 1233, 138, 131, 10, 266, 266, 1572, 1572, 206, 206, 3533, 206, 4874, 1444, 3533, 206, 206, 7397, 206, 3252, 206, 2314, 91, 206, 7567, 841, 5346, 3252, 206, 841, 3366, 517, 517, 3252, 344, 344, 1278, 3950, 57, 57, 597, 7160, 121, 7334, 631, 292, 41, 41, 8944, 1991, 1408, 1408, 1408, 1462, 3, 166, 8745, 17, 2332, 1574, 7443, 50, 17, 27, 429, 9225, 713, 4099, 4099, 4099, 75, 555, 5932, 8870, 7627, 7627, 5661, 3088, 26, 288, 262 };
    static const std::vector<bark_vocab::id> ans2 = { 3205, 6179, 7731, 6972, 5722, 602, 441, 125, 147, 991, 1573, 402, 402, 6774, 1913, 8020, 8572, 8572, 1722, 5681, 1133, 4694, 1133, 7517, 9575, 8125, 5905, 6486, 1797, 6486, 5138, 5138, 4150, 2630, 2879, 59, 28, 28, 385, 1741, 4042, 9898, 302, 9600, 7231, 5673, 5475, 321, 171, 321, 164, 1025, 4681, 6202, 6752, 8288, 6747, 7656, 9804, 9804, 2411, 178, 50, 441, 6401, 5899, 79, 6511, 6511, 9629, 6511, 6154, 2224, 2224, 73, 73, 9814, 6303, 1997, 1997, 7396, 8062, 825, 441};
    static const std::vector<bark_vocab::id> ans3 = { 3946, 8514, 7741, 9262, 5153, 4400, 4509, 512, 1136, 4631, 8486, 4631, 3954, 7234, 993, 4412, 993, 9161, 332, 8209, 5565, 4224, 4344, 6152, 6152, 2704, 2285, 4438, 232, 131, 10, 5038, 2430, 59, 28, 28, 28, 28, 1310, 4449, 5920, 9449, 2002, 9693, 7939, 4049, 4049, 6059, 210, 100, 10, 10, 282, 3968, 988, 9790, 1728, 2587, 4405, 2948, 232, 232, 100, 3621, 8680, 417, 10, 2595, 7352, 9725, 6445, 2428, 41, 41, 10, 41, 9261, 4212, 3963, 6261, 8210, 9588, 934, 441, 1025, 2875, 8558, 6968, 116, 41, 41, 7789, 5721, 5721, 267, 2116, 579, 100, 1133, 3446, 2599, 7503, 3390, 3390, 4485, 657, 1385, 1385, 7691, 7557, 5272, 8887, 10, 6619, 3592, 6394, 5272, 5272, 8887, 1841, 602, 441, 217, 4542, 5861, 5861, 3803, 4542, 4542, 4542, 6675, 7204, 131, 100, 790, 2832, 266, 6115, 4209, 1739, 1739, 1444, 8659, 1739, 1739, 1739, 1133, 1739, 1739, 2556, 2556, 413, 413, 10, 3373, 7966, 2330, 1588, 409, 2942, 59, 28, 28, 28, 10, 28, 3160, 9569, 5920, 5887, 9693, 6290, 3458, 1242, 50, 210, 2977, 1433, 1433, 6150, 6150, 1136, 6413, 9693, 3441, 9598, 9061, 7949, 9137, 5615, 131, 100, 652, 7863, 7344, 8899, 7765, 50, 10, 100, 7399, 9915, 7557, 4509, 8486, 6264, 6133, 6133, 6133, 6619, 5210, 5210, 9629, 2555, 2339, 9486, 1425, 2762, 2466, 1079, 10 };

    static std::map<std::vector<bark_vocab::id>, std::vector<bark_vocab::id>> _k_tests = {
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

    printf("%s: reading bark text model\n", __func__);
    if(!gpt_model_load(fname, model)) {
        fprintf(stderr, "%s: invalid model file '%s'\n", __func__, fname.c_str());
        return 1;
    }

    for (const auto & test_kv : k_tests()) {
        std::vector<bark_vocab::id> pad = pad_input(test_kv.first);
        std::vector<bark_vocab::id> res = bark_forward_text_encoder(
            pad, model, n_threads, 1.0f, true, min_eos_p);

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
