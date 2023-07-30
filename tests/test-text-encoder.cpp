#include "bark.h"

#include <cstdio>
#include <string>
#include <map>
#include <random>
#include <vector>

static const std::map<std::vector<bark_vocab::id>, std::vector<bark_vocab::id>> & k_tests()
{
    static std::map<std::vector<bark_vocab::id>, std::vector<bark_vocab::id>> _k_tests = {
        { { 71742, 20181, 21404 }, {1863, 2151, 2365, 5897,  176,  176,  176, 1136, 7142, 3954, 3954, 1906, 4631, 7803, 9598, 3458, 7916,  429,   41,   41, 1904, 5505, 774,  774, 2680, 2680, 6583, 6583,  891, 6583, 1260, 3062, 3062, 1260, 3062, 1260, 1260, 5980, 1260, 3794, 3794, 3794, 3794, 8594, 7449, 4087,   10,  131, 5222, 2630, 6566,  137,  107,   28,  107, 107,  385, 1634,  441, 4978, 3284, 4978,   56,  663,  461, 6248, 10, 6248, 1710, 6248,   55, 6248, 2494, 6248, 1710,   10, 1590, 4896, 2300, } },  // hello world
        { { 20579, 20172, 20199, 33733 }, { 1863,  126, 2365, 1300, 8665, 8844, 1573,  483, 5407, 8020, 4172, 1722,   59,   59, 3238, 9284, 4133, 4133,  327,   92,  327, 7035, 4284,  321,   41, 5674,   39, 3586, 1972, 4071, 9804, 8583,  288, 17,  113, 9414, 5419,  388, 3831, 3663, 2224, 2224, 3286, 1997, 1239,  825,  175,  266,   10,  266,  266, 1147, 8829,  206,  276, 206,  206,  276,  206,  206,  206,   91,  350 } },  // this is an audio
        { { 21113,  35307,  10165,  62571,  10165,  23622,  20236,  20959,  52867 }, { 1863,  126, 2365, 1300, 8665, 8844, 3584, 4334, 9655, 6981, 2255, 2255, 3507,   50,   27, 1041, 6107, 9891, 9891, 9891,  321,   41, 1018, 1516, 8104, 8104, 8104, 7759, 6126,  262,   10,   27,  583, 4045, 4107, 4107, 9844,  230, 3718,  230,  230, 1710, 8165,   10, 5278,  206,  206,  206,   56,  206,  206, 2381, 5800,  206,  528, 9862,  528, 2539,  193,  206,  528,  193,   56, 5198, 9260, 9260, 8735, 6961, 5772,  302, 5516, 2907,   59,   28,  254, 9569, 6007, 6007, 5887,  709, 2479, 2479, 6733, 8928,  201, 6543,  201,  201, 201,  482,   97,  138,  230,  230,  147,   56,   56,  230, 5800, 147,  193, 1743,  206, 6032,  206, 7800, 7056, 3340,   52,   56, 4488,   56, 4645, 2009, 4645, 2009,  147, 8435, 8435, 6273,  985, 41,    5, 3968,  988, 1728, 1728, 1728, 6463, 6191,   27,   27, 27, 3621,  417, 2595, 9725, 4431, 4431,  821,  821,  508,   41, 1732, 4976, 8530, 9261, 4212, 4106, 4106,  441, 1025, 4691,  100, 5332, 1372, 6684, 6684, 6968,  305,  305, 5982, 4418, 5721, 4710, 4637,  579, 4856, 9648, 7503,  753, 3390, 3390, 3390,  259,  259, 7075, 6661, 2066,  178,  178,  198 } },  // You cannot, sir, take from me anything
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

std::vector<bark_vocab::id> forward_text_encoder(
        const gpt_model & model, 
        const std::vector<bark_vocab::id> & tokens,
        const int n_threads,
        const float min_eos_p) {
    std::vector<bark_vocab::id> out;
    std::vector<bark_vocab::id> input = tokens;

    int n_past = 0;
    float eos_p = 0;

    bool merge_ctx;

    std::vector<float> logits;
    std::mt19937 rng(0);

    // dry run to estimate mem_per_token
    size_t mem_per_token = 0;
    gpt_eval(model, n_threads, 0, false, { 0, 1, 2, 3 }, logits, mem_per_token);

    for (int i = 0; i < 768; i++) {
        merge_ctx = i == 0;
        gpt_eval(model, n_threads, n_past, merge_ctx, input, logits, mem_per_token);

        float logits_pad_token = logits[SEMANTIC_PAD_TOKEN];
        logits.resize(SEMANTIC_VOCAB_SIZE);
        logits.push_back(logits[logits_pad_token]);

        n_past += input.size();
        if (i == 0)
            n_past -= 256;  // first step, context are merged

        input.clear();

        bark_vocab::id sampled_id = gpt_sample(logits, 1.0f, rng, &eos_p);
        input.push_back(sampled_id);
        out.push_back(sampled_id);

        if ((sampled_id == SEMANTIC_VOCAB_SIZE) || (eos_p > min_eos_p))
            break;
    }

    return out;
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

    int ix = 1;

    printf("%s: reading bark text model\n", __func__);
    if(!gpt_model_load(fname, model)) {
        fprintf(stderr, "%s: invalid model file '%s'\n", __func__, fname.c_str());
        return 1;
    }

    for (const auto & test_kv : k_tests()) {
        std::vector<bark_vocab::id> pad = pad_input(test_kv.first);
        std::vector<bark_vocab::id> res = forward_text_encoder(model, pad, n_threads, min_eos_p);

        bool correct = res.size() == test_kv.second.size();

        for (int i = 0; i < (int) res.size() && correct; ++i) {
            if (res[i] != test_kv.second[i]) {
                correct = false;
            }
        }

        if (!correct) {
            fprintf(stderr, "%s : failed test '%d'\n", __func__, ix);
            fprintf(stderr, "%s : expected tokens: ", __func__);
            for (const auto & t : test_kv.second) {
                fprintf(stderr, "%6d, ", t);
            }
            fprintf(stderr, "\n");
            fprintf(stderr, "%s : got tokens:      ", __func__);
            for (const auto & t : res) {
                fprintf(stderr, "%6d, ", t);
            }
            fprintf(stderr, "\n");

            return 3;
        }

        ix += 1;
    }

    fprintf(stderr, "%s : tests passed successfully.\n", __func__);

    return 0;
}
