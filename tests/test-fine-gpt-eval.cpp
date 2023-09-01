#include <string>
#include <vector>

#include "bark.h"
#include "common.h"


static const std::vector<std::tuple<std::string, int>> test_args = {
    { "./data/fine_gpt_eval/test_fine_gpt_eval_1.bin", 2 },   // prompt: Hello, my name is Suno. And, uh - and I like pizza. [laughs] But I also have other interests such as playing tic tac toe.
    { "./data/fine_gpt_eval/test_fine_gpt_eval_2.bin", 3 },   // prompt: Buenos días Miguel. Tu colega piensa que tu alemán es extremadamente malo. But I suppose your english isn't terrible.
    { "./data/fine_gpt_eval/test_fine_gpt_eval_3.bin", 4 },   // prompt: ♪ In the jungle, the mighty jungle, the lion barks tonight ♪
    { "./data/fine_gpt_eval/test_fine_gpt_eval_4.bin", 5 },   // prompt: I have a silky smooth voice, and today I will tell you about the exercise regimen of the common sloth.
    { "./data/fine_gpt_eval/test_fine_gpt_eval_5.bin", 6 },   // prompt: You cannot, my good sir, take that away from me without having me retaliate in the most ferocious way.
    { "./data/fine_gpt_eval/test_fine_gpt_eval_6.bin", 7 },   // prompt: C’est un roc ! c’est un pic ! c’est un cap ! Que dis-je, c’est un cap ? C’est une péninsule !
};

static const int n_threads = 4;

template<typename T>
std::vector<T> flatten(std::vector<std::vector<T>> const &vec) {
    std::vector<T> flattened;
    for (auto const &v: vec) {
        flattened.insert(flattened.end(), v.begin(), v.end());
    }
    return flattened;
}

int main() {
    const std::string fname = "../ggml_weights/ggml_weights_fine.bin";

    gpt_model model;
    if(!gpt_model_load(fname, model)) {
        fprintf(stderr, "%s: invalid model file '%s'\n", __func__, fname.c_str());
        return 1;
    }

    bark_codes tokens;
    std::vector<float> gt_logits, logits;

    // dry run to estimate mem_per_token
    bark_sequence decoy = { 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8 };
    fine_gpt_eval(model, decoy.data(), decoy.size(), nullptr, n_threads, 2);

    for (int i = 0; i < (int) test_args.size(); i++) {
        std::string path = std::get<0>(test_args[i]);
        int codebook_ix  = std::get<1>(test_args[i]);

        tokens.clear();
        gt_logits.clear();
        logits.clear();

        load_test_data(path, tokens, gt_logits);
        tokens = transpose(tokens);

        std::vector<int> tokens_vec = flatten(tokens);

        logits.resize(1024*1056);
        fine_gpt_eval(model, tokens_vec.data(), tokens_vec.size(), logits.data(), n_threads, codebook_ix);

        printf("\n");
        printf("%s: %s\n", __func__, path.c_str());
        if (!run_test(gt_logits, logits)) {
            printf("%s:     test %d failed.\n", __func__, i+1);
        } else {
            printf("%s:     test %d passed.\n", __func__, i+1);
        }
    }

    return 0;
}