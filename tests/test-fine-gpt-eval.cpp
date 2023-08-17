#include <string>
#include <vector>

#include "bark.h"
#include "common.h"


static const std::tuple<std::vector<std::string>, int> test_args = {
    { "./data/gpt_eval/test_fine_gpt_eval_1.bin", 2 },   // prompt: Hello, my name is Suno. And, uh - and I like pizza. [laughs] But I also have other interests such as playing tic tac toe.
    { "./data/gpt_eval/test_fine_gpt_eval_2.bin", 3 },   // prompt: Buenos días Miguel. Tu colega piensa que tu alemán es extremadamente malo. But I suppose your english isn't terrible.
    { "./data/gpt_eval/test_fine_gpt_eval_3.bin", 4 },   // prompt: ♪ In the jungle, the mighty jungle, the lion barks tonight ♪
    { "./data/gpt_eval/test_fine_gpt_eval_4.bin", 5 },   // prompt: I have a silky smooth voice, and today I will tell you about the exercise regimen of the common sloth.
    { "./data/gpt_eval/test_fine_gpt_eval_5.bin", 6 },   // prompt: You cannot, my good sir, take that away from me without having me retaliate in the most ferocious way.
    { "./data/gpt_eval/test_fine_gpt_eval_6.bin", 7 },   // prompt: Ceci est un texte en français pour tester le bon fonctionnement de bark.
    { "./data/gpt_eval/test_fine_gpt_eval_7.bin", 8 },   // prompt: C’est un roc ! c’est un pic ! c’est un cap ! Que dis-je, c’est un cap ? C’est une péninsule !
};

static const int n_threads = 4;

int main() {
    const std::string fname = "../ggml_weights/ggml_weights_fine.bin";

    gpt_model model;
    if(!gpt_model_load(fname, model)) {
        fprintf(stderr, "%s: invalid model file '%s'\n", __func__, fname.c_str());
        return 1;
    }

    bark_codes tokens;
    logit_matrix gt_logits, logits;

    // dry run to estimate mem_per_token
    size_t mem_per_token = 0;
    {
        int n_past = 0;
        fine_gpt_eval(model, n_threads, 2, { 0, 1, 2, 3 }, logits, mem_per_token);
    }

    for (int i = 0; i < (int) test_args.size(); i++) {
        tokens.clear();
        gt_logits.clear();
        logits.clear();

        std::string path = std::get<0>(test_args[i]);
        int codebook_ix  = std::get<1>(test_args[i]);

        load_test_data(path, tokens, gt_logits);

        int n_past = 0;
        fine_gpt_eval(model, n_threads, codebook_ix, tokens, logits, mem_per_token);

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