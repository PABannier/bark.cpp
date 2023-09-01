#include <string>
#include <vector>

#include "bark.h"
#include "common.h"


static const std::vector<std::tuple<std::string, bool>> test_args = {
    { "./data/gpt_eval/test_gpt_eval_1_no_merge.bin", false },  // prompt: Hello, my name is Suno. And, uh - and I like pizza. [laughs] But I also have other interests such as playing tic tac toe.
    { "./data/gpt_eval/test_gpt_eval_2_no_merge.bin", false },  // prompt: Buenos días Miguel. Tu colega piensa que tu alemán es extremadamente malo. But I suppose your english isn't terrible.
    { "./data/gpt_eval/test_gpt_eval_3_no_merge.bin", false },  // prompt: ♪ In the jungle, the mighty jungle, the lion barks tonight ♪

    { "./data/gpt_eval/test_gpt_eval_1_merge.bin", true },     // prompt: I have a silky smooth voice, and today I will tell you about the exercise regimen of the common sloth.
    { "./data/gpt_eval/test_gpt_eval_2_merge.bin", true },     // prompt: You cannot, my good sir, take that away from me without having me retaliate in the most ferocious way.
    { "./data/gpt_eval/test_gpt_eval_3_merge.bin", true },     // prompt: Ceci est un texte en français pour tester le bon fonctionnement de bark.
};

static const int n_threads = 4;

int main() {
    const std::string fname = "../ggml_weights/ggml_weights_text.bin";

    gpt_model model;
    if(!gpt_model_load(fname, model)) {
        fprintf(stderr, "%s: invalid model file '%s'\n", __func__, fname.c_str());
        return 1;
    }

    bark_sequence tokens;
    logit_sequence gt_logits, logits;

    auto & hparams = model.hparams;
    int n_vocab = hparams.n_out_vocab;
    logits.resize(n_vocab);

    // dry run to estimate mem_per_token
    {
        int n_past = 0;
        bark_token decoy[4] = { 0, 1, 2, 3 };
        gpt_eval(model, decoy, 4, nullptr, &n_past, false, n_threads);
    }

    for (int i = 0; i < (int) test_args.size(); i++) {
        tokens.clear();
        gt_logits.clear();

        std::string path = std::get<0>(test_args[i]);
        bool merge_ctx   = std::get<1>(test_args[i]);

        load_test_data(path, tokens, gt_logits);

        int n_past = 0;
        gpt_eval(model, tokens.data(), tokens.size(), logits.data(), &n_past, merge_ctx, n_threads);

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