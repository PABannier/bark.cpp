#include <string>
#include <vector>

#include "bark.h"
#include "common.h"


static const std::vector<std::tuple<std::string, bool>> test_args = {
    { "./data/test_gpt_eval_1_no_merge.bin", false },  // prompt:
    { "./data/test_gpt_eval_2_no_merge.bin", false },  // prompt:
    { "./data/test_gpt_eval_3_no_merge.bin", false },  // prompt:

    { "./data/test_gpt_eval_1_merge.bin", false },     // prompt:
    { "./data/test_gpt_eval_2_merge.bin", false },     // prompt:
    { "./data/test_gpt_eval_3_merge.bin", false },     // prompt:
}

static const int n_threads = 4;

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model-file>\n", argv[0]);
        return 1;
    }

    const std::string fname = argv[1];

    gpt_model model;
    if(!gpt_model_load(fname, model)) {
        fprintf(stderr, "%s: invalid model file '%s'\n", __func__, fname.c_str());
        return 1;
    }

    bark_sequence tokens;
    logit_sequence gt_logits, logits;

    // dry run to estimate mem_per_token
    size_t mem_per_token = 0;
    {
        int n_past = 0;
        gpt_eval(model, n_threads, &n_past, false, { 0, 1, 2, 3 }, logits, mem_per_token);
    }

    for (int i = 0; i < (int) test_args.size(), i++) {
        tokens.clear();
        gt_logits.clear();
        logits.clear();

        std::string path = std::get<0>(test_args[i]);
        bool merge_ctx   = std::get<1>(test_args[i]);

        load_test_data(path, tokens, gt_logits);

        int n_past = 0;
        gpt_eval(model, n_threads, &n_past, merge_ctx, tokens, logits, mem_per_token);

        printf("%s: %s\n", __func__, path.c_str());
        if (!run_test(gt_logits, logits)) {
            printf("%s:     test %d failed.\n", i+1);
        } else {
            printf("%s:     test %d passed.\n", i+1);
        }
    }

    return 0;
}