#include <cstdio>
#include <string>
#include <random>
#include <vector>

#include "bark.h"
#include "common.h"

static const std::vector<std::string> test_data = {
    // "./data/encodec/test_pass_encodec_1.bin",   // prompt: El hombre que se levanta es aún más grande que el que no ha caído.
    // "./data/encodec/test_pass_encodec_2.bin",   // prompt: ♪ Heal the world, Make it a better place, For you and for me, and the entire human race ♪
    // "./data/encodec/test_pass_encodec_3.bin",   // prompt: En été, mieux vaut suer que trembler.
    "./data/encodec/test_pass_encodec_4.bin"    // prompt: we are living on earth
};

int main() {
    const std::string fname = "../ggml_weights/ggml_weights_codec.bin";

    encodec_model model;
    if(!encodec_model_load(fname, model)) {
        fprintf(stderr, "%s: invalid model file '%s'\n", __func__, fname.c_str());
        return 1;
    }

    bark_codes tokens;
    audio_arr_t gt_audio_arr;

    for (int i = 0; i < (int) test_data.size(); i++) {
        tokens.clear();
        gt_audio_arr.clear();

        std::string path = test_data[i];
        load_test_data(path, tokens, gt_audio_arr);

        audio_arr_t audio_arr = bark_forward_encodec(transpose(tokens), model);

        printf("\n");
        printf("%s: %s\n", __func__, path.c_str());
        if (!run_test(gt_audio_arr, audio_arr)) {
            printf("%s:     test %d failed.\n", __func__, i+1);
        } else {
            printf("%s:     test %d passed.\n", __func__, i+1);
        }
    }

    return 0;
}
