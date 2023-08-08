/* This test checks that the forward pass as defined in `bark_forward_encodec`
yields the same output as the original bark implementation.
*/
#include <cstdio>
#include <string>
#include <random>
#include <vector>

#include "bark.h"
#include "common.h"

static const std::vector<std::string> test_data = {
    "./data/encodec/test_pass_encodec_1.bin",
};

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model-file>\n", argv[0]);
        return 1;
    }

    const std::string fname = argv[1];

    encodec_model model;
    const int n_threads = 1;  // TODO: should be 4

    bool success = true;

    if(!encodec_model_load(fname, model)) {
        fprintf(stderr, "%s: invalid model file '%s'\n", __func__, fname.c_str());
        return 1;
    }

    for (int i = 0; i < (int) test_data.size(); i++) {
        bark_codes input;
        audio_arr_t truth;
        std::string path = test_data[i];

        load_test_data(path, input, truth);
        bark_codes input_t = transpose(input);

        audio_arr_t output = bark_forward_encodec(input_t, model, n_threads);

        fprintf(stderr, "input_t = [%zu, %zu]\n", input_t.size(), input_t[0].size());
        fprintf(stderr, "output  = [%zu, 1]\n", output.size());
        fprintf(stderr, "truth   = [%zu, 1]\n", truth.size());

        fprintf(stderr, "%s", path.c_str());
        if (!run_test_on_sequence(truth, output)) {
            success = false;
            fprintf(stderr, "   TEST %d FAILED.\n", i+1);
        } else {
            fprintf(stderr, "   TEST %d PASSED.\n", i+1);
        }
    }

    if (success)
        fprintf(stderr, "%s : tests passed successfully.\n", __func__);

    return 0;
}