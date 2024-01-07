/* Usage:

```bash
    ./bin/test-fine-encoder ../ggml_weights/
```
*/
#include <cstdio>
#include <string>
#include <vector>

#include "bark.h"

const int n_threads = 4;
const bark_verbosity_level verbosity = bark_verbosity_level::MEDIUM;

const bark_codes coarse_tokens = {
    { 395, 395, 395, 395, 475, 395, 475, 395, 395, 395, 395, 395, 819, 395, 395, 395, 395, 395, 395, 819, 819, 395, 395, 395, 395, 395, 395, 395, 395, 395, 537, 887, 537, 499, 835, 475, 404, 475, 395, 475, 855, 257, 475, 404, 779, 779, 395, 395, 23, 59, 881, 59, 901, 151, 860, 819, 819, 819, 373, 819, 819, 635, 1011, 373, 798, 819, 373, 819, 709, 819, 819, 819, 635, 323, 192, 901, 59, 942, 871, 208, 430, 604, 834, 430, 475, 475, 395, 475, 537, 233, 747, 428, 683, 112, 402, 216, 683, 112, 402, 216, 216, 99, 683, 112, 402, 216, 216, 683, 112, 428, 428, 690, 942, 871, 208, 228, 904, 404, 404, 499, 404, 475, 395, 475, 257, 835, 475, 475, 475, 395, 475, 257, 475, 475, 855, 887, 392, 216, 683, 112, 112, 402, 11, 11, 11, 323, 91, 904, 404, 855, 404, 779, 677, 475, 59, 59, 151, 276, 23, 276, 276, 347, 347, 879, 753, 325, 879, 1011, 753, 276, 276, 753, 276, 228, 855, 835, 475, 475, 475, 475, 106, 475, 395, 537, 835, 257, 404, 835, 475, 887, 475, 475, 475, 855, 475, 475, 475, 475, 475, 475, 475, 475, 475, 475 },
    { 969, 928, 928, 913, 928, 43, 424, 913, 518, 200, 200, 544, 544, 200, 200, 200, 424, 200, 424, 544, 969, 200, 964, 200, 913, 969, 544, 200, 200, 544, 646, 200, 913, 648, 969, 518, 544, 424, 913, 518, 424, 544, 913, 424, 544, 913, 913, 544, 73, 504, 591, 952, 591, 655, 1007, 429, 603, 857, 4, 857, 896, 1010, 504, 35, 955, 67, 4, 1010, 857, 857, 857, 857, 961, 964, 381, 955, 952, 955, 386, 403, 601, 961, 765, 544, 913, 424, 765, 424, 928, 453, 403, 505, 833, 478, 478, 478, 478, 478, 478, 478, 478, 95, 478, 478, 478, 478, 478, 478, 95, 663, 136, 386, 386, 891, 770, 896, 516, 937, 544, 747, 928, 969, 913, 424, 363, 424, 424, 424, 424, 646, 913, 544, 928, 424, 544, 463, 478, 185, 776, 300, 685, 685, 371, 663, 513, 105, 1007, 770, 1007, 969, 544, 964, 648, 519, 717, 591, 833, 364, 364, 105, 364, 770, 200, 364, 519, 519, 519, 519, 519, 745, 942, 519, 829, 928, 859, 937, 913, 424, 544, 424, 424, 518, 200, 648, 928, 544, 544, 424, 424, 646, 913, 424, 913, 544, 913, 913, 913, 518, 928, 913, 913, 913, 913, 518},
};

std::vector<std::vector<int> > transpose(const std::vector<std::vector<int> > data) {
    // this assumes that all inner vectors have the same size and
    // allocates space for the complete result in advance
    std::vector<std::vector<int> > result(data[0].size(),
                                          std::vector<int>(data.size()));
    for (std::vector<int>::size_type i = 0; i < data[0].size(); i++)
        for (std::vector<int>::size_type j = 0; j < data.size(); j++) {
            result[i][j] = data[j][i];
        }
    return result;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model-file>\n", argv[0]);
        return 1;
    }

    const std::string weights_dir = argv[1];

    // initialize bark context
    struct bark_context * bctx = bark_load_model(weights_dir.c_str(), verbosity);
    if (!bctx) {
        fprintf(stderr, "%s: Could not load model\n", __func__);
        exit(1);
    }

    bctx->coarse_tokens = transpose(coarse_tokens);

    // generate fine tokens
    if (!bark_forward_fine_encoder(bctx, n_threads, verbosity)) {
        fprintf(stderr, "%s: failed to forward fine encoder\n", __func__);
        return 1;
    }

    // print fine tokens
    fprintf(stderr, "shape of fine tokens: [%zu, %zu]\n", bctx->fine_tokens.size(), bctx->fine_tokens[0].size());

    bark_codes ft = transpose(bctx->fine_tokens);
    // bark_codes ft = bctx->fine_tokens;

    for (size_t i = 0; i < ft.size(); i++) {
        for (size_t j = 0; j < ft[i].size(); j++) {
            fprintf(stderr, "%d ", ft[i][j]);
        }
        fprintf(stderr, "\n");
    }

    return 0;
}