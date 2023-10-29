/* Usage:

```bash
    ./bin/test-tokenizer ../ggml_weights/ggml_vocab.bin
```
*/
#include <cstdio>
#include <string>
#include <map>
#include <vector>

#include "bark.h"

static const std::map<std::string, bark_sequence> & k_tests()
{
    static std::map<std::string, bark_sequence> _k_tests = {
        { "Hello world!",                       { 31178, 11356,   106,                                    }, },
        { "Hello world",                        { 31178, 11356,                                           }, },
        { " Hello world!",                      { 31178, 11356,   106,                                    }, },
        { "this is an audio generated by bark", { 10531, 10124, 10151, 23685, 48918, 10155, 18121, 10174, }, },
        { "we are living on earth",             { 11951, 10301, 14625, 10135, 39189                       }, }
    };
    return _k_tests;
};

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model-file>\n", argv[0]);
        return 1;
    }

    const std::string fname = argv[1];

    fprintf(stderr, "%s : reading vocab from: '%s'\n", __func__, fname.c_str());

    bark_vocab vocab;
    int max_ctx_size = 256;

    if (!bark_vocab_load(fname.c_str(), &vocab, 119547)) {
        fprintf(stderr, "%s: invalid vocab file '%s'\n", __func__, fname.c_str());
        exit(1);
    }

    for (const auto & test_kv : k_tests()) {
        bark_sequence res(test_kv.first.size());
        int n_tokens;
        bert_tokenize(&vocab, test_kv.first.c_str(), res.data(), &n_tokens, max_ctx_size);
        res.resize(n_tokens);

        bool correct = res.size() == test_kv.second.size();

        for (int i = 0; i < (int) res.size() && correct; ++i) {
            if (res[i] != test_kv.second[i]) {
                correct = false;
            }
        }

        if (!correct) {
            fprintf(stderr, "%s : failed test: '%s'\n", __func__, test_kv.first.c_str());
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
    }

    fprintf(stderr, "%s : tests passed successfully.\n", __func__);

    return 0;
}
