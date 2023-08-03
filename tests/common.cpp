#include <fstream>
#include <vector>
#include <tuple>

#include "bark-util.h"
#include "common.h"

inline bool all_close(logit_sequence s1, logit_sequence s2, float tol) {
    if (s1.size() != s2.size()) { return false; }
    for (int i = 0; i < (int) s1.size(); i++) {
        if (fabs(s1[i] - s2[i]) > tol) {
            return false;
        }
    }
    return true;
}

bool run_test_on_sequence(logit_sequence truth, logit_sequence logits, bool merge_ctx) {
    logit_sequence result;
    result.insert(result.end(), logits.begin(), logits.begin() + 50);
    result.insert(result.end(), logits.end() - 50, logits.end());

    if (!all_close(result, truth, TOL)) {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s : failed test (merge_ctx=%d) \n", __func__, merge_ctx);
        fprintf(stderr, "%s : expected tokens (n=%zu): ", __func__, truth.size());
        for (const auto & l : truth) {
            fprintf(stderr, "%.4f ", l);
        }
        fprintf(stderr, "\n");
        fprintf(stderr, "%s : got tokens (n=%zu):      ", __func__, result.size());
        for (const auto & l : result) {
            fprintf(stderr, "%.4f ", l);
        }
        fprintf(stderr, "\n");

        return false;
    }

    return true;
}

void load_test_data(std::string fname, std::vector<int>& input, std::vector<float>& logits) {
    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        throw;
    }

    // input
    {
        int32_t n_dims;
        read_safe(fin, n_dims);

        int32_t nelements = 1;
        int32_t ne[3] = { 1, 1, 1 };
        for (int i = 0; i < n_dims; i++) {
            read_safe(fin, ne[i]);
            nelements *= ne[i];
        }

        input.reserve(nelements);
        fin.read(reinterpret_cast<char *>(input.data()), nelements*sizeof(int32_t));
    }

    // logits
    {
        int32_t n_dims;
        read_safe(fin, n_dims);

        int32_t nelements = 1;
        int32_t ne[3] = { 1, 1, 1 };
        for (int i = 0; i < n_dims; i++) {
            read_safe(fin, ne[i]);
            nelements *= ne[i];
        }

        logits.reserve(nelements);
        fin.read(reinterpret_cast<char *>(logits.data()), nelements*sizeof(float));
    }

    assert(fin.eof());
}