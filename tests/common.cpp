#include <fstream>
#include <vector>
#include <tuple>

#include "bark-util.h"
#include "common.h"

inline bool all_close(
    logit_sequence s1, logit_sequence s2, float * max_violation, int * n_violations) {
    if (s1.size() != s2.size()) { return false; }
    for (int i = 0; i < (int) s1.size(); i++) {
        float violation = fabs(s1[i] - s2[i]);
        *max_violation = std::max(*max_violation, violation);
        if (*max_violation > ABS_TOL + REL_TOL * fabs(s2[i]))
            *n_violations += 1;
    }
    return *n_violations == 0;
}

bool run_test_on_sequence(logit_sequence truth, logit_sequence result) {
    float max_violation = 0.0f;
    int n_violations = 0;
    if (!all_close(result, truth, &max_violation, &n_violations)) {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s : failed test\n", __func__);
        fprintf(stderr, "       abs_tol=%.4f, rel_tol=%.4f, abs max viol=%.4f, viol=%.1f%%", ABS_TOL, REL_TOL, max_violation, (float)n_violations/truth.size()*100);
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

        input.resize(nelements);
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

        logits.resize(nelements);
        fin.read(reinterpret_cast<char *>(logits.data()), nelements*sizeof(float));
    }

    assert(fin.eof());
}