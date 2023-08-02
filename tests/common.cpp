#include "bark.h"
#include <vector>
#include <tuple>

#define TOL 0.05f

typedef std::vector<float> logit_sequence;
typedef std::tuple<bark_sequence, logit_sequence, logit_sequence> test_data_t;

inline bool all_close(logit_sequence s1, logit_sequence s2, float tol) {
    if (s1.size() != s2.size()) { return false; }
    for (int i = 0; i < (int) s1.size(); i++) {
        if (fabs(s1[i] - s2[i]) > tol) {
            return false;
        }
    }
    return true;
}

bool run_test(logit_sequence truth, logit_sequence logits, bool merge_ctx) {
    logit_sequence result;
    result.insert(result.end(), logits.begin(), logits.begin() + 50);
    result.insert(result.end(), logits.end() - 50, logits.end());

    if (!all_close(result, truth, TOL)) {
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
