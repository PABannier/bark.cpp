#include <fstream>
#include <vector>
#include <tuple>

#include "bark-util.h"
#include "common.h"

template <typename T, typename U>
inline bool all_equal(std::vector<T> s1, std::vector<U> s2, int * n_violations) {
    if (s1.size() != s2.size()) { return false; }
    for (int i = 0; i < (int) s1.size(); i++) {
        if (s1[i] != s2[i])
            *n_violations += 1;
    }
    return *n_violations == 0;
}

template bool all_equal(std::vector<int> s1, std::vector<int> s2, int * n_violations);
template bool all_equal(std::vector<float> s1, std::vector<float> s2, int * n_violations);

template <typename T, typename U>
inline bool all_close(
    std::vector<T> s1, std::vector<U> s2, float * max_violation, int * n_violations) {
    if (s1.size() != s2.size()) { return false; }
    for (int i = 0; i < (int) s1.size(); i++) {
        float violation = fabs(s1[i] - s2[i]);
        *max_violation = std::max(*max_violation, violation);
        if (*max_violation > ABS_TOL + REL_TOL * fabs(s2[i]))
            *n_violations += 1;
    }
    return *n_violations == 0;
}

template bool all_close(std::vector<int> s1, std::vector<int> s2, float * max_violation, int * n_violations);
template bool all_close(std::vector<float> s1, std::vector<float> s2, float * max_violation, int * n_violations);

inline bool all_close_nested(
    std::vector<std::vector<float>> s1, std::vector<std::vector<float>> s2,
    float * max_violation, int * n_violations) {
    if (s1.size() != s2.size()) { return false; }
    for (int i = 0; i < (int) s1.size(); i++) {
        if (s1[i].size() != s2[i].size()) { return false; }
        for (int j = 0; j < (int) s1[i].size(); j++) {
            float violation = fabs(s1[i][j] - s2[i][j]);
            *max_violation = std::max(*max_violation, violation);
            if (*max_violation > ABS_TOL + REL_TOL * fabs(s2[i][j]))
                *n_violations += 1;
        }
    }
    return *n_violations == 0;
}

bool run_test_on_sequence(std::vector<float> truth, std::vector<float> result) {
    float max_violation = 0.0f;
    int n_violations = 0;
    if (!all_close(result, truth, &max_violation, &n_violations)) {
        if (n_violations == 0) {
            fprintf(stderr, "%s : wrong shape (%zu != %zu).\n", __func__, truth.size(), result.size());
        } else {
            fprintf(stderr, "\n");
            fprintf(stderr, "       abs_tol=%.4f, rel_tol=%.4f, abs max viol=%.4f, viol=%.1f%%", ABS_TOL, REL_TOL, max_violation, (float)n_violations/truth.size()*100);
            fprintf(stderr, "\n");
        }
        return false;
    }
    return true;
}

bool run_test_on_sequence(std::vector<int> truth, std::vector<int> result) {
    int n_violations = 0;
    if (!all_equal(result, truth, &n_violations)) {
        if (n_violations == 0) {
            fprintf(stderr, "%s : wrong shape (%zu != %zu).\n", __func__, truth.size(), result.size());
        } else {
            fprintf(stderr, "\n");
            fprintf(stderr, "       viol=%.1f%%", (float)n_violations/truth.size()*100);
            fprintf(stderr, "\n");
        }
        return false;
    }
    return true;
}

bool run_test_on_codes(logit_matrix truth, logit_matrix result) {
    float max_violation = 0.0f;
    int n_violations = 0;
    if (!all_close_nested(result, truth, &max_violation, &n_violations)) {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s : failed test\n", __func__);
        if (n_violations == 0) {
            fprintf(stderr, "%s : wrong shape (%zu != %zu).\n", __func__, truth.size(), result.size());
        } else {
            fprintf(stderr, "       abs_tol=%.4f, rel_tol=%.4f, abs max viol=%.4f, viol=%.1f%%", ABS_TOL, REL_TOL, max_violation, (float)n_violations/(truth.size()*truth[0].size())*100);
            fprintf(stderr, "\n");
        }
        return false;
    }
    return true;
}

template <typename T, typename U>
void load_test_data(std::string fname, std::vector<T>& input, std::vector<U>& output) {
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
        fin.read(reinterpret_cast<char *>(input.data()), nelements*sizeof(T));
    }

    // output
    {
        int32_t n_dims;
        read_safe(fin, n_dims);

        int32_t nelements = 1;
        int32_t ne[3] = { 1, 1, 1 };
        for (int i = 0; i < n_dims; i++) {
            read_safe(fin, ne[i]);
            nelements *= ne[i];
        }

        output.resize(nelements);
        fin.read(reinterpret_cast<char *>(output.data()), nelements*sizeof(U));
    }

    assert(fin.eof());
}

template void load_test_data(std::string fname, std::vector<int>& input, std::vector<float>& output);
template void load_test_data(std::string fname, std::vector<int32_t>& input, std::vector<int32_t>& output);

void load_nested_test_data(
        std::string fname,
        std::vector<std::vector<int>>& input,
        logit_matrix& logits) {
    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        throw;
    }

    // input
    {
        int32_t n_dims;
        read_safe(fin, n_dims);

        int32_t ne[3] = { 1, 1, 1 };
        for (int i = 0; i < n_dims; i++) { read_safe(fin, ne[i]); }

        for (int i = 0; i < ne[0]; i++) {
            std::vector<int> _tmp(ne[1]);
            fin.read(reinterpret_cast<char *>(_tmp.data()), ne[1]*sizeof(int32_t));
            input.push_back(_tmp);
        }
    }

    // logits
    {
        int32_t n_dims;
        read_safe(fin, n_dims);

        int32_t ne[3] = { 1, 1, 1 };
        for (int i = 0; i < n_dims; i++) { read_safe(fin, ne[i]); }

        for (int i = 0; i < ne[0]; i++) {
            std::vector<float> _tmp(ne[1]);
            fin.read(reinterpret_cast<char *>(_tmp.data()), ne[1]*sizeof(float));
            logits.push_back(_tmp);
        }
    }

    assert(fin.eof());
}